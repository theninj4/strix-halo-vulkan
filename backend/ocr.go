package backend

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/ocr"
	"strix-halo-vulkan/util"
	"strix-halo-vulkan/vk"
)

// OCROptions is what cmd/serve's -ocr flags come to.
type OCROptions struct {
	// Model is the PaddleOCR-VL-1.6 checkpoint directory.
	Model string
	// Device runs it; there is no CPU path.
	Device *Device
	// MaxPixels is the largest image the tower is staged for (0: the
	// checkpoint's 1,003,520). A request's max_pixels above it is refused.
	MaxPixels int
}

// OCRModelID is the name the chat door answers to, and the one vLLM serves
// the checkpoint under in PaddleOCR's own deployment
// (`paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B`).
const OCRModelID = "PaddleOCR-VL-1.6-0.9B"

// OCR is PaddleOCR-VL's element-level recognition (OCR.md O6): an
// api.CompletionBackend over ocr.Engine, served the way vLLM serves the
// checkpoint, because that is the door PaddleOCR's page pipeline already
// knows (`--vl_rec_backend vllm-server`).
//
// A request is one user message holding one image and the task prompt, which
// is every request the checkpoint was trained on and every one PaddleX sends.
// Generation is greedy (PaddleX asks `temperature: 0`); vLLM's extras are
// honoured -- repetition_penalty, skip_special_tokens, mm_processor_kwargs'
// pixel bounds -- and every sampling knob that would change the answer is
// refused rather than ignored.
//
// Each device pass -- the tower, the prefill, every decode step -- takes the
// device lock on its own, so speech and the rest interleave with a page's
// generation a token at a time.
type OCR struct {
	opt OCROptions
	mu  sync.Mutex // guards eng against Close
	eng *ocr.Engine
}

// NewOCR loads the tokenizer, the tower and ERNIE: 2.3 GB on the device,
// about two seconds.
func NewOCR(opt OCROptions) (*OCR, error) {
	if opt.Device == nil {
		return nil, fmt.Errorf("backend: ocr needs the device (it has no CPU path)")
	}
	if opt.MaxPixels <= 0 {
		opt.MaxPixels = ocr.DefaultMaxPixels
	}
	o := ocr.DefaultOptions()
	o.MaxPatches = max(opt.MaxPixels/(ocr.PatchSize*ocr.PatchSize), ocr.MaxPatches)
	b := &OCR{opt: opt}
	err := opt.Device.Do(func(dev *vk.Device) error {
		e, err := ocr.Load(dev, opt.Model, o)
		b.eng = e
		return err
	})
	if err != nil {
		return nil, err
	}
	b.eng.Do = func(fn func() error) error {
		return opt.Device.Do(func(*vk.Device) error { return fn() })
	}
	return b, nil
}

// DeviceBytes is what the model holds on the device.
func (b *OCR) DeviceBytes() int { return b.eng.DeviceBytes() }

// Close releases the device objects.
func (b *OCR) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.eng != nil {
		_ = b.opt.Device.Do(func(*vk.Device) error { b.eng.Destroy(); return nil })
		b.eng = nil
	}
}

// Models is the one id.
func (b *OCR) Models() []api.Model {
	return []api.Model{{ID: OCRModelID, Object: "model", OwnedBy: "PaddlePaddle"}}
}

// ocrRequest is what a chat request comes to: the image's URL and the
// prompt text.
func ocrRequest(req *api.CompletionRequest, maxPixels int) (url, prompt string, err error) {
	switch {
	case len(req.Tools) > 0:
		return "", "", unsupported("this model has no tools")
	case req.Temperature != nil && *req.Temperature != 0:
		return "", "", unsupported("temperature %g: this model decodes greedily (temperature 0), as PaddleOCR asks it to", *req.Temperature)
	case req.TopK != nil && *req.TopK != 1 && *req.TopK != -1 && *req.TopK != 0:
		return "", "", unsupported("top_k %d: this model decodes greedily", *req.TopK)
	case req.MinP != nil && *req.MinP != 0:
		return "", "", unsupported("min_p: this model decodes greedily")
	case req.RepeatPenalty != nil:
		return "", "", unsupported("repeat_penalty is llama-server's windowed penalty; this model takes vLLM's repetition_penalty")
	case req.PresencePenalty != nil && *req.PresencePenalty != 0:
		return "", "", unsupported("presence_penalty is not implemented for this model")
	case len(req.Stop) > 0:
		return "", "", unsupported("stop sequences are not implemented for this model")
	case len(req.ChatTemplateKwargs) > 0 || req.ReasoningEffort != "":
		return "", "", unsupported("this model's template takes no variables and it does not reason")
	case req.RepetitionPenalty != nil && *req.RepetitionPenalty <= 0:
		return "", "", unsupported("repetition_penalty must be positive; 1 is off")
	}
	// top_p is a no-op under greedy decoding, whatever its value, so it is
	// accepted: PaddleX's fastdeploy path sends top_p 0.
	if m := req.MMProcessorKwargs; m != nil {
		if m.MaxPixels > maxPixels {
			return "", "", unsupported("mm_processor_kwargs.max_pixels %d: this server staged the tower for %d", m.MaxPixels, maxPixels)
		}
		if m.MinPixels < 0 || m.MaxPixels < 0 || (m.MaxPixels > 0 && m.MinPixels > m.MaxPixels) {
			return "", "", unsupported("mm_processor_kwargs: min_pixels %d, max_pixels %d", m.MinPixels, m.MaxPixels)
		}
	}
	if len(req.Messages) != 1 || (req.Messages[0].Role != "user" && req.Messages[0].Role != "") {
		return "", "", unsupported("%d messages: this model reads one user message, one image and its task prompt "+
			"(\"OCR:\", \"Table Recognition:\", ...), which is how it was trained", len(req.Messages))
	}
	var texts []string
	for _, part := range req.Messages[0].Content {
		switch {
		case part.Type == "image_url" && part.ImageURL != nil:
			if url != "" {
				return "", "", unsupported("more than one image: this model reads one image a request")
			}
			url = part.ImageURL.URL
		case part.Type == "" || part.Type == "text":
			texts = append(texts, part.Text)
		}
	}
	if url == "" {
		return "", "", unsupported("no image: this model recognises the text in an image (an image_url part)")
	}
	// The template writes every text part after the image, back to back.
	return url, strings.Join(texts, ""), nil
}

// Complete recognises the request's image.
func (b *OCR) Complete(ctx context.Context, req *api.CompletionRequest, emit func(api.Delta) error) (*api.CompletionResult, error) {
	enter := time.Now()
	url, prompt, err := ocrRequest(req, b.opt.MaxPixels)
	if err != nil {
		return nil, err
	}
	// Fetched before the device is asked for: a URL is a download of unknown
	// length. OpenAI fetches an image_url, so this server does too.
	data, err := util.FetchImage(ctx, url)
	if err != nil {
		return nil, unsupported("the image: %v", err)
	}
	img, _, err := pixels.Decode(data)
	if err != nil {
		return nil, unsupported("%v", err)
	}
	b.mu.Lock()
	eng := b.eng
	b.mu.Unlock()
	if eng == nil {
		return nil, fmt.Errorf("the OCR model is closed")
	}
	skip := req.SkipSpecialTokens == nil || *req.SkipSpecialTokens
	r := ocr.Request{Image: img, Prompt: prompt, MaxTokens: req.Budget()}
	if m := req.MMProcessorKwargs; m != nil {
		r.MinPixels, r.MaxPixels = m.MinPixels, m.MaxPixels
	}
	if req.RepetitionPenalty != nil {
		r.RepetitionPenalty = *req.RepetitionPenalty
	}

	// The stream: the whole prefix is decoded after each token and what is
	// new is sent, held back while it ends inside a byte-fallback character
	// (U+FFFD until its last byte arrives). The final decode settles the
	// rest, so the deltas concatenate to exactly the buffered answer.
	var ids []int32
	sent := ""
	send := func(text string) error {
		if !strings.HasPrefix(text, sent) || len(text) == len(sent) {
			return nil
		}
		d := text[len(sent):]
		sent = text
		return emit(api.Delta{Content: d})
	}
	var first time.Duration
	r.OnToken = func(id int32) error {
		if len(ids) == 0 {
			first = time.Since(enter)
		}
		ids = append(ids, id)
		text := eng.Tok.Decode(ids, skip)
		if strings.HasSuffix(text, "�") {
			return nil
		}
		return send(text)
	}
	res, err := eng.Recognize(ctx, r)
	if err != nil {
		return nil, err
	}
	final := eng.Tok.Decode(res.IDs, skip)
	if !strings.HasPrefix(final, sent) {
		// Not reachable with this decoder (a prefix's text is a prefix of
		// the whole's once its byte runs are complete), and a stream that
		// had diverged could not be repaired by another delta.
		return nil, fmt.Errorf("ocr: the streamed text is not a prefix of the answer")
	}
	if err := send(final); err != nil {
		return nil, err
	}
	log.Printf("%socr: %dx%d image, %d prompt + %d tokens (%s): tower %v, prefill %v, decode %v, first token %v",
		logID(ctx), img.W, img.H, res.PromptTokens, len(res.IDs), res.Finish, res.Tower.Round(time.Millisecond),
		res.Prefill.Round(time.Millisecond), res.Decode.Round(time.Millisecond), first.Round(time.Millisecond))
	return &api.CompletionResult{
		FinishReason: res.Finish,
		Usage: api.Usage{
			PromptTokens:     res.PromptTokens,
			CompletionTokens: len(res.IDs),
			TotalTokens:      res.PromptTokens + len(res.IDs),
		},
	}, nil
}
