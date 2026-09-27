package ocr

// Element-level recognition (OCR.md O5): an image and a task prompt in, the
// text out. This is the HF checkpoint alone, and the unit the chat door
// (O6) and the page pipeline (O8) both call.
//
// A request is the host side (Process, BuildPrompt), the tower (GPUTower)
// and the language model (LM): the image rows the tower returns replace the
// prompt's placeholder rows, the prompt is prefilled with its last row's
// logits, and generation is greedy (OCR.md decision 6) until </s> or the
// cap. One request holds the device at a time for now; a page's regions as
// slots of one decode step is O11.
//
// What is *not* here, because the server PaddleOCR talks to does not do it
// either: Spotting's 2x Lanczos upscale of small crops
// (pre_process_for_spotting), the per-label pixel bounds, and the
// post-processing of the text (OTSL to HTML, formula delimiters, repetition
// truncation). Those are PaddleX's pipeline, O8.

import (
	"context"
	"fmt"
	"sync"
	"time"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// DefaultMaxTokens is PaddleX's PADDLEOCR_VL_MAX_NEW_TOKENS.
const DefaultMaxTokens = 8192

// Options size an engine.
type Options struct {
	// MaxPatches is the tower's budget (0: the checkpoint's max_pixels,
	// 5,120). A request above it is refused, not resized.
	MaxPatches int
	// LM sizes the language model; MaxLen bounds prompt + generation.
	LM LMOptions
}

// DefaultOptions fits the checkpoint's largest image and PaddleX's cap.
func DefaultOptions() Options { return Options{LM: DefaultLMOptions()} }

// Engine is PaddleOCR-VL resident on a device.
type Engine struct {
	Tok   *Tokenizer
	tower *GPUTower
	lm    *LM
	o     Options
	mu    sync.Mutex

	// Do, if set, runs each device pass: the tower, the prefill and every
	// decode step. A server sets it to its device lock so another model's
	// work interleaves with a long generation step by step, rather than
	// waiting out a page (backend/ocr.go). Nil runs them directly.
	Do func(func() error) error
}

func (e *Engine) do(fn func() error) error {
	if e.Do != nil {
		return e.Do(fn)
	}
	return fn()
}

// Load stages the tokenizer, the tower and the language model from dir.
func Load(dev *vk.Device, dir string, o Options) (*Engine, error) {
	if o.MaxPatches <= 0 {
		o.MaxPatches = MaxPatches
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	tw, err := LoadTower(dir, -1)
	if err != nil {
		return nil, err
	}
	gt, err := NewGPUTower(dev, tw, o.MaxPatches)
	if err != nil {
		return nil, err
	}
	lm, err := LoadLM(dev, dir, o.LM)
	if err != nil {
		gt.Destroy()
		return nil, err
	}
	return &Engine{Tok: tok, tower: gt, lm: lm, o: o}, nil
}

// Destroy releases the device state.
func (e *Engine) Destroy() {
	e.tower.Destroy()
	e.lm.Destroy()
}

// DeviceBytes is what the engine holds on the device.
func (e *Engine) DeviceBytes() int {
	return e.tower.WeightBytes() + e.tower.ActivationBytes() + e.lm.DeviceBytes()
}

// Request is one element: an image and the text after it (one of Tasks'
// values, or any prompt).
type Request struct {
	Image  *pixels.RGB
	Prompt string
	// MinPixels and MaxPixels bound the resize (0: the checkpoint's).
	MinPixels, MaxPixels int
	// MaxTokens caps generation (0: DefaultMaxTokens), and is lowered to
	// what the cache has left after the prompt.
	MaxTokens int
	// RepetitionPenalty is vLLM's (and HF's): every token already in the
	// prompt or the output has its logit divided by it when positive and
	// multiplied by it when not. 0 or 1 is off. PaddleX sends it only when
	// its caller sets one.
	RepetitionPenalty float64
	// OnToken, if set, sees each generated id as it is chosen (</s> too).
	// An error from it stops the generation and is Recognize's.
	OnToken func(id int32) error
}

// Result is the generation and where the time went.
type Result struct {
	IDs  []int32 // generated, </s> included when it ended there
	Text string  // decoded, specials skipped
	// Finish is "stop" (</s>) or "length" (the cap).
	Finish       string
	PromptTokens int
	GridH, GridW int // the patch grid

	Process, Tower, Prefill, Decode time.Duration
}

// Recognize runs one request to the end.
func (e *Engine) Recognize(ctx context.Context, req Request) (*Result, error) {
	if req.Image == nil {
		return nil, fmt.Errorf("ocr: no image")
	}
	minPx, maxPx := req.MinPixels, req.MaxPixels
	if minPx <= 0 {
		minPx = DefaultMinPixels
	}
	if maxPx <= 0 {
		maxPx = DefaultMaxPixels
	}
	if minPx > maxPx {
		return nil, fmt.Errorf("ocr: min_pixels %d over max_pixels %d", minPx, maxPx)
	}
	t0 := time.Now()
	im, err := Process(req.Image, minPx, maxPx)
	if err != nil {
		return nil, err
	}
	if n := im.GridH * im.GridW; n > e.o.MaxPatches {
		return nil, fmt.Errorf("ocr: a %dx%d image is %d patches; this engine is staged for %d (max_pixels %d)",
			req.Image.W, req.Image.H, n, e.o.MaxPatches, e.o.MaxPatches*PatchSize*PatchSize)
	}
	p, err := e.Tok.BuildPrompt(req.Prompt, im.MergedH(), im.MergedW())
	if err != nil {
		return nil, err
	}
	maxTok := req.MaxTokens
	if maxTok <= 0 {
		maxTok = DefaultMaxTokens
	}
	if room := e.o.LM.MaxLen - len(p.IDs); maxTok > room {
		if room <= 0 {
			return nil, fmt.Errorf("ocr: a %d-token prompt; the cache holds %d", len(p.IDs), e.o.LM.MaxLen)
		}
		maxTok = room
	}
	res := &Result{PromptTokens: len(p.IDs), GridH: im.GridH, GridW: im.GridW, Process: time.Since(t0)}

	e.mu.Lock()
	defer e.mu.Unlock()
	t0 = time.Now()
	var vis *qwen.Mat
	err = e.do(func() error {
		var err error
		vis, err = e.tower.Forward(ctx, im)
		return err
	})
	if err != nil {
		return nil, err
	}
	res.Tower = time.Since(t0)
	if vis.Rows != p.ImageLen {
		return nil, fmt.Errorf("ocr: the tower gave %d rows for %d image tokens", vis.Rows, p.ImageLen)
	}

	rows := make([]Row, len(p.IDs))
	for i, id := range p.IDs {
		rows[i] = Row{ID: id, Pos: i, Rope: p.Pos[i]}
		if j := i - p.ImageAt; p.ImageAt >= 0 && j >= 0 && j < p.ImageLen {
			rows[i].Embed = vis.Row(j)
		}
	}
	t0 = time.Now()
	var logits []float32
	for lo := 0; lo < len(rows); lo += e.o.LM.Rows {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		hi := min(lo+e.o.LM.Rows, len(rows))
		n := 0
		if hi == len(rows) {
			n = 1
		}
		err := e.do(func() error {
			lg, _, err := e.lm.Pass(rows[lo:hi], n)
			if n > 0 && err == nil {
				logits = lg[0]
			}
			return err
		})
		if err != nil {
			return nil, err
		}
	}
	res.Prefill = time.Since(t0)

	t0 = time.Now()
	eos := e.Tok.EOS()
	res.Finish = "length"
	var seen []bool
	penalty := float32(req.RepetitionPenalty)
	if penalty > 0 && penalty != 1 {
		seen = make([]bool, e.lm.Vocab())
		for _, id := range p.IDs {
			seen[id] = true
		}
	}
	for {
		if seen != nil {
			for i, s := range seen {
				if !s {
					continue
				}
				if logits[i] > 0 {
					logits[i] /= penalty
				} else {
					logits[i] *= penalty
				}
			}
		}
		id := int32(argmaxF32(logits))
		res.IDs = append(res.IDs, id)
		if seen != nil {
			seen[id] = true
		}
		if req.OnToken != nil {
			if err := req.OnToken(id); err != nil {
				return nil, err
			}
		}
		if id == eos {
			res.Finish = "stop"
			break
		}
		if len(res.IDs) >= maxTok {
			break
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		j := len(res.IDs) - 1
		pos := p.Next + int32(j)
		err := e.do(func() error {
			lg, _, err := e.lm.Pass([]Row{{ID: id, Pos: len(p.IDs) + j, Rope: [3]int32{pos, pos, pos}}}, 1)
			if err == nil {
				logits = lg[0]
			}
			return err
		})
		if err != nil {
			return nil, err
		}
	}
	res.Decode = time.Since(t0)
	res.Text = e.Tok.Decode(res.IDs, true)
	return res, nil
}

// argmaxF32 is the first index of the largest value: greedy's tie rule,
// torch.argmax's.
func argmaxF32(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}
