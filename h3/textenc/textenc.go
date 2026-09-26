// Package textenc is MiniMax-H3's text conditioning (VIDEO.md M2): the
// hidden state after decoder layer 49 of the Qwen3-VL-32B conditioner,
// *unnormalised*, over the prompt tokenised verbatim — no chat template, no
// special tokens, nothing dropped.
//
// The conditioner is Qwen-Image-2.1's Qwen3-VL at four times the size, so
// this package is only the parts that differ: which layer, and the
// presentation. The transformer itself is zimage/qwen under
// qimage/textenc's nested-config adapter; loading Layers of its 64 layers
// makes the last one's output the conditioning, because zimage/qwen never
// applies a final norm.
package textenc

import (
	"fmt"

	"strix-halo-vulkan/h3/plan"
	qtextenc "strix-halo-vulkan/qimage/textenc"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// Layers is how many decoder layers run: the pipeline reads
// hidden_states[50], the output of layer index 49. Layers 50–63 and the
// language-model head are never loaded.
const Layers = 50

// Width is the conditioning's feature width, the transformer's text_dim.
const Width = 5120

// LoadConfig reads the checkpoint's text_encoder/config.json.
func LoadConfig(dir string) (*qwen.Config, error) { return qtextenc.LoadConfig(dir) }

// EncodePrompt tokenises a t2va prompt: the string itself, as
// `tokenizer(prompt, add_special_tokens=False)` does. The repository's
// tokenizer adds `<d>`/`</d>` for dialogue, which Context-IR prompts use.
func EncodePrompt(tok *tokenizer.Tokenizer, prompt string) ([]int32, error) {
	return tok.Encode(prompt)
}

// Keyframes (fl2va, VIDEO.md M10).
//
// An fl2va presentation is the prompt verbatim *preceded* by one block per
// keyframe, with still no template and no special tokens around it:
//
//	"<Picture 1>: " <|vision_start|> <|image_pad|>×n <|vision_end|> … prompt
//
// The vision block's rows are tagged *video*, not text: that is what the
// transformer's AdaLN keys off (plan.VideoTag), and the label's rows stay
// text. The conditioner sees the keyframe exactly as a Qwen-Image edit
// reference is seen — the tower's merged rows scattered into the pads,
// deepstack added after layers 0–2, 3-D positions — so this is only the
// presentation, and qimage/textenc's EditPrompt carries the rest.

// Presentation is a tokenized fl2va presentation: the conditioner's input,
// and the per-row modality tag the transformer reads for it.
type Presentation struct {
	*qtextenc.EditPrompt
	Tags []int32
}

// ImageGrid is a keyframe's patch grid on an h × w canvas. The canvas is a
// multiple of 32 and inside the processor's pixel bounds, so the
// processor's smart_resize leaves it as it is and the grid is the canvas
// over the patch.
func ImageGrid(h, w, patch int) qtextenc.Grid {
	return qtextenc.Grid{T: 1, H: h / patch, W: w / patch}
}

// NewPresentation builds the fl2va presentation for len(grids) keyframes,
// in packed order (first, then last).
func NewPresentation(tok *tokenizer.Tokenizer, prompt string, grids []qtextenc.Grid, merge int) (*Presentation, error) {
	ids := func(s string) (int32, error) {
		id, ok := tok.ID(s)
		if !ok {
			return 0, fmt.Errorf("textenc: the tokenizer has no %s", s)
		}
		return id, nil
	}
	start, err := ids("<|vision_start|>")
	if err != nil {
		return nil, err
	}
	end, err := ids("<|vision_end|>")
	if err != nil {
		return nil, err
	}
	pad, err := ids(qtextenc.ImagePad)
	if err != nil {
		return nil, err
	}
	var seq, tags []int32
	for i, g := range grids {
		label, err := tok.Encode(fmt.Sprintf("<Picture %d>: ", i+1))
		if err != nil {
			return nil, err
		}
		seq = append(seq, label...)
		for range label {
			tags = append(tags, plan.TextTag)
		}
		n := g.Tokens(merge)
		seq = append(seq, start)
		for j := 0; j < n; j++ {
			seq = append(seq, pad)
		}
		seq = append(seq, end)
		for j := 0; j < n+2; j++ {
			tags = append(tags, plan.VideoTag)
		}
	}
	p, err := tok.Encode(prompt)
	if err != nil {
		return nil, err
	}
	seq = append(seq, p...)
	for range p {
		tags = append(tags, plan.TextTag)
	}
	ep, err := qtextenc.NewPromptIDs(seq, grids, merge, pad)
	if err != nil {
		return nil, err
	}
	return &Presentation{EditPrompt: ep, Tags: tags}, nil
}

// VisionPixels is the processor's view of one RGB keyframe, [H·W·3] bytes
// to [3, H, W] planes: Qwen2VLImageProcessorFast fuses the 1/255 rescale
// into its mean and std (both 0.5), so each value is (v − 127.5) / 127.5 in
// float32. qimage/vision's Patchify takes it from there.
func VisionPixels(rgb []byte, h, w int) ([]float32, error) {
	if len(rgb) != h*w*3 {
		return nil, fmt.Errorf("textenc: %d bytes for a %dx%d RGB frame", len(rgb), w, h)
	}
	out := make([]float32, 3*h*w)
	for i := 0; i < h*w; i++ {
		for c := 0; c < 3; c++ {
			out[c*h*w+i] = (float32(rgb[i*3+c]) - 127.5) / 127.5
		}
	}
	return out, nil
}
