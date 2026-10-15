package textenc

import (
	"fmt"

	qtextenc "strix-halo-vulkan/qimage/textenc"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// GPU is Encoder on the device: zimage/qwen's GPUEncoder over the first 32
// layers, the same one pad row appended to the real tokens, its attention
// the PAD_LAST build so that row does not see its own key (as the
// reference's padding mask keeps it from doing), and the eight taps read
// back between layers.
type GPU struct {
	Enc *qwen.GPUEncoder
	Tok *tokenizer.Tokenizer
	cfg *qwen.Config
	max int
}

// NewGPU stages the encoder for prompts of up to maxTokens real tokens.
func NewGPU(dev *vk.Device, dir string, maxTokens int) (*GPU, error) {
	cfg, err := qtextenc.LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	enc, err := qwen.NewGPUEncoder(dev, set, cfg, Taps[len(Taps)-1], maxTokens+1, nil)
	if err != nil {
		return nil, err
	}
	if err := enc.UsePadLastAttention(); err != nil {
		enc.Destroy()
		return nil, err
	}
	tok, err := tokenizer.Load(dir)
	if err != nil {
		enc.Destroy()
		return nil, err
	}
	return &GPU{Enc: enc, Tok: tok, cfg: cfg, max: maxTokens}, nil
}

// Destroy releases the device encoder.
func (g *GPU) Destroy() { g.Enc.Destroy() }

// IDs tokenizes a prompt as Encoder.IDs does.
func (g *GPU) IDs(prompt string) ([]int32, error) {
	e := Encoder{Tok: g.Tok}
	return e.IDs(prompt)
}

// Encode is Encoder.Encode on the device: [Window(n), 8*hidden].
func (g *GPU) Encode(ids []int32) (*qwen.Mat, error) {
	n := len(ids)
	if n == 0 || n > g.max {
		return nil, fmt.Errorf("textenc: %d tokens, staged for 1..%d", n, g.max)
	}
	x, err := g.Enc.Embeddings(append(append([]int32(nil), ids...), PadID))
	if err != nil {
		return nil, err
	}
	if err := g.Enc.UploadEmbeds(x); err != nil {
		return nil, err
	}
	if err := g.Enc.SetRoPE(padRoPE(g.cfg.HeadDim, n, g.cfg.RopeTheta)); err != nil {
		return nil, err
	}
	if err := g.Enc.SetPlan(qwen.PlanFor(n + 1)); err != nil {
		return nil, err
	}
	// hidden_states[k] is layer k-1's output.
	h := g.cfg.HiddenSize
	taps := map[int]int{}
	for j, k := range Taps {
		taps[k-1] = j
	}
	out := qwen.NewMat(Window(n), len(Taps)*h)
	err = g.Enc.RunHooked(func(layer int) error {
		j, ok := taps[layer]
		if !ok {
			return nil
		}
		hs := g.Enc.Read(g.Enc.TensorX(), h)
		for r := 0; r < out.Rows; r++ {
			copy(out.Row(r)[j*h:(j+1)*h], hs.Row(min(r, n)))
		}
		return nil
	})
	return out, err
}
