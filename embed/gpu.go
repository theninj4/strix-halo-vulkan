package embed

import (
	"fmt"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// GPU runs the model on the device.
//
// The transformer is `qwen.GPUEncoder` — PIPELINE.md stage 5c's Vulkan text
// encoder — with every layer loaded rather than NumLayers-1, and this type is
// the three things that sit outside it: the final norm, the pooling and the
// normalisation. All three are host arithmetic over **one row of 1024
// floats**, which is why there is no shader here. A dispatch for them would
// cost more in launch and barrier (about 1.5 µs, `overhead`) than the 1024
// multiply-adds do on a core, and it would need a read-back anyway.
//
// The read-back is the one thing done carefully. The activation arena reads
// at 0.2 GB/s, so pulling the whole [tokens, 1024] hidden state back for a
// vector that is one row of it would cost 0.6 ms at 27 tokens and 4.4 ms at
// 200 — several times the forward pass. ReadRow takes the row.
type GPU struct {
	Cfg *qwen.Config
	Enc *qwen.GPUEncoder
	Tok *Tokenizer

	// AutoPlan re-plans every run for its own length with PlanFor. It is on
	// by default and it is free -- every rung reads the same staged weight,
	// so this changes which pipeline a dispatch names and nothing else. Set
	// it false to hold a plan across lengths, which is what the ladder in
	// cmd/embed does.
	AutoPlan bool

	norm *qwen.RMSNorm
}

// PlanFor is the measured schedule for *this* model: which GEMM rung wins at
// a given sequence length.
//
// It exists because `qwen.PlanFor` is the wrong table here. That one was
// fitted on Z-Image's Qwen3-4B, where hidden is 2560 and the FFN 9728; at
// hidden 1024 every projection is a quarter the width, a tile of the shape
// that filled the machine there launches a quarter the workgroups here, and
// the ladder moves. Measured at 31, 122, 252, 382 and 602 tokens,
// `qwen.PlanFor` is **1.13-1.23x off the best rung at every one of them** and
// never wins; what wins is a tile that grows with the run:
//
//	tokens   winner              vs qwen.PlanFor
//	31       reg16x64_bt16_k8    1.22x
//	122      reg16x64_bt16_k8    1.20x
//	252      reg32x64_bt16       1.20x
//	382      reg64_bt16          1.13x
//	602      reg64_bt16          1.00x   (the tables agree at last)
//
// The boundaries are placed between measured points rather than on them.
func PlanFor(tokens int) qwen.GEMMPlan {
	switch {
	case tokens <= 187:
		return qwen.UniformGEMMPlan(qwen.GEMMReg16x64K8)
	case tokens <= 317:
		return qwen.UniformGEMMPlan(qwen.GEMMReg32x64)
	default:
		return qwen.UniformGEMMPlan(qwen.GEMMReg64)
	}
}

// NewGPU loads the checkpoint onto the device, sized for runs of at most
// maxTokens. The weights are staged one layer at a time and the safetensors
// set is closed before this returns: what stays resident is 0.88 GB of fp16
// bank on the device plus the fp32 embedding table on the host.
func NewGPU(dev *vk.Device, dir string, maxTokens int) (*GPU, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	norm, err := LoadFinalNorm(set, cfg)
	if err != nil {
		return nil, err
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	// The plan is named rather than left to qwen.PlanFor, which turns the
	// encoder's own AutoPlan off; PlanFor above replaces it per run.
	enc, err := qwen.NewGPUEncoder(dev, set, cfg, cfg.NumLayers, maxTokens, PlanFor(maxTokens))
	if err != nil {
		return nil, err
	}
	// A layer a command buffer rather than qwen's eight dispatches: a submit
	// and its fence are ~40 us, a lone text makes 84 of them at eight, and
	// this is 11.9 -> 9.6 ms for a 27-token query (TestGPUBatchLadder). A
	// layer of this model is ~4 ms of GPU at 2048 rows, well inside the
	// watchdog that eight was chosen for on the 4B.
	enc.PerSubmit = 1 << 10
	return &GPU{Cfg: cfg, Enc: enc, Tok: tok, norm: norm, AutoPlan: true}, nil
}

// Destroy releases the device objects.
func (g *GPU) Destroy() {
	if g.Enc != nil {
		g.Enc.Destroy()
	}
}

// Tokens is the longest run this instance was built for.
func (g *GPU) Tokens() int { return g.Enc.Tokens() }

// EmbedIDs runs the stack and returns the unit vector. The ids must carry the
// appended end-of-text token, which Tokenizer.Encode puts there.
func (g *GPU) EmbedIDs(ids []int32) ([]float32, error) {
	if err := g.plan(len(ids)); err != nil {
		return nil, err
	}
	if err := g.Enc.RunIDs(ids); err != nil {
		return nil, err
	}
	row := g.Enc.ReadRow(g.Enc.TensorX(), len(ids)-1, g.Cfg.HiddenSize)
	return Normalize(g.normRow(row)), nil
}

// EmbedBatch runs several texts in one pass (E7) and returns their unit
// vectors in order. The projections read the weights once for all of them and
// attention runs per text (qwen.GPUEncoder.RunBatch), so a vector is what
// EmbedIDs returns for the same text under the same plan. Fits says whether a
// set of lengths goes in one call.
func (g *GPU) EmbedBatch(seqs [][]int32) ([][]float32, error) {
	rows := 0
	for _, s := range seqs {
		rows += len(s)
	}
	if err := g.plan(rows); err != nil {
		return nil, err
	}
	offs, err := g.Enc.RunBatch(seqs)
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(seqs))
	for i, s := range seqs {
		row := g.Enc.ReadRow(g.Enc.TensorX(), offs[i]+len(s)-1, g.Cfg.HiddenSize)
		out[i] = Normalize(g.normRow(row))
	}
	return out, nil
}

// Fits reports whether texts of these token counts fit one EmbedBatch.
func (g *GPU) Fits(lens []int) bool { return g.Enc.BatchFits(lens) }

// Embed is the whole thing: text to a unit vector.
func (g *GPU) Embed(text string) ([]float32, error) {
	ids, err := g.Tok.Encode(text)
	if err != nil {
		return nil, err
	}
	return g.EmbedIDs(ids)
}

// Hidden runs the stack and reads the *whole* normed hidden state back,
// which is what transformers calls last_hidden_state. It is the validation
// path -- it pays the arena's slow read for every row -- and EmbedIDs is the
// one a caller wants.
func (g *GPU) Hidden(ids []int32) (*qwen.Mat, error) {
	if err := g.plan(len(ids)); err != nil {
		return nil, err
	}
	x, err := g.Enc.Forward(ids)
	if err != nil {
		return nil, err
	}
	if err := g.norm.ApplyInPlace(x); err != nil {
		return nil, fmt.Errorf("embed: final norm: %w", err)
	}
	return x, nil
}

// plan installs the schedule for a run of the given length, unless the
// caller has taken the wheel.
func (g *GPU) plan(tokens int) error {
	if !g.AutoPlan {
		return nil
	}
	return g.Enc.SetPlan(PlanFor(tokens))
}

// normRow applies the final norm to one row, in place.
func (g *GPU) normRow(row []float32) []float32 {
	m := &qwen.Mat{Rows: 1, Cols: len(row), Data: row}
	if err := g.norm.ApplyInPlace(m); err != nil {
		// Unreachable: the row is HiddenSize wide because ReadRow was told
		// to read that many, and the norm weight is checked against
		// HiddenSize at load.
		panic(err)
	}
	return row
}
