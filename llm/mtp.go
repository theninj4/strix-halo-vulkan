package llm

// The MTP draft head — `blk.48` of the separate 2.78 GB checkpoint (LLM.md
// P5a, research/p5-mtp-rollback.md).
//
// **This is a measuring instrument, not the product path.** Its whole job is
// to answer the one question P5's design pass left open: what fraction of the
// trunk's own next token the draft head predicts correctly. That number, and
// nothing else, decides whether the R-row decode kernels of P5b are worth
// building. So the wiring below is arranged to be *checkable* rather than
// fast: the four `nextn` tensors are evaluated on the host, one draft step is
// a dozen separate submits, and the trunk's 10240-wide residual makes a round
// trip through a mapped pointer. A product loop would do none of that.
//
// **What the checkpoint says, and it is more than was assumed.**
// `nextn_predict_layers` is 1 and `block_count` 49, so the head is one extra
// layer; `mtp.layer_types` is `["full_attention"]` and `ple_layer_ids` is
// `[2]`, so it carries **no recurrent state and no convolution ring** — its
// entire per-sequence state is a one-layer KV cache, which `SetPast` rewinds
// for free (gpu_attn.go: a cell past the end is masked out of every score).
// And `mtp_use_dedicated_embeddings` is false, so the shard's bundled
// `token_embd` (Q4_K) and `output` (Q6_K) are copies of the trunk's, staged
// at other widths because the shard has to stand alone. This type uses the
// trunk's, which is both correct and 1.27 GB of residency not spent.
//
// **There is no oracle for the wiring, which is new.** D5 — llama.cpp is the
// oracle — has an answer for every other block in this vertical and none for
// this one: `src/models/qwen4exp.cpp` contains no `nextn` symbol, so the
// tensors are never created and `blk.48` is never loaded, and transformers'
// `modeling_qwen4_exp.py` carries
// `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` — it discards them on
// load. The nearest relative that *is* implemented is `qwen3next.cpp`'s
// `graph_mtp`, and it is a model without hyper-connections.
//
// So the wiring here is a **hypothesis with a gate**. `nextn.eh_proj` is
// `[5120, 2560]`, i.e. K = 2*n_embd, the same shape every architecture in
// llama.cpp creates — but `nextn.hnorm` is `[10240]` rather than `[2560]`,
// because this model's residual is four hyper-connection streams. The reading
// that leaves no tensor unexplained is that `hnorm` is `[2560, 4]` — a
// per-stream gamma, exactly as `qwen4exp.cpp` creates `hc_head_norm` with
// `{n_embd, hc}` and `TENSOR_ALLOW_RESHAPE` — and that **`eh_proj` runs once
// per stream**:
//
//	e       = rms(embd(x_{i+1}), enorm)                      [2560]
//	h_s     = rms(h_i[s], hnorm[s])       for s = 0..3       [2560] each
//	res_s   = eh_proj . concat(e, h_s)    for s = 0..3       [2560] each
//
// which produces a fresh `[2560, 4]` hyper-connection state for the layer to
// run on. `blk.48`'s own `hc_attn_*` and `hc_ffn_*` then drive it exactly as
// a trunk layer's do, and `blk.48.nextn.hc_head_{norm,down,up}` is the output
// mixer feeding the lm head — **that half is not a hypothesis**: the shard
// carries no `output_norm` and no `output_hc_*` at all, so the only mixer it
// has for the head is that one.
//
// The gate is the acceptance rate. A correct wiring predicts the trunk's own
// next token often; every wrong one predicts it at about the vocabulary's
// base rate. No numerical oracle is needed to tell those apart, which is why
// P5a comes before P5b and P5c.
//
// **One deviation from the checkpoint, and it is inert at the contexts this
// runs at.** `attention.compress_ratios[48]` is **0**, where every
// full-attention layer of the trunk says 4 — even though `blk.48` ships the
// indexer's `q_proj`/`k_proj` and their norms. Ratio zero is what the trunk's
// *linear* layers say, so the array does not describe this layer, and nothing
// in the checkpoint says what the MTP layer's selection budget is. `Ratio`
// below is therefore taken from the trunk, and the consequence is nil where
// it matters: `selWidth` is `min(top_k + ratio - 1, nKV)` = 2051, so at
// n_ctx <= 2048 the selection names every cell and `AttnGPU` does not
// dispatch it at all (gpu_attn.go's `sparse`). Above that the number would be
// a guess, and `NewMTPHead` says so rather than guessing.

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"strix-halo-vulkan/vk"
)

// MTPHead is `blk.48` staged on the device, plus the four `nextn` tensors on
// the host, plus a borrowed lm head.
type MTPHead struct {
	dev *vk.Device
	m   *Model
	cfg Config

	hc   *HCGPU
	attn *AttnGPU
	moe  *MoEGPU
	move *mover
	// head is the **trunk's**, not this checkpoint's: see the file comment.
	head *HeadGPU

	// The `nextn` block, on the host. `ehProj` is ggml memory order — 2560
	// rows of 5120 values, which is the row-major [out][in] a dot product
	// wants with no transpose.
	enorm, hnorm, ehProj []float32
	eps                  float32

	nEmbd, hcN, wide int
	layer            int

	// past is how many cells this head's own KV cache holds. It is the only
	// state the head carries, and `Rewind` is the whole of its rollback.
	past int
	nKV  int

	// Steps counts draft steps and the two durations split what one costs:
	// `SeedTime` is the host `nextn` block — four 13.1 M-weight matvecs on
	// the CPU — and `GPUTime` the dozen unrecorded submits that are the
	// layer. Both are instrument costs rather than the product path's, and
	// the split is here so the write-up can say which is which instead of
	// quoting one number that is mostly neither.
	Steps    int
	SeedTime time.Duration
	GPUTime  time.Duration
}

// MTPLayer is the block index the head lives at: one past the trunk's last.
func MTPLayer(trunk Config) int { return trunk.NLayer }

// NewMTPHead stages the draft head from its own checkpoint.
//
// `trunkCfg` is the *trunk's* config — the head's shard states
// `block_count = 49` where the trunk states 48, and the layer to stage is the
// trunk's count. `head` is the trunk's staged lm head, borrowed rather than
// restaged; `maxTok` and `nKV` size this head's own arenas and cache.
func NewMTPHead(dev *vk.Device, m *Model, trunkCfg Config, head *HeadGPU, maxTok, nKV int) (*MTPHead, error) {
	if head == nil {
		return nil, fmt.Errorf("llm: the MTP head borrows the trunk's lm head, and none was given")
	}
	l := MTPLayer(trunkCfg)
	if m.Config.NLayer != l+1 {
		return nil, fmt.Errorf("llm: the draft checkpoint states %d blocks against the trunk's %d, so its head is not blk.%d",
			m.Config.NLayer, trunkCfg.NLayer, l)
	}
	if m.Config.NEmbd != trunkCfg.NEmbd || m.Config.HC != trunkCfg.HC {
		return nil, fmt.Errorf("llm: the draft head is %d wide over %d streams, the trunk %d over %d",
			m.Config.NEmbd, m.Config.HC, trunkCfg.NEmbd, trunkCfg.HC)
	}
	g := &MTPHead{
		dev: dev, m: m, cfg: m.Config, head: head, layer: l,
		nEmbd: m.Config.NEmbd, hcN: m.Config.HC, wide: m.Config.HC * m.Config.NEmbd,
		eps: m.Config.RMSEps, nKV: nKV,
	}
	if err := g.stage(dev, trunkCfg, maxTok, nKV); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

func (g *MTPHead) stage(dev *vk.Device, trunkCfg Config, maxTok, nKV int) error {
	m, l := g.m, g.layer
	var err error
	if g.move, err = newMover(dev); err != nil {
		return err
	}

	// The three mixers, in the order a one-layer graph indexes them: 2*0
	// attn, 2*0+1 ffn, 2*nLayer = 2 the head's. The head mixer is
	// `nextn.hc_head_*` and has no inject, because there is no block after
	// it to scatter into — the same shape as the trunk's `output_hc_*`.
	var mixers []HCWeights
	for _, side := range []string{"attn", "ffn"} {
		w, err := m.HCWeights(l, side)
		if err != nil {
			return fmt.Errorf("llm: draft hc %s: %w", side, err)
		}
		mixers = append(mixers, w)
	}
	hw, err := g.nextnMixer()
	if err != nil {
		return err
	}
	mixers = append(mixers, hw)
	// The draft head stages on L8a's int8 bank rather than on
	// `LLM_DENSE_BANK`'s plan: the plan's widths were chosen by 145-chunk
	// perplexity runs over the *trunk*, and the published imatrix has no row
	// for any `blk.48` tensor, so a 4.5-bit stage here would be an
	// uncalibrated fit nobody has graded. It is 1.9 GB either way.
	if g.hc, err = NewHCGPU(dev, m.Config.HCConfig(), maxTok, mixers, HCOpts{Q8: true}); err != nil {
		return fmt.Errorf("llm: draft hc: %w", err)
	}
	mixers = nil
	freeHost()

	acfg, ok, err := m.AttnConfig(l)
	if err != nil {
		return fmt.Errorf("llm: draft attn config: %w", err)
	}
	if !ok {
		return fmt.Errorf("llm: the draft checkpoint has no attention tensors at blk.%d", l)
	}
	// See the file comment: the checkpoint's own ratio for this layer is 0,
	// which is what its *linear* layers say, so it does not describe this
	// one. The trunk's is used and the selection is refused above the
	// context where it would start to bite.
	if acfg.Ratio <= 0 {
		tr, tok, terr := g.trunkRatio(trunkCfg)
		if terr != nil {
			return terr
		}
		if !tok {
			return fmt.Errorf("llm: neither the draft head nor the trunk states a compress ratio")
		}
		acfg.Ratio = tr
		if w := acfg.TopK + acfg.Ratio - 1; nKV > w {
			return fmt.Errorf("llm: the draft head's compress ratio is not in its checkpoint (blk.%d says 0), "+
				"so its QSA selection is only safe where it is the identity: n_ctx %d is past %d",
				l, nKV, w)
		}
	}
	aw, err := m.AttnWeights(l)
	if err != nil {
		return fmt.Errorf("llm: draft attn weights: %w", err)
	}
	if g.attn, err = NewAttnGPU(dev, acfg, maxTok, nKV, []AttnWeights{aw}, true); err != nil {
		return fmt.Errorf("llm: draft attn: %w", err)
	}
	aw = AttnWeights{}
	freeHost()

	mw, err := m.MoEWeights(l)
	if err != nil {
		return fmt.Errorf("llm: draft moe weights: %w", err)
	}
	// The draft's experts stage at the checkpoint's own widths for the same
	// reason its dense half does: `LLM_MOE_BANK`'s transcode is calibrated
	// against the published imatrix, which has no row for any `blk.48`
	// tensor, so an empty plan here is the honest stage rather than an
	// uncalibrated one. `WithMoEBankPlan` overrides the environment, which
	// the trunk in the same process is still reading.
	if g.moe, err = NewMoEGPU(dev, m.MoEConfig(), maxTok, []MoEWeights{mw},
		WithMoEBankPlan(MoEBankPlan{})); err != nil {
		return fmt.Errorf("llm: draft moe: %w", err)
	}
	mw = MoEWeights{}
	freeHost()

	return g.stageNextn()
}

// trunkRatio finds a compress ratio the trunk states, so the draft head's
// missing one is borrowed rather than invented.
func (g *MTPHead) trunkRatio(trunkCfg Config) (int, bool, error) {
	v, ok := g.m.Set.Ints(g.m.Set.Arch() + ".attention.compress_ratios")
	if !ok {
		return 0, false, nil
	}
	for i := 0; i < len(v) && i < trunkCfg.NLayer; i++ {
		if v[i] > 0 {
			return int(v[i]), true, nil
		}
	}
	return 0, false, nil
}

// nextnMixer loads `blk.N.nextn.hc_head_*` as the head mixer.
func (g *MTPHead) nextnMixer() (HCWeights, error) {
	p := fmt.Sprintf("blk.%d.nextn.hc_head_", g.layer)
	w := HCWeights{Name: p}
	var err error
	if w.Norm, err = g.m.F32(p + "norm.weight"); err != nil {
		return w, fmt.Errorf("llm: draft head mixer: %w", err)
	}
	if w.Down, err = g.m.F32(p + "down.weight"); err != nil {
		return w, fmt.Errorf("llm: draft head mixer: %w", err)
	}
	if w.Up, err = g.m.F32(p + "up.weight"); err != nil {
		return w, fmt.Errorf("llm: draft head mixer: %w", err)
	}
	return w, nil
}

// stageNextn dequantises the four tensors the `nextn` block is, and checks
// their shapes against the hypothesis the file comment states. A shape that
// does not fit is an error here rather than a wrong answer later.
func (g *MTPHead) stageNextn() error {
	p := fmt.Sprintf("blk.%d.nextn.", g.layer)
	var err error
	if g.enorm, err = g.m.F32(p + "enorm.weight"); err != nil {
		return fmt.Errorf("llm: draft enorm: %w", err)
	}
	if g.hnorm, err = g.m.F32(p + "hnorm.weight"); err != nil {
		return fmt.Errorf("llm: draft hnorm: %w", err)
	}
	if g.ehProj, err = g.m.F32(p + "eh_proj.weight"); err != nil {
		return fmt.Errorf("llm: draft eh_proj: %w", err)
	}
	switch {
	case len(g.enorm) != g.nEmbd:
		return fmt.Errorf("llm: nextn.enorm is %d values, want n_embd %d", len(g.enorm), g.nEmbd)
	case len(g.hnorm) != g.wide:
		return fmt.Errorf("llm: nextn.hnorm is %d values, want hc*n_embd %d — "+
			"the per-stream reading this head is built on does not hold", len(g.hnorm), g.wide)
	case len(g.ehProj) != 2*g.nEmbd*g.nEmbd:
		return fmt.Errorf("llm: nextn.eh_proj is %d values, want [%d, %d] — "+
			"the once-per-stream reading this head is built on does not hold",
			len(g.ehProj), 2*g.nEmbd, g.nEmbd)
	}
	return nil
}

// Wiring is which reading of the `nextn` block a run uses.
//
// It exists because there is no oracle (see the file comment) and the arms
// are one line apart, so the honest thing is to **measure them against each
// other** rather than to pick one and hope. `Flip` is the concatenation
// order: llama.cpp's `graph_mtp` does `ggml_concat(e_norm, h_norm, dim=0)`,
// which puts the embedding first, and a checkpoint converted from a reference
// that did it the other way would look exactly like a wrong model. The choice
// of *what* `h` is — the wide residual, the trunk's collapsed `result_norm`
// broadcast into the streams, or zero — is the caller's, because it is a fact
// about the trunk and not about this block.
type Wiring struct {
	Name string
	Flip bool
}

// Seed is the `nextn` block, on the host: a wide hidden state for token i and
// the embedding of token i+1, into a fresh wide residual for the draft layer.
//
// `h` is [hc*nEmbd] and `embd` [nEmbd]; the result is [hc*nEmbd].
func (g *MTPHead) Seed(h, embd []float32, w Wiring) ([]float32, error) {
	if len(h) != g.wide {
		return nil, fmt.Errorf("llm: the hidden state is %d values, want %d", len(h), g.wide)
	}
	if len(embd) != g.nEmbd {
		return nil, fmt.Errorf("llm: the embedding is %d values, want %d", len(embd), g.nEmbd)
	}
	n, k := g.nEmbd, 2*g.nEmbd
	// e once, h per stream, and the concatenation is materialised per stream
	// because `matvec` wants one contiguous operand.
	e := make([]float32, n)
	groupedRMSNorm(e, embd, g.enorm, n, 1, g.eps)
	hn := make([]float32, g.wide)
	groupedRMSNorm(hn, h, g.hnorm, n, g.hcN, g.eps)

	out := make([]float32, g.wide)
	x := make([]float32, k)
	for s := 0; s < g.hcN; s++ {
		if w.Flip {
			copy(x, hn[s*n:(s+1)*n])
			copy(x[n:], e)
		} else {
			copy(x, e)
			copy(x[n:], hn[s*n:(s+1)*n])
		}
		parMatvec(out[s*n:(s+1)*n], g.ehProj, x, n, k)
	}
	return out, nil
}

// Broadcast repeats a 2560-wide hidden state into every stream, which is what
// `hc_init` does with the token embedding at the top of the trunk — and the
// competing reading of what the MTP block is handed.
func (g *MTPHead) Broadcast(x []float32) ([]float32, error) {
	if len(x) != g.nEmbd {
		return nil, fmt.Errorf("llm: the hidden state is %d values, want %d", len(x), g.nEmbd)
	}
	out := make([]float32, g.wide)
	for s := 0; s < g.hcN; s++ {
		copy(out[s*g.nEmbd:], x)
	}
	return out, nil
}

// Wide and NEmbd are this head's two widths, for a caller building a hidden
// state to hand it.
func (g *MTPHead) Wide() int  { return g.wide }
func (g *MTPHead) NEmbd() int { return g.nEmbd }

// parMatvec is `matvec` over a pool of goroutines. eh_proj is 13.1 M weights
// and a draft step does four of them, which is 50 ms on one core and under
// three on sixteen — the difference between an experiment that runs and one
// that does not.
func parMatvec(dst, w, x []float32, n, k int) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		matvec(dst, w, x, n, k)
		return
	}
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for c := 0; c < n; c += chunk {
		hi := c + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				row := w[i*k : (i+1)*k]
				var acc float32
				for j, v := range row {
					acc += v * x[j]
				}
				dst[i] = acc
			}
		}(c, hi)
	}
	wg.Wait()
}

// Past is how many cells the draft head's own cache holds, and Rewind moves
// it. That is the whole of this head's rollback: it carries no recurrent
// state and no convolution ring, and a KV cell past the end is masked out of
// every score rather than merely unread.
func (g *MTPHead) Past() int { return g.past }

func (g *MTPHead) Rewind(n int) error {
	if n < 0 || n > g.past {
		return fmt.Errorf("llm: rewind to %d of %d", n, g.past)
	}
	g.past = n
	return nil
}

// Reset starts a fresh sequence.
func (g *MTPHead) Reset() { g.past = 0 }

// Step runs one draft: the trunk's residual after token i, the id of token
// i+1, and the position token i+1 sits at. It returns the logits for token
// i+2 and the draft layer's own wide residual, which is what a second draft
// step chains on in place of the trunk's.
//
// `embd` is the embedding of token i+1 out of the **trunk's** `token_embd`
// (`mtp_use_dedicated_embeddings` is false).
func (g *MTPHead) Step(h, embd []float32, pos int, w Wiring) (logits, res, mixed []float32, err error) {
	if pos < 0 || pos >= g.nKV {
		return nil, nil, nil, fmt.Errorf("llm: draft token at position %d of a %d-cell cache", pos, g.nKV)
	}
	t0 := time.Now()
	seed, err := g.Seed(h, embd, w)
	if err != nil {
		return nil, nil, nil, err
	}
	g.SeedTime += time.Since(t0)
	t0 = time.Now()
	defer func() { g.GPUTime += time.Since(t0) }()
	g.past = pos
	if err := g.hc.Upload(seed, 1); err != nil {
		return nil, nil, nil, err
	}
	if err := g.attn.SetPast(pos); err != nil {
		return nil, nil, nil, err
	}

	// One layer, in the trunk's own order: mix, attention, combine, mix,
	// MoE, combine — then the head mixer and the borrowed projection. The
	// combines are issued rather than held, because P1a's fusion is a
	// scheduling optimisation and this path is not timed.
	if err := g.hc.Run(0, false); err != nil {
		return nil, nil, nil, fmt.Errorf("llm: draft attn mix: %w", err)
	}
	if err := g.attn.Resize(1); err != nil {
		return nil, nil, nil, err
	}
	if err := g.move.Move(g.attn.InPort(), g.hc.MixedPort(), 1); err != nil {
		return nil, nil, nil, err
	}
	if err := g.attn.Run(0); err != nil {
		return nil, nil, nil, fmt.Errorf("llm: draft attn: %w", err)
	}
	if err := g.move.Move(g.hc.BlockOutPort(), g.attn.OutPort(), 1); err != nil {
		return nil, nil, nil, err
	}
	if err := g.hc.RunCombine(0); err != nil {
		return nil, nil, nil, fmt.Errorf("llm: draft attn combine: %w", err)
	}

	if err := g.hc.Run(1, false); err != nil {
		return nil, nil, nil, fmt.Errorf("llm: draft ffn mix: %w", err)
	}
	if err := g.moe.Resize(1); err != nil {
		return nil, nil, nil, err
	}
	if err := g.move.Move(g.moe.InPort(), g.hc.MixedPort(), 1); err != nil {
		return nil, nil, nil, err
	}
	if err := g.moe.Run(0); err != nil {
		return nil, nil, nil, fmt.Errorf("llm: draft moe: %w", err)
	}
	if err := g.move.Move(g.hc.BlockOutPort(), g.moe.OutPort(), 1); err != nil {
		return nil, nil, nil, err
	}
	if err := g.hc.RunCombine(1); err != nil {
		return nil, nil, nil, fmt.Errorf("llm: draft ffn combine: %w", err)
	}

	// The head mixer is this block's own (`nextn.hc_head_*`), staged at
	// index 2*nLayer = 2; the projection is the trunk's.
	res = g.hc.Res()
	if err := g.hc.Run(2, false); err != nil {
		return nil, nil, nil, fmt.Errorf("llm: draft head mixer: %w", err)
	}
	if err := g.head.Resize(1); err != nil {
		return nil, nil, nil, err
	}
	if err := g.move.Move(g.head.InPort(), g.hc.MixedPort(), 1); err != nil {
		return nil, nil, nil, err
	}
	if err := g.head.Run(); err != nil {
		return nil, nil, nil, fmt.Errorf("llm: draft head: %w", err)
	}
	g.past = pos + 1
	g.Steps++
	return g.head.Logits(), res, g.hc.Mixed(), nil
}

// WeightBytes and ActivationBytes are what this head cost in device memory,
// the borrowed lm head excluded.
func (g *MTPHead) WeightBytes() int {
	n := 0
	if g.hc != nil {
		n += g.hc.WeightBytes()
	}
	if g.attn != nil {
		n += g.attn.WeightBytes()
	}
	if g.moe != nil {
		n += g.moe.WeightBytes()
	}
	return n
}

func (g *MTPHead) ActivationBytes() int {
	n := 0
	if g.hc != nil {
		n += g.hc.ActivationBytes()
	}
	if g.attn != nil {
		n += g.attn.ActivationBytes()
	}
	if g.moe != nil {
		n += g.moe.ActivationBytes()
	}
	return n
}

// Destroy releases everything but the borrowed lm head.
func (g *MTPHead) Destroy() {
	if g.moe != nil {
		g.moe.Destroy()
		g.moe = nil
	}
	if g.attn != nil {
		g.attn.Destroy()
		g.attn = nil
	}
	if g.hc != nil {
		g.hc.Destroy()
		g.hc = nil
	}
	if g.move != nil {
		g.move.Destroy()
		g.move = nil
	}
	g.enorm, g.hnorm, g.ehProj = nil, nil, nil
}

// Chain is `depth` draft steps, the first off the trunk's residual and every
// later one off the draft layer's own — which is what a speculative round of
// depth M would run, and the only part of P5 that needs the draft head's KV
// rewound at all.
//
// `h` is the trunk's wide residual after token i, `first` the id of token
// i+1, and `pos` the position token i+1 sits at. The returned ids are the
// head's guesses for positions pos+1, pos+2, … — so guess k is compared
// against whatever the trunk's argmax at position pos+k turns out to be.
//
// The rewind is `Step`'s own: it sets `past` from the position it is given,
// and a KV cell past the end is masked out of every score, so the cells a
// rejected chain left behind are unreadable rather than stale and the next
// round overwrites the ones it needs. That is the whole of this head's
// rollback, and it is why P5a can measure the acceptance rate before P5c
// exists.
func (g *MTPHead) Chain(h []float32, embed *Model, first int32, pos, depth int, w Wiring, collapsed bool) ([]int32, error) {
	out := make([]int32, 0, depth)
	cur, state := first, h
	for k := 0; k < depth; k++ {
		e, err := embed.Embedding(cur)
		if err != nil {
			return nil, fmt.Errorf("llm: token_embd for the draft: %w", err)
		}
		logits, res, mixed, err := g.Step(state, e, pos+k, w)
		if err != nil {
			return nil, err
		}
		cur = Argmax(logits)
		out = append(out, cur)
		// What the next step chains on has to match what the first one was
		// handed: an arm that reads the trunk's collapsed `result_norm`
		// chains on the draft block's own collapsed output, and one that
		// reads the wide residual chains on the wide one.
		if collapsed {
			if state, err = g.Broadcast(mixed); err != nil {
				return nil, err
			}
		} else {
			state = append(state[:0:0], res...)
		}
	}
	return out, nil
}
