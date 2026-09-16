package llm

// The graph (LLM.md L6b): the five blocks in llama.cpp's own order, over one
// resident model, from a token list to logits.
//
// L6a put every weight on the device — 84.20 GB in 68 buffers — but nothing
// had ever run two blocks in sequence. Each of them was built against the
// oracle in isolation: llama.cpp's own value for the tensor it reads went in,
// and its value for the tensor it writes came out. This file is where the
// inputs stop being the reference's and start being ours, which is the only
// way the thing that has been built since L2 is a model rather than five
// kernels.
//
// The order is `llama_model_qwen4exp::graph::graph`, and it is not the
// obvious one:
//
//	res = repeat(embed(ids), hc)            "hc_init"
//	for each layer:
//	    if it is the PLE layer: res += ple(res, ngram_embd)
//	    mixed, inject = hc_mix(res, attn)   the layer's *attention* mixer
//	    out = deltanet(mixed) | attention(mixed)
//	    res = hc_combine(res, out, inject)  "hc_combine-L"
//	    mixed, inject = hc_mix(res, ffn)
//	    out = moe(mixed)                    "ffn_out-L"
//	    res = hc_combine(res, out, inject)  "l_last-L"
//	norm = hc_mix(res[last], head)          "result_norm" -- there is no
//	logits = output * norm                  separate output norm
//
// Two things about that are worth saying out loud. The final mixer and the
// head run on **one row**: llama.cpp's `inp_out_ids` drops every token but
// the last before them, and the trace agrees — `result_output` is one row of
// 248320 floats for a seven-token prompt. And the residual never leaves the
// hyper-connection block's arena: `llm_hc_combine.comp` updates it in place,
// so what crosses a block boundary here is the 2560-wide `hc_mixed` going
// out and the 2560-wide block output coming back, not the 10240-wide stream.
//
// **The glue is a host round trip, and this is the first cut.** Every block
// owns its own arenas, so `hc_mixed` is read out of one fp32 arena, narrowed
// to halves and written into another — on this machine that is a memcpy
// between two mapped pointers of the same physical RAM, but it is a memcpy
// the reference does not do. L6b measures it rather than hiding it; fusing it
// away is a shared-arena change to five files and belongs to whatever comes
// after the number.

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"time"

	"strix-halo-vulkan/vk"
)

// GraphOpts are the choices that change what the graph allocates.
type GraphOpts struct {
	// MaxTokens is the longest prompt the arenas hold. Every block is sized
	// for it, and the MoE block's 0.87 GB of arenas scale with it.
	MaxTokens int
	// NKV is the attention cache's cell count. Zero takes the reference's
	// own rounding, MaxTokens up to 256 cells, which is what the 7-token
	// trace was dumped at.
	NKV int
	// Layers truncates the model to its first N layers. It is not a model —
	// the last mixer and the head still run — but it is how a graph gets
	// staged in seconds instead of 33 of them while the order is being
	// checked, and how a block's contribution is isolated by leaving the
	// ones after it out.
	Layers int
	// NoHead leaves `output.weight` unstaged, which saves 1.27 GB and 20
	// seconds for a caller that wants `result_norm` and not logits.
	NoHead bool
}

// blockKind says which sublayer a layer's attention half is, and where in its
// block's staged list it sits.
type blockKind struct {
	attn bool // false: gated DeltaNet
	idx  int
}

// Graph is the whole model on the device: the five blocks, the head, and the
// order that makes them one forward pass.
type Graph struct {
	m   *Model
	cfg Config

	hc   *HCGPU
	ple  *PLEGPU
	dn   *DeltaNetGPU
	attn *AttnGPU
	moe  *MoEGPU
	head *HeadGPU
	move *mover

	pleCfg PLEConfig
	hasPLE bool

	kinds  []blockKind
	nLayer int
	maxTok int
	nKV    int

	// The sequence, which is what L7 added. `past` is how many tokens of it
	// are already behind the graph — cells in the attention cache, positions
	// behind the PLE convolution, and tokens folded into every DeltaNet
	// layer's recurrent state — and `ids` is the whole of it, because the
	// n-gram gather is a trigram and a continuing run's first token still has
	// two predecessors.
	past int
	ids  []int32

	// Staged is what each block cost to put on the device, in the order they
	// were staged, for the caller that wants to print a plan.
	Staged []StagedBlock

	// Stats accumulates across every Prefill since ResetStats.
	Stats GraphStats

	// rec is the pass being recorded, installed on every block for the
	// duration of one forward pass and torn down by `flush` (record.go). It
	// is what makes a pass **one command buffer** rather than ~490: the
	// blocks' Run methods append to it instead of submitting.
	rec *recorder
}

// record installs the recorder on every block, so that everything from here
// to the matching `flush` is recorded rather than submitted. It is idempotent
// per pass and paired with `flush` on every return path, which is why the
// callers are the three entry points and not the layer loop.
func (g *Graph) record() {
	if g.rec == nil {
		g.rec = &recorder{}
	}
	g.hc.rec = g.rec
	g.move.rec = g.rec
	if g.ple != nil {
		g.ple.rec = g.rec
	}
	if g.dn != nil {
		g.dn.rec = g.rec
	}
	if g.attn != nil {
		g.attn.rec = g.rec
	}
	if g.moe != nil {
		g.moe.rec = g.rec
	}
	if g.head != nil {
		g.head.rec = g.rec
	}
}

// flush submits everything recorded and folds the GPU time each block's
// dispatches took into Stats. Every block goes back to submitting for itself
// afterwards, because the per-block benchmarks and the tests drive them
// directly and must not depend on a graph having been here.
//
// Nothing may read a device arena between `record` and here: the dispatches
// have not run. That is the one rule this arrangement adds, and the three
// entry points below are written around it — every `Mixed()`, `Logits()` and
// `Res()` is after the flush.
func (g *Graph) flush() error {
	if g.rec == nil {
		return nil
	}
	n := g.rec.len()
	byOwner, total, err := g.rec.submit()
	g.rec = nil
	g.hc.rec, g.move.rec = nil, nil
	if g.ple != nil {
		g.ple.rec = nil
	}
	if g.dn != nil {
		g.dn.rec = nil
	}
	if g.attn != nil {
		g.attn.rec = nil
	}
	if g.moe != nil {
		g.moe.rec = nil
	}
	if g.head != nil {
		g.head.rec = nil
	}
	if err != nil {
		return err
	}
	g.Stats.Dispatches += n
	g.Stats.GPU += total
	g.Stats.HC += byOwner[ownHC]
	g.Stats.PLE += byOwner[ownPLE]
	g.Stats.DeltaNet += byOwner[ownDN]
	g.Stats.Attn += byOwner[ownAttn]
	g.Stats.MoE += byOwner[ownMoE]
	g.Stats.Head += byOwner[ownHead]
	g.Stats.Move += byOwner[ownMove]
	return nil
}

// PinSchedule fixes the hyper-connection block's kernel choice so that the
// graph runs the same arithmetic whatever the chunk length, and puts it back
// on the measured schedule when off.
//
// It exists for one equality and it is the equality L7b is about. Every rung
// of every ladder in this vertical used to be **bit-exact** against its
// siblings — they change how the work is tiled and never how a sum is
// associated — so "a prompt in chunks is the prompt whole" could be asserted
// to the last place even though a one-token chunk plans different kernels
// than a 128-token one. L7d's split-K GEMV is the first rung that is not:
// dividing a 10240-long dot product 160 ways and adding the pieces back is a
// different association of the same products, and at one token it is the rung
// the schedule picks. So the exactness claim is now conditional on the
// schedule, and this is what lets a test say which half it is testing.
func (g *Graph) PinSchedule(on bool) error {
	if !on {
		g.hc.AutoPlan()
		return nil
	}
	return g.hc.SetPlan(HCDownM1, HCUpM4)
}

// ResetStats clears the accumulated timings.
func (g *Graph) ResetStats() { g.Stats = GraphStats{} }

// since adds the elapsed time to one of the stat fields and returns now, so a
// sequence of phases is a chain of one-line calls rather than a pile of
// start/stop pairs.
// blk routes a host interval to a block's stat field — except while a pass is
// being recorded, when the host is only building push constants and the
// block's real cost comes back from the GPU marks (record.go). Then the
// interval is what recording it cost, which is glue.
func (g *Graph) blk(dst *time.Duration, t0 time.Time) time.Time {
	if g.rec != nil {
		return since(&g.Stats.Glue, t0)
	}
	return since(dst, t0)
}

func since(dst *time.Duration, t0 time.Time) time.Time {
	now := time.Now()
	*dst += now.Sub(t0)
	return now
}

// GraphStats is where a prefill's wall clock went.
//
// The split that matters here is **block against glue**. Every block in this
// vertical owns its own arenas, so `hc_mixed` leaves one fp32 arena, is
// narrowed to halves on the host and is written into another — a memcpy
// between two mapped pointers of the same physical RAM on this machine, but
// one llama.cpp does not do. 96 of those a graph, each 2560 wide, plus the
// block outputs coming back. Fusing them away is a shared-arena change to
// five files; L6b's job is to say what they cost, not to hide them.
type GraphStats struct {
	Total time.Duration
	// Per block, the wall clock around its dispatches.
	HC, PLE, DeltaNet, Attn, MoE, Head time.Duration
	// The host side: the two gathers, and whatever still crosses a block
	// boundary through host memory.
	Gather, Glue time.Duration
	// Move is the arena-to-arena moves, on the device (L6c).
	Move time.Duration
	// Runs is how many prefills these totals cover.
	Runs int
	// GPU is what the pass's one command buffer took end to end, and
	// Dispatches how many dispatches went into it (L7d). The per-block rows
	// above are GPU time inside it and so sum to a little less than GPU
	// itself — the difference is what the barriers between dispatches cost.
	GPU        time.Duration
	Dispatches int
}

// Blocks is the time inside the five blocks and the head.
func (s GraphStats) Blocks() time.Duration {
	return s.HC + s.PLE + s.DeltaNet + s.Attn + s.MoE + s.Head
}

// StagedBlock is one row of the staging plan.
type StagedBlock struct {
	Name    string
	Layers  int
	Buffers int
	Weights int
	Arenas  int
	Elapsed time.Duration
}

// NewGraph stages the whole model and builds the order.
//
// The staging order is L6a-5's and for L6a-5's reason: the dense blocks are
// dequantised to f32 on the host on the way to the device, and the 36
// DeltaNet layers are 8.35 GB of transient floats at once, so they go first
// and their host copies are dropped between them. The 77 GB expert bank needs
// no host floats at all and goes last, which makes the peak the resident
// model rather than the resident model plus a dequantisation.
func NewGraph(dev *vk.Device, m *Model, opts GraphOpts) (*Graph, error) {
	c := m.Config
	if opts.MaxTokens <= 0 {
		return nil, fmt.Errorf("llm: graph MaxTokens is %d", opts.MaxTokens)
	}
	nLayer := c.NLayer
	if opts.Layers > 0 && opts.Layers < nLayer {
		nLayer = opts.Layers
	}
	nKV := opts.NKV
	if nKV < roundUpInt(opts.MaxTokens, 256) {
		nKV = roundUpInt(opts.MaxTokens, 256)
	}
	g := &Graph{m: m, cfg: c, nLayer: nLayer, maxTok: opts.MaxTokens, nKV: nKV}

	if err := g.stage(dev, opts); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// stage puts every block on the device, in the order that keeps the peak down.
func (g *Graph) stage(dev *vk.Device, opts GraphOpts) error {
	m, c := g.m, g.cfg
	var err error
	if g.move, err = newMover(dev); err != nil {
		return err
	}
	mark := func(name string, layers, buffers, weights, arenas int, start time.Time) {
		g.Staged = append(g.Staged, StagedBlock{name, layers, buffers, weights, arenas, time.Since(start)})
	}

	// 1. The mixers: two a layer and the head's. Indexed 2L (attn), 2L+1
	//    (ffn), and 2*nLayer for the head — which is why the head is staged
	//    with them and not as a block of its own.
	start := time.Now()
	var mixers []HCWeights
	for l := 0; l < g.nLayer; l++ {
		for _, side := range []string{"attn", "ffn"} {
			w, err := m.HCWeights(l, side)
			if err != nil {
				return fmt.Errorf("llm: hc layer %d %s: %w", l, side, err)
			}
			mixers = append(mixers, w)
		}
	}
	head, err2 := m.HCWeights(-1, "")
	if err2 != nil {
		return fmt.Errorf("llm: hc head mixer: %w", err2)
	}
	mixers = append(mixers, head)
	if g.hc, err = NewHCGPU(dev, c.HCConfig(), g.maxTok, mixers, HCOpts{}); err != nil {
		return fmt.Errorf("llm: hc: %w", err)
	}
	mark("hyper-conn", len(mixers), g.hc.Buffers(), g.hc.WeightBytes(), g.hc.ActivationBytes(), start)
	mixers = nil
	freeHost()

	// 2. The PLE n-gram block, if this checkpoint has one. It is a single
	//    layer — layer 1 — and the 28.80 GB table it gathers from stays
	//    mmap'd (D2).
	pleCfg, ok, err := m.PLEConfig()
	if err != nil {
		return fmt.Errorf("llm: ple config: %w", err)
	}
	if ok && pleRuns(pleCfg, g.nLayer) {
		start = time.Now()
		w, err := m.PLEWeights(pleCfg.Layers[0])
		if err != nil {
			return fmt.Errorf("llm: ple layer %d: %w", pleCfg.Layers[0], err)
		}
		if g.ple, err = NewPLEGPU(dev, pleCfg, g.maxTok, w, PLEOpts{}); err != nil {
			return fmt.Errorf("llm: ple: %w", err)
		}
		g.pleCfg, g.hasPLE = pleCfg, true
		mark("ple n-gram", len(pleCfg.Layers), g.ple.Buffers(), g.ple.WeightBytes(), g.ple.ActivationBytes(), start)
		freeHost()
	}

	// 3 and 4. The two kinds of attention half, and the map from a layer to
	//    which of them it is. The interval is four, so 36 layers are the
	//    gated DeltaNet and 12 are full attention with the QSA indexer.
	g.kinds = make([]blockKind, g.nLayer)
	start = time.Now()
	var dnCfg DeltaNetConfig
	var dns []DeltaNetWeights
	for l := 0; l < g.nLayer; l++ {
		cfg, ok, err := m.DeltaNetConfig(l)
		if err != nil {
			return fmt.Errorf("llm: deltanet config %d: %w", l, err)
		}
		if !ok {
			continue
		}
		w, err := m.DeltaNetWeights(l)
		if err != nil {
			return fmt.Errorf("llm: deltanet layer %d: %w", l, err)
		}
		g.kinds[l] = blockKind{attn: false, idx: len(dns)}
		dnCfg, dns = cfg, append(dns, w)
	}
	if len(dns) > 0 {
		if g.dn, err = NewDeltaNetGPU(dev, dnCfg, g.maxTok, dns); err != nil {
			return fmt.Errorf("llm: deltanet: %w", err)
		}
		mark("deltanet", len(dns), g.dn.Buffers(), g.dn.WeightBytes(), g.dn.ActivationBytes(), start)
	}
	dns = nil
	freeHost()

	start = time.Now()
	var atCfg AttnConfig
	var ats []AttnWeights
	for l := 0; l < g.nLayer; l++ {
		cfg, ok, err := m.AttnConfig(l)
		if err != nil {
			return fmt.Errorf("llm: attn config %d: %w", l, err)
		}
		if !ok {
			continue
		}
		w, err := m.AttnWeights(l)
		if err != nil {
			return fmt.Errorf("llm: attn layer %d: %w", l, err)
		}
		g.kinds[l] = blockKind{attn: true, idx: len(ats)}
		atCfg, ats = cfg, append(ats, w)
	}
	if len(ats) > 0 {
		if g.attn, err = NewAttnGPU(dev, atCfg, g.maxTok, g.nKV, ats); err != nil {
			return fmt.Errorf("llm: attn: %w", err)
		}
		mark("attention", len(ats), g.attn.Buffers(), g.attn.WeightBytes(), g.attn.ActivationBytes(), start)
	}
	ats = nil
	freeHost()

	// 5. The head, before the 77 GB rather than after it: it is 2.54 GB of
	//    host floats in flight a slab at a time, and the bank it fills is the
	//    last dense allocation there is.
	if !opts.NoHead {
		start = time.Now()
		t, err := m.Set.Get("output.weight")
		if err != nil {
			return fmt.Errorf("llm: output.weight: %w", err)
		}
		if g.head, err = NewHeadGPU(dev, c.NEmbd, t, 1); err != nil {
			return fmt.Errorf("llm: head: %w", err)
		}
		mark("lm head", 1, g.head.Buffers(), g.head.WeightBytes(), g.head.ActivationBytes(), start)
		freeHost()
	}

	// 6. The expert banks: 97% of the parameters, staged byte for byte out
	//    of the mmap'd checkpoint, one buffer a layer.
	start = time.Now()
	var moes []MoEWeights
	for l := 0; l < g.nLayer; l++ {
		w, err := m.MoEWeights(l)
		if err != nil {
			return fmt.Errorf("llm: moe layer %d: %w", l, err)
		}
		moes = append(moes, w)
	}
	if g.moe, err = NewMoEGPU(dev, m.MoEConfig(), g.maxTok, moes); err != nil {
		return fmt.Errorf("llm: moe: %w", err)
	}
	mark("moe", len(moes), g.moe.Buffers(), g.moe.WeightBytes(), g.moe.ActivationBytes(), start)
	moes = nil
	freeHost()
	return nil
}

// Layers is how many layers this graph runs, NKV its cache, MaxTokens its
// arenas.
func (g *Graph) Layers() int    { return g.nLayer }
func (g *Graph) NKV() int       { return g.nKV }
func (g *Graph) MaxTokens() int { return g.maxTok }

// Head is the staged output projection, or nil when the graph was built
// without one.
func (g *Graph) Head() *HeadGPU { return g.head }

// Past is how many tokens of the current sequence are behind the graph, and
// Ids is that sequence.
func (g *Graph) Past() int    { return g.past }
func (g *Graph) Ids() []int32 { return g.ids }

// Reset starts a fresh sequence: the attention cache from cell zero, the PLE
// convolution from position zero, and every DeltaNet layer's recurrent state
// and window cleared (L7b).
//
// Only the DeltaNet's state is actually written. The other two are *masked*
// rather than cleared — a cell past the end of the sequence is excluded from
// every score, a convolution tap before position zero contributes nothing —
// so clearing them would be work that changes no output.
func (g *Graph) Reset() error {
	g.past = 0
	g.ids = g.ids[:0]
	if g.attn != nil {
		g.attn.Reset()
	}
	if g.ple != nil {
		g.ple.Reset()
	}
	if g.dn != nil {
		for l := 0; l < g.nLayer; l++ {
			if k := g.kinds[l]; !k.attn {
				if err := g.dn.Reset(k.idx); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// Forward runs the whole model over a prompt **as a fresh sequence** and
// returns the logits of its last token, which is the only row llama.cpp
// computes at prefill.
//
// It also returns `result_norm` — the final mixer's output, the tensor the
// head reads — because a logit disagreement has exactly two places to be and
// the caller should not have to run the graph twice to find out which.
func (g *Graph) Forward(ids []int32) (logits, norm []float32, err error) {
	if err := g.Reset(); err != nil {
		return nil, nil, err
	}
	return g.Extend(ids)
}

// Extend runs the model over the next tokens of the sequence the graph is
// already holding, and returns the last one's logits.
//
// It is Forward without the reset, and it is the whole of decode: a prompt is
// one call with many tokens, a generated token is a call with one. What makes
// the two the same computation is L7a's cache and L7b's two carried
// histories, and what asserts it is TestGraphIsAChunkSplit.
func (g *Graph) Extend(ids []int32) (logits, norm []float32, err error) {
	if g.head == nil {
		return nil, nil, fmt.Errorf("llm: this graph was staged without a head")
	}
	// One command buffer for the whole of it, head included (L7d): the
	// layers, the head mixer and the projection are recorded and submitted
	// once, and the two read-backs below are after the flush because nothing
	// has run until then.
	g.record()
	defer func() {
		if ferr := g.flush(); ferr != nil && err == nil {
			logits, norm, err = nil, nil, ferr
		}
	}()
	if err := g.hidden(ids); err != nil {
		return nil, nil, err
	}
	top := time.Now()
	t0 := top
	if err := g.head.Resize(1); err != nil {
		return nil, nil, err
	}
	t0 = since(&g.Stats.Glue, t0)
	if err := g.move.Move(g.head.InPort(), g.hc.MixedPort(), 1); err != nil {
		return nil, nil, err
	}
	t0 = g.blk(&g.Stats.Move, t0)
	if err := g.head.Run(); err != nil {
		return nil, nil, err
	}
	g.blk(&g.Stats.Head, t0)
	// The flush is where the pass actually runs, and what it takes is
	// Stats.GPU plus the hand-over — so the clock restarts after it rather
	// than folding a whole forward pass into the glue.
	if err := g.flush(); err != nil {
		return nil, nil, err
	}
	t0 = time.Now()
	logits, norm = g.head.Logits(), g.hc.Mixed()
	since(&g.Stats.Glue, t0)
	g.Stats.Total += time.Since(top)
	return logits, norm, nil
}

// Hidden runs every layer and the final mixer over a fresh sequence, and
// returns `result_norm` for the last token: [nEmbd].
func (g *Graph) Hidden(ids []int32) ([]float32, error) {
	if err := g.Reset(); err != nil {
		return nil, err
	}
	return g.HiddenExtend(ids)
}

// HiddenExtend is Hidden without the reset: the next tokens of the sequence
// the graph is already holding.
func (g *Graph) HiddenExtend(ids []int32) ([]float32, error) {
	top := time.Now()
	g.record()
	if err := g.hidden(ids); err != nil {
		_ = g.flush()
		return nil, err
	}
	if err := g.flush(); err != nil {
		return nil, err
	}
	t0 := time.Now()
	norm := g.hc.Mixed()
	since(&g.Stats.Glue, t0)
	g.Stats.Total += time.Since(top)
	return norm, nil
}

// hidden is HiddenExtend's body, without the read-back and without the
// submit: every layer, then the final mixer over the last token alone. It is
// separate so that Extend can go on recording through the head rather than
// flushing twice (L7d).
func (g *Graph) hidden(ids []int32) error {
	if err := g.appendN(ids, g.nLayer); err != nil {
		return err
	}
	top := time.Now()
	t0 := top
	// `inp_out_ids`: everything below is per token, so the reference drops
	// every row but the last before the final mixer. Here that is a move of
	// one row of the residual to the front of the same arena, and the mixer
	// then runs as a one-token pass — the same arithmetic on 1/nTok of the
	// work, which is what makes `result_norm` comparable against a trace
	// that holds one row.
	last := g.hc.ResRowPort(len(ids) - 1)
	if err := g.hc.Resize(1); err != nil {
		return err
	}
	t0 = since(&g.Stats.Glue, t0)
	if err := g.move.Move(g.hc.ResRowPort(0), last, 1); err != nil {
		return err
	}
	t0 = g.blk(&g.Stats.Move, t0)
	if err := g.hc.Run(2*g.nLayer, false); err != nil {
		return fmt.Errorf("llm: head mixer: %w", err)
	}
	g.blk(&g.Stats.HC, t0)
	return nil
}

// Residual is the wide residual as the last Prefill left it: [T][hc*nEmbd],
// which is `l_last` of the last layer the graph ran.
//
// It is the tensor a truncated graph is checked against. `Layers: 4` against
// `l_last-3` is the whole order — the embedding, the init, the PLE block at
// layer 1, three DeltaNet layers, one full-attention layer, four MoE blocks
// and eight mixers — for a tenth of the staging and none of the 77 GB the
// other 44 layers need.
func (g *Graph) Residual() []float32 { return g.hc.Res() }

// Prefill runs every layer over a prompt **as a fresh sequence**, leaving the
// wide residual in the hyper-connection block's arena.
func (g *Graph) Prefill(ids []int32) error {
	if err := g.Reset(); err != nil {
		return err
	}
	return g.Append(ids)
}

// PrefillN runs the first nLayer layers of a fresh sequence and stops.
//
// It is how the graph is asked a question the trace can answer. The oracle
// holds `l_last-0` through `l_last-3` and nothing deeper, so "does our error
// accumulate with depth or is there a bug at the bottom" is a question about
// four numbers — and four numbers are only a trend if the same staged model
// produces all of them.
func (g *Graph) PrefillN(ids []int32, nLayer int) error {
	if err := g.Reset(); err != nil {
		return err
	}
	return g.AppendN(ids, nLayer)
}

// Append runs every layer over the next tokens of the sequence the graph is
// holding, and advances it.
func (g *Graph) Append(ids []int32) error { return g.AppendN(ids, g.nLayer) }

// AppendN runs the first nLayer layers over the next tokens of the sequence.
//
// Three things make this a continuation rather than a second prompt, and all
// three are somewhere else: the attention cache and the indexer's pooled
// blocks (L7a), the PLE convolution's ring (L7b), and every DeltaNet layer's
// recurrent state and window, which were already carried across a batch split
// at L3 and only ever needed the graph to stop clearing them. What is left
// here is the n-gram gather — a trigram, so the first token of a continuing
// run still reads two tokens that are not in it, which is why the graph keeps
// the whole id list and hashes over all of it.
func (g *Graph) AppendN(ids []int32, nLayer int) error {
	top := time.Now()
	g.record()
	if err := g.appendN(ids, nLayer); err != nil {
		_ = g.flush()
		return err
	}
	err := g.flush()
	g.Stats.Total += time.Since(top)
	return err
}

// appendN is that body without the record/flush pair, for the entry points
// that go on recording past the layers (L7d).
func (g *Graph) appendN(ids []int32, nLayer int) error {
	if nLayer <= 0 || nLayer > g.nLayer {
		return fmt.Errorf("llm: %d layers, this graph staged %d", nLayer, g.nLayer)
	}
	nTok := len(ids)
	if nTok <= 0 || nTok > g.maxTok {
		return fmt.Errorf("llm: %d tokens, the arenas are built for %d", nTok, g.maxTok)
	}
	if g.past+nTok > g.nKV {
		return fmt.Errorf("llm: %d tokens at position %d of a %d-cell context", nTok, g.past, g.nKV)
	}

	top := time.Now()
	t0 := top

	// The embedding gather, and the n-gram one beside it. Both are host-side
	// reads out of the mmap'd checkpoint: `token_embd` is 675 MB of which a
	// prompt touches nTok rows, and the n-gram table is 28.80 GB of which it
	// touches nTok*16 — and that one has to stay on the host, because D2
	// keeps the table off the device entirely.
	embd, err := g.m.Embeddings(ids)
	if err != nil {
		return fmt.Errorf("llm: token_embd: %w", err)
	}
	g.ids = append(g.ids, ids...)
	var pleEmbd []float32
	if g.hasPLE {
		// Hashed over the whole sequence, gathered for its tail: the trigram
		// of the first token of this run reaches two tokens behind it.
		rows := PLERows(g.pleCfg, g.ids)[g.past*g.pleCfg.NHeads:]
		pleEmbd, err = g.m.PLEGather(rows, g.pleCfg.NHeads, g.pleCfg.HeadDim)
		if err != nil {
			return fmt.Errorf("llm: per_layer_token_embd: %w", err)
		}
	}
	t0 = since(&g.Stats.Gather, t0)

	// The wide residual starts as hc identical copies of the embedding, and
	// from here it lives in the hyper-connection block's arena: every
	// combine updates it in place and nothing else in the graph writes it.
	// It is per *batch* and not per sequence — the stream is a function of
	// the token, and what carries between runs is the three histories above.
	// This is the last host *write* of the pass — a write costs 70 GB/s
	// where a read costs 25 (L6b-4), and the embedding is on the host
	// anyway.
	if err := g.hc.UploadInit(embd, nTok); err != nil {
		return err
	}
	if g.attn != nil {
		if err := g.attn.SetPast(g.past); err != nil {
			return err
		}
	}
	if g.ple != nil {
		if err := g.ple.SetPast(g.past); err != nil {
			return err
		}
	}
	if g.dn != nil {
		if err := g.dn.SetPast(g.past); err != nil {
			return err
		}
	}
	t0 = since(&g.Stats.Glue, t0)

	for l := 0; l < nLayer; l++ {
		if g.hasPLE && g.pleCfg.IsPLE(l) {
			// The one block that reads and writes the wide residual, so the
			// only place the 10240-wide tensor crosses a boundary — twice,
			// once a graph.
			if err := g.ple.UploadEmbd(pleEmbd, nTok); err != nil {
				return err
			}
			t0 = since(&g.Stats.Glue, t0)
			if err := g.move.Move(g.ple.ResPort(), g.hc.ResPort(), nTok); err != nil {
				return err
			}
			t0 = g.blk(&g.Stats.Move, t0)
			if err := g.ple.Run(); err != nil {
				return err
			}
			t0 = g.blk(&g.Stats.PLE, t0)
			if err := g.move.Move(g.hc.ResPort(), g.ple.ResPort(), nTok); err != nil {
				return err
			}
			t0 = g.blk(&g.Stats.Move, t0)
		}

		// The attention half.
		if err := g.hc.Run(2*l, false); err != nil {
			return fmt.Errorf("llm: layer %d attn mix: %w", l, err)
		}
		t0 = g.blk(&g.Stats.HC, t0)
		t1, err := g.sublayer(l, nTok, t0)
		if err != nil {
			return err
		}
		t0 = t1
		if err := g.hc.RunCombine(2 * l); err != nil {
			return fmt.Errorf("llm: layer %d attn combine: %w", l, err)
		}
		t0 = g.blk(&g.Stats.HC, t0)

		// The FFN half, which every layer has.
		if err := g.hc.Run(2*l+1, false); err != nil {
			return fmt.Errorf("llm: layer %d ffn mix: %w", l, err)
		}
		t0 = g.blk(&g.Stats.HC, t0)
		if err := g.moe.Resize(nTok); err != nil {
			return err
		}
		t0 = since(&g.Stats.Glue, t0)
		if err := g.move.Move(g.moe.InPort(), g.hc.MixedPort(), nTok); err != nil {
			return err
		}
		t0 = g.blk(&g.Stats.Move, t0)
		if err := g.moe.Run(l); err != nil {
			return fmt.Errorf("llm: layer %d moe: %w", l, err)
		}
		t0 = g.blk(&g.Stats.MoE, t0)
		if err := g.move.Move(g.hc.BlockOutPort(), g.moe.OutPort(), nTok); err != nil {
			return err
		}
		t0 = g.blk(&g.Stats.Move, t0)
		if err := g.hc.RunCombine(2*l + 1); err != nil {
			return fmt.Errorf("llm: layer %d ffn combine: %w", l, err)
		}
		t0 = g.blk(&g.Stats.HC, t0)
	}
	g.past += nTok
	g.Stats.Runs++
	return nil
}

// sublayer runs one layer's attention half — the gated DeltaNet in 36 layers
// of 48, full attention with the QSA indexer in the other 12 — leaving its
// output in the hyper-connection block's arena for the combine, and returning
// the clock it left off at.
//
// Nothing here is a host tensor. The mixer's output is moved into the layer's
// fp16 A operand by one dispatch and the layer's output is moved back by
// another; `Resize` is only the length of the run and whatever a Move will
// not reach.
func (g *Graph) sublayer(l, nTok int, t0 time.Time) (time.Time, error) {
	k := g.kinds[l]
	var in, out Port
	var run func() error
	if k.attn {
		if err := g.attn.Resize(nTok); err != nil {
			return t0, err
		}
		in, out = g.attn.InPort(), g.attn.OutPort()
		run = func() error { return g.attn.Run(k.idx) }
	} else {
		if err := g.dn.Resize(nTok); err != nil {
			return t0, err
		}
		in, out = g.dn.InPort(), g.dn.OutPort()
		run = func() error { return g.dn.Run(k.idx) }
	}
	t0 = since(&g.Stats.Glue, t0)
	if err := g.move.Move(in, g.hc.MixedPort(), nTok); err != nil {
		return t0, err
	}
	t0 = g.blk(&g.Stats.Move, t0)
	if err := run(); err != nil {
		return t0, fmt.Errorf("llm: layer %d sublayer: %w", l, err)
	}
	if k.attn {
		t0 = g.blk(&g.Stats.Attn, t0)
	} else {
		t0 = g.blk(&g.Stats.DeltaNet, t0)
	}
	if err := g.move.Move(g.hc.BlockOutPort(), out, nTok); err != nil {
		return t0, err
	}
	return g.blk(&g.Stats.Move, t0), nil
}

// Destroy releases every block.
func (g *Graph) Destroy() {
	if g.move != nil {
		g.move.Destroy()
	}
	if g.head != nil {
		g.head.Destroy()
	}
	if g.moe != nil {
		g.moe.Destroy()
	}
	if g.attn != nil {
		g.attn.Destroy()
	}
	if g.dn != nil {
		g.dn.Destroy()
	}
	if g.ple != nil {
		g.ple.Destroy()
	}
	if g.hc != nil {
		g.hc.Destroy()
	}
}

// pleRuns reports whether any PLE layer falls inside a truncated graph.
func pleRuns(c PLEConfig, nLayer int) bool {
	for _, l := range c.Layers {
		if int(l) < nLayer {
			return true
		}
	}
	return false
}

// freeHost returns the floats a block was staged from to the operating system
// before the next block asks for its own.
//
// It matters here and nowhere else in this package, for L6a-5's reason: the
// dense half is dequantised to f32 on the way to the device and the 36
// DeltaNet layers are 8.35 GB of that at once, so a staging that does not
// give them back peaks at the resident model plus a dequantisation.
func freeHost() {
	runtime.GC()
	debug.FreeOSMemory()
}
