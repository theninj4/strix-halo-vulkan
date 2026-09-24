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
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"sync"
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
	// HeadRows is how many rows of logits the head's arena holds. One is
	// what the reference computes and what every caller before L8c wanted:
	// llama.cpp's graph ends with `inp_out_ids`, so prefill projects the
	// last token alone. A perplexity run wants every row, and a row is
	// 0.99 MB of f32 — 2.03 GB for a 2048-token chunk — so it names the
	// slab it works in here rather than asking for the batch.
	HeadRows int
	// Speculative stages the second slot of every carried tensor that P5c's
	// rollback ping-pongs: each DeltaNet layer's recurrent state and
	// convolution ring, and the PLE block's ring. **120.75 MB at 48 layers**,
	// and only a speculative loop has any use for it — a product decode never
	// rewinds a pass — so it is opt-in and `Speculate(true)` refuses on a
	// graph that was not built for it. Nothing else about the graph changes:
	// with it off the scan's destination is the slot it read and
	// `llm_seq_hist.comp` takes the pure-write arm it always had, which is
	// why a `-gen` decode measures the same rate either way.
	Speculative bool
	// Slots stages that many sequences' carried state — the KV cache, every
	// DeltaNet layer's recurrent state and ring, the PLE ring — so that one
	// graph can hold several conversations and switch between them with
	// UseSlot (CONCURRENCY.md C1). Zero and one are one sequence. It is the
	// ping-pong's allocation used for sequences, so it refuses Speculative.
	// At the served context a slot is ~7.4 GB, nearly all of it KV. Each slot
	// has its own cache buffers (C6), so the 4 GiB descriptor range caps a
	// slot's NKV at ~349k cells, not all slots' together.
	Slots int
	// DenseFP16 stages the dense weights as halves, which is what every
	// block did before L8. It is the control, not the default: the bank L8
	// stages is the checkpoint's own int8 with an fp16 scale per 32 elements,
	// which is half the bytes and — for a Q8_0 tensor — the same numbers
	// (bank.go). `LLM_DENSE_FP16=1` sets it from the environment, on the
	// precedent of L6b's `LLM_ARENA_UNCACHED`.
	DenseFP16 bool
}

// denseQ8 is whether the dense banks are L8's. The environment wins, so that
// a run can be repeated on the old bank without rebuilding a caller.
func (o GraphOpts) denseQ8() bool { return !o.DenseFP16 && DenseQ8() }

// DenseQ8 is that choice for a caller with no options of its own — the block
// benchmarks and `-resident`. L8's bank is the default and `LLM_DENSE_FP16=1`
// is the control, on the precedent of L6b's `LLM_ARENA_UNCACHED`.
//
// **A width simulation forces the fp16 arm** (sim.go). L8a's int8 bank is an
// identity only for a Q8_0 tensor, so staging a simulated Q4 weight through
// it would quantise twice and the number measured would be neither format's.
// The fp16 tiling holds the candidate's own halves exactly, which is the
// whole basis of the simulation, so it is the arm a sim has to use — at the
// cost of residency and tok/s being the fp16 arm's, which a sim cannot
// measure anyway.
func DenseQ8() bool { return os.Getenv("LLM_DENSE_FP16") != "1" && DensePlan().Off() }

// HCFuse is whether P1a's combine+norm fusion is in use: one dispatch per
// mixer boundary (`llm_hc_cn.comp`) instead of a scatter and a norm.
// `LLM_HC_NOFUSE=1` puts the pair back, which is the control the write-up's
// numbers are measured against and the arm
// TestHCFusionIsTheCombineThenTheNorm compares against.
//
// The two arms compute the **same bits**, which is what makes the control
// worth keeping: the combine is elementwise and the fused kernel keeps the
// norm's 256-thread partition and its 256-way tree, so nothing about the
// order of a sum changes. What changes is 94 dispatches a pass and one read
// of the wide residual per boundary.
func HCFuse() bool { return os.Getenv("LLM_HC_NOFUSE") != "1" }

// DecodeGEMV is whether the one-token kernels L8d added are in use: the MoE's
// expert GEMVs and split-K router, and the dense split-K GEMV under the gated
// DeltaNet's two projections. They are the default; `LLM_DECODE_GEMM=1` puts
// every one of them back on the cooperative-matrix GEMM that ran before L8d,
// which is the control the write-up's numbers are measured against — and the
// only way to run the two side by side over one prompt, since a block chooses
// its rung from the batch and not from a flag at the call site.
//
// The two paths do not compute the same numbers, and the difference is the
// order of a sum rather than the weights: a cooperative-matrix accumulator
// adds sixteen k an instruction in an order the extension does not define,
// where a GEMV lane adds them serially and a split-K reduce adds the slabs
// afterwards. It is f32 round-off over a 2560-long chain — rms 1.7e-05 on a
// projection whose scale is 31.5 — and at temperature zero over a hundred
// tokens it is enough to re-word a sentence. See L8d's gate.
func DecodeGEMV() bool { return os.Getenv("LLM_DECODE_GEMM") != "1" }

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
	// pleAhead is a PrefetchPLE still faulting pages in (P17). A gather waits
	// for it rather than faulting the same pages beside it, and Destroy waits
	// for it because it reads the model's mapping.
	pleAhead sync.WaitGroup

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
	// spans are the sequence's images, in cell order (LLM-VISION.md V5):
	// what makes a cell's rotary position differ from the cell. ropeHW is,
	// per slot, the end of the rotary rows that may not be a text table's,
	// so a sequence without images rewrites rows only below it.
	spans  []ImageSpan
	ropeHW []int

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

	// The decode step, recorded once (P1c). A one-token pass is the same
	// dispatches with the same push constants every time — the position moved
	// into the arenas at P1c, and TestDecodeDispatchDiff is the proof — so
	// the first one-token Extend captures its sequence into a reusable
	// command buffer and every later one submits that instead of re-encoding
	// 1407 dispatches. preKinds and preOwner are the captured labels, so the
	// replay feeds Stats exactly as the recording path does.
	pre      *vk.Prerecorded
	preKinds []string
	preOwner []string
	// preEpoch is the decode plan `pre` was captured under, and capEpoch the
	// one the armed capture is for. **A recorded buffer is only replayable
	// while the plan that built it still holds**, and since P15 one knob in
	// that plan moves with the depth: the attention block takes its gathered
	// arm once the cache is deeper than twice the selection's width, which is
	// two more dispatches and a different grid. So the epoch is compared
	// before every replay and the one step that crosses is re-recorded. It
	// crosses at most once in a sequence, because the depth only grows.
	preEpoch, capEpoch int
	// capture arms the next flush to build `pre` from what it submits.
	capture bool

	// Speculation (P5c, research/p5-mtp-rollback.md §3). `spec` is whether
	// passes are rewindable; the four fields under it are the one pass in
	// flight, remembered so that Rewind can put the sequence back exactly
	// where the pass found it. `specBlocks` is the attention block's pooled
	// indexer rows, the one thing here that does not rewind by itself.
	spec       bool
	specArmed  bool
	specPast   int
	specRows   int
	specIds    int
	specSpans  int
	specBlocks []uint16

	// Sequence slots (CONCURRENCY.md C1). `slot` is the live one; `past`,
	// `ids` and the recorded decode step above are its, and `parked` holds
	// every other slot's copy of them. The recorded step is per slot because
	// its push constants carry the slot's cache, state and ring offsets.
	slots  int
	slot   int
	parked []seqState
}

// seqState is the host's half of a sequence that is not the live one: what
// UseSlot puts away and brings back. The device's half stays where it is,
// in the slot's own region of each block's arenas.
type seqState struct {
	past               int
	ids                []int32
	spans              []ImageSpan
	pre                *vk.Prerecorded
	preKinds, preOwner []string
	preEpoch           int
}

// Prerecord is whether decode replays a recorded command buffer. On by
// default; LLM_NO_PRERECORD=1 re-records every token, which is the control
// the P1c gate compares tokens against.
func Prerecord() bool { return os.Getenv("LLM_NO_PRERECORD") != "1" }

// record installs the recorder on every block, so that everything from here
// to the matching `flush` is recorded rather than submitted. It is idempotent
// per pass and paired with `flush` on every return path, which is why the
// callers are the three entry points and not the layer loop.
func (g *Graph) record(rows int) {
	if g.rec == nil {
		g.rec = &recorder{}
	}
	g.rec.rows = rows
	// The depth the attention dispatches of this pass will run at, which is
	// the other half of what one command buffer may hold (nsPerRowCell).
	g.rec.past = g.past
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
	// The capture, before submit resets the recorder's slices. The pass has
	// to run once through the ordinary path anyway, so the sequence is taken
	// from what is about to be submitted — the same dispatches, byte for
	// byte — and the prerecorded buffer is built only if that submit
	// succeeds.
	var capD []vk.MultiDispatch
	var capK, capO []string
	if g.capture {
		g.capture = false
		capD = append([]vk.MultiDispatch(nil), g.rec.d...)
		capK = append([]string(nil), g.rec.kind...)
		capO = append([]string(nil), g.rec.owner...)
	}
	t0 := time.Now()
	byOwner, byKind, total, err := g.rec.submit()
	submit := time.Since(t0)
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
	if capD != nil {
		// One buffer for the whole step, where the live path chunks at
		// maxBatch: a decode step is ~27 ms of GPU against the ring
		// watchdog's 2 s, and the prerecorded pool sizes its own marks.
		pre, perr := vk.NewPrerecorded(capD, true, true)
		if perr != nil {
			// The pass itself succeeded; losing the fast path costs 6% of a
			// step, not correctness, so say so and go on re-recording.
			fmt.Fprintf(os.Stderr, "llm: decode prerecord failed, re-recording each token: %v\n", perr)
		} else {
			g.pre, g.preKinds, g.preOwner = pre, capK, capO
			g.preEpoch = g.capEpoch
		}
	}
	g.Stats.Dispatches += n
	g.Stats.GPU += total
	g.Stats.Submit += submit
	if g.Stats.Kinds == nil {
		g.Stats.Kinds = make(map[string]DispatchStat, len(byKind))
	}
	for k, v := range byKind {
		s := g.Stats.Kinds[k]
		s.Count, s.GPU = s.Count+v.Count, s.GPU+v.GPU
		g.Stats.Kinds[k] = s
	}
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
// **L8d widened it.** The one-token kernels that stage replaces — the MoE's
// expert GEMVs, its split-K router and the split-K GEMV under the DeltaNet's
// two projections — are not bit-exact against the GEMMs they replace either,
// and for the same reason: a cooperative-matrix accumulator sums sixteen k an
// instruction in an order the extension does not define, a GEMV lane sums them
// serially, and a split-K reduce adds the slabs afterwards. So the pin now
// covers four blocks rather than one — **five with L8e**, which puts the
// full-attention layer's two projections on the same kernel, and **six with
// P8**, which splits the attention itself the same way and for the same
// reason: a decode step's key axis is cut across workgroups and folded back
// together, where a prefill walks it once.
func (g *Graph) PinSchedule(on bool) error {
	// The pin changes which pipelines a one-token pass plans, so a captured
	// decode step no longer matches what the graph would record; the next
	// one-token Extend captures afresh. **Every slot's**, not only the live
	// one's: a parked slot's step was recorded under the old plan too, and
	// replaying it after the pin is a run on the schedule the pin was meant
	// to exclude (found by TestGraphImageDecode, LLM-VISION.md V5).
	g.dropPrerecorded()
	for i := range g.parked {
		if g.parked[i].pre != nil {
			g.parked[i].pre.Destroy()
			g.parked[i].pre, g.parked[i].preKinds, g.parked[i].preOwner = nil, nil, nil
		}
	}
	if g.moe != nil {
		g.moe.PinGemv(on)
	}
	if g.dn != nil {
		g.dn.PinGemv(on)
	}
	if g.attn != nil {
		g.attn.PinGemv(on)
		// And P8's split decode attention, which is the sixth kernel this
		// pin covers and reassociates for the plainest reason of all: it
		// cuts the key axis across workgroups and folds each slice's online
		// softmax together afterwards, where the unsplit kernel folds them
		// as it walks. Measured at the layer's own output, the two differ by
		// 1.3e-04 rms against the 9.2e-04 that layer already sits from
		// llama.cpp — but "differ" is what matters to an equality, so it
		// goes off with the rest of them.
		if on {
			g.attn.SetSplits(1)
			// And P14-2's gather, which is the seventh kernel this pin covers
			// and the only one that could not have been made chunk-invariant:
			// its list is the union of a query tile's sixteen rows, so a chunk
			// that ends inside the tile gathers a different list and the online
			// softmax folds the same terms in a different order. See
			// AttnGPU.SetGather.
			g.attn.SetGather(false)
		} else {
			g.attn.SetSplits(0)
			g.attn.AutoGather()
		}
	}
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
	// Submit is the **host** wall clock around the submit and the fence
	// wait — GPU is the part of it the timestamps saw, so `Submit - GPU` is
	// what handing one command buffer over costs (P1). It is separated from
	// the rest of the host side because it is the one interval a
	// pre-recorded command buffer would not remove.
	Submit time.Duration
	// Kinds is the same GPU time split by each dispatch's own label rather
	// than by the block it came from (P1). A decode step is ~490 dispatches
	// over about thirty distinct labels, so this is the resolution at which
	// "where does the token go" has an answer.
	Kinds map[string]DispatchStat
}

// Blocks is the time inside the five blocks and the head.
func (s GraphStats) Blocks() time.Duration {
	return s.HC + s.PLE + s.DeltaNet + s.Attn + s.MoE + s.Head
}

// Clone is a snapshot that the next pass cannot change.
//
// Every other field is a scalar and copies with the struct, but `Kinds` is a
// map: `st := g.Stats` shares it, so dispatches recorded *after* the snapshot
// still land in it while the scalars beside them stay put. A caller that keeps
// several snapshots — the depth sweep keeps one a depth a phase — then reads
// per-label times that include work belonging to a later measurement, and the
// scalar columns next to them disagree. Take this instead of the struct.
func (s GraphStats) Clone() GraphStats {
	if s.Kinds == nil {
		return s
	}
	k := make(map[string]DispatchStat, len(s.Kinds))
	for name, v := range s.Kinds {
		k[name] = v
	}
	s.Kinds = k
	return s
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
	slots := max(opts.Slots, 1)
	if slots > 1 && opts.Speculative {
		return nil, fmt.Errorf("llm: GraphOpts.Slots and Speculative share the carried state's slot index; pick one")
	}
	g := &Graph{m: m, cfg: c, nLayer: nLayer, maxTok: opts.MaxTokens, nKV: nKV,
		slots: slots, parked: make([]seqState, slots), ropeHW: make([]int, slots)}

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
	// L8c-6's bank, where the plan names this family. `hc_attn_down` is the
	// tensor `simFamily` maps to "hyper_conn" and the checkpoint ships it as
	// Q8_0, which is what the `true` says; `inject` is F32 and stays in the
	// fp16 tail whatever this is (D13).
	hcOpts := HCBankOpts()
	hcOpts.Q8 = opts.denseQ8()
	if g.hc, err = NewHCGPU(dev, c.HCConfig(), g.maxTok, mixers, hcOpts); err != nil {
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
		// P2's bank, where the plan names this family. `ple_key` is the
		// tensor `simFamily` maps to "ple_proj" and the checkpoint ships it
		// as Q8_0, which is what the `true` says — the one block that never
		// got L8a's int8 stage either, so the two-valued default is new here
		// too.
		pleOpts := PLEOpts{Bank: bankOf(opts.denseQ8()), Layer: pleCfg.Layers[0]}
		if q, ok := DenseBankPlan().For(fmt.Sprintf("blk.%d.ple_key.weight", pleCfg.Layers[0]), true); ok {
			b, err := bankOfSim(q)
			if err != nil {
				return err
			}
			pleOpts.Bank, pleOpts.Sim = b, q
		}
		pleOpts.Speculative = opts.Speculative
		pleOpts.Slots = g.slots
		if g.ple, err = NewPLEGPU(dev, pleCfg, g.maxTok, w, pleOpts); err != nil {
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
		// L8c-5's bank, where the plan names this family. `attn_qkv` is the
		// tensor `simFamily` maps to "deltanet" and the checkpoint ships it
		// as Q8_0, which is what the `true` says — the block's own two F32
		// matrices stay in the fp16 tail whatever this is (D13).
		dnBank, dnSim := bankOf(opts.denseQ8()), QuantSim{}
		if q, ok := DenseBankPlan().For("blk.0.attn_qkv.weight", true); ok {
			b, err := bankOfSim(q)
			if err != nil {
				return err
			}
			dnBank, dnSim = b, q
		}
		var dnOpts []DNOption
		if g.slots > 1 {
			dnOpts = append(dnOpts, WithDNSlots(g.slots))
		}
		if opts.Speculative {
			dnOpts = append(dnOpts, WithDNSpeculative())
		}
		if g.dn, err = NewDeltaNetGPUBank(dev, dnCfg, g.maxTok, dns, dnBank, dnSim, dnOpts...); err != nil {
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
		// L8c-7's bank, where the plan names this family. `attn_q` is the
		// tensor `simFamily` maps to "full_attn" and the checkpoint ships it
		// as Q8_0, which is what the `true` says. The indexer's two BF16
		// projections ride the same plane at the same width — the fp16 tail
		// left with D13's payoff (P2) — so a plan that names `qsa_indexer`
		// at a different width than `full_attn` is refused rather than
		// averaged: they are one fused matrix.
		atBank, atSim := bankOf(opts.denseQ8()), QuantSim{}
		if q, ok := DenseBankPlan().For("blk.0.attn_q.weight", true); ok {
			b, err := bankOfSim(q)
			if err != nil {
				return err
			}
			atBank, atSim = b, q
		}
		if q, ok := DenseBankPlan().For("blk.0.indexer.q_proj.weight", false); ok {
			if _, isQK := qkBank(atBank); !isQK {
				return fmt.Errorf("llm: qsa_indexer is staged in the fused projection's plane, so it needs full_attn on the same bank")
			}
			if q != atSim {
				return fmt.Errorf("llm: full_attn is %s and qsa_indexer is %s, and they share one plane", atSim, q)
			}
		}
		if g.attn, err = NewAttnGPUBank(dev, atCfg, g.maxTok, g.nKV, ats, atBank, atSim,
			WithAttnSlots(g.slots)); err != nil {
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
		headRows := opts.HeadRows
		if headRows <= 0 {
			// One, or a row a slot, so that DecodeRows can advance every
			// sequence the graph holds in one pass (CONCURRENCY.md C5).
			headRows = max(1, opts.Slots)
		}
		// L8c-4's bank, where the plan names this family. Everything else
		// is still the checkpoint's own width (D13): the 4.5-bit bank has a
		// kernel for a matrix whose k is a multiple of 256, which is every
		// dense weight here but the 320-wide hyper-connection pair.
		// `true` is "the checkpoint ships it as Q8_0", which `output.weight`
		// does — it is what `LLM_DENSE_BANK_SRC` would select on if this
		// plan ever grew one.
		bank, sim := bankOf(opts.denseQ8()), QuantSim{}
		if q, ok := DenseBankPlan().For(t.Name, true); ok {
			b, err := bankOfSim(q)
			if err != nil {
				return err
			}
			bank, sim = b, q
		}
		if g.head, err = NewHeadGPUBank(dev, c.NEmbd, t, headRows, bank, sim); err != nil {
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

// Attn is the staged full-attention block, or nil when the graph was built
// without one. It is exported for the **selection**: the QSA bitmask a pass
// leaves behind is what says how much of the key axis the attention kernel's
// block skip can actually skip, and at prefill that is the whole of the depth
// term (P11). `cmd/llm -depth` reads it.
func (g *Graph) Attn() *AttnGPU { return g.attn }

// Past is how many tokens of the current sequence are behind the graph, and
// Ids is that sequence.
func (g *Graph) Past() int    { return g.past }
func (g *Graph) Ids() []int32 { return g.ids }

// Slots is how many sequences the graph was staged to hold, and Slot the one
// every call below runs on.
func (g *Graph) Slots() int { return g.slots }
func (g *Graph) Slot() int  { return g.slot }

// UseSlot makes sequence slot s the live one (CONCURRENCY.md C1): every
// Forward, Extend, Reset, Past and Ids after it is that sequence's, and the
// one that was live is left exactly where it stood, to be picked up by a
// later UseSlot.
//
// **Switching moves nothing on the device.** Each slot's KV cache, DeltaNet
// states and rings live in their own region of the blocks' arenas, so a
// switch is four blocks re-pointing their offsets, the position written to
// three arena dwords, and the host swapping its id list and recorded decode
// step. That is what makes interleaving conversations a token at a time
// cost nothing but the interleaving.
func (g *Graph) UseSlot(s int) error {
	if s < 0 || s >= g.slots {
		return fmt.Errorf("llm: sequence slot %d of %d", s, g.slots)
	}
	if s == g.slot {
		return nil
	}
	if g.specArmed {
		return fmt.Errorf("llm: a speculative pass is in flight; commit or rewind it first")
	}
	g.parked[g.slot] = seqState{
		past: g.past, ids: g.ids, spans: g.spans,
		pre: g.pre, preKinds: g.preKinds, preOwner: g.preOwner, preEpoch: g.preEpoch,
	}
	in := g.parked[s]
	g.parked[s] = seqState{}
	g.past, g.ids, g.spans = in.past, in.ids, in.spans
	g.pre, g.preKinds, g.preOwner, g.preEpoch = in.pre, in.preKinds, in.preOwner, in.preEpoch
	g.slot = s
	if g.attn != nil {
		if err := g.attn.UseSlot(s); err != nil {
			return err
		}
		if err := g.attn.SetPast(g.past); err != nil {
			return err
		}
	}
	if g.ple != nil {
		if err := g.ple.UseSlot(s); err != nil {
			return err
		}
		if err := g.ple.SetPast(g.past); err != nil {
			return err
		}
	}
	if g.dn != nil {
		if err := g.dn.UseSlot(s); err != nil {
			return err
		}
		if err := g.dn.SetPast(g.past); err != nil {
			return err
		}
	}
	return nil
}

// Checkpoint is one sequence's carried state at one position, held on the
// host: what Graph.Restore needs to take a slot back there after it has run
// on (CONCURRENCY.md C4).
//
// A DeltaNet state has no inverse, so without one a conversation that
// diverges from what its slot holds anywhere before the end is prefilled
// from token zero. A voice command is the same long system prompt with a new
// utterance, and every one of them re-read the prompt.
//
// **It is the recurrences and not the cache.** What is copied is every
// DeltaNet layer's state and ring (~120 MB at 36 layers) and the PLE ring.
// The KV cells and raw indexer keys below `past` are never rewritten by a run
// that continues from there, so they stay on the device. That is why a
// checkpoint can only be restored into the slot it was taken in, and only
// while that slot still holds its tokens (Restore checks both). Cells at or
// past `past` are masked out of every score until they are rewritten.
//
// **The pooled indexer table needs nothing either, though the rollback
// snapshots it (P5c).** A continuation's pass rebuilds the blocks it
// completes before it scores any, and every block at or past the last whole
// one is scored as one broadcast value under a -inf or 1e9 bias
// (llm_attn_score.comp, P7), which no stored row can change the selection of.
// So the blocks an abandoned continuation completed are never read.
// TestGraphCheckpointIsThePrefill measured it: a restore that also put the
// frontier row back over them gave the same logits bit for bit, with the
// detour reaching past the real continuation.
type Checkpoint struct {
	past  int
	ids   []int32
	spans []ImageSpan
	dn    []float32
	ple   []float32
}

// Past is the position the checkpoint was taken at, and Ids the tokens it
// holds.
func (c *Checkpoint) Past() int    { return c.past }
func (c *Checkpoint) Ids() []int32 { return c.ids }

// Checkpoint copies the live slot's carried state out. A non-nil `into` is
// reused, buffers and all, so that a slot re-checkpointing every request does
// not allocate 120 MB a time.
func (g *Graph) Checkpoint(into *Checkpoint) (*Checkpoint, error) {
	if g.specArmed {
		return nil, fmt.Errorf("llm: a speculative pass is in flight; commit or rewind it first")
	}
	if g.past == 0 || len(g.ids) != g.past {
		return nil, fmt.Errorf("llm: nothing to checkpoint at position %d with %d ids", g.past, len(g.ids))
	}
	c := into
	if c == nil {
		c = &Checkpoint{}
	}
	c.past = g.past
	c.ids = append(c.ids[:0], g.ids...)
	c.spans = append(c.spans[:0], g.spans...)
	if g.dn != nil {
		c.dn = resized(c.dn, g.dn.CarriedLen())
		if err := g.dn.SaveCarried(c.dn); err != nil {
			return nil, err
		}
	}
	if g.ple != nil {
		c.ple = resized(c.ple, g.ple.CarriedLen())
		if err := g.ple.SaveCarried(c.ple); err != nil {
			return nil, err
		}
	}
	return c, nil
}

// Restore takes the live slot back to a checkpoint taken in it. It refuses
// unless the slot's sequence still starts with the checkpoint's tokens,
// because that is what says the cache cells below it were written by those
// tokens and not by some other conversation since.
func (g *Graph) Restore(c *Checkpoint) error {
	if g.specArmed {
		return fmt.Errorf("llm: a speculative pass is in flight; commit or rewind it first")
	}
	if c == nil || c.past == 0 || len(g.ids) != g.past || c.past > g.past ||
		!slices.Equal(g.ids[:c.past], c.ids) {
		return fmt.Errorf("llm: this slot no longer holds the checkpoint's tokens")
	}
	// An image's cells are all the same pad id, so the ids alone cannot say
	// the slot holds the checkpoint's pictures: the spans, hashes and all,
	// have to agree too.
	if kept, err := truncateSpans(g.spans, c.past); err != nil || !slices.Equal(kept, c.spans) {
		return fmt.Errorf("llm: this slot no longer holds the checkpoint's images")
	}
	if g.dn != nil {
		if err := g.dn.RestoreCarried(c.dn); err != nil {
			return err
		}
	}
	if g.ple != nil {
		if err := g.ple.RestoreCarried(c.ple); err != nil {
			return err
		}
	}
	return g.rewindPosition(c.past, c.past)
}

func resized(b []float32, n int) []float32 {
	if cap(b) < n {
		return make([]float32, n)
	}
	return b[:n]
}

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
	g.spans = nil
	g.specArmed, g.specBlocks = false, nil
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

// Speculate arms the rollback: while it is on, a pass may be thrown away
// (P5c, research/p5-mtp-rollback.md §3).
//
// **What a pass mutates that outlives it is four things, and only two of them
// need anything.** Every DeltaNet layer's recurrent state and both
// convolution rings are written into a second slot and the committed one is
// left alone, so Rewind is *doing nothing* and Commit is flipping a pair of
// integers. The KV cache and the indexer's raw keys need no rollback at all —
// a cell past the end of the sequence is masked out of every score, so a
// rewound cell is unreadable rather than stale — and the host's id list is a
// slice. The one thing left is the pooled indexer table, whose rows are final
// once written, and Rewind restores those.
//
// Exactly one pass may be in flight: the ping-pong has two slots, so a second
// uncommitted pass would read the state its predecessor was supposed to leave
// and find the committed one instead. Commit or Rewind before running another.
//
// It also drops the pre-recorded decode step, because a speculative pass has
// a different destination in its push constants and P1c's buffer is the bytes
// of a pass that is committed as it runs.
func (g *Graph) Speculate(on bool) error {
	if on == g.spec {
		return nil
	}
	if on && g.slots > 1 {
		return fmt.Errorf("llm: this graph's slots are sequences (GraphOpts.Slots); it cannot also speculate")
	}
	// The draft head (llm/mtp.go) has a one-layer cache of its own with a
	// text rotary table, so behind an image it would rope 176 or 1 000
	// positions off. It is parked (spec-loop 0.95x); refuse rather than port
	// it (LLM-VISION.md Q5).
	if on && len(g.spans) > 0 {
		return fmt.Errorf("llm: this sequence holds an image, and the draft head has no image positions")
	}
	if g.specArmed {
		return fmt.Errorf("llm: a speculative pass is in flight; commit or rewind it first")
	}
	// The blocks first, because a graph that was not staged for it refuses
	// here and must be left exactly as it was.
	if g.dn != nil {
		if err := g.dn.Speculate(on); err != nil {
			return err
		}
	}
	if g.ple != nil {
		if err := g.ple.Speculate(on); err != nil {
			if g.dn != nil {
				_ = g.dn.Speculate(!on)
			}
			return err
		}
	}
	g.dropPrerecorded()
	g.spec = on
	return nil
}

// Speculating is whether passes are rewindable.
func (g *Graph) Speculating() bool { return g.spec }

// Commit accepts the speculative pass just run, whole.
//
// It is the only thing that moves the ping-pong: what the pass wrote becomes
// the committed state and the committed ring, and the slot it read becomes
// the scratch the *next* pass writes. `past` and the id list already moved
// when the pass ran, because a committed pass is an ordinary one.
func (g *Graph) Commit() error {
	if !g.specArmed {
		return fmt.Errorf("llm: nothing to commit; no speculative pass has run")
	}
	if g.dn != nil {
		g.dn.CommitSlot()
	}
	if g.ple != nil {
		g.ple.CommitSlot()
	}
	g.specArmed, g.specBlocks = false, nil
	return nil
}

// Rewind discards the speculative pass just run: the sequence goes back to
// the position and the ids it started from, and every carried history with it.
//
// The pass's tokens are *not* re-run here. A caller that accepted some of them
// re-runs the accepted prefix at the front of its next pass, which is what the
// design pass costs out (§3.3c): the next pass is rows wide whatever happens,
// so carrying the accepted rows into it is cheaper than snapshotting a
// recurrent state per token.
func (g *Graph) Rewind() error {
	if !g.specArmed {
		return fmt.Errorf("llm: nothing to rewind; no speculative pass has run")
	}
	// The pooled indexer rows first, while `specPast`/`specRows` still say
	// which ones the pass claimed.
	if g.attn != nil && g.specBlocks != nil {
		if err := g.attn.RestoreBlocks(g.specPast, g.specRows, g.specBlocks); err != nil {
			return err
		}
	}
	g.past = g.specPast
	g.ids = g.ids[:g.specIds]
	g.spans = g.spans[:g.specSpans]
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
	g.specArmed, g.specBlocks = false, nil
	return nil
}

// rewindPosition is Rewind's position half alone: the sequence goes back to
// where it was and every carried history stays where the pass left it.
//
// **Nothing in the product calls it and nothing should.** It exists because
// TestSpeculationRewindIsTheSequence needs a control that can fail (P5b's
// finding 2): a rewind that restores the position and not the histories is
// exactly the bug the ping-pong exists to prevent, and a gate that passes
// against it is not evidence that anything was rolled back.
func (g *Graph) rewindPosition(past, ids int) error {
	if past > g.past || ids > len(g.ids) {
		return fmt.Errorf("llm: rewinding to %d/%d from %d/%d", past, ids, g.past, len(g.ids))
	}
	spans, err := truncateSpans(g.spans, past)
	if err != nil {
		return err
	}
	g.past, g.ids, g.spans = past, g.ids[:ids], spans
	g.specArmed, g.specBlocks = false, nil
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
	return nil
}

// arm remembers what a speculative pass is about to overwrite. It runs at the
// top of appendN, before anything has moved.
func (g *Graph) arm(nTok int) error {
	if !g.spec {
		return nil
	}
	if g.specArmed {
		return fmt.Errorf("llm: a speculative pass is already in flight; commit or rewind it first")
	}
	g.specPast, g.specRows, g.specIds, g.specBlocks = g.past, nTok, len(g.ids), nil
	g.specSpans = len(g.spans)
	if g.attn != nil && g.past > 0 {
		snap, err := g.attn.SnapshotBlocks(g.past, nTok)
		if err != nil {
			return err
		}
		g.specBlocks = snap
	}
	g.specArmed = true
	return nil
}

// ExtendRows is Extend with every row's logits rather than the last one's, and
// it is what a speculative verification pass runs.
//
// llama.cpp's graph ends with `inp_out_ids`, so `Extend` moves the last row of
// the residual to the front and runs the final mixer and the head on one row.
// A verification pass needs them all: row t's logits are what says whether the
// draft's guess for token t+1 was the trunk's own. The residual comes back
// beside them — [rows][hc*nEmbd], the same tensor `Residual` returns — because
// the draft head's next round is seeded from the row the round commits on.
//
// Both slices are into mapped arenas the next pass overwrites.
func (g *Graph) ExtendRows(ids []int32) (logits, res []float32, err error) {
	if g.head == nil {
		return nil, nil, fmt.Errorf("llm: this graph was staged without a head")
	}
	n := len(ids)
	if n <= 0 || n > g.head.MaxRows() {
		return nil, nil, fmt.Errorf("llm: %d rows, the head arena holds %d — GraphOpts.HeadRows", n, g.head.MaxRows())
	}
	top := time.Now()
	g.record(n)
	defer func() {
		if ferr := g.flush(); ferr != nil && err == nil {
			logits, res, err = nil, nil, ferr
		}
	}()
	if err := g.appendN(ids, g.nLayer); err != nil {
		return nil, nil, err
	}
	t0 := time.Now()
	// The final mixer over every row, which is the one line `hidden` does
	// differently: there it is one row after a move, here it is per token on
	// nTok times the work.
	if err := g.hc.Run(2*g.nLayer, false); err != nil {
		return nil, nil, fmt.Errorf("llm: head mixer: %w", err)
	}
	t0 = g.blk(&g.Stats.HC, t0)
	if err := g.head.Resize(n); err != nil {
		return nil, nil, err
	}
	t0 = since(&g.Stats.Glue, t0)
	if err := g.move.Move(g.head.InPort(), g.hc.MixedPort(), n); err != nil {
		return nil, nil, err
	}
	t0 = g.blk(&g.Stats.Move, t0)
	if err := g.head.Run(); err != nil {
		return nil, nil, err
	}
	g.blk(&g.Stats.Head, t0)
	if err := g.flush(); err != nil {
		return nil, nil, err
	}
	t0 = time.Now()
	logits, res = g.head.Logits(), g.hc.Res()
	since(&g.Stats.Glue, t0)
	g.Stats.Total += time.Since(top)
	return logits, res, nil
}

// batchSabotage, when set, runs after DecodeRows has written the blocks'
// position tables. Nothing in the product sets it; it is how
// TestGraphDecodeRowsIsEachSlot's controls break one block's table to show
// the gate can see it (the same arrangement as rewindPosition).
var batchSabotage func(g *Graph)

// DecodeRows advances several sequences one token each in one pass
// (CONCURRENCY.md C5): row r is token ids[r] of the sequence in slot
// slots[r]. It returns every row's logits, [len(slots)][vocab], and leaves
// the live slot live.
//
// The rows share every weight read — the projections, the MoE and the head
// run over all of them at once, which is what makes a three-row pass cost
// 1.78 steps rather than three — and each row's attention, DeltaNet and PLE
// run on its own sequence's cache, state and ring at its own position
// (batch.go). A single row is an ordinary Extend on that slot, recorded
// decode step and all.
func (g *Graph) DecodeRows(slots []int, ids []int32) (logits []float32, err error) {
	n := len(slots)
	switch {
	case n == 0 || n != len(ids):
		return nil, fmt.Errorf("llm: %d slots and %d tokens", n, len(ids))
	case g.head == nil:
		return nil, fmt.Errorf("llm: this graph was staged without a head")
	case n > g.head.MaxRows():
		return nil, fmt.Errorf("llm: %d rows, the head arena holds %d — GraphOpts.HeadRows", n, g.head.MaxRows())
	case g.spec:
		return nil, fmt.Errorf("llm: a speculating graph cannot batch sequences")
	}
	if n == 1 {
		live := g.slot
		if err := g.UseSlot(slots[0]); err != nil {
			return nil, err
		}
		l, _, err := g.Extend(ids)
		if err == nil {
			// Extend's logits are a view of an arena the next pass writes.
			l = append([]float32(nil), l...)
		}
		if uerr := g.UseSlot(live); uerr != nil && err == nil {
			err = uerr
		}
		return l, err
	}

	// Every row's sequence where the others are: the live one's position and
	// ids go into its parked entry for the length of the pass, and back.
	live := g.slot
	g.parked[live].past, g.parked[live].ids, g.parked[live].spans = g.past, g.ids, g.spans
	defer func() {
		g.past, g.ids, g.spans = g.parked[live].past, g.parked[live].ids, g.parked[live].spans
		g.parked[live].past, g.parked[live].ids, g.parked[live].spans = 0, nil, nil
	}()
	rows := make([]batchRow, n)
	for r, s := range slots {
		if s < 0 || s >= g.slots {
			return nil, fmt.Errorf("llm: sequence slot %d of %d", s, g.slots)
		}
		rows[r] = batchRow{slot: s, past: g.parked[s].past}
	}
	if err := checkBatch(rows, g.slots, g.nKV); err != nil {
		return nil, err
	}
	for _, r := range rows {
		if err := g.prepRope(r.slot, r.past, 1, g.parked[r.slot].spans); err != nil {
			return nil, err
		}
	}

	top := time.Now()
	t0 := top
	embd, err := g.m.Embeddings(ids)
	if err != nil {
		return nil, fmt.Errorf("llm: token_embd: %w", err)
	}
	var pleEmbd []float32
	if g.hasPLE {
		// Each row's n-gram is its own sequence's, and reaches NGram-1
		// tokens behind it.
		var rowsPLE []int32
		for r, s := range slots {
			h := g.parked[s].ids
			tail := append(append([]int32(nil), h[max(0, len(h)-g.pleCfg.NGram):]...), ids[r])
			rowsPLE = append(rowsPLE, PLERowsFrom(g.pleCfg, tail, len(tail)-1)...)
		}
		g.pleAhead.Wait()
		if pleEmbd, err = g.m.PLEGather(rowsPLE, g.pleCfg.NHeads, g.pleCfg.HeadDim); err != nil {
			return nil, fmt.Errorf("llm: per_layer_token_embd: %w", err)
		}
	}
	t0 = since(&g.Stats.Gather, t0)
	if err := g.hc.UploadInit(embd, n); err != nil {
		return nil, err
	}
	type batcher interface{ SetBatch([]batchRow) error }
	var set []batcher
	if g.attn != nil {
		set = append(set, g.attn)
	}
	if g.dn != nil {
		set = append(set, g.dn)
	}
	if g.ple != nil {
		set = append(set, g.ple)
	}
	defer func() {
		for _, b := range set {
			_ = b.SetBatch(nil)
		}
	}()
	for _, b := range set {
		if err := b.SetBatch(rows); err != nil {
			return nil, err
		}
	}
	if batchSabotage != nil {
		batchSabotage(g)
	}
	t0 = since(&g.Stats.Glue, t0)

	g.record(n)
	defer func() {
		if ferr := g.flush(); ferr != nil && err == nil {
			logits, err = nil, ferr
		}
	}()
	if t0, err = g.layers(n, g.nLayer, pleEmbd, t0); err != nil {
		return nil, err
	}
	if err := g.hc.Run(2*g.nLayer, false); err != nil {
		return nil, fmt.Errorf("llm: head mixer: %w", err)
	}
	t0 = g.blk(&g.Stats.HC, t0)
	if err := g.head.Resize(n); err != nil {
		return nil, err
	}
	if err := g.move.Move(g.head.InPort(), g.hc.MixedPort(), n); err != nil {
		return nil, err
	}
	t0 = g.blk(&g.Stats.Move, t0)
	if err := g.head.Run(); err != nil {
		return nil, err
	}
	g.blk(&g.Stats.Head, t0)
	if err := g.flush(); err != nil {
		return nil, err
	}
	for r, s := range slots {
		st := &g.parked[s]
		st.ids = append(st.ids, ids[r])
		st.past++
	}
	g.Stats.Runs++
	logits = g.head.Logits()
	g.Stats.Total += time.Since(top)
	return logits, nil
}

// PassExperts is how many **distinct** experts the last pass's last layer
// routed to, across every row of it.
//
// It is the numerator of the one growth law a multi-row pass has that a
// one-row pass does not: the dense families read the same bytes whatever the
// row count and the routed experts do not (research/p5-mtp-rollback.md §1.2).
// The MoE block's arenas are shared across layers, so what is readable after a
// pass is layer 47's selection and nothing else — a sample, and the only one
// available without a read-back per layer.
func (g *Graph) PassExperts() int {
	if g.moe == nil {
		return 0
	}
	seen := make(map[int32]bool, 32)
	for _, e := range g.moe.TopK() {
		seen[e] = true
	}
	return len(seen)
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
	return g.ExtendInput(Input{IDs: ids})
}

// ExtendInput is Extend over an Input, which may carry images
// (LLM-VISION.md V5, V6). A pass with an image is never a one-token decode
// step, so it never takes the recorded path.
func (g *Graph) ExtendInput(in Input) (logits, norm []float32, err error) {
	ids := in.IDs
	if g.head == nil {
		return nil, nil, fmt.Errorf("llm: this graph was staged without a head")
	}
	// P1c: a one-token pass is the same dispatches with the same bytes every
	// time, so the first one is captured (`capture`, read by flush) and every
	// later one replays it. The capture rides the ordinary path, which is
	// what makes the two arms comparable: the recorded buffer *is* the
	// sequence the first token submitted. `past > 0` because a fresh
	// sequence is a different shape: at position zero the attention block
	// rebuilds its whole pooled table, so its grid is NBlocks rather than
	// one (blockRange).
	if len(ids) == 1 && len(in.Images) == 0 && g.past > 0 && Prerecord() {
		epoch := g.decodeEpoch()
		if g.pre != nil && g.preEpoch == epoch {
			return g.extendPrerecorded(ids[0])
		}
		// The plan moved under the buffer (or there is no buffer yet). One
		// step pays the ordinary path and captures the new one.
		g.dropPrerecorded()
		g.capture, g.capEpoch = true, epoch
	}
	// One command buffer for the whole of it, head included (L7d): the
	// layers, the head mixer and the projection are recorded and submitted
	// once, and the two read-backs below are after the flush because nothing
	// has run until then.
	g.record(len(ids))
	defer func() {
		if ferr := g.flush(); ferr != nil && err == nil {
			logits, norm, err = nil, nil, ferr
		}
	}()
	if err := g.hiddenIn(in); err != nil {
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

// extendPrerecorded is Extend for one token over the captured command buffer
// (P1c): the host work of a decode step — the two gathers, the uploads, the
// position — and then one submit of the sequence the first decode token
// recorded, instead of re-encoding 1407 dispatches to build it again.
//
// It mirrors the host side of appendN + hidden + Extend in their order, and
// only that side: everything the dispatches used to be re-recorded for is
// either byte-identical every step (TestDecodeDispatchDiff) or reaches the
// GPU through a mapped write here. The Resizes stay because a pass that
// followed a prefill would otherwise run with the prompt's row count in the
// blocks' host-side state — they are field sets and pad clears, not
// dispatches.
func (g *Graph) extendPrerecorded(id int32) (logits, norm []float32, err error) {
	if g.past+1 > g.nKV {
		return nil, nil, fmt.Errorf("llm: a token at position %d of a %d-cell context", g.past, g.nKV)
	}
	if err := g.prepRope(g.slot, g.past, 1, g.spans); err != nil {
		return nil, nil, err
	}
	top := time.Now()
	t0 := top
	embd, err := g.m.Embeddings([]int32{id})
	if err != nil {
		return nil, nil, fmt.Errorf("llm: token_embd: %w", err)
	}
	g.ids = append(g.ids, id)
	var pleEmbd []float32
	if g.hasPLE {
		rows := PLERowsFrom(g.pleCfg, g.ids, g.past)
		pleEmbd, err = g.m.PLEGather(rows, g.pleCfg.NHeads, g.pleCfg.HeadDim)
		if err != nil {
			return nil, nil, fmt.Errorf("llm: per_layer_token_embd: %w", err)
		}
	}
	t0 = since(&g.Stats.Gather, t0)

	if err := g.hc.UploadInit(embd, 1); err != nil {
		return nil, nil, err
	}
	if g.attn != nil {
		if err := g.attn.SetPast(g.past); err != nil {
			return nil, nil, err
		}
		if err := g.attn.Resize(1); err != nil {
			return nil, nil, err
		}
	}
	if g.ple != nil {
		if err := g.ple.SetPast(g.past); err != nil {
			return nil, nil, err
		}
		if err := g.ple.UploadEmbd(pleEmbd, 1); err != nil {
			return nil, nil, err
		}
	}
	if g.dn != nil {
		if err := g.dn.SetPast(g.past); err != nil {
			return nil, nil, err
		}
		if err := g.dn.Resize(1); err != nil {
			return nil, nil, err
		}
	}
	if err := g.moe.Resize(1); err != nil {
		return nil, nil, err
	}
	if err := g.head.Resize(1); err != nil {
		return nil, nil, err
	}
	t0 = since(&g.Stats.Glue, t0)

	total, each, err := g.pre.Submit()
	submit := time.Since(t0)
	if err != nil {
		return nil, nil, fmt.Errorf("llm: prerecorded decode step: %w", err)
	}
	g.Stats.Dispatches += g.pre.Count()
	g.Stats.GPU += total
	g.Stats.Submit += submit
	if g.Stats.Kinds == nil {
		g.Stats.Kinds = make(map[string]DispatchStat, len(g.preKinds))
	}
	for i, d := range each {
		s := g.Stats.Kinds[g.preKinds[i]]
		s.Count, s.GPU = s.Count+1, s.GPU+d
		g.Stats.Kinds[g.preKinds[i]] = s
		switch g.preOwner[i] {
		case ownHC:
			g.Stats.HC += d
		case ownPLE:
			g.Stats.PLE += d
		case ownDN:
			g.Stats.DeltaNet += d
		case ownAttn:
			g.Stats.Attn += d
		case ownMoE:
			g.Stats.MoE += d
		case ownHead:
			g.Stats.Head += d
		case ownMove:
			g.Stats.Move += d
		}
	}
	g.past++
	g.Stats.Runs++

	t0 = time.Now()
	logits, norm = g.head.Logits(), g.hc.Mixed()
	since(&g.Stats.Glue, t0)
	g.Stats.Total += time.Since(top)
	return logits, norm, nil
}

// ForwardRows runs the model over a prompt as a fresh sequence and hands
// every row's logits from `first` on to fn, a head-arena slab at a time.
//
// It is Forward without `inp_out_ids`, and it exists because perplexity is
// the one question in this vertical that needs a logit per token. The
// reference's graph drops every row but the last before the final mixer, so
// `hidden` moves one row to the front and runs the mixer and the head on it;
// here the mixer runs over the whole batch — it is per token, so that is the
// same arithmetic on nTok times the work — and the head then runs over slabs
// of GraphOpts.HeadRows, because a row of logits is 0.99 MB and nothing
// reads two of them at once.
//
// fn sees a slice into the head's mapped arena and must not keep it: the
// next slab overwrites it.
func (g *Graph) ForwardRows(ids []int32, first int, fn func(t int, logits []float32) error) error {
	if g.head == nil {
		return fmt.Errorf("llm: this graph was staged without a head")
	}
	if first < 0 || first >= len(ids) {
		return fmt.Errorf("llm: first row %d of %d tokens", first, len(ids))
	}
	if err := g.Reset(); err != nil {
		return err
	}
	top := time.Now()
	g.record(len(ids))
	if err := g.appendN(ids, g.nLayer); err != nil {
		_ = g.flush()
		return err
	}
	t0 := time.Now()
	if err := g.hc.Run(2*g.nLayer, false); err != nil {
		_ = g.flush()
		return fmt.Errorf("llm: head mixer: %w", err)
	}
	g.blk(&g.Stats.HC, t0)
	if err := g.flush(); err != nil {
		return err
	}

	// The head, a slab at a time. Each slab is its own command buffer
	// because the logits have to be read between them, and nothing may read
	// a device arena while a pass is being recorded (flush's one rule).
	slab, vocab := g.head.MaxRows(), g.head.Vocab()
	for r0 := first; r0 < len(ids); r0 += slab {
		n := minInt(slab, len(ids)-r0)
		g.record(n)
		err := func() error {
			if err := g.head.Resize(n); err != nil {
				return err
			}
			if err := g.move.Move(g.head.InPort(), g.hc.MixedRowPort(r0), n); err != nil {
				return err
			}
			return g.head.Run()
		}()
		if ferr := g.flush(); err == nil {
			err = ferr
		}
		if err != nil {
			return err
		}
		lg := g.head.Logits()
		for i := 0; i < n; i++ {
			if err := fn(r0+i, lg[i*vocab:(i+1)*vocab]); err != nil {
				return err
			}
		}
	}
	g.Stats.Total += time.Since(top)
	return nil
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
	return g.HiddenExtendInput(Input{IDs: ids})
}

// HiddenExtendInput is HiddenExtend over an Input, which may carry images.
func (g *Graph) HiddenExtendInput(in Input) ([]float32, error) {
	ids := in.IDs
	top := time.Now()
	g.record(len(ids))
	if err := g.hiddenIn(in); err != nil {
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
func (g *Graph) hidden(ids []int32) error { return g.hiddenIn(Input{IDs: ids}) }

func (g *Graph) hiddenIn(in Input) error {
	ids := in.IDs
	if err := g.appendIn(in, g.nLayer); err != nil {
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

// PrefetchPLE faults in the n-gram table's pages for `next`, the batch that
// will follow `prev`, on a goroutine of its own, and returns at once (P17).
//
// A prompt's rows are 16 a token at random offsets into a 28.80 GB mapping
// that is mostly not in the page cache, so a fresh 2048-token chunk is ~32 000
// major faults: **16 ms of host gather a chunk at depth 0, where the bench's
// warm-up pass has touched the same text, and 93 ms at 32 000 cells, where it
// has not** — 3-4% of a prefill pass, spent with the GPU idle. But a prompt's
// tokens are all known before its first chunk runs, so the next chunk's faults
// can be taken while the device works on this one, when the host is otherwise
// waiting on a fence. The gather it does is discarded; what it leaves behind is
// warm pages, and the real gather then reads them at ~12 µs a token.
//
// It changes no value anywhere — the rows are the table's, whoever faulted
// them — and a wrong guess costs page cache and nothing else. `prev` supplies
// the n-gram's predecessors; only its last NGram-1 tokens are read.
// LLM_PLE_PREFETCH=0 is the control arm.
func (g *Graph) PrefetchPLE(prev, next []int32) {
	if !g.hasPLE || len(next) == 0 || os.Getenv("LLM_PLE_PREFETCH") == "0" {
		return
	}
	keep := minInt(len(prev), g.pleCfg.NGram-1)
	ids := make([]int32, 0, keep+len(next))
	ids = append(ids, prev[len(prev)-keep:]...)
	ids = append(ids, next...)
	rows := PLERowsFrom(g.pleCfg, ids, keep)
	g.pleAhead.Add(1)
	go func() {
		defer g.pleAhead.Done()
		_, _ = g.m.PLEGather(rows, g.pleCfg.NHeads, g.pleCfg.HeadDim)
	}()
}

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
	return g.AppendInputN(Input{IDs: ids}, nLayer)
}

// AppendInputN is AppendN over an Input, which may carry images.
func (g *Graph) AppendInputN(in Input, nLayer int) error {
	ids := in.IDs
	top := time.Now()
	g.record(len(ids))
	if err := g.appendIn(in, nLayer); err != nil {
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
	return g.appendIn(Input{IDs: ids}, nLayer)
}

// appendIn is appendN over an Input, which may carry images.
func (g *Graph) appendIn(in Input, nLayer int) error {
	ids := in.IDs
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
	spans, err := g.inputSpans(in)
	if err != nil {
		return err
	}
	if len(in.Images) > 0 && g.spec {
		return fmt.Errorf("llm: a speculating graph takes text only")
	}
	// What a rewindable pass has to be able to put back (P5c). Before
	// anything moves, and a no-op unless Speculate is on.
	if err := g.arm(nTok); err != nil {
		return err
	}
	// The rotary rows this pass reads, while nothing is in flight.
	if err := g.prepRope(g.slot, g.past, nTok, spans); err != nil {
		return err
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
	// The images' rows over the pads' (V6). The hyper-connection init reads
	// nothing else of the input, so this is the whole of the scatter.
	for _, im := range in.Images {
		if im.Embd != nil {
			copy(embd[im.At*g.m.Config.NEmbd:], im.Embd)
		}
	}
	g.ids = append(g.ids, ids...)
	var pleEmbd []float32
	if g.hasPLE {
		// The tail's rows only. A position's n-gram depends on the two tokens
		// behind it and nothing further, so `PLERowsFrom` walks from `past`
		// and reads back across the boundary — which is the same rows the
		// whole-sequence form returned and, at 128 000 cells, 2.05 million
		// fewer of them (P15-1).
		rows := PLERowsFrom(g.pleCfg, g.ids, g.past)
		g.pleAhead.Wait()
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

	if _, err := g.layers(nTok, nLayer, pleEmbd, t0); err != nil {
		return err
	}
	g.past += nTok
	g.spans = spans
	g.Stats.Runs++
	return nil
}

// Input is one pass's rows: token ids and, among them, images
// (LLM-VISION.md V5, V6).
type Input struct {
	IDs    []int32
	Images []InputImage
}

// InputImage is one image's run of rows in Input.IDs: GridH x GridW merged
// tokens from row At, in raster order, every one of whose ids must be the
// image-pad token, because that is what the PLE hashes at an image's cells
// (HF's rule; TestPLEImagePrompt). Hash identifies the pixels.
type InputImage struct {
	At           int
	GridH, GridW int
	Hash         uint64
	// Embd is the vision tower's merged rows, [GridH*GridW][NEmbd], which
	// replace the pad's token embedding at the image's rows (V6): the rows
	// HF scatters into `inputs_embeds` at the image-pad slots. nil keeps the
	// pad's own embedding, which only a test that is about positions wants.
	Embd []float32

	// textPositions ropes the image's rows as text, 0..n-1 past the last
	// position, which is what vLLM's qwen4_exp does. It is the control
	// LLM-VISION.md V0 ruled against (TestGraphImagePrefix) and nothing else.
	textPositions bool
}

// inputSpans is the sequence's spans once `in` has run, checked.
func (g *Graph) inputSpans(in Input) ([]ImageSpan, error) {
	if len(in.Images) == 0 {
		return g.spans, nil
	}
	spans := slices.Clip(g.spans)
	next := 0
	for _, im := range in.Images {
		s := ImageSpan{Start: g.past + im.At, End: g.past + im.At + im.GridH*im.GridW,
			GridH: im.GridH, GridW: im.GridW, Hash: im.Hash}
		if err := s.valid(); err != nil {
			return nil, err
		}
		if im.At < next || im.At+im.GridH*im.GridW > len(in.IDs) {
			return nil, fmt.Errorf("llm: an image at rows [%d, %d) of a %d-row pass, after row %d",
				im.At, im.At+im.GridH*im.GridW, len(in.IDs), next)
		}
		for r := im.At; r < im.At+im.GridH*im.GridW; r++ {
			if g.hasPLE && in.IDs[r] != g.pleCfg.Image {
				return nil, fmt.Errorf("llm: row %d of an image holds id %d, not the image pad %d",
					r, in.IDs[r], g.pleCfg.Image)
			}
		}
		if w := g.m.Config.NEmbd; im.Embd != nil && len(im.Embd) != im.GridH*im.GridW*w {
			return nil, fmt.Errorf("llm: an image's rows are %d values, want %dx%d rows of %d",
				len(im.Embd), im.GridH, im.GridW, w)
		}
		next = im.At + im.GridH*im.GridW
		if im.textPositions {
			continue
		}
		spans = append(spans, s)
	}
	return spans, nil
}

// prepRope writes the rotary rows of cells [from, from+n) of a slot's table
// when they are not the text table's: when the sequence has an image, or when
// an earlier sequence in the slot left image rows there. A text-only
// sequence in a clean slot writes nothing, which is every run before V5.
func (g *Graph) prepRope(slot, from, n int, spans []ImageSpan) error {
	if g.attn == nil || (len(spans) == 0 && from >= g.ropeHW[slot]) {
		return nil
	}
	if err := g.attn.SetRopeRows(slot, from, cellPositions(spans, from, n)); err != nil {
		return err
	}
	switch {
	case len(spans) > 0:
		g.ropeHW[slot] = max(g.ropeHW[slot], from+n)
	case from+n >= g.ropeHW[slot]:
		// A text sequence wrote every row it holds as it ran, so what is
		// below `from` is clean already.
		g.ropeHW[slot] = from
	}
	return nil
}

// layers is a pass's layer loop, over whatever the embedding upload and the
// blocks' positions (or batch rows) have set up: nTok rows through the first
// nLayer layers, ending with the last combine flushed. It returns the clock
// it left off at.
func (g *Graph) layers(nTok, nLayer int, pleEmbd []float32, t0 time.Time) (time.Time, error) {
	// P1a's fusion, as a scheduling rule rather than a kernel choice. A
	// combine and the next mixer's norm are the same 2560 values per
	// (token, stream) read, written and read again, so `RunCombineMix` does
	// both in one dispatch — but only where nothing else touches the wide
	// residual in between. So a combine is *held* rather than issued, and
	// whatever comes next either absorbs it or flushes it: the PLE block,
	// which moves the residual out of this arena and back, flushes; the end
	// of the pass flushes, because the final mixer is reached through a row
	// move in `hidden`. Everything else absorbs, which is 94 of the 96.
	pending := -1
	flushHC := func() error {
		if pending < 0 {
			return nil
		}
		m := pending
		pending = -1
		return g.hc.RunCombine(m)
	}
	fuse := HCFuse()
	mixHC := func(m int) error {
		if pending < 0 {
			return g.hc.Run(m, false)
		}
		if !fuse {
			if err := flushHC(); err != nil {
				return err
			}
			return g.hc.Run(m, false)
		}
		p := pending
		pending = -1
		return g.hc.RunCombineMix(p, m)
	}

	for l := 0; l < nLayer; l++ {
		if g.hasPLE && g.pleCfg.IsPLE(l) {
			// The one block that reads and writes the wide residual, so the
			// only place the 10240-wide tensor crosses a boundary — twice,
			// once a graph.
			if err := flushHC(); err != nil {
				return t0, fmt.Errorf("llm: layer %d ple combine: %w", l, err)
			}
			t0 = g.blk(&g.Stats.HC, t0)
			if err := g.ple.UploadEmbd(pleEmbd, nTok); err != nil {
				return t0, err
			}
			t0 = since(&g.Stats.Glue, t0)
			if err := g.move.Move(g.ple.ResPort(), g.hc.ResPort(), nTok); err != nil {
				return t0, err
			}
			t0 = g.blk(&g.Stats.Move, t0)
			if err := g.ple.Run(); err != nil {
				return t0, err
			}
			t0 = g.blk(&g.Stats.PLE, t0)
			if err := g.move.Move(g.hc.ResPort(), g.ple.ResPort(), nTok); err != nil {
				return t0, err
			}
			t0 = g.blk(&g.Stats.Move, t0)
		}

		// The attention half.
		if err := mixHC(2 * l); err != nil {
			return t0, fmt.Errorf("llm: layer %d attn mix: %w", l, err)
		}
		t0 = g.blk(&g.Stats.HC, t0)
		t1, err := g.sublayer(l, nTok, t0)
		if err != nil {
			return t0, err
		}
		t0 = t1
		pending = 2 * l

		// The FFN half, which every layer has.
		if err := mixHC(2*l + 1); err != nil {
			return t0, fmt.Errorf("llm: layer %d ffn mix: %w", l, err)
		}
		t0 = g.blk(&g.Stats.HC, t0)
		if err := g.moe.Resize(nTok); err != nil {
			return t0, err
		}
		t0 = since(&g.Stats.Glue, t0)
		if err := g.move.Move(g.moe.InPort(), g.hc.MixedPort(), nTok); err != nil {
			return t0, err
		}
		t0 = g.blk(&g.Stats.Move, t0)
		if err := g.moe.Run(l); err != nil {
			return t0, fmt.Errorf("llm: layer %d moe: %w", l, err)
		}
		t0 = g.blk(&g.Stats.MoE, t0)
		if err := g.move.Move(g.hc.BlockOutPort(), g.moe.OutPort(), nTok); err != nil {
			return t0, err
		}
		t0 = g.blk(&g.Stats.Move, t0)
		pending = 2*l + 1
	}
	if err := flushHC(); err != nil {
		return t0, fmt.Errorf("llm: last combine: %w", err)
	}
	return g.blk(&g.Stats.HC, t0), nil
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

// Prerecorded reports whether a captured decode step is live — the P1c fast
// path, engaged from the second one-token Extend on.
func (g *Graph) Prerecorded() bool { return g.pre != nil }

// decodeEpoch names the decode plan a one-token Extend would record right now.
// Only the attention block has a depth-dependent plan (P15's gather); every
// other block's decode step is the same dispatches at every position.
func (g *Graph) decodeEpoch() int {
	if g.attn == nil {
		return 0
	}
	return g.attn.DecodeEpoch(g.past + 1)
}

// dropPrerecorded releases the captured decode step, if any; the next
// one-token Extend records and captures a fresh one.
func (g *Graph) dropPrerecorded() {
	if g.pre != nil {
		g.pre.Destroy()
		g.pre, g.preKinds, g.preOwner = nil, nil, nil
	}
}

// Destroy releases every block.
func (g *Graph) Destroy() {
	g.pleAhead.Wait()
	g.dropPrerecorded()
	for i := range g.parked {
		if g.parked[i].pre != nil {
			g.parked[i].pre.Destroy()
			g.parked[i].pre = nil
		}
	}
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
