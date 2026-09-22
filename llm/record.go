package llm

// One command buffer a **pass**, and the attribution that has to survive it:
// LLM.md L7d.
//
// L7c-5 took the graph from a submit a dispatch to a submit a block — ~1130
// to ~490 — and priced the rest at 31-43 us each, which is ~15-21 ms of a
// 99 ms decode token. Nothing in the shim ever prevented going further: it
// binds each dispatch's own pipeline, descriptor set and push constants, so a
// sequence may mix blocks as freely as it mixes kernels of one block. What
// stopped it was the *measurement*. Every number this vertical has been tuned
// by comes from the host wall clock around a block's `Run`, and a block that
// no longer submits has no wall clock of its own.
//
// So the recorder moves both. Each block's `Run` appends to it instead of
// submitting when one is installed, and the submit asks the shim for a
// timestamp after **every** dispatch (vk.DispatchMultiMarked) rather than only
// at the two ends. The per-block figures that come back are then GPU time
// inside one command buffer rather than host time around many, which is the
// same thing llama.cpp's own `GGML_VK_PERF_LOGGER` reports for its graph — the
// baseline half of every comparison in LLM.md — and strictly the better
// number: it cannot be inflated by a fence wait, a Go allocation or a page
// fault on the host.

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"strix-halo-vulkan/vk"
)

// The block names a dispatch can be attributed to. They are the rows of
// GraphStats, and a dispatch carries one because a command buffer does not.
const (
	ownHC   = "hc"
	ownPLE  = "ple"
	ownDN   = "dn"
	ownAttn = "attn"
	ownMoE  = "moe"
	ownHead = "head"
	ownMove = "move"
)

// maxBatch is how many dispatches go into one command buffer. It is a bound
// on the shim's query pool (SHIM_QUERY_SLOTS) and not a tuning knob: a whole
// prefill pass is ~1200 dispatches, so this is one submit for a decode token
// and two for a long prompt, against the ~490 either used to take.
// LLM_MAX_BATCH overrides it, which is how P0 asked whether a hang was about
// the size of one submit rather than about any dispatch in it.
var maxBatch = envInt("LLM_MAX_BATCH", 1024)

// minBatch keeps the row-scaled bound below from degenerating: a submit costs
// ~40 us, so even at 32 dispatches the boundary is under 2% of a pass.
const minBatch = 32

// submitBudget is how much GPU time one command buffer may hold.
//
// **P0.** It is not the query pool and it is not the dispatch count: a submit
// that occupies the gfx ring for more than ~2 s is killed by amdgpu's ring
// watchdog. The kernel says `ring gfx_0.0.0 timeout, signaled seq=N, emitted
// seq=N+3`, resets the ring, and the reset *force-signals the fence* -- so
// vkQueueSubmit returns, vkWaitForFences returns VK_SUCCESS, its own 20-second
// timeout never fires, and the only trace left is the timestamps the killed
// dispatches never wrote. Reading those with VK_QUERY_RESULT_WAIT_BIT is an
// unbounded userspace poll inside RADV, which is the 100%-of-a-core spin that
// looked like a hang in the submit (see shim_query_results_deadline).
//
// Measured at 48 layers, 1024 dispatches a submit: 2560 rows is 1.886 s and
// runs, 2688 rows is 1.964 s and runs, 2816 rows is 2.033 s and is reset. The
// same 1273 dispatches in *one* submit are 787 ms at 512 rows and run, and are
// reset at 2688 rows -- where two submits of 1024 and 249 had just run the same
// work. So the bound is time, and half the cliff is the budget.
const submitBudget = 1000 * time.Millisecond

// A pass's cost is affine in the rows, not proportional to them: fitting the
// whole-pass figures of 789.5 ms at 512 rows, 2342.4 at 2560, 2434.2 at 2688
// and 2687.5 at 3072 over 1273 dispatches gives 322 us a dispatch plus 582 ns
// a dispatch-row, and predicts all four to within 1.5%. It puts a 1024-dispatch
// submit at 2816 rows at 2.008 s, which is the row count that is reset.
const (
	usPerDispatch    = 322
	nsPerDispatchRow = 582
)

// nsPerRowCell is the term P0's fit could not have had, because P0 only ever
// prefilled from cell zero: **a dispatch's cost depends on how deep the cache
// already is, and attention is the whole of it** (P11-5). Every other block
// of the model is flat in depth; `attn.attn`, `attn.select`, `attn.score` and
// `attn.expand` are not, and together they are 97% of what a prefill token
// gains between depth zero and 64k.
//
// From P12's shipped ladder at 2048 rows: 1207.7 tok/s at depth zero is a
// 1.696 s pass, 667.7 at 64 000 cells is a 3.067 s one, so the depth term is
// 1.371 s over rows*cells = 2048*64000, which is 10.46 ns each. 11 is that
// with a little margin, and margin is what this constant is for -- a submit
// that runs long is a ring reset and a dead process, where one that runs
// short is ~40 us of fence wait.
//
// **This is what stopped 128k completing.** At 64 000 cells a 2048-row pass
// is 3.07 s in two chunks of ~1.53 s, already three quarters of the way to
// amdgpu's 2 s watchdog with a budget that believed it was spending 0.86; at
// ~115 000 the same two chunks cross it and the ring is reset mid-fill. The
// symptom is P0's -- "N of 1025 timestamp slots never became ready" -- with
// depth rather than row count as the variable that reached it.
const nsPerRowCell = 11

// attnDispatchesAtFullDepth is how many dispatches carry that term in the
// whole model: the shipped checkpoint has 12 full-attention layers of 48 and
// each records six or seven, which is the graph the fit above was measured
// over. Charging `whole/78` to each attention dispatch a pass *actually*
// holds is what scales the model to the graph -- a four-layer prefix with one
// attention layer records six of them and is charged six seventy-eighths of
// the whole term, which is what it costs.
const attnDispatchesAtFullDepth = 78

// dispatchCost is what one dispatch of a pass over this many rows costs.
func dispatchCost(rows int) time.Duration {
	return usPerDispatch*time.Microsecond + time.Duration(nsPerDispatchRow*rows)*time.Nanosecond
}

// depthCost is what one *attention* dispatch adds on top of that when the
// cache already holds `past` cells. It is zero on a fresh sequence, which is
// every measurement P0 took.
func depthCost(rows, past int) time.Duration {
	if past <= 0 || rows <= 0 {
		return 0
	}
	whole := time.Duration(int64(rows)*int64(past)*nsPerRowCell) * time.Nanosecond
	return whole / attnDispatchesAtFullDepth
}

// batchFor is how many dispatches fit in submitBudget at this many rows. At
// one row -- every decode step -- it is maxBatch, so nothing about decode
// changes; it starts biting at ~1700 rows and is ~200 at 8192.
func batchFor(rows int) int {
	if rows < 1 {
		rows = 1
	}
	n := int(submitBudget / dispatchCost(rows))
	return min(max(n, minBatch), maxBatch)
}

// BatchFor is that bound at a given number of rows, exported for a caller
// reporting how many command buffers a pass took.
func BatchFor(rows int) int { return batchFor(rows) }

// batchTimes prints each submit's GPU time, which is how the budget above is
// checked against the ring rather than against the model that predicts it.
var batchTimes = os.Getenv("LLM_BATCH_TIMES") != ""

func envInt(name string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil && v > 0 {
		return v
	}
	return def
}

// recorder collects a pass's dispatches with the block each came from.
type recorder struct {
	d     []vk.MultiDispatch
	owner []string
	// kind is the block's own name for each dispatch -- "moe.up",
	// "attn.qkv" -- which is what P1 needed and what every block's `graph`
	// method has always built for its error messages. A block name is where
	// the time is *not*: the MoE is 11.6 ms of a 31.7 ms token and says
	// nothing about which of its nine kernels that is.
	kind []string
	// rows is what the pass is passing, which is what a dispatch's cost --
	// and so how many of them fit in one submit -- scales with.
	rows int
	// past is how many cells the cache already holds, which is the *other*
	// thing a dispatch's cost scales with and the one P0's fit never saw
	// (see nsPerRowCell). Only the attention dispatches pay it.
	past int
}

// add appends one block's sequence, with the block's own label per dispatch.
// A nil recorder is the direct path, which is what every block does when the
// graph is not batching and what the per-block benchmarks always do.
//
// `kinds` may be shorter than `d` or nil -- the mover has one dispatch and no
// label of its own -- in which case the owner's name stands in.
func (r *recorder) add(owner string, kinds []string, d []vk.MultiDispatch) bool {
	if r == nil {
		return false
	}
	r.d = append(r.d, d...)
	for i := range d {
		r.owner = append(r.owner, owner)
		k := owner
		if i < len(kinds) {
			k = owner + "." + kinds[i]
		}
		r.kind = append(r.kind, k)
	}
	return true
}

func (r *recorder) len() int {
	if r == nil {
		return 0
	}
	return len(r.d)
}

// captureSink, when set, is handed a copy of every recorded sequence at the
// moment it is submitted — pipeline, grid and push-constant bytes, with each
// dispatch's label. It exists for P1c's first question: which dispatches of a
// decode step actually differ from the last step's, asked by capturing two
// consecutive steps and diffing them. Test-only; nothing in the model sets it.
var captureSink func(d []vk.MultiDispatch, kind []string)

// DispatchStat is one kernel's share of a pass: how many dispatches carried
// that label and what they cost on the GPU.
type DispatchStat struct {
	Count int
	GPU   time.Duration
}

// submit runs everything recorded, in order, and returns the GPU time each
// block's dispatches took, and the same split by the label each carried. It
// chunks at batchFor(rows), which costs one more fence wait per chunk and
// keeps the marks.
func (r *recorder) submit() (map[string]time.Duration, map[string]DispatchStat, time.Duration, error) {
	byOwner := make(map[string]time.Duration, 8)
	byKind := make(map[string]DispatchStat, 64)
	var total time.Duration
	if captureSink != nil {
		d := make([]vk.MultiDispatch, len(r.d))
		for i, md := range r.d {
			d[i] = md
			d[i].PushConstants = append([]byte(nil), md.PushConstants...)
		}
		captureSink(d, append([]string(nil), r.kind...))
	}
	r.dumpRange()
	for i := 0; i < len(r.d); {
		j := r.chunkEnd(i)
		// Before the submit, because `observe` moves the scale the model is
		// read through and the two have to be the same prediction.
		want := r.chunkCost(i, j)
		el, each, err := vk.DispatchMultiMarked(r.d[i:j], true)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("llm: dispatches %d-%d: %w", i, j-1, err)
		}
		if batchTimes {
			fmt.Fprintf(os.Stderr, "batch %d-%d: %v on the GPU, modelled %v\n", i, j-1, el, want)
		}
		r.observe(want, el)
		total += el
		if each == nil {
			// The shim refused the marks, which at this size it cannot; the
			// pass still ran, so report it as one block rather than losing it.
			byOwner[r.owner[i]] += el
			s := byKind[r.kind[i]]
			s.Count, s.GPU = s.Count+j-i, s.GPU+el
			byKind[r.kind[i]] = s
		} else {
			for k, d := range each {
				byOwner[r.owner[i+k]] += d
				s := byKind[r.kind[i+k]]
				s.Count, s.GPU = s.Count+1, s.GPU+d
				byKind[r.kind[i+k]] = s
			}
		}
		i = j
	}
	r.d, r.owner, r.kind = r.d[:0], r.owner[:0], r.kind[:0]
	return byOwner, byKind, total, nil
}

// chunkCost is what the model above says dispatches [i, j) cost: the affine
// row term for every one of them and the depth term for the attention ones,
// scaled by whatever the last submit actually measured.
func (r *recorder) chunkCost(i, j int) time.Duration {
	base := dispatchCost(r.rows)
	deep := depthCost(r.rows, r.past)
	var c time.Duration
	for k := i; k < j; k++ {
		c += base
		if r.owner[k] == ownAttn {
			c += deep
		}
	}
	return time.Duration(float64(c) * costScale())
}

// chunkEnd is where the next command buffer ends: as many dispatches as fit
// in submitBudget, counted by their own predicted cost rather than by a
// single number for the whole pass.
//
// **Why a cost walk and not a count.** P0's `batchFor(rows)` divided the
// budget by one dispatch's cost, which is right exactly when every dispatch
// costs the same -- and at depth they do not. Attention carries a term
// proportional to `rows * past` and every other block carries none, so a
// chunk that happens to land on the twelve attention layers is several times
// a chunk that lands on the MoE. Walking the recorded sequence and cutting
// where the *accumulated* prediction reaches the budget charges each
// dispatch what its own label costs, needs no second constant for how the
// blocks are interleaved, and degenerates to exactly P0's answer at past = 0.
func (r *recorder) chunkEnd(i int) int {
	base := dispatchCost(r.rows)
	deep := depthCost(r.rows, r.past)
	scale := costScale()
	budget, acc := submitBudget, time.Duration(0)
	j := i
	for ; j < len(r.d) && j-i < maxBatch; j++ {
		c := base
		if r.owner[j] == ownAttn {
			c += deep
		}
		acc += time.Duration(float64(c) * scale)
		if acc > budget && j-i >= minBatch {
			break
		}
	}
	return max(j, min(i+1, len(r.d)))
}

// The adaptive half, and the reason this is safe to be wrong about.
//
// The constants above are a fit, and a fit is a hypothesis: a kernel that
// gets faster makes them pessimistic (more submits, ~40 us each, nobody
// notices) and one that gets slower makes them fatal (a ring reset and a dead
// process). So every submit checks itself -- the recorder already gets its
// GPU time back for the attribution -- and the ratio it measures scales the
// next prediction. One submit is enough to converge, a prefill is many
// submits, and the scale is clamped so that a single anomalous reading
// cannot take the budget anywhere dangerous.
//
// It is deliberately global and deliberately sticky: `g.rec` is a fresh
// recorder per pass, and what this has to survive is exactly the pass
// boundary -- the fill loop that walks a 128k prompt is hundreds of them.
//
// **And it only ever corrects upwards**, which is not a detail. A decode step
// is 1443 dispatches over one row, so the affine model charges it 322 us each
// and predicts ~330 ms where the whole step is ~36 ms on the GPU -- the
// per-dispatch constant was fitted at prefill row counts and is an order out
// at one row, which costs nothing because `maxBatch` bounds a decode step
// anyway. A scale that believed those measurements would converge towards
// 0.25 across a conversation's decode steps and then hand the *next prefill*
// four times the budget it asked for, which is a ring reset. Clamping the
// floor at 1.0 makes the correction only ever more conservative: the model is
// already 10-20% pessimistic at prefill, and what this exists to catch is the
// day some kernel becomes slower than the fit, not the day one is faster.
var (
	costScaleMu  sync.Mutex
	costScaleVal = 1.0
)

func costScale() float64 {
	costScaleMu.Lock()
	defer costScaleMu.Unlock()
	return costScaleVal
}

// observe folds one submit's measured GPU time into the scale. Only a chunk
// big enough to be about the dispatches rather than about the fence wait is
// worth learning from.
func (r *recorder) observe(modelled, actual time.Duration) {
	if modelled <= 0 || actual <= 0 || modelled < 10*time.Millisecond {
		return
	}
	ratio := float64(actual) / float64(modelled)
	costScaleMu.Lock()
	defer costScaleMu.Unlock()
	// Up fast, down slow: under-predicting is the direction that resets the
	// ring, so a submit that ran long is believed at once and one that ran
	// short is averaged in. The floor is 1.0 -- see above.
	next := costScaleVal * ratio
	if next < costScaleVal {
		next = 0.75*costScaleVal + 0.25*next
	}
	costScaleVal = min(max(next, 1), 8)
}

// dumpRange prints the block and grid of a slice of the recorded sequence
// before it is submitted, which is the only moment a dispatch that hangs the
// GPU can still be named. LLM_DISPATCH_DUMP=lo:hi selects the range; the
// dispatch a pass stopped at is the one after the last timestamp slot that
// came back (slot k is written after dispatch k-1).
func (r *recorder) dumpRange() {
	spec := os.Getenv("LLM_DISPATCH_DUMP")
	if spec == "" {
		return
	}
	lo, hi := 0, len(r.d)
	if a, b, ok := strings.Cut(spec, ":"); ok {
		if v, err := strconv.Atoi(a); err == nil {
			lo = v
		}
		if v, err := strconv.Atoi(b); err == nil {
			hi = v
		}
	}
	lo, hi = max(lo, 0), minInt(hi, len(r.d))
	fmt.Fprintf(os.Stderr, "dispatch dump: %d recorded, showing %d-%d\n", len(r.d), lo, hi-1)
	for i := lo; i < hi; i++ {
		fmt.Fprintf(os.Stderr, "  %4d %-5s groups %dx%d pc %d\n",
			i, r.owner[i], r.d[i].GroupsX, r.d[i].GroupsY, len(r.d[i].PushConstants))
	}
}
