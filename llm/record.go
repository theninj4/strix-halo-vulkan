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

// dispatchCost is what one dispatch of a pass over this many rows costs.
func dispatchCost(rows int) time.Duration {
	return usPerDispatch*time.Microsecond + time.Duration(nsPerDispatchRow*rows)*time.Nanosecond
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
	batch := batchFor(r.rows)
	for i := 0; i < len(r.d); i += batch {
		j := minInt(i+batch, len(r.d))
		el, each, err := vk.DispatchMultiMarked(r.d[i:j], true)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("llm: dispatches %d-%d: %w", i, j-1, err)
		}
		if batchTimes {
			fmt.Fprintf(os.Stderr, "batch %d-%d: %v on the GPU\n", i, j-1, el)
		}
		total += el
		if each == nil {
			// The shim refused the marks, which at this size it cannot; the
			// pass still ran, so report it as one block rather than losing it.
			byOwner[r.owner[i]] += el
			s := byKind[r.kind[i]]
			s.Count, s.GPU = s.Count+j-i, s.GPU+el
			byKind[r.kind[i]] = s
			continue
		}
		for k, d := range each {
			byOwner[r.owner[i+k]] += d
			s := byKind[r.kind[i+k]]
			s.Count, s.GPU = s.Count+1, s.GPU+d
			byKind[r.kind[i+k]] = s
		}
	}
	r.d, r.owner, r.kind = r.d[:0], r.owner[:0], r.kind[:0]
	return byOwner, byKind, total, nil
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
