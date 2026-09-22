package llm

// P1c's first question, asked empirically: which dispatches of a decode step
// differ from the previous step's?
//
// The pre-recorded decode command buffer (LLM2 idea 2, priced by P1 at
// 1.127 ms of host recording plus 0.828 ms of hand-over a token) rests on the
// claim that the decode graph is shape-stable — same dispatches, same
// pipelines, same grids, and push constants that vary only where the position
// enters. This test captures two consecutive one-token Extends off the
// four-layer prefix and diffs them dispatch for dispatch, naming every push
// field that moved via reflection over the push block. The output is the work
// list: every named field is one the shaders must learn to read from a
// per-token scalars buffer before the command buffer can be recorded once.
//
//	go test ./llm/ -v -run TestDecodeDispatchDiff   # four layers, ~9 GB
//
// The Extends feed *different* token ids on purpose: a token-dependent
// push constant or grid would be fatal to the whole idea (the token must
// enter only through the arenas the host writes), and this is the test that
// would catch it.

import (
	"bytes"
	"fmt"
	"reflect"
	"sort"
	"testing"

	"strix-halo-vulkan/vk"
)

// TestPrerecordedDecodeIsTheRecordedDecode is P1c's gate: greedy decode over
// the captured command buffer produces bit-for-bit the logits — and so the
// tokens — of the arm that re-records every step. Both arms run off one
// staged four-layer graph, the control first so its own first decode token
// cannot have been a replay; the prerecorded arm's fresh prefill after a
// Reset also exercises the capture surviving a sequence boundary, which is
// what serve does between requests.
//
// The warm-up run exists because the comparison needs a repeatable baseline
// and the *first* run over freshly allocated arenas is not it: a fresh-arena
// run diverges from every later run of the same tokens at ~1.6e-5 in the
// logits from about the fifth decode step, prerecording or not (measured
// control-vs-control while this gate was being written). That drift predates
// P1c and is not its business; runs two onward are bit-repeatable, so the
// gate compares those.
func TestPrerecordedDecodeIsTheRecordedDecode(t *testing.T) {
	const steps = 48
	g, _, ids := graphFixture(t, GraphOpts{Layers: 4})

	run := func(pre bool) ([]int32, [][]float32) {
		if pre {
			t.Setenv("LLM_NO_PRERECORD", "")
		} else {
			t.Setenv("LLM_NO_PRERECORD", "1")
		}
		logits, _, err := g.Forward(ids)
		if err != nil {
			t.Fatal(err)
		}
		var got []int32
		var all [][]float32
		for i := 0; i < steps; i++ {
			id := int32(0)
			for j := range logits {
				if logits[j] > logits[id] {
					id = int32(j)
				}
			}
			got = append(got, id)
			if logits, _, err = g.Extend([]int32{id}); err != nil {
				t.Fatal(err)
			}
			all = append(all, append([]float32(nil), logits...))
		}
		return got, all
	}

	run(false) // the warm-up: see above
	wantIds, wantLogits := run(false)
	if g.Prerecorded() {
		t.Fatal("the control armed the prerecorded path")
	}
	gotIds, gotLogits := run(true)
	if !g.Prerecorded() {
		t.Fatal("the prerecorded arm never captured a decode step")
	}

	for i := range wantIds {
		if gotIds[i] != wantIds[i] {
			t.Fatalf("token %d: %d against the control's %d", i, gotIds[i], wantIds[i])
		}
	}
	for s := range wantLogits {
		for j := range wantLogits[s] {
			if gotLogits[s][j] != wantLogits[s][j] {
				t.Fatalf("step %d logit %d: %g against the control's %g — the replay is not the recording",
					s, j, gotLogits[s][j], wantLogits[s][j])
			}
		}
	}
	t.Logf("%d greedy tokens and %d logit rows, bit-identical to the re-recording arm", len(wantIds), len(wantLogits))
}

// pcFieldNames is the push block's fields in declaration order, one per
// dword, which is what maps a differing byte back to a name.
func pcFieldNames() []string {
	pt := reflect.TypeOf(push{})
	names := make([]string, pt.NumField())
	for i := range names {
		names[i] = pt.Field(i).Name
	}
	return names
}

func TestDecodeDispatchDiff(t *testing.T) {
	// The question here is what the *recording* path produces each step, so
	// the replay that P1c builds on this very stability is switched off.
	t.Setenv("LLM_NO_PRERECORD", "1")
	g, _, ids := graphFixture(t, GraphOpts{Layers: 4})

	if _, _, err := g.Forward(ids); err != nil {
		t.Fatal(err)
	}

	var d []vk.MultiDispatch
	var kind []string
	captureSink = func(cd []vk.MultiDispatch, ck []string) {
		d = append(d, cd...)
		kind = append(kind, ck...)
	}
	defer func() { captureSink = nil }()

	grab := func(id int32) ([]vk.MultiDispatch, []string) {
		d, kind = nil, nil
		if _, _, err := g.Extend([]int32{id}); err != nil {
			t.Fatal(err)
		}
		return d, kind
	}

	// Forty steps against the first, not two against each other: a pooled
	// indexer block completes only every Ratio tokens and the DeltaNet's
	// convolution ring has a period of its own, so two adjacent steps could
	// agree by coincidence where step n and step n+Ratio do not.
	const steps = 40
	a, ak := grab(ids[0])
	t.Logf("%d dispatches a step", len(a))

	names := pcFieldNames()
	fieldName := func(dword int) string {
		if dword < len(names) {
			return names[dword]
		}
		return fmt.Sprintf("dword%d", dword)
	}

	// kind -> the set of push fields that differed, with a count of dispatches.
	type diff struct {
		count  int
		fields map[string]int
	}
	diffs := map[string]*diff{}
	varies := map[int]bool{}
	for s := 1; s < steps; s++ {
		b, bk := grab(ids[s%len(ids)])
		if len(a) != len(b) {
			t.Fatalf("decode is not shape-stable: %d dispatches at step 1, %d at step %d", len(a), len(b), s+1)
		}
		for i := range a {
			if ak[i] != bk[i] {
				t.Fatalf("dispatch %d is %s at step 1, %s at step %d", i, ak[i], bk[i], s+1)
			}
			if a[i].Pipeline != b[i].Pipeline {
				t.Errorf("dispatch %d (%s): different pipeline at step %d", i, ak[i], s+1)
			}
			if a[i].GroupsX != b[i].GroupsX || a[i].GroupsY != b[i].GroupsY {
				t.Errorf("dispatch %d (%s): grid %dx%d at step 1, %dx%d at step %d", i, ak[i],
					a[i].GroupsX, a[i].GroupsY, b[i].GroupsX, b[i].GroupsY, s+1)
			}
			if bytes.Equal(a[i].PushConstants, b[i].PushConstants) {
				continue
			}
			varies[i] = true
			dd := diffs[ak[i]]
			if dd == nil {
				dd = &diff{fields: map[string]int{}}
				diffs[ak[i]] = dd
			}
			pa, pb := a[i].PushConstants, b[i].PushConstants
			for w := 0; w*4+4 <= len(pa); w++ {
				if !bytes.Equal(pa[w*4:w*4+4], pb[w*4:w*4+4]) {
					dd.fields[fieldName(w)]++
				}
			}
		}
	}
	for _, dd := range diffs {
		dd.count = 0
	}
	for i := range varies {
		diffs[ak[i]].count++
	}

	t.Logf("%d of %d dispatches carry identical push constants over %d steps", len(a)-len(varies), len(a), steps)
	kinds := make([]string, 0, len(diffs))
	for k := range diffs {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	for _, k := range kinds {
		dd := diffs[k]
		fields := make([]string, 0, len(dd.fields))
		for f, n := range dd.fields {
			fields = append(fields, fmt.Sprintf("%s(%d)", f, n))
		}
		sort.Strings(fields)
		t.Logf("  %-16s %3d dispatches differ: %v", k, dd.count, fields)
	}
}

// TestPrerecordedDecodeCrossesTheGatherBound is P15's gate on the one thing
// that arrangement adds to P1c: **a decode plan that moves with the depth.**
//
// Every knob in a decode step was constant until the gather, so the first
// one-token Extend could capture its command buffer and every later one could
// replay it without asking whether it was still the right buffer. The gathered
// arm is not constant — it turns on once the cache is deeper than twice the
// selection's width, because below that the selection excludes nothing and the
// compaction is pure overhead (gathersAtDepth) — so the graph compares
// `decodeEpoch` before each replay and re-records the step that crosses.
//
// **What this asserts is the re-recording and not the answer, and that is
// deliberate.** The obvious test — drive a sequence across the bound with the
// replay on and off and diff the logits — cannot fail at any depth a short
// fixture can reach: below the selection's 2051-cell width every live cell is
// selected, so the gathered list *is* the key axis in the same order, the two
// kernels fold the same partials the same way, and a replay carrying the wrong
// one is bit-identical to the right one. That version of this test was written
// first and passed with the epoch comparison deleted. So the assertion here is
// on `preEpoch` itself: the buffer the graph is replaying must be the buffer
// the current depth asks for, checked after every step.
//
// `LLM_ATTN_GATHER_MIN` puts the bound where a short fixture can cross it. The
// cache has to be deeper than 2051 cells for the block to be sparse at all,
// which is why NKV is set here and nowhere else in this file.
func TestPrerecordedDecodeCrossesTheGatherBound(t *testing.T) {
	const steps = 24
	const bound = 12
	t.Setenv("LLM_ATTN_GATHER_MIN", fmt.Sprint(bound))
	t.Setenv("LLM_NO_PRERECORD", "")
	g, _, ids := graphFixture(t, GraphOpts{Layers: 4, NKV: 4096})
	if g.attn == nil {
		t.Skip("no attention layer in this prefix")
	}
	if !g.attn.Sparse() {
		t.Skip("the selection does not bite at this cache size")
	}
	// The crossing has to be *inside* the run, or this test is the one above.
	if lo, hi := len(ids)+1, len(ids)+steps; bound < lo || bound > hi {
		t.Fatalf("the bound is %d cells and the run covers %d..%d — nothing crosses", bound, lo, hi)
	}

	logits, _, err := g.Forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[int]int{}
	for i := 0; i < steps; i++ {
		id := int32(0)
		for j := range logits {
			if logits[j] > logits[id] {
				id = int32(j)
			}
		}
		if logits, _, err = g.Extend([]int32{id}); err != nil {
			t.Fatal(err)
		}
		// After the step, the cache holds `past` cells and the buffer that just
		// ran has to have been the one that depth asks for.
		want := g.attn.DecodeEpoch(g.past)
		if g.pre == nil {
			t.Fatalf("step %d (cell %d): no buffer was captured", i, g.past)
		}
		if g.preEpoch != want {
			t.Fatalf("step %d (cell %d): replayed the epoch-%d buffer where the depth "+
				"asks for epoch %d — the plan moved and the recording did not",
				i, g.past, g.preEpoch, want)
		}
		seen[want]++
	}
	if len(seen) < 2 {
		t.Fatalf("only epoch %v occurred over %d steps — the bound is not being crossed "+
			"and this test asserted nothing", seen, steps)
	}
	t.Logf("%d steps across a gather bound at %d cells: epoch 0 on %d of them, epoch 1 on %d, "+
		"and the replayed buffer matched the depth on every one",
		steps, bound, seen[0], seen[1])
}
