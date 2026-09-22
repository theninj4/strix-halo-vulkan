package llm

import (
	"math"
	"testing"
)

// P13's gate: compacting the live key blocks into a list does not change the
// answer, and it is an **equality** rather than a tolerance.
//
// It has to be. The list holds exactly the blocks `llm_attn_wmma.comp`'s
// `blkAny` skip would not have skipped, in the order the loop would have met
// them — so the same key blocks pass through the same online softmax in the
// same order, and every rescale is the same rescale. Nothing is reassociated
// and nothing is approximated; what is deleted is the *visit* to the blocks
// P7 already established contribute nothing. A tolerance here would pass a
// list that dropped a live block, which at 3.2% density is a change no rms
// against llama.cpp would ever notice.
//
// The fixture is the 4 k one for the reason every selection test uses it: at
// 256 cells the selection names every cell and the list would be the whole
// axis, which tests nothing. At 4096 cells against a width of 2051 it bites.
func TestAttnGPUBlockListDoesNotChangeTheAnswer(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	if !g.Sparse() {
		t.Fatalf("the fixture is not selecting: width %d of %d cells", g.selWidth(), g.NKV())
	}
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	in := src.Vals

	// One pinned plan for both arms, and the key split off: both are chosen
	// from the run's length, and a rung or a slice count that moved between
	// the arms would make this a comparison of two kernels rather than of one
	// kernel with and without the list.
	attn, gemm, outGemm := g.Plan()
	if err := g.SetPlan(attn, gemm, outGemm); err != nil {
		t.Fatal(err)
	}
	g.SetSplits(1)
	defer g.SetSplits(0)

	for _, tc := range []struct {
		name   string
		chunks []int
	}{
		{"one pass", []int{nTok}},
		{"512 at a time", even(nTok, 512)},
		{"a prompt and then one token at a time", append([]int{nTok - 33}, even(33, 1)...)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("LLM_ATTN_BLOCK_LIST", "0")
			want := runChunks(t, g, in, c.NEmbd, tc.chunks)
			t.Setenv("LLM_ATTN_BLOCK_LIST", "1")
			got := runChunks(t, g, in, c.NEmbd, tc.chunks)
			if len(got) != len(want) {
				t.Fatalf("%d values with the list, %d without", len(got), len(want))
			}
			bad, first := 0, -1
			for i := range got {
				if got[i] != want[i] {
					bad++
					if first < 0 {
						first = i
					}
				}
			}
			if bad != 0 {
				t.Fatalf("%d of %d values differ, first at token %d component %d: %v with the list against %v without",
					bad, len(got), first/c.NEmbd, first%c.NEmbd, got[first], want[first])
			}
			t.Logf("%d tokens over %d chunks: identical to the last place", nTok, len(tc.chunks))
		})
	}
}

// TestAttnGPUBlockListIsDispatched is the precondition the equality above
// cannot see: a list that was never built, or an attention kernel that was
// handed NO_W and walked the axis anyway, passes it by being the control arm
// twice over.
func TestAttnGPUBlockListIsDispatched(t *testing.T) {
	g, _, _, nTok, done := attnGPU4k(t)
	defer done()
	g.SetSplits(1)
	defer g.SetSplits(0)
	// And P14-2's gather off: it is the same compaction one granularity finer
	// and the two are exclusive, so the graph builds this one only where the
	// gather is not running.
	g.SetGather(false)
	defer g.AutoGather()
	if err := g.SetPast(0); err != nil {
		t.Fatal(err)
	}
	if err := g.Resize(nTok); err != nil {
		t.Fatal(err)
	}

	t.Setenv("LLM_ATTN_BLOCK_LIST", "1")
	_, kinds, err := g.graph(0)
	if err != nil {
		t.Fatal(err)
	}
	if !hasKind(kinds, "blocks") {
		t.Errorf("no compaction dispatch in %v", kinds)
	}

	// And the default, which is off since P13-3 measured the compaction inert
	// at prefill: an empty environment must not dispatch it.
	t.Setenv("LLM_ATTN_BLOCK_LIST", "")
	_, kinds, err = g.graph(0)
	if err != nil {
		t.Fatal(err)
	}
	if hasKind(kinds, "blocks") {
		t.Errorf("the default still dispatches the compaction: %v", kinds)
	}
}

func hasKind(kinds []string, want string) bool {
	for _, k := range kinds {
		if k == want {
			return true
		}
	}
	return false
}

// TestAttnGPURungsAgree is what P13's new rungs need and the ladder never
// had: a correctness gate over *every* build of llm_attn_wmma.comp, not just
// the shipped one.
//
// The 16-cell rung (qt1_kt1) is why it exists. A key block narrower than a
// mask word is the first shape in this file where `base` is not a multiple of
// 32, so the staged bitmask slice is a *half* of a word and every `selected`
// bit moves by `base & 31`. Get that wrong and the kernel is still a
// perfectly well-formed attention — over the wrong cells — which every
// tolerance against llama.cpp would pass by being roughly a dense causal
// attention.
//
// It is a tolerance and not an equality, and it has to be: a rung is a
// different number of online-softmax rescales in a different order, and fp32
// addition is not associative. The bar is P8's, for the same reason — the
// dense bank this layer reads is itself 5.4e-3 rms from the f32 model.
func TestAttnGPURungsAgree(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	if !g.Sparse() {
		t.Fatalf("the fixture is not selecting: width %d of %d cells", g.selWidth(), g.NKV())
	}
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	in := src.Vals
	_, gemm, outGemm := g.Plan()
	g.SetSplits(1)
	defer g.SetSplits(0)

	run := func(k AttnKernel) []float32 {
		if err := g.SetPlan(k, gemm, outGemm); err != nil {
			t.Fatal(err)
		}
		return runChunks(t, g, in, c.NEmbd, []int{nTok})
	}
	want := run(AttnQT1KT2)
	for _, k := range AttnKernels() {
		if k == AttnQT1KT2 {
			continue
		}
		got := run(k)
		var maxAbs, sum2, ref2 float64
		for i := range got {
			d := math.Abs(float64(got[i]) - float64(want[i]))
			maxAbs = math.Max(maxAbs, d)
			sum2 += d * d
			ref2 += float64(want[i]) * float64(want[i])
		}
		rms := math.Sqrt(sum2 / ref2)
		t.Logf("%-8s against qt1_kt2: rms %.3g, max abs %.3g", k, rms, maxAbs)
		if rms > 1e-3 {
			t.Errorf("%s: rms %.3g against the shipped rung, over the 1e-3 a reassociation "+
				"can account for — this is a different set of cells, not a different order", k, rms)
		}
	}
	if err := g.SetPlan(AttnQT1KT2, gemm, outGemm); err != nil {
		t.Fatal(err)
	}
}
