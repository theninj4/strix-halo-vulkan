package llm

import (
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// TestBatchForIsUnderTheWatchdog is the unit check P0's fix needs, because the
// constants it is made of are a fit in milliseconds and the bound is a
// time.Duration: a factor of a thousand either way still compiles, still runs,
// and is only visible as a ring reset 40 seconds into a staged model.
//
// The cliff measured at 48 layers is 2 s a submit: 1024 dispatches is 1.964 s
// at 2688 rows and runs, 2.033 s at 2816 rows and is reset.
func TestBatchForIsUnderTheWatchdog(t *testing.T) {
	const cliff = 2 * time.Second

	// The fit against what was measured, whole passes of 1273 dispatches.
	for _, c := range []struct {
		rows int
		want time.Duration
	}{
		{512, 789500 * time.Microsecond},
		{2560, 2342400 * time.Microsecond},
		{2688, 2434200 * time.Microsecond},
		{3072, 2687500 * time.Microsecond},
	} {
		got := 1273 * dispatchCost(c.rows)
		off := float64(got-c.want) / float64(c.want)
		if off < -0.02 || off > 0.02 {
			t.Errorf("rows %d: modelled %v against a measured %v, %.1f%% off", c.rows, got, c.want, 100*off)
		}
	}

	// And the bound it buys, at every row count a prefill can reach.
	for _, rows := range []int{1, 2, 64, 512, 1024, 2048, 2688, 2816, 3072, 4096, 8192, 16384, 262144} {
		n := batchFor(rows)
		if n < minBatch || n > maxBatch {
			t.Errorf("rows %d: batch %d is outside [%d, %d]", rows, n, minBatch, maxBatch)
		}
		if held := time.Duration(n) * dispatchCost(rows); held > cliff/2 && n > minBatch {
			t.Errorf("rows %d: %d dispatches hold the ring for %v, over the %v budget", rows, n, held, cliff/2)
		}
	}

	// Decode must not pay for any of this.
	if n := batchFor(1); n != maxBatch {
		t.Errorf("one row batches at %d, want the full %d", n, maxBatch)
	}
}

// TestChunkEndIsUnderTheWatchdogAtDepth is the same check for the term P0
// could not have had. `batchFor` alone is a function of the rows, so it hands
// back the same 660 dispatches at 2048 rows whether the cache is empty or
// holds 128 000 cells — and at 64 000 those 660 already run for ~1.5 s of the
// 2 s cliff. This walks a synthetic 48-layer pass and asserts that no chunk
// the recorder would cut is modelled over the budget, at every depth a
// 128k context reaches.
func TestChunkEndIsUnderTheWatchdogAtDepth(t *testing.T) {
	const cliff = 2 * time.Second

	// A pass shaped like the shipped graph: 1273 dispatches with 78 of them
	// attention, spread as the twelve full-attention layers of forty-eight
	// are — one run of six every fourth layer.
	pass := func() *recorder {
		r := &recorder{}
		for i := 0; i < 1273; i++ {
			r.d = append(r.d, vk.MultiDispatch{})
			own := ownMoE
			if (i/26)%4 == 3 && i%26 < 6 {
				own = ownAttn
			}
			r.owner = append(r.owner, own)
			r.kind = append(r.kind, own)
		}
		return r
	}

	for _, rows := range []int{1, 512, 2048, 4096, 8192} {
		for _, past := range []int{0, 8000, 32000, 64000, 131072} {
			r := pass()
			r.rows, r.past = rows, past
			chunks := 0
			for i := 0; i < len(r.d); chunks++ {
				j := r.chunkEnd(i)
				if j <= i {
					t.Fatalf("rows %d past %d: chunkEnd(%d) = %d, no progress", rows, past, i, j)
				}
				if held := r.chunkCost(i, j); held > cliff/2 && j-i > minBatch {
					t.Errorf("rows %d past %d: dispatches %d-%d hold the ring for %v, over the %v budget",
						rows, past, i, j-1, held, cliff/2)
				}
				i = j
			}
			if chunks == 0 {
				t.Errorf("rows %d past %d: no chunks", rows, past)
			}
		}
	}

	// A decode step is one row, so depth must not chunk it any harder than
	// the query pool already does: 1407 dispatches at 131 072 cells is ~36 ms
	// on the GPU, three per cent of the budget, and the only bound that may
	// cut it is maxBatch. This is the line that says nothing about decode
	// moved.
	r := pass()
	r.rows, r.past = 1, 131072
	if j := r.chunkEnd(0); j != min(maxBatch, len(r.d)) {
		t.Errorf("one row at 131 072 cells chunks at %d of %d, want the full %d",
			j, len(r.d), min(maxBatch, len(r.d)))
	}
}

// TestObserveNeverLoosensTheBudget is the hazard the adaptive correction
// creates and the one line that closes it.
//
// The affine model's 322 us a dispatch was fitted at prefill row counts. A
// decode step is ~1443 dispatches over *one* row, so it is predicted at
// ~330 ms of GPU where the step is ~36 ms — an order out, and harmless,
// because maxBatch bounds a decode step anyway. But a scale that learned from
// those measurements would converge downwards across a conversation's decode
// steps and then hand the next prefill several times the budget it asked
// for, which is a ring reset 40 seconds into a 128k prompt.
func TestObserveNeverLoosensTheBudget(t *testing.T) {
	defer func() {
		costScaleMu.Lock()
		costScaleVal = 1
		costScaleMu.Unlock()
	}()
	r := &recorder{}

	// A hundred decode steps, each measured at a ninth of what it was
	// modelled — which is what a one-row pass actually does.
	for i := 0; i < 100; i++ {
		r.observe(330*time.Millisecond, 36*time.Millisecond)
	}
	if s := costScale(); s < 1 {
		t.Fatalf("decode steps pulled the scale to %.3f; a prefill would then be given "+
			"%.1fx the budget it asked for", s, 1/s)
	}

	// And it still tightens when a submit really does run long.
	r.observe(500*time.Millisecond, 900*time.Millisecond)
	if s := costScale(); s < 1.7 {
		t.Errorf("a submit at 1.8x its prediction moved the scale to %.3f, want >= 1.7", s)
	}
}
