package llm

import (
	"testing"
	"time"
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
