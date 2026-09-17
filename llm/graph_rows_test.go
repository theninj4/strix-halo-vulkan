package llm

// L8c's instrument: a logit per token, and the claim that each of them is the
// prompt truncated there.
//
//	go test ./llm/ -v -run TestGraphForwardRows
//
// Every entry point before this one computed the last row alone, because that
// is what the reference computes: `inp_out_ids` drops every row but the last
// before the final mixer, so `Forward` moves one row to the front of the
// residual and runs the mixer and the head over it. Perplexity needs all of
// them, and `ForwardRows` gets them by not doing the move — the mixer is per
// token, so running it over the batch is the same arithmetic on nTok times
// the work, and the head then runs over slabs because a row of logits is
// 0.99 MB.
//
// What has to be true for a perplexity to mean anything is that **row t is
// the model's prediction after t+1 tokens** — that nothing below the head
// lets a row see its right-hand neighbours. That is a causality claim about
// five blocks, and it is checkable without a reference: run the prompt
// truncated at t and compare. L7b established the same property for
// `result_norm` across a chunk split; this is it through the head, at four
// rows of a batch, and it is the reason the slab loop is allowed to run the
// head over a window that starts in the middle of the prompt.

import (
	"testing"
	"time"
)

func TestGraphForwardRows(t *testing.T) {
	const (
		layers = 4
		nTok   = 128
		// Small enough that the window [first, nTok) is four slabs, so the
		// loop that moves a row range into the head's arena is exercised
		// rather than being one dispatch that happens to start at zero.
		headRows = 16
	)
	m, tr := fixtures4k(t)
	all, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	if len(all) < nTok {
		t.Skipf("the 4k trace has %d tokens", len(all))
	}
	ids := all[:nTok]

	dev, done := newTestDevice(t)
	t.Cleanup(done)
	start := time.Now()
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: nTok, Layers: layers, HeadRows: headRows})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	t.Logf("%d layers, %d tokens, head arena %d rows, staged in %s",
		g.Layers(), nTok, headRows, time.Since(start).Round(time.Millisecond))

	// Pinned, for the same reason TestGraphIsAChunkSplit pins: the one-token
	// runs below plan the decode kernels, which associate a dot product
	// differently from the GEMMs a 128-token batch plans. The claim here is
	// about causality, not about round-off, so both sides run the prefill
	// schedule.
	if err := g.PinSchedule(true); err != nil {
		t.Fatal(err)
	}

	first := nTok / 2
	rows := map[int][]float32{}
	seen := 0
	if err := g.ForwardRows(ids, first, func(tk int, lg []float32) error {
		seen++
		rows[tk] = append([]float32(nil), lg...)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if want := nTok - first; seen != want {
		t.Fatalf("ForwardRows handed back %d rows, want %d", seen, want)
	}

	// Four rows: the first of the window, the one after it, one inside a
	// slab and the last — which is the only one any other entry point can
	// produce, and so the only one that was ever checked before.
	for _, tk := range []int{first, first + 1, nTok - 7, nTok - 1} {
		got := rows[tk]
		if got == nil {
			t.Fatalf("no row %d", tk)
		}
		want, _, err := g.Forward(ids[:tk+1])
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(want) {
			t.Fatalf("row %d is %d logits, the %d-token prompt gives %d", tk, len(got), tk+1, len(want))
		}
		bad, at := 0, -1
		for i := range got {
			if got[i] != want[i] {
				bad++
				if at < 0 {
					at = i
				}
			}
		}
		if bad != 0 {
			r, cerr := compare(got, want)
			if cerr != nil {
				t.Fatal(cerr)
			}
			t.Errorf("row %d of the batch against the %d-token prompt: %d of %d logits differ, first at %d (%v)",
				tk, tk+1, bad, len(got), at, r)
			continue
		}
		t.Logf("row %d: identical to the %d-token prompt's logits, to the last place", tk, tk+1)
	}
}
