package llm

import (
	"testing"
	"time"
)

// L7b's gate: the whole graph, continued.
//
// L7a made the attention layer's cache a chunk split — bit for bit, at every
// schedule down to one token at a time. This is the same claim about the
// model: a prompt run in pieces has to be the prompt run whole, which means
// every history in it carries. There are four, and three of them are ours:
//
//	the KV cache and the indexer's pooled blocks      L7a
//	the PLE convolution's nine-row ring               L7b
//	every DeltaNet layer's [128,128,48] recurrent
//	  state and three-row window                      L3, and L7b per layer
//	the n-gram trigram, which is the host's           PLERows over the whole
//	                                                  id list, sliced
//
// The fourth is the one a cache design does not suggest: the first token of a
// continuing run still hashes two tokens that are not in it, so the graph
// keeps the sequence rather than the batch.
//
// Four layers, because four is the whole architecture — three DeltaNet
// layers, the PLE block in front of layer 1, one full-attention layer with
// its indexer, four MoE blocks and eight mixers — at 7 GB instead of 85.
//
// **The one-token schedule is the control as well as the gate.** A chunk of
// one has nothing of its own to read: every convolution tap reaches behind
// it, its trigram is two tokens it does not contain, its attention is a
// cache it did not write, and its recurrent state is 500 tokens of somebody
// else's arithmetic. A history that was not carried could not produce the
// right answer by accident there, and the last subtest below is the same run
// with the sequence reset before each token, which is what a dropped history
// looks like.
func TestGraphIsAChunkSplit(t *testing.T) {
	const (
		layers = 4
		nTok   = 512
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
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: nTok, Layers: layers, NoHead: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	t.Logf("%d layers, %d tokens, staged in %s", g.Layers(), nTok, time.Since(start).Round(time.Millisecond))

	// Pinned, for the three equalities below. The ladders this vertical is
	// made of are bit-exact against each other — they tile the same
	// arithmetic differently — with one exception, and it is the rung the
	// schedule picks at exactly the chunk length this test cares about:
	// L7d's split-K GEMV associates a 10240-long sum 160 ways where the GEMM
	// associates it in one sweep. So the equality is asserted against a fixed
	// schedule, and what the decode schedule costs is the fourth subtest.
	if err := g.PinSchedule(true); err != nil {
		t.Fatal(err)
	}

	// The whole prompt in one run, and the two tensors it leaves: the wide
	// residual of every token, and the one row the head would read.
	norm, err := g.Hidden(ids)
	if err != nil {
		t.Fatal(err)
	}
	wantNorm := append([]float32(nil), norm...)

	for _, tc := range []struct {
		name   string
		chunks []int
	}{
		{"128 at a time", even(nTok, 128)},
		{"9 at a time, the length of the PLE ring", even(nTok, 9)},
		{"a prompt and then one token at a time", append([]int{nTok - 17}, even(17, 1)...)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := g.Reset(); err != nil {
				t.Fatal(err)
			}
			var got []float32
			at := 0
			for i, n := range tc.chunks {
				if i == len(tc.chunks)-1 {
					got, err = g.HiddenExtend(ids[at : at+n])
				} else {
					err = g.Append(ids[at : at+n])
				}
				if err != nil {
					t.Fatalf("chunk %d of %d tokens at position %d: %v", i, n, at, err)
				}
				at += n
			}
			if g.Past() != nTok {
				t.Fatalf("the graph is at position %d after %d tokens", g.Past(), nTok)
			}
			bad, first := 0, -1
			for i := range got {
				if got[i] != wantNorm[i] {
					bad++
					if first < 0 {
						first = i
					}
				}
			}
			if bad != 0 {
				t.Errorf("result_norm: %d of %d values differ, first at %d: %v against %v",
					bad, len(got), first, got[first], wantNorm[first])
				return
			}
			t.Logf("%d chunks, %d tokens: result_norm identical to the last place", len(tc.chunks), nTok)
		})
	}

	// What the decode schedule costs, on the hardest of the three splits.
	// It is a different association of the same products and nothing else —
	// the histories are the same histories — so the bar is fp32 round-off
	// over a 10240-long dot product, amplified by L6b-3's x1.085 a layer and
	// by seventeen steps of recurrence, and **not** the 1.785e+00 the control
	// below reports for a history that was not carried.
	t.Run("one token at a time, on the decode schedule", func(t *testing.T) {
		if err := g.PinSchedule(false); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := g.PinSchedule(true); err != nil {
				t.Fatal(err)
			}
		}()
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		var got []float32
		if err := g.Append(ids[:nTok-17]); err != nil {
			t.Fatal(err)
		}
		for i := nTok - 17; i < nTok; i++ {
			if got, err = g.HiddenExtend(ids[i : i+1]); err != nil {
				t.Fatal(err)
			}
		}
		r, err := compare(got, wantNorm)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("result_norm, split-K down projection: %v", r)
		if r.rms > 1e-3 {
			t.Errorf("the decode schedule moves result_norm by %.3e rms, which is more than a "+
				"reassociated dot product can account for (%v)", r.rms, r)
		}
	})

	// The control. Every token its own *sequence* rather than its own batch:
	// the same dispatches, the same weights, the same order, and none of the
	// four histories. If the equalities above were passing because nothing
	// carried, this would match too.
	t.Run("control, every token a fresh sequence", func(t *testing.T) {
		var got []float32
		for i := range ids {
			if got, err = g.Hidden(ids[i : i+1]); err != nil {
				t.Fatal(err)
			}
		}
		r, err := compare(got, wantNorm)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("result_norm, histories dropped: %v", r)
		if r.rms < 1e-2 {
			t.Errorf("dropping every history moves result_norm by only %.3e, "+
				"so the equalities above are not evidence that anything carried", r.rms)
		}
	})
}
