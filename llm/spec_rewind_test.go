package llm

import (
	"testing"
	"time"
)

// P5c's gate: a pass that is thrown away leaves the model exactly as it found
// it (research/p5-mtp-rollback.md §3.5).
//
// Speculation runs tokens the model has not agreed to. A round drafts, runs a
// verification pass over the candidates, and then either keeps the pass or
// discards it — and "discards it" has to mean the sequence is byte for byte
// what it would have been had the pass never run. There are four histories
// that could make that false, and two of them do:
//
//	the KV cache and the indexer's raw keys    free — a cell past the end of
//	                                           the sequence is masked out of
//	                                           every score, so a rewound cell
//	                                           is unreadable and not stale
//	the pooled indexer block table             **not** free — a block is
//	                                           written the one time a run
//	                                           completes its cells and never
//	                                           again, so Rewind restores it
//	the two convolution rings                  **not** free — see below
//	every DeltaNet layer's recurrent state     **not** free — the pass folded
//	                                           tokens into it that are gone
//
// **The rings are the finding the design pass nearly missed, and one rejected
// token is enough.** A ring is addressed by absolute position modulo its
// length, so the DeltaNet's −3 tap reads slot `p mod 3` — the slot position p
// itself writes — and the PLE's dilated −6 and −3 taps alias at depth 3 and 6
// of a nine-slot ring. A single rejected draft therefore corrupts a tap the
// *re-run of the accepted prefix* will read, and the failure is silent: wrong
// activations, no error, and a perplexity delta nobody would attribute.
//
// So the schedule below is not "run some tokens and rewind". It is the shape a
// speculative loop actually runs — a rewindable pass whose tail is a token the
// model rejects, the rewind, and then the accepted prefix re-run and committed
// — repeated until the whole prompt is consumed, against the same prompt run
// in one pass.
//
// **And the control is the point.** Comparing the rolled-back arm against the
// reference only says something if an arm *without* the rollback would fail,
// so the last subtest runs the identical schedule with the position rewound
// and the histories left where the rejected pass put them. P5b's rule: a
// control has to be able to fail.
func TestSpeculationRewindIsTheSequence(t *testing.T) {
	const (
		layers = 4
		nTok   = 512
		// Where the speculative rounds start. The prefix is one ordinary
		// prefill, so every round below runs against a sequence with real
		// history behind it — a ring whose taps all reach back, a recurrent
		// state 480 tokens deep, and a KV cache that is not empty.
		prefix = 480
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
	// A few cells past the prompt, because a verification pass runs rows the
	// sequence does not keep and they still have to fit in the cache.
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: nTok, NKV: nTok + 16, Layers: layers, NoHead: true,
		Speculative: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	t.Logf("%d layers, %d tokens, staged in %s", g.Layers(), nTok, time.Since(start).Round(time.Millisecond))

	// Pinned, for the same reason TestGraphIsAChunkSplit pins: the decode
	// GEMV associates a long sum differently from the GEMM and the schedule
	// below changes row counts round by round. The claim here is about
	// histories, so the arithmetic is held still and the equality is exact.
	if err := g.PinSchedule(true); err != nil {
		t.Fatal(err)
	}

	// The reference: the whole prompt in one pass.
	norm, err := g.Hidden(ids)
	if err != nil {
		t.Fatal(err)
	}
	want := append([]float32(nil), norm...)

	// A token this prompt does not have at the position it is offered at, so
	// that every speculative pass below is genuinely rejected. It is a real
	// id — a draft head's wrong guess is a plausible token, not a poison
	// value — and the point is only that it is not the one the sequence has.
	wrong := ids[7]

	// One round of the schedule: `draft` rows of rejected candidates behind
	// `keep` true ones, then the rewind, then the true rows committed.
	type round struct{ keep, draft int }
	rounds := []round{
		{1, 1}, // depth 1, rejected: the two-row pass P5 is built around
		{2, 1}, // the accepted prefix re-run at the front of the next pass
		{1, 2}, // three rows, so the ring's oldest tap is inside the pass
		{2, 2},
	}

	run := func(t *testing.T, rollback bool) []float32 {
		t.Helper()
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		// The prefill is an ordinary pass and is committed as it runs; the
		// rollback goes on afterwards, which is also the order a served loop
		// does it in — a prompt is not speculated.
		if err := g.Append(ids[:prefix]); err != nil {
			t.Fatal(err)
		}
		if err := g.Speculate(rollback); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := g.Speculate(false); err != nil {
				t.Fatal(err)
			}
		}()
		var got []float32
		at, slot, flips := prefix, -1, 0
		if g.dn != nil {
			slot = g.dn.StateSlot()
		}
		for i := 0; at < nTok; i++ {
			r := rounds[i%len(rounds)]
			keep := minInt(r.keep, nTok-at)
			// The verification pass: the tokens the round will keep, then
			// the drafts it will not.
			pass := append([]int32(nil), ids[at:at+keep]...)
			for k := 0; k < r.draft; k++ {
				pass = append(pass, wrong)
			}
			was, wasIds := g.Past(), len(g.Ids())
			if err := g.Append(pass); err != nil {
				t.Fatalf("round %d, speculative pass of %d rows at %d: %v", i, len(pass), at, err)
			}
			if rollback {
				err = g.Rewind()
			} else {
				err = g.rewindPosition(was, wasIds)
			}
			if err != nil {
				t.Fatalf("round %d: %v", i, err)
			}
			if g.Past() != was {
				t.Fatalf("round %d: the graph is at %d after a rewind to %d", i, g.Past(), was)
			}
			// The accepted prefix, committed. The last one runs the final
			// mixer so that there is something to compare.
			if at+keep == nTok {
				got, err = g.HiddenExtend(ids[at : at+keep])
			} else {
				err = g.Append(ids[at : at+keep])
			}
			if err != nil {
				t.Fatalf("round %d, committed pass of %d rows at %d: %v", i, keep, at, err)
			}
			if rollback {
				if err := g.Commit(); err != nil {
					t.Fatalf("round %d: %v", i, err)
				}
				if g.dn != nil && g.dn.StateSlot() != slot {
					slot, flips = g.dn.StateSlot(), flips+1
				}
			}
			at += keep
		}
		if g.Past() != nTok {
			t.Fatalf("the graph is at position %d after %d tokens", g.Past(), nTok)
		}
		if rollback {
			// The ping-pong has to have moved. If it never did, every pass
			// wrote the slot it read and the arm below is not measuring a
			// rollback at all.
			if flips == 0 {
				t.Errorf("the state slot never flipped, so no pass was ever written to the other one")
			}
			t.Logf("%d committed rounds, %d state-slot flips", flips, flips)
		}
		return append([]float32(nil), got...)
	}

	t.Run("rejected passes, rolled back", func(t *testing.T) {
		got := run(t, true)
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
			t.Errorf("result_norm: %d of %d values differ, first at %d: %v against %v",
				bad, len(got), first, got[first], want[first])
			return
		}
		t.Logf("result_norm identical to the last place after %d rejected passes", (nTok-prefix)/2)
	})

	// The control. The same schedule, the same rejected tokens, the same
	// re-runs — and the position rewound without the histories. If the arm
	// above were passing because nothing a rejected pass writes is ever read,
	// this would pass too.
	t.Run("control, the position rewound and the histories left", func(t *testing.T) {
		got := run(t, false)
		r, err := compare(got, want)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("result_norm, a rejected pass left in the histories: %v", r)
		if r.rms < 1e-3 {
			t.Errorf("leaving a rejected pass in the convolution rings and the recurrent state moves "+
				"result_norm by only %.3e, so the equality above is not evidence that anything was rolled back", r.rms)
		}
	})
}

// TestSpeculationNeedsItsSlots is the other half of P5c's staging choice: a
// graph built without `GraphOpts.Speculative` has one slot for every carried
// tensor, and `Speculate(true)` has to **refuse** rather than quietly write
// the state it is supposed to be preserving.
//
// It is the cheapest test in the file and it guards 120.75 MB: the product
// path does not allocate the second slot, so the only thing standing between
// a decode graph and a silently corrupted rollback is this refusal.
func TestSpeculationNeedsItsSlots(t *testing.T) {
	m, _ := fixtures4k(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: 8, NKV: 64, Layers: 2, NoHead: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	if err := g.Speculate(true); err == nil {
		t.Fatal("a graph staged without Speculative accepted Speculate(true)")
	} else {
		t.Logf("refused: %v", err)
	}
	if g.Speculating() {
		t.Error("the graph is speculating after a refused Speculate(true)")
	}
	// And the refusal leaves the blocks where it found them, so an ordinary
	// pass after it is an ordinary pass.
	if g.dn != nil && g.dn.Speculating() {
		t.Error("the DeltaNet block is speculating after a refused Speculate(true)")
	}
}
