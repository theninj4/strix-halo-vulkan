package llm

// P20f's first gate: speculation on a graph that holds several sequences
// (research/p20-llamacpp-mtp.md §11).
//
//	go test ./llm/ -v -run TestSpeculationSlotsAreSequences   # four layers, ~9 GB

import "testing"

// TestSpeculationSlotsAreSequences runs P20c/P20b's keep-prefix schedule on
// sequence slot 0 of a two-slot graph while slot 1 advances between its
// rounds — alone, and every third round in a batched pass with slot 0
// (DecodeRows, C5) — and compares each sequence's last row with the same
// tokens run plainly.
//
// Until P20f the carried state's slot index was either a sequence or a
// speculation plane. Now sequence s owns planes [s*P, (s+1)*P), each sequence
// remembers which of its planes is committed, and a batched row reads its
// sequence's committed plane rather than its first. The schedule is arranged
// so that every one of those can be wrong: slot 0's committed plane moves on
// almost every round, and slot 1 runs in a batch while it has.
func TestSpeculationSlotsAreSequences(t *testing.T) {
	const (
		layers  = 4
		nA      = 512
		prefixA = 480
		prefixB = 400
	)
	m, tr := fixtures4k(t)
	all, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	if len(all) < 1200 {
		t.Skipf("the 4k trace has %d tokens", len(all))
	}
	A, B := all[:nA], all[600:1100]

	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: nA, NKV: 1024, Layers: layers, Slots: 2,
		Speculative: true, SpecRows: 3, HeadGEMM: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	if err := g.PinSchedule(true); err != nil {
		t.Fatal(err)
	}
	use := func(s int) {
		t.Helper()
		if err := g.UseSlot(s); err != nil {
			t.Fatal(err)
		}
	}
	wrong := A[7]
	type round struct{ rows, keep int }
	rounds := []round{{2, 1}, {3, 2}, {3, 3}, {2, 2}, {3, 1}, {3, 2}, {2, 1}, {3, 3}}

	use(1)
	if err := g.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := g.Append(B[:prefixB]); err != nil {
		t.Fatal(err)
	}
	use(0)
	if err := g.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := g.Append(A[:prefixA]); err != nil {
		t.Fatal(err)
	}
	if err := g.SpeculateFirst(true); err != nil {
		t.Fatal(err)
	}

	at, nb, batched, moved := prefixA, prefixB, 0, 0
	for i := 0; at < nA-2; i++ {
		// Slot 0: one speculative round.
		use(0)
		if err := g.Speculate(true); err != nil {
			t.Fatal(err)
		}
		r := rounds[i%len(rounds)]
		keep := minInt(r.keep, nA-2-at)
		n := r.rows
		if keep < r.keep {
			n = keep
		}
		pass := append([]int32(nil), A[at:at+keep]...)
		for len(pass) < n {
			pass = append(pass, wrong)
		}
		if err := g.Append(pass); err != nil {
			t.Fatalf("round %d, a pass of %d rows at %d: %v", i, n, at, err)
		}
		if keep == n {
			err = g.Commit()
		} else {
			err = g.Keep(keep)
		}
		if err != nil {
			t.Fatalf("round %d: %v", i, err)
		}
		at += keep
		if err := g.Speculate(false); err != nil {
			t.Fatal(err)
		}
		if g.dn != nil && g.dn.StateSlot() != 0 {
			moved++
		}

		// Slot 1: a token, alone or beside slot 0's next one.
		if i%3 == 2 && at < nA-2 {
			if g.dn != nil && g.dn.StateSlot() != 0 {
				batched++
			}
			if _, err := g.DecodeRows([]int{0, 1}, []int32{A[at], B[nb]}); err != nil {
				t.Fatalf("round %d, batched: %v", i, err)
			}
			at++
		} else {
			use(1)
			if err := g.Append(B[nb : nb+1]); err != nil {
				t.Fatal(err)
			}
		}
		nb++
	}
	if moved == 0 || batched == 0 {
		t.Fatalf("slot 0's committed plane moved on %d rounds and was off plane 0 in %d batched passes; "+
			"the gate needs both", moved, batched)
	}
	t.Logf("slot 0 off its first plane after %d rounds, in %d batched passes; slot 1 at %d", moved, batched, nb)

	use(0)
	gotA, err := g.HiddenExtend(A[at : at+1])
	if err != nil {
		t.Fatal(err)
	}
	gotA = append([]float32(nil), gotA...)
	use(1)
	gotB, err := g.HiddenExtend(B[nb : nb+1])
	if err != nil {
		t.Fatal(err)
	}
	gotB = append([]float32(nil), gotB...)
	if err := g.SpeculateFirst(false); err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct {
		name string
		slot int
		ids  []int32
		got  []float32
	}{{"slot 0, speculated", 0, A[:at+1], gotA}, {"slot 1, between", 1, B[:nb+1], gotB}} {
		use(c.slot)
		want, err := g.Hidden(c.ids)
		if err != nil {
			t.Fatal(err)
		}
		bad, first := 0, -1
		for i := range want {
			if c.got[i] != want[i] {
				bad++
				if first < 0 {
					first = i
				}
			}
		}
		if bad != 0 {
			r, _ := compare(c.got, want)
			t.Errorf("%s: result_norm %d of %d values differ, first at %d (%v)", c.name, bad, len(want), first, r)
		} else {
			t.Logf("%s: result_norm identical over %d tokens", c.name, len(c.ids))
		}
	}
}

// TestServedSpeculationIsPlain is P20f's gate for the served loop's pieces
// (Speculator.Prime, Begin, Round, Owe, Settle): two sequences on a two-slot
// graph, each prefilled in chunks with the draft primed from each chunk's
// residual, decoded in rounds that alternate between the slots with a batched
// plain step of both every fourth turn (C5; the draft owes those rows), then a
// second turn on slot 0 that continues what the first left — against the same
// tokens decoded plainly, one greedy step at a time. Every pass is at most
// three rows, so every projection runs a decode GEMV, which is row-exact, and
// the texts must agree token for token.
//
// The schedule is deliberately **not** pinned: DecodeRows does not take
// Graph.PinSchedule, so a pinned batched row rounds unlike a pinned solo step
// (it read token 8 wrong at four layers), while unpinned both are the GEMVs.
//
// On the four-layer prefix the draft head (trained against 48) is rarely
// right, so this is a gate on the mechanics — priming, slots, settling — and
// not on acceptance; the keep-prefix arithmetic has its own gates above.
func TestServedSpeculationIsPlain(t *testing.T) {
	const (
		layers = 4
		chunk  = 96
		gen    = 24
	)
	m, tr := fixtures4k(t)
	all, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	if len(all) < 1200 {
		t.Skipf("the 4k trace has %d tokens", len(all))
	}
	draft, err := Open(mtpDraftPath)
	if err != nil {
		t.Skipf("draft checkpoint: %v", err)
	}
	defer draft.Close()

	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: chunk, NKV: 1024, Layers: layers, Slots: 2,
		Speculative: true, SpecRows: 3})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	d, err := StageDraft(dev, draft, g)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Destroy)
	if err := d.SetVocab(DraftVocabDefault); err != nil {
		t.Fatal(err)
	}
	use := func(s int) {
		t.Helper()
		if err := g.UseSlot(s); err != nil {
			t.Fatal(err)
		}
	}

	// plain prefills ids on slot s in chunks and decodes n greedy tokens.
	plain := func(s int, ids []int32, n int) []int32 {
		t.Helper()
		use(s)
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		var l []float32
		for at := 0; at < len(ids); at += chunk {
			if l, _, err = g.Extend(ids[at:min(at+chunk, len(ids))]); err != nil {
				t.Fatal(err)
			}
		}
		var out []int32
		for len(out) < n {
			id := Argmax(l)
			out = append(out, id)
			if l, _, err = g.Extend([]int32{id}); err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	A1, B := all[:300], all[600:850]
	wantA1 := plain(0, A1, gen)
	wantB := plain(1, B, gen)
	// The second turn: the first turn's text, then more prompt.
	A2 := append(append(append([]int32(nil), A1...), wantA1[:gen-1]...), all[400:470]...)
	wantA2 := plain(0, A2, gen)

	sps := make([]*Speculator, 2)
	for s := range sps {
		if sps[s], err = NewSlotSpeculator(g, d, m, Wiring{Name: "res"}, s, SpecDepthDefault); err != nil {
			t.Fatal(err)
		}
	}
	// prefill runs ids on slot s from position `from`, priming the draft;
	// it returns the token the last chunk names.
	prefill := func(s int, ids []int32, from int) int32 {
		t.Helper()
		use(s)
		if err := g.Speculate(false); err != nil {
			t.Fatal(err)
		}
		if from == 0 {
			if err := g.Reset(); err != nil {
				t.Fatal(err)
			}
		}
		var l []float32
		for at := from; at < len(ids); at += chunk {
			c := ids[at:min(at+chunk, len(ids))]
			var res []float32
			if l, res, err = g.ExtendResidual(c); err != nil {
				t.Fatal(err)
			}
			if err := sps[s].Prime(c, at, res); err != nil {
				t.Fatal(err)
			}
		}
		return Argmax(l)
	}
	got := make([][]int32, 2)
	begin := func(s int, first int32) {
		t.Helper()
		got[s] = []int32{first}
		if err := sps[s].Begin(); err != nil {
			t.Fatal(err)
		}
	}
	round := func(s int) {
		t.Helper()
		use(s)
		if err := g.Speculate(true); err != nil {
			t.Fatal(err)
		}
		// Greedy, stopping at the budget as the server does, so the trunk
		// holds exactly the turn's text but its last token.
		have := len(got[s])
		out, err := sps[s].Round(got[s][have-1], nil, func(row int, l []float32) (int32, bool) {
			return Argmax(l), have+row+1 >= gen
		})
		if err != nil {
			t.Fatal(err)
		}
		got[s] = append(got[s], out...)
	}
	check := func(name string, got, want []int32) {
		t.Helper()
		if len(got) < len(want) {
			t.Fatalf("%s: %d tokens, want %d", name, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: token %d is %d, plain decode says %d\n got %v\nwant %v", name, i, got[i], want[i],
					got[:len(want)], want)
			}
		}
		t.Logf("%s: %d tokens, plain's", name, len(want))
	}

	// A batched plain step of both slots, as the server takes when two
	// conversations decode at once: the trunk runs each slot's pending
	// token and each speculator owes the row (Speculator.Owe).
	var hrow []float32
	stepped := 0
	batched := func() {
		t.Helper()
		if err := g.Speculate(false); err != nil {
			t.Fatal(err)
		}
		in := []int32{got[0][len(got[0])-1], got[1][len(got[1])-1]}
		l, err := g.DecodeRows([]int{0, 1}, in)
		if err != nil {
			t.Fatal(err)
		}
		vocab := len(l) / 2
		for s := range 2 {
			hrow = g.ResidualRow(s, hrow)
			sps[s].Owe(in[s], hrow)
			got[s] = append(got[s], Argmax(l[s*vocab:(s+1)*vocab]))
		}
		stepped++
	}

	begin(1, prefill(1, B, 0))
	begin(0, prefill(0, A1, 0))
	for i := 0; len(got[0]) < gen || len(got[1]) < gen; i++ {
		if i%4 == 3 && len(got[0]) < gen && len(got[1]) < gen {
			batched()
			continue
		}
		for s := range 2 {
			if len(got[s]) < gen {
				round(s)
			}
		}
	}
	if stepped == 0 {
		t.Fatal("no batched step ran")
	}
	check("slot 0, turn 1", got[0], wantA1)
	check("slot 1", got[1], wantB)

	// Slot 0's next turn continues what the first left on the trunk — the
	// prompt and the turn's tokens but its last — the way the server's will:
	// settle the draft, then prefill the rest from where the trunk is.
	use(0)
	if err := g.Speculate(false); err != nil {
		t.Fatal(err)
	}
	if err := sps[0].Settle(); err != nil {
		t.Fatal(err)
	}
	held := g.Ids()
	if n := commonPrefixLen(held, A2); n != len(held) {
		t.Fatalf("slot 0 holds %d tokens of which %d are the second turn's prefix", len(held), n)
	}
	begin(0, prefill(0, A2, len(held)))
	for len(got[0]) < gen {
		round(0)
	}
	check("slot 0, turn 2", got[0], wantA2)
	for s, sp := range sps {
		t.Logf("slot %d: %d rounds, %d tokens, a₁ %.0f%%", s, sp.Stats.Rounds, sp.Stats.Tokens,
			100*sp.Stats.Acceptance())
	}
}

func commonPrefixLen(a, b []int32) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}
