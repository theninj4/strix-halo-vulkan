package llm

// The speculative decode loop (LLM.md P5c, research/p5-mtp-rollback.md §3).
//
// P5a measured the draft head — `a₁ = 74.0%` against the trunk's own argmax —
// and found the optimum depth to be **one**, because the MoE's expert set
// grows with the rows a verification pass carries and the draft's lm head is
// four fifths of a draft step. P5b then built the kernels a two-row pass
// needs, taking it from 3.24 decode steps to 1.17. This file is the round the
// two of them make possible.
//
// **A round at depth one, and why it is not the textbook one.** The trunk is
// at position P with the token `x_P` known and not yet consumed. The draft
// head is handed the trunk's residual and `x_P` and names a candidate for
// `x_{P+1}`; the verification pass runs the two of them as rows P and P+1;
// row P's logits say what `x_{P+1}` really was.
//
//	accepted   the candidate was right, so the pass is the sequence: commit
//	           it, and row P+1's logits name x_{P+2} as well. Two tokens for
//	           one pass and one draft.
//	rejected   the pass folded a token the model did not emit into every
//	           recurrent state and both convolution rings. Rewind it — which
//	           costs nothing, because a rewindable pass writes the slot the
//	           committed state is not in — and the round still learned
//	           `x_{P+1}`, from row P.
//
// **The round after a rejection is the part the design pass got wrong for
// this depth, and it is the whole difference between 1.25x and 1.33x.** After
// a rejection two tokens are known and neither is in the trunk, so the next
// pass is `[x_P, x_{P+1}]` — two certain rows, no draft, one new token. At
// general M that re-run is free, because the next pass is M+1 rows whatever
// happens and the accepted prefix rides in front of the drafts (§3.3c). At
// **R = 2 it is not**, because the prefix fills the pass and displaces the
// speculation. The alternative is for the scan to snapshot S after the first
// row so that a partial accept commits at P+1 — about 113 MB of extra write,
// 0.02 of a step — and it is priced in the write-up rather than built, because
// it is a store inside the one loop in this model with a 512-deep carried
// dependency and `llm_dn_scan.comp`'s header says what a store there costs.
//
// **P20c built that snapshot** (`Partial`, research/p20-llamacpp-mtp.md §7):
// row 0 of a pass is the token the trunk already emitted, so the pass folds
// it alone into the slots it read while the whole pass goes to the other one,
// and a rejection is Graph.KeepFirst — at P+1, drafting again next round. With
// it the chain below collapses to one round type, pass + draft for `1 + a`
// tokens. Off, it is the chain:
//
// The round is then a two-state chain and its multiplier is not `E[tokens]/cost`
// of one round type:
//
//	speculating   cost pass + draft,  2 tokens with probability a, else 1
//	              and a recovery round next
//	recovering    cost pass,          1 token, then speculating again
//
// **The loop is lossless by construction and measured to be so.** Every token
// it emits is an argmax of the trunk's own logits over the trunk's own
// sequence; the draft head only ever chooses which row the trunk runs next,
// never what it emits. What could still break that is the rollback, and the
// loop carries its own check for it: a recovery pass re-runs a token whose
// value is already known, so `row 0`'s argmax has to name that token again.
// If a rejected pass left anything behind, it will not.

import (
	"fmt"
	"time"

	"strix-halo-vulkan/vk"
)

// SpecStats is what a run of the loop cost and where it went.
type SpecStats struct {
	// Rounds is verification passes, Drafted the ones that carried a
	// candidate (the others are recoveries), Accepted the ones whose
	// candidate was right, and Tokens what the loop emitted in all — the
	// prompt's first token included.
	Rounds, Drafted, Accepted, Tokens int
	// Wall is the whole loop; Draft is the draft head, Verify the
	// verification pass, and Decide the host between them — the two argmaxes
	// over 248320 logits and the residual copy the next draft is seeded from.
	Wall, Draft, Verify, Decide time.Duration
	// Prime is the draft's catch-up over the prompt at Start (P20a), outside
	// Wall like the prefill it follows.
	Prime time.Duration
	// Experts is the number of **distinct** experts the last layer's router
	// named across a pass's two rows, summed over the rounds.
	//
	// It is here because it is the one quantity that decides what a
	// verification pass costs and the only one P5 could not measure on decode
	// traffic: a token reads ten of 512 experts and two tokens do not read the
	// same ten, so the MoE's routed half — 1.327 GB of a 4.132 GB token — is
	// multiplied by `E(2)/10`. The design pass read **17** off a two-token
	// prefill of a repeated prompt, which is not the same question as two
	// consecutive tokens of real text.
	//
	// One layer of 48, read back from the arena the MoE block shares across
	// them, so it is a sample rather than the model's total — and a sample of
	// one layer per round over hundreds of rounds.
	Experts, ExpertRounds int
	// DraftedAt and AcceptedAt are per draft position (P20b): index i counts
	// the rounds whose (i+1)th draft was checked — every earlier one was
	// accepted — and the ones where it was right too, so AcceptedAt[i] /
	// DraftedAt[i] is the conditional acceptance a_{i+1}.
	DraftedAt, AcceptedAt [4]int
}

// ExpertsPerPass is the measured E(2): distinct experts a two-row pass routes
// to, in one layer.
func (s SpecStats) ExpertsPerPass() float64 {
	if s.ExpertRounds == 0 {
		return 0
	}
	return float64(s.Experts) / float64(s.ExpertRounds)
}

// Acceptance is the fraction of drafted rounds whose candidate the trunk
// agreed with: `a₁` measured on the loop's own traffic rather than on a
// teacher-forced corpus.
func (s SpecStats) Acceptance() float64 {
	if s.Drafted == 0 {
		return 0
	}
	return float64(s.Accepted) / float64(s.Drafted)
}

// TokensPerRound is what the two-state chain actually delivered.
func (s SpecStats) TokensPerRound() float64 {
	if s.Rounds == 0 {
		return 0
	}
	return float64(s.Tokens) / float64(s.Rounds)
}

// Speculator is the trunk, the draft head, and the round.
type Speculator struct {
	g *Graph
	d *MTPHead
	m *Model
	w Wiring

	// pend is what is known and not yet in the trunk: one token while the
	// loop is speculating, two after a rejection. h is the trunk's wide
	// residual at the position **before** pend[0], which is what the draft
	// head is seeded from, and it is a copy because the next pass overwrites
	// the arena it came out of.
	pend []int32
	h    []float32
	// dh is the draft's own residual a deeper draft chains on (P20b).
	dh []float32

	// CatchUp is P20a (research/p20-llamacpp-mtp.md §3.1): the draft's cache
	// is written over the prompt at Start and over every committed row, so
	// its attention never reads a cell nobody wrote. Off is P5c's loop, which
	// primed nothing and wrote a cell only on a speculating round — the
	// control.
	CatchUp bool
	// oweH and oweX are the rows the draft's cache is behind by: the trunk's
	// residual before each owed position and the token at it. A committed
	// two-row pass leaves exactly one — position P+1, from row 0's residual
	// and row 1's token — because the draft step at P was already seeded with
	// the true residual and the true token.
	oweH []float32
	oweX []int32

	// Partial is P20c (research/p20-llamacpp-mtp.md §3.3): every pass also
	// folds row 0 alone into the committed slots (Graph.SpeculateFirst), so a
	// rejected draft keeps row 0 and the next round drafts again — there is
	// no recovery round. Off is P20a's two-state chain, the control.
	Partial bool
	// Depth is how many tokens a round drafts (P20b): the draft head chains
	// on its own residual, the verification pass is Depth+1 rows, and the
	// longest agreeing prefix is kept (Graph.Keep). Above one it needs
	// Partial and a graph staged with GraphOpts.SpecRows of Depth+1.
	Depth int
	// DraftVocab, when nonzero, restricts the draft to the first DraftVocab
	// ids (§9): BPE id order is merge order, a frequency ranking on the
	// tokenizer's own corpus, so the subset is a prefix. The trunk still
	// decides every token, so the loop stays lossless; a token outside the
	// subset is only never proposed.
	DraftVocab int
	// Trace, when set, sees every keep-prefix round: the position of its
	// row 0, the rows it verified, and how many it kept.
	Trace func(at int, rows []int32, keep int)

	vocab int
	wide  int
	// slot is the sequence slot this speculator drafts for (P20f), or -1 for
	// the single-sequence loop, which never switches the draft's cache.
	slot int

	Stats SpecStats
}

// NewSpeculator wires a staged trunk to a staged draft head.
//
// The graph has to have been built with room for a two-row pass — `MaxTokens`
// and `HeadRows` at least two — because a verification pass wants every row's
// logits and llama.cpp's graph computes one.
func NewSpeculator(g *Graph, d *MTPHead, m *Model, w Wiring) (*Speculator, error) {
	if g == nil || d == nil || m == nil {
		return nil, fmt.Errorf("llm: the speculative loop needs a trunk, a draft head and a model")
	}
	if g.Head() == nil {
		return nil, fmt.Errorf("llm: the speculative loop needs the trunk's lm head")
	}
	if g.MaxTokens() < 2 {
		return nil, fmt.Errorf("llm: the trunk's arenas hold %d rows and a verification pass is 2", g.MaxTokens())
	}
	if g.Head().MaxRows() < 2 {
		return nil, fmt.Errorf("llm: the head's arena holds %d rows of logits and a verification pass needs 2 "+
			"— GraphOpts.HeadRows", g.Head().MaxRows())
	}
	return &Speculator{g: g, d: d, m: m, w: w, vocab: g.Head().Vocab(), wide: d.Wide(), Depth: 1, slot: -1}, nil
}

// The served loop (P20f, research/p20-llamacpp-mtp.md §11). The server does
// not hand a speculator a whole prompt: its prefill is cut into chunks the
// scheduler interleaves with other conversations', a slot keeps what it holds
// between requests, and a token is drawn by the request's own sampler. So a
// speculator a slot is driven in pieces, each run with the trunk's slot live:
//
//	Prime   after each prefill chunk, from the chunk's residual
//	Begin   once the prompt is in
//	Round   for each round of decode, drawing tokens with a pick function
//	Owe     for each plain step the trunk runs on the slot instead
//	Settle  when the request ends, so the draft's cache covers every
//	        committed row and the next turn primes from where this left
//	Rewind  when the trunk is restored to a checkpoint
//
// Between them the speculator's draft is caught up to the trunk's position:
// every cell below it written, and `h` the trunk's residual one row before.

// NewSlotSpeculator is NewSpeculator for one of a multi-slot trunk's
// sequences, with P20's served settings: the draft catches up over every row,
// a rejection keeps row 0, and the round is Depth drafts deep.
func NewSlotSpeculator(g *Graph, d *MTPHead, m *Model, w Wiring, slot, depth int) (*Speculator, error) {
	if slot < 0 || slot >= g.Slots() || slot >= d.Slots() {
		return nil, fmt.Errorf("llm: speculator slot %d; the trunk holds %d and the draft %d", slot, g.Slots(), d.Slots())
	}
	s, err := NewSpeculator(g, d, m, w)
	if err != nil {
		return nil, err
	}
	s.slot, s.CatchUp, s.Partial, s.Depth = slot, true, true, max(depth, 1)
	if g.MaxTokens() < s.Depth+1 || g.Head().MaxRows() < s.Depth+1 {
		return nil, fmt.Errorf("llm: depth %d is a %d-row pass; the trunk's arenas hold %d and its head %d",
			s.Depth, s.Depth+1, g.MaxTokens(), g.Head().MaxRows())
	}
	return s, nil
}

func (s *Speculator) useSlot() error {
	if s.slot < 0 {
		return nil
	}
	return s.d.UseSlot(s.slot)
}

// Prime writes the draft's cells at positions at..at+len(ids)-1 from a chunk
// the trunk has just run: `res` is ExtendResidual's, row r the trunk's
// residual at at+r. Position `at` is seeded from the residual the speculator
// already holds — the row before it — or zeros at position 0, as Start does.
func (s *Speculator) Prime(ids []int32, at int, res []float32) error {
	if err := s.useSlot(); err != nil {
		return err
	}
	if len(res) < len(ids)*s.wide {
		return fmt.Errorf("llm: %d rows of residual for %d ids", len(res)/s.wide, len(ids))
	}
	if at > 0 && len(s.h) != s.wide {
		return fmt.Errorf("llm: priming from position %d without the residual before it", at)
	}
	if at > 0 && at != s.d.Past() {
		return fmt.Errorf("llm: priming the draft at %d; its cache holds %d", at, s.d.Past())
	}
	t0 := time.Now()
	defer func() { s.Stats.Prime += time.Since(t0) }()
	n, chunk := len(ids), s.d.MaxRows()
	hs := make([]float32, 0, chunk*s.wide)
	es := make([]float32, 0, chunk*s.d.NEmbd())
	for p0 := 0; p0 < n; p0 += chunk {
		k := minInt(chunk, n-p0)
		hs, es = hs[:0], es[:0]
		for p := p0; p < p0+k; p++ {
			switch {
			case p > 0:
				hs = append(hs, res[(p-1)*s.wide:p*s.wide]...)
			case at > 0:
				hs = append(hs, s.h...)
			default:
				hs = append(hs, make([]float32, s.wide)...)
			}
			e, err := s.m.Embedding(ids[p])
			if err != nil {
				return fmt.Errorf("llm: token_embd for the draft: %w", err)
			}
			es = append(es, e...)
		}
		if _, _, _, err := s.d.Rows(hs, es, k, at+p0, s.w, false); err != nil {
			return fmt.Errorf("llm: priming the draft at %d: %w", at+p0, err)
		}
	}
	s.h = append(s.h[:0], res[(n-1)*s.wide:n*s.wide]...)
	s.oweH, s.oweX = s.oweH[:0], s.oweX[:0]
	return nil
}

// Begin checks that the draft is caught up to the trunk and arms the
// partial accept, before the first Round.
func (s *Speculator) Begin() error {
	if err := s.useSlot(); err != nil {
		return err
	}
	if got, want := s.d.Past()+len(s.oweX), s.g.Past(); got != want {
		return fmt.Errorf("llm: the draft is at %d and the trunk at %d", got, want)
	}
	if len(s.h) != s.wide {
		return fmt.Errorf("llm: beginning without the trunk's residual")
	}
	return s.g.SpeculateFirst(true)
}

// Round is one round of the served loop from `first`, the token named last
// and not yet run: Depth drafts, one verification pass, and the tokens `pick`
// drew, the last of which is pending in the same sense. The trunk must be
// speculating (Graph.Speculate) with this speculator's slot live.
func (s *Speculator) Round(first int32, out []int32, pick func(row int, logits []float32) (int32, bool)) ([]int32, error) {
	return s.RoundDraw(first, out, nil, func(row int, logits []float32, _ int32, _ *Dist) (int32, bool) {
		return pick(row, logits)
	})
}

// RoundDraw is Round with the draft's pick named too (P20g): `propose` draws
// each draft from the draft's logits (nil: the argmax) and returns the
// distribution it drew from, and `pick` is handed the draft proposed after
// its row and that distribution — nil past the last draft — so it can verify
// by speculative sampling (Sampler.Verify) rather than draw and compare.
func (s *Speculator) RoundDraw(first int32, out []int32, propose func(logits []float32) (int32, *Dist),
	pick func(row int, logits []float32, draft int32, q *Dist) (int32, bool)) ([]int32, error) {
	if len(s.h) != s.wide {
		return nil, fmt.Errorf("llm: a round without the trunk's residual")
	}
	if len(s.oweX)+1 > s.d.MaxRows() {
		// A long run of plain steps (Owe) is more than one draft call.
		if err := s.settleOwed(); err != nil {
			return nil, err
		}
	}
	s.pend = append(s.pend[:0], first)
	return s.round(out[:0], propose, pick)
}

// Owe records a plain decode step the trunk ran on this slot — token x at its
// position, and `h` the trunk's residual at that row — so the draft stays
// caught up through steps that did not speculate (the server batches the
// conversations decoding together, CONCURRENCY.md C5). The row is written
// into the draft's cache by the next round's draft call or by Settle.
func (s *Speculator) Owe(x int32, h []float32) {
	s.oweH = append(s.oweH, s.h...)
	s.oweX = append(s.oweX, x)
	s.h = append(s.h[:0], h...)
}

// Settle writes the draft's cells for the rows it owes — the committed rows
// of the last round but its last — so that its cache covers everything the
// trunk holds. Nothing is pending after it.
func (s *Speculator) Settle() error {
	s.pend = s.pend[:0]
	if err := s.useSlot(); err != nil {
		return err
	}
	if err := s.settleOwed(); err != nil {
		return err
	}
	// A deeper draft wrote cells past the trunk; they are masked once the
	// position says so.
	if s.d.Past() > s.g.Past() {
		return s.d.Rewind(s.g.Past())
	}
	return nil
}

// settleOwed writes the owed rows, a draft arena's worth at a time; they end
// at the trunk's position.
func (s *Speculator) settleOwed() error {
	if err := s.useSlot(); err != nil {
		return err
	}
	n, chunk := len(s.oweX), s.d.MaxRows()
	at := s.g.Past() - n
	es := make([]float32, 0, minInt(n, chunk)*s.d.NEmbd())
	for p0 := 0; p0 < n; p0 += chunk {
		k := minInt(chunk, n-p0)
		es = es[:0]
		for _, x := range s.oweX[p0 : p0+k] {
			e, err := s.m.Embedding(x)
			if err != nil {
				return fmt.Errorf("llm: token_embd for the draft: %w", err)
			}
			es = append(es, e...)
		}
		if _, _, _, err := s.d.Rows(s.oweH[p0*s.wide:(p0+k)*s.wide], es, k, at+p0, s.w, false); err != nil {
			return fmt.Errorf("llm: settling the draft at %d: %w", at+p0, err)
		}
	}
	s.oweH, s.oweX = s.oweH[:0], s.oweX[:0]
	return nil
}

// Residual is the trunk's residual one row before its position, which a
// checkpoint keeps beside the trunk's state so Rewind can resume from it.
func (s *Speculator) Residual() []float32 { return append([]float32(nil), s.h...) }

// Rewind takes the draft back to position `past` — the trunk has just been
// restored there — resuming from `h`, the residual Residual returned when the
// checkpoint was taken. The cells below `past` must still be the ones the
// checkpoint's tokens wrote, which is the caller's to know.
func (s *Speculator) Rewind(past int, h []float32) error {
	if err := s.useSlot(); err != nil {
		return err
	}
	if len(h) != s.wide {
		return fmt.Errorf("llm: rewinding the draft without the residual before %d", past)
	}
	if err := s.d.Rewind(past); err != nil {
		return err
	}
	s.h = append(s.h[:0], h...)
	s.pend, s.oweH, s.oweX = s.pend[:0], s.oweH[:0], s.oweX[:0]
	return nil
}

// Start runs the prompt as an ordinary prefill and returns the token the
// trunk emits from it. The loop is speculative from here on.
func (s *Speculator) Start(ids []int32) (int32, error) {
	if len(ids) == 0 {
		return 0, fmt.Errorf("llm: an empty prompt")
	}
	if err := s.g.Speculate(false); err != nil {
		return 0, err
	}
	if err := s.g.SpeculateFirst(s.Partial); err != nil {
		return 0, err
	}
	s.d.Reset()
	s.Stats = SpecStats{}
	s.oweH, s.oweX = s.oweH[:0], s.oweX[:0]
	t0 := time.Now()
	var first int32
	if s.CatchUp {
		logits, res, err := s.g.ForwardResidual(ids)
		if err != nil {
			return 0, err
		}
		first = Argmax(logits)
		if err := s.prime(ids, res); err != nil {
			return 0, err
		}
		s.h = append(s.h[:0], res[(len(ids)-1)*s.wide:]...)
	} else {
		logits, _, err := s.g.Forward(ids)
		if err != nil {
			return 0, err
		}
		first = Argmax(logits)
		// `Forward` ends in `hidden`, which moves the last row of the
		// residual to the front and runs the final mixer on it — so the
		// arena holds one row and it is the one the draft wants.
		s.h = append(s.h[:0], s.g.Residual()...)
	}
	s.pend = append(s.pend[:0], first)
	s.Stats.Tokens = 1
	s.Stats.Wall += time.Since(t0) - s.Stats.Prime
	return first, s.g.Speculate(true)
}

// prime writes the draft's cell at every prompt position: position p from
// the trunk's residual at p-1 and the token at p, a draft-arena's rows at a
// time. Position 0 has no predecessor and is seeded from a zero residual, so
// the cell is a function of the prompt rather than of whatever the previous
// sequence left there.
func (s *Speculator) prime(ids []int32, res []float32) error {
	t0 := time.Now()
	defer func() { s.Stats.Prime += time.Since(t0) }()
	n, chunk := len(ids), s.d.MaxRows()
	hs := make([]float32, 0, chunk*s.wide)
	es := make([]float32, 0, chunk*s.d.NEmbd())
	for p0 := 0; p0 < n; p0 += chunk {
		k := minInt(chunk, n-p0)
		hs, es = hs[:0], es[:0]
		for p := p0; p < p0+k; p++ {
			if p == 0 {
				hs = append(hs, make([]float32, s.wide)...)
			} else {
				hs = append(hs, res[(p-1)*s.wide:p*s.wide]...)
			}
			e, err := s.m.Embedding(ids[p])
			if err != nil {
				return fmt.Errorf("llm: token_embd for the draft: %w", err)
			}
			es = append(es, e...)
		}
		if _, _, _, err := s.d.Rows(hs, es, k, p0, s.w, false); err != nil {
			return fmt.Errorf("llm: priming the draft at %d: %w", p0, err)
		}
	}
	return nil
}

// Next runs one round and returns the tokens it determined: two when the
// draft was right, one otherwise.
//
// The slice is reused between calls.
func (s *Speculator) Next(out []int32) ([]int32, error) {
	out = out[:0]
	if len(s.pend) == 0 {
		return nil, fmt.Errorf("llm: the loop has not been started")
	}
	if s.Partial {
		return s.nextPrefix(out)
	}
	if s.Depth > 1 {
		return nil, fmt.Errorf("llm: depth %d needs the partial accept (Speculator.Partial)", s.Depth)
	}
	top := time.Now()
	defer func() { s.Stats.Wall += time.Since(top) }()
	at := s.g.Past()

	// 1. The candidate, or the accepted prefix a rejection left behind.
	rows := make([]int32, 0, 2)
	rows = append(rows, s.pend[0])
	drafted := len(s.pend) == 1
	if drafted {
		t0 := time.Now()
		// The owed rows ride in front of the drafting one, at the
		// positions just before it.
		hs := append(s.oweH, s.h...)
		xs := append(s.oweX, s.pend[0])
		es := make([]float32, 0, len(xs)*s.d.NEmbd())
		for _, x := range xs {
			e, err := s.m.Embedding(x)
			if err != nil {
				return nil, fmt.Errorf("llm: token_embd for the draft: %w", err)
			}
			es = append(es, e...)
		}
		lg, _, _, err := s.d.Rows(hs, es, len(xs), at-len(s.oweX), s.w, true)
		if err != nil {
			return nil, fmt.Errorf("llm: draft at %d: %w", at, err)
		}
		s.oweH, s.oweX = hs[:0], xs[:0]
		rows = append(rows, s.draftArgmax(lg))
		s.Stats.Draft += time.Since(t0)
	} else {
		rows = append(rows, s.pend[1])
	}

	// 2. The verification pass, over both rows.
	t0 := time.Now()
	logits, res, err := s.g.ExtendRows(rows)
	if err != nil {
		return nil, fmt.Errorf("llm: verification pass at %d: %w", at, err)
	}
	s.Stats.Verify += time.Since(t0)
	t0 = time.Now()
	s.Stats.Rounds++

	// 3. Row 0's argmax is the trunk's own next token, whatever the draft
	//    said. On a recovery pass it is a token that is already known, and
	//    that is this loop's standing check on the rollback: if the rejected
	//    pass had left anything in a ring or a state, the re-run would not
	//    reproduce it.
	first := Argmax(logits[:s.vocab])
	if !drafted && first != s.pend[1] {
		return nil, fmt.Errorf("llm: re-running position %d named %d where the pass that was rewound named %d — "+
			"the rollback did not restore the sequence", at+1, first, s.pend[1])
	}
	out = append(out, first)
	keep := !drafted || rows[1] == first
	if drafted {
		s.Stats.Drafted++
		if keep {
			s.Stats.Accepted++
		}
	}
	if keep {
		// Both rows are the sequence, so the pass is the sequence. Row 1's
		// logits are a free token and row 1's residual seeds the next draft.
		if !drafted {
			// A recovery pass emits only its second row's token: the first
			// was determined by the round that was rewound.
			out = out[:0]
		}
		out = append(out, Argmax(logits[s.vocab:]))
		s.h = append(s.h[:0], res[s.wide:2*s.wide]...)
		if s.CatchUp {
			s.oweH = append(s.oweH, res[:s.wide]...)
			s.oweX = append(s.oweX, rows[1])
		}
		s.pend = append(s.pend[:0], out[len(out)-1])
		if err := s.g.Commit(); err != nil {
			return nil, err
		}
	} else if s.Partial {
		// P20c: row 0 is already in the committed slots, so the sequence is
		// at P+1 with `first` known and not yet in the trunk — exactly a
		// speculating round's starting point. Row 0's residual seeds the
		// next draft with `first`, and the draft owes nothing: its cell at P
		// was written this round from the true pair.
		if err := s.g.KeepFirst(); err != nil {
			return nil, err
		}
		s.h = append(s.h[:0], res[:s.wide]...)
		s.pend = append(s.pend[:0], first)
	} else {
		// The pass ran a token the model did not emit. Throw it away; the
		// next round re-runs `[pend[0], first]` and drafts nothing.
		if err := s.g.Rewind(); err != nil {
			return nil, err
		}
		s.pend = append(s.pend[:1], first)
	}
	s.Stats.Tokens += len(out)
	if n := s.g.PassExperts(); n > 0 {
		s.Stats.Experts += n
		s.Stats.ExpertRounds++
	}
	s.Stats.Decide += time.Since(t0)
	return out, nil
}

// draftArgmax is the draft's pick, over DraftVocab ids when it is set.
func (s *Speculator) draftArgmax(lg []float32) int32 {
	return Argmax(s.draftLogits(lg))
}

// draftLogits is the draft's row cut to the DraftVocab ids it proposes from.
func (s *Speculator) draftLogits(lg []float32) []float32 {
	if s.DraftVocab > 0 && s.DraftVocab < len(lg) {
		lg = lg[:s.DraftVocab]
	}
	return lg
}

// nextPrefix is a round of the keep-prefix loop (P20c, P20b): draft Depth
// tokens, verify them in one Depth+1-row pass, keep the longest prefix the
// trunk agrees with, and emit one token more than the drafts it accepted.
//
// After keeping k rows the trunk is at P+k with the token row k-1 named
// pending, the draft is seeded from row k-1's residual, and the draft owes
// the cells P+1..P+k-1: each from the row before it and the token at it.
func (s *Speculator) nextPrefix(out []int32) ([]int32, error) {
	return s.round(out, nil, func(_ int, logits []float32, _ int32, _ *Dist) (int32, bool) { return Argmax(logits), false })
}

// round is nextPrefix with the trunk's token at each row named by `pick`
// rather than its argmax (P20f). Row i's token is drawn from row i's logits
// over a prefix that is the sequence — every earlier draft agreed with what
// was drawn — so a sampler that draws each row in order is drawing from the
// trunk's own distribution whatever the draft said: the loop is lossless at
// any temperature, and only its acceptance falls. `pick` returning stop ends
// the round at that row (an end-of-generation token, the budget): the rows
// after it are not kept even where the draft agreed.
func (s *Speculator) round(out []int32, propose func([]float32) (int32, *Dist),
	pick func(row int, logits []float32, draft int32, q *Dist) (int32, bool)) ([]int32, error) {
	if propose == nil {
		propose = func(lg []float32) (int32, *Dist) { return Argmax(lg), nil }
	}
	top := time.Now()
	defer func() { s.Stats.Wall += time.Since(top) }()
	at := s.g.Past()
	depth := max(s.Depth, 1)
	if err := s.useSlot(); err != nil {
		return nil, err
	}

	// 1. The drafts: the owed rows and the first draft in one call, then
	//    each further draft from the draft's own residual.
	t0 := time.Now()
	rows := make([]int32, 0, depth+1)
	rows = append(rows, s.pend[0])
	hs := append(s.oweH, s.h...)
	xs := append(s.oweX, s.pend[0])
	es := make([]float32, 0, len(xs)*s.d.NEmbd())
	for _, x := range xs {
		e, err := s.m.Embedding(x)
		if err != nil {
			return nil, fmt.Errorf("llm: token_embd for the draft: %w", err)
		}
		es = append(es, e...)
	}
	lg, res, _, err := s.d.Rows(hs, es, len(xs), at-len(s.oweX), s.w, true)
	if err != nil {
		return nil, fmt.Errorf("llm: draft at %d: %w", at, err)
	}
	s.oweH, s.oweX = hs[:0], xs[:0]
	qs := make([]*Dist, 0, depth)
	d, q := propose(s.draftLogits(lg))
	rows, qs = append(rows, d), append(qs, q)
	for k := 1; k < depth; k++ {
		// A copy: the residual is in the arena the next call seeds.
		s.dh = append(s.dh[:0], res...)
		e, err := s.m.Embedding(rows[k])
		if err != nil {
			return nil, fmt.Errorf("llm: token_embd for the draft: %w", err)
		}
		if lg, res, _, err = s.d.Rows(s.dh, e, 1, at+k, s.w, true); err != nil {
			return nil, fmt.Errorf("llm: draft %d at %d: %w", k+1, at+k, err)
		}
		d, q := propose(s.draftLogits(lg))
		rows, qs = append(rows, d), append(qs, q)
	}
	s.Stats.Draft += time.Since(t0)

	// 2. The verification pass.
	if s.slot >= 0 && s.g.Slot() != s.slot {
		return nil, fmt.Errorf("llm: the speculator is slot %d's and the trunk's live slot is %d", s.slot, s.g.Slot())
	}
	t0 = time.Now()
	logits, vres, err := s.g.ExtendRows(rows)
	if err != nil {
		return nil, fmt.Errorf("llm: verification pass at %d: %w", at, err)
	}
	s.Stats.Verify += time.Since(t0)
	t0 = time.Now()
	s.Stats.Rounds++
	s.Stats.Drafted++

	// 3. Row i's argmax is the trunk's token at P+i+1; the pass is the
	//    sequence up to the first draft it disagrees with.
	n := len(rows)
	for i := 0; i < n; i++ {
		draft, q := int32(-1), (*Dist)(nil)
		if i+1 < n {
			draft, q = rows[i+1], qs[i]
		}
		g, stop := pick(i, logits[i*s.vocab:(i+1)*s.vocab], draft, q)
		out = append(out, g)
		if stop || i+1 == n {
			break
		}
		s.Stats.DraftedAt[min(i, 3)]++
		if rows[i+1] != g {
			break
		}
		s.Stats.AcceptedAt[min(i, 3)]++
	}
	keep := len(out)
	if keep > 1 {
		s.Stats.Accepted++
	}
	if s.Trace != nil {
		s.Trace(at, rows, keep)
	}
	if err := s.g.Keep(keep); err != nil {
		return nil, err
	}
	if s.CatchUp {
		for i := 1; i < keep; i++ {
			s.oweH = append(s.oweH, vres[(i-1)*s.wide:i*s.wide]...)
			s.oweX = append(s.oweX, rows[i])
		}
	}
	s.h = append(s.h[:0], vres[(keep-1)*s.wide:keep*s.wide]...)
	s.pend = append(s.pend[:0], out[keep-1])
	s.Stats.Tokens += len(out)
	if n := s.g.PassExperts(); n > 0 {
		s.Stats.Experts += n
		s.Stats.ExpertRounds++
	}
	s.Stats.Decide += time.Since(t0)
	return out, nil
}

// Stop leaves the graph on the ordinary decode path, so that a caller can go
// on generating without the rollback — and so that P1c's pre-recorded step is
// captured again.
func (s *Speculator) Stop() error { return s.g.Speculate(false) }

// StageDraft opens a draft checkpoint and stages `blk.48` beside a trunk that
// is already on the device, borrowing the trunk's lm head.
//
// It is here rather than in the command because both the loop's benchmark and
// anything that serves the loop need the same three lines and the same
// argument about `maxTok`: the draft runs one row at depth one, whatever the
// verification pass is.
func StageDraft(dev *vk.Device, draft *Model, g *Graph) (*MTPHead, error) {
	return NewMTPHeadSlots(dev, draft, g.cfg, g.Head(), DraftRows, g.NKV(), g.Slots())
}

// SpecDepthDefault is the drafts a round (P20e, measured 2026-10-01): on
// SPEED-Bench's qualitative prompts depth 2 reads 1.48x against depth 1's
// 1.34x (a₁ 0.85, a₂|a₁ ~0.84); on the 73%-acceptance prose prompt it gives
// back 0.04 (1.20x against 1.24x). A graph for it is staged with
// GraphOpts.SpecRows = depth+1.
const SpecDepthDefault = 2

// DraftVocabDefault is the draft's vocabulary prefix (§9, measured
// 2026-10-01): the first 65 536 ids cover 96.6% of wikitext and 98.3% of Go
// source, the loop's acceptance is unchanged on prose (108/148) and 92.0 →
// 85.7% on memorised wikitext, and the draft step is 3.7 → 2.2 ms.
const DraftVocabDefault = 65536

// DraftRows is how many rows the draft's arenas hold: the prompt is primed
// that many positions a pass, and a round's owed rows plus its drafting row
// (two at depth one) fit with room to spare.
const DraftRows = 64

// ResetGraphStats and GraphStats expose the trunk's own attribution, so that a
// multiplier that is not the projected one can be charged to a block.
func (s *Speculator) ResetGraphStats()       { s.g.ResetStats() }
func (s *Speculator) GraphStats() GraphStats { return s.g.Stats }
