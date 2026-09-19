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
// So the round is a two-state chain and its multiplier is not `E[tokens]/cost`
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

	vocab int
	wide  int

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
	return &Speculator{g: g, d: d, m: m, w: w, vocab: g.Head().Vocab(), wide: d.Wide()}, nil
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
	s.d.Reset()
	s.Stats = SpecStats{}
	t0 := time.Now()
	logits, _, err := s.g.Forward(ids)
	if err != nil {
		return 0, err
	}
	first := Argmax(logits)
	// `Forward` ends in `hidden`, which moves the last row of the residual to
	// the front and runs the final mixer on it — so the arena holds one row
	// and it is the one the draft wants.
	s.h = append(s.h[:0], s.g.Residual()...)
	s.pend = append(s.pend[:0], first)
	s.Stats.Tokens = 1
	s.Stats.Wall += time.Since(t0)
	return first, s.g.Speculate(true)
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
	top := time.Now()
	defer func() { s.Stats.Wall += time.Since(top) }()
	at := s.g.Past()

	// 1. The candidate, or the accepted prefix a rejection left behind.
	rows := make([]int32, 0, 2)
	rows = append(rows, s.pend[0])
	drafted := len(s.pend) == 1
	if drafted {
		t0 := time.Now()
		embd, err := s.m.Embedding(s.pend[0])
		if err != nil {
			return nil, fmt.Errorf("llm: token_embd for the draft: %w", err)
		}
		lg, _, _, err := s.d.Step(s.h, embd, at, s.w)
		if err != nil {
			return nil, fmt.Errorf("llm: draft at %d: %w", at, err)
		}
		rows = append(rows, Argmax(lg))
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
		s.pend = append(s.pend[:0], out[len(out)-1])
		if err := s.g.Commit(); err != nil {
			return nil, err
		}
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
	return NewMTPHead(dev, draft, g.cfg, g.Head(), 1, g.NKV())
}

// ResetGraphStats and GraphStats expose the trunk's own attribution, so that a
// multiplier that is not the projected one can be charged to a block.
func (s *Speculator) ResetGraphStats()       { s.g.ResetStats() }
func (s *Speculator) GraphStats() GraphStats { return s.g.Stats }
