package lm

// The two phases end to end (MUSIC.md A7c), as generate_with_stop_condition
// runs them with thinking on and upstream's defaults:
//
//  1. Think: unless the request gives bpm, key, time signature and duration
//     (HasAll), the LM writes a CoT under the FSM, one row, no CFG, and the
//     metadata is parsed out of it (ParseCoT).
//  2. Codes: the metadata is re-serialised (Meta.CoT), the conditional
//     prompt ends in it and the unconditional one in an empty reasoning
//     block, and the pair decode together under CFG, the same token appended
//     to both, until 5 codes a second have been written and EOS is forced.

import (
	"fmt"
	"math/rand/v2"
	"strconv"
	"time"

	"strix-halo-vulkan/zimage/tokenizer"
)

// Planner runs the LM's phases on a resident GPU.
type Planner struct {
	G  *GPU
	Tk *tokenizer.Tokenizer
	S  Sampling
	v  *fsmVocab

	// Between, when set, runs after every decode step: a server yields the
	// device there, and an error (a cancelled request) stops the phase.
	Between func() error
	// Step, when set, sees each phase's progress: phase 1 or 2, steps done,
	// and the phase's length (0 for phase 1, whose length is the model's).
	Step func(phase, done, total int)
}

// after is the hooks' call after a step.
func (p *Planner) after(phase, done, total int) error {
	if p.Step != nil {
		p.Step(phase, done, total)
	}
	if p.Between != nil {
		return p.Between()
	}
	return nil
}

// NewPlanner builds the FSM's tables from the LM's tokenizer.
func NewPlanner(g *GPU, tk *tokenizer.Tokenizer) (*Planner, error) {
	v, err := newFSMVocab(tk)
	if err != nil {
		return nil, err
	}
	return &Planner{G: g, Tk: tk, S: DefaultSampling, v: v}, nil
}

// Stats is where a phase's time went.
type Stats struct {
	Prompt, Tokens int
	Prefill, Steps time.Duration
}

func (s Stats) String() string {
	per := 0.0
	if s.Tokens > 0 {
		per = float64(s.Steps.Microseconds()) / 1e3 / float64(s.Tokens)
	}
	return fmt.Sprintf("%d prompt tokens in %v, %d steps in %v (%.1f ms a step)", s.Prompt,
		s.Prefill.Round(time.Millisecond), s.Tokens, s.Steps.Round(time.Millisecond), per)
}

// maxCoT is _compute_max_new_tokens for phase 1: no target duration, so the
// 3,500-token cap (DURATION_MAX·5 + 500).
const maxCoT = durationMax*CodesPerSecond + 500

// prefill runs every row's prompt but its last token, all in passes of at
// most the arena's rows.
func (p *Planner) prefill(rows [][]int32) (time.Duration, error) {
	var all []Tok
	for s, r := range rows {
		if len(r) >= p.G.o.MaxLen {
			return 0, fmt.Errorf("lm: a %d-token prompt; the cache holds %d", len(r), p.G.o.MaxLen)
		}
		for i, id := range r[:len(r)-1] {
			all = append(all, Tok{ID: id, Slot: s, Pos: i})
		}
	}
	t0 := time.Now()
	for len(all) > 0 {
		n := min(len(all), p.G.o.Rows)
		if _, _, err := p.G.Pass(all[:n], Range{}); err != nil {
			return 0, err
		}
		all = all[n:]
	}
	return time.Since(t0), nil
}

// Think is phase 1. With every meta given it returns them without running
// the LM, as upstream does. The text is what the LM wrote.
func (p *Planner) Think(caption, lyrics string, user Meta, rng *rand.Rand) (Meta, string, Stats, error) {
	var st Stats
	if user.HasAll() {
		out := Meta{}
		for k, v := range user {
			out[k] = v
		}
		return out, "", st, nil
	}
	ids, err := p.Tk.Encode(Phase1Prompt(caption, lyrics))
	if err != nil {
		return nil, "", st, err
	}
	st.Prompt = len(ids)
	if st.Prefill, err = p.prefill([][]int32{ids}); err != nil {
		return nil, "", st, err
	}
	f := newFSM(p.v, user)
	step := []Tok{{ID: ids[len(ids)-1], Slot: 0, Pos: len(ids) - 1}}
	var gen []int32
	t0 := time.Now()
	for len(gen) < maxCoT {
		if step[0].Pos >= p.G.o.MaxLen {
			return nil, "", st, fmt.Errorf("lm: the CoT outgrew the cache")
		}
		out, _, err := p.G.Pass(step, Range{Lo: 0, Hi: p.G.vocab})
		if err != nil {
			return nil, "", st, err
		}
		scores := out[0]
		f.Apply(scores)
		topP(scores, p.S.TopP)
		tok := int32(draw(scores, p.S.Temperature, rng))
		f.Update(tok)
		gen = append(gen, tok)
		if tok == ImEndID || tok == EndOfTextID {
			break
		}
		if err := p.after(1, len(gen), 0); err != nil {
			return nil, "", st, err
		}
		step[0].ID = tok
		step[0].Pos++
	}
	st.Steps, st.Tokens = time.Since(t0), len(gen)
	text, err := p.Tk.Decode(gen)
	if err != nil {
		return nil, "", st, err
	}
	return ParseCoT(text), text, st, nil
}

// TargetCodes is the codes phase's length: `int(duration * 5)` of the
// request's duration, or the CoT's when the request gave none.
func TargetCodes(seconds float64, meta Meta) (int, error) {
	if seconds <= 0 {
		var err error
		if seconds, err = parseFloat(meta["duration"]); err != nil || seconds <= 0 {
			return 0, fmt.Errorf("lm: no duration in the request or the CoT (%q)", meta["duration"])
		}
	}
	return int(seconds * CodesPerSecond), nil
}

// Codes is phase 2: the audio codes (0..63999), five a second.
func (p *Planner) Codes(caption, lyrics string, meta Meta, target int, rng *rand.Rand) ([]int, Stats, error) {
	var st Stats
	cond, err := p.Tk.Encode(Phase2Prompt(caption, lyrics, meta.CoT()))
	if err != nil {
		return nil, st, err
	}
	uncond, err := p.Tk.Encode(Phase2Uncond())
	if err != nil {
		return nil, st, err
	}
	st.Prompt = len(cond) + len(uncond)
	if len(cond)+target+1 > p.G.o.MaxLen {
		return nil, st, fmt.Errorf("lm: %d prompt tokens and %d codes; the cache holds %d", len(cond), target, p.G.o.MaxLen)
	}
	if st.Prefill, err = p.prefill([][]int32{cond, uncond}); err != nil {
		return nil, st, err
	}
	step := []Tok{{ID: cond[len(cond)-1], Slot: 0, Pos: len(cond) - 1},
		{ID: uncond[len(uncond)-1], Slot: 1, Pos: len(uncond) - 1}}
	var codes []int
	t0 := time.Now()
	for len(codes) <= target {
		out, _, err := p.G.Pass(step, Range{Lo: ImEndID, Hi: CodeBase + NumCodes})
		if err != nil {
			return nil, st, err
		}
		scores := codeScores(out[0], out[1], p.S.CFG, len(codes), target)
		topP(scores, p.S.TopP)
		c := draw(scores, p.S.Temperature, rng)
		st.Tokens++
		if c == NumCodes {
			break // EOS
		}
		codes = append(codes, c)
		if err := p.after(2, len(codes), target); err != nil {
			return nil, st, err
		}
		for s := range step {
			step[s].ID = int32(CodeBase + c)
			step[s].Pos++
		}
	}
	st.Steps = time.Since(t0)
	return codes, st, nil
}

func parseFloat(s string) (float64, error) { return strconv.ParseFloat(pyStrip(s), 64) }

// CoTBudget bounds the CoT's tokens as the codes phase's prompt carries it:
// the FSM caps the caption at 512 and the other fields are ~40.
const CoTBudget = captionMaxTokens + 88

// Positions is how much of a slot's cache a request needs at most: the
// phase-1 prompt, the CoT the codes prompt adds to it, and the codes.
func (p *Planner) Positions(caption, lyrics string, seconds float64) (int, error) {
	ids, err := p.Tk.Encode(Phase1Prompt(caption, lyrics))
	if err != nil {
		return 0, err
	}
	return len(ids) + CoTBudget + int(seconds*CodesPerSecond) + 1, nil
}

// MaxLen is the positions a slot's cache holds.
func (p *Planner) MaxLen() int { return p.G.o.MaxLen }
