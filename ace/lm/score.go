package lm

// The LM's half of upstream's auto score (MUSIC.md A14b),
// core/scoring/lm_score.calculate_pmi_score_per_condition: the song's codes
// go in A13's understand prompt, and the LM is teacher-forced on each
// condition the song was asked for.
//
//   - The caption and the lyrics are scored by PMI: the mean log p of the
//     condition's text given the codes, less the same given the codes
//     string "NO USER INPUT", squashed by sigmoid(PMI / scale).
//   - bpm, duration, genres, keyscale, language and timesignature are each
//     scored by top-10 recall: a target position scores 1 − (rank − 1)/10
//     when its token is in the LM's top 10, else 0, averaged.
//   - The reward weighs caption 0.5, lyrics 0.3 and the metadata's mean 0.2,
//     renormalised over the ones present.
//
// Upstream runs one forward over prompt + target per condition. Here the
// two prompts are prefilled once, into slots 0 and 1, and each target is a
// teacher-forced pass from the prompt's last token, which rewrites the
// cache past the prompt as it goes: the same logits, ten short passes
// instead of ten full ones.

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// ScoreScale is the PMI's sigmoid scale: Gradio's shipped default
// (user_preferences.py), against calculate_pmi_score_per_condition's 0.1.
const ScoreScale = 0.5

// scoreTopK is the recall's k (scoring.py passes 10).
const scoreTopK = 10

// recallKeys are metadata_recall_keys, in the order upstream scores them.
var recallKeys = []string{"bpm", "duration", "genres", "keyscale", "language", "timesignature"}

// Conditions are what a song was asked for, as the score reads them.
type Conditions struct {
	Caption, Lyrics string
	// Meta holds the recalled fields: bpm, duration, genres, keyscale,
	// language, timesignature. bpm and duration are ints to yaml.dump when
	// they are digits (parse_lm_output's types); the rest are strings.
	Meta Meta
}

// Score is the LM's verdict on a song.
type Score struct {
	// Conditions are the per-condition scores in [0, 1]: "caption" and
	// "lyrics" by PMI, the metadata fields by top-10 recall.
	Conditions map[string]float64
	// PMI is the caption's and the lyrics' raw PMI, in nats a token.
	PMI map[string]float64
	// Reward is calculate_reward_score's weighted sum.
	Reward float64
	Stats  Stats
}

// fieldYAML is `yaml.dump({k: v}, allow_unicode=True, sort_keys=True).strip()`.
func fieldYAML(k, v string) string {
	if (k == "bpm" || k == "duration") && isDigits(v) {
		v = strings.TrimLeft(v, "0")
		if v == "" {
			v = "0"
		}
		return k + ": " + v
	}
	return strings.TrimSpace(k + ":" + yamlScalar(v, len(k)+1))
}

// ScoreTargets are the texts upstream teacher-forces, in its order, and
// whether each is scored by PMI (against the unconditional prompt too) or
// by recall.
func ScoreTargets(c Conditions) (keys, texts []string, pmi []bool) {
	for _, k := range recallKeys {
		if v, ok := c.Meta[k]; ok {
			keys = append(keys, k)
			texts = append(texts, "<think>\n"+fieldYAML(k, v)+"\n</think>\n")
			pmi = append(pmi, false)
		}
	}
	keys = append(keys, "caption")
	texts = append(texts, "<think>\n"+fieldYAML("caption", c.Caption)+"\n</think>\n")
	pmi = append(pmi, true)
	if c.Lyrics != "" {
		keys = append(keys, "lyrics")
		texts = append(texts, "<think>\n</think>\n# Lyric\n"+c.Lyrics+"\n")
		pmi = append(pmi, true)
	}
	return keys, texts, pmi
}

// ScorePrompts are the conditional prompt over the codes and the
// unconditional one: build_formatted_prompt_for_understanding over the
// codes, and over the codes string "NO USER INPUT".
func (p *Planner) ScorePrompts(codes []int) (cond, uncond []int32, err error) {
	if cond, err = p.UnderstandPrompt(codes); err != nil {
		return nil, nil, err
	}
	uncond, err = p.Tk.Encode(chatPrompt(UnderstandInstruction, NoUserInput))
	return cond, uncond, err
}

// Teach is the teacher-forced pass over one target: the prompt (already
// prefilled but its last token, in slot) and then the target, each row
// scoring the token after it. It returns the log p and rank of every target
// token.
func (p *Planner) Teach(slot int, prompt, target []int32) (logp []float64, rank []int, took time.Duration, err error) {
	if len(target) == 0 {
		return nil, nil, 0, nil
	}
	if len(prompt)+len(target) > p.G.o.MaxLen {
		return nil, nil, 0, fmt.Errorf("lm: %d prompt and %d target tokens; the cache holds %d", len(prompt), len(target), p.G.o.MaxLen)
	}
	// Row i is the token at position P−1+i; it predicts target[i].
	ids := append([]int32{prompt[len(prompt)-1]}, target[:len(target)-1]...)
	for lo := 0; lo < len(ids); lo += ScoreRows {
		hi := min(lo+ScoreRows, len(ids))
		toks := make([]Tok, hi-lo)
		for i := range toks {
			toks[i] = Tok{ID: ids[lo+i], Slot: slot, Pos: len(prompt) - 1 + lo + i}
		}
		lp, rk, d, err := p.G.Teacher(toks, target[lo:hi])
		if err != nil {
			return nil, nil, took, err
		}
		logp, rank, took = append(logp, lp...), append(rank, rk...), took+d
	}
	return logp, rank, took, nil
}

// recall is _calculate_topk_recall's position-weighted score.
func recall(rank []int) float64 {
	if len(rank) == 0 {
		return 0
	}
	s := 0.0
	for _, r := range rank {
		if r <= scoreTopK {
			s += 1 - float64(r-1)/scoreTopK
		}
	}
	return s / float64(len(rank))
}

func mean(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

// Score is calculate_pmi_score_per_condition and calculate_reward_score
// over a song's codes. scale is the PMI's sigmoid scale (ScoreScale).
func (p *Planner) Score(codes []int, c Conditions, scale float64) (Score, error) {
	sc := Score{Conditions: map[string]float64{}, PMI: map[string]float64{}}
	if len(codes) == 0 {
		return sc, fmt.Errorf("lm: no audio codes to score")
	}
	cond, uncond, err := p.ScorePrompts(codes)
	if err != nil {
		return sc, err
	}
	sc.Stats.Prompt = len(cond) + len(uncond)
	if sc.Stats.Prefill, err = p.prefill([][]int32{cond, uncond}); err != nil {
		return sc, err
	}
	t0 := time.Now()
	keys, texts, pmi := ScoreTargets(c)
	for i, k := range keys {
		target, err := p.Tk.Encode(texts[i])
		if err != nil {
			return sc, err
		}
		lp, rank, _, err := p.Teach(0, cond, target)
		if err != nil {
			return sc, err
		}
		sc.Stats.Tokens += len(target)
		if !pmi[i] {
			sc.Conditions[k] = recall(rank)
			continue
		}
		ulp, _, _, err := p.Teach(1, uncond, target)
		if err != nil {
			return sc, err
		}
		sc.Stats.Tokens += len(target)
		v := mean(lp) - mean(ulp)
		sc.PMI[k] = v
		sc.Conditions[k] = 1 / (1 + math.Exp(-v/scale))
	}
	sc.Stats.Steps = time.Since(t0)
	sc.Reward = Reward(sc.Conditions)
	return sc, nil
}

// Reward is calculate_reward_score: caption 0.5, lyrics 0.3, and the mean
// of the rest 0.2, renormalised over the components present.
func Reward(scores map[string]float64) float64 {
	type part struct{ v, w float64 }
	var parts []part
	if v, ok := scores["caption"]; ok {
		parts = append(parts, part{v, 0.5})
	}
	if v, ok := scores["lyrics"]; ok {
		parts = append(parts, part{v, 0.3})
	}
	var meta []float64
	for _, k := range recallKeys { // the dict's order, so the sum's
		if v, ok := scores[k]; ok {
			meta = append(meta, v)
		}
	}
	if len(meta) > 0 {
		parts = append(parts, part{mean(meta), 0.2})
	}
	tw, s := 0.0, 0.0
	for _, pt := range parts {
		tw += pt.w
	}
	if tw == 0 {
		return 0
	}
	for _, pt := range parts {
		s += pt.v * pt.w / tw
	}
	return s
}
