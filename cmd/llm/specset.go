package main

// `-spec -spec-set prompts.jsonl` (P20e, research/p20-llamacpp-mtp.md §10):
// the speculative loop over a set of prompts, the way llama.cpp PR 29761
// measured it on SPEED-Bench — each prompt through the chat template at
// temperature zero, the plain and speculative arms interleaved per prompt, and
// the multiplier reported per category and over the set. One staging for the
// whole set, sized for its longest prompt.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/zimage/tokenizer"
)

// setPrompt is one line of the set: `{"id", "category", "text"}`.
type setPrompt struct {
	ID       string `json:"id"`
	Category string `json:"category"`
	Text     string `json:"text"`
	ids      []int32
}

// loadSet reads and tokenizes the set through the chat template.
func loadSet(path string, tok interface {
	Encode(string) ([]int32, error)
}) ([]setPrompt, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []setPrompt
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		if strings.TrimSpace(sc.Text()) == "" {
			continue
		}
		var p setPrompt
		if err := json.Unmarshal(sc.Bytes(), &p); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		if p.ids, err = tok.Encode(tokenizer.ChatPrompt(p.Text)); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s holds no prompts", path)
	}
	return out, sc.Err()
}

// setTally is what one prompt, or a sum of them, cost in each arm.
type setTally struct {
	plainTok, specTok   int
	plainWall, specWall time.Duration
	rounds, drafted     int
	accepted            int
	prompts             int
}

func (t *setTally) add(u setTally) {
	t.plainTok += u.plainTok
	t.specTok += u.specTok
	t.plainWall += u.plainWall
	t.specWall += u.specWall
	t.rounds += u.rounds
	t.drafted += u.drafted
	t.accepted += u.accepted
	t.prompts += u.prompts
}

func (t setTally) line(name string) string {
	pr := float64(t.plainTok) / t.plainWall.Seconds()
	sr := float64(t.specTok) / t.specWall.Seconds()
	return fmt.Sprintf("%-14s %3d %9.2f %9.2f %7.2fx %7.1f%% %7.3f",
		name, t.prompts, pr, sr, sr/pr, 100*float64(t.accepted)/float64(max(t.drafted, 1)),
		float64(t.specTok)/float64(max(t.rounds, 1)))
}

// runSet runs every prompt of the set, both arms, o.passes times each.
func runSet(o specOpts, set []setPrompt, g *llm.Graph, sp *llm.Speculator,
	tok interface{ Decode([]int32) (string, error) }, eog map[int32]bool) error {

	// One untimed round of each first, as the single-prompt form does.
	if _, err := plainLoop(g, set[0].ids, 4, eog); err != nil {
		return err
	}
	if _, err := specLoop(sp, set[0].ids, 4, eog); err != nil {
		return err
	}
	var texts *os.File
	if path := os.Getenv("SPEC_SET_TEXT"); path != "" {
		var err error
		if texts, err = os.Create(path); err != nil {
			return err
		}
		defer texts.Close()
	}

	rows := [][]string{{"id", "category", "prompt_tokens", "arm", "pass", "tokens", "ms", "tok_s",
		"rounds", "drafted", "accepted"}}
	cats := map[string]*setTally{}
	var all setTally
	plainSelf, specSelf, lossless := 0, 0, 0
	for i, p := range set {
		var u setTally
		var plain, spec []specRun
		for pass := 0; pass < o.passes; pass++ {
			r, err := plainLoop(g, p.ids, o.n, eog)
			if err != nil {
				return fmt.Errorf("%s: %w", p.ID, err)
			}
			s, err := specLoop(sp, p.ids, o.n, eog)
			if err != nil {
				return fmt.Errorf("%s: %w", p.ID, err)
			}
			plain, spec = append(plain, r), append(spec, s)
			u.plainTok += len(r.ids)
			u.plainWall += r.wall
			u.specTok += len(s.ids)
			u.specWall += s.wall
			u.rounds += s.stats.Rounds
			u.drafted += s.stats.Drafted
			u.accepted += s.stats.Accepted
			for _, x := range []struct {
				arm string
				r   specRun
			}{{"plain", r}, {"spec", s}} {
				rows = append(rows, []string{p.ID, p.Category, strconv.Itoa(len(p.ids)), x.arm, strconv.Itoa(pass),
					strconv.Itoa(len(x.r.ids)), fmt.Sprintf("%.1f", float64(x.r.wall.Microseconds())/1000),
					fmt.Sprintf("%.3f", x.r.rate()), strconv.Itoa(x.r.stats.Rounds),
					strconv.Itoa(x.r.stats.Drafted), strconv.Itoa(x.r.stats.Accepted)})
			}
		}
		u.prompts = 1
		// The controls, per prompt: does each arm reproduce itself, and
		// where does speculation first leave the plain text.
		ps := firstDiff(plain[0].ids, plain[len(plain)-1].ids) < 0
		ss := firstDiff(spec[0].ids, spec[len(spec)-1].ids) < 0
		at := firstDiff(plain[0].ids, spec[0].ids)
		if ps {
			plainSelf++
		}
		if ss {
			specSelf++
		}
		if at < 0 {
			lossless++
		}
		leave := "same text"
		if at >= 0 {
			leave = fmt.Sprintf("leaves plain at %d", at)
		}
		fmt.Printf("%2d %-12s %5d prompt tok %4d out  plain %6.2f  spec %6.2f  %.2fx  a1 %3d/%3d  plain self %v, spec self %v, %s\n",
			i+1, p.Category, len(p.ids), len(plain[0].ids),
			float64(u.plainTok)/u.plainWall.Seconds(), float64(u.specTok)/u.specWall.Seconds(),
			(float64(u.specTok)/u.specWall.Seconds())/(float64(u.plainTok)/u.plainWall.Seconds()),
			u.accepted, u.drafted, ps, ss, leave)
		if texts != nil {
			if txt, err := tok.Decode(plain[0].ids); err == nil {
				fmt.Fprintf(texts, "=== %d %s %s (plain, %d tokens)\n%s\n\n", i+1, p.Category, p.ID, len(plain[0].ids), txt)
			}
		}
		if cats[p.Category] == nil {
			cats[p.Category] = &setTally{}
		}
		cats[p.Category].add(u)
		all.add(u)
	}

	fmt.Printf("\n%-14s %3s %9s %9s %8s %8s %7s\n", "category", "n", "plain", "spec", "x", "a1", "tok/rnd")
	names := make([]string, 0, len(cats))
	for c := range cats {
		names = append(names, c)
	}
	sort.Strings(names)
	for _, c := range names {
		fmt.Println(cats[c].line(c))
	}
	fmt.Println(all.line("overall"))
	fmt.Printf("\nplain reproduces itself on %d of %d prompts, spec on %d; spec is the plain text on %d\n",
		plainSelf, len(set), specSelf, lossless)
	if o.csv != "" {
		if err := writeCSV(o.csv, rows); err != nil {
			return err
		}
		fmt.Printf("wrote %s\n", o.csv)
	}
	return nil
}
