package main

// P5c — the speculative loop, measured against a same-hour plain one
// (LLM.md P5, research/p5-mtp-rollback.md).
//
//	go run ./cmd/llm -spec -n 128 -ctx 512                    # the multiplier
//	go run ./cmd/llm -spec -n 128 -spec-prefix 1024 -ctx 2048  # a long context
//	go run ./cmd/llm -spec -n 64 -layers 4                     # the loop, in 9 GB
//	SPEC_PLAIN_ONLY=1 go run ./cmd/llm -spec -n 128            # the control
//
// **2048 cells is the ceiling and it is the draft head's**, not the trunk's:
// `blk.48` states a compress ratio of 0, so `NewMTPHead` refuses any context
// where the QSA selection is not provably the identity (llm/mtp.go).
//
// **Two things are being measured and one of them is not a rate.** The gate
// P5 set is that speculation is *lossless* — the text is the text the plain
// loop produces, token for token, at temperature zero — and that is checked by
// running both arms over the same prompt and comparing the ids. The other is
// the multiplier, and it is quoted against a plain loop measured **in the same
// process minutes apart**, because P1c's environment finding says a decode
// rate from a different day is not a control.
//
// The arms are interleaved (`-spec-passes`, default 2 of each) for the same
// reason every whole-model number in this vertical is: the machine's state
// drifts, and two arms taken back to back bound the drift where two runs an
// hour apart do not.
//
// **The prompt matters more here than anywhere else in the vertical.** The
// multiplier is a function of the acceptance rate and the acceptance rate is a
// function of how predictable the text is — P5a found that a greedy trunk left
// to free-run degenerates into repeating a paragraph within a few dozen tokens
// and reads as 92% acceptance, which is a measurement of the sampler. So this
// prints the text, and the acceptance it reports is the loop's own traffic
// rather than a teacher-forced corpus's.

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/zimage/tokenizer"
)

type specOpts struct {
	model, draft string
	prompt, file string
	prefix       int
	n, ctx       int
	layers       int
	passes       int
	wire         string
	chat         bool
	csv          string
}

// specRun is one arm's result.
type specRun struct {
	ids   []int32
	wall  time.Duration
	stats llm.SpecStats
	graph llm.GraphStats
}

func (r specRun) rate() float64 { return float64(len(r.ids)) / r.wall.Seconds() }

func speculate(o specOpts) error {
	m, err := llm.Open(o.model)
	if err != nil {
		return err
	}
	defer m.Close()
	tok, err := llm.LoadTokenizer(m.Set)
	if err != nil {
		return err
	}
	var ids []int32
	if o.prefix > 0 {
		// The long-context arm: a real corpus rather than a repeated
		// paragraph, because the acceptance rate is a property of the text
		// and a prompt that repeats itself is a prompt the draft head finds
		// easy (P5a). `-spec-prefix N` takes the first N tokens of -ppl-file.
		buf, err := os.ReadFile(o.file)
		if err != nil {
			return err
		}
		all, err := tok.Encode(string(buf))
		if err != nil {
			return err
		}
		if len(all) < o.prefix {
			return fmt.Errorf("%s tokenizes to %d tokens, -spec-prefix asks for %d", o.file, len(all), o.prefix)
		}
		ids = all[:o.prefix]
	} else {
		text := o.prompt
		if o.chat {
			text = tokenizer.ChatPrompt(text)
		}
		var err error
		if ids, err = tok.Encode(text); err != nil {
			return err
		}
	}
	if len(ids) == 0 {
		return fmt.Errorf("the prompt tokenizes to nothing")
	}
	eog := llm.EndOfGeneration(m.Set, tok)

	wire, err := parseArms(o.wire)
	if err != nil {
		return err
	}
	if len(wire) != 1 || wire[0].zero {
		return fmt.Errorf("-spec runs one wiring and it cannot be the negative control; got %q", o.wire)
	}
	arm := wire[0]

	draft, err := llm.Open(o.draft)
	if err != nil {
		return fmt.Errorf("draft head: %w", err)
	}
	defer draft.Close()

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()

	ctx := o.ctx
	if want := len(ids) + o.n + 4; ctx < want {
		ctx = want
	}
	fmt.Printf("%s\n  + %s\n", o.model, o.draft)
	fmt.Printf("%d layers, draft blk.%d, %d tokens of prompt, %d to generate, %d cells of context, wiring %s\n",
		m.Config.NLayer, llm.MTPLayer(m.Config), len(ids), o.n, ctx, arm.name)

	// The arenas hold the prompt, because the prefill is one batch — and the
	// head's hold two rows, because a verification pass wants both.
	start := time.Now()
	g, err := llm.NewGraph(dev, m, llm.GraphOpts{
		MaxTokens: max(len(ids), 2), NKV: ctx, Layers: o.layers, HeadRows: 2,
		Speculative: true,
	})
	if err != nil {
		return err
	}
	defer g.Destroy()
	reportStaging(g, time.Since(start))

	start = time.Now()
	d, err := llm.StageDraft(dev, draft, g)
	if err != nil {
		return err
	}
	defer d.Destroy()
	fmt.Printf("draft staged in %v: %.2f GB of weights, %.2f GB of arenas (the lm head is the trunk's)\n\n",
		time.Since(start).Round(time.Millisecond),
		float64(d.WeightBytes())/1e9, float64(d.ActivationBytes())/1e9)

	sp, err := llm.NewSpeculator(g, d, m, arm.wiring)
	if err != nil {
		return err
	}

	// The two arms, interleaved. One untimed round of each first: the first
	// pass of anything on this machine is not like the others (the banks are
	// cold and the Go heap has not grown), and the pre-recorded decode step
	// is captured on the first one-token Extend.
	if _, err := plainLoop(g, ids, 4, eog); err != nil {
		return err
	}
	if _, err := specLoop(sp, ids, 4, eog); err != nil {
		return err
	}

	var plain, spec []specRun
	for p := 0; p < o.passes; p++ {
		r, err := plainLoop(g, ids, o.n, eog)
		if err != nil {
			return err
		}
		fmt.Printf("plain %d/%d: %d tokens in %s — %.2f tok/s\n",
			p+1, o.passes, len(r.ids), r.wall.Round(time.Millisecond), r.rate())
		plain = append(plain, r)

		// **The control on the control.** `SPEC_PLAIN_ONLY=1` drops the
		// speculative arm and runs the plain loop against itself, which is
		// the only way to find out whether "the same 128 ids" is a claim
		// about speculation or about the machine. It is not idle: it is what
		// found that a fresh sequence here is not independent of the one
		// before it (research/p5c-speculative-loop.md §5).
		if os.Getenv("SPEC_PLAIN_ONLY") == "1" {
			continue
		}
		s, err := specLoop(sp, ids, o.n, eog)
		if err != nil {
			return err
		}
		fmt.Printf("spec  %d/%d: %d tokens in %s — %.2f tok/s, %d rounds, %d drafted, %d accepted (%.1f%%)\n",
			p+1, o.passes, len(s.ids), s.wall.Round(time.Millisecond), s.rate(),
			s.stats.Rounds, s.stats.Drafted, s.stats.Accepted, 100*s.stats.Acceptance())
		spec = append(spec, s)
	}

	// 1. Lossless, or not — **pairwise, and the plain arms against each other
	//    first**. "The speculative arm reproduces the plain one" is only a
	//    claim about speculation if the plain loop reproduces itself, and a
	//    greedy loop 128 tokens deep does not have to: two logits a
	//    ten-thousandth apart decide a token and the rest of the paragraph
	//    follows the one that won. So the control is printed as a row of the
	//    same table rather than assumed.
	fmt.Println()
	all := append(append([]specRun{}, plain...), spec...)
	names := make([]string, 0, len(all))
	for i := range plain {
		names = append(names, fmt.Sprintf("plain%d", i))
	}
	for i := range spec {
		names = append(names, fmt.Sprintf("spec%d", i))
	}
	for i := 0; i < len(all); i++ {
		for j := i + 1; j < len(all); j++ {
			at := firstDiff(all[i].ids, all[j].ids)
			if at < 0 {
				fmt.Printf("%-7s = %-7s  the same %d ids, token for token\n", names[i], names[j], len(all[i].ids))
				continue
			}
			fmt.Printf("%-7s ≠ %-7s  first at token %d: %d against %d\n",
				names[i], names[j], at, idAt(all[i].ids, at), idAt(all[j].ids, at))
		}
	}
	want := plain[len(plain)-1].ids
	if txt, err := tok.Decode(want); err == nil {
		fmt.Printf("\n--- %d generated tokens (%s) ---\n%s\n---\n", len(want), names[len(plain)-1], txt)
	}

	if len(spec) == 0 {
		return nil
	}
	// 2. The multiplier.
	pr, prSpread := meanRate(plain)
	sr, srSpread := meanRate(spec)
	st := totalStats(spec)
	fmt.Printf("\nplain %.2f tok/s (spread %.2f over %d passes), speculative %.2f (spread %.2f) — **%.2fx**\n",
		pr, prSpread, len(plain), sr, srSpread, sr/pr)
	fmt.Printf("  a1 %.1f%% over %d drafted rounds, %.3f tokens a round, %.2f rounds a token\n",
		100*st.Acceptance(), st.Drafted, st.TokensPerRound(), float64(st.Rounds)/float64(st.Tokens))
	fmt.Printf("  a two-row pass routes to %.1f distinct experts of 512 in a layer, against 10 at one row "+
		"— the design pass read 17 off a two-token prefill\n", st.ExpertsPerPass())
	step := 1000 / pr // ms
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / float64(max(st.Rounds, 1)) }
	fmt.Printf("  a round is %.2f ms: draft %.2f (%.3f of a step), verify %.2f (%.3f), decide %.2f (%.3f); "+
		"a plain step is %.2f ms\n",
		ms(st.Wall), ms(st.Draft), ms(st.Draft)/step, ms(st.Verify), ms(st.Verify)/step,
		ms(st.Decide), ms(st.Decide)/step, step)
	if st.Drafted > 0 {
		// The two-state chain, from the measured acceptance and the measured
		// costs — so that the number above can be checked against the model
		// P5c's write-up argues from rather than only reported.
		a := st.Acceptance()
		pass, dr := ms(st.Verify)/step, ms(st.Draft)/step
		round := pass + dr
		tokens := 1 + a
		fmt.Printf("  the chain: speculating %.3f steps for %.3f tokens, recovering %.3f for 1 — "+
			"%.3f steps a token, %.2fx\n",
			round, tokens, pass,
			(round+(1-a)*pass)/2, 2/(round+(1-a)*pass))
	}

	// 3. Where the difference went, per block, per token of output. The two
	//    arms run the same weights over a different number of rows, so a
	//    ratio that is not the one P5b projected has to be attributable to a
	//    block rather than to "speculation".
	fmt.Printf("\n%-12s %10s %10s %8s   per output token\n", "block", "plain ms", "spec ms", "spec/plain")
	pg, sg := sumGraph(plain), sumGraph(spec)
	pn, sn := countIds(plain), countIds(spec)
	for _, r := range []struct {
		name string
		p, s time.Duration
	}{
		{"hyper-conn", pg.HC, sg.HC}, {"ple n-gram", pg.PLE, sg.PLE},
		{"deltanet", pg.DeltaNet, sg.DeltaNet}, {"attention", pg.Attn, sg.Attn},
		{"moe", pg.MoE, sg.MoE}, {"lm head", pg.Head, sg.Head}, {"move", pg.Move, sg.Move},
		{"= on the GPU", pg.GPU, sg.GPU},
		{"gather", pg.Gather, sg.Gather}, {"glue", pg.Glue, sg.Glue}, {"submit", pg.Submit, sg.Submit},
	} {
		pv, sv := perToken(r.p, pn), perToken(r.s, sn)
		if pv == 0 && sv == 0 {
			continue
		}
		fmt.Printf("%-12s %10.2f %10.2f %8s\n", r.name, pv, sv, ratio(pv, sv))
	}
	fmt.Printf("%-12s %10d %10d\n", "dispatches", pg.Dispatches/max(pn, 1), sg.Dispatches/max(sn, 1))

	// And per dispatch label, for the block the ratio lands on. A two-row
	// pass reads the same dense bytes as a one-row one and a larger set of
	// experts, so anything here that is not the routed pair growing with the
	// expert set is shape rather than bytes.
	kinds := map[string]bool{}
	for k := range pg.Kinds {
		kinds[k] = true
	}
	for k := range sg.Kinds {
		kinds[k] = true
	}
	type row struct {
		k    string
		p, s float64
	}
	var rs []row
	for k := range kinds {
		rs = append(rs, row{k, perToken(pg.Kinds[k].GPU, pn), perToken(sg.Kinds[k].GPU, sn)})
	}
	sort.Slice(rs, func(i, j int) bool { return rs[i].s > rs[j].s })
	fmt.Printf("\n%-18s %10s %10s %8s %8s %8s   per output token\n",
		"dispatch", "plain ms", "spec ms", "spec/pl", "plain n", "spec n")
	for i, r := range rs {
		if i >= 14 {
			break
		}
		fmt.Printf("%-18s %10.2f %10.2f %8s %8d %8d\n", r.k, r.p, r.s, ratio(r.p, r.s),
			pg.Kinds[r.k].Count/max(pn, 1), sg.Kinds[r.k].Count/max(sn, 1))
	}

	if o.csv == "" {
		return nil
	}
	rows := [][]string{{"arm", "pass", "tokens", "ms", "tok_s", "rounds", "drafted", "accepted"}}
	for i, r := range plain {
		rows = append(rows, specRow("plain", i, r))
	}
	for i, r := range spec {
		rows = append(rows, specRow("spec", i, r))
	}
	if err := writeCSV(o.csv, rows); err != nil {
		return err
	}
	fmt.Printf("\nwrote %s\n", o.csv)
	return nil
}

func specRow(arm string, i int, r specRun) []string {
	return []string{arm, strconv.Itoa(i), strconv.Itoa(len(r.ids)),
		fmt.Sprintf("%.1f", float64(r.wall.Microseconds())/1000),
		fmt.Sprintf("%.3f", r.rate()),
		strconv.Itoa(r.stats.Rounds), strconv.Itoa(r.stats.Drafted), strconv.Itoa(r.stats.Accepted)}
}

// plainLoop is the ordinary decode path: the prompt in one batch and then one
// token at a time, greedy. It is the control the multiplier is quoted against
// and the text the speculative arm has to reproduce.
//
// The prefill is **outside** the clock in both arms. Speculation does not
// touch it and including it would dilute the ratio by however long the prompt
// is.
func plainLoop(g *llm.Graph, ids []int32, n int, eog map[int32]bool) (specRun, error) {
	if err := g.Speculate(false); err != nil {
		return specRun{}, err
	}
	logits, _, err := g.Forward(ids)
	if err != nil {
		return specRun{}, err
	}
	// The prefill is outside both the clock and the attribution: it is the
	// same work in both arms and speculation does not touch it.
	g.ResetStats()
	out := make([]int32, 0, n)
	t0 := time.Now()
	for len(out) < n {
		id := llm.Argmax(logits)
		out = append(out, id)
		if eog[id] || len(out) == n {
			break
		}
		if logits, _, err = g.Extend([]int32{id}); err != nil {
			return specRun{}, err
		}
	}
	return specRun{ids: out, wall: time.Since(t0), graph: g.Stats}, nil
}

// specLoop is the same generation through the speculative round.
//
// It stops on the **first** end-of-generation id in a round's output rather
// than after it, so that the two arms emit the same list: a round that
// accepted its draft can produce a token past the end and the plain loop
// never sees one.
func specLoop(sp *llm.Speculator, ids []int32, n int, eog map[int32]bool) (specRun, error) {
	first, err := sp.Start(ids)
	if err != nil {
		return specRun{}, err
	}
	defer sp.Stop()
	sp.ResetGraphStats()
	out := make([]int32, 0, n)
	out = append(out, first)
	t0 := time.Now()
	var buf []int32
	for len(out) < n && !eog[out[len(out)-1]] {
		buf, err = sp.Next(buf)
		if err != nil {
			return specRun{}, err
		}
		for _, id := range buf {
			out = append(out, id)
			if len(out) == n || eog[id] {
				break
			}
		}
	}
	wall := time.Since(t0)
	return specRun{ids: out, wall: wall, stats: sp.Stats, graph: sp.GraphStats()}, nil
}

func firstDiff(a, b []int32) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}

func idAt(a []int32, i int) int32 {
	if i < len(a) {
		return a[i]
	}
	return -1
}

func sumGraph(rs []specRun) llm.GraphStats {
	var t llm.GraphStats
	for _, r := range rs {
		t.HC += r.graph.HC
		t.PLE += r.graph.PLE
		t.DeltaNet += r.graph.DeltaNet
		t.Attn += r.graph.Attn
		t.MoE += r.graph.MoE
		t.Head += r.graph.Head
		t.Move += r.graph.Move
		t.GPU += r.graph.GPU
		t.Gather += r.graph.Gather
		t.Glue += r.graph.Glue
		t.Submit += r.graph.Submit
		t.Dispatches += r.graph.Dispatches
		if t.Kinds == nil {
			t.Kinds = map[string]llm.DispatchStat{}
		}
		for k, v := range r.graph.Kinds {
			s := t.Kinds[k]
			s.Count, s.GPU = s.Count+v.Count, s.GPU+v.GPU
			t.Kinds[k] = s
		}
	}
	return t
}

func countIds(rs []specRun) int {
	n := 0
	for _, r := range rs {
		n += len(r.ids)
	}
	return n
}

func perToken(d time.Duration, n int) float64 {
	if n == 0 {
		return 0
	}
	return float64(d.Microseconds()) / 1000 / float64(n)
}

func ratio(a, b float64) string {
	if a == 0 {
		return "-"
	}
	return fmt.Sprintf("%.2fx", b/a)
}

func meanRate(rs []specRun) (mean, spread float64) {
	if len(rs) == 0 {
		return 0, 0
	}
	lo, hi := rs[0].rate(), rs[0].rate()
	for _, r := range rs {
		mean += r.rate()
		lo, hi = min(lo, r.rate()), max(hi, r.rate())
	}
	return mean / float64(len(rs)), hi - lo
}

func totalStats(rs []specRun) llm.SpecStats {
	var t llm.SpecStats
	for _, r := range rs {
		t.Rounds += r.stats.Rounds
		t.Drafted += r.stats.Drafted
		t.Accepted += r.stats.Accepted
		t.Tokens += r.stats.Tokens
		t.Wall += r.stats.Wall
		t.Draft += r.stats.Draft
		t.Verify += r.stats.Verify
		t.Decide += r.stats.Decide
		t.Experts += r.stats.Experts
		t.ExpertRounds += r.stats.ExpertRounds
	}
	return t
}

// specPromptDefault is a paragraph of ordinary prose rather than the
// three-word prompt `-gen` defaults to, because a multiplier measured on text
// a greedy trunk loops in is measuring the sampler (P5a).
const specPromptDefault = "The history of computing is usually told as a story about machines, " +
	"but it is at least as much a story about the people who decided what the machines were for. "

func specWireList(s string) string {
	if strings.TrimSpace(s) == "" {
		return "res"
	}
	return s
}
