package main

// Generation (LLM.md L7c): the prompt, then one token at a time, in Go on
// Vulkan.
//
//	go run ./cmd/llm -gen -n 64                      # greedy, the default prompt
//	go run ./cmd/llm -gen -prompt 'The capital of France is' -n 32
//	go run ./cmd/llm -gen -n 128 -temp 0.7 -top-p 0.8 -seed 1
//	go run ./cmd/llm -gen -n 16 -layers 4            # the loop, in 7 GB
//
// The acceptance criterion is llama.cpp's own text, so the default is greedy:
//
//	$L/build/bin/llama-completion -m $M -n 64 --temp 0 -p '…'
//
// Two rates are reported and they are different questions. **Prefill** is the
// prompt in one batch, which L6c left at 941.3 tok/s against llama.cpp's
// 391.42. **Decode** is a batch of one, 48 layers deep, and it is a different
// machine limit: every dense weight and ten experts of 512 are read for one
// token, 6.334 GB of them, so 242 GB/s is 38.2 tok/s and llama.cpp reaches
// 25.15 (LLM.md's budget). Prefill is an arithmetic problem and decode is a
// bandwidth one; nothing about being 2.40x at the first says anything about
// the second.

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/zimage/tokenizer"
)

// llamaDecode is llama.cpp's own tg128 on this checkpoint and this machine
// (L1's baseline), and llamaCeiling the rate the bytes allow at 242 GB/s.
const (
	llamaDecode  = 25.15
	llamaCeiling = 38.2
)

// genOpts is what the -gen flags come to.
type genOpts struct {
	model, prompt string
	n, ctx        int
	layers        int
	temp, topP    float64
	topK          int
	seed          int64
	chat          bool
	top           int
	csv           string
	attrib        bool
}

// hostPhases is where a decode loop's wall clock goes **outside** the graph
// (P1). The graph accounts for itself — GPU, gather, glue, hand-over — and
// the four below are what the loop does between two of its calls, which no
// figure in this vertical has ever separated from "hand-over".
type hostPhases struct {
	sample time.Duration // argmax or the temperature ladder over 151k logits
	detok  time.Duration // the id back to its piece
	emit   time.Duration // the write and the flush to stdout
	extend time.Duration // the whole of the next forward pass
}

// generate prefills a prompt and then decodes n tokens.
func generate(o genOpts) error {
	m, err := llm.Open(o.model)
	if err != nil {
		return err
	}
	defer m.Close()
	tok, err := llm.LoadTokenizer(m.Set)
	if err != nil {
		return err
	}
	text := o.prompt
	if o.chat {
		text = tokenizer.ChatPrompt(text)
	}
	ids, err := tok.Encode(text)
	if err != nil {
		return err
	}
	if len(ids) == 0 {
		return fmt.Errorf("the prompt tokenizes to nothing")
	}
	eog := llm.EndOfGeneration(m.Set, tok)

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()

	// The arenas hold the prompt; the cache holds the prompt and everything
	// generated after it. They are separate numbers because a decode step is
	// a batch of one however long the context is.
	ctx := o.ctx
	if want := len(ids) + o.n; ctx < want {
		ctx = want
	}
	fmt.Printf("%s\n%d layers, %d tokens of prompt, %d to generate, %d cells of context\n",
		o.model, m.Config.NLayer, len(ids), o.n, ctx)

	start := time.Now()
	g, err := llm.NewGraph(dev, m, llm.GraphOpts{MaxTokens: len(ids), NKV: ctx, Layers: o.layers})
	if err != nil {
		return err
	}
	defer g.Destroy()
	reportStaging(g, time.Since(start))

	s := llm.NewSampler(float32(o.temp), o.topK, float32(o.topP), o.seed)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	fmt.Fprint(out, text)

	// The prompt, in one batch.
	g.ResetStats()
	t0 := time.Now()
	logits, _, err := g.Forward(ids)
	if err != nil {
		return err
	}
	prefill := time.Since(t0)
	prefillStats := g.Stats

	// Then one token at a time, each its own forward pass over a cache the
	// last one extended.
	g.ResetStats()
	t0 = time.Now()
	var got []int32
	var hp hostPhases
	for i := 0; i < o.n; i++ {
		t := time.Now()
		id := s.Sample(logits)
		t = since(&hp.sample, t)
		if o.top > 0 {
			reportTop(out, tok, logits, o.top, id)
			t = time.Now()
		}
		got = append(got, id)
		if eog[id] {
			break
		}
		piece, err := tok.Decode([]int32{id})
		if err != nil {
			return err
		}
		t = since(&hp.detok, t)
		fmt.Fprint(out, piece)
		out.Flush()
		t = since(&hp.emit, t)
		if i == o.n-1 {
			break
		}
		if logits, _, err = g.Extend([]int32{id}); err != nil {
			return err
		}
		since(&hp.extend, t)
	}
	decode := time.Since(t0)
	fmt.Fprintln(out)
	out.Flush()

	fmt.Printf("\nprefill %d tokens in %s — %.1f tok/s, %.2fx llama.cpp's %.2f\n",
		len(ids), prefill.Round(time.Millisecond),
		float64(len(ids))/prefill.Seconds(), float64(len(ids))/prefill.Seconds()/llamaPrefill, llamaPrefill)
	reportPhases(prefillStats, prefill)

	rate := float64(len(got)) / decode.Seconds()
	fmt.Printf("\ndecode %d tokens in %s — %.2f tok/s, %.2fx llama.cpp's %.2f, %.0f%% of the %.1f the bytes allow\n",
		len(got), decode.Round(time.Millisecond), rate, rate/llamaDecode, llamaDecode,
		100*rate/llamaCeiling, llamaCeiling)
	if p := llm.MoEBankPlanFromEnv(); !p.Off() {
		fmt.Printf("moe bank: %s\n", p)
	}
	if p := llm.DenseBankPlan(); !p.Off() {
		// `llamaCeiling` is the **checkpoint's** width — 6.334 GB a token at
		// 242 GB/s — and a run on L8c-4's bank does not read that many bytes,
		// so the percentage above is against the wrong denominator and says
		// so rather than being quietly re-derived. `cmd/gguf -width` is the
		// arithmetic; this line is the warning.
		fmt.Printf("    (the %.1f is the checkpoint's own width; this run stages %s narrower — see cmd/gguf -width)\n",
			llamaCeiling, p)
	}
	reportPhases(g.Stats, decode)
	if o.attrib {
		reportAttribution(g.Stats, hp, decode, len(got))
		if o.csv != "" {
			// With -attrib the CSV **is** the attribution table: one row a
			// phase and one a dispatch label, which is the artifact P1 is
			// for. The one-row summary below is what -gen -csv writes
			// without it, and the two do not fit in one file.
			return writeAttribCSV(o.csv, g.Stats, hp, decode, len(got))
		}
	}
	if o.csv == "" {
		return nil
	}
	st := g.Stats
	ms := func(d time.Duration) string {
		return fmt.Sprintf("%.1f", float64(d.Microseconds())/1000/float64(max(len(got), 1)))
	}
	return writeCSV(o.csv, [][]string{
		{"prompt_tokens", "gen_tokens", "ctx", "prefill_tok_s", "decode_tok_s", "against_llama",
			"ms_per_token", "hc_ms", "ple_ms", "dn_ms", "attn_ms", "moe_ms", "head_ms", "move_ms", "gather_ms", "glue_ms"},
		{
			fmt.Sprint(len(ids)), fmt.Sprint(len(got)), fmt.Sprint(ctx),
			fmt.Sprintf("%.1f", float64(len(ids))/prefill.Seconds()),
			fmt.Sprintf("%.2f", rate), fmt.Sprintf("%.3f", rate/llamaDecode),
			fmt.Sprintf("%.1f", decode.Seconds()*1000/float64(max(len(got), 1))),
			ms(st.HC), ms(st.PLE), ms(st.DeltaNet), ms(st.Attn), ms(st.MoE),
			ms(st.Head), ms(st.Move), ms(st.Gather), ms(st.Glue),
		},
	})
}

// reportPhases prints where a run's wall clock went, as a share of the wall
// clock the caller measured rather than of the graph's own total — so that
// whatever the graph does not account for shows up as the gap.
//
// **The first rows are GPU time and the last three are host time** (L7d). A
// pass is one command buffer, so a block no longer has a wall clock of its
// own; what it has is the sum of its dispatches' timestamps inside that
// command buffer, which is the same figure llama.cpp's GGML_VK_PERF_LOGGER
// reports on the other side of every comparison here. `on the GPU` is the
// whole command buffer end to end, so the gap between it and the rows above
// is what the barriers between dispatches cost, and `hand-over` is what the
// submit and the fence wait take on top of it.
func reportPhases(st llm.GraphStats, wall time.Duration) {
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
	total := float64(wall.Microseconds()) / 1000
	row := func(name string, d time.Duration) {
		if d <= 0 {
			return
		}
		fmt.Printf("    %-12s %8.1f ms  %5.1f%%\n", name, ms(d), 100*ms(d)/total)
	}
	for _, r := range []struct {
		name string
		d    time.Duration
	}{
		{"hyper-conn", st.HC}, {"ple n-gram", st.PLE}, {"deltanet", st.DeltaNet},
		{"attention", st.Attn}, {"moe", st.MoE}, {"lm head", st.Head},
		{"move", st.Move},
	} {
		row(r.name, r.d)
	}
	row("= on the GPU", st.GPU)
	row("gather", st.Gather)
	row("glue", st.Glue)
	row("hand-over", wall-st.GPU-st.Gather-st.Glue)
	if n := st.Dispatches / max(st.Runs, 1); n > 0 {
		b := llm.BatchFor(1)
		fmt.Printf("    %d dispatches a pass, in %d command buffer(s)\n",
			n, (n+b-1)/b)
	}
}

// reportTop prints the sampled token beside the alternatives it beat, which
// is how a disagreement with another implementation is located: a different
// token out of the same top ten is a tolerance, out of a different ten is a
// bug.
func reportTop(w *bufio.Writer, tok interface{ Decode([]int32) (string, error) }, logits []float32, n int, picked int32) {
	w.Flush()
	var b strings.Builder
	for i, id := range llm.TopN(logits, n) {
		piece, err := tok.Decode([]int32{id})
		if err != nil {
			piece = "?"
		}
		mark := " "
		if id == picked {
			mark = "*"
		}
		if i > 0 {
			b.WriteString("  ")
		}
		fmt.Fprintf(&b, "%s%d %q %.3f", mark, id, piece, logits[id])
	}
	fmt.Fprintf(os.Stderr, "\n[%s]\n", b.String())
}

// since adds the elapsed time to one phase and returns now, so the decode
// loop is a chain of one-line calls rather than a pile of start/stop pairs.
func since(dst *time.Duration, t0 time.Time) time.Time {
	now := time.Now()
	*dst += now.Sub(t0)
	return now
}

// reportAttribution is P1: where one decode token's 31.7 ms goes, at the
// resolution of a dispatch label rather than of a block, with the host side
// beside it and a residual that says how much the two miss the wall clock by.
//
// **The two halves answer different questions.** The GPU table is the sum of
// the timestamps the shim wrote after each dispatch, so it is what the
// kernels cost; the host table is wall clock around the loop's own calls. The
// graph's GPU time is a *part* of `extend`, not beside it, so the host rows
// print `extend` broken into the graph's own four and the rest of the loop
// beside them. Everything sums to the measured wall clock by construction,
// which is what makes the residual meaningful.
func reportAttribution(st llm.GraphStats, hp hostPhases, wall time.Duration, tokens int) {
	n := float64(max(tokens, 1))
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / n }
	step := float64(wall.Microseconds()) / 1000 / n

	fmt.Printf("\nper-token attribution, %d tokens, %.2f ms a step\n", tokens, step)

	// The host side first, because it is the one that adds up to the step.
	fmt.Printf("\n  host, one step\n")
	hrow := func(name string, d time.Duration) {
		fmt.Printf("    %-16s %7.3f ms  %5.1f%%\n", name, ms(d), 100*ms(d)/step)
	}
	hrow("gather (ple)", st.Gather)
	hrow("record", st.Glue)
	hrow("submit+fence", st.Submit)
	hrow("sample", hp.sample)
	hrow("detokenize", hp.detok)
	hrow("emit", hp.emit)
	rest := wall - st.Gather - st.Glue - st.Submit - hp.sample - hp.detok - hp.emit
	hrow("unattributed", rest)

	// Then the submit, split: what the GPU said it was doing, and what it
	// cost to hand the command buffer over.
	fmt.Printf("\n  inside submit+fence\n")
	srow := func(name string, d time.Duration) {
		fmt.Printf("    %-16s %7.3f ms  %5.1f%%\n", name, ms(d), 100*ms(d)/step)
	}
	srow("dispatches", st.GPU)
	srow("hand-over", st.Submit-st.GPU)

	if len(st.Kinds) == 0 {
		return
	}
	// And the dispatches, by label. Sorted by cost, because the question is
	// which kernel owns the step and not which block runs first.
	type row struct {
		name string
		s    llm.DispatchStat
	}
	rows := make([]row, 0, len(st.Kinds))
	var sum time.Duration
	var count int
	for k, v := range st.Kinds {
		rows = append(rows, row{k, v})
		sum += v.GPU
		count += v.Count
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].s.GPU > rows[j].s.GPU })
	passes := float64(max(st.Runs, 1))
	// A mark-to-mark interval is the dispatch *and* the barrier in front of
	// it, so these sum to the whole command buffer by construction — there
	// is no barrier residual to print, and `us each` is a kernel plus its
	// share of the barriers.
	fmt.Printf("\n  dispatches by label — %.0f a pass over %d labels, %.3f ms, %.1f%% of the step\n",
		float64(count)/passes, len(rows), ms(sum), 100*ms(sum)/step)
	fmt.Printf("    %-24s %8s %8s %9s %7s\n", "label", "n/pass", "ms/step", "us each", "share")
	for _, r := range rows {
		each := float64(r.s.GPU.Nanoseconds()) / 1000 / float64(max(r.s.Count, 1))
		fmt.Printf("    %-24s %8.1f %8.3f %9.1f %6.1f%%\n",
			r.name, float64(r.s.Count)/passes, ms(r.s.GPU), each, 100*ms(r.s.GPU)/step)
	}
}

// writeAttribCSV is reportAttribution as a file: one row a phase and one a
// dispatch label, each with its per-token milliseconds and its share of the
// step, so the table can be re-read without re-running the model.
func writeAttribCSV(path string, st llm.GraphStats, hp hostPhases, wall time.Duration, tokens int) error {
	n := float64(max(tokens, 1))
	passes := float64(max(st.Runs, 1))
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / n }
	step := float64(wall.Microseconds()) / 1000 / n
	rows := [][]string{{"section", "label", "n_per_pass", "ms_per_token", "us_each", "share_pct"}}
	add := func(section, label string, count float64, d time.Duration, each float64) {
		e := ""
		if each > 0 {
			e = fmt.Sprintf("%.1f", each)
		}
		c := ""
		if count > 0 {
			c = fmt.Sprintf("%.1f", count)
		}
		rows = append(rows, []string{section, label, c,
			fmt.Sprintf("%.3f", ms(d)), e, fmt.Sprintf("%.1f", 100*ms(d)/step)})
	}
	add("step", "wall", 0, wall, 0)
	for _, r := range []struct {
		name string
		d    time.Duration
	}{
		{"gather", st.Gather}, {"record", st.Glue}, {"submit+fence", st.Submit},
		{"sample", hp.sample}, {"detokenize", hp.detok}, {"emit", hp.emit},
		{"unattributed", wall - st.Gather - st.Glue - st.Submit - hp.sample - hp.detok - hp.emit},
	} {
		add("host", r.name, 0, r.d, 0)
	}
	add("submit", "dispatches", 0, st.GPU, 0)
	add("submit", "hand-over", 0, st.Submit-st.GPU, 0)
	type row struct {
		name string
		s    llm.DispatchStat
	}
	ks := make([]row, 0, len(st.Kinds))
	for k, v := range st.Kinds {
		ks = append(ks, row{k, v})
	}
	sort.Slice(ks, func(i, j int) bool { return ks[i].s.GPU > ks[j].s.GPU })
	for _, r := range ks {
		add("dispatch", r.name, float64(r.s.Count)/passes, r.s.GPU,
			float64(r.s.GPU.Nanoseconds())/1000/float64(max(r.s.Count, 1)))
	}
	return writeCSV(path, rows)
}
