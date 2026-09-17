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
	"strings"
	"time"

	"strix-halo-vulkan/gguf"
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
}

// generate prefills a prompt and then decodes n tokens.
func generate(o genOpts) error {
	m, err := llm.Open(o.model)
	if err != nil {
		return err
	}
	defer m.Close()
	tok, err := LoadTokenizer(m.Set)
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
	eog := endOfGeneration(m.Set, tok)

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
	for i := 0; i < o.n; i++ {
		id := s.Sample(logits)
		if o.top > 0 {
			reportTop(out, tok, logits, o.top, id)
		}
		got = append(got, id)
		if eog[id] {
			break
		}
		piece, err := tok.Decode([]int32{id})
		if err != nil {
			return err
		}
		fmt.Fprint(out, piece)
		out.Flush()
		if i == o.n-1 {
			break
		}
		if logits, _, err = g.Extend([]int32{id}); err != nil {
			return err
		}
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

// endOfGeneration is the set of tokens that stop the loop.
//
// It is **not** `tokenizer.ggml.eos_token_id` alone, and the difference is
// visible on the first prompt anyone tries: this checkpoint's EOS is 248046
// and what the model actually emits to finish a completion is 248044,
// `<|endoftext|>`. llama.cpp marks a control token end-of-generation by
// *name* as well as by the metadata key — `llama_vocab` flags `<|endoftext|>`,
// `<|im_end|>` and `<|eot_id|>` wherever the vocabulary has them — so a loop
// that trusts the key alone runs past the end of the text and then repeats
// one token until -n is exhausted.
func endOfGeneration(set *gguf.Set, tok *tokenizer.Tokenizer) map[int32]bool {
	eog := map[int32]bool{}
	if v, ok := set.Uint("tokenizer.ggml.eos_token_id"); ok {
		eog[int32(v)] = true
	}
	for _, k := range []string{"eot_token_id", "eom_token_id"} {
		if v, ok := set.Uint("tokenizer.ggml." + k); ok {
			eog[int32(v)] = true
		}
	}
	for _, name := range []string{"<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end|>"} {
		if id, ok := tok.ID(name); ok {
			eog[id] = true
		}
	}
	return eog
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
		fmt.Printf("    %d dispatches a pass, in %d command buffer(s)\n",
			n, (n+llm.MaxBatch-1)/llm.MaxBatch)
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
