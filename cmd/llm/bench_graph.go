package main

// The graph, end to end (LLM.md L6b): five blocks in llama.cpp's own order,
// from tokens to logits, timed against the 388.60 tok/s of L1's baseline.
//
//	go run ./cmd/llm -graph -tokens 512            # the number to beat
//	go run ./cmd/llm -graph -tokens 128,512,2048   # the ladder
//	go run ./cmd/llm -graph -tokens 512 -prompt 'The capital of France is'
//	go run ./cmd/llm -graph -tokens 512 -layers 4  # the shape, in 7 GB
//
// It reports two things, and the second is the point. **tok/s** is the gate:
// llama.cpp does 391.42 ± 1.13 at its own best ubatch on this checkpoint and
// this machine (L2a-4). And **where the wall clock went** — because this
// first cut's blocks each own their arenas, so `hc_mixed` is read out of one
// fp32 arena, narrowed to halves on the host and written into another, 96
// times a graph. That is not arithmetic and it is not the reference's; it is
// what the next stage deletes, and it can only be deleted once it has a
// number.

import (
	"fmt"
	"strconv"
	"time"

	"strix-halo-vulkan/llm"
)

// graphBenchIters is how many prefills each length is timed over, after one
// that is thrown away.
//
// The first pass is not like the others: nothing of the 85 GB is in any cache
// and the Go heap has not yet grown to hold a run's host buffers. Two timed
// passes is what the rest of this vertical quotes (LLM.md's reproducibility
// convention) and it is enough here, where the spread between them is under
// a percent.
const graphBenchIters = 2

// graphBench stages the model and prefills a prompt of each length.
func graphBench(model, prompt string, toks []int, nLayers, ctx int, csvPath string) error {
	m, err := llm.Open(model)
	if err != nil {
		return err
	}
	defer m.Close()

	maxTok := 0
	for _, t := range toks {
		maxTok = max(maxTok, t)
	}
	if maxTok <= 0 {
		return fmt.Errorf("no token counts to run")
	}
	ids, err := graphPrompt(m, prompt, maxTok)
	if err != nil {
		return err
	}

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()

	fmt.Printf("%s\n%d layers, n_embd %d, %d experts of %d wide, %d used\n",
		model, m.Config.NLayer, m.Config.NEmbd, m.Config.NExpert, m.Config.FFNExpert, m.Config.NExpertUsed)

	start := time.Now()
	g, err := llm.NewGraph(dev, m, llm.GraphOpts{MaxTokens: maxTok, NKV: ctx, Layers: nLayers})
	if err != nil {
		return err
	}
	defer g.Destroy()
	reportStaging(g, time.Since(start))

	rows := [][]string{{"tokens", "layers", "ms", "tok_s", "against_llama",
		"hc_ms", "ple_ms", "dn_ms", "attn_ms", "moe_ms", "head_ms", "move_ms", "gather_ms", "glue_ms"}}
	for _, n := range toks {
		if n > maxTok {
			continue
		}
		// One untimed pass, then the ones that count.
		if _, _, err := g.Forward(ids[:n]); err != nil {
			return err
		}
		g.ResetStats()
		for i := 0; i < graphBenchIters; i++ {
			if _, _, err := g.Forward(ids[:n]); err != nil {
				return err
			}
		}
		rows = append(rows, reportGraphRun(g, n))
	}
	if csvPath == "" {
		return nil
	}
	return writeCSV(csvPath, rows)
}

// graphPrompt produces at least n tokens to prefill.
//
// A real prompt is what the caller should give: the MoE block's routing is
// nothing like balanced (L5a-4) and the QSA selection only starts excluding
// cells past 2051, so both of them cost what the *text* makes them cost. When
// there is not enough text the prompt is repeated rather than padded with one
// id, because a run of identical tokens routes to one expert and is not a
// measurement of this model.
func graphPrompt(m *llm.Model, prompt string, n int) ([]int32, error) {
	tok, err := LoadTokenizer(m.Set)
	if err != nil {
		return nil, err
	}
	ids, err := tok.Encode(prompt)
	if err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("the prompt tokenizes to nothing")
	}
	for len(ids) < n {
		ids = append(ids, ids[:min(len(ids), n-len(ids))]...)
	}
	return ids[:n], nil
}

// reportStaging prints what went on the device.
func reportStaging(g *llm.Graph, wall time.Duration) {
	fmt.Printf("\n%-12s %7s %8s %12s %11s %11s\n", "block", "layers", "buffers", "weights", "arenas", "staged in")
	var w, a, bufs int
	for _, s := range g.Staged {
		fmt.Printf("%-12s %7d %8d %9.2f GB %8.1f MB %11s\n", s.Name, s.Layers, s.Buffers,
			float64(s.Weights)/1e9, float64(s.Arenas)/1e6, s.Elapsed.Round(time.Millisecond))
		w, a, bufs = w+s.Weights, a+s.Arenas, bufs+s.Buffers
	}
	fmt.Printf("%-12s %7d %8d %9.2f GB %8.1f MB %11s\n\n", "total", g.Layers(), bufs,
		float64(w)/1e9, float64(a)/1e6, wall.Round(time.Millisecond))
}

// llamaPrefill is llama.cpp's own prefill rate on this checkpoint and this
// machine: L2a-4's re-measurement at its best ubatch, 391.42 ± 1.13 tok/s,
// which agrees with L1's separate 388.60 ± 1.89 to 0.7%.
const llamaPrefill = 391.42

// reportGraphRun prints one length's timing and returns it as a CSV row.
func reportGraphRun(g *llm.Graph, n int) []string {
	st := g.Stats
	runs := float64(max(st.Runs, 1))
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / runs }
	total := ms(st.Total)
	rate := float64(n) / (total / 1000)

	fmt.Printf("%d tokens, %d layers: %.1f ms, %.1f tok/s — %.2fx llama.cpp's %.2f\n",
		n, g.Layers(), total, rate, rate/llamaPrefill, llamaPrefill)
	for _, r := range []struct {
		name string
		d    time.Duration
	}{
		{"hyper-conn", st.HC}, {"ple n-gram", st.PLE}, {"deltanet", st.DeltaNet},
		{"attention", st.Attn}, {"moe", st.MoE}, {"lm head", st.Head},
		{"move", st.Move}, {"gather", st.Gather}, {"glue", st.Glue},
	} {
		if r.d == 0 {
			continue
		}
		fmt.Printf("    %-11s %8.1f ms  %5.1f%%\n", r.name, ms(r.d), 100*ms(r.d)/total)
	}
	// The line the next stage is scoped against: what the graph would do if
	// the arena-to-arena moves were not there at all.
	blocks := ms(st.Blocks())
	fmt.Printf("    blocks alone %.1f ms = %.1f tok/s; the moves are %.1f ms (%.1f%%) and the host %.1f ms (%.1f%%)\n",
		blocks, float64(n)/(blocks/1000), ms(st.Move), 100*ms(st.Move)/total,
		ms(st.Gather)+ms(st.Glue), 100*(ms(st.Gather)+ms(st.Glue))/total)

	return []string{
		strconv.Itoa(n), strconv.Itoa(g.Layers()),
		fmt.Sprintf("%.1f", total), fmt.Sprintf("%.1f", rate),
		fmt.Sprintf("%.3f", rate/llamaPrefill),
		fmt.Sprintf("%.1f", ms(st.HC)), fmt.Sprintf("%.1f", ms(st.PLE)),
		fmt.Sprintf("%.1f", ms(st.DeltaNet)), fmt.Sprintf("%.1f", ms(st.Attn)),
		fmt.Sprintf("%.1f", ms(st.MoE)), fmt.Sprintf("%.1f", ms(st.Head)),
		fmt.Sprintf("%.1f", ms(st.Move)),
		fmt.Sprintf("%.1f", ms(st.Gather)), fmt.Sprintf("%.1f", ms(st.Glue)),
	}
}
