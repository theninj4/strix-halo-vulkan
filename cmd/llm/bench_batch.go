package main

// -batch: what a batched decode pass costs (CONCURRENCY.md C5).
//
//	LLM_BANK_CACHE=… go run ./cmd/llm -batch 1,2,3 -batch-depth 7000
//
// It stages the graph with a slot a row, prefills each slot with its own
// `depth` tokens of wikitext (different text, so the rows route to different
// experts the way concurrent conversations do), then times Graph.DecodeRows
// at each row count. One row is the ordinary decode step, recorded replay and
// all, so it is the baseline the others are read against. The GEMV bench
// (-graph -tokens 1,2,3) prices the same row counts as one sequence at a
// shallow depth; this one is the product's shape.

import (
	"fmt"
	"os"
	"time"

	"strix-halo-vulkan/llm"
)

const batchBenchSteps = 16

func batchBench(model, corpus string, rows []int, depth, chunk, ctx, nLayers int, csvPath string) error {
	m, err := llm.Open(model)
	if err != nil {
		return err
	}
	defer m.Close()
	maxR := 0
	for _, r := range rows {
		maxR = max(maxR, r)
	}
	if maxR <= 0 {
		return fmt.Errorf("no row counts to run")
	}
	text, err := os.ReadFile(corpus)
	if err != nil {
		return err
	}
	tok, err := llm.LoadTokenizer(m.Set)
	if err != nil {
		return err
	}
	// ~4.3 characters a token in wikitext; cut generously and trim.
	chars := depth * 6
	if chars*maxR > len(text) {
		return fmt.Errorf("%s holds %d characters, not %d prompts of %d", corpus, len(text), maxR, chars)
	}
	prompts := make([][]int32, maxR)
	for s := range prompts {
		ids, err := tok.Encode(string(text[s*chars : (s+1)*chars]))
		if err != nil {
			return err
		}
		if len(ids) < depth {
			return fmt.Errorf("prompt %d is %d tokens, want %d", s, len(ids), depth)
		}
		prompts[s] = ids[:depth]
	}

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()
	start := time.Now()
	nKV := depth + (len(rows)+1)*(batchBenchSteps+2) + 64
	if ctx > 0 {
		// The server's shape: a cache sized for the context it serves, not
		// for this run. P7 made the attention cost the depth rather than
		// the cache, and this is how to check that it still does.
		nKV = ctx
	}
	g, err := llm.NewGraph(dev, m, llm.GraphOpts{MaxTokens: chunk, NKV: nKV, Layers: nLayers, Slots: maxR})
	if err != nil {
		return err
	}
	defer g.Destroy()
	reportStaging(g, time.Since(start))

	for s, p := range prompts {
		if err := g.UseSlot(s); err != nil {
			return err
		}
		for at := 0; at < len(p); at += chunk {
			c := p[at:min(at+chunk, len(p))]
			if at == 0 {
				_, _, err = g.Forward(c)
			} else {
				_, _, err = g.Extend(c)
			}
			if err != nil {
				return err
			}
		}
	}
	if err := g.UseSlot(0); err != nil {
		return err
	}
	fmt.Printf("%d slots prefilled to %d tokens each\n", maxR, depth)

	// The -graph table's columns, so the two CSVs read alike; its
	// against_llama column is a prefill ratio and means nothing here.
	table := [][]string{{"rows", "layers", "ms", "tok_s", "against_llama",
		"hc_ms", "ple_ms", "dn_ms", "attn_ms", "moe_ms", "head_ms", "move_ms", "gather_ms", "glue_ms"}}
	labels := [][]string{{"rows", "label", "n_per_pass", "ms_per_pass", "us_each", "pct_of_gpu"}}
	for _, n := range rows {
		slots := make([]int, n)
		ids := make([]int32, n)
		for r := range slots {
			slots[r] = r
			ids[r] = prompts[r][len(prompts[r])-1]
		}
		// Greedy, so each step routes the way generated text does: a row fed
		// the same token every step would read the same experts every step,
		// and a batch's cost is the union of its rows' experts.
		step := func() error {
			l, err := g.DecodeRows(slots, ids)
			if err != nil {
				return err
			}
			vocab := len(l) / n
			for r := range ids {
				ids[r] = argmax(l[r*vocab : (r+1)*vocab])
			}
			return nil
		}
		// Two untimed passes: the first one-row step records the replay.
		for range 2 {
			if err := step(); err != nil {
				return err
			}
		}
		g.ResetStats()
		experts := 0
		for range batchBenchSteps {
			if err := step(); err != nil {
				return err
			}
			experts += g.PassExperts()
		}
		table = append(table, reportGraphRun(g, n))
		fmt.Printf("    %.1f distinct experts in the last layer's pass, on average\n",
			float64(experts)/batchBenchSteps)
		labels = append(labels, graphLabelRows(g.Stats.Clone(), n)...)
	}
	if csvPath == "" {
		return nil
	}
	if err := writeCSV(csvPath, table); err != nil {
		return err
	}
	return writeCSV(labelCSVPath(csvPath), labels)
}

func argmax(l []float32) int32 {
	id := 0
	for j := range l {
		if l[j] > l[id] {
			id = j
		}
	}
	return int32(id)
}
