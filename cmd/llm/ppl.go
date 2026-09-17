package main

// Perplexity (LLM.md L8c): the accuracy instrument this vertical has been
// grading itself against a *tensor* for, and now grades itself against a
// corpus.
//
//	go run ./cmd/llm -ppl -model $M
//	go run ./cmd/llm -ppl -chunks 8 -csv results/l8c_ppl.csv
//
// L1 measured the checkpoint's own perplexity with llama.cpp — `PPL 4.0340
// +/- 0.02283` over `wiki.test.raw` at `-c 2048 -b 2048` — and every stage
// since has compared *tensors* against a dump, because nothing our kernels
// did changed what the model computes. L8c is the first stage that does: the
// dense half is about to stop being the checkpoint's 8.5 bits a weight. So
// the delta its gate asks for needs two numbers from the same corpus, and
// this is the one that is ours.
//
// **The protocol is llama.cpp's, line for line**, because a perplexity is
// only comparable to the protocol that produced it
// (`tools/perplexity/perplexity.cpp` at the oracle's own build, cff184438):
//
//   - the whole file tokenized once, with no BOS — this checkpoint's
//     `tokenizer.ggml.add_bos_token` is false, so llama.cpp's per-chunk BOS
//     substitution does not fire;
//   - `n_chunk = len(tokens) / n_ctx`, each chunk a *fresh* sequence with the
//     cache cleared, which is what `Graph.ForwardRows` resets;
//   - the logits of positions `[n_ctx/2, n_ctx-1)` scored against the token
//     one place to their right — the second half of the window, so that
//     every graded token has at least 1024 tokens of context;
//   - `n_ctx - 1 - n_ctx/2` = 1023 tokens counted a chunk;
//   - NLL from a float32 max, a float64 sum of `expf(logit - max)`, and
//     `PPL = exp(sum nll / count)`.
//
// The uncertainty is llama.cpp's too: the sample standard error of the
// per-token NLL, `sqrt((<nll^2> - <nll>^2)/(count-1))`, reported on the
// log scale and as the multiplicative band it puts around the perplexity.

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"

	"strix-halo-vulkan/llm"
)

// llamaPPL is llama.cpp's own number on this checkpoint, this corpus and this
// protocol (L1): `-c 2048 -b 2048` over `models/wikitext-2-raw/wiki.test.raw`.
// It is the number L8c's gate is a delta from, and it was measured on the
// build behind every trace here — a rebuild invalidates it (LLM.md's open
// question on the pinned oracle).
const llamaPPL = 4.0340

// oursPPL is L8c-0's: the same protocol over our own graph on the bank we
// run today — the checkpoint's own Q4_K/Q5_K/Q5_1/Q8_0 experts and its Q8_0
// dense weights. **This is the number a candidate width is a delta from**
// (D17), because ours and the reference's already differ by -0.13% at
// identical weights for reasons that are settled and are not the bank.
const oursPPL = 4.0289

// pplOpts is what the -ppl flags come to.
type pplOpts struct {
	model, file string
	ctx         int
	chunks      int
	headRows    int
	layers      int
	csv         string
}

// perplexity runs the corpus and prints a running estimate per chunk.
func perplexity(o pplOpts) error {
	m, err := llm.Open(o.model)
	if err != nil {
		return err
	}
	defer m.Close()
	tok, err := llm.LoadTokenizer(m.Set)
	if err != nil {
		return err
	}
	buf, err := os.ReadFile(o.file)
	if err != nil {
		return err
	}
	ids, err := tok.Encode(string(buf))
	if err != nil {
		return err
	}
	if len(ids) < 2*o.ctx {
		return fmt.Errorf("%s tokenizes to %d tokens, a %d-cell context needs %d",
			o.file, len(ids), o.ctx, 2*o.ctx)
	}
	nChunk := len(ids) / o.ctx
	if o.chunks > 0 && o.chunks < nChunk {
		nChunk = o.chunks
	}
	first := o.ctx / 2
	perChunk := o.ctx - 1 - first

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()

	fmt.Printf("%s\n%s: %d tokens, %d chunks of %d, scoring %d each from position %d\n",
		o.model, o.file, len(ids), nChunk, o.ctx, perChunk, first)
	if p := llm.DenseBankPlan(); !p.Off() {
		// The *real* bank, not a simulation of one (L8c-4). It is printed on
		// the same line the simulation gets, and for the same reason: a
		// perplexity is only attributable if the run says which weights it
		// was measured over.
		fmt.Printf("dense bank: %s — %v\n\n", p, p.Widths())
	}
	if p := llm.DensePlan(); !p.Off() {
		// L8c-1: a candidate bank, staged as the halves it would produce.
		// Printed before staging rather than after, because a run this long
		// with the wrong bank is worth catching in the first second.
		fmt.Printf("dense width simulation, on the fp16 arm: %s\n", p)
		for _, w := range p.Widths() {
			fmt.Printf("    %s\n", w)
		}
	}

	start := time.Now()
	g, err := llm.NewGraph(dev, m, llm.GraphOpts{
		MaxTokens: o.ctx, NKV: o.ctx, Layers: o.layers, HeadRows: o.headRows,
	})
	if err != nil {
		return err
	}
	defer g.Destroy()
	reportStaging(g, time.Since(start))
	if p := llm.DensePlan(); !p.Off() {
		// What the simulation actually reached. A family listed at 0.000B —
		// or missing — is a staging path the hook does not cover, which is
		// how `lm_head` once screened as free (sim.go).
		fmt.Printf("simulated: %s\n", llm.SimTallyLine())
		if u := llm.SimUncalibratedLine(); u != "" {
			fmt.Printf("           %s\n", u)
		}
		fmt.Println()
	}

	var nll, nll2 float64
	count := 0
	rows := [][]string{{"chunk", "tokens", "nll", "ppl", "stderr", "seconds"}}
	t0 := time.Now()
	for c := 0; c < nChunk; c++ {
		chunk := ids[c*o.ctx : (c+1)*o.ctx]
		tc := time.Now()
		var reduce time.Duration
		// The last row predicts a token in the next chunk, which llama.cpp
		// does not score, so it is computed and dropped rather than asked
		// for: the head's slab is a whole number of rows either way.
		err := g.ForwardRows(chunk, first, func(t int, logits []float32) error {
			if t+1 >= len(chunk) {
				return nil
			}
			tr := time.Now()
			v := negLogSoftmax(logits, chunk[t+1])
			reduce += time.Since(tr)
			nll, nll2 = nll+v, nll2+v*v
			count++
			return nil
		})
		if err != nil {
			return fmt.Errorf("chunk %d: %w", c, err)
		}
		av := nll / float64(count)
		se := stderr(nll, nll2, count)
		el := time.Since(tc)
		fmt.Printf("[%3d/%d] ppl %8.4f  nll %7.4f  +/- %.4f  %5.1fs (%.1fs reduce)  (eta %s)\n",
			c+1, nChunk, math.Exp(av), av, math.Exp(av)*se, el.Seconds(), reduce.Seconds(),
			(time.Duration(nChunk-c-1) * time.Since(t0) / time.Duration(c+1)).Round(time.Second))
		rows = append(rows, []string{
			fmt.Sprint(c + 1), fmt.Sprint(count),
			fmt.Sprintf("%.6f", av), fmt.Sprintf("%.4f", math.Exp(av)),
			fmt.Sprintf("%.4f", math.Exp(av)*se), fmt.Sprintf("%.2f", el.Seconds()),
		})
	}
	if count == 0 {
		return fmt.Errorf("no tokens scored")
	}
	av := nll / float64(count)
	ppl, band := math.Exp(av), math.Exp(av)*stderr(nll, nll2, count)
	fmt.Printf("\nFinal estimate: PPL = %.4f +/- %.5f  over %d tokens in %s\n",
		ppl, band, count, time.Since(t0).Round(time.Second))
	// The two reference numbers are whole-corpus numbers, so a partial run
	// has nothing to compare against: L8c-0 measured a four-chunk estimate
	// reading +0.32% where the corpus reads -0.13%, and printing a delta
	// from a different number of tokens would invite exactly that mistake.
	if nChunk < len(ids)/o.ctx {
		fmt.Printf("%d of %d chunks — a partial run is a screen, not a delta; "+
			"the whole corpus is llama.cpp 4.0340 and ours 4.0289\n",
			nChunk, len(ids)/o.ctx)
		return writePPL(o.csv, rows)
	}
	fmt.Printf("llama.cpp on the same corpus, context and chunking: PPL = %.4f +/- %.5f\n",
		llamaPPL, 0.02283)
	fmt.Printf("delta: %+.4f (%+.2f%%)\n", ppl-llamaPPL, 100*(ppl-llamaPPL)/llamaPPL)
	// D17: a width is graded against our own bank, not the reference's, since
	// the two already differ by -0.13% at identical weights (L8c-0).
	fmt.Printf("ours on the checkpoint's own widths (L8c-0):        PPL = %.4f\n", oursPPL)
	fmt.Printf("delta: %+.4f (%+.2f%%)  <- the number a width is graded on\n",
		ppl-oursPPL, 100*(ppl-oursPPL)/oursPPL)
	return writePPL(o.csv, rows)
}

func writePPL(path string, rows [][]string) error {
	if path == "" {
		return nil
	}
	if err := writeCSV(path, rows); err != nil {
		return err
	}
	fmt.Printf("wrote %s\n", path)
	return nil
}

// stderr is llama.cpp's own uncertainty on the mean NLL: the sample standard
// error over the counted tokens, on the log scale.
func stderr(nll, nll2 float64, count int) float64 {
	if count < 2 {
		return 0
	}
	av := nll / float64(count)
	v := nll2/float64(count) - av*av
	if v <= 0 {
		return 0
	}
	return math.Sqrt(v / float64(count-1))
}

// negLogSoftmax is `-(logits[tok] - max - log(sum exp(logits - max)))`, in
// llama.cpp's own arithmetic: a float32 max, `expf` summands and a float64
// accumulator. The sum is over 248320 values, which is the one place in this
// file worth spreading over the cores.
func negLogSoftmax(logits []float32, tok int32) float64 {
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	n := runtime.NumCPU()
	if n > 16 {
		n = 16
	}
	sums := make([]float64, n)
	var wg sync.WaitGroup
	span := (len(logits) + n - 1) / n
	for i := 0; i < n; i++ {
		lo, hi := i*span, minInt((i+1)*span, len(logits))
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(i, lo, hi int) {
			defer wg.Done()
			var s float64
			for _, v := range logits[lo:hi] {
				s += float64(expf(v - maxLogit))
			}
			sums[i] = s
		}(i, lo, hi)
	}
	wg.Wait()
	var sum float64
	for _, s := range sums {
		sum += s
	}
	return float64(maxLogit) + math.Log(sum) - float64(logits[tok])
}

// expf is C's `expf`: the float32 exponential. Go has only the float64 one,
// and rounding its result is not the same function — but it is the same to
// well inside a float32 ulp, and what matters here is that the summands are
// float32-sized, which is what keeps the sum comparable to the reference's.
func expf(x float32) float32 { return float32(math.Exp(float64(x))) }

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
