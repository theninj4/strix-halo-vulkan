// Command quanterr measures what each quantization format does to the
// *output* of a real projection, fed real activations — IDEAS §7's first
// item, and LLM.md's L0d.
//
//	go run ./cmd/quanterr
//	go run ./cmd/quanterr -layers 4 -tokens 32 -blocks 32,64,128,256
//
// Everything else in this repo ranks formats by GFLOP/s, and that ranking is
// settled: W4A8 decode reads 99-103% of the bus (§1.7), a 4-bit MoE prefill is
// 2.10x its fp16 twin (§2.2), and L0c took the fine scale block from a 14.2%
// tax to 1.0%. None of it says whether the numbers coming out are any good,
// and two decisions rest on that alone — W4A8 against W4A16 (a 3x throughput
// cliff, §1.1), and symmetric against asymmetric at equal bits per weight.
//
// The model is Z-Image's Qwen3-4B text encoder, because it is the Qwen3 this
// machine already has locally and its hidden size is **2560 — the same as
// qwen3.8-flash-next's**, so its projections are the target's dense tensors at
// the same width. `models/Qwen3-Embedding-0.6B` is the same family at 1024 and
// would say less.
//
// Real activations rather than a Gaussian matrix is the whole point: outliers
// are a property of a trained model, they are the entire risk in int8, and a
// random test matrix has none. Stage 6 met the same fact from the other
// direction — SwiGLU overflows fp16 "on a real prompt and never on a random
// one".
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"strix-halo-vulkan/bench"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// The default prompt is prose rather than a keyword list because activation
// statistics are what is being measured and a degenerate prompt gives
// degenerate ones.
const defaultPrompt = "A photograph of a red fox standing in deep snow at dawn, " +
	"its breath visible in the cold air, shot on an 85mm lens with a shallow " +
	"depth of field; the low sun picks out individual hairs along its back."

func main() {
	encDir := flag.String("encoder", "models/Z-Image-Turbo/text_encoder", "Qwen3 checkpoint to draw weights from")
	tokDir := flag.String("tokenizer", "models/Z-Image-Turbo/tokenizer", "tokenizer directory")
	prompt := flag.String("prompt", defaultPrompt, "prompt whose activations are measured")
	layers := flag.Int("layers", 2, "how many decoder layers to run (and draw sites from)")
	maxTokens := flag.Int("tokens", 24, "cap on tokens kept per site; the matmuls are float64 and O(tokens)")
	blocksFlag := flag.String("blocks", "32,64,128,256", "quantization block sizes to sweep")
	describe := flag.Bool("describe", false, "print each site's weight statistics and exit, instead of measuring error")
	sitesFlag := flag.String("sites", "q,o,gate,down", "which projections to measure: q,k,v,o,gate,up,down")
	flag.Parse()

	blocks, err := parseInts(*blocksFlag)
	if err != nil {
		log.Fatalf("-blocks: %v", err)
	}
	want := map[string]bool{}
	for _, s := range strings.Split(*sitesFlag, ",") {
		want[strings.TrimSpace(s)] = true
	}

	tok, err := tokenizer.Load(*tokDir)
	if err != nil {
		log.Fatal(err)
	}
	ids, err := tok.EncodePrompt(*prompt)
	if err != nil {
		log.Fatal(err)
	}
	model, err := qwen.Load(*encDir, *layers)
	if err != nil {
		log.Fatal(err)
	}

	tr := qwen.Trace{}
	if _, err := model.Forward(ids, tr); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d tokens through %d layers of %s\n", len(ids), *layers, *encDir)

	sites := collectSites(model, tr, *layers, *maxTokens, want)
	if len(sites) == 0 {
		log.Fatal("no sites selected")
	}

	if *describe {
		for _, s := range sites {
			describeWeights(s.Name, s.W)
		}
		return
	}

	var rows []bench.QuantErrorRow
	var outliers []bench.QuantOutliers
	for _, s := range sites {
		fmt.Fprintf(os.Stderr, "  %s: [%d,%d] weights, %d tokens\n", s.Name, s.Rows, s.Cols, s.Tokens)
		rows = append(rows, bench.AnalyseQuantSite(s, blocks)...)
		outliers = append(outliers, bench.AnalyseQuantOutliers(s))
	}
	bench.PrintQuantError(os.Stdout, rows, outliers)
}

// collectSites pairs each projection's weight with the activations the trace
// recorded arriving at it. The pairing is the part that has to be right: a
// weight measured against the wrong tensor would still produce a plausible
// error table.
func collectSites(m *qwen.Model, tr qwen.Trace, layers, maxTokens int, want map[string]bool) []bench.QuantSite {
	var out []bench.QuantSite
	for i := 0; i < layers && i < len(m.Layers); i++ {
		l := m.Layers[i]
		p := fmt.Sprintf("layer%d.", i)
		for _, c := range []struct {
			key  string
			lin  *qwen.Linear
			from string
		}{
			{"q", l.Q, "input_layernorm"},
			{"k", l.K, "input_layernorm"},
			{"v", l.V, "input_layernorm"},
			{"o", l.O, "attn_ctx"},
			{"gate", l.Gate, "post_attention_layernorm"},
			{"up", l.Up, "post_attention_layernorm"},
			{"down", l.Down, "swiglu"},
		} {
			if !want[c.key] {
				continue
			}
			x := tr[p+c.from]
			if x == nil {
				log.Fatalf("trace has no %q — the layer did not record its %s input", p+c.from, c.key)
			}
			if x.Cols != c.lin.In {
				log.Fatalf("%s%s: trace tensor is %d wide, the projection takes %d", p, c.key, x.Cols, c.lin.In)
			}
			t := x.Rows
			if t > maxTokens {
				t = maxTokens
			}
			out = append(out, bench.QuantSite{
				Name:   fmt.Sprintf("L%d.%s", i, c.key),
				Rows:   c.lin.Out,
				Cols:   c.lin.In,
				W:      c.lin.Weight,
				X:      x.Data[:t*x.Cols],
				Tokens: t,
			})
		}
	}
	return out
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		v, err := strconv.Atoi(f)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// describeWeights prints the statistics that distinguish a hard tensor from a
// broken one. It exists because L0d's first full sweep produced a site whose
// weight-reconstruction error was identical to four digits at every block
// size, which no real quantizer can do, and the question "what is in that
// tensor" needed answering before any of the table could be believed.
func describeWeights(name string, w []float32) {
	var lo, hi float64 = math.Inf(1), math.Inf(-1)
	var sum2 float64
	var nan, inf, zero int
	for _, v := range w {
		d := float64(v)
		switch {
		case math.IsNaN(d):
			nan++
			continue
		case math.IsInf(d, 0):
			inf++
			continue
		case d == 0:
			zero++
		}
		if d < lo {
			lo = d
		}
		if d > hi {
			hi = d
		}
		sum2 += d * d
	}
	n := float64(len(w) - nan - inf)
	fmt.Printf("  %-10s n=%d min=%.4g max=%.4g rms=%.4g nan=%d inf=%d zero=%d\n",
		name, len(w), lo, hi, math.Sqrt(sum2/n), nan, inf, zero)

	// How many blocks would have an fp16 scale below the smallest *normal*
	// fp16, where the format keeps only a handful of mantissa bits. Q8's
	// scale is maxAbs/127 and Q4's is maxAbs/7, so Q8's is 18x smaller and
	// falls off that cliff 18x sooner — which is the shape of the anomaly
	// L0d's first sweep turned up, and the reason this counter exists.
	const fp16MinNormal = 6.103515625e-05
	for _, block := range []int{32, 64, 128, 256} {
		var sub8, sub4, zero8 int
		blocks := len(w) / block
		for b := 0; b < blocks; b++ {
			var maxAbs float64
			for _, v := range w[b*block : (b+1)*block] {
				if a := math.Abs(float64(v)); a > maxAbs {
					maxAbs = a
				}
			}
			if maxAbs/127 < fp16MinNormal {
				sub8++
			}
			if maxAbs/127 < 6e-8 {
				zero8++
			}
			if maxAbs/7 < fp16MinNormal {
				sub4++
			}
		}
		fmt.Printf("      block %3d: %6.2f%% of Q8 scales subnormal in fp16 (%.3f%% flush to zero), "+
			"%.2f%% of Q4 scales subnormal\n",
			block, 100*float64(sub8)/float64(blocks), 100*float64(zero8)/float64(blocks),
			100*float64(sub4)/float64(blocks))
	}
}
