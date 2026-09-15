package main

// The full-attention layer's benchmark: LLM.md L2f, against the per-op
// attribution of llama.cpp's own graph that L2a produced.
//
//	go run ./cmd/llm -attn                        # the default sweep, 64..2048
//	go run ./cmd/llm -attn -tokens 512 -ladder    # every rung at the reference's ubatch
//	go run ./cmd/llm -attn -csv results/l2f_attn.csv
//
// What is timed is GPU dispatch time, swept across every staged layer. One
// layer's weights are 103 MB — 3.2x the 32 MiB MALL on their own — so even a
// single staged layer is read cold, but the sweep is kept because the
// *activations* are not: at 512 tokens the fused projection's output is 28.6
// MB, which is exactly the size the MALL would hold for a repeated dispatch
// and would not for a real graph running twelve different layers.
//
// The cache is 2048 cells by default and not the prompt's length, because that
// is what llama.cpp's measured graph had: its `MUL_MAT f32 m=512 n=2048 k=128`
// is the indexer scoring 512 blocks, i.e. 2048 cells, and its FLASH_ATTN_EXT
// reads `k(256,2048,2,1)`. Sizing ours the same way is what makes the indexer
// lines comparable at all.

import (
	"fmt"
	"strconv"
	"time"

	"strix-halo-vulkan/llm"
)

// The lines of llama.cpp's 512-token prefill graph at `-ub 512` that are
// nameably one full-attention layer, from results/l2a_prefill_ops.csv. Twelve
// of the 48 layers are this one, so a line with 12 dispatches is one per
// layer; 24 is two, 48 is four.
//
// Three attributions are worth stating rather than trusting:
//
//   - `MUL_MAT q8_0 m=2560 n=512 k=6144` runs 48 times, but only 12 of those
//     are `attn_output`: the other 36 are the DeltaNet layers' `ssm_out`,
//     which is the same shape. A quarter of that line is taken.
//   - `ROPE` runs 48 times, which is four per full-attention layer — the
//     query, the key, the indexer's query and its pooled key. It is split in
//     half between the pack and the indexer.
//   - `TOP_K` and `TOPK_QSA GET_ROWS` are the selection, which this graph does
//     not do at all (at any prompt shorter than top_k it names every cell and
//     is bit-identical to not selecting). They are reported beside the
//     comparison and excluded from it, which is the conservative direction.
//
// Everything else this layer spends — the pooling itself, the gate's SIGMOID
// and MUL, the CONT that makes the gated context contiguous, the CPYs into the
// KV cache — is inside MUL (479), SIGMOID (326), CONT (251) and CPY (85),
// which every other block of the model also uses. None of it is attributed
// here, and every one of those dispatches is *absent* from our graph rather
// than faster in it.
var llamaAttn = []llamaOp{
	{"qkv", "MUL_MAT q8_0 m=12288 n=512 k=2560 (query and gate)", 12, 20384.4},
	{"qkv", "MUL_MAT q8_0 m=512 n=512 k=2560 (key, value)", 24, 2780.0},
	{"qkv", "MUL_MAT bf16 m=512 n=512 k=2560 (indexer query)", 12, 3491.2},
	{"qkv", "MUL_MAT bf16 m=128 n=512 k=2560 (indexer key)", 12, 1535.6},
	{"pack", "RMS_NORM_MUL RMS_NORM(256,24,512,1) (query)", 12, 2033.8},
	{"pack", "RMS_NORM_MUL RMS_NORM(256,2,512,1) (key)", 12, 163.3},
	{"pack", "ROPE (query, key: half of 48)", 24, 567.4},
	{"idx", "RMS_NORM_MUL RMS_NORM(128,4,512,1) (indexer query)", 12, 272.9},
	{"idx", "RMS_NORM_MUL RMS_NORM(128,512,1,1) (indexer key)", 12, 84.8},
	{"idx", "ROPE (indexer query, pooled key: half of 48)", 24, 567.4},
	{"score", "MUL_MAT f32 m=512 n=2048 k=128", 12, 460.3},
	{"score", "RELU", 12, 269.4},
	{"attn", "FLASH_ATTN_EXT dst(256,24,512,1) k(256,2048,2,1)", 12, 25180.3},
	{"out", "MUL_MAT q8_0 m=2560 n=512 k=6144 (12 of that line's 48)", 12, 8558.9},
}

// The selection, which our graph does not contain. Reported, not compared.
var llamaAttnTopK = []llamaOp{
	{"select", "TOP_K K=2048 (2048,512,1,1)", 12, 2194.6},
	{"select", "TOPK_QSA GET_ROWS", 12, 206.7},
}

// llamaFlashGFLOPs is what llama.cpp's flash-attention kernel *achieved* on
// the work it issued, from the same CSV row.
//
// It is here because the `attn` line is the one comparison in this table that
// is not like for like, and saying so needs a number. The reference's kernel
// walks the whole [512, 2048] rectangle its cache defines — ggml's own FLOP
// count says so, 2*24*512*2048*256*2 over 2098 us — while ours walks the
// causal triangle of a 512-token prefill, an eighth of that. So the ms ratio
// on that row is mostly a difference in *work*, and the rate is what compares
// the two kernels.
const llamaFlashGFLOPs = 12280.9

// layersPerGraph is how many full-attention layers a forward pass runs: 12 of
// the 48, the interval being 4.
const layersPerGraph = 12

func attnBench(model string, tokens []int, ctx, nLayers, iters int, ladder bool, csvPath string) error {
	m, err := llm.Open(model)
	if err != nil {
		return err
	}
	defer m.Close()

	// The full-attention layers, found by asking the checkpoint rather than by
	// assuming the interval.
	var idx []int
	var cfg llm.AttnConfig
	for l := 0; l < m.Config.NLayer && len(idx) < nLayers; l++ {
		c, ok, err := m.AttnConfig(l)
		if err != nil {
			return err
		}
		if ok {
			if len(idx) == 0 {
				cfg = c
			}
			idx = append(idx, l)
		}
	}
	if len(idx) == 0 {
		return fmt.Errorf("checkpoint has no full-attention layers")
	}
	fmt.Printf("%s\nlayers %v of %d, %d heads of %d over %d kv, indexer %d x %d, top_k %d, ratio %d\n",
		model, idx, m.Config.NLayer, cfg.NHead, cfg.HeadDim, cfg.NHeadKV,
		cfg.IdxHeads, cfg.IdxDim, cfg.TopK, cfg.Ratio)

	var ws []llm.AttnWeights
	for _, l := range idx {
		w, err := m.AttnWeights(l)
		if err != nil {
			return fmt.Errorf("layer %d: %w", l, err)
		}
		ws = append(ws, w)
	}

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()

	maxTok := 0
	for _, t := range tokens {
		maxTok = max(maxTok, t)
	}
	nKV := max(ctx, roundUp(maxTok, 256))
	g, err := llm.NewAttnGPU(dev, cfg, maxTok, nKV, ws)
	if err != nil {
		return err
	}
	defer g.Destroy()
	fmt.Printf("%d layers staged: %.1f MB of weights, %.1f MB of arenas for %d tokens in %d cells (%d blocks)\n\n",
		g.Layers(), float64(g.WeightBytes())/1e6, float64(g.ActivationBytes())/1e6,
		maxTok, g.NKV(), g.NBlocks())

	var plans [][3]string
	if ladder {
		for _, a := range llm.AttnKernels() {
			for _, k := range llm.GEMMKernels() {
				for _, o := range llm.GEMMKernels() {
					plans = append(plans, [3]string{string(a), string(k), string(o)})
				}
			}
		}
	}

	var rows [][]string
	rows = append(rows, []string{"tokens", "cells", "attn_kernel", "gemm_kernel", "out_kernel",
		"dispatch", "us_per_layer", "us_per_graph", "gflops", "gbps"})
	for _, tok := range tokens {
		xn := make([]float32, tok*cfg.NEmbd)
		for i := range xn {
			// Anything with a realistic magnitude: this measures dispatch
			// time, and the only value-dependent cost here is a denormal.
			xn[i] = float32((i%97)-48) * 0.01
		}
		if err := g.Upload(xn, tok); err != nil {
			return err
		}
		// Upload has already chosen the measured schedule for this length;
		// naming it back through SetPlan would freeze it for every length
		// after, which is what the empty `plans` case is avoiding.
		todo := plans
		if todo == nil {
			todo = [][3]string{{}}
		}
		for _, p := range todo {
			if p[0] != "" {
				if err := g.SetPlan(llm.AttnKernel(p[0]), llm.GEMMKernel(p[1]), llm.GEMMKernel(p[2])); err != nil {
					return err
				}
			}
			a, k, o := g.Plan()
			p = [3]string{string(a), string(k), string(o)}
			st, err := g.ProfileSweep(iters)
			if err != nil {
				return err
			}
			fmt.Printf("T = %-5d %s / %s / %s\n", tok, p[0], p[1], p[2])
			var total time.Duration
			for _, s := range st {
				total += s.GPU
				us := float64(s.GPU.Nanoseconds()) / 1e3
				gf, gb := attnRates(s.Kind, cfg, tok, g.NKV(), s.GPU)
				fmt.Printf("  %-6s %9.1f us  x%d = %7.1f ms", s.Kind, us, layersPerGraph,
					us*layersPerGraph/1e3)
				if gf > 0 {
					fmt.Printf("  %8.0f GFLOP/s", gf)
				}
				if gb > 0 {
					fmt.Printf("  %6.0f GB/s", gb)
				}
				fmt.Println()
				rows = append(rows, []string{
					strconv.Itoa(tok), strconv.Itoa(g.NKV()), p[0], p[1], p[2], s.Kind,
					fmt.Sprintf("%.3f", us), fmt.Sprintf("%.1f", us*layersPerGraph/1e3),
					fmt.Sprintf("%.1f", gf), fmt.Sprintf("%.1f", gb),
				})
			}
			us := float64(total.Nanoseconds()) / 1e3
			fmt.Printf("  %-6s %9.1f us  x%d = %7.1f ms\n", "layer", us, layersPerGraph,
				us*layersPerGraph/1e3)
			rows = append(rows, []string{
				strconv.Itoa(tok), strconv.Itoa(g.NKV()), p[0], p[1], p[2], "layer",
				fmt.Sprintf("%.3f", us), fmt.Sprintf("%.1f", us*layersPerGraph/1e3), "", "",
			})
			if tok == 512 && plans == nil {
				// The causal FLOP count of one attention dispatch: two GEMMs
				// over half the score matrix.
				flops := 2 * float64(cfg.NHead) * float64(tok) * float64(tok) * float64(cfg.HeadDim)
				reportAttnAgainstLlama(st, total, flops)
			}
			fmt.Println()
		}
	}

	if csvPath != "" {
		if err := writeCSV(csvPath, rows); err != nil {
			return err
		}
		fmt.Printf("wrote %s\n", csvPath)
	}
	return nil
}

// reportAttnAgainstLlama puts the measured layer beside the lines of
// llama.cpp's own graph it replaces. Both sides are one prefill graph of 512
// tokens over a 2048-cell cache.
func reportAttnAgainstLlama(st []llm.Stage, total time.Duration, flops float64) {
	byKind := map[string]float64{}
	disp := map[string]int{}
	var refTotal float64
	var refDisp int
	for _, op := range llamaAttn {
		byKind[op.kind] += op.usTotal
		disp[op.kind] += op.dispatches
		refTotal += op.usTotal
		refDisp += op.dispatches
	}
	fmt.Println("\n  against llama.cpp's own graph at -ub 512 (results/l2a_prefill_ops.csv):")
	fmt.Printf("  %-6s %6s %12s %6s %12s %8s\n", "", "disp", "llama.cpp", "disp", "ours", "")
	var ours float64
	for _, s := range st {
		us := float64(s.GPU.Nanoseconds()) / 1e3 * layersPerGraph
		ours += us
		ref := byKind[s.Kind]
		fmt.Printf("  %-6s %6d %9.1f ms %6d %9.1f ms  %6.2fx\n",
			s.Kind, disp[s.Kind], ref/1e3, layersPerGraph, us/1e3, ref/us)
	}
	fmt.Printf("  %-6s %6d %9.1f ms %6d %9.1f ms  %6.2fx\n",
		"layer", refDisp, refTotal/1e3, layersPerGraph*len(st), ours/1e3, refTotal/ours)
	fmt.Printf("  %.1f%% of llama.cpp's %.0f ms graph becomes %.1f%%, a %.1f ms saving\n",
		100*refTotal/llamaGraphUs, llamaGraphUs/1e3, 100*ours/llamaGraphUs, (refTotal-ours)/1e3)

	// The one row that is not like for like, priced both ways.
	for _, st := range st {
		if st.Kind != "attn" || flops <= 0 {
			continue
		}
		rate := flops / st.GPU.Seconds() / 1e9
		equiv := flops / (llamaFlashGFLOPs * 1e9) * 1e6 * layersPerGraph // us per graph
		fmt.Printf("  — `attn` is the row that is not like for like. The reference's kernel\n")
		fmt.Printf("    issues a [512, 2048] rectangle where ours walks a causal triangle, so\n")
		fmt.Printf("    %.2fx is mostly work. On rate it is %.1f TFLOP/s against %.1f, %.2fx;\n",
			byKind["attn"]/(st.GPU.Seconds()*1e6*layersPerGraph), rate/1e3, llamaFlashGFLOPs/1e3,
			rate/llamaFlashGFLOPs)
		fmt.Printf("    at our shape and its own rate the reference would be %.1f ms, making\n", equiv/1e3)
		fmt.Printf("    the layer %.1f ms against our %.1f — %.2fx.\n",
			(refTotal-byKind["attn"]+equiv)/1e3, ours/1e3, (refTotal-byKind["attn"]+equiv)/ours)
	}

	var sel float64
	var selDisp int
	for _, op := range llamaAttnTopK {
		sel += op.usTotal
		selDisp += op.dispatches
	}
	fmt.Printf("  — and %d dispatches of selection (%.1f ms, TOP_K and its GET_ROWS) are left\n", selDisp, sel/1e3)
	fmt.Println("    out of the comparison entirely: at this width the selection names every")
	fmt.Println("    cell, so it is work neither side needs and only one side does. Counting")
	fmt.Printf("    it would make the reference %.1f ms and the ratio %.2fx.\n", (refTotal+sel)/1e3, (refTotal+sel)/ours)
	fmt.Println("  — the pooling, the gate's sigmoid and multiply, the CONT behind them and")
	fmt.Println("    the CPYs into the KV cache are inside MUL/SIGMOID/CONT/CPY, which every")
	fmt.Println("    other block shares; none of it is attributed and all of it is absent here.")
}

// attnRates reports what a dispatch achieved: arithmetic for the three
// matmuls, bandwidth for the two passes that only move activations.
func attnRates(kind string, c llm.AttnConfig, tok, nKV int, d time.Duration) (gflops, gbps float64) {
	if d <= 0 {
		return 0, 0
	}
	s := d.Seconds()
	t := float64(tok)
	embd := float64(c.NEmbd)
	qkvN := float64(c.QWidth() + 2*c.KVWidth() + c.IdxHeads*c.IdxDim + c.IdxDim)
	switch kind {
	case "qkv":
		return 2 * t * qkvN * embd / s / 1e9, (qkvN*embd*2 + t*embd*2) / s / 1e9
	case "pack":
		// The three tensors out of the fp32 projection and into fp16 tiles.
		w := float64((c.NHead + 2*c.NHeadKV) * c.HeadDim)
		return 0, (t*w*4 + t*w*2) / s / 1e9
	case "attn":
		// Causal: half the score matrix and half the value average.
		h, hd := float64(c.NHead), float64(c.HeadDim)
		return 2 * h * t * t * hd / s / 1e9, 0
	case "out":
		g := float64(c.GateWidth())
		return 2 * t * embd * g / s / 1e9, (embd*g*2 + t*g*2) / s / 1e9
	case "score":
		nb := float64((nKV + c.Ratio - 1) / c.Ratio)
		return 2 * t * nb * float64(c.IdxHeads*c.IdxDim) / s / 1e9, 0
	}
	return 0, 0
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }
