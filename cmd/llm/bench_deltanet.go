package main

// The gated DeltaNet's benchmark: LLM.md L3b, against the per-op attribution
// of llama.cpp's own graph that L2a produced.
//
//	go run ./cmd/llm -dn                        # the default sweep, 64..2048
//	go run ./cmd/llm -dn -tokens 512 -ladder    # every scan rung at the reference's ubatch
//	go run ./cmd/llm -dn -csv results/l3b_dn.csv
//
// What is timed is GPU dispatch time, swept across every staged layer, for the
// reason the other two benches give: one layer's weights are 116 MB and are
// read cold whatever happens, but at 512 tokens the fused projection's output
// is 33.8 MB — just past the MALL — and a repeated dispatch would keep the
// activations warm in a way a real graph running 36 different layers does not.
//
// `-ladder` sweeps the scan's nineteen rungs against the GEMM schedule this
// length has already chosen; `-gemm-ladder` adds the 9-way cross product over
// both row blocks on top. The scan's is the interesting one and it is the
// question L3a left open: LPC is how many lanes own one column of the
// [128, 128] state, and moving it from 64 to 1 trades occupancy for registers
// and for a proportional cut in how many times each head's q and k row is
// re-read. The two ends are 6.3x apart on identical arithmetic.

import (
	"fmt"
	"strconv"
	"time"

	"strix-halo-vulkan/llm"
)

// The lines of llama.cpp's 512-token prefill graph at `-ub 512` that are
// nameably one linear-attention layer, from results/l2a_prefill_ops.csv.
// Thirty-six of the 48 layers are this one, so a line with 36 dispatches is
// one per layer and 72 is two.
//
// Two attributions are worth stating rather than trusting:
//
//   - `MUL_MAT q8_0 m=10240 n=512 k=2560` runs 37 times, not 36. The extra one
//     is layer 1's `ple_key`, which is the same shape; 36/37 of the line is
//     taken.
//   - `MUL_MAT q8_0 m=2560 n=512 k=6144` runs 48 times, but only 36 of those
//     are `ssm_out`: the other 12 are the full-attention layers' `attn_output`,
//     which is the same shape. Three quarters of that line is taken, and L2f
//     takes the other quarter.
//
// `RMS_NORM_MUL RMS_NORM(128,48,512,1)` is unambiguous — 48 heads of 128 over
// 512 tokens, 36 dispatches — and is the gated output norm. What sits beside
// it and is *not* attributed is the sigmoid of z, the multiply against it, the
// CONTs behind both and the SiLU the convolution carries, all of which live in
// the shared MUL (479), SIGMOID (326) and CONT (251) lines that every other
// block of the model also uses. Every one of those dispatches is **absent**
// from our graph rather than faster in it, so leaving them out is the
// conservative direction.
var llamaDN = []llamaOp{
	{"qkv", "MUL_MAT q8_0 m=10240 n=512 k=2560 (36 of that line's 37)", 36, 48485.8},
	{"qkv", "MUL_MAT q8_0 m=6144 n=512 k=2560 (the output gate z)", 36, 34977.8},
	{"qkv", "MUL_MAT f32 m=48 n=512 k=2560 (ssm_alpha, ssm_beta)", 72, 7841.0},
	{"conv", "SSM_CONV_SILU SSM_CONV", 36, 6785.2},
	{"conv", "L2_NORM (q and k)", 72, 4548.5},
	{"conv", "SOFTPLUS", 36, 88.4},
	{"scan", "GATED_DELTA_NET", 36, 15783.0},
	{"norm", "RMS_NORM_MUL RMS_NORM(128,48,512,1)", 36, 8584.6},
	{"out", "MUL_MAT q8_0 m=2560 n=512 k=6144 (36 of that line's 48)", 36, 25676.8},
}

// dnLayersPerGraph is how many linear-attention layers a forward pass runs: 36
// of the 48, the full-attention interval being 4.
const dnLayersPerGraph = 36

// llamaScanUsPerLayer is what the reference's own scan costs per layer at
// ubatch 512, which is the number L3a-2 said the kernel has to beat.
const llamaScanUsPerLayer = 438.416

// dnBench runs the sweep. `ladder` walks the scan's rungs against the GEMM
// schedule Upload already chose, because the scan is the open question and the
// two projections are the same kernel L2f already laddered; `gemmLadder` adds
// the 9-way cross product over both row blocks on top of it, which is what
// checks that L2f's schedule carries to a differently sized weight.
func dnBench(model string, tokens []int, nLayers, iters int, ladder, gemmLadder bool, csvPath string) error {
	m, err := llm.Open(model)
	if err != nil {
		return err
	}
	defer m.Close()

	// The linear-attention layers, found by asking the checkpoint rather than
	// by assuming the interval.
	var idx []int
	var cfg llm.DeltaNetConfig
	for l := 0; l < m.Config.NLayer && len(idx) < nLayers; l++ {
		c, ok, err := m.DeltaNetConfig(l)
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
		return fmt.Errorf("checkpoint has no linear-attention layers")
	}
	// The build's spelling of the L2 norm, not the model's: what this
	// measures has to be what the tests check (research/l3-deltanet.md).
	cfg.QKNorm = llm.L2Max
	fmt.Printf("%s\nlayers %v of %d, %d value heads and %d key heads of %d, conv k=%d, state %.1f MB a layer\n",
		model, idx, m.Config.NLayer, cfg.NHeadV, cfg.NHeadK, cfg.HeadDim, cfg.Conv,
		float64(cfg.StateSize()*4)/1e6)

	var ws []llm.DeltaNetWeights
	for _, l := range idx {
		w, err := m.DeltaNetWeights(l)
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
	g, err := llm.NewDeltaNetGPU(dev, cfg, maxTok, ws)
	if err != nil {
		return err
	}
	defer g.Destroy()
	fmt.Printf("%d layers staged: %.1f MB of weights, %.1f MB of arenas for %d tokens\n\n",
		g.Layers(), float64(g.WeightBytes())/1e6, float64(g.ActivationBytes())/1e6, maxTok)

	var plans [][3]string
	if ladder || gemmLadder {
		scans := llm.DNKernels()
		if gemmLadder && !ladder {
			scans = []llm.DNKernel{llm.DefaultDNKernel()}
		}
		for _, s := range scans {
			if !gemmLadder {
				plans = append(plans, [3]string{string(s), "", ""})
				continue
			}
			for _, k := range llm.GEMMKernels() {
				for _, o := range llm.GEMMKernels() {
					plans = append(plans, [3]string{string(s), string(k), string(o)})
				}
			}
		}
	}

	var rows [][]string
	rows = append(rows, []string{"tokens", "scan_kernel", "operands", "regs_per_lane", "workgroups", "qk_reread",
		"gemm_kernel", "out_kernel", "dispatch", "us_per_layer", "us_per_graph", "gflops", "gbps"})
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
		todo := plans
		if todo == nil {
			todo = [][3]string{{}}
		}
		for _, p := range todo {
			if p[0] != "" {
				// The measured schedule for *this* length, re-derived rather
				// than read back: SetPlan turns the auto-plan off, so a rung
				// named once would otherwise freeze at every length after.
				k, o := llm.GEMMKernelFor(tok), llm.OutGEMMKernelFor(tok)
				if p[1] != "" {
					k, o = llm.GEMMKernel(p[1]), llm.GEMMKernel(p[2])
				}
				if err := g.SetPlan(llm.DNKernel(p[0]), k, o); err != nil {
					return err
				}
			}
			s, k, o := g.Plan()
			p = [3]string{string(s), string(k), string(o)}
			regs, wgs, reread := llm.DNShape(llm.DNKernel(p[0]), cfg)
			arm := "LDS"
			if llm.DNOperandsInRegisters(llm.DNKernel(p[0])) {
				arm = "reg"
			}
			if llm.DNPrefetches(llm.DNKernel(p[0])) {
				arm += "+pf"
			}
			st, err := g.ProfileSweep(iters)
			if err != nil {
				return err
			}
			fmt.Printf("T = %-5d %-5s (%3d regs/lane, %4d workgroups, q/k %s read %3dx) / %s / %s\n",
				tok, p[0], regs, wgs, arm, reread, p[1], p[2])
			var total time.Duration
			for _, s := range st {
				total += s.GPU
				us := float64(s.GPU.Nanoseconds()) / 1e3
				gf, gb := dnRates(s.Kind, cfg, tok, reread, s.GPU)
				fmt.Printf("  %-6s %9.1f us  x%d = %7.1f ms", s.Kind, us, dnLayersPerGraph,
					us*dnLayersPerGraph/1e3)
				if gf > 0 {
					fmt.Printf("  %8.0f GFLOP/s", gf)
				}
				if gb > 0 {
					fmt.Printf("  %6.0f GB/s", gb)
				}
				if s.Kind == "scan" {
					fmt.Printf("   (llama.cpp: %.1f us, %.2fx)", llamaScanUsPerLayer, llamaScanUsPerLayer/us)
				}
				fmt.Println()
				rows = append(rows, []string{
					strconv.Itoa(tok), p[0], arm, strconv.Itoa(regs), strconv.Itoa(wgs), strconv.Itoa(reread),
					p[1], p[2], s.Kind,
					fmt.Sprintf("%.3f", us), fmt.Sprintf("%.1f", us*dnLayersPerGraph/1e3),
					fmt.Sprintf("%.1f", gf), fmt.Sprintf("%.1f", gb),
				})
			}
			us := float64(total.Nanoseconds()) / 1e3
			fmt.Printf("  %-6s %9.1f us  x%d = %7.1f ms\n", "layer", us, dnLayersPerGraph,
				us*dnLayersPerGraph/1e3)
			rows = append(rows, []string{
				strconv.Itoa(tok), p[0], arm, strconv.Itoa(regs), strconv.Itoa(wgs), strconv.Itoa(reread),
				p[1], p[2], "layer",
				fmt.Sprintf("%.3f", us), fmt.Sprintf("%.1f", us*dnLayersPerGraph/1e3), "", "",
			})
			if tok == 512 && plans == nil {
				reportDNAgainstLlama(st, total)
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

// reportDNAgainstLlama puts the measured layer beside the lines of llama.cpp's
// own graph it replaces. Both sides are one prefill graph of 512 tokens.
func reportDNAgainstLlama(st []llm.Stage, total time.Duration) {
	byKind := map[string]float64{}
	disp := map[string]int{}
	var refTotal float64
	var refDisp int
	for _, op := range llamaDN {
		byKind[op.kind] += op.usTotal
		disp[op.kind] += op.dispatches
		refTotal += op.usTotal
		refDisp += op.dispatches
	}
	fmt.Println("\n  against llama.cpp's own graph at -ub 512 (results/l2a_prefill_ops.csv):")
	fmt.Printf("  %-6s %6s %12s %6s %12s %8s\n", "", "disp", "llama.cpp", "disp", "ours", "")
	var ours float64
	for _, s := range st {
		us := float64(s.GPU.Nanoseconds()) / 1e3 * dnLayersPerGraph
		ours += us
		ref := byKind[s.Kind]
		fmt.Printf("  %-6s %6d %9.1f ms %6d %9.1f ms  %6.2fx\n",
			s.Kind, disp[s.Kind], ref/1e3, dnLayersPerGraph, us/1e3, ref/us)
	}
	fmt.Printf("  %-6s %6d %9.1f ms %6d %9.1f ms  %6.2fx\n",
		"layer", refDisp, refTotal/1e3, dnLayersPerGraph*len(st), ours/1e3, refTotal/ours)
	fmt.Printf("  %.1f%% of llama.cpp's %.0f ms graph becomes %.1f%%, a %.1f ms saving\n",
		100*refTotal/llamaGraphUs, llamaGraphUs/1e3, 100*ours/llamaGraphUs, (refTotal-ours)/1e3)
	fmt.Println("  — the SiLU the convolution carries, the sigmoid of z, the multiply against")
	fmt.Println("    it and the CONTs behind all three are inside MUL/SIGMOID/CONT, which")
	fmt.Println("    every other block shares; none of it is attributed and all of it is")
	fmt.Println("    absent here rather than faster.")
}

// dnRates reports what a dispatch achieved: arithmetic for the three matmul
// shapes, bandwidth for the two passes that only move activations, and both
// for the scan — whose whole ladder is a trade between them.
func dnRates(kind string, c llm.DeltaNetConfig, tok, reread int, d time.Duration) (gflops, gbps float64) {
	if d <= 0 {
		return 0, 0
	}
	s := d.Seconds()
	t := float64(tok)
	embd := float64(c.NEmbd)
	cw := float64(c.ConvWidth())
	inner := float64(c.Inner)
	hd := float64(c.HeadDim)
	hv := float64(c.NHeadV)
	qkvN := cw + inner + 2*hv
	switch kind {
	case "qkv":
		return 2 * t * qkvN * embd / s / 1e9, (qkvN*embd*2 + t*embd*2) / s / 1e9
	case "conv":
		// Four taps out of the projection's output and one row back in. The
		// taps overlap, so the honest count is one read and one write.
		return 0, 2 * t * cw * 4 / s / 1e9
	case "scan":
		// Per state element per token: g*s and the kv accumulation, then
		// g*s + k*delta and the q accumulation. Eight flops, counting an
		// fma as two.
		flops := 8 * hv * t * hd * hd
		// What it moves: q and k re-read once per workgroup of the head, the
		// value and the output once each, and the state once each way.
		bytes := float64(reread)*hv*t*2*hd*4 + 2*hv*t*hd*4 + 2*hv*hd*hd*4
		return flops / s / 1e9, bytes / s / 1e9
	case "norm":
		return 0, (2*t*inner*4 + t*inner*2) / s / 1e9
	case "out":
		return 2 * t * embd * inner / s / 1e9, (embd*inner*2 + t*inner*2) / s / 1e9
	}
	return 0, 0
}
