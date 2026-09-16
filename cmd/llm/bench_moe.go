package main

// The MoE block's benchmark: LLM.md L5b, against the per-op attribution of
// llama.cpp's own graph that L2a produced.
//
//	go run ./cmd/llm -moe                      # the default sweep, 64..4096
//	go run ./cmd/llm -moe -tokens 512 -ladder  # every row block at the reference's ubatch
//	go run ./cmd/llm -moe -csv results/l5b_moe.csv
//
// **The input is llama.cpp's own, and that is not a convenience.** Everything
// this kernel costs depends on the routing: how much of the 1.57 GB bank a
// run reads is how many experts the prompt touches, and how much of each tile
// is padding is how the rows fall across them. L5a-4 measured the real
// distribution — 274 of 512 experts at a 512-token ubatch, one of them taking
// 95% of the tokens — and a synthetic activation routes very nearly uniformly,
// which reads *twice* the bank and hides the imbalance the schedule exists
// for. So the sweep reads `hc_mixed-3` out of the 4 k trace and takes its
// first T rows; without the trace it falls back to synthetic input and says
// so, and the numbers it then prints are not the ones to quote.

import (
	"fmt"
	"strconv"
	"time"

	"strix-halo-vulkan/llm"
)

// The lines of llama.cpp's 512-token prefill graph at `-ub 512` that are
// nameably the MoE block, from results/l2a_prefill_ops.csv. Every one of the
// 48 layers has one of these blocks, so a line with 48 dispatches is one per
// layer and 96 is two.
//
// Three attributions are worth stating rather than trusting:
//
//   - The routed matmuls are split by **format**: gate and up are Q4_K on 46
//     layers and Q5_K on layer 2; down is Q5_1 on 43 and Q8_0 on 4. All of
//     them are one op — `MUL_MAT_ID` — and the whole of each line is taken.
//   - `MUL_MAT q8_0 m=640 n=512 k=2560` (94) and `m=2560 n=512 k=640` (47)
//     are the **shared** expert's three matmuls, which are dense because it
//     runs for every token. Nothing else in the model has those shapes.
//   - `MUL_MAT f32 m=1 n=512 k=2560` is `shared_expert_gate`: 47 dispatches
//     of a [2560, 1] F32 matrix, **9.1 ms a graph at 13.6 GFLOP/s**, which is
//     L2a's `inject` scandal in miniature and which this kernel deletes by
//     making it a 513th column of the router.
//
// `GLU` is unambiguous — 96 dispatches, two per layer, `ggml_swiglu_split`
// over the routed half and the shared one — and so are ARGSORT, SUM_ROWS,
// CLAMP and DIV, which are the routing's tail at one per layer. What is
// *not* attributed is the MUL that applies the ten weights, the MUL that
// applies the shared gate, the SIGMOID behind it and the ADD that sums the
// two halves: all of those live in the shared MUL (479), SIGMOID (326) and
// ADD (231) lines that every other block of the model also uses. Every one of
// those dispatches is **absent** from our graph rather than faster in it —
// the weights and the gate are applied in a store, and the sum is the
// combine — so leaving them out is the conservative direction.
var llamaMoE = []llamaOp{
	{"up", "MUL_MAT_ID q4_K m=640 k=2560 (gate and up, 46 layers)", 92, 352733.0},
	{"up", "MUL_MAT_ID q5_K m=640 k=2560 (gate and up, layer 2)", 2, 12584.4},
	{"down", "MUL_MAT_ID q5_1 m=2560 k=640 (43 layers)", 43, 142510.0},
	{"down", "MUL_MAT_ID q8_0 m=2560 k=640 (4 layers)", 4, 16394.7},
	{"up", "GLU (ggml_swiglu_split, both halves)", 96, 7011.5},
	{"shexp.up", "MUL_MAT q8_0 m=640 n=512 k=2560 (shared gate and up)", 94, 12299.7},
	{"shexp.down", "MUL_MAT q8_0 m=2560 n=512 k=640 (shared down)", 47, 4139.2},
	{"router", "MUL_MAT f32 m=512 n=512 k=2560 (ffn_gate_inp)", 47, 6104.7},
	{"router", "MUL_MAT f32 m=1 n=512 k=2560 (shared_expert_gate)", 47, 9060.1},
	{"route", "ARGSORT", 47, 1775.2},
	{"route", "TOPK_MOE_EARLY_SOFTMAX_NORM SOFT_MAX", 48, 378.8},
	{"route", "SUM_ROWS (weights_sum)", 48, 249.2},
	{"route", "CLAMP", 48, 115.0},
	{"route", "DIV (the normalise)", 47, 112.7},
	{"perm.up", "GET_ROWS (routing_gather)", 135, 1965.3},
}

// The same lines at `-ub 2048`. L2a-4 established that 512 is llama.cpp's
// best ubatch and so the baseline to quote — but the MoE is the one block
// that gets *cheaper* per token as the chunk grows on both sides, and the
// glue that makes 2048 lose is somebody else's, so the pair is worth having.
var llamaMoE2048 = []llamaOp{
	{"up", "MUL_MAT_ID q4_K m=640 k=2560 (gate and up, 46 layers)", 92, 1069360.0},
	{"up", "MUL_MAT_ID q5_K m=640 k=2560 (gate and up, layer 2)", 2, 32384.0},
	{"down", "MUL_MAT_ID q5_1 m=2560 k=640 (43 layers)", 43, 475347.0},
	{"down", "MUL_MAT_ID q8_0 m=2560 k=640 (4 layers)", 4, 48828.6},
	{"up", "GLU (ggml_swiglu_split, both halves)", 96, 33322.5},
	{"shexp.up", "MUL_MAT q8_0 m=640 n=2048 k=2560 (shared gate and up)", 94, 45994.2},
	{"shexp.down", "MUL_MAT q8_0 m=2560 n=2048 k=640 (shared down)", 47, 15082.2},
	{"router", "MUL_MAT f32 m=512 n=2048 k=2560 (ffn_gate_inp)", 47, 14873.3},
	{"router", "MUL_MAT f32 m=1 n=2048 k=2560 (shared_expert_gate)", 47, 11043.8},
	{"route", "ARGSORT", 47, 6030.1},
	{"route", "TOPK_MOE_EARLY_SOFTMAX_NORM SOFT_MAX", 48, 1213.9},
	{"route", "SUM_ROWS (weights_sum)", 48, 684.3},
	{"route", "CLAMP", 48, 121.3},
	{"route", "DIV (the normalise)", 47, 119.1},
	{"perm.up", "GET_ROWS (routing_gather)", 135, 4036.9},
}

// The whole graph at each of those two ubatches, summed over every op line.
const llamaGraph2048Us = 4559130.0

// moeLayersPerGraph is how many of these blocks a forward pass runs: all 48.
const moeLayersPerGraph = 48

// moeBenchLayer is which layer the sweep stages. Layer 3 is the one every
// other test in this vertical uses, and its formats — Q4_K gate and up, Q5_1
// down — are what 43 of the 48 layers have. The five that do not (layer 2's
// Q5_K pair, four Q8_0 downs) are more expensive on the reference's side too,
// by 1.2-1.6x on the lines above.
const moeBenchLayer = 3

// moeBench runs the sweep. `ladder` walks the 3x3 cross of row blocks: the up
// projection gathers BM scattered rows into LDS per K-step and the down
// projection reads BM contiguous ones, so the two sit on opposite sides of
// the question L2f-4 and L3b-7 both found inverting a schedule.
func moeBench(model string, tokens []int, nLayers, iters int, ladder bool, csvPath string) error {
	m, err := llm.Open(model)
	if err != nil {
		return err
	}
	defer m.Close()

	cfg := m.MoEConfig()
	var idx []int
	var ws []llm.MoEWeights
	for l := moeBenchLayer; l < m.Config.NLayer && len(idx) < nLayers; l++ {
		w, err := m.MoEWeights(l)
		if err != nil {
			return fmt.Errorf("layer %d: %w", l, err)
		}
		idx = append(idx, l)
		ws = append(ws, w)
	}
	fmt.Printf("%s\nlayers %v of %d, %d experts of %d wide, %d used, shared %d\n",
		model, idx, m.Config.NLayer, cfg.NExpert, cfg.FFNExpert, cfg.NExpertUsed, cfg.FFNShared)

	x, real, err := moeInput(cfg)
	if err != nil {
		return err
	}
	maxTok := 0
	for _, t := range tokens {
		maxTok = max(maxTok, t)
	}
	if maxTok*cfg.NEmbd > len(x) {
		return fmt.Errorf("the input has %d tokens, the sweep asks for %d", len(x)/cfg.NEmbd, maxTok)
	}

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()

	start := time.Now()
	g, err := llm.NewMoEGPU(dev, cfg, maxTok, ws)
	if err != nil {
		return err
	}
	defer g.Destroy()
	gateF, upF, downF := g.Formats(0)
	fmt.Printf("%d layers staged in %v: %.2f GB of quantised bank (%s/%s gate-up, %s down), %.1f MB of arenas for %d tokens\n",
		g.Layers(), time.Since(start).Round(time.Millisecond),
		float64(g.WeightBytes())/1e9, gateF, upF, downF,
		float64(g.ActivationBytes())/1e6, maxTok)
	if !real {
		fmt.Println("*** synthetic input: the routing is near-uniform, so the bank traffic is")
		fmt.Println("*** roughly twice a real prompt's and the tile imbalance is absent.")
	}
	fmt.Println()

	// The ladder. At one token it is a different cross as well as a longer
	// one: `llm_moe_gemv.comp`'s six rungs are a *second kernel* for the same
	// two modes (L8d), they may be mixed with the GEMM's because every rung
	// cuts its tile list to the same sixteen rows, and they are refused above
	// one token — so the two crosses are walked separately and then the six
	// mixtures that pair each kernel's best with the other's.
	type moePlan struct {
		up, down llm.MoEKernel
		router   llm.MoERouterKernel
	}
	plansFor := func(tok int) []moePlan {
		if !ladder {
			return []moePlan{{}}
		}
		var p []moePlan
		for _, u := range llm.MoEUpKernels() {
			for _, d := range llm.MoEKernels() {
				p = append(p, moePlan{u, d, ""})
			}
		}
		if tok == 1 {
			for _, u := range llm.MoEDecodeKernels() {
				for _, d := range llm.MoEDecodeKernels() {
					p = append(p, moePlan{u, d, ""})
				}
			}
			for _, u := range llm.MoEDecodeKernels() {
				p = append(p, moePlan{u, llm.MoEM1, ""})
			}
			for _, d := range llm.MoEDecodeKernels() {
				p = append(p, moePlan{llm.MoEN1M1, d, ""})
			}
			// And the router's own ladder, which does not cross the experts'
			// — it is a different matrix and a different kernel (L8d-3), so
			// it is walked once against the plan the experts settle on.
			du, dd := llm.MoEPlanFor(1)
			for _, r := range append([]llm.MoERouterKernel{llm.MoERouterGEMM}, llm.MoERouterKernels()...) {
				p = append(p, moePlan{du, dd, r})
			}
		}
		return p
	}

	var rows [][]string
	rows = append(rows, []string{"tokens", "up_kernel", "down_kernel", "experts_touched",
		"up_tiles", "up_rows_exec", "down_tiles", "down_rows_exec",
		"dispatch", "us_per_layer", "ms_per_graph", "gflops", "weight_gbps"})
	for _, tok := range tokens {
		if err := g.Upload(x[:tok*cfg.NEmbd], tok); err != nil {
			return err
		}
		for _, p := range plansFor(tok) {
			if p.up != "" {
				if err := g.SetPlan(p.up, p.down); err != nil {
					return err
				}
			}
			if p.router != "" {
				if err := g.SetRouter(p.router); err != nil {
					return err
				}
			}
			up, down := g.Plan()
			// The schedule has to exist before it can be reported, and it is
			// the device that builds it.
			if err := g.Run(0); err != nil {
				return err
			}
			touched := g.Touched()
			upTiles, upExec, realRows := g.Schedule(up)
			dnTiles, dnExec, _ := g.Schedule(down)
			fmt.Printf("T = %-5d %s/%s/%s (%d and %d waves a workgroup)  %d of %d experts touched, %d rows: %d up tiles (%d rows, %.2fx), %d down tiles (%d rows, %.2fx)\n",
				tok, up, down, g.Router(), llm.MoEWaves(up), llm.MoEWaves(down), touched, cfg.NExpert, realRows,
				upTiles, upExec, float64(upExec)/float64(realRows),
				dnTiles, dnExec, float64(dnExec)/float64(realRows))

			st, err := g.ProfileSweep(iters)
			if err != nil {
				return err
			}
			var total time.Duration
			for _, s := range st {
				total += s.GPU
				us := float64(s.GPU.Nanoseconds()) / 1e3
				gf, gb := moeRates(g, s.Kind, cfg, tok, touched, upExec, dnExec, s.GPU)
				fmt.Printf("  %-10s %9.1f us  x%d = %7.1f ms", s.Kind, us, moeLayersPerGraph,
					us*moeLayersPerGraph/1e3)
				if gf > 0 {
					fmt.Printf("  %8.0f GFLOP/s", gf)
				}
				if gb > 0 {
					fmt.Printf("  %6.0f GB/s of bank", gb)
				}
				fmt.Println()
				rows = append(rows, []string{
					strconv.Itoa(tok), string(up), string(down) + "/" + string(g.Router()), strconv.Itoa(touched),
					strconv.Itoa(upTiles), strconv.Itoa(upExec), strconv.Itoa(dnTiles), strconv.Itoa(dnExec),
					s.Kind, fmt.Sprintf("%.3f", us),
					fmt.Sprintf("%.1f", us*moeLayersPerGraph/1e3),
					fmt.Sprintf("%.1f", gf), fmt.Sprintf("%.1f", gb),
				})
			}
			us := float64(total.Nanoseconds()) / 1e3
			fmt.Printf("  %-10s %9.1f us  x%d = %7.1f ms\n", "block", us, moeLayersPerGraph,
				us*moeLayersPerGraph/1e3)
			rows = append(rows, []string{
				strconv.Itoa(tok), string(up), string(down) + "/" + string(g.Router()), strconv.Itoa(touched),
				strconv.Itoa(upTiles), strconv.Itoa(upExec), strconv.Itoa(dnTiles), strconv.Itoa(dnExec),
				"block", fmt.Sprintf("%.3f", us),
				fmt.Sprintf("%.1f", us*moeLayersPerGraph/1e3), "", "",
			})
			if !ladder && tok == 512 {
				reportMoEAgainstLlama(st, llamaMoE, 512, llamaGraphUs)
			}
			if !ladder && tok == 2048 {
				reportMoEAgainstLlama(st, llamaMoE2048, 2048, llamaGraph2048Us)
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

// moeInput returns the block's input. It prefers llama.cpp's own — the 4 k
// trace's `hc_mixed-3`, written the **second** time in the layer, which is
// the FFN mixer's output and not the attention mixer's — because the routing
// a synthetic activation produces is not the routing this kernel is
// scheduled against.
func moeInput(cfg llm.MoEConfig) ([]float32, bool, error) {
	tr, err := llm.OpenTrace("reference/out/llm4k")
	if err == nil {
		if in, err := tr.Get("hc_mixed-3", 1); err == nil {
			return in.Vals, true, nil
		}
	}
	x := make([]float32, 4096*cfg.NEmbd)
	for i := range x {
		x[i] = float32((i%97)-48) * 0.01
	}
	return x, false, nil
}

// reportMoEAgainstLlama puts the measured block beside the lines of
// llama.cpp's own graph it replaces. Both sides are one prefill graph of 512
// tokens.
func reportMoEAgainstLlama(st []llm.Stage, ops []llamaOp, ubatch int, graphUs float64) {
	byKind := map[string]float64{}
	disp := map[string]int{}
	var refTotal float64
	var refDisp int
	for _, op := range ops {
		byKind[op.kind] += op.usTotal
		disp[op.kind] += op.dispatches
		refTotal += op.usTotal
		refDisp += op.dispatches
	}
	fmt.Printf("\n  against llama.cpp's own graph at -ub %d (results/l2a_prefill_ops.csv):\n", ubatch)
	fmt.Printf("  %-10s %6s %12s %6s %12s %8s\n", "", "disp", "llama.cpp", "disp", "ours", "")
	var ours float64
	for _, s := range st {
		us := float64(s.GPU.Nanoseconds()) / 1e3 * moeLayersPerGraph
		ours += us
		ref := byKind[s.Kind]
		x := ""
		if ref > 0 {
			x = fmt.Sprintf("%6.2fx", ref/us)
		}
		fmt.Printf("  %-10s %6d %9.1f ms %6d %9.1f ms  %s\n",
			s.Kind, disp[s.Kind], ref/1e3, moeLayersPerGraph, us/1e3, x)
	}
	fmt.Printf("  %-10s %6d %9.1f ms %6d %9.1f ms  %6.2fx\n",
		"block", refDisp, refTotal/1e3, moeLayersPerGraph*len(st), ours/1e3, refTotal/ours)
	fmt.Printf("  %.1f%% of llama.cpp's %.0f ms graph becomes %.1f%%, a %.1f ms saving\n",
		100*refTotal/graphUs, graphUs/1e3, 100*ours/graphUs, (refTotal-ours)/1e3)
	fmt.Println("  — the ten weights' MUL, the shared gate's SIGMOID and MUL and the ADD that")
	fmt.Println("    sums the two halves are inside MUL/SIGMOID/ADD, which every other block")
	fmt.Println("    shares; none of it is attributed and all of it is absent here rather than")
	fmt.Println("    faster — the weights ride a store and the sum is the combine.")
	fmt.Println("  — our layer is the Q4_K/Q5_1 pair that 43 of the 48 layers have; the five")
	fmt.Println("    that differ are 1.2-1.6x more expensive on the reference's side as well.")
}

// moeRates reports what a dispatch achieved: arithmetic over the rows it
// actually executed (padding included, because it is executed), and the rate
// it pulled the **quantised bank** at, which is the traffic that cannot be
// avoided. The A operand's re-read across column blocks is deliberately not
// counted in the bandwidth figure — it is a schedule cost, not a floor, and
// counting it would flatter a kernel that reads its weights badly.
func moeRates(g *llm.MoEGPU, kind string, c llm.MoEConfig, tok, touched, upExec, dnExec int, d time.Duration) (gflops, gbps float64) {
	if d <= 0 {
		return 0, 0
	}
	s := d.Seconds()
	embd, ff := float64(c.NEmbd), float64(c.FFNExpert)
	upPair, down := g.ExpertBytes(0)
	shUpPair, shDown := g.SharedBytes(0)
	switch kind {
	case "router":
		n := float64(c.NExpert + 1)
		return 2 * float64(tok) * n * embd / s / 1e9,
			(n*embd*2 + float64(tok)*embd*2) / s / 1e9
	case "up":
		return 2 * float64(upExec) * ff * embd * 2 / s / 1e9,
			float64(touched) * float64(upPair) / s / 1e9
	case "down":
		return 2 * float64(dnExec) * embd * ff / s / 1e9,
			float64(touched) * float64(down) / s / 1e9
	case "shexp.up":
		return 2 * float64(tok) * ff * embd * 2 / s / 1e9, float64(shUpPair) / s / 1e9
	case "shexp.down":
		return 2 * float64(tok) * embd * ff / s / 1e9, float64(shDown) / s / 1e9
	case "combine":
		slots := float64(c.NExpertUsed + 1)
		return 0, float64(tok) * embd * 4 * (slots + 1) / s / 1e9
	}
	return 0, 0
}
