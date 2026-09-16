package main

// The hyper-connection block's benchmark: LLM.md L2c, against the per-op
// attribution of llama.cpp's own graph that L2a produced.
//
//	go run ./cmd/llm -hc                       # the default ladder, 64..2048
//	go run ./cmd/llm -hc -tokens 512 -ladder   # every rung at the reference's ubatch
//	go run ./cmd/llm -hc -csv results/l2c_hc.csv
//
// What is timed is GPU dispatch time, swept across every staged mixer so the
// weights are read as cold as the real graph reads them — one mixer is 13.4 MB
// and would otherwise sit in the 32 MiB MALL. Wall clock around a run is not a
// measurement of the block: it carries the upload and the read-back, and this
// arena reads at 0.2 GB/s.

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/vk"
)

// llamaOp is one line of llama.cpp's prefill graph at `-ub 512`, from
// results/l2a_prefill_ops.csv — its best ubatch, so the right baseline to
// quote. Totals are per graph of 512 tokens, microseconds.
type llamaOp struct {
	kind       string
	op         string
	dispatches int
	usTotal    float64
}

// The four lines of that graph that are nameably this block, plus the one
// piece of glue whose dispatch count identifies it beyond doubt: REPEAT is
// 98 dispatches, which is the mixer count, and it is the combine broadcasting
// one scatter weight per stream across 2560 features.
//
// The rest of the block's glue — the gamma multiply, the gate's sigmoid, the
// xn*gate multiply, the collapse's adds, the combine's scale/sigmoid/mul/add
// — is inside MUL (479), ADD (231), SIGMOID (326) and MULTI_ADD (170), which
// the model's other blocks also use. L2a attributes the whole elementwise
// category at 30.6% of the graph; nothing here needs that attribution to be
// exact, because every one of those dispatches is *absent* from our graph
// rather than faster in it.
var llamaHC = []llamaOp{
	{"norm", "RMS_NORM(2560,4,512,1)", 98, 16707.5},
	{"down", "MUL_MAT q8_0 m=320 n=512 k=10240", 95, 38055.6},
	{"down", "MUL_MAT f32 m=4 n=512 k=10240 (inject)", 95, 125231.0},
	{"up", "MUL_MAT q8_0 m=10240 n=512 k=320", 95, 30726.0},
	{"combine", "REPEAT", 98, 15928.3},
}

// llamaGraphUs is that whole graph, summed over every op line at `-ub 512`.
const llamaGraphUs = 1164730.0

// mixersPerGraph is how many mixers a forward pass runs: two per layer plus
// the one at the head. llama.cpp's dispatch counts are 95 and 98 because the
// head mixer has no inject and its norm is dumped separately; ours is one
// number, and the difference is inside the noise of the comparison.
const mixersPerGraph = 97

func hcBench(model string, tokens []int, mixers, iters int, ladder bool, csvPath string) error {
	m, err := llm.Open(model)
	if err != nil {
		return err
	}
	defer m.Close()
	cfg := m.Config.HCConfig()
	fmt.Printf("%s\n%d layers, n_embd %d, hc %d x %d, low rank %d\n",
		model, m.Config.NLayer, cfg.NEmbd, cfg.HC, cfg.NEmbd, cfg.LowRank)

	// Real weights, from as many mixers as asked for: the sweep needs the
	// staged set to overflow the MALL, and at 13.4 MB a mixer that is three.
	var ws []llm.HCWeights
	for i := 0; len(ws) < mixers; i++ {
		for _, side := range []string{"attn", "ffn"} {
			if len(ws) == mixers {
				break
			}
			w, err := m.HCWeights(i/2, side)
			if err != nil {
				return fmt.Errorf("mixer %d: %w", len(ws), err)
			}
			ws = append(ws, w)
		}
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
	g, err := llm.NewHCGPU(dev, cfg, maxTok, ws, llm.HCOpts{})
	if err != nil {
		return err
	}
	defer g.Destroy()
	fmt.Printf("%d mixers staged: %.1f MB of weights, %.1f MB of arenas for %d tokens\n\n",
		g.Mixers(), float64(g.WeightBytes())/1e6, float64(g.ActivationBytes())/1e6, maxTok)

	// Nil means the measured schedule, llm.PlanFor, chosen per length.
	// Nil means the measured schedule; a ladder is built per length, because
	// the GEMV rungs only answer at one token (L7d).
	ladderFor := func(tok int) [][2]llm.HCKernel {
		var out [][2]llm.HCKernel
		for _, d := range llm.DownKernelsAt(tok) {
			for _, u := range llm.UpKernels() {
				out = append(out, [2]llm.HCKernel{d, u})
			}
		}
		return out
	}

	var rows [][]string
	rows = append(rows, []string{"tokens", "down_kernel", "up_kernel", "dispatch", "us_per_mixer",
		"us_per_graph", "gflops", "gbps"})
	for _, tok := range tokens {
		res := make([]float32, tok*cfg.Wide())
		for i := range res {
			// Anything with a realistic magnitude: this measures dispatch
			// time, and the only value-dependent cost here is a denormal.
			res[i] = float32((i%97)-48) * 0.01
		}
		if err := g.Upload(res, tok); err != nil {
			return err
		}
		if err := g.UploadBlockOut(make([]float32, tok*cfg.NEmbd)); err != nil {
			return err
		}
		var todo [][2]llm.HCKernel
		if ladder {
			todo = ladderFor(tok)
		} else {
			d, u := llm.PlanFor(tok)
			todo = [][2]llm.HCKernel{{d, u}}
		}
		for _, p := range todo {
			if err := g.SetPlan(p[0], p[1]); err != nil {
				return err
			}
			st, err := g.ProfileSweep(true, iters)
			if err != nil {
				return err
			}
			fmt.Printf("T = %-5d %s / %s\n", tok, p[0], p[1])
			var total time.Duration
			for _, s := range st {
				total += s.GPU
				us := float64(s.GPU.Nanoseconds()) / 1e3
				gf, gb := hcRates(s.Kind, cfg, tok, s.GPU)
				fmt.Printf("  %-8s %8.1f us  x%d = %7.1f ms", s.Kind, us, mixersPerGraph,
					us*mixersPerGraph/1e3)
				if gf > 0 {
					fmt.Printf("  %8.0f GFLOP/s", gf)
				}
				if gb > 0 {
					fmt.Printf("  %6.0f GB/s", gb)
				}
				fmt.Println()
				rows = append(rows, []string{
					strconv.Itoa(tok), string(p[0]), string(p[1]), s.Kind,
					fmt.Sprintf("%.3f", us), fmt.Sprintf("%.1f", us*mixersPerGraph/1e3),
					fmt.Sprintf("%.1f", gf), fmt.Sprintf("%.1f", gb),
				})
			}
			us := float64(total.Nanoseconds()) / 1e3
			fmt.Printf("  %-8s %8.1f us  x%d = %7.1f ms\n", "block", us, mixersPerGraph,
				us*mixersPerGraph/1e3)
			rows = append(rows, []string{
				strconv.Itoa(tok), string(p[0]), string(p[1]), "block",
				fmt.Sprintf("%.3f", us), fmt.Sprintf("%.1f", us*mixersPerGraph/1e3), "", "",
			})
			if tok == 512 && !ladder {
				reportAgainstLlama(st, total)
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

// reportAgainstLlama puts the measured block beside the lines of llama.cpp's
// own graph it replaces. Both sides are one prefill graph of 512 tokens.
func reportAgainstLlama(st []llm.Stage, total time.Duration) {
	byKind := map[string]float64{}
	for _, op := range llamaHC {
		byKind[op.kind] += op.usTotal
	}
	var refTotal float64
	for _, v := range byKind {
		refTotal += v
	}
	fmt.Println("\n  against llama.cpp's own graph at -ub 512 (results/l2a_prefill_ops.csv):")
	fmt.Printf("  %-8s %12s %12s %8s\n", "", "llama.cpp", "ours", "")
	var ours float64
	for _, s := range st {
		us := float64(s.GPU.Nanoseconds()) / 1e3 * mixersPerGraph
		ours += us
		ref := byKind[s.Kind]
		fmt.Printf("  %-8s %9.1f ms %9.1f ms  %6.2fx\n", s.Kind, ref/1e3, us/1e3, ref/us)
	}
	fmt.Printf("  %-8s %9.1f ms %9.1f ms  %6.2fx\n", "block", refTotal/1e3, ours/1e3, refTotal/ours)
	fmt.Printf("  %.1f%% of llama.cpp's %.0f ms graph becomes %.1f%%, a %.1f ms saving\n",
		100*refTotal/llamaGraphUs, llamaGraphUs/1e3, 100*ours/llamaGraphUs, (refTotal-ours)/1e3)
	fmt.Println("  — and the glue those lines do not count (the gamma multiply, the gate's")
	fmt.Println("    sigmoid, the xn*gate multiply, the collapse's adds, the combine's four")
	fmt.Println("    elementwise passes) is absent from our graph rather than faster in it.")
}

// hcRates reports what a dispatch achieved: arithmetic for the two matmuls,
// bandwidth for the two passes that only move the residual.
func hcRates(kind string, c llm.HCConfig, tok int, d time.Duration) (gflops, gbps float64) {
	if d <= 0 {
		return 0, 0
	}
	s := d.Seconds()
	t, wide, lr := float64(tok), float64(c.Wide()), float64(c.LowRank)
	switch kind {
	case "norm":
		// The wide residual in fp32 and back out as fp16.
		return 0, (t*wide*4 + t*wide*2) / s / 1e9
	case "down":
		n := lr + float64(coopTile)
		return 2 * t * n * wide / s / 1e9, (n*wide*2 + t*wide*2) / s / 1e9
	case "up":
		return 2 * t * wide * lr / s / 1e9, (wide*lr*2 + t*wide*2) / s / 1e9
	case "combine":
		// Read the residual, read the block output, write the residual.
		return 0, (2*t*wide*4 + t*wide/4*4) / s / 1e9
	}
	return 0, 0
}

// coopTile is the fragment extent the inject columns are padded up to.
const coopTile = 16

func openDevice() (*vk.Device, func(), error) {
	inst, err := vk.NewInstance("llm")
	if err != nil {
		return nil, nil, err
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		return nil, nil, fmt.Errorf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == 0x1586 { // Strix Halo
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		return nil, nil, err
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		return nil, nil, err
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		inst.Destroy()
		return nil, nil, err
	}
	fmt.Printf("device: %s\n", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }, nil
}

func writeCSV(path string, rows [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if err := w.WriteAll(rows); err != nil {
		return err
	}
	w.Flush()
	return w.Error()
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		n, err := strconv.Atoi(f)
		if err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no token counts in %q", s)
	}
	return out, nil
}

// pleOps are the lines of llama.cpp's 512-token prefill graph that are the
// PLE block, from results/l2a_prefill_ops.csv. Microseconds per graph.
//
// It runs once, at layer 1, so it is the cheapest block in the model to get
// wrong and the least worth tuning: the key projection is one dispatch of the
// 37 that share `MUL_MAT q8_0 m=10240 n=512 k=2560`, and the value one of the
// 24 at `m=512 n=512 k=2560`. The per-dispatch means are what those lines
// cost, and they are quoted here as one dispatch each rather than as the line
// total, which is the whole point.
var pleOps = []llamaOp{
	{"kv", "MUL_MAT q8_0 m=10240 n=512 k=2560 (ple_key, 1 of 37)", 1, 1346.8},
	{"kv", "MUL_MAT q8_0 m=512 n=512 k=2560 (ple_value, 1 of 24)", 1, 115.8},
	{"gate", "RMS_NORM(2560,4,512) x3 (1 of 98 each)", 3, 3 * 170.5},
}

func pleBench(model string, tokens []int, iters int, csvPath string) error {
	m, err := llm.Open(model)
	if err != nil {
		return err
	}
	defer m.Close()
	cfg, ok, err := m.PLEConfig()
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("%s states no PLE module", model)
	}
	layer := cfg.Layers[0]
	w, err := m.PLEWeights(layer)
	if err != nil {
		return err
	}
	fmt.Printf("%s\nPLE at layer %d: %d-gram, %d heads of %d, conv %d dilated by %d\n",
		model, layer, cfg.NGram, cfg.NHeads, cfg.HeadDim, cfg.Conv, cfg.NGram)

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()
	maxTok := 0
	for _, t := range tokens {
		maxTok = max(maxTok, t)
	}
	g, err := llm.NewPLEGPU(dev, cfg, maxTok, w, llm.PLEOpts{})
	if err != nil {
		return err
	}
	defer g.Destroy()
	fmt.Printf("staged: %.1f MB of weights, %.1f MB of arenas for %d tokens\n\n",
		float64(g.WeightBytes())/1e6, float64(g.ActivationBytes())/1e6, maxTok)

	rows := [][]string{{"tokens", "dispatch", "kernel", "us", "scheduled"}}
	for _, tok := range tokens {
		res := make([]float32, tok*cfg.Wide())
		embd := make([]float32, tok*cfg.EmbdWidth())
		for i := range res {
			res[i] = float32((i%97)-48) * 0.01
		}
		for i := range embd {
			embd[i] = float32((i%61)-30) * 0.001
		}
		if err := g.Upload(res, embd, tok); err != nil {
			return err
		}
		var total time.Duration
		fmt.Printf("T = %-5d\n", tok)
		// The fused projection's rung is swept: its B is 65.5 MB against a
		// 32 MiB MALL, so what BM costs is the weight read M/BM times.
		for _, k := range llm.PLEKernels() {
			if err := g.SetKernel(k); err != nil {
				return err
			}
			st, err := g.Profile(iters)
			if err != nil {
				return err
			}
			sched := k == llm.PLEKernelFor(tok)
			for i, s := range st {
				us := float64(s.GPU.Nanoseconds()) / 1e3
				// Only the fused projection has a ladder; the other two
				// dispatches are the same kernel under every rung, so they are
				// recorded once, under the scheduled one.
				if i > 0 && !sched {
					continue
				}
				if sched {
					total += s.GPU
					fmt.Printf("  %-6s %8.1f us\n", s.Kind, us)
				} else {
					fmt.Printf("  %-6s %8.1f us  (%s)\n", s.Kind, us, k)
				}
				rows = append(rows, []string{strconv.Itoa(tok), s.Kind, string(k),
					fmt.Sprintf("%.3f", us), fmt.Sprintf("%t", sched)})
			}
		}
		fmt.Printf("  %-6s %8.1f us\n", "block", float64(total.Nanoseconds())/1e3)
		rows = append(rows, []string{strconv.Itoa(tok), "block", string(llm.PLEKernelFor(tok)),
			fmt.Sprintf("%.3f", float64(total.Nanoseconds())/1e3), "true"})
		if tok == 512 {
			var ref float64
			for _, op := range pleOps {
				ref += op.usTotal
			}
			ours := float64(total.Nanoseconds()) / 1e3
			fmt.Printf("\n  against the same five lines of llama.cpp's -ub 512 graph: %.1f us -> %.1f us, %.2fx\n",
				ref, ours, ref/ours)
			fmt.Printf("  which is %.2f%% of its %.0f ms graph, against %.2f%%\n\n",
				100*ref/llamaGraphUs, llamaGraphUs/1e3, 100*ours/llamaGraphUs)
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
