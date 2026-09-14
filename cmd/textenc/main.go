// Command textenc runs Z-Image's text encoder: a prompt in, the hidden state
// the DiT's cap_embedder consumes out.
//
// It is the stage-5 slice, on either implementation. `-gpu` runs the Vulkan
// encoder (stage 5c) and everything else the CPU reference (stage 5b), which
// is the oracle it was debugged against.
//
// What it measures is a model on the *other* half of the roofline from the
// DiT. At T tokens a projection here is T flop per byte of weight against
// this device's 235 crossover (research/3.4-model-shapes.md), so a prompt of
// 8-512 tokens is memory-bound: the floor is the 7.06 GB of fp16 weights read
// once, whatever T is, and the interesting number is what fraction of the bus
// a run reaches rather than what fraction of the matrix cores.
//
//	go run ./cmd/textenc -gpu -prompt "a cat on a windowsill"
//	go run ./cmd/textenc -gpu -ladder -tokens 24,105,512
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

const strixHaloDeviceID = 0x1586

func main() {
	log.SetFlags(0)
	encDir := flag.String("encoder", "models/Z-Image-Turbo/text_encoder", "text encoder checkpoint")
	tokDir := flag.String("tokenizer", "models/Z-Image-Turbo/tokenizer", "tokenizer directory")
	prompt := flag.String("prompt", "a cat sitting on a windowsill at golden hour, 85mm lens", "prompt")
	layers := flag.Int("layers", 0, "decoder layers to run; 0 is what the pipeline uses")
	reps := flag.Int("reps", 3, "timed repetitions; the best is reported")
	gpu := flag.Bool("gpu", false, "run the Vulkan encoder instead of the CPU reference")
	ladder := flag.Bool("ladder", false, "sweep the GEMM ladder at every -tokens length")
	tokens := flag.String("tokens", "", "comma-separated sequence lengths to time instead of the prompt's own")
	verbose := flag.Bool("v", false, "print the per-dispatch profile")
	flag.Parse()

	tok, err := tokenizer.Load(*tokDir)
	must(err)
	ids, err := tok.EncodePrompt(*prompt)
	must(err)

	cfg, err := qwen.LoadConfig(*encDir)
	must(err)
	n := *layers
	if n == 0 {
		n = cfg.EncoderLayers()
	}
	lengths := []int{len(ids)}
	if *tokens != "" {
		lengths = parseLengths(*tokens)
	}
	maxLen := 0
	for _, l := range lengths {
		maxLen = max(maxLen, l)
	}

	fmt.Printf("prompt   %q\n", *prompt)
	fmt.Printf("tokens   %d, including the chat template's 8\n", len(ids))
	fmt.Printf("encoder  Qwen3, %d layers, hidden %d, %d/%d heads of %d, ffn %d\n",
		cfg.NumLayers, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateSize)
	fmt.Printf("running  %d layers -- hidden_states[-2], no final norm, no lm_head\n", n)

	if *gpu {
		runGPU(*encDir, cfg, n, ids, lengths, maxLen, *reps, *ladder, *verbose)
		return
	}
	runCPU(*encDir, cfg, n, ids, lengths, *reps, *verbose, tok)
}

// runCPU is the fp32 reference: the oracle, and the thing stage 5c has to
// beat by two orders of magnitude for the pipeline to be worth having.
func runCPU(dir string, cfg *qwen.Config, layers int, ids []int32, lengths []int, reps int, verbose bool, tok *tokenizer.Tokenizer) {
	start := time.Now()
	model, err := qwen.Load(dir, layers)
	must(err)
	fmt.Printf("load     %v (fp32 on the host)\n", time.Since(start).Round(time.Millisecond))

	for _, T := range lengths {
		run := fit(ids, T)
		best := time.Duration(math.MaxInt64)
		var out *qwen.Mat
		for i := 0; i < reps; i++ {
			start = time.Now()
			out, err = model.Forward(run, nil)
			must(err)
			best = min(best, time.Since(start))
		}
		report("cpu", cfg, layers, len(run), best, out)
	}
	_ = verbose
	_ = tok
}

// runGPU builds the Vulkan encoder once and times it. The weights are staged
// in the layout every rung reads, so the ladder re-plans rather than
// reloading: one 7 GB upload for the whole sweep.
func runGPU(dir string, cfg *qwen.Config, layers int, ids []int32, lengths []int, maxLen, reps int, ladder, verbose bool) {
	inst, err := vk.NewInstance("textenc")
	must(err)
	defer inst.Destroy()
	devices, err := inst.PhysicalDevices()
	must(err)
	if len(devices) == 0 {
		log.Fatal("no Vulkan devices")
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	must(err)
	sgs, err := phys.SubgroupSizeControl()
	must(err)
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	must(err)
	defer dev.Destroy()
	fmt.Printf("device   %s\n", phys.Name)

	start := time.Now()
	set, err := safetensors.OpenSet(dir)
	must(err)
	g, err := qwen.NewGPUEncoder(dev, set, cfg, layers, maxLen, nil)
	set.Close()
	must(err)
	defer g.Destroy()
	load := time.Since(start)

	bytes, _ := g.Banks()
	var bankMB []string
	for _, b := range bytes {
		bankMB = append(bankMB, strconv.Itoa(b>>20))
	}
	fmt.Printf("load     %v -- %.2f GB of fp16 weights in %d banks (%s MB), %d MB of activations\n",
		load.Round(time.Millisecond), float64(g.WeightBytes())/1e9, len(bytes),
		strings.Join(bankMB, "+"), g.ActivationBytes()>>20)
	fmt.Printf("kernels  %s attention, %d dispatches per layer; projections re-planned per length\n",
		g.Attention(), len(g.Labels()))

	for _, T := range lengths {
		run := fit(ids, T)
		if ladder {
			sweep(g, cfg, layers, run, reps)
			continue
		}
		best := time.Duration(math.MaxInt64)
		var out *qwen.Mat
		for i := 0; i < reps; i++ {
			start = time.Now()
			o, err := g.Forward(run)
			must(err)
			best = min(best, time.Since(start))
			out = o
		}
		report("gpu", cfg, layers, len(run), best, out)
		if verbose {
			profile(g, run, layers)
		}
	}
}

// sweep times every rung of the GEMM ladder at one sequence length. The
// figure that decides it is not TFLOP/s -- at these lengths the kernel is
// memory-bound and most of its flops are on padded rows -- but the fraction
// of the 236 GB/s bus the weight traffic reaches.
func sweep(g *qwen.GPUEncoder, cfg *qwen.Config, layers int, ids []int32, reps int) {
	fmt.Printf("\nladder at %d tokens (best of %d)\n", len(ids), reps)
	fmt.Printf("  %-22s %10s %10s %10s\n", "kernel", "wall", "GB/s", "vs best")
	type row struct {
		name string
		best time.Duration
	}
	var rows []row
	g.AutoPlan = false
	defer func() { g.AutoPlan = true }()
	for _, k := range qwen.GEMMKernels() {
		must(g.SetPlan(qwen.UniformGEMMPlan(k)))
		best := time.Duration(math.MaxInt64)
		for i := 0; i < reps; i++ {
			start := time.Now()
			_, err := g.Forward(ids)
			must(err)
			best = min(best, time.Since(start))
		}
		rows = append(rows, row{string(k), best})
	}
	fastest := rows[0].best
	for _, r := range rows {
		fastest = min(fastest, r.best)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].best < rows[j].best })
	weights := g.WeightReadBytes() * float64(layers)
	for _, r := range rows {
		fmt.Printf("  %-22s %10v %10.1f %9.2fx\n", r.name, r.best.Round(time.Microsecond),
			weights/r.best.Seconds()/1e9, float64(r.best)/float64(fastest))
	}
	must(g.SetPlan(nil))
}

// profile times every dispatch on the GPU and sums by kind. Wall clock around
// Forward also carries the host gather and the read-back, and this arena's
// reads run at 0.2 GB/s (research/stage-3-dit-attention.md), so the two
// numbers answer different questions.
func profile(g *qwen.GPUEncoder, ids []int32, layers int) {
	stages, _, err := g.Profile(ids)
	must(err)
	byKind := map[string]time.Duration{}
	var order []string
	for _, s := range stages {
		if _, seen := byKind[s.Kind]; !seen {
			order = append(order, s.Kind)
		}
		byKind[s.Kind] += s.GPU
	}
	total := qwen.Elapsed(stages)
	fmt.Printf("\n  %d dispatches, %v of GPU time\n", len(stages), total.Round(time.Microsecond))
	sort.SliceStable(order, func(i, j int) bool { return byKind[order[i]] > byKind[order[j]] })
	for _, k := range order {
		fmt.Printf("  %-14s %10v  %5.1f%%\n", k, byKind[k].Round(time.Microsecond),
			100*float64(byKind[k])/float64(total))
	}
}

// report prints one timing, with the two ceilings that bound it: the weights
// read once at 236 GB/s, and the arithmetic at the 55.5 TFLOP/s the matrix
// cores reach. Which one is closer is the whole point of the stage.
func report(where string, cfg *qwen.Config, layers, T int, best time.Duration, out *qwen.Mat) {
	var sum, sumSq, absmax float64
	for _, v := range out.Data {
		sum += float64(v)
		sumSq += float64(v) * float64(v)
		absmax = math.Max(absmax, math.Abs(float64(v)))
	}
	rms := math.Sqrt(sumSq / float64(len(out.Data)))

	perLayer := float64(cfg.HiddenSize) * float64(
		2*cfg.NumHeads*cfg.HeadDim+2*cfg.NumKVHeads*cfg.HeadDim+3*cfg.IntermediateSize)
	params := perLayer * float64(layers)
	flops := 2 * params * float64(T)
	weights := 2 * params

	fmt.Printf("\n%s  %4d tokens  %v   %.1f GFLOP at %.1f GFLOP/s\n",
		where, T, best.Round(time.Microsecond), flops/1e9, flops/best.Seconds()/1e9)
	fmt.Printf("     hidden %s  sum %+.4g  rms %.4g  absmax %.5g\n", out, sum, rms, absmax)
	fmt.Printf("     %.2f GB of fp16 weights: %.1f GB/s, %.0f%% of the 236 GB/s bus; floor %v\n",
		weights/1e9, weights/best.Seconds()/1e9, 100*weights/best.Seconds()/236e9,
		time.Duration(weights/236e9*1e9).Round(time.Microsecond))
}

// fit stretches or truncates a real prompt to a given token count, for the
// timing shapes that have no prompt behind them. The ids are repeated, which
// changes nothing about the work: every length runs the same weights, and a
// causal attention's cost depends on the count and not the content.
func fit(ids []int32, n int) []int32 {
	if n == len(ids) {
		return ids
	}
	out := make([]int32, n)
	for i := range out {
		out[i] = ids[i%len(ids)]
	}
	return out
}

func parseLengths(s string) []int {
	var out []int
	for _, part := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(part))
		must(err)
		if v <= 0 {
			log.Fatalf("token count %d", v)
		}
		out = append(out, v)
	}
	return out
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
