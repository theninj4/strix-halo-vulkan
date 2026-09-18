package main

// Residency: every layer of every block on the device at once (LLM.md L6a).
//
//	go run ./cmd/llm -resident                  # the whole model, 512-token arenas
//	go run ./cmd/llm -resident -tokens 2048
//	go run ./cmd/llm -resident -dense           # the 8 GB half, without the 77
//
// L5b left one thing standing between the five blocks and L6's graph: nothing
// had ever held more than a handful of layers. Every block in this vertical
// lays its weights out as **one arena with a per-layer offset**, and
// `maxStorageBufferRange` on this device is 4 GiB - 4 — so "all 48 layers
// resident" is not a loop over the staging code, it is a question about
// whether the layout can express the model at all. This is the tool that
// answers it by doing it: it stages every mixer, every layer and every expert
// bank, and prints what that cost in bytes, in buffers and in wall clock,
// beside the checkpoint's own inventory and the machine's memory.
//
// The dense blocks are staged **first and their host floats dropped between
// them**, because the peak is not the total: a layer's weights are
// dequantised to f32 on the host before they are packed into halves, and the
// 36 DeltaNet layers are 8.35 GB of them at once. The MoE bank needs no host
// floats at all — it is memcpy'd out of the mmap'd checkpoint — so staging it
// last means the peak is the resident model and not the resident model plus a
// dequantisation.

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/llm"
)

// residentBlock is one row of the plan: what a block type staged, and what it
// cost to stage it.
type residentBlock struct {
	name    string
	layers  int
	buffers int
	weights int
	arenas  int
	staged  time.Duration
}

// residency stages the whole model and reports the plan.
//
// nothing here is timed as a kernel — no dispatch runs — because the question
// is residency: whether 48 layers of five block types fit this device's
// buffers, this machine's memory and this vertical's uint32 offsets at the
// same time.
func residency(model string, maxTok, nKV int, bankLayers []int, denseOnly bool, csvPath string) error {
	m, err := llm.Open(model)
	if err != nil {
		return err
	}
	defer m.Close()
	c := m.Config

	fmt.Printf("%s\n%d layers, n_embd %d, %d experts of %d wide, %d used\n",
		model, c.NLayer, c.NEmbd, c.NExpert, c.FFNExpert, c.NExpertUsed)

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()

	fmt.Printf("arenas for %d tokens in a %d-cell cache\n\n", maxTok, nKV)

	before := memAvailable()
	after := int64(0)
	var plan []residentBlock

	// 1. The hyper-connection mixers: two a layer and the head's, 97 in all.
	//    Every layer runs two of them and the head one more, so this is the
	//    one block whose count is not the layer count.
	start := time.Now()
	var mixers []llm.HCWeights
	for l := 0; l < c.NLayer; l++ {
		for _, side := range []string{"attn", "ffn"} {
			w, err := m.HCWeights(l, side)
			if err != nil {
				return fmt.Errorf("hc layer %d %s: %w", l, side, err)
			}
			mixers = append(mixers, w)
		}
	}
	head, err := m.HCWeights(-1, "")
	if err != nil {
		return fmt.Errorf("hc head mixer: %w", err)
	}
	mixers = append(mixers, head)
	hc, err := llm.NewHCGPU(dev, c.HCConfig(), maxTok, mixers, llm.HCBankOpts())
	if err != nil {
		return fmt.Errorf("hc: %w", err)
	}
	defer hc.Destroy()
	plan = append(plan, residentBlock{"hyper-conn", len(mixers), hc.Buffers(),
		hc.WeightBytes(), hc.ActivationBytes(), time.Since(start)})
	mixers = nil
	drop()

	// 2. The PLE n-gram block. `qwen4exp.ple.layers` is a single entry on this
	//    checkpoint — layer 1 — and the 28.80 GB table it gathers from is D2's
	//    off-heap tensor, which is a quarter of the checkpoint and never
	//    touches the device.
	pleCfg, ok, err := m.PLEConfig()
	if err != nil {
		return fmt.Errorf("ple config: %w", err)
	}
	if ok {
		start = time.Now()
		pw, err := m.PLEWeights(pleCfg.Layers[0])
		if err != nil {
			return fmt.Errorf("ple layer %d: %w", pleCfg.Layers[0], err)
		}
		pleOpts := llm.PLEOpts{Bank: llm.BankFor(llm.DenseQ8()), Layer: pleCfg.Layers[0]}
		if q, ok := llm.DenseBankPlan().For(fmt.Sprintf("blk.%d.ple_key.weight", pleCfg.Layers[0]), true); ok {
			pleOpts.Bank, pleOpts.Sim = llm.BankQ4K, q
		}
		ple, err := llm.NewPLEGPU(dev, pleCfg, maxTok, pw, pleOpts)
		if err != nil {
			return fmt.Errorf("ple: %w", err)
		}
		defer ple.Destroy()
		plan = append(plan, residentBlock{"ple n-gram", len(pleCfg.Layers), ple.Buffers(),
			ple.WeightBytes(), ple.ActivationBytes(), time.Since(start)})
		drop()
	}

	// 3. The gated DeltaNet: 36 of the 48 layers, and the largest dense bank
	//    in the model — one fused [16512, 2560] projection a layer.
	start = time.Now()
	var dnCfg llm.DeltaNetConfig
	var dns []llm.DeltaNetWeights
	for l := 0; l < c.NLayer; l++ {
		cfg, ok, err := m.DeltaNetConfig(l)
		if err != nil {
			return fmt.Errorf("deltanet config %d: %w", l, err)
		}
		if !ok {
			continue
		}
		w, err := m.DeltaNetWeights(l)
		if err != nil {
			return fmt.Errorf("deltanet layer %d: %w", l, err)
		}
		dnCfg, dns = cfg, append(dns, w)
	}
	dn, err := llm.NewDeltaNetGPU(dev, dnCfg, maxTok, dns, llm.DenseQ8())
	if err != nil {
		return fmt.Errorf("deltanet: %w", err)
	}
	defer dn.Destroy()
	plan = append(plan, residentBlock{"deltanet", len(dns), dn.Buffers(),
		dn.WeightBytes(), dn.ActivationBytes(), time.Since(start)})
	dns = nil
	drop()

	// 4. The full-attention layers: the other 12, with the QSA indexer and the
	//    only KV cache in the model.
	start = time.Now()
	var atCfg llm.AttnConfig
	var ats []llm.AttnWeights
	for l := 0; l < c.NLayer; l++ {
		cfg, ok, err := m.AttnConfig(l)
		if err != nil {
			return fmt.Errorf("attn config %d: %w", l, err)
		}
		if !ok {
			continue
		}
		w, err := m.AttnWeights(l)
		if err != nil {
			return fmt.Errorf("attn layer %d: %w", l, err)
		}
		atCfg, ats = cfg, append(ats, w)
	}
	at, err := llm.NewAttnGPU(dev, atCfg, maxTok, nKV, ats, llm.DenseQ8())
	if err != nil {
		return fmt.Errorf("attn: %w", err)
	}
	defer at.Destroy()
	plan = append(plan, residentBlock{"attention", len(ats), at.Buffers(),
		at.WeightBytes(), at.ActivationBytes(), time.Since(start)})
	ats = nil
	drop()

	// 5. The MoE block: 97% of the parameters, and the reason any of this is
	//    a question. One quantised bank a layer, one buffer a bank.
	//
	//    It is staged in a loop, because a width is a measurement here rather
	//    than a setting: the dense blocks above stay where they are and only
	//    the bank is restaged, so `-bank 4,12,24,48` prices residency against
	//    itself with everything else held still.
	if denseOnly {
		after = memAvailable()
		reportResidency(plan, before, after, denseOnly)
		return writeResidentCSV(csvPath, residentCSV(plan, nil))
	}
	var rows [][]string
	for i, n := range widths(bankLayers, c.NLayer) {
		start = time.Now()
		// The staged range always contains moeBenchLayer, because that is
		// the layer the measurement below compares against L5b and a
		// narrower stage that left it out would be measuring a different
		// layer's routing rather than a narrower residency.
		first := 0
		if n <= moeBenchLayer {
			first = moeBenchLayer - n + 1
		}
		var moes []llm.MoEWeights
		for l := first; l < first+n; l++ {
			w, err := m.MoEWeights(l)
			if err != nil {
				return fmt.Errorf("moe layer %d: %w", l, err)
			}
			moes = append(moes, w)
		}
		moe, err := llm.NewMoEGPU(dev, m.MoEConfig(), maxTok, moes)
		if err != nil {
			return fmt.Errorf("moe: %w", err)
		}
		moes = nil
		full := append(plan[:len(plan):len(plan)], residentBlock{"moe", n, moe.Buffers(),
			moe.WeightBytes(), moe.ActivationBytes(), time.Since(start)})
		drop()
		after = memAvailable()
		if i == 0 {
			reportResidency(full, before, after, denseOnly)
		}

		// And the closing question, which only a resident model can answer:
		// does the arrangement cost anything against the two-buffer stage
		// L5b measured? Same kernel, same layer, same input.
		us, err := residentMoECost(moe, m.MoEConfig(), maxTok, weightsOf(full), first)
		if err != nil {
			moe.Destroy()
			return err
		}
		rows = append(rows, residentCSV(full, us)...)
		moe.Destroy()
		drop()
	}
	return writeResidentCSV(csvPath, rows)
}

// writeResidentCSV writes the plan, header and all, or does nothing when no
// path was named.
func writeResidentCSV(path string, rows [][]string) error {
	if path == "" || len(rows) == 0 {
		return nil
	}
	head := []string{"banks", "resident_gb", "block", "layers", "buffers",
		"weight_gb", "arena_mb", "staged_s", "moe_layer3_us"}
	return writeCSV(path, append([][]string{head}, rows...))
}

// widths turns the -bank flag into the list of bank counts to stage. Zero, or
// anything past the layer count, is the whole model.
func widths(spec []int, nLayer int) []int {
	if len(spec) == 0 {
		return []int{nLayer}
	}
	out := make([]int, 0, len(spec))
	for _, n := range spec {
		if n <= 0 || n > nLayer {
			n = nLayer
		}
		out = append(out, n)
	}
	return out
}

// residentCSV is the plan as rows, with the measured block beside every one of
// them so that a row can be read without the header for context.
func residentCSV(plan []residentBlock, us map[int]float64) [][]string {
	var rows [][]string
	banks, resident := 0, weightsOf(plan)
	for _, p := range plan {
		if p.name == "moe" {
			banks = p.layers
		}
	}
	for _, p := range plan {
		rows = append(rows, []string{
			strconv.Itoa(banks), strconv.FormatFloat(float64(resident)/1e9, 'f', 3, 64),
			p.name, strconv.Itoa(p.layers), strconv.Itoa(p.buffers),
			strconv.FormatFloat(float64(p.weights)/1e9, 'f', 4, 64),
			strconv.FormatFloat(float64(p.arenas)/1e6, 'f', 1, 64),
			strconv.FormatFloat(p.staged.Seconds(), 'f', 3, 64),
			strconv.FormatFloat(us[moeBenchLayer], 'f', 1, 64),
		})
	}
	return rows
}

// weightsOf sums a plan's staged weight bytes.
func weightsOf(plan []residentBlock) int {
	var n int
	for _, p := range plan {
		n += p.weights
	}
	return n
}

// moeResidentIters is how many times each dispatch is repeated for the closing
// measurement. Repetition warms nothing here — one layer's bank is 1.6 GB
// against a 32 MiB MALL, so the second pass reads the weights as cold as the
// first — so it buys only the noise floor, and ten is where it stops moving.
const moeResidentIters = 10

// residentMoECost times the MoE block on the first and the last staged layer,
// with every layer of the model on the device.
//
// It is the price of L6a's arrangement, isolated: the bank index is a push
// constant read through an array of moeMaxBanks descriptors, and if selecting
// one of 48 costs more than selecting one of 2 this is where it shows.
func residentMoECost(g *llm.MoEGPU, cfg llm.MoEConfig, maxTok, resident, first int) (map[int]float64, error) {
	x, real, err := moeInput(cfg)
	if err != nil {
		return nil, err
	}
	if maxTok*cfg.NEmbd > len(x) {
		return nil, fmt.Errorf("the input has %d tokens, the arenas are %d", len(x)/cfg.NEmbd, maxTok)
	}
	if err := g.Upload(x[:maxTok*cfg.NEmbd], maxTok); err != nil {
		return nil, err
	}
	fmt.Printf("\nthe block at %d tokens, with %d expert banks and %.1f GB resident",
		maxTok, g.Layers(), float64(resident)/1e9)
	if !real {
		fmt.Printf(" (synthetic input: the routing is near-uniform and the numbers are not the ones to quote)")
	}
	fmt.Println(":")
	// moeBenchLayer first, because that is the layer L5b measured and the
	// input is its own activation — the routing a prompt produces is a
	// property of the *layer's* router, so another layer's block is a
	// different amount of work and not a comparable number.
	layers := []int{moeBenchLayer}
	if last := first + g.Layers() - 1; last != moeBenchLayer {
		layers = append(layers, last)
	}
	out := map[int]float64{}
	for _, l := range layers {
		st, err := g.Profile(l-first, moeResidentIters)
		if err != nil {
			return nil, err
		}
		us := float64(llm.Elapsed(st).Microseconds())
		out[l] = us
		gate, up, down := g.Formats(l - first)
		fmt.Printf("  layer %-2d  %s/%s gate-up, %s down   %8.1f us   x%d = %6.1f ms a graph\n",
			l, gate, up, down, us, g.Layers(), us*float64(g.Layers())/1000)
	}
	fmt.Printf("  L5b measured layer %d at 10824.7 us with two banks staged, 519.6 ms a graph.\n", moeBenchLayer)
	fmt.Println("  The descriptor array is the same width either way and the dispatch reads the")
	fmt.Println("  same 1.6 GB, so a difference is what else is resident and not the indirection.")
	return out, nil
}

// reportResidency prints the plan beside the checkpoint's own inventory.
func reportResidency(plan []residentBlock, before, after int64, denseOnly bool) {
	fmt.Printf("%-12s %7s %8s %12s %11s %11s\n", "block", "layers", "buffers", "weights", "arenas", "staged in")
	var w, a, bufs int
	var total time.Duration
	for _, p := range plan {
		fmt.Printf("%-12s %7d %8d %9.2f GB %8.1f MB %11s\n", p.name, p.layers, p.buffers,
			float64(p.weights)/1e9, float64(p.arenas)/1e6, p.staged.Round(time.Millisecond))
		w += p.weights
		a += p.arenas
		bufs += p.buffers
		total += p.staged
	}
	fmt.Printf("%-12s %7s %8d %9.2f GB %8.1f MB %11s\n", "total", "", bufs,
		float64(w)/1e9, float64(a)/1e6, total.Round(time.Millisecond))
	fmt.Println()
	fmt.Printf("  %.2f GB of weights and %.2f GB of arenas in %d buffers\n",
		float64(w)/1e9, float64(a)/1e9, bufs)
	if !denseOnly {
		// The inventory's figure for everything but the n-gram table, which
		// stays mmap'd (D2), less the two tensors *this tool* does not stage:
		// the embedding is a host-side gather, and the lm head belongs to the
		// graph (llm/gpu_head.go, L6b) rather than to any block, so residency
		// as measured here is the five blocks and not the whole 85.47 GB the
		// graph puts on the device. What is left of the gap is the dense half
		// becoming halves.
		const core, unstaged = 82.52, 0.68 + 0.68
		fmt.Printf("  against the checkpoint's %.2f GB resident core, less the %.2f GB of embedding\n",
			core, unstaged)
		fmt.Printf("  and lm head that nothing stages yet: %+.2f GB on %.2f, %.2fx — the dense\n",
			float64(w)/1e9-(core-unstaged), core-unstaged, float64(w)/1e9/(core-unstaged))
		fmt.Printf("  weights are staged as halves and the 77 GB expert bank is not\n")
	}
	fmt.Printf("  MemAvailable: %.1f GB before, %.1f GB after — %.1f GB gone; this process is resident in %.1f GB\n",
		float64(before)/1e9, float64(after)/1e9, float64(before-after)/1e9, float64(rss())/1e9)
}

// drop returns the host floats a block was staged from to the operating
// system before the next block asks for its own. It matters here and nowhere
// else in this vertical: a whole model's dense weights are dequantised to f32
// on the way to the device, and 36 DeltaNet layers of that is 8.35 GB that
// would otherwise still be held when the next 4 GB arrives.
func drop() {
	runtime.GC()
	debug.FreeOSMemory()
}

// rss is this process's resident set, from /proc/self/statm. The device
// buffers are system RAM on this machine — there is no other kind — so a
// staged model shows up here, and it is the number to put beside the
// inventory's 82.52 GB.
func rss() int64 {
	buf, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	f := strings.Fields(string(buf))
	if len(f) < 2 {
		return 0
	}
	pages, err := strconv.ParseInt(f[1], 10, 64)
	if err != nil {
		return 0
	}
	return pages * int64(os.Getpagesize())
}

// memAvailable reads MemAvailable from /proc/meminfo, in bytes.
//
// It is the right field rather than MemFree: this machine's weights come out
// of a 111 GB mmap'd checkpoint, so most of what is not free is page cache
// that the next allocation can have.
func memAvailable() int64 {
	buf, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(buf), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		f := strings.Fields(line)
		if len(f) < 2 {
			return 0
		}
		kb, err := strconv.ParseInt(f[1], 10, 64)
		if err != nil {
			return 0
		}
		return kb * 1024
	}
	return 0
}
