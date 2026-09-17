// Command embed runs Qwen3-Embedding-0.6B: text in, a 1024-dimensional unit
// vector out.
//
// `-gpu` (the default) runs the Vulkan stack, `-cpu` the fp32 reference it
// was debugged against. With no arguments it runs the model card's own
// example -- two queries against two documents -- and prints the similarity
// matrix, which is the end-to-end check that the whole vertical is right.
//
//	go run ./cmd/embed
//	go run ./cmd/embed -text "the quick brown fox"
//	go run ./cmd/embed -query "what is the capital of China?" \
//	                   -doc "The capital of China is Beijing." -doc "Gravity is a force."
//	go run ./cmd/embed -profile
//
// What it measures is a model on the memory-bound half of the roofline: at T
// tokens a projection is T flop per byte of weight against this device's 235
// crossover (research/3.4-model-shapes.md), so a text of a few dozen tokens
// is bounded by the 0.88 GB of fp16 weights read once, whatever T is. The
// number to watch is therefore the fraction of the 236 GB/s bus a run
// reaches, not the fraction of the matrix cores.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"time"

	"strix-halo-vulkan/embed"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

const strixHaloDeviceID = 0x1586

// stringList is a flag that may be given more than once.
type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ", ") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

// The model card's example, which is what a bare run reproduces.
var cardQueries = []string{"What is the capital of China?", "Explain gravity"}
var cardDocs = []string{
	"The capital of China is Beijing.",
	"Gravity is a force that attracts two bodies towards each other. It gives " +
		"weight to physical objects and is responsible for the movement of planets " +
		"around the sun.",
}

func main() {
	log.SetFlags(0)
	dir := flag.String("model", "models/Qwen3-Embedding-0.6B", "checkpoint directory")
	var texts, queries, docs stringList
	flag.Var(&texts, "text", "a text to embed; repeatable")
	flag.Var(&queries, "query", "a query to embed, with the instruction prefix; repeatable")
	flag.Var(&docs, "doc", "a document to embed, with no instruction; repeatable")
	task := flag.String("task", embed.DefaultTask, "the instruction a -query is prefixed with")
	cpu := flag.Bool("cpu", false, "run the fp32 CPU reference instead of the device")
	dim := flag.Int("dim", 0, "truncate and renormalise to this many components (MRL); 0 keeps 1024")
	reps := flag.Int("reps", 5, "timed repetitions; the best is reported")
	maxTokens := flag.Int("maxtokens", 0, "arena size in tokens; 0 fits the inputs")
	profile := flag.Bool("profile", false, "print the per-dispatch GPU profile of the first run")
	ladder := flag.Bool("ladder", false, "sweep the GEMM ladder on the longest input instead of embedding")
	show := flag.Int("show", 4, "components of each vector to print")
	flag.Parse()

	if len(texts) == 0 && len(queries) == 0 && len(docs) == 0 {
		queries, docs = cardQueries, cardDocs
	}
	// Queries carry the instruction and documents do not; that asymmetry is
	// the model's, not this command's (EMBEDDING.md E0).
	var inputs []string
	var kinds []string
	for _, q := range queries {
		inputs = append(inputs, embed.Instruct(*task, q))
		kinds = append(kinds, "query")
	}
	for _, d := range docs {
		inputs = append(inputs, d)
		kinds = append(kinds, "doc")
	}
	for _, t := range texts {
		inputs = append(inputs, t)
		kinds = append(kinds, "text")
	}

	cfg, err := embed.LoadConfig(*dir)
	must(err)
	fmt.Printf("model    Qwen3-Embedding-0.6B: %d layers, hidden %d, %d/%d heads of %d, ffn %d\n",
		cfg.NumLayers, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateSize)

	if *cpu {
		runCPU(*dir, inputs, kinds, *dim, *reps, *show)
		return
	}
	runGPU(*dir, cfg, inputs, kinds, *dim, *reps, *maxTokens, *profile, *ladder, *show)
}

// runCPU is the fp32 oracle: 2.4 GB on the host and a couple of seconds a
// text, kept because it is what the device path is checked against.
func runCPU(dir string, inputs, kinds []string, dim, reps, show int) {
	start := time.Now()
	m, err := embed.Load(dir)
	must(err)
	fmt.Printf("load     %v (fp32 on the host)\n", time.Since(start).Round(time.Millisecond))

	vecs := make([][]float32, len(inputs))
	for i, text := range inputs {
		ids, err := m.Tok.Encode(text)
		must(err)
		best := time.Duration(math.MaxInt64)
		for r := 0; r < reps; r++ {
			start = time.Now()
			v, err := m.EmbedIDs(ids)
			must(err)
			best = min(best, time.Since(start))
			vecs[i] = v
		}
		vecs[i] = embed.Truncate(vecs[i], dim)
		report("cpu", kinds[i], text, len(ids), best, vecs[i], show)
	}
	scores(inputs, kinds, vecs)
}

func runGPU(dir string, cfg *qwen.Config, inputs, kinds []string, dim, reps, maxTokens int, prof, ladder bool, show int) {
	dev, done := openDevice()
	defer done()

	tok, err := embed.LoadTokenizer(dir)
	must(err)
	ids := make([][]int32, len(inputs))
	longest := 0
	for i, text := range inputs {
		ids[i], err = tok.Encode(text)
		must(err)
		longest = max(longest, len(ids[i]))
	}
	if maxTokens == 0 {
		maxTokens = longest
	}

	start := time.Now()
	g, err := embed.NewGPU(dev, dir, maxTokens)
	must(err)
	defer g.Destroy()
	fmt.Printf("load     %v -- %.2f GB of fp16 weights, arenas for %d tokens\n",
		time.Since(start).Round(time.Millisecond), float64(g.Enc.WeightBytes())/1e9, maxTokens)
	fmt.Printf("kernels  %s attention, %d dispatches per layer, %d layers\n",
		g.Enc.Attention(), len(g.Enc.Labels()), cfg.NumLayers)

	if ladder {
		longestIdx := 0
		for i := range ids {
			if len(ids[i]) > len(ids[longestIdx]) {
				longestIdx = i
			}
		}
		sweep(g, cfg, ids[longestIdx], reps)
		return
	}

	vecs := make([][]float32, len(inputs))
	for i := range inputs {
		best := time.Duration(math.MaxInt64)
		for r := 0; r < reps; r++ {
			start = time.Now()
			v, err := g.EmbedIDs(ids[i])
			must(err)
			best = min(best, time.Since(start))
			vecs[i] = v
		}
		vecs[i] = embed.Truncate(vecs[i], dim)
		reportGPU("gpu", kinds[i], inputs[i], len(ids[i]), best, vecs[i], show, cfg, g)
		if prof && i == 0 {
			profileRun(g, ids[i])
		}
	}
	scores(inputs, kinds, vecs)
}

func report(where, kind, text string, tokens int, best time.Duration, v []float32, show int) {
	fmt.Printf("\n%s  %-5s %4d tokens  %v   %s\n", where, kind, tokens,
		best.Round(time.Microsecond), quote(text))
	fmt.Printf("     %d-d unit vector: %s\n", len(v), head(v, show))
}

// reportGPU adds the two ceilings the run sits between: the weights read once
// at 236 GB/s, and the arithmetic at the 55.5 TFLOP/s the matrix cores reach.
// At these lengths the first is the binding one, which is the point.
func reportGPU(where, kind, text string, tokens int, best time.Duration, v []float32,
	show int, cfg *qwen.Config, g *embed.GPU) {
	report(where, kind, text, tokens, best, v, show)
	perLayer := float64(cfg.HiddenSize) * float64(
		2*cfg.NumHeads*cfg.HeadDim+2*cfg.NumKVHeads*cfg.HeadDim+3*cfg.IntermediateSize)
	params := perLayer * float64(cfg.NumLayers)
	flops := 2 * params * float64(tokens)
	weights := 2 * params // fp16
	fmt.Printf("     %.1f GFLOP at %.1f GFLOP/s; %.2f GB of weights at %.1f GB/s (%.0f%% of 236)\n",
		flops/1e9, flops/best.Seconds()/1e9, weights/1e9,
		weights/best.Seconds()/1e9, 100*weights/best.Seconds()/1e9/236)
}

// sweep times every rung of the GEMM ladder on one input. The figure that
// decides it is not GFLOP/s -- at a few dozen tokens most of a tile's rows
// are padding -- but the fraction of the 236 GB/s bus the weight traffic
// reaches. The weights are staged in the layout every rung reads, so this
// re-plans rather than reloading.
func sweep(g *embed.GPU, cfg *qwen.Config, ids []int32, reps int) {
	fmt.Printf("\nGEMM ladder at %d tokens (best of %d)\n", len(ids), reps)
	fmt.Printf("  %-22s %10s %10s %10s\n", "kernel", "wall", "GB/s", "vs best")
	type row struct {
		name string
		best time.Duration
	}
	g.AutoPlan = false
	defer func() { g.AutoPlan = true }()
	var rows []row
	for _, k := range qwen.GEMMKernels() {
		if err := g.Enc.SetPlan(qwen.UniformGEMMPlan(k)); err != nil {
			fmt.Printf("  %-22s %10s\n", k, "unbuildable")
			continue
		}
		best := time.Duration(math.MaxInt64)
		for i := 0; i < reps; i++ {
			start := time.Now()
			_, err := g.EmbedIDs(ids)
			must(err)
			best = min(best, time.Since(start))
		}
		rows = append(rows, row{string(k), best})
	}
	// And the two schedules a run can actually get: this package's own table
	// and the text encoder's, which is what "use the plan we already have"
	// would pick.
	for _, c := range []struct {
		name string
		plan qwen.GEMMPlan
	}{{"embed.PlanFor (default)", embed.PlanFor(len(ids))}, {"qwen.PlanFor (z-image)", qwen.PlanFor(len(ids))}} {
		must(g.Enc.SetPlan(c.plan))
		best := time.Duration(math.MaxInt64)
		for i := 0; i < reps; i++ {
			start := time.Now()
			_, err := g.EmbedIDs(ids)
			must(err)
			best = min(best, time.Since(start))
		}
		rows = append(rows, row{c.name, best})
	}

	fastest := rows[0].best
	for _, r := range rows {
		fastest = min(fastest, r.best)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].best < rows[j].best })
	perLayer := float64(cfg.HiddenSize) * float64(
		2*cfg.NumHeads*cfg.HeadDim+2*cfg.NumKVHeads*cfg.HeadDim+3*cfg.IntermediateSize)
	weights := 2 * perLayer * float64(cfg.NumLayers)
	for _, r := range rows {
		fmt.Printf("  %-22s %10v %10.1f %9.2fx\n", r.name, r.best.Round(time.Microsecond),
			weights/r.best.Seconds()/1e9, float64(r.best)/float64(fastest))
	}
	must(g.Enc.SetPlan(embed.PlanFor(len(ids))))
}

// profileRun times every dispatch on the GPU and sums by kind. Wall clock
// around a run also carries the host-side embedding gather, the RoPE table
// and the read-back, so the two numbers answer different questions and the
// gap between them is the host's share.
func profileRun(g *embed.GPU, ids []int32) {
	stages, _, err := g.Enc.Profile(ids)
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

// scores prints the query-by-document cosine matrix, which is the shape the
// model card reports and the only end-to-end check this command makes.
func scores(inputs, kinds []string, vecs [][]float32) {
	var q, d []int
	for i, k := range kinds {
		switch k {
		case "query":
			q = append(q, i)
		case "doc":
			d = append(d, i)
		}
	}
	if len(q) == 0 || len(d) == 0 {
		if len(vecs) > 1 {
			fmt.Printf("\ncosine between the %d inputs\n", len(vecs))
			for i := range vecs {
				for j := range vecs {
					fmt.Printf("  %7.4f", embed.Cosine(vecs[i], vecs[j]))
				}
				fmt.Printf("   %s\n", quote(inputs[i]))
			}
		}
		return
	}
	fmt.Printf("\nquery x document cosine\n")
	for _, i := range q {
		for _, j := range d {
			fmt.Printf("  %7.4f", embed.Cosine(vecs[i], vecs[j]))
		}
		fmt.Printf("   %s\n", quote(inputs[i]))
	}
}

func head(v []float32, n int) string {
	if n > len(v) {
		n = len(v)
	}
	parts := make([]string, n)
	for i := 0; i < n; i++ {
		parts[i] = fmt.Sprintf("%+.5f", v[i])
	}
	return "[" + strings.Join(parts, " ") + " …]"
}

func quote(s string) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	if len(s) > 64 {
		s = s[:64] + "…"
	}
	return `"` + s + `"`
}

func openDevice() (*vk.Device, func()) {
	inst, err := vk.NewInstance("embed")
	must(err)
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
	fmt.Printf("device   %s\n", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
