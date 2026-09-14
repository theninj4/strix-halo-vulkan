// Command ditstack times the whole Z-Image DiT -- all 34 blocks, resident --
// for one denoising step, and reports what it costs per image.
//
// cmd/ditblock times one block and sweeps the GEMM ladder over the model's
// three shapes; this runs the real thing, so what it adds is everything a
// stack has that a block does not: 12.0 GB of fp16 weights split across three
// storage buffers because one addresses 4.29 GB here, the two unmodulated
// context refiners, and the three phases running at three different sequence
// lengths.
//
// Those phases are the transformer's own (diffusers'
// ZImageTransformer2DModel.forward): the noise refiners run over the image
// tokens, the context refiners over the caption, and the 30 layers over the
// two concatenated. A 1024x1024 image is 4096 image tokens, so the layers --
// which are 30 of the 34 blocks -- run at 4096 + the caption length, not at
// 4096.
//
// Everything reported is GPU timestamps per dispatch. Wall clock around a run
// also carries the host write of x and the read-back of the result, and this
// arena's reads run at 0.2 GB/s (research/stage-3-dit-attention.md).
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/dit"
)

const strixHaloDeviceID = 0x1586

// steps is Z-Image-Turbo's NFE: the scheduler runs the whole stack eight
// times per image.
const steps = 8

// phase is one of the transformer's three runs over the stack, each with its
// own blocks and its own sequence length.
type phase struct {
	name   string
	blocks []int
	rows   int
}

func main() {
	log.SetFlags(0)
	dir := flag.String("transformer", "models/Z-Image-Turbo/transformer", "checkpoint")
	image := flag.Int("image", 4096, "image tokens; 4096 is a 1024x1024 latent grid at patch size 2")
	caption := flag.Int("caption", 128, "caption tokens")
	reps := flag.Int("reps", 3, "timed repetitions; the best is reported")
	verbose := flag.Bool("v", false, "print every dispatch of the first block of each phase")
	unfusedLayout := flag.Bool("unfused-layout", false,
		"keep `pack v` and `narrow ctx` as their own dispatches, which is stage 9's graph")
	flag.Parse()

	cfg, err := dit.LoadConfig(*dir)
	must(err)
	set, err := safetensors.OpenSet(*dir)
	must(err)
	defer set.Close()

	inst, err := vk.NewInstance("ditstack")
	must(err)
	defer inst.Destroy()
	devs, err := inst.PhysicalDevices()
	must(err)
	phys := &devs[0]
	for i := range devs {
		if devs[i].DeviceID == strixHaloDeviceID {
			phys = &devs[i]
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

	// The unified sequence is the longest thing the stack ever runs, so the
	// arenas and the rotary table are built for it and the two shorter phases
	// use a prefix.
	unified := *caption + *image
	rope, err := dit.NewRoPE(ids(*caption, *image, cfg.AxesLens[0]), cfg.AxesDims, cfg.AxesLens, cfg.RopeTheta)
	must(err)

	fmt.Printf("device: %s\n", phys.Name)
	fmt.Printf("dim %d, %d heads x %d, %d+%d+%d blocks, %d caption + %d image = %d tokens, %d steps\n",
		cfg.Dim, cfg.NHeads, cfg.Dim/cfg.NHeads, cfg.NRefiner, cfg.NRefiner, cfg.NLayers,
		*caption, *image, unified, steps)

	start := time.Now()
	g, err := dit.NewGPUStack(dev, set, cfg, rope, unified, nil)
	must(err)
	defer g.Destroy()
	g.FuseLayout = !*unfusedLayout
	load := time.Since(start)

	banks, byBlock := g.Banks()
	fmt.Printf("\n%d blocks resident in %.1f s: %.2f GB of weights (%d fp16 banks",
		g.Len(), load.Seconds(), float64(g.WeightBytes())/1e9, len(banks))
	for i, b := range banks {
		n := 0
		for _, bb := range byBlock {
			if bb == i {
				n++
			}
		}
		fmt.Printf(", %.2f GB x %d", float64(b)/1e9, n)
	}
	fmt.Printf("), %.2f GB of activations\n\n", float64(g.ActivationBytes())/1e9)

	// The three phases, in the order the transformer runs them. StackBlocks
	// lays the stack out the same way, so the indices follow from the counts.
	nr := cfg.NRefiner
	phases := []phase{
		{"noise_refiner", seq(0, nr), *image},
		{"context_refiner", seq(nr, 2*nr), *caption},
		{"layers", seq(2*nr, 2*nr+cfg.NLayers), unified},
	}

	rng := rand.New(rand.NewSource(1))
	adaln := make([]float32, 256)
	for i := range adaln {
		adaln[i] = float32(rng.NormFloat64())
	}

	fmt.Printf("%-16s %7s %8s %12s %12s %12s %12s %9s\n",
		"PHASE", "BLOCKS", "TOKENS", "PER BLOCK", "TOTAL", "GEMMS", "ATTN", "TFLOP/S")
	var step, layersGPU time.Duration
	for _, p := range phases {
		x := dit.NewMat(p.rows, cfg.Dim)
		for i := range x.Data {
			x.Data[i] = float32(rng.NormFloat64())
		}
		if _, err := g.Apply(x, adaln, p.blocks); err != nil { // warm the pipelines
			log.Fatalf("%s: %v", p.name, err)
		}
		var best []dit.Stage
		for i := 0; i < *reps; i++ {
			st, _, err := g.Profile(x, adaln, p.blocks)
			must(err)
			if best == nil || dit.Elapsed(st) < dit.Elapsed(best) {
				best = st
			}
		}
		var gemm, attn, total time.Duration
		for _, s := range best {
			total += s.GPU
			switch {
			case strings.HasPrefix(s.Kind, "gemm"):
				gemm += s.GPU
			case s.Kind == "attention":
				attn += s.GPU
			}
		}
		step += total
		if p.name == phases[len(phases)-1].name {
			layersGPU = total
		}
		flops := g.FLOPs(p.rows) * float64(len(p.blocks))
		fmt.Printf("%-16s %7d %8d %12s %12s %12s %12s %9.1f\n",
			p.name, len(p.blocks), p.rows, ms(total/time.Duration(len(p.blocks))), ms(total),
			ms(gemm), ms(attn), flops/total.Seconds()/1e12)
		if *verbose {
			for _, s := range best {
				if s.Block != p.blocks[0] {
					break
				}
				fmt.Printf("    %-14s %10s\n", s.Kind, ms(s.GPU))
			}
		}
	}

	fmt.Printf("\n%-16s %s per step, %s per image over %d steps\n",
		"TOTAL", ms(step), ms(step*steps), steps)

	// The layers phase as wall clock, to price what the GPU timestamps leave
	// out: 30 blocks is 540 dispatches submitted in batches of eight, so 68
	// fence waits, and 93% of the step's work. The upload and the read-back
	// are deliberately outside it -- a pipeline does those once per image, not
	// once per step -- which is what GPUStack.Run is for.
	layers := phases[len(phases)-1]
	x := dit.NewMat(layers.rows, cfg.Dim)
	for i := range x.Data {
		x.Data[i] = float32(rng.NormFloat64())
	}
	if _, err := g.Apply(x, adaln, layers.blocks[:1]); err != nil {
		log.Fatal(err)
	}
	var wall time.Duration
	for i := 0; i < *reps; i++ {
		t0 := time.Now()
		must(g.Run(layers.blocks))
		if d := time.Since(t0); wall == 0 || d < wall {
			wall = d
		}
	}
	fmt.Printf("%-16s layers as wall clock %s against %s of dispatches, %+.1f%% (submission and fences)\n",
		"", ms(wall), ms(layersGPU), 100*(wall.Seconds()/layersGPU.Seconds()-1))
}

// ids is the positional grid the transformer builds: the caption tokens run
// along axis 0, the image tokens tile a square grid on axes 1 and 2. Only the
// rotary table depends on them, and it is the same size either way, so the
// exact split is immaterial to a timing run -- but the *lengths* are not,
// since the three phases run over different parts of it.
func ids(caption, image, capLen int) [][3]int32 {
	side := 1
	for side*side < image {
		side++
	}
	out := make([][3]int32, caption+image)
	for i := 0; i < caption; i++ {
		out[i] = [3]int32{int32(i % capLen), 0, 0}
	}
	for i := 0; i < image; i++ {
		out[caption+i] = [3]int32{0, int32(i / side), int32(i % side)}
	}
	return out
}

func seq(lo, hi int) []int {
	out := make([]int, 0, hi-lo)
	for i := lo; i < hi; i++ {
		out = append(out, i)
	}
	return out
}

func ms(d time.Duration) string {
	if d >= time.Second {
		return fmt.Sprintf("%.2f s", d.Seconds())
	}
	return fmt.Sprintf("%.2f ms", float64(d.Microseconds())/1000)
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
