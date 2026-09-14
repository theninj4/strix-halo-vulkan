// Command ditblock times a whole Z-Image DiT block on the GPU, dispatch by
// dispatch, and sweeps the projection GEMM's ladder over the model's three
// shapes.
//
// It is the stage-4 counterpart of cmd/ditbench, which times the attention
// stack alone. Everything here is GPU timestamps: a wall-clock figure around
// a block also carries the host write of x and the read-back of the result,
// and this arena's reads run at 0.2 GB/s, which is how stage 3c came to
// measure ten different kernels at the same 347 ms
// (research/stage-3-dit-attention.md).
//
// 4096 tokens is a 1024x1024 image's latent grid; 320 is the sequence the
// diffusers reference dump uses, and the refiner blocks run on the caption
// stream at about that length.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"slices"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/dit"
)

const strixHaloDeviceID = 0x1586

// blocks and steps are what a whole image costs: 34 transformer blocks
// (30 layers plus two context and two noise refiners) and 8 denoising steps
// for Z-Image-Turbo.
const (
	blocks = 34
	steps  = 8
)

func main() {
	log.SetFlags(0)
	dir := flag.String("transformer", "models/Z-Image-Turbo/transformer", "checkpoint")
	sizes := flag.String("tokens", "320,1024,4096", "comma-separated sequence lengths")
	only := flag.String("kernels", "", "comma-separated GEMM kernels; default every build, plus the mixed default plan")
	reps := flag.Int("reps", 3, "timed repetitions per configuration; the best is reported")
	verbose := flag.Bool("v", false, "print every dispatch, not just the summary")
	flag.Parse()

	cfg, err := dit.LoadConfig(*dir)
	must(err)
	set, err := safetensors.OpenSet(*dir)
	must(err)
	defer set.Close()
	blk, err := dit.LoadBlock(set, "layers.0", cfg)
	must(err)

	inst, err := vk.NewInstance("ditblock")
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

	fmt.Printf("device: %s\ndim %d, %d heads x %d, ffn %d, %d blocks x %d steps\n\n",
		phys.Name, cfg.Dim, cfg.NHeads, cfg.Dim/cfg.NHeads, blk.FFN.W1.Out, blocks, steps)

	// The ladder, plus "default": the per-shape plan DefaultGEMMPlan picks,
	// which is the one the pipeline would actually run.
	type arm struct {
		name string
		plan dit.GEMMPlan
	}
	arms := []arm{{"default", dit.DefaultGEMMPlan()}}
	for _, k := range dit.GEMMKernels() {
		arms = append(arms, arm{string(k), dit.UniformGEMMPlan(k)})
	}
	if *only != "" {
		want := strings.Split(*only, ",")
		arms = slices.DeleteFunc(arms, func(a arm) bool { return !slices.Contains(want, a.name) })
	}

	for _, tok := range strings.Split(*sizes, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(tok))
		if err != nil {
			continue
		}
		// Positional ids: a caption run along axis 0 then a square image grid
		// on axes 1 and 2, which is the shape patchify_and_embed produces.
		// Only the rotary table depends on them and it is the same size
		// either way, so the exact split is immaterial to a timing run.
		ids := make([][3]int32, n)
		for i := range ids {
			ids[i] = [3]int32{int32(i % cfg.AxesLens[0]), 0, 0}
		}
		rope, err := dit.NewRoPE(ids, cfg.AxesDims, cfg.AxesLens, cfg.RopeTheta)
		must(err)

		rng := rand.New(rand.NewSource(1))
		x := dit.NewMat(n, cfg.Dim)
		for i := range x.Data {
			x.Data[i] = float32(rng.NormFloat64())
		}
		adaln := make([]float32, blk.AdaLN.In)
		for i := range adaln {
			adaln[i] = float32(rng.NormFloat64())
		}

		// Per-shape times, for the arms that run one kernel everywhere. The
		// DiT has exactly three GEMM shapes and results/shapes.csv says they
		// do not agree on a kernel, so this table -- not the block total --
		// is what a plan is built from.
		type shape struct {
			name  string
			flops float64
		}
		shapes := []shape{
			{"qkv/o", 2 * float64(n) * float64(cfg.Dim) * float64(cfg.Dim)},
			{"ff.w13", 2 * float64(n) * float64(cfg.Dim) * float64(blk.FFN.W1.Out)},
			{"ff.w2", 2 * float64(n) * float64(blk.FFN.W1.Out) * float64(cfg.Dim)},
		}
		shapeOf := map[string]string{
			"gemm q": "qkv/o", "gemm k": "qkv/o", "gemm v": "qkv/o", "gemm o": "qkv/o",
			"gemm w1": "ff.w13", "gemm w3": "ff.w13", "gemm w2": "ff.w2",
		}
		perShape := map[string]map[string][]time.Duration{}

		fmt.Printf("== %d tokens ==\n", n)
		fmt.Printf("%-18s %10s %10s %10s %10s %9s %9s\n",
			"PLAN", "BLOCK", "GEMMS", "ATTN", "OTHER", "TFLOP/S", "IMAGE")
		for _, a := range arms {
			g, err := dit.NewGPUBlock(dev, blk, rope, n, a.plan)
			if err != nil {
				fmt.Printf("%-18s failed: %v\n", a.name, err)
				continue
			}
			if _, err := g.Apply(x, adaln); err != nil { // warm the pipelines
				fmt.Printf("%-18s failed: %v\n", a.name, err)
				g.Destroy()
				continue
			}
			var best []dit.Stage
			for i := 0; i < *reps; i++ {
				st, _, err := g.Profile(x, adaln)
				must(err)
				if best == nil || dit.Elapsed(st) < dit.Elapsed(best) {
					best = st
				}
			}
			var gemm, attn, total time.Duration
			uniform := a.name != "default"
			for _, s := range best {
				total += s.GPU
				switch {
				case strings.HasPrefix(s.Kind, "gemm"):
					gemm += s.GPU
					if uniform {
						if perShape[a.name] == nil {
							perShape[a.name] = map[string][]time.Duration{}
						}
						sh := shapeOf[s.Kind]
						perShape[a.name][sh] = append(perShape[a.name][sh], s.GPU)
					}
				case s.Kind == "attention":
					attn += s.GPU
				}
			}
			flops := g.FLOPs()
			fmt.Printf("%-18s %10s %10s %10s %10s %9.1f %9s\n",
				a.name, ms(total), ms(gemm), ms(attn), ms(total-gemm-attn),
				flops/total.Seconds()/1e12, ms(total*blocks*steps))
			if *verbose {
				for _, s := range best {
					fmt.Printf("    %-14s %10s\n", s.Kind, ms(s.GPU))
				}
			}
			g.Destroy()
		}

		if len(perShape) > 0 {
			fmt.Printf("\n%-18s", "BY SHAPE")
			for _, sh := range shapes {
				fmt.Printf(" %9s %8s", sh.name, "")
			}
			fmt.Println()
			for _, a := range arms {
				byShape, ok := perShape[a.name]
				if !ok {
					continue
				}
				fmt.Printf("%-18s", a.name)
				for _, sh := range shapes {
					ds := byShape[sh.name]
					if len(ds) == 0 {
						fmt.Printf(" %18s", "-")
						continue
					}
					// The fastest of the projections that share a shape:
					// they are the same GEMM with a different weight, so the
					// spread between them is run-to-run noise.
					best := ds[0]
					for _, d := range ds[1:] {
						best = min(best, d)
					}
					fmt.Printf(" %6.1f T/s %8s", sh.flops/best.Seconds()/1e12, ms(best))
				}
				fmt.Println()
			}
		}
		fmt.Println()
	}
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
