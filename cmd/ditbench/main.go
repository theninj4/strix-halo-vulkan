// Command ditbench times the Vulkan DiT attention stack at a sweep of
// sequence lengths. 4096 tokens is a 1024x1024 image's latent grid, which is
// what the real model runs at.
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

func main() {
	log.SetFlags(0)
	dir := flag.String("transformer", "models/Z-Image-Turbo/transformer", "checkpoint")
	sizes := flag.String("tokens", "320,1024,2048,4096", "comma-separated sequence lengths")
	only := flag.String("kernels", "", "comma-separated kernel names; default every kernel the device has")
	flag.Parse()

	cfg, err := dit.LoadConfig(*dir)
	must(err)
	set, err := safetensors.OpenSet(*dir)
	must(err)
	defer set.Close()
	blk, err := dit.LoadBlock(set, "layers.0", cfg)
	must(err)

	inst, err := vk.NewInstance("ditbench")
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
	fmt.Printf("device: %s\ndim %d, %d heads x %d\n\n", phys.Name, cfg.Dim, cfg.NHeads, cfg.Dim/cfg.NHeads)

	fmt.Printf("%-8s %-18s %6s %11s %11s %8s %12s %12s\n",
		"TOKENS", "KERNEL", "AI", "ATTN", "GRAPH", "WALL", "GFLOP", "GFLOP/S")
	for _, tok := range strings.Split(*sizes, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(tok))
		if err != nil {
			continue
		}
		ids := make([][3]int32, n)
		for i := range ids {
			ids[i] = [3]int32{int32(i % cfg.AxesLens[0]), 0, 0}
		}
		rope, err := dit.NewRoPE(ids, cfg.AxesDims, cfg.AxesLens, cfg.RopeTheta)
		must(err)

		g, err := dit.NewGPUAttention(dev, blk.Attn, rope, n)
		if err != nil {
			fmt.Printf("%-8d failed: %v\n", n, err)
			continue
		}

		rng := rand.New(rand.NewSource(1))
		mk := func() *dit.Mat {
			m := dit.NewMat(n, cfg.Dim)
			for i := range m.Data {
				m.Data[i] = float32(rng.NormFloat64())
			}
			return m
		}
		q, k, v := mk(), mk(), mk()

		for _, kernel := range g.Kernels() {
			if *only != "" && !slices.Contains(strings.Split(*only, ","), string(kernel)) {
				continue
			}
			g.Kernel = kernel
			if _, err := g.Apply(q, k, v, false); err != nil { // warm
				fmt.Printf("%-8d %-18s failed: %v\n", n, kernel, err)
				continue
			}
			// Three figures, because they differ by two orders of magnitude
			// and only one of them is the kernel:
			//
			//   attn  the attention dispatch, timed on the GPU. This is what
			//         the ladder is being compared on.
			//   graph every dispatch Apply issues, so the fp16 pack shows up
			//         as the ~30% of GPU time it is. Stage 4 removes it by
			//         having the projections write the packed layout directly.
			//   wall  the same call from the CPU's side, which at 4096 tokens
			//         is 97% the 63 MB read-back of the result: the arena is
			//         device-local host-visible memory, where writes run at
			//         11.5 GB/s and reads at 0.2. Nothing in the real pipeline
			//         reads a block's output back to the host, so this column
			//         is a property of the harness, not of the kernel.
			attn, graph, wall := time.Duration(1)<<62, time.Duration(1)<<62, time.Duration(1)<<62
			for i := 0; i < 3; i++ {
				t0 := time.Now()
				_, err := g.Apply(q, k, v, false)
				must(err)
				wall = min(wall, time.Since(t0))

				st, _, err := g.Profile(q, k, v, false)
				must(err)
				var total, one time.Duration
				for _, s := range st {
					total += s.GPU
					if s.Kind == "attention" {
						one = s.GPU
					}
				}
				attn, graph = min(attn, one), min(graph, total)
			}
			// Attention is two GEMMs -- q.k^T and p.v -- each 2*n*n*headDim
			// per head, so 4*n*n*dim over all of them. (Earlier runs of this
			// file used 6*n*n*dim, which counted a pass that does not exist;
			// every GFLOP/s figure from before 2026-09-14 is 1.5x optimistic.)
			fl := 4 * float64(n) * float64(n) * float64(cfg.Dim)
			ai := "-"
			if v := g.Intensity(kernel); v > 0 {
				ai = strconv.Itoa(int(v))
			}
			fmt.Printf("%-8d %-18s %6s %11s %11s %8s %12.1f %12.0f\n",
				n, kernel, ai, attn.Round(time.Microsecond), graph.Round(time.Microsecond),
				wall.Round(time.Millisecond), fl/1e9, fl/attn.Seconds()/1e9)
		}
		g.Destroy()
	}
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
