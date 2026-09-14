// Command ditbench times the Vulkan DiT attention stack at a sweep of
// sequence lengths. 4096 tokens is a 1024x1024 image's latent grid, which is
// what the real model runs at.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
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
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{})
	must(err)
	defer dev.Destroy()
	fmt.Printf("device: %s\ndim %d, %d heads x %d\n\n", phys.Name, cfg.Dim, cfg.NHeads, cfg.Dim/cfg.NHeads)

	fmt.Printf("%-8s %10s %12s %12s %14s\n", "TOKENS", "ACT MB", "WALL", "GFLOP", "GFLOP/S")
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

		if _, err := g.Apply(q, k, v, false); err != nil { // warm
			fmt.Printf("%-8d failed: %v\n", n, err)
			g.Destroy()
			continue
		}
		best := time.Duration(1) << 62
		for i := 0; i < 3; i++ {
			t0 := time.Now()
			_, err := g.Apply(q, k, v, false)
			must(err)
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		// Two passes of q.k plus the weighted sum of v, over every head.
		fl := 3 * 2 * float64(n) * float64(n) * float64(cfg.Dim)
		fmt.Printf("%-8d %10.1f %12s %12.1f %14.0f\n",
			n, float64(g.ActivationBytes())/1e6, best.Round(time.Millisecond), fl/1e9, fl/best.Seconds()/1e9)
		g.Destroy()
	}
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
