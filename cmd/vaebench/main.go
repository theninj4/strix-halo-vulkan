// Command vaebench times the Vulkan VAE decoder at a sweep of latent sizes
// and reports the activation arena each one needs. The target is a 128x128
// latent, which is the 1024x1024 image Z-Image-Turbo generates.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/vae"
)

const strixHaloDeviceID = 0x1586

func main() {
	log.SetFlags(0)
	dir := flag.String("vae", "models/Z-Image-Turbo/vae", "VAE checkpoint directory")
	sizes := flag.String("sizes", "16,32,64", "comma-separated latent sizes")
	flag.Parse()

	inst, err := vk.NewInstance("vaebench")
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
	fmt.Printf("device: %s\n\n", phys.Name)

	cpu, err := vae.LoadDecoder(*dir, vae.FluxConfig())
	must(err)

	fmt.Printf("%-8s %-12s %10s %12s %12s\n", "LATENT", "IMAGE", "DISPATCH", "ACT MB", "WALL")
	for _, tok := range strings.Split(*sizes, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(tok))
		if err != nil {
			continue
		}
		g, err := vae.NewGPUDecoder(dev, cpu, n, n)
		if err != nil {
			fmt.Printf("%-8d failed: %v\n", n, err)
			continue
		}
		nd, err := g.Dispatches(n, n)
		must(err)

		latent := vae.NewTensor(1, 16, n, n)
		rng := rand.New(rand.NewSource(1))
		for i := range latent.Data {
			latent.Data[i] = float32(rng.NormFloat64())
		}

		if _, err := g.Apply(latent); err != nil { // warm
			fmt.Printf("%-8d failed: %v\n", n, err)
			g.Destroy()
			continue
		}
		best := time.Duration(1) << 62
		for i := 0; i < 3; i++ {
			t0 := time.Now()
			_, err := g.Apply(latent)
			must(err)
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		fmt.Printf("%-8d %-12s %10d %12.1f %12s\n",
			n, fmt.Sprintf("%dx%d", n*8, n*8), nd,
			float64(g.ActivationBytes())/1e6, best.Round(time.Millisecond))
		g.Destroy()
	}
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
