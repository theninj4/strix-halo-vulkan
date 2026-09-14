// Command vaebench times the Vulkan VAE decoder at a sweep of latent sizes
// and reports the activation arena each one needs. The target is a 128x128
// latent, which is the 1024x1024 image Z-Image-Turbo generates.
//
// With -ladder it instead sweeps the mid block's two kernels at one size and
// reports each build's own dispatches on GPU timestamps -- which is the
// measurement that matters for them, since a wall clock over the whole decode
// puts 3.0 s of convolution in front of a 24 ms attention.
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
	ladder := flag.Bool("ladder", false, "sweep the mid block's kernels at -size instead of timing decodes")
	size := flag.Int("size", 128, "latent size for -ladder")
	reps := flag.Int("reps", 3, "runs per configuration; the best is reported")
	attn := flag.String("attn", "", "mid-block attention kernel")
	gemm := flag.String("gemm", "", "mid-block projection kernel")
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
	// The mid block's matrix-core path (stage 7) needs all three; without
	// them NewGPUDecoder falls back to stage 2b's fp32 kernels.
	feat, err := phys.SupportedFeatures()
	must(err)
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16:             feat.Float16,
		CoopMatrix:          feat.CoopMatrix,
		SubgroupSizeControl: feat.SubgroupSizeControl,
	})
	must(err)
	defer dev.Destroy()
	fmt.Printf("device: %s\n\n", phys.Name)

	cpu, err := vae.LoadDecoder(*dir, vae.FluxConfig())
	must(err)

	if *ladder {
		runLadder(dev, cpu, *size, *reps)
		return
	}

	fmt.Printf("%-8s %-12s %10s %12s %12s\n", "LATENT", "IMAGE", "DISPATCH", "ACT MB", "WALL")
	for _, tok := range strings.Split(*sizes, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(tok))
		if err != nil {
			continue
		}
		g, err := vae.NewGPUDecoderOpts(dev, cpu, n, n, vae.Options{
			Attn: vae.AttnKernel(*attn), GEMM: vae.GEMMKernel(*gemm),
		})
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

// runLadder times every build of the mid block's two kernels. Each is
// measured on its own dispatches rather than on the decode: GPU timestamps,
// best of reps, the attention kernel's one dispatch and the four projections'
// separately, so a kernel that is 1% of the decode is still measured to 1%.
func runLadder(dev *vk.Device, cpu *vae.Decoder, n, reps int) {
	latent := vae.NewTensor(1, 16, n, n)
	rng := rand.New(rand.NewSource(1))
	for i := range latent.Data {
		latent.Data[i] = float32(rng.NormFloat64())
	}

	fmt.Printf("latent %dx%d -> image %dx%d, best of %d\n\n", n, n, n*8, n*8, reps)
	fmt.Printf("%-22s %-22s %12s %12s %12s %12s\n", "ATTN", "GEMM", "ATTENTION", "TFLOP/s", "4x PROJ", "TFLOP/s")

	run := func(o vae.Options) {
		g, err := vae.NewGPUDecoderOpts(dev, cpu, n, n, o)
		if err != nil {
			fmt.Printf("%-22s %-22s failed: %v\n", o.Attn, o.GEMM, err)
			return
		}
		defer g.Destroy()
		var attn, gemm time.Duration
		var attnF, gemmF float64
		for i := 0; i < reps; i++ {
			st, err := g.Profile(latent)
			must(err)
			var a, m time.Duration
			var af, mf float64
			for _, s := range st {
				switch {
				case strings.HasPrefix(s.Kind, "attention"):
					a += s.GPU
					af += s.Flops
				case strings.HasPrefix(s.Kind, "gemm"), strings.HasPrefix(s.Kind, "linear"):
					m += s.GPU
					mf += s.Flops
				}
			}
			if i == 0 || a < attn {
				attn, attnF = a, af
			}
			if i == 0 || m < gemm {
				gemm, gemmF = m, mf
			}
		}
		ak, gk := g.Kernels()
		fmt.Printf("%-22s %-22s %12s %12.1f %12s %12.1f\n", ak, gk,
			attn.Round(time.Microsecond), attnF/attn.Seconds()/1e12,
			gemm.Round(time.Microsecond), gemmF/gemm.Seconds()/1e12)
	}

	for _, k := range vae.AttnKernels() {
		run(vae.Options{Attn: k})
	}
	fmt.Println()
	for _, k := range vae.GEMMKernels() {
		run(vae.Options{GEMM: k})
	}
}
