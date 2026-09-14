package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"sort"
	"time"

	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/vae"
)

func main() {
	log.SetFlags(0)
	size := flag.Int("size", 32, "latent size")
	topN := flag.Int("top", 12, "slowest dispatches to list")
	attn := flag.String("attn", "", "mid-block attention kernel (\"scalar\" for stage 2b's fp32 path)")
	gemm := flag.String("gemm", "", "mid-block projection kernel")
	flag.Parse()
	inst, _ := vk.NewInstance("prof")
	defer inst.Destroy()
	devs, _ := inst.PhysicalDevices()
	phys := &devs[0]
	for i := range devs {
		if devs[i].DeviceID == 0x1586 {
			phys = &devs[i]
		}
	}
	qf, _ := phys.ComputeQueueFamily()
	// The mid block's matrix-core path (stage 7) needs all three; without
	// them NewGPUDecoder falls back to stage 2b's fp32 kernels.
	feat, err := phys.SupportedFeatures()
	if err != nil {
		log.Fatal(err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16:             feat.Float16,
		CoopMatrix:          feat.CoopMatrix,
		SubgroupSizeControl: feat.SubgroupSizeControl,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Destroy()

	cpu, err := vae.LoadDecoder("models/Z-Image-Turbo/vae", vae.FluxConfig())
	if err != nil {
		log.Fatal(err)
	}
	n := *size
	g, err := vae.NewGPUDecoderOpts(dev, cpu, n, n, vae.Options{
		Attn: vae.AttnKernel(*attn), GEMM: vae.GEMMKernel(*gemm),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer g.Destroy()
	lat := vae.NewTensor(1, 16, n, n)
	rng := rand.New(rand.NewSource(1))
	for i := range lat.Data {
		lat.Data[i] = float32(rng.NormFloat64())
	}
	st, err := g.Profile(lat)
	if err != nil {
		fmt.Printf("FAILED after %d dispatches: %v\n", len(st), err)
		if len(st) > 0 {
			last := st[len(st)-1]
			fmt.Printf("last completed: #%d %s in %s\n", last.Index, last.Kind, last.GPU)
		}
		return
	}
	var total time.Duration
	byKind := map[string]time.Duration{}
	for _, s := range st {
		total += s.GPU
		k := s.Kind
		for i, c := range k {
			if c == ' ' {
				k = k[:i]
				break
			}
		}
		byKind[k] += s.GPU
	}
	ak, gk := g.Kernels()
	fmt.Printf("latent %dx%d -> image %dx%d, %d dispatches, GPU total %s (attn %s, gemm %s)\n\n",
		n, n, n*8, n*8, len(st), total.Round(time.Millisecond), ak, gk)

	type kv struct {
		k string
		d time.Duration
	}
	var ks []kv
	for k, d := range byKind {
		ks = append(ks, kv{k, d})
	}
	sort.Slice(ks, func(i, j int) bool { return ks[i].d > ks[j].d })
	fmt.Println("by kind:")
	for _, e := range ks {
		fmt.Printf("  %-12s %10s  %5.1f%%\n", e.k, e.d.Round(time.Millisecond), 100*float64(e.d)/float64(total))
	}

	sort.Slice(st, func(i, j int) bool { return st[i].GPU > st[j].GPU })
	fmt.Println("\nslowest dispatches:")
	for i := 0; i < *topN && i < len(st); i++ {
		s := st[i]
		gf := s.Flops / s.GPU.Seconds() / 1e9
		fmt.Printf("  %-34s %9s  %5.1f%%  %8.0f GFLOP/s\n", s.Kind, s.GPU.Round(time.Microsecond), 100*float64(s.GPU)/float64(total), gf)
	}
}
