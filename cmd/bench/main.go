// Command bench runs the Vulkan compute benchmark suite for
// neural-network-inference-shaped operations (GEMM, GEMV, elementwise,
// reductions) across implementation flavours (naive/tiled/subgroup/
// cooperative-matrix; fp32/fp16/int8/int4 weights) on the local GPU,
// picking the Strix Halo iGPU if present.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"strix-halo-vulkan/bench"
	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func main() {
	sizesFlag := flag.String("sizes", "256,512,1024,2048,4096", "comma-separated matrix/vector dimensions to sweep (gemv, gemm, reductions)")
	bwSizesFlag := flag.String("bwsizes", "1048576,4194304,16777216,67108864", "comma-separated element counts to sweep for bandwidth/elementwise (needs millions of elements to leave the dispatch-overhead-bound regime and reach steady-state memory bandwidth)")
	blocksFlag := flag.String("blocks", "32,64,128", "comma-separated quantization block sizes to sweep")
	warmup := flag.Uint("warmup", 3, "untimed warmup dispatches before each timed measurement")
	iters := flag.Uint("iters", 20, "back-to-back timed dispatches averaged per measurement")
	csvPath := flag.String("csv", "", "optional path to write results as CSV")
	skip := flag.String("skip", "", "comma-separated op families to skip (bandwidth,elementwise,gemv,gemm,reduce)")
	flag.Parse()

	sizes, err := parseInts(*sizesFlag)
	if err != nil {
		log.Fatalf("-sizes: %v", err)
	}
	bwSizes, err := parseInts(*bwSizesFlag)
	if err != nil {
		log.Fatalf("-bwsizes: %v", err)
	}
	blocks, err := parseInts(*blocksFlag)
	if err != nil {
		log.Fatalf("-blocks: %v", err)
	}
	skipSet := map[string]bool{}
	for _, s := range strings.Split(*skip, ",") {
		if s = strings.TrimSpace(s); s != "" {
			skipSet[s] = true
		}
	}

	if err := run(sizes, bwSizes, blocks, uint32(*warmup), uint32(*iters), *csvPath, skipSet); err != nil {
		log.Fatal(err)
	}
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("%q: %w", part, err)
		}
		out = append(out, v)
	}
	return out, nil
}

func run(sizes, bwSizes, blocks []int, warmup, iters uint32, csvPath string, skip map[string]bool) error {
	instance, err := vk.NewInstance("strix-halo-bench")
	if err != nil {
		return err
	}
	defer instance.Destroy()

	devices, err := instance.PhysicalDevices()
	if err != nil {
		return err
	}
	phys := pickDevice(devices)
	fmt.Printf("device: %s (vendor 0x%04x, device 0x%04x)\n", phys.Name, phys.VendorID, phys.DeviceID)

	feat, err := phys.SupportedFeatures()
	if err != nil {
		return err
	}
	fmt.Printf("features: fp16=%v int8=%v integerDotProduct=%v coopMatrix=%v\n",
		feat.Float16, feat.Int8, feat.IntegerDotProduct, feat.CoopMatrix)

	queueFamily, err := phys.ComputeQueueFamily()
	if err != nil {
		return err
	}
	dev, err := vk.NewDevice(phys, queueFamily, vk.DeviceFeatures{
		Float16: true, Int8: true, IntegerDotProduct: true, CoopMatrix: true,
	})
	if err != nil {
		return err
	}
	defer dev.Destroy()

	fmt.Printf("sweeping sizes=%v bwsizes=%v blocks=%v warmup=%d iters=%d\n\n", sizes, bwSizes, blocks, warmup, iters)

	var results []bench.Result
	run := func(name string, fn func() ([]bench.Result, error)) error {
		if skip[name] {
			fmt.Printf("== %s (skipped) ==\n", name)
			return nil
		}
		fmt.Printf("== %s ==\n", name)
		r, err := fn()
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		bench.PrintTable(os.Stdout, r)
		fmt.Println()
		results = append(results, r...)
		return nil
	}

	if err := run("bandwidth", func() ([]bench.Result, error) { return bench.RunBandwidth(dev, bwSizes, warmup, iters) }); err != nil {
		return err
	}
	if err := run("elementwise", func() ([]bench.Result, error) { return bench.RunElementwise(dev, bwSizes, warmup, iters) }); err != nil {
		return err
	}
	if err := run("gemv", func() ([]bench.Result, error) { return bench.RunGEMV(dev, phys, sizes, blocks, warmup, iters) }); err != nil {
		return err
	}
	if err := run("gemm", func() ([]bench.Result, error) { return bench.RunGEMM(dev, phys, sizes, blocks, warmup, iters) }); err != nil {
		return err
	}
	if err := run("reduce", func() ([]bench.Result, error) { return bench.RunReductions(dev, sizes, warmup, iters) }); err != nil {
		return err
	}

	if csvPath != "" {
		if err := bench.WriteCSV(csvPath, results); err != nil {
			return err
		}
		fmt.Printf("wrote %d results to %s\n", len(results), csvPath)
	}
	return nil
}

// pickDevice prefers the Strix Halo iGPU by deviceID, falling back to the
// first device the instance reports.
func pickDevice(devices []vk.PhysicalDevice) *vk.PhysicalDevice {
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			return &devices[i]
		}
	}
	return &devices[0]
}
