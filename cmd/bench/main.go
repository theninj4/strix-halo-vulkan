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
	"time"

	"strix-halo-vulkan/bench"
	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func main() {
	sizesFlag := flag.String("sizes", "256,512,1024,2048,4096", "comma-separated matrix/vector dimensions to sweep (gemv, gemm, reductions)")
	// Each bandwidth/elementwise element is one fp32 read plus one fp32
	// write, so the footprint touched per iteration is 8 bytes per element:
	// this list runs from 2MB up to 512MB, deliberately fine-grained around
	// this chip's ~32MB last-level cache so the cache-to-DRAM cliff can be
	// located rather than merely straddled.
	bwSizesFlag := flag.String("bwsizes", "262144,524288,1048576,2097152,3145728,4194304,6291456,8388608,12582912,16777216,33554432,67108864", "comma-separated element counts to sweep for bandwidth/elementwise (footprint per iteration is 8 bytes/element; needs millions of elements to leave the dispatch-overhead-bound regime and reach steady-state memory bandwidth)")
	blocksFlag := flag.String("blocks", "32,64,128", "comma-separated quantization block sizes to sweep")
	warmup := flag.Uint("warmup", 3, "untimed warmup dispatches before each timed measurement")
	iters := flag.Uint("iters", 20, "back-to-back timed dispatches averaged per measurement")
	csvPath := flag.String("csv", "", "optional path to write results as CSV")
	skip := flag.String("skip", "", "comma-separated op families to skip (peak,overhead,bandwidth,stride,elementwise,gemv,gemv_cold,gemm,reduce)")
	stridePadsFlag := flag.String("stridepads", "", "comma-separated row-padding values in bytes for the strided-read sweep (default: IDEAS §5.1b's sweep across the 2 KB channel-interleave period); each must be a multiple of 16")
	strideFootprintsFlag := flag.String("stridefootprints", "", "comma-separated touched footprints in MB for the strided-read sweep (default: one under and one over the 32 MB last-level cache)")
	strideRowBytesFlag := flag.String("striderowbytes", "", "comma-separated bytes-touched-per-row for the strided-read sweep (default: a K=4096 and a K=1024 fp16 weight row); each must be a multiple of 1024")
	coldFootprintsFlag := flag.String("coldfootprints", "16,64,256", "comma-separated weight footprints in MB (fp16-equivalent) for the DRAM-resident gemv_cold sweep; entries well above the ~32MB last-level cache are the ones that measure real decode")
	coldN := flag.Int("coldn", 4096, "reduction length N for the gemv_cold sweep; its row count M is derived from each footprint")
	warmClock := flag.Duration("warmclock", 3*time.Second, "maximum ALU-heavy warmup before each timed measurement; stops as soon as the GPU reaches its top advertised clock, so a hot GPU costs one short burst (0 disables)")
	clockSample := flag.Duration("clocksample", time.Millisecond, "sampling period for the sclk/power counters recorded with each measurement")
	cus := flag.Int("cus", 40, "compute-unit count, used only to express the measured peak rates as ops/clock/CU")
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
	coldFootprints, err := parseInts(*coldFootprintsFlag)
	if err != nil {
		log.Fatalf("-coldfootprints: %v", err)
	}
	stridePads := bench.StridePadsBytes
	if *stridePadsFlag != "" {
		if stridePads, err = parseInts(*stridePadsFlag); err != nil {
			log.Fatalf("-stridepads: %v", err)
		}
	}
	strideFootprints := bench.StrideFootprintsMB
	if *strideFootprintsFlag != "" {
		if strideFootprints, err = parseInts(*strideFootprintsFlag); err != nil {
			log.Fatalf("-stridefootprints: %v", err)
		}
	}
	strideRowBytes := bench.StrideRowBytesList
	if *strideRowBytesFlag != "" {
		if strideRowBytes, err = parseInts(*strideRowBytesFlag); err != nil {
			log.Fatalf("-striderowbytes: %v", err)
		}
	}
	skipSet := map[string]bool{}
	for _, s := range strings.Split(*skip, ",") {
		if s = strings.TrimSpace(s); s != "" {
			skipSet[s] = true
		}
	}

	cfg := config{
		sizes: sizes, bwSizes: bwSizes, blocks: blocks,
		coldFootprints: coldFootprints, coldN: *coldN,
		stridePads: stridePads, strideFootprints: strideFootprints, strideRowBytes: strideRowBytes,
		warmup: uint32(*warmup), iters: uint32(*iters),
		warmClock: *warmClock, clockSample: *clockSample, cus: *cus,
		csvPath: *csvPath, skip: skipSet,
	}
	if err := run(cfg); err != nil {
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

// config is one invocation's worth of sweep parameters, gathered into a
// struct because run's parameter list had outgrown being readable.
type config struct {
	sizes, bwSizes, blocks []int
	coldFootprints         []int
	coldN                  int
	stridePads             []int
	strideFootprints       []int
	strideRowBytes         []int
	warmup, iters          uint32
	warmClock, clockSample time.Duration
	cus                    int
	csvPath                string
	skip                   map[string]bool
}

func run(cfg config) error {
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

	// Instrumentation is installed before any measurement: it both lifts
	// the GPU off its idle clock ahead of each timed batch and records the
	// clock alongside every result, so a number taken at a low clock is
	// visible as such instead of looking like a slow kernel.
	inst, err := bench.NewInstruments(dev, cfg.warmClock, cfg.clockSample)
	if err != nil {
		return err
	}
	defer inst.Destroy()
	bench.SetInstruments(inst)

	fmt.Printf("sweeping sizes=%v bwsizes=%v blocks=%v warmup=%d iters=%d\n", cfg.sizes, cfg.bwSizes, cfg.blocks, cfg.warmup, cfg.iters)
	fmt.Printf("instrumentation: %s\n\n", inst.Describe())

	var results []bench.Result
	run := func(name string, fn func() ([]bench.Result, error)) error {
		if cfg.skip[name] {
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

	// peak and overhead run first: they establish the ceiling and the floor
	// that every subsequent number should be read against.
	if err := run("peak", func() ([]bench.Result, error) { return bench.RunPeak(dev, phys, cfg.warmup, cfg.iters) }); err != nil {
		return err
	}
	bench.PrintPeakSummary(os.Stdout, results, cfg.cus)
	if len(results) > 0 {
		fmt.Println()
	}
	if err := run("overhead", func() ([]bench.Result, error) { return bench.RunOverhead(dev, cfg.warmup, cfg.iters) }); err != nil {
		return err
	}
	if err := run("bandwidth", func() ([]bench.Result, error) { return bench.RunBandwidth(dev, cfg.bwSizes, cfg.warmup, cfg.iters) }); err != nil {
		return err
	}
	// stride follows bandwidth because it is that number's qualifier: the
	// contiguous sweep bandwidth reports is the best case, and this says
	// what the same bytes cost at a channel-aliased stride or in a gather.
	if err := run("stride", func() ([]bench.Result, error) {
		return bench.RunStride(dev, cfg.strideFootprints, cfg.strideRowBytes, cfg.stridePads, cfg.warmup, cfg.iters)
	}); err != nil {
		return err
	}
	bench.PrintStrideSummary(os.Stdout, results)
	if err := run("elementwise", func() ([]bench.Result, error) { return bench.RunElementwise(dev, cfg.bwSizes, cfg.warmup, cfg.iters) }); err != nil {
		return err
	}
	if err := run("gemv", func() ([]bench.Result, error) {
		return bench.RunGEMV(dev, phys, cfg.sizes, cfg.blocks, cfg.warmup, cfg.iters)
	}); err != nil {
		return err
	}
	if err := run("gemv_cold", func() ([]bench.Result, error) {
		return bench.RunGEMVCold(dev, phys, cfg.coldFootprints, cfg.coldN, cfg.blocks, cfg.warmup, cfg.iters)
	}); err != nil {
		return err
	}
	if err := run("gemm", func() ([]bench.Result, error) {
		return bench.RunGEMM(dev, phys, cfg.sizes, cfg.blocks, cfg.warmup, cfg.iters)
	}); err != nil {
		return err
	}
	if err := run("reduce", func() ([]bench.Result, error) { return bench.RunReductions(dev, cfg.sizes, cfg.warmup, cfg.iters) }); err != nil {
		return err
	}

	if cfg.csvPath != "" {
		if err := bench.WriteCSV(cfg.csvPath, results); err != nil {
			return err
		}
		fmt.Printf("wrote %d results to %s\n", len(results), cfg.csvPath)
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
