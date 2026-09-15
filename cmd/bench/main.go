// Command bench runs the Vulkan compute benchmark suite for
// neural-network-inference-shaped operations (GEMM, GEMV, elementwise,
// reductions) across implementation flavours (naive/tiled/subgroup/
// cooperative-matrix; fp32/fp16/int8/int4 weights) on the local GPU,
// picking the Strix Halo iGPU if present.
//
// A run targets named op families rather than sweeping everything: the full
// suite takes long enough, and produces enough rows, that running it to
// answer one question is mostly waste. Each family's results are written to
// their own CSV under -resultsdir, so re-running one family refreshes only
// that file.
//
//	go run ./cmd/bench -list          # what can be targeted
//	go run ./cmd/bench gemv gemv_cold # two families
//	go run ./cmd/bench all            # everything, the old behaviour
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"strix-halo-vulkan/bench"
	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func main() {
	def := bench.DefaultParams()
	list := flag.Bool("list", false, "list the op families a run can target, then exit")
	resultsDir := flag.String("resultsdir", "results", "directory to write one CSV per targeted family into (empty disables CSV output)")
	sizesFlag := flag.String("sizes", intsToFlag(def.Sizes), "comma-separated matrix/vector dimensions to sweep (gemv, gemm, gemm_wmma, reduce)")
	bwSizesFlag := flag.String("bwsizes", intsToFlag(def.BWSizes), "comma-separated element counts to sweep for bandwidth/elementwise (footprint per iteration is 8 bytes/element; needs millions of elements to leave the dispatch-overhead-bound regime and reach steady-state memory bandwidth)")
	blocksFlag := flag.String("blocks", intsToFlag(def.Blocks), "comma-separated quantization block sizes to sweep")
	warmup := flag.Uint("warmup", uint(def.Warmup), "untimed warmup dispatches before each timed measurement")
	iters := flag.Uint("iters", uint(def.Iters), "back-to-back timed dispatches averaged per measurement")
	stridePadsFlag := flag.String("stridepads", "", "comma-separated row-padding values in bytes for the strided-read sweep (default: IDEAS §5.1b's sweep across the 2 KB channel-interleave period); each must be a multiple of 16")
	strideFootprintsFlag := flag.String("stridefootprints", "", "comma-separated touched footprints in MB for the strided-read sweep (default: one under and one over the 32 MB last-level cache)")
	strideRowBytesFlag := flag.String("striderowbytes", "", "comma-separated bytes-touched-per-row for the strided-read sweep (default: a K=4096 and a K=1024 fp16 weight row); each must be a multiple of 1024")
	coldFootprintsFlag := flag.String("coldfootprints", intsToFlag(def.ColdFootprints), "comma-separated weight footprints in MB (fp16-equivalent) for the DRAM-resident gemv_cold sweep; entries well above the ~32MB last-level cache are the ones that measure real decode")
	coldN := flag.Int("coldn", def.ColdN, "reduction length N for the gemv_cold sweep; its row count M is derived from each footprint")
	bankGiB := flag.Int("bankgib", def.BankGiB, "total weight bank to allocate for the bank family, in GiB; allocation stops early and the sweep shortens if the device will not give this much")
	bankReadMiB := flag.Int("bankreadmib", def.BankReadMiB, "bytes read per timed step in the bank family, in MiB; held constant while the bank grows, so GB/s isolates the working set")
	warmClock := flag.Duration("warmclock", 3*time.Second, "maximum ALU-heavy warmup before each timed measurement; stops as soon as the GPU reaches its top advertised clock, so a hot GPU costs one short burst (0 disables)")
	clockSample := flag.Duration("clocksample", time.Millisecond, "sampling period for the sclk/power counters recorded with each measurement")
	cus := flag.Int("cus", def.CUs, "compute-unit count, used only to express the measured peak rates as ops/clock/CU")
	flag.Usage = usage
	flag.Parse()

	if *list {
		printFamilies(os.Stdout)
		return
	}

	families, err := bench.SelectFamilies(flag.Args())
	if err != nil {
		fmt.Fprintf(os.Stderr, "bench: %v\n\n", err)
		printFamilies(os.Stderr)
		os.Exit(2)
	}

	p := def
	if p.Sizes, err = parseInts(*sizesFlag); err != nil {
		log.Fatalf("-sizes: %v", err)
	}
	if p.BWSizes, err = parseInts(*bwSizesFlag); err != nil {
		log.Fatalf("-bwsizes: %v", err)
	}
	if p.Blocks, err = parseInts(*blocksFlag); err != nil {
		log.Fatalf("-blocks: %v", err)
	}
	if p.ColdFootprints, err = parseInts(*coldFootprintsFlag); err != nil {
		log.Fatalf("-coldfootprints: %v", err)
	}
	if *stridePadsFlag != "" {
		if p.StridePads, err = parseInts(*stridePadsFlag); err != nil {
			log.Fatalf("-stridepads: %v", err)
		}
	}
	if *strideFootprintsFlag != "" {
		if p.StrideFootprints, err = parseInts(*strideFootprintsFlag); err != nil {
			log.Fatalf("-stridefootprints: %v", err)
		}
	}
	if *strideRowBytesFlag != "" {
		if p.StrideRowBytes, err = parseInts(*strideRowBytesFlag); err != nil {
			log.Fatalf("-striderowbytes: %v", err)
		}
	}
	p.ColdN = *coldN
	p.BankGiB = *bankGiB
	p.BankReadMiB = *bankReadMiB
	p.Warmup = uint32(*warmup)
	p.Iters = uint32(*iters)
	p.CUs = *cus

	cfg := config{
		params:      p,
		families:    families,
		warmClock:   *warmClock,
		clockSample: *clockSample,
		resultsDir:  *resultsDir,
	}
	if err := run(cfg); err != nil {
		log.Fatal(err)
	}
}

func usage() {
	fmt.Fprintf(flag.CommandLine.Output(), "usage: bench [flags] <family> [family...]\n\n")
	printFamilies(flag.CommandLine.Output())
	fmt.Fprintf(flag.CommandLine.Output(), "\nflags:\n")
	flag.PrintDefaults()
}

func printFamilies(w io.Writer) {
	fmt.Fprintln(w, "op families a run can target:")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	for _, f := range bench.Families() {
		fmt.Fprintf(tw, "  %s\t%s\n", f.Name, f.Desc)
	}
	fmt.Fprintf(tw, "  %s\t%s\n", "all", "every family above, in the order listed")
	tw.Flush()
}

func intsToFlag(v []int) string {
	parts := make([]string, len(v))
	for i, n := range v {
		parts[i] = strconv.Itoa(n)
	}
	return strings.Join(parts, ",")
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

// config is one invocation: which families to run, the sweeps to run them
// over, and where to put the results.
type config struct {
	params                 bench.Params
	families               []bench.Family
	warmClock, clockSample time.Duration
	resultsDir             string
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
	fmt.Printf("features: fp16=%v int8=%v integerDotProduct=%v coopMatrix=%v subgroupSizeControl=%v\n",
		feat.Float16, feat.Int8, feat.IntegerDotProduct, feat.CoopMatrix, feat.SubgroupSizeControl)

	// IDEAS §6.2: the wave32 rows are only meaningful if the device will let
	// a pipeline pin its wave size, so report the range rather than leave a
	// silently-skipped family to be noticed in the CSV.
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		return err
	}
	fmt.Printf("subgroup sizes: supported=%v range=%d..%d fullSubgroups=%v\n",
		sgs.Supported, sgs.MinSubgroupSize, sgs.MaxSubgroupSize, sgs.ComputeFullSubgroups)

	queueFamily, err := phys.ComputeQueueFamily()
	if err != nil {
		return err
	}
	dev, err := vk.NewDevice(phys, queueFamily, vk.DeviceFeatures{
		Float16: true, Int8: true, IntegerDotProduct: true, CoopMatrix: true,
		SubgroupSizeControl: sgs.Supported,
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

	p := cfg.params
	fmt.Printf("running %s\n", strings.Join(familyNames(cfg.families), ", "))
	fmt.Printf("sweeping sizes=%v bwsizes=%v blocks=%v warmup=%d iters=%d\n", p.Sizes, p.BWSizes, p.Blocks, p.Warmup, p.Iters)
	fmt.Printf("instrumentation: %s\n\n", inst.Describe())

	// Each family's CSV is written as soon as that family finishes, so a run
	// that dies partway (or is interrupted) still leaves the families that
	// did complete on disk.
	var written []string
	for _, f := range cfg.families {
		fmt.Printf("== %s ==\n", f.Name)
		results, err := f.Run(dev, phys, p)
		if err != nil {
			return fmt.Errorf("%s: %w", f.Name, err)
		}
		bench.PrintTable(os.Stdout, results)
		if f.Summary != nil {
			f.Summary(os.Stdout, results, p)
		}
		if cfg.resultsDir != "" {
			path, err := bench.WriteFamilyCSV(cfg.resultsDir, f.Name, results)
			if err != nil {
				return fmt.Errorf("%s: %w", f.Name, err)
			}
			fmt.Printf("\nwrote %d results to %s\n", len(results), path)
			written = append(written, filepath.Base(path))
		}
		fmt.Println()
	}
	if len(written) > 0 {
		fmt.Printf("results in %s/: %s\n", cfg.resultsDir, strings.Join(written, " "))
	}
	return nil
}

func familyNames(families []bench.Family) []string {
	names := make([]string, len(families))
	for i, f := range families {
		names[i] = f.Name
	}
	return names
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
