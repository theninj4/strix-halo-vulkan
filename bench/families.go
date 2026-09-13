package bench

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"strix-halo-vulkan/vk"
)

// Params is one invocation's worth of sweep parameters. Every family reads
// the subset it needs from the same struct so the CLI can hand the registry
// a single value rather than knowing each family's argument list.
type Params struct {
	Sizes   []int // matrix/vector dimensions (gemv, gemm, reduce)
	BWSizes []int // element counts (bandwidth, elementwise)
	Blocks  []int // quantization block sizes

	ColdFootprints []int // weight footprints in MB (gemv_cold)
	ColdN          int   // reduction length (gemv_cold)

	StridePads       []int // row padding in bytes (stride)
	StrideFootprints []int // touched footprints in MB (stride)
	StrideRowBytes   []int // bytes touched per row (stride)

	Warmup uint32
	Iters  uint32
	CUs    int // compute units, only to express peak rates as ops/clock/CU
}

// DefaultParams are the sweeps each family runs when the CLI is given no
// overriding flags.
func DefaultParams() Params {
	return Params{
		Sizes: []int{256, 512, 1024, 2048, 4096},
		// Each bandwidth/elementwise element is one fp32 read plus one fp32
		// write, so the footprint touched per iteration is 8 bytes per
		// element: this list runs from 2MB up to 512MB, deliberately
		// fine-grained around this chip's ~32MB last-level cache so the
		// cache-to-DRAM cliff can be located rather than merely straddled.
		BWSizes: []int{262144, 524288, 1048576, 2097152, 3145728, 4194304,
			6291456, 8388608, 12582912, 16777216, 33554432, 67108864},
		Blocks:           []int{32, 64, 128},
		ColdFootprints:   []int{16, 64, 256},
		ColdN:            4096,
		StridePads:       StridePadsBytes,
		StrideFootprints: StrideFootprintsMB,
		StrideRowBytes:   StrideRowBytesList,
		Warmup:           3,
		Iters:            20,
		CUs:              40,
	}
}

// Family is one independently-runnable group of measurements: the unit a
// bench invocation targets and the unit one CSV file holds. Splitting the
// suite this way is what keeps a run cheap — the whole suite takes long
// enough, and produces enough rows, that running it to answer one question
// is mostly waste.
type Family struct {
	Name string
	Desc string
	Run  func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error)
	// Summary optionally prints a family-specific digest under the results
	// table — the cross-row reading (a ceiling, a grid) that the flat table
	// cannot show. It is given only this family's results.
	Summary func(w io.Writer, results []Result, p Params)
}

// families is in run order: peak and overhead come first because they
// establish the ceiling and the floor every other number should be read
// against, and stride follows bandwidth because it is that number's
// qualifier.
var families = []Family{
	{
		Name: "peak",
		Desc: "instruction-issue ceilings (fp32/fp16 FMA, int8 dot, coopmat) from register-resident kernels",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunPeak(dev, phys, p.Warmup, p.Iters)
		},
		Summary: func(w io.Writer, results []Result, p Params) { PrintPeakSummary(w, results, p.CUs) },
	},
	{
		Name: "overhead",
		Desc: "dispatch + barrier cost of a shader that does nothing",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunOverhead(dev, p.Warmup, p.Iters)
		},
	},
	{
		Name: "bandwidth",
		Desc: "contiguous read+write sweep across the cache-to-DRAM cliff",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunBandwidth(dev, p.BWSizes, p.Warmup, p.Iters)
		},
	},
	{
		Name: "stride",
		Desc: "what a fixed byte set costs as row stride and request shape vary",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunStride(dev, p.StrideFootprints, p.StrideRowBytes, p.StridePads, p.Warmup, p.Iters)
		},
		Summary: func(w io.Writer, results []Result, p Params) { PrintStrideSummary(w, results) },
	},
	{
		Name: "elementwise",
		Desc: "fp32/fp16 elementwise kernels over the bandwidth sweep",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunElementwise(dev, p.BWSizes, p.Warmup, p.Iters)
		},
	},
	{
		Name: "gemv",
		Desc: "y = W*x (decode shape) across naive/subgroup/wave32 and fp32/fp16/q8/q4/W8A8/W4A8",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunGEMV(dev, phys, p.Sizes, p.Blocks, p.Warmup, p.Iters)
		},
	},
	{
		Name: "gemv_cold",
		Desc: "the same GEMV against weights several times larger than the last-level cache",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunGEMVCold(dev, phys, p.ColdFootprints, p.ColdN, p.Blocks, p.Warmup, p.Iters)
		},
	},
	{
		Name: "gemm",
		Desc: "C = A*B across naive/tiled/cooperative-matrix/W8A8 and fp32/fp16/q8/q4",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunGEMM(dev, phys, p.Sizes, p.Blocks, p.Warmup, p.Iters)
		},
	},
	{
		Name: "gemm_wmma",
		Desc: "the hand-tuned WMMA GEMM ladder (register/LDS/workgroup tilings)",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunGEMMWMMA(dev, phys, p.Sizes, p.Warmup, p.Iters)
		},
	},
	{
		Name: "shapes",
		Desc: "the models in GOALS.md at their own (M;N;K) instead of the square sweep",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunShapes(dev, phys, p.Warmup, p.Iters)
		},
		Summary: func(w io.Writer, results []Result, p Params) { PrintShapesSummary(w, results, p) },
	},
	{
		Name: "moe",
		Desc: "the grouped/MoE GEMM, the per-expert dispatch loop it replaces, and the gather/combine around it",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunMoE(dev, phys, p.Warmup, p.Iters)
		},
		Summary: func(w io.Writer, results []Result, p Params) { PrintMoESummary(w, results, p) },
	},
	{
		Name: "reduce",
		Desc: "softmax/rmsnorm reductions, shared-memory and subgroup",
		Run: func(dev *vk.Device, phys *vk.PhysicalDevice, p Params) ([]Result, error) {
			return RunReductions(dev, phys, p.Sizes, p.Warmup, p.Iters)
		},
	},
}

// Families returns the registry in run order.
func Families() []Family { return families }

// FamilyNames returns every family's name, in run order.
func FamilyNames() []string {
	names := make([]string, len(families))
	for i, f := range families {
		names[i] = f.Name
	}
	return names
}

// SelectFamilies resolves the named targets to families, in registry run
// order regardless of the order they were named, with duplicates dropped.
// The name "all" expands to the whole registry. An unknown name is an error
// rather than a silently empty run.
func SelectFamilies(names []string) ([]Family, error) {
	want := map[string]bool{}
	for _, n := range names {
		n = strings.TrimSpace(n)
		if n == "" {
			continue
		}
		if n == "all" {
			return families, nil
		}
		if _, ok := lookupFamily(n); !ok {
			return nil, fmt.Errorf("unknown target %q (have: %s, all)", n, strings.Join(FamilyNames(), ", "))
		}
		want[n] = true
	}
	if len(want) == 0 {
		return nil, fmt.Errorf("no target named (have: %s, all)", strings.Join(FamilyNames(), ", "))
	}
	var out []Family
	for _, f := range families {
		if want[f.Name] {
			out = append(out, f)
		}
	}
	return out, nil
}

func lookupFamily(name string) (Family, bool) {
	for _, f := range families {
		if f.Name == name {
			return f, true
		}
	}
	return Family{}, false
}

// WriteFamilyCSV writes one family's results to dir/<name>.csv, creating dir
// if needed, and returns the path written. One file per family (rather than
// one appended file for the whole suite) is what lets a run that targets a
// single family refresh only that family's numbers and leave the rest of the
// committed results alone.
func WriteFamilyCSV(dir, name string, results []Result) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, name+".csv")
	if err := WriteCSV(path, results); err != nil {
		return "", err
	}
	return path, nil
}
