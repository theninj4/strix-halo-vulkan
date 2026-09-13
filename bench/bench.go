// Package bench times the Vulkan compute kernels in shaders/ across the
// "flavours" (naive/tiled/subgroup/cooperative-matrix; fp32/fp16/int8/int4
// weights) that matter for neural-network inference, using GPU-side
// timestamp queries (vk.ComputePipeline.DispatchTimed) rather than
// wall-clock time.
package bench

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"strings"
	"text/tabwriter"

	"strix-halo-vulkan/vk"
)

// Result is one measured data point.
type Result struct {
	Op           string
	Variant      string
	WeightFormat string // "fp32", "fp16", "q8", "q4", or "" if not applicable
	BlockSize    int    // quantization block size, 0 if not applicable
	Size         int    // case-defined primary size (vector/matrix dimension)
	// Detail is an optional free-form description of anything else that
	// distinguishes this case (e.g. the fixed dimension of a non-square
	// shape). Must contain no commas, since it becomes a CSV field.
	Detail    string
	NsPerIter float64
	GFLOPS    float64 // 0 if not a compute-bound op
	GBPS      float64 // 0 if not a bandwidth-bound op
	// Clocks records the GPU's shader clock and power draw while this
	// measurement ran. Zero when no instrumentation was installed (see
	// SetInstruments). Without it a result taken at the 636 MHz idle clock
	// is indistinguishable from a slow kernel measured at 2900 MHz.
	Clocks ClockStats
}

// maxBatchNs bounds how long any single DispatchTimed call (a batch of
// back-to-back dispatches recorded into one command buffer) is allowed to
// run. Slow kernels (e.g. naive GEMM at large N) can take seconds per
// iteration; batching a fixed iteration count regardless of that risks
// exceeding the kernel driver's GPU-hang watchdog (amdgpu's default TDR is
// commonly ~10s) and getting VK_ERROR_DEVICE_LOST, which poisons the device
// for every case still to run. 500ms leaves ample margin under that.
const maxBatchNs = 500_000_000

// TimeDispatch measures nanoseconds/iteration for one dispatch shape. It
// first probes with a single iteration (which also serves as one warmup
// dispatch) to estimate per-iteration cost, then caps both the remaining
// warmup dispatches and the timed batch to maxBatchNs worth of iterations —
// so a very slow kernel is measured with fewer iterations rather than
// risking a driver timeout.
//
// If instrumentation is installed (SetInstruments), the GPU is first driven
// with ALU-heavy work to lift it off its idle clock, and the shader clock
// and power are sampled throughout the timed batch and returned alongside
// the timing. Both steps exist because this part idles at 636 MHz of a
// 2900 MHz maximum, so an un-warmed measurement can understate a kernel by
// several times — and silently, unless the clock is recorded with it.
func TimeDispatch(pipe *vk.ComputePipeline, groupsX, groupsY, groupsZ, warmup, iters uint32, pushConstants []byte) (float64, ClockStats, error) {
	if err := instruments.Warm(); err != nil {
		return 0, ClockStats{}, fmt.Errorf("clock warmup: %w", err)
	}
	probe, err := pipe.DispatchTimed(groupsX, groupsY, groupsZ, 1, pushConstants)
	if err != nil {
		return 0, ClockStats{}, err
	}
	probeNs := float64(probe.Nanoseconds())
	if probeNs <= 0 {
		probeNs = 1
	}
	capToBudget := func(n uint32) uint32 {
		maxByBudget := uint32(maxBatchNs / probeNs)
		if maxByBudget < 1 {
			maxByBudget = 1
		}
		if n > maxByBudget {
			return maxByBudget
		}
		return n
	}

	if warmup > 1 {
		if _, err := pipe.DispatchTimed(groupsX, groupsY, groupsZ, capToBudget(warmup-1), pushConstants); err != nil {
			return 0, ClockStats{}, err
		}
	}
	actualIters := capToBudget(iters)
	stop := instruments.Watch()
	d, err := pipe.DispatchTimed(groupsX, groupsY, groupsZ, actualIters, pushConstants)
	clocks := stop()
	if err != nil {
		return 0, ClockStats{}, err
	}
	return float64(d.Nanoseconds()) / float64(actualIters), clocks, nil
}

// TimeDispatchSequence is TimeDispatch for a sequence of dispatches that
// differ in their push constants — IDEAS §3.5's one-dispatch-per-expert
// baseline. One "iteration" is the whole sequence, so the returned
// nanoseconds are per *sequence*, which is what the grouped kernel's single
// dispatch is being compared against.
//
// The budget cap works the same way, and matters more here: a 512-dispatch
// sequence recorded 70 times is 36000 commands in one command buffer, and
// the probe is what keeps that from becoming a driver timeout on a shape
// that turns out to be slow.
func TimeDispatchSequence(pipe *vk.ComputePipeline, groupsX []uint32, groupsY, groupsZ, warmup, iters uint32, pushConstants [][]byte) (float64, ClockStats, error) {
	if err := instruments.Warm(); err != nil {
		return 0, ClockStats{}, fmt.Errorf("clock warmup: %w", err)
	}
	probe, err := pipe.DispatchSequenceTimed(groupsX, groupsY, groupsZ, 1, pushConstants)
	if err != nil {
		return 0, ClockStats{}, err
	}
	probeNs := float64(probe.Nanoseconds())
	if probeNs <= 0 {
		probeNs = 1
	}
	capToBudget := func(n uint32) uint32 {
		maxByBudget := uint32(maxBatchNs / probeNs)
		if maxByBudget < 1 {
			maxByBudget = 1
		}
		if n > maxByBudget {
			return maxByBudget
		}
		return n
	}

	if warmup > 1 {
		if _, err := pipe.DispatchSequenceTimed(groupsX, groupsY, groupsZ, capToBudget(warmup-1), pushConstants); err != nil {
			return 0, ClockStats{}, err
		}
	}
	actualIters := capToBudget(iters)
	stop := instruments.Watch()
	d, err := pipe.DispatchSequenceTimed(groupsX, groupsY, groupsZ, actualIters, pushConstants)
	clocks := stop()
	if err != nil {
		return 0, ClockStats{}, err
	}
	return float64(d.Nanoseconds()) / float64(actualIters), clocks, nil
}

// PrintTable writes a human-readable results table to w.
func PrintTable(w io.Writer, results []Result) {
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "OP\tVARIANT\tFORMAT\tBLOCK\tSIZE\tNS/ITER\tGFLOP/S\tGB/S\tSCLK MHZ\tW")
	for _, r := range results {
		gflops, gbps, block, sclk, power := "-", "-", "-", "-", "-"
		if r.GFLOPS > 0 {
			gflops = fmt.Sprintf("%.2f", r.GFLOPS)
		}
		if r.GBPS > 0 {
			gbps = fmt.Sprintf("%.2f", r.GBPS)
		}
		if r.BlockSize > 0 {
			block = fmt.Sprintf("%d", r.BlockSize)
		}
		if r.Clocks.Samples > 0 {
			// min/max as well as mean: a wide spread means the clock moved
			// during the measurement, which invalidates the comparison
			// rather than merely annotating it.
			sclk = fmt.Sprintf("%.0f (%.0f-%.0f)", r.Clocks.SclkMHz, r.Clocks.SclkMinMHz, r.Clocks.SclkMaxMHz)
			power = fmt.Sprintf("%.1f", r.Clocks.PowerW)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%.1f\t%s\t%s\t%s\t%s\n",
			r.Op, r.Variant, r.WeightFormat, block, r.Size, r.NsPerIter, gflops, gbps, sclk, power)
	}
	tw.Flush()
}

// WriteCSV writes results to path as
// op,variant,weight_format,block_size,size,ns_per_iter,gflops,gbps
// followed by sclk_mhz,sclk_mhz_min,sclk_mhz_max,power_w (zero when no
// instrumentation was installed) and detail. The new columns are appended
// rather than interleaved so existing column-indexed analysis of the first
// eight keeps working.
func WriteCSV(path string, results []Result) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Fprintln(f, "op,variant,weight_format,block_size,size,ns_per_iter,gflops,gbps,sclk_mhz,sclk_mhz_min,sclk_mhz_max,power_w,detail")
	for _, r := range results {
		fmt.Fprintf(f, "%s,%s,%s,%d,%d,%.3f,%.4f,%.4f,%.1f,%.1f,%.1f,%.2f,%s\n",
			r.Op, r.Variant, r.WeightFormat, r.BlockSize, r.Size, r.NsPerIter, r.GFLOPS, r.GBPS,
			r.Clocks.SclkMHz, r.Clocks.SclkMinMHz, r.Clocks.SclkMaxMHz, r.Clocks.PowerW,
			strings.ReplaceAll(r.Detail, ",", ";"))
	}
	return nil
}

func randomFloats(n int) []float32 {
	r := rand.New(rand.NewSource(42))
	out := make([]float32, n)
	for i := range out {
		out[i] = r.Float32()*2 - 1 // [-1, 1)
	}
	return out
}

func groupsFor(n, localSize int) uint32 {
	return uint32((n + localSize - 1) / localSize)
}

func approxEqual(a, b, relTol float32) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	scale := a
	if scale < 0 {
		scale = -scale
	}
	if b < 0 && -b > scale {
		scale = -b
	}
	if scale < 1 {
		scale = 1
	}
	return diff <= relTol*scale
}
