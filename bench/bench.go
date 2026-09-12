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
	NsPerIter    float64
	GFLOPS       float64 // 0 if not a compute-bound op
	GBPS         float64 // 0 if not a bandwidth-bound op
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
func TimeDispatch(pipe *vk.ComputePipeline, groupsX, groupsY, groupsZ, warmup, iters uint32, pushConstants []byte) (float64, error) {
	probe, err := pipe.DispatchTimed(groupsX, groupsY, groupsZ, 1, pushConstants)
	if err != nil {
		return 0, err
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
			return 0, err
		}
	}
	actualIters := capToBudget(iters)
	d, err := pipe.DispatchTimed(groupsX, groupsY, groupsZ, actualIters, pushConstants)
	if err != nil {
		return 0, err
	}
	return float64(d.Nanoseconds()) / float64(actualIters), nil
}

// PrintTable writes a human-readable results table to w.
func PrintTable(w io.Writer, results []Result) {
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "OP\tVARIANT\tFORMAT\tBLOCK\tSIZE\tNS/ITER\tGFLOP/S\tGB/S")
	for _, r := range results {
		gflops, gbps, block := "-", "-", "-"
		if r.GFLOPS > 0 {
			gflops = fmt.Sprintf("%.2f", r.GFLOPS)
		}
		if r.GBPS > 0 {
			gbps = fmt.Sprintf("%.2f", r.GBPS)
		}
		if r.BlockSize > 0 {
			block = fmt.Sprintf("%d", r.BlockSize)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%.1f\t%s\t%s\n",
			r.Op, r.Variant, r.WeightFormat, block, r.Size, r.NsPerIter, gflops, gbps)
	}
	tw.Flush()
}

// WriteCSV writes results to path in the format described in the plan:
// op,variant,weight_format,block_size,size,ns_per_iter,gflops,gbps.
func WriteCSV(path string, results []Result) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Fprintln(f, "op,variant,weight_format,block_size,size,ns_per_iter,gflops,gbps")
	for _, r := range results {
		fmt.Fprintf(f, "%s,%s,%s,%d,%d,%.3f,%.4f,%.4f\n",
			r.Op, r.Variant, r.WeightFormat, r.BlockSize, r.Size, r.NsPerIter, r.GFLOPS, r.GBPS)
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
