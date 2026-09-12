package bench

import (
	"fmt"
	"math"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// reduceRows is fixed across the sweep; `sizes` sweeps the row width (n) —
// the dimension the max/sum reductions run over (e.g. a transformer's
// hidden size).
const reduceRows = 32

func cpuRMSNorm(x, weight []float32, rows, n int, eps float32) []float32 {
	y := make([]float32, rows*n)
	for r := 0; r < rows; r++ {
		base := r * n
		var sumSq float32
		for i := 0; i < n; i++ {
			v := x[base+i]
			sumSq += v * v
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(n) + eps)))
		for i := 0; i < n; i++ {
			y[base+i] = (x[base+i] / rms) * weight[i]
		}
	}
	return y
}

func cpuSoftmax(x []float32, rows, n int) []float32 {
	y := make([]float32, rows*n)
	for r := 0; r < rows; r++ {
		base := r * n
		m := x[base]
		for i := 1; i < n; i++ {
			if x[base+i] > m {
				m = x[base+i]
			}
		}
		var sum float32
		for i := 0; i < n; i++ {
			sum += float32(math.Exp(float64(x[base+i] - m)))
		}
		for i := 0; i < n; i++ {
			y[base+i] = float32(math.Exp(float64(x[base+i]-m))) / sum
		}
	}
	return y
}

// RunReductions measures RMSNorm and softmax — the normalization ops in
// every transformer block — across shared-memory-tree and subgroup-op
// reduction strategies.
func RunReductions(dev *vk.Device, sizes []int, warmup, iters uint32) ([]Result, error) {
	var results []Result

	rmsShared, err := dev.NewShaderModule(shaders.RMSNormShared)
	if err != nil {
		return nil, err
	}
	defer rmsShared.Destroy()
	rmsSub, err := dev.NewShaderModule(shaders.RMSNormSubgroup)
	if err != nil {
		return nil, err
	}
	defer rmsSub.Destroy()
	smShared, err := dev.NewShaderModule(shaders.SoftmaxShared)
	if err != nil {
		return nil, err
	}
	defer smShared.Destroy()
	smSub, err := dev.NewShaderModule(shaders.SoftmaxSubgroup)
	if err != nil {
		return nil, err
	}
	defer smSub.Destroy()

	for _, n := range sizes {
		r, err := runRMSNorm(dev, rmsShared, "shared", n, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("rmsnorm shared size=%d: %w", n, err)
		}
		results = append(results, r)

		r, err = runRMSNorm(dev, rmsSub, "subgroup", n, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("rmsnorm subgroup size=%d: %w", n, err)
		}
		results = append(results, r)

		r, err = runSoftmax(dev, smShared, "shared", n, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("softmax shared size=%d: %w", n, err)
		}
		results = append(results, r)

		r, err = runSoftmax(dev, smSub, "subgroup", n, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("softmax subgroup size=%d: %w", n, err)
		}
		results = append(results, r)
	}
	return results, nil
}

const rmsEps = float32(1e-6)

func runRMSNorm(dev *vk.Device, mod *vk.ShaderModule, variant string, n int, warmup, iters uint32) (Result, error) {
	rows := reduceRows
	xData := randomFloats(rows * n)
	wData := randomFloats(n)

	x, err := dev.NewBuffer(rows * n * 4)
	if err != nil {
		return Result{}, err
	}
	defer x.Destroy()
	x.WriteFloat32(xData)

	w, err := dev.NewBuffer(n * 4)
	if err != nil {
		return Result{}, err
	}
	defer w.Destroy()
	w.WriteFloat32(wData)

	y, err := dev.NewBuffer(rows * n * 4)
	if err != nil {
		return Result{}, err
	}
	defer y.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{Buffers: []*vk.Buffer{x, w, y}, PushConstantSize: 12})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := newPC().U32(uint32(rows)).U32(uint32(n)).F32(rmsEps).Bytes()

	if _, err := pipe.DispatchTimed(uint32(rows), 1, 1, 1, pc); err != nil {
		return Result{}, err
	}
	got := y.ReadFloat32(rows * n)
	want := cpuRMSNorm(xData, wData, rows, n, rmsEps)
	if err := compareVec(got, want, 1e-3); err != nil {
		return Result{}, fmt.Errorf("correctness check failed: %w", err)
	}

	ns, err := TimeDispatch(pipe, uint32(rows), 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	bytesMoved := float64(rows) * float64(n) * 4 * 3 // read x, read weight, write y
	return Result{
		Op: "rmsnorm", Variant: variant, Size: n,
		NsPerIter: ns,
		GBPS:      bytesMoved / (ns / 1e9) / 1e9,
	}, nil
}

func runSoftmax(dev *vk.Device, mod *vk.ShaderModule, variant string, n int, warmup, iters uint32) (Result, error) {
	rows := reduceRows
	xData := randomFloats(rows * n)

	x, err := dev.NewBuffer(rows * n * 4)
	if err != nil {
		return Result{}, err
	}
	defer x.Destroy()
	x.WriteFloat32(xData)

	y, err := dev.NewBuffer(rows * n * 4)
	if err != nil {
		return Result{}, err
	}
	defer y.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{Buffers: []*vk.Buffer{x, y}, PushConstantSize: 8})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := newPC().U32(uint32(rows)).U32(uint32(n)).Bytes()

	if _, err := pipe.DispatchTimed(uint32(rows), 1, 1, 1, pc); err != nil {
		return Result{}, err
	}
	got := y.ReadFloat32(rows * n)
	want := cpuSoftmax(xData, rows, n)
	if err := compareVec(got, want, 1e-3); err != nil {
		return Result{}, fmt.Errorf("correctness check failed: %w", err)
	}

	ns, err := TimeDispatch(pipe, uint32(rows), 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	bytesMoved := float64(rows) * float64(n) * 4 * 2 // one logical read + one write
	return Result{
		Op: "softmax", Variant: variant, Size: n,
		NsPerIter: ns,
		GBPS:      bytesMoved / (ns / 1e9) / 1e9,
	}, nil
}
