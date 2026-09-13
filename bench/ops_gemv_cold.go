package bench

import (
	"fmt"
	"math"
	"os"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// ColdFootprintsMB is the swept weight footprint, in MB of *fp16-equivalent*
// weights — i.e. the size the same logical layer would occupy at fp16, so
// that every format in a sweep row holds the same number of weights and the
// only thing differing is bytes-per-weight. The quantized formats therefore
// touch a quarter (q4) or half (q8/w8a8) of the stated figure.
//
// The point of the sweep is to straddle this chip's ~32MB MALL cache: the
// small entries reproduce the cache-resident regime the existing square
// sweep measures, the large ones force every weight byte to come from DRAM
// the way real inference does.
var ColdFootprintsMB = []int{16, 64, 256}

// RunGEMVCold measures the decode-shape GEMV against weights far larger
// than the last-level cache, which is the regime real inference runs in and
// the existing RunGEMV sweep does not reach.
//
// Why the existing numbers need this correction: RunGEMV sweeps square
// M=N, so its largest case (4096) holds 33.5MB of fp16 weights, 16.8MB at
// int8 and 8.4MB at int4 — all at or under the 32MB MALL — and then the
// timing loop re-reads that same matrix hundreds of times. The reported
// 518 GB/s for W8A8 is over twice this machine's 236 GB/s of DRAM
// bandwidth, which is only possible out of cache. Decode streams the whole
// model per token from DRAM, so measuring one cache-resident layer
// over-reports it and, worse, flatters the formats that read *more* bytes,
// since bytes are free when they come from cache.
//
// Rather than round-robin several matrices, this simply makes one matrix
// big: within a single dispatch each weight is read exactly once, so there
// is no intra-dispatch reuse to defeat — a matrix several times the cache
// size guarantees every iteration refills from DRAM.
//
// Deliberately skips the CPU reference check. These kernels are the same
// ones RunGEMV already verifies against a CPU reference at small sizes, and
// a reference here would mean materialising a multi-hundred-MB fp32
// dequantized copy plus an O(M*N) CPU pass per case. Only a cheap sanity
// check on the output runs, which still catches gross breakage (wrong
// buffer size, unbound descriptor) without repeating the verification.
func RunGEMVCold(dev *vk.Device, phys *vk.PhysicalDevice, footprintsMB []int, N int, blocks []int, warmup, iters uint32) ([]Result, error) {
	feat, err := phys.SupportedFeatures()
	if err != nil {
		return nil, err
	}

	f16Mod, err := dev.NewShaderModule(shaders.GEMVSubgroupF16)
	if err != nil {
		return nil, err
	}
	defer f16Mod.Destroy()
	q8Mod, err := dev.NewShaderModule(shaders.GEMVSubgroupQ8)
	if err != nil {
		return nil, err
	}
	defer q8Mod.Destroy()
	q4Mod, err := dev.NewShaderModule(shaders.GEMVSubgroupQ4)
	if err != nil {
		return nil, err
	}
	defer q4Mod.Destroy()
	var w8a8Mod *vk.ShaderModule
	// The wave32 W4A8 arms (IDEAS §6.2) matter more here than in RunGEMV:
	// this is the DRAM-resident sweep, so it is where a wave-size effect on
	// decode would actually show up rather than being absorbed by the MALL.
	coldW4A8Variants := filterWaveVariants(w4a8Variants, mustSubgroupSizeControl(phys),
		func(v w4a8Variant) (string, uint32) { return "gemv_cold " + v.name + " w4a8", v.waveSize })
	w4a8Mods := make([]*vk.ShaderModule, len(coldW4A8Variants))
	if feat.IntegerDotProduct {
		if w8a8Mod, err = dev.NewShaderModule(shaders.GEMVW8A8); err != nil {
			return nil, err
		}
		defer w8a8Mod.Destroy()
		for i, v := range coldW4A8Variants {
			if w4a8Mods[i], err = dev.NewShaderModule(v.spirv); err != nil {
				return nil, err
			}
			defer w4a8Mods[i].Destroy()
		}
	} else {
		fmt.Fprintln(os.Stderr, "gemv_cold w8a8/w4a8: shaderIntegerDotProduct not supported, skipping")
	}

	var results []Result
	for _, fpMB := range footprintsMB {
		M := fpMB * 1024 * 1024 / (N * 2)
		if M < 1 {
			return nil, fmt.Errorf("footprint %dMB is too small for N=%d", fpMB, N)
		}

		r, err := runGEMVColdCase(dev, f16Mod, coldCase{
			format: "fp16", M: M, N: N,
			weightBytes: M * N * 2,
			fillWeights: fillFloat16Pattern,
		}, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("gemv_cold fp16 footprint=%dMB: %w", fpMB, err)
		}
		results = append(results, r)

		for _, block := range blocks {
			if N%block != 0 {
				continue
			}
			r, err := runGEMVColdCase(dev, q8Mod, coldCase{
				format: "q8", M: M, N: N, block: block,
				weightBytes: M * N,
				fillWeights: fillInt8Pattern,
			}, warmup, iters)
			if err != nil {
				return nil, fmt.Errorf("gemv_cold q8 block=%d footprint=%dMB: %w", block, fpMB, err)
			}
			results = append(results, r)

			r, err = runGEMVColdCase(dev, q4Mod, coldCase{
				format: "q4", M: M, N: N, block: block,
				weightBytes: M * N / 2,
				fillWeights: fillNibblePattern,
			}, warmup, iters)
			if err != nil {
				return nil, fmt.Errorf("gemv_cold q4 block=%d footprint=%dMB: %w", block, fpMB, err)
			}
			results = append(results, r)

			if w8a8Mod == nil {
				continue
			}
			r, err = runGEMVColdCase(dev, w8a8Mod, coldCase{
				format: "w8a8", M: M, N: N, block: block,
				weightBytes: M * N,
				fillWeights: fillInt8Pattern,
				kind:        coldInt8Acts,
			}, warmup, iters)
			if err != nil {
				return nil, fmt.Errorf("gemv_cold w8a8 block=%d footprint=%dMB: %w", block, fpMB, err)
			}
			results = append(results, r)

			for i, v := range coldW4A8Variants {
				if !w4a8BlockOK(block, v.weightsPerLoad) || N%v.weightsPerLoad != 0 {
					continue
				}
				r, err = runGEMVColdCase(dev, w4a8Mods[i], coldCase{
					format: "w4a8", variant: v.name, M: M, N: N, block: block,
					weightBytes: M * N / 2,
					fillWeights: fillNibblePattern,
					kind:        coldW4A8,
					waveSize:    v.waveSize,
				}, warmup, iters)
				if err != nil {
					return nil, fmt.Errorf("gemv_cold w4a8 %s block=%d footprint=%dMB: %w", v.name, block, fpMB, err)
				}
				results = append(results, r)
			}
		}
	}
	return results, nil
}

// coldKind selects the calling convention of the shader under test: which
// buffers it binds and what its push constants look like.
type coldKind int

const (
	// coldFloatActs is the fp16/q8/q4 subgroup shaders: fp32 activations,
	// {M,N,block}.
	coldFloatActs coldKind = iota
	// coldInt8Acts is gemv_w8a8.comp: activations packed as int8 words plus
	// a scalar scale, {M,N,block,xScale}.
	coldInt8Acts
	// coldW4A8 is gemv_w4a8.comp: int8 activations as above, plus a fifth
	// binding holding the per-block activation sums that cancel the packed
	// nibbles' +8 bias, and log2(block) in place of block.
	coldW4A8
)

// coldCase is one (format, shape) pair to time.
type coldCase struct {
	format string
	// variant names the kernel flavour for the results table; empty means
	// "subgroup", which all of these are except the wide-load W4A8 one.
	variant     string
	kind        coldKind
	M, N, block int
	weightBytes int
	fillWeights func(buf []byte)
	// waveSize pins the pipeline's subgroup size, and must match the -DWAVE
	// the shader module was built with (IDEAS §6.2). Zero is the default 64.
	// Every kernel here reduces a whole row with subgroupAdd, so running a
	// 32-thread binary at the default would measure a half-idle wave64 and
	// label it wave32.
	waveSize uint32
}

func runGEMVColdCase(dev *vk.Device, mod *vk.ShaderModule, c coldCase, warmup, iters uint32) (Result, error) {
	wBuf, err := dev.NewBuffer(c.weightBytes)
	if err != nil {
		return Result{}, err
	}
	defer wBuf.Destroy()
	weights := make([]byte, c.weightBytes)
	c.fillWeights(weights)
	wBuf.WriteBytes(weights)
	weights = nil // a few hundred MB; let it go before the next allocation

	// One fp16 scale per block per row. fp16 rather than fp32 matches every
	// quantized shader's Scales binding.
	scaleCount := 1
	if c.block > 0 {
		scaleCount = c.M * (c.N / c.block)
	}
	// The fp16 path's Scales binding is an unused dummy; keep a 4-byte
	// floor so it is never a degenerate 2-byte allocation.
	scalesBytes := scaleCount * 2
	if scalesBytes < 4 {
		scalesBytes = 4
	}
	scalesBuf, err := dev.NewBuffer(scalesBytes)
	if err != nil {
		return Result{}, err
	}
	defer scalesBuf.Destroy()
	scalesBuf.WriteBytes(fillFloat16Const(scaleCount, 0.01))

	var xBuf *vk.Buffer
	var xBytes []byte
	const xScale = 0.01
	if c.kind == coldFloatActs {
		if xBuf, err = dev.NewBuffer(c.N * 4); err != nil {
			return Result{}, err
		}
		xBuf.WriteFloat32(randomFloats(c.N))
	} else {
		// N int8 lanes packed 4-per-word, the layout gemv_w8a8.comp and
		// gemv_w4a8.comp both read.
		if xBuf, err = dev.NewBuffer(c.N); err != nil {
			return Result{}, err
		}
		xBytes = make([]byte, c.N)
		fillInt8Pattern(xBytes)
		xBuf.WriteBytes(xBytes)
	}
	defer xBuf.Destroy()

	yBuf, err := dev.NewBuffer(c.M * 4)
	if err != nil {
		return Result{}, err
	}
	defer yBuf.Destroy()

	// All three shaders bind {W, Scales, X, Y} in that order; W4A8 adds a
	// fifth binding for the per-block activation sums. Those are computed
	// from the same bytes the x buffer was filled with, so the bias
	// correction is the real one even though this sweep skips the CPU
	// reference (see the doc comment).
	buffers := []*vk.Buffer{wBuf, scalesBuf, xBuf, yBuf}
	pushSize := uint32(12)
	pc := newPC().U32(uint32(c.M)).U32(uint32(c.N)).U32(uint32(c.block))
	switch c.kind {
	case coldInt8Acts:
		pushSize = 16
		pc = pc.F32(xScale)
	case coldW4A8:
		pushSize = 16
		pc = newPC().U32(uint32(c.M)).U32(uint32(c.N)).U32(log2u(c.block)).F32(xScale)
		sums := int8BlockSums(bytesToInt8(xBytes), c.block)
		sumsBuf, err := dev.NewBuffer(len(sums) * 4)
		if err != nil {
			return Result{}, err
		}
		defer sumsBuf.Destroy()
		sumsBuf.WriteBytes(int32SliceToBytes(sums))
		buffers = append(buffers, sumsBuf)
	}
	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:              buffers,
		PushConstantSize:     pushSize,
		RequiredSubgroupSize: c.waveSize,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pcBytes := pc.Bytes()

	groupsX := uint32(c.M) // the subgroup kernels use one workgroup per output row
	if _, err := pipe.DispatchTimed(groupsX, 1, 1, 1, pcBytes); err != nil {
		return Result{}, err
	}
	if err := sanityCheckOutput(yBuf, c.M); err != nil {
		return Result{}, err
	}

	ns, clocks, err := TimeDispatch(pipe, groupsX, 1, 1, warmup, iters, pcBytes)
	if err != nil {
		return Result{}, err
	}

	variant := c.variant
	if variant == "" {
		variant = "subgroup"
	}
	bytesRead := float64(c.weightBytes + scaleCount*2)
	flops := float64(2 * c.M * c.N)
	return Result{
		Op: "gemv_cold", Variant: variant, WeightFormat: c.format, BlockSize: c.block, Size: c.M,
		Detail:    fmt.Sprintf("N=%d;weightMB=%.1f", c.N, bytesRead/(1024*1024)),
		NsPerIter: ns,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
		GBPS:      bytesRead / (ns / 1e9) / 1e9,
		Clocks:    clocks,
	}, nil
}

// sanityCheckOutput is the cheap stand-in for the CPU reference check the
// small-size sweeps do: it only proves the kernel ran and wrote plausible
// numbers, which is enough to catch a mis-sized buffer or an unbound
// descriptor without an O(M*N) host-side matmul.
func sanityCheckOutput(y *vk.Buffer, M int) error {
	probe := 64
	if probe > M {
		probe = M
	}
	got := y.ReadFloat32(probe)
	nonZero := false
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("sanity check failed: y[%d] = %v", i, v)
		}
		if v != 0 {
			nonZero = true
		}
	}
	if !nonZero {
		return fmt.Errorf("sanity check failed: first %d outputs are all zero", probe)
	}
	return nil
}

// The fill* helpers write cheap deterministic patterns straight into the
// byte layout each shader expects, instead of generating float32 weights and
// quantizing them. At these footprints the float32 intermediate would be
// gigabytes, and since none of these kernels branch on data values, timing
// is data-independent — the only requirement is normal, non-zero values
// (no denormals or NaNs, which can carry their own arithmetic penalties).

func fillFloat16Pattern(buf []byte) {
	// A repeating run of small positive halves, written once and then
	// doubled to fill, which is far faster than per-element conversion.
	pattern := float32SliceToFloat16Bytes(patternFloats(1024))
	fillRepeating(buf, pattern)
}

func fillInt8Pattern(buf []byte) {
	pattern := make([]byte, 1024)
	for i := range pattern {
		pattern[i] = byte(int8(i%63 + 1))
	}
	fillRepeating(buf, pattern)
}

// fillNibblePattern fills Q4-packed bytes, each holding two nibbles that
// the shaders read as values in [0,15] and re-centre to [-8,7]. Nibbles are
// kept away from 8 (which maps to zero) so no product vanishes.
func fillNibblePattern(buf []byte) {
	pattern := make([]byte, 1024)
	for i := range pattern {
		lo := uint8(i%7 + 9)
		hi := uint8(i%7 + 1)
		pattern[i] = lo | hi<<4
	}
	fillRepeating(buf, pattern)
}

func fillFloat16Const(n int, v float32) []byte {
	return float32SliceToFloat16Bytes(constFloats(n, v))
}

func patternFloats(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 0.5 + float32(i%16)*0.03125
	}
	return out
}

// fillRepeating tiles pattern across buf, doubling the filled prefix each
// pass so a multi-hundred-MB buffer costs a handful of large copies rather
// than one copy per pattern length.
func fillRepeating(buf, pattern []byte) {
	if len(buf) == 0 || len(pattern) == 0 {
		return
	}
	n := copy(buf, pattern)
	for n < len(buf) {
		n += copy(buf[n:], buf[:n])
	}
}
