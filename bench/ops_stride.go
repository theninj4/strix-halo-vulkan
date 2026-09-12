package bench

import (
	"fmt"
	"io"
	"sort"
	"text/tabwriter"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

const (
	strideLocalSize = 256
	// strideTargetBatchNs is how long one timed batch should run. The
	// cache-resident cases are the reason it exists: 16 MiB at ~965 GB/s is
	// a 17us dispatch, and the suite's default 20 iterations average that
	// over 340us. Each case therefore requests as many iterations as its
	// own probe dispatch says will fill this budget — TimeDispatch still
	// caps the batch at maxBatchNs, so a slow case is unaffected and no
	// case can run long enough to worry the GPU-hang watchdog.
	strideTargetBatchNs = 50_000_000
	// strideBatches is how many timed batches each case runs, keeping the
	// fastest. The cache-resident cases are why: a 16 MiB working set is
	// half the MALL, so anything else touching memory — the display this
	// iGPU also drives, another process on the GPU — evicts part of it and
	// the case falls back towards DRAM. Measured with a second benchmark
	// process running alongside, ~10% of these cases land at 0.3-0.5x with
	// sclk pinned at 2815 MHz and fclk at 1850 MHz, on a different (stride,
	// shape) cell every run — so it is neither clock nor the address
	// pattern, which two cases in the same column share exactly. Sole use
	// of the GPU removes the large dropouts and leaves ~10% one-sided
	// scatter. Contention can only ever make a case slower, so the fastest
	// batch is the estimator that answers "what does the memory system
	// deliver", which is what this family asks. DRAM-resident cases neither
	// need it nor are changed by it: they reproduce to within 2% either way.
	strideBatches = 3
	// stridePushConstantSize must match strided_read.comp's push-constant
	// block. Declaring a smaller range than the shader's block is invalid,
	// and a larger one leaves a field reading out of range — which RADV
	// answers with plausible numbers rather than an error.
	stridePushConstantSize = 12
)

// StrideRowBytesList is the swept *touched* bytes per row, held constant
// while the stride varies so that a pad adds only a gap and never changes
// how much is read — exactly what padding a real matrix's leading dimension
// does. Three entries because the engine's rule has to hold for any K, and
// the row length turned out to be half the mechanism rather than a control:
// 8192 B is one row of a K=4096 fp16 weight matrix (the shape §2.3 measured
// the aliasing penalty on) and is long enough to reach every channel by
// itself whatever the stride; 2048 B and 1024 B are K=1024 and K=512 rows,
// short enough that an aliased stride leaves whole channels untouched, which
// costs 2x and 4x and hits a plainly contiguous read as hard as a gather.
// See strideChannelCoverage.
var StrideRowBytesList = []int{8192, 2048, 1024}

// StridePadsBytes is the swept gap between the end of one touched row and
// the start of the next, in bytes — so the row stride is the row length
// plus the pad. Chosen to answer two questions at once, and kept as
// the committed sweep because the answers turned out to be sharp:
//
//   - is the penalty *periodic* in the stride, and with what period? §2.3
//     inferred 2 KB from a GEMM, from three strides. This sweep has nine
//     points inside one period and three multiples of 4 KB, and says the
//     period is **4 KB**: 10240 and 14336 are multiples of 2 KB and run at
//     full speed, while 8192, 12288 and 16384 — the multiples of 4 KB — are
//     the only slow strides. 4 KB is one rotation of sixteen 256 B chunks,
//     which is what a 256-bit LPDDR5X bus of 16-bit sub-channels gives.
//   - where inside one period is the optimum, and how fast is the recovery?
//     The 16/64/128/256/512 B points resolve that: one 256 B channel chunk
//     of pad recovers most of it and 128 B is already within noise of the
//     best, so the engine's rule can be "pad to 256 B past a multiple of
//     4 KB" without a per-shape search.
//
// Every entry is a multiple of 16 so each row start stays 16-byte aligned:
// unaligned rows would make the loads narrower than buffer_load_b128 and
// the sweep would measure alignment instead of aliasing.
var StridePadsBytes = []int{0, 16, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192}

// StrideFootprintsMB is the swept *touched* footprint, in MiB. 16 MiB sits
// under this chip's 32 MiB MALL (which §0.4 measured at 805 GB/s) with room
// for the largest stride's span; 64 MiB is twice the MALL, so every byte
// comes from DRAM (236 GB/s, and spot-checked at 256 MiB where every number
// is within 1% of the 64 MiB one). Running both is the point, because the
// two levels answer differently. DRAM loses up to 1.32x to an aliased gather
// and follows strideChannelCoverage exactly. The MALL delivers 930-965 GB/s
// for a pure read, is indifferent to request shape, and keeps a
// fully-covering working set resident however far it is spread — but under
// partial coverage it keeps it only while the *span* is also inside 32 MiB,
// and past that the DRAM law reappears undiminished. See IDEAS §5.1b's
// table; the mechanism there is inferred, not measured.
var StrideFootprintsMB = []int{16, 64}

// strideShape is one request shape of shaders/strided_read.comp, i.e. one
// lane->address mapping over an identical set of touched bytes.
// lanesPerRow consecutive 16-byte chunks of a row go to consecutive lanes,
// so one 64-lane load instruction spans rowsPerReq = 64/lanesPerRow
// distinct rows. That is the quantity the aliasing mechanism is about: a
// request whose addresses are all `stride` apart can put all of them in one
// memory channel, and a contiguous request never can.
type strideShape struct {
	name        string
	spirv       []byte
	lanesPerRow int
}

func (s strideShape) rowsPerReq() int { return 64 / s.lanesPerRow }

// strideShapes runs from a plain contiguous sweep to a maximally divergent
// gather. `rows16` is the shape IDEAS §5.1b asks for (16 addresses of 16
// bytes each); `rows32` is what a 16x16 fp16 coopMatLoad actually issues on
// a wave32 subgroup (32 lanes x 16 B over 16 rows, two lanes per row);
// `rows1` is the contiguous control that the `bandwidth` family measures.
var strideShapes = []strideShape{
	{name: "rows1", spirv: shaders.StridedReadLPR64, lanesPerRow: 64},
	{name: "rows4", spirv: shaders.StridedReadLPR16, lanesPerRow: 16},
	{name: "rows16", spirv: shaders.StridedReadLPR4, lanesPerRow: 4},
	{name: "rows32", spirv: shaders.StridedReadLPR2, lanesPerRow: 2},
	{name: "rows64", spirv: shaders.StridedReadLPR1, lanesPerRow: 1},
}

// RunStride measures read bandwidth over a fixed set of bytes as a function
// of row stride and request shape — IDEAS §5.1b, the ceiling half of §2.3.
//
// The `bandwidth` family only measures a contiguous sweep, so its 236 GB/s
// (DRAM) and 805 GB/s (MALL) are best-case figures. §2.3 showed a GEMM
// gaining up to 1.6x purely from moving an operand's leading dimension
// 256 B off a power-of-two, but a GEMM cannot separate the stride's effect
// from its own tiling. This can: every case reads the same bytes with the
// same number of 16-byte load instructions, and only the stride between
// rows and the lane->address mapping change.
//
// Two independent effects come out of it, and only the first is §2.3's:
// a request whose addresses are all one stride apart loses up to 1.32x when
// that stride is a multiple of the 4 KB channel-interleave rotation, and a
// row shorter than gcd(stride, 4096) leaves whole channels unaddressed,
// which costs 2x or 4x and hits a contiguous read just as hard. See
// strideChannelCoverage for the second, which is the larger of the two.
func RunStride(dev *vk.Device, footprintsMB, rowBytesList, pads []int, warmup, iters uint32) ([]Result, error) {
	mods := make([]*vk.ShaderModule, len(strideShapes))
	for i, s := range strideShapes {
		m, err := dev.NewShaderModule(s.spirv)
		if err != nil {
			return nil, err
		}
		defer m.Destroy()
		mods[i] = m
	}

	maxPad := 0
	for _, p := range pads {
		if p > maxPad {
			maxPad = p
		}
		if p%16 != 0 {
			return nil, fmt.Errorf("stride pad %d is not a multiple of 16, which would unalign the rows", p)
		}
	}

	var results []Result
	for _, rowBytes := range rowBytesList {
		if rowBytes%(16*64) != 0 {
			// Every shape assigns up to 64 consecutive 16-byte chunks of a
			// row to one request, so a row has to divide into whole
			// requests or the shapes would cover different byte sets.
			return nil, fmt.Errorf("stride row length %dB is not a multiple of 1024", rowBytes)
		}
		chunksPerRow := rowBytes / 16
		for _, fpMB := range footprintsMB {
			rows := fpMB * 1024 * 1024 / rowBytes
			if rows < 1 {
				return nil, fmt.Errorf("footprint %dMB is smaller than one %dB row", fpMB, rowBytes)
			}
			if rows*chunksPerRow%strideLocalSize != 0 {
				return nil, fmt.Errorf("footprint %dMB at %dB rows needs %d threads, not a multiple of the %d-thread workgroup",
					fpMB, rowBytes, rows*chunksPerRow, strideLocalSize)
			}
			// One buffer per (row length, footprint), sized for the largest
			// stride in the sweep and reused by every (stride, shape) case:
			// the strides then differ in nothing but the push-constant word,
			// and no case pays for a fresh allocation the driver might place
			// differently.
			src, err := dev.NewBuffer(rows * (rowBytes + maxPad))
			if err != nil {
				return nil, err
			}
			groups := uint32(rows * chunksPerRow / strideLocalSize)
			dst, err := dev.NewBuffer(int(groups) * 4)
			if err != nil {
				src.Destroy()
				return nil, err
			}
			fillStridePattern(src)

			for _, pad := range pads {
				stride := rowBytes + pad
				want := expectedStrideSum(rows, stride, chunksPerRow)
				for i, shape := range strideShapes {
					r, err := runStrideCase(dev, mods[i], shape, src, dst,
						rows, stride, chunksPerRow, groups, want, warmup, iters)
					if err != nil {
						src.Destroy()
						dst.Destroy()
						return nil, fmt.Errorf("stride %s row=%dB footprint=%dMB stride=%dB: %w",
							shape.name, rowBytes, fpMB, stride, err)
					}
					results = append(results, r)
				}
			}
			src.Destroy()
			dst.Destroy()
		}
	}
	return results, nil
}

func runStrideCase(dev *vk.Device, mod *vk.ShaderModule, shape strideShape, src, dst *vk.Buffer,
	rows, stride, chunksPerRow int, groups uint32, want uint32, warmup, iters uint32) (Result, error) {
	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{src, dst},
		PushConstantSize: stridePushConstantSize,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := newPC().U32(uint32(rows)).U32(uint32(chunksPerRow)).U32(uint32(stride / 16)).Bytes()
	if len(pc) != stridePushConstantSize {
		return Result{}, fmt.Errorf("push constants are %d bytes, shader declares %d", len(pc), stridePushConstantSize)
	}

	// One untimed dispatch first, checked against a host sum over the same
	// (row, chunk) set. The check is over coverage rather than over the
	// shader's lane mapping, so it catches a mapping that reads a chunk
	// twice or skips one — which would otherwise look like a bandwidth
	// result rather than a bug.
	probe, err := pipe.DispatchTimed(groups, 1, 1, 1, pc)
	if err != nil {
		return Result{}, err
	}
	var got uint32
	for _, v := range dst.ReadUint32(int(groups)) {
		got += v
	}
	if got != want {
		return Result{}, fmt.Errorf("checksum mismatch: got %d want %d (touched set not covered exactly once)", got, want)
	}

	// The verification dispatch doubles as the cost estimate for how many
	// iterations this case needs to fill strideTargetBatchNs.
	if probeNs := float64(probe.Nanoseconds()); probeNs > 0 {
		if want := uint32(strideTargetBatchNs / probeNs); want > iters {
			iters = want
		}
	}

	var ns float64
	var clocks ClockStats
	for b := 0; b < strideBatches; b++ {
		batchNs, batchClocks, err := TimeDispatch(pipe, groups, 1, 1, warmup, iters, pc)
		if err != nil {
			return Result{}, err
		}
		if b == 0 || batchNs < ns {
			ns, clocks = batchNs, batchClocks
		}
	}

	bytesTouched := float64(rows) * float64(chunksPerRow) * 16
	span := rows * stride
	return Result{
		Op: "stride", Variant: shape.name, Size: stride,
		NsPerIter: ns,
		Clocks:    clocks,
		GBPS:      bytesTouched / (ns / 1e9) / 1e9,
		Detail: fmt.Sprintf("touch=%dMiB row=%dB span=%dMiB rows=%d lanes/row=%d rows/req=%d stride_mod_4k=%d",
			int(bytesTouched)/(1024*1024), chunksPerRow*16, span/(1024*1024), rows,
			shape.lanesPerRow, shape.rowsPerReq(), stride%strideAliasChunk),
	}, nil
}

// strideFillMultiplier is Knuth's multiplicative hash constant, used to give
// every 4-byte word a value that depends on its address. A constant fill
// would let a checksum pass even if the kernel read the wrong rows.
const strideFillMultiplier = 2654435761

func strideWord(i int) uint32 { return uint32(i) * strideFillMultiplier }

func fillStridePattern(b *vk.Buffer) {
	words := unsafe.Slice((*uint32)(b.MappedPointer()), b.Size()/4)
	for i := range words {
		words[i] = strideWord(i)
	}
}

// expectedStrideSum is the sum the kernel must produce: every 4-byte word of
// every touched chunk, added once, with uint32 wraparound (which is what the
// shader's uint arithmetic does too). Derived from the (rows, stride, chunks)
// geometry alone, deliberately not from the shader's lane mapping.
func expectedStrideSum(rows, stride, chunksPerRow int) uint32 {
	var sum uint32
	strideWords := stride / 4
	for r := 0; r < rows; r++ {
		base := r * strideWords
		for w := 0; w < chunksPerRow*4; w++ {
			sum += strideWord(base + w)
		}
	}
	return sum
}

// PrintStrideSummary pivots the stride family into the grid the experiment
// is actually about: request shape down the side, row stride across, GB/s in
// the cells, with each row also given as a ratio against its own unpadded
// stride so the aliasing penalty is readable without dividing by hand. One
// grid per (touched footprint, row length), since those are the two outer
// sweeps and the comparison is only meaningful inside one of them.
func PrintStrideSummary(w io.Writer, results []Result) {
	type group struct {
		footprint string
		rowBytes  int
	}
	type key struct {
		group group
		shape string
	}
	cells := map[key]map[int]float64{}
	strides := map[group]map[int]bool{}
	var groups []group
	seen := map[group]bool{}
	for _, r := range results {
		if r.Op != "stride" {
			continue
		}
		g := group{strideFootprintOf(r), strideRowBytesOf(r)}
		if !seen[g] {
			seen[g] = true
			groups = append(groups, g)
			strides[g] = map[int]bool{}
		}
		k := key{g, r.Variant}
		if cells[k] == nil {
			cells[k] = map[int]float64{}
		}
		cells[k][r.Size] = r.GBPS
		strides[g][r.Size] = true
	}
	if len(groups) == 0 {
		return
	}

	for _, g := range groups {
		var strideList []int
		for s := range strides[g] {
			strideList = append(strideList, s)
		}
		sort.Ints(strideList)

		fmt.Fprintf(w, "stride sweep, %s touched in %dB rows (GB/s, and x vs the same shape at stride %d)\n", g.footprint, g.rowBytes, g.rowBytes)
		tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
		fmt.Fprint(tw, "ROWS/REQ")
		for _, s := range strideList {
			fmt.Fprintf(tw, "\t%d%s", s, strideAliasMark(s))
		}
		fmt.Fprintln(tw)
		fmt.Fprint(tw, "model")
		for _, s := range strideList {
			fmt.Fprintf(tw, "\t%.2fx", strideChannelCoverage(g.rowBytes, s))
		}
		fmt.Fprintln(tw)
		for _, shape := range strideShapes {
			row := cells[key{g, shape.name}]
			if row == nil {
				continue
			}
			fmt.Fprintf(tw, "%d", shape.rowsPerReq())
			baseline := row[g.rowBytes]
			for _, s := range strideList {
				v, ok := row[s]
				if !ok {
					fmt.Fprint(tw, "\t-")
					continue
				}
				if baseline > 0 {
					fmt.Fprintf(tw, "\t%.0f (%.2fx)", v, v/baseline)
				} else {
					fmt.Fprintf(tw, "\t%.0f", v)
				}
			}
			fmt.Fprintln(tw)
		}
		tw.Flush()
		fmt.Fprintf(w, "(* = stride is a multiple of %d B, one full rotation of sixteen %d B channel chunks.\n", strideAliasChunk, strideChannelChunk)
		fmt.Fprintf(w, " model = the fraction of those channels this (row, stride) pattern touches at all,\n")
		fmt.Fprintf(w, " which bounds every shape; what a row loses below its own model figure is the\n")
		fmt.Fprintf(w, " per-request penalty for a gather whose addresses all land in one channel.)\n\n")
	}
}

// The two constants the whole family measures. strideChannelChunk is the
// interleave granularity — consecutive 256 B chunks of an address range go
// to consecutive memory channels — and strideAliasChunk is one full rotation
// through all sixteen of them. Strides that are a multiple of the rotation
// are the slow ones; every stride that is not, *including* multiples of 2 KB
// which §2.3 had guessed were the bad case, runs at full speed.
const (
	strideChannelChunk = 256
	strideChannelCount = 16
	strideAliasChunk   = strideChannelChunk * strideChannelCount // 4096
)

// strideChannelCoverage is the fraction of the sixteen channels a
// (rowBytes, stride) access pattern touches at all, and so — measured to
// within 2% at every point of this sweep — the fraction of peak DRAM
// bandwidth it can reach.
//
// Row k starts at k*stride, so row starts land on the multiples of
// g = gcd(stride, 4096) within one rotation, and each row covers rowBytes
// consecutive bytes from its start. The union of those is everything if
// rowBytes >= g, and otherwise a rowBytes/g fraction of the rotation —
// which is a fraction of the *channels*, hence of the bus.
//
// This is the larger of the family's two effects and the one with no
// request-shape escape: it applies to a plainly contiguous sweep. A K=1024
// fp16 weight matrix (2048 B rows) whose stride someone aligned to 4096 B
// for tidiness reads at half of this chip's DRAM bandwidth, and at 4096 B
// rows with an 8192 B stride it would be a quarter. The engine's rule
// follows directly: keep gcd(stride, 4096) no larger than the bytes actually
// read per row, which padding the stride to 256 B past a multiple of 4 KB
// satisfies for every shape (it makes the gcd 256).
func strideChannelCoverage(rowBytes, stride int) float64 {
	g := gcdInt(stride, strideAliasChunk)
	if rowBytes >= g {
		return 1
	}
	return float64(rowBytes) / float64(g)
}

func gcdInt(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// strideAliasMark flags the strides aliasing predicts will be slow, so the
// prediction is visible in the table next to the result rather than only in
// the commentary.
func strideAliasMark(stride int) string {
	if stride%strideAliasChunk == 0 {
		return "*"
	}
	return ""
}

// strideFootprintOf and strideRowBytesOf read back the two outer sweep
// variables from Detail, which is where they live because Result has one
// Size field and the swept axis in it is the stride.
func strideFootprintOf(r Result) string {
	var fp string
	if _, err := fmt.Sscanf(r.Detail, "touch=%s", &fp); err != nil {
		return "?"
	}
	return fp
}

func strideRowBytesOf(r Result) int {
	var fp string
	var rowBytes int
	if _, err := fmt.Sscanf(r.Detail, "touch=%s row=%dB", &fp, &rowBytes); err != nil {
		return 0
	}
	return rowBytes
}
