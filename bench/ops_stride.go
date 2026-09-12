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

// strideShape is one case of shaders/strided_read.comp, i.e. one
// lane->address mapping over an identical set of touched bytes. Three
// things distinguish them:
//
//   - lanesPerRow: how many consecutive 16-byte chunks of a row go to
//     consecutive lanes, so one 64-lane load instruction spans
//     rowsPerReq = 64/lanesPerRow distinct rows. That is the quantity the
//     per-request aliasing mechanism is about: a request whose addresses
//     are all `stride` apart can put all of them in one memory channel, and
//     a contiguous request never can.
//   - crossWave: whether consecutive *requests* advance by a row group or
//     by a column block. The shapes above vary addresses only inside one
//     request; this varies which addresses different waves hold at the same
//     moment, which is the thing IDEAS §2.3 suspected mattered to the LDS
//     staging loads and the one pattern the original sweep could not reach.
//   - loadsPerWave: how many loads one request issues along its own row
//     before retiring. 1 leaves every wave phase-locked with its
//     neighbours; at chunksPerRow/lanesPerRow a wave streams a whole row,
//     which is the shape the GEMV kernels have. Only meaningful with
//     crossWave, and at the top of the ladder the two traversals coincide
//     (one column group, nothing left to permute).
type strideShape struct {
	name         string
	spirv        []byte
	lanesPerRow  int
	crossWave    bool
	loadsPerWave int
}

func (s strideShape) rowsPerReq() int { return 64 / s.lanesPerRow }

// loads is loadsPerWave with the zero value read as 1, so the common shapes
// can leave it out of their literals.
func (s strideShape) loads() int {
	if s.loadsPerWave < 1 {
		return 1
	}
	return s.loadsPerWave
}

// bytesInFlight is the contiguous run of bytes one request holds in one row,
// which is the quantity strideChannelCoverage turns out to be about once the
// traversal is cross-wave: the addresses outstanding at any moment are then
// these runs, one per row, one stride apart.
func (s strideShape) bytesInFlight() int { return s.lanesPerRow * 16 * s.loads() }

// label is how the shape appears in the pivot grid: the rows one request
// spans, plus markers for the cross-wave traversal and for a request that
// issues more than one load.
func (s strideShape) label() string {
	out := fmt.Sprintf("%d", s.rowsPerReq())
	if s.crossWave {
		out += " xw"
	}
	if s.loads() > 1 {
		out += fmt.Sprintf(" L%d", s.loads())
	}
	return out
}

// traversal names the axis for CSV/Detail consumers: `walk` follows one row
// with consecutive requests, `xw` follows one column.
func (s strideShape) traversal() string {
	if s.crossWave {
		return "xw"
	}
	return "walk"
}

// strideShapes runs from a plain contiguous sweep to a maximally divergent
// gather, each in both traversals. `rows16` is the shape IDEAS §5.1b asks
// for (16 addresses of 16 bytes each); `rows32` is what a 16x16 fp16
// coopMatLoad actually issues on a wave32 subgroup (32 lanes x 16 B over 16
// rows, two lanes per row); `rows1` is the contiguous control that the
// `bandwidth` family measures, and `rows1_xw` is the GEMV/W4A8 pattern —
// each wave reading one row contiguously, consecutive waves one stride
// apart. The pairs are adjacent in the list so each traversal comparison is
// also adjacent in time, which keeps the ~10% scatter of the cache-resident
// cases from landing preferentially on one side of it.
var strideShapes = []strideShape{
	{name: "rows1", spirv: shaders.StridedReadLPR64, lanesPerRow: 64},
	{name: "rows1_xw", spirv: shaders.StridedReadXWLPR64, lanesPerRow: 64, crossWave: true},
	{name: "rows4", spirv: shaders.StridedReadLPR16, lanesPerRow: 16},
	{name: "rows4_xw", spirv: shaders.StridedReadXWLPR16, lanesPerRow: 16, crossWave: true},
	{name: "rows16", spirv: shaders.StridedReadLPR4, lanesPerRow: 4},
	{name: "rows16_xw", spirv: shaders.StridedReadXWLPR4, lanesPerRow: 4, crossWave: true},
	{name: "rows32", spirv: shaders.StridedReadLPR2, lanesPerRow: 2},
	{name: "rows32_xw", spirv: shaders.StridedReadXWLPR2, lanesPerRow: 2, crossWave: true},
	{name: "rows64", spirv: shaders.StridedReadLPR1, lanesPerRow: 1},
	{name: "rows64_xw", spirv: shaders.StridedReadXWLPR1, lanesPerRow: 1, crossWave: true},
	// The ladder from a phase-locked one-load-per-wave dispatch up to one
	// wave streaming a whole row, which is the GEMV kernels' shape. Only
	// the row lengths with enough column blocks run each rung; the rest are
	// skipped per case (see RunStride), and at the rung where a wave covers
	// the whole row the traversal no longer exists as a choice.
	{name: "rows1_xw_l2", spirv: shaders.StridedReadXWL2, lanesPerRow: 64, crossWave: true, loadsPerWave: 2},
	{name: "rows1_xw_l4", spirv: shaders.StridedReadXWL4, lanesPerRow: 64, crossWave: true, loadsPerWave: 4},
	{name: "rows1_xw_l8", spirv: shaders.StridedReadXWL8, lanesPerRow: 64, crossWave: true, loadsPerWave: 8},
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
// Three effects come out of it, and only the first is §2.3's: a request
// whose addresses are all one stride apart loses up to 1.32x when that
// stride is a multiple of the 4 KB channel-interleave rotation; a row
// shorter than gcd(stride, 4096) leaves whole channels unaddressed, which
// costs 2x or 4x and hits a contiguous read just as hard; and — the largest,
// which subsumes the second — whenever the bytes the *concurrent* requests
// hold in one row cover less than gcd(stride, 4096), the same channels go
// unaddressed, which costs up to 8x and needs no padding at all to happen.
// See strideChannelCoverage for the law and strideInFlightBytes for the
// quantity it is really a function of.
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
			// Both traversals tile the matrix as (row group x column block),
			// so the row count has to divide into whole row groups or the
			// last group would run off the end — which the checksum would
			// report as missing coverage rather than as a geometry error.
			for _, shape := range strideShapes {
				if rows%shape.rowsPerReq() != 0 {
					return nil, fmt.Errorf("footprint %dMB at %dB rows is %d rows, not a multiple of the %d rows shape %s spans per request",
						fpMB, rowBytes, rows, shape.rowsPerReq(), shape.name)
				}
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
			// Sized for the shape with the most workgroups — one load per
			// request. A shape whose requests issue several loads runs a
			// proportionally smaller grid and writes into the front of the
			// same buffer.
			groups := uint32(rows * chunksPerRow / strideLocalSize)
			dst, err := dev.NewBuffer(int(groups) * 4)
			if err != nil {
				src.Destroy()
				return nil, err
			}
			fillStridePattern(src)

			// Discard one case before timing anything in this group. The
			// fill above writes the whole buffer from the host — up to
			// 128 MiB of dirty lines — and whatever that leaves behind
			// (writeback stealing DRAM bandwidth is the likeliest
			// candidate) costs the first *cache-resident* case 13-17%,
			// reproducibly by position rather than by pattern: it measured
			// exactly 782 GB/s in all three row-length groups of one run,
			// and its shape-identical twin measured 942 immediately after.
			// The clock rules itself out at 1% between them. A case is
			// three 50 ms batches, which is enough for the effect to be
			// gone by the second, and the 64 MiB groups never showed it.
			if len(pads) > 0 {
				warmStride := rowBytes + pads[0]
				if _, err := runStrideCase(dev, mods[0], strideShapes[0], src, dst,
					rows, warmStride, chunksPerRow, groups,
					expectedStrideSum(rows, warmStride, chunksPerRow), warmup, iters); err != nil {
					src.Destroy()
					dst.Destroy()
					return nil, fmt.Errorf("stride warm row=%dB footprint=%dMB stride=%dB: %w",
						rowBytes, fpMB, warmStride, err)
				}
			}

			for _, pad := range pads {
				stride := rowBytes + pad
				want := expectedStrideSum(rows, stride, chunksPerRow)
				for i, shape := range strideShapes {
					// A multi-load request covers loadsPerWave whole column
					// blocks, so a row too short to hold one of those groups
					// has no such shape: skip it rather than mis-shaping the
					// grid. (1 KB rows are one column block wide, so every
					// rung of the ladder above the first is skipped there —
					// and their rows1 case already *is* one wave per row.)
					if chunksPerRow%(shape.lanesPerRow*shape.loads()) != 0 {
						continue
					}
					shapeGroups := uint32(rows * chunksPerRow / shape.loads() / strideLocalSize)
					if shapeGroups == 0 {
						continue
					}
					r, err := runStrideCase(dev, mods[i], shape, src, dst,
						rows, stride, chunksPerRow, shapeGroups, want, warmup, iters)
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
		Detail: fmt.Sprintf("touch=%dMiB row=%dB span=%dMiB rows=%d lanes/row=%d rows/req=%d traversal=%s loads/req=%d inflight=%dB stride_mod_4k=%d",
			int(bytesTouched)/(1024*1024), chunksPerRow*16, span/(1024*1024), rows,
			shape.lanesPerRow, shape.rowsPerReq(), shape.traversal(), shape.loads(),
			shape.bytesInFlight(), stride%strideAliasChunk),
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
			fmt.Fprint(tw, shape.label())
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
		fmt.Fprintf(w, " per-request penalty for a gather whose addresses all land in one channel.\n")
		fmt.Fprintf(w, " xw = the cross-wave traversal: consecutive requests advance down the rows, so\n")
		fmt.Fprintf(w, " the loads in flight together are one stride apart instead of contiguous.)\n\n")

		printStrideTraversalGrid(w, g.rowBytes, strideList, func(shape strideShape) map[int]float64 {
			return cells[key{g, shape.name}]
		})
	}
}

// printStrideTraversalGrid is the third axis on its own: for each request
// shape, the cross-wave traversal's bandwidth divided by the walk-along-a-row
// traversal's at the same stride, over an identical set of bytes read by an
// identical number of identical instructions. Anything away from 1.00x is
// the cost (or gain) of *when* addresses are outstanding rather than of what
// they are — the one thing IDEAS §5.1b's original sweep held fixed, and the
// pattern every GEMV kernel in the suite actually has.
//
// Each cell carries the prediction beside the measurement, from the same
// strideChannelCoverage the main grid's model row uses but evaluated on the
// bytes one request holds instead of on the whole row — which is what the
// coverage law turns out to be about (see strideInFlightBytes). It is exact
// for the contiguous-request shapes, which is what real kernels issue, and
// only indicative for the gathers: they beat it by 1.4-8x where it predicts
// a severe penalty, and fall short of it (0.53-0.93x) where it predicts
// none, since the per-request penalty of the second mechanism is still
// there and is much larger cross-wave than the 1.32x a walk shows.
//
// It is printed as its own grid rather than as a second parenthetical in the
// main one because the two ratios answer different questions: there, each
// row against its own unpadded stride (the aliasing penalty); here, one
// traversal against the other (the concurrency penalty).
func printStrideTraversalGrid(w io.Writer, rowBytes int, strideList []int, cellsFor func(strideShape) map[int]float64) {
	var pairs [][2]strideShape
	for _, xw := range strideShapes {
		if !xw.crossWave {
			continue
		}
		for _, walk := range strideShapes {
			if !walk.crossWave && walk.lanesPerRow == xw.lanesPerRow {
				pairs = append(pairs, [2]strideShape{walk, xw})
			}
		}
	}
	if len(pairs) == 0 {
		return
	}

	fmt.Fprintf(w, "  cross-wave traversal effect, %dB rows (xw / walk at the same stride)\n", rowBytes)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprint(tw, "  SHAPE")
	for _, s := range strideList {
		fmt.Fprintf(tw, "\t%d%s", s, strideAliasMark(s))
	}
	fmt.Fprintln(tw)
	degenerate := false
	for _, p := range pairs {
		walkRow, xwRow := cellsFor(p[0]), cellsFor(p[1])
		if walkRow == nil || xwRow == nil {
			continue
		}
		// One request covering a whole row leaves the two traversals with
		// nothing to permute: there is a single column group, so the
		// mappings are identical and the cell is a repeatability check
		// rather than a measurement. That is also the top rung of the
		// loads-per-request ladder, and it is the GEMV kernels' shape.
		same := p[1].bytesInFlight() == rowBytes
		if same {
			degenerate = true
		}
		fmt.Fprintf(tw, "  %s", p[1].label())
		for _, s := range strideList {
			a, okA := walkRow[s]
			b, okB := xwRow[s]
			if !okA || !okB || a <= 0 {
				fmt.Fprint(tw, "\t-")
				continue
			}
			mark := ""
			if same {
				mark = "="
			}
			model := strideChannelCoverage(strideInFlightBytes(p[1], rowBytes), s) /
				strideChannelCoverage(strideInFlightBytes(p[0], rowBytes), s)
			fmt.Fprintf(tw, "\t%.2fx%s (%.2f)", b/a, mark, model)
		}
		fmt.Fprintln(tw)
	}
	tw.Flush()
	fmt.Fprintf(w, "  (bracketed = predicted, from the same coverage law read on the bytes one request\n")
	fmt.Fprintf(w, "   holds rather than on the whole row. Exact for the contiguous shapes; only\n")
	fmt.Fprintf(w, "   indicative for the gathers, which beat it where it predicts a severe penalty\n")
	fmt.Fprintf(w, "   (neighbouring requests refill lines it counts as untouched) and miss it where it\n")
	fmt.Fprintf(w, "   predicts none (the per-request penalty it omits is far larger cross-wave).\n")
	if degenerate {
		fmt.Fprintf(w, "   = : one request already covers a whole %dB row, so both traversals compile to\n", rowBytes)
		fmt.Fprintf(w, "   the same mapping and the cell measures run-to-run scatter, not traversal.\n")
		fmt.Fprintf(w, "   That case is one wave per row, which is what the GEMV kernels do.\n")
	}
	fmt.Fprintln(w, "  )")
	fmt.Fprintln(w)
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

// strideInFlightBytes is the contiguous run of bytes, inside one row, that
// the requests outstanding at any one moment cover — which is the quantity
// strideChannelCoverage is really about, and the thing the traversal axis
// changes. Walking along a row, consecutive requests continue where the
// previous one stopped, so the run is the whole row and the law reads as
// IDEAS §5.1b first stated it. Cross-wave, consecutive requests are a
// stride apart, so the run is only what a single request holds: lanesPerRow
// chunks, times however many loads it issues before retiring.
//
// That is why "a densely packed tensor is never at risk" (§5.1b's first
// corollary) does not survive this axis: at 1 KB rows packed at a 1 KB
// stride, a cross-wave traversal holding 256 B per row reads at a quarter
// of DRAM bandwidth, with no padding anywhere to blame.
func strideInFlightBytes(s strideShape, rowBytes int) int {
	if s.crossWave && s.bytesInFlight() < rowBytes {
		return s.bytesInFlight()
	}
	return rowBytes
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
