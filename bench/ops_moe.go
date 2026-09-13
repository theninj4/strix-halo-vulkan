package bench

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// This file is IDEAS §3.5: the grouped / MoE GEMM, which §3.4 promoted to the
// top of the backlog on two measurements rather than on the model card.
//
// qwen3.8-flash-next routes each token to 10 of 512 experts, so its FFN is
// not one GEMM but 512 of them over variable-sized row groups. §3.4 measured
// what the suite's best kernel does with the resulting shape — 2048 tokens
// over 512 experts is 40 rows each — and got 7600 useful GFLOP/s, 14% of the
// WMMA ceiling, with 62% of the dispatch being tile padding. It also counted
// the dispatches: 1440 of a decode token's 1861 matmuls are expert
// projections.
//
// What is measured here, and why each part is needed to answer that:
//
//  1. **The grouped kernel.** -DGROUPED=1 on shaders/gemm_wmma.comp, which
//     replaces "derive this workgroup's tile origin from gl_WorkGroupID" with
//     "read it out of a table". Every expert's tiles then fit in one
//     dispatch. The inner loop, the operand layout, the hoisting rung and the
//     register allocation are all unchanged — RADV_DEBUG=shaderstats prices
//     the grouped binaries at the same VGPR counts as their non-grouped
//     twins — so the comparison below is between two schedules of the same
//     kernel, not between two kernels.
//  2. **The one-dispatch-per-expert baseline**, the thing it is supposed to
//     beat, run as the *same binary over the same table* with each dispatch
//     taking one expert's slice. That needs a command buffer whose dispatches
//     differ in their push constants, which is why vk grew
//     DispatchSequenceTimed. The barrier between consecutive dispatches is
//     kept, because that serialisation is exactly what the baseline is being
//     charged for.
//  3. **The M tile as a knob.** At 40 rows an expert fills 62% of a 64-row
//     tile and 62% of two 32-row tiles, but 83% of three 16-row ones. BM=16
//     buys that back and pays in arithmetic intensity (10.7 against 32), so
//     the trade has to be priced rather than argued.
//  4. **The gather and the combine**, which a grouped GEMM makes necessary:
//     its A operand needs each expert's rows contiguous, and real routing
//     does not deliver them that way. Two extra passes over the activations
//     per layer is not obviously cheap next to a matmul that is already
//     memory-bound, and nothing else in this suite measures them.
//
// The weights here are fp16, which is what the WMMA path takes. That makes
// the prefill overwhelmingly memory-bound — a 512-expert fp16 bank is 1.7 GB
// and one pass over it at the full 236 GB/s bus is 7.1 ms — and it is the
// reason the summary reports weight bandwidth next to GFLOP/s. A Q4 bank
// would be a quarter of that, which is IDEAS §2.2's int8/Q4 arm and is what
// this measurement hands a motive back to.

// The MoE structure of qwen3.8-flash-next, from its config.json (the same
// source bench/modelshapes.go reads): 512 experts, 10 active per token, a
// 2560-wide residual stream and a 640-wide expert intermediate.
const (
	moeExperts = 512
	moeTopK    = 10
	moeHidden  = 2560
	moeFFN     = 640
	// moeLayers is how many MoE blocks one forward pass runs, which is what
	// turns a per-layer measurement into a per-pass budget.
	moeLayers = 48
)

// moeTokens is the batch sweep. 2048 is §3.4's prompt chunk and the only one
// the model card implies; the others are here because rows-per-expert is
// tokens*topk/experts and that ratio is the whole question — 1 token means 10
// experts of one row each (decode), 512 means 10 rows, 8192 means 160, which
// is where an expert's group stops being a fringe shape.
var moeTokens = []int{1, 512, 2048, 8192}

// moeShape is one of the two matrices in an MoE FFN block. Both are per
// expert; the bank holds moeExperts of them end to end.
type moeShape struct {
	layer string
	N, K  int
}

var moeShapes = []moeShape{
	// gate and up are two matrices of this shape, run back to back.
	{layer: "moe.gate_up", N: moeFFN, K: moeHidden},
	{layer: "moe.down", N: moeHidden, K: moeFFN},
}

// moeVariant is one grouped build of gemm_wmma.comp. Same fields as
// wmmaVariant means the same things (see bench/ops_gemm_wmma.go); B is always
// column-major here because an expert bank is [E*N, K], which is both how the
// weights already sit and the only layout whose per-expert offset is a row
// offset rather than a per-K-step one.
type moeVariant struct {
	name       string
	spirv      []byte
	bm, bn, bk int
	waveSize   uint32
	// pad is the leading-dimension padding in halves, applied to both
	// operands (unlike wmmaVariant, which separates them so that §2.3's
	// ablation can attribute an effect to one; here they are always the same
	// because both operands' rows are K halves long).
	pad int
	// gcdTarget, when non-zero, picks the pad per shape instead: the smallest
	// one that makes gcd(row bytes, 4096) equal this. That is the axis the
	// stride arm sweeps, and it has to be expressed as a target rather than
	// as a pad because the same pad lands on a different gcd at each K —
	// which is precisely how the suite's fixed +256 B ended up being the
	// wrong stride for the down projection.
	gcdTarget int
	// strideArm marks the rows that vary only the stride: they run grouped
	// only, and only at the real prefill batch, because the schedule and the
	// batch are not what they are asking about.
	strideArm bool
}

// padHalves is the padding this variant uses at reduction length K, and
// whether it is expressible at all (a gcd target the strides cannot reach is
// skipped rather than silently approximated).
func (v moeVariant) padHalves(K int) (int, bool) {
	if v.gcdTarget == 0 {
		return v.pad, true
	}
	return padForGCD(K, v.gcdTarget)
}

// padForGCD is the smallest row padding, in halves, that puts a K-half row at
// a stride whose gcd with the 4 KB channel rotation is exactly target. The
// step of 8 halves keeps every row 16-byte aligned, without which coopMatLoad
// drops back to scalar 16-bit loads and the comparison measures alignment
// instead of stride (IDEAS §2.3).
func padForGCD(K, target int) (int, bool) {
	for p := 0; p <= 8192; p += 8 {
		if gcdInt((K+p)*2, strideAliasChunk) == target {
			return p, true
		}
	}
	return 0, false
}

func (v moeVariant) ld(K int) int {
	p, _ := v.padHalves(K)
	return K + p
}
func (v moeVariant) lda(K int) int { return v.ld(K) }
func (v moeVariant) ldb(K int) int { return v.ld(K) }
func (v moeVariant) intensity() float64 {
	return float64(v.bm*v.bn) / float64(v.bm+v.bn)
}
func (v moeVariant) wave() int {
	if v.waveSize == 0 {
		return 64
	}
	return int(v.waveSize)
}

// moeVariants has two halves, the same way bench/ops_gemm_wmma.go's does.
//
// The first varies *geometry*. The first three are §3.4's two crowned kernels
// plus the wave-size twin that separates them, so the MoE numbers land on the
// same axis the rest of the suite is measured on; the last two are the shapes
// only a 40-row group motivates — 16 rows per tile, so a 40-row group wastes
// 8 rows instead of 24, bought with half the arithmetic intensity.
//
// The second varies *stride* at fixed geometry, which is here because the
// coverage law (§5.1b) says the +256 B pad every kernel in this suite carries
// is the wrong pad for these two shapes, and says it loudest for the one
// §3.4 could not explain. The law is that an access pattern reaches
// min(1, C/gcd(stride, 4096)) of peak, with C the contiguous run of one row
// the requests in flight hold — 32 B per K-tile, so 128 B at BK_TILES=4:
//
//	shape        pad    row bytes  gcd(.,4096)  C/gcd
//	moe.gate_up  +256B  5376       256          0.50
//	 (K=2560)    +16B   5136        16          1.00
//	             none   5120       1024         0.13
//	moe.down     +256B  1536        512         0.25
//	 (K=640)     +16B   1296        16          1.00
//	             none   1280        256         0.50
//
// So on the down projection the pad the suite has been carrying is predicted
// to be *worse than no pad at all*, and a 16-byte pad better than either.
// That shape is §3.4's second open item — the MoE down projection is the one
// decode reduction length that reads 88-91% of the bus and nothing moved it —
// and these three rows are the same test on the GEMM side of it. Same SPIR-V,
// same tile, one different push-constant word.
var moeVariants = []moeVariant{
	// §2.7's winner, which takes every rectangle with M >= 1024 in §3.4.
	{name: "moe_reg64_bt_hka4_padab128", spirv: shaders.GEMMWMMAMoEReg64BTHKA4, bm: 64, bn: 64, bk: 64, pad: 128},
	// §6.2's AI-16 arm at both wave sizes. The wave32 one is §3.4's winner at
	// every M below 1024, i.e. at everything an expert ever sees.
	{name: "moe_reg32_bt_hkab4_padab128", spirv: shaders.GEMMWMMAMoEReg32BTHKAB4, bm: 32, bn: 32, bk: 64, pad: 128},
	{name: "moe_reg32_bt_hkab4_w32_padab128", spirv: shaders.GEMMWMMAMoEReg32BTHKAB4W32, bm: 32, bn: 32, bk: 64, waveSize: 32, pad: 128},
	{name: "moe_reg16x32_bt_hkab4_w32_padab128", spirv: shaders.GEMMWMMAMoEReg16x32BTHKAB4W32, bm: 16, bn: 32, bk: 64, waveSize: 32, pad: 128},
	{name: "moe_reg16x64_bt_hkab4_w32_padab128", spirv: shaders.GEMMWMMAMoEReg16x64BTHKAB4W32, bm: 16, bn: 64, bk: 64, waveSize: 32, pad: 128},

	// The stride arm. Same SPIR-V, same tile, one different push-constant
	// word; what varies is gcd(row stride, 4096), swept over every power of
	// two the 16-half padding step can reach. Two tiles, so that whatever it
	// finds is attributable to the stride rather than to the kernel it was
	// found on.
	//
	// Both shapes are in it and they start from opposite ends — the gate/up
	// projection's natural stride is 1024 and the down projection's is 256 —
	// so the sweep separates "pad by 256 B" from "land on a gcd of 256",
	// which every measurement in this suite so far has been unable to tell
	// apart because every K it swept was a power of two.
	{name: "moe_reg64_bt_hka4_gcd16", spirv: shaders.GEMMWMMAMoEReg64BTHKA4, bm: 64, bn: 64, bk: 64, gcdTarget: 16, strideArm: true},
	{name: "moe_reg64_bt_hka4_gcd64", spirv: shaders.GEMMWMMAMoEReg64BTHKA4, bm: 64, bn: 64, bk: 64, gcdTarget: 64, strideArm: true},
	{name: "moe_reg64_bt_hka4_gcd128", spirv: shaders.GEMMWMMAMoEReg64BTHKA4, bm: 64, bn: 64, bk: 64, gcdTarget: 128, strideArm: true},
	{name: "moe_reg64_bt_hka4_gcd256", spirv: shaders.GEMMWMMAMoEReg64BTHKA4, bm: 64, bn: 64, bk: 64, gcdTarget: 256, strideArm: true},
	{name: "moe_reg64_bt_hka4_gcd512", spirv: shaders.GEMMWMMAMoEReg64BTHKA4, bm: 64, bn: 64, bk: 64, gcdTarget: 512, strideArm: true},
	{name: "moe_reg64_bt_hka4_gcd1024", spirv: shaders.GEMMWMMAMoEReg64BTHKA4, bm: 64, bn: 64, bk: 64, gcdTarget: 1024, strideArm: true},
	{name: "moe_reg64_bt_hka4_gcd2048", spirv: shaders.GEMMWMMAMoEReg64BTHKA4, bm: 64, bn: 64, bk: 64, gcdTarget: 2048, strideArm: true},
	{name: "moe_reg16x64_bt_hkab4_w32_gcd16", spirv: shaders.GEMMWMMAMoEReg16x64BTHKAB4W32, bm: 16, bn: 64, bk: 64, waveSize: 32, gcdTarget: 16, strideArm: true},
	{name: "moe_reg16x64_bt_hkab4_w32_gcd64", spirv: shaders.GEMMWMMAMoEReg16x64BTHKAB4W32, bm: 16, bn: 64, bk: 64, waveSize: 32, gcdTarget: 64, strideArm: true},
	{name: "moe_reg16x64_bt_hkab4_w32_gcd128", spirv: shaders.GEMMWMMAMoEReg16x64BTHKAB4W32, bm: 16, bn: 64, bk: 64, waveSize: 32, gcdTarget: 128, strideArm: true},
	{name: "moe_reg16x64_bt_hkab4_w32_gcd256", spirv: shaders.GEMMWMMAMoEReg16x64BTHKAB4W32, bm: 16, bn: 64, bk: 64, waveSize: 32, gcdTarget: 256, strideArm: true},
	{name: "moe_reg16x64_bt_hkab4_w32_gcd512", spirv: shaders.GEMMWMMAMoEReg16x64BTHKAB4W32, bm: 16, bn: 64, bk: 64, waveSize: 32, gcdTarget: 512, strideArm: true},
	{name: "moe_reg16x64_bt_hkab4_w32_gcd1024", spirv: shaders.GEMMWMMAMoEReg16x64BTHKAB4W32, bm: 16, bn: 64, bk: 64, waveSize: 32, gcdTarget: 1024, strideArm: true},
	{name: "moe_reg16x64_bt_hkab4_w32_gcd2048", spirv: shaders.GEMMWMMAMoEReg16x64BTHKAB4W32, bm: 16, bn: 64, bk: 64, waveSize: 32, gcdTarget: 2048, strideArm: true},
}

// moeStrideTokens is the batch the stride arm runs at: the real prefill chunk
// and nothing else.
const moeStrideTokens = 2048

// moeMinIters matches the shapes family's floor and for the same reason: the
// small-token cases here run for tens of microseconds, and at that length the
// suite's default of 20 iterations has been measured disagreeing with itself
// by up to 105% on its worst cell.
const moeMinIters = 200

// RunMoE measures the grouped GEMM, the per-expert baseline it replaces, and
// the gather/combine passes it needs (IDEAS §3.5).
func RunMoE(dev *vk.Device, phys *vk.PhysicalDevice, warmup, iters uint32) ([]Result, error) {
	if iters < moeMinIters {
		iters = moeMinIters
	}
	shape, ok, err := findCoopMatShape(phys, vk.ComponentFloat16, vk.ComponentFloat32)
	if err != nil {
		return nil, err
	}
	if !ok || shape.M != 16 || shape.N != 16 || shape.K != 16 {
		fmt.Fprintln(os.Stderr, "moe: no 16x16x16 cooperative-matrix shape, skipping")
		return nil, nil
	}

	routings := make(map[int]moeRouting, len(moeTokens))
	for _, t := range moeTokens {
		routings[t] = routeTokens(t)
	}

	results, err := runMoERoute(dev, routings, warmup, iters)
	if err != nil {
		return nil, err
	}

	variants := filterWaveVariants(moeVariants, mustSubgroupSizeControl(phys),
		func(v moeVariant) (string, uint32) { return "moe " + v.name, v.waveSize })

	for _, s := range moeShapes {
		rs, err := runMoEShape(dev, s, variants, routings, warmup, iters)
		if err != nil {
			return nil, err
		}
		results = append(results, rs...)
	}
	return results, nil
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

// moeRouting is one batch's top-k assignment: for each expert, which token
// rows were routed to it.
//
// The assignment is uniform-random rather than exactly balanced, because
// balance is the thing under test. Uniform top-k over 512 experts makes each
// expert's count Poisson-ish around tokens*topk/experts, so at the real
// prefill an expert gets 40 rows on average and 60-odd at the tail — and a
// per-expert dispatch loop pays for the tail twice over, once in tile padding
// and once in the dispatch that runs alone.
type moeRouting struct {
	tokens    int
	perExpert [][]int32 // token index, in expert order
	touched   int       // experts with at least one row
	maxRows   int
}

func routeTokens(tokens int) moeRouting {
	r := rand.New(rand.NewSource(20250913))
	out := moeRouting{tokens: tokens, perExpert: make([][]int32, moeExperts)}
	pick := make([]int, moeTopK)
	for t := 0; t < tokens; t++ {
		// Distinct experts per token, which is what top-k means; with
		// topk=10 of 512 a rejection loop retries about 1% of the time.
		for i := 0; i < moeTopK; i++ {
			for {
				e := r.Intn(moeExperts)
				dup := false
				for j := 0; j < i; j++ {
					if pick[j] == e {
						dup = true
						break
					}
				}
				if !dup {
					pick[i] = e
					break
				}
			}
		}
		for _, e := range pick {
			out.perExpert[e] = append(out.perExpert[e], int32(t))
		}
	}
	for _, rows := range out.perExpert {
		if len(rows) > 0 {
			out.touched++
		}
		if len(rows) > out.maxRows {
			out.maxRows = len(rows)
		}
	}
	return out
}

// rows is the total number of real (non-padding) expert rows, i.e. how many
// row-times-matrix products the layer actually needs.
func (r moeRouting) rows() int { return r.tokens * moeTopK }

// moeLayout places each expert's group in the gathered activation buffer,
// rounded up to the kernel's M tile so the kernel never has to mask: a
// cooperative-matrix A fragment reads all 16 of its rows whether they mean
// anything or not, and the gather writes zeros into the ones that don't.
type moeLayout struct {
	start []int // first row of expert e's group
	count []int // real rows in expert e's group
	total int   // padded rows in total
}

func layoutGroups(r moeRouting, bm int) moeLayout {
	lay := moeLayout{start: make([]int, moeExperts), count: make([]int, moeExperts)}
	for e, rows := range r.perExpert {
		lay.start[e] = lay.total
		lay.count[e] = len(rows)
		lay.total += roundUpTo(len(rows), bm)
	}
	return lay
}

// moeTiles is the tile table the grouped kernel reads, plus the slicing the
// per-expert baseline needs to walk the same table one expert at a time.
type moeTiles struct {
	table  []uint32 // 4 words per tile: A row, B row in the bank, C column
	base   []uint32 // first tile index of expert e
	counts []uint32 // tiles belonging to expert e
}

func (t moeTiles) total() int { return len(t.table) / 4 }

// buildTiles walks experts outermost so that consecutive workgroups in the
// grouped dispatch share an expert's weights. That ordering is free and it is
// the same lever IDEAS §2.4 calls a swizzle; it is not swept here, but it is
// the reason the grouped arm gets any weight reuse at all.
func buildTiles(r moeRouting, lay moeLayout, v moeVariant, N int) moeTiles {
	t := moeTiles{base: make([]uint32, moeExperts), counts: make([]uint32, moeExperts)}
	nTiles := N / v.bn
	for e := 0; e < moeExperts; e++ {
		t.base[e] = uint32(t.total())
		if lay.count[e] == 0 {
			continue
		}
		mTiles := (lay.count[e] + v.bm - 1) / v.bm
		for mt := 0; mt < mTiles; mt++ {
			for nt := 0; nt < nTiles; nt++ {
				t.table = append(t.table,
					uint32(lay.start[e]+mt*v.bm), // row in the gathered A and in C
					uint32(e*N+nt*v.bn),          // row in the [E*N, K] bank
					uint32(nt*v.bn),              // column in C
					0)
			}
		}
		t.counts[e] = uint32(mTiles * nTiles)
	}
	return t
}

// ---------------------------------------------------------------------------
// The grouped GEMM and its per-expert baseline
// ---------------------------------------------------------------------------

// moePushConstantSize is gemm_wmma.comp's block with GROUPED's extra word.
const moePushConstantSize = 28

func moePushConstants(N, K, ldb, lda, tileBase int) []byte {
	pc := newPC().U32(0).U32(uint32(N)).U32(uint32(K)).U32(0).
		U32(uint32(ldb)).U32(uint32(lda)).U32(uint32(tileBase)).Bytes()
	if len(pc) != moePushConstantSize {
		panic(fmt.Sprintf("moe push constants: built %d bytes, layout declares %d",
			len(pc), moePushConstantSize))
	}
	return pc
}

// moePatternBytes is the size of the repeating fp16 block the weight bank is
// filled with. The bank is 1.7-2.0 GB and only its addresses matter to a
// timing run; what does matter is that every half is a real small number, so
// that no stride through it can turn up an infinity.
const moePatternBytes = 1 << 20

func runMoEShape(dev *vk.Device, s moeShape, variants []moeVariant, routings map[int]moeRouting, warmup, iters uint32) ([]Result, error) {
	// Every buffer is allocated once per shape, at the largest extent any
	// (variant, token count) needs, and reused. The leading dimensions are
	// push constants, so a variant with a different pad addresses the same
	// allocation differently rather than needing its own — which is the only
	// reason a 2 GB bank can be swept over five kernels at all.
	maxLda, maxLdb, maxRows, maxTiles := 0, 0, 0, 0
	for _, v := range variants {
		if _, ok := v.padHalves(s.K); !ok {
			continue
		}
		if l := v.lda(s.K); l > maxLda {
			maxLda = l
		}
		if l := v.ldb(s.K); l > maxLdb {
			maxLdb = l
		}
		for _, t := range moeTokens {
			lay := layoutGroups(routings[t], v.bm)
			if lay.total > maxRows {
				maxRows = lay.total
			}
			if n := buildTiles(routings[t], lay, v, s.N).total(); n > maxTiles {
				maxTiles = n
			}
		}
	}

	aBuf, err := dev.NewBuffer(maxRows * maxLda * 2)
	if err != nil {
		return nil, fmt.Errorf("moe %s A buffer: %w", s.layer, err)
	}
	defer aBuf.Destroy()
	bBuf, err := dev.NewBuffer(moeExperts * s.N * maxLdb * 2)
	if err != nil {
		return nil, fmt.Errorf("moe %s expert bank (%.2f GB): %w", s.layer,
			float64(moeExperts*s.N*maxLdb*2)/1e9, err)
	}
	defer bBuf.Destroy()
	scalesBuf, err := dev.NewBuffer(4)
	if err != nil {
		return nil, err
	}
	defer scalesBuf.Destroy()
	cBuf, err := dev.NewBuffer(maxRows * s.N * 4)
	if err != nil {
		return nil, fmt.Errorf("moe %s C buffer: %w", s.layer, err)
	}
	defer cBuf.Destroy()
	tileBuf, err := dev.NewBuffer(maxTiles * 16)
	if err != nil {
		return nil, err
	}
	defer tileBuf.Destroy()

	pattern := float32SliceToFloat16Bytes(randomFloats(moePatternBytes / 2))
	aBuf.FillRepeating(pattern)
	bBuf.FillRepeating(pattern)

	var results []Result
	for _, v := range variants {
		if _, ok := v.padHalves(s.K); !ok {
			fmt.Fprintf(os.Stderr, "moe %s %s: no 16-half pad reaches gcd %d at K=%d, skipping\n",
				s.layer, v.name, v.gcdTarget, s.K)
			continue
		}
		mod, err := dev.NewShaderModule(v.spirv)
		if err != nil {
			return nil, err
		}
		defer mod.Destroy()
		// One correctness check per variant, at a synthetic size small enough
		// for a host reference. The timed cases below run over a bank filled
		// with a repeating pattern and are never checked — this is what
		// stands between a mistyped tile-table entry and a plausible number.
		if err := verifyMoEGrouped(dev, mod, v); err != nil {
			return nil, fmt.Errorf("moe %s correctness check: %w", v.name, err)
		}

		pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers:              []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf, tileBuf},
			PushConstantSize:     moePushConstantSize,
			RequiredSubgroupSize: v.waveSize,
		})
		if err != nil {
			return nil, fmt.Errorf("moe %s pipeline: %w", v.name, err)
		}
		defer pipe.Destroy()

		for _, tokens := range moeTokens {
			if v.strideArm && tokens != moeStrideTokens {
				continue
			}
			r := routings[tokens]
			lay := layoutGroups(r, v.bm)
			tiles := buildTiles(r, lay, v, s.N)
			tileBuf.WriteBytes(uint32SliceToBytes(tiles.table))

			base := moeResult(s, v, r, lay, tiles)

			// Grouped: every expert's tiles in one dispatch.
			pc := moePushConstants(s.N, s.K, v.ldb(s.K), v.lda(s.K), 0)
			ns, clocks, err := TimeDispatch(pipe, uint32(tiles.total()), 1, 1, warmup, iters, pc)
			if err != nil {
				return nil, fmt.Errorf("moe %s %s grouped t=%d: %w", s.layer, v.name, tokens, err)
			}
			results = append(results, base.finish("grouped", 1, ns, clocks))

			if v.strideArm {
				// The stride rows are not asking about the schedule.
				continue
			}

			// Per expert: the same table, one dispatch per expert that has
			// rows, with a barrier between each.
			var groups []uint32
			var pcs [][]byte
			for e := 0; e < moeExperts; e++ {
				if tiles.counts[e] == 0 {
					continue
				}
				groups = append(groups, tiles.counts[e])
				pcs = append(pcs, moePushConstants(s.N, s.K, v.ldb(s.K), v.lda(s.K), int(tiles.base[e])))
			}
			ns, clocks, err = TimeDispatchSequence(pipe, groups, 1, 1, warmup, iters, pcs)
			if err != nil {
				return nil, fmt.Errorf("moe %s %s per-expert t=%d: %w", s.layer, v.name, tokens, err)
			}
			results = append(results, base.finish("per_expert", len(groups), ns, clocks))
		}
	}
	return results, nil
}

// moeCase is everything about a (shape, variant, batch) that does not depend
// on which schedule ran it, so that the grouped and per-expert rows differ in
// exactly one field plus the timing.
type moeCase struct {
	shape   moeShape
	variant moeVariant
	tokens  int
	useful  float64 // FLOPs the model needs
	exec    float64 // FLOPs the padded tiles actually run
	weights float64 // bytes of expert weights one pass reads
	acts    float64 // bytes of A read and C written
	padRows int
	realRow int
	tiles   int
}

func moeResult(s moeShape, v moeVariant, r moeRouting, lay moeLayout, tiles moeTiles) moeCase {
	return moeCase{
		shape: s, variant: v, tokens: r.tokens,
		useful: 2 * float64(r.rows()) * float64(s.N) * float64(s.K),
		exec:   2 * float64(tiles.total()) * float64(v.bm) * float64(v.bn) * float64(s.K),
		// Only the experts that were routed to are read, which at one token
		// is 10 of 512 and at 2048 is all of them.
		weights: float64(r.touched) * float64(s.N) * float64(s.K) * 2,
		acts:    float64(lay.total)*float64(s.K)*2 + float64(lay.total)*float64(s.N)*4,
		padRows: lay.total, realRow: r.rows(), tiles: tiles.total(),
	}
}

func (c moeCase) finish(mode string, dispatches int, ns float64, clocks ClockStats) Result {
	return Result{
		Op: "moe", Variant: c.variant.name, WeightFormat: "fp16", Size: c.tokens,
		Detail: fmt.Sprintf("layer=%s;mode=%s;tokens=%d;rows=%d;padrows=%d;tile=%dx%dx%d;wave=%d;ai=%.0f;"+
			"strideB=%dB;gcd4K=%d;tiles=%d;dispatches=%d;wgperdispatch=%.0f;useful=%.0f%%;"+
			"exec_gflops=%.0f;weightGBps=%.0f;actGBps=%.0f",
			c.shape.layer, mode, c.tokens, c.realRow, c.padRows,
			c.variant.bm, c.variant.bn, c.variant.bk, c.variant.wave(), c.variant.intensity(),
			c.variant.ldb(c.shape.K)*2, gcdInt(c.variant.ldb(c.shape.K)*2, strideAliasChunk),
			c.tiles, dispatches, float64(c.tiles)/float64(dispatches), 100*c.useful/c.exec,
			c.exec/(ns/1e9)/1e9, c.weights/(ns/1e9)/1e9, c.acts/(ns/1e9)/1e9),
		NsPerIter: ns,
		// Useful FLOP/s, as in the shapes family: the FLOPs the model needs
		// over the time the padded dispatch took. The rate the kernel itself
		// achieved is exec_gflops in the detail, and the rate that actually
		// binds at these shapes is weightGBps.
		GFLOPS: c.useful / (ns / 1e9) / 1e9,
		GBPS:   (c.weights + c.acts) / (ns / 1e9) / 1e9,
		Clocks: clocks,
	}
}

// verifyMoEGrouped checks one grouped variant against a host reference at the
// smallest size that still exercises what the tile table can get wrong: three
// experts with deliberately awkward row counts (one that half-fills its last
// M tile, one with a single row, one that fills exactly two), two N tiles so
// the C column base and the B row base have to be different numbers, and two
// K slabs so the accumulator carries.
func verifyMoEGrouped(dev *vk.Device, mod *vk.ShaderModule, v moeVariant) error {
	const experts = 3
	N, K := v.bn*2, v.bk*2
	counts := []int{v.bm + 1, 1, v.bm * 2}

	r := moeRouting{tokens: 0, perExpert: make([][]int32, moeExperts), touched: experts}
	for e := 0; e < experts; e++ {
		for i := 0; i < counts[e]; i++ {
			r.perExpert[e] = append(r.perExpert[e], int32(r.tokens))
			r.tokens++
		}
	}
	lay := layoutGroups(r, v.bm)
	tiles := buildTiles(r, lay, v, N)

	lda, ldb := v.lda(K), v.ldb(K)
	aData := randomFloats(lay.total * K)
	// Padding rows read as zero, which is what the gather writes into them.
	for e := 0; e < moeExperts; e++ {
		for i := lay.count[e]; i < roundUpTo(lay.count[e], v.bm); i++ {
			row := lay.start[e] + i
			for k := 0; k < K; k++ {
				aData[row*K+k] = 0
			}
		}
	}
	// The bank as the host sees it: experts*N rows of K, each expert's block
	// being that expert's [N,K] weight matrix.
	bData := randomFloats(experts * N * K)

	aBuf, err := dev.NewBuffer(lay.total * lda * 2)
	if err != nil {
		return err
	}
	defer aBuf.Destroy()
	bBuf, err := dev.NewBuffer(moeExperts * N * ldb * 2)
	if err != nil {
		return err
	}
	defer bBuf.Destroy()
	scalesBuf, err := dev.NewBuffer(4)
	if err != nil {
		return err
	}
	defer scalesBuf.Destroy()
	cBuf, err := dev.NewBuffer(lay.total * N * 4)
	if err != nil {
		return err
	}
	defer cBuf.Destroy()
	tileBuf, err := dev.NewBuffer(len(tiles.table) * 4)
	if err != nil {
		return err
	}
	defer tileBuf.Destroy()

	aBuf.WriteBytes(float32SliceToFloat16Bytes(padRows(aData, lay.total, K, lda)))
	bBuf.WriteBytes(float32SliceToFloat16Bytes(padRows(bData, experts*N, K, ldb)))
	tileBuf.WriteBytes(uint32SliceToBytes(tiles.table))

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:              []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf, tileBuf},
		PushConstantSize:     moePushConstantSize,
		RequiredSubgroupSize: v.waveSize,
	})
	if err != nil {
		return err
	}
	defer pipe.Destroy()

	pc := moePushConstants(N, K, ldb, lda, 0)
	if _, err := pipe.DispatchTimed(uint32(tiles.total()), 1, 1, 1, pc); err != nil {
		return err
	}
	got := cBuf.ReadFloat32(lay.total * N)

	refA, refB := float16RoundTrip(aData), float16RoundTrip(bData)
	for e := 0; e < experts; e++ {
		// The kernel's B operand is expert e's block read column-major, i.e.
		// the [N,K] matrix transposed back to the [K,N] cpuGEMM wants.
		bE := transpose(refB[e*N*K:(e+1)*N*K], N, K)
		rows := lay.count[e]
		want := cpuGEMM(refA[lay.start[e]*K:(lay.start[e]+rows)*K], bE, rows, N, K)
		if err := compareMat(got[lay.start[e]*N:(lay.start[e]+rows)*N], want, 5e-2); err != nil {
			return fmt.Errorf("expert %d (%d rows): %w", e, rows, err)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Gather and combine
// ---------------------------------------------------------------------------

// moeRoutePushConstantSize matches shaders/moe_route.comp's block.
const moeRoutePushConstantSize = 16

// moeNoRow is the padding sentinel, matching NO_ROW in moe_route.comp.
const moeNoRow = 0xffffffff

// runMoERoute measures the two data-movement passes a grouped GEMM makes
// necessary. They are run at the residual width (2560), which is what both
// the gather into gate_up and the combine out of down actually move, once per
// MoE layer each.
//
// The gather is run at every distinct M tile in the variant list, because the
// padding it has to write is a function of that tile and is the one part of
// its cost the kernel choice controls.
func runMoERoute(dev *vk.Device, routings map[int]moeRouting, warmup, iters uint32) ([]Result, error) {
	bms := map[int]bool{}
	for _, v := range moeVariants {
		bms[v.bm] = true
	}
	var bmList []int
	for bm := range bms {
		bmList = append(bmList, bm)
	}
	sort.Ints(bmList)

	maxTokens := 0
	for _, t := range moeTokens {
		if t > maxTokens {
			maxTokens = t
		}
	}
	maxRows := 0
	for _, t := range moeTokens {
		for _, bm := range bmList {
			if n := layoutGroups(routings[t], bm).total; n > maxRows {
				maxRows = n
			}
		}
	}

	const K = moeHidden
	srcBuf, err := dev.NewBuffer(maxRows * K * 2)
	if err != nil {
		return nil, err
	}
	defer srcBuf.Destroy()
	idxBuf, err := dev.NewBuffer(maxRows * 4)
	if err != nil {
		return nil, err
	}
	defer idxBuf.Destroy()
	wtBuf, err := dev.NewBuffer(maxTokens * moeTopK * 4)
	if err != nil {
		return nil, err
	}
	defer wtBuf.Destroy()
	dstBuf, err := dev.NewBuffer(maxRows * K * 2)
	if err != nil {
		return nil, err
	}
	defer dstBuf.Destroy()

	pattern := float32SliceToFloat16Bytes(randomFloats(moePatternBytes / 2))
	srcBuf.FillRepeating(pattern)
	weights := make([]float32, maxTokens*moeTopK)
	for i := range weights {
		weights[i] = 1.0 / moeTopK
	}
	wtBuf.WriteFloat32(weights)

	modes := []struct {
		name  string
		spirv []byte
	}{{"gather", shaders.MoEGather}, {"combine", shaders.MoECombine}}

	var results []Result
	for _, m := range modes {
		mod, err := dev.NewShaderModule(m.spirv)
		if err != nil {
			return nil, err
		}
		defer mod.Destroy()
		pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers:          []*vk.Buffer{srcBuf, idxBuf, wtBuf, dstBuf},
			PushConstantSize: moeRoutePushConstantSize,
		})
		if err != nil {
			return nil, err
		}
		defer pipe.Destroy()

		for _, tokens := range moeTokens {
			r := routings[tokens]
			for _, bm := range bmList {
				lay := layoutGroups(r, bm)
				var rows int
				var idx []uint32
				var readBytes, writeBytes float64
				if m.name == "gather" {
					rows = lay.total
					idx = gatherIndices(r, lay)
					readBytes = float64(r.rows()) * K * 2
					writeBytes = float64(rows) * K * 2
				} else {
					// The combine is per token and does not depend on the M
					// tile, so it is run once — at the first tile size — and
					// not repeated per bm.
					if bm != bmList[0] {
						continue
					}
					rows = tokens
					idx = combineIndices(r, lay)
					readBytes = float64(r.rows()) * K * 2
					writeBytes = float64(tokens) * K * 2
				}
				idxBuf.WriteBytes(uint32SliceToBytes(idx))

				pc := newPC().U32(uint32(rows)).U32(K).U32(K).U32(K).Bytes()
				if len(pc) != moeRoutePushConstantSize {
					panic("moe route push constants: size mismatch")
				}
				ns, clocks, err := TimeDispatch(pipe, uint32(rows), 1, 1, warmup, iters, pc)
				if err != nil {
					return nil, fmt.Errorf("moe %s tokens=%d bm=%d: %w", m.name, tokens, bm, err)
				}
				results = append(results, Result{
					Op: "moe", Variant: m.name, WeightFormat: "fp16", Size: tokens,
					Detail: fmt.Sprintf("layer=route;mode=%s;tokens=%d;rows=%d;padrows=%d;bm=%d;readMB=%.0f;writeMB=%.0f",
						m.name, tokens, r.rows(), rows, bm, readBytes/1e6, writeBytes/1e6),
					NsPerIter: ns,
					GBPS:      (readBytes + writeBytes) / (ns / 1e9) / 1e9,
					Clocks:    clocks,
				})
			}
		}
	}
	return results, nil
}

// gatherIndices maps each row of the grouped layout back to its token, with
// the sentinel in the rows that pad a group up to the M tile.
func gatherIndices(r moeRouting, lay moeLayout) []uint32 {
	idx := make([]uint32, lay.total)
	for i := range idx {
		idx[i] = moeNoRow
	}
	for e, rows := range r.perExpert {
		for i, tok := range rows {
			idx[lay.start[e]+i] = uint32(tok)
		}
	}
	return idx
}

// combineIndices is the inverse: for each token, the moeTopK rows of the
// expert-output buffer that hold its partial results.
func combineIndices(r moeRouting, lay moeLayout) []uint32 {
	idx := make([]uint32, r.tokens*moeTopK)
	for i := range idx {
		idx[i] = moeNoRow
	}
	next := make([]int, r.tokens)
	for e, rows := range r.perExpert {
		for i, tok := range rows {
			idx[int(tok)*moeTopK+next[tok]] = uint32(lay.start[e] + i)
			next[tok]++
		}
	}
	return idx
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

// PrintMoESummary reads the flat table the three ways it has to be read: what
// the grouped schedule is worth against dispatching an expert at a time, what
// the M tile is worth against the padding it removes, and what a whole MoE
// layer — gather, two matmuls, combine — costs once the winners are picked.
func PrintMoESummary(w io.Writer, results []Result, p Params) {
	printMoEGroupedVsPerExpert(w, results)
	printMoETileGrid(w, results)
	printMoEOccupancy(w, results)
	printMoEStrideGrid(w, results)
	printMoERoute(w, results)
	printMoELayerBudget(w, results)
}

func printMoEGroupedVsPerExpert(w io.Writer, results []Result) {
	type cell struct {
		variant string
		gflops  float64
		ns      float64
		gbps    float64
		disp    string
	}
	best := map[string]map[string]cell{} // "layer|tokens" -> mode -> best
	var order []string
	for _, r := range results {
		mode := detailField(r.Detail, "mode")
		if mode != "grouped" && mode != "per_expert" {
			continue
		}
		// Geometry variants only: the stride arm runs grouped-only, so
		// letting it into the grouped column would compare the best stride
		// against a fixed one and call the difference a scheduling win.
		if moeIsStrideArm(r.Variant) {
			continue
		}
		key := detailField(r.Detail, "layer") + "|" + detailField(r.Detail, "tokens")
		if _, ok := best[key]; !ok {
			best[key] = map[string]cell{}
			order = append(order, key)
		}
		c := cell{variant: r.Variant, gflops: r.GFLOPS, ns: r.NsPerIter,
			disp: detailField(r.Detail, "dispatches")}
		c.gbps, _ = strconv.ParseFloat(detailField(r.Detail, "weightGBps"), 64)
		if old, ok := best[key][mode]; !ok || c.gflops > old.gflops {
			best[key][mode] = c
		}
	}
	if len(order) == 0 {
		return
	}
	fmt.Fprintf(w, "\ngrouped vs one dispatch per expert (%d experts top-%d, fp16 weights)\n", moeExperts, moeTopK)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "LAYER\tTOKENS\tROWS/EXPERT\tGROUPED US\tPER-EXPERT US\tSPEEDUP\tGROUPED GFLOP/S\tWEIGHT GB/S\tBEST KERNEL\tDISPATCHES")
	for _, k := range order {
		g, okG := best[k][mode1]
		pe, okP := best[k]["per_expert"]
		if !okG || !okP {
			continue
		}
		parts := strings.SplitN(k, "|", 2)
		tokens, _ := strconv.Atoi(parts[1])
		fmt.Fprintf(tw, "%s\t%d\t%.1f\t%.1f\t%.1f\t%.2fx\t%.0f\t%.0f\t%s\t%s -> 1\n",
			parts[0], tokens, float64(tokens*moeTopK)/moeExperts,
			g.ns/1e3, pe.ns/1e3, pe.ns/g.ns, g.gflops, g.gbps, g.variant, pe.disp)
	}
	tw.Flush()
	fmt.Fprintf(w, "  weight GB/s is the expert bank read once per pass against the %.0f GB/s DRAM bus:\n"+
		"  at these shapes that is the binding constraint, not the %.1f TFLOP/s of matrix cores\n",
		dramPeakGBPS, wmmaPeakTFLOPS)
}

// mode1 is the grouped mode's name, pulled out so the two spellings in the
// table above cannot drift apart.
const mode1 = "grouped"

func printMoETileGrid(w io.Writer, results []Result) {
	const tokens = "2048"
	type row struct {
		useful  string
		gflops  float64
		exec    float64
		gbps    float64
		padrows string
	}
	grid := map[string]map[string]row{} // variant -> layer -> row
	var variants, layers []string
	for _, r := range results {
		if detailField(r.Detail, "mode") != mode1 || detailField(r.Detail, "tokens") != tokens {
			continue
		}
		if v, ok := lookupMoEVariant(r.Variant); ok && v.strideArm {
			continue
		}
		layer := detailField(r.Detail, "layer")
		if _, ok := grid[r.Variant]; !ok {
			grid[r.Variant] = map[string]row{}
			variants = append(variants, r.Variant)
		}
		if !containsString(layers, layer) {
			layers = append(layers, layer)
		}
		e, _ := strconv.ParseFloat(detailField(r.Detail, "exec_gflops"), 64)
		g, _ := strconv.ParseFloat(detailField(r.Detail, "weightGBps"), 64)
		grid[r.Variant][layer] = row{
			useful: detailField(r.Detail, "useful"), gflops: r.GFLOPS, exec: e, gbps: g,
			padrows: detailField(r.Detail, "padrows"),
		}
	}
	if len(variants) == 0 {
		return
	}
	fmt.Fprintf(w, "\nthe M tile at the real prefill shape (2048 tokens, %.0f rows per expert)\n",
		float64(2048*moeTopK)/moeExperts)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprint(tw, "KERNEL\tTILE\tWAVE\tAI")
	for _, l := range layers {
		fmt.Fprintf(tw, "\t%s USEFUL\t%s GFLOP/S\t%s GB/S", l, l, l)
	}
	fmt.Fprintln(tw)
	for _, name := range variants {
		v, ok := lookupMoEVariant(name)
		if !ok {
			continue
		}
		fmt.Fprintf(tw, "%s\t%dx%dx%d\t%d\t%.0f", name, v.bm, v.bn, v.bk, v.wave(), v.intensity())
		for _, l := range layers {
			r := grid[name][l]
			fmt.Fprintf(tw, "\t%s\t%.0f\t%.0f", r.useful, r.gflops, r.gbps)
		}
		fmt.Fprintln(tw)
	}
	tw.Flush()
	fmt.Fprintln(w, "  useful is the share of the tiles' rows that are real tokens rather than group padding;")
	fmt.Fprintf(w, "  every row here carries the suite's +%d B stride pad, which the next table shows is the wrong\n"+
		"  one for these shapes\n", strideChannelChunk)
}

// printMoEStrideGrid is the stride arm: what gcd(row stride, 4096) is worth,
// at fixed tile, on two shapes whose natural strides sit at opposite ends of
// the range. It is here rather than in the stride family because these are
// the first two *non-power-of-two* reduction lengths anything in this suite
// has run a GEMM at, and they are the only ones that can separate "pad by
// 256 B" from "land on a gcd of 256".
func printMoEStrideGrid(w io.Writer, results []Result) {
	type key struct{ kernel, layer string }
	type cell struct {
		stride, gcd  int
		gflops, gbps float64
	}
	rows := map[key][]cell{}
	var kernels, layers []string
	for _, r := range results {
		v, ok := lookupMoEVariant(r.Variant)
		if !ok || !v.strideArm {
			continue
		}
		base := strings.SplitN(r.Variant, "_gcd", 2)[0]
		layer := detailField(r.Detail, "layer")
		k := key{base, layer}
		if _, seen := rows[k]; !seen && !containsString(kernels, base) {
			kernels = append(kernels, base)
		}
		if !containsString(layers, layer) {
			layers = append(layers, layer)
		}
		stride, _ := strconv.Atoi(strings.TrimSuffix(detailField(r.Detail, "strideB"), "B"))
		gcd, _ := strconv.Atoi(detailField(r.Detail, "gcd4K"))
		gbps, _ := strconv.ParseFloat(detailField(r.Detail, "weightGBps"), 64)
		rows[k] = append(rows[k], cell{stride, gcd, r.GFLOPS, gbps})
	}
	if len(kernels) == 0 {
		return
	}
	fmt.Fprintf(w, "\nrow stride against the %d B channel rotation, at %d tokens (grouped, same SPIR-V per kernel)\n",
		strideAliasChunk, moeStrideTokens)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "KERNEL\tLAYER\tK\tGCD\tSTRIDE B\tPAD B\tUSEFUL GFLOP/S\tWEIGHT GB/S\tCOVER")
	for _, kern := range kernels {
		for _, layer := range layers {
			cells := rows[key{kern, layer}]
			sort.Slice(cells, func(i, j int) bool { return cells[i].gcd < cells[j].gcd })
			K := 0
			for _, sh := range moeShapes {
				if sh.layer == layer {
					K = sh.K
				}
			}
			for _, c := range cells {
				fmt.Fprintf(tw, "%s\t%s\t%d\t%d\t%d\t%d\t%.0f\t%.0f\t%.2f\n",
					kern, layer, K, c.gcd, c.stride, c.stride-K*2, c.gflops, c.gbps,
					strideChannelCoverage(32*4, c.stride))
			}
		}
	}
	tw.Flush()
	fmt.Fprintf(w, "  pad B = 0 is the tensor as it comes; %d B is the pad every other kernel in this suite carries\n",
		strideChannelChunk)
	fmt.Fprintln(w, "  cover is IDEAS §5.1b's law, which orders these rows by gcd ascending — read it against what they did")
}

// printMoEOccupancy is the attribution for the table above: why the
// per-expert loop loses, and why it sometimes doesn't. A dispatch that covers
// one expert launches (its M tiles) x (N/BN) workgroups, and this part needs
// about 80 waves in flight to run a cooperative-matrix kernel at rate (§0.1).
// So the prediction is that the grouped schedule's win is a function of *that
// number* and not of the dispatch count — the launch itself costs ~300 ns
// (§4.1), which against a 20 us dispatch is nothing.
//
// Sorted by workgroups per expert dispatch, so the relationship is the
// ordering of the table rather than a claim about it.
func printMoEOccupancy(w io.Writer, results []Result) {
	type row struct {
		kernel, layer string
		tokens, wg    int
		speedup       float64
	}
	grouped := map[string]float64{}
	perExpert := map[string]struct {
		ns float64
		wg int
	}{}
	for _, r := range results {
		if moeIsStrideArm(r.Variant) {
			continue
		}
		key := r.Variant + "|" + detailField(r.Detail, "layer") + "|" + detailField(r.Detail, "tokens")
		switch detailField(r.Detail, "mode") {
		case mode1:
			grouped[key] = r.NsPerIter
		case "per_expert":
			wg, _ := strconv.Atoi(detailField(r.Detail, "wgperdispatch"))
			perExpert[key] = struct {
				ns float64
				wg int
			}{r.NsPerIter, wg}
		}
	}
	var rows []row
	for key, pe := range perExpert {
		g, ok := grouped[key]
		if !ok || g == 0 {
			continue
		}
		parts := strings.Split(key, "|")
		tokens, _ := strconv.Atoi(parts[2])
		rows = append(rows, row{parts[0], parts[1], tokens, pe.wg, pe.ns / g})
	}
	if len(rows) == 0 {
		return
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].wg < rows[j].wg })
	fmt.Fprintln(w, "\nwhy the per-expert loop loses: workgroups in one expert's dispatch vs what grouping is worth")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "WG/DISPATCH\tSPEEDUP\tLAYER\tTOKENS\tKERNEL")
	for _, r := range rows {
		fmt.Fprintf(tw, "%d\t%.2fx\t%s\t%d\t%s\n", r.wg, r.speedup, r.layer, r.tokens, r.kernel)
	}
	tw.Flush()
	fmt.Fprintf(w, "  this part wants ~%d waves resident to run WMMA at rate (IDEAS §0.1); one workgroup here is\n"+
		"  one wave, so a dispatch under that number leaves the machine idle and the barrier after it\n"+
		"  makes that idleness serial\n", moeWavesForRate)
}

// moeWavesForRate is §0.1's finding that the cooperative-matrix path needs two
// waves per CU — 80 on this 40-CU part — before it issues at full rate.
const moeWavesForRate = 80

func printMoERoute(w io.Writer, results []Result) {
	var rows []Result
	for _, r := range results {
		if m := detailField(r.Detail, "mode"); m == "gather" || m == "combine" {
			rows = append(rows, r)
		}
	}
	if len(rows) == 0 {
		return
	}
	fmt.Fprintf(w, "\nthe gather and combine a grouped GEMM needs (residual width %d, fp16)\n", moeHidden)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "PASS\tTOKENS\tBM\tROWS OUT\tREAD MB\tWRITE MB\tUS\tGB/S")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%.1f\t%.0f\n",
			r.Variant, detailField(r.Detail, "tokens"), detailField(r.Detail, "bm"),
			detailField(r.Detail, "padrows"), detailField(r.Detail, "readMB"),
			detailField(r.Detail, "writeMB"), r.NsPerIter/1e3, r.GBPS)
	}
	tw.Flush()
	fmt.Fprintf(w, "  the combine reads fp16 here; with the GEMM's fp32 C it would read twice this (IDEAS §2.6)\n")
}

// printMoELayerBudget adds the pieces up into the number §3.4 asked for: what
// one MoE block costs at prefill, and therefore what the 48 of them do to a
// prompt chunk.
func printMoELayerBudget(w io.Writer, results []Result) {
	const tokens = 2048
	tok := strconv.Itoa(tokens)
	best := map[string]float64{}     // layer|mode -> fastest ns
	winner := map[string]string{}    // layer|mode -> the kernel that was fastest
	gatherNs := map[string]float64{} // bm -> ns
	var combine float64
	for _, r := range results {
		if detailField(r.Detail, "tokens") != tok {
			continue
		}
		mode := detailField(r.Detail, "mode")
		switch mode {
		case mode1, "per_expert":
			if moeIsStrideArm(r.Variant) {
				continue
			}
			key := detailField(r.Detail, "layer") + "|" + mode
			if cur, ok := best[key]; !ok || r.NsPerIter < cur {
				best[key] = r.NsPerIter
				winner[key] = r.Variant
			}
		case "gather":
			gatherNs[detailField(r.Detail, "bm")] = r.NsPerIter
		case "combine":
			combine = r.NsPerIter
		}
	}
	gateUpG, ok1 := best["moe.gate_up|"+mode1]
	downG, ok2 := best["moe.down|"+mode1]
	gateUpP, ok3 := best["moe.gate_up|per_expert"]
	downP, ok4 := best["moe.down|per_expert"]
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return
	}
	// The gather is charged at the M tile the winning gate_up kernel forces,
	// because that tile is what decides how many padding rows it has to
	// write — a smaller tile that wins the matmul also makes the gather
	// cheaper, and averaging over tile sizes would hide that.
	gather := gatherNs[moeWinnerBM(winner["moe.gate_up|"+mode1])]
	// gate and up are two matrices of the same shape; down is one.
	grouped := 2*gateUpG + downG + gather + combine
	perExpert := 2*gateUpP + downP + gather + combine

	fmt.Fprintf(w, "\none MoE block at %d tokens, and the %d of them in a forward pass\n", tokens, moeLayers)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "SCHEDULE\tGATHER\tGATE+UP\tDOWN\tCOMBINE\tBLOCK\tx48 LAYERS\tDISPATCHES/BLOCK")
	fmt.Fprintf(tw, "grouped\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f s\t%d\n",
		gather/1e6, 2*gateUpG/1e6, downG/1e6, combine/1e6, grouped/1e6, float64(moeLayers)*grouped/1e9, 5)
	fmt.Fprintf(tw, "per expert\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f s\t%d\n",
		gather/1e6, 2*gateUpP/1e6, downP/1e6, combine/1e6, perExpert/1e6, float64(moeLayers)*perExpert/1e9,
		2+3*moeExperts)
	tw.Flush()
	fmt.Fprintf(w, "  grouped is %.2fx the whole block, and the routing passes are %.0f%% of its time\n",
		perExpert/grouped, 100*(gather+combine)/grouped)
	fmt.Fprintf(w, "  at fp16 the bank alone is %.2f GB per block, %.1f ms at the %.0f GB/s bus — which is the\n"+
		"  floor both schedules are working against, and the argument for a Q4 grouped kernel (IDEAS §2.2)\n",
		moeBankBytes()/1e9, moeBankBytes()/(dramPeakGBPS*1e9)*1e3, dramPeakGBPS)
	// Both rows above carry the suite's fixed stride pad, so that the two
	// schedules are comparable. The stride is a separate, multiplicative win.
	fmt.Fprintf(w, "  both rows use the fixed +%d B pad; the best stride in the table above is a further %.2fx on\n"+
		"  gate/up and %.2fx on down, which would take the grouped block to %.2f ms\n",
		strideChannelChunk, moeStrideGain(results, "moe.gate_up"), moeStrideGain(results, "moe.down"),
		(2*gateUpG/moeStrideGain(results, "moe.gate_up")+downG/moeStrideGain(results, "moe.down")+gather+combine)/1e6)
}

// moeBankBytes is the fp16 weight footprint of one MoE block: gate, up and
// down for every expert.
// moeIsStrideArm reports whether a result came from the stride sweep, which
// the two schedule tables have to leave out because it is grouped-only.
func moeIsStrideArm(name string) bool {
	v, ok := lookupMoEVariant(name)
	return ok && v.strideArm
}

// moeStrideGain is what moving a layer's row stride off the suite's fixed pad
// and onto the best one measured is worth, as a ratio, so the budget can say
// what it is leaving on the table without mixing the two arms into one row.
func moeStrideGain(results []Result, layer string) float64 {
	var fixed, best float64
	for _, r := range results {
		if detailField(r.Detail, "layer") != layer ||
			detailField(r.Detail, "mode") != mode1 ||
			detailField(r.Detail, "tokens") != strconv.Itoa(moeStrideTokens) {
			continue
		}
		if moeIsStrideArm(r.Variant) {
			if r.GFLOPS > best {
				best = r.GFLOPS
			}
		} else if r.GFLOPS > fixed {
			fixed = r.GFLOPS
		}
	}
	if fixed == 0 || best == 0 {
		return 1
	}
	return best / fixed
}

// moeWinnerBM is the M tile of a named variant, as the string the gather rows
// are keyed by.
func moeWinnerBM(name string) string {
	if v, ok := lookupMoEVariant(name); ok {
		return strconv.Itoa(v.bm)
	}
	return ""
}

func moeBankBytes() float64 {
	return float64(moeExperts) * (2*float64(moeFFN)*float64(moeHidden) + float64(moeHidden)*float64(moeFFN)) * 2
}

// wmmaPeakTFLOPS is §0.1's measured cooperative-matrix issue ceiling, the
// number the prefill rates in this file are *not* bounded by.
const wmmaPeakTFLOPS = 55.5

func lookupMoEVariant(name string) (moeVariant, bool) {
	for _, v := range moeVariants {
		if v.name == name {
			return v, true
		}
	}
	return moeVariant{}, false
}

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}
