package bench

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// This file is IDEAS §2.2 — the Q4 grouped GEMM — run against the shape §3.5
// left it with a number on.
//
// §3.5 measured the fp16 grouped kernel at **93% of the ceiling its format
// implies**: 190 GB/s of a 236 GB/s bus, 5.03 GB of expert weights per MoE
// block, 21.3 ms of unavoidable DRAM traffic inside a 28-30 ms block. There
// is nothing left in that kernel. The only remaining lever on the phase §3.4
// called 98% memory-bound is to read fewer bytes, and §1.1 already showed
// 4-bit weights feeding this chip's bus at 99-103% in the GEMV.
//
// So the question here is not "does Q4 save bytes" — it saves exactly 4x of
// them — but **what binds once they are saved**. The arithmetic (see the
// header of shaders/gemm_wmma_q4.comp) says the executed FLOPs take over:
// 108 GFLOP of padded tiles per gate_up pass against 1.9 ms of weight
// traffic. Which makes two predictions this file is built to test:
//
//  1. The speedup is ~2x, not 4x, because the matrix cores become the
//     constraint rather than the bus.
//  2. **§3.5's finding 2 inverts.** There, a 16-row tile bought 84% useful
//     rows instead of 62% and was *slower*, because the padding cost FLOPs
//     and FLOPs were free. Here they are not supposed to be free, so the
//     geometry arm is the same five tiles, unchanged, so the two formats'
//     tables can be read against each other row for row.
//
// Three further variants change one thing each against the first geometry,
// to price the parts of this kernel that the fp16 one does not have: the
// scale block (QBLOCK=128 takes the scales from 6.2% of the bank to 1.6%),
// the LDS row pad (without it the B slab's rows are exactly one rotation of
// the 32 LDS banks apart), and double buffering.
//
// The stride arm is the same test §3.5 ran on the fp16 bank, re-aimed: a
// 4-bit row is half the bytes of an fp16 one, so the two shapes land in
// different places against the 4 KB channel rotation. Unpadded, gate_up's
// 1280 B row is at gcd 256 — inside §5.1b's [128, 256] window — while down's
// 320 B row is at gcd 64, below it, which is the end §3.5 measured costing
// 5-17%. Both are swept, and for down the padding needed to climb into the
// window costs real bytes (384 B/row is 1.2x the traffic, 768 B is 2.4x), so
// the arm is a genuine trade rather than a free fix.

// moeQ4Variant is one build of gemm_wmma_q4.comp. The geometry fields mean
// what they mean in moeVariant; the rest are the axes only a 4-bit kernel
// has.
type moeQ4Variant struct {
	name       string
	spirv      []byte
	bm, bn, bk int
	waveSize   uint32
	// qblock is nibbles per fp16 scale along a row of B, baked into the
	// binary (it is a shift in the scale index, not a runtime divide).
	qblock int
	// ldsPad and doubleBuf are likewise compile-time; they are carried here
	// only so the summary can name what a row varied.
	ldsPad    int
	doubleBuf bool
	// scaleLayout is gemm_wmma_q4.comp's SCALE_LAYOUT: 0 row-major,
	// 1 k-major, 2 tile-blocked. LLM.md L0c. The host has to pack the plane
	// the same way, which is the only reason this is not purely a -D.
	scaleLayout int
	// padElems is B's row padding in *elements* (nibbles), used when
	// gcdTarget is zero. Zero means the bank as it comes.
	padElems int
	// gcdTarget picks the padding per shape instead: the smallest one that
	// lands gcd(row bytes, 4096) exactly there. See §5.1b rule 1.
	gcdTarget int
	strideArm bool
}

// moeQ4AGCD is the gcd the *activation* rows are padded to, for every row in
// this file. §3.5's finding 3 is that the suite's fixed +256 B pad is the
// wrong statement of its own rule — what matters is landing
// gcd(stride, 4096) in [128, 256] B — and the MoE shapes are the first here
// that can tell the two apart. The fp16 arm carries the old fixed pad so that
// it stays comparable with every other family; this one does not, because it
// is new and there is no reason to reproduce a known-wrong stride in it.
const moeQ4AGCD = 256

// q4PadStep is the granularity of a Q4 row pad, in elements. 32 nibbles is
// 16 bytes, which is one uvec4 staging load: anything finer would break the
// 16-byte alignment the kernel's buffer_load_b128 needs, and anything finer
// than QBLOCK would leave a scale block straddling the row end.
const q4PadStep = 32

func (v moeQ4Variant) wave() int {
	if v.waveSize == 0 {
		return 64
	}
	return int(v.waveSize)
}

func (v moeQ4Variant) intensity() float64 {
	return float64(v.bm*v.bn) / float64(v.bm+v.bn)
}

// ldb is B's row stride in elements (nibbles) at reduction length K, and
// whether this variant's stride is expressible there at all.
func (v moeQ4Variant) ldb(K int) (int, bool) {
	if v.gcdTarget == 0 {
		return K + v.padElems, true
	}
	for p := 0; p <= 16384; p += q4PadStep {
		if (K+p)%v.qblock != 0 {
			continue
		}
		if gcdInt((K+p)/2, strideAliasChunk) == v.gcdTarget {
			return K + p, true
		}
	}
	return 0, false
}

// lda is A's row stride in halves: the corrected rule, applied.
func (v moeQ4Variant) lda(K int) int {
	p, ok := padForGCD(K, moeQ4AGCD)
	if !ok {
		return K
	}
	return K + p
}

// moeQ4Variants: the geometry arm is §3.5's five tiles unchanged, then one
// row per Q4-only axis against the first of them, then the stride sweep.
var moeQ4Variants = []moeQ4Variant{
	{name: "moe_q4_reg64", spirv: shaders.GEMMWMMAQ4MoEReg64, bm: 64, bn: 64, bk: 64, qblock: 32, ldsPad: 8},
	{name: "moe_q4_reg32", spirv: shaders.GEMMWMMAQ4MoEReg32, bm: 32, bn: 32, bk: 64, qblock: 32, ldsPad: 8},
	{name: "moe_q4_reg32_w32", spirv: shaders.GEMMWMMAQ4MoEReg32W32, bm: 32, bn: 32, bk: 64, waveSize: 32, qblock: 32, ldsPad: 8},
	{name: "moe_q4_reg16x32_w32", spirv: shaders.GEMMWMMAQ4MoEReg16x32W32, bm: 16, bn: 32, bk: 64, waveSize: 32, qblock: 32, ldsPad: 8},
	{name: "moe_q4_reg16x64_w32", spirv: shaders.GEMMWMMAQ4MoEReg16x64W32, bm: 16, bn: 64, bk: 64, waveSize: 32, qblock: 32, ldsPad: 8},

	// One axis each against moe_q4_reg64.
	{name: "moe_q4_reg64_qb128", spirv: shaders.GEMMWMMAQ4MoEReg64QB128, bm: 64, bn: 64, bk: 64, qblock: 128, ldsPad: 8},
	{name: "moe_q4_reg64_ldspad0", spirv: shaders.GEMMWMMAQ4MoEReg64LDSPad0, bm: 64, bn: 64, bk: 64, qblock: 32, ldsPad: 0},
	{name: "moe_q4_reg64_db", spirv: shaders.GEMMWMMAQ4MoEReg64DB, bm: 64, bn: 64, bk: 64, qblock: 32, ldsPad: 8, doubleBuf: true},

	// LLM.md L0c — the scale plane's layout, against both scale-block sizes.
	// `moe_q4_reg64` and `moe_q4_reg64_qb128` above are the row-major pair
	// these subtract from, so the table reads as a 3x2: three layouts, two
	// values of QBLOCK, everything else held at §2.2's winning geometry.
	{name: "moe_q4_reg64_smk", spirv: shaders.GEMMWMMAQ4MoEReg64SMK, bm: 64, bn: 64, bk: 64, qblock: 32, ldsPad: 8, scaleLayout: 1},
	{name: "moe_q4_reg64_smk_qb128", spirv: shaders.GEMMWMMAQ4MoEReg64SMKQB128, bm: 64, bn: 64, bk: 64, qblock: 128, ldsPad: 8, scaleLayout: 1},
	{name: "moe_q4_reg64_smt", spirv: shaders.GEMMWMMAQ4MoEReg64SMT, bm: 64, bn: 64, bk: 64, qblock: 32, ldsPad: 8, scaleLayout: 2},
	{name: "moe_q4_reg64_smt_qb128", spirv: shaders.GEMMWMMAQ4MoEReg64SMTQB128, bm: 64, bn: 64, bk: 64, qblock: 128, ldsPad: 8, scaleLayout: 2},

	// The stride arm, grouped-only at the real prefill batch.
	{name: "moe_q4_reg64_gcd16", spirv: shaders.GEMMWMMAQ4MoEReg64, bm: 64, bn: 64, bk: 64, qblock: 32, ldsPad: 8, gcdTarget: 16, strideArm: true},
	{name: "moe_q4_reg64_gcd64", spirv: shaders.GEMMWMMAQ4MoEReg64, bm: 64, bn: 64, bk: 64, qblock: 32, ldsPad: 8, gcdTarget: 64, strideArm: true},
	{name: "moe_q4_reg64_gcd128", spirv: shaders.GEMMWMMAQ4MoEReg64, bm: 64, bn: 64, bk: 64, qblock: 32, ldsPad: 8, gcdTarget: 128, strideArm: true},
	{name: "moe_q4_reg64_gcd256", spirv: shaders.GEMMWMMAQ4MoEReg64, bm: 64, bn: 64, bk: 64, qblock: 32, ldsPad: 8, gcdTarget: 256, strideArm: true},
}

func lookupMoEQ4Variant(name string) (moeQ4Variant, bool) {
	for _, v := range moeQ4Variants {
		if v.name == name {
			return v, true
		}
	}
	return moeQ4Variant{}, false
}

// runMoEShapeQ4 is runMoEShape's Q4 twin: same routing, same tile table, same
// two schedules, with the expert bank at 4 bits per weight plus its scales.
func runMoEShapeQ4(dev *vk.Device, s moeShape, variants []moeQ4Variant, routings map[int]moeRouting, tokens []int, warmup, iters uint32) ([]Result, error) {
	maxLda, maxLdb, maxRows, maxTiles, minQBlock := 0, 0, 0, 0, 1<<30
	for _, v := range variants {
		ldb, ok := v.ldb(s.K)
		if !ok {
			continue
		}
		if ldb > maxLdb {
			maxLdb = ldb
		}
		if l := v.lda(s.K); l > maxLda {
			maxLda = l
		}
		if v.qblock < minQBlock {
			minQBlock = v.qblock
		}
		for _, t := range tokens {
			lay := layoutGroups(routings[t], v.bm)
			if lay.total > maxRows {
				maxRows = lay.total
			}
			if n := buildTiles(routings[t], lay, v.bm, v.bn, s.N).total(); n > maxTiles {
				maxTiles = n
			}
		}
	}
	if maxLdb == 0 {
		return nil, nil
	}

	aBuf, err := dev.NewBuffer(maxRows * maxLda * 2)
	if err != nil {
		return nil, fmt.Errorf("moe q4 %s A buffer: %w", s.layer, err)
	}
	defer aBuf.Destroy()
	bBytes := moeExperts * s.N * maxLdb / 2
	bBuf, err := dev.NewBuffer(bBytes)
	if err != nil {
		return nil, fmt.Errorf("moe q4 %s expert bank (%.2f GB): %w", s.layer, float64(bBytes)/1e9, err)
	}
	defer bBuf.Destroy()
	scalesBuf, err := dev.NewBuffer(moeExperts * s.N * (maxLdb / minQBlock) * 2)
	if err != nil {
		return nil, fmt.Errorf("moe q4 %s scales: %w", s.layer, err)
	}
	defer scalesBuf.Destroy()
	cBuf, err := dev.NewBuffer(maxRows * s.N * 4)
	if err != nil {
		return nil, fmt.Errorf("moe q4 %s C buffer: %w", s.layer, err)
	}
	defer cBuf.Destroy()
	tileBuf, err := dev.NewBuffer(maxTiles * 16)
	if err != nil {
		return nil, err
	}
	defer tileBuf.Destroy()

	// Timed runs only care about addresses, but every byte still has to
	// decode to something finite: the activations and the scales are real
	// fp16 numbers, and the nibbles can be anything at all.
	halves := float32SliceToFloat16Bytes(randomFloats(moePatternBytes / 2))
	aBuf.FillRepeating(halves)
	scalesBuf.FillRepeating(halves)
	bBuf.FillRepeating(randomBytes(moePatternBytes))

	var results []Result
	for _, v := range variants {
		ldb, ok := v.ldb(s.K)
		if !ok {
			fmt.Fprintf(os.Stderr, "moe q4 %s %s: no %d-element pad reaches gcd %d at K=%d, skipping\n",
				s.layer, v.name, q4PadStep, v.gcdTarget, s.K)
			continue
		}
		mod, err := dev.NewShaderModule(v.spirv)
		if err != nil {
			return nil, err
		}
		defer mod.Destroy()
		if err := verifyMoEGroupedQ4(dev, mod, v); err != nil {
			return nil, fmt.Errorf("moe q4 %s correctness check: %w", v.name, err)
		}

		pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers:              []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf, tileBuf},
			PushConstantSize:     moePushConstantSize,
			RequiredSubgroupSize: v.waveSize,
		})
		if err != nil {
			return nil, fmt.Errorf("moe q4 %s pipeline: %w", v.name, err)
		}
		defer pipe.Destroy()

		for _, t := range tokens {
			if v.strideArm && t != moeStrideTokens {
				continue
			}
			r := routings[t]
			lay := layoutGroups(r, v.bm)
			tiles := buildTiles(r, lay, v.bm, v.bn, s.N)
			tileBuf.WriteBytes(uint32SliceToBytes(tiles.table))

			base := moeQ4Result(s, v, r, lay, tiles, ldb)

			pc := moeQ4PushConstants(s.N, s.K, moeExperts*s.N, ldb, v.lda(s.K), 0)
			ns, clocks, err := TimeDispatch(pipe, uint32(tiles.total()), 1, 1, warmup, iters, pc)
			if err != nil {
				return nil, fmt.Errorf("moe q4 %s %s grouped t=%d: %w", s.layer, v.name, t, err)
			}
			results = append(results, base.finish(mode1, 1, ns, clocks))

			if v.strideArm {
				continue
			}

			var groups []uint32
			var pcs [][]byte
			for e := 0; e < moeExperts; e++ {
				if tiles.counts[e] == 0 {
					continue
				}
				groups = append(groups, tiles.counts[e])
				pcs = append(pcs, moeQ4PushConstants(s.N, s.K, moeExperts*s.N, ldb, v.lda(s.K), int(tiles.base[e])))
			}
			ns, clocks, err = TimeDispatchSequence(pipe, groups, 1, 1, warmup, iters, pcs)
			if err != nil {
				return nil, fmt.Errorf("moe q4 %s %s per-expert t=%d: %w", s.layer, v.name, t, err)
			}
			results = append(results, base.finish("per_expert", len(groups), ns, clocks))
		}
	}
	return results, nil
}

// moeQ4Case is moeCase with the weight bytes counted at 4 bits plus scales,
// which is the whole point: everything else about the measurement is
// deliberately identical so the two formats' rows subtract.
type moeQ4Case struct {
	shape   moeShape
	variant moeQ4Variant
	ldb     int
	tokens  int
	useful  float64
	exec    float64
	nibbles float64 // bytes of packed weights one pass reads
	scales  float64 // bytes of fp16 scales alongside them
	acts    float64
	padRows int
	realRow int
	tiles   int
}

func moeQ4Result(s moeShape, v moeQ4Variant, r moeRouting, lay moeLayout, tiles moeTiles, ldb int) moeQ4Case {
	return moeQ4Case{
		shape: s, variant: v, ldb: ldb, tokens: r.tokens,
		useful:  2 * float64(r.rows()) * float64(s.N) * float64(s.K),
		exec:    2 * float64(tiles.total()) * float64(v.bm) * float64(v.bn) * float64(s.K),
		nibbles: float64(r.touched) * float64(s.N) * float64(s.K) / 2,
		scales:  float64(r.touched) * float64(s.N) * float64(s.K/v.qblock) * 2,
		acts:    float64(lay.total)*float64(s.K)*2 + float64(lay.total)*float64(s.N)*4,
		padRows: lay.total, realRow: r.rows(), tiles: tiles.total(),
	}
}

func (c moeQ4Case) finish(mode string, dispatches int, ns float64, clocks ClockStats) Result {
	weights := c.nibbles + c.scales
	strideB := c.ldb / 2
	return Result{
		Op: "moe", Variant: c.variant.name, WeightFormat: "q4", BlockSize: c.variant.qblock,
		Size: c.tokens,
		Detail: fmt.Sprintf("layer=%s;mode=%s;tokens=%d;rows=%d;padrows=%d;tile=%dx%dx%d;wave=%d;ai=%.0f;"+
			"qblock=%d;ldspad=%d;db=%t;strideB=%dB;gcd4K=%d;tiles=%d;dispatches=%d;wgperdispatch=%.0f;"+
			"useful=%.0f%%;exec_gflops=%.0f;weightGBps=%.0f;scalepct=%.1f%%;actGBps=%.0f",
			c.shape.layer, mode, c.tokens, c.realRow, c.padRows,
			c.variant.bm, c.variant.bn, c.variant.bk, c.variant.wave(), c.variant.intensity(),
			c.variant.qblock, c.variant.ldsPad, c.variant.doubleBuf,
			strideB, gcdInt(strideB, strideAliasChunk),
			c.tiles, dispatches, float64(c.tiles)/float64(dispatches),
			100*c.useful/c.exec, c.exec/(ns/1e9)/1e9, weights/(ns/1e9)/1e9,
			100*c.scales/weights, c.acts/(ns/1e9)/1e9),
		NsPerIter: ns,
		GFLOPS:    c.useful / (ns / 1e9) / 1e9,
		GBPS:      (weights + c.acts) / (ns / 1e9) / 1e9,
		Clocks:    clocks,
	}
}

// verifyMoEGroupedQ4 is verifyMoEGrouped's Q4 twin, checking the same three
// awkward expert shapes — and, on top of what that one checks, the nibble
// order, the scale indexing and the fp16 bit trick the dequant is built on.
//
// The reference is dequantizeQ4 put back through fp16, because that is
// exactly what the kernel computes: (1024+n) - 1032 is exact in fp16 for
// every nibble, and the multiply by the fp16 scale is one rounded fp16
// operation. So a mismatch here is a bug and not a quantization artefact.
func verifyMoEGroupedQ4(dev *vk.Device, mod *vk.ShaderModule, v moeQ4Variant) error {
	const experts = 3
	N, K := v.bn*2, v.bk*2
	if K%v.qblock != 0 {
		K = v.qblock * 2
	}
	counts := []int{v.bm + 1, 1, v.bm * 2}

	r := moeRouting{tokens: 0, perExpert: make([][]int32, moeExperts), touched: experts}
	for e := 0; e < experts; e++ {
		for i := 0; i < counts[e]; i++ {
			r.perExpert[e] = append(r.perExpert[e], int32(r.tokens))
			r.tokens++
		}
	}
	lay := layoutGroups(r, v.bm)
	tiles := buildTiles(r, lay, v.bm, v.bn, N)

	lda, ok := v.ldb(K)
	if !ok {
		return fmt.Errorf("no pad reaches gcd %d at the check's K=%d", v.gcdTarget, K)
	}
	ldb := lda
	lda = v.lda(K)

	aData := randomFloats(lay.total * K)
	for e := 0; e < moeExperts; e++ {
		for i := lay.count[e]; i < roundUpTo(lay.count[e], v.bm); i++ {
			row := lay.start[e] + i
			for k := 0; k < K; k++ {
				aData[row*K+k] = 0
			}
		}
	}
	bData := randomFloats(experts * N * K)
	packed, scales := quantizeQ4(bData, experts*N, K, v.qblock)

	aBuf, err := dev.NewBuffer(lay.total * lda * 2)
	if err != nil {
		return err
	}
	defer aBuf.Destroy()
	bBuf, err := dev.NewBuffer(moeExperts * N * ldb / 2)
	if err != nil {
		return err
	}
	defer bBuf.Destroy()
	scalesBuf, err := dev.NewBuffer(moeExperts * N * (ldb / v.qblock) * 2)
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
	bBuf.WriteBytes(padPackedQ4Rows(packed, experts*N, K, ldb))
	// Packed over the plane's *allocated* height, not just the three experts
	// that carry data, so the k-major layout's stride is the one the kernel
	// is told about.
	scalesBuf.WriteBytes(packScalePlane(scales, experts*N, moeExperts*N, K/v.qblock, ldb/v.qblock, v.scaleLayout, v.bn))
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

	pc := moeQ4PushConstants(N, K, moeExperts*N, ldb, lda, 0)
	if _, err := pipe.DispatchTimed(uint32(tiles.total()), 1, 1, 1, pc); err != nil {
		return err
	}
	got := cBuf.ReadFloat32(lay.total * N)

	refA := float16RoundTrip(aData)
	refB := float16RoundTrip(dequantizeQ4(packed, scales, experts*N, K, v.qblock))
	for e := 0; e < experts; e++ {
		bE := transpose(refB[e*N*K:(e+1)*N*K], N, K)
		rows := lay.count[e]
		want := cpuGEMM(refA[lay.start[e]*K:(lay.start[e]+rows)*K], bE, rows, N, K)
		if err := compareMat(got[lay.start[e]*N:(lay.start[e]+rows)*N], want, 5e-2); err != nil {
			return fmt.Errorf("expert %d (%d rows): %w", e, rows, err)
		}
	}
	return nil
}

// padPackedQ4Rows re-lays a rows x cols nibble matrix at a wider row stride,
// in elements. Both cols and ldElems must be even, which q4PadStep enforces.
func padPackedQ4Rows(packed []uint8, rows, cols, ldElems int) []byte {
	out := make([]byte, rows*ldElems/2)
	for r := 0; r < rows; r++ {
		copy(out[r*ldElems/2:r*ldElems/2+cols/2], packed[r*cols/2:(r+1)*cols/2])
	}
	return out
}

// padScaleRows does the same for the fp16 scale plane, whose row is one entry
// per quantization block and whose stride follows B's.
// moeQ4PushConstants is moePushConstants with the slot the fp16 kernel leaves
// dead carrying the scale plane's leading dimension, which SCALE_LAYOUT=1
// needs and the other two ignore.
func moeQ4PushConstants(N, K, snrows, ldb, lda, tileBase int) []byte {
	pc := newPC().U32(0).U32(uint32(N)).U32(uint32(K)).U32(uint32(snrows)).
		U32(uint32(ldb)).U32(uint32(lda)).U32(uint32(tileBase)).Bytes()
	if len(pc) != moePushConstantSize {
		panic(fmt.Sprintf("moe q4 push constants: built %d bytes, layout declares %d",
			len(pc), moePushConstantSize))
	}
	return pc
}

// packScalePlane writes the fp16 scale plane in one of gemm_wmma_q4.comp's
// three SCALE_LAYOUTs (LLM.md L0c). srcRows rows carry data; the plane is
// planeRows tall, because the k-major layout's stride is the whole bank's
// height and the kernel is told that number, not the populated one.
//
// The three expressions here must stay the mirror image of the shader's
// SCALE_IDX. That is what the correctness check is for: a layout that the two
// sides disagree about produces wrong numbers, not slow ones.
func packScalePlane(scales []uint16, srcRows, planeRows, blocksPerRow, ldBlocks, layout, bn int) []byte {
	if planeRows < srcRows {
		planeRows = srcRows
	}
	out := make([]byte, planeRows*ldBlocks*2)
	for r := 0; r < srcRows; r++ {
		for b := 0; b < blocksPerRow; b++ {
			var i int
			switch layout {
			case 1:
				i = b*planeRows + r
			case 2:
				i = ((r/bn)*ldBlocks+b)*bn + r%bn
			default:
				i = r*ldBlocks + b
			}
			binary.LittleEndian.PutUint16(out[i*2:], scales[r*blocksPerRow+b])
		}
	}
	return out
}

func padScaleRows(scales []uint16, rows, blocksPerRow, ldBlocks int) []byte {
	out := make([]byte, rows*ldBlocks*2)
	for r := 0; r < rows; r++ {
		for b := 0; b < blocksPerRow; b++ {
			binary.LittleEndian.PutUint16(out[(r*ldBlocks+b)*2:], scales[r*blocksPerRow+b])
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

// printMoEQ4Grid is the head-to-head: every Q4 geometry against the best fp16
// grouped kernel at the same layer and batch, at the real prefill shape.
func printMoEQ4Grid(w io.Writer, results []Result) {
	const tokens = "2048"
	type cell struct{ ns, gflops, exec, gbps float64 }
	grid := map[string]map[string]cell{}
	fp16 := map[string]cell{}
	var variants, layers []string
	for _, r := range results {
		if detailField(r.Detail, "mode") != mode1 || detailField(r.Detail, "tokens") != tokens {
			continue
		}
		layer := detailField(r.Detail, "layer")
		e, _ := strconv.ParseFloat(detailField(r.Detail, "exec_gflops"), 64)
		g, _ := strconv.ParseFloat(detailField(r.Detail, "weightGBps"), 64)
		c := cell{r.NsPerIter, r.GFLOPS, e, g}
		switch r.WeightFormat {
		case "fp16":
			if moeIsStrideArm(r.Variant) {
				continue
			}
			if old, ok := fp16[layer]; !ok || c.gflops > old.gflops {
				fp16[layer] = c
			}
		case "q4":
			if v, ok := lookupMoEQ4Variant(r.Variant); !ok || v.strideArm {
				continue
			}
			if _, ok := grid[r.Variant]; !ok {
				grid[r.Variant] = map[string]cell{}
				variants = append(variants, r.Variant)
			}
			grid[r.Variant][layer] = c
			if !containsString(layers, layer) {
				layers = append(layers, layer)
			}
		}
	}
	if len(variants) == 0 {
		return
	}
	fmt.Fprintf(w, "\nQ4 expert bank vs fp16, grouped, at the real prefill shape (%s tokens, %.0f rows per expert)\n",
		tokens, float64(2048*moeTopK)/moeExperts)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprint(tw, "KERNEL\tTILE\tWAVE\tQBLOCK")
	for _, l := range layers {
		fmt.Fprintf(tw, "\t%s MS\t%s GFLOP/S\t%s EXEC\t%s GB/S\t%s vs FP16", l, l, l, l, l)
	}
	fmt.Fprintln(tw)
	for _, name := range variants {
		v, _ := lookupMoEQ4Variant(name)
		fmt.Fprintf(tw, "%s\t%dx%dx%d\t%d\t%d", name, v.bm, v.bn, v.bk, v.wave(), v.qblock)
		for _, l := range layers {
			c := grid[name][l]
			ratio := 0.0
			if b, ok := fp16[l]; ok && c.gflops > 0 {
				ratio = c.gflops / b.gflops
			}
			fmt.Fprintf(tw, "\t%.2f\t%.0f\t%.0f\t%.0f\t%.2fx", c.ns/1e6, c.gflops, c.exec, c.gbps, ratio)
		}
		fmt.Fprintln(tw)
	}
	fmt.Fprint(tw, "best fp16\t-\t-\t-")
	for _, l := range layers {
		c := fp16[l]
		fmt.Fprintf(tw, "\t%.2f\t%.0f\t%.0f\t%.0f\t1.00x", c.ns/1e6, c.gflops, c.exec, c.gbps)
	}
	fmt.Fprintln(tw)
	tw.Flush()
	fmt.Fprintf(w, "  GB/S is the bank as read — packed nibbles plus fp16 scales — against the %.0f GB/s bus;\n"+
		"  EXEC is the padded tiles' own FLOP rate against the %.1f TFLOP/s of matrix cores, and which of\n"+
		"  those two is near its ceiling is what says whether Q4 has moved this shape off the memory system\n",
		dramPeakGBPS, wmmaPeakTFLOPS)
}

// printMoEQ4ScaleLayout is LLM.md **L0c**: the scale plane's layout against
// the scale block size, three by two, everything else held at §2.2's winning
// geometry.
//
// The number this exists to explain is §2.2 finding 4 — QBLOCK=128 beats
// QBLOCK=32 by 1.24x between instruction-identical binaries — and the reading
// is one comparison: **k-major at QBLOCK=32 against row-major at QBLOCK=128**.
// If they meet, the 1.24x was locality, a finer scale block is free once the
// plane is laid out for the staging step, and phase 2 can have block-32
// accuracy at block-128 speed. If they do not, the cause is still open and a
// finer block still costs.
func printMoEQ4ScaleLayout(w io.Writer, results []Result) {
	const tokens = "2048"
	names := map[int]string{0: "row-major", 1: "k-major", 2: "tile-blocked"}
	// [layout][qblock][layer] -> ms
	ms := map[[3]string]float64{}
	var layers []string
	for _, r := range results {
		if r.WeightFormat != "q4" || detailField(r.Detail, "mode") != mode1 ||
			detailField(r.Detail, "tokens") != tokens {
			continue
		}
		v, ok := lookupMoEQ4Variant(r.Variant)
		if !ok || v.strideArm || v.bn != 64 || v.bm != 64 || v.ldsPad != 8 || v.doubleBuf {
			continue
		}
		layer := detailField(r.Detail, "layer")
		ms[[3]string{strconv.Itoa(v.scaleLayout), strconv.Itoa(v.qblock), layer}] = r.NsPerIter / 1e6
		if !containsString(layers, layer) {
			layers = append(layers, layer)
		}
	}
	if len(layers) == 0 {
		return
	}
	fmt.Fprintf(w, "\nL0c — the scale plane's layout (%s tokens, 64x64x64 tile, LDS_PAD=8)\n", tokens)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprint(tw, "SCALE LAYOUT")
	for _, l := range layers {
		fmt.Fprintf(tw, "\t%s qb32\t%s qb128\t%s qb32 gain", l, l, l)
	}
	fmt.Fprintln(tw)
	for _, lay := range []int{0, 1, 2} {
		fmt.Fprintf(tw, "%s", names[lay])
		for _, l := range layers {
			a := ms[[3]string{strconv.Itoa(lay), "32", l}]
			b := ms[[3]string{strconv.Itoa(lay), "128", l}]
			base := ms[[3]string{"0", "32", l}]
			gain := "-"
			if a > 0 && base > 0 {
				gain = fmt.Sprintf("%.2fx", base/a)
			}
			fmt.Fprintf(tw, "\t%.2f\t%.2f\t%s", a, b, gain)
		}
		fmt.Fprintln(tw)
	}
	tw.Flush()
	fmt.Fprint(w, "  ms per dispatch; \"qb32 gain\" is against row-major qb32, the configuration §2.2 measured.\n"+
		"  The comparison that settles it is k-major qb32 against row-major qb128.\n")
}

// printMoEQ4StrideGrid is §5.1b's rule applied to a 4-bit row, which is half
// the bytes of the fp16 one and so lands somewhere else against the 4 KB
// rotation — and where, unlike the fp16 arm, climbing into the window costs
// real traffic.
func printMoEQ4StrideGrid(w io.Writer, results []Result) {
	type cell struct {
		stride, gcd, pad int
		gflops, gbps     float64
	}
	rows := map[string][]cell{}
	var layers []string
	for _, r := range results {
		if r.WeightFormat != "q4" {
			continue
		}
		v, ok := lookupMoEQ4Variant(r.Variant)
		if !ok || !v.strideArm {
			continue
		}
		layer := detailField(r.Detail, "layer")
		if !containsString(layers, layer) {
			layers = append(layers, layer)
		}
		stride, _ := strconv.Atoi(strings.TrimSuffix(detailField(r.Detail, "strideB"), "B"))
		gcd, _ := strconv.Atoi(detailField(r.Detail, "gcd4K"))
		gbps, _ := strconv.ParseFloat(detailField(r.Detail, "weightGBps"), 64)
		K := 0
		for _, sh := range moeShapes {
			if sh.layer == layer {
				K = sh.K
			}
		}
		rows[layer] = append(rows[layer], cell{stride, gcd, stride - K/2, r.GFLOPS, gbps})
	}
	if len(layers) == 0 {
		return
	}
	fmt.Fprintf(w, "\nQ4 row stride against the %d B channel rotation, at %d tokens (grouped, one binary)\n",
		strideAliasChunk, moeStrideTokens)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "LAYER\tGCD\tSTRIDE B\tPAD B\tEXTRA TRAFFIC\tUSEFUL GFLOP/S\tWEIGHT GB/S")
	for _, l := range layers {
		cells := rows[l]
		sort.Slice(cells, func(i, j int) bool { return cells[i].gcd < cells[j].gcd })
		for _, c := range cells {
			fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%.2fx\t%.0f\t%.0f\n",
				l, c.gcd, c.stride, c.pad, float64(c.stride)/float64(c.stride-c.pad), c.gflops, c.gbps)
		}
	}
	tw.Flush()
	fmt.Fprintln(w, "  a 4-bit row is half an fp16 one, so the pad that reaches a given gcd is twice as large a")
	fmt.Fprintln(w, "  share of it: on down (320 B/row) the window at [128, 256] B costs 1.20-2.40x the traffic")
}

// printMoEQ4Budget is printMoELayerBudget's Q4 column: what the block costs
// once the bank is 4-bit, which is the number IDEAS §2.2 exists to produce.
func printMoEQ4Budget(w io.Writer, results []Result) {
	const tokens = 2048
	tok := strconv.Itoa(tokens)
	best := map[string]float64{}
	winner := map[string]string{}
	gatherNs := map[string]float64{}
	var combine float64
	for _, r := range results {
		if detailField(r.Detail, "tokens") != tok {
			continue
		}
		switch detailField(r.Detail, "mode") {
		case mode1:
			if r.WeightFormat == "q4" {
				if v, ok := lookupMoEQ4Variant(r.Variant); !ok || v.strideArm {
					continue
				}
			} else if moeIsStrideArm(r.Variant) {
				continue
			}
			key := r.WeightFormat + "|" + detailField(r.Detail, "layer")
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
	q4Up, ok1 := best["q4|moe.gate_up"]
	q4Down, ok2 := best["q4|moe.down"]
	fpUp, ok3 := best["fp16|moe.gate_up"]
	fpDown, ok4 := best["fp16|moe.down"]
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return
	}
	q4BM := ""
	if v, ok := lookupMoEQ4Variant(winner["q4|moe.gate_up"]); ok {
		q4BM = strconv.Itoa(v.bm)
	}
	q4Block := 2*q4Up + q4Down + gatherNs[q4BM] + combine
	fpBlock := 2*fpUp + fpDown + gatherNs[moeWinnerBM(winner["fp16|moe.gate_up"])] + combine

	fmt.Fprintf(w, "\none MoE block at %d tokens: what the weight format is worth\n", tokens)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "FORMAT\tBANK/BLOCK\tGATHER\tGATE+UP\tDOWN\tCOMBINE\tBLOCK\tx48 LAYERS\tBEST KERNEL")
	fmt.Fprintf(tw, "fp16\t%.2f GB\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f s\t%s\n",
		moeBankBytes()/1e9, gatherNs[moeWinnerBM(winner["fp16|moe.gate_up"])]/1e6, 2*fpUp/1e6, fpDown/1e6,
		combine/1e6, fpBlock/1e6, float64(moeLayers)*fpBlock/1e9, winner["fp16|moe.gate_up"])
	fmt.Fprintf(tw, "q4\t%.2f GB\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f ms\t%.2f s\t%s\n",
		moeQ4BankBytes(32)/1e9, gatherNs[q4BM]/1e6, 2*q4Up/1e6, q4Down/1e6,
		combine/1e6, q4Block/1e6, float64(moeLayers)*q4Block/1e9, winner["q4|moe.gate_up"])
	tw.Flush()
	fmt.Fprintf(w, "  Q4 is %.2fx the whole block and %.2fx the matmuls alone; the bank's own floor falls from\n"+
		"  %.1f ms to %.1f ms at the %.0f GB/s bus, so the gap between those two ratios is what did not follow\n"+
		"  the bytes down\n",
		fpBlock/q4Block, (2*fpUp+fpDown)/(2*q4Up+q4Down),
		moeBankBytes()/(dramPeakGBPS*1e9)*1e3, moeQ4BankBytes(32)/(dramPeakGBPS*1e9)*1e3, dramPeakGBPS)
}

// moeQ4BankBytes is the 4-bit weight footprint of one MoE block, scales
// included, at a given quantization block size.
func moeQ4BankBytes(qblock int) float64 {
	weights := float64(moeExperts) * (2*float64(moeFFN)*float64(moeHidden) + float64(moeHidden)*float64(moeFFN))
	return weights/2 + weights/float64(qblock)*2
}
