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

// This file is the item IDEAS §3.5 left open and §2.2 made worth doing: the
// **grouped GEMV**, which is what a MoE model's decode step actually needs and
// what nothing in this suite had.
//
// Why the existing decode rows are not a decode measurement. §3.5's `moe`
// family runs its grouped kernel at one token as well as at 512/2048/8192,
// but that kernel is a *GEMM*: its A operand is a 16-row cooperative-matrix
// fragment, so a one-row group is 1/16th of a tile and the other 15 rows are
// zeros the machine still loads, multiplies and accumulates. The weights are
// read once either way — which is why §3.5 called the waste free — but A's
// traffic is multiplied by the tile height (a BM=64 tile reads 64 rows of K
// halves to use one), and the FLOPs by the same factor. And the 10 experts a
// single token touches are 8-25 MB of 4-bit weights per projection, which sits
// *inside* the 32 MiB MALL, so the timing loop re-reads them out of cache.
//
// What this arm changes:
//
//  1. **The kernel.** `gemv_w4a8.comp -DGROUPED=1`: one subgroup per output
//     row, one workgroup per (row, routed pair), the pair's bank row,
//     activation row and output row read from a table. No tile, so no padding
//     and no wasted A traffic — and no gather pass either, because the gather
//     in the GEMM path exists only to make each expert's rows *contiguous* for
//     a fragment load, and a subgroup can read whichever row a table names.
//  2. **The batch.** A decode step is one token per sequence, so the axis that
//     matters is how many sequences are in flight: 1, 4, 16 and 64. That is
//     also what moves the measurement off the MALL — 10 experts at one token
//     is 8 MB of gate_up, 640 pairs at 64 tokens touch ~400 experts and
//     328 MB. Rows per expert stay at 1.0-1.6 throughout, so every one of them
//     is a GEMV shape; the batch is not a way of sneaking a GEMM in.
//  3. **The row stride**, which is the second open item in the handoff and
//     costs nothing to fold in here. §3.5's finding 3 amended §5.1b's first
//     engine rule from "pad by 256 B" to "land gcd(stride, 4096) in
//     [128, 256] B", and the down projection's 4-bit row is 320 B at gcd 64 —
//     below the plateau. The bank here is [E*N, ldw] with ldw a push constant,
//     so the same binary sweeps it. Every stride in the sweep is a multiple of
//     64 B, so each row still starts on a cache line and the pad is never
//     *read*: unlike §2.2's arm, which had to buy coverage with 1.2-2.4x the
//     traffic, this one buys it with nothing but footprint.
//
// The baseline it is measured against is the same one §3.5 used, in both
// senses: the same kernel scheduled one dispatch at a time, and the Q4 grouped
// GEMM (§2.2) run over the same routing at the same batches, which is the
// thing "decode uses the GEMM at M=1" means in practice.

// moeDecodeTokens is the decode batch sweep: sequences decoding at once, each
// contributing one token. See the head of this file for why it runs to 64 —
// the footprint, not the shape, is what needs the larger batches.
var moeDecodeTokens = []int{1, 4, 16, 64}

// moeGEMVStrideTokens is the batch the stride arm runs at: the largest one, so
// that what it measures is a DRAM stride effect rather than a MALL one.
const moeGEMVStrideTokens = 64

// mallBytes is this part's last-level cache, the line a decode footprint has
// to be read against (IDEAS §0.3/§5.1b). Under it, a repeated dispatch is
// measuring the cache; over it, DRAM.
const mallBytes = 32 << 20

// moeGEMVPadStep is the granularity of a weight-bank row pad, in elements
// (nibbles). 128 nibbles is 64 B — one cache line — which is the property that
// makes the stride arm a clean experiment: every row of every stride in it
// starts on a line boundary, so the padding changes which channels a row
// spans without changing how many bytes come off the bus.
const moeGEMVPadStep = 128

// moeGEMVVariant is one build of gemv_w4a8.comp -DGROUPED=1, plus the two
// things the host varies without recompiling (the quantization block, which
// is a pushed log2, and the bank's row stride).
type moeGEMVVariant struct {
	name  string
	spirv []byte
	// weightsPerLoad is 8*VEC: the weights one lane consumes per loop step,
	// which is the axis §1.7's rule is stated in.
	weightsPerLoad int
	waveSize       uint32
	// block is nibbles per fp16 scale. It must be at least weightsPerLoad, so
	// the load-width sweep is run at 128 — the only block every width can
	// take — and the cost of a 32-nibble block is priced by one extra row.
	block int
	// gcdTarget, when non-zero, picks the bank's row stride per shape: the
	// smallest 64 B-aligned one whose gcd with the 4 KB channel rotation is
	// exactly this. Zero means the natural stride, K.
	gcdTarget int
	strideArm bool
}

func (v moeGEMVVariant) wave() int {
	if v.waveSize == 0 {
		return 64
	}
	return int(v.waveSize)
}

// vec is the shader's VEC, i.e. uints of weights per lane per step.
func (v moeGEMVVariant) vec() int { return v.weightsPerLoad / 8 }

// coverage is §1.7's C: the contiguous run of one weight row, in bytes, that
// one lane-step of a wave holds. The rule is that the bus is reached when this
// is the whole row (K/2 bytes) and missed when it is not.
func (v moeGEMVVariant) coverage(K int) int {
	lanes := v.wave()
	if steps := K / v.weightsPerLoad; steps < lanes {
		lanes = steps
	}
	return lanes * v.vec() * 4
}

// ldw is the bank's row stride in elements (nibbles) at reduction length K,
// and whether this variant's target is reachable there. The constraints are
// the shader's: a multiple of 32 nibbles so every row starts on a uvec4, and
// a multiple of the quantization block so no scale block straddles a row end.
// The pad step of 64 B is stricter than both and is there for the reason in
// the file header.
func (v moeGEMVVariant) ldw(K int) (int, bool) {
	if v.gcdTarget == 0 {
		return K, K%moeGEMVPadStep == 0 && K%v.block == 0
	}
	// Capped at 4x the row: past that the arm is measuring a footprint change
	// rather than a stride one, and the bank stops fitting in a buffer.
	for p := 0; p <= 3*K; p += moeGEMVPadStep {
		if (K+p)%v.block != 0 {
			continue
		}
		if gcdInt((K+p)/2, strideAliasChunk) == v.gcdTarget {
			return K + p, true
		}
	}
	return 0, false
}

// moeGEMVVariants: the load-width grid first, then the two axes that are not
// load width, then the stride arm.
//
// §1.7's rule predicts a different winner for each of the two expert shapes,
// which is the first thing this grid is for: one lane-step should cover a
// whole weight row, VEC = K/(8*WAVE), so the 4-bit K=640 row (320 B) wants
// VEC=4 at wave64 and the K=2560 one (1280 B) wants VEC=8. Both predictions
// are inside the swept range in both directions, so the grid can be wrong in
// either.
var moeGEMVVariants = []moeGEMVVariant{
	{name: "moe_gemv_v1", spirv: shaders.GEMVW4A8Grouped, weightsPerLoad: 8, block: 128},
	{name: "moe_gemv_v4", spirv: shaders.GEMVW4A8GroupedVec4, weightsPerLoad: 32, block: 128},
	{name: "moe_gemv_v8", spirv: shaders.GEMVW4A8GroupedVec8, weightsPerLoad: 64, block: 128},
	{name: "moe_gemv_v16", spirv: shaders.GEMVW4A8GroupedVec16, weightsPerLoad: 128, block: 128},

	// The wave32 controls. §1.7 concluded the wave size was never the cause of
	// anything once C is held fixed; these two hold C fixed against the wave64
	// arms above (v4_w32 covers what v1 does at wave64 at K=640, v8_w32 what
	// v4 does) and are here to say whether that still holds at a shape whose
	// rows are addressed through a table.
	{name: "moe_gemv_v4_w32", spirv: shaders.GEMVW4A8GroupedVec4W32, weightsPerLoad: 32, waveSize: 32, block: 128},
	{name: "moe_gemv_v8_w32", spirv: shaders.GEMVW4A8GroupedVec8W32, weightsPerLoad: 64, waveSize: 32, block: 128},

	// The scale plane, priced against moe_gemv_v4. A 32-nibble block is 6.25%
	// of the bank's bytes against 1.56% at 128, which is the whole difference
	// here — unlike §2.2's finding 4, where two instruction-identical GEMM
	// binaries differed by 1.24x and the bytes did not explain it. A GEMV
	// reads the scale plane the same way it reads the weights, so if the bytes
	// are the mechanism this row costs 4.4% and nothing else.
	{name: "moe_gemv_v4_qb32", spirv: shaders.GEMVW4A8GroupedVec4, weightsPerLoad: 32, block: 32},

	// The stride arm (the handoff's "K=640 is explained, and the fix is to
	// stop padding it"). One binary, one push-constant word, every 64 B-
	// aligned stride whose gcd with the 4 KB rotation is a given power of two.
	{name: "moe_gemv_v4_gcd64", spirv: shaders.GEMVW4A8GroupedVec4, weightsPerLoad: 32, block: 128, gcdTarget: 64, strideArm: true},
	{name: "moe_gemv_v4_gcd128", spirv: shaders.GEMVW4A8GroupedVec4, weightsPerLoad: 32, block: 128, gcdTarget: 128, strideArm: true},
	{name: "moe_gemv_v4_gcd256", spirv: shaders.GEMVW4A8GroupedVec4, weightsPerLoad: 32, block: 128, gcdTarget: 256, strideArm: true},
	{name: "moe_gemv_v4_gcd512", spirv: shaders.GEMVW4A8GroupedVec4, weightsPerLoad: 32, block: 128, gcdTarget: 512, strideArm: true},
	{name: "moe_gemv_v4_gcd1024", spirv: shaders.GEMVW4A8GroupedVec4, weightsPerLoad: 32, block: 128, gcdTarget: 1024, strideArm: true},
}

func lookupMoEGEMVVariant(name string) (moeGEMVVariant, bool) {
	for _, v := range moeGEMVVariants {
		if v.name == name {
			return v, true
		}
	}
	return moeGEMVVariant{}, false
}

// moeGEMVPushConstantSize matches gemv_w4a8.comp's GROUPED block: the four
// words the plain kernel takes plus the two strides and the table base.
const moeGEMVPushConstantSize = 28

func moeGEMVPushConstants(N, K, block, ldw, lda, groupBase int, xScale float32) []byte {
	pc := newPC().U32(uint32(N)).U32(uint32(K)).U32(log2u(block)).F32(xScale).
		U32(uint32(ldw)).U32(uint32(lda)).U32(uint32(groupBase)).Bytes()
	if len(pc) != moeGEMVPushConstantSize {
		panic(fmt.Sprintf("moe gemv push constants: built %d bytes, layout declares %d",
			len(pc), moeGEMVPushConstantSize))
	}
	return pc
}

// moeGroups is the table the grouped GEMV reads: one entry per routed
// (token, expert) pair, plus the slicing the one-dispatch-at-a-time baseline
// walks it with.
//
// Experts are outermost, as in buildTiles, so that consecutive entries — and
// therefore the workgroups that run next to each other — share an expert's
// weights. A pair is a whole matrix's worth of rows, so unlike the GEMM's tile
// table there is nothing inside an entry to order.
type moeGroups struct {
	table  []uint32 // 4 words per pair: (bank row of the expert, token row, output row)
	base   []uint32 // first pair index of expert e
	counts []uint32 // pairs routed to expert e
	pairs  int
}

func buildMoEGroups(r moeRouting, N int) moeGroups {
	g := moeGroups{base: make([]uint32, moeExperts), counts: make([]uint32, moeExperts)}
	for e := 0; e < moeExperts; e++ {
		g.base[e] = uint32(g.pairs)
		for _, tok := range r.perExpert[e] {
			g.table = append(g.table, uint32(e*N), uint32(tok), uint32(g.pairs), 0)
			g.pairs++
		}
		g.counts[e] = uint32(len(r.perExpert[e]))
	}
	return g
}

// runMoEShapeGEMV runs the grouped GEMV over one of the two expert matrices.
//
// The buffers are allocated once at the largest extent any (variant, batch)
// needs and addressed differently by each, the same way runMoEShape does it —
// the strides are push constants, so a 1 GB bank is swept rather than rebuilt.
func runMoEShapeGEMV(dev *vk.Device, s moeShape, variants []moeGEMVVariant, routings map[int]moeRouting, tokens []int, warmup, iters uint32) ([]Result, error) {
	maxLdw, minBlock, maxPairs, maxTokens := 0, 1<<30, 0, 0
	usable := variants[:0:0]
	for _, v := range variants {
		ldw, ok := v.ldw(s.K)
		if !ok {
			fmt.Fprintf(os.Stderr, "moe gemv %s %s: no %d-element pad reaches gcd %d at K=%d, skipping\n",
				s.layer, v.name, moeGEMVPadStep, v.gcdTarget, s.K)
			continue
		}
		if !w4a8BlockOK(v.block, v.weightsPerLoad) || s.K%v.weightsPerLoad != 0 {
			fmt.Fprintf(os.Stderr, "moe gemv %s %s: block=%d does not admit %d weights per load at K=%d, skipping\n",
				s.layer, v.name, v.block, v.weightsPerLoad, s.K)
			continue
		}
		usable = append(usable, v)
		if ldw > maxLdw {
			maxLdw = ldw
		}
		if v.block < minBlock {
			minBlock = v.block
		}
	}
	if len(usable) == 0 {
		return nil, nil
	}
	for _, t := range tokens {
		if p := routings[t].rows(); p > maxPairs {
			maxPairs = p
		}
		if t > maxTokens {
			maxTokens = t
		}
	}

	bankBytes := moeExperts * s.N * maxLdw / 2
	wBuf, err := dev.NewBuffer(bankBytes)
	if err != nil {
		return nil, fmt.Errorf("moe gemv %s expert bank (%.2f GB): %w", s.layer, float64(bankBytes)/1e9, err)
	}
	defer wBuf.Destroy()
	scalesBuf, err := dev.NewBuffer(moeExperts * s.N * (maxLdw / minBlock) * 2)
	if err != nil {
		return nil, fmt.Errorf("moe gemv %s scales: %w", s.layer, err)
	}
	defer scalesBuf.Destroy()
	xBuf, err := dev.NewBuffer(maxTokens * s.K)
	if err != nil {
		return nil, err
	}
	defer xBuf.Destroy()
	yBuf, err := dev.NewBuffer(maxPairs * s.N * 4)
	if err != nil {
		return nil, err
	}
	defer yBuf.Destroy()
	sumsBuf, err := dev.NewBuffer(maxTokens * (s.K / minBlock) * 4)
	if err != nil {
		return nil, err
	}
	defer sumsBuf.Destroy()
	groupBuf, err := dev.NewBuffer(maxPairs * 16)
	if err != nil {
		return nil, err
	}
	defer groupBuf.Destroy()

	// Only the addresses matter to a timed run, but every byte still has to
	// decode to something finite: nibbles can be any bit pattern at all, the
	// scales are real fp16 numbers, and the block sums are int32 of the
	// magnitude a block of int8 activations would actually produce.
	wBuf.FillRepeating(randomBytes(moePatternBytes))
	scalesBuf.FillRepeating(float32SliceToFloat16Bytes(randomFloats(moePatternBytes / 2)))
	xBuf.FillRepeating(randomBytes(moePatternBytes))
	sumsBuf.FillRepeating(int32SliceToBytes(randomSmallInts(moePatternBytes/4, 1<<13)))

	var results []Result
	for _, v := range usable {
		ldw, _ := v.ldw(s.K)
		mod, err := dev.NewShaderModule(v.spirv)
		if err != nil {
			return nil, err
		}
		defer mod.Destroy()
		if err := verifyMoEGroupedGEMV(dev, mod, v); err != nil {
			return nil, fmt.Errorf("moe gemv %s correctness check: %w", v.name, err)
		}

		pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers:              []*vk.Buffer{wBuf, scalesBuf, xBuf, yBuf, sumsBuf, groupBuf},
			PushConstantSize:     moeGEMVPushConstantSize,
			RequiredSubgroupSize: v.waveSize,
		})
		if err != nil {
			return nil, fmt.Errorf("moe gemv %s pipeline: %w", v.name, err)
		}
		defer pipe.Destroy()

		for _, t := range tokens {
			if v.strideArm && t != moeGEMVStrideTokens {
				continue
			}
			r := routings[t]
			g := buildMoEGroups(r, s.N)
			groupBuf.WriteBytes(uint32SliceToBytes(g.table))
			base := moeGEMVResult(s, v, r, g, ldw)

			// The grid is (output row, pair): x first, so the workgroups that
			// issue together sweep one expert's rows in address order.
			pc := moeGEMVPushConstants(s.N, s.K, v.block, ldw, s.K, 0, 1.0/127.0)
			ns, clocks, err := TimeDispatch(pipe, uint32(s.N), uint32(g.pairs), 1, warmup, iters, pc)
			if err != nil {
				return nil, fmt.Errorf("moe gemv %s %s grouped t=%d: %w", s.layer, v.name, t, err)
			}
			results = append(results, base.finish(mode1, 1, ns, clocks))

			if v.strideArm {
				continue
			}

			// The baseline is one dispatch per *pair* rather than per expert,
			// which at decode is nearly the same thing — a batch of 64 routes
			// 640 pairs over ~400 experts — and is the schedule an engine
			// without a table would write. (It is also the only one this
			// vk.DispatchSequenceTimed can express: the grid's y extent is
			// fixed across a sequence, and a per-expert dispatch would need it
			// to be that expert's pair count.)
			groupsX := make([]uint32, g.pairs)
			pcs := make([][]byte, g.pairs)
			for i := range groupsX {
				groupsX[i] = uint32(s.N)
				pcs[i] = moeGEMVPushConstants(s.N, s.K, v.block, ldw, s.K, i, 1.0/127.0)
			}
			ns, clocks, err = TimeDispatchSequence(pipe, groupsX, 1, 1, warmup, iters, pcs)
			if err != nil {
				return nil, fmt.Errorf("moe gemv %s %s per-pair t=%d: %w", s.layer, v.name, t, err)
			}
			results = append(results, base.finish("per_pair", g.pairs, ns, clocks))
		}
	}
	return results, nil
}

// randomSmallInts fills n int32s with values in [-mag, mag], for buffers whose
// contents have to be plausible numbers but not particular ones.
func randomSmallInts(n, mag int) []int32 {
	r := rand.New(rand.NewSource(44))
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(r.Intn(2*mag+1) - mag)
	}
	return out
}

// moeGEMVCase is everything about a (shape, variant, batch) that does not
// depend on the schedule, so the grouped and per-pair rows differ in one field
// plus the timing.
type moeGEMVCase struct {
	shape   moeShape
	variant moeGEMVVariant
	ldw     int
	tokens  int
	pairs   int
	experts int
	useful  float64 // FLOPs the model needs
	issued  float64 // weight bytes the kernel asks for (a pair re-reads its expert)
	touched float64 // weight bytes distinct across the dispatch — the residency figure
	acts    float64 // int8 activations read plus fp32 outputs written
}

func moeGEMVResult(s moeShape, v moeGEMVVariant, r moeRouting, g moeGroups, ldw int) moeGEMVCase {
	// Bytes of one expert matrix as the kernel reads it: K nibbles per row
	// plus one fp16 scale per block. The row pad is never read — every stride
	// in the sweep is 64 B-aligned, so it costs footprint and not traffic.
	perExpert := float64(s.N) * (float64(s.K)/2 + float64(s.K/v.block)*2)
	return moeGEMVCase{
		shape: s, variant: v, ldw: ldw, tokens: r.tokens, pairs: g.pairs, experts: r.touched,
		useful:  2 * float64(g.pairs) * float64(s.N) * float64(s.K),
		issued:  float64(g.pairs) * perExpert,
		touched: float64(r.touched) * perExpert,
		acts:    float64(g.pairs)*float64(s.K) + float64(g.pairs)*float64(s.N)*4,
	}
}

func (c moeGEMVCase) finish(mode string, dispatches int, ns float64, clocks ClockStats) Result {
	strideB := c.ldw / 2
	return Result{
		Op: "moe", Variant: c.variant.name, WeightFormat: "w4a8", BlockSize: c.variant.block,
		Size: c.tokens,
		Detail: fmt.Sprintf("layer=%s;mode=%s;tokens=%d;pairs=%d;experts=%d;rowsperexpert=%.2f;"+
			"vec=%d;wave=%d;cover=%dB;rowB=%d;strideB=%dB;gcd4K=%d;touchedMB=%.0f;resident=%s;"+
			"dispatches=%d;wgperdispatch=%d;weightGBps=%.0f;actGBps=%.0f;perpairus=%.2f",
			c.shape.layer, mode, c.tokens, c.pairs, c.experts,
			float64(c.pairs)/float64(c.experts),
			c.variant.vec(), c.variant.wave(), c.variant.coverage(c.shape.K), c.shape.K/2,
			strideB, gcdInt(strideB, strideAliasChunk), c.touched/1e6, c.residency(),
			dispatches, c.shape.N*c.pairs/dispatches,
			c.issued/(ns/1e9)/1e9, c.acts/(ns/1e9)/1e9, ns/1e3/float64(c.pairs)),
		NsPerIter: ns,
		GFLOPS:    c.useful / (ns / 1e9) / 1e9,
		GBPS:      (c.issued + c.acts) / (ns / 1e9) / 1e9,
		Clocks:    clocks,
	}
}

// residency says where this case's weights are coming from on the second and
// later iterations of the timing loop. It is a recorded field rather than a
// footnote because at one token they come from cache, and that is the trap
// §3.5's decode rows fell into.
//
// The three-way split, rather than a single 32 MiB threshold, is because the
// measurement showed the threshold is not where the cliff is: a 34 MB
// footprint — nominally larger than the cache — still reads 410 GB/s, 1.7x the
// bus, so the MALL is serving most of it. Only past about twice the cache does
// the rate fall to the bus and stay there. "edge" is the band in between, and
// nothing in a summary should treat it as a DRAM measurement.
func (c moeGEMVCase) residency() string {
	switch {
	case c.touched < mallBytes:
		return "mall"
	case c.touched < 2*mallBytes:
		return "edge"
	}
	return "dram"
}

// verifyMoEGroupedGEMV checks one grouped build against a host reference at
// the smallest size that still exercises everything the grouping can get
// wrong: a padded bank stride (so a row's nibbles, its scales and its
// neighbours are all addressed off ldw rather than K), two pairs on one expert
// and one on another (so the table's bank row is not a function of the pair
// index), a token read by two different experts, and an expert whose rows are
// not the first in the bank.
//
// The reference is built from the pre-repack Q4 arrays, so it checks
// repackQ4ToW4A8's permutation as well as the kernel — the same reference
// verifyGEMVW4A8 uses, at a shape it cannot reach.
func verifyMoEGroupedGEMV(dev *vk.Device, mod *vk.ShaderModule, v moeGEMVVariant) error {
	const experts, N = 3, 5
	K := v.block * 2
	ldw := K + v.block // padded, and still a whole number of uvec4s and blocks
	const tokens = 4
	pairs := [][2]int{{0, 0}, {0, 2}, {1, 1}, {2, 3}, {2, 0}} // (expert, token)

	bData := randomFloats(experts * N * K)
	packed, scales := quantizeQ4(bData, experts*N, K, v.block)
	words := repackQ4ToW4A8(packed, experts*N, K)

	xData := randomFloats(tokens * K)
	// One scale for every activation in the batch, which is what the kernel's
	// single xScale push constant means.
	packedX, xScales := quantizeQ8(xData, 1, tokens*K, tokens*K)
	xScale := float16ToFloat32(xScales[0])

	wBuf, err := dev.NewBuffer(moeExperts * N * ldw / 2)
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	scalesBuf, err := dev.NewBuffer(moeExperts * N * (ldw / v.block) * 2)
	if err != nil {
		return err
	}
	defer scalesBuf.Destroy()
	xBuf, err := dev.NewBuffer(tokens * K)
	if err != nil {
		return err
	}
	defer xBuf.Destroy()
	yBuf, err := dev.NewBuffer(len(pairs) * N * 4)
	if err != nil {
		return err
	}
	defer yBuf.Destroy()
	sumsBuf, err := dev.NewBuffer(tokens * (K / v.block) * 4)
	if err != nil {
		return err
	}
	defer sumsBuf.Destroy()
	groupBuf, err := dev.NewBuffer(len(pairs) * 16)
	if err != nil {
		return err
	}
	defer groupBuf.Destroy()

	wBuf.WriteBytes(padW4A8Rows(words, experts*N, K, ldw))
	scalesBuf.WriteBytes(padScaleRows(scales, experts*N, K/v.block, ldw/v.block))
	xBuf.WriteBytes(int8SliceToBytes(packedX))
	sumsBuf.WriteBytes(int32SliceToBytes(int8BlockSums(packedX, v.block)))
	var table []uint32
	for i, p := range pairs {
		table = append(table, uint32(p[0]*N), uint32(p[1]), uint32(i), 0)
	}
	groupBuf.WriteBytes(uint32SliceToBytes(table))

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:              []*vk.Buffer{wBuf, scalesBuf, xBuf, yBuf, sumsBuf, groupBuf},
		PushConstantSize:     moeGEMVPushConstantSize,
		RequiredSubgroupSize: v.waveSize,
	})
	if err != nil {
		return err
	}
	defer pipe.Destroy()

	pc := moeGEMVPushConstants(N, K, v.block, ldw, K, 0, xScale)
	if _, err := pipe.DispatchTimed(uint32(N), uint32(len(pairs)), 1, 1, pc); err != nil {
		return err
	}
	got := yBuf.ReadFloat32(len(pairs) * N)

	refW := dequantizeQ4(packed, scales, experts*N, K, v.block)
	refX := dequantizeQ8(packedX, xScales, 1, tokens*K, tokens*K)
	for i, p := range pairs {
		want := cpuGEMV(refW[p[0]*N*K:(p[0]+1)*N*K], refX[p[1]*K:(p[1]+1)*K], N, K)
		if err := compareVec(got[i*N:(i+1)*N], want, 2e-2); err != nil {
			return fmt.Errorf("pair %d (expert %d token %d): %w", i, p[0], p[1], err)
		}
	}
	return nil
}

// padW4A8Rows re-lays repackQ4ToW4A8's dense words at a wider row stride, in
// elements (nibbles). ldElems must be a multiple of 8.
func padW4A8Rows(words []uint32, rows, cols, ldElems int) []byte {
	wordsPerRow, strideWords := cols/8, ldElems/8
	out := make([]uint32, rows*strideWords)
	for r := 0; r < rows; r++ {
		copy(out[r*strideWords:r*strideWords+wordsPerRow], words[r*wordsPerRow:(r+1)*wordsPerRow])
	}
	return uint32SliceToBytes(out)
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

// PrintMoEGEMVSummary reads the decode rows the four ways they have to be
// read: what the load width is worth against §1.7's rule, what the grouped
// schedule is worth, what the bank's row stride is worth, and what the honest
// kernel is worth against the GEMM the suite has been measuring decode with.
func PrintMoEGEMVSummary(w io.Writer, results []Result) {
	printMoEGEMVLoadWidth(w, results)
	printMoEGEMVSchedule(w, results)
	printMoEGEMVStride(w, results)
	printMoEGEMVvsGEMM(w, results)
	printMoEGEMVBudget(w, results)
}

// moeGEMVRows selects this arm's rows: the grouped GEMV, one mode, with the
// stride arm optionally excluded.
func moeGEMVRows(results []Result, mode string, stride bool) []Result {
	var out []Result
	for _, r := range results {
		if r.WeightFormat != "w4a8" || detailField(r.Detail, "mode") != mode {
			continue
		}
		v, ok := lookupMoEGEMVVariant(r.Variant)
		if !ok || v.strideArm != stride {
			continue
		}
		out = append(out, r)
	}
	return out
}

// printMoEGEMVLoadWidth is §1.7's rule at the expert shapes: the bus is
// reached when one lane-step covers a whole weight row, and these two rows are
// 320 B and 1280 B, so the rule names VEC=4 and VEC=8 at wave64.
func printMoEGEMVLoadWidth(w io.Writer, results []Result) {
	type key struct{ variant, layer, tokens string }
	cells := map[key]float64{}
	var variants, layers, tokens []string
	for _, r := range moeGEMVRows(results, mode1, false) {
		layer, tok := detailField(r.Detail, "layer"), detailField(r.Detail, "tokens")
		gbps, _ := strconv.ParseFloat(detailField(r.Detail, "weightGBps"), 64)
		cells[key{r.Variant, layer, tok}] = gbps
		if !containsString(variants, r.Variant) {
			variants = append(variants, r.Variant)
		}
		if !containsString(layers, layer) {
			layers = append(layers, layer)
		}
		if !containsString(tokens, tok) {
			tokens = append(tokens, tok)
		}
	}
	if len(variants) == 0 {
		return
	}
	sort.Slice(tokens, func(i, j int) bool { return atoiOr(tokens[i]) < atoiOr(tokens[j]) })

	fmt.Fprintf(w, "\ngrouped W4A8 decode GEMV: weight bytes off the bus (GB/s of %.0f), by load width\n", dramPeakGBPS)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprint(tw, "KERNEL\tVEC\tWAVE\tBLOCK")
	for _, l := range layers {
		for _, t := range tokens {
			fmt.Fprintf(tw, "\t%s t=%s", shortLayer(l), t)
		}
	}
	fmt.Fprintln(tw, "\tCOVER/ROW")
	for _, name := range variants {
		v, _ := lookupMoEGEMVVariant(name)
		fmt.Fprintf(tw, "%s\t%d\t%d\t%d", name, v.vec(), v.wave(), v.block)
		for _, l := range layers {
			for _, t := range tokens {
				fmt.Fprintf(tw, "\t%.0f", cells[key{name, l, t}])
			}
		}
		var cov []string
		for _, l := range layers {
			K := moeLayerK(l)
			cov = append(cov, fmt.Sprintf("%s %d/%d B", shortLayer(l), v.coverage(K), K/2))
		}
		fmt.Fprintf(tw, "\t%s\n", strings.Join(cov, "  "))
	}
	tw.Flush()
	fmt.Fprintf(w, "  these are bytes *asked for*; where several pairs share an expert the MALL serves the\n"+
		"  repeats, so only the t=16 and t=64 columns — which touch 3.5x and 9.4x the %d MB cache — are\n"+
		"  DRAM rates. COVER/ROW is §1.7's C against the 4-bit row: the rule says the bus is reached when\n"+
		"  they are equal\n", mallBytes>>20)
}

// printMoEGEMVSchedule is §3.5's grouped-vs-per-dispatch question asked of a
// kernel whose single expert dispatch already fills the machine — a GEMV
// launches one workgroup per output row, so 640 or 2560 of them per pair,
// where a 64-row GEMM tile launches 10.
func printMoEGEMVSchedule(w io.Writer, results []Result) {
	type key struct{ variant, layer, tokens string }
	grouped, perPair := map[key]float64{}, map[key]float64{}
	var keys []key
	for _, r := range moeGEMVRows(results, mode1, false) {
		k := key{r.Variant, detailField(r.Detail, "layer"), detailField(r.Detail, "tokens")}
		grouped[k] = r.NsPerIter
		keys = append(keys, k)
	}
	for _, r := range moeGEMVRows(results, "per_pair", false) {
		perPair[key{r.Variant, detailField(r.Detail, "layer"), detailField(r.Detail, "tokens")}] = r.NsPerIter
	}
	if len(keys) == 0 {
		return
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].layer != keys[j].layer {
			return keys[i].layer < keys[j].layer
		}
		if keys[i].tokens != keys[j].tokens {
			return atoiOr(keys[i].tokens) < atoiOr(keys[j].tokens)
		}
		return keys[i].variant < keys[j].variant
	})

	fmt.Fprintln(w, "\none grouped dispatch against one dispatch per routed pair (same binary, same table)")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "LAYER\tTOKENS\tKERNEL\tPAIRS\tGROUPED MS\tPER-PAIR MS\tSPEEDUP")
	for _, k := range keys {
		p, ok := perPair[k]
		if !ok {
			continue
		}
		pairs := atoiOr(k.tokens) * moeTopK
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%.3f\t%.3f\t%.2fx\n",
			shortLayer(k.layer), k.tokens, k.variant, pairs, grouped[k]/1e6, p/1e6, p/grouped[k])
	}
	tw.Flush()
	fmt.Fprintln(w, "  §3.5 found grouping worth 1.1-4.1x on the GEMM and tied it to how many workgroups one")
	fmt.Fprintln(w, "  expert's dispatch launches; a GEMV's is one per output row — 640 or 2560 per pair, past")
	fmt.Fprintln(w, "  the 80 waves §0.1 says this part needs resident — so on that mechanism this should be")
	fmt.Fprintln(w, "  worth §4.1's 300 ns launch and nothing more. It measured 690-1555 ns per dispatch")
	fmt.Fprintln(w, "  removed, median 1163: what a barrier costs a memory-bound dispatch is the drain, and")
	fmt.Fprintln(w, "  §4.1's empty shader has nothing to drain (IDEAS §1.8 finding 3)")
}

// printMoEGEMVStride is the handoff's "K=640 is explained, and the fix is to
// stop padding it", carried from the GEMM to the GEMV. Every stride here is
// 64 B-aligned, so the pad is never read: this is coverage at constant traffic.
func printMoEGEMVStride(w io.Writer, results []Result) {
	type cell struct {
		gcd, strideB int
		gbps, gflops float64
		touchedMB    float64
	}
	rows := map[string][]cell{}
	var layers []string
	for _, r := range moeGEMVRows(results, mode1, true) {
		layer := detailField(r.Detail, "layer")
		if !containsString(layers, layer) {
			layers = append(layers, layer)
		}
		gbps, _ := strconv.ParseFloat(detailField(r.Detail, "weightGBps"), 64)
		touched, _ := strconv.ParseFloat(detailField(r.Detail, "touchedMB"), 64)
		rows[layer] = append(rows[layer], cell{
			gcd:     atoiOr(detailField(r.Detail, "gcd4K")),
			strideB: atoiOr(trimSuffixB(detailField(r.Detail, "strideB"))),
			gbps:    gbps, gflops: r.GFLOPS, touchedMB: touched,
		})
	}
	if len(layers) == 0 {
		return
	}
	fmt.Fprintf(w, "\nbank row stride against the %d B channel rotation, at %d tokens (one binary, one pushed word)\n",
		strideAliasChunk, moeGEMVStrideTokens)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "LAYER\tGCD\tSTRIDE B\tPAD B\tFOOTPRINT\tWEIGHT GB/S\tUSEFUL GFLOP/S")
	for _, l := range layers {
		cells := rows[l]
		sort.Slice(cells, func(i, j int) bool { return cells[i].gcd < cells[j].gcd })
		rowB := moeLayerK(l) / 2
		for _, c := range cells {
			fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%.0f MB\t%.0f\t%.0f\n",
				shortLayer(l), c.gcd, c.strideB, c.strideB-rowB, c.touchedMB, c.gbps, c.gflops)
		}
	}
	tw.Flush()
	fmt.Fprintln(w, "  the pad is footprint only — every stride is a multiple of 64 B, so a row still starts on a")
	fmt.Fprintln(w, "  cache line and the padding bytes are never fetched. §5.1b's rule says the plateau is at")
	fmt.Fprintln(w, "  gcd 128-256 B; down's natural 320 B row sits at 64, below it, and gate_up's 1280 B at 256")
}

// printMoEGEMVvsGEMM is the point of the file: the honest decode kernel
// against the grouped GEMM at M=1, which is what every decode row in this
// family measured before it existed.
func printMoEGEMVvsGEMM(w io.Writer, results []Result) {
	type key struct{ layer, tokens string }
	type best struct {
		variant   string
		ns, gbps  float64
		actGBps   float64
		touchedMB float64
		resident  string
	}
	gemv, gemm := map[key]best{}, map[key]best{}
	seen := map[key]bool{}
	note := func(m map[key]best, k key, b best) {
		if cur, ok := m[k]; !ok || b.ns < cur.ns {
			m[k] = b
		}
		seen[k] = true
	}
	for _, r := range results {
		if detailField(r.Detail, "mode") != mode1 {
			continue
		}
		k := key{detailField(r.Detail, "layer"), detailField(r.Detail, "tokens")}
		if !containsString(moeDecodeTokenStrings(), k.tokens) {
			continue
		}
		gbps, _ := strconv.ParseFloat(detailField(r.Detail, "weightGBps"), 64)
		act, _ := strconv.ParseFloat(detailField(r.Detail, "actGBps"), 64)
		switch r.WeightFormat {
		case "w4a8":
			v, ok := lookupMoEGEMVVariant(r.Variant)
			if !ok || v.strideArm {
				continue
			}
			touched, _ := strconv.ParseFloat(detailField(r.Detail, "touchedMB"), 64)
			note(gemv, k, best{r.Variant, r.NsPerIter, gbps, act, touched, detailField(r.Detail, "resident")})
		case "q4":
			if v, ok := lookupMoEQ4Variant(r.Variant); !ok || v.strideArm {
				continue
			}
			note(gemm, k, best{r.Variant, r.NsPerIter, gbps, act, 0, ""})
		}
	}
	if len(seen) == 0 {
		return
	}
	keys := make([]key, 0, len(seen))
	for k := range seen {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].layer != keys[j].layer {
			return keys[i].layer < keys[j].layer
		}
		return atoiOr(keys[i].tokens) < atoiOr(keys[j].tokens)
	})

	fmt.Fprintln(w, "\nthe honest decode kernel against the grouped GEMM at M=1 (both Q4, both grouped, same routing)")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "LAYER\tTOKENS\tFOOTPRINT\tGEMV MS\tGEMV DISTINCT GB/S\tGEMV ISSUED GB/S\tGEMV ACT GB/S\tGEMM MS\tGEMM DISTINCT GB/S\tGEMM ACT GB/S\tGEMV/GEMM\tGEMV KERNEL")
	for _, k := range keys {
		a, ok1 := gemv[k]
		b, ok2 := gemm[k]
		if !ok1 || !ok2 {
			continue
		}
		// Both kernels read each touched expert once from DRAM; the GEMV's
		// weightGBps counts what it *asks* for, which at more than one pair
		// per expert includes re-reads the MALL serves. touchedMB/ns is the
		// distinct rate, which is what the GEMM's column already is.
		fmt.Fprintf(tw, "%s\t%s\t%.0f MB %s\t%.3f\t%.0f\t%.0f\t%.0f\t%.3f\t%.0f\t%.0f\t%.2fx\t%s\n",
			shortLayer(k.layer), k.tokens, a.touchedMB, a.resident,
			a.ns/1e6, a.touchedMB*1e6/a.ns, a.gbps, a.actGBps,
			b.ns/1e6, b.gbps, b.actGBps, b.ns/a.ns, a.variant)
	}
	tw.Flush()
	fmt.Fprintln(w, "  both read the same weight bytes; what the GEMM adds is a 16-to-64-row A fragment per")
	fmt.Fprintln(w, "  one-row group, so its activation traffic and its FLOPs are multiplied by the tile height")
}

// printMoEGEMVBudget turns the per-matrix rows into what GOALS.md is asking
// about: what one token's MoE FFN costs, and what that is per second.
//
// Two corrections are baked into how it reads the table, and both matter by
// about 3x:
//
//   - It divides by the experts a dispatch *touched*, not by the pairs it ran,
//     because a batch of 64 routes 640 pairs over ~370 experts and the
//     duplicate pairs are MALL hits. A real decode step reads 10 experts that
//     nothing has touched for 47 layers, so the quantity that transfers is the
//     cost of one expert's weights arriving from DRAM.
//   - It ignores the cache-resident batches entirely for the headline, since
//     10 experts of one projection is 8 MB and fits in the MALL. The t=1 row
//     is printed underneath precisely to show what that flatters by.
func printMoEGEMVBudget(w io.Writer, results []Result) {
	type pick struct {
		variant, tokens string
		nsPerExpert     float64
		experts, pairs  int
	}
	best := map[string]pick{} // layer -> cheapest expert, DRAM-resident
	hot := map[string]pick{}  // layer -> the same at one token, all MALL
	for _, r := range moeGEMVRows(results, mode1, false) {
		layer := detailField(r.Detail, "layer")
		experts := atoiOr(detailField(r.Detail, "experts"))
		if experts == 0 {
			continue
		}
		p := pick{r.Variant, detailField(r.Detail, "tokens"), r.NsPerIter / float64(experts),
			experts, atoiOr(detailField(r.Detail, "pairs"))}
		target := best
		if detailField(r.Detail, "resident") != "dram" {
			if p.tokens != "1" {
				continue
			}
			target = hot
		}
		if cur, ok := target[layer]; !ok || p.nsPerExpert < cur.nsPerExpert {
			target[layer] = p
		}
	}
	up, ok1 := best["moe.gate_up"]
	down, ok2 := best["moe.down"]
	if !ok1 || !ok2 {
		return
	}
	bytes := float64(moeLayers*moeTopK) * (2*float64(moeFFN)*float64(moeHidden) + float64(moeHidden)*float64(moeFFN)) / 2
	floor := bytes / (dramPeakGBPS * 1e9) * 1e3

	fmt.Fprintln(w, "\nwhat one decode token's MoE FFN costs (48 blocks, 10 experts each, 4-bit bank)")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "MEASURED AT\tBATCH\tPAIRS/EXPERT\tGATE+UP/EXPERT\tDOWN/EXPERT\tBLOCK\tx48 LAYERS\tFFN-ONLY TOK/S\t% OF BUS\tKERNEL")
	row := func(label string, u, d pick) {
		// One MoE block per token: gate and up are two matrices of the gate_up
		// shape and down is one, each read for every expert the token routes
		// to.
		block := float64(moeTopK) * (2*u.nsPerExpert + d.nsPerExpert)
		pass := float64(moeLayers) * block
		fmt.Fprintf(tw, "%s\tt=%s/%s\t%.2f/%.2f\t%.1f us\t%.1f us\t%.3f ms\t%.2f ms\t%.0f\t%.0f%%\t%s\n",
			label, u.tokens, d.tokens,
			float64(u.pairs)/float64(u.experts), float64(d.pairs)/float64(d.experts),
			2*u.nsPerExpert/1e3, d.nsPerExpert/1e3, block/1e6, pass/1e6, 1e9/pass,
			100*floor/(pass/1e6), u.variant)
	}
	row("DRAM-resident", up, down)
	if h1, ok := hot["moe.gate_up"]; ok {
		if h2, ok := hot["moe.down"]; ok {
			row("t=1 (MALL)", h1, h2)
		}
	}
	tw.Flush()
	fmt.Fprintf(w, "  the FFN alone reads %.2f GB of 4-bit weights per token, which is %.2f ms at the %.0f GB/s\n"+
		"  bus and is the only ceiling that can matter at this shape. PAIRS/EXPERT is how many of the\n"+
		"  batch's pairs shared an expert; the per-expert cost is what a decode step pays because its own\n"+
		"  10 experts are cold\n", bytes/1e9, floor, dramPeakGBPS)
}

// ---------------------------------------------------------------------------
// Small helpers the summaries share
// ---------------------------------------------------------------------------

func atoiOr(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return v
}

func trimSuffixB(s string) string {
	if len(s) > 0 && s[len(s)-1] == 'B' {
		return s[:len(s)-1]
	}
	return s
}

func shortLayer(layer string) string {
	if len(layer) > 4 && layer[:4] == "moe." {
		return layer[4:]
	}
	return layer
}

func moeLayerK(layer string) int {
	for _, s := range moeShapes {
		if s.layer == layer {
			return s.K
		}
	}
	return 0
}

func moeDecodeTokenStrings() []string {
	out := make([]string, len(moeDecodeTokens))
	for i, t := range moeDecodeTokens {
		out[i] = strconv.Itoa(t)
	}
	return out
}
