package llm

// The MoE block on the device: LLM.md L5b, and the last block of this model
// to get a kernel.
//
// It is the largest line in L2a's attribution — **35.7% of llama.cpp's
// 512-token prefill graph**, against the hyper-connection block's 14.9%, the
// full-attention layer's 5.7% and the gated DeltaNet's 13.1% — and it is the
// one block whose weights cannot be staged the way the other four are. One
// layer's three routed banks are 2.52 billion weights: 5.03 GB dequantised to
// halves, 241 GB across the model, against **1.57 GB and 75 GB as they ship**.
// So `llm_moe_gemm.comp` reads the checkpoint's own Q4_K, Q5_K, Q5_1 and Q8_0
// blocks and unpacks its own slab into LDS per K-step, and the bank here is a
// byte-for-byte copy of the GGUF's.
//
// Nine dispatches:
//
//	router    [nExpert+1, nEmbd] * xn -> logits         llm_gemm.comp MODE 2
//	route     softmax, top-10, normalise, shared gate   one workgroup a token
//	perm x2   counting sort + the two tile schedules    one workgroup
//	up        gate and up, silu(gate)*up fused          grouped, Q4_K/Q5_K
//	down      down, weighted, scattered                 grouped, Q5_1/Q8_0
//	shexp x2  the same two kernels, one group, Q8_0
//	combine   the eleven contributions summed
//
// Four tensors the reference materialises never exist here. `ffn_moe_gate`
// and `ffn_moe_up` are consumed on the accumulators; `ffn_moe_weighted` is
// the down projection's store; and `shared_expert_gate` is a 513th column of
// the router's matrix rather than its own [2560, 1] matmul — L2a's argument
// for `inject` at a fifth site, and the one place this block computes
// something the reference computes differently, since one output column keeps
// that gate on the f32 vector path at any prompt length (L3a-5).
//
// **What the permutation is for.** L5a-4 measured the routing and it is not
// balanced: at a 512-token ubatch **274 of 512 experts are touched, one takes
// 484 of the 512 tokens and 273 of the others have fewer than 19 rows**. So
// the schedule cannot be a grid — it is a list of (expert, row block) records
// the device builds, over which the GEMM dispatches a static upper bound with
// an early return. And the permutation's row index is
// `token * (used + 1) + slot`, which does three things at once: the up
// kernel's gathered rows become the down kernel's *contiguous* ones, the
// routing weight is indexed by the same number as the row, and a token's
// eleven contributions — ten routed and the shared expert's, in the last
// slot — are contiguous for the combine.

import (
	"fmt"
	"time"
	"unsafe"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// moeBN is the compiled column block of every rung of the grouped GEMM. It is
// not a ladder axis: the kernel stages one B row per lane and the wave is 64
// wide, so BN is the wave. Both output widths divide it — 640 is ten blocks
// and 2560 is forty.
const moeBN = 64

// moeWave is the pinned subgroup width, as it is for every cooperative-matrix
// kernel in this vertical.
const moeWave = 64

// MoEKernel names one rung of the grouped GEMM's ladder. A rung is two
// numbers: how many accumulator tiles a wave holds in the M direction, and how
// many **waves** split a tile's rows between them.
//
// The second is the one that matters here, and L5b measured why. A workgroup
// unpacks a whole BN x BK slab of Q4_K into LDS per K-step and then multiplies
// it by BM rows of A — so a narrow tile pays the same dequant for fewer rows,
// and a wide one pays for it in registers and in occupancy. Splitting the rows
// across waves of one workgroup buys both at once: the slab is unpacked once
// for all of them, and the workgroup's LDS is amortised over four waves rather
// than one on a kernel that was running at about **one wave a SIMD**.
//
// A plan names one rung for each mode, because the up projection gathers its
// A rows through LDS and the down projection reads them contiguously — the
// asymmetry L2f-4 and L3b-7 both found inverting a schedule on two matrices
// of the same kernel. What couples them is only the row *alignment*: the
// permutation pads every expert to the wider of the two, so a mixed plan pays
// the wider rung's padding on both sides.
type MoEKernel string

const (
	MoEM1   MoEKernel = "m1"   // BM 16: one wave, one tile of rows
	MoEM2   MoEKernel = "m2"   // BM 32: one wave, two tiles
	MoEM4   MoEKernel = "m4"   // BM 64: one wave, four tiles
	MoEW2M1 MoEKernel = "w2m1" // BM 32: two waves of one tile, sharing the slab
	MoEW4M1 MoEKernel = "w4m1" // BM 64: four waves of one tile
	// The decode rungs, and the third number a rung can move: **BN**, the
	// output columns a workgroup covers (L7d). Every rung above leaves it at
	// 64, which is right when there are enough row blocks to fill the
	// machine and wrong at one token, where the routed pair is 10 experts x
	// (640/64) = 100 workgroups and the shared expert is ten. Narrowing BN
	// multiplies the grid and divides the slab a workgroup unpacks, for the
	// same total unpack — which is what a kernel short of workgroups rather
	// than of rows needs. Up mode only: the down mode's N is 2560.
	MoEN2M1 MoEKernel = "n2m1" // BM 16, BN 32
	MoEN1M1 MoEKernel = "n1m1" // BM 16, BN 16

	// And the rungs that are **not the same kernel**: llm_moe_gemv.comp,
	// L8d. Narrowing BN took the up mode from 71 to 93 GB/s and no further,
	// because what is left is not the grid — it is that a workgroup unpacks
	// its slab into LDS per K-step, behind a barrier, to multiply it by one
	// real row. The GEMV drops the slab, the barrier and the cooperative
	// matrix: LPR lanes share one output column and walk its row of the bank
	// in stride, straight into registers, with every load of the row in
	// flight at once.
	//
	// A rung is `v<LPR>` with an optional `w4`, which is four waves a
	// workgroup sharing the one A vector in LDS. Both modes have all six and
	// they may be mixed with each other and with the GEMM rungs, because
	// every one of them cuts its tile list to the same sixteen rows.
	//
	// **They read one row of a tile**, which is every real row a tile has
	// when the batch is one token — ten distinct experts, one row each — and
	// is not when it is two. The host refuses them above one token.
	MoEV16   MoEKernel = "v16"   // 16 lanes a column, 4 columns a workgroup
	MoEV32   MoEKernel = "v32"   // 32 lanes, 2 columns
	MoEV64   MoEKernel = "v64"   // a whole wave a column
	MoEV16W4 MoEKernel = "v16w4" // four waves: 16 columns a workgroup
	MoEV32W4 MoEKernel = "v32w4" // 8 columns
	MoEV64W4 MoEKernel = "v64w4" // 4 columns
)

// MoEKernels lists the GEMM rungs both modes have, narrowest first.
func MoEKernels() []MoEKernel {
	return []MoEKernel{MoEM1, MoEM2, MoEM4, MoEW2M1, MoEW4M1}
}

// MoEUpKernels lists every GEMM rung the up mode can run, which is the list
// above plus the narrow-N pair.
func MoEUpKernels() []MoEKernel {
	return append(MoEKernels(), MoEN2M1, MoEN1M1)
}

// MoEDecodeKernels lists the GEMV rungs, which both modes share and which
// only a one-token batch may stand on.
func MoEDecodeKernels() []MoEKernel {
	return []MoEKernel{MoEV16, MoEV32, MoEV64, MoEV16W4, MoEV32W4, MoEV64W4}
}

// MoERouterKernel names how `ffn_gate_inp` runs. `MoERouterGEMM` is
// llm_gemm.comp MODE 2 at the `GEMMKernelFor` rung, which is right at prefill
// and is nine workgroups at one token; the rest are llm_moe_router.comp's
// split-K, by how many ways it cuts K (L8d-3).
type MoERouterKernel string

const (
	MoERouterGEMM MoERouterKernel = "gemm"
	MoERouterK8   MoERouterKernel = "k8"
	MoERouterK10  MoERouterKernel = "k10"
	MoERouterK20  MoERouterKernel = "k20"
	MoERouterK40  MoERouterKernel = "k40"
)

// MoERouterKernels lists the split-K rungs. Every one of them divides 40,
// which is what makes 160 k-tiles a whole number of four-tile steps.
func MoERouterKernels() []MoERouterKernel {
	return []MoERouterKernel{MoERouterK8, MoERouterK10, MoERouterK20, MoERouterK40}
}

// moeRouterSlabs is a rung's split, and zero for the GEMM.
func moeRouterSlabs(k MoERouterKernel) int {
	switch k {
	case MoERouterK8:
		return 8
	case MoERouterK10:
		return 10
	case MoERouterK20:
		return 20
	case MoERouterK40:
		return 40
	}
	return 0
}

// moeRouterMaxSlabs sizes the partial-sum arena: f32 [KSLABS][routerN], which
// at the widest rung is 92 KB and is allocated whatever rung runs.
const moeRouterMaxSlabs = 40

// MoEIsGemv reports whether a rung is llm_moe_gemv.comp rather than
// llm_moe_gemm.comp. It is what says a plan needs a one-token batch.
func MoEIsGemv(k MoEKernel) bool {
	switch k {
	case MoEV16, MoEV32, MoEV64, MoEV16W4, MoEV32W4, MoEV64W4:
		return true
	}
	return false
}

// moeBNOf is a rung's column block. It is what the host has to know to shape
// the grid, and it is compiled into the SPIR-V.
// A GEMV rung has no column block in the GEMM's sense; what it has is NCOL,
// the output columns one workgroup owns, which is `waves * 64 / LPR`. It
// enters the grid at exactly the same place, so the two kinds of rung share
// this function and `graph` needs no branch.
func moeBNOf(k MoEKernel) int {
	switch k {
	case MoEN1M1:
		return 16
	case MoEN2M1:
		return 32
	case MoEV16:
		return 4
	case MoEV32:
		return 2
	case MoEV64:
		return 1
	case MoEV16W4:
		return 16
	case MoEV32W4:
		return 8
	case MoEV64W4:
		return 4
	}
	return moeBN
}

func moeBM(k MoEKernel) int {
	switch k {
	case MoEM1, MoEN1M1, MoEN2M1:
		return 16
	case MoEM2, MoEW2M1:
		return 32
	case MoEM4, MoEW4M1:
		return 64
	}
	if MoEIsGemv(k) {
		// The tile list is the GEMM's narrowest, so a GEMV plan may be mixed
		// with one and the permutation's alignment does not move.
		return 16
	}
	return 0
}

// MoEWaves is how many waves a rung's workgroup holds, which is what decides
// whether the dequantised slab is shared and how much of the machine is busy.
func MoEWaves(k MoEKernel) int {
	switch k {
	case MoEW2M1:
		return 2
	case MoEW4M1, MoEV16W4, MoEV32W4, MoEV64W4:
		return 4
	}
	return 1
}

// moeBMMin and moeBMMax bound the ladder, which is what the arenas are sized
// against: the narrowest rung makes the most tiles and the widest overhangs
// an expert's rows the furthest.
const moeBMMin = 16
const moeBMMax = 64

// MoEPlanFor is the measured schedule (results/l5b_moe_ladder.csv), and it is
// a trade between two things the routing decides:
//
//	tokens   winner        runner-up          why
//	   512   m2/m2 1.00x   m2/m1 1.05x        2.33x padded rows
//	  2048   m4/m4 1.00x   m4/m2 1.00x        1.90x padded rows
//
// A wider row block amortises the slab a workgroup unpacks over more rows and
// costs padding, and how much padding depends on how skewed the routing is:
// at 512 tokens 274 experts share 5120 rows and a 64-row block executes
// **3.90x** the rows that exist, at 2048 it is 1.90x and at 4096 1.46x. So
// the ladder moves outward with the chunk, and the boundary sits where the
// two effects cross. Getting it wrong costs 1.03-1.13x on either side.
// At **one token** neither effect is the one that decides: ten experts have
// one row each, so every rung makes exactly ten tiles and a wider row block
// buys no reuse and costs only padding. What is left is the grid, and the
// narrowest is the widest grid — `n1m1/m1` is 400 up workgroups against
// `m2/m2`'s 100 and the shared expert's 40 against ten, which is **1.28x on
// the block** (L7d): 612.3 us to 479.9, with the up mode at 93 GB/s of bank
// against 71 and the shared expert's at 46 against 23.
// MoERouterFor is the router's rung. The split-K kernel is a **decode** kernel
// in the same sense the expert GEMVs are — it reads one token's row — so
// above one token the block goes back to the GEMM, which is the right kernel
// there anyway: at a 512-token ubatch `ffn_gate_inp` is 512 rows of A and the
// grid is no longer nine workgroups.
func MoERouterFor(tokens int) MoERouterKernel {
	if tokens >= 1 && tokens <= GEMVMaxRows && DecodeGEMV() {
		return MoERouterK40
	}
	return MoERouterGEMM
}

// SetRouter chooses the router's rung, for the ladder. It is separate from
// SetPlan because the router is a different matrix from the experts and its
// ladder does not cross theirs.
func (g *MoEGPU) SetRouter(k MoERouterKernel) error {
	if k != MoERouterGEMM && moeRouterSlabs(k) == 0 {
		return fmt.Errorf("llm: no MoE router kernel %q (have %v)", k, append([]MoERouterKernel{MoERouterGEMM}, MoERouterKernels()...))
	}
	if k != MoERouterGEMM && g.rows > GEMVMaxRows {
		return fmt.Errorf("llm: the MoE router rung %q carries at most %d rows (P5b); this batch is %d",
			k, GEMVMaxRows, g.rows)
	}
	g.routerGemv = k
	g.autoPlan = false
	return nil
}

// Router reports the rung in use.
func (g *MoEGPU) Router() MoERouterKernel { return g.routerGemv }

// PinGemv holds the block on llm_moe_gemm.comp and the GEMM router whatever
// the batch, and releases it to the measured schedule when off. See
// Graph.PinSchedule.
func (g *MoEGPU) PinGemv(on bool) {
	g.pinGemv = on
	if !g.autoPlan {
		return
	}
	if on {
		g.up, g.down = moeGEMMPlanFor(g.rows)
		g.routerGemv = MoERouterGEMM
		g.shUp, g.shDown = g.up, g.down
	} else {
		g.up, g.down = MoEPlanFor(g.rows)
		g.shUp, g.shDown = MoESharedPlanFor(g.rows)
		g.routerGemv = MoERouterFor(g.rows)
	}
	g.syncShared()
}

// moePlan is the schedule for a batch, honouring the pin.
func (g *MoEGPU) moePlan(nTok int) (MoEKernel, MoEKernel, MoERouterKernel) {
	if g.pinGemv {
		up, down := moeGEMMPlanFor(nTok)
		return up, down, MoERouterGEMM
	}
	up, down := MoEPlanFor(nTok)
	return up, down, MoERouterFor(nTok)
}

// moeSharedPlan is the shared expert's pair for a batch, honouring the pin —
// which puts it back on whatever the routed pair runs, because the pin's whole
// job is to make the block one kernel family again.
func (g *MoEGPU) moeSharedPlan(nTok int) (MoEKernel, MoEKernel) {
	if g.pinGemv {
		up, down, _ := g.moePlan(nTok)
		return up, down
	}
	return MoESharedPlanFor(nTok)
}

// moeCheckGemv is the one thing a GEMV plan needs that a GEMM plan does not.
//
// **P5b raised the bound from one row to GEMVMaxRows.** The rung reads ROWS
// rows from a tile's base and the permutation pads the rest with a sentinel
// whose activation row is zeros (llm_moe_perm.comp), so a tile that holds
// fewer real rows computes finite nonsense into output slots the combine does
// not read — exactly what the GEMM does with the same rows. Past the bound it
// would still silently drop rows, which is what this refuses.
func moeCheckGemv(up, down MoEKernel, rows int) error {
	if rows <= GEMVMaxRows {
		return nil
	}
	for _, k := range []MoEKernel{up, down} {
		if MoEIsGemv(k) {
			return fmt.Errorf("llm: the MoE GEMV rung %q carries at most %d rows of a tile (P5b); this batch is %d",
				k, GEMVMaxRows, rows)
		}
	}
	return nil
}

func MoEPlanFor(tokens int) (MoEKernel, MoEKernel) {
	if tokens >= 1 && tokens <= GEMVMaxRows && DecodeGEMV() {
		return MoEV64W4, MoEV16W4
	}
	return moeGEMMPlanFor(tokens)
}

// moeGEMMPlanFor is that plan with the decode rungs taken out: the cooperative
// -matrix pair a batch of this length would run if llm_moe_gemv.comp did not
// exist. `tokens == 1` is L8d's control, the narrow-BN GEMM rung L7d left the
// plan on.
//
// **It is a function because the pin needs it and asking MoEPlanFor stopped
// being the same question at P5b.** Before then the GEMV was named at one row
// only, so `moePlan`'s pinned branch could ask MoEPlanFor for any batch past
// one and be sure of a GEMM; P5b raised the bound to GEMVMaxRows and the pin
// quietly stopped pinning at two rows — Graph.PinSchedule went on returning
// nil and the schedule went on changing under it. P5c found it as a two-token
// chunk split that did not reproduce the whole prompt.
func moeGEMMPlanFor(tokens int) (MoEKernel, MoEKernel) {
	switch {
	case tokens == 1:
		return MoEN1M1, MoEM1
	case tokens <= 1024:
		return MoEM2, MoEM2
	}
	return MoEM4, MoEM4
}

// MoESharedPlanFor is the **shared expert's** pair, and at one token it is not
// the routed pair's (L8e-2, read off results/l8e_moe.csv one dispatch at a
// time rather than by block total).
//
// **Only the down mode moved, and the reason the up mode did not is D16.**
// The ladder below is measured on a two-layer bank whose ten routed experts
// are 32 MB — the MALL exactly — re-read by every iteration of the sweep, and
// it says the routed up mode wants v16w4 (63.7 us a layer at 288 GB/s, which
// is already past this machine's 242 GB/s bus and so cannot be a DRAM rate).
// In the whole model each expert is read once from DRAM, and there the same
// change is **42.1 ms worse over 64 tokens** where the micro-bench predicted
// 43.6 better. So the routed pair keeps v64w4/v16w4.
//
// The two are the same grouped GEMM over the same [2560, 640] and [640, 2560]
// shapes, and they ran on one field until now. But a GEMV rung is LPR — how
// many lanes share an output column and walk its row of the bank in stride —
// and what that wants is a row's **payload words**, which is a fact about the
// format and not about the shape. Q4_K packs a 2560-long row into 320 payload
// dwords; Q8_0 packs the same row into 640. The routed banks are Q4_K (Q5_K on
// layer 2) and Q5_1, the shared expert's three are Q8_0 everywhere — so the
// ladders invert, us a layer at one token:
//
//	up     v16w4  63.7   v32w4  78.0   v64w4  78.1     routed, Q4_K
//	sh.up  v16w4  35.8   v32w4  19.4   v64w4  12.0     shared, Q8_0
//
// One field had to pick the better *joint* rung, which was v64w4 at 90.1 us
// against v16w4's 100.7. Two fields pick 63.7 + 12.0 = **75.7**. The down mode
// is the same story an order of magnitude smaller: the routed one wants v16w4
// (34.7) and the shared one v32w4 (5.68 against v16w4's 7.13).
func MoESharedPlanFor(tokens int) (MoEKernel, MoEKernel) {
	if tokens >= 1 && tokens <= GEMVMaxRows && DecodeGEMV() {
		return MoEV64W4, MoEV32W4
	}
	return MoEPlanFor(tokens)
}

// moeFmt is which of the checkpoint's quantised formats a bank ships in, and
// it is a fact about the *layer*: `ffn_down_exps` is Q5_1 on 43 of the 48
// layers and Q8_0 on the other five, gate and up are Q4_K except on layer 2
// where they are Q5_K, and the shared expert is Q8_0 everywhere.
type moeFmt int

const (
	fmtQ4K moeFmt = iota
	fmtQ5K
	fmtQ51
	fmtQ80
	// P4c's two. No tensor ships in either; they exist because
	// `ffn_down_exps` has a row of 640, which no K-quant divides, so the only
	// way below its 6.26 bits is a block-32 format the bank is transcoded
	// into (`moebank.go`).
	fmtQ41
	fmtIQ4NL
)

func (f moeFmt) String() string {
	switch f {
	case fmtQ4K:
		return "q4k"
	case fmtQ5K:
		return "q5k"
	case fmtQ51:
		return "q51"
	case fmtQ41:
		return "q41"
	case fmtIQ4NL:
		return "iq4nl"
	}
	return "q80"
}

// moeFmtOf maps a GGUF type onto a build of the kernel, and refuses anything
// else rather than reading it as something it is not.
func moeFmtOf(t *gguf.Tensor) (moeFmt, error) {
	f, err := moeFmtOfType(t.Type)
	if err != nil {
		return 0, fmt.Errorf("llm: %s is ggml type %d, which the MoE kernel has no build for", t.Name, t.Type)
	}
	return f, nil
}

// moeFmtOfType is the same map without a tensor to name in the error, for
// P4's transcode target.
func moeFmtOfType(t gguf.Type) (moeFmt, error) {
	switch t {
	case gguf.Q4_K:
		return fmtQ4K, nil
	case gguf.Q5_K:
		return fmtQ5K, nil
	case gguf.Q5_1:
		return fmtQ51, nil
	case gguf.Q8_0:
		return fmtQ80, nil
	case gguf.Q4_1:
		return fmtQ41, nil
	case gguf.IQ4_NL:
		return fmtIQ4NL, nil
	}
	return 0, fmt.Errorf("llm: ggml type %d has no MoE kernel build", t)
}

// moeSPIRV is the cross of the two modes with the formats each actually
// meets. The up mode never sees a Q5_1 — no gate or up projection in this
// checkpoint is one — and the down mode never sees a Q4_K or Q5_K, so
// building the full cross would be eight dead pipelines.
var moeSPIRV = map[string][]byte{
	"up_q4k_m1":     shaders.LLMMoEUpQ4KM1,
	"up_q4k_m2":     shaders.LLMMoEUpQ4KM2,
	"up_q4k_m4":     shaders.LLMMoEUpQ4KM4,
	"up_q4k_w2m1":   shaders.LLMMoEUpQ4KW2M1,
	"up_q4k_w4m1":   shaders.LLMMoEUpQ4KW4M1,
	"up_q5k_m1":     shaders.LLMMoEUpQ5KM1,
	"up_q5k_m2":     shaders.LLMMoEUpQ5KM2,
	"up_q5k_m4":     shaders.LLMMoEUpQ5KM4,
	"up_q5k_w2m1":   shaders.LLMMoEUpQ5KW2M1,
	"up_q5k_w4m1":   shaders.LLMMoEUpQ5KW4M1,
	"up_q80_m1":     shaders.LLMMoEUpQ80M1,
	"up_q80_m2":     shaders.LLMMoEUpQ80M2,
	"up_q80_m4":     shaders.LLMMoEUpQ80M4,
	"up_q80_w2m1":   shaders.LLMMoEUpQ80W2M1,
	"up_q80_w4m1":   shaders.LLMMoEUpQ80W4M1,
	"up_q4k_n2m1":   shaders.LLMMoEUpQ4KN2M1,
	"up_q4k_n1m1":   shaders.LLMMoEUpQ4KN1M1,
	"up_q5k_n2m1":   shaders.LLMMoEUpQ5KN2M1,
	"up_q5k_n1m1":   shaders.LLMMoEUpQ5KN1M1,
	"up_q80_n2m1":   shaders.LLMMoEUpQ80N2M1,
	"up_q80_n1m1":   shaders.LLMMoEUpQ80N1M1,
	"down_q51_m1":   shaders.LLMMoEDownQ51M1,
	"down_q51_m2":   shaders.LLMMoEDownQ51M2,
	"down_q51_m4":   shaders.LLMMoEDownQ51M4,
	"down_q51_w2m1": shaders.LLMMoEDownQ51W2M1,
	"down_q51_w4m1": shaders.LLMMoEDownQ51W4M1,
	"down_q80_m1":   shaders.LLMMoEDownQ80M1,
	"down_q80_m2":   shaders.LLMMoEDownQ80M2,
	"down_q80_m4":   shaders.LLMMoEDownQ80M4,
	"down_q80_w2m1": shaders.LLMMoEDownQ80W2M1,
	"down_q80_w4m1": shaders.LLMMoEDownQ80W4M1,

	// P4c's two down formats, the same five rungs each. They are built
	// because the *bank* can be transcoded into them, not because the
	// checkpoint ships one.
	"down_q41_m1":     shaders.LLMMoEDownQ41M1,
	"down_q41_m2":     shaders.LLMMoEDownQ41M2,
	"down_q41_m4":     shaders.LLMMoEDownQ41M4,
	"down_q41_w2m1":   shaders.LLMMoEDownQ41W2M1,
	"down_q41_w4m1":   shaders.LLMMoEDownQ41W4M1,
	"down_iq4nl_m1":   shaders.LLMMoEDownIQ4NLM1,
	"down_iq4nl_m2":   shaders.LLMMoEDownIQ4NLM2,
	"down_iq4nl_m4":   shaders.LLMMoEDownIQ4NLM4,
	"down_iq4nl_w2m1": shaders.LLMMoEDownIQ4NLW2M1,
	"down_iq4nl_w4m1": shaders.LLMMoEDownIQ4NLW4M1,

	// The decode rungs, which are a different kernel rather than a different
	// shape of the same one: llm_moe_gemv.comp, LLM.md L8d. Same two modes,
	// same four formats, same tile list — no LDS slab, no barrier in the K
	// loop and no cooperative matrix.
	"up_q4k_v16":       shaders.LLMMoEGVUpQ4KV16,
	"up_q4k_v32":       shaders.LLMMoEGVUpQ4KV32,
	"up_q4k_v64":       shaders.LLMMoEGVUpQ4KV64,
	"up_q4k_v16w4":     shaders.LLMMoEGVUpQ4KV16W4,
	"up_q4k_v32w4":     shaders.LLMMoEGVUpQ4KV32W4,
	"up_q4k_v64w4":     shaders.LLMMoEGVUpQ4KV64W4,
	"up_q5k_v16":       shaders.LLMMoEGVUpQ5KV16,
	"up_q5k_v32":       shaders.LLMMoEGVUpQ5KV32,
	"up_q5k_v64":       shaders.LLMMoEGVUpQ5KV64,
	"up_q5k_v16w4":     shaders.LLMMoEGVUpQ5KV16W4,
	"up_q5k_v32w4":     shaders.LLMMoEGVUpQ5KV32W4,
	"up_q5k_v64w4":     shaders.LLMMoEGVUpQ5KV64W4,
	"up_q80_v16":       shaders.LLMMoEGVUpQ80V16,
	"up_q80_v32":       shaders.LLMMoEGVUpQ80V32,
	"up_q80_v64":       shaders.LLMMoEGVUpQ80V64,
	"up_q80_v16w4":     shaders.LLMMoEGVUpQ80V16W4,
	"up_q80_v32w4":     shaders.LLMMoEGVUpQ80V32W4,
	"up_q80_v64w4":     shaders.LLMMoEGVUpQ80V64W4,
	"down_q51_v16":     shaders.LLMMoEGVDownQ51V16,
	"down_q51_v32":     shaders.LLMMoEGVDownQ51V32,
	"down_q51_v64":     shaders.LLMMoEGVDownQ51V64,
	"down_q51_v16w4":   shaders.LLMMoEGVDownQ51V16W4,
	"down_q51_v32w4":   shaders.LLMMoEGVDownQ51V32W4,
	"down_q51_v64w4":   shaders.LLMMoEGVDownQ51V64W4,
	"down_q80_v16":     shaders.LLMMoEGVDownQ80V16,
	"down_q80_v32":     shaders.LLMMoEGVDownQ80V32,
	"down_q80_v64":     shaders.LLMMoEGVDownQ80V64,
	"down_q80_v16w4":   shaders.LLMMoEGVDownQ80V16W4,
	"down_q80_v32w4":   shaders.LLMMoEGVDownQ80V32W4,
	"down_q80_v64w4":   shaders.LLMMoEGVDownQ80V64W4,
	"down_q41_v16":     shaders.LLMMoEGVDownQ41V16,
	"down_q41_v32":     shaders.LLMMoEGVDownQ41V32,
	"down_q41_v64":     shaders.LLMMoEGVDownQ41V64,
	"down_q41_v16w4":   shaders.LLMMoEGVDownQ41V16W4,
	"down_q41_v32w4":   shaders.LLMMoEGVDownQ41V32W4,
	"down_q41_v64w4":   shaders.LLMMoEGVDownQ41V64W4,
	"down_iq4nl_v16":   shaders.LLMMoEGVDownIQ4NLV16,
	"down_iq4nl_v32":   shaders.LLMMoEGVDownIQ4NLV32,
	"down_iq4nl_v64":   shaders.LLMMoEGVDownIQ4NLV64,
	"down_iq4nl_v16w4": shaders.LLMMoEGVDownIQ4NLV16W4,
	"down_iq4nl_v32w4": shaders.LLMMoEGVDownIQ4NLV32W4,
	"down_iq4nl_v64w4": shaders.LLMMoEGVDownIQ4NLV64W4,
}

// moeRouterSPIRV is the decode router's two dispatches at each split (L8d-3).
// It is a table of its own because the router is not a rung of the grouped
// GEMM: a different matrix, a different kernel and a ladder that does not
// cross the experts'.
var moeRouterSPIRV = map[string][]byte{
	"router_k8":    shaders.LLMMoERouterK8,
	"router_k8_r":  shaders.LLMMoERouterK8R,
	"router_k10":   shaders.LLMMoERouterK10,
	"router_k10_r": shaders.LLMMoERouterK10R,
	"router_k20":   shaders.LLMMoERouterK20,
	"router_k20_r": shaders.LLMMoERouterK20R,
	"router_k40":   shaders.LLMMoERouterK40,
	"router_k40_r": shaders.LLMMoERouterK40R,
}

// moeLayerWeights is where one layer's FFN half sits. The router is halves in
// the fp16 bank; everything else is the checkpoint's own bytes in the
// quantised bank, addressed by byte offset because a Q8_0 block is 34 bytes.
type moeLayerWeights struct {
	router                        uint32 // fp16 bank, halves
	bank                          int    // which of qbufs holds the six below
	gate, up, down                uint32 // quantised bank, bytes
	shGate, shUp, shDown          uint32
	gateFmt, upFmt, downFmt       moeFmt
	shGateFmt, shUpFmt, shDownFmt moeFmt
	// to is the ggml type each of the six is *staged* in, in the order the
	// reserve loop lists them (gate, up, down, shGate, shUp, shDown). It
	// equals the checkpoint's own type unless P4's `LLM_MOE_BANK` narrowed
	// it, and `stage` transcodes exactly where the two differ — one decision,
	// made once in `reserve` where the size depends on it, and read back
	// rather than made again.
	to [6]gguf.Type
}

// moeCombineWG is llm_moe_combine.comp's workgroup, and the column block its
// grid is cut to. It is a constant on both sides.
const moeCombineWG = 64

// moeMaxBanks is how many buffers binding 5 (and binding 6, the same buffers
// read as `uvec4`) holds, and it is the checkpoint's layer count: **a layer's
// quantised bank is a buffer of its own**.
//
// It is not a tidiness choice. `maxStorageBufferRange` on this device is
// 4 GiB - 4 and one layer's three routed banks plus its shared expert's are
// 1.61 GB, so two layers is the most one buffer could hold and 48 of them is
// 77 GB — the arena-with-an-offset arrangement every other block here uses
// cannot express residency (LLM.md L6a). The count is compiled into
// llm_moe_gemm.comp as NBANK, so it is a constant on both sides and
// TestMoEGPUBankArray is what holds them together; a stage of fewer layers
// pads the array with a placeholder, which no dispatch ever names.
const moeMaxBanks = 48

// MoEGPU runs the MoE block on the device.
type MoEGPU struct {
	// rec, when set, collects this block's dispatches into the pass's one
	// command buffer instead of submitting them (record.go).
	rec *recorder
	dev *vk.Device
	cfg MoEConfig

	wbuf, abuf, hbuf, bank *vk.Buffer
	// One quantised bank a layer, bound as one array of moeMaxBanks
	// descriptors. Beyond len(layers) they are the placeholder.
	qbufs []*vk.Buffer
	pipes map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	layers []moeLayerWeights
	// bankPlan is P4's `LLM_MOE_BANK`: which of the six MoE families are
	// staged narrower than the checkpoint ships them. Read once at
	// construction so that `reserve`'s sizes and `stage`'s bytes cannot
	// disagree about it, and so a `Formats` line names what actually ran.
	bankPlan MoEBankPlan

	up, down MoEKernel
	// The shared expert's own pair. It is the same two kernels over the same
	// shape, so it rode the routed pair's rung until L8e-2 — and at one token
	// the two want **opposite** ends of the LPR ladder, because a lane group
	// is sized to a row's payload words and the routed bank is Q4_K where the
	// shared one is Q8_0 (MoESharedPlanFor).
	shUp, shDown MoEKernel
	router       GEMMKernel
	routerGemv   MoERouterKernel
	autoPlan     bool
	// pinGemv holds the block on llm_moe_gemm.comp and the GEMM router
	// whatever the batch: see Graph.PinSchedule and DeltaNetGPU.pinGemv.
	pinGemv bool

	tokens, arenaRows, rows int
	lda, ldCtx              int

	// fp32 activation arena. The permutation, its inverse, the tile lists and
	// the top-k are uints in the same buffer, read through binding 4.
	aLogits, aWeights, aGated, aOut uint32
	aPerm, aBook                    uint32
	aTilesUp, aTilesDown            uint32
	aShTilesUp, aShTilesDown        uint32
	aRouterPart                     uint32
	actElems                        int
	// fp16 activation arena.
	hXn, hSwiglu uint32
	hElems       int

	qBytes int
}

// bankSet is the array binding 5 and binding 6 are written from: the staged
// layers' buffers, then the placeholder as many times as it takes to fill
// moeMaxBanks. Every descriptor of an array has to be written whether or not
// a shader ever indexes it.
func (g *MoEGPU) bankSet() []*vk.Buffer {
	set := make([]*vk.Buffer, moeMaxBanks)
	for i := range set {
		if i < len(g.qbufs) {
			set[i] = g.qbufs[i]
		} else {
			set[i] = g.wbuf
		}
	}
	return set
}

// routerN is the fused router's output width: the 512 experts, the shared
// expert's gate as a 513th column, and the pad up to the plain GEMM's column
// block.
func (g *MoEGPU) routerN() int { return roundUpInt(g.cfg.NExpert+1, attnBN) }

// NewMoEGPU stages layers onto the device and builds every pipeline. The
// quantised banks are copied byte for byte out of the mmap'd checkpoint, so
// one layer costs 1.57 GB of device memory and two is the most a 4 GiB
// buffer holds.
// MoEOption adjusts a block before it allocates. There is one, and it exists
// because P4's bank plan has to be varied *within* a process — the gate is
// two stagings of the same layer compared bit for bit, and `LLM_MOE_BANK` is
// read through a `sync.Once` like every other plan here.
type MoEOption func(*MoEGPU)

// WithMoEBankPlan stages this block under an explicit plan rather than the
// environment's.
func WithMoEBankPlan(p MoEBankPlan) MoEOption { return func(g *MoEGPU) { g.bankPlan = p } }

func NewMoEGPU(dev *vk.Device, cfg MoEConfig, maxTokens int, layers []MoEWeights, opts ...MoEOption) (*MoEGPU, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("llm: %d tokens", maxTokens)
	}
	if len(layers) == 0 {
		return nil, fmt.Errorf("llm: no MoE layers to stage")
	}
	// Both are compiled in: the route kernel caches the probabilities in LDS
	// and the permutation kernel scans the experts in fixed chunks, so a
	// wider checkpoint would be answered wrongly rather than slowly.
	if cfg.NExpert > 512 {
		return nil, fmt.Errorf("llm: the route and permutation kernels are built for at most 512 experts, this checkpoint says %d", cfg.NExpert)
	}
	if len(layers) > moeMaxBanks {
		return nil, fmt.Errorf("llm: %d layers to stage, and binding 5 is an array of %d — rebuild llm_moe_gemm.comp with -DNBANK=%d",
			len(layers), moeMaxBanks, len(layers))
	}
	if cfg.NExpertUsed > 16 {
		return nil, fmt.Errorf("llm: the route kernel is built for at most 16 experts a token, this checkpoint says %d", cfg.NExpertUsed)
	}
	if cfg.FFNShared%moeBN != 0 {
		return nil, fmt.Errorf("llm: shared expert width %d is not a multiple of the column block %d",
			cfg.FFNShared, moeBN)
	}
	if cfg.FFNExpert%moeBN != 0 || cfg.NEmbd%moeBN != 0 {
		return nil, fmt.Errorf("llm: ffn %d and n_embd %d must be multiples of the column block %d",
			cfg.FFNExpert, cfg.NEmbd, moeBN)
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("llm: this device has no 16x16x16 fp16 cooperative matrix")
	}
	g := &MoEGPU{
		dev: dev, cfg: cfg,
		pipes:  make(map[string]*vk.ComputePipeline),
		tokens: maxTokens, rows: maxTokens,
		lda:        cfg.NEmbd + gemmPad,
		ldCtx:      cfg.FFNExpert + gemmPad,
		router:     GEMMKernelFor(maxTokens),
		routerGemv: MoERouterFor(maxTokens),
		autoPlan:   true,
		bankPlan:   MoEBankPlanFromEnv(),
	}
	for _, opt := range opts {
		opt(g)
	}
	g.up, g.down = MoEPlanFor(maxTokens)
	g.shUp, g.shDown = MoESharedPlanFor(maxTokens)
	align := moeBMMax
	for _, v := range gemmVariants {
		align = maxInt(align, v.bm)
	}
	g.arenaRows = roundUpInt(maxTokens, align)

	if err := g.alloc(layers); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(layers); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// pad is the alignment every expert's row range is rounded up to: the wider
// of the two modes' row blocks, so that both tile lists divide it exactly and
// no tile ever straddles two experts. It is what lets the GEMM store whole
// cooperative-matrix fragments straight to global.
// pad is the permuted row space's alignment: the widest row block any of the
// four expert dispatches runs, because every one of them reads the same
// permutation and a tile list has to divide the space it indexes.
func (g *MoEGPU) pad() int {
	return maxInt(maxInt(moeBM(g.up), moeBM(g.down)), maxInt(moeBM(g.shUp), moeBM(g.shDown)))
}

// maxRows is the static upper bound on the permuted row space: the shared
// expert's group at the front, every routed row, and at most one alignment of
// slack per expert **that has a row**. The last clause is the whole bound at
// decode: a batch of `rows` tokens names at most `rows * used` experts, so
// ten of the 512 can be padded and not 512 (L8d-1).
func (g *MoEGPU) maxRows(rows int) int {
	return roundUpInt(rows, moeBMMax) + rows*g.cfg.NExpertUsed + g.touchable(rows)*moeBMMax
}

// touchable is how many experts a batch can possibly route to.
func (g *MoEGPU) touchable(rows int) int {
	return minInt(g.cfg.NExpert, rows*g.cfg.NExpertUsed)
}

// maxTiles is the static upper bound on (expert, row block) records at a
// given rung. **The GEMM's grid is this**, and everything past the schedule's
// real length returns before it reads a weight — so it is not a bookkeeping
// number, it is a dispatch: `gemmN/BN` workgroups for every tile it names,
// and an empty workgroup is about 0.5 ns.
//
// **L8d-1: it was `maxRows/bm`, which at one token is 2052 tiles where eleven
// exist.** `maxRows` allows every one of the 512 experts an alignment of
// slack, because a 512-token ubatch really can touch 274 of them — but the
// row space and the *tile list* are not the same bound, and a batch that
// routes ten rows makes at most ten padded experts however wide the padding
// is. At the GEMM's own BN of 64 that was 82 080 workgroups to run 11, which
// cost about 33 us a layer and was invisible because it was inside the
// dispatch it padded; at the GEMV rungs, whose whole point is a narrower BN,
// it is the dispatch. The bound below is exact at one token.
func (g *MoEGPU) maxTiles(bm, rows int) int {
	pad := g.pad()
	return roundUpInt(rows, pad)/bm + roundUpInt(rows*g.cfg.NExpertUsed, bm)/bm + g.touchable(rows)*(pad/bm)
}

// alloc lays out the five arenas. Nothing is read here — every size follows
// from the config — so the layout can be checked before a weight is touched.
func (g *MoEGPU) alloc(layers []MoEWeights) error {
	c := g.cfg
	rows := g.arenaRows
	used := c.NExpertUsed

	var err error
	// Binding 0 is unused by this block: every weight it reads is either in
	// the fp16 bank or in the quantised one. It is still bound, because the
	// descriptor layout is the vertical's.
	if g.wbuf, err = g.dev.NewBuffer(256); err != nil {
		return fmt.Errorf("llm: moe fp32 weight arena: %w", err)
	}

	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	g.aLogits = alloc(rows * g.routerN())
	// Four tensors of (token, slot), in one run: the weights as floats, then
	// the top-k, the inverse permutation as uints. The route, permutation and
	// combine kernels derive where each starts rather than being told.
	g.aWeights = alloc(rows * (3*used + 2))
	maxRows := g.maxRows(rows)
	g.aGated = alloc(maxRows * c.NEmbd)
	g.aOut = alloc(rows * c.NEmbd)
	// The permutation covers the whole padded row space: the shared expert's
	// group at the front and every routed expert's padded range behind it.
	g.aPerm = alloc(maxRows)
	// The counting sort's four per-expert arrays: counts, offsets, the
	// scatter cursor and the tile base.
	g.aBook = alloc(4 * c.NExpert)
	g.aTilesUp = alloc(2 + 3*g.maxTiles(moeBMMin, rows))
	g.aTilesDown = alloc(2 + 3*g.maxTiles(moeBMMin, rows))
	// The shared expert is one group of the same grouped GEMM — same shape,
	// same kernel, one expert — so it has a schedule of its own, and that one
	// is host-written because a dense group's is the identity.
	shTiles := 2 + 3*(roundUpInt(rows, moeBMMax)/moeBMMin)
	g.aShTilesUp = alloc(shTiles)
	// The decode router's partial sums, f32 [KSLABS][routerN] (L8d-3). 92 KB
	// at the widest rung, and allocated whichever rung runs — the arena plan
	// is fixed at construction and the rung is not.
	g.aRouterPart = alloc(GEMVMaxRows * moeRouterMaxSlabs * g.routerN())
	g.aShTilesDown = alloc(shTiles)
	if g.abuf, err = newArena(g.dev, g.actElems*4); err != nil {
		return fmt.Errorf("llm: moe fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	// One row past the batch, because that is where a padding row's
	// permutation entry points and it has to be zero.
	g.hXn = halloc((rows + moeBMMax) * g.lda)
	g.hSwiglu = halloc(maxRows * g.ldCtx)
	if g.hbuf, err = newArena(g.dev, g.hElems*2); err != nil {
		return fmt.Errorf("llm: moe fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once: the A operands' pad columns, the sentinel token's row and
	// every swiglu row no tile ever writes. None of these kernels
	// bounds-checks, and a stale half can be a NaN.
	g.hbuf.Zero()
	g.abuf.Zero()

	perRouter := g.routerN() * c.NEmbd
	if g.bank, err = g.dev.NewBuffer(len(layers) * perRouter * 2); err != nil {
		return fmt.Errorf("llm: moe fp16 weight bank (%d MB): %w", (len(layers)*perRouter*2)>>20, err)
	}

	// The quantised bank: the checkpoint's own bytes, laid end to end. Every
	// tensor here is a whole number of four-byte words, which is what lets
	// the shader read it as uints.
	//
	// **One buffer a layer**, because 48 of them are 77 GB and a storage
	// buffer on this device is at most 4 GiB - 4 (L6a). So `off` restarts at
	// every layer and the offsets a dispatch carries are inside that layer's
	// own buffer, which the bank index names.
	g.layers = make([]moeLayerWeights, len(layers))
	total := 0
	for i, w := range layers {
		g.layers[i].router = uint32(i * perRouter)
		g.layers[i].bank = i
		off := 0
		for j, t := range []struct {
			dst  *uint32
			fmt  *moeFmt
			name string
			ten  *gguf.Tensor
		}{
			{&g.layers[i].gate, &g.layers[i].gateFmt, "ffn_gate_exps", w.Gate.T},
			{&g.layers[i].up, &g.layers[i].upFmt, "ffn_up_exps", w.Up.T},
			{&g.layers[i].down, &g.layers[i].downFmt, "ffn_down_exps", w.Down.T},
			{&g.layers[i].shGate, &g.layers[i].shGateFmt, "ffn_gate_shexp", w.GateShexpT},
			{&g.layers[i].shUp, &g.layers[i].shUpFmt, "ffn_up_shexp", w.UpShexpT},
			{&g.layers[i].shDown, &g.layers[i].shDownFmt, "ffn_down_shexp", w.DownShexpT},
		} {
			if t.ten == nil {
				return fmt.Errorf("llm: layer %d has no %s", i, t.name)
			}
			// P4: the plan may narrow this tensor. The decision is made here
			// because the *size* depends on it, and `stage` reads it back.
			typ := t.ten.Type
			if to, move := g.bankPlan.For(t.ten.Name, typ); move {
				if int(t.ten.Dims[0])%to.BlockElems() != 0 {
					return fmt.Errorf("llm: %s has rows of %d and %s blocks are %d elements — "+
						"there is no such format for this tensor (P4)",
						t.ten.Name, t.ten.Dims[0], to, to.BlockElems())
				}
				typ = to
			}
			f, err := moeFmtOfType(typ)
			if err != nil {
				return fmt.Errorf("llm: %s staged as %s: %w", t.ten.Name, typ, err)
			}
			nbytes, err := typ.SizeOf(t.ten.Elems())
			if err != nil {
				return err
			}
			if nbytes%4 != 0 {
				return fmt.Errorf("llm: %s is %d bytes, not a whole number of words", t.ten.Name, nbytes)
			}
			// Sixteen-aligned, because the Q4_K and Q5_K arms read their
			// nibbles as `uvec4` and a super-block is a multiple of sixteen
			// bytes only if the row is, and the row only if the tensor is.
			*t.dst = uint32(off)
			*t.fmt = f
			g.layers[i].to[j] = typ
			off += roundUpInt(int(nbytes), 16)
		}
		buf, err := g.dev.NewBuffer(off)
		if err != nil {
			return fmt.Errorf("llm: moe quantised bank, layer %d (%d MB): %w", i, off>>20, err)
		}
		g.qbufs = append(g.qbufs, buf)
		total += off
	}
	g.qBytes = total
	return nil
}

func (g *MoEGPU) build() error {
	// Seven bindings against the rest of the vertical's five: the quantised
	// bank at binding 5, and the same buffers again at binding 6 as sixteen-
	// byte words, because the unpack issues half as many loads through it.
	// Only this block's shaders declare either, and the router's plain GEMM
	// arm declares neither — a descriptor set may have bindings its shader
	// never names, which is what lets one sequence mix them.
	//
	// The last two bindings are **arrays** of moeMaxBanks descriptors, one a
	// layer, so the seven bindings are 5 + 2*48 = 101 buffers (L6a).
	banks := g.bankSet()
	bufs := append([]*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank, g.abuf}, banks...)
	bufs = append(bufs, banks...)
	counts := []uint32{1, 1, 1, 1, 1, moeMaxBanks, moeMaxBanks}
	pcSize := uint32(unsafe.Sizeof(push{}))
	for name, spirv := range map[string][]byte{
		"route":   shaders.LLMMoERoute,
		"perm":    shaders.LLMMoEPerm,
		"combine": shaders.LLMMoECombine,
	} {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: bufs, Counts: counts, PushConstantSize: pcSize}); err != nil {
			return err
		}
	}
	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("llm: subgroup size control: %w", err)
	}
	if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < moeWave {
		return fmt.Errorf("llm: the grouped GEMM needs a pinned %d-wide subgroup", moeWave)
	}
	spec := vk.PipelineSpec{Buffers: bufs, Counts: counts, PushConstantSize: pcSize, RequiredSubgroupSize: moeWave}
	if sgs.MaxComputeWorkgroupSubgroups < 4 {
		return fmt.Errorf("llm: the widest rung needs four %d-wide subgroups a workgroup, this device allows %d",
			moeWave, sgs.MaxComputeWorkgroupSubgroups)
	}
	// One module, one pipeline per row count (P5b): `ROWS` is a
	// specialization constant in llm_moe_gemv.comp and llm_moe_router.comp,
	// so the seventy-odd `.spv` of these two families do not double. The
	// grouped GEMM below takes its row block from its own name and needs
	// none of this.
	for _, fam := range []map[string][]byte{moeSPIRV, moeRouterSPIRV} {
		for name, spirv := range fam {
			for rows := 1; rows <= GEMVMaxRows; rows++ {
				rspec := spec
				rspec.SpecConstants = gemvSpec(rows)
				if err := g.pipeline(gemvRowName(name, rows), spirv, rspec); err != nil {
					return err
				}
			}
		}
	}
	for _, v := range gemmVariants {
		if err := g.pipeline(string(v.name), v.spirv, spec); err != nil {
			return err
		}
	}
	return nil
}

func (g *MoEGPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("llm: shader moe.%s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("llm: pipeline moe.%s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// stage copies each layer's weights. The router and the shared expert's gate
// are one fused [nExpert+1, nEmbd] fp16 matrix; the six quantised tensors are
// memcpy'd out of the mmap'd checkpoint, which is 1.57 GB a layer and the
// reason this constructor takes seconds rather than milliseconds.
func (g *MoEGPU) stage(layers []MoEWeights) error {
	c := g.cfg
	for i, w := range layers {
		if len(w.Router) != c.NExpert*c.NEmbd {
			return fmt.Errorf("llm: layer %d router is %d values, want %d", i, len(w.Router), c.NExpert*c.NEmbd)
		}
		if len(w.SharedGate) != c.NEmbd {
			return fmt.Errorf("llm: layer %d shared gate is %d values, want %d", i, len(w.SharedGate), c.NEmbd)
		}
		router := make([]uint16, g.routerN()*c.NEmbd)
		tileB(router, w.Router, c.NExpert, c.NEmbd, func(r int) int { return r })
		nE := c.NExpert
		tileB(router, w.SharedGate, 1, c.NEmbd, func(int) int { return nE })
		g.bank.WriteUint16At(int(g.layers[i].router), router)

		for j, t := range []struct {
			off uint32
			ten *gguf.Tensor
		}{
			{g.layers[i].gate, w.Gate.T},
			{g.layers[i].up, w.Up.T},
			{g.layers[i].down, w.Down.T},
			{g.layers[i].shGate, w.GateShexpT},
			{g.layers[i].shUp, w.UpShexpT},
			{g.layers[i].shDown, w.DownShexpT},
		} {
			data := t.ten.Data
			if to := g.layers[i].to[j]; to != t.ten.Type {
				var err error
				if data, err = TranscodeMoE(t.ten, to, g.bankPlan.Mode()); err != nil {
					return err
				}
			}
			g.qbufs[g.layers[i].bank].WriteBytesAt(int(t.off), data)
		}
	}
	return nil
}

// SetPlan chooses the two rungs. Every rung is built and they all read the
// same staged bank, so this moves a pipeline and restages nothing.
func (g *MoEGPU) SetPlan(up, down MoEKernel) error {
	if moeBM(up) == 0 {
		return fmt.Errorf("llm: no MoE up kernel %q (have %v)", up, append(MoEUpKernels(), MoEDecodeKernels()...))
	}
	if moeBM(down) == 0 || (!MoEIsGemv(down) && moeBNOf(down) != moeBN) {
		return fmt.Errorf("llm: no MoE down kernel %q (have %v)", down, append(MoEKernels(), MoEDecodeKernels()...))
	}
	// The GEMV reads one row of a tile, and a tile has exactly one real row
	// only when the batch does: a token's top-k names ten *distinct* experts,
	// so at one token every tile is one row and at two it may be two.
	if err := moeCheckGemv(up, down, g.rows); err != nil {
		return err
	}
	g.up, g.down = up, down
	g.shUp, g.shDown = up, down
	g.autoPlan = false
	// The shared expert's schedule is host-built and cut to the same row
	// blocks, so it has to move with them. A plan set after an Upload would
	// otherwise run the new rungs against the old rung's tile records.
	g.syncShared()
	return nil
}

// SetSharedPlan chooses the shared expert's two rungs on their own, for the
// ladder and for the plan L8e-2 measured. Call it **after** SetPlan, which
// puts both pairs on the rungs it is given — so a caller that names only the
// routed pair still gets the one-field behaviour this split replaced.
func (g *MoEGPU) SetSharedPlan(up, down MoEKernel) error {
	if moeBM(up) == 0 {
		return fmt.Errorf("llm: no MoE up kernel %q (have %v)", up, append(MoEUpKernels(), MoEDecodeKernels()...))
	}
	if moeBM(down) == 0 || (!MoEIsGemv(down) && moeBNOf(down) != moeBN) {
		return fmt.Errorf("llm: no MoE down kernel %q (have %v)", down, append(MoEKernels(), MoEDecodeKernels()...))
	}
	if err := moeCheckGemv(up, down, g.rows); err != nil {
		return err
	}
	g.shUp, g.shDown = up, down
	g.autoPlan = false
	g.syncShared()
	return nil
}

// Plan reports the routed rungs in use.
func (g *MoEGPU) Plan() (MoEKernel, MoEKernel) { return g.up, g.down }

// SharedPlan reports the shared expert's two rungs.
func (g *MoEGPU) SharedPlan() (MoEKernel, MoEKernel) { return g.shUp, g.shDown }

// Layers and Tokens report what was staged.
func (g *MoEGPU) Layers() int { return len(g.layers) }
func (g *MoEGPU) Tokens() int { return g.rows }

// WeightBytes is what a run reads at most: the quantised bank plus the fp16
// router. What it reads in fact is smaller and depends on the prompt, because
// L5a-4's routing touches about half the experts at a 512-token ubatch.
func (g *MoEGPU) WeightBytes() int {
	n := g.bank.Size()
	for _, b := range g.qbufs {
		n += b.Size()
	}
	return n
}
func (g *MoEGPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// RouterBytes is what one token's routing reads: the fused router matrix, in
// the width it is **staged** in rather than the width the checkpoint ships.
//
// P4 exists because those two were confused. `ffn_gate_inp` is F32 in the
// GGUF — 0.252 GB over 48 layers, 4% of the decode budget for 0.06 B of
// parameters — and every budget line in `LLM.md`, `cmd/gguf`'s re-pricing and
// P1's attribution table quoted that number. But `stage` has narrowed it to
// halves since L5b: `g.bank` is `routerN*NEmbd*2` bytes a layer and the two
// router kernels read no other copy. So the true figure is **0.142 GB a
// token** (2.95 MB a layer, the 513 columns padded to 576), and D3's "the
// router to fp16, +1.5 tok/s of ceiling" was banked before it was proposed.
//
// It matters twice. The ceiling is under-quoted by 0.11 GB a token, and P1's
// 329.8 GB/s — excluded from the loss column under D16 as "not a DRAM
// measurement" — is **205 GB/s** on the right byte count, which is an
// ordinary streaming rate beside `deltanet`'s 201.5 and `moe.up`'s 200.6.
func (g *MoEGPU) RouterBytes() int { return g.bank.Size() / len(g.layers) }

// Buffers is how many device allocations the block holds: the four every
// block here has, and one quantised bank a layer (L6a).
func (g *MoEGPU) Buffers() int { return 4 + len(g.qbufs) }

// Upload narrows the block's input into the fp16 arena and writes the shared
// expert's own permutation and schedule, which are the identity over the
// run's tokens and so are host-side.
func (g *MoEGPU) Upload(x []float32, nTok int) error {
	c := g.cfg
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, staged for %d", nTok, g.tokens)
	}
	if len(x) < nTok*c.NEmbd {
		return fmt.Errorf("llm: input is %d values, want %d", len(x), nTok*c.NEmbd)
	}
	g.rows = nTok
	if g.autoPlan {
		g.router = GEMMKernelFor(nTok)
		g.up, g.down, g.routerGemv = g.moePlan(nTok)
		g.shUp, g.shDown = g.moeSharedPlan(nTok)
	} else if err := moeCheckGemv(g.up, g.down, nTok); err != nil {
		return err
	} else if err := moeCheckGemv(g.shUp, g.shDown, nTok); err != nil {
		return err
	}
	rowsPad := roundUpInt(nTok, moeBMMax)
	xn := make([]uint16, rowsPad*g.lda)
	narrowRows(xn, x, nTok, c.NEmbd, g.lda)
	g.hbuf.WriteUint16At(int(g.hXn), xn)
	g.syncShared()
	return nil
}

// syncShared writes the shared expert's rows and its two tile lists.
//
// It is one group of the same grouped GEMM — the same shape as a routed
// expert, the same two kernels, one expert — and it takes the **front** of the
// permuted row space because it runs for every token, so its rows are the
// identity and need no counting sort. The slack up to the alignment gets the
// same sentinel a routed expert's padding does.
func (g *MoEGPU) syncShared() {
	used := g.cfg.NExpertUsed
	slots := used + 1
	pad := g.pad()
	reserve := roundUpInt(g.rows, pad)
	perm := make([]uint32, reserve)
	for t := 0; t < g.rows; t++ {
		perm[t] = uint32(t*slots + used)
	}
	for t := g.rows; t < reserve; t++ {
		perm[t] = uint32(g.rows * slots)
	}
	g.writeUints(g.aPerm, perm)
	// Only the shared expert's column of the inverse is host-written; the
	// permutation kernel writes the other ten.
	invOff := int(g.aWeights) + g.rows*slots + g.rows*used
	inv := unsafe.Slice((*uint32)(g.abuf.MappedPointer()), invOff+g.rows*slots)
	for t := 0; t < g.rows; t++ {
		inv[invOff+t*slots+used] = uint32(t)
	}
	for _, t := range []struct {
		off uint32
		bm  int
	}{{g.aShTilesUp, moeBM(g.shUp)}, {g.aShTilesDown, moeBM(g.shDown)}} {
		n := reserve / t.bm
		rec := make([]uint32, 2+3*n)
		rec[0] = uint32(n)
		rec[1] = uint32(reserve)
		for mb := 0; mb < n; mb++ {
			rec[2+3*mb+1] = uint32(mb * t.bm)
			rec[2+3*mb+2] = uint32(t.bm)
		}
		g.writeUints(t.off, rec)
	}
}

// poisonPad writes v across the sentinel token's activation row — the row
// every padding row of the schedule reads. A run is supposed to be completely
// insensitive to it; TestMoEGPUPaddingIsInert is what makes that a claim
// rather than an assumption.
func (g *MoEGPU) poisonPad(v float32) {
	row := make([]uint16, g.lda)
	h := safetensors.F32ToF16(v)
	for i := range row {
		row[i] = h
	}
	g.hbuf.WriteUint16At(int(g.hXn)+g.rows*g.lda, row)
}

// moeExpertPipe names the pipeline one of the two expert modes runs: the
// bank's format, the rung, and — for a GEMV rung — **the row count it was
// specialized to** (P5b).
//
// **The last part is what P5c found missing.** P5b made `ROWS` a
// specialization constant and built one pipeline per row count, but the four
// expert dispatches went on naming the bare pipeline, which is the one
// specialized to a single row. So a two-row batch ran the one-row kernel over
// a two-row schedule and every tile's second row was left holding whatever the
// arena last had. The router's two dispatches were named correctly, which is
// why the router was the one arm of the block that agreed.
//
// It is invisible to a block test that drives `Run` twice in a row and
// compares the second run's row 1 against the first's — the arena still holds
// what the reference pass wrote there — and it is what a whole-graph two-token
// chunk split sees at once, because the graph's arena holds a 480-token
// prefill instead.
func moeExpertPipe(mode string, f moeFmt, k MoEKernel, rows int) string {
	name := fmt.Sprintf("%s_%s_%s", mode, f, k)
	if MoEIsGemv(k) {
		return gemvRowName(name, rows)
	}
	return name
}

func (g *MoEGPU) writeUints(off uint32, src []uint32) {
	dst := unsafe.Slice((*uint32)(g.abuf.MappedPointer()), int(off)+len(src))
	copy(dst[off:], src)
}

// graph builds one layer's dispatch sequence with a label per dispatch, and
// is shared by Run and Profile so that what the profiler times is what a run
// executes.
func (g *MoEGPU) graph(layer int) ([]vk.MultiDispatch, []string, error) {
	if layer < 0 || layer >= len(g.layers) {
		return nil, nil, fmt.Errorf("llm: layer %d of %d", layer, len(g.layers))
	}
	c := g.cfg
	w := g.layers[layer]
	rv, _ := gemmVariantFor(g.router)
	bmUp, bmDown := moeBM(g.up), moeBM(g.down)

	base := push{
		Tokens: uint32(g.rows), NEmbd: uint32(c.NEmbd),
		LDA: uint32(g.lda), LDCtx: uint32(g.ldCtx),
		XnOff: g.hXn, QKVOff: g.aLogits, OutOff: g.aOut, GatedOff: g.aGated,
		MoEWeightOff: g.aWeights,
		// The used count in the low sixteen bits and the layer's bank index
		// in the high, because the push block is full at 64 uints and the
		// GEMM needs to know which buffer of the array it reads (L6a).
		MoEUsed: uint32(c.NExpertUsed) | uint32(w.bank)<<16,
		GateOff: noW,
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc push) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	// 1. The router, with the shared expert's gate as one more output column.
	//    At one token it is a split-K GEMV in two dispatches and at every
	//    other length the GEMM (L8d-3); both write the same f32 logits row.
	router := base
	router.OutOff, router.BOff = g.aLogits, w.router
	router.GemmM = uint32(roundUpInt(g.rows, rv.bm))
	router.GemmN, router.GemmK = uint32(g.routerN()), uint32(c.NEmbd)
	if ks := moeRouterSlabs(g.routerGemv); ks > 0 {
		router.GammaOff = g.aRouterPart
		add(gemvRowName(fmt.Sprintf("router_%s", g.routerGemv), g.rows), "router",
			uint32(ks), uint32(g.routerN()/16), router)
		add(gemvRowName(fmt.Sprintf("router_%s_r", g.routerGemv), g.rows), "router.sum",
			uint32(roundUpInt(g.routerN(), moeWave)/moeWave), uint32(g.rows), router)
	} else {
		add(string(g.router), "router", uint32(g.routerN()/attnBN),
			uint32(roundUpInt(g.rows, rv.bm)/rv.bm), router)
	}

	// 2. The softmax, the ten argmaxes, the normalise and the shared gate.
	route := base
	route.GemmN, route.GemmK = uint32(g.routerN()), uint32(c.NExpert)
	route.MoEPermOff = g.aPerm
	add("route", "route", uint32(g.rows), 1, route)

	// 3-4. The counting sort, and the tile schedules it implies. It runs
	//      twice because the two modes are cut to different row blocks and a
	//      tile list is only consistent with the permutation that outlives
	//      it — so both passes happen before either GEMM reads one.
	//
	//      **Unless the two row blocks are the same number**, which is every
	//      decode step: all six GEMV rungs cut their tile list to sixteen
	//      rows, so the second pass recomputes the first pass's permutation
	//      and emits the same records into a second buffer (L8e-3). Then
	//      there is one pass and `downTiles` points the down mode at the up
	//      mode's list.
	passes := []struct {
		off uint32
		bm  int
		lbl string
	}{{g.aTilesUp, bmUp, "perm.up"}}
	if bmDown != bmUp {
		passes = append(passes, struct {
			off uint32
			bm  int
			lbl string
		}{g.aTilesDown, bmDown, "perm.down"})
	}
	for _, t := range passes {
		p := base
		p.GemmK, p.GemmM, p.GemmN = uint32(c.NExpert), uint32(t.bm), uint32(g.pad())
		p.MoEPermOff, p.MoETileOff, p.MoEBOff2 = g.aPerm, t.off, g.aBook
		add("perm", t.lbl, 1, 1, p)
	}

	// 5. gate and up for every routed row, with the swiglu on the
	//    accumulators.
	up := base
	up.MoEPermOff, up.MoETileOff = g.aPerm, g.aTilesUp
	up.BOff, up.MoEBOff2 = w.gate, w.up
	up.CtxOff = g.hSwiglu
	up.GemmN, up.GemmK = uint32(c.FFNExpert), uint32(c.NEmbd)
	add(moeExpertPipe("up", w.gateFmt, g.up, g.rows), "up",
		uint32(c.FFNExpert/moeBNOf(g.up)), uint32(g.maxTiles(bmUp, g.rows)), up)

	// 6. down, weighted and scattered to its (token, slot).
	down := base
	down.MoEPermOff, down.MoETileOff = g.aPerm, g.downTiles()
	down.BOff = w.down
	down.CtxOff = g.hSwiglu
	down.GemmN, down.GemmK = uint32(c.NEmbd), uint32(c.FFNExpert)
	add(moeExpertPipe("down", w.downFmt, g.down, g.rows), "down",
		uint32(c.NEmbd/moeBNOf(g.down)), uint32(g.maxTiles(bmDown, g.rows)), down)

	// 7-8. The shared expert: the same two kernels over one group whose
	//      permutation is the identity, because it is the same shape as a
	//      routed expert and runs for every token.
	shUp := base
	shUp.MoEPermOff, shUp.MoETileOff = g.aPerm, g.aShTilesUp
	shUp.BOff, shUp.MoEBOff2 = w.shGate, w.shUp
	shUp.CtxOff = g.hSwiglu
	shUp.GemmN, shUp.GemmK = uint32(c.FFNShared), uint32(c.NEmbd)
	add(moeExpertPipe("up", w.shGateFmt, g.shUp, g.rows), "shexp.up",
		uint32(c.FFNShared/moeBNOf(g.shUp)), uint32(roundUpInt(g.rows, g.pad())/moeBM(g.shUp)), shUp)

	shDown := base
	shDown.MoEPermOff, shDown.MoETileOff = g.aPerm, g.aShTilesDown
	shDown.BOff = w.shDown
	shDown.CtxOff = g.hSwiglu
	shDown.GemmN, shDown.GemmK = uint32(c.NEmbd), uint32(c.FFNShared)
	add(moeExpertPipe("down", w.shDownFmt, g.shDown, g.rows), "shexp.down",
		uint32(c.NEmbd/moeBNOf(g.shDown)), uint32(roundUpInt(g.rows, g.pad())/moeBM(g.shDown)), shDown)

	// 9. `ffn_out`: the ten and the shared expert's, in slot order.
	add("combine", "combine", uint32(roundUpInt(c.NEmbd, moeCombineWG)/moeCombineWG), uint32(g.rows), base)
	return d, kinds, nil
}

// Run executes one layer's block over whatever Upload left in the arenas.
func (g *MoEGPU) Run(layer int) error {
	d, kinds, err := g.graph(layer)
	if err != nil {
		return err
	}
	// One command buffer for the whole block, not one a dispatch: a submit
	// and a fence wait is ~150 us here and a decode step is a batch of one,
	// so what the sequence costs is how many times it is handed over
	// (LLM.md L7c). And when the graph is recording a whole pass, not even
	// one a block (L7d).
	if g.rec.add(ownMoE, kinds, d) {
		return nil
	}
	if _, err := vk.DispatchMultiTimed(d, 1, 1, true); err != nil {
		return fmt.Errorf("llm: moe, %d dispatches (%v): %w", len(d), kinds, err)
	}
	return nil
}

// Profile times each dispatch on the GPU over iters back-to-back repetitions.
//
// Unlike the other four blocks, repeating a dispatch here does not warm
// anything that matters: one layer's bank is **1.57 GB** against a 32 MiB
// MALL, so the second repetition reads the weights as cold as the first.
// What repetition does hide is the host-side staging, which is the point.
func (g *MoEGPU) Profile(layer, iters int) ([]Stage, error) {
	if iters <= 0 {
		iters = 1
	}
	d, kinds, err := g.graph(layer)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(d))
	for i := range d {
		dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, uint32(iters), true)
		if err != nil {
			return out, fmt.Errorf("llm: moe dispatch %d (%s): %w", i, kinds[i], err)
		}
		out = append(out, Stage{Kind: kinds[i], GPU: dur / time.Duration(iters)})
	}
	return out, nil
}

// ProfileSweep times each kind of dispatch across every staged layer.
func (g *MoEGPU) ProfileSweep(iters int) ([]Stage, error) {
	if iters <= 0 {
		iters = 1
	}
	_, kinds, err := g.graph(0)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(kinds))
	for k := range kinds {
		var total time.Duration
		for l := range g.layers {
			d, _, err := g.graph(l)
			if err != nil {
				return nil, err
			}
			dur, err := vk.DispatchMultiTimed(d[k:k+1], 1, uint32(iters), true)
			if err != nil {
				return out, fmt.Errorf("llm: %s sweep, layer %d: %w", kinds[k], l, err)
			}
			total += dur
		}
		out = append(out, Stage{Kind: kinds[k], GPU: total / time.Duration(len(g.layers)*iters)})
	}
	return out, nil
}

// The tensors a run leaves behind, in the layouts moe.go's CPU reference
// uses — so a disagreement lands on a node of llama.cpp's own graph.

// Logits is the fused router's output, [T][nExpert+1]: `ffn_moe_logits` and,
// in its last column, the raw `shared_expert_gate`.
func (g *MoEGPU) Logits() []float32 {
	raw := g.abuf.ReadFloat32At(int(g.aLogits), g.rows*g.routerN())
	out := make([]float32, g.rows*(g.cfg.NExpert+1))
	for t := 0; t < g.rows; t++ {
		copy(out[t*(g.cfg.NExpert+1):], raw[t*g.routerN():t*g.routerN()+g.cfg.NExpert+1])
	}
	return out
}

// TopK is the selection, [T][used], in the order the ten argmaxes took them.
func (g *MoEGPU) TopK() []int32 {
	used := g.cfg.NExpertUsed
	raw := g.abuf.ReadUint32At(int(g.aWeights)+g.rows*(used+1), g.rows*used)
	out := make([]int32, len(raw))
	for i, v := range raw {
		out[i] = int32(v)
	}
	return out
}

// Weights is [T][used+1]: the ten normalised routing weights and, in the last
// slot, sigmoid(shared_expert_gate).
func (g *MoEGPU) Weights() []float32 {
	return g.abuf.ReadFloat32At(int(g.aWeights), g.rows*(g.cfg.NExpertUsed+1))
}

// Counts is the routing histogram the permutation kernel built: how many rows
// each expert has. It is L5a-4's distribution, measured on the device.
func (g *MoEGPU) Counts() []uint32 {
	return g.abuf.ReadUint32At(int(g.aBook), g.cfg.NExpert)
}

// Offsets is where each expert's padded row range starts, behind the shared
// expert's reserve.
func (g *MoEGPU) Offsets() []uint32 {
	return g.abuf.ReadUint32At(int(g.aBook)+g.cfg.NExpert, g.cfg.NExpert)
}

// Perm is the permuted row space: the shared expert's group at the front, then
// every routed expert's padded range. Each entry is
// `token * (used + 1) + slot`, and a padding row is the sentinel
// `tokens * (used + 1)` — a token one past the batch, whose activation row is
// zero.
func (g *MoEGPU) Perm() []uint32 {
	return g.abuf.ReadUint32At(int(g.aPerm), g.Rows())
}

// InvPerm is where a (token, slot) landed: [T][used+1], the shared expert's
// slot last.
func (g *MoEGPU) InvPerm() []uint32 {
	slots := g.cfg.NExpertUsed + 1
	return g.abuf.ReadUint32At(int(g.aWeights)+g.rows*slots+g.rows*g.cfg.NExpertUsed, g.rows*slots)
}

// Reserve is where the routed rows begin: the shared expert's group, rounded
// up to the schedule's alignment.
func (g *MoEGPU) Reserve() int { return roundUpInt(g.rows, g.pad()) }

// Rows is how many rows the permuted space holds, padding included.
func (g *MoEGPU) Rows() int {
	_, exec, _ := g.Schedule(g.up)
	return g.Reserve() + exec
}

// downTiles is the list the down mode reads: its own, or the up mode's when
// the two are cut to the same row block and the second sort pass was elided
// (L8e-3).
func (g *MoEGPU) downTiles() uint32 {
	if moeBM(g.down) == moeBM(g.up) {
		return g.aTilesUp
	}
	return g.aTilesDown
}

// Tiles reports how many (expert, row block) records each schedule holds
// against the static bound the grid covers, which is what says how much of
// the dispatch is empty.
func (g *MoEGPU) Tiles() (up, down, boundUp, boundDown int) {
	u := g.abuf.ReadUint32At(int(g.aTilesUp), 1)[0]
	d := g.abuf.ReadUint32At(int(g.downTiles()), 1)[0]
	return int(u), int(d), g.maxTiles(moeBM(g.up), g.rows), g.maxTiles(moeBM(g.down), g.rows)
}

// Swiglu is `ffn_moe_swiglu` in permutation order, [rows][ffn], and
// Unpermute turns it back into the reference's [T][used][ffn].
func (g *MoEGPU) Swiglu() []float32 {
	n := g.Rows()
	raw := g.hbuf.ReadUint16At(int(g.hSwiglu), n*g.ldCtx)
	out := make([]float32, n*g.cfg.FFNExpert)
	for r := 0; r < n; r++ {
		for j := 0; j < g.cfg.FFNExpert; j++ {
			out[r*g.cfg.FFNExpert+j] = safetensors.F16ToF32(raw[r*g.ldCtx+j])
		}
	}
	return out
}

// ShSwiglu is the shared expert's `silu(gate)*up`, [T][ffnShared] — the front
// of the same buffer, because it is one group of the same grouped GEMM.
func (g *MoEGPU) ShSwiglu() []float32 {
	raw := g.hbuf.ReadUint16At(int(g.hSwiglu), g.rows*g.ldCtx)
	out := make([]float32, g.rows*g.cfg.FFNShared)
	for r := 0; r < g.rows; r++ {
		for j := 0; j < g.cfg.FFNShared; j++ {
			out[r*g.cfg.FFNShared+j] = safetensors.F16ToF32(raw[r*g.ldCtx+j])
		}
	}
	return out
}

// Unpermute reads a [rows][width] tensor in permutation order back into the
// reference's [T][used][width], which is what a tensor comparison needs. The
// shared expert's slot is dropped, because the reference names that tensor
// separately.
func (g *MoEGPU) Unpermute(src []float32, width int) []float32 {
	used := g.cfg.NExpertUsed
	slots := used + 1
	inv := g.InvPerm()
	out := make([]float32, g.rows*used*width)
	for t := 0; t < g.rows; t++ {
		for k := 0; k < used; k++ {
			r := int(inv[t*slots+k])
			copy(out[(t*used+k)*width:(t*used+k+1)*width], src[r*width:(r+1)*width])
		}
	}
	return out
}

// Weighted is `ffn_moe_weighted` beside `ffn_shexp_gated`: [T][used+1][nEmbd],
// the down projection's output times its slot's weight.
//
// Neither the multiply nor this layout exists on the device — the down
// projection stores whole cooperative-matrix fragments in permutation order
// and the combine carries the weight — so this is the reference's tensor
// reconstructed from the two things that do exist, which is what a comparison
// against llama.cpp's graph needs and what a kernel should not be made to
// produce.
func (g *MoEGPU) Weighted() []float32 {
	c := g.cfg
	slots := c.NExpertUsed + 1
	raw := g.abuf.ReadFloat32At(int(g.aGated), g.Rows()*c.NEmbd)
	w := g.Weights()
	inv := g.InvPerm()
	out := make([]float32, g.rows*slots*c.NEmbd)
	for t := 0; t < g.rows; t++ {
		for k := 0; k < slots; k++ {
			r := int(inv[t*slots+k])
			wt := w[t*slots+k]
			src := raw[r*c.NEmbd : (r+1)*c.NEmbd]
			dst := out[(t*slots+k)*c.NEmbd:]
			for j, v := range src {
				dst[j] = v * wt
			}
		}
	}
	return out
}

// Out is `ffn_out`, [T][nEmbd] — the block's output.
func (g *MoEGPU) Out() []float32 {
	return g.abuf.ReadFloat32At(int(g.aOut), g.rows*g.cfg.NEmbd)
}

// Destroy releases every object the block owns.
func (g *MoEGPU) Destroy() {
	for _, p := range g.pipes {
		if p != nil {
			p.Destroy()
		}
	}
	g.pipes = nil
	for _, m := range g.mods {
		m.Destroy()
	}
	g.mods = nil
	for _, b := range append([]*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank}, g.qbufs...) {
		if b != nil {
			b.Destroy()
		}
	}
	g.wbuf, g.abuf, g.hbuf, g.bank, g.qbufs = nil, nil, nil, nil, nil
}

// Schedule reports what a rung's tile list costs: how many (expert, row
// block) records it holds and how many rows those records execute against
// how many are real. The gap is L5a-4's distribution paying for a static
// tiling — 273 of the 274 experts a 512-token ubatch touches have fewer than
// 19 rows, so most tiles are mostly padding — and it is what says whether a
// wide row block is reuse or waste.
func (g *MoEGPU) Schedule(k MoEKernel) (tiles, executed, real int) {
	bm, pad := moeBM(k), g.pad()
	for _, c := range g.Counts() {
		padded := roundUpInt(int(c), pad)
		tiles += padded / bm
		executed += padded
		real += int(c)
	}
	return tiles, executed, real
}

// Touched is how many of the experts this prompt's routing reaches. L5a-4
// measured 274 of 512 at a 512-token ubatch and 435 at 4096; it is a fact
// about the prompt, and it is what decides how much of the bank a run reads.
func (g *MoEGPU) Touched() int {
	n := 0
	for _, c := range g.Counts() {
		if c > 0 {
			n++
		}
	}
	return n
}

// ExpertBytes is how many bytes of the quantised bank one expert costs, in
// the checkpoint's own formats: the gate/up pair a swiglu tile reads, and the
// down matrix a down tile reads. The whole bank divided by the expert count,
// so it follows the layer's formats rather than assuming them.
func (g *MoEGPU) ExpertBytes(layer int) (upPair, down int) {
	c := g.cfg
	w := g.layers[layer]
	upPair = c.FFNExpert * (moeRowBytes(w.gateFmt, c.NEmbd) + moeRowBytes(w.upFmt, c.NEmbd))
	down = c.NEmbd * moeRowBytes(w.downFmt, c.FFNExpert)
	return upPair, down
}

// SharedBytes is the same for the always-on expert, which runs for every
// token and is the same shape as a routed one.
func (g *MoEGPU) SharedBytes(layer int) (upPair, down int) {
	c := g.cfg
	w := g.layers[layer]
	// It used to assume Q8_0 on all three, which was true of every
	// checkpoint width until P4's transcode moved two of them.
	upPair = c.FFNShared * (moeRowBytes(w.shGateFmt, c.NEmbd) + moeRowBytes(w.shUpFmt, c.NEmbd))
	down = c.NEmbd * moeRowBytes(w.shDownFmt, c.FFNShared)
	return upPair, down
}

// moeRowBytes is one row of k elements in a bank format.
func moeRowBytes(f moeFmt, k int) int {
	switch f {
	case fmtQ4K:
		return k / 256 * 144
	case fmtQ5K:
		return k / 256 * 176
	case fmtQ51:
		return k / 32 * 24
	case fmtQ41:
		return k / 32 * 20
	case fmtIQ4NL:
		return k / 32 * 18
	}
	return k / 32 * 34
}

// Formats names the layer's three routed banks, which decide which build of
// the grouped GEMM runs and how many bytes an expert costs.
func (g *MoEGPU) Formats(layer int) (gate, up, down string) {
	w := g.layers[layer]
	return w.gateFmt.String(), w.upFmt.String(), w.downFmt.String()
}

// InPort is the block's input as the router's and every routed tile's A
// operand wants it: fp16 [T][lda], `hc_mixed` narrowed. The row count is the
// widest row block a rung uses, because the grouped GEMM has no bounds check.
func (g *MoEGPU) InPort() Port {
	return Port{Buf: g.hbuf, Off: g.hXn, Stride: g.lda, Width: g.cfg.NEmbd,
		Rows: roundUpInt(g.rows, moeBMMax), Half: true}
}

// OutPort is the block's output, `ffn_out`: fp32 [T][nEmbd].
func (g *MoEGPU) OutPort() Port {
	return Port{Buf: g.abuf, Off: g.aOut, Stride: g.cfg.NEmbd, Width: g.cfg.NEmbd}
}

// Resize sets the length of the run and writes the shared expert's own
// permutation and schedule, which are the identity over the run's tokens and
// so stay host-side however the input arrives.
func (g *MoEGPU) Resize(nTok int) error {
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, staged for %d", nTok, g.tokens)
	}
	if !g.autoPlan {
		if err := moeCheckGemv(g.up, g.down, nTok); err != nil {
			return err
		}
		if err := moeCheckGemv(g.shUp, g.shDown, nTok); err != nil {
			return err
		}
	}
	g.rows = nTok
	if g.autoPlan {
		g.router = GEMMKernelFor(nTok)
		g.up, g.down, g.routerGemv = g.moePlan(nTok)
		g.shUp, g.shDown = g.moeSharedPlan(nTok)
	}
	g.syncShared()
	return nil
}
