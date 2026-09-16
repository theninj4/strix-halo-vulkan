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
)

// MoEKernels lists the rungs, narrowest first.
func MoEKernels() []MoEKernel {
	return []MoEKernel{MoEM1, MoEM2, MoEM4, MoEW2M1, MoEW4M1}
}

func moeBM(k MoEKernel) int {
	switch k {
	case MoEM1:
		return 16
	case MoEM2, MoEW2M1:
		return 32
	case MoEM4, MoEW4M1:
		return 64
	}
	return 0
}

// MoEWaves is how many waves a rung's workgroup holds, which is what decides
// whether the dequantised slab is shared and how much of the machine is busy.
func MoEWaves(k MoEKernel) int {
	switch k {
	case MoEW2M1:
		return 2
	case MoEW4M1:
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
func MoEPlanFor(tokens int) (MoEKernel, MoEKernel) {
	if tokens <= 1024 {
		return MoEM2, MoEM2
	}
	return MoEM4, MoEM4
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
)

func (f moeFmt) String() string {
	switch f {
	case fmtQ4K:
		return "q4k"
	case fmtQ5K:
		return "q5k"
	case fmtQ51:
		return "q51"
	}
	return "q80"
}

// moeFmtOf maps a GGUF type onto a build of the kernel, and refuses anything
// else rather than reading it as something it is not.
func moeFmtOf(t *gguf.Tensor) (moeFmt, error) {
	switch t.Type {
	case gguf.Q4_K:
		return fmtQ4K, nil
	case gguf.Q5_K:
		return fmtQ5K, nil
	case gguf.Q5_1:
		return fmtQ51, nil
	case gguf.Q8_0:
		return fmtQ80, nil
	}
	return 0, fmt.Errorf("llm: %s is ggml type %d, which the MoE kernel has no build for", t.Name, t.Type)
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
}

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
	dev *vk.Device
	cfg MoEConfig

	wbuf, abuf, hbuf, bank *vk.Buffer
	// One quantised bank a layer, bound as one array of moeMaxBanks
	// descriptors. Beyond len(layers) they are the placeholder.
	qbufs []*vk.Buffer
	pipes map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	layers []moeLayerWeights

	up, down MoEKernel
	router   GEMMKernel
	autoPlan bool

	tokens, arenaRows, rows int
	lda, ldCtx              int

	// fp32 activation arena. The permutation, its inverse, the tile lists and
	// the top-k are uints in the same buffer, read through binding 4.
	aLogits, aWeights, aGated, aOut uint32
	aPerm, aBook                    uint32
	aTilesUp, aTilesDown            uint32
	aShTilesUp, aShTilesDown        uint32
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
func NewMoEGPU(dev *vk.Device, cfg MoEConfig, maxTokens int, layers []MoEWeights) (*MoEGPU, error) {
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
		lda:      cfg.NEmbd + gemmPad,
		ldCtx:    cfg.FFNExpert + gemmPad,
		router:   GEMMKernelFor(maxTokens),
		autoPlan: true,
	}
	g.up, g.down = MoEPlanFor(maxTokens)
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
func (g *MoEGPU) pad() int { return maxInt(moeBM(g.up), moeBM(g.down)) }

// maxRows is the static upper bound on the permuted row space: the shared
// expert's group at the front, every routed row, and at most one alignment of
// slack per expert.
func (g *MoEGPU) maxRows(rows int) int {
	return roundUpInt(rows, moeBMMax) + rows*g.cfg.NExpertUsed + g.cfg.NExpert*moeBMMax
}

// maxTiles is the static upper bound on (expert, row block) records at a
// given rung. The GEMM's grid is this, and everything past the schedule's
// real length returns before it reads a weight.
func (g *MoEGPU) maxTiles(bm, rows int) int {
	return g.maxRows(rows) / bm
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
		for _, t := range []struct {
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
			f, err := moeFmtOf(t.ten)
			if err != nil {
				return err
			}
			if len(t.ten.Data)%4 != 0 {
				return fmt.Errorf("llm: %s is %d bytes, not a whole number of words", t.ten.Name, len(t.ten.Data))
			}
			// Sixteen-aligned, because the Q4_K and Q5_K arms read their
			// nibbles as `uvec4` and a super-block is a multiple of sixteen
			// bytes only if the row is, and the row only if the tensor is.
			*t.dst = uint32(off)
			*t.fmt = f
			off += roundUpInt(len(t.ten.Data), 16)
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
	for name, spirv := range moeSPIRV {
		if err := g.pipeline(name, spirv, spec); err != nil {
			return err
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

		for _, t := range []struct {
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
			g.qbufs[g.layers[i].bank].WriteBytesAt(int(t.off), t.ten.Data)
		}
	}
	return nil
}

// SetPlan chooses the two rungs. Every rung is built and they all read the
// same staged bank, so this moves a pipeline and restages nothing.
func (g *MoEGPU) SetPlan(up, down MoEKernel) error {
	for _, k := range []MoEKernel{up, down} {
		if moeBM(k) == 0 {
			return fmt.Errorf("llm: no MoE kernel %q (have %v)", k, MoEKernels())
		}
	}
	g.up, g.down = up, down
	g.autoPlan = false
	// The shared expert's schedule is host-built and cut to the same row
	// blocks, so it has to move with them. A plan set after an Upload would
	// otherwise run the new rungs against the old rung's tile records.
	g.syncShared()
	return nil
}

// Plan reports the rungs in use.
func (g *MoEGPU) Plan() (MoEKernel, MoEKernel) { return g.up, g.down }

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
		g.up, g.down = MoEPlanFor(nTok)
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
	}{{g.aShTilesUp, moeBM(g.up)}, {g.aShTilesDown, moeBM(g.down)}} {
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
	router := base
	router.OutOff, router.BOff = g.aLogits, w.router
	router.GemmM = uint32(roundUpInt(g.rows, rv.bm))
	router.GemmN, router.GemmK = uint32(g.routerN()), uint32(c.NEmbd)
	add(string(g.router), "router", uint32(g.routerN()/attnBN),
		uint32(roundUpInt(g.rows, rv.bm)/rv.bm), router)

	// 2. The softmax, the ten argmaxes, the normalise and the shared gate.
	route := base
	route.GemmN, route.GemmK = uint32(g.routerN()), uint32(c.NExpert)
	route.MoEPermOff = g.aPerm
	add("route", "route", uint32(g.rows), 1, route)

	// 3-4. The counting sort, and the two tile schedules it implies. It runs
	//      twice because the two modes are cut to different row blocks and a
	//      tile list is only consistent with the permutation that outlives
	//      it — so both passes happen before either GEMM reads one.
	for _, t := range []struct {
		off uint32
		bm  int
		lbl string
	}{{g.aTilesUp, bmUp, "perm.up"}, {g.aTilesDown, bmDown, "perm.down"}} {
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
	add(fmt.Sprintf("up_%s_%s", w.gateFmt, g.up), "up",
		uint32(c.FFNExpert/moeBN), uint32(g.maxTiles(bmUp, g.rows)), up)

	// 6. down, weighted and scattered to its (token, slot).
	down := base
	down.MoEPermOff, down.MoETileOff = g.aPerm, g.aTilesDown
	down.BOff = w.down
	down.CtxOff = g.hSwiglu
	down.GemmN, down.GemmK = uint32(c.NEmbd), uint32(c.FFNExpert)
	add(fmt.Sprintf("down_%s_%s", w.downFmt, g.down), "down",
		uint32(c.NEmbd/moeBN), uint32(g.maxTiles(bmDown, g.rows)), down)

	// 7-8. The shared expert: the same two kernels over one group whose
	//      permutation is the identity, because it is the same shape as a
	//      routed expert and runs for every token.
	shUp := base
	shUp.MoEPermOff, shUp.MoETileOff = g.aPerm, g.aShTilesUp
	shUp.BOff, shUp.MoEBOff2 = w.shGate, w.shUp
	shUp.CtxOff = g.hSwiglu
	shUp.GemmN, shUp.GemmK = uint32(c.FFNShared), uint32(c.NEmbd)
	add(fmt.Sprintf("up_%s_%s", w.shGateFmt, g.up), "shexp.up",
		uint32(c.FFNShared/moeBN), uint32(roundUpInt(g.rows, g.pad())/bmUp), shUp)

	shDown := base
	shDown.MoEPermOff, shDown.MoETileOff = g.aPerm, g.aShTilesDown
	shDown.BOff = w.shDown
	shDown.CtxOff = g.hSwiglu
	shDown.GemmN, shDown.GemmK = uint32(c.NEmbd), uint32(c.FFNShared)
	add(fmt.Sprintf("down_%s_%s", w.shDownFmt, g.down), "shexp.down",
		uint32(c.NEmbd/moeBN), uint32(roundUpInt(g.rows, g.pad())/bmDown), shDown)

	// 9. `ffn_out`: the ten and the shared expert's, in slot order.
	add("combine", "combine", uint32(g.rows), 1, base)
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
	// (LLM.md L7c).
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

// Tiles reports how many (expert, row block) records each schedule holds
// against the static bound the grid covers, which is what says how much of
// the dispatch is empty.
func (g *MoEGPU) Tiles() (up, down, boundUp, boundDown int) {
	u := g.abuf.ReadUint32At(int(g.aTilesUp), 1)[0]
	d := g.abuf.ReadUint32At(int(g.aTilesDown), 1)[0]
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
	rowBytes := func(f moeFmt, k int) int {
		switch f {
		case fmtQ4K:
			return k / 256 * 144
		case fmtQ5K:
			return k / 256 * 176
		case fmtQ51:
			return k / 32 * 24
		}
		return k / 32 * 34
	}
	upPair = c.FFNExpert * (rowBytes(w.gateFmt, c.NEmbd) + rowBytes(w.upFmt, c.NEmbd))
	down = c.NEmbd * rowBytes(w.downFmt, c.FFNExpert)
	return upPair, down
}

// SharedBytes is the same for the always-on expert, which runs for every
// token and is the same shape as a routed one.
func (g *MoEGPU) SharedBytes(layer int) (upPair, down int) {
	c := g.cfg
	upPair = c.FFNShared * (c.NEmbd / 32 * 34) * 2
	down = c.NEmbd * (c.FFNShared / 32 * 34)
	return upPair, down
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
	g.rows = nTok
	if g.autoPlan {
		g.router = GEMMKernelFor(nTok)
		g.up, g.down = MoEPlanFor(nTok)
	}
	g.syncShared()
	return nil
}
