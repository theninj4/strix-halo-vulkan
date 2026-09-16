// Shared binding and push-constant layout for the qwen3.8-flash-next
// kernels (LLM.md L2).
//
// Same four-arena arrangement the DiT and the text encoder use
// (dit_common.glsl): one fp32 weight arena, one fp32 activation arena, one
// fp16 activation arena and one fp16 weight bank, with "which tensor"
// expressed as an offset so that a whole block is a dispatch sequence over
// one descriptor set. vk.DispatchMultiTimed records a sequence into a single
// command buffer only if every pipeline in it was built over the same buffers
// with the same push-constant size, so the block below is one size for every
// kernel in this vertical.
//
// It is a separate file from dit_common.glsl rather than a reuse of it
// because the fields are a contract with a *shader*, and a hyper-connection
// block has no heads, no head dim and no rotary table, while it does have a
// stream count, a low rank and three activation tensors that do not exist
// anywhere else. Naming one of those `span` or `aux1` is how a graph gets
// silently wrong (the reason zimage/qwen keeps its own copy of the DiT's).

layout(binding = 0) readonly buffer Weights { float wbuf[]; };
layout(binding = 1) buffer Act { float act[]; };
layout(binding = 2) buffer HAct { float16_t hact[]; };
layout(binding = 3) readonly buffer W16 { float16_t w16[]; };
// The fp32 activation arena again, read as uints. The QSA selection is a
// bitmask — one bit per cache cell per token — and a bitmask is not a float:
// punning it through `uintBitsToFloat` would put NaN payloads in a storage
// buffer and trust them to survive, where binding the same VkBuffer at a
// second descriptor costs nothing and says what the tensor is. Every pipeline
// in this vertical declares it so that a sequence can still be recorded into
// one command buffer; only llm_attn_select.comp writes it and only
// llm_attn_wmma.comp reads it.
layout(binding = 4) buffer ActU { uint actu[]; };
// The quantised expert bank, read as raw words. **Only the MoE block
// declares it** (`-DMOE_QBANK`), because every other kernel in this vertical
// is built over five buffers and a shader that names a binding its descriptor
// set does not have is a pipeline that will not create.
//
// It is `uint` and not `float16_t` for the reason L5b exists: one layer's
// three expert banks are 2.52 billion weights, 5.03 GB dequantised to halves
// and 75 GB across the model, so they stay in the checkpoint's own Q4_K, Q5_K,
// Q5_1 and Q8_0 blocks and the kernel unpacks its own slab per K-step. A
// tensor is addressed by its **byte** offset here, not its element offset,
// because a Q8_0 block is 34 bytes and nothing about these formats is
// four-aligned below the row.
#ifdef MOE_QBANK
// **It is an array of buffers, one a layer.** `maxStorageBufferRange` on this
// device is 4 GiB - 4 and the whole bank is 77 GB (LLM.md L6a), so residency
// is not a matter of one arena with a per-layer offset: a layer's 1.61 GB is
// a buffer of its own, all NBANK of them are bound at one binding, and the
// layer index selects. The index is dynamically uniform — one layer per
// dispatch — so this is core Vulkan and not descriptor indexing.
#ifndef NBANK
#define NBANK 1
#endif
layout(binding = 5) readonly buffer QBank { uint w[]; } qb[NBANK];
// The same buffer again, as sixteen-byte words. A Q4_K super-block is 144
// bytes and a Q5_K one 176, both multiples of sixteen, so the 32 bytes of
// nibbles a lane unpacks per K-step are two aligned `uvec4` loads where they
// would otherwise be eight `uint` ones — and this kernel turned out to be
// bound by how many load instructions its unpack issues, not by the bytes
// they fetch. Q5_1's 24-byte and Q8_0's 34-byte blocks are not aligned to
// sixteen and go on reading the `uint` view.
layout(binding = 6) readonly buffer QBank4 { uvec4 w[]; } qb4[NBANK];
// The kernel goes on spelling them `qbank` and `qbank4`: which buffer of the
// array is the same fact for every load a dispatch issues, so it belongs at
// the declaration rather than at the 30-odd call sites.
#define qbank  qb[MOE_BANK].w
#define qbank4 qb4[MOE_BANK].w
#endif

layout(push_constant) uniform PC {
    // The wide residual, fp32 in the activation arena: [T][hc*nEmbd], which
    // is ggml's [nEmbd, hc, T] read the same way round (ne[0] is fastest).
    uint resOff;
    // The normalised activation, fp16: [T][lda], stream c at column c*nEmbd.
    // Both matmuls of the block read it, which is the fusion.
    uint xnOff;
    // The low-rank gate input after silu(x/hc), fp16: [T][ldaLo].
    uint loOff;
    // The scatter weights, fp32: [T][gemmN - lowRank]. The down projection's
    // last tile writes them, so the row stride is the tile and not hc.
    uint injOff;
    // The block's fp32 output: `mixed` out of the mix, `res` out of the
    // combine, whose other input is the block output at outOff.
    uint outOff;
    // Per-stream RMSNorm gamma, fp32: [hc*nEmbd].
    uint gammaOff;
    // The packed fp16 weight, in the bank.
    uint bOff;
    // Validation only: where to write the 10240-wide gate the fused kernel
    // otherwise never materialises, or NO_W to leave it unwritten.
    uint gateOff;

    uint tokens;
    uint nEmbd;
    uint hc;
    uint lowRank;
    uint lda;       // xn row stride, halves
    uint ldaLo;     // lo row stride, halves
    uint gemmM;     // rows of A and C, padded up to the tile
    uint gemmN;     // columns of B and C
    uint gemmK;     // the reduction extent
    uint eps;       // float bits

    // The PLE block (llm_ple_gate.comp, llm_ple_conv.comp). It runs once, at
    // layer 1, so nothing here is on a hot path — but it reads the same wide
    // residual and writes back into it, so it belongs in the same contract
    // rather than in arenas of its own.
    uint kvOff;      // f32 key and value from one fused projection, [T][gemmN]
    uint gatedOff;   // f32 value * gate, broadcast over the streams
    uint normOff;    // f32 grouped-normed gated value: the convolution's input
    uint convOff;    // f32 depthwise weights, [hc*nEmbd][kern], channel-major
    uint convOutOff; // f32 silu(conv), or NO_W — the residual add does not need it
    uint gammaQOff;  // f32 query norm, over the residual
    uint gammaCOff;  // f32 conv norm, over the gated value
    uint kern;       // conv taps
    uint dil;        // conv dilation, which is the n-gram size

    // The full-attention layer and the QSA indexer (llm_attn_pack.comp,
    // llm_attn_idx.comp, llm_attn_score.comp, llm_attn_wmma.comp). Twelve of
    // the 48 layers, and the only ones with a KV cache.
    //
    // Every tensor here is named for what it is rather than folded onto a
    // spare field, for the reason the header gives: this block has four
    // separate norm gammas and five activation tensors that exist nowhere
    // else in the vertical, and a graph that reads the indexer's key norm
    // through a field called `gammaOff` is wrong in a way no tolerance
    // catches. 51 uints is 204 bytes against the device's 256.
    uint qkvOff;     // f32 [T][gemmN]: the one fused projection's output --
                     // query, gate, key, value, indexer query, indexer key
    uint qOff;       // fp16 packed query planes, fragment tiles
    uint kOff;       // fp16 packed key planes, same tiling as the query
    uint vOff;       // fp16 packed value planes, each tile transposed
    uint ctxOff;     // fp16 gated context: the output projection's A operand
    uint idxKOff;    // fp16 [nBlocks][idxDim], pooled, normed and rotated
    uint idxQOff;    // fp16 [T][idxHeads][idxDim]
    uint scoreOff;   // f32 [T][nBlocks], the rectified per-block score
    uint cellOff;    // f32 [T][nKV], biased, expanded and causally masked
    uint ropeOff;    // f32 rotary table: [nKV][rot/2] cos, then the same sins
    uint gammaKOff;  // f32 [headDim], the key's per-head norm
    uint gammaIQOff; // f32 [idxDim], the indexer query's
    uint gammaIKOff; // f32 [idxDim], the indexer key's
    uint heads;      // query heads
    uint kvHeads;    // key/value heads; heads/kvHeads share one cache head
    uint headDim;
    uint nKV;        // cache cells -- the whole context, not this batch
    uint plane;      // padded token rows per packed head plane
    uint ldCtx;      // the context's row stride in halves, gateWidth + pad
    uint idxHeads;
    uint idxDim;
    uint selOff;     // uint [T][(nKV+31)/32] cell bitmask, or NO_W for dense
    uint selWidth;   // top_k + ratio - 1, clamped to nKV: what the select asks for
    uint ratio;      // compress_ratio: cells pooled into one indexer block
    uint rotDims;    // n_rot -- 64 of the 256 head dims rotate
    uint attnScale;  // float bits: 1/sqrt(headDim) * log2(e), folded into q

    // **Continuing a sequence needs three more fields and the block has
    // none** (L7). 64 uints is 256 bytes and that is this device's whole
    // push-constant range, so the things a run that follows another knows
    // that a fresh one does not are said by fields their blocks already do
    // not use — the same arrangement `moeUsed` documents above, and the same
    // rule: the mapping lives here, and nothing spells the borrowed field's
    // own name.
    //
    //   lowRank   SEQ_PAST:    tokens of this sequence already behind the
    //             run. For the attention layer that is cells in the KV cache,
    //             so token t is cell SEQ_PAST + t at position SEQ_PAST + t;
    //             for the PLE block it is how far back its convolution may
    //             reach. Zero is a fresh sequence. Only the hyper-connection
    //             block uses `lowRank` as itself.
    //   injOff    SEQ_HIST:    the ring a convolution over the token axis
    //             reads behind itself, addressed by **position modulo its
    //             length**: f32 [(kern-1)*dil][hc*nEmbd] for the PLE block,
    //             f32 [kern-1][gemmN] per layer for the gated DeltaNet.
    //             `injOff` is the hyper-connection block's scatter weights
    //             and neither of those blocks has any.
    //   loOff     SEQ_SRC:     the tensor llm_seq_hist.comp copies those rows
    //             *out* of, for the one dispatch that writes the ring. It is
    //             the same field ATTN_IDXRAW is, because the two are never
    //             in the same push block: one is the attention layer's and
    //             the other belongs to the two convolutions.
    //   loOff     ATTN_IDXRAW: fp16 [nKV][idxDim], the indexer's *raw* key
    //             per cell. It is a cache because a pooled block spans
    //             `ratio` cells and at decode those arrive in `ratio`
    //             different batches; llm_attn_pack.comp writes this batch's
    //             cells and llm_attn_idx.comp pools out of it.
    //
    // kOff, vOff and idxKOff are unchanged as fields and changed as tensors:
    // they now address **one staged layer's** cache rather than a shared
    // arena, because a cache belongs to the layer that filled it. The key and
    // value planes are nKV cells rather than `plane` token rows; idxKOff is
    // the pooled table, and a block that is already complete is never
    // recomputed.

    // The gated DeltaNet (llm_dn_conv.comp, llm_dn_scan.comp,
    // llm_dn_norm.comp). Thirty-six of the 48 layers, and the only ones with
    // a recurrent state.
    //
    // Six fields, because the rest of this block already says what the layer
    // needs: `qkvOff` is the one fused projection again ([q | k | v | z |
    // alpha | beta], six of llama.cpp's four matrices), `gemmN` its row
    // stride, `convOff`/`kern` the depthwise taps as the PLE block states
    // them, `normOff` the convolution's normalised output, `outOff` the
    // recurrence's, `ctxOff` the fp16 A operand of the output projection,
    // `gammaOff` the shared head norm and `heads`/`kvHeads`/`headDim` the
    // 48/16/128 the whole layer is shaped around.
    uint ssmGateOff;  // f32 [T][heads]: the log decay, softplus(a)*A
    uint ssmBetaOff;  // f32 [T][heads]: sigmoid(beta)
    uint ssmStateOff; // f32 [heads][headDim][headDim], s[h][j][i] = S[i][j]
    uint ssmAOff;     // f32 [heads]: ssm_a, already -exp(A_log)
    uint ssmDTOff;    // f32 [heads]: ssm_dt.bias
    uint ssmNorm;     // 1 = ggml_l2_norm's max(|x|, eps), 0 = #28068's rsqrt

    // The MoE block (llm_moe_route.comp, llm_moe_perm.comp,
    // llm_moe_gemm.comp, llm_moe_combine.comp). Every one of the 48 layers
    // has one, 97% of the checkpoint's parameters are in them, and L2a put
    // them at 35.7% of the prefill graph.
    //
    // **Five fields, and the block is out of room**: 64 uints is 256 bytes,
    // which is this device's whole push-constant range. So the rest of what
    // the MoE needs is said by fields that already mean it, and the mapping
    // is written down here rather than inferred at four call sites:
    //
    //   xnOff/lda     the block's fp16 input, [T][lda] -- the A operand of
    //                 the router and of every routed gate/up tile
    //   qkvOff        f32 [T][gemmN]: the one fused projection's output
    //                 again, which here is the router's 512 logits with the
    //                 shared expert's one-column gate as a 513th
    //   ctxOff/ldCtx  fp16 [rows][ffn]: silu(gate)*up, which is exactly what
    //                 that field already is -- the A operand of the layer's
    //                 output projection
    //   gatedOff      f32 [T][used+1][nEmbd]: "value * gate", which here is
    //                 the down projection's output times its routing weight,
    //                 one row per (token, slot) and the last slot the shared
    //                 expert's
    //   outOff        f32 [T][nEmbd]: ffn_out, the block's output
    //   bOff          the **byte** offset of a dispatch's first quantised
    //                 matrix in the bank above -- `gate` for a swiglu tile,
    //                 `down` for a down tile
    //   gemmM         the row block the tile list is cut to, which the
    //                 permutation kernel has to know and every GEMM rung
    //                 compiles in
    //   gemmN         the bank's output width per expert, so that expert e's
    //                 output row n is bank row e*gemmN + n -- and, for the
    //                 route and permutation kernels, the expert count
    //   gemmK         the reduction extent, which is also what says how many
    //                 quantised blocks a row holds
    uint moePermOff;   // uint: topk[T][used], then the expert-major permutation
    uint moeTileOff;   // uint: the tile count, the per-expert counts, offsets
                       // and tile bases, then 3 uints per (expert, row block)
    uint moeWeightOff; // f32 [T][used+1]: the normalised routing weights, and
                       // sigmoid(shared_expert_gate) in the last slot

    uint moeBOff2;     // the byte offset of a swiglu tile's second matrix, `up`
    // Experts per token in the low sixteen bits -- the slot stride is that
    // plus one -- and the **bank index** in the high sixteen.
    //
    // The two ride in one uint because the block above is the last one there
    // was room for: 64 uints is 256 bytes and that is this device's whole
    // push-constant range. L6a's array of bank buffers needs a layer index at
    // every GEMM dispatch and there is no 65th field to put it in, so it goes
    // where there is room -- the used count is at most sixteen. MOE_USED and
    // MOE_BANK are how the kernels read them; nothing spells `pc.moeUsed`.
    uint moeUsed;
} pc;

#define MOE_USED (pc.moeUsed & 0xffffu)
#define MOE_BANK (pc.moeUsed >> 16u)

const uint NO_W = 0xffffffffu;

// The four fields a *continuing* run borrows (L7). See the notes in the push
// block: 64 uints is 256 bytes and there was no room for a 65th.
#define SEQ_PAST    pc.lowRank
#define SEQ_HIST    pc.injOff
#define SEQ_SRC     pc.loOff
#define ATTN_IDXRAW pc.loOff
