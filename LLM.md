# LLM — the qwen3.8-flash-next vertical

> **Current work from 2026-09-16.** `SPEECH.md` is finished-ish (74.2x real
> time, T7 open); `PIPELINE.md` (z-image) is parked at 14.26 s an image. Same
> rules as both: this file is **rewritten** each session rather than appended
> to, history goes to `TODO.md`, closed findings to `research/`. Stage numbers
> are **L0, L1, …**; `§N.M` still addresses `IDEAS.md`.

**Target**: `Qwen/Qwen3.8-Flash-Next` — 180 B params, 6 B active, "a preview of
the Qwen4 architecture" — generating text end to end in Go on Vulkan. The
third vertical, and 200x the parameters of the other two put together.

**Status: L0, L1, the whole of L2, L3, L4 and L5, and now L6a — every block of
this model runs on the GPU, and all 48 layers of all of them are resident at
once.** The checkpoint is downloaded, llama.cpp runs
it, the Go side reads it, the prefill mystery is solved, and there is a kernel
for every layer and every sub-layer: the hyper-connection block in four
dispatches where the reference has sixteen (4.24x, 14.9% of llama.cpp's whole
prefill graph), the PLE n-gram block beside it with its trigram hash
bit-exact, the full-attention layer with its QSA indexer **and** its selection
in seven dispatches where the reference has twenty-eight (L2f and L4b, 2.32x),
the gated DeltaNet — three quarters of the layers — in five where the
reference has eleven (L3b, 1.37x, the recurrence itself at 1.06x), and now
**L5b: the MoE block, 35.7% of the graph and 97% of the parameters, in nine
dispatches against about 845**.

**Between them the five blocks have taken 1015.5 ms of llama.cpp's 1164.7 ms
prefill graph down to 714.8** — 87.2% of it replaced by 61.4%, a 25.8% saving
on the whole graph. 114 GB in 18 minutes;
`models/Qwen3.8-Flash-Next-GGUF/` holds the four `UD-Q4_K_XL` shards and the
2.79 GB MTP head.

**L5b is the first kernel here whose weights never become floats.** One
layer's three expert banks are 2.52 billion weights — 5.03 GB as halves and
241 GB across the model, against 1.57 GB and 75 GB as they ship — so the bank
is staged byte for byte out of the mmap'd checkpoint and each workgroup
unpacks its own Q4_K/Q5_K/Q5_1/Q8_0 slab into LDS per K-step. The schedule is
L5a-4's routing made concrete: a device-built *list* of (expert, row block)
records over a static bound, with every expert's rows padded up to the block
so a cooperative-matrix store can go straight to global. **567.4 ms of
llama.cpp's graph becomes 522.3 — 1.09x at its own best ubatch — and 1758.4
becomes 1131.6 at ubatch 2048, 1.55x**, with `ffn_out` at 8.132e-05 rms over
all 4096 tokens and the top-10-of-512 selection the reference's set for set
*and order for order*. Along the way the router turned out to be **bit-exact**
against llama.cpp on all 2 097 152 logits, and so did `shared_expert_gate` —
which corrects L5a-1: fusing that gate onto the router's matrix is not a
deviation but a reproduction.

**L6a has since put the whole model on the device: 84.20 GB of weights and
0.87 GB of arenas in 68 buffers, staged in 33 seconds, with 41 GB of this
machine left over.** That needed one structural change and it was not in a
kernel. Every block here lays its weights out as one arena with a per-layer
offset, `maxStorageBufferRange` on this device is **4 GiB - 4**, and a
layer's expert bank is 1.61 GB — so binding 5 became an **array of buffers,
one a layer**, indexed by a push constant that had to ride in the top sixteen
bits of an existing field because the block was already 64 uints against a
256-byte limit. **Residency turns out to be free in the bank's size**: 2, 4,
12, 24 and 48 banks span 10.4 to 84.2 GB resident and 11062-11242 us for the
same block, a 1.6% spread with no trend, which is L0a holding at the size the
model actually is. **Next: L6b, the graph that runs the five blocks in order,
and logits against llama.cpp.**

## The number to beat

`llama-bench`, build `cff184438`, Vulkan on RADV STRIX_HALO, the model as
shipped, `-ngl -1`, mmap on, two repetitions, nothing else on the GPU:

| test | tok/s | against |
|---|---:|---|
| pp512 | 313.62 ± 1.74 | |
| **pp2048** | **388.60 ± 1.89** | L2a re-measures **391.42 ± 1.13** at its best ubatch, and attributes the 5x |
| pp8192 | 392.95 ± 1.04 | prefill plateaus, it does not scale with the chunk |
| tg32 | 25.12 ± 0.03 | |
| **tg128** | **25.15 ± 0.02** | **65.8%** of the 242 GB/s bus |

A real generation agrees: `llama-completion -n 64 --temp 0` reports 24.66
tok/s and coherent text. Reproducibility across two separate runs is 0.5% on
pp512 and 0.8% on tg128. And the accuracy reference, on the same checkpoint:
**PPL = 4.0340 ± 0.02283** over wikitext-2's 145 chunks at n_ctx 2048 — the
figure L8's re-quantisation has to stay near, at the same corpus, context and
chunking. [Write-up](research/l1-baseline.md)

**So the target moves.** Phase 1's job is the 38.2 tok/s the bytes allow, not
llama.cpp's 25.15 — decode leaves a third of the bus unused in the reference
implementation. Phase 2's ~67 is then **2.7x the reference**, not 1.8x.

**L2a has since re-measured the prefill row**: `-ub 512` is llama.cpp's best
ubatch and it does **391.42 ± 1.13**, 0.7% from L1's separate 388.60 ± 1.89.
256/1024/2048 give 347.71 / 380.91 / 336.72 — bigger is *worse*, for a reason
finding 5 below explains.

## What L0 established

> **L0a: the DRAM bus does not care how big the weight bank is.** 236.4 GB/s
> reading a **64 GiB** bank at random against 237.1 GB/s reading a 1 GiB one,
> `rand/seq` 1.00x in every cell at both of the model's real expert slab
> sizes. That *is* §0.4's 236 GB/s ceiling, not merely near it — no TLB cliff,
> no page-table cost, no penalty for scattering. So decode stays linear in
> bits/weight at the size a 180 B model needs, which is what every number
> below depends on. At `-bankgib 80`: 23 buffers, 85.9 GB, every page written,
> **237 GB/s at every prefix** — past UD-Q4_K_XL's 82.52 GB resident core.
> [Write-up](research/l0a-bank-range.md)
>
> **L0b: neither does the memory type, and the heap sizes are fiction.** All
> eight types that can back a storage buffer read **236.0-237.4 GB/s** — 0.57%
> across 32 cells, heap 0 / heap 1 = **1.0004**. Type 3 reserved **105.0 GiB**
> against a heap RADV calls 83.79. **The ceiling is physical RAM.**
> [Write-up](research/5.1-memory-types.md) · `results/bank.csv`
>
> **L0c: §2.2's unexplained 1.24x was locality in the scale plane, and it is
> gone.** A k-major scale plane takes `gate_up` at QBLOCK=32 from **4.71 ms to
> 3.72, 1.27x**, past row-major QBLOCK=128's 3.79. **The fine scale block
> accuracy wants now costs 1.0% instead of 14.2%.**
> [Write-up](research/l0c-scale-plane.md) · `results/moe.csv`
>
> **L0d: W4A8 is safe, and above ~5 bits/weight the activations are the
> floor.** int8-per-token activations cost **1.13x** the error of fp16 ones,
> so the 3x cliff between W4A8 and W4A16 is bought cheaply — but int8
> activations *alone* contribute 2.87e-2 where an 8-bit weight contributes
> 3.4e-3, so **W8A8 is 99% activation error**. Asymmetric Q4 is worth 5% at
> equal bits, not 2x. And a bug fell out: `quantizeQ8`'s fp16 block scale goes
> subnormal under maxAbs 7.75e-3, worth **14x** on one real tensor in fourteen.
> [Write-up](research/l0d-quant-error.md)

## What L2a established — the 5x, attributed

`GGML_VK_PERF_LOGGER=1` puts a timestamp query around every dispatch in
llama.cpp's Vulkan graph. Both of L1's candidates are **wrong**, and the
answer is better than either. [Write-up](research/l2a-prefill-attribution.md)
· `results/l2a_prefill_ops.csv`

> **L2a-1: the architecture is innocent.** `GATED_DELTA_NET` is **1.6%** of a
> prefill graph — 36 dispatches, 74.1 ms of 4556. The whole DeltaNet core is
> 2.7%, full attention plus the QSA indexer and its top-2048 selection 1.5%.
> **The candidate "this architecture has a great deal of prefill that is not
> matmul" accounts for 4.5% and cannot explain a 5x.** Nor is it host
> overhead: **93% of the un-instrumented wall clock is GPU kernel time**.
>
> **L2a-2: it is the hyper-connection block, and half of that is free.**
> Per graph: MoE experts **35.7%**, dense matmul **26.2%**, elementwise glue
> **30.6%** (2453 dispatches), tiny-N F32 matmul **12.2%** at `-ub 512`. The
> worst single line is `MUL_MAT f32 m=4 n=512 k=10240` — `hc_*_inject`, a
> `[10240, 4]` F32 projection run 95 times a graph — at **31.8 GFLOP/s, 0.06%
> of peak, 10.3% of prefill**. It reads the same `xn` that `hc.down` reads, so
> it is four more output columns on a matrix that already has 320: **fusing it
> deletes a tenth of prefill.** `CONCAT` is the other scandal, at 48 GB/s
> falling to **13 GB/s** as the ubatch grows.
>
> **L2a-3: against our own measured kernels the chunk is 2.25-2.53x.**
> `results/shapes.csv`'s best variant per shape beats llama.cpp by **5.08x at
> `hc.down`** (N=320), 2.96x at `moe.shared` (N=640), 2.15x at `attn.o`, and
> only 1.11-1.15x on the two widest-N shapes — **we win exactly where N is
> small.** §2.2's already-built Q4 grouped MoE block is **642 ms against 1626,
> 2.53x**. Total: **711 tok/s with the glue as it stands, ~1150 with three
> quarters of it fused into epilogues.** So **L6 is an afternoon, and ~1150
> — not ~2000 — is what it should be scoped against.**
>
> **L2a-4: `-ub 512` is a MALL-sized optimum, so do not raise it.** 256 / 512
> / 1024 / 2048 give 347.71 / **391.42** / 380.91 / 336.72. Per token the MoE
> does get cheaper with a bigger ubatch (0.78x) — but the glue gets **1.57x**
> more expensive, because the `[10240, T]` **F32** residual is 20.97 MB at
> T=512, **just inside the 32 MiB MALL**, and 83.9 MB at T=2048. `MUL` falls
> from 362 GB/s to 294, `CONCAT` from 48 to 13. **Keep the residual fp16 and
> fuse the traffic away rather than cache it.**
>
> **L2a-5: decode's gap is dispatches, not GEMV.** llama.cpp's decode matmuls
> are already at **198-230 GB/s** (`lm_head` 675 MB in 2.93 ms = 230). The
> 6.334 GB a token needs 26.2 ms and it spends 41.3; the missing 15.1 ms is
> **2896 dispatches that read no weights, costing 8.9 ms — 22% of the step**,
> an order of magnitude past L1's 0.62 ms estimate for the same graph.
> **L7 is won by not dispatching 3837 kernels, not by a better GEMV.**

## What L2b established — the block runs, and the oracle is not exact

`llm/` is the vertical's package: a `Config` read from the checkpoint, the
hyper-connection block, and `reference/eval_dump.c`, which writes **whole**
tensors of a real llama.cpp pass where `llama-eval-callback` prints three per
axis. [Write-up](research/l2b-hyper-connections.md) · `reference/out/llm/`

> **L2b-1: the block matches, and four of its seven mixers match to f32
> round-off.** `token_embd`'s gather and `hc_init` are **bit-exact**; the
> combine is 1e-09 rms; the F32-weighted `hc_norm` and `hc_inject` are 1e-07
> and **3e-06 relative** on values to \|71.7\|; and `hc_gate` — through two
> Q8_0 matmuls, a SiLU and a sigmoid — is **6.2e-08 rms in four of the seven
> mixers of the four dumped layers**. Nothing wrong about the formula, the
> weight indexing or the layout survives that.
>
> **L2b-2: llama.cpp evaluates a Q8_0 matmul over int8 activations, and
> modelling it is worth 233x.** `quantize_q8_1.comp`: blocks of 32, `d =
> amax/127`, `round(x * 1/d)` **with the f32 reciprocal**, and the scale
> *stored* fp16. `hc_gate-0` goes from **3.143e-03 rms to 1.347e-05**.
> Rounding the activations, the weights or both to fp16 changes **nothing**
> (3.14e-03), so this is not a precision effect — it is a different
> arithmetic. Quantising with an already-fp16 scale, the obvious misreading,
> is 35x worse than the real thing.
>
> **L2b-3: so L2's gate changes.** "Matches `llama-eval-callback` to fp16
> tolerance" is unreachable *and* meaningless downstream of a Q8_0 matmul.
> The criterion is an **rms bound under the reference's own numerics**
> (`llm.Numerics`, `Exact` vs `RefQ8`), and maxAbs is a logged diagnostic
> rather than a bound, because an int8 grid makes the error heavy-tailed:
> one flipped rounding among `lo`'s 320 values moves all 10240 gate outputs.
> Three of the seven mixers keep 6e-06 to 3.4e-04 rms for that reason,
> bounded but not explained.
>
> **L2b-4: and D6 is cheaper than it looked.** The baseline **PPL 4.0340**
> was measured with **W8A8 already running on the dense tensors**. L0d priced
> int8-per-token activations at 1.13x fp16's error and called W8A8 "99%
> activation error"; that is not a projection about our bank, it is what the
> reference implementation does. **D3 and D6 give up no activation axis the
> reference keeps**, and L8's perplexity comparison is like-for-like on it.

## What L2c established — the block runs on the GPU, and it is 4.24x

`shaders/llm_hc_norm.comp`, `llm_gemm.comp` (two of its three epilogues) and
`llm_hc_combine.comp`, driven by `llm/gpu.go`. The first kernel of the
vertical, and the one L2a put at the front of the queue.
[Write-up](research/l2c-hc-kernel.md) · `results/l2c_hc.csv`

> **L2c-1: four dispatches against sixteen, and 226.6 ms becomes 53.4.** Per
> 512-token graph, against the lines of llama.cpp's own graph that are nameably
> this block: the grouped norm 16.7 → **5.3 ms**, `down` + `inject` 163.3 →
> **24.7**, `up` 30.7 → **16.8**, the combine's `REPEAT` 15.9 → **6.6**.
> **4.24x, and 14.9% of the whole 1164.7 ms graph removed by one block** — and
> the rest of the block's glue (the gamma multiply, the gate's sigmoid, the
> `xn*gate` multiply, the collapse's adds, the combine's four elementwise
> passes) is **absent from our graph rather than faster in it**, so that is the
> conservative reading. If nothing else changed, the reference's 391.4 tok/s
> would become **~460**.
>
> **L2c-2: two tensors never exist, and that is where the time goes.**
> `inject` is four more output columns on a `[336, 10240]` fused weight — L2a's
> worst single line, 10.3% of prefill at 31.8 GFLOP/s, deleted. And the
> `[10240, T]` gate — 21 MB a mixer at 512 tokens, the whole MALL — is consumed
> inside the up projection's epilogue, which applies the sigmoid to its
> accumulators and collapses the four streams against `xn` through LDS. That
> needs the four streams of a feature in one workgroup, which is a **weight
> permutation** (`packUpB`), and `TestHCGPUUnpermutedUpIsWrong` is the negative
> control: unpermuted is a plausible tensor built from the wrong four columns.
>
> **L2c-3: it is 38x nearer the f32 model than the oracle is.** All eight
> mixers of layers 0-3, each from llama.cpp's own residual, agree with L2b's
> CPU reference to **1e-04 rms** — `maxRel` on `hc_norm` is 4.9e-04, which *is*
> fp16's step. Against llama.cpp the residual is 2.3e-03 to 6.5e-03, which is
> **L2b's figure for the CPU reference**, not ours: the reference is the side
> computing in int8. Every rung of both ladders is bit-identical to every
> other, which is what checks the M padding at a 7-token prompt.
>
> **L2c-4: the MALL cliff is ours too, and fp16 residual is now worth 1.5x.**
> Between 512 and 1024 tokens `norm` goes **575 → 167 GB/s** and `combine`
> **695 → 170** — above the DRAM bus on the near side, below it on the far —
> because the fp32 residual is 21.0 MB at T=512 and 41.9 MB at T=1024 against a
> 32 MiB MALL. Per token the block is cheapest at 512 and **1.5x** more
> expensive at 2048. L2a inferred "keep the residual in fp16" from llama.cpp's
> profile; this measures it on our own kernel. Not taken here — the residual
> accumulates across 97 combines, so it is an accuracy decision for L6/L8.

## What L2d established — the n-gram block, and a hash that has no tolerance

`per_layer_token_embd` is a quarter of the checkpoint and it is not a matrix.
[Write-up](research/l2d-ple.md) · `results/l2d_ple.csv`

> **L2d-1: the trigram hash is bit-exact.** Sixteen rows of a **320 001 536-row**
> table per token, chosen by a 64-bit mix of the token and its two
> predecessors against sixteen near-prime vocabularies whose offsets tile the
> table exactly. `ple_embd` matches llama.cpp to **0 ulp** — which is the only
> tolerance a lookup has, since a wrong index returns a real embedding
> belonging to another n-gram. The block on top of it matches to **7.7e-08
> rms** (`ple_gate-1`), 2.1e-09 (`gated`) and 1.3e-08 (`conv_out`).
>
> **L2d-2: it closes L2c's one open comparison.** Layer 1's attn mixer reads a
> residual this block has written into, which is why it could not be checked
> against the trace before. Fed through the block it lands where the other
> seven do — 2.4e-04 on `hc_norm`, 2.9e-03 on the gate — so **the two stages
> compose**.
>
> **L2d-3: three dispatches against about thirty, and it is 0.19% of a graph.**
> The key and value projections read the same gathered embedding, so they are
> one fused `[12800, 2560]` matrix; everything between them and the convolution
> is one workgroup per (token, stream); and the dilated depthwise convolution
> carries its SiLU and the residual add. On the nameable lines we are **0.88x
> — slower** (2234 us against 1974), and the projections alone are **1.06x
> while reading twice the weight bytes**, because llama.cpp's are Q8_0 and ours
> are fp16. This block runs **once**, so none of that matters; it was built to
> be correct and to be on the device.
>
> **L2d-4: two levers, and both were free.** The fused projection's B is
> **65.5 MB, twice the MALL**, so it is read M/BM times: BM 32 → 128 is
> **3.3x** at 2048 tokens for identical arithmetic. And making the *channel*
> the fast grid axis instead of the token — so the resident workgroups cover
> one token's 40 KB row rather than one channel block of forty tokens — is
> **2.9-4.0x** on the convolution and 1.14-1.26x on the gate. Same work, same
> bytes; §5.1b's traversal penalty in a kernel written without thinking about
> it.

## What L2e established — the attention layer, and two more oracle numerics

`llm/attn.go`: the fused query/gate projection, interleaved M-RoPE, the QSA
indexer and causal GQA, checked against layer 3 of the dump.
[Write-up](research/l2e-attention.md)

> **L2e-1: it matches, and the chain composes.** The indexer's pooled keys to
> 1.7e-07 rms, its score to **3.5e-07 relative**, the fused query/gate split to
> 1.2e-06, the attention output to 2.4e-04. And from `l_last-2` through our
> mixer, our indexer, our attention and our combine, **`hc_combine-3` agrees
> to 9.75e-06 rms** — four stages, none of the intermediates llama.cpp's.
>
> **L2e-2: the indexer's key cache is fp16, worth 2072x.** The keys are
> rounded to halves *between* being projected and being pooled, which is why
> `indexer_k_raw` is exact and `indexer_k_pooled` three lines later was
> 3.49e-04 until it was modelled. `llama_memory_hybrid_idx` passes the
> context's `type_k` straight through.
>
> **L2e-3: a matmul between two F32 tensors is an fp16 matmul, worth 69x.**
> The indexer's score is F32 x F32 and the reference computes it on the fp16
> matrix cores: 3.52e-03 rms in f32, **the same 3.52e-03 in float64**, and
> 5.11e-05 with both operands rounded to halves. **So the fp16 kernels L2c and
> L2d run are not giving up anything the reference keeps** — they are the same
> arithmetic.
>
> **L2e-4: the block structure is the cache's, not the prompt's.** Seven tokens
> in a 256-cell cache (flash attention pads it) make **one** real block of 4;
> the other 63 pool cell 0 four times because `blk_cells` is zero-filled, and
> the unpooled tail goes to a spare block carrying a 1e9 "always visible"
> marker — finite, so it can never meet a -inf and make a NaN.
>
> **L2e-5: and the selection cannot be tested yet.** `top_k + ratio - 1` is
> 2051 against 256 cells, so it names every cell; running the attention with
> it is bit-identical to running it without. llama.cpp's `TOP_K` is also a
> *selection and not a sort* — it returns the identity permutation where the
> scores are not monotone — which is unobservable at this width. **L4's 4k
> context is where both become testable.**

## What L2f established — the layer on the device, and a ladder that inverts

`llm/gpu_attn.go`, `shaders/llm_attn_{pack,idx,score,wmma}.comp`: six
dispatches, the plain GEMM arm twice and four kernels of this layer's own.
[Write-up](research/l2f-attention-gpu.md) · `results/l2f_attn.csv`

> **L2f-1: six dispatches against nineteen, and 66.3 ms becomes 27.6.** Per
> 512-token graph, against the lines of llama.cpp's own graph that are nameably
> this layer: the five projections 28.2 → **18.4 ms**, the norms and ropes 3.7
> → **0.8**, the indexer's score 0.7 → **0.8**, flash attention 25.2 → **2.1**,
> the output projection 8.6 → **5.6**. **2.40x, and 5.7% of the whole graph
> becoming 2.4%.** The selection llama.cpp also runs — `TOP_K` and its
> `GET_ROWS`, 24 dispatches and 2.4 ms — is left out of the comparison
> entirely, and so is every elementwise line this block shares with the others.
> **L4b puts the selection back in on both sides**: one dispatch of ours
> against those 24, and the layer 68.8 ms against 29.6, **2.32x**.
>
> **L2f-2: one of those rows is not like for like, and saying so needs a
> number.** llama.cpp's flash kernel issues the whole **[512, 2048]** rectangle
> its cache defines — ggml's own FLOP count says so — where ours walks the
> causal triangle of a 512-token prefill, an eighth of the work. On **rate**
> the two kernels are **18.9 TFLOP/s against 12.3, 1.54x**, and repricing that
> row at our shape makes the layer **1.60x** rather than 2.40x. Both are real:
> 2.40x is what a 2048-token prefill in four chunks costs either side, 1.60x is
> what the kernels are worth at equal work.
>
> **L2f-3: six of the reference's matrices are one.** The query, its gate, the
> key, the value and both of the indexer's projections read the same normalised
> residual, so they are one `[13952, 2560]` weight — L2a's argument for
> `inject` and L2d's for the PLE value, at the largest scale in the model. Then
> the per-head norm, the interleaved M-RoPE and the fragment tiling for q, k
> and v are **one dispatch over a (token tile, plane) grid**, because all three
> are column ranges of that output ending in the same layout. And **the output
> gate folds into the attention epilogue**, which already holds the tile for
> the softmax divide — a `SIGMOID`, a `MUL`, a `CONT` and a `[512, 6144]` round
> trip through DRAM, deleted.
>
> **L2f-4: the two projections want opposite schedules on the same kernel.**
> The fused projection's B is **71.4 MB, 2.2x the MALL**, so it is read M/BM
> times and the widest row block wins everywhere above 64 tokens, by 3.2x at
> 2048. The output projection's B is **31.5 MB and fits**, and on that side of
> the line reuse buys nothing while occupancy does: at 64 tokens BM=128 is the
> **worst** rung by 2.1x, at 512 BM=64 wins, and only at 1024 does BM=128 win
> back. **Same kernel, same arithmetic, inverted schedule, decided by which
> side of one cache the weight falls on** — so there are two schedules, and the
> pair is the exact winner of a 27-plan ladder at five of six lengths.
>
> **L2f-5: and the attention ladder ends one rung narrower than every other in
> the repo.** qt1_kt2 **170 us**, qt1_kt4 223, qt2_kt4 241 — monotonic in how
> much a wave holds live. §3.3's conclusion is unchanged (intensity is inert,
> the register file decides); headDim **256 against the DiT's 128** just makes
> it decide sooner, a wave already carrying 16 query fragments and 16 output
> accumulators across the key loop.
>
> **L2f-6: 5.9e-05 rms, and 19.3x nearer the f32 model than the oracle.** Every
> tensor agrees — the six column ranges to 1.7e-04, the pooled indexer key to
> 3.1e-04, the score to **3.8e-07 relative**, the cell mask **exactly** — and
> the layer's output sits at 5.9e-05 against L2e's f32 reference where
> llama.cpp sits at 1.14e-03. That gap is the reference's int8 activations
> under four Q8_0 projections, not ours. One numeric is now **reproduced rather
> than modelled**: the indexer's fp16 key cache is fp16 here because the arena
> is.

## What L3a established — every layer now has a reference, and the oracle has a bug

`llm/deltanet.go`: the fused qkv projection, the depthwise causal convolution,
the L2 normalisation, the two F32 gate projections and the delta rule itself,
against all eighteen tensors llama.cpp names inside layer 0.
[Write-up](research/l3-deltanet.md)

> **L3a-1: it matches, and the recurrence is at f32 round-off.**
> `attn_output-0` is **3.11e-09 rms**, the 786 432-value `new_state-0`
> **1.91e-08**, and `linear_attn_out-0` **1.31e-07** — with `state_predelta-0`
> checked to be exactly zero, so those are the recurrence's own numbers and
> not the fixture's. Layers 1 and 2, whose input the PLE and hyper-connection
> blocks have already written, agree at 6e-10 on the output; their
> `linear_attn_out` sits at 1.2-1.5e-05 and all of that gap appears at the
> last Q8_0 matmul, which is L2b-3's heavy-tailed int8 grid over a wider input.
>
> **L3a-2: the reference does not chunk, so neither does the gate.**
> `cparams.fused_gdn_ch` sends a multi-token batch to `ggml_gated_delta_net`,
> whose Vulkan kernel walks the tokens **one at a time**, one workgroup per
> head, with the state in registers across the loop.
> `build_delta_net_chunking` — 270 lines of `solve_tri` and cumulative-sum
> decay masks — is in the tree and is not what ran; `q_conv_predelta` being
> dumped with **16 heads and not 48** proves it, because the non-fused path
> repeats q and k first. So the chunked form is L3's *performance* question,
> and what it has to beat is **438 us a layer at ubatch 512** (2058 at 2048).
> *(L3b corrects the shape: the dispatch is `{H, n_seqs, S_v}` with one state
> column per workgroup, which on a 64-wide wave is **6144** workgroups and not
> 48 — the source's "one workgroup per head" is the loop's structure, not the
> grid's.)* **vLLM answers it the other way** — `chunk_gated_delta_rule` at
> chunk size 64 for prefill, a fused recurrent update only for decode — so L3b
> is a real open question that two teams have split on, and
> flash-linear-attention's four stages (`scaled_dot_kkt` → `solve_tril` →
> `delta_h` → `fwd_o`) are its blueprint. **L3b answers it: neither team's
> choice is worth much here, because the recurrence is 13% of the layer.**
>
> **L3a-3: the 48 value heads read the 16 key heads by `h % 16` — because
> this checkpoint's V heads are permuted.** ggml's own two broadcast rules
> disagree here (`ggml_repeat` tiles, `ggml_mul_mat`'s batch broadcast
> divides), they agree on 16 of the 48 heads, and the fused op tiles:
> **3.11e-09 against 2.71e-03, 870 000x.** **vLLM divides** — `i_h // (H //
> Hg)` in flash-linear-attention — and both are right, because llama.cpp's
> converter (`_LinearAttentionVReorderBase`) transposes the (16, 3) head grid
> into tiled order across **seven tensor families** so that ggml's cheap
> broadcast lands on the right head. So `h % 16` is a fact about the **GGUF**,
> not the architecture — a live hazard for L8, below. The state is also
> **stored transposed** (ne[0] is the key axis), which is not a quirk: it
> makes all three of a token's operations walk contiguous columns, and that is
> what lets the state stay in registers.
>
> **L3a-4: the oracle's build has a GDN normalisation bug, and it is worth
> 15x.** `build_gdn_l2_norm` divided by `max(|x|, eps)` until llama.cpp commit
> 5fdfa6282, *"models : fix GDN normalization from `max` to `rsqrt`"*
> (#28068), and this repo's build — **cff184438** — predates it. That build is
> also L1's `pp2048 388.60`, `tg128 25.15` and **`PPL 4.0340`**. Reproducing
> the bug takes `k_conv_predelta` from 9.16e-07 to **6.01e-08** and the layer
> output from 1.91e-06 to 1.31e-07, so `DeltaNetConfig.QKNorm` selects the
> spelling, the model's `rsqrt` is the default and the tests set `max`.
> **vLLM independently computes `1/sqrt(sum + 1e-6)`**, so #28068 is a fix
> rather than a change of convention. This is the first time the oracle has
> been wrong about the **model** rather than merely imprecise about the
> arithmetic.
>
> **L3a-5: L2e-3's fp16 F32 matmul is a dispatch threshold at 8 columns, and
> it changes what every tolerance in L2 means.** `ssm_alpha` and `ssm_beta`
> are F32 weights read by an F32 activation — L2e-3's situation exactly — and
> modelling them in fp16 is **86x worse** (2.15e-06 against 1.84e-04).
> `ggml_vk_mul_mat` sends a matmul to the f32 **vector** path whenever the
> output has at most `mul_mat_vec_max_cols = 8` columns and to the fp16 coopmat
> GEMM otherwise; the dump's prompt is **7 tokens**, and the indexer's score
> has 4x7 = 28 columns. One line explains both. **So at a real 512-token
> ubatch every projection in the model crosses to the fp16 path** — a tolerance
> measured against a 7-token dump is not a tolerance at 512, and that now
> applies to L2b, L2d, L2e and L3 alike.
>
> **L3a-6: and two things cross a batch boundary, not one.** 3 + 4 tokens
> reproduce 7 **bit-identically**, output and state both — but only because
> `DeltaNetState` carries the convolution's three-column window beside the
> [128, 128, 48] state. A kernel that carries the state and forgets the window
> is wrong in a way nothing but that test sees, and at L7 it would be wrong on
> every token. Two smaller things fell out: ggml's `softplus` adds in f32 and
> so flushes to **exactly zero** below -16.6, where `math.Log1p` is more
> accurate and wrong; and nine of the trace's tensors are **non-contiguous
> views** that `eval_dump.c` writes as the parent region they span, so
> `ReadDump` now gathers with the recorded strides (`q_conv-0`: 1.61e-01 rms
> read flat, 1.04e-07 read properly).

## What L3b established — the layer on the device, and a ladder that is 6.3x

`llm/gpu_deltanet.go`, `shaders/llm_dn_{conv,scan,norm}.comp`: five dispatches,
the plain GEMM arm twice and three kernels of this layer's own.
[Write-up](research/l3b-deltanet-gpu.md) · `results/l3b_dn.csv`

> **L3b-1: five dispatches against eleven, and 152.8 ms becomes 111.5.** Per
> 512-token graph, against the lines of llama.cpp's own graph that are nameably
> this layer: the four projections 91.3 → **71.3 ms**, the convolution with its
> SiLU, both L2 norms and the softplus 11.4 → **7.0**, the recurrence 15.8 →
> **14.9**, the gated norm 8.6 → **2.2**, the output projection 25.7 → **16.1**.
> **1.37x, and 13.1% of the whole graph becoming 9.6%** — a second run gives
> 113.2 ms and 1.35x. The SiLU, the sigmoid of z, the multiply against it and
> the CONTs behind all three are left out of the comparison entirely — they live in the shared MUL/SIGMOID/CONT lines and
> are *absent* from our graph rather than faster in it.
>
> **L3b-2: four of the reference's matrices are one, and two of them are the
> F32 scandal in miniature.** `attn_qkv`, `attn_gate` and the two **F32**
> [2560, 48] gate projections read the same block input, so they are one
> [16512, 2560] weight — L2a's argument for `inject` at a fourth site. The
> F32 pair is 72 dispatches a graph, **7.8 ms at 1155 GFLOP/s for 0.25 MB of
> weights**, and here it is 96 more columns on a matrix that already has
> 16384. That puts alpha and beta on the fp16 matrix cores, which is a
> deviation **from the 7-token dump and not from the reference**: L3a-5's
> 8-column threshold means llama.cpp does the same at any real ubatch, and
> **L1's PPL 4.0340 was measured with it**. Priced rather than hidden: 1.23e-03
> rms on the log decay, and at most **1.105%** on the per-token forgetting
> factor.
>
> **L3b-3: the scan's ladder is a re-read-to-ALU crossover, and its ends are
> 6.3x apart.** LPC — how many lanes own one column of the [128, 128] state —
> sets the registers a lane holds (128/LPC), the workgroups the layer
> dispatches, and how many times each head's q and k row is re-read per token,
> **all at once**. llama.cpp's own kernel is the top rung: `{H, n_seqs, S_v}`
> with one column a workgroup is **6144 workgroups and a 128-fold re-read** on
> a 64-wide wave — not the "48 workgroups" L3a inferred from the source — and
> ours at that shape costs 2766 us. Five rungs sit pinned at **1.18-1.31 TB/s
> while their flop rate varies 4x**, and the ablation is direct: **deleting the
> operand loads takes the widest rung from 2766 us to 1013**, where deleting
> both subgroup reductions moves it 5%. The winner is 16 state elements a lane
> and 768 workgroups — **413 us against the reference's 438, 1.06x**, and 1.16x
> at ubatch 2048.
>
> **L3b-4: registers beat LDS for the operands until they do not, and the
> turn is at the same rung.** Keeping q and k beside the state is 1.15x at
> LPC 16 and 1.08x at LPC 8, and **loses by 1.33x at LPC 4**, where 96
> registers a lane costs more occupancy than an LDS round trip costs latency.
> Two things that should have helped and did not: sharing the staged operands
> across four waves of a workgroup cuts the re-read 4x and ties, and a one-deep
> software pipeline is worth nothing at the wide end and 1.4x *against* at the
> winner.
>
> **L3b-5: so the chunk is priced rather than built, and the question was
> mis-scoped.** The recurrence is **13% of this layer**; the fused projection
> is **64%**. flash-linear-attention's four stages come to 3.70 GFLOP at 512
> tokens against the scan's 3.22 — **1.15x the arithmetic, all of it matmul
> shaped** — so at §2.7's best measured 39.0 TFLOP/s it would be 95 us against
> 413, a **4.3x ceiling**, and at an honest 10-15 TFLOP/s for 64x64x128 tiles
> with a serial triangular solve and an 8-step serial carry, **1.1-1.6x**.
> A *perfect* chunked kernel saves 10% of the layer and **1.0% of the prefill
> graph**, and puts a recurrent state through fp16 operands where the scan is
> at 2.4e-06. **Revisit at decode, not at prefill.**
>
> **L3b-6: it matches, and the state carries bit-identically.** `attn_output`
> is **2.4e-06 rms** against both L3a's CPU reference and llama.cpp — 200x
> tighter than the tensors feeding it, because the recurrence has no matmul in
> it — and the fifteen rungs agree with each other to 2.6e-10 to 7.1e-10.
> **3 + 4 tokens reproduce 7 exactly**, output and state, because the
> convolution's window is three rows of *negative token index* in front of the
> projection's own output: no branch, no second tensor. The control that zeroes
> it moves the output by 4.0e-02.
>
> **L3b-7: and L2f-4's MALL rule holds on two more weights.** The fused
> projection's B is **84.5 MB**, 2.6x the MALL, and wants the widest row block
> everywhere — by 2.6x at 512 tokens and 3.3x at 2048. `ssm_out`'s is
> **31.5 MB** and fits, and picks the narrow rung at 512 and the wide one at
> 2048 — the same rungs at the same lengths as `attn_output`, which is the same
> shape. Same kernel, opposite schedules, decided by which side of one cache
> the weight falls on.

## What L4a established — the selection, and an arithmetic that changes at 8 columns

The 4k dump: 4096 tokens of wikitext-2 in a 4096-cell cache, one ubatch, the
same pinned build. 8.0 GB, 116 tensors, ~4 minutes — and the filter takes
layers 0 and 3, so **the whole MoE block comes with it**.
[Write-up](research/l4-qsa.md) · `reference/out/llm4k/`

> **L4a-1: the selection is a radix select, and ours is the reference's.**
> `topk_radix_select.comp` is **not a sort**: four 8-bit radix passes over an
> order-preserving float→uint map find the key of the width-th largest value,
> then it emits every cell **strictly above** it and fills the rest from the
> cells **equal** to it in `atomicAdd` order. Ported pass for pass, `topK`
> reproduces `indexer_top_k-3` on **all 4096 rows** — and, unexpectedly, **set
> for set on the 2045 rows where the selection bites**, because filling the tie
> in ascending cell index is what a wave resolving an atomic in lane order
> does over `ratio` consecutive cells. It is also what makes a 4096-token
> fixture affordable: four passes over a row where a partial selection sort
> costs 2051 of them.
>
> **L4a-2: the order is not reproducible and the selection is.** A second dump
> of the same prompt gives **byte-identical** per-cell scores, a top-k whose
> order differs on **4096 of 4096** rows, whose set differs on **1967**, and
> whose set *restricted to causally visible cells* differs on **none**. All the
> run-to-run difference is among the -inf cells the mask drops anyway. **So a
> kernel may emit the selection in whatever order is cheapest**, and must
> reproduce the set, which it can.
>
> **L4a-3: block granularity is now checked rather than read.** vLLM supplied
> the claim in L3 and llama.cpp only implies it; at 4 k it is a property of the
> reference's own output. Past the biting point every row is `(i+1) mod ratio`
> tail cells + **exactly 512** whole blocks + a split of `(width-tail) mod
> ratio` cells off **the lowest-scoring block in the selection** — 3 cells 512
> times, 2 cells 511, 1 cell 511, and **0 cells 511 times**. The selection
> first drops a visible cell at token **2051**, which is the width.
>
> **L4a-4: and 0.0022% of it survives our own scores.** A top-k is
> discontinuous, so two blocks within our error can swap across the threshold.
> Run end to end from `hc_mixed-3` through our own indexer: **138 of 6 298 621**
> visible cells differ, over **30 of 4096** tokens, worst row 8 cells — down
> from 2354 over 471 tokens before L4a-6.
>
> **L4a-5: every quantised matmul accumulates in fp16 at a real ubatch, and
> that is the tolerance.** At 4096 columns **100.0% of the values the reference
> writes out of a quantised matmul are exactly IEEE halves** — `Qcur_full`,
> `Kcur`, `Vcur`, `attn_output`, the DeltaNet's `z` and `linear_attn_out`, the
> MoE's `ffn_gate`/`ffn_up`, `ffn_shexp`. That is `.f16acc`, an fp16
> *accumulator*, and the exceptions confirm it: `hc_inject`, the F32 router and
> `shared_expert_gate` are **0.0%**. The consequence is that no model of the
> operands can close the gap — exact f32, fp16 operands and the reference's own
> int8 activations land within **1.06x** of each other and all ~**5.4e-03 rms**
> away on a tensor whose own rms is 1.21. **L2c-3, L2f-6 and L3b-6's reading
> holds an order of magnitude larger and from a different cause: at prefill the
> reference is the side losing precision.** A 1e-6 tolerance on a projection is
> a fact about a 7-token dump.
>
> **L4a-6: and the indexer's BF16 weights meet a bf16 activation, worth
> 2061x.** Above the 8-column threshold a BF16 `src0` against an F32 `src1`
> sets `y_non_contig`, so the *activation* is converted to BF16 and a
> BF16 x BF16 kernel runs — eight mantissa bits. `indexer_k_raw-3` goes from
> **9.179e-04 rms to 4.454e-07**, and the control is that **fp16 is 1.02x
> *worse* than f32**: this is a different numeric from L2e-3's, not the same
> one again. With it, the whole indexer chain comes back (24-69x on the pooled
> key, the normed key, the query and the score) — and **L2e-3 is separately
> re-confirmed at 4096 columns**, the F32 x F32 score being 7.2x better in fp16
> than in f32 and 59x better than in bf16.
>
> **L4a-7: two subnormal bugs, both invisible to five stages of comparison.**
> The fp16 census caught `ffn_shexp-3` at 99.9% where everything else was
> 100.0%, and the 11 047 exceptions were all the smallest magnitudes and all
> negative: **`f16Round` returned |x| for every negative subnormal half**,
> building it as `-0.0 + q*2^-24`. Fixing that exposed **`f16` widening every
> subnormal half to *half* its value** — exponent field 113+e where the
> arithmetic gives 114+e. `TestHalfPrecisionRoundTrips` now walks all 65 536
> halves and all 65 536 bfloats.

## What L4b established — the selection on the device, and sparsity priced

`llm/gpu_attn.go`, `shaders/llm_attn_select.comp`: a seventh dispatch that
only exists past 2051 cells, and a bitmask the attention kernel reads beside
the causal mask. [Write-up](research/l4b-qsa-gpu.md) · `results/l4b_qsa.csv`

> **L4b-1: one dispatch against twenty-four, and 2.40 ms becomes 1.24.** Per
> 512-token graph, against the two lines that are the selection: `TOP_K`
> (2.19 ms) and `TOPK_QSA GET_ROWS` (0.21) against our one kernel — **1.94x**,
> and 8.89 ms against 3.42 at ubatch 2048, **2.59x**. The second of the
> reference's ops exists only because the first emits an **index list**: it
> spends a `GET_ROWS` turning that back into an f16 mask for a *dense* flash
> attention. Ours writes the mask, which is 32x smaller than the list. With
> the selection counted on both sides the layer is **68.8 ms of llama.cpp's
> graph against 29.6, 2.32x**; a real run at that shape does not select at
> all, and then it is **2.47x**, because the reference runs an identity here
> and `sparse` declines to. L2f's 2.40x, which excluded it from both sides,
> re-measures at 2.38x and is unchanged.
>
> **L4b-2: the bucket search was the kernel, and the histogram never was.**
> The reference walks 256 radix buckets **serially on lane 0** while the other
> 255 wait, four times a row — 1024 dependent LDS reads on the critical path
> of a kernel whose input is 8 KB. The same answer is a suffix sum plus an
> `atomicMax`, and as a subgroup scan over four pinned waves it is **192.0 →
> 103.3 us, 1.86x**. Per-wave histograms — the standard fix for LDS atomic
> contention, and the obvious suspect — are **1.06x against**. What is left is
> 42-61 GB/s, 4-6x off the bus, and 0.1% of a prefill graph.
>
> **L4b-3: the selection costs the attention kernel 1.21x and saves it
> nothing.** At 4096 tokens in a 4096-cell cache the same kernel on the same
> arithmetic is **9950 us dense and 12046 reading the bitmask** — because
> without a selection only the diagonal key block passes through the P mask
> loop, and with one all 128 do. The first version, bit-testing the arena per
> element instead of staging the block's 16 mask words in LDS, was **18546 us,
> 1.90x**. So "the sparsity is semantics at prefill, not a saving" is now a
> number, and it is negative. Two traps the dense path cannot reach:
> `exp2(-inf - -inf)` is a **NaN** when a key block has nothing selected in
> it, and a pad row has no selection at all.
>
> **L4b-4: exact against the CPU selection, 0.0381% against llama.cpp's — and
> the 17x is upstream.** The bitmask is `topK`'s selection **cell for cell on
> all 4096 rows** over the device's own scores, with no tolerance, because a
> bitmask has no order for the tie fill to get wrong. Against llama.cpp it is
> 2400 of 6 298 621 visible cells where L4a's CPU path was 138 — and the cause
> is **L2f-3's fusion**: the indexer's two BF16 weights are fp16 column ranges
> of the layer's one fused matrix, so the device cannot take L4a-6's bf16
> activation. `indexer_k_raw-3` sits at **9.319e-04 rms, which is L4a-6's own
> *unmodelled* 9.179e-04**, 2093x its modelled 4.454e-07, and 24x on the
> score. One weight instead of six, priced.
>
> **L4b-5: the layer at 4 k, and the control.** `attn_output-3` is **9.872e-04
> rms** against llama.cpp at 4096 tokens — *inside* the 1.14e-03 the same
> comparison gives at 7 tokens, so the 17x does not propagate — and
> `attn_gated-3` 3.475e-04. The same layer with the selection off is
> 1.958e-03, **2.0x further**: the bitmask is read, and reading it is what
> moves the layer towards the reference. A kernel that loaded the mask and
> ignored it would pass every tolerance by being a dense causal attention,
> which at 7 tokens is the right answer and at 4096 is a different model.
>
> **L4b-6: the attention ladder's top two invert with length, and sparsity is
> not why.** At 4096 tokens qt1_kt2 is 9950 us, **qt2_kt4 10770** and qt1_kt4
> 13969, where L2f-5 measured 170/223/241 monotonic at 512. The inversion is
> in the dense column too. The winner does not move at either length or
> either density, which is what `DefaultAttnKernel` rests on.


## What L5a established — the MoE block, and a routing that is not balanced

`llm/moe.go`: the router, the softmax, the top-10 of 512, the normalised
weights, the three expert matmuls, the shared expert and its sigmoid gate —
against all fourteen tensors llama.cpp names inside the FFN half.
[Write-up](research/l5a-moe.md)

> **L5a-1: the selection is the reference's, set for set on all 4096 tokens.**
> 0 of 40 960 slots differ as a set — which is the number that matters,
> because a top-k is discontinuous and an rms on the weights would hide a
> swapped expert entirely. Two slots differ in *order*, on one row, and the
> reference's own probabilities for that pair are **4.35e-07 apart relative**
> against a router whose maxRel is 3.4e-05, so the test demands that any
> disagreement in order be a tie the router's precision cannot resolve. The
> router itself is 3.6e-05 rms, and it is L2e-3's numeric rather than
> L4a-5's: fp16 operands with an **f32** accumulator, because `.f16acc`
> belongs to the quantised kernels and L4a-5 put `ffn_moe_logits` at 0.0%
> exact halves. *(L5b-5 corrects the other half of this finding: the claim
> that `shared_expert_gate`'s single output column keeps it on the f32 vector
> path at any prompt length is wrong — the threshold counts the columns of
> the output, which is the token count. At a real ubatch it is on the matrix
> cores like everything else, and the device reproduces it bit-for-bit.)*
>
> **L5a-2: the block matches at the tolerance the fp16 accumulator sets.**
> `ffn_out-3` is **1.378e-04 rms**, with the three routed matmuls at
> 2.7-2.9e-03 on values of order 1 — the oracle's precision, not ours — and
> `shared_expert_gate` at **4.152e-05** because it is the one tensor here with
> no quantised weight in it — not, as this finding first said, because one
> output column keeps it on the f32 vector path; see L5b-5.
>
> **L5a-3: modelling the reference's arithmetic makes the fit *worse* here,
> by 1.4-1.8x.** `Exact` f32 is 2.217e-03 against llama.cpp where `RefQ8` is
> 3.054e-03 on `ffn_moe_gate`, 0.57-0.73x across three tensors. Everywhere
> else in this vertical modelling the reference's int8 activations was the
> largest correction available — 233x at L2b-2, 69x at L2e-3, 2072x at
> L2e-2, 2061x at L4a-6 — and here it is worth less than nothing, because the
> reference has already lost eleven mantissa bits in the accumulator and our
> quantisation error adds in quadrature rather than cancelling. **So L5b has
> nothing to reproduce**, and a tensor comparison at the MoE is a bound rather
> than a fit.
>
> **L5a-4: the routing is nothing like balanced, and that is what L5b is
> sized by.** L2a's budget assumed "each expert is read 10 times in 512". The
> mean is right and nothing else is: at ubatch 512 only **274 of 512 experts
> are touched**, the widest bucket is **484 tokens — 95% of the ubatch** — and
> **expert 454 is ranked first by 274 of the 512 tokens**. At 2048 and 4096 it
> is 399 and 435 experts touched with the top bucket at 60% and 46%. Every
> bucket is cross-checked against llama.cpp's own `ffn_moe_topk`, so the skew
> is the model's. Four consequences: half the bank is read per ubatch rather
> than all of it, a workgroup-per-expert kernel has a **26x** load imbalance
> that cannot be scheduled statically, arithmetic intensity varies **484x**
> between experts inside one dispatch (which is the argument for a grouped
> GEMM with a variable row count over a gather-GEMV), and at decode the top
> expert is the same 0.5 MB of Q4_K 95% of the time and would sit in the MALL
> if anything kept it there.
>
> **L5a-5: three controls, because three things here have a plausible wrong
> version every tolerance would pass.** Reading the expert bank interleaved
> rather than expert-major is **175x** worse; `silu(up)*gate` instead of
> `silu(gate)*up` is **59x** — the smallest margin of the three, because silu
> is near-linear away from zero; and the `weights_sum` clamp guards a division
> by zero the data never reaches (the smallest sum on this prompt is 0.0554
> against the clamp's 6.1e-05), so it is asserted from the graph and measured
> to be the identity on all 4096 rows.


## What L5b established — the block on the device, and two optimisations that were worth nothing

`llm/gpu_moe.go`, `shaders/llm_moe_{gemm,route,perm,combine}.comp`: nine
dispatches, the plain GEMM arm once and four kernels of this block's own.
[Write-up](research/l5b-moe-gpu.md) · `results/l5b_moe.csv`,
`results/l5b_moe_ladder.csv`

> **L5b-1: nine dispatches against about 845, and 567.4 ms becomes 522.3.**
> Per 512-token graph, against the lines of llama.cpp's own graph that are
> nameably this block: the router with the shared expert's gate 15.2 →
> **4.7 ms** (3.22x), the routing's tail 2.6 → **0.7** (3.70x), the gather
> 2.0 → **1.8**, gate/up with its `GLU` 372.3 → **345.4** (1.08x), down
> 158.9 → **134.0** (1.19x), the shared expert 16.4 → **13.6** (1.21x), and
> a **combine the reference does not have at all**, 22.1 ms, counted in full.
> **1.09x, and 48.7% of the whole graph becoming 44.8%** — a second run gives
> 522.2 ms, 0.3% apart. At `-ub 2048` the same comparison is **1758.4 ms
> against 1131.6, 1.55x**, because the MoE is the one block that gets cheaper
> per token on *both* sides as the chunk grows and the glue that makes 2048 a
> worse ubatch overall (L2a-4) belongs to other blocks. The `MUL` that
> applies the ten weights, the `MUL` and `SIGMOID` of the shared gate and the
> `ADD` that sums the two halves are left out of the comparison entirely —
> they live in the shared MUL/SIGMOID/ADD lines and are *absent* from our
> graph rather than faster in it.
>
> **L5b-2: the bank never becomes floats, and that is what makes the kernel
> possible.** Every other GEMM in this vertical takes a B operand the host has
> dequantised to halves and packed into §2.8 fragment tiles; one layer's three
> routed banks are 2.52 billion weights — **5.03 GB at fp16 and 241 GB across
> the model**, against 1.57 and 75 as they ship — so that cannot happen here.
> The bank is staged byte for byte out of the mmap'd checkpoint and each
> workgroup unpacks its own BN x BK slab into LDS per K-step, from **Q4_K,
> Q5_K, Q5_1 or Q8_0** under a `-DQFMT`, which is §2.2's arrangement for
> §2.2's reason: a cooperative-matrix fragment's lane layout is not exposed,
> so nibbles cannot be unpacked into registers and declared a fragment.
>
> **L5b-3: L5a-4's routing is the schedule, and padding the rows is what buys
> the epilogue.** 274 of 512 experts at ubatch 512 with one taking 95% of the
> tokens makes a grid of (experts x row blocks) almost entirely empty, so the
> tiles are a device-built **list** over a static upper bound with an early
> return — 372 real records against a bound of 1216. And every expert's range
> is padded up to the row block, the slack named by a **sentinel** permutation
> entry pointing at a token one past the batch: that is what lets a
> cooperative-matrix store, which covers sixteen rows and cannot be masked, go
> straight to global rather than through an LDS staging buffer. It executes
> 1.09-3.90x the rows that exist depending on the rung and the chunk, and
> **poisoning the sentinel's activation row with 1000 moves `ffn_out` by
> exactly zero** — which is the control that scheme needs and that no tensor
> comparison provides.
>
> **L5b-4: two optimisations worked, two did not, and the two that did not are
> the finding.** Deleting the LDS epilogue is **1.35x** on the whole block and
> widening the unpack's loads is **3.1x on Q8_0** — both about instruction
> count rather than about bytes or arithmetic. But **sharing the dequantised
> slab across two or four waves of one workgroup is 1.15-1.24x *against***, at
> the same row block, even though it halves the unpack per row and doubles the
> waves a CU holds on a kernel running at about one wave a SIMD: occupancy is
> not what this kernel is short of, which is the answer L3b-4 got for the
> scan's staged operands. And **making the K-step cover a whole Q4_K nibble
> group is 1.16x against** even though it removes a real 2x read
> amplification, because the doubled slab takes MODE 0's LDS from 12.8 KB to
> 23 and the workgroups a CU holds from five to two. A third: the
> conflict-free LDS pad is **5.4x worse** than the four-way-conflicting one,
> because 16-byte row alignment is what the fragment load needs and every
> aligned pad has `gcd >= 4` with the bank count.
>
> **L5b-5: the router is bit-exact, and that corrects L5a-1.**
> `ffn_moe_logits-3` is **0 ulp** on all 2 097 152 values where L5a's CPU
> reference is at 3.606e-05 — same operands, same accumulator, and the same
> 16-wide accumulation order. **`shared_expert_gate-3` is bit-exact too.**
> L5a-1 read the reference as keeping that one on the f32 *vector* path "at
> any prompt length" because it has one output column, and inferred that
> fusing it onto the router's matrix as a 513th column would be a deviation to
> price, in L3b-2's spirit. `mul_mat_vec_max_cols` counts the columns of the
> **output**, which is the token count, so at any real ubatch a [2560, 1] F32
> weight is on the fp16 matrix cores like everything else. The fusion costs
> nothing and it deletes 47 dispatches running at **13.6 GFLOP/s** — L2a's
> `inject` scandal in miniature, 9.1 ms a graph.
>
> **L5b-6: it matches, and the selection now matches in order too.**
> `ffn_out-3` is **8.132e-05 rms** against llama.cpp over all 4096 tokens —
> inside L5a's 1.378e-04 over 48 — with the swiglu at 5.5e-04 on values to
> 8.8, which is L4a-5's fp16 accumulator and not ours. The selection is the
> reference's **set for set on all 4096 tokens and 0 of 40 960 slots differ in
> sequence**, where the CPU reference had two: the tie that separated them is
> 4.35e-07 apart relative, the CPU's logits differ from the reference's by
> more than that and the device's do not differ at all. The permutation has a
> gate of its own with no tolerance in it — a bijection over all 40 960
> (token, slot) pairs, every row in the range of the expert that chose it, the
> histogram the selection's bucket for bucket, and the inverse round-tripping
> on all 45 056 slots.
>
> **L5b-7: and the next 2.5x is the unpack, or a bigger ubatch.** `up` runs at
> **10.8 TFLOP/s and 70 GB/s** of quantised bank against §2.7's 39 TFLOP/s and
> L0a's 236 GB/s. Its own byte floor is 2.9 ms against 7.2 measured, with the
> arithmetic able to hide under it. But the arithmetic intensity is the
> **routing's**, not the kernel's — 274 experts read 505 MB to serve 5120 rows,
> about 17 rows an expert — so the lever that actually moves is the chunk,
> which is why the block is 1.09x at 512 and 1.55x at 2048. L2a-4's "do not
> raise the ubatch" is a statement about the glue, not about this.


## What L6a established — the whole model resident, and a bank that is an array

`cmd/llm/resident.go`, `llm/gpu_moe.go`, `vk/shim.c`,
`shaders/llm_common.glsl`: 84.20 GB in 68 buffers, and binding 5 as an array.
[Write-up](research/l6a-residency.md) · `results/l6a_resident.csv`

> **L6a-1: 84.20 GB of weights and 0.87 GB of arenas in 68 buffers, staged in
> 33 seconds, with 41 GB of the machine left over.** 97 hyper-connection
> mixers (1.31 GB), the PLE block (0.07), 36 DeltaNet layers (4.18), 12
> attention layers (1.24) and 48 expert banks (77.41). Against the
> checkpoint's 82.52 GB resident core it is **1.04x** once the 0.68 GB
> embedding and the 0.68 GB lm head — which nothing stages yet — come off the
> reference figure; the **+3.04 GB** is the dense half staged as *halves*,
> 6.79 GB of fp16 against the 3.67 GB of Q8_0 and F32 those tensors ship as,
> 1.85x. The 28.80 GB n-gram table is still mmap'd (D2) and 41 GB of
> MemAvailable is left for its working set. **The DeltaNet bank is the near
> miss**: 36 fused projections in **4.18 GB in one buffer**, 2.7% under this
> device's cap.
>
> **L6a-2: the expert bank is an array of buffers, one a layer, because the
> arena-with-an-offset arrangement cannot express this model.**
> `maxStorageBufferRange` here is **4 GiB - 4** and a layer's bank is 1.61 GB,
> which is also the most a `uint32` byte offset addresses — the same wall
> twice. So binding 5 and binding 6 became `qb[NBANK]` and `qb4[NBANK]`,
> indexed by a **dynamically uniform** push constant, which is core Vulkan and
> not descriptor indexing: one dispatch is one layer. Three consequences.
> `shim_create_compute_pipeline` gained a per-binding `descriptorCount`, so
> seven bindings are 5 + 48 + 48 = **101 buffers** and `counts == NULL` is
> byte for byte the old path. NBANK is **compiled in at 48** — a spec constant
> cannot portably size a descriptor array and the SPIR-V is pre-compiled — so
> a short stage pads the array with the block's 256-byte placeholder. And the
> layer index rides in the **top sixteen bits of `moeUsed`**, because the push
> block is 64 uints against this device's whole 256-byte range and L5b's own
> comment on it reads *"Five fields, and the block is out of room"*.
>
> **L6a-3: the index is load-bearing, and the control says so by 203x.** Every
> other test here stages layer 3 alone, where the bank index is zero and an
> ignored index is indistinguishable from a read one. `TestMoEGPUBankArray`
> stages layers 0-3 and asks for the last: **8.133e-05 rms** against
> llama.cpp, against **1.652e-02** — 203x — for the same run with the index
> forced to zero. Layer 0's experts against layer 3's routing is a perfectly
> plausible tensor of the right shape and magnitude, which is why the control
> is in the test and not in a comment.
>
> **L6a-4: residency is free in the bank's size.** 2, 4, 12, 24 and 48 banks —
> **10.4 to 84.2 GB resident** — run layer 3's block in **11134, 11062, 11162,
> 11067 and 11242 us**: an eightfold range of bytes, a 1.6% spread, no trend.
> That is L0a (*"the DRAM bus does not care how big the weight bank is"*)
> holding at 68 live buffers and the size the model actually is, rather than
> at a probe's 23. What is left is a **flat ~2.5% step** against L5b's
> isolated 10824.7 us that appears as soon as the other four blocks are staged
> and does not grow with the bank — not the count and not the indirection,
> both constant across that column. At a ~1% noise band it is written down
> rather than chased: L6b runs all five blocks in one graph anyway.
>
> **L6a-5: staging order is worth 8 GB.** The expert bank is `memcpy` out of
> the mmap'd checkpoint at about 2.6 GB/s and needs no host floats at all; the
> dense half is **dequantised to f32 on the host** first, and the 36 DeltaNet
> layers are **8.35 GB of transient floats at once**. So the dense blocks go
> first, each block's floats are returned to the OS before the next asks, and
> the 77 GB goes last — the peak is the resident model rather than the
> resident model plus a dequantisation.

---

## The three findings that set the direction

**1. `llama.cpp` already implements this architecture, is built with Vulkan on
this box, and supports `qwen4exp`.** `/home/kube/repos/llama.cpp` has
`src/models/qwen4exp.cpp` (1279 lines: DeltaNet, the QSA indexer, PLE n-gram
hashing, hyper-connections) and 50 built tools. Confirmed at L1 by running
them: it is a reference implementation, an oracle (`llama-eval-callback`), an
accuracy criterion (`llama-perplexity`) and a performance baseline
(`llama-bench`) at once. **This vertical does not need a Python reference
dump**, which is what S1 and T1 spent their first session on.

> **And there is now a second implementation to argue with.** `../vllm` has
> `vllm/models/qwen4_exp/` (NVIDIA and AMD paths), `qwen_gdn_linear_attn.py`
> and a vendored flash-linear-attention. It **cannot** be a second oracle — it
> runs bf16 Triton on CUDA against our Q4_K_XL on Vulkan, so there is no
> tensor-level comparison without running a 180 B model here — but it is an
> **arbiter of formulas**, which is what D5's oracle turned out to need. In
> one pass it confirmed the GDN normalisation bug (L3a-4), explained the head
> map and found a hazard for L8 (L3a-3), disagreed about chunking (L3a-2), and
> settled what the QSA selection *is* without the 4k dump L4 was waiting for.
> The rule it earns: **llama.cpp for numbers, vLLM for formulas.**

**2. A quarter of UD-Q4_K_XL is a lookup table that should never touch the
GPU.** Of its 111.32 GB, **28.80 GB** is `per_layer_token_embd` — a 320 M-row
hash table read 16 rows of 160 values at a time, **1.41 KB per token**. Leave
it mmap'd on the host and the resident core is **82.52 GB**.

> **L1 confirmed llama.cpp does exactly this.** `qwen4exp.cpp` creates the
> table with `TENSOR_READ_LAZY` — *"read rows on demand instead of loading
> whole tensor; requires mmap"* — and `free` shows **77 GiB resident** during
> a run against the inventory's 76.86 GiB core. So D2 is not a deviation from
> the reference; it is what the reference does, and the baseline above is a
> like-for-like comparison.

**3. The stock quant is tuned for accuracy-per-byte, not for tok/s on a
bandwidth-bound APU, and it costs 1.7x.** Unsloth put every dense tensor at
Q8_0 and the routers at F32. On a machine where decode is pinned to the DRAM
bus that is the single most expensive choice available, because **76% of the
bytes read per token are dense** — each expert is read 10 times in 512, every
dense weight is read every time.

| | resident core | GB/token | **tok/s ceiling** | measured |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL as shipped | 82.52 GB | 6.334 | **38.2** | llama.cpp: **25.15** |
| our own bank, ~4.25 bits on everything streamed | ~67 GB | ~3.6 | **~67** | — |

---

## Direction

**Phase 1 — make it run on UD-Q4_K_XL.** Consume the GGUF natively. **Done at
L1**: the reader, the five dequant paths and the tokenizer all exist and are
checked against llama.cpp. **L2 has the dense skeleton on the device** — the
hyper-connection block, the PLE n-gram block and the full-attention layer with
its indexer — **L3 has the linear-attention layer**, **L4 has the QSA
selection**, and **L5 has the MoE block**, all checked tensor-for-tensor
against `llama-eval-callback`. **Every block of the model now has a kernel.**
The acceptance criterion is *it generates the same text as llama.cpp*. **L2a
sizes it**: ~1150 tok/s prefill against 391.4, and 38.2 tok/s decode against
25.15, both reachable with kernels that already exist plus the epilogue fusion
the hyper-connection block needs. **L2c, L2f, L3b, L4b and L5b between them
have taken 1015.5 ms of llama.cpp's 1164.7 ms prefill graph down to 714.8 —
87.2% of it replaced by 61.4%, a 25.8% saving on the whole graph.** **L6a has
the arena plan**: 84.20 GB in 68 buffers, all 48 layers of all five blocks
resident at once, with residency measured to cost nothing that scales with the
bank. What is left is **L6b** — the graph that runs them in order over one set
of arenas, the embedding gather and the lm head — and its gate, logits against
llama.cpp.

**Phase 2 — our own bank.** Re-quantise to the repo's W4A8 layout (§1.1's
repack) at widths chosen for this bus rather than for a generic machine:
~4.25 bits on everything streamed, fp16 routers, the n-gram table left as
IQ4_NL on disk, and — L0c — **fp16 scales per 32 nibbles in a k-major plane**.
Target ~67 tok/s against phase 1's 38.2 ceiling and the reference's 25.15.
Source it either by transcoding the GGUF (cheap, double-quantises) or from
the 360 GB bf16 (clean — and **unsloth publish their imatrix**,
`imatrix_unsloth.gguf`, 580 MB, so the calibration is free).

**Phase 3 — the throughput levers that are worth more than the format.** The
MTP head is downloaded (2.79 GB, `mtp-…-Q4_K_M.gguf`); a draft step is ~12% of
a full one, so self-speculation is worth ~1.5-1.8x. Batching amortises the
dense 76% completely. Both dwarf the difference between 4 and 5 bits.

---

## The machine, measured

| limit | value | where from |
|---|---|---|
| DRAM bandwidth | **236 GB/s** peak; **242 GB/s** best real decode GEMV | §0, §1.7 |
| MALL | 32 MiB, 805 GB/s copy / 965 read | §0.4, §5.1b |
| WMMA fp16 | 55.5 TFLOP/s; best real kernel 39.0 (70%) | §0.1, §2.7 |
| memory type, for a GPU read | **irrelevant**, all 8 within 0.57% | §5.1 |
| the real capacity ceiling | **physical RAM**, 117.7 GiB, less the OS | §5.1 |
| a resident 82.52 GB model | **77 GiB, steady**, with 39 GiB left for page cache | L1 |
| `maxBufferSize` / `maxMemoryAllocationSize` | **4 GiB − 4 B** | `vulkaninfo` |
| `maxDescriptorSetStorageBuffers` | 8 388 606 | `vulkaninfo` |
| GTT / system RAM | 117.7 GiB (`amdgpu.gttsize=126976` already set) | `/proc/cmdline` |
| disk free | 1.1 TB after the download | `df` |

The 4 GiB buffer cap is the one structural consequence: a 60-80 GB bank is
**~100 buffers**, one per layer per projection (a layer's Q4 expert pair is
1.26 GB, comfortably under). Descriptor counts are not a constraint.

---

## The model, from its own tensor table

`go run ./cmd/gguf -tensors` over the real checkpoint — 1224 tensors, not a
config.json transcription. `general.size_label` is `512x56B`;
`general.description` is "A Preview of the Qwen4 Architecture".

```
arch            qwen4exp   (Qwen4ExpForConditionalGeneration, multimodal)
layers          48 = 36 linear_attention + 12 full_attention (interval 4)
hidden          2560       residual stream 10240 (hyper-connections, hc_count 4, low_rank 320)
vocab           248320     context 262144 (1 M with YaRN)   tie_word_embeddings false
MoE             512 experts, 10 active + 1 shared, ffn 640, router in F32
```

A GGUF states a weight as `[in, out]`, so every shape below is `[K, N]`:

| tensor | shape | type | per pass |
|---|---|---|---|
| `attn_q` | [2560, **12288**] | Q8_0 | 12 — query *and* gate, 24 heads x 256 x 2 |
| `attn_k`, `attn_v` | [2560, 512] each | Q8_0 | 12 each — 2 KV heads x 256 |
| `attn_output` | [6144, 2560] | Q8_0 | 12 |
| `indexer.q_proj` | [2560, 512] | **BF16** | 12 — QSA, 4 heads x 128, top_k 2048 |
| `indexer.k_proj` | [2560, 128] | BF16 | 12 |
| `attn_qkv` | [2560, **10240**] | Q8_0 | 36 — DeltaNet's *fused* input projection |
| `attn_gate` | [2560, 6144] | Q8_0 | 36 |
| `ssm_out` | [6144, 2560] | Q8_0 | 36 |
| `ssm_alpha`, `ssm_beta` | [2560, 48] each | F32 | 36 each |
| `ssm_conv1d` | [4, 10240] | F32 | 36 — depthwise, k=4 |
| `ssm_a`, `ssm_dt.bias` | [48] | F32 | 36 |
| `hc_{attn,ffn}_{down,up}` | [10240, 320] / [320, 10240] | Q8_0 | **97 each** incl. `output_hc_*` |
| `hc_{attn,ffn}_inject` | [10240, 4] | F32 | 96 |
| `ple_key` | [2560, 10240] | Q8_0 | 1 — layer 1 only |
| `ple_value` | [2560, 2560] | Q8_0 | 1 |
| `per_layer_token_embd` | [160, **320001536**] | IQ4_NL | gather, 1.41 KB/token |
| `ffn_gate_exps`, `ffn_up_exps` | [2560, 640, 512] | Q4_K (Q5_K on layer 2) | 48 |
| `ffn_down_exps` | [640, 2560, 512] | **Q5_1** (Q8_0 on 5 layers) | 48 |
| `ffn_gate_inp` | [2560, 512] | F32 | 48 — the router |
| `ffn_*_shexp` | [2560, 640] / [640, 2560] | Q8_0 | 48 — the always-on shared expert |
| `token_embd`, `output` | [2560, 248320] | Q8_0 | gather / 1 |

Other metadata worth having to hand: `ssm` state 128, groups 16, dt_rank 48,
inner 6144; rope mrope interleaved, sections [11,11,10], 64 of 256 dims, theta
1e7; PLE ngram_size 3, heads_per_ngram 8 → 16 hash heads with prime vocabs
~20 000 0xx; vision tower 27 layers, hidden 1152, in a separate
`mmproj-*.gguf` (0.9 GB, not downloaded).

> **`bench/modelshapes.go` has been corrected to this table** (L1). It had
> `attn.q` at 6144, one fused 512-wide KV, a 2048/6144 DeltaNet pair that does
> not exist, and no rows at all for the hyper-connections, the indexer or the
> PLE block — **220 matmuls a token** and 13.4% of the weights. It now reads
> **6.665 B weights a token against the checkpoint's 6.671 B**, 0.09% apart,
> and `shapes` re-run over it prices the 2081 dispatches at **0.62 ms of
> launch cost, 5% of a 4-bit decode step**.

### What UD-Q4_K_XL actually contains

1224 tensors, 176.944 B params, **111.32 GB, 5.03 bits/weight** — produced by
`cmd/gguf` in 47 ms and cross-checked tensor-for-tensor against
`reference/gguf_inventory.py` (**0 disagreements**):

| group | params | GB | bits/w | read | quant mix (GB) |
|---|---:|---:|---:|---|---|
| moe_experts | 120.796 B | 77.02 | 5.10 | 10/512 | Q4_K 44.4, **Q5_1 27.1**, Q8_0 4.5, Q5_K 1.2 |
| **ngram_ple_table** | 51.200 B | **28.80** | 4.50 | **gather** | IQ4_NL 28.8 |
| deltanet | 2.087 B | 2.25 | 8.62 | every tok | Q8_0 |
| hyper_conn | 0.641 B | 0.70 | 8.68 | every tok | Q8_0 |
| lm_head | 0.636 B | 0.68 | 8.50 | every tok | Q8_0 |
| embed (lookup) | 0.636 B | 0.68 | 8.50 | gather | Q8_0 |
| full_attn | 0.598 B | 0.64 | 8.50 | every tok | Q8_0 |
| moe_router | 0.063 B | 0.25 | **32.00** | every tok | F32 |
| moe_shared | 0.236 B | 0.25 | 8.51 | every tok | Q8_0 |
| qsa_indexer | 0.020 B | 0.04 | 16.00 | every tok | BF16 |
| ple_proj | 0.033 B | 0.04 | 8.55 | every tok | Q8_0 |

Two things worth reading twice. **`ffn_down_exps` is Q5_1 on 43 of 48 layers
and Q8_0 on the other 5** while gate/up are Q4_K — that is unsloth's imatrix
telling them the down projection is the sensitive one, and it is 27 GB of the
77. And **the routers are F32**: 0.25 GB read every single token, 4% of the
decode budget for 0.06 B of parameters.

### Where the bytes go at decode

    dense, every token      4.830 GB   76%
    experts, 10 of 512      1.504 GB   24%
                            -------
    per token               6.334 GB   -> 38.2 tok/s at 242 GB/s
                                       -> llama.cpp reaches 25.15 (159 GB/s)

Capacity is an expert problem (97% of params). **Speed is a dense problem.**
The usual "keep attention at Q8, it's only 2% of the model" instinct is exactly
backwards on this machine: it costs 2% of memory and **a factor of 1.7 in
tok/s**. Conversely, going below 4 bits on the *experts* buys very little —
they are only a quarter of the traffic.

### Context is cheap here, which is the good news

12 full-attention layers x 2 KV heads x 256, and QSA caps what a step *reads*
at `top_k = 2048` however long the context is:

| context | KV fp16 | + indexer | DeltaNet state |
|---:|---:|---:|---:|
| 32 768 | 0.81 GB | 0.05 | 113 MB (constant) |
| 131 072 | 3.22 GB | 0.20 | 113 MB |
| 262 144 | 6.44 GB | 0.40 | 113 MB |

82.52 GB of weights plus 6.8 GB of KV is comfortable in 117.7 GiB, and L1
measured 77 GiB resident with 39 GiB of page cache still live beside it.
Capacity has stopped being the binding constraint on this model at any format
below Q8. Bandwidth is the whole story.

---

## What exists to build on

| piece | state |
|---|---|
| `gguf/` | **L1: the GGUF reader.** mmap'd, sharded, the split convention, the value grammar, openable from any shard. 1224/1224 tensors against the Python inventory. |
| `gguf/dequant.go` | **L1: Q4_K, Q5_K, Q5_1, Q8_0, IQ4_NL, F32, F16, BF16** — bit-exact against ggml's own `to_float` via `reference/dequant_ref.c`. |
| `cmd/gguf` | inventory, quant mix, decode budget, `-check` against `tensors.json`. |
| `cmd/llm` | the vertical's driver. `-tokenize` today; generation at L7. |
| `zimage/tokenizer/` | **L1: `FromVocab`** builds the BPE from GGUF metadata, and `split()` now carries both Qwen pre-tokenizers. 213/213 against `llama-tokenize`. |
| `llm/` | **L2b: the vertical's package.** `Config` from the checkpoint's own metadata, on-demand dequantisation, the `token_embd` gather (bit-exact), and the hyper-connection block — `HCInit`, `HCMix`, `HCCombine` — with a `Numerics` switch between the exact model and the reference's int8 arithmetic. |
| `reference/eval_dump.c` | **L2b: the oracle.** Whole tensors of a real llama.cpp pass, where `llama-eval-callback` prints three per axis and a sum. `llm/evaldump.go` reads them back as a `Trace`. **L4a** added `-f` (a prompt from a file) and `-nt` (an exact token count), and pins the build's headers. |
| `reference/out/llm4k/` | **L4a: the second trace**, 4096 tokens in a 4096-cell cache in one ubatch — 8.0 GB, 116 tensors. The only place the QSA selection exists, the only place the reference's real prefill arithmetic is visible, and it carries the whole MoE block for L5. The 7-token trace stays beside it: the empty blocks, the spare block and the f32 vector path are only legible there. |
| `llm/gpu.go` | **L2c: the block on the device.** Four arenas, the fused `[336, 10240]` down/inject weight, the up projection's row permutation, an M ladder per projection with a measured `PlanFor` schedule, and a sweep profiler that reads every staged mixer so the weights are as cold as a real graph's. |
| `shaders/llm_gemm.comp` | **L2c/L2d/L2f: the vertical's GEMM**, one kernel and one epilogue per mode — MODE 0 down+inject+silu, MODE 1 up+sigmoid+collapse, MODE 2 plain — because in this architecture the epilogue is where the time goes. |
| `shaders/llm_hc_*.comp`, `llm_ple_*.comp` | **L2c/L2d: the blocks.** `llm_hc_norm` (grouped RMSNorm → fp16 A operand), `llm_hc_combine`, `llm_ple_gate` (both norms, the signed-sqrt gate, the broadcast and the conv norm in one pass) and `llm_ple_conv` (four dilated taps, the SiLU and the residual add), over `llm_common.glsl`'s binding contract. |
| `llm/attn.go` | **L2e: the full-attention layer.** The fused query/gate projection, interleaved M-RoPE, the QSA indexer's pool/score/select with the reference's own cache block structure, and causal GQA — with the fp16 KV cache and the fp16 F32-matmul the reference turns out to use. **L4a** replaced the selection with a port of the reference's radix select, and made the indexer's two BF16 projections take a bf16 activation above the 8-column threshold. |
| `llm/gpu_attn.go` | **L2f: that layer on the device, in six dispatches.** The fused `[13952, 2560]` projection for six of llama.cpp's matrices, a host-built rotary table, two BM ladders because the two projections fall on opposite sides of the MALL, and a sweep profiler over every staged layer. |
| `shaders/llm_attn_select.comp` | **L4b: the QSA selection.** One workgroup a token, the row's keys cached in LDS, llama.cpp's four radix passes with its serial bucket walk replaced by a subgroup suffix sum, and a **per-cell bitmask** out instead of an index list — which deletes the reference's `GET_ROWS`. Dispatched only where `top_k + ratio - 1` is fewer cells than the cache holds; `SetSparse` forces it either way, which is how it is priced and how the dense control runs. |
| `shaders/llm_attn_*.comp` | **L2f: the layer's four kernels.** `llm_attn_pack` (per-head norm + interleaved M-RoPE + the fragment tiling for q, k and v in one grid), `llm_attn_idx` (the indexer's pooled key and query over two addressings), `llm_attn_score` (the rectified score, its bias, the cells and the causal mask) and `llm_attn_wmma` (causal GQA at headDim 256, **with the output gate in its epilogue**, and **L4b's bitmask staged a key block at a time** beside the causal mask). |
| `llm/moe.go` | **L5a: the MoE block.** `MoEConfig` from the checkpoint, the F32 router with fp16 operands and an f32 accumulator, the softmax/argsort/top-10/clamp/normalise chain, an `ExpertBank` that dequantises one expert's [640, 2560] matrix on demand out of a 77 GB tensor nothing can hold as floats, the routed half walked **by expert** rather than by token, and the shared expert with its one-column gate on the f32 vector path. |
| `cmd/llm -resident` | **L6a: the whole model on the device.** Stages every mixer, every layer and every expert bank, and prints the plan — bytes, buffers, wall clock — beside the checkpoint's inventory and the machine's memory. `-bank 48,24,12,4,2` restages the bank alone to price residency against itself; `-dense` leaves the 77 GB out. |
| `vk.PipelineSpec.Counts` | **L6a: a binding may be an array of buffers.** `shim_create_compute_pipeline` takes a per-binding `descriptorCount` and the flat buffer list is their concatenation; `nil` is the one-buffer-per-binding arrangement every other kernel in the repo uses, unchanged. |
| `llm/gpu_moe.go` | **L5b: the MoE block on the device, in nine dispatches.** The fused `[nExpert+1, nEmbd]` router with the shared expert's gate as its 513th column, a **quantised** bank staged byte for byte out of the checkpoint (1.57 GB a layer, against 5.03 as halves), a seventh binding that is the same bank read as `uvec4`, the permuted row space with the shared expert's group at its front, the measured `MoEPlanFor` schedule and a sweep profiler. **L6a** made the quantised bank one buffer a layer, bound as one array of `moeMaxBanks`, with the layer index in the top sixteen bits of `moeUsed`. |
| `shaders/llm_moe_gemm.comp` | **L5b: the grouped quantised GEMM.** Two modes — gate/up with `silu(gate)*up` on the accumulators, down in permutation order — crossed with the four formats the bank ships in (Q4_K, Q5_K, Q5_1, Q8_0) and five row-block rungs. Each workgroup unpacks its own BN x BK slab into LDS per K-step, and every store is a whole cooperative-matrix fragment because the permutation pads each expert's rows to the block. |
| `shaders/llm_moe_{route,perm,combine}.comp` | **L5b: the routing.** `llm_moe_route` is the softmax over 512, ten workgroup argmaxes reproducing `ggml_argsort`'s DESC comparator without materialising the other 502 ranks, the normalised weights and the shared gate's sigmoid — one workgroup a token, `llm_attn_select.comp`'s shape. `llm_moe_perm` is a counting sort, the padded offsets, the sentinel fill, the inverse permutation and the two tile schedules, in one workgroup. `llm_moe_combine` is the eleven contributions weighted and summed in the reference's order. |
| `llm/deltanet.go` | **L3a: the gated DeltaNet, 36 of the 48 layers.** The fused [2560, 10240] qkv projection, the depthwise causal conv, the L2 norm under either of llama.cpp's two spellings of it (`QKNorm`), the two F32 gate projections and the delta rule itself — with a `DeltaNetState` carrying both the [128, 128, 48] recurrent state *and* the convolution's window, bit-identically across a batch split. |
| `llm/gpu_deltanet.go` | **L3b: that layer on the device, in five dispatches.** The fused `[16512, 2560]` projection for four of llama.cpp's matrices — including both F32 gate projections — a recurrent state per staged layer, the convolution's window as `Conv-1` rows of negative token index in front of the projection's own output, and a sweep profiler over every staged layer. |
| `shaders/llm_dn_*.comp` | **L3b: the layer's three kernels.** `llm_dn_conv` (the depthwise causal convolution, its SiLU, the per-head L2 norm of q and k under either spelling, and the softplus/sigmoid pair, over a (plane, token) grid), `llm_dn_scan` (the delta rule — **a fifteen-rung ladder on LPC and QKREG**, whose ends are 6.3x apart) and `llm_dn_norm` (the gated RMS norm straight into the output matmul's A operand). |
| `llm/ple.go`, `llm/gpu_ple.go` | **L2d: the n-gram block.** The trigram hash and the host-side gather out of the mmap'd 28.80 GB table (D2), the CPU block, and the three-dispatch device path. |
| `zimage/qwen/` | a working Qwen3 transformer on the GPU in Go — RMSNorm, RoPE, GQA, SwiGLU, four shared arenas. The skeleton for L2. |
| `shaders/gemv_w4a8.comp` | decode GEMV at **99-103% of the bus** (§1.1, §1.7), plus 25 grouped / M-blocked / N-blocked MoE builds (§1.8-§1.12). |
| `shaders/gemm_wmma_q4.comp` | Q4 prefill GEMM, **2.10x fp16** on a MoE block (§2.2), with L0c's k-major scale plane. |
| `shaders/moe_route/gather/combine` | top-k routing and the permutation around it. |
| `shaders/qwen_attn_wmma_*`, `qwen_rope.comp` | causal WMMA attention at 38.8 TFLOP/s (§3.3); NeoX rotary. |
| `rmsnorm_shared.comp` | 95% of the bus. |
| `shaders/bank_gather.comp` + the `bank` family | L0a and L0b's probe. |
| `bench/modelshapes.go` | **L1: corrected** to the real tensor table. |

## What has to be built

1. ~~**Hyper-connections**~~ — **done at L2b and L2c**, the block the whole
   prefill question turned on: 4-branch gated residual at width 10240, every
   layer, 194 low-rank projections a token, now four dispatches and 4.24x.
2. ~~**Gated DeltaNet — 36 of the 48 layers**~~ — **done at L3a and L3b.**
   The CPU reference matches all eighteen tensors of layer 0 with the
   recurrence at 3.11e-09 rms, and the kernel runs the whole layer in five
   dispatches at 1.37x the reference, the scan itself at 1.06x. §3.6's
   question — scan or chunk — is **answered by pricing rather than by
   building**: the recurrence is 13% of this layer against the fused
   projection's 64%, and a perfect chunked kernel would save 1.0% of the
   prefill graph (L3b-5).
3. ~~**QSA sparse attention**~~ — **done at L2e, L2f, L4a and L4b.** The CPU
   reference, the layer's four kernels, the selection settled as a radix
   select at 4 k, and the selection as a kernel: a per-cell bitmask that
   `llm_attn_wmma.comp` reads beside the causal mask, **1.94x llama.cpp's
   TOP_K and its GET_ROWS together**. What the architecture is named for turns
   out to **cost** 1.21x at prefill, because the reference runs dense flash
   attention over a mask and so do we (L4b-3); it is an optimisation at
   decode, not here. Still not in `IDEAS.md`; it needs a section.
4. ~~**PLE n-gram**~~ — **done at L2d**: trigram hashing into 16 heads over a
   320 M-row mmap'd table, `layer_multipliers`, conv1d k=4, key/value
   projections, bit-exact and in three dispatches.
5. ~~**The MoE block**~~ — **done at L5a and L5b.** The CPU reference matches
   all fourteen tensors of the FFN half at 4096 tokens, and the kernel runs
   the block in nine dispatches against about 845 — **1.09x at llama.cpp's
   own best ubatch and 1.55x at 2048** — over a bank that stays in the
   checkpoint's own Q4_K, Q5_K, Q5_1 and Q8_0 blocks because 241 GB of halves
   is not a thing this machine holds. L5a-4's 26x imbalance is answered by a
   device-built tile *list* rather than a grid, and by padding each expert's
   rows to the block so the epilogue needs no LDS at all (L5b-3).
6. ~~**Residency**~~ — **done at L6a**: 84.20 GB in 68 buffers, and a bank
   that is an array because a storage buffer here is at most 4 GiB - 4. What
   is left of the graph is the order, one set of arenas, and the head.
7. **KV cache** and a decode loop. The mrope is **done at L2e and L2f**
   (interleaved, sections [11,11,10], 64 of 256 dims, NeoX on a text batch);
   what is left is the ring buffer, the indexer cache's own incremental
   pooling, and a rotary table that does not want a row per context cell.
8. **The re-quantiser** (phase 2) and **MTP speculative decoding** (phase 3).

---

## Task list

### L0 — answer the format question by measurement, not arithmetic

- [x] **L0.0** the inventory without downloading the model —
      `reference/gguf_inventory.py`, 35 MB of range requests.
- [x] **L0a** *Does the bus survive a 64-82 GB working set?* **It does not
      notice.** [Write-up](research/l0a-bank-range.md)
- [x] **L0b** *Is heap 0 as fast as heap 1?* **No type or heap is faster than
      any other, and the heap sizes are not limits.**
      [Write-up](research/5.1-memory-types.md)
- [x] **L0c** *The blocked scale plane.* **1.27x, and it changes the format.**
      [Write-up](research/l0c-scale-plane.md)
- [x] **L0d** *Per-format error metrics* (§7). **W4A8, symmetric, per-token
      activation scales.** [Write-up](research/l0d-quant-error.md)

### L1 — get the checkpoint, and a baseline  *(done)*

- [x] Download `UD-Q4_K_XL` (4 shards, 111.32 GB) and `mtp-…-Q4_K_M.gguf`
      (2.79 GB). **18 minutes**, `reference/fetch_llm_checkpoint.sh`, resumable.
- [x] `llama-bench` and `llama-completion`: **pp2048 388.60, tg128 25.15**.
- [x] Go GGUF reader, checked tensor-for-tensor: **1224/1224, 0
      disagreements**.
- [x] Dequant paths for the five formats, **bit-exact against ggml**.
- [x] Tokenizer: 248320 vocab, the multimodal specials and the `qwen35`
      pre-tokenizer — **213/213 against `llama-tokenize`**.
- [x] Correct `bench/modelshapes.go` to the real inventory.
- [x] `llama-perplexity` on wikitext-2 (145 chunks at n_ctx 2048, 15.2
      minutes): **PPL = 4.0340 ± 0.02283**, the accuracy reference for
      everything phase 2 does.
- [x] Re-run the `shapes` family against the corrected table — 313 rows. It
      prices a decode step at 3.33 GB and **73 tok/s at 4 bits before
      attention**, with **2081 dispatches costing 0.62 ms, 5%** (§4.1). The
      old table had 1861 and read 13.4% too few weights.

### L2 — the dense skeleton  *(done)*

- [x] **L2a — settle the prefill question.** Done by per-op attribution of
      llama.cpp's own graph rather than by pricing one layer: **DeltaNet is
      1.6%, attention+QSA 1.5%, and the 5x is MoE (35.7%), dense matmul
      (26.2%), glue (30.6%) and tiny-N F32 (12.2%)**. Against our own kernels
      the chunk is 2.25-2.53x, so **the target is ~1150 tok/s, not ~2000**.
      [Write-up](research/l2a-prefill-attribution.md) · `results/l2a_prefill_ops.csv`
- [x] **L2b — the hyper-connection block, CPU reference.** `llm/`: the block,
      a `Config` read from the checkpoint, and **`reference/eval_dump.c`**, a
      whole-tensor oracle (143 tensors, 35.7 MB, all of layers 0-3 — enough
      for L2, L3 *and* L4 from one model load). Every mixer and combine of the
      four dumped layers matches; four of seven gates to **6.2e-08 rms**. And
      it found what the oracle is really computing — see L2b-2 above.
      [Write-up](research/l2b-hyper-connections.md)
- [x] **L2c — the fused hyper-connection kernel.** L2a-2's specification,
      built: grouped RMSNorm into the GEMM's fp16 A layout, `hc.down` **with
      `inject`'s four columns fused onto it**, the low-rank gate's sigmoid and
      the 4-branch collapse inside the up projection's epilogue, and the
      combine in one pass. **Four dispatches against sixteen, 226.6 ms of
      llama.cpp's 512-token graph against 53.4 — 4.24x**, and it reproduces
      L2b's CPU reference to 1e-04 rms on all eight mixers of layers 0-3.
      [Write-up](research/l2c-hc-kernel.md) · `results/l2c_hc.csv`
- [x] **L2d — the PLE gather and the n-gram block.** The host-side trigram
      hash (**bit-exact**, 112 rows of a 320 M-row table), the block in Go
      (**7.7e-08 rms**) and on the device in three dispatches against about
      thirty. Layer 1's mixer, the one L2c could not compare, now matches too.
      [Write-up](research/l2d-ple.md) · `results/l2d_ple.csv`
- [x] **L2e — one full-attention layer with mrope, and the QSA indexer beside
      it.** The CPU reference: the fused query/gate projection, IMRoPE (which
      is **NeoX on text**, asserted rather than assumed), the block-pooled
      indexer and causal GQA. Every dumped tensor of layer 3 matches, and the
      four-stage chain from `l_last-2` lands at **9.75e-06 rms**. Two more
      oracle numerics fell out. [Write-up](research/l2e-attention.md)
- [x] **L2f — the GPU port of that layer.** Six dispatches against the
      reference's nineteen: one fused `[13952, 2560]` projection for six of its
      matrices, one pass doing the per-head norm, the M-RoPE and the fragment
      tiling for q, k and v together, the indexer's pool/norm/rotate in one
      kernel and its score/bias/mask in another, and causal GQA on the matrix
      cores **with the output gate in its epilogue**. **66.3 ms of llama.cpp's
      512-token graph becomes 27.6 — 2.40x, 1.60x at equal attention work** —
      and the layer's output is 5.9e-05 rms against the f32 model, 19.3x nearer
      it than the oracle. [Write-up](research/l2f-attention-gpu.md) ·
      `results/l2f_attn.csv`
- [x] Gate: layer 3's output matches the reference. **Not "to fp16
      tolerance"** — L2b-3: an rms bound under `llm.RefQ8`, since the
      reference is computing in int8 and the error is heavy-tailed.
      `hc_combine-3`, the residual after the attention half, is at 9.75e-06 rms
      on the CPU and `attn_output-3` at 1.14e-03 against llama.cpp on the
      device. `l_last-3` needs the MoE half, which is L5.

### L3 — Gated DeltaNet  *(done)*

- [x] **L3a — CPU reference, fp32 state, against the dump.** All eighteen
      tensors of layer 0's graph match: the recurrence **3.11e-09 rms**, its
      786 432-value state **1.91e-08**, the layer's output **1.31e-07**, with
      layers 1 and 2 beside it. Five things the architecture does not settle
      and the reference does — it **does not chunk**, the head map is
      **modulo**, the state is **stored transposed**, `softplus` **flushes to
      zero** in f32, and the build's `build_gdn_l2_norm` **predates #28068's
      fix** and reproducing that is worth 15x. Plus L2e-3 sharpened into a
      threshold: the fp16 F32 matmul starts at **8 output columns**, so it is
      absent from this 7-token dump and present at a 512-token ubatch.
      [Write-up](research/l3-deltanet.md)
- [x] Gate: layer 0 matches; state **bit-identical** in fp32 across a chunk
      boundary — 3 + 4 tokens reproduce 7 exactly, output and state, with the
      convolution's window carried beside the recurrent state and a control
      showing the carry matters (4.5e-02 rms without it).
- [x] **L3b — the GPU kernel.** Five dispatches against eleven: one fused
      [16512, 2560] projection for four of llama.cpp's matrices, one pass doing
      the convolution, its SiLU, both L2 norms and the two per-head scalars,
      the delta rule with the state in registers, the gated norm straight into
      the output matmul's A operand, and the output projection. **152.8 ms of
      llama.cpp's 512-token graph becomes 111.5 — 1.37x** — and the scan itself
      is **413 us against 438**. §3.6's question is answered by a price rather
      than a build: the recurrence is 13% of this layer, the projection 64%,
      and a perfect chunked kernel saves 1.0% of the prefill graph.
      [Write-up](research/l3b-deltanet-gpu.md) · `results/l3b_dn.csv`
- [x] Gate: layers 0, 1 and 2 match — `attn_output` at **2.4e-06 rms** against
      both the CPU reference and llama.cpp — and 3 + 4 tokens reproduce 7
      **bit-identically** on the device, with the control that zeroes the
      convolution's window moving the output by 4.0e-02.

### L4 — QSA

- [x] **L4a — re-dump at 4 k, and settle the selection on the CPU.** 4096
      tokens in a 4096-cell cache, one ubatch, 8.0 GB, the same pinned build
      — and the filter brings the whole MoE block with it, so L5's fixture
      exists already. `eval_dump.c` grew `-f` and `-nt`, and is now built
      against headers extracted from the *build's* commit rather than from a
      working tree that has moved on. The selection is a **radix select**
      reproduced pass for pass (all 4096 rows algorithmically, **set for set
      on the 2045 biting rows**), its block granularity is checked rather than
      read, its order is measured to be irreproducible and its visible set to
      be stable, and **0.0022% of it survives our own scores**. Two findings
      that are not about QSA: **every quantised matmul accumulates in fp16 at
      a real ubatch** (so the prefill tolerance is 5e-3, and the reference is
      the side losing precision), and the indexer's **BF16 weights meet a bf16
      activation**, worth 2061x. Plus two subnormal bugs.
      [Write-up](research/l4-qsa.md)
- [x] Gate: our selection is the reference's, from the reference's own scores
      — `TestQSASelectionIsARadixSelect`, with
      `TestQSASelectionIsStableAcrossRuns` and
      `TestQSASelectionIsBlockGranular` beside it.
- [x] **L4b — the selection on the device, and the gathered attention.**
      `llm_attn_select.comp`: one workgroup a token, the row's keys in LDS,
      four radix passes, a **per-cell bitmask** out — and `llm_attn_wmma.comp`
      reading it beside the causal mask, staged a key block at a time. **2.40
      ms of llama.cpp's 512-token graph becomes 1.24, 1.94x**, and its second
      op (a `GET_ROWS` turning an index list back into a mask) is deleted
      rather than beaten. The reference's serial 256-bucket walk is 1.86x of
      it. And the thing the architecture is named for is now priced: at
      prefill the selection **costs** the attention kernel 1.21x, because
      `build_attn_qsa` runs dense flash attention over a mask and so do we.
      [Write-up](research/l4b-qsa-gpu.md) · `results/l4b_qsa.csv`
- [x] Gate: layer 3's `attn_output` matches at 4 k context, on the device,
      where the selection bites — **9.872e-04 rms** against llama.cpp, inside
      the 7-token figure — with the bitmask **cell for cell** the CPU
      selection's on all 4096 rows and a dense control 2.0x further away.

### L5 — the MoE block

- [x] **L5a — the block in Go, against the 4k dump.** `llm/moe.go`: the F32
      router, the softmax over 512, the argsort top-10, the normalised weights
      with the reference's clamp, the three expert matmuls walked **by expert**
      so each bank is gathered once, the shared expert and its one-column
      sigmoid gate. All fourteen tensors match — `ffn_out-3` at **1.378e-04
      rms** — and the selection is the reference's **set for set on all 4096
      tokens**. Two findings that are not about correctness: modelling the
      reference's int8 activations makes the fit **worse** here (L5a-3), and
      the routing is **not balanced** — 274 of 512 experts at ubatch 512, one
      of them taking 95% of the tokens (L5a-4).
      [Write-up](research/l5a-moe.md)
- [x] Gate: layers 3's whole FFN half matches from llama.cpp's own
      `hc_mixed-3` (the **second** occurrence — the block runs twice a layer),
      with three negative controls: the expert bank's row order (175x), the
      SwiGLU's two halves (59x) and the `weights_sum` clamp, which is invisible
      in the data and asserted from the graph.
- [x] **L5b — the kernel.** Nine dispatches: the router with the shared
      expert's gate as a 513th column, the softmax/top-10/normalise in one
      workgroup a token, a counting sort and its two tile schedules, a
      **grouped GEMM over the checkpoint's own quantised blocks** (Q4_K, Q5_K,
      Q5_1, Q8_0 — because 241 GB of halves is not a thing this machine
      holds), and the weighted combine. L5a-4's 26x imbalance is answered by a
      device-built tile *list* over a static bound, and by padding each
      expert's rows to the row block so a cooperative-matrix store goes
      straight to global. **567.4 ms of llama.cpp's 512-token graph becomes
      522.3 — 1.09x — and 1758.4 becomes 1131.6 at ubatch 2048, 1.55x.**
      Two optimisations that should have worked did not, and are written down
      (L5b-4). [Write-up](research/l5b-moe-gpu.md) · `results/l5b_moe.csv`
- [x] Gate: the block matches on the device at 4096 tokens — **`ffn_out-3` at
      8.132e-05 rms**, inside L5a's figure over 48 tokens — with the selection
      the reference's set for set *and order for order*, the permutation
      checked as an exact bijection, and the padding shown to be inert by
      poisoning the row it reads. Rates reported against L0a: `up` at 70 GB/s
      of quantised bank and 10.8 TFLOP/s, `down` at 121 GB/s. 35.7% of
      llama.cpp's graph in `moe_experts` alone, 48.7% counting everything
      nameably this block, becomes 44.8%.
- [ ] **L5c — decode's shape, when L7 needs it.** The grouped Q4 GEMV
      (§1.8-§1.12) against a one-token routing, where L5a-4's hot expert is
      the same 0.5 MB of Q4_K 95% of the time and would sit in the MALL if
      anything kept it there. Not built at L5b, because at prefill the
      grouped GEMM is the whole block.

### L6 — the whole stack, prefill only

- [x] **L6a — residency.** All 48 layers of all five blocks on the device at
      once: **84.20 GB of weights and 0.87 GB of arenas in 68 buffers, staged
      in 33 s**, the n-gram table still mmap'd, 41 GB of the machine left
      over. The MoE bank is an **array of buffers, one a layer**, because
      `maxStorageBufferRange` is 4 GiB - 4 and a layer is 1.61 GB — a
      per-binding `descriptorCount` in the shim, a block-instance array in the
      GLSL, NBANK compiled in at 48, and the layer index in the top sixteen
      bits of `moeUsed` because the push block was full. **Residency is free
      in the bank's size** (L6a-4). [Write-up](research/l6a-residency.md) ·
      `results/l6a_resident.csv`
- [x] Gate: the bank index is load-bearing — layer 3 read from bank 3 is
      **8.133e-05 rms** against llama.cpp, the same run with the index forced
      to zero is **203x further away**, and every L5b gate still passes with
      the array in place.
- [ ] **L6b — the graph.** The five blocks in llama.cpp's own order over one
      set of arenas, plus the two tensors nothing stages yet: the embedding
      gather and the lm head.
- [ ] Gate: logits match llama.cpp; prefill tok/s against 388.60.

### L7 — decode

- [ ] KV cache, ring buffer, sampling, the generation loop.
- [ ] Gate: it generates the same text as llama.cpp; tok/s against the 38.2
      ceiling and against 25.15.

### L8 — phase 2, our own bank

- [ ] Choose per-tensor widths from L0d and L1's **PPL 4.0340**.
- [ ] Re-quantise (transcode from the GGUF, or from bf16 with unsloth's
      published imatrix) into the §1.1 W4A8 layout with an L0c scale plane.
      **If from bf16: apply llama.cpp's V-head reorder first** (L3a-3) — seven
      tensor families per linear layer, or `kHeadOfV` is wrong on 32 of 48
      heads in 36 of 48 layers.
- [ ] Gate: ~67 tok/s, ≤67 GB resident, perplexity within a stated delta of
      **4.0340** on the same corpus, context and chunking.

### L9 — phase 3, and shipping

- [ ] MTP speculative decoding (the separate GGUF). Target 1.5-1.8x.
- [ ] Batching, then the HTTP API `GOALS.md` asks for.
- [ ] Vision tower, if wanted.

---

## How to run things

    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
    L=/home/kube/repos/llama.cpp

    # ours
    go run ./cmd/gguf $M                      # the inventory and the decode budget
    go run ./cmd/gguf -tensors -kv $M         # every tensor, every metadata key
    go run ./cmd/llm -tokenize 'hello world'  # generation arrives at L7
    go run ./cmd/llm -hc -model $M -tokens 512          # L2c's block, against L2a's table
    go run ./cmd/llm -hc -model $M -ladder -csv results/l2c_hc.csv
    go run ./cmd/llm -ple -model $M -csv results/l2d_ple.csv   # L2d's block
    go run ./cmd/llm -attn -model $M                    # L2f's layer, 64..2048
    go run ./cmd/llm -attn -model $M -tokens 512 -ladder
    go run ./cmd/llm -attn -model $M -ladder -csv results/l2f_attn.csv
    go run ./cmd/llm -attn -model $M -tokens 512,2048 -sel on   # L4b: price the
    go run ./cmd/llm -attn -model $M -tokens 4096 -ctx 4096     # selection, then
    go run ./cmd/llm -attn -model $M -tokens 4096 -ctx 4096 -sel off  # the control
    go run ./cmd/llm -dn -model $M -tokens 512                 # L3b's layer
    go run ./cmd/llm -dn -model $M -ladder -csv results/l3b_dn.csv
    go run ./cmd/llm -dn -model $M -tokens 512,2048 -gemm-ladder
    go run ./cmd/llm -moe -model $M -tokens 512,2048              # L5b's block
    go run ./cmd/llm -moe -model $M -tokens 512,2048 -ladder -iters 5 \
        -csv results/l5b_moe_ladder.csv
    go run ./cmd/llm -resident -model $M                      # L6a: the whole model
    go run ./cmd/llm -resident -model $M -dense               # the 6.8 GB half alone
    go run ./cmd/llm -resident -model $M -bank 48,24,12,4,2 \
        -csv results/l6a_resident.csv                         # residency, priced

    # the reference implementation, the oracle, the baseline
    less $L/src/models/qwen4exp.cpp                  # 1279 lines, the whole architecture
    $L/build/bin/llama-bench -m $M -p 512,2048 -n 128 -r 2
    $L/build/bin/llama-eval-callback -m $M -p 'hello' -n 1   # per-tensor dumps
    $L/build/bin/llama-perplexity -m $M -f models/wikitext-2-raw/wiki.test.raw -c 2048 -b 2048

    # L2a's tool: a timestamp query around every dispatch in the Vulkan graph.
    # Costs 4-7% and prints one per-op-shape table per graph compute.
    GGML_VK_PERF_LOGGER=1 $L/build/bin/llama-bench -m $M -p 2048 -n 16 -r 1 -ub 512

    # L2b's tool: whole intermediate tensors of a real pass, to reference/out/llm.
    # See research/l2b-hyper-connections.md for the build line and the filter.
    /tmp/eval_dump -m $M -o reference/out/llm -c 64 -p '…' -n '<regex>'
    go test ./llm/ -v                        # the block against that trace,
                                             # CPU and GPU, plus the controls
    go test ./llm/ -v -run TestDeltaNet      # L3a: the linear layers
    go test ./llm/ -v -run TestDeltaNetGPU   # L3b: its kernels, the ladder, the carry

    # L4a's fixture: 4096 tokens, 4096 cells, one ubatch, 8.0 GB, ~4 minutes.
    # The headers come from the *build's* commit — the checkout has moved past it.
    H=$(mktemp -d) && (cd $L && git archive cff184438 include ggml/include | tar -x -C $H)
    gcc -O2 -o /tmp/eval_dump reference/eval_dump.c -I$H/include -I$H/ggml/include \
        -L$L/build/bin -lllama -lggml-base -Wl,-rpath,$L/build/bin
    mkdir -p reference/out/llm4k
    head -c 40000 models/wikitext-2-raw/wiki.test.raw | head -n -1 > reference/out/llm4k/prompt.txt
    /tmp/eval_dump -m $M -o reference/out/llm4k -f reference/out/llm4k/prompt.txt \
        -nt 4096 -c 4096 -ub 4096 \
        -n '^(model\.input_embed|ple_embd|hc_init)$|-(0|3)$|^ple_(gate|gated_value|conv_out)-1$'
    go test ./llm/ -v -run TestQSA           # L4a: the selection, at the length it exists
    go test ./llm/ -v -run 'TestAttnGPUSelection|TestAttnGPULayer4k|TestAttnGPUIndexer4k'
                                             # L4b: the same on the device, plus
                                             # the dense control and the bf16 price
    go test ./llm/ -v -run 'TestMatMuls|TestF16Acc|TestIndexerProjection|TestHalfPrecision'
    go test ./llm/ -v -run TestMoE           # L5a: the MoE block, its controls,
                                             # and the routing distribution
    go test ./llm/ -v -run TestMoEGPU        # L5b: the same on the device, the
                                             # permutation, the ladder's
                                             # agreement and the padding control
    go test ./llm/ -v -run TestMoEGPUBankArray  # L6a: the bank index, and the
                                             # control that shows it is read

    # re-fetch (resumable, checks sizes)
    reference/fetch_llm_checkpoint.sh

## Decisions

| # | decision | why |
|---|---|---|
| D1 | **Consume UD-Q4_K_XL first, don't quantise from bf16.** | 114 GB instead of 360, an imatrix-calibrated checkpoint, and a bit-exact oracle. Phase order: run it, profile it, then optimise (`TODO.md`). |
| D2 | **The n-gram table lives off-heap, mmap'd.** | 28.80 GB of capacity for 1.41 KB/token of bandwidth. **L1: llama.cpp already does this** — `TENSOR_READ_LAZY` — so it is the reference behaviour, not a deviation. |
| D3 | **Target ~4.25 bits on everything *streamed*, not just the experts.** | 76% of decode bytes are dense. L0d adds the accuracy half: above ~5 bits the int8 activations are the floor anyway. |
| D4 | **Do not go below 4 bits on the experts.** | They are 24% of the traffic; Q3 buys ~11% for a real accuracy hit. |
| D6 | **W4A8, with per-token activation scales.** | L0d: int8-per-token costs **1.13x** the error of fp16 activations, against a 3x throughput cliff. Per-token is free in the RMSNorm epilogue (§3.1). |
| D7 | **Symmetric Q4, not asymmetric.** | L0d: 1.043-1.053x at equal bits, where §7 predicted 2x. Revisit only if L9's perplexity asks for it. |
| D8 | **fp16 scales per 32 nibbles, in a k-major plane.** | L0c made the fine block cost 1.0% instead of 14.2% at prefill. The remaining cost is decode bytes — 11.1% of the bank against 3.0% at per-128 — the one axis still worth trading if tok/s falls short. |
| D5 | **`llama.cpp` is the oracle, not a Python dump.** | It is built, it is Vulkan, it supports `qwen4exp`, and L1 got correctness, accuracy and a baseline out of one binary. |

## Open questions

- **L8's bf16 path has to replicate a head permutation, or it is a different
  model.** L3a-3 and vLLM between them: llama.cpp's converter reorders the 48
  V heads from HF's grouped order into tiled order — `in_proj_qkv`'s V rows,
  `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, `conv1d`'s V
  channels and `out_proj`'s columns, all seven together — so that `h % 16`
  recovers the right key head. Phase 2's stated alternative is to re-quantise
  "from the 360 GB bf16 (clean — and unsloth publish their imatrix)", and that
  source is in **HF's** order, where `h % 16` is wrong on 32 of the 48 heads.
  It would be wrong in **36 of the 48 layers** with nothing but perplexity to
  notice. Either replicate `_reorder_v_heads` or switch `kHeadOfV` to divide;
  never half of each. Transcoding the GGUF (D1's default) is unaffected.
- ~~**Does the DeltaNet kernel want the scan or the chunk?**~~ **Answered at
  L3b, by pricing rather than by building.** The scan, at the rung this
  hardware wants, is **413 us a layer against the reference's 438** — and it is
  **13% of the layer**, where the fused input projection is 64%. The chunked
  form is 1.15x the arithmetic on the matrix cores: a 4.3x ceiling at §2.7's
  best measured rate, 1.1-1.6x at an honest one for 64x64x128 tiles with a
  serial triangular solve in the middle, and a *perfect* version saves 1.0% of
  the prefill graph while putting a recurrent state through fp16 operands.
  **What is still open is the same question at decode**, where a chunk is a
  token and the whole argument changes; that is L7's.
- **Why is llama.cpp's own scan shape 6.3x better in its kernel than in
  ours?** L3b-3 ran the reference's decomposition — one state column per
  workgroup, 6144 workgroups, q and k re-read 128 times — and got 2766 us
  where llama.cpp's own kernel does 438 at what should be the same work. The
  operand traffic is most of ours (deleting it: 2766 → 1013 us) and the
  subgroup reductions are not (5%), but 1013 is still 2.3x the reference's
  whole dispatch. Ours wins at a different rung and by more than this costs,
  so it is written down rather than chased — but it is the one place in this
  vertical where the reference does something we cannot reproduce.

- ~~**Why is prefill 393 and not ~2000?**~~ **Answered by L2a**: neither
  candidate. It is the hyper-connection block's glue and its `[10240, 4]` F32
  `inject` projection, plus a MoE kernel 2.53x off ours. See L2a above.
- ~~**How much of the 30.6% glue actually fuses?**~~ **L2c answers it for the
  first block** (4.24x on the nameable lines, the optimistic end of L2a's
  711-to-1159 spread), **L2f for the second** (2.40x, 1.60x at equal attention
  work) **and L3b for the linear layers** (1.37x on the layer, 3.90x on the
  gated norm alone). All three dense blocks are in and all three land inside
  that spread. **L5b closes the last of it**: the MoE's own glue — the GLU,
  the two weight multiplies, the shared gate's sigmoid and the two-half add —
  is fused into an accumulator, a store and a read that had to happen anyway,
  and the block is 1.09x at ubatch 512 and 1.55x at 2048. All five blocks are
  in, and the whole nameable graph is 87.2% replaced by 61.4%.
- ~~**How does a grouped MoE kernel schedule a 26x bucket imbalance?**~~
  **Answered at L5b: with a list, not a grid, and by padding.** The tiles are
  one device-built record per (expert, row block) that has rows, over a static
  upper bound with an early return — and every expert's range is rounded up to
  the row block so the epilogue can store whole cooperative-matrix fragments.
  **§2.2's 2.53x does not survive the real distribution**: it was swept at a
  uniform split, and at the measured one the block is **1.09x at ubatch 512
  and 1.55x at 2048**, because 274 experts read 505 MB of bank to serve 5120
  rows and the arithmetic intensity is the routing's rather than the tile's.
  The padding it costs is 2.33x the rows at 512 and 1.90x at 2048, and it is
  nearly free time, because a cold expert's tile is reading 0.9 MB of Q4_K to
  multiply seventeen rows either way.
- **Why is the grouped GEMM 2.5x off its own byte floor?** L5b-7: `up` moves
  686 MB of quantised bank in 7.2 ms, 70 GB/s of a 236 GB/s bus, with 78
  GFLOP of matrix work that would take 2.0 ms at §2.7's best rate and could
  hide entirely under the loads. What sits in between is the unpack — 1.22 G
  weights a dispatch, each a shift, a mask, a convert, an fma and a 16-bit LDS
  store, issued by the same wave that is waiting on the load. Three things
  that should have closed it did not (L5b-4): more waves a workgroup, a wider
  K-step and a conflict-free LDS pad are 1.15-1.24x, 1.16x and **5.4x**
  against respectively. What has not been tried is issuing the loads a K-step
  ahead of the unpack that consumes them.
- **Would a float atomic delete the combine?** L5b's down projection stores
  whole fragments in permutation order because a scattered store would need an
  LDS round trip, so a token's eleven contributions are summed by a ninth
  dispatch — 22.1 ms a graph at ubatch 512 and 66.4 at 2048, reading a 58 MB
  tensor that exists for no other reason. `VK_EXT_shader_atomic_float` would
  delete both, at the cost of a summation order that is no longer the
  reference's — which L5a's tie analysis is the reason to be careful about.
  The extension is not enabled in `vk.DeviceFeatures` today.
- **Does the hot expert want to be treated as a second shared expert?** At
  ubatch 512 expert 454 is chosen by 95% of tokens and ranked *first* by 54%.
  It has the same shape as the shared expert, which runs for 100%. Running the
  two together as one dense pair and routing only the other nine would delete
  a gather for half the block's tokens — and would be wrong on the 5% that do
  not select it, so it is a fast path with a correction rather than a
  simplification. **Still unmeasured after L5b**, which treats it as one more
  group of the grouped GEMM like any other — sixteen tiles of it at ubatch 512
  against one for each of the other 273.
- **Should our kernels follow the reference onto an fp16 accumulator — part
  two.** L5a-3 is the sharpest data point yet: at the MoE, modelling the
  reference's arithmetic makes the fit **worse** (0.57-0.73x), because the
  operand error and the accumulator error are independent. So an f32
  accumulator is strictly nearer the model here and there is nothing to
  reproduce — which leaves the question purely one of throughput, and L2f-5
  and L3b-4 both found register pressure binding.
- **Should the indexer's two projections leave the fused matrix?** They are
  the only BF16 weights in the model, and L2f-3 made them fp16 column ranges
  of the layer's one [13952, 2560] matrix — which is most of why the
  projection is 1.53x. **L4b-4 prices what that costs**: the device cannot
  take L4a-6's bf16 activation, so `indexer_k_raw` sits at L4a-6's own
  *unmodelled* 9.3e-04 rather than its modelled 4.5e-07, the score is 24x
  further from the reference and the selection 17x. The layer's output does
  not currently care (L4b-5, 9.87e-04 rms at 4 k against 1.14e-03 at 7
  tokens); whether the *model* does is L8's perplexity run. Un-fusing them
  would cost a dispatch — llama.cpp spends 5.03 ms a graph on those two lines.
- **The select kernel is 4-6x off the bus, and it is 0.1% of a graph.** Three
  of its four radix passes walk every cell to test a prefix almost none of
  them match; a compaction after pass 1 would fix that, and so would a
  **block-granular** histogram, since all `ratio` cells of a whole block carry
  one score and a 4096-cell row is 1024 distinct values plus a split tail.
  Written down rather than done, on L2f-5's precedent.
- **QSA becomes an optimisation at decode, and only there.** L4b-3 measured
  the selection *costing* the attention kernel 1.21x at prefill, because
  `build_attn_qsa` runs dense flash attention over a mask and so do we, and a
  32-cell key block is 8 indexer blocks — at ~50% density the chance that all
  eight are unselected is ~0.4%, so nothing can be skipped. At decode one
  token reads 2051 of up to 262 144 cells, the bitmask is 32 KB, and skipping
  unselected key blocks is the whole point. L7's.
- **Do the two projection ladders generalise?** L2f-4 found the same kernel
  wanting opposite BM schedules on two weights that differ only in falling
  either side of the 32 MiB MALL — widest-wins at 71.4 MB, narrowest-wins at
  31.5. **L3b-7 checks the DeltaNet's two and the rule holds**: the 84.5 MB
  fused projection wants the widest rung by 2.6x at 512 tokens and 3.3x at
  2048, and `ssm_out` at 31.5 MB picks the narrow rung at 512 and the wide one
  at 2048 — the same rungs at the same lengths as `attn_output`, which is the
  same shape. Still unchecked: the MoE bank and the lm_head.
- **The indexer's score is the one line we lose on**, 0.8 ms against
  llama.cpp's 0.7 (and 1.4 against 0.7 at a 4096-cell cache, which is twice
  the blocks). It is a scalar workgroup per token at 4.1 TFLOP/s where the
  matrix cores do 39, and it is an obvious cooperative-matrix rewrite — but it
  is 0.07% of a prefill graph, so it is written down rather than done.
- **Should our kernels follow the reference onto an fp16 accumulator?** L4a-5
  says the reference does, everywhere a weight is quantised, and that it costs
  it ~0.44% relative on every projection at prefill. Ours accumulate in f32 and
  are therefore *nearer the model and further from the oracle* — which is
  fine for a tensor comparison and is an open question for **throughput**:
  an fp16 accumulator halves accumulator register pressure, which L2f-5 and
  L3b-4 both found to be the binding constraint on the ladder. Worth a rung on
  the attention and projection ladders before L6 freezes them. The accuracy
  half is L8's: **`PPL 4.0340` was measured with fp16 accumulation on**, so
  matching it does not require avoiding it.
- **The rotary table wants a row per cache cell**, which is right for prefill
  and wrong for a 262144-cell context (67 MB). L7 either builds it per batch,
  as llama.cpp does, or goes back to computing the angle per lane and pays for
  a float32 `pow` at a large position.
- **Should the residual be fp16?** L2c measured the MALL cliff on our own
  kernel (575 → 167 GB/s on the norm between 512 and 1024 tokens), **L2f found
  it again on a different kernel** (the attention pack, 379 → 172 GB/s across
  the same boundary), and it is worth ~1.5x on this block at a long ubatch. But the residual accumulates
  across 97 combines, so it is L6's decision and L8's perplexity run, not a
  kernel's.
- **Q8_0 weights in the block's kernels.** L2c dequantises to fp16 at upload —
  13.4 MB a mixer against 7.1 — and the down projection is weight-read-bound,
  so reading the checkpoint's Q8_0 directly is the obvious next lever. It is
  also what phase 2's W4A8 bank will need here anyway. **L2d sharpens it**: its
  fused projection is 1.06x llama.cpp's while reading *twice* the weight bytes,
  in a dispatch that is 80% weight-bound.
- **How many other kernels have the grid the wrong way round?** L2d found
  2.9-4.0x on a convolution and 1.26x on a reduction by swapping which grid
  axis is fast, so that the resident workgroups cover one token's row instead
  of one channel block of forty tokens. Nothing else in this vertical has been
  checked for it, and §5.1b says the penalty is a property of the traversal
  rather than of the kernel.
- ~~**When does the selection start to bite?**~~ **Answered at L4a, by the
  dump vLLM's formula was waiting for.** At token **2051** — the width — and
  the shape is vLLM's: `(i+1) mod ratio` tail cells, **exactly 512** whole
  blocks, and a split of at most `ratio-1` cells off the lowest-scoring block
  in the selection. It is `topk_radix_select.comp` rather than a sort, ours
  reproduces it **set for set on all 2045 biting rows**, the reference's
  *order* is irreproducible run to run while its visible set is stable, and
  **0.0022% of it moves** when the scores are our own. What is left is only
  the kernel, which is **L4b**: 1.94x the reference's two ops, exact against
  the CPU selection on all 4096 rows, 0.0381% off llama.cpp's — and a
  selection that costs the attention kernel 1.21x rather than saving it
  anything. [Write-up](research/l4b-qsa-gpu.md)
- ~~**The recurrent state.**~~ **L3a-6 settles the DeltaNet half and L3b-6
  does it on the device**: both the [128, 128, 48] state and the convolution's
  three-column window carry, and 3 + 4 tokens reproduce 7 bit-identically —
  in Go and in the kernel — with a control showing the carry matters. On the
  device the window is not a second tensor at all: it is `Conv-1` rows of
  *negative token index* in front of the fused projection's own output. L2d's PLE convolution reaches 9 tokens back and is still only zeros
  at position zero; wiring both into a ring buffer is L7.
- ~~**What else is the backend computing in fp16?**~~ **L3a-5 answers it with
  a rule**: `ggml_vk_mul_mat` takes the f32 *vector* path at up to
  `mul_mat_vec_max_cols = 8` output columns and the fp16 coopmat GEMM above
  it. So the answer is "everything wider than 8 columns" — which is every
  projection in the model at a real ubatch, and **none of them in the 7-token
  dump**. ~~What is now open is the consequence.~~ **L4a measured it**, and it
  is worse than "an fp16 operand": on the other side of the threshold every
  quantised matmul accumulates in **fp16**, so the reference sits ~5.4e-03 rms
  from the f32 model and no operand model of ours can get closer. The BF16
  indexer projections take a **bf16** activation there, which is modelled and
  worth 2061x. **Every 1e-6 tolerance in L2b, L2d, L2e and L3a is a fact about
  a 7-token dump**; at prefill the number is 5e-3 and the gap is the oracle's.
  What is still open is only whether the same holds for the *elementwise*
  lines, which have no threshold and were never measured at width.
- **The oracle's build is pinned, the checkout is not, and they have
  diverged.** `/home/kube/repos/llama.cpp` is at **`d1d3c3396`** while
  `build/bin` is still **`cff184438`**, so reading `src/models/qwen4exp.cpp`
  today reads a *different implementation* from the one the oracle runs — it
  already carries #28068's GDN fix and a rms_norm fusion. `include/llama.h`
  and `ggml/include/ggml.h` have moved too, so **L4a builds `eval_dump.c`
  against headers extracted from the build's own commit** (`git archive
  cff184438 include ggml/include`) rather than from the working tree, and
  every source citation should be `git show cff184438:…`. L3a-4 found that
  `cff184438` — the build behind the trace, `pp2048 388.60`, `tg128 25.15` and
  **`PPL 4.0340`** — predates llama.cpp's fix to `build_gdn_l2_norm` (#28068),
  and that reproducing the pre-fix `max(|x|, eps)` is worth 15x on a DeltaNet
  layer. `TestDeltaNetL2NormIsTheBuilds` fails loudly if the build moves. The
  open part: **a rebuild invalidates the trace and the baseline together**, so
  regenerating one means regenerating both, and L8's perplexity comparison
  has to be against a number from the same binary as the checkpoint it grades.
- **Why is `attn_pregate` 1.03e-04 where everything around it is 1e-06?** Its
  inputs are all modelled and rounding the query to fp16 moves it by 3%. What
  is left is the flash-attention kernel's own accumulation order, which the
  dump enables by default. Bounded, not explained — the same category as
  L2b's three residual mixers.
- **Where does llama.cpp's missing 1.5 s at `-ub 2048` go?** The GPU graph is
  4.556 s and the wall clock 6.083 s, against ≤7% host time at `-ub 512`. It
  does not scale with graph count. Not ours to fix, but it means the honest
  baseline is the `-ub 512` one.
- **Does 111.32 GB — the whole GGUF, n-gram table included — hold live?** 80
  GiB is demonstrably residable (L0a) and 105 GiB reservable (L0b); L1 ran at
  77 GiB with 39 GiB of page cache beside it. If the whole thing fits, D2
  becomes an efficiency choice rather than a necessity and the host-side PLE
  gather can be deferred past L2.
- **Why do three of the seven hyper-connection mixers keep a residual?** Four
  reproduce llama.cpp's gate to 6.2e-08 rms under `RefQ8`; `blk.0.hc_attn` and
  `blk.2.hc_attn` sit at ~2e-05 and `blk.3.hc_ffn` at 3.4e-04, with identical
  code, identical Q8_0 weight types and an `hc_norm` input agreeing to 8e-08.
  An int8 grid amplifying a last-bit accumulation difference fits, but a count
  of near-boundary `lo` values does not track it. Bounded, not explained.
- **What is `down_proj`'s activation error really costing?** L0d settled W4A8
  at 1.13x mean, 1.6x worst, and the worst site is `down`, whose SwiGLU input
  has no norm in front of it. Finer activation groups on that one input, or a
  rotation, are the unmeasured mitigations.
- **Fix `quantizeQ8`'s subnormal scale** — two lines (floor it at fp16's
  smallest normal), but it changes the CPU reference every W8A8 correctness
  check is built from. Becomes urgent if phase 2 transcodes UD-Q4_K_XL's Q8_0
  dense tensors through it.
- **Does the scale plane's layout matter at *decode* too?** L0c is all GEMM.
  §1.8 measured the grouped GEMV paying 1.045-1.116x on the same axis and put
  it down to bytes alone, but it has not been tried with a k-major plane.
- **How much of the Q8_0 dense allocation is actually needed?** D3 assumes
  ~4.25 bits is enough; **4.0340** against L8's re-measurement is the only
  honest test.
- **`§3.4 finding 4` says prefill is "98% memory-bound by weight bytes";
  §2.2's direct measurement of a Q4 block says 2.2x above its memory floor.**
  L2a settles which is nearer: at 2048 tokens the expert bank is 64 GB, a
  **273 ms** floor, and §2.2's kernel takes 642 — so **2.4x above the bus**,
  and §3.4's "98% memory-bound" is the one to retire. The model-level estimate
  built on it was 5x optimistic; L2a's bottom-up 1768 ms replaces it.
- **Batch and speculation.** At batch 4 the dense 76% amortises completely.
  Worth knowing whether the API will ever serve more than one stream before
  optimising the batch-1 path to death.
