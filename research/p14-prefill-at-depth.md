<!-- P14. 900 tok/s at 128k: the selection over blocks instead of cells, the
     per-cell gather and the value layout that had to move for it, the indexer's
     score on the matrix cores, and the buffer range that was silent when it was
     exceeded. Cited from llm/gpu_attn.go, shaders/llm_attn_selblk.comp,
     shaders/llm_attn_gather.comp, shaders/llm_attn_wmma.comp,
     shaders/llm_attn_score_wmma.comp and shaders/llm_attn_pack.comp. -->

[← TODO.md](../TODO.md) · [research index](README.md) · [P0](p0-ring-watchdog.md) · [P7](p7-context-depth.md) · [P11](p11-prefill.md) · [P12](p12-prefill-round-two.md) · [P13](p13-long-context-prefill.md)

# P14 — the gather, and 128k prefills past 900 tok/s

> **The target is met.** A 128 000-cell prefill runs at **826.0 tok/s** at ubatch
> 2048 against P13's 563.6 (**1.47x**), at **886.1** at 4096, and at
> **946.1** at 8192 — which is the 900 tok/s `GOALS.md` asked for. The
> falloff from depth zero goes **0.50x to 0.72x** at the same ubatch, and decode
> at 128k does not move (22.18 against 22.27). Four changes and **four measured
> refusals**, and the refusals are the interesting half: after the gather the
> attention kernel is bound by neither its matrix cores (doubling their work
> costs 2.7%), nor its gather traffic (halving it buys 4%), nor its barriers.

## Where P13 left it

P13 closed the arithmetic and named the route. At 128 000 cells, ubatch 2048,
48 layers (ms a token):

	floor (everything flat in depth)   0.803
	attn.attn                          0.566
	attn.select                        0.233
	attn.score                         0.078
	attn.expand                        0.051
	                                 = 1.731, and 563.6 on the clock

900 tok/s is 1.111. P13's own summary of what that needed was three things —
the selection change, a wider ubatch, and the gather — and priced the gather at
2.8x on `attn.attn` from a union of "~4 574 cells" against the 14 976 the
16-cell rung reads.

**Both of those cell counts were wrong**, which is the first thing P14
measured. `AttnGPU.SelUnion` prices the question directly, on the selection a
pass actually left behind, and at 128 000 cells a sixteen-row query tile reads
**15 145 cells** where its rows' union holds **7 049** and one row selects
2 051. So the gather is worth **2.15x**, not 3.3x; the union is 3.4x one row's
reach, not 2.2x. That shortfall is why P14 is four changes and not two.

## P14-1: the selection over block scores, and the expansion is deleted

`llm_attn_select.comp` radix-selects over `llm_attn_expand.comp`'s tensor: one
f32 a cache cell a token, `rows * nKV` of them, which at 128 000 cells and a
2048-row batch is **1.14 GB of arena** and five streams of it per token — four
radix passes and the emit.

That tensor has `nKV/ratio + 1` distinct values in it. All `ratio` cells of a
pooled block carry one block score (`llm_attn_score.comp`) and every cell the
causal mask drops carries -inf, so the select can run over the **block** scores
with a per-block **weight** — how many of its cells this token can see, which is
closed form — and reach the same threshold, the same ties and the same bitmask.
`ratio` is 4 here, so that is a quarter of the traffic and the expansion is then
not computed at all.

`shaders/llm_attn_selblk.comp` is that kernel. Three things have to hold and all
three are arithmetic rather than hope:

1. **The threshold.** A weighted histogram over blocks counts exactly the cells
   the per-cell histogram counted, because every cell of block `b` has key `b`
   and nothing else does. The total weight is exactly `ncols`.
2. **The tie fill.** The reference appends cells equal to the threshold in
   ascending *cell* index (L4a), and the emit still walks cells ascending — it is
   the keys that come from blocks. Two blocks whose scores happen to be equal are
   one histogram bucket and two runs in the emit, and the emit takes the lower
   run first, as the per-cell code did.
3. **The dead cells.** Everything past this token's position is -inf, whose key
   `0x007FFFFF` is the smallest there is, and they are all *after* the live
   cells — so they enter as one weighted entry and fill from the lowest index.
   No block can collide with that key: the only bias that makes one -inf is the
   `b >= nBid` arm, and those blocks have weight zero.

The emit is still the one pass per cell, because the output is — but it is no
longer a read per cell. Cells are grouped `ratio` to a block and a word is
`32/ratio` blocks, so the key is fetched when the block changes; and a word
wholly past this token's position is every cell -inf with no read at all.

Measured — 4 layers, one full-attention layer, `-pp 2048 -ctx 136000`, scaled to
the 48 layers that count, with `attn.attn` and `attn.score` as the controls:

| ms a token at 128 000 | cell select | block select | |
|---|---:|---:|---:|
| `attn.select` | 0.2808 | **0.0204** | **13.8x** |
| `attn.expand` | 0.0456 | — | gone |
| `attn.attn` (control) | 0.9516 | 0.9528 | 1.00x |
| `attn.score` (control) | 0.0792 | 0.0804 | 1.00x |

**13.8x and not the 4x the traffic argument predicts**, because two other things
went with it: the histogram is bounded at the last block a cell can reach, so a
shallow cache never visits the rest of the table, and the 16 KB of LDS that used
to cache 4 096 *cells* now caches 4 096 *blocks*, which is 16 384 cells.

`TestAttnGPUSelectBlocksIsTheCellSelect` is the gate and it is an equality —
30 405 462 selected cells over four chunk schedules, and the layer's own output,
identical to the last bit. The schedules are the point: at 4096 tokens in a
4096-cell cache the whole cache is whole blocks, so `nBid >= nBlocks` and the
*incomplete tail* never exists; a run in chunks of seven reaches a cell count
that is not a multiple of `ratio` at every step, which is where the tail block's
weight and the causal boundary inside a partly visible block are both live.

The old kernel stays as the control arm (`LLM_ATTN_EXPAND_CELLS=1`), which is
also what `Cells()` needs — and the arena is allocated only on that arm, because
that is the point.

## P14-2: the per-cell gather

P13 compacted the *blocks* the selection leaves anything in and measured 1.00x.
The kernel it fed still read every cell of a live block, and the 2.15x between
15 145 and 7 049 is what a block list cannot reach: the granularity a
cooperative-matrix tile can skip at is sixteen cells and the selection's runs are
four.

So the list is per cell. `shaders/llm_attn_gather.comp` emits, per query tile,
the ascending union of its rows' selections and a per-row bitmask over those
positions; `llm_attn_wmma.comp -DGATHER` then runs a **dense** attention over the
list — no axis walk, no block skip, no `blkAny` vote, and no causal comparison
anywhere. Two dispatches and not one, because the mask reads the list and a
pipeline barrier is the only handoff worth trusting between two passes over a
buffer.

Three things make it affordable, and none of them existed in P13.

**The value plane had to move.** A gathered cell's sixteen dims for one head-dim
group are 32 contiguous bytes and a selected run is four consecutive cells, so a
run is one 128-byte line — *in the key plane*. The value was stored transposed
(one dim's sixteen cells contiguous) because the reduction axis of `p.v` is the
cell and `coopMatLoad`'s ColumnMajor read the B operand straight; a run was then
four halves out of each of sixteen 32-byte rows, which is the whole tile read for
a quarter of it. RowMajor over a cell-major tile reads the **same matrix** —
element (k=cell, n=dim) at `k*16 + n` rather than `n*16 + k` — so the fix is one
line in `llm_attn_pack.comp` and one in the load, it is a permutation inside the
512-byte tile, and every exact gate stayed exact. It costs the *block* kernel
about 7% of `attn.attn` (0.9528 → 1.0236 ms a token at 128 000) and buys the
gather half its traffic; decode, which reads the same plane through P8's split
build, did not move at all.

**The staging is a head-dim group, not a chunk.** Staging a whole 32-cell chunk's
256 dims is 16 KB of LDS, and a single-wave workgroup that takes 16 KB is four
resident per 64 KB where this kernel gets twenty. Occupancy is the one thing a
prefill cannot trade (P13-3), so the staging rides the loop that was already over
the groups: `KTIL * 256` halves, 1 KB at the shipped rung, double-buffered so the
next group's global loads overlap the current group's matrix ops.

**The causal test is in the mask.** The gather ANDs each row's own extent into the
bits it writes, so the loop has one predicate where the block kernel had three,
and the row max loses its `break`.

### What it measures

4 layers, `-pp 2048 -tg 4 -ctx 136000`, `attn.attn` in ms a token scaled to 48
layers, three gathered rungs against the block kernel:

| | 64 000 | 128 000 |
|---|---:|---:|
| block kernel, 16-cell rung | 0.5796 | 1.0236 |
| gathered `qt1_kt1` | — | — |
| **gathered `qt1_kt2`** | **0.3372** | **0.4536** |
| gathered `qt1_kt4` | 0.4512 | 0.5940 |
| the gather's own two dispatches | 0.0144 | 0.0228 |

**2.15x net at 128 000**, which is exactly the cell ratio — this kernel's cost is
linear in the cells it visits and nothing else, which is the third time that has
been the answer (P13-2 predicted 1.49x from a cell count and measured 1.46x).

And the pass, 4 layers at 128 000 cells: 5 483 → 7 290 tok/s, two passes
agreeing to 0.3%.

**The rung is wider than the block kernel's and that is not a contradiction.**
P13 took the narrow 16-cell block because at depth the kernel was short of cells
it was *allowed to skip*; a gathered axis has nothing left to skip, so the block
width is decided by reuse again — which is the ladder L2f measured before depth
was in the picture. 32 cells wins: `qt1_kt1` gives up a cooperative-matrix tile
of reuse and leaves half the lanes idle in the staging loop, and `qt1_kt4` doubles
the LDS and the per-chunk register pressure for nothing.

### The one invariant it gives up

**The gather is the first kernel in this vertical that is not chunk-invariant,
and unlike P8's split it cannot be made so.** L7a's gate is that a token at cell
`pos` comes back bit for bit whether it arrived alone or seventh of seven, and
every ladder here holds it: a rung tiles the same arithmetic differently, P7's
skip drops blocks that contribute nothing, and P8's split strides *absolute* key
blocks precisely so that a slice does not depend on how the prompt was cut.

A gathered list is the union of a query tile's sixteen rows. A chunk that ends
inside the tile has fewer rows to union, so the same absolute tile gathers a
different list, the list is partitioned into different key chunks, and the online
softmax folds its rescales in a different order. The *set* each row attends over
is identical either way and so is every nonzero term in its accumulators — a cell
in the union that this row did not select has `P = 0` and leaves `blkMax` at -inf,
whose correction is exactly 1.0 — but the grouping moves, and fp32 addition is
not associative.

So it goes off with the other reassociating kernels under `Graph.PinSchedule`,
and the chunk-equality gates pin it. What is left in the shipped path is narrower
than it sounds: a prompt prefilled twice at the same ubatch is bit-identical, and
decode never takes this kernel at all — only changing the ubatch moves the last
places.

The gates that replace the equality:

- `TestAttnGPUGatherSelectsTheSameCells` is **exact**, on the part that can be:
  every tile, every row, the list is ascending and holds exactly the union and
  the mask is exactly the selection AND the row's causal extent. 6 298 621 live
  (row, cell) pairs over two chunk shapes — the same number L4a-4 counted when it
  priced the reference's own disagreement.
- `TestAttnGPUGatherIsTheBlockKernel` is the tolerance, and it is
  `TestAttnGPUSplitMatchesDense`'s argument: the two kernels are **4.1e-06 rms**
  apart where each is **9.895e-04** from llama.cpp, so they are 240x nearer each
  other than either is to the reference, and both are exactly the same distance
  from it.

## P14-3: the indexer's score on the matrix cores

After P14-1 and P14-2, `attn.score` is the second largest label in the attention
block at depth — 0.077 ms a token at 128 000 cells against the gathered
attention's 0.30 — and it is a scalar f32 dot product: one lane per pooled block,
walking four heads of 128 dims with `d += float(q) * float(k)`. It was written
where it did not matter, because at depth zero a 2048-row batch scores 512
blocks.

It is a GEMM and it always was: `[T, 128] x [128, nBlocks]` per head, with a
rectifier on the product and a sum over the four heads behind it. Both operands
are already fp16 in the arena and already laid out for the two `coopMatLoad`
layouts it wants — the query as a RowMajor A at stride `idxHeads * idxDim`, the
pooled key as a ColumnMajor B at stride `idxDim` — so
`shaders/llm_attn_score_wmma.comp` loads the query tile's 4 x 8 A-fragments
**once** into registers and spends eight `coopMatMulAdd` per head per block tile.
The rectifier is why there are four accumulators and not one 512-wide reduction.

It is **not** bit-identical to the scalar kernel and it is marginally nearer the
reference: research/l2e-attention.md established that llama.cpp evaluates this
f32 tensor on the fp16 matrix cores, and the kernel's disagreement with
llama.cpp's own selection went 9 762 → 9 760 of 6 298 621 visible cells. That
number is the gate, as a *size* rather than a zero, because a top-k is
discontinuous.

It **is** chunk-invariant, which is why it ships where the gather needs a pin: a
matrix product's rows are independent, so `C[m][n]` is the same sum over the same
`k` in the same order whichever row of the fragment `m` lands in.

`LLM_ATTN_SCORE=scalar` is the control arm.

## P14-4: the buffer range that was silent when it was exceeded

This one is not an optimisation and it cost a sweep.

P13 noted that `maxStorageBufferRange` is 4 GiB - 4 on this device and that the
KV planes share a buffer with the attention arenas, "which caps this at about
148k cells". What the note did not say is what happens *past* the cap, and the
answer is nothing: a VkBuffer larger than the range is legal to create, binding
more of it than the range at a descriptor is not, and the driver **clamps rather
than failing**. A kernel reading past the range gets zeros.

`cmd/llm -depth` raises `-ctx` to whatever the sweep needs, and a five-depth
sweep to 128 000 at ubatch 4096 needs `128000 + 5*(4096+8) = 148520` cells. That
allocated a 4.39 GB fp16 arena, the descriptor reached 4.29 GB of it, and the
deepest layers attended over a cache of zeros. The run did not crash, did not
warn, and came back **1 124 tok/s at 128 000 cells** — 1.43x the correct number,
because an attention over zeros is a cheap attention and a degenerate indexer
selects a contiguous window. The tell, once there was an instrument for it, was
`SelUnion`: a sixteen-row tile's union read **2 063** cells against one row's
2 051, where a healthy run reads 7 049.

`checkBufferRange` is that note turned into an error, on both attention arenas,
naming the cell cap it computes from the cache's own geometry. **The failure mode
it catches is a benchmark that gets faster**, which is the one direction nobody
checks.

## P14-5: four things that are measured inert, and the probe that explains them

The gathered kernel is 2.15x and it is **34% of this device's matrix-core peak**,
so it looked like there was another 2x sitting in it. There is not, and four arms
say why. Each is one build, `-pp 2048 -ctx 136000 -layers 4`, `attn.attn` in ms a
token at 128 000 cells scaled to 48 layers:

| arm | `attn.attn` | |
|---|---:|---:|
| the shipped gathered kernel | 0.4524 | 1.00x |
| a subgroup staging barrier instead of a workgroup one | 0.4536 | 1.00x |
| two head-dim groups a staging barrier (`GRP=2`) | 0.4512 | 1.00x |
| four (`GRP=4`) | 0.6072 | **0.75x** |
| two query heads a workgroup, sharing the staged key (`HPW=2`) | 0.4344 | 1.04x |
| four (`HPW=4`) | 0.6012 | **0.75x** |
| **the QK matrix work, doubled** | **0.4644** | **0.97x** |

The last row is the probe and it is the whole answer. **Doubling the matrix work
costs 2.7%**, so the matrix cores are ~5% of this kernel and nothing that makes
them busier or idler will move it. The barrier arms then say the same for
latency — for the fourth time in this vertical, after P11-4's hoist, P12-4's
reduction tree and P13-3's block list: *latency that matters at decode does not
matter at prefill, because prefill has occupancy.* And `HPW=2`, which halves both
the global gather reads and the LDS writes by sharing them across two query heads
of one kv head, buys **4%** — so the gather's own traffic is not the cost either.

What is left, by elimination, is the **LDS round-trip for the fragments and the
per-head softmax scaffolding**, neither of which any of these arms touches: a key
chunk reads 32 KB back out of LDS into cooperative matrices per head however it
got there, and the row max, the rescale, the exponential and the mask all run per
head over `BM x BN`. Both scale with the union, which is why the kernel's cost is
linear in the cells it visits and why 2.15x fewer cells was 2.15x.

The two knobs stay as a ladder with a measured answer (`LLM_ATTN_GATHER_GRP`,
`LLM_ATTN_GATHER_HEADS`, both defaulting to 1) because the 0.75x arms are the
informative ones: LDS is occupancy, and this kernel is single waves that want to
be many.

## Where 128k stands

48 layers, shipped banks, `wiki.test.raw`, one sequence walked forward,
`-pp 2048 -tg 8 -ctx 139000`, against P13's table at the same shape:

| depth | P13 pp | P14 pp | | P13 tg | P14 tg |
|---:|---:|---:|---:|---:|---:|
| 0 | 1129.3 | 1154.2 | 1.02x | 29.29 | 29.31 |
| 32 000 | 825.8 | **915.6** | 1.11x | 25.84 | 25.80 |
| 64 000 | 719.6 | **892.1** | 1.24x | 24.68 | 24.46 |
| 96 000 | 607.9 | **832.3** | 1.37x | 23.24 | 22.80 |
| **128 000** | **563.6** | **826.0** | **1.47x** | 22.94 | 22.18 |

The falloff from depth zero is **0.72x** where P13 left it at 0.50x and P0 could
not reach the depth at all. **Decode is the control and does not move**: the
gather is prefill's kernel, P14-1's select is bit-identical, and P14-3's score is
a rounding away.

And against the **ubatch**, at 128 000 cells alone (two runs each; `-ctx 139000`
has room for any of these at one depth, and for none but 2048 across five):

| ubatch | pp tok/s at 128 000 | decode |
|---:|---:|---:|
| 2048 | 826.0 | 22.18 |
| 4096 (the shipped `-llm-batch`) | **886.1** | 20.74 |
| 8192 | **946.1** | 18.48 |

That is P12's trade at depth and it is the same trade: a wider ubatch buys the
flat floor and costs decode, so 8192 is a batch summariser's setting and 4096
stays the interactive default.

Where a prefill token goes at 128 000 cells and ubatch 2048, against P13's
budget (ms a token):

	                    P13      P14
	floor              0.803    0.818     everything flat in depth
	attn.attn          0.566    0.303     the gather
	attn.select        0.233    0.021     the block select
	attn.score         0.078    0.034     the matrix cores
	attn.expand        0.051    —         deleted
	attn.gather + mask   —      0.016     what the gather costs
	                 = 1.731  = 1.192

## What is left

- **The union is 3.4x one row's selection** and that is the gathered kernel's
  floor, because the fragment is sixteen rows and the rows are consecutive
  tokens. 2 051 cells a row against 7 049 a tile at 128 000 cells: a per-row
  attention would do 3.4x less work and cannot use a matrix core. Nothing in
  this vertical closes that gap; it is the shape of the hardware against the
  shape of QSA.
- **The floor is now most of a prefill token at depth.** At 128 000 cells and
  ubatch 2048 the attention block is under a third of it and `moe.up` alone is a
  quarter; P11 and P12 established that the MoE prefill GEMM is bound by its
  per-tile slab unpack and that gutting the whole Q4_K scale path is only 1.21x.
- **`maxStorageBufferRange` caps the cache at about 148k cells** and now says so.
  256k needs L6a's array-of-buffers, one a layer — and the guard is what makes
  that a decision rather than a silent wrong answer.
- **`attn.gathmask` is 0.015 ms a token at 128 000 cells**, 5% of the gathered
  attention, and it is the one term here that is a scattered read by nature: the
  per-row mask over the gathered positions. It is already done once for all
  twenty-four heads and already caches the mask word across a run.

## Reproducing

	go build -o /tmp/pp ./cmd/llm        # once, never `go run` per arm
	export LLM_BANK_CACHE=models/Qwen3.8-Flash-Next-GGUF/bank-cache
	export LLM_DENSE_BANK=deltanet=q4_k/32,hyper_conn=q5_k/32,lm_head=q5_k/32,full_attn=q5_k/32,qsa_indexer=q5_k/32
	export LLM_MOE_BANK=gate_shexp=q4_k,up_shexp=q4_k,down_shexp=q5_1,down_exps=iq4_nl
	M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

	# 128k alone, which is what a wide ubatch has room for at this -ctx.
	# **Do not sweep five depths at ubatch 4096**: the bench raises -ctx to
	# 148520 cells and P14-4 is the error that now stops it.
	/tmp/pp -model $M -depth -depths 128000 -pp 2048 -tg 8 -ctx 139000

	# the fast loop, 40 seconds an arm, and it reproduces the attn.attn ratios
	/tmp/pp -model $M -depth -depths 0,64000,128000 -pp 2048 -tg 4 -ctx 136000 -layers 4

	# the control arms
	LLM_ATTN_GATHER=0            # P13's block kernel
	LLM_ATTN_GATHER_KERNEL=qt1_kt4
	LLM_ATTN_EXPAND_CELLS=1      # P13's cell select, and what Cells() needs
	LLM_ATTN_SCORE=scalar        # P14-3's control

And the whole-model check, which is the cheapest bit-exactness test in the
repo and is worth spelling out here because three of P14's four changes claim
to be bit-exact. `TestGraphLogits` runs 7 tokens through all 48 layers against
llama.cpp's own dump, and it fails on `main` for a reason that predates this
work (nine of ten top tokens, argmax agreeing) — so what matters is that its
*diagnostics* reproduce the base **to the digit**, and they do:
`maxAbs 2.507e+00 at 141557`, `rms 5.148e-01`, top ten
`[561 11751 1049 1061 248046 271 9338 198 19592 220]`. That is the block
select, the value plane's new layout and the arena changes all confirmed
byte-identical through the whole model, and it is the only failure in
`go test ./llm/` (666 s, `-timeout 60m`; `LLM_BANK_CACHE` must be **absolute**
or the run writes 23 GB into `llm/models`).

The gates:

	go test ./llm/ -run TestAttnGPUSelectBlocksIsTheCellSelect      # P14-1, exact
	go test ./llm/ -run TestAttnGPUSelectBlocksSkipsTheExpansion    # P14-1's arena
	go test ./llm/ -run TestAttnGPUGatherSelectsTheSameCells        # P14-2, exact on the set
	go test ./llm/ -run TestAttnGPUGatherIsTheBlockKernel           # P14-2, the tolerance
	go test ./llm/ -run TestAttnGPUGatherIsDispatched               # P14-2, the shape
	go test ./llm/ -run TestAttnGPUSelectionAgainstTheReference     # P14-3's size
	go test ./llm/ -run TestAttnGPU                                 # the block
	go test ./llm/ -run TestGraphIsAChunkSplit                      # the graph
