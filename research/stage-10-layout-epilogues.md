# Stage 10 — the block's two layout passes, as epilogues

**Result: the DiT block is 16 dispatches, not 18.** `pack v` and `narrow ctx`
computed nothing — each read a tensor and wrote the same numbers in the shape
the next kernel wanted — and both are now the store instruction of the kernel
that produced the tensor. A block goes **53.17 ms → 51.80 ms** at 4224 tokens,
a step **1.70 s → 1.66 s**, and an image **14.49 s → 14.26 s**.

| | unfused | fused |
|---|---|---|
| dispatches per block | 18 | **16** |
| one block, layers phase (4224 tokens) | 53.17 ms | **51.80 ms** |
| one step, 34 blocks | 1.70 s | **1.66 s** |
| an image, 1024², 8 steps | 14.49 s | **14.26 s** |

Two runs each, agreeing to 52.82/53.52 and 51.78/51.82 ms a block, and to
14.43/14.55 and 14.18/14.34 s an image. Both columns are the same binary:
`go run ./cmd/ditstack -unfused-layout` and `go run ./cmd/zimage
-unfused-layout` keep the old graph, which is the oracle the new one is
compared against.

## The item was mispriced by a factor of a thousand, and the fix was arithmetic

`PIPELINE.md` and four handoffs in `TODO.md` carried this item as **"~1.0 s an
image of pure layout"**, and made it the largest single thing left in the
pipeline at 7%. It is **~0.29 s**, and 2%.

The measurement it came from was right. `research/stage-4-dit-graph.md` wrote
it down correctly — "`pack v` (0.59 ms) is pure layout and `gemm v` could do it
in its epilogue; `narrow ctx` (0.41 ms) is the same for attention's output.
Together ~1.0 **ms**" — and the next document to quote it turned the unit into
seconds. Nothing re-derived it afterwards, because 1.0 s an image is a
plausible number for a thing that costs 1.0 ms in a 50 ms block run 272 times:
34 blocks x 8 steps x 1.07 ms is 0.29 s, and the error is that the block count
was already in the millisecond figure's future, not its past.

So the lesson is not about layout at all. **A number that moves between
documents should carry the dimension it was measured in**, because the one
thing a reader cannot check is a unit that was never wrong in the file it came
from. This item was at the top of the backlog for four sessions on the
strength of a units slip.

It was still worth doing — 0.23 s an image for two compile-time flags is the
best ratio left in the block — but it is not what the budget said it was, and
what *is* the largest item left has changed as a result. See the end of this
file.

## What the two passes were

Both existed because a kernel's natural output is not its consumer's natural
input, and the block had no way to say so:

- **`pack v`** read `v`'s fp32 C, `[tokens, 3840]`, and wrote it back as fp16
  16x16 fragment tiles, per head, each tile transposed —
  `dit_pack_f16.comp` mode 1, which is the layout `p.v` needs because that
  multiply reduces over the *token* axis (stage 3c).
- **`narrow ctx`** read attention's fp32 context, `[tokens, 3840]`, and wrote
  it back as fp16 at the output projection's padded row stride.

Per block per step at 4096 tokens that is 63 MB written, 63 MB read and 32 MB
written, twice over: **252 MB of traffic that moves no information**. Across
34 blocks and 8 steps, **68 GB an image**. At this part's 236 GB/s that is 0.29
s — which is both what the two dispatches measured (0.64 + 0.43 = 1.07 ms a
block) and what removing them returned. The saving is the bandwidth arithmetic
and nothing else.

## Both are store layouts, not transposes

The reason this is two compile-time flags rather than two kernels is that in
both cases the data is **already in the right registers**, in the right
16x16 shape, and only the address arithmetic of the store is wrong.

**`dit_gemm.comp -DC_PACK=1`.** A wave holds its slice of C as 16x16 fp32
accumulator tiles. A pack output tile is also 16x16. The two are cut on the
same boundaries, because a 16-column slice of C lies inside one head whenever
`headDim` is a multiple of 16 — it is 128 — so accumulator tile (i, j) *is*
pack tile (token tile, component tile) of head `n0 / headDim`. And the
transpose is not a transpose: a column-major `coopMatStore` with stride 16 puts
element (token, component) at `d*16 + m`, which is exactly what mode 1 writes.
So the epilogue is a different `coopMatStore` and eleven lines of index
arithmetic.

**`dit_attention_wmma.comp -DOUT_F16=1`.** The epilogue's only work is
dividing each row of O by its softmax denominator, and the kernel already has
the machinery for a per-row scale: §3.3's third awkwardness, where the online
rescale becomes a matrix whose every row is constant and `O * Ccorr` is a
component-wise multiply. `1/rowSum` is the same shape of thing. So the fp32
path's staging loop — store each output tile to LDS, then four lanes per row
writing four floats each — is replaced by one `O * Cinv` and one
`coopMatStore` of the fp16 result at the A operand's row stride. The fused
kernel is **shorter** than the one it replaces, which is unusual for a fusion
and is the sign that the staging loop was only ever there to reach a per-row
scalar.

The pad rows past the sequence are written rather than branched around. They
are the A operand's pad rows: the output projection computes their products
and nothing reads them, and for `v` the attention kernel's tail mask already
zeroes the columns of P that would weight them, so what is in them cannot
reach a real token.

## Where the time went

The dispatch profile of one block in the layers phase, `go run ./cmd/ditstack
-v`, 4224 tokens:

| dispatch | unfused | fused |
|---|---|---|
| `gemm v` | 3.31 ms | **3.08 ms** |
| `pack v` | 0.64 ms | — |
| `attention` | 7.21 ms | 7.19 ms |
| `narrow ctx` | 0.43 ms | — |
| block total | 53.17 ms | **51.80 ms** |

The two removed dispatches are 1.07 ms of the 1.37 ms. The rest is `gemm v`,
which now writes 32 MB of fp16 where it wrote 63 MB of fp32 — a quarter of a
millisecond at the bus. Attention does the same and does not move: its store
was never on its critical path, which is consistent with stage 3c finding the
kernel bound by its register file rather than by memory.

## The correctness argument, and the one thing it caught

Neither epilogue changes an arithmetic operation, so the bar here is higher
than a tolerance: the question is what *exactly* may differ and why.

**`v` is bit-identical**, 1,474,560 halves of the packed plane, every head.
The pack narrowed the fp32 C; the projection narrows the accumulator the fp32
C came from; same float, same rounding.

**The context is not**, and the 84 halves in 1,228,800 that differ are the
finding. Every one of them is an **exact fp16 tie in the fp32 context** —
`TestGPUBlockFusedLayout` asserts that, element by element, rather than
bounding the difference. What that means is a **double rounding in the old
path**: it rounded `o/rowSum` to fp32, stored it, and a second dispatch
rounded that to fp16, and where the fp32 result landed exactly halfway between
two halves, round-to-even broke a tie that the fp32 rounding had itself
manufactured. The fused path hands the fp16 store the unrounded product, so it
never sees the tie.

That was established by experiment, not by reading the ISA. Adding an fp32
store of the same product to the fused kernel — which forces the intermediate
to exist — makes **all 84 differences disappear**, and the fp32 contexts of the
two builds are bit-identical to begin with, so the difference cannot be
upstream of the epilogue. The compiler is contracting the multiply and the
narrow into one mixed-precision instruction.

So the fused path is the *more* correctly rounded of the two. Its cost:

| | |
|---|---|
| packed `v` | bit-identical, 1.47 M halves |
| fp16 context | 84 of 1.23 M differ (6.8e-5), all exact fp16 ties, 1 ulp each |
| block output | rel 9.9e-4 of RMS (stage 4b's own fusions: 1.6e-3) |
| pipeline vs diffusers, 256² | 0.0158 → **0.0215**, against a 3e-2 bound |

The last row is the one to read carefully. It is **not** an accuracy loss —
per operation this path rounds better — it is a chaotic trajectory landing
somewhere else: 84 halves of one operand, one ulp each, amplified through
eight denoising steps, the same way stage 9's end-to-end figure moved 0.0252
to 0.0158 when *it* changed what a step computed. What it does do is eat a
third of the remaining margin against that bound, and the bound is where the
measurement put it rather than where anything requires it to be.

## Why q and k are not the same case

They look symmetric with `v` and they are not. Between the projection and the
fragment pack, q and k get a per-head RMS norm and a rotary rotation — and the
norm is a **reduction over the head**, which a GEMM epilogue cannot do: the
wave holding accumulator tile (i, j) holds 16 of the head's 128 components,
and the other seven tiles are in other waves or other workgroups. Something
has to read the tensor back whatever the store writes. That is what
`dit_qk_pack.comp` is, and stage 4b already fused the three passes it replaced
into one.

The same argument says which *other* operands in this engine could take an
epilogue: exactly those whose consumer wants a layout and not a reduction.

## What is now the largest item left

With these two gone, a block at 4224 tokens is 16 dispatches, and it splits
**74% GEMM** (37.97 ms across seven projections), **14% attention** (7.19 ms)
and **12% elementwise** (5.93 ms across eight passes). An image is 14.26 s, of
which the DiT is 13.37 s, 94%.

There is no layout left to remove. What is left is the arithmetic:

- **The GEMMs at 73-76% of the WMMA ceiling.** 1.24 s a step, **9.9 s an
  image, 69% of everything**. The quarter of the ceiling they do not reach is
  2.6 s an image, which is an order of magnitude more than every other item in
  the pipeline put together — and also the hardest, since stage 4 already
  applied the three things that got them here.
- **Attention at 38 TFLOP/s** against the GEMMs' 40.6-42.0: 1.9 s an image,
  worth ~0.2 s if it caught up. Its store is now fp16 and its epilogue no
  longer touches LDS, and neither moved it at all, which is a second
  measurement agreeing with stage 3c that the register file is what binds it.
- **The elementwise passes**, 5.93 ms a block and **1.6 s an image**: eight
  passes that each read the residual stream and write it. `swiglu` (1.45 ms)
  and the two gated residuals (0.89 and 0.85) are half of it. Unlike the two
  this stage removed they do compute something, so what they want is fusion
  into a neighbour rather than deletion — and the neighbour on one side is
  always a GEMM, which is where §2.6 and this stage both ended up.
- Outside the DiT, the VAE's elementwise fusions are still ~200 ms, 1.4%.

## How it was measured

    go run ./cmd/ditstack -v -reps 3 [-unfused-layout]
    go run ./cmd/zimage -reps 2 [-unfused-layout]
    go test ./zimage/dit -run TestGPUBlockFusedLayout -v
    go test ./zimage/pipeline -run TestPipelineAgainstDiffusers -v

Two runs of each arm, nothing else on the GPU. `-unfused-layout` is a live
switch on the graph rather than a build: both kernels are compiled and both
pipelines built, which is what lets one test run the two paths back to back
over the same arenas.
