<!-- Pipeline stage findings, not an IDEAS section. Referenced from
     PIPELINE.md, which stays short by pointing here. -->

[← PIPELINE.md](zimage-pipeline.md) · [research index](README.md) · pipeline stage 3

# Stage 3 — the DiT block and its attention

`zimage/dit` implements Z-Image's transformer block on CPU and its attention
stack on Vulkan. Both matched diffusers on their first run.

**Stage 3c is done**: attention runs on the matrix cores at **38.8 TFLOP/s,
70% of this part's measured WMMA ceiling** — the same fraction the best GEMM
in the suite reaches — which is **30.4x** the scalar kernel it replaces. One
block's attention at 4096 tokens went from 202 ms to **6.64 ms**.

Two corrections to what this file said before, both about measurement rather
than about a kernel:

- **The GFLOP/s figures were 1.5x optimistic.** `cmd/ditbench` counted
  `6*n*n*dim` flops for a pass that does not exist. Attention is two GEMMs,
  `2*n*n*headDim` each per head, so `4*n*n*dim`. The scalar flash kernel's
  "707 GFLOP/s" is 477 at the old wall-clock timing and 1276 at the GPU
  timing below; the table at the end of this file is in the new units.
- **The wall clock was never measuring the kernel.** `Apply` copies q, k and
  v in and reads the result back, and read-back out of the device-local
  host-visible arena runs at **0.2 GB/s** against 11.5 GB/s for writes — so at
  4096 tokens, 63 MB of read-back is 344 ms and the whole GPU graph is 8.5.
  Every tile geometry in the WMMA ladder measured 347 ms wall and looked
  identical; on GPU timestamps they span 6.6 to 19.3 ms. `GPUAttention.Profile`
  now times each dispatch, and nothing in the pipeline reads a block's output
  back to the host, so the 0.2 GB/s is a property of the harness. It is,
  however, a property every future harness will have, and stage 6's PNG path
  will need a host-cached staging buffer rather than a direct read.

  **Refined by stage 9** ([`stage-9-head-and-tail.md`](stage-9-head-and-tail.md)):
  it is a property of the harness in a stricter sense than this paragraph
  knew. `vk.NewBuffer` falls back from the device-local host-visible heap to
  the plain host-visible one when the first cannot serve the request, and on
  this device that happens at about **8 GB of total allocation** — so a
  benchmark reads at 0.18 GB/s and the pipeline, which has 20.5 GB of weights
  resident, reads the *same arena* at 15 GB/s. The 63 MB read this paragraph
  prices at 344 ms costs the pipeline **6.5 ms**. Measured with `cmd/bus`.

## Stage 3b — the scalar kernels, and why they could not get there

Kept because the simple one is the oracle the matrix-core kernels are checked
against (`TestAttentionKernelsAgree`), and because the three fixes below are
what turned §3.3's instruction from a guess into a measurement.

Three successive structural fixes bought **1.53x in total** on the wall clock,
which is **2.47x** on GPU time — the per-step ratios below are the wall-clock
ones and are all understated by the same constant 344 ms of read-back sitting
in both ends of each comparison. The third fix is the one that says why no
fourth will help:

| Change | Effect |
|---|---|
| Baseline: one query row per workgroup | 500 ms GPU @ 4096 tokens, 0.52 TFLOP/s |
| QB=8 queries share an LDS key tile | **1.09x** — and it *lost* occupancy |
| Remove the LDS staging again | 1.11x |
| One key per thread, all QB queries in registers | 1.26x → 202 ms GPU, 1.28 TFLOP/s |

- The LDS tile's 16 KB took the CU from six resident workgroups to two, and
  the latency hiding lost paid for the reuse gained. Occupancy is the budget
  that decides this kernel, not shared-memory capacity.
- After the register-blocking fix the kernel runs at **2.67 FLOP/byte where
  ~28 is needed** to saturate 22.9 TFLOP/s against the MALL. Reaching that by
  register blocking alone would need a query block near 32-64, which LDS
  cannot hold.
- So [IDEAS §3.3](ideas.md)'s original instruction — build attention's two
  matmuls out of the register-blocked WMMA kernel — is now *measured* advice
  rather than a guess. The scalar-FMA path tops out an order of magnitude
  short.

**Cost**: 546 ms wall / 202 ms GPU for one block's attention at 4096 tokens,
i.e. ~55 s per image over 34 blocks and 8 steps on GPU time alone. The linears
are 79% of a block's FLOPs but run ~15x faster on the WMMA path, so attention
was ~95% of the DiT's time until stage 3c below, which took it to 4%.

## Two hazards a port hits here

Both are covered by negative controls that catch them at **12,000x to
100,000x** the tolerance:

- **RoPE pairs adjacent components** — (2j, 2j+1) as one complex number. The
  other convention in common use pairs j with j+headDim/2; it is equally
  plausible, differs only in index arithmetic, and is wrong everywhere
  (measured at 12.6 against a 2e-4 bound).
- **The q/k RMS norms are per head**, over 128 components, not over the full
  3840. Normalising over the full width measures 2.43.

Two more the controls cover: adaLN's four chunks are concatenated
(scale_msa, gate_msa, scale_mlp, gate_mlp) rather than interleaved, and the
gated residuals are easy to drop.

## The checkpoint disagrees with the config

The DiT has **34 attention blocks, not 30** — 30 `layers` plus two
`context_refiner` and two `noise_refiner`, all the same module. And each
carries an `adaLN_modulation` of `[15360, 256]` that `bench/modelshapes.go`
does not model at all. Both were found by reading the weights rather than
`config.json`.

## Stage 3c — attention on the matrix cores

`shaders/dit_attention_wmma.comp` is flash attention with both matmuls issued
as `coopMatMulAdd`, one wave per workgroup, fp16 operands and fp32
accumulators. At 4096 tokens, 30 heads, head dim 128 (GPU timestamps, best of
three, two full runs agreeing to 0.991-1.007 on every row at this size):

| kernel | VGPRs | spilled | LDS | attention | TFLOP/s | % of WMMA ceiling |
|---|---|---|---|---|---|---|
| `simple` (fp32, one query per workgroup) | | | | 500 ms | 0.52 | — |
| `flash` (fp32, register-blocked) | | | | 202 ms | 1.28 | — |
| `wmma_qt4_kt4` | 256 | **361** | 17408 | 19.3 ms | 13.3 | 24% |
| `wmma_qt2_kt8` | 256 | 140 | 16384 | 11.3 ms | 22.8 | 41% |
| `wmma_qt2_kt2` | 252 | 0 | 7168 | 10.9 ms | 23.5 | 42% |
| `wmma_qt2_kt4` | 256 | 50 | 16384 | 9.9 ms | 26.0 | 47% |
| `wmma_qt1_kt4` | 192 | 0 | 5120 | 9.3 ms | 27.7 | 50% |
| `wmma_qt1_kt8` | 252 | 0 | 7168 | 9.1 ms | 28.4 | 51% |
| `wmma_qt2_kt4_w32` | 256 | 125 | 9216 | 8.6 ms | 30.1 | 54% |
| `wmma_qt1_kt8_w32` | 256 | 20 | 7168 | 7.2 ms | 35.8 | 64% |
| `wmma_qt1_kt2_w32` | 192 | 0 | 4096 | 7.0 ms | 36.7 | 66% |
| **`wmma_qt1_kt4_w32`** | 240 | **0** | 5120 | **6.64 ms** | **38.8** | **70%** |

`qt` is query tiles per wave, `kt` key tiles per key block, `_w32` a pinned
wave size of 32. Register and LDS figures are RADV's own
(`RADV_DEBUG=shaderstats,nocache`), per variant.

### Arithmetic intensity did not decide this one

Which is the surprise, because it decided everything in §2.1 and §2.7. Only
`qt` moves intensity here — per key block a wave reads `kt*headDim/16`
fragments of each of k and v and issues `2*qt*kt*headDim/16` multiplies, so
`kt` cancels out of the ratio and `qt` alone sets it at `16*qt` FLOP/byte — and
the three values of `qt` span 16 to 64 FLOP/byte to land **within 1.06x of each
other** at wave64 (27.7, 26.0, and 13.3 only because qt=4 spills). The ladder
is decided by two other things entirely:

1. **The register file.** Every variant that spills loses, monotonically in
   how much: 361 spilled VGPRs costs 2.9x, 140 costs 1.24x against its
   unspilled sibling, 0 wins. The winner sits at 240 of 256 VGPRs with
   nothing spilled, which is the same cliff §2.7 ended on.
2. **Wave size**, worth a clean **1.40x** at identical tiling (9.30 → 6.64 ms,
   `qt1_kt4` to `qt1_kt4_w32`). This is the third arm of §6.2's three-way
   split and it behaves like the second one: the kernel is fragment-load
   dominated, wave32 halves the per-instruction operand footprint, and it has
   the registers to spare where the GEMM's best kernel did not.

The reason intensity is inert is the same reason LDS staging lost in the GEMM:
**the caches already supply the reuse.** Every query block of a head streams
that head's whole k and v — 2 MB in fp16 — and the workgroups covering one
head are adjacent in the dispatch, so the aggregate 15.4 GB of fragment reads
at 4096 tokens is served at an effective 2.2 TB/s, well past the MALL's 805.
Register blocking exists to buy reuse out of DRAM, and there is no DRAM
traffic here to buy it out of.

### The layout is the load, and the load is 32 bytes

Getting the *tiling* right was worth nothing until the operands were stored
the way a fragment load reads them. A 16x16 fp16 `coopMatLoad` holds 32 bytes
of each of the 16 rows it touches, so reading one out of a matrix with any
natural row stride puts sixteen 32-byte chunks, one stride apart, in a single
instruction: the contiguous run in flight is 32 B against a `gcd` of 256 or
more, which is §5.1b's mechanism 3 and the regime the GEMM found 2-4x in.

With q, k and v in their natural per-head layouts (headDim contiguous, and v
transposed for the second GEMM's reduction axis) the kernel reached **1.56x**
the scalar one and *every tile geometry tied* — the signature of a kernel bound
by neither registers nor MMA issue. `shaders/dit_pack_f16.comp` now stores each
operand **as fragments**: tile (t, c) of a head is one contiguous 512-byte
block and `coopMatLoad(..., stride 16)` reads it with full coverage, no
hoisting, no LDS staging and no extra registers. That change alone is the
difference between 1.56x and 30x.

The pack costs three dispatches, 1.8 ms of the 8.5 ms graph at 4096 tokens.
Stage 4 removes it: the q/k/v projections are GEMMs, and a GEMM can write the
fragment-tile layout as easily as a row-major one.

### Three things a WMMA flash kernel has to work around

All three come from the same place — `GL_KHR_cooperative_matrix` gives an
opaque element layout — and all three have a cheap answer:

1. **An accumulator cannot be a multiply operand.** Conversions are allowed
   only between matrices of the same Use, so P, which arrives in an
   accumulator, must go through memory to come back as an A operand. That LDS
   round trip is unavoidable, and it is also where the fp16 narrowing and the
   tail mask happen, so it earns its place.
2. **Element access carries no row index.** `m[i]` and `exp2` over it are
   legal and are how the exponential is applied in registers, but which
   (row, column) element `i` is is implementation-defined, so no row-wise
   reduction can be done that way. The row max comes from a single 16x16 fp32
   staging tile in LDS, reused per key tile, which is why the winner needs
   5 KB of LDS and not the 16 KB that cost the scalar flash kernel its
   occupancy.
3. **The online-softmax rescale is per row, and a per-row scale is not a
   scalar.** But component-wise arithmetic between two same-typed matrices
   *is* supported, so the correction becomes a matrix whose every row is
   constant, built in LDS once per key block and applied as `O = O * Ccorr`.
   The big accumulator never leaves registers.

The row sums fall out of the same trick from the other side: appending a
column of ones to V makes `P . ones` a matrix whose every column is the row
sum of P, so the softmax denominator accumulates alongside O, takes the same
`* Ccorr` rescale, and touches LDS only in the epilogue. It costs `qt*kt` of
the `2*qt*kt*headDim/16` multiplies per block — 6% at head dim 128 — and
removes a reduction pass over every score.

### fp16 is safe here, and it was not in the VAE

The matrix cores take fp16 operands and nothing else, so this is not a choice.
Against the diffusers reference at 320 tokens the whole stack measures
**6.7e-3** of the tensor's RMS where the fp32 kernels measure 1.1e-5, and the
bound is set at 1.5e-2. The worst element is 0.031 off a value of 52 — 6e-4
relative to itself, i.e. fp16's 4.9e-4 quantum — and looks eleven times worse
than that only because this project normalises by RMS (4.73) rather than per
element. All ten variants agree to the last digit of it: the error follows the
arithmetic, not the tiling.

Stage 2 found fp16 could not hold the VAE's intermediates at all (1.16e7
against a 65504 limit). The difference is structural, not a matter of luck:
attention's softmax weights are bounded by 1 by construction and every
accumulator stays fp32.

### The negative controls

Three, each a single change to an otherwise identical dispatch, and each
caught by orders of magnitude (`TestGPUWMMANegativeControls`):

| control | rel | vs the 1.5e-2 bound |
|---|---|---|
| v packed in the natural layout instead of transposed | 8.54 | 569x |
| q without the log2(e) that makes `exp2` the right base | 4.24 | 283x |
| the tail mask compiled out (`-DNO_TAIL_MASK`) | 0.398 | 27x |

The first is the hazard this stage was most at risk of: the two GEMMs reduce
over different axes, so q and k want one layout and v wants its transpose, and
getting it wrong is silent — the shapes match and every load is in bounds. The
third is a real build of the real shader, so what it proves is that a sequence
length which is not a multiple of the key block genuinely needs the mask: a
pad key scores zero, not minus infinity, and without masking every row is
divided by a denominator counting keys that do not exist. `TestGPUWMMATails`
runs 300, 337 and 512 tokens for the same reason.

### What it costs per image now

8.46 ms of GPU time for one block's attention stack at 4096 tokens — 6.64 ms
of attention, 1.8 ms of packing — is **2.30 s** over 34 blocks and 8 steps,
against 55 s for the scalar kernel. Attention is no longer the DiT's problem;
`results/shapes.csv` says the linear layers are ~1.62 s per step, i.e. ~13 s
per image, so stage 4 is where the remaining time is.
