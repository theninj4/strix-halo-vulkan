# Stage 4 — the DiT block as a graph, and three things a GEMM's memory wants

**Result: a whole Z-Image DiT block runs on the GPU in 49.3 ms at 4096
tokens** — 18 dispatches, every matrix operation on the matrix cores,
validated stage by stage against the diffusers reference. That is **13.4 s per
image** for the transformer (34 blocks x 8 steps) and 19.0 s with the VAE,
against the ~16.9 s the per-shape budget predicted for its linears and
attention *alone*.

It got there in two halves. **4a** built the graph and asked where a weight
should live; **4b** asked what else the memory system was being told, and
found two more things it wanted — and the three compound:

| the block at 4096 tokens, one change at a time | block | per image | |
|---|---|---|---|
| `results/shapes.csv`'s pick: §2.7's crown kernel, stage 4a's graph | 90.0 ms | 24.5 s | |
| the four-wave 128x256 tile with the weight as **16x16 fragment tiles** (§2.8) | 67.9 | 18.5 | 1.33x |
| + the grid walked in **bands of 8 columns** (§2.4) | 54.5 | 14.8 | 1.25x |
| + the **elementwise tail fused**, 26 dispatches to 18 | 49.9 | 13.6 | 1.09x |
| + an **fp16 C** for the FFN's gate and up (§2.6) | **49.3** | **13.4** | 1.01x |

Not one of the first four rows changes the arithmetic: same flops, same 16x16x16
cooperative-matrix tiles, same fp32 accumulators. The tile geometry moves once
(row 1 to row 2, single-wave 64x64 to four-wave 128x256); everything else is
the *order* of things — the order of a weight's bytes in memory, the order the
workgroups are launched in, the order the passes are run in. The two kernel
levers in the middle are about one quantity, how many bytes of a row are in
flight and how much of what is in flight is shared, and that quantity has now
been worth 1.33x, 1.25x and — in stage 3c, on the same argument applied to an
activation — 30x. The last row does change it, deliberately, and what that
costs is measured below.

Where it ends up: **42.0, 41.1 and 40.6 TFLOP/s** on the model's three GEMM
shapes, 73-76% of this part's 55.5 TFLOP/s WMMA ceiling, on one kernel. They
started 35.3, 14.1 and 26.7.

## What was built

`zimage/dit/gpublock.go` — `GPUBlock`, one whole block as a dispatch sequence
over four arenas (fp32 weights, fp32 activations, fp16 activations, fp16
weights). The unit is the *block* rather than the layer stack because a block
is what `reference/dump_dit_block.py` dumps stage by stage, so it is the
largest thing that can still be validated against something other than itself.

Five new shaders:

- `shaders/dit_gemm.comp` — the projection GEMM. §2.7's winning arm ported onto
  the DiT's binding and push-constant layout, with the LDS arms, the grouped
  tile table and the benchmark's own bindings left behind, and with **three B
  layouts** as a build-time knob: `[N, K+pad]` read column-major, `[K, N+pad]`
  read row-major, and 16x16 fragment tiles.
- `dit_adaln.comp` — the modulation projection and its four chunks, one
  workgroup per output row.
- `dit_scale_f16.comp`, `dit_swiglu_f16.comp`, `dit_gate_add.comp` — the
  elementwise passes that keep the GEMMs fed.

`cmd/ditblock` times the block dispatch by dispatch and sweeps the GEMM builds
over the model's three shapes. `vk.Buffer` gained `WriteUint16At` /
`ReadUint16At`, and `safetensors` exported `F32ToF16` / `F16ToF32`, since the
fp16 arenas are now filled by a Go-side packer rather than by a straight copy.

Stage 4b added two build-time knobs to the same GEMM and three fused passes
beside it:

- `SWZ`, the workgroup swizzle (§2.4): bands of SWZ columns walked top to
  bottom instead of the grid's own row-major order. Seven builds, SWZ = 2, 4,
  8 and 16 against two B layouts.
- `C_F16`, an fp16 store (§2.6), with `dit_swiglu_f16.comp` gaining the
  matching `IN_F16` read. Offered only to the FFN's gate and up projections,
  through a companion table rather than the kernel ladder, because it is a
  property of the consumer and of the values — `dit.ff.w2`'s output reaches
  6e5 against fp16's 65504.
- `dit_norm_scale_f16.comp`, `dit_norm_gate_add.comp` and `dit_qk_pack.comp`,
  which collapse seven dispatches into three. `GPUBlock.Fused` keeps both
  paths, because the unfused one is what the stagewise walk needs and what the
  fused one is checked against.

## The graph

18 dispatches fused, 26 unfused, and both are kept. The unfused one is the
graph stage 4a validated: every norm is its own dispatch, so every norm is a
tensor the diffusers dump can be compared against. The fused one is what runs.

```
fused (18):
  adaln  attn in  gemm q  gemm k  gemm v  qkpack q  qkpack k  pack v
  attention  narrow ctx  gemm o  gate msa  ffn in  gemm w1  gemm w3
  swiglu  gemm w2  gate mlp

unfused (26): ... rmsnorm x + attn in, rmsnorm q/k + rope q/k + pack q/k,
  rmsnorm attn + gate msa, rmsnorm ffn + ffn in, rmsnorm ff + gate mlp
```

Each fused pass exists because the pair it replaces writes an fp32 tensor to
DRAM and reads it straight back. `dit_qk_pack.comp` is the one worth reading:
the per-head RMS norm, the rotary embedding, the softmax scale and the
fragment-tile pack all have the same natural unit — *a head of one token* — so
one workgroup owning 16 tokens of one head owns everything three dispatches
needed, and the intermediate never leaves LDS.

## Correctness

`TestGPUBlockAgainstDiffusers` walks all thirteen stage boundaries the
reference dumps, each by re-running the graph's prefix (`GPUBlock.RunTo`)
rather than reading the arena after `Apply` — most of these tensors do not
survive to the end of the graph, since two norms share one scratch tensor and
four of the five run in place.

Error, normalised by the tensor's RMS: the fp32 paths (modulation,
`attention_norm1`) at 1e-6, the projections at 1.2e-3, the rotated q and k at
1.6e-3 to 3.2e-3, `attn_ctx` and `attn_out` at 1.3e-2 to 1.5e-2, and **the
block's output at 1.7e-2**. The bound is 8e-2, a little over twice the worst
stage.

Two stages — `attention_norm2` at 3.7e-2 and `ffn_norm2` at 3.1e-2 — are the
largest figures in the table, and they are the *normalisation* rather than the
arithmetic. `attention_norm2`'s worst element is 0.0108 off a value of 5.757,
1.9e-3 of itself and about four fp16 quanta at that magnitude; the tensor's RMS
is 0.295, because an RMS norm's output inherits the learned weight's dynamic
range and most of this one's elements are twenty times smaller than its
largest. Normalising by the RMS is still the right default — these activations
cross zero constantly — but on a tensor with that spread it reports twenty
times the per-element figure.

Three negative controls, at **1160x, 2828x and 15828x** the bound: the
modulation's four unlabelled chunks swapped, one of the twenty-six dispatches
dropped, and a weight staged in a layout its kernel does not read. And
`TestGPUBlockKernelsAgree`: all eleven builds — three weight layouts, two
tilings, five swizzles — produce the same output to every digit logged, which
is what makes the residue attributable to fp16 and not to a kernel.

Stage 4b's two changes are checked separately, because only one of them is
arithmetic-preserving:

- **The fusions.** `TestGPUBlockFusedMatchesUnfused` compares the two graphs
  at the points where the fused passes' own output is still visible, before
  the fp16 GEMMs downstream can amplify a difference into something that has
  to be argued about. The fp16 A operand out of norm+scale+narrow is
  **bit-identical**. The attention context, downstream of the q/k fusion,
  differs by 5.1e-4 — that pass keeps the norm and the rotation in fp32
  registers where the unfused one rounds to fp32 in the arena between them.
  The block's output differs by 1.6e-3, since an fp16 operand one ulp apart is
  a different GEMM input. Bound 5e-3.
- **The fp16 C**, which *is* a precision trade and is measured rather than
  asserted free. `TestGPUBlockFP16FFN`: the block's error against diffusers
  goes 1.7e-2 to 1.8e-2, and the narrowing alone moves the worst element by
  1.8e-4 of itself. `TestFFNIntermediatesFitFP16` is the range check it
  depends on — stage 2 found this model's VAE could not hold its intermediates
  in fp16 at all, so "fp16 is the fast path" is a claim that has to be made per
  tensor: `w1` and `w3` peak at 250 and 508, 262x and 129x under the limit,
  and `w2` at 6e5, which is why only two of the three have an fp16-C build.

## The measurement

`go run ./cmd/ditblock -tokens 320,1024,4096`, best of three per
configuration, GPU timestamps per dispatch, run twice. The two runs agree to
**0.998-1.020** on every block total and to 1-3% on every per-shape cell.

### The block, at 4096 tokens

| plan | block | GEMMs | attention | elementwise | per image |
|---|---|---|---|---|---|
| **`wg128x256_bt16_swz8`** (the default) | **49.3 ms** | 35.7 | 6.7 | 6.8 | **13.4 s** |
| `wg128x256_bt16_swz4` | 50.7 | 36.5 | 6.8 | 7.5 | 13.8 s |
| `wg128x256_bt16_swz16` | 51.3 | 36.9 | 6.8 | 7.6 | 14.0 s |
| `wg128x256_bt16_swz2` | 53.7 | 39.4 | 6.7 | 7.5 | 14.6 s |
| `wg128x256_bt16` (no swizzle) | 63.8 | 49.5 | 6.8 | 7.5 | 17.4 s |
| `wg128x256_swz8` (row-major B) | 66.1 | 51.6 | 6.9 | 7.7 | 18.0 s |
| `wg128x256` | 69.9 | 55.6 | 6.7 | 7.6 | 19.0 s |
| `reg64_bt16` | 80.4 | 66.2 | 6.8 | 7.4 | 21.9 s |
| `reg64_hka4` (§2.7's crown) | 86.3 | 72.0 | 6.8 | 7.5 | 23.5 s |

(The 0.7 ms of elementwise every non-default row carries is not noise: the
fp16-C companion exists only for the default kernel, so every other arm runs
SwiGLU against fp32 inputs. It is the same 0.85 ms the fp16 C is worth below.)

### TFLOP/s by shape and build

M=4096, both runs. `_bt16` is the fragment-tiled weight, `_swzN` the
N-column band, `hka4` §2.7's hoisted four-tile K-slab; `reg64` is one wave on
a 64x64 accumulator grid and `wg128x256` four waves on 128x256.

| build | B layout | `qkv`/`o` | `ff.w13` | `ff.w2` |
|---|---|---|---|---|
| `reg64_hka4` | [N,K] | 33.7-35.3 | 14.1-14.5 | 26.7-27.3 |
| `reg64_hka4_bt16` | tiles | 37.1-38.0 | 13.7-13.8 | 30.7-31.2 |
| `reg64_bt16` | tiles | 36.0 | 17.7 | 22.0-22.2 |
| `wg128x256` | [K,N] | 28.0-28.2 | 24.0-24.2 | 28.2-28.3 |
| `wg128x256_swz8` | [K,N] | 28.1-28.3 | 28.0 | 29.0-29.1 |
| `wg128x256_bt16` | tiles | 41.1-41.3 | 22.3-22.6 | 39.2-40.0 |
| `wg128x256_bt16_swz2` | tiles | 37.9-38.2 | 36.0-36.4 | 37.7-37.9 |
| `wg128x256_bt16_swz4` | tiles | 41.0 | 39.2-39.5 | 40.7-40.8 |
| **`wg128x256_bt16_swz8`** | tiles | **41.9-42.0** | **40.9-41.1** | **40.6** |
| `wg128x256_bt16_swz16` | tiles | 41.1-41.3 | 38.3-39.0 | 39.7-40.2 |

### The elementwise tail, at 4096 tokens

| pass | unfused | fused |
|---|---|---|
| `rmsnorm x` + `attn in` | 1.12 ms | 0.59 |
| `rmsnorm q/k` + `rope q/k` + `pack q/k` | 3.66 | 1.40 |
| `pack v` | 0.59 | 0.59 |
| `narrow ctx` | 0.41 | 0.41 |
| `rmsnorm attn` + `gate msa` | 1.47 | 0.86 |
| `rmsnorm ffn` + `ffn in` | 1.12 | 0.52 |
| `swiglu` | 2.19 | 1.37 (fp16 in) |
| `rmsnorm ff` + `gate mlp` | 1.46 | 0.82 |
| `adaln` | 0.17 | 0.17 |
| **total** | **12.2 ms** | **6.7 ms** |

## What it says

**1. The fragment-tiled weight is worth 1.43-1.46x, and it is not universal.**
Holding the kernel fixed and changing only the weight's arrangement,
`wg128x256` goes 28.0 -> 41.1 on the attention projections and 28.3 -> 39.2 on
`ff.w2`, by making a fragment load contiguous — §5.1b's coverage law, the
mechanism stage 3c measured 30x of on activations. On `ff.w13` it *loses* 5%.
So stage 3c's "strictly dominates §2.7's hoist" is withdrawn: the tiling is a
shape-dependent lever like every other one here.

**2. With a tiled B, hoisting A can hurt.** `reg64_bt16` against
`reg64_hka4_bt16` on `ff.w13`: **17.7 against 13.7**, i.e. removing §2.7's
lever is worth 1.29x once B is tiled. The two are not independent — both exist
to raise how many bytes of a row a wave has outstanding, and the hoist pays in
registers that the tiled layout no longer needs spent. On `qkv`/`o` the hoist
still wins slightly, so it is an interaction to carry rather than a rule.

**3. The shape that resisted everything was resisting the launch order.**
`ff.w13` sat at 24.2 TFLOP/s where the others reached 41, and the swizzle
takes it to **41.1 — 1.84x, and the largest single number in this file**. The
mechanism is exact and was worked out before the code was written:

- `ff.w13` (N=10240, K=3840) and `ff.w2` (N=3840, K=10240) have **identical
  tile-level traffic**, 3.52 GB at M=4096, and identical flops. They measured
  264 and 439 GB/s of effective bandwidth against it.
- The grid is walked with `gl_WorkGroupID.x` — the N direction — fastest, so
  the workgroups resident at one moment are a *row* of C: one A slab shared by
  all of them, and B streamed past. `w13`'s grid is 40 columns wide and `w2`'s
  is 15, so at ~64 workgroups in flight `w2` keeps four grid rows live and
  amortises each B slab across four A slabs; `w13` keeps 1.6.
- 2.52 GB of B traffic amortised 1.6x is 1.57 GB, plus 1.26 GB of A, at 236
  GB/s: **12 ms**, against 13.3 measured. The same arithmetic for `w2` gives
  8 ms against 8.0 measured.
- Walking bands of 8 columns top to bottom instead makes the resident set a
  tall block: 2 B slabs against 32 A slabs, and `w13`'s whole A is 30 MB,
  which fits the 32 MB MALL. 7.87 ms measured.

**Eight columns is the optimum and it is interior**: 2, 4, 8, 16 give 36.0,
39.2, **41.1**, 38.3 on `w13`. Narrow bands do not amortise B; wide ones push
the A working set past the MALL.

**4. The swizzle's value depends on the layout it is applied to** — 1.16x on a
row-major B, **1.84x on a tiled one**. Neither ordering nor layout is worth
its full value without the other, which is the third interaction in this file
and the reason the ladder is a cross product rather than a list.

**5. One kernel now wins all three shapes** (42.0, 41.1, 40.6 TFLOP/s), where
stage 4a needed a per-shape plan. The shapes disagreed only while the losing
ones were bound by something other than their arithmetic; with the memory
system given what it wants, they converge to 73-76% of the ceiling.

**6. Fusing the tail is worth 1.09x and is arithmetic-preserving.** 12.2 ms of
elementwise work becomes 6.7. The largest piece is `dit_qk_pack.comp` — 3.66
ms to 1.40 — and it is possible because the per-head norm, the rotation, the
scale and the fragment pack share one natural unit, a head of one token. The
fp16 A operand it produces upstream is bit-identical to the unfused path's.

**7. An fp16 C is worth 0.85 ms and costs 6% of the block's error.** SwiGLU
goes 2.19 ms to 1.37 by reading fp16 (§2.6). The block's error against
diffusers goes 1.7e-2 to 1.8e-2. It applies to two of the FFN's three
projections and not the third, which is a per-tensor range question, not a
policy.

## What this leaves open

The block is now 72% GEMM, 14% attention, 14% elementwise, and the GEMMs are
at 73-76% of the ceiling. In order of what is left:

- ~~**34 blocks (stage 4c).**~~ **Done**, and it came out as predicted:
  the arena is split into three banks and a block names the one it reads from,
  with nothing about the kernels changed. The only thing the prediction missed
  is that a bank is a *pipeline* rather than a push constant. See
  [`stage-4c-dit-stack.md`](stage-4c-dit-stack.md) — 12.54 GB resident,
  49.27 ms a block inside the stack against 49.3 alone.
- **The projections writing fragment tiles directly.** `pack v` (0.59 ms) is
  pure layout and `gemm v` could do it in its epilogue; `narrow ctx` (0.41 ms)
  is the same for attention's output. Together ~1.0 ms, and unlike the q/k
  chain neither needs a cross-tile reduction to do it.
- **The swizzle in the benchmark kernel.** §2.4 is closed here on the DiT's
  own shapes, but `shaders/gemm_wmma.comp` and `results/gemm_wmma.csv` do not
  have it, so the suite's numbers are now 1.25x behind what the same hardware
  does in the pipeline.
- **Attention at 38 TFLOP/s** against the GEMMs' 41, now that the GEMMs have
  passed it. Its grid is (tokens/16, heads) and its locality argument is a
  different one; whether a swizzle applies there has not been asked.
