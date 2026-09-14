# PIPELINE — the z-image-turbo vertical slice

> **This file is rewritten, not appended.** It states where the pipeline is
> *now* and what happens next. History belongs in `TODO.md` (session
> handoffs) and `research/` (closed findings); if a paragraph here is about
> the past, it is in the wrong file. Keep it under ~200 lines — it earns a
> few more with each stage that closes, and loses them when a stage's detail
> moves to `research/`.

**Status**: stages 1, 2, 3 and 4 done. **The whole DiT runs on the GPU** — all
34 blocks resident, 12.54 GB of weights across three storage buffers because
one addresses 4.29 GB here — at **1.60 s per denoising step, 12.8 s per
image** at 1024x1024, validated against diffusers. With the VAE that is
**18.4 s**, under the 20-30 s stage 4 set out to accept. A block inside the
stack costs the same 49.3 ms it cost alone, and a run with the weights in six
banks is bit-identical to one with them in a single buffer. Next is stage 5,
the text encoder.
**Target**: `prompt → PNG` for Z-Image-Turbo at 1024x1024, 8 steps, fp16.

## Why this exists

Phase 1 — fifteen-plus sessions of microbenchmarks — ended 2026-09-13 with no
convergence criterion (`TODO.md` carries the argument). **The unit of work is
now "a thing that runs", not "an experiment":** a profiler chooses the
optimisation targets, the backlog does not. Stage 4 is what that looks like —
it went looking for the kernel `results/shapes.csv` said would win, found it
the *slowest* in the ladder once the weight and the launch order were what the
memory system wanted, and closed three backlog items on the way without ever
having gone looking for them.

## The model, as the checkpoint actually describes it

Read with `go run ./cmd/inspect models/Z-Image-Turbo/<part>`. These differ
from `bench/modelshapes.go`, which was transcribed from `config.json` by
hand and has gaps that change the arithmetic:

| | |
|---|---|
| DiT blocks | **34**, not 30 — 30 `layers` + 2 `context_refiner` + 2 `noise_refiner` |
| Width / heads | dim 3840, 30 heads, head_dim 128, `qk_norm` on `[128]` |
| Attention | four matrices `to_q`/`to_k`/`to_v`/`to_out.0`, each `[3840, 3840]`, **no biases** |
| FFN | SwiGLU: `w1`,`w3` `[10240, 3840]` gate/up, `w2` `[3840, 10240]` down |
| Modulation | `adaLN_modulation` `[15360, 256]` + bias per layer; chunks are scale_msa, gate_msa, scale_mlp, gate_mlp, the scales offset by one and the gates through tanh |
| RoPE | 3D, `axes_dims [32, 48, 48]`, `axes_lens [1536, 512, 512]`, theta 256 |
| Latents | 16 channels, patch size 2 — 1024x1024 is 128x128 latents is **4096 tokens** |
| VAE | Flux `AutoencoderKL`: `block_out_channels [128,256,512,512]`, 2 layers/block, GroupNorm(32), SiLU, mid-block attention, 8x upsample, scale 0.3611, shift 0.1159 |
| Text encoder | Qwen3-4B: 36 layers, hidden 2560, FFN 9728, GQA 32q/8kv, head_dim 128, vocab 151936 |
| Scheduler | `FlowMatchEulerDiscrete`, shift 3.0, 1000 train timesteps, 8 NFE |
| Topology | **three phases, three lengths**: noise refiners over the image tokens, context refiners over the caption, then the 30 layers over the two concatenated. The context refiners are `modulation=False` — no adaLN, no scale, ungated residuals |

Sizes: transformer 6.155 B **F32** on disk (24.62 GB); text encoder 4.022 B
and VAE 0.084 B, both **BF16**. As fp16 the three total **20.5 GB**, so the
whole pipeline is resident in 128 GB unified memory at once.

## Stages

| # | Stage | State | Notes |
|---|---|---|---|
| 1 | Checkpoint loader | **done** | `safetensors/`, `cmd/inspect` |
| 2a | VAE decoder, CPU reference | **done** | `zimage/vae/`, validated stagewise against diffusers |
| 2b | VAE decoder, Vulkan | **done** | 10 shaders, 119 dispatches, 1024² in 5.6 s |
| 3a | DiT block, CPU reference | **done** | `zimage/dit/`, validated stagewise incl. RoPE and qk-norm |
| 3b | Attention, Vulkan, scalar | **done** | 4 shaders; 1.28 TFLOP/s, kept as the oracle |
| 3c | Attention on WMMA | **done** | 38.8 TFLOP/s = 70% of ceiling, 30.4x |
| 4a | The block as a graph | **done** | `GPUBlock`, 26 dispatches, 65.6 ms; §2.8 |
| 4b | Fuse the tail, fix `ff.w13` | **done** | 18 dispatches, **49.3 ms**; §2.4, §2.6 |
| 4c | 34 blocks | **done** | `GPUStack`, 3 weight banks, **1.60 s/step**, 12.8 s/image |
| 5 | Text encoder + tokenizer | next | Qwen3-4B, GQA; BPE from `tokenizer/` |
| 6 | Scheduler + driver | | FlowMatchEuler, 8 steps; the three phases, PNG out |

Each stage gets a CPU implementation in Go first: it separates "do I understand
the architecture" from "is the shader right", and once it matches diffusers it
is the oracle the GPU port is debugged against.

## The rule for every stage

Validate against the **diffusers reference**, not against a shape table.
`reference/dump_vae.py` and `reference/dump_dit_block.py` are the pattern: run
a fixed input through diffusers in fp32 on CPU, dump **every submodule's
output**, and walk the graph stage by stage against it. `GPUBlock.RunTo` is
what makes that possible on an 18-dispatch graph whose intermediates are
overwritten before it ends: it re-runs the prefix up to a labelled dispatch.

Three rules around that, each of which has caught something:

- **A negative control per test**, deliberately breaking the implementation
  and asserting the break is caught. Without one the test proves only that it
  runs. `zimage/vae`'s land 141x-14000x above tolerance, stage 3c's 27x-569x,
  stage 4a's **1160x-15828x**.
- **When an optimisation removes a stage boundary, keep the slow path** and
  check the fast one against it at the last point its own output is visible.
  Stage 4b's fusions are bit-identical there; its fp16 store is not, and is
  measured separately for it rather than folded into one bound.
- **Normalise the error by the tensor's RMS**, not per element — but know what
  that costs: two of stage 4a's stages report 3e-2 where their worst element
  is 2e-3 of itself, because an RMS norm's output inherits the learned
  weight's dynamic range.

## What the kernels already give us

Measured on this device (`research/`), and load-bearing for every stage:

- **fp16 is the fast path.** WMMA fp16 is 55.5 TFLOP/s; WMMA int8 is the
  *same* rate, not double (§0). The DiT's projections reach **42.0, 76%**.
- **Do not quantize the DiT for speed.** At M=4096 its intensity is `4*M` =
  16384 flop/byte against a 235 crossover, so it is compute-bound and 4-bit
  weights buy nothing (§3.4). Quantize for footprint only.
- **Coverage law** (§5.1b): achievable bandwidth is
  `min(1, C/gcd(stride, 4096))`, C being the contiguous run the in-flight
  requests hold. Land `gcd` in [128, 256] B. Getting this wrong costs 2-8x.
- **Order, twice.** Store every WMMA operand as 16x16 fragment tiles (§2.8) —
  free on a weight, 1.43-1.46x — and walk the grid in bands of 8 columns
  rather than rows (§2.4), which keeps the resident workgroups a *block* of C
  instead of a row that streams all of B past one A slab: **1.84x** on the one
  shape wide enough to suffer, and only 1.16x without the tiling. They are one
  lever measured in two places.
- **Kernel choice is per shape only while something else is wrong.** The DiT's
  three shapes disagreed until both of the above were applied; then one kernel
  won all three. `results/shapes.csv`'s pick measures 23.5 s/image against
  13.4.
- **Time dispatches, not `Apply`.** Reads out of the device-local
  host-visible arena run at **0.2 GB/s** (writes: 11.5), so any wall-clock
  figure that includes a read-back is measuring the read-back.

## Measured baseline to beat — the DiT

`go run ./cmd/ditstack -image 4096 -caption 128`, GPU timestamps per dispatch,
best of three, runs agreeing to 0.997-1.008. Wall clock is +0.4% over the sum,
so 77 fence waits per step are not a cost.

| phase | blocks | tokens | per block | total |
|---|---|---|---|---|
| `noise_refiner` | 2 | 4096 | 49.27 ms | 98.6 ms |
| `context_refiner` | 2 | 128 | 2.60 ms | 5.2 ms |
| `layers` | 30 | 4224 | 49.81 ms | 1.49 s |
| **one step** | 34 | | | **1.60 s** |
| **one image**, 8 steps | | | | **12.8 s** |

A block is 72% GEMM, 14% attention, 14% elementwise. TFLOP/s by shape at 4096,
on the one kernel that wins all three (`wg128x256_bt16_swz8`): `qkv`/`o`
**42.0**, `ff.w13` **41.1**, `ff.w2` **40.6** — 73-76% of the 55.5 TFLOP/s
ceiling. Per-block times by length, from `cmd/ditblock`:

| tokens | image | block | GEMMs | attention | elementwise |
|---|---|---|---|---|---|
| 320 | — | 5.2 ms | 4.7 | 0.06 | 0.5 |
| 1024 | 256² | 11.7 ms | 9.8 | 0.45 | 1.4 |
| 4096 | **1024²** | **49.3 ms** | **35.7** | **6.7** | **6.8** |

Residency: 12.54 GB of weights (three fp16 banks of 4.25, 4.25 and 3.54 GB,
holding 12, 12 and 10 blocks, plus a 0.51 GB fp32 arena) and 1.29 GB of
activations, against an 83.79 GiB device-local host-visible heap. 15.1 s to
load.

## Measured baseline to beat — VAE

| latent | image | activations | wall |
|---|---|---|---|
| 16² | 128² | 49 MB | 49 ms |
| 32² | 256² | 197 MB | 217 ms |
| 64² | 512² | 789 MB | 1.02 s |
| 128² | **1024²** | 3.16 GB | **5.6 s** |

Everything is fp32 and that wall clock includes a read-back. At 30% of the
image it is now the second-largest item and has had no optimisation pass; the
lever is fp16 onto the WMMA path, which halves the arena and should move
conv3x3 past its 3.3 TFLOP/s. Stage 2 found fp16 cannot hold *these*
intermediates (1.16e7 against 65504), so it needs the per-tensor range check
stage 4b used on the FFN, not a blanket narrowing.

## Where the image budget stands

| | per image, 1024², 8 steps |
|---|---|
| DiT, measured end to end | **12.8 s** |
| — of which projections | 9.3 s |
| — of which attention | 1.7 s |
| — of which elementwise | 1.8 s |
| VAE decode, measured | 5.6 s |
| **total** | **18.4 s** |

The DiT's levers are spent: it is 72% GEMM at 73-76% of the WMMA ceiling. What
is left inside it is ~1.0 s of pure layout (`pack v` and `narrow ctx`, which a
GEMM and the attention kernel could do in their epilogues) and attention's
38 TFLOP/s, now *behind* the GEMMs it used to lead.

**The VAE is now 30% of the image and has had no optimisation pass at all.**
The lever is fp16 onto the WMMA path, which halves the arena and should move
conv3x3 past its 3.3 TFLOP/s. Stage 2 found fp16 cannot hold *these*
intermediates (1.16e7 against 65504), so it needs the per-tensor range check
stage 4b used on the FFN, not a blanket narrowing.

## What each stage found

One file per stage in `research/`, indexed in `research/README.md`. The three
that are load-bearing for what is left:

- **[Stage 2, the VAE](research/stage-2-vae-decoder.md)** — fp16 cannot hold
  this model's VAE intermediates (1.16e7 against 65504); register blocking beat
  every other conv fix 6.1x; two hard device limits, 4.29 GB per storage buffer
  and a watchdog that kills a 5.5 s command buffer.
- **[Stage 3, attention](research/stage-3-dit-attention.md)** — 70% of the WMMA
  ceiling, but only once every operand is stored as fragment tiles. Three
  hazards a port hits: RoPE pairs *adjacent* components, the q/k norms are *per
  head*, and v is the operand that needs transposing.
- **[Stage 4, the block](research/stage-4-dit-graph.md)** — 90.0 ms/block to
  **49.3**, none of it arithmetic. The interactions are the finding: the
  swizzle is worth 1.84x on a tiled weight and 1.16x on a row-major one, the
  hoisted K-slab becomes a *loss* on a tiled weight, and once all of it is
  applied the three shapes that disagreed on a kernel stop disagreeing.
- **[Stage 4c, the stack](research/stage-4c-dit-stack.md)** — splitting a
  12.0 GB weight arena across storage buffers costs **one pipeline per bank
  and nothing per dispatch**, because the buffer is in the descriptor set. Six
  banks are bit-identical to one. The fp32 CPU chain (4.2e-4) is what makes
  the fp16 chain's compounded 1.8e-1 readable as rounding rather than wiring.
