# PIPELINE — the z-image-turbo vertical slice

> **This file is rewritten, not appended.** It states where the pipeline is
> *now* and what happens next. History belongs in `TODO.md` (session
> handoffs) and `research/` (closed findings); if a paragraph here is about
> the past, it is in the wrong file. Keep it under ~150 lines.

**Status**: stages 1, 2 and 3 done. The VAE decoder runs on Vulkan at
1024x1024 in **5.6 s**. The DiT block matches diffusers on CPU and GPU, and
its attention now runs on the matrix cores at **38.8 TFLOP/s, 70% of this
part's WMMA ceiling** — 6.64 ms per block at 4096 tokens, 30.4x the scalar
kernel, 2.3 s per image over 34 blocks and 8 steps. Next is stage 4: the
DiT graph, where the linear layers are now all of the remaining time.
**Target**: `prompt → PNG` for Z-Image-Turbo at 1024x1024, 8 steps, fp16.
**Accept for now**: ~20-30 s/image. Correctness first, then profile.

## Why this exists

Phase 1 — fifteen-plus sessions of microbenchmarks — ended 2026-09-13 with no
convergence criterion; `TODO.md` carries the argument. **The unit of work is
now "a thing that runs", not "an experiment":** a profiler chooses the
optimisation targets, the IDEAS backlog does not. Stage 3c is what that looks
like — the profiler said the 347 ms every variant appeared to take was 344 ms
of host read-back, and the kernel was somewhere else entirely.

## The model, as the checkpoint actually describes it

Read with `go run ./cmd/inspect models/Z-Image-Turbo/<part>`. These differ
from `bench/modelshapes.go`, which was transcribed from `config.json` by
hand and has gaps that change the arithmetic:

| | |
|---|---|
| DiT blocks | **34**, not 30 — 30 `layers` + 2 `context_refiner` + 2 `noise_refiner` |
| Width / heads | dim 3840, 30 heads, head_dim 128, `qk_norm` on `[128]` |
| Attention | four matrices `to_q`/`to_k`/`to_v`/`to_out.0`, each `[3840, 3840]` — not a fused qkv |
| FFN | SwiGLU: `w1`,`w3` `[10240, 3840]` gate/up, `w2` `[3840, 10240]` down |
| Modulation | `adaLN_modulation` `[15360, 256]` + bias per layer — **not modelled anywhere before** |
| RoPE | 3D, `axes_dims [32, 48, 48]`, `axes_lens [1536, 512, 512]`, theta 256 |
| Latents | 16 channels, patch size 2 — 1024x1024 is 128x128 latents is **4096 tokens** |
| VAE | Flux `AutoencoderKL`: `block_out_channels [128,256,512,512]`, 2 layers/block, GroupNorm(32), SiLU, mid-block attention, 8x upsample, scale 0.3611, shift 0.1159 |
| Text encoder | Qwen3-4B: 36 layers, hidden 2560, FFN 9728, GQA 32q/8kv, head_dim 128, vocab 151936 |
| Scheduler | `FlowMatchEulerDiscrete`, shift 3.0, 1000 train timesteps, 8 NFE |

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
| 3c | Attention on WMMA | **done** | 38.8 TFLOP/s = 70% of ceiling, 30.4x (see below) |
| 4 | DiT graph | next | 34 blocks, + adaLN, + SwiGLU, + the projections as WMMA GEMMs |
| 5 | Text encoder + tokenizer | | Qwen3-4B, GQA; BPE from `tokenizer/` |
| 6 | Scheduler + driver | | FlowMatchEuler, 8 steps; PNG out |

Each stage gets a CPU implementation in Go first: it separates "do I understand
the architecture" from "is the shader right", and once it matches diffusers it
is the oracle the GPU port is debugged against — which stage 3c then used twice
over, since the scalar kernels it replaced are themselves oracles now.

## The rule for every stage

Validate against the **diffusers reference**, not against a shape table.
`reference/dump_vae.py` is the pattern: decode a fixed input with diffusers
in fp32 on CPU, dump **every submodule's output**, and have the Go test walk
the graph stage by stage. Stagewise is the point — an end-to-end check tells
you the image is wrong, this tells you which of the 40 convolutions is.

Every such test needs a **negative control** that deliberately breaks the
implementation and asserts the break is caught; without one the test proves
only that it runs. `zimage/vae`'s land 141x-14000x above tolerance, stage 3c's
27x-569x, and that is also how each tolerance got set rather than by taste
(fp32 paths 2e-4 against 6.2e-5 of measured drift; the fp16 WMMA path 1.5e-2
against 6.7e-3). Normalise the error by the tensor's **RMS**, not per element —
these activations cross zero constantly, and a per-element relative error
reports 0.19 for a tensor that agrees to seven digits.

## What the kernels already give us

Measured on this device (`research/`), and load-bearing for every stage —
stage 2b used three of these directly:

- **fp16 is the fast path.** WMMA fp16 is 55.5 TFLOP/s; WMMA int8 is the
  *same* rate, not double (§0). Best GEMM is 70% of it (§2.7).
- **Do not quantize the DiT for speed.** At M=4096 its intensity is `4*M` =
  16384 flop/byte against a 235 crossover, so it is compute-bound and 4-bit
  weights buy nothing (§3.4). Quantize for footprint only.
- **Coverage law** (§5.1b): achievable bandwidth is
  `min(1, C/gcd(stride, 4096))`, C being the contiguous run the in-flight
  requests hold. Land `gcd` in [128, 256] B. Getting this wrong costs 2-8x
  and it is how stage 2b's attention lost 20x.
- **Raise arithmetic intensity in registers** (§2.1) before moving bytes
  closer. Worth 6.1x on conv2d.
- **Kernel choice splits at M** (§3.4): `wmma_reg64_bt_hka4_padab128` above
  M=1024, `wmma_reg32_bt_hkab4_w32_padab128` (wave32) below.
- **Store WMMA operands as 16x16 fragment tiles** (stage 3c). A fragment load
  holds 32 B of each of the 16 rows it touches, so *any* natural row stride
  puts the coverage law at 32/gcd — an eighth of the bus or worse. Packing each
  operand as contiguous 512 B tiles fixes it with no registers, no LDS and no
  hoisting, and it was the difference between 1.56x and 30x on attention. Every
  GEMM stage 4 adds is exposed to this.
- **Time dispatches, not `Apply`.** Reads out of the device-local
  host-visible arena run at **0.2 GB/s** (writes: 11.5), so any wall-clock
  figure that includes a read-back is measuring the read-back.

## Measured baseline to beat — VAE

| latent | image | activations | wall |
|---|---|---|---|
| 16² | 128² | 49 MB | 49 ms |
| 32² | 256² | 197 MB | 217 ms |
| 64² | 512² | 789 MB | 1.02 s |
| 128² | **1024²** | 3.16 GB | **5.6 s** |

Everything is fp32. The next lever is fp16 onto the WMMA path, which halves
the arena and should move conv3x3 past its 3.3 TFLOP/s — a separate change with
its own validation, and stage 3c is now the evidence that fp16 operands with
fp32 accumulators hold up when the intermediates are bounded.

## What each stage found

Closed findings live in `research/`, not here:

- **[Stage 2 — the VAE decoder](research/stage-2-vae-decoder.md)**: fp16
  cannot hold this model's intermediates (1.16e7 against a 65504 limit),
  register-blocking beat every other conv fix 6.1x, §5.1b's coverage law
  predicted a 20x attention bug exactly, and two hard device limits — 4.29 GB
  per storage buffer, and a watchdog that kills a 5.5 s command buffer.
- **[Stage 3 — the DiT block and its attention](research/stage-3-dit-attention.md)**:
  the scalar path tops out an order of magnitude short (2.67 FLOP/byte where
  ~28 is needed) and the matrix-core kernel reaches 70% of the WMMA ceiling —
  but only once every operand is stored **as 16x16 fragment tiles**, which was
  the difference between 1.56x and 30x. Arithmetic intensity turned out to be
  inert here and the register file and wave size decisive; fp16 operands are
  safe where the VAE's were not; and there are three hazards a port hits —
  RoPE pairs *adjacent* components, the q/k norms are *per head*, and v is the
  operand that needs transposing, not k.

## Measured baseline to beat — DiT attention

GPU timestamps for the attention dispatch alone (`go run ./cmd/ditbench`,
which prints the dispatch, the whole graph and the wall clock side by side).
Earlier numbers in this repo are not comparable: they counted `6*n*n*dim` flops
where attention does `4*n*n*dim`, and they were wall-clock, which here is
mostly read-back.

| tokens | image | `flash` (fp32) | `wmma_qt1_kt4_w32` | TFLOP/s | of ceiling |
|---|---|---|---|---|---|
| 320 | — | 1.8 ms | 0.13 ms | 12.2 | 22% |
| 1024 | 256² | 16.8 ms | 0.47 ms | 34.4 | 62% |
| 2048 | — | 56.4 ms | 1.70 ms | 38.0 | 68% |
| 4096 | **1024²** | **202 ms** | **6.64 ms** | **38.8** | **70%** |

The 320-token row is not reproducible across runs (the dispatch is ~130 µs and
the clock has not ramped); everything from 1024 up agrees to 0.991-1.007
between two full sweeps. The winner is the *lowest*-intensity geometry at
wave32: arithmetic intensity, which decided every GEMM in the suite, does
nothing here, and the register file and the wave size decide it instead.

## Measured baseline to beat — DiT, and where the image budget now stands

From `results/shapes.csv`, DiT linear layers only, 1024x1024, per step, best
kernel per shape: **~1.62 s**, weighted 26.9 TFLOP/s (48% of ceiling), for 30
blocks — the real 34-block figure is ~13% higher. `dit.ff.w13` is half the
step's time at 23.4 TFLOP/s and is the worst shape in the model; the §2.7 crown
kernel loses 1.66x on it to an older tile, which is stage 4's first question.

| | per image, 1024², 8 steps |
|---|---|
| DiT linears (34 blocks, from `shapes.csv`) | ~14.6 s |
| DiT attention stack, measured (8.5 ms x 34 x 8) | **2.3 s** |
| VAE decode, measured | 5.6 s |
| **total, nothing tuned, nothing fused** | **~22 s** |

Which is inside the 20-30 s this stage set out to accept, with attention at 4%
of it. Of the attention figure, 1.8 ms per block is the fp16 pack, which stage
4 deletes by having the projections write the packed layout directly; the
linears are the whole rest of the budget.
