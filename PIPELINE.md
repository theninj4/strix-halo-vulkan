# PIPELINE — the z-image-turbo vertical slice

> **This file is rewritten, not appended.** It states where the pipeline is
> *now* and what happens next. History belongs in `TODO.md` (session
> handoffs) and `research/` (closed findings); if a paragraph here is about
> the past, it is in the wrong file. Keep it under ~150 lines.

**Status**: stages 1 and 2 done. The VAE decoder runs on Vulkan at
1024x1024 in **5.6 s**, matching diffusers to 2.4e-5.
**Target**: `prompt → PNG` for Z-Image-Turbo at 1024x1024, 8 steps, fp16.
**Accept for now**: ~20-30 s/image. Correctness first, then profile.

## Why this exists

Phase 1 — fifteen-plus sessions of microbenchmarks — is over. Its last
iteration bought 1.09-1.13x on one cell while two pillars sat at zero, and
the backlog in `IDEAS.md` generates new items faster than it closes them, so
it has no convergence criterion. The evidence, taken 2026-09-13:

- `bench/` was 11,658 of ~13,900 Go lines; `vk/`, the actual engine, is 653,
  and every `api/` endpoint returns `NotImplemented`.
- **No checkpoint had ever been loaded.** Every number in `results/` is
  against randomly generated tensors of the right shape.
- `IDEAS.md` §7 (accuracy) has zero work done, and IDEAS itself calls real
  perplexity "ultimately the only test that matters". It stays blocked until
  a model runs.
- The measured rules (`VEC`, `NROWS`, `MROWS`, the gcd window, the wave-size
  crossover) live only in prose. Nothing outside `bench/` knows which kernel
  wins, so every session re-derives them by re-reading 240 KB.

**The unit of work is now "a thing that runs", not "an experiment."** A
profiler chooses the optimisation targets; the IDEAS backlog does not.

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
| 3 | Attention | next | IDEAS §3.3; 4096 tokens, 30 heads, 3D RoPE, qk_norm |
| 4 | DiT graph | | 34 blocks over kernels already near ceiling, + adaLN, + SwiGLU |
| 5 | Text encoder + tokenizer | | Qwen3-4B, GQA; BPE from `tokenizer/` |
| 6 | Scheduler + driver | | FlowMatchEuler, 8 steps; PNG out |

**Stage 2 is first on purpose**: it can be validated against a latent
decoded by diffusers before any of the DiT exists, and it is the largest
genuinely new piece — convolution is an operator class the benchmark suite
never covered. Each stage gets a CPU implementation in Go first: it
separates "do I understand the architecture" from "is the shader right", and
once it matches diffusers it is the oracle the GPU port is debugged against.

## The rule for every stage

Validate against the **diffusers reference**, not against a shape table.
`reference/dump_vae.py` is the pattern: decode a fixed input with diffusers
in fp32 on CPU, dump **every submodule's output**, and have the Go test walk
the graph stage by stage. Stagewise is the point — an end-to-end check tells
you the image is wrong, this tells you which of the 40 convolutions is.

Every such test needs a **negative control** that deliberately breaks the
implementation and asserts the break is caught. Without one the test proves
only that it runs. `zimage/vae`'s control perturbs an output channel, the
padding, a residual and a projection, and they land 141x to 14000x above the
tolerance — which is also how the tolerance got set, rather than by taste.

Set tolerances from measurement: `zimage/vae` sees 6.2e-5 legitimate drift
over 40 convolutions of independent float32 summation order, and bounds at
2e-4. Normalise the error by the tensor's **RMS**, not per-element — the
activations here cross zero constantly, and a per-element relative error
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

## Measured baseline to beat — VAE

| latent | image | activations | wall |
|---|---|---|---|
| 16² | 128² | 49 MB | 49 ms |
| 32² | 256² | 197 MB | 217 ms |
| 64² | 512² | 789 MB | 1.02 s |
| 128² | **1024²** | 3.16 GB | **5.6 s** |

Everything is fp32. The obvious next lever is fp16 weights and activations
onto the WMMA path, which halves the arena and should move conv3x3 well past
its current 3.3 TFLOP/s — but it is a separate change with its own
validation, because doing it at the same time as the correctness port would
leave any discrepancy with two possible causes.

## Measured baseline to beat — DiT

From `results/shapes.csv`, DiT linear layers only, 1024x1024, per step, best
kernel per shape: **~1.62 s**, weighted 26.9 TFLOP/s (48% of ceiling). Eight
steps is **~12.9 s** — and that is for 30 blocks, so the real 34-block
figure is ~13% higher, before attention, VAE, adaLN or any elementwise pass.
`dit.ff.w13` is half the step's time at 23.4 TFLOP/s and is the worst shape
in the model; the §2.7 crown kernel loses 1.66x on it to an older tile.
