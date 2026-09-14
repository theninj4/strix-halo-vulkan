# PIPELINE — the z-image-turbo vertical slice

> **This file is rewritten, not appended.** It states where the pipeline is
> *now* and what happens next. History belongs in `TODO.md` (session
> handoffs) and `research/` (closed findings); if a paragraph here is about
> the past, it is in the wrong file. Keep it under ~200 lines — it earns a
> few more with each stage that closes, and loses them when a stage's detail
> moves to `research/`.

**Status**: **the target is met and the first optimisation pass has landed.**
`go run ./cmd/zimage -prompt "..."` writes a 1024x1024 PNG in **17.8 s** —
tokenizer, text encoder, 34 DiT blocks over eight denoising steps, VAE, PNG —
and reproduces diffusers' fp32 CPU pipeline to **2.5e-2** of the decoded image.
Two runs agree to 17.84 s and 17.79 s. Every weight is resident: 7.17 GB of
text encoder, 12.54 GB of transformer and 4.56 GB of activations, 21.6 s to
load, and then images cost only their own time.

Stage 7 took the profiler's top two items in the VAE — the mid block's
attention and its four projections — onto the matrix cores: **1.43 s to 15.1 ms
and 503 ms to 1.12 ms**, the decode from 5.65 s to **3.62 s**. What the
profiler says now is that the VAE is **85% one convolution kernel**, 3.02 s at
3.0-3.2 TFLOP/s, and that everything else in this pipeline is either at 70-76%
of a ceiling or under 1% of an image.

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
| 5a | Tokenizer | **done** | `zimage/tokenizer`, byte-level BPE, 20 cases id for id |
| 5b | Text encoder, CPU reference | **done** | `zimage/qwen`, 35 of 36 layers, **2.08 s**; `cmd/textenc` |
| 5c | Text encoder, Vulkan | **done** | 21 dispatches/layer, 2 banks, **57 ms**; the winning tile moves with T |
| 6 | Scheduler + driver | **done** | `zimage/pipeline`, `cmd/zimage`; **19.8 s per image**, 2.5e-2 against diffusers |
| 7 | VAE mid block on WMMA | **done** | attention **94x**, projections **449x**; decode 5.65 s → **3.62 s**, image → **17.8 s**; §7 |

Each stage gets a CPU implementation in Go first: it separates "do I understand
the architecture" from "is the shader right", and once it matches diffusers it
is the oracle the GPU port is debugged against.

## The rule for every stage

Validate against the **library reference**, not against a shape table.
`reference/dump_vae.py` and `reference/dump_dit_block.py` are the pattern: run
a fixed input through diffusers in fp32 on CPU, dump **every submodule's
output**, and walk the graph stage by stage against it. Stage 5's oracle is
`transformers` rather than `diffusers` (`reference/dump_qwen.py`,
`reference/dump_tokenizer.py`) — same pattern, and it does one thing more:
where the port relies on an assumption about the library (which hidden state
`[-2]` is, whether padding changes an answer) the dump *measures* the
assumption and the test reads the number. `GPUBlock.RunTo` is
what makes that possible on an 18-dispatch graph whose intermediates are
overwritten before it ends: it re-runs the prefix up to a labelled dispatch.

Five rules around that, each of which has caught something:

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
  weight's dynamic range. Stage 5 is where that bill came due: Qwen3's
  massive activations put the absmax **109-238x** above the RMS, so the RMS
  alone fails a stage whose worst element is off by 9e-6 of itself. The
  denominator needs a floor — `max(|want|, rms)` per element — and **which
  denominator is right is a property of the model**, so look before
  inheriting one. Stage 6 needs a third answer: a denoising trajectory is
  chaotic, so it is bounded in **relative L2** over the whole tensor.
- **Validate the composition too, and do it at a small size.** Every stage's
  own oracle passed while the pipeline made a black image, because what was
  wrong was between them. `reference/dump_zimage_run.py` runs the whole
  diffusers pipeline in fp32 on the CPU at **256x256** — 3.2 s a step, the
  full eight steps and the VAE in half a minute — because the composition
  does not know how big the image is. It is the cheapest test in the project
  and it found three bugs in one sitting.
- **A real prompt is not a random tensor.** Two of stage 6's three bugs only
  appear on real inputs: SwiGLU's product overflows fp16 at 7.1e4 where the
  random fixture peaks at 5.1e4, and the VAE's watchdog cap only fails at the
  full image. Random-input validation is necessary and it is not sufficient.

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
- **Both halves of the roofline are in this pipeline, and they want opposite
  kernels.** The DiT is compute-bound at 16384 flop/byte; the text encoder
  runs the same kind of layer at `T` flop/byte, which for a prompt is 8-512
  against the same 235 crossover. Measured: the encoder's winning tile is 16
  or 32 rows at 24 tokens, 64 at 128, and the DiT's own 128x256 at 256+ —
  1.24x for taking the DiT's default at a short prompt, 1.92x for taking the
  narrowest at a long one. Re-planning per run is free (`qwen.PlanFor`),
  because every rung reads the same staged weight.
- **Kernel choice is per shape only while something else is wrong.** The DiT's
  three shapes disagreed until both of the above were applied; then one kernel
  won all three. `results/shapes.csv`'s pick measures 23.5 s/image against
  13.4.
- **Time dispatches, not `Apply`.** Reads out of the device-local
  host-visible arena run at **0.2 GB/s** (writes: 11.5), so any wall-clock
  figure that includes a read-back is measuring the read-back.

## Measured baselines, per stage

The stage-level figures the budget above is made of. Each stage's own
measurement harness is still there — `cmd/ditstack`, `cmd/vaebench`,
`cmd/textenc -gpu` — and each stage's research note has the sweep behind it.

**The DiT** (`go run ./cmd/ditstack -image 4096 -caption 128`, GPU timestamps,
best of three, runs agreeing to 0.997-1.008):

| phase | blocks | tokens | per block | total |
|---|---|---|---|---|
| `noise_refiner` | 2 | 4096 | 49.27 ms | 98.6 ms |
| `context_refiner` | 2 | 128 | 2.60 ms | 5.2 ms |
| `layers` | 30 | 4224 | 49.81 ms | 1.49 s |
| **one step** | 34 | | | **1.60 s** |

A block is 72% GEMM, 14% attention, 14% elementwise, and one kernel
(`wg128x256_bt16_swz8`) wins all three of its shapes at **40.6-42.0 TFLOP/s**,
73-76% of the ceiling. Per block by length: 5.2 ms at 320 tokens, 11.7 ms at
1024, 49.3 ms at 4096. Residency is 12.54 GB in three fp16 banks of 4.25, 4.25
and 3.54 GB holding 12, 12 and 10 blocks, plus a 0.51 GB fp32 arena and 1.29 GB
of activations, against an 83.79 GiB device-local host-visible heap.

In the pipeline a step is **1.76 s**, not 1.60: 58 ms of CPU head and tail and
~100 ms of the residual stream crossing the bus. Both go away together.

**The VAE** (`go run ./cmd/vaebench`), fp32 except the mid block, wall clock
including a read-back:

| latent | image | activations | wall | before stage 7 |
|---|---|---|---|---|
| 16² | 128² | 49 MB | 47 ms | 49 ms |
| 32² | 256² | 195 MB | 192 ms | 217 ms |
| 64² | 512² | 780 MB | 856 ms | 1.02 s |
| 128² | **1024²** | 3.12 GB | **3.62 s** | 5.59 s |

The profile is now **one kernel**: conv3x3 is **85% of the decode**, 3.02 s at
3.0-3.2 TFLOP/s against a 55.5 TFLOP/s fp16 ceiling, over 40-odd dispatches of
which the largest is 404 ms. Everything else together is 15%, and the mid
block — the whole of stage 7's 2 s — is 0.5%.

The lever left is the same one, now aimed at the only operator that has it:
conv's implicit GEMM onto fp16 and the matrix cores. Stage 2's constraint still
holds — fp16 cannot hold *these* intermediates (1.16e7 against 65504) — but
stage 7 measured the shape of the answer: conv's *inputs* peak at 497 while its
accumulators need fp32, which is what the matrix cores do natively.

**The text encoder** (`go run ./cmd/textenc -gpu`), 35 layers, 7.07 GB of fp16
weights in two banks, best of three, two runs agreeing to 1.004:

| tokens | GPU wall | GFLOP/s | GB/s of weight | CPU fp32 |
|---|---|---|---|---|
| 24 | **57.1 ms** | 2979 | 124 (53% of the bus) | 2.08 s |
| 128 | **71.4 ms** | 12670 | 99 (42%) | 9.17 s at 105 |
| 512 | 170 ms | 21227 | 42 (18%) | — |

One run walks the roofline: the weights are read once whatever T is, so the
GB/s column falling *is* the GFLOP/s column rising. At 24 tokens the profile is
96% GEMM and 0.4% attention, so the causal grouped-query kernel was worth
making correct and is not worth tuning. Wall clock includes the read-back,
which the pipeline pays once per image rather than once per step.

## Where the image budget stands

`go run ./cmd/zimage -width 1024`, wall clock, two runs agreeing to 17.84 s
and 17.79 s.

| | per image, 1024², 8 steps | share |
|---|---|---|
| denoising | **14.13 s** | 79.3% |
| — of which the 34 blocks | 13.6 s | 76.4% |
| — of which the CPU head and tail | 0.44 s | 2.5% |
| — of which the stream across the bus | ~0.8 s | 4.5% |
| VAE decode | **3.60 s** | 20.2% |
| text encoder, 17 tokens | 0.06 s | 0.4% |
| caption refiners, once per image | 0.007 s | 0.04% |
| **total** | **17.82 s** | |

Loading is 21.6 s and happens once: 7.17 GB of text encoder, 12.54 GB of
transformer, 4.56 GB of activation arenas.

Two items are left and they are very different sizes.

**conv3x3, 3.02 s, 17% of the image and 85% of the VAE.** 3.0-3.2 TFLOP/s
against a 55.5 TFLOP/s fp16 ceiling, and unlike everything else in this
pipeline it is not a GEMM with the wrong kernel in front of it — its implicit
GEMM has never been written. This is the whole of the remaining headroom worth
the name, and `research/stage-7-vae-mid-block.md` ends on what stands in its
way.

**The head and tail onto the device, 1.24 s, 7.0%.** They are the same item:
`x_embedder` and the final layer run on the CPU (0.44 s), and because they do,
the [4128, 3840] residual stream crosses the bus twice a step (~0.8 s) where
the [4096, 64] latent is 1 MB. Two GEMMs at shapes the existing kernel already
covers — K=64 and N=64 — plus a LayerNorm.

The DiT's own levers are spent: 72% GEMM at 73-76% of the WMMA ceiling, and
what is left inside it is ~1.0 s of pure layout (`pack v` and `narrow ctx`,
which a GEMM and the attention kernel could do in their epilogues) and
attention's 38 TFLOP/s, now *behind* the GEMMs it used to lead. The text
encoder is 0.3% of an image and the 1.9x still between it and the memory bus
is worth less than a percent.

## What each stage found

One file per stage in `research/`, indexed in `research/README.md`. The ones
that are load-bearing for what is left:

- **[Stage 2, the VAE](research/stage-2-vae-decoder.md)** — fp16 cannot hold
  this model's VAE intermediates (1.16e7 against 65504); register blocking beat
  every other conv fix 6.1x; two hard device limits, 4.29 GB per storage buffer
  and a reset watchdog that kills a multi-second command buffer.
- **[Stage 3, attention](research/stage-3-dit-attention.md)** — 70% of the WMMA
  ceiling, but only once every operand is stored as fragment tiles. Three
  hazards a port hits: RoPE pairs *adjacent* components, the q/k norms are *per
  head*, and v is the operand that needs transposing.
- **[Stage 4, the block](research/stage-4-dit-graph.md)** — 90.0 ms/block to
  **49.3**, none of it arithmetic. The interactions are the finding: the
  swizzle is worth 1.84x on a tiled weight and 1.16x on a row-major one, the
  hoisted K-slab becomes a *loss* on a tiled weight, and once all of it is
  applied the three shapes that disagreed on a kernel stop disagreeing.
- **[Stage 5, the text encoder](research/stage-5-text-encoder.md)** — the
  pipeline runs **35 of 36 layers** and un-normalised output; right padding to
  512 is measurably a no-op; **2.08 s to 57 ms**. The encoder sits on the
  **memory-bound** half of the roofline, so unlike the DiT its winning kernel
  *moves with the prompt length* and re-planning per run is free. Three
  conventions invert between it and the DiT — NeoX RoPE, causal attention,
  grouped heads — and Qwen3's massive activations break an RMS-normalised
  error bound.
- **[Stage 6, the pipeline](research/stage-6-pipeline.md)** — `prompt → PNG`,
  **19.8 s**, matching diffusers to 2.5e-2 of the image. Every bug this stage
  had was in the *composition* and invisible to each component's own oracle:
  SwiGLU's product **overflows fp16 on a real prompt and never on a random
  one** (and the fix is free, because its consumer is an RMS norm), a new push
  constant on a shared shader silently zeroed the text encoder's
  feed-forward, and a dispatches-per-submit cap is a proxy for a time cap that
  stops being a good one when the dispatches differ 100x. Also: the context
  refiners take neither the timestep nor the latents, so **two of the three
  phases run once per image rather than once per step**.
- **[Stage 7, the VAE mid block](research/stage-7-vae-mid-block.md)** — the
  first target a profiler picked: attention **1.43 s → 15.1 ms** (36.4
  TFLOP/s) and the four projections **503 ms → 1.12 ms**. The head is 512 wide,
  four times the DiT's, so stage 3c's kernel spills 170 VGPRs at it and the
  *workgroup* has to own the head instead of the wave — the two matmuls then
  split along the same axis in opposite senses. wave32 is 1.56x at identical
  tiling, again. And the finding: this block's softmax is so saturated that a
  kernel **dropping three quarters of every dot product decodes the same
  image**, so it is validated on its own inputs rather than through the
  picture.
- **[Stage 4c, the stack](research/stage-4c-dit-stack.md)** — splitting a
  12.0 GB weight arena across storage buffers costs **one pipeline per bank
  and nothing per dispatch**, because the buffer is in the descriptor set. Six
  banks are bit-identical to one. The fp32 CPU chain (4.2e-4) is what makes
  the fp16 chain's compounded 1.8e-1 readable as rounding rather than wiring.
