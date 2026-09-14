# PIPELINE — the z-image-turbo vertical slice

> **PARKED 2026-09-14, and this file is the record of where.** The current
> work is [`SPEECH.md`](SPEECH.md) — the parakeet and kokoro verticals — and
> this slice resumes after them. Nothing below is stale: it is the state the
> pipeline is in, and the three items under "Where the image budget stands"
> are where to pick it up. The short version: an image is **14.26 s**, the DiT
> is **94%** of it, and the only thing left worth more than a percent is the
> **quarter of the WMMA ceiling the seven projections do not reach — 2.6 s**.

> **This file is rewritten, not appended.** It states where the pipeline is
> *now* and what happens next. History belongs in `TODO.md` (session
> handoffs) and `research/` (closed findings); if a paragraph here is about
> the past, it is in the wrong file. Keep it under ~200 lines — it earns a
> few more with each stage that closes, and loses them when a stage's detail
> moves to `research/`.

**Status**: **the target is met, and the block has no layout work left in it.**
`go run ./cmd/zimage -prompt "..."` writes a 1024x1024 PNG in **14.3 s** —
tokenizer, text encoder, 34 DiT blocks over eight denoising steps, VAE, PNG —
and reproduces diffusers' fp32 CPU pipeline to **2.2e-2** of the decoded image.
Two runs agree to 14.18 s and 14.34 s. Every weight is resident: 7.17 GB of
text encoder, 12.54 GB of transformer, 0.30 GB of VAE and 4.90 GB of
activations, 21.9 s to load, and then images cost only their own time.

Stage 9 took the last host arithmetic out of the denoising loop, the patch
embedder and the final layer: **59 ms of CPU a step to 5-10**. Stage 10 then
removed the two dispatches in the block that moved no information — `pack v`
and `narrow ctx`, each of them a tensor read back and written out in the shape
the next kernel wanted — by making them the *store instruction* of the kernel
above: **18 dispatches a block to 16**, 53.17 ms to **51.80**, an image 14.49 s
to **14.26**.

So the whole of what is left is the DiT's arithmetic. The DiT is **94% of an
image**, and inside a block it is 74% GEMM at 73-76% of the WMMA ceiling, 14%
attention and 12% elementwise. Nothing outside it is worth a percent.

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
| 8 | conv2d on WMMA | **done** | implicit GEMM, **12.4x**; decode 3.62 s → **876 ms**, image → **15.0 s**; §8 |
| 9 | Head and tail on the device | **done** | host per step **59 ms → 5-10 ms**, image → **14.5 s**; §9 |
| 10 | The layout passes as epilogues | **done** | 18 dispatches → **16**, image → **14.26 s**; §10 |

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
- **Time dispatches, not `Apply`** — but know which side of the heap you are
  on. A host read of the device-local host-visible arena runs at **0.18 GB/s**
  (writes: 29), so a wall-clock figure with a read-back in it is measuring the
  read-back. That heap's budget on this device is about **8 GB**, though, and
  `vk.NewBuffer` silently falls back to the plain host-visible one past it,
  where the same read runs at **15 GB/s** — so a single-stage benchmark and
  the 20.5 GB pipeline are on opposite sides of it and disagree by 83x on what
  a read-back costs. `cmd/bus` measures it; stage 9 found it.

## Measured baselines, per stage

The stage-level figures the budget above is made of. Each stage's own
measurement harness is still there — `cmd/ditstack`, `cmd/vaebench`,
`cmd/textenc -gpu` — and each stage's research note has the sweep behind it.

**The DiT** (`go run ./cmd/ditstack -image 4096 -caption 128`, GPU timestamps,
best of three, two runs agreeing to 51.78 and 51.82 ms a layer block):

| phase | blocks | tokens | per block | total |
|---|---|---|---|---|
| `noise_refiner` | 2 | 4096 | 49.4 ms | 98.8 ms |
| `context_refiner` | 2 | 128 | 2.57 ms | 5.1 ms |
| `layers` | 30 | 4224 | **51.80 ms** | 1.55 s |
| **one step** | 34 | | | **1.66 s** |

A block is 16 dispatches: **74% GEMM** (37.97 ms), 14% attention (7.19 ms) and
12% elementwise (5.93 ms across eight passes). One kernel
(`wg128x256_bt16_swz8`) wins all three of its shapes at **40.6-42.0 TFLOP/s**,
73-76% of the ceiling. Residency is 12.54 GB in three fp16 banks of 4.25, 4.25
and 3.54 GB holding 12, 12 and 10 blocks, plus a 0.51 GB fp32 arena and 1.29 GB
of activations, against an 83.79 GiB device-local host-visible heap.

The layers phase measures 1.56 s of wall clock against 1.55 s of dispatches,
and in the pipeline a step is 1.67-1.70 s against the same 1.66 — 10 ms of
host and 9 of bus, and the ~50 ms remainder is the one thing in this budget
still unaccounted for (stage 9 ruled out submit batching for it).

**The VAE** (`go run ./cmd/vaebench`), every convolution and the mid block on
the matrix cores, wall clock including a read-back:

| latent | image | fp32 arena | fp16 arena | wall | before stage 8 |
|---|---|---|---|---|---|
| 16² | 128² | 46 MB | 9 MB | 13 ms | 47 ms |
| 32² | 256² | 185 MB | 34 MB | 46 ms | 192 ms |
| 64² | 512² | 738 MB | 135 MB | 204 ms | 856 ms |
| 128² | **1024²** | 2.95 GB | 539 MB | **876 ms** | 3.62 s |

The wall column includes a read-back that the *pipeline* does not pay — 12.58
MB at the 0.18 GB/s a 3.5 GB program gets, 71 ms — which is why a decode costs
805 ms inside an image and 876 on its own (stage 9).

There is **no arithmetic left in this decoder**: conv3x3 is 244 ms at 41-42
TFLOP/s (30%) and the next four items are bandwidth-bound elementwise passes
running back to back over the same tensors — group norm 244 ms, the conv pack
160, SiLU 61, the adds 40, **63% together**. So what it wants next is not a
kernel but **fusion**, the same shape of change as stage 10: a SiLU whose only
consumer is a convolution could write the blocked fp16 form *instead of* its
fp32 output. ~200 ms of 805, and 1.4% of an image.

**The text encoder** (`go run ./cmd/textenc -gpu`), 35 layers, 7.07 GB of fp16
weights in two banks: **57.1 ms** at 24 tokens, 71.4 at 128, 170 at 512,
against 2.08 s and 9.17 s on the CPU. One run walks the roofline — the weights
are read once whatever T is, so GB/s of weight falling (124, 99, 42) *is*
GFLOP/s rising (3.0, 12.7, 21.2 T). 0.4% of an image; detail in
[`research/stage-5-text-encoder.md`](research/stage-5-text-encoder.md).

## Where the image budget stands

`go run ./cmd/zimage -width 1024`, wall clock, two runs agreeing to 14.18 s
and 14.34 s.

| | per image, 1024², 8 steps | share |
|---|---|---|
| denoising | **13.37 s** | 93.8% |
| — of which the 34 blocks | 13.3 s | 93.1% |
| — of which the host | 0.08 s | 0.6% |
| VAE decode | **0.81 s** | 5.7% |
| text encoder, 19 tokens | 0.06 s | 0.4% |
| caption refiners, once per image | 0.006 s | 0.04% |
| **total** | **14.26 s** | |

Loading is 21.9 s and happens once: 7.17 GB of text encoder, 12.54 GB of
transformer, 0.30 GB of VAE weights, 4.90 GB of activation arenas.

**The image is the DiT and nothing else.** Everything outside it — tokenizer,
text encoder, caption refiners, VAE, and now the head and tail — is 6.4%
together, and no item in it is worth more than 1.3%. There is no boundary work
left: the host does a patchify and an unpatchify on a megabyte, and the bus
carries 1 MB each way per step.

And inside the block there is no *layout* left either — stage 10 took the last
of it. Everything remaining is arithmetic or the traffic arithmetic needs:

- **The seven projections, 9.9 s an image (69%), at 73-76% of the WMMA
  ceiling.** The quarter they do not reach is 2.6 s, which is an order of
  magnitude more than everything else in this list together. It is also the
  hardest: stage 4 already applied the fragment-tiled weight (§2.8), the
  banded grid (§2.4) and the hoisted K-slab (§2.7), and the ablation behind
  `results/gemm_wmma.csv` is 1.25x behind what the pipeline already does.
- **Attention at 38 TFLOP/s**, 1.9 s an image, against the GEMMs' 40.6-42.0.
  Worth ~0.2 s if it caught up. Stage 10 halved its store and removed its LDS
  staging and neither moved it, which agrees with stage 3c that the register
  file is what binds it.
- **The eight elementwise passes, 5.93 ms a block and 1.6 s an image.** Each
  reads the residual stream and writes it; `swiglu` (1.45 ms) and the two
  gated residuals (0.89, 0.85) are half. Unlike stage 10's two they compute
  something, so what they want is fusion into a neighbour — and the neighbour
  is usually a GEMM, which is where §2.6 and stage 10 both ended.

Below those: the VAE's elementwise fusions are ~200 ms (1.4%) and the text
encoder's remaining 1.9x is worth less than a percent.

**One correction worth carrying**, because it sat at the top of this list for
four sessions: the layout item was priced here at "~1.0 s an image" and was
**~0.29 s**. The measurement it came from (stage 4) said 1.0 **ms a block**
and the unit was lost in transcription. A number quoted between documents
should carry the dimension it was measured in.

## What each stage found

One file per stage in `research/`, indexed in `research/README.md`. The ones
that are load-bearing for what is left:

- **[Stage 10, the layout epilogues](research/stage-10-layout-epilogues.md)**
  — the two dispatches in the block that moved no information, absorbed into
  the store of the kernel above: **18 a block to 16**, an image 14.49 s to
  14.26. Possible at all because in both cases the data was already in the
  right registers in the right 16x16 shape, so a transposed fragment tile is
  a *column-major `coopMatStore` of stride 16* and not a transpose. Two other
  things fell out: the item had been mispriced 3.4x by a units slip
  (`1.0 ms a block` quoted as `1.0 s an image`), and the old path was doing a
  **double rounding** — which is why 84 halves in 1.2 M of the context differ
  and why every one of them is an exact fp16 tie.
- **[Stage 2, the VAE](research/stage-2-vae-decoder.md)** — two hard device
  limits that still bind: 4.29 GB per storage buffer, and a reset watchdog
  that kills a multi-second command buffer. Its absmax survey (497 through the
  decoder, 1.16e7 in the mid block's scores) is what stages 7 and 8 narrowed
  against, and is now asserted by `TestConvInputsFitFP16`.
- **[Stage 3, attention](research/stage-3-dit-attention.md)** — 70% of the WMMA
  ceiling, but only once every operand is stored as fragment tiles. Three port
  hazards: RoPE pairs *adjacent* components, the q/k norms are *per head*, and
  v is the operand that needs transposing.
- **[Stage 4, the block](research/stage-4-dit-graph.md)** — 90.0 ms/block to
  **49.3**, none of it arithmetic. The interactions are the finding: the
  swizzle is worth 1.84x on a tiled weight and 1.16x on a row-major one, the
  hoisted K-slab becomes a *loss* on a tiled weight, and once all of it is
  applied the three shapes that disagreed on a kernel stop disagreeing.
- **[Stage 5, the text encoder](research/stage-5-text-encoder.md)** — **2.08 s
  to 57 ms**, and the encoder sits on the **memory-bound** half of the
  roofline, so unlike the DiT its winning kernel *moves with the prompt
  length* and re-planning per run is free. Three conventions invert between it
  and the DiT — NeoX RoPE, causal attention, grouped heads — and Qwen3's
  massive activations break an RMS-normalised error bound.
- **[Stage 6, the pipeline](research/stage-6-pipeline.md)** — every bug this
  stage had was in the *composition* and invisible to each component's own
  oracle: SwiGLU's product **overflows fp16 on a real prompt and never on a
  random one**, a new push constant on a shared shader silently zeroed the
  text encoder's feed-forward, and a dispatches-per-submit cap is a proxy for
  a time cap that stops being one when the dispatches differ 100x. Also: the
  context refiners take neither the timestep nor the latents, so **two of the
  three phases run once per image rather than once per step**.
- **[Stage 7, the VAE mid block](research/stage-7-vae-mid-block.md)** —
  attention **1.43 s → 15.1 ms**, the four projections **503 ms → 1.12 ms**.
  The finding that outlives the stage: this block's softmax is so saturated
  that a kernel **dropping three quarters of every dot product decodes the
  same image**, so no end-to-end tolerance can validate it.
- **[Stage 9, the head and tail](research/stage-9-head-and-tail.md)** — the
  host's share of a step **59 ms → 5-10 ms**. A bias and a *pad token* become
  extra columns of K, so `y = Wx + b` and `y = x_pad_token` come out of one
  biasless GEMM. And the correction that matters beyond this stage: the
  engine's "a host read-back runs at 0.2 GB/s" holds only below **~8 GB of
  allocation**, where `vk.NewBuffer` still gets the device-local heap; the
  20.5 GB pipeline reads the same arena **83x faster**, which is why the 0.8 s
  an image this stage was partly priced on was never there (`cmd/bus`).
- **[Stage 8, conv2d](research/stage-8-vae-conv.md)** — the last operator in
  the VAE, **3.02 s → 244 ms** at 41-42 TFLOP/s. The finding is the layout: a
  patch fragment is contiguous only if the **channel** axis is the tiled one,
  because tiling the pixel axis — what every other operand in this engine does
  — makes the `dw = ±1` window straddle two tiles. Also: §2.7's K-slab hoist
  does **nothing** once both operands are fully covered.
- **[Stage 4c, the stack](research/stage-4c-dit-stack.md)** — splitting a
  12.0 GB weight arena across storage buffers costs **one pipeline per bank
  and nothing per dispatch**; six banks are bit-identical to one.
