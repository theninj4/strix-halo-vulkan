# TODO / session handoff — Strix Halo Vulkan inference benchmark suite

## Where we are

Goal: benchmark the Vulkan compute operations neural-network inference is
built from (GEMM, GEMV, elementwise, softmax/RMSNorm) across implementation
"flavours" on this specific machine's iGPU (AMD Radeon 8060S / RDNA3.5 /
RADV STRIX_HALO), to figure out how to build an inference engine tuned for
it — with a specific focus on whether ~4-bit quantized weights can be made
to perform well.

Everything below builds clean (`go build ./...`, `gofmt -l .`,
`go vet ./...` all clean) and has passed its correctness checks on real
hardware. The measurements live in **`results/`, one CSV per op family**
(`results/gemv.csv`, `results/stride.csv`, …) — the single 1900-row
`results.csv` was split into them, row for row, when the CLI stopped
sweeping everything by default. A run now has to name its targets:

    go run ./cmd/bench -blocks 32,64,128,256,512,1024 gemv gemm

`go run ./cmd/bench -list` prints the twelve families; `all` is the old
whole-suite behaviour. The non-default block list above is what the committed
files have always used, so keep it if they are to stay comparable row for
row. Run with nothing else on the GPU (see the measurement lessons below).
The whole suite takes ~55 minutes, of which `stride` is ~14 (816 cases), the
GEMM families ~10, `moe` ~7 (measured twice this session) and `shapes` ~5 —
which is the reason for
targeting: refreshing `gemv` alone is a couple of minutes and rewrites only
`results/gemv.csv`. Note `gemm_wmma` (the 57-variant WMMA ablation, 285 rows)
is its own family, split out of `gemm`, since it is tuned on its own, and
`shapes` and `moe` sweep nothing at all — they run the kernels the other
families picked over the model dimensions in `bench/modelshapes.go` and over
qwen3.8-flash-next's 512-expert bank, so `-sizes` and `-blocks` do not apply
to them. `moe` allocates the largest buffers in the suite (a 512-expert fp16
bank is 1.8-4.0 GB depending on the row stride under test), so give it
headroom; its Q4 arm (IDEAS §2.2) runs in a second pass after the fp16 banks
are freed, and its grouped-GEMV arm (IDEAS §1.8-§1.11) in a third, so no two
banks are ever live together. (The wall-clock figures this file has carried for
`moe` — ~18 minutes, then ~11, then ~7 — are worth re-timing rather than
trusting: the family is now **896 rows in ~9 minutes**, having grown three
more GEMV builds and eight mixed-width dispatch plans since.)

**`SPEECH.md`** is the current work: the parakeet (speech-to-text) and
kokoro (text-to-speech) verticals. **`PIPELINE.md`** is the z-image-turbo
slice, **parked** at 14.26 s an image with its resume points stated at the
top. Both are rewritten each session rather than appended to, and the
current one is the file to read first.

Three documents carry the older analysis: **`IDEAS.md`** is the prioritised
experiment backlog (~30 items, each with hypothesis / change / expected
gain / how to measure, and marked up with what has since been measured),
plus the measured roofline and the order of attack. **`research/`** is the
findings archive — one file per *completed* section, indexed in
`research/README.md`; when an item closes, its write-up moves there and
`IDEAS.md` keeps the heading, a one-line result and a link. **`GOALS.md`**
is the long-term target (five models, HTTP API in Go).

`§N.M` is the stable address across all of them and is cited from the code
itself (293 comments in `shaders/`, `bench/`, `cmd/`, `vk/`). Never
renumber a section; give a new experiment the next free number.

## Phase 2 — build the pipeline, profile it, then optimise

The microbenchmark phase ended 2026-09-13. The current plan, the model's
real dimensions and the stage list live in **[`PIPELINE.md`](PIPELINE.md)**,
which is *rewritten* each session rather than appended to. Session handoffs
still accumulate below.

**Session 2026-09-13 (the pivot):** `safetensors/` (mmap checkpoint reader, F32/F16/BF16,
sharded and single-file) and `cmd/inspect`. Verified against 10,574 golden
narrowing cases from CPython's `struct 'e'`, an exhaustive 65,536-pattern
fp16 round trip, and a value-for-value cross-check of real tensors against
an independent Python read. The golden table caught a real bug: the
underflow boundary was one exponent too high, so values in [2^-25, 2^-24)
flushed to zero instead of rounding up to the smallest subnormal.

This is the first time the project has loaded a real checkpoint, and it
immediately corrected the hand-transcribed shape table — the DiT has **34
attention blocks, not 30**, and `adaLN_modulation` was not modelled at all.
See `PIPELINE.md` for the full inventory.

### Session 2026-09-14 — stage 3c: attention on the matrix cores

**Result: 38.8 TFLOP/s, 70% of this part's WMMA ceiling, 30.4x the kernel it
replaces.** One DiT block's attention at 4096 tokens went from 202 ms to
**6.64 ms** of GPU time; the whole attention stack including the fp16 pack and
the q/k norms and RoPE is 8.5 ms, i.e. 2.3 s per image over 34 blocks and 8
steps against 55 s before. Written up in
[`research/stage-3-dit-attention.md`](research/stage-3-dit-attention.md); IDEAS
§3.3 is closed.

**Built:**

- `shaders/dit_attention_wmma.comp` — flash attention with both matmuls on
  `coopMatMulAdd`, one wave per workgroup, fp16 operands, fp32 accumulators,
  online softmax. Ten variants over two knobs (`QT` query tiles per wave,
  `KTIL` key tiles per block) plus two pinned-wave32 arms, and an eleventh
  built with `-DNO_TAIL_MASK` as a negative control.
- `shaders/dit_pack_f16.comp` — packs an fp32 activation into **16x16 fragment
  tiles**, which is the finding of the session (below).
- `zimage/dit/gpu.go` — `Kernel` replaces the old `Flash bool`, the ladder is a
  variant table as in `bench/ops_gemm_wmma.go`, and `Profile` times each
  dispatch on the GPU the way `vae.GPUDecoder.Profile` does.
- `vk.Device.Physical()` and `.Features()`, so a package holding only a
  `*Device` can ask whether fp16/coopmat were actually enabled.

**Three things that were not expected, in order of how much they cost to
find:**

1. **The wall clock was measuring the read-back, not the kernel.** All ten
   variants measured 347 ms and looked identical. Reading 63 MB back out of the
   device-local host-visible arena runs at **0.2 GB/s** (writes: 11.5), so 344
   ms of that 347 was one `ReadFloat32At`. On GPU timestamps the variants span
   6.6 to 19.3 ms. Anything measured with a read-back in it is measuring the
   read-back — including, mildly, the VAE's 5.6 s.
2. **The operand layout, not the tiling.** With q, k and v in natural per-head
   layouts the kernel got 1.56x and *every geometry tied*. A 16x16 fp16
   fragment load holds 32 B of each of 16 rows, so any natural row stride puts
   §5.1b's coverage at 32/gcd. Storing each operand as contiguous 512 B
   fragment tiles took it to 30x, with no registers, no LDS and no hoisting —
   it strictly dominates §2.7's HOIST lever and belongs in every WMMA kernel.
3. **Arithmetic intensity is inert here.** 16, 32 and 64 FLOP/byte land within
   1.06x; the caches already supply the reuse. The register file and the wave
   size decide it instead: every spilling variant loses monotonically in how
   much it spills, and wave32 is a clean 1.40x at identical tiling.

**How it was measured:** `go run ./cmd/ditbench` (tokens 320/1024/2048/4096,
every kernel, best of three, GPU timestamps), twice. Rows at >=1024 tokens
agree to median 1.000, p10-p90 0.874-1.018; at 4096 alone, 0.991-1.007. The
320-token rows swing up to 4x between runs — the dispatch is ~130 µs and the
clock has not ramped — so do not quote them. Register and LDS figures per
variant come from `RADV_DEBUG=shaderstats,nocache` with one pipeline created
per process; without `nocache` a second run is a cache hit and prints nothing.

**Tolerances:** the fp16 path measures 6.7e-3 of the tensor's RMS against the
diffusers reference where the fp32 kernels measure 1.1e-5, and is bounded at
1.5e-2. That is what fp16's 4.9e-4 quantum predicts once the RMS
normalisation is accounted for, all ten variants agree to the last digit of it,
and the three negative controls land 27x, 283x and 569x above the bound.

**Next — stage 4, the DiT graph.** Attention is 4% of the image budget now and
the linear layers are ~14.6 s of the ~22 s total, so that is where the work is.
Two things this session leaves for it: (1) the projections should write the
fragment-tile layout directly, which deletes the 1.8 ms/block pack; (2)
`results/shapes.csv` says `dit.ff.w13` is half a step's time at 23.4 TFLOP/s
and that §2.7's crown kernel *loses* 1.66x on it to an older tile — worth
knowing whether the fragment-tile layout is what that kernel was missing.

### Session 2026-09-14 (later) — stage 4a: the DiT block as a graph

**Result: a whole DiT block runs on the GPU in 65.6 ms at 4096 tokens** — 26
dispatches, every matrix operation on the matrix cores, validated stage by
stage against diffusers. **17.8 s per image** for the transformer and 23.4 s
with the VAE, inside the 20-30 s stage 4 set out to accept. Written up in
[`research/stage-4-dit-graph.md`](research/stage-4-dit-graph.md); IDEAS §2.8 is
new and closed, §2.4 and §2.6 gained the measurements that justify them.

**Built:**

- `shaders/dit_gemm.comp` — the projection GEMM. §2.7's winning arm on the
  DiT's binding and push-constant layout, with **three B layouts** as a
  build-time knob (`[N, K+pad]` column-major, `[K, N+pad]` row-major, 16x16
  fragment tiles) and five builds over two tilings.
- Four elementwise shaders: `dit_adaln.comp` (the modulation projection and
  its four chunks), `dit_scale_f16.comp`, `dit_swiglu_f16.comp`,
  `dit_gate_add.comp`.
- `zimage/dit/gpublock.go` — `GPUBlock`: four arenas (fp32/fp16 x
  weights/activations), a `GEMMPlan` that picks a kernel *per projection* and
  stages each weight in the layout that kernel reads, and `RunTo`, which runs
  the graph's prefix up to a labelled dispatch so a 26-dispatch graph can still
  be validated stagewise.
- `cmd/ditblock` — times the block dispatch by dispatch and sweeps the ladder
  over the model's three shapes.
- `dit_common.glsl`'s push-constant block gained six GEMM fields (88 bytes of
  the device's 256); `vk.Buffer` gained `WriteUint16At`/`ReadUint16At`;
  `safetensors` exported `F32ToF16`/`F16ToF32`.

**Three things that were not expected:**

1. **`results/shapes.csv`'s crown kernel is the slowest of five in situ.**
   `wmma_reg64_bt_hka4_padab128` — §2.7's winner, 39.0 TFLOP/s in the suite —
   gives 24.5 s/image against the new default plan's 17.8. The table was not
   wrong; it was measured before the weight could be stored as fragment tiles,
   and that changes which geometry wins on two of the three shapes.
2. **A fragment-tiled weight is 1.46x, and stage 3c's "strictly dominates" is
   withdrawn.** Same kernel, only the weight's arrangement changed: 28.0 →
   41.0 TFLOP/s on `dit.qkv`/`o` (**74% of the WMMA ceiling**, past §2.7's
   70%), 28.3 → 40.5 on `dit.ff.w2`, and **23.0 against 24.3 on
   `dit.ff.w13` — a 5% loss**. It also interacts with §2.7: with a tiled B,
   *removing* the hoisted K-slab is worth 1.3-1.5x on `w13`.
3. **18% of a block does no arithmetic.** 12.1 ms of the 65.6 is the
   elementwise tail — norms, narrowings, the rotary embedding, the fragment
   pack, SwiGLU, the two gated residuals — 3.3 s per image. The fusions are
   priced in `research/stage-4-dit-graph.md`; `swiglu` alone is already at 193
   GB/s of the 236 GB/s bus and needs §2.6's fp16 C rather than a fusion.

**How it was measured:** `go run ./cmd/ditblock -tokens 320,1024,4096`, best of
three per configuration, GPU timestamps per dispatch, run twice. The runs agree
to **0.994-1.006** on every block total and to 1-2% on every per-shape cell;
the two cells outside that (`reg64_bt16` on `ff.w13`, `reg64_hka4_bt16` on
`ff.w2`) are losing cells and change no choice.

**Tolerances:** the whole-block path measures 1.7e-2 of the tensor's RMS
against the diffusers reference and is bounded at 8e-2 — looser than stage
3c's 1.5e-2 because all seven projections now narrow *both* operands rather
than only attention's. Two intermediate stages report 3.1e-2 and 3.7e-2, and
that is the RMS normalisation rather than the arithmetic: the worst element of
`attention_norm2` is 1.9e-3 of itself, about four fp16 quanta, on a tensor
whose RMS is twenty times below its largest elements. Three negative controls
land at **1160x, 2828x and 15828x** the bound, and all five GEMM builds — three
weight layouts, two tilings — agree to every digit logged.

**Next — stage 4b, then 4c.** In order of what the profile says: (1) `ff.w13`
is 27 ms of the block's 47 ms of GEMM at 24.3 TFLOP/s where its neighbours
reach 41, it *degrades with M* (27.3 at 1024 tokens, 23.0 at 4096), and its
weight is 157 MB against a 32 MB MALL — which is §2.4's workgroup swizzle, and
worth **2.9 s per image**; (2) the elementwise fusions, worth ~1.0 s; (3)
§2.6's fp16 C, which halves both `w13`'s write and `swiglu`'s read. Then stage
4c, 34 blocks, whose only new problem is that 12.3 GB of fp16 weights do not
fit one 4.29 GB storage buffer.

### Session 2026-09-14 (later still) — stage 4b: three levers, none of them arithmetic

**Result: the DiT block goes 65.6 ms -> 49.3 ms at 4096 tokens, and the image
budget 17.8 s -> 13.4 s** (19.0 s with the VAE). The block's three GEMM shapes
now sit at **42.0, 41.1 and 40.6 TFLOP/s — 73-76% of the WMMA ceiling — on a
single kernel**, where stage 4a needed a per-shape plan and its worst shape was
at 24.2. Written up alongside 4a in
[`research/stage-4-dit-graph.md`](research/stage-4-dit-graph.md); IDEAS **§2.4
and §2.6 are closed**, and §2.8 is amended.

**Built:**

- `SWZ` in `shaders/dit_gemm.comp` (§2.4): bands of SWZ columns walked top to
  bottom instead of the grid's row-major order. Seven builds over two B
  layouts and SWZ = 2, 4, 8, 16.
- `C_F16` in the same kernel and `IN_F16` in `dit_swiglu_f16.comp` (§2.6),
  reached through a companion table rather than the kernel ladder — it is a
  property of the consumer, not the tiling.
- `dit_norm_scale_f16.comp`, `dit_norm_gate_add.comp`, `dit_qk_pack.comp`:
  seven dispatches become three. `GPUBlock.Fused` and `.FP16FFN` keep both
  paths, and the graph is 18 dispatches fused against 26.
- `TestGPUBlockFusedMatchesUnfused`, `TestGPUBlockFP16FFN`,
  `TestFFNIntermediatesFitFP16`.

**What it found:**

1. **`ff.w13` was resisting the launch order, not the layout** — 24.2 -> 41.1
   TFLOP/s, **1.84x**, the largest number in the stage. The mechanism was
   derived before the code was written and predicted the measurement to within
   10%: `w13` and `w2` have *identical* tile-level traffic (3.52 GB) and
   reached 264 and 439 GB/s against it, because `gl_WorkGroupID.x` is the
   fastest axis and `w13`'s grid is 40 columns wide against `w2`'s 15 — so
   `w2` keeps four grid rows in flight and amortises each B slab four ways,
   `w13` keeps 1.6.
2. **The optimum band is interior**: 2/4/8/16 give 36.0/39.2/**41.1**/38.3.
   Narrow bands do not amortise B; wide ones push A past the 32 MB MALL.
3. **Ordering and layout are one lever, not two.** The swizzle is worth 1.84x
   against a fragment-tiled weight and only 1.16x against a row-major one, and
   the tiled weight *lost* on `w13` until the swizzle was there. Neither is
   worth its full value alone, which is why the ladder is a cross product.
4. **Per-shape kernel plans were a symptom.** With both applied, one kernel
   wins all three shapes; `DefaultGEMMPlan` is uniform again.
5. **The elementwise tail fuses cleanly and the biggest piece is q/k**: 12.2 ms
   -> 6.7, of which `rmsnorm q/k + rope + pack` 3.66 -> 1.40. That one works
   because all four steps share a natural unit — a head of one token — so the
   intermediate never leaves LDS.
6. **An fp16 C is worth 0.85 ms and 6% of the block's error**, and it is a
   per-tensor question: `w1`/`w3` peak at 250 and 508 against fp16's 65504,
   `w2` at 6e5.

**How it was measured:** `go run ./cmd/ditblock -tokens 320,1024,4096`, best of
three, GPU timestamps per dispatch, run twice — agreeing to **0.998-1.020** on
every block total and 1-3% per shape.

**Correctness:** the eleven GEMM builds all agree to every digit logged. The
fusions are checked against the unfused graph at the last point their own
output is visible: the fp16 A operand is **bit-identical**, the attention
context differs by 5.1e-4, the block by 1.6e-3 (bound 5e-3). The fp16 C is
measured rather than asserted free — the block's error against diffusers goes
1.7e-2 to 1.8e-2.

**Next — stage 4c, and then the VAE.** 4c is 34 blocks, whose only new problem
is that 12.3 GB of fp16 weights do not fit one 4.29 GB storage buffer. After
that the DiT is 72% GEMM at 73-76% of the ceiling and the next-largest item in
the image is the **VAE's 5.6 s**, which is 30% of the total and has had no
optimisation pass at all. Two smaller things are left inside the block:
`pack v` and `narrow ctx` are 1.0 s per image of pure layout that a GEMM and
the attention kernel could do in their epilogues, and attention's 38 TFLOP/s
is now *behind* the GEMMs it used to lead.

### Session 2026-09-14 (fourth) — stage 4c: the whole DiT resident, in three storage buffers

**Result: all 34 blocks live on the device at once — 12.54 GB of weights — and
one denoising step is 1.60 s, which is 12.8 s per image** at 1024x1024 and 8
steps (18.4 s with the VAE). A block inside the stack costs **49.27 ms at 4096
tokens**, the same 49.3 ms stage 4b measured for one block alone, and a run
with the weights in six banks is **bit-identical** to the same run with them in
one. Written up in
[`research/stage-4c-dit-stack.md`](research/stage-4c-dit-stack.md).

**Built:**

- `zimage/dit/gpustack.go` (the old `gpublock.go`, renamed): `GPUStack`, N
  blocks over shared pipelines and shared activation arenas, with the fp16
  projection weights split into **banks** — one storage buffer addresses 4.29
  GB here and the DiT's weights are 12.0 GB, so the 34 blocks come out 12, 12,
  10. A block is a row of offsets plus a bank index. `GPUBlock` is now this
  type with one block in it (`gpublock.go`), so every stage-4b test applies
  unchanged.
- `Apply(x, adaln, sel)`, `Run(sel)` (no upload, no read-back), `Profile`,
  `SetRoPE`, `Banks()`, and a run length that may be **shorter** than the one
  the arenas were built for.
- The unmodulated block, which is two of the 34: `pc.aux2 = 0` added to
  `dit_gate_add.comp` and `dit_norm_gate_add.comp` (every other `.spv` is
  byte-identical after `go generate`), the same flag on the CPU `Block.Apply`,
  and the `adaln` dispatch not issued at all — 17 dispatches, not 18.
- `reference/dump_dit_stack.py`, `cmd/ditstack`, `zimage/dit/gpustack_test.go`
  (five tests: against diffusers per block and chained, against the CPU chain,
  banks-agree, the bank plan without a device, and three controls).

**What it found:**

1. **Splitting the weight arena is free per dispatch and costs one pipeline
   per bank.** The buffer a GEMM reads is in its *descriptor set*, not its push
   constants, so a bank is a pipeline — six instead of two for the real stack.
   Nothing in the dispatch path changes, because `vk.DispatchMultiTimed`
   already binds each dispatch's own set (IDEAS §1.12 built it for that).
2. **The transformer runs three phases at three lengths**, which stage 4b's
   "34 x 8 x 49.3 ms" quietly assumed away: noise refiners over the image
   tokens, context refiners over the caption (2.60 ms a block, 19x cheaper),
   the 30 layers over the two concatenated — so 30 of 34 run at 4224, not
   4096. The two nearly cancel: 12.8 s measured against 13.4 s projected.
3. **The fp32 CPU chain is what makes the fp16 chain's error readable.** Six
   chained blocks drift to 1.8e-1 of the tensor's RMS, over twice a single
   block's bound, and nothing about that figure alone says rounding rather than
   wiring. The identical chain in fp32 lands at 4.2e-4 — worst element 1.6e-5
   of itself — and each block run on the *reference's* input is inside the
   single-block bound. Both halves were needed.
4. **Sizing a weight for a layout it is never read in cost 0.35 GB.** The
   single-block code sized every weight for the largest of the three B layouts
   so the `wrongBLayout` control could not run off the end; across 34 blocks
   that is 2.9% of the arena, so it is now conditional on the control.
5. **Wall clock is +0.3-0.4% over the sum of the dispatch timings.** 612
   dispatches in batches of eight is 77 fence waits against 1.6 s of GPU work.
6. **A block is sub-linear in tokens between 4096 and 4224** — 49.27 to 49.81
   ms, +1.1% for +3.1% of GEMM rows and +6.4% of attention. Unexplained; the
   candidate is §5.1b and the fact that 4096 is the one power of two in the
   model. One cell, 2.6%, so nothing is concluded from it yet.

**How it was measured:** `go run ./cmd/ditstack -image 4096 -caption 128`, best
of three, GPU timestamps per dispatch, run twice plus a third with `-v`. The
runs agree to **0.997-1.008** on every cell.

**Correctness:** five tests, three controls at **263x, 351x and 496x** the
bound, and the bank split checked to be exactly identical rather than within a
tolerance. Loading is 15.1 s for 24.6 GB of fp32 narrowed and packed into 12.0
GB of fp16 (packing parallelised over output rows, 1.56x), peak host memory one
block.

**Next — stage 5, the text encoder**, and the VAE. The DiT's levers are spent:
it is 72% GEMM at 73-76% of the WMMA ceiling, and the next-largest item in the
image is the **VAE's 5.6 s**, 30% of the total, with no optimisation pass at
all. Inside the DiT what is left is ~1.0 s of pure layout (`pack v`, `narrow
ctx`) and attention's 38 TFLOP/s, now behind the GEMMs.

### Session 2026-09-14 (fifth) — stage 5a/5b: the tokenizer and the text encoder on the CPU

**Result: a prompt now becomes the tensor the DiT's `cap_embedder` consumes.**
The tokenizer reproduces HF's fast Qwen2 tokenizer **id for id on all 20
corpus cases**, raw and through the chat template; the encoder matches
transformers to **8e-6** over the 35 layers Z-Image actually runs. `go run
./cmd/textenc` is the slice end to end: **2.08 s** for a 24-token prompt,
9.17 s for a 105-token one, against a **30 ms** GPU floor. Written up in
[`research/stage-5-text-encoder.md`](research/stage-5-text-encoder.md).

**Built:**

- `zimage/tokenizer` (`tokenizer.go`, `split.go`) — byte-level BPE read out
  of `tokenizer.json` (vocab, merges, added tokens), the pre-tokenizer
  hand-rolled because Go's RE2 has no lookahead, and the chat template as the
  one hand-transcribed thing in the package.
- `zimage/qwen` (`model.go`, `load.go`) — Qwen3-4B as an encoder: NeoX RoPE,
  per-head q/k norms, causal GQA (32q/8kv), SwiGLU, and a `Trace` that names
  every intermediate so a test can walk the graph.
- `reference/dump_tokenizer.py`, `reference/dump_qwen.py` — the two oracles.
  The qwen dump also *settles* two assumptions rather than asserting them
  (below). This is the first stage whose reference is `transformers` rather
  than `diffusers`; it was installed into `.venv` this session.
- `cmd/textenc` — prompt in, hidden state and the regime arithmetic out.

**Four things worth carrying forward:**

1. **The pipeline runs 35 of the 36 layers, and the output is un-normalised.**
   `hidden_states[-2]` is layer 34's output: no last layer, no final norm, no
   lm_head. transformers v5 collects hidden states with a hook on the decoder
   layer rather than v4's list, so the index could have shifted — the dump
   runs 35 layers by hand and measures the difference at **exactly 0**.
2. **The pipeline's padding to 512 is a measured no-op.** Right padding under
   causal attention cannot reach a real token; the dump compares the padded
   run masked back down against an unpadded one and gets **exactly 0**, so
   `Forward` runs at the prompt's own length — 21x less work at 24 tokens.
3. **Qwen3's massive activations break an RMS-normalised error bound.** The
   absmax is **109-238x** the RMS on these hidden states, so an fp32
   summation-order difference on one outlier reads as 7.5e-4 of the tensor
   and 9.4e-6 of the element it sits on. `hidden_2` failed the 2e-4 bound
   every other stage passed. The denominator needs a per-element floor,
   `max(|want|, rms)`; with it every stage lands at 2e-6 to 2e-5 and the four
   negative controls still fail by 6500x-160000x. Third pipeline stage, third
   answer to "normalise by what" — it is a property of the model.
4. **A negative control is only as good as the corpus.** Breaking the merge
   table is caught by 19 of the tokenizer's 20 cases and breaking the byte
   alphabet by 19; breaking special-token handling is caught by **one** — the
   single case with `<|im_end|>` written out in it, and the chat template
   rides entirely on that mechanism.

**How it was measured:** `go run ./cmd/textenc -reps 2` at two prompt lengths;
the forward pass is pure CPU fp32 over 32 cores and the two lengths agree on
the rate to 1%. Correctness is `go test ./zimage/tokenizer ./zimage/qwen`:
20 cases x 2 (raw and templated), a round trip, the NFC gap pinned as a gap,
14 stages of layer 0, four hidden states of the full stack, and two prompts
end to end.

**Next — stage 5c, and it is not stage 4 again.** The DiT is compute-bound at
16384 flop/byte; this model runs the same kind of layer at `T` flop/byte,
which for a prompt is 8-512 against the same 235 crossover. At T=24 the DiT's
winning `wg128x256` tile would fill 24 of its 128 rows. The kernel question is
§1.7/§1.9's (GEMV, and where an M block turns over between M=1 and M=256), the
weights are 7.06 GB of fp16 — two banks at this device's 4.29 GB limit, stage
4c's machinery unchanged — and unlike the DiT this model *is* worth
quantizing, because here 4-bit weights buy time as well as footprint.

### Session 2026-09-14 (sixth) — stage 5c: the text encoder on the GPU

**Result: 2.08 s to 57 ms, 36x**, which takes the text encoder from 11% of
the image to **0.3%** and the whole pipeline to **18.5 s**. 35 layers,
7.07 GB of fp16 weights in two storage buffers, 21 dispatches per layer,
matching transformers to **7.9e-4**. Correct on its first run against the
reference dump, stage by stage. Written up in
[`research/stage-5-text-encoder.md`](research/stage-5-text-encoder.md).

**Built:**

- `zimage/qwen/gpu.go` — `GPUEncoder` over the DiT's four-arena binding
  layout and push-constant block, with stage 4c's bank machinery unchanged.
  `SetPlan`/`PlanFor`/`AutoPlan` re-plan per run.
- `shaders/qwen_rope.comp` — NeoX rotary, its own file rather than a flag on
  `dit_rope.comp`: which convention a model wants is a property of the
  checkpoint, and a build flag would let a graph pick the wrong one silently.
- `CAUSAL` and `GQA` on `dit_attention_wmma.comp`, plus five narrow-M rungs
  of `dit_gemm.comp`. The DiT's own builds come out instruction for
  instruction identical (checked with `spirv-dis`, ids normalised).
- `zimage/qwen/load.go` gains `LoadLayer`/`LoadEmbedding`, so the GPU path
  stages one layer at a time — 404 MB of host memory, not 14.1 GB.
- `cmd/textenc -gpu [-ladder] [-tokens ...] [-v]`.

**Four things worth carrying forward:**

1. **The winning kernel moves with the prompt length, and that is new.** In
   the DiT, once the weight was fragment-tiled and the grid swizzled, one
   kernel won all three shapes. Here seven rungs over the *same* weights
   cannot agree, because T is not a shape the kernel sees — it is the
   arithmetic intensity itself. 16-row and 32-row tiles win at 16-64 tokens,
   a 64-row tile at 128, and the DiT's own 128x256 at 256+. Taking the DiT's
   default at a 24-token prompt costs **1.24x**; taking the narrowest at 512
   costs **1.92x**. Re-planning is free — every rung reads the same staged
   weight — so `AutoPlan` applies the measured table per run and `SetPlan`
   makes the ladder one 7 GB load instead of seven.
2. **The run walks the roofline in one sweep.** 24 to 512 tokens: 2979 to
   21227 GFLOP/s while the weight bandwidth falls 124 to 42 GB/s. Same
   weights, read once, under 21x the work. It is the clearest measurement of
   §3.4's crossover the project has: below it the GB/s column is what moves,
   above it the GFLOP/s column.
3. **Two bandwidth hypotheses falsified, cheaply.** At 24 tokens the best
   rung streams a never-reused weight at 124 GB/s of a 236 GB/s bus. Wider
   in N (eight B fragments per K step) and deeper in K (an eight-tile slab)
   both came out **slower**, 1.08x and 1.10x, so it is not per-wave request
   count. What the profile does show is a *size* effect: 49.8 MB dispatches
   reach 157 GB/s and 5.2 MB ones reach 64. The lever that follows is
   concatenating q/k/v into one [6144, 2560] weight — one dispatch instead of
   three, priced at 5.6% of the run.
4. **A causal mask has to be applied twice.** Once when the row max is taken
   and once on P. Masking only P leaves `exp2(real - future_max)`
   underflowing to zero in fp16 whenever a masked score beats every real one
   by more than ~24 in log2 units, and the kernel then divides by a row sum
   of zero. This is why the row-max scan breaks at the diagonal.

**How it was measured:** `go run ./cmd/textenc -gpu`, best of three, twice —
the two runs agree to 1.004 at 24 tokens and 1.000 at 128. GPU timestamps per
dispatch for the breakdown; wall clock also carries the host gather and the
read-back, which at 128 tokens is 6.5 ms of 71.

**Correctness:** five tests — one layer walked stage by stage against the
reference dump (14 stages, worst 4.1e-3), all 35 layers end to end for two
prompts (7.9e-4), every ladder rung agreeing to the last bit, four banks
bit-identical to one, and five negative controls at **430x to 135000x** the
bound.

**Next — stage 6, the scheduler and the driver.** The three pieces all run;
what does not exist is the thing that runs them in order: FlowMatchEuler over
8 steps, the transformer's three phases, `cap_embedder` fed from the encoder's
output *on the device* rather than through a read-back, and a PNG at the end.
The optimisation backlog is in the research note and none of it is on the
critical path — the VAE's 5.6 s is 30% of the image and has had no pass at
all, and it is the only large item left.

### Session 2026-09-14 (seventh) — stage 6: the pipeline

**Result: `prompt → PNG`.** `go run ./cmd/zimage -prompt "a red fox sitting in
fresh snow, photograph"` writes a 1024x1024 image in **19.8 s** and reproduces
diffusers' fp32 CPU pipeline to **2.5e-2** of the decoded image. Two runs agree
to 19.79 s and 19.79 s. Written up in
[`research/stage-6-pipeline.md`](research/stage-6-pipeline.md).

| stage | wall | share |
|---|---|---|
| text encoder (17 tokens, tokenizer included) | 62 ms | 0.3% |
| caption refiners, **once per image** | 6 ms | 0.03% |
| denoising, 8 x 1.76 s over 4128 tokens | 14.05 s | 71.0% |
| VAE decode | 5.65 s | 28.6% |
| **total** | **19.79 s** | |

**Built:**

- `zimage/dit/head.go` — everything the transformer does around its blocks:
  `t_embedder`, `x_embedder`, `cap_embedder`, the two learned pad tokens, the
  final layer, patchify/unpatchify and the 3-D position ids. 130 MB of small
  linear layers and almost entirely convention.
- `zimage/pipeline` — `FlowMatchEuler` (the schedule, the conditioning value
  and the Euler step) and `Pipeline`, which holds every stage resident and
  runs them in order. `Generate`/`GenerateFrom`, a `Step` progress callback
  carrying the latents, and `Timings`.
- `cmd/zimage` — prompt in, PNG out, with the per-stage wall clock.
- `GPUStack.Upload(x, at, rows)` / `SetAdaLN` / `Rows` — what the three phases
  need and `Apply` could not express: write a tensor at a row offset and state
  the run's length, so the caption's 32 rows sit behind the image's 4096 in one
  arena with no copy.
- `GPUStack.FFScale` and a scale on `shaders/dit_swiglu_f16.comp`.
- `reference/dump_zimage.py` (the head, with the transformer built at
  `n_layers=0`), `reference/dump_zimage_run.py` (the whole pipeline in fp32 on
  the CPU at 256x256) and `reference/dump_zimage_step.py` (one forward pass
  with hooks on each phase boundary).

**Four things worth carrying forward:**

1. **SwiGLU overflows fp16 on a real prompt and never on a random one.** The
   gate and up projections peak at 276 and 256, their product is 7.1e4 against
   65504, and **one infinity in a row of an A operand makes that whole row of
   the GEMM's output a NaN** — so the latent was entirely NaN by step 2 and the
   image was black. On stage 4b's *random* fixture the unscaled peak is already
   5.12e4, 78% of the range: the margin was 28% and a real prompt spends it.
   The fix is free because it never has to be undone — w2's output is read by
   an RMS norm and by nothing else, and an RMS norm is invariant to a positive
   scale on its input, so `FFScale` = 1/16 lives in one shader and no consumer
   is told. `TestGPUBlockFFScale` holds that invariance to 8e-7.
2. **A new push constant on a shared shader is a landmine, because there is no
   default that means "not set".** Adding the scale zeroed `zimage/qwen`'s
   feed-forward, which builds the same shader and left `pc.scale` at 0. The
   encoder did not produce zeros: it produced a tensor with a plausible RMS
   and shape, missing only its massive activations — absmax **56.8 where the
   reference has 13753** — and the image was colourful noise. Qwen's scale has
   to be 1, because its SwiGLU output feeds a GEMM whose output feeds a
   *residual add* rather than a norm.
3. **Validate the composition, at a small size.** Every component's oracle
   passed while the pipeline made a black image. The whole diffusers pipeline
   in fp32 on the CPU at 256x256 is 3.2 s a step — eight steps and the VAE in
   half a minute — because the composition does not know how big the image is.
   It found all three of this stage's bugs in one sitting. The bound it uses is
   **relative L2**, not max-over-RMS: a denoising trajectory is chaotic, so one
   element of the last latent can be far out while the image is the same image.
   The measured per-step sequence is 1.4e-4 rising to 2.1e-2, i.e. the fp16
   chain entering at 1.4e-4 and the feedback multiplying it by ~1.6 a step.
4. **A dispatches-per-submit cap is a proxy for a time cap, and stops being a
   good one when the dispatches differ 100x.** The VAE's 8 was fine at every
   size stage 2 measured and returns `VK_ERROR_DEVICE_LOST` at 1024x1024,
   because one dispatch — the mid-block attention over 16384 rows — is **1.51 s
   on its own, 27% of the decode**. It is now 4, which is the same wall clock
   as 2 to within 1%.

**How it was measured:** `go run ./cmd/zimage -width 1024 -reps 2`, wall clock,
two process runs. `go run ./cmd/vaeprof -size 128` for the decode's profile.

**Correctness:** `go test ./zimage/pipeline` is the stage's own — eight steps
beside diffusers' from the same initial latent (per-step relative L2 1.4e-4 to
2.1e-2, image 2.5e-2) and three negative controls at 27x-56x the bound. Note
the third, **caption positions from 0 instead of 1, lands at only 2-3x**: the
off-by-one the checkpoint invites is the one the tests barely catch, which is
the argument for keeping the end-to-end bound as tight as the measurement
allows. `go test ./zimage/dit` gains the head's seven tests (four controls at
219x-48474x) and `TestGPUBlockFFScale`.

**Next — optimisation, with the profiler in front of it.** Two items:

- **The VAE is 28.6% of the image and has had no pass at all.** fp32
  throughout, conv3x3 at 3.0-3.2 TFLOP/s against a 55.5 ceiling, one attention
  dispatch at 27% of the decode. This is the whole of the remaining headroom
  worth the name.
- **The head and tail onto the device, 1.3 s, 6.4%.** One item, not two: the
  CPU head and tail cost 0.47 s, and *because* they are on the CPU the
  [4128, 3840] residual stream crosses the bus twice a step (~0.8 s) where the
  [4096, 64] latent is 1 MB. Two GEMMs at shapes the kernel already covers
  (K=64, N=64) plus a LayerNorm.

The DiT's own levers are spent (72% GEMM at 73-76% of the WMMA ceiling) and the
text encoder is 0.3% of an image. `GOALS.md`'s other four models have not been
started, and the HTTP API has not either.

### Session 2026-09-14 (eighth) — stage 7: the VAE mid block on the matrix cores

**Result: the first optimisation pass chosen by a profiler.** The decode's
second- and third-largest costs were the mid block's attention (**1.42 s in one
dispatch, 26%**) and its four projections (**503 ms at 62 GFLOP/s**), both
still stage 2b's fp32 correctness kernels. On fp16 operands and
`coopMatMulAdd` they are **15.1 ms** (36.4 TFLOP/s) and **1.12 ms**
(30.6 TFLOP/s) — 94x and 449x. The decode is 5.65 s → **3.62 s** and an image
19.79 s → **17.82 s**. Written up in
[`research/stage-7-vae-mid-block.md`](research/stage-7-vae-mid-block.md).

| | before | after |
|---|---|---|
| mid-block attention, 16384 rows x 512 | 1.429 s | **15.1 ms** |
| the four 512x512 projections | 503 ms | **1.12 ms** |
| VAE decode, 1024², GPU total | 5.53 s | **3.56 s** |
| VAE decode, wall | 5.65 s | **3.62 s** |
| an image, 1024², 8 steps | 19.79 s | **17.82 s** |

**Built:**

- `shaders/vae_attention_wmma.comp` — flash attention where the **workgroup**
  owns the head rather than the wave. Seven builds over QT/KTIL/WAVE plus two
  negative controls.
- `shaders/vae_pack_f16.comp` (two builds, natural and transposed tiles) and
  `shaders/vae_narrow_f16.comp` — the fp16 operands, with the projections'
  biases folded into the pack.
- `zimage/vae/gpu_wmma.go` — the kernel ladders, the fragment-tile weight
  staging, and the mid block's graph. `Options`/`NewGPUDecoderOpts`,
  `AttnKernels()`, `GEMMKernels()`.
- `vae_common.glsl`'s push-constant block grew the six GEMM fields at
  `dit_common.glsl`'s byte offsets, and `vae_rows_to_nchw_add.comp` an optional
  bias. The VAE's pipelines are now built over four buffers.
- `cmd/vaebench -ladder` sweeps both kernels on GPU timestamps;
  `cmd/vaeprof -attn/-gemm` profiles a chosen pair. Both commands (and the
  package's tests) now ask for fp16, cooperative matrices and subgroup-size
  control, and fall back to stage 2b's fp32 path without them.

**Four things worth carrying forward:**

1. **A head dimension is a register budget, and 4x the DiT's does not fit.**
   `dit_attention_wmma.comp` keeps `HEAD_DIM/16` output accumulators per query
   tile — 8 at the DiT's 128, **32** at the VAE's 512 — and built at 512 it
   asks for **424 VGPRs against a 256-register file, spilling 170 into 30 KB of
   scratch**. `RADV_DEBUG=shaderstats` said so before a line of host code was
   written. The fix is to give the head to the workgroup: four waves, each
   owning an eighth of the component axis, which splits `S = Q.K^T` (a
   *reduction* over that axis, so the partials are summed in LDS) and
   `O += P.V` (*indexed* by it, so each wave owns its own output tiles) in
   opposite directions. 192 VGPRs, no spill, 24 KB of LDS.
2. **This attention cannot be validated end to end, and that is a property of
   the model.** Stage 2 found the mid block's softmax is a hard one-hot whose
   argmax is set by `||k_j||`. Stage 7 hit the consequence: the
   `NO_CROSS_WAVE` control — which **throws away three quarters of every dot
   product** — decodes the reference image to **8.2e-4**, *inside* the
   tolerance the correct kernel meets, while `NO_RESCALE` lands 634x outside
   it. A kernel whose Q/K addressing ignored the wave index would have passed
   every image test in the package. So the kernel is tested on its own, on
   unit-normal rows where the distribution is soft, and there the same control
   is **833x** out. `TestGPUWMMAControls` now asserts *both* directions, so
   that if the image ever becomes score-sensitive the test says to promote the
   control rather than widen a bound.
3. **Reusing a tuned kernel is cheaper than forking it, and the price is a
   push-constant convention.** The projections run on `dit_gemm.comp`
   *unmodified*; what that cost was making `vae_common.glsl`'s block the same
   88 bytes as `dit_common.glsl`'s with the six GEMM fields at the same
   offsets (`inOff` and `outOff` already coincided). The alternative was
   forking a kernel that carries §2.1, §2.3, §2.4 and §2.7 behind it. Its
   missing bias went into the two passes that were already reading those
   tensors — the pack and the residual add — for no extra dispatch.
4. **wave32, a fourth time.** 15.1 ms against 23.6 at identical tiling,
   **1.56x**, and here it is the only thing separating the ladder's winner from
   its middle. §2.7's hoisted K-slab also flips sign: a *win* of 4% on these
   projections where stage 4a measured it a loss on the DiT's `ff.w13`.

**How it was measured:** `go run ./cmd/vaebench -ladder -size 128` (GPU
timestamps, best of three, every build), run twice and agreeing to 1.003 on the
winner; `go run ./cmd/vaeprof -size 128` for the decode's profile;
`go run ./cmd/vaebench -sizes 16,32,64,128` and `go run ./cmd/zimage -reps 2`
for wall clock (17.84 s and 17.79 s). Register and LDS figures per build come
from `RADV_DEBUG=shaderstats,nocache go run ./cmd/probe`.

**Correctness:** `go test ./zimage/vae` gains five tests — the fp16 path
against diffusers over every build (8.2e-4), against the fp32 path on the same
device (8.2e-4), at a latent whose row count is not a multiple of the key block
(1.7e-4), the two image-level controls, and the kernel test above (2.0e-3 with
the controls at 833x and 2718x). The fp32 tests are unchanged and now name
`Attn: AttnScalar` explicitly. `go test ./zimage/pipeline` still measures
0.0252 against diffusers at 256², i.e. the mid block's fp16 does not show up in
the end-to-end number at all.

**Next — conv3x3, which is now 85% of the decode** (3.02 s at 3.0-3.2 TFLOP/s,
40-odd dispatches, the largest 404 ms) and 17% of an image. Unlike everything
else optimised so far it is not a GEMM with the wrong kernel in front of it:
its implicit GEMM has never been written. Stage 2's overflow constraint is
what stands in the way and stage 7 measured the shape of the answer — conv's
inputs peak at 497 while its accumulators need fp32, which is what the matrix
cores do natively. After that the only item left in the image is the DiT's CPU
head and tail (1.24 s, 7.0%).

### Session 2026-09-14 (ninth) — stage 8: conv2d on the matrix cores

**Result: the VAE is done.** conv3x3 was 85% of the decode at 3.0-3.2 TFLOP/s
and is now **244 ms at 41-42 TFLOP/s**; the decode is 3.62 s → **876 ms** and
an image **17.82 s → 15.00 s**. The largest dispatch in the whole pipeline
went from 404 ms to 30.3 ms. Written up in
[`research/stage-8-vae-conv.md`](research/stage-8-vae-conv.md).

| | before | after |
|---|---|---|
| `conv3x3 512->512 @512x512` | 404.2 ms, 3060 GFLOP/s | **30.3 ms, 40841** |
| all conv3x3 | 3.017 s | **244 ms** |
| all conv1x1 | 153 ms | **12 ms** |
| VAE decode, GPU total | 3.557 s | **801 ms** |
| VAE decode, wall | 3.62 s | **876 ms** |
| an image, 1024², 8 steps | 17.82 s | **15.00 s** |

(The 876 ms is `cmd/vaebench`'s, and stage 9 found 71 ms of it is a read-back
the pipeline does not pay -- 12.58 MB at the 0.18 GB/s a small program gets.
In an image the decode is **805 ms**.)

**Built:**

- `shaders/vae_conv_wmma.comp` — conv as an implicit GEMM, `C[OC, HW] =
  A[OC, taps*C] * B[taps*C, HW]`, with B addressed rather than materialised
  (im2col at 1024² would be 4.7 GB). Nine builds over BM/BN/BK/WAVE, plus a
  negative control.
- `shaders/vae_pack_conv.comp` — the blocked fp16 layout
  `[ceil(C/16)][H+2][W+2][16]`, plus a second build as the padding control.
- `zimage/vae/gpu_conv.go` — the ladder, the filter packing, the expanded
  biases, `ConvKernel`/`ConvKernels()`, and the conv half of the graph.
- `Options` grew `Conv`, and `chooseKernels` now resolves the mid block and
  the convolutions **independently**, so a test can narrow one and not the
  other. Every stage-7 test says `Conv: ConvScalar` explicitly.
- `zimage/vae/gpu_conv_test.go` — six tests: the fp16 headroom on real
  activations and on the filters, every build against the scalar path, the
  whole fp16 decoder against diffusers, a 96x96 image, and the two controls.
- `cmd/vaebench -convladder`, `-conv`; `cmd/vaeprof -conv`; the fp16 arena in
  both outputs; `Pipeline.Residency` grew a VAE column; `cmd/probe`'s
  push-constant range grew to 128 bytes (the VAE block is 88 and did not fit).

**Five things worth carrying forward:**

1. **Which axis you tile is decided by the access pattern, not by the
   operand.** Every WMMA kernel in this engine stores its operands as 16x16
   fragment tiles, and for conv that is *wrong on the pixel axis*: a tiled
   pixel axis makes the `dw = ±1` tap window straddle two tiles, which a
   fragment load cannot express. Tiling the **channel** axis instead —
   16 channels contiguous per pixel — makes a patch fragment 512 contiguous
   bytes **at any pixel offset**, which is a column-major load of stride 16.
   That one choice is the stage.
2. **Put the padding in the data.** A one-pixel zero border on the packed
   layout means every address the kernel forms is a real address holding
   either an activation or a zero, so nine taps are nine adds — no masking, no
   branch, no K-axis bounds check. It costs 0.4% of the tensor at 1024².
3. **§2.7 is a fix for an under-covered load, not a general lever.** The
   K-slab knob measured 1.00x here, because both operands are already fully
   covered 512 B reads and there is nothing left for more outstanding bytes to
   fix. Stage 3c got the same answer for the same reason.
4. **A test at a deliberately awkward size earns its keep.** The kernel clamps
   every pixel address into its own channel plane so nothing can leave the
   buffer; the clamp lands on the bottom border row, which is zeros —
   *provided the padded row is at least 16 pixels wide*. At the 12x12 latent
   `TestGPUConvOddSize` decodes it is 14, so the clamp reached back into the
   last real row and the last output row got two columns of the wrong pixels.
   Every build failed identically at the same element, which is what said the
   bug was in the shared addressing. The fix is `max(W+2, 16)`. Neither the
   1024² image nor the 16x16 reference latent can have this bug.
5. **A bump allocator has to give blocks back to the bump pointer.**
   `arena.release` only ever pushed onto the free list, so a sequence of
   allocate-free-allocate in *increasing* size — which is exactly what the
   packed activations are, 17 MB then 68 then 270 then 539 — stranded every
   block behind the next. The fp16 arena was the sum (933 MB) instead of the
   largest (539 MB). Four lines; the fp32 arena gained 168 MB too, which
   matters against the 4.29 GB storage-buffer limit.

**How it was measured:** `go run ./cmd/vaebench -convladder -size 128` (GPU
timestamps, best of two then best of three, the runs agreeing to 1.005 on the
winner); `go run ./cmd/vaeprof -size 128` with and without `-conv scalar` for
the distribution; `go run ./cmd/vaebench -sizes 16,32,64,128` and
`go run ./cmd/zimage -reps 2` for wall clock (15.00 s and 15.04 s). Register
and LDS figures from `RADV_DEBUG=shaderstats,nocache go run ./cmd/probe`:
144 VGPRs at wave64, 192 at wave32, no spills, 1-4 KB of LDS.

**Correctness:** `go test ./zimage/vae` gains six tests. The fp16 convolutions
land **8.2e-3** from the fp32 graph — a chain of 40-odd of them, where stage
7's 8.2e-4 was one block — and the whole fp16 decoder lands **3.3e-3** from
diffusers; the bound is a max over elements, so the two are not additive and
the second is legitimately smaller. `go test ./zimage/pipeline` still measures
**0.0252** against diffusers at 256², i.e. none of this shows up end to end.
Both controls fire at **312x** and **271x** — and `pad_clamp`, which touches
only the image's frame, is the opposite of stage 7's result: a decoder's
border is not information-bottlenecked the way its mid-block softmax is.

**Next.** The VAE is 5.4% of an image and has no arithmetic left in it: its
largest kernel is now a **group norm**, and four bandwidth-bound elementwise
passes (groupnorm, pack, silu, add) are 63% of the decode. They run back to
back over the same gigabyte, so what it wants is **fusion** — specifically a
SiLU whose only consumer is a convolution writing the blocked fp16 form
instead of its fp32 output, worth ~200 ms of 876. But that is 1.3% of an
image. The item that is actually next is **the DiT's CPU head and tail**
(1.24 s, 8.3%): two GEMMs at shapes the existing kernel already covers plus a
LayerNorm, which also stops the [4128, 3840] residual stream crossing the bus
twice a step. `PIPELINE.md` has both.

Housekeeping: `PIPELINE.md` is 330 lines against its own ~200-line budget,
and the next stage that closes should pay some of that back by moving closed
detail into `research/`.

### Session 2026-09-14 (thirteenth) — stages S1-S5: parakeet transcribes on the CPU

**Result: `go run ./cmd/asr testdata/jfk.wav` prints the right words.** 11 s of
audio in, the reference's transcript out — *exactly*, emission for emission —
in 2.68 s of CPU, which is 4.1x real time before a single shader exists. The
whole STT vertical up to the Vulkan port is done: `SPEECH.md` S1 through S5,
rewritten with the results and with S6's target shapes.

**Built:**

- **`audio/`** — WAV in and out (16-bit PCM only, deliberately), a radix-2 FFT
  in float64, a centred STFT that reproduces `torch.stft(center=True,
  pad_mode="constant")`, and a Slaney mel filterbank that reproduces
  `librosa.filters.mel` to a float32 ulp. The inverse transform is written and
  round-trip tested but unused: it is kokoro's iSTFT.
- **`parakeet/`** — the model in Go: the front end, the `dw_striding`
  subsampling stack, 24 FastConformer layers with relative-position attention,
  the LSTM prediction network, the joint, the TDT greedy loop and a
  decode-only Metaspace tokenizer.
- **`cmd/asr`** — WAV in, transcript and stage profile out; `-v` gives the
  per-emission trace with timestamps, which is how the alignment was eyeballed
  ("▁And" at 0.24 s, "▁country" at 10.2 s).
- **`reference/dump_parakeet.py`** — 59 tensors and a manifest, walking the
  whole model rather than its ends, with four self-checks in it.
- **`testdata/jfk.wav`** — the fixture, 11.000 s, public domain, its decode
  pinned by a sha256 against CPython's `wave`.
- **`safetensors`** gained **I64**, carried but not convertible, so
  `cmd/inspect` can open the checkpoint and still report the 24 BatchNorm
  `num_batches_tracked` scalars honestly rather than skipping them.

**Five things the checkpoint said that the survey had not:**

1. **The subsampling is separable.** NeMo's `dw_striding`: `conv2d(1→256,
   stride 2)` then *twice* `depthwise(256, stride 2) + pointwise(256→256)`.
   The collapsed shape table showed `[256, 1, 3, 3]` five times and hid the
   grouping.
2. **The attention is full, not windowed.** No `att_context_size`, and the
   mask transformers builds is padding only. At 138 frames that is free; at
   3000 (30 s) it is a 3000x3000 score matrix per head, and it is why a long
   clip is a different problem.
3. **A clip's two length formulas disagree by one.** 176000 samples → 1101 mel
   frames of which 1100 are valid → 138 encoder frames, all valid. The
   trailing frame is zeroed, excluded from the normalisation statistics, and
   absorbed by the subsampling.
4. **The blank's embedding row is zeros**, so the first prediction step can
   only be validated through the LSTM state it leaves behind.
5. **`bias_ih` and `bias_hh` are both stored** for every LSTM layer although
   only their sum can matter at inference — a CuDNN artefact, summed once at
   load.

**The one open kernel question, answered before a shader was written.**
Conformer relative-position attention is two score matrices summed —
`(q+bias_u)·k^T` over key positions and `(q+bias_v)·rel_k^T` over *relative*
offsets — and transformers folds the `[T, 2T-1]` second one onto the `[T, T]`
grid with `_rel_shift`, a left pad and a reinterpret of the flat buffer. The
closed form, checked exactly (max abs 0) against the reference for all eight
heads, is

    shifted[i][j] = raw[i][T-1-i+j]

i.e. column j of row i is the score for relative offset i-j, read off a
diagonal. The shader needs that index, not the pad-and-reinterpret. There is a
test pinning it against the mistake it is usually confused with — the middle-T
slice — which agrees on exactly one row.

**Every bound has a negative control.** The stage comparisons are relative
(the stages' rms spans 0.02 to 400), and each bound is paired with mutations
that have to miss it by at least 10x: the window convention, the preemphasis,
the mel scale, the filterbank normalisation, the content bias, the BatchNorm
fold, the GLU halves, and the sin/cos interleave against RoPE's half-split.
They miss by 119x to 6411x. The end of the chain needs no tolerance argument
at all — the transcript is a string, and it is the right one.

**Two places the *reference* is the imprecise side**, both computed in float64
here and rounded once: the power spectrum, where float32 loses three digits
between the loudest bins and those six decades down, and the position
embeddings, where torch holds the angle in float32 and one ulp at an offset of
137 frames is already 1e-5.

**The profile, which is S6's brief.** The encoder is 91.5% of the time and
~180 GFLOP (89 G multiply-adds) for eleven seconds; the front end is 0.7% and
the decode 7.7%. Per layer the work is 63% feed forward, 16% q/k/v/o, 12%
convolution pointwise, 8% `relative_k_proj` and 2% attention proper — the GEMM
ladder this repository already has, at **M = 138**. Three things it does not
have: stride-2 conv2d over a 1-channel input (2.8 GFLOP, runs once, a
correctness problem rather than a performance one), the rel-shift index inside
the score kernel, and LayerNorm with a mean (which `dit_final_norm.comp` has).

**The ladder was re-measured at these shapes** rather than assumed, after the
parakeet rows in `bench/modelshapes.go` were corrected against the checkpoint
— they were missing `relative_k_proj` (8% of the layer, and an M of 2T-1) and
the subsampling linear, and had the projector running per emitted token when
it runs once per frame. The winner at M=138, 384 and 767 is the same rung,
**`wmma_reg32_bt_hkab4_w32_padab128`, the wave32 narrow-M build**, and at
M=138 it beats the wave64 rungs by nearly 2x (23.1 against 12.7 TFLOP/s) and
the DiT's 128x256 tile by 2.3x (10.6), most of that being tile padding: 138
rounds to 160 rows at BM=32 and to 256 at BM=128. At M=1024 `reg64_bt_hka4`
takes the lead back, so `qwen.PlanFor`'s shape — pick the rung from the
sequence length — is the plan here too.

A projection's intensity counting weight traffic alone is `2M/bytes`, i.e.
**exactly M flop/byte at fp16**, so against the 235 crossover the encoder is
memory-bound below 235 frames (18.7 s of audio) and compute-bound above it —
the inverse of the DiT, every shape of which was compute-bound. The measured
29.2 TFLOP/s at M=138 on `[138,1024]x[1024,4096]` is 90% of that shape's
bandwidth ceiling and 53% of the matrix rate, which is the same story from the
other side. **The target for S6**: 177 GFLOP at ~26 TFLOP/s is 6.8 ms and the
1.25 GB of fp16 weights are 5.3 ms at 236 GB/s, so an 11 s clip should encode
in ~7 ms against 2.45 s on the CPU — 350x, and ~1500x real time.

**fp16 was tested, not assumed.** `Model.SetF16` narrows every matrix-core
operand on the CPU reference — weights in place, activations into each
projection, norms and residuals left alone — and **the transcript and the
whole decode trace are unchanged**. The cost is 4.3% relative drift at the
encoder output against fp32's 0.017%, accumulating layer by layer (0.7% at
layer 0, 1.4% at layer 12). The consequence for S6 is a validation strategy
rather than a tolerance: **the bound is the transcript**, and per-stage
tensors locate a fault rather than certify its absence.

### Session 2026-09-14 (twelfth) — z-image parked, and the two speech verticals surveyed

**No code.** The z-image slice is parked at **14.26 s an image** and the next
work is the two audio models in `GOALS.md`. `SPEECH.md` is the plan and is now
the file to read first; `PIPELINE.md` keeps the z-image state with its resume
points at the top.

**Why park here.** What is left in z-image is one deep problem — the last
quarter of the WMMA ceiling on seven GEMMs, 2.6 s an image — and it will keep.
Parakeet and Kokoro are 0.63 B and 0.08 B against 6.2 B, they reuse most of
the engine, and each closes a whole capability rather than a percent.

**What the survey found.** Both checkpoints are already on disk and were read
rather than assumed (`SPEECH.md` has the inventories):

1. **Parakeet ships as HF `transformers`, not NeMo.** `config.json` says
   `ParakeetForTDT`, there is a `model.safetensors`, and the `.venv`'s
   `transformers 5.17.0` has `ParakeetForTDT`/`ParakeetProcessor`. **The oracle
   exists today with no conversion work** — the same position stage 5 was in
   with Qwen3, and the `.nemo` and `.gguf` beside it are not needed.
2. **`safetensors` cannot open it**, and for one reason: 24 of its 723 tensors
   are BatchNorm `num_batches_tracked`, scalar **I64**, which inference never
   reads. That is the first concrete task and the note states the decision to
   make.
3. **Its joint head is 8198 wide, not 8193** — 8193 token logits (blank at
   8192) followed by **5 duration logits**, which `generation_config.json`
   confirms by suppressing 8193-8197 from the token argmax. That is the whole
   of TDT in one tensor shape.
4. **Its tokenizer is Metaspace BPE, not byte-level**, so `zimage/tokenizer`
   does not transfer — but ASR only ever *decodes*, which is a vocab lookup
   and a `▁ -> space` rule. Encoding is not needed at all.
5. **Kokoro is a pickle**, five submodules, and two thirds of its 82 M
   parameters are the iSTFTNet vocoder. Its ALBERT is 25 tensors for 12 layers
   because the layers share weights. Its voices are `[510, 1, 256]`, one style
   vector per phoneme-sequence length.
6. **G2P is kokoro's real dependency and it is not a kernel.** The
   recommendation in the note is to make the first vertical take **phonemes,
   not text**, so the whole network is validatable on day one and G2P becomes
   a separable stage that can start as an `espeak-ng` shell-out.

**The recommendation: parakeet first**, though it is 8x larger. Its oracle
needs no conversion, its encoder is 90% shapes the GEMM ladder already wins at
the head dimension the attention kernel is compiled for, it has exactly one
open kernel question (Conformer relative-position attention, which
`dit_attention_wmma.comp` does not compute), and **its correctness bound is an
exact string** — the cheapest oracle this project has had.

**The interesting engineering question**, flagged in the note rather than
answered: every kernel here was tuned at M=4096 with 12.5 GB resident, and
neither of these models is a residency problem — parakeet is 1.25 GB at fp16
and a 30 s clip is **375 encoder frames**. That puts it near stage 5c's 235
flop/byte crossover, where the text encoder found that **the winning tile
moves with the sequence length**. These two will test that finding rather than
inherit it.

### Session 2026-09-14 (eleventh) — stage 10: the two layout passes become epilogues

**Result: a DiT block is 16 dispatches, not 18.** `pack v` and `narrow ctx`
moved no information — each read a tensor and wrote the same numbers in the
shape the next kernel wanted — and both are now the *store instruction* of the
kernel that produced the tensor. A block goes **53.17 ms → 51.80 ms** at 4224
tokens, a step **1.70 s → 1.66 s**, an image **14.49 s → 14.26 s** (two runs,
14.18 and 14.34). Written up in
[`research/stage-10-layout-epilogues.md`](research/stage-10-layout-epilogues.md).

Both figures are the same binary: `go run ./cmd/ditstack -unfused-layout` and
`go run ./cmd/zimage -unfused-layout` keep stage 9's graph, which is the oracle
the new one is compared against.

**Built:**

- `shaders/dit_gemm.comp -DC_PACK=1` — the projection stores its accumulator
  tiles straight into `dit_pack_f16.comp`'s mode-1 layout: fp16, per head, each
  16x16 tile transposed. Eleven lines of index arithmetic and a different
  `coopMatStore`, because the transpose is not a transpose — column-major with
  stride 16 *is* mode 1.
- `shaders/dit_attention_wmma.comp -DOUT_F16=1` — the kernel writes the output
  projection's fp16 A operand instead of an fp32 context. It came out
  **shorter** than the path it replaces: the per-row `1/rowSum` becomes a
  row-constant matrix (the same trick §3.3 already uses for the online
  rescale), so `O * Cinv` and one `coopMatStore` replace the fp32 epilogue's
  LDS staging loop.
- `GPUStack.FuseLayout` (default on) and the `cpackFor` / `wmmaVariant.of16`
  companion tables, alongside the existing `cf16For`. Both kernels are
  compiled and both pipelines built, so the switch is live on the graph — which
  is what lets one test run the two paths back to back over the same arenas.
- `-unfused-layout` on `cmd/ditstack` and `cmd/zimage`;
  `Options.UnfusedLayout` on the pipeline; `TestGPUBlockFusedLayout`.

**Four things worth carrying forward:**

1. **The item was mispriced by a factor of a thousand, and the fix is
   arithmetic.** `PIPELINE.md` and four handoffs here carried this as "~1.0 s
   an image of pure layout" and made it the largest thing left at 7%. It is
   **~0.29 s** and 2%. Stage 4 measured it correctly — "together ~1.0 **ms**"
   a block — and the unit was lost the first time the number was quoted
   somewhere else. Nothing re-derived it for four sessions, because 1.0 s an
   image is entirely plausible for a thing that costs 1.0 ms in a block run
   272 times. **A number quoted between documents should carry the dimension
   it was measured in.**
2. **The saving is the bandwidth arithmetic, exactly.** The two passes moved
   252 MB a block a step at 4096 tokens — 68 GB an image — which at 236 GB/s
   is 0.29 s. That is what they measured (0.64 + 0.43 ms a block) and what
   removing them returned. Nothing here was a cache effect or a launch cost.
3. **The old path was doing a double rounding.** `v` comes out bit-identical
   over 1.47 M halves, but 84 halves in 1.23 M of the context differ by one
   ulp — and **every one of them is an exact fp16 tie in the fp32 context**.
   The old path rounded `o/rowSum` to fp32, stored it, and narrowed that to
   fp16; where the fp32 result landed exactly between two halves,
   round-to-even broke a tie the fp32 rounding had manufactured. The fused
   path hands the store the unrounded product. Established by experiment:
   adding an fp32 store of the same product to the fused kernel — forcing the
   intermediate to exist — makes all 84 differences disappear, and the two
   builds' fp32 contexts are bit-identical, so it is not upstream. The fused
   path is the *more* correctly rounded of the two.
4. **Attention's store was never on its critical path.** Stage 10 halved its
   output bytes and deleted its LDS staging loop, and it measures 7.21 → 7.19
   ms — nothing. A second measurement agreeing with stage 3c that the register
   file is what binds that kernel, and the reason its remaining 1.1x is not a
   memory problem.

**How it was measured:** `go run ./cmd/ditstack -v -reps 3` and
`go run ./cmd/zimage -reps 2`, each with and without `-unfused-layout`, two
runs of each arm. Layer blocks agree to 51.78/51.82 fused and 52.82/53.52
unfused; images to 14.18/14.34 and 14.43/14.55.

**Correctness:** `TestGPUBlockFusedLayout` asserts bit-identity for the packed
`v` plane and, for the context, that *every* differing half is an fp16 tie —
which is a stronger statement than a tolerance, since anything that was not a
rounding difference would fail it. The block's output moves 9.9e-4 of its RMS
(stage 4b's own fusions: 1.6e-3). End to end against diffusers at 256² the
pipeline goes **0.0158 → 0.0215** against a 3e-2 bound: a chaotic trajectory
landing elsewhere, not an accuracy loss — but it does eat a third of the
remaining margin, which is worth knowing before the next thing perturbs an
operand.

**Next.** There is no layout left in the block. What remains is arithmetic and
the traffic it needs: the seven projections are **9.9 s an image, 69% of
everything, at 73-76% of the WMMA ceiling** — the quarter they do not reach is
2.6 s and is an order of magnitude more than every other item together;
attention is 1.9 s at 38 TFLOP/s against the GEMMs' 40.6-42.0, worth ~0.2 s;
and the eight elementwise passes are 1.6 s, each reading the residual stream
and writing it, wanting fusion into a neighbour rather than deletion.
`PIPELINE.md` has the budget.

Housekeeping, carried on: `PIPELINE.md` is **351 lines** against its own
~200-line budget. This session added a stage (+36) and paid back 25 by moving
the VAE stages' detail into `research/`, which is the right direction and not
far enough. The two sections with the most duplication left are "What the
kernels already give us" and "What each stage found", both of which restate
`research/README.md`.

### Session 2026-09-14 (tenth) — stage 9: the head and tail on the device

**Result: no model arithmetic runs on the host inside the denoising loop any
more.** The patch embedder and the final layer moved onto the matrix cores:
the host's share of a step went **59 ms → 5-10 ms**, a step 1.75 s →
**1.70 s**, an image **14.89 s → 14.49 s** (two runs, 14.49 and 14.54).
Written up in
[`research/stage-9-head-and-tail.md`](research/stage-9-head-and-tail.md).

Both figures are the same binary: `go run ./cmd/zimage -cpuhead` keeps stage
6's host path, which is the oracle the device path is compared against.

**Built:**

- `shaders/dit_final_norm.comp` — LayerNorm (the mean *is* subtracted, which
  nothing else in this model does), the final layer's adaLN scale, and the
  narrowing into the fp16 arena, in one pass. Plus a `NO_MEAN` control.
- `zimage/dit/gpuhead.go` — `GPUHead`: two GEMMs on `dit_gemm.comp` rungs the
  block never uses, the block's own `dit_adaln.comp` for the final layer's
  modulation, and 5 MB of its own weights. It borrows the *stack's* two
  activation arenas, because the embedder writes the residual stream and the
  tail reads it: same tensor, therefore same buffer.
- `GPUStack` grew three activation slots and `writeAdaLN`, which puts the
  timestep embedding in the arena in both the forms the model wants — as it
  is for every block, and through a SiLU for the final layer. That asymmetry
  is diffusers'.
- `Options.CPUHead` and `cmd/zimage -cpuhead`; `zimage/dit/gpuhead_test.go`
  (four tests) and `TestPipelineHeadOnDevice`.
- **`cmd/bus`** — a probe for what a host read and write of a mapped buffer
  cost, and how that changes with how much has already been allocated. It is
  the tool that found the item below.

**Four things worth carrying forward:**

1. **A bias -- and a *pad token* -- can be a column of K.** `dit_gemm.comp`
   has no bias and stage 7's rule (put it in a pass already reading the
   tensor) has nowhere to go at the head, because the embedder's consumer is
   the block graph. So A gets a column holding 1 for a real token against a B
   row holding the bias, and **a second column holding 1 for a *padded* token**
   against a B row holding the learned `x_pad_token`. `y = Wx + b` and
   `y = x_pad_token` then come out of the same GEMM with no branch, no second
   dispatch and no host write of the padding. K goes 64 → 128, which the
   64-wide slab was rounding up to anyway.
2. **The engine's read-back number was wrong for the pipeline, by 83x.**
   `vk.NewBuffer` falls back from the device-local host-visible heap to the
   plain host-visible one when the first cannot serve a request, and on this
   device that heap's budget is about **8 GB** against the 83.79 GiB it
   reports. Below it a host read is 0.18 GB/s; above it, 15. The pipeline has
   20.5 GB of weights resident, so its 63 MB read-back costs **6.5 ms, not
   344**. Two things in the repo's own numbers fall out of it: `cmd/vaebench`
   reports a 1024² decode at 876 ms where the pipeline reports 805, and the
   71 ms difference is exactly 12.58 MB at 0.18 GB/s; and the **0.8 s an image
   of "residual stream crossing the bus" never existed** -- it was the
   remainder of a subtraction, not a measurement.
3. **A bound inherited from a looser path is a control that has stopped
   controlling.** Taking the block's `fp16RelTol` (1.5e-2) for the head would
   have left `NO_MEAN` -- an RMS norm where the model has a LayerNorm -- only
   **3x** outside it. Set from their own measurements the two bounds are 6e-3
   and 2e-3, and the control is 22x out. Its host twin in `head_test.go` is
   the same 4.4e-2 at 219x, against an fp32 bound.
4. **What is not the missing 100 ms.** A step is 1.70 s of wall against the
   stack's 1.60 s of GPU timestamps. It is not the host (5-10 ms), not the bus
   (9 ms), and **not submit batching**: `perSubmit` 8 → 18 (one submit per
   block) → 64 measures 1.69 and 1.68 against 1.70, inside the noise. Left
   open rather than guessed at, which is what the last item was for.

**How it was measured:** `go run ./cmd/zimage -reps 2` with and without
`-cpuhead`; `go run ./cmd/bus -buffer 1200 -chunk 63 -pre N` for the heap
threshold, bisected to between 6 and 7 GB.

**Correctness:** the embedder lands **3.4e-3** from diffusers' `x_prepared`
and the tail **8.6e-4** from its `final`, each against a bound set from its
own measurement; the padded-stream case is checked against the host path at a
token count that needs padding, which is the only test the pad-token column
has. `TestPipelineHeadOnDevice` runs a whole image both ways (latent 0.0135,
image 0.0187), and the pipeline's end-to-end number against diffusers moved
0.0252 → **0.0158** -- which is a chaotic trajectory landing differently, not
an accuracy claim.

**Next.** The image is **94% the DiT** and nothing outside it is worth a
percent. Inside the block: ~1.0 s an image of pure layout (`pack v` and
`narrow ctx`, both of which exist only to put an operand in the next kernel's
shape -- a GEMM and the attention kernel could each absorb one in an
epilogue), and attention at 38 TFLOP/s against the GEMMs' 40.6-42.0. Below
those, the VAE's elementwise fusion is ~200 ms (1.3%). `PIPELINE.md` has the
budget.

### Phase 1's last session: IDEAS §1.12 — the M block read off the routing histogram, and an expert belongs to one dispatch

The handoff's item 1 was "`down` at 256 sequences is the last cell of the
decode path under the bus — size `MROWS` from the routing histogram per
dispatch, and measure the **selection**, not another binary", with the two
cheap corner builds beside it. All of it is done, the cell closed
(**66% → 72% of the bus**, the MoE FFN at a serving batch **821 → 848 tok/s**),
and the interesting part is not the selection — which works and is free — but
what the second half of it found: **the pad-minimal plan loses**, and the
reason is a locality law no cost model written in groups and slots can see.

**New code:**
- `shaders/gemv_w4a8.comp`: the corner arm's row list extended from `NROWS<=4`
  to 8 (four macro rows, and the `#error` that forbade it replaced by a
  power-of-two check). Every pre-existing `.spv` is byte-identical after
  `go generate` (cmp-verified over all 138).
- Three builds in `shaders/shaders.go`: `v4_m8_n4`, `v4_m2_n8`, `v4_m4_n8` —
  the three corner cells §1.11 ruled out on registers it had not yet measured.
- `vk/shim.[ch]` + `vk/engine.go`: **`shim_dispatch_multi_timed` /
  `vk.DispatchMultiTimed`**, a timed sequence whose dispatches differ in their
  *pipeline* and grid as well as their push constants, with the
  within-iteration barrier as a flag. (Pipelines cross cgo **by value** — a Go
  slice of `*C.ShimComputePipeline` is a Go pointer to Go pointers and the cgo
  checker rejects it; a slice of the structs themselves holds only Vulkan
  handles and is fine.) `bench/bench.go` gets `TimeDispatchMulti` beside it.
- `bench/ops_moe_select.go` (new, ~600 lines): the cost-model fit
  (`T = groups*w + slots*s`, non-negative least squares over the four M widths
  of one (layer, batch, `NROWS`) cell), the two planners, the mixed-width table
  builder, the run pass, a GPU equivalence check, and two summary tables.
- `bench/ops_moe_select_test.go` (new): the planner and table invariants as
  host-side tests — including the one the buffer sizing depends on (only the
  last group of an expert can be padded, so an expert's pad is under
  `max(widths)`) and the two plans' defining difference (`whole` never puts an
  expert in two parts; `split`'s count of when it does is recounted from the
  table it built).
- `bench/ops_moe_gemv.go`: eight `mixed` variants (four row blocks x two
  plans), which are plans rather than builds — no spirv, skipped by the
  build-shaped tables, and recorded under the plain `grouped` mode so that the
  summaries which pick the fastest decode kernel at a batch have to choose
  between them and the fixed builds.

**What it found.** Full detail in IDEAS §1.12; the short version:

1. **The rule works and is free.** A two-term cost model picks the oracle width
   in 28 of 40 (layer, batch, `NROWS`) cells and is within 1% in 32, against a
   1.11-1.66x spread across the four widths at t=256. **Calibrating once at
   t=4 is as good as re-fitting at the batch** (28/40, 33/40) and is better in
   the cell where they disagree most.
2. **§1.11's engine number was a cache-resident one.** The group-to-slot ratio
   is 45:1 on gate_up and 6.5:1 on down *at t=4*, and **1.0-2.7:1 at t=256** on
   both. At a serving batch one expert's weight read costs about what one pad
   slot costs, which is why over-blocking stops paying there.
3. **Mixed widths beat every fixed width on the cell that was under the
   bus — if an expert stays inside one dispatch.** `whole` (one width per
   expert, repeated) is **1.09-1.13x** the best fixed build on `down` at t=256
   and 1.00-1.02x on gate_up; `split` (the pad-minimal decomposition, 5% pad
   against the fixed build's 23%) is **0.72-0.96x**. Below t=256 the two plans
   are the same table and both are worth 1.02-1.07x at t=64.
4. **Each group that lands in a different dispatch from its expert's first
   costs 3.26-5.88 us** — against **3.58 us** for a cold DRAM read of that
   expert's 844,800 bytes, and against 0.45-1.67 us for a group inside the
   part. The cost model's error tracks it exactly (-18 to -46% for `split`,
   -5 to -17% for `whole`).
5. **It is not the schedule.** A full barrier between every part costs
   **+0.2% to +1.3%** at t=256 — three barriers at §1.8's 1163 ns on a 2.5 ms
   dispatch — so the parts overlapping or not is a rounding error.
6. **Of the two cheap builds, `NROWS=8` with an M block is the one that pays**:
   `m4_n8` is the fastest fixed build at down's t=256 (2.733 ms against
   `m4_n4`'s 2.761) and `m2_n8` ties it; both lose 1.09-1.15x on gate_up, as
   §1.10's `NROWS >= 8*VEC*WAVE/K` predicts. `m8_n4` is a wash on gate_up and
   1.05x slower on down. The 32-cell corner's cost is **occupancy, not
   spills**: 48 VGPRs / 32 subgroups per SIMD plain, 84 / 18 at (4,4) and
   (2,8), 144 / 10 at (8,4) and (4,8), with zero scratch anywhere.
7. **What it is worth.** `down` at t=256: 2.761 → **2.530 ms**, 157 → **169
   GB/s** distinct, **72% of the bus**; the MoE FFN at 256 sequences **848
   tok/s** against the Q4 GEMM's 545, with `down` itself back from the GEMM
   (0.99x → **1.07x**). Single-stream decode is untouched (the M axis is a
   loss at one row per expert, so the plan picks width 1 and the mixed
   dispatch is one part).

**Also worth knowing:** the family ran **three times** this session (the first
before the `whole` planner existed). Runs 2 and 3 agree at a **median ratio of
0.9999** over all 896 common rows, p10-p90 inside 0.65%, with 33 rows outside
±3% — every one of them a ≤20 us t=1 dispatch or one of the known-noisy GEMM
cells, as in previous sessions. Committed `results/moe.csv` is run 3.
Correctness: all 29 grouped-GEMV builds pass `verifyMoEGroupedGEMV` against the
CPU reference, and every mixed plan is checked on the real bank against the
width-1 build of its own row block (same output-row numbering, same reduction
order, so they agree to 1e-4) before it is timed.

**How to re-run:** `go generate ./...` then `go run ./cmd/bench moe` (the
family ignores `-sizes`/`-blocks`). 896 rows in **~9 minutes**.

### Phase 1, previous session: IDEAS §1.11 — both blocks in one wave, and decode's throughput end reaches the bus

The handoff's item 1 was "close the decode GEMV's last corner: `NROWS` x
`MROWS` in one build", with two smaller probes beside it. All three are done.
The two blocks **compose**: at 256 sequences in flight the corner is **1.12x
each of its own axes on `gate_up` and 1.19x on `down`**, **1.10x the best
single-axis build of any width in the sweep** at both shapes, and 1.54x/2.11x
the unblocked kernel. At 16-64 tokens it ties the best single-axis build
within 1%; below that the M block is a loss and the corner loses with it.

**New code:**
- `shaders/gemv_w4a8.comp`: a third `main`, taken when `MROWS > 1 &&
  NROWS > 1` — the combination the file used to `#error` on. A workgroup is a
  group of `MROWS` slots naming one expert (§1.9's grid) and a subgroup is
  `NROWS` consecutive weight rows of it (§1.10's), so a wave holds
  `MROWS*NROWS` cells; both operands are hoisted, and `W4A8_MNXY` dots a
  weight `uvec4` against an activation pair that are *both* already in
  registers. A *cell* owns an accumulator and a partial, a *row* owns its
  weight loads and scales, a *slot* owns its activation loads and block sums,
  so there are three instantiators (`MN_ROWS`, `MN_SLOTS`, `MN_CELLS`) over one
  set of sites, all taking `(R, S)` and ignoring the half they do not use.
  Cells are emitted slot-outermost so a slot's `NROWS` stores stay adjacent
  inside the elected lane's block. This is the one place in the file that
  builds names by `##` pasting — a cell is indexed by two numbers and the
  single-axis arms' one-name-per-argument style does not extend to a product —
  but nothing is indexed dynamically, so the scratch-spill trap `W4A8_GROUP`
  documents is still avoided (confirmed: no scratch access in any build).
  The two single-axis arms' conditions became `MROWS > 1 && NROWS == 1` and
  `NROWS > 1 && MROWS == 1`, so **every pre-existing `.spv` is byte-identical**
  after `go generate` (md5-verified across the whole tree).
- Nine builds in `shaders/shaders.go`: the corner at VEC=4 (m2n2, m4n2, m8n2,
  m2n4, m4n4), the (VEC, NROWS) grid (v1_n4, v8_n4, v16_n4) and the wave32 arm
  (v4_n4_w32). `NROWS=8` is not built against an M block (past the register
  knee on its own).
- `bench/ops_moe_gemv.go`: `mrows` and `nrows` now combine on a variant — the
  grid divisor and the slot-table builder were already independent, so the
  change is the variant rows plus a **`moeGEMVSibling(v, mrows, nrows)`**
  lookup (`moeGEMVUnblocked` is now `moeGEMVSibling(v, 1, 1)`) and a new
  **`printMoEGEMVCorner`** table that reads every corner build against the
  plain kernel *and* against each of its own axes. The M-block table now
  excludes corner builds and the row-block table prints one base row per
  distinct load width and wave, since the grid sweeps those now.

**What it found.** Full detail in IDEAS §1.11; the short version:

1. **Composition, not saturation and not competition.** `c` — the corner's
   speedup over the *product* of the speedups of its own two axes at its own
   two widths — stays between 0.73 and 1.05 at every cell. At t=256 the corner
   is 1.54x the unblocked kernel on gate_up and 2.12x on down, against 1.39x
   and 1.93x for the best single-blocked build at VEC=4 (`m8` and `n8`).
2. **`gate_up` reaches 96% of the DRAM bus at a serving batch** — 228 GB/s of
   distinct weight bytes against §1.9's 206 (87%) and the unblocked kernel's
   148 (63%). Every cell of the decode path is now at 93-100% of the bus
   except **`down` at t=256, which is at 66%** and is the one thing left.
   About a quarter of that gap is pad slots: its best build runs 829 groups of
   4 against 2560 pairs, so 756 of 3316 slots (23%) are padding.
3. **An N block makes the M block's pads 4x cheaper on `gate_up` and 1.2x on
   `down`.** Fitting `T = groups*w + slots*s` at t=4 (where every expert has
   one pair, so weight traffic is fixed): gate_up's slot goes 0.167 → 0.042 µs
   as `NROWS` goes 1 → 4, down's 0.346 → 0.288. A pad costs a per-wave part
   (its activation row, its share of the launch), which `NROWS` divides, plus a
   per-output-row part (§1.10's `subgroupAdd` and store), which it does not —
   and a down slot is 2560 output rows against gate_up's 640. §1.9's engine
   rule inherits the split: block when it removes one group per `w/s` pads,
   which is now **~45 pads on gate_up** (was 10.5) and **6.5 on down**.
4. **The row block partly substitutes for the load width.** `NROWS=4` is worth
   1.67x at VEC=1 and 1.37x at VEC=4 on gate_up at t=256, and `v1_n4` (2.059
   ms) is the fastest single-axis build there — at a width §1.7's rule calls
   the worst available (a VEC=1 lane-step covers 256 B of a 1280 B row). What
   the memory system responds to looks like bytes in flight per wave, which is
   `VEC` x `NROWS`.
5. **wave32 + N block pays in exactly one cell**: 1.065x at down's t=256
   (3.111 ms against 3.313), a wash at t=16/64, and 0.77x on gate_up — where
   §1.7's coverage rule explains it (a 32-lane VEC=4 step covers down's whole
   320 B row but only 512 B of gate_up's 1280 B one). The wave size is still
   not an independent lever.
6. **No spills.** From `RADV_DEBUG=asm`: 47 VGPRs plain, 59 at (2, 4), 83 at
   (4, 4), 107 at (8, 2), and **zero scratch operations anywhere in the
   family** — so the register-competition branch of the hypothesis did not
   happen, and (8, 4) is worth building rather than being ruled out.
7. **What it is worth.** The MoE FFN per token of the batch, best kernel of
   each family: no block 210/274/459 tok/s at 16/64/256 sequences, M block
   212/303/692, N block 221/321/746, **both 223/326/818**, the Q4 grouped GEMM
   120/185/545. So 1.50x the GEMM at t=256 where §1.9 measured 1.27x, and the
   per-projection crossover §1.9 opened closes: `down` at 5.05 rows per expert
   goes 0.77x → 0.90x → **0.98x** the GEMM, `gate_up` 1.70x → **1.88x**.
   Single-stream decode is untouched at **5.14 ms over 48 layers, 195 tok/s,
   97% of the bus floor** — the M axis is a loss at one row per expert, so the
   corner is not a latency lever and was not expected to be.

**Also worth knowing:** the family ran **twice** and the two runs agree at a
**median ratio of 1.0001** over all 706 common rows (p10-p90 within 0.6%);
every one of the 22 cells outside ±3% is either a ≤20 µs t=1 dispatch or one of
the known-noisy GEMM cells, as in previous sessions. Committed
`results/moe.csv` is the second run. Correctness: all 26 grouped-GEMV builds
(the five corner ones included) pass `verifyMoEGroupedGEMV` against the CPU
reference. GEMV rows ran at 2690-2900 MHz and 104-144 W.

**How to re-run:** `go generate ./...` then `go run ./cmd/bench moe` (the
family ignores `-sizes`/`-blocks`). Both runs took **~7 minutes** for 714 rows
with the nine new builds — against the ~11 minutes the last two sessions
reported for 564 and 624 rows, so the wall-clock figures in this file are
worth re-timing rather than trusted.

### Two sessions ago: IDEAS §1.10 — an N block for the grouped GEMV, and §1.8's last open cell closes

The handoff's item 1 was "explain the down projection's 198 GB/s against
gate_up's 235, and the probe is the `ROWS` knob `GROUPED` forbids". It is done.
The cause is a **fixed cost per output row inside the wave**, not the grid:
dividing the workgroup count alone is worth **1.02x** at the cell in question,
while dividing it together with the wave count — one subgroup taking several of
an expert's weight rows — is worth **1.17x** there and takes down's
distinct-weight rate from 196 to **230 GB/s** against gate_up's 235.

**New code:**
- `shaders/gemv_w4a8.comp`: `GROUPED` no longer `#error`s on `ROWS > 1` (the
  arm that divides the workgroup count and changes nothing else), and a new
  **`NROWS`** knob gives one subgroup `NROWS` consecutive weight rows of one
  expert against one activation row. The activations are hoisted out of the dot
  macro into `N_LOAD_X` and `W4A8_XY` does one row's eight
  `dotPacked4x8AccSatEXT` against operands already in registers — the transpose
  of `MROWS`' `W4A8_ROW` hoist. An `N_ROWS(SITE)` macro instantiates every row
  at the five places one appears, so a width is one `#define`, nothing is
  indexed dynamically, and all `NROWS` stores are issued consecutively under a
  single `subgroupElect` (which is the point of the arm: `NROWS` consecutive
  floats from one lane instead of one 4-byte store per wave). It is written out
  separately from the `MROWS` `main` rather than sharing a body, so **every
  pre-existing `.spv` is byte-identical** after `go generate` (checksum-verified
  across the whole `shaders/` tree, including this session's comment edits).
  `NROWS` requires `GROUPED`, does not combine with `MROWS`, and needs `N` to
  divide it — the `rowBase >= pc.M` guard covers a partial workgroup, not a
  partial block inside a wave.
- Six builds in `shaders/shaders.go`: VEC=4 at ROWS 2/4/8 and NROWS 2/4/8, the
  width that wins the unblocked arm at both expert shapes.
- `bench/ops_moe_gemv.go`: `rows`/`nrows` on the variant with `rowsPerWG()`
  dividing the grid's x extent; a shape check that skips a build whose block
  does not divide `N`; a new **`xreq`** figure (activation bytes *asked for* —
  every wave re-reads its token's row out of cache, so it is `N/NROWS` times
  what DRAM supplies) and `waves` in every row's detail; `verifyMoEGroupedGEMV`
  sized to `3 * rowsPerWG()` output rows for a blocked build so the second and
  third block origins are both checked; and a new `printMoEGEMVRowBlock` table
  printing both arms against the plain build with WG/PAIR, WAVES, ISSUED,
  DISTINCT and XREQ side by side.

**What it found.** Full detail in IDEAS §1.10; the short version:

1. **It is the wave, not the grid.** At t=16 (DRAM-resident, 1.15 rows per
   expert): `ROWS` 2/4/8 → 201/202/203 GB/s per distinct expert against the
   plain build's 196, `NROWS` 2/4/8 → 218/**230**/227, against gate_up's 235.
   The gap closes from 0.83 to **0.98**.
2. **The fit prices it**: `T(n) = A/n + B` from `n=1,2` gives `A` = **19.8% of
   down's time** and **-0.4% of gate_up's**, and predicts down's `n=4` cell to
   0.2%. One `subgroupAdd`, one store and one activation row, paid four times
   as often against a quarter of the loads — exactly the fixed per-output-row
   cost §1.9 finding 3 inferred from its pad slots.
3. **Lane occupancy is falsified a second time, from the other direction.**
   §1.8 tested it by halving the wave; the complementary cell was already in
   the sweep — `VEC=1` at down is 80 loads over 64 lanes, the only fully
   occupied build in the grid, and it is **1.05x slower** than VEC=4, while
   `NROWS=4` leaves the 44 idle lanes idle and is 1.17x faster. The comment in
   the shader that claimed `NROWS` fills the lanes was wrong and has been
   corrected: the load loop is still strided by the wave size.
4. **The N block beats the M block on `down` at every batch** (1.16-1.58x),
   ties it on `gate_up` within 1% at t≥16, needs **nothing from routing** (N is
   a model constant) and **never pads**. §1.9's "read the block off the
   routing" now applies to the M axis only.
5. **It is a latency lever**, which the M block was not: the single-stream MoE
   FFN goes from 5.43 ms to **5.15 ms over 48 layers, 194 tok/s, 97% of the
   5.00 ms bus floor** (§1.8 measured 91%). Against the Q4 grouped GEMM the
   `down` ratios move with it: 1.51x at t=16 (was 1.31x), 1.38x at t=64 (was
   1.16x), 0.90x at t=256 (was 0.77x).
6. **At a serving batch both blocks pay for §1.9's reason instead.** `NROWS=4`
   is worth 1.38x on gate_up at t=256 despite gate_up having no per-row
   deficit at all, and `XREQ` is the column that moves: 1447 → 499 GB/s asked
   for. Sharing one activation row across weight rows and sharing one weight
   row across tokens relieve the same MALL.
7. **The sizing rule**, which is §1.7's read along the other axis:
   `NROWS >= 8*VEC*WAVE/K` — one wave-step should cover the lanes, as one
   lane-step should cover a weight row. 4 at down's K=640, 1 at gate_up's 2560,
   which is what the grid measures. `NROWS=8` is past the register knee
   (eight accumulators, eight scale streams, an eight-float store).

**Also worth knowing:** the family was run **twice** this session and the two
runs agree at a **median ratio of 1.0000** over all 624 rows (p10-p90 within
0.6%); every cell outside ±3% is a 12-20 µs t=1 dispatch, as in previous
sessions. Committed `results/moe.csv` is the second run. GEMV rows ran at
2813-2900 MHz.

**How to re-run:** `go generate ./...` then `go run ./cmd/bench moe` (the
family ignores `-sizes`/`-blocks`). It measured **~11 minutes** for 624 rows
with the six row-block builds added.

### Three sessions ago: IDEAS §1.9 — an M block for the grouped GEMV, and decode's throughput end splits from its latency end

The handoff's item 1 was "give the grouped GEMV an M block, which is the
crossover §1.8 measured". It is done, it is worth **1.39-1.66x at 256
sequences in flight**, and the two things it found that were not in the plan
are that the block is a **loss** at a single stream (0.39x at its widest) and
that what it removes is **cache** traffic rather than bus traffic.

**New code:**
- `shaders/gemv_w4a8.comp`: an `MROWS` knob on the grouped arm. A *group* is
  `MROWS` consecutive table slots naming one expert; a workgroup loads that
  expert's weight row once per step and dots it against `MROWS` activation
  rows, each carrying its own int32 accumulator, activation-row base,
  block-sum base and output row. The weight load is hoisted out of the dot
  macro — `W4A8_ROW` takes an already-loaded `uvec4` — and every slot is
  instantiated by an `M_ROWS(SITE)` macro at each of the four places a slot
  appears, so adding a block width is one `#define` rather than four unrolled
  blocks, and nothing is indexed dynamically (the scratch-spill trap
  `W4A8_GROUP` already documents).

  Short groups are **padded, not branched**: an expert whose routed count is
  not a multiple of `MROWS` gets slots that repeat its last pair's activation
  row and write to a scratch output row past the real ones. That keeps the
  inner loop identical for every slot and makes the pad cost measurable on its
  own — which is where finding 3 below comes from.

  `MROWS=1` is textually the previous file (`#if MROWS > 1` carries a whole
  alternative `main`, `#else` the original verbatim), and **all eighteen
  existing `gemv_w4a8*.spv` are byte-identical** after `go generate`
  (`cmp`-verified), so `results/gemv.csv`, `results/gemv_cold.csv` and
  `results/shapes.csv` stay comparable row for row.
- Four builds in `shaders/shaders.go`: VEC=4 at MROWS 2/4/8, plus VEC=8 at
  MROWS=2 as the control for whether the two knobs compete for registers (they
  do not appear to: `v8_m2` tracks `v4_m2` within 2% everywhere except t=256,
  where it is 7-9% behind).
- `bench/ops_moe_gemv.go`: `buildMoEGroups` takes the block width and emits
  slots grouped by expert with the scratch-row padding; the grid's y extent
  becomes the group count; weight traffic is now counted per *group*, FLOPs
  per *pair* and output writes per *slot*, with `mrows`, `groups`, `slots` and
  `padslots` in every row's detail. New `printMoEGEMVMBlock` table, and
  `printMoEGEMVvsGEMM` gained a second table that reads both kernels per token
  of the batch. The per-pair baseline is skipped for blocked builds (it would
  measure two things at once) and above 64 tokens (2560 dispatches per
  iteration for an answer §1.8 already has).
- `moeDecodeTokens` gained **t=256**, where routing gives an expert 5.05 rows
  — the first batch in the sweep with something for a block of 4 to amortize.
  The Q4 GEMM arm runs there too, since it is the comparison.

**What it found.** Full detail in IDEAS §1.9; the short version:

1. **The block pays in proportion to what routing supplies, and the sign flips
   at ~1.7 rows per expert.** Against the same VEC unblocked: 1.21-1.66x at
   5.05 rows (t=256), 1.06-1.17x at 1.72 (t=64), a wash at 1.15, and
   **0.39-0.98x at 1.00**, where an `MROWS=8` build does eight slots of work
   for one pair's output. The best width is the one nearest the mean rows per
   expert: 8 for gate_up at t=256, 4 for down.
2. **What it removes is cache traffic, not DRAM traffic** — the finding that
   reframes §1.8's issued-GB/s column. Unblocked at 5.05 rows per expert the
   kernel asks for **746 GB/s**, 3.2x the bus, while DRAM underneath runs at
   **148**; four fifths of the requests are re-reads of an expert a previous
   pair pulled in, and the MALL serving them at its own ~750-965 GB/s is what
   sets the time. `MROWS=8` cuts issue 4.7x and DRAM rises to **206 GB/s, 87%
   of the bus**. So an expert's duplicate pairs were never free.
3. **A pad slot costs a tenth of a pair.** At t=1 and t=4 every expert has
   exactly one pair, so a blocked build reads identical weights and does `m`
   slots of work — the pad priced with weight traffic held fixed. Fitting
   `T = groups*w + slots*s` at t=4 gives **w=1.92 µs, s=0.182 µs** at gate_up
   (10.5:1) and **w=2.53 µs, s=0.390 µs** at down (6.5:1). The engine rule
   follows arithmetically: block when it removes one group per 6-10 pads it
   creates, i.e. `MROWS` ≈ mean rows per expert, erring low.
4. **The GEMM crossover moves out 4x and becomes per-projection.** §1.8's
   `down` tie at t=64 is now a 1.16x win. At t=256 it splits: gate_up stays
   **1.70x** ahead of the GEMM, `down` goes to **0.77x** — a 16-row tile
   covers 5.05 rows in one pass where an `MROWS=4` block needs 1.62 and
   `MROWS=8` pays 1792 pad slots to reach 1.06. A compile-time block is a
   coarser instrument than a tile that rounds up for free.
5. **Serving throughput.** The FFN alone, per token of the batch: **694 tok/s
   at 256 sequences** against 546 for the GEMM, 303 at 64, 212 at 16. The
   single-stream budget is unchanged — 5.43 ms over 48 layers, 184 tok/s, 92%
   of the bus floor — which is the point: this is a throughput lever with no
   latency effect.
6. **Second data point on §1.8's open cell.** down's slot costs 2.1x gate_up's
   where its output rows are 4x as many and its reduction a quarter as long —
   an ordering the dot products get backwards and a fixed per-output-row cost
   (a `subgroupAdd` and a store) gets right. Both open items now point at the
   same probe: `ROWS > 1` under `GROUPED`, giving one workgroup several
   *output* rows.

**Also worth knowing:** all **470** pre-existing rows reproduce against the
committed file at a **median ratio of 0.9998** (p10-p90 within 0.6%), the only
outlier being the 12 µs one-token `moe_gemv_v16` cell at 0.843x — so §1.8's,
§2.2's and §3.5's numbers survive the change. The GEMV rows run at 2812-2900
MHz and 96-140 W.

**How to re-run:** `go generate ./...` then
`go run ./cmd/bench -blocks 32,64,128,256,512,1024 moe`. The family measured
**~11 minutes** this session (564 rows), with the t=256 batch and the four
M-block builds included.

### Four sessions ago: IDEAS §1.8 — the grouped GEMV, and decode stops being measured with the wrong kernel

The handoff's item 1 was "a grouped GEMV for decode — now the largest gap in
the file", and item 3 was "carry §3.5's stride window back to the GEMV kernels
and re-run §3.4's decode arm with it". Both are done, in one arm, and the
second answered **the opposite of what it was set up to find**.

**New code:**
- `shaders/gemv_w4a8.comp`: a `GROUPED` mode. One extra binding holds one entry
  per routed (token, expert) pair — the pair's row in the `[E*N, ldw]` bank,
  its activation row, its output row — and the grid becomes two-dimensional,
  x the output row inside an expert's matrix and y the pair, so the pair index
  costs no division and consecutive workgroups walk one expert's rows in
  address order. Grouped mode also takes a bank row stride and per-row
  activations as pushed words, which is what lets one binary sweep the stride
  and lets a pair read its own token's vector.

  **No gather pass is needed**, and that is worth knowing rather than
  inferring: `moe_route.comp`'s gather exists because a cooperative-matrix A
  fragment reads 16 *consecutive* rows and real routing does not put an
  expert's tokens next to each other. A subgroup reads whichever row a table
  names, so the 0.90 ms gather (§3.5 finding 4) is simply absent from the
  decode path.

  `GROUPED=0` is textually the file the twelve existing binaries were built
  from — every `#if GROUPED` has an `#else` carrying the original line
  verbatim — and all twelve are **byte-identical** after `go generate`
  (`cmp`-verified), so `results/gemv.csv`, `results/gemv_cold.csv` and
  `results/shapes.csv` stay comparable row for row.
- Six builds in `shaders/shaders.go`: VEC 1/4/8/16 at wave64 and VEC 4/8 at
  wave32.
- `bench/ops_moe_gemv.go`: the `moe_gemv` arm — §3.5's routing at decode
  batches of 1/4/16/64 sequences, the grouped and one-dispatch-per-pair
  schedules, a load-width grid, a quantization-block row, a five-point stride
  sweep, a correctness check against the pre-repack Q4 arrays (so it checks
  `repackQ4ToW4A8`'s permutation as well as the kernel, at a padded stride with
  two pairs on one expert and one token read by two experts), and five summary
  tables.
- `bench/ops_moe_q4.go`, `bench/ops_moe.go`: `runMoEShapeQ4` takes a token
  list, and `RunMoE` hands it the union of the prefill and decode batches, so
  the Q4 *GEMM* is measured at the same four decode batches over the same
  routing. The head-to-head is then one routing through two kernels rather than
  two measurements.

**What it found.** Full detail in IDEAS §1.8; the short version:

1. **The GEMV is 1.30-3.26x the grouped GEMM at M=1**, which is what every
   decode row in this family used to be. What it removes is not weight bytes —
   both read each touched expert once — but activations and padding: the GEMM's
   best tile here is the smallest built and is still **7-11% useful rows**, it
   moves 17-41 GB/s of activations against the GEMV's 2-6, and its weight rate
   tops out at 128-171 GB/s where the GEMV reaches **235 GB/s, 100% of the
   bus**, on gate_up.
2. **A decode token's MoE FFN: 5.47 ms over 48 layers, 183 tok/s, 91% of its
   own bus floor**, against 9.56 ms / 105 tok/s for the GEMM. The floor is the
   1.18 GB of 4-bit weights 48 blocks × 10 experts read, 5.00 ms at 236 GB/s.
   Measured per *distinct* expert, not per pair, because a real step's 10
   experts have not been touched for 47 layers.
3. **The measurement's own trap, quantified.** One token touches 8 MB of a
   projection's experts, well inside the 32 MiB MALL, and measures 2.03 ms per
   token — **2.7x optimistic**. Even 34 MB, nominally larger than the cache,
   still reads 410 GB/s. Only past ~2x the cache does the rate settle at the
   bus, which is why the arm sweeps the batch out to 315 MB and why every row
   carries a `resident=mall|edge|dram` field.
4. **Grouping is worth 1.29-1.81x, and at decode it is not occupancy.** Every
   per-pair dispatch here launches 640 or 2560 workgroups, 8-32x the 80 waves
   §0.1 says this part needs resident, so §3.5's mechanism predicts the launch
   cost alone. Dividing the gap by the dispatches it removes gives **690-1555
   ns, median 1163**, against §4.1's **300 ns** for an empty shader with a
   barrier. The difference is the drain: a memory-bound dispatch's last waves
   are still waiting on DRAM when the barrier stops the next one starting.
5. **Do not pad an expert bank.** This is the reverse of what item 3 expected.
   Every stride in the sweep is 64 B-aligned, so the padding bytes are never
   read and the distinct traffic is identical (315 MB) in every row — a pure
   coverage experiment, unlike §2.2's, which had to buy coverage with 1.2-2.4x
   the traffic. The **unpadded** stride wins at both shapes, and on `down` it
   wins at gcd **64**, below the [128, 256] plateau §3.5 established on the
   GEMM: padding into the window costs 1.12-1.15x and padding to gcd 1024 costs
   2.4x. The plateau is a property of a *tiled* read, where a fragment holds
   32 B of a row; a kernel that walks whole matrices wants its rows contiguous,
   because at the natural stride an expert is one 845 KB run that consecutive
   workgroups continue.
6. **The crossover back to the GEMM is at ~1.2 rows per expert**, not at the
   M=16 a tile holds. This GEMV runs one grid per *pair*, so two tokens on one
   expert read its weights twice; at t=64 (1.72 pairs per expert) the down
   projection ties outright while gate_up still wins 1.82x. That is §1.5's
   question answered at the shape that has it, and the fix is an M-blocked
   GEMV rather than the GEMM.
7. **Load width is worth 3-5% here** (§3.4's prediction, at the MoE shapes):
   a 4-bit expert row is 1280 B or 320 B, gcd 256 and 64 with the 4 KB
   rotation, so even VEC=1's 256 B lane-step covers it. VEC=16 is the worst
   cell at both layers.
8. **The scale plane costs its bytes and nothing else** on this kernel:
   QBLOCK=32 against 128 adds 4.7% of the bank and 1.045-1.116x of time. It
   narrows §2.2's finding 4 by elimination — the GEMM's unexplained 1.24x on
   the same axis is not traffic.
9. **One open cell.** down reads 198 GB/s per distinct expert where gate_up
   reads 235, for identical bytes. Its rows are 4x as many and a quarter as
   long. The lane-count explanation is testable and **fails**: at VEC=4 a
   640-nibble row is 20 lane-steps for a 64-lane wave, but the wave32 arms —
   the same 20 steps over 32 lanes — measure within 1%.

**Also worth knowing:** the run was done twice. Against the committed file,
252 pre-existing rows reproduce at a **median ratio of 0.9999** with the worst
cell at 1.10x, so §2.2's and §3.5's numbers survive the refactor; run-to-run on
the new rows the median is 0.9998, with the only outliers (0.76x, 1.17x) in the
11-19 µs one-token cells. The GEMV rows run at 2812-2900 MHz and 96-140 W.

**How to re-run:** `go generate ./...` then
`go run ./cmd/bench -blocks 32,64,128,256,512,1024 moe`. The family is now
~18 minutes (the decode batches added to the Q4 GEMM arm are most of the
growth) and the GEMV arm allocates up to 1.34 GB for its own bank — the widest
stride in the sweep — after the GEMM banks are freed.

### Five sessions ago: IDEAS §2.2 — the Q4 grouped GEMM, and 4x the bytes buying 2.1x the time

The handoff's item 1 was "§2.2's Q4/int8 arm — now the largest number in the
file", promoted there by §3.5's measurement that MoE prefill sits at 93% of
the ceiling its *format* implies and that the ceiling is entirely weight
bytes. It is now built, and it delivers **2.10x a whole MoE block** — 28.13
ms to 13.37 ms, 1.35 s to 0.64 s across 48 layers — with the mechanism
changing halfway through: the phase stops being memory-bound.

**New code:**
- `shaders/gemm_wmma_q4.comp`: §2.7's winning kernel with one operand
  rerouted. A stays fp16 read from global by `coopMatLoad` with `HOIST_A`;
  B is a 4-bit `[E*N, K]` expert bank, dequantized into an LDS tile one
  K-slab at a time and read back with the same column-major fragment load the
  `_bt` variants use. Grouped, off the same tile table §3.5 built.

  **Why a separate file rather than another -D on `gemm_wmma.comp`**: a
  cooperative-matrix fragment's lane layout is not exposed.
  `GL_KHR_cooperative_matrix` defines `m[i]` and `m.length()` but not which
  (row, col) an element is, so 4-bit weights cannot be unpacked into
  registers and declared a fragment. LDS is the only place that is not DRAM
  where `coopMatLoad` has a defined layout to read. This is worth knowing
  before anyone tries the same trick for another format.

  The unpack does no integer-to-float conversion: fp16 `1024.0` is `0x6400`
  and its ulp there is exactly 1.0, so OR-ing the +8-biased nibble into the
  mantissa gives `1024+n` exactly; subtract 1032, multiply by the block
  scale, two nibbles at a time in an `f16vec2`.
- Eight builds in `shaders/shaders.go`: §3.5's five geometries unchanged (so
  the two formats' tables subtract row for row), plus one axis each for the
  scale block (`QBLOCK` 32 vs 128), the LDS row pad and double buffering.
- `bench/ops_moe_q4.go`: the Q4 arm of the `moe` family — same routing, same
  tile table, same grouped and per-expert schedules, weight bytes counted as
  packed nibbles plus fp16 scales, a host-side Q4 bank packer with padded row
  strides, a correctness check against `dequantizeQ4` put back through fp16
  (which is exactly what the kernel computes, so a mismatch is a bug and not
  a quantization artefact), and three summary tables.
- `bench/ops_moe.go`: `buildTiles` takes `(bm, bn)` instead of a variant, and
  the fp16 summary printers are guarded on `WeightFormat` so the Q4 rows do
  not walk into tables labelled fp16. `bench/bench.go`: `randomBytes`.

**What it found.** Full detail in IDEAS §2.2; the short version:

1. **2.10x the block, 2.24x the matmuls**, best-configuration against
   best-configuration at 2048 tokens. gate_up 8.77 -> 3.82 ms, down
   9.18 -> 4.31 ms. Against the geometry arm alone it is 2.32x and 2.52x;
   the difference is that fp16 still had 1.18x of stride left on `down`
   (§3.5 finding 3) and Q4 has none.
2. **The constraint moved, which is the whole reason 4x of bytes buys 2.1x
   of time.** Weight traffic 190 -> 113 GB/s of a 236 GB/s bus (80% -> 48%);
   the padded tiles' own rate 12137 -> 28104 GFLOP/s (22% -> **51% of the
   matrix cores**). It is now at half of each ceiling and pinned against
   neither — what is left is the dequant loop body itself.
3. **§3.5's finding 2 inverts, as predicted.** The 16-row tile that was
   *slower* at fp16 (84% useful rows, but padding cost FLOPs and FLOPs were
   free) is the fastest QBLOCK=32 geometry at both layers, and wins by
   executing fewer FLOPs. It is 1.02-1.05x, and it is beaten outright by (4).
4. **The scale block is worth 1.24x, between two binaries that are
   instruction-for-instruction identical** — same 252 VGPRs, same 5064-byte
   code, same 802 instructions, same VALU and VMEM counts. Nominal bytes do
   not explain it either (6.2% of the bank vs 1.6%). Unattributed; the
   candidate is scale-plane locality and the test is a blocked scale layout.
5. **The LDS row pad is the single biggest knob: 1.68x and 1.83x.** A `BK`-
   half slab row is 128 B, exactly one rotation of the 32 LDS banks, so a
   16-row fragment load hits one bank 16 times. Eight halves of pad fixes it.
   Invisible in the instruction stream — 28 bytes of code separate the two.
6. **§5.1b's coverage window loses to the traffic that buying it costs.** On
   `down`, padding the 320 B 4-bit row from its natural gcd 64 *into* the
   [128, 256] window costs 1.20x and 2.40x the traffic and measures 1.07x
   and 1.16x slower. The window prices bandwidth; a kernel that is not
   bandwidth-bound has nothing to spend it on. Read the rule as "prefer a
   stride already in the window", not "pad until you reach it".
7. **Grouping is worth *more* at Q4**: 5.2-8.1x at 10 workgroups per expert
   dispatch where fp16 measured 3.2-4.2x. Which is what §3.5's occupancy
   explanation predicts and the launch-cost one does not — the idle time an
   under-filled dispatch leaves is fixed, so it is a bigger share of a faster
   kernel.
8. **Double buffering is worth nothing here** (4.54 vs 4.73 ms), costs the
   last 4 VGPRs and a third of the occupancy. It was 1.23x on
   `gemm_wmma.comp`'s LDS path; the staged operand here is a quarter the size.
9. **A negative result worth keeping: folding the +8 bias into the scale
   multiply is wrong.** `fma(q, s, -1032*s)` would halve the dequant's
   arithmetic and **failed the correctness check by 8%** (-0.661 vs -0.717):
   `1032*s` is ~145 where the result is ~1, so rounding that addend to fp16
   puts 0.07 of error on a quantity of magnitude 1. The bias has to come off
   at nibble magnitude. The valid form carries it to the fp32 epilogue as a
   per-K-block row sum of A, which is what `gemv_w4a8.comp` does in int32.
   The arm was built, run, caught by the check, and removed; the reasoning
   is in the shader's header so it is not re-proposed.

**Also worth knowing:** the Q4 rows run at 2796-2870 MHz and 118-133 W where
the fp16 ones run at 2880 MHz and 104-106 W. More arithmetic per byte is more
power and this part pays 1-3% of clock for it, so the ratios above are very
slightly understated.

**How to re-run:** `go generate ./...` then
`go run ./cmd/bench -blocks 32,64,128,256,512,1024 moe`. The family is now
~8 minutes and allocates up to ~3.7 GB at once (the fp16 `down` bank at the
widest swept stride); the Q4 shapes are run in their own loop after the fp16
ones so the two banks are never live together.

### Six sessions ago: IDEAS §3.5 — the grouped/MoE GEMM, and three answers that were not the expected ones

The handoff's item 1 was "§3.5, the grouped/MoE GEMM", promoted to the top of
`IDEAS.md` by §3.4 on two measurements: 40 rows per expert running at 14% of
the WMMA ceiling with 62% tile padding, and 1440 of a decode token's 1861
matmuls being expert projections. It is now done, and **neither of those was
the mechanism.**

**New code:**
- `shaders/gemm_wmma.comp`: a `GROUPED` mode. One extra binding holds a tile
  table — (row in the gathered activations, row in the `[E*N, K]` expert
  bank, column in the output) — and the tile origin is read out of it
  instead of derived from `gl_WorkGroupID`, plus one push-constant word
  naming the first table entry a dispatch covers. Nothing below the origin
  changes: `RADV_DEBUG=shaderstats` prices the grouped binaries at exactly
  the VGPR counts of their non-grouped twins (252/144/168), and the
  non-grouped SPIR-V is instruction-for-instruction identical (the ids shift
  by one `#if`; `spirv-dis` with ids renumbered diffs clean, and every file
  is the same size to the byte).
- `shaders/moe_route.comp`: the gather and combine passes a grouped GEMM
  makes necessary, since a coopmat fragment reads 16 rows at one stride and
  real routing does not put an expert's tokens next to each other.
- `vk/shim.c`, `vk/engine.go`: `DispatchSequenceTimed` — a command buffer of
  N dispatches with *differing* push constants and a barrier between each.
  That is what the per-expert baseline is, and `DispatchTimed` cannot express
  it (one constant block per command buffer). `bench.TimeDispatchSequence`
  wraps it with the same clock-warmup and 500 ms budget cap.
- `vk/engine.go`: `Buffer.FillRepeating`, because a 512-expert fp16 bank is
  1.8-4.0 GB and building a host-side copy of it costs more than the
  measurement.
- `bench/ops_moe.go`: the `moe` family. Uniform-random top-10 routing over
  512 experts at 1/512/2048/8192 tokens; five tile geometries and a
  seven-point stride sweep; grouped and per-expert arms of each; the
  gather/combine passes; and five summary tables.
- `bench/families.go`: `moe` registered between `shapes` and `reduce`.

**Finding 1: grouping is worth 1.08-4.14x, and it is occupancy.** Sort every
(kernel, layer, batch) cell by how many workgroups *one expert's* dispatch
launches:

| workgroups per expert dispatch | grouped / per-expert |
|---|---|
| 10 | 3.18-4.14x |
| 20 | 1.58-3.27x |
| 30-41 | 1.12-2.54x |
| 80-119 | 1.09-1.27x |
| 209-837 | 0.75-1.14x |

80 waves is what §0.1 measured this 40-CU part needs resident to issue WMMA
at rate, and one workgroup here is one wave. A per-expert dispatch that
cannot fill the machine is slow, and the barrier after it makes the idleness
serial. §4.1's 300 ns launch is 1.5% of a 20 µs dispatch and explains none of
it — so the dispatch count, which is what promoted this item, was the wrong
lever. Two cells **invert** (0.74-0.75x, both `moe.gate_up` at 8192 tokens in
the two narrowest-BN kernels, reproducible across three runs) and are not
explained.

**Finding 2: the tile padding is real and costs nothing.** A 16-row tile
takes a 40-row group from 62% useful to 84% and is *slower* — 7146 against
7596 useful GFLOP/s on gate_up. At this shape the binding constraint is
weight bytes: the grouped kernel moves **190 GB/s of the 236 GB/s bus**, so
what a smaller tile buys in useful rows it loses in operand bytes re-read.
Against the memory-bound ceiling the fp16 bank implies (1.68 GB of weights
plus 252 MB of activations, 8.2 ms at 236 GB/s, 8.2 TFLOP/s useful) the
grouped kernel is at **93%**. §3.4's "14% of the WMMA ceiling" was the wrong
ceiling for a shape that never sees the matrix cores, and its "5.1x behind
M=2048" compared two MALL-resident dispatches.

**Finding 3, which nobody asked for: "pad every stride by 256 B" is the wrong
statement of its own rule.** These are the first non-power-of-two reduction
lengths anything in the suite has run a GEMM at (2560 and 640), and they
separate "pad by 256 B" from "land on a gcd of 256". Sweeping
`gcd(row stride, 4096)` over every power of two, at fixed tile, two kernels,
two shapes (GB/s of the 236 GB/s bus):

| gcd | 16 | 64 | **128** | **256** | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| gate_up K=2560, reg64 | 181 | 186 | **192** | 190 | 190 | 173 | 133 |
| gate_up K=2560, reg16x64 | 149 | 165 | **179** | 179 | 173 | 137 | 89 |
| down K=640, reg64 | 167 | 165 | 175 | **175** | 152 | 118 | 98 |
| down K=640, reg16x64 | 164 | 166 | 182 | **183** | 155 | 114 | 86 |

A plateau at 128-256 B with a cliff on both sides. The 640-wide down
projection is at gcd 256 **unpadded**, and the +256 B pad every other kernel
in this suite carries moves it to 512 for a measured **1.19x loss**. The
confound is broken by construction: gate_up's gcd-1024 row is the unpadded
stride and down's gcd-256 row is too, so this is not "large stride is slow".
`IDEAS.md` §5.1b's first engine rule is amended accordingly, and this also
answers §3.4's second open item (K=640 at 88-91% of the bus) on the GEMM
side.

**Finding 4: the routing passes are 5% of the block.** Gather 0.90 ms
(368 GB/s — above the DRAM bus, because it re-reads a 10 MB token buffer ten
times out of the MALL), combine 0.52 ms, against 28.1 ms of matmul.

**Finding 5: the budget.** One MoE block at 2048 tokens is **29.99 ms grouped
against 38.49 ms expert-at-a-time**, 5 dispatches against 1538; times 48
layers, **1.44 s against 1.85 s** per prompt chunk. All fp16, where the bank
alone is 5.03 GB per block and 21.3 ms of unavoidable traffic — so what is
left of MoE prefill is the *format*, and a Q4 grouped GEMM (§2.2) is now the
largest number in `IDEAS.md`.

**Not done, deliberately:** the decode rows run the grouped GEMM at M=1,
which is free on bandwidth but wastes 15 of 16 rows of every MMA tile, and
their 10-expert working set is 33 MB — inside the MALL, so they are not a
decode measurement. The honest decode kernel is a grouped *GEMV*.

**One thing this session found by accident, worth knowing before trusting a
small cell.** Adding `GROUPED` shifts the SPIR-V ids in the non-grouped
binaries without changing a single instruction (`spirv-dis` with ids
renumbered diffs clean; every file is the same size to the byte), so
`gemm_wmma` was re-run as a regression check. Median ratio 1.0005 over all
285 rows — and two cells off by 0.23x and 1.51x. Re-measuring both at 200
iterations, twice: `wmma_reg32_hka8_pada128` at N=256 is 7551/7553 against
the committed 7526 (the 0.23x was a one-off low cell in the *check* run), and
`wmma_reg64_bt` at N=512 is 14754/14745 against the committed **9790** — so
that committed cell is the wrong one, and it is wrong by 1.5x. Both are the
default 20 iterations on dispatches of tens of microseconds, which is exactly
the regime §3.4 raised the `shapes` floor to 200 for. **`results/gemm_wmma.csv`
at N=256 and N=512 should not be trusted to better than ~1.5x**; the file is
left as measured rather than partially refreshed, and the fix is to re-run
that family with `-iters 200`.

### Seven sessions ago: IDEAS §3.4 — the models' real shapes, and two answers that move

The handoff's item 1 was "`VEC=32`, when §3.4 says a target model needs it",
and item 1's own instruction was to do §3.4 first. That is now done, and it
answered item 1 in the negative and moved a second answer nobody had asked
about.

**New code:**
- `bench/modelshapes.go`: 60 weight matrices — the four models in `GOALS.md`
  plus Z-Image's Qwen3 text encoder — with, for each, the token count it runs
  at and how many times it runs per forward pass. Read out of the models' own
  configs, with the provenance in the file header: Z-Image from the local
  `models/Z-Image-Turbo/` checkout (its FFN width is stated only in the
  safetensors headers, `feed_forward.w1.weight` = [10240, 3840]), the rest
  from their Hugging Face `config.json`. It is data, not code; nothing in it
  is inferred except the token counts, which no config states and which are
  documented one by one.
- `bench/ops_shapes.go`: the `shapes` family. A decode arm (W4A8 GEMV at
  four load widths over every reduction length a model decodes at, each with
  a DRAM-resident twin so a small matrix is not measured out of the MALL), a
  prefill arm (five WMMA variants over the 38 deduplicated rectangles, M and
  N padded up to the tile and the padding charged to the reported rate), and
  a three-part summary: the load-width grid against §1.7's rule, the winner
  per rectangle, and what one forward pass of each model costs.
- Two small refactors so this family reuses rather than copies: `timeColdGEMV`
  split out of `runGEMVColdCase` (`bench/ops_gemv_cold.go`), and
  `runWMMACase`/`buildWMMAPipelineWith` split out of `timeGEMMWMMA`
  (`bench/ops_gemm_wmma.go`). Both leave their original callers producing
  identical rows.
- `bench/families.go`: `shapes` registered between `gemm_wmma` and `reduce`.

**Finding 1: `VEC=32` is retired, and the load-width rule is nearly free at
real reduction lengths.** The models decode over K = 640, 1024, 2560 and
6144. None is a power of two, so each 4-bit weight row (320, 512, 1280,
3072 B) has a *smaller* gcd with the 4 KB channel rotation than the square
sweep's rows did — and §1.7's threshold is met by narrower loads, often the
narrowest. DRAM-resident, against the 236 GB/s bus:

| K | rowB | gcd | vec1 | vec4 | vec8 | vec16 |
|---|---|---|---|---|---|---|
| 640 | 320 | 64 | 209 | **215** | 213 | 196 |
| 1024 | 512 | 512 | 211 | **239** | 237 | 222 |
| 2560 | 1280 | 256 | 239 | 241 | 241 | **242** |
| 6144 | 3072 | 1024 | 236 | **239** | 238 | 239 |

K=2560 is most of the text model's decode traffic and it reads 101-102% of
the bus at every width. Only K=1024 gains from a wider load (1.13x). The
largest reduction length any model decodes over is 6144, so there is no model
for `VEC=32` to serve.

**Finding 2, which nobody asked for: prefill has two winners and the split is
M.** §2.7's `wmma_reg64_bt_hka4_padab128` wins all 16 rectangles with
M ≥ 1024 and loses all 22 below it — by up to 2.8x — to
`wmma_reg32_bt_hkab4_w32_padab128`, §6.2's AI-16 wave32 arm, which the square
sweep only ever crowned at N=1024. The attribution is in the same data: at
equal tile, rung and pads, wave32 is 1.2-2.3x its wave64 twin at *every*
shape (§6.2's effect), and at equal wave size the AI-16 tile beats AI-32 by
1.43x at M=128. So it is the small tile for occupancy plus wave32 for the
fragment registers. The square sweep cannot see this because it ties M to N
and K.

**Finding 3: the padding hazard is M, not K.** Every K in all five models is
a multiple of 64 — the predicted "K not a multiple of TILE_K" problem does not
occur once. Every N is too, except Parakeet's 8198-wide joint output, which
pads to 8224 for 0.3% waste. The shape that hurts is the MoE expert at
prefill: 2048 tokens over 512 experts is **40 rows each**, 62% useful in a
64-row tile, and the best kernel returns **7600 useful GFLOP/s — 14% of the
WMMA ceiling and 5.1x behind the same matrix at M=2048**.

**Finding 4: the budget.** Adding the shapes up, a qwen3.8-flash-next token
reads **2.89 GB of 4-bit weights** over 1861 matmuls; at the best measured
DRAM-resident rate (242 GB/s) that is **11.9 ms = 84 tok/s** before
attention, with 0.56 ms (5%) of dispatch overhead on top at §4.1's 300 ns.
Its *prefill* reads 61.8 GB for 2048 tokens — the whole 512-expert bank — and
is **98% memory-bound by weight bytes**, because the batch reaches each
expert divided by 512. Every other model in `GOALS.md` is compute-bound at
its natural batch.

**On iteration count.** This family uses a 200-iteration floor rather than the
suite's 20. Its dispatches are tens of microseconds, so 20 of them is under a
millisecond of timed work, and at that length a few cells per run came back
2x low at a pinned 2814 MHz. Two runs at 20 iterations disagree by up to 105%
on their worst cell; two at 200 disagree by at most 15%, median 0.2% either
way. `TimeDispatch` still caps the batch to its 500 ms budget, so the long
cases are unaffected.

**Not done, deliberately:** attention (§3.3) is still absent, so the 84 tok/s
above is a weights-only floor; the KV cache at long context is not in it. And
the decode arm measures W4A8 only — the shapes family is not the place to
re-run a format ablation.

### Eight sessions ago: IDEAS §1.7 — the wave32 W4A8 win, explained and superseded

The handoff's item 1 was "explain the wave32 W4A8 win, and see how far it
goes": §6.2 had measured the DRAM-resident W4A8 decode GEMV at 225.7 of
236 GB/s at wave32 against 211 at wave64, reproducible to under 1%, specific
to the `VEC=4` arm, and unexplained — §5.1b's coverage law predicted the
wrong sign. It named three probes. Two were run and answered it; the third
was retired by the answer.

**The answer: it was never the wave size.** The quantity that predicts every
cell is **C, the contiguous run of one weight row that a lane-step holds** =
`min(WAVE, N/(8·VEC)) · VEC · 4` bytes. Hold C fixed and vary the wave size
and the bandwidth is identical — at N=8192, `vec8` at wave64 and `vec16` at
wave32 both hold 2048 B and measure **198.83 and 198.93 GB/s**. The 1.07x is a
*deficit* in one cell, N=4096 at `VEC=4`, where a wave64 step holds exactly
half a row so every wave in the grid addresses the same half of the 4 KB
channel rotation at once; it does not occur at N=2048 or N=8192, and two
unrelated changes each erase it (`ROWS=2`, and `VEC=8`).

**And the rule that replaces it finished the decode path.** §5.1b's
`coverage = min(1, C/gcd(stride, 4096))` with the weight row as the stride
says the bus is reached exactly when a lane-step covers a whole row, i.e.
**`VEC = N/(8·WAVE)`**. That threshold is hit on the nose at all three
reduction lengths swept:

| N | best kernel | GB/s | % of the 236 GB/s bus | previous best |
|---|---|---|---|---|
| 2048 | `vec4` wave64 | 243.0 | 103% | — (never measured) |
| 4096 | `vec16` wave32 / `vec8` wave64 | 238.9 / 234.8 | 101% | 225.6 |
| 8192 | `vec16` wave64 | 234.6 | 99% | 164.3 |

The N=8192 row is the one that matters: it is the shape a real model decodes
at, the previous kernel got 70% of the bus there, and this is **1.43x**. The
cache-resident `gemv` sweep gained too, which was not predicted: its best
W4A8 row at N=4096 goes from 563.5 GB/s to **734.0**, 1.30x. The
law is exact about the cliff edge and far too pessimistic below it (it
predicts 3% of peak where the measurement is 57%), which is expected — it was
derived for one wave's requests, and here thousands of waves on adjacent rows
cover the rotation between them.

**New code:**
- `shaders/gemv_w4a8.comp`: `VEC` now accepts 8 and 16 as well as 1 and 4,
  and the wide path is a `W4A8_GROUP(G)` macro instantiated once per `uvec4`
  of weights. The macro is not a loop on purpose: a `for (g < VEC/4)` loop
  survives `glslc -O` rolled, and a rolled group loop indexes its operands
  dynamically, which puts them in scratch — the one thing a load-width arm
  must not do. New `ROWS` `-D` puts several subgroups (several output rows) in
  one workgroup via `gl_SubgroupID`, with `LANE`/`LANES` macros selecting the
  subgroup-relative or workgroup-relative builtins; they are macros rather
  than locals so the `ROWS=1` expansion is unchanged.
- `shaders/shaders.go`: eight new binaries — `_v8`, `_v16`, `_v4_r2`,
  `_v8_r2` and their `_w32` pairs.
- `bench/ops_w4a8.go`: the variant table is now a 2x2x2 over (wave size) x
  (load width) x (rows per workgroup) plus the two `VEC=16` arms, with
  `w4a8Groups(M, rowsPerWG)` for the grid width.
- `bench/ops_gemv_cold.go`: `coldCase.rowsPerWG`, threaded to the dispatch.

**Caveat on comparability:** the `VEC=1` and `VEC=4` SPIR-V changed when the
wide path became a macro (the `xs[]` staging array is gone; RADV's ISA is
equivalent — same `buffer_load_b128` and `v_dot4` counts, 48 VGPRs, no
spills). `results/gemv.csv` and `results/gemv_cold.csv` were both regenerated,
so every row in them is from the current source. The earlier steps of this
session did hold byte-identity (`cmp`-verified) while `ROWS` was added; it was
given up deliberately when `VEC=16` needed the macro.

**One residue.** At N=2048 the wave32 `vec8` arm satisfies both clauses of the
rule — whole row per step, every lane issuing — and still reads 219.2 against
wave64 `vec4`'s 243.0. At equal C and equal lane occupancy, 64 outstanding
requests per wave beat 32 by 10%. Not chased further; the engine rule (use
wave64, pick `VEC = N/512`) is on the right side of it either way.

**Not run, and why:** the third probe §6.2 asked for was the `stride` family
at both wave sizes, ~14 minutes each. It existed to isolate a wave-size effect
from everything else a GEMV does. There is no longer a wave-size effect to
isolate, so it was retired rather than run.

### Nine sessions ago: IDEAS §6.2 — wave32, and the payoff landed somewhere else

The handoff's item 1 was §6.2, promoted by §2.7 on a specific argument: the
best lever in the file stops at the 256-VGPR wave64 budget, and wave32 was
supposed to halve what a coopmat fragment costs, which is the currency that
lever spends. Wave size is now a per-pipeline knob, every kernel whose
workgroup *is* one subgroup can be built for either, and the answer came out
in three pieces — none of which is the one the promotion predicted.

**New code:**
- `vk/shim.c`/`shim.h`: `VK_EXT_subgroup_size_control` is queried
  (`shim_query_subgroup_size_control` → min/max size, `computeFullSubgroups`,
  and whether COMPUTE accepts a required size), enabled at device creation
  when asked for, and `shim_create_compute_pipeline` takes a
  `requiredSubgroupSize` — passed as
  `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` alongside
  `REQUIRE_FULL_SUBGROUPS`, so a shader whose `local_size_x` is not a multiple
  of the size asked for fails at pipeline creation rather than silently
  running a partly-inactive wave through `subgroupAdd`. Zero keeps the
  driver's default and builds byte-identically to before.
- `vk/engine.go`: `DeviceFeatures.SubgroupSizeControl`,
  `PhysicalDevice.SubgroupSizeControl()`, `PipelineSpec.RequiredSubgroupSize`.
- `shaders/*.comp`: `WAVE` is a `-D` on `gemm_wmma.comp`,
  `gemv_subgroup.comp`, `gemv_w4a8.comp`, `gemv_w8a8.comp`,
  `rmsnorm_subgroup.comp` and `softmax_subgroup.comp`. The wave64 SPIR-V is
  byte-identical to what it was before (`cmp`-verified), so every pre-existing
  row is comparable.
- `shaders/shaders.go`: 16 new `_w32` binaries — seven WMMA tiles, the four
  `gemv_subgroup` precisions, both W4A8 load widths, W8A8, and both
  reductions.
- `bench/wavesize.go`: `filterWaveVariants` drops any variant naming a size
  the device won't allow, with a note on stderr, so the wave64 half of the
  suite is still a complete run on an older driver. Used by all five families.
- `bench/ops_*.go`: a `waveSize` field on the `wmma`/`gemv`/`w4a8`/`w8a8`/
  `reduce`/`coldCase` variant tables, threaded to the pipeline. `ops_w8a8.go`
  and `ops_reduce.go` became table-driven to take it.
- `cmd/probe/main.go`: optional second argument pins the wave size, which is
  the only way to price a variant's registers at wave32.
- `cmd/bench/main.go`: prints the device's subgroup-size range next to the
  feature line.

**What it found** (full detail in IDEAS §6.2):

1. **The GEMM: 0.67x-2.13x, and the suite's headline does not move.** The
   fragment-dominated AI-16 grid gains **1.9-2.1x at every size**
   (`reg32_bt_hkab4_padab128`: 11417 → 24313 GFLOP/s at N=4096), and wave32
   takes the N=1024 crown at 33603. But §2.7's winner `reg64_bt_hka4_padab128`
   **spills 62 VGPRs at wave32** and loses 18-25%, so N=2048 and N=4096 stay
   with wave64 at 39039 and 31183. A second run agrees to ≤1.5% everywhere.
2. **Why, and it is half of what was predicted.** The calibration is the three
   *non-coopmat* kernels, whose register use per unit work cannot change with
   the wave: they report exactly 48 → 96 VGPRs, so 2x is the no-change
   baseline. Every coopmat variant comes in under it, and the most
   fragment-heavy (`reg32_bt_hka8`, eighteen fragments against four
   accumulators) reports the *identical* 192 at both sizes — half the per-work
   cost. So the fragment saving is real. The accumulators do not shrink and the
   256 ceiling is in the same halving units, so the roof does not lift.
3. **The one engine-relevant win is on decode, not prefill.** DRAM-resident
   W4A8 GEMV at `VEC=4`, 68 MB of weights, three runs:
   **209.8 / 210.2 / 212.8 GB/s at wave64 → 225.6 / 226.3 / 225.1 at wave32**,
   i.e. 89% → **96% of the 236 GB/s bus**, under 1% spread. It is specific to
   the wide-load arm (at `VEC=1` wave32 is 0.98x) and it is **unexplained** —
   §5.1b's coverage law predicts the wrong sign. See "Next steps" item 1.
4. **Every cache-resident GEMV loses**, 0.58-0.95x, and so do the reductions.
   Which closed §3.7 for free: `shared` 223.7 GB/s at 256 threads per row,
   `subgroup` 68.5 at 64, `subgroup_w32` 35.7 at 32 — a line through the
   origin in threads per row, so the 3.3x gap is the lane count and has
   nothing to do with `subgroupAdd`.

**One caveat on the committed CSV.** The cache-resident `gemv` square sweep
carries the one-sided dropout noise the README documents for MALL-resident
cases, and one wave32 cell in this run is a dropout:
`gemv,subgroup_w32,w4a8,block=64,size=1024` reads 61.5 GB/s and reran at
270-281. A second run put a different cell (`subgroup_vec4_w32` at N=2048)
low instead. Read the pattern across sizes, not any single cell. The
DRAM-resident `gemv_cold` rows reproduce to under 1% and are the ones to
trust.

### Ten sessions ago: IDEAS §2.7 — hoisting the K-slab, and §2.3's residue closed

The handoff's item 1 was "§2.3's residue, re-aimed by the traversal axis":
the 1.6x that transposed-B stayed behind row-major at N=4096 **with both
strides padded**, which §2.3 ruled out three explanations for and could not
attribute. §5.1b's mechanism 3 had supplied the first candidate — a coopmat
fragment load holds 32 B of a row, a tiled GEMM's waves retire after one, so
the contiguous run in flight is an eighth of the `gcd` even a padded stride
leaves — and said the test needed no new probe, just more of a row in flight
inside the WMMA kernel. That is now done, and it is the largest single win
the GEMM has had since register blocking.

**New code:**
- `shaders/gemm_wmma.comp` gains a fifth axis, `HOIST_A` / `HOIST_B`: one
  operand's fragment loads for the *whole* K-slab are issued before the
  slab's first MMA, instead of one K-tile's worth at a time. The byte set,
  the MMA count, the tile, the accumulator grid and the arithmetic intensity
  are all unchanged; only how much of each row is outstanding at one moment
  moves. It is per operand because the register file, not the idea, sets how
  far it goes. The non-hoisted path is left byte-identical — the existing
  twelve variants recompile to the same SPIR-V, verified with `cmp`.
- `shaders/shaders.go` gains eleven variants: a transposed-B `reg32`
  baseline, the rung-2/4/8 ladder on both grids, and the row-major controls. Each
  generate line's VGPR cost is recorded in the comment above them.
- `bench/ops_gemm_wmma.go` gains 22 rows — the ladder unpadded and with
  §2.3's +256 B pad, plus the row-major arm — bringing the WMMA ablation to
  48 variants.

**The finding: the residue was the traversal, and it reverses.** New suite
best **38990 GFLOP/s at N=2048 — 70% of the measured 55.5 TFLOP/s WMMA
ceiling**, up from 28510 and 51%. At N=4096 the transposed-B kernel goes from
15697 to **30609**, past row-major's 24466: the kernel with the better
instruction stream is now the faster one at every size, which is what §2.1
predicted before the memory system got in the way. Both runs of the sweep
agree to ≤1% on those rows.

**The control was already in the suite, and it had been misread.**
`wmma_reg64_bt_k64` is BK_TILES=4 without hoisting — the same 64
`buffer_load_b128` and 64 `v_wmma` per loop body as the hoisted kernel, the
same bytes. `RADV_DEBUG=asm` via `cmd/probe` counts what the scheduler
actually leaves outstanding before the first `s_waitcnt`: **16 loads for the
unhoisted kernel, 45 for the hoisted one**, and 8734 vs 18363 GFLOP/s
unpadded at N=4096. So "deepening the K-slab does nothing", which this file
recorded as a refutation twice, was the control measuring its own no-op.
Depth is not the variable; concurrency is.

**Two things make it attributable rather than merely true.** The lever is
worth 2.88x on the transposed-B layout (both operands K-contiguous gathers)
and 1.14x on row-major (whose B fragments are gathered along N and cannot
deepen) — it acts on exactly the loads the mechanism names. And the *stride
pad's* value decays as the kernel holds more of a row: 1.44x at C=32 B,
1.24x at 64 B, 1.09-1.10x at 256 B, which is the coverage law's own
prediction observed from outside the probe that produced it. What this does
**not** separate is channel coverage from plain latency hiding; more loads in
flight buys both, and the experiment has no lever that buys only one.

**Where it stops: registers.** On wave64 an accumulator is ~8 VGPRs and a
fragment ~4, so WM=WN=4's sixteen accumulators take 128 of 256. Hoisting both
operands at BK_TILES=4 spills (140 VGPRs, 3 KB scratch), at BK_TILES=8
catastrophically (380-488, 70 KB). The AI-32 ladder therefore stops at rung 4
and rung 8 needs the four-accumulator AI-16 grid. **This promotes §6.2
(wave32)** from a curiosity to the next item: wave32 halves the per-fragment
register cost of the same tile, which is the exact currency this lever
spends, and WMMA is natively a wave32 shape.

**One cell is unexplained and reproducible:** `reg32_bt_hkab4` is *slower*
padded than unpadded (0.88x at N=4096, 0.87x at N=2048, both runs). Every
other rung on that ladder goes the other way.

### Eleven sessions ago: IDEAS §5.1b's follow-up — the traversal axis

§5.1b left one access pattern unmeasured and called it the sharpest item in
the file: **concurrent-but-contiguous requests from *different* waves at an
aliased stride.** Every shape in the `stride` family varied the addresses
inside one request, and its contiguous shape deliberately walked *along* a
row, so consecutive waves were never a stride apart — which is exactly the
GEMV/W4A8 pattern. It is now measured, and it did not answer the question it
was asked; it replaced the model the whole item was built on.

**New code:**
- `shaders/strided_read.comp` gains two compile-time axes and keeps
  everything else fixed. `CROSS_WAVE` swaps the two loop orders over the
  (row-group, column-block) grid so consecutive requests advance *down* the
  rows: a permutation of the identical (row, chunk) set, so the byte set, the
  footprint, the instruction count and the host checksum are all unchanged
  and only which addresses are outstanding together moves. `LOADS_PER_WAVE`
  makes one request issue *n* loads along its own row before retiring, which
  is what a real kernel's inner loop does; at the top of that ladder one wave
  streams a whole row and consecutive waves take consecutive rows, which is
  the GEMV kernels' shape exactly.
- `shaders/shaders.go` gains eight variants: the five request shapes in the
  cross-wave traversal, plus the 2/4/8-loads-per-wave rungs of the ladder for
  the contiguous shape.
- `bench/ops_stride.go`: `strideShape` carries `crossWave` and
  `loadsPerWave`, cases whose rows are too short for a rung are skipped per
  (row length, footprint), each shape sizes its own grid, and
  `PrintStrideSummary` prints a second grid — the cross-wave ratio per shape
  per stride, each cell with the model's prediction beside it.
  `strideInFlightBytes` is the new model, and the old `strideChannelCoverage`
  is unchanged: it turned out to be the same function of a different
  argument.
- `bench/ops_stride.go` also discards one case per (row length, footprint)
  group before timing anything in it — see the 782 GB/s note below, which is
  what that is for.

**The finding: §5.1b's coverage law is a law about concurrency.** The
achievable fraction of peak bandwidth is

> **coverage = min(1, C / gcd(stride, 4096))**, where **C** is the contiguous
> run of bytes, inside one row, that the requests outstanding at one moment
> hold.

Walking along a row, consecutive requests continue where the last stopped, so
C is the whole row and the law reads as §5.1b stated it. Cross-wave, C is
only what one request holds. Same bytes, same instructions, 8192 B rows,
64 MiB touched, contiguous 1 KB requests: the walk reads **238-244 GB/s at
every stride in the sweep**, the cross-wave traversal reads **63** at 8192,
**124** at 10240 and **229** at 8448 — model 0.25 / 0.50 / 1.00, measured
0.26 / 0.51 / 0.95. It predicts every contiguous-shape cell to within 2% at
the aliased strides and within 6% elsewhere.

**The ladder proves the mechanism and clears the GEMV kernels.** At stride
8192, giving each wave 1 / 2 / 4 / 8 KB of its row in flight measures
63 / 122 / 239 / 238 GB/s against a model of 0.25 / 0.50 / 1.00 / 1.00, and
the 2048 B-row group says it again one rung shorter (120 GB/s with 1 KB in
flight, 237 with the whole row, at a *dense* 2048 B stride). C is
a property of the *wave*, not of the instruction, and 4 KB — one interleave
rotation — is immune at every stride. The top rung *is* one wave per row with
consecutive waves on consecutive rows, and it runs at the full bus at every
stride: **decode was never exposed**, consistent with W4A8's 89%, and the
1.12x that bounded this experiment for decode is not there to be had.

**It costs §5.1b a corollary.** "A densely packed tensor is never at risk"
was derived with C = rowBytes and is false for any kernel whose waves do not
walk along a row. With **no padding anywhere**, 64 MiB touched: 1024 B rows
at a 1024 B stride read 243 GB/s with 1024 B in flight per row and **61**
with 256 B; 2048 B rows at a 2048 B stride read 243, **120** and **31** with
2048, 1024 and 256 B in flight. An eighth of
the bus, on a tensor with nothing wrong with it.

**Two smaller results.**
- The shapes where both traversals compile to the same mapping (one request
  already covers a whole row) are printed with `=` and are a free
  repeatability check. At 64 MiB they are 1.00x in all twelve cells. At
  16 MiB the same pair measured 782 and 942 in the last run before the fix —
  which **retires the 782 GB/s MALL oddity** §5.1b could not explain, and
  better than as scatter: 782 was the *first timed case of each 16 MiB
  group*, in all three row-length groups to three digits, and only in the
  cache-resident ones. The host fills the buffer immediately before, so that
  case measures the fill's residue — writeback stealing DRAM bandwidth is
  the likeliest candidate, and the clock rules itself out at 1% between the
  twins. **Discard the first cache-resident case after a host fill**, which
  `RunStride` now does per group; those cells read 884-950 in the committed
  run, inside the ~10% scatter every cache-resident case carries, and the
  identical twins are 2.3% apart.
- **The MALL is not immune to this one.** §5.1b found it indifferent to
  request *shape*; it is not indifferent to traversal. 2048 B rows at a dense
  2048 B stride, 16 MiB touched: **484** GB/s cross-wave against the 868-942
  the walk shapes read (model 0.50), and **128** at 256 B in flight (model
  0.125), with the whole-row rung back at 928. The same law scaled to the
  MALL's peak — though not every MALL cell fits it, so that item's
  slice-structure question stays open.

**What the model does not cover**, stated because the grid prints it: for
shapes whose single request already spans many rows it misses in both
directions. Where it predicts a severe penalty they beat it by 1.4-8x (the
64-address gather at stride 8192 measures 0.03x against a model of 0.004),
because their rows start at every multiple of `gcd` and neighbouring requests
partly refill the lines it counts as untouched; where it predicts none they
fall short of it, to 0.53-0.93x, because the per-request penalty it omits is
far larger cross-wave than the 1.32x a walking traversal shows. It is exact
where C is a genuine contiguous run, which is what real kernels issue.

**Engine consequence: rule 1 got much more valuable and did not change.**
Padding every row stride to 256 B past a multiple of 4 KB makes
`gcd(stride, 4096) = 256`, which is at or below what *any* request shape here
holds in one row — so coverage is full however a kernel traverses the tensor.
It was filed as a 1.3x tidy-up against gathers; it is a defence against a 4x
that needs no padding to occur. The second escape, when the stride is not the
engine's to choose, is depth: 4 KB of a row in flight per wave.

### Twelve sessions ago: IDEAS §5.1b — the strided-bandwidth probe

`IDEAS.md` §5.1b asked for the memory system to be measured directly instead
of inferred through a GEMM: fixed bytes touched, sweeping (a) the stride
against the channel-interleave period and (b) request *shape* at constant
stride. **Both are answered, and there turned out to be two mechanisms
rather than one — the larger of which nothing in the suite was looking for,
because it needs no gather and no kernel at all to trigger.**

**New code:**
- `shaders/strided_read.comp` — reads `rows` rows of which the first
  `rowBytes` are touched, spaced `stride` bytes apart, one `uvec4`
  (`buffer_load_b128`) per lane. Every touched chunk is read exactly once in
  every variant, so the byte set, the footprint and the load-instruction
  count are identical across the whole sweep and only two things move: the
  stride (a push constant) and the lane→address mapping. `LANES_PER_ROW`
  sets the latter — how many consecutive 16-byte chunks of one row go to
  consecutive lanes — so one 64-lane request spans `64/LANES_PER_ROW`
  distinct rows: 1 (a contiguous 1 KB sweep) through 64 (a 64-address,
  16-bytes-each gather), with 32 being the shape a 16×16 fp16 `coopMatLoad`
  issues. Five variants, one `-D` apart.
- `bench/ops_stride.go` — the `stride` family: three sweeps (row length ×
  touched footprint × stride), `PrintStrideSummary`'s pivot grid, and
  `strideChannelCoverage`, the closed-form model below, printed as a `model`
  row so each grid checks itself against it.
- `bench/ops_stride.go` verifies every case against a host checksum over its
  (row, chunk) set, computed from the geometry and deliberately *not* from
  the shader's lane mapping — so a mapping that read a chunk twice or skipped
  one fails instead of reporting a plausible bandwidth. It caught nothing,
  which is the point: without it a bandwidth number is unfalsifiable.
- `vk/engine.go` gains `Buffer.ReadUint32` and `Buffer.MappedPointer` (the
  latter to fill a 128 MiB buffer in place rather than via a host-side copy).
- `cmd/bench/main.go` gains `-stridepads`, `-striderowbytes`,
  `-stridefootprints` and the `stride` skip name.

**Finding 1 — channel coverage, worth 2-4x, and it hits contiguous reads.**
The interleave is 256 B across **sixteen** channels, a 4 KB rotation. Row *k*
starts at `k*stride`, so row starts land only on multiples of
`g = gcd(stride, 4096)`, and each row covers `rowBytes` from its start. When
`rowBytes < g` the union **never addresses some channels**, and the
achievable fraction of peak is exactly the fraction of channels touched:

> **coverage = min(1, rowBytes / gcd(stride, 4096))**

which tracked the measurement to within 2% at every point of the sweep. At
1024 B rows and 64 MiB touched, strides of 2048 and 4096 B measure 0.50x and
0.25x of peak for *every* shape including the fully contiguous one, while
1280 and 3072 B are at 1.00x. Nothing in §2.3 could see this: a K=4096 fp16
row is 8192 B, already ≥ any `g`, so its coverage is full at every stride.

**Finding 2 — per-request aliasing, worth up to 1.34x, and it needs a
gather.** With coverage full (8192 B rows), a 4 KB-multiple stride costs a
contiguous read *nothing* (243 GB/s at every stride in the sweep) and costs
more the more rows one request spans: the unpadded-to-best ratio is 1.01x for
1 row per request, 1.06x for 4, 1.17x for 16 and 32, and **1.32x for 64**.
This is §2.3's effect, isolated from tiling.

**The period is 4 KB, not the 2 KB §2.3 inferred.** With nine strides inside
one rotation: 10240 and 14336 B *are* multiples of 2 KB and run at the full
rate, while 8192, 12288 and 16384 — the multiples of 4 KB — are the only slow
strides. §2.3 had three strides and no way to tell the two periods apart.

**§2.3's sweep shape is now explained as two terms.** Its GEMM peaked at
+256 B and declined after, which read as an optimum inside a 2 KB period. But
B is `N*(K+padB)*2` bytes, so at N=K=4096 the unpadded case is *exactly* the
32 MiB MALL and every pad step pushes it out: 8192 B → 32 MiB, 8448 → 33,
9216 → 36, 10240 → 40, 12288 → 48 MiB. At constant bytes touched the probe
shows no decline at all — a flat plateau from +128 B across a whole period.
So that curve is a de-aliasing step (complete by +128-256 B) times a monotone
footprint cost. Same engine rule, different reason, and the rule is now
"+256 B past a multiple of 4 KB **and no more**".

**Two corollaries decide where the padding rule earns its 3%.** A densely
packed tensor is never at risk: `stride == rowBytes` forces
`gcd(stride, 4096) ≤ rowBytes`, so coverage is 1 by construction and this
mechanism is *created* by padding — only by badly chosen padding. But any
kernel reading a *strip* of a larger tensor is exposed, and that is what
tiling is: a blocked GEMM sweeping a 1 KB-wide panel of a K=4096 fp16 matrix
has `rowBytes` 1024 against a stride of 8192, `gcd` 4096, coverage **0.25**.
Padding that matrix to an 8448 B stride restores coverage to 1 for every
panel width. So the padding rule is a *prerequisite* for §2.4 and for the
MALL-blocking half of §5.1b, not a 3% optimisation next to them.

**Request shape is free, which retires §2.3's open question.** At a
de-aliased stride a 64-address gather reads at 239 GB/s against a contiguous
sweep's 243 — within 2%, over identical bytes. So the [N,K] layout a real
`Linear` weight already has is usable above MALL size with no transpose, and
the residue §2.3 could not explain (transposed-B still 1.6x behind at N=4096
*with both strides padded*) is **not** a memory-request-shape effect. §2.3
had already ruled out request count and cache-line survival; the remaining
place to look is the kernel.

**The MALL behaves differently from DRAM on both counts.** At 16 MiB touched
it delivers 930-965 GB/s for a pure read — above the 805 GB/s §0.4 measured
with a read+write copy — and it is immune to the gather penalty (the 64-row
gather is ~0.93x at *every* stride, aliased or not). The coverage law does
*not* carry over: a fully-covering working set stays resident however far it
is spread (16 MiB scattered over a 144 MiB span still gets MALL bandwidth),
while under partial coverage it stays resident only while the span is also
inside 32 MiB — 2048 B rows at a 4096 B stride show *no* penalty at a 32 MiB
span, the same rows at 8192 B (64 MiB span) fall to DRAM × 0.5, and 1024 B
rows at 4096 B fall to DRAM × 0.25. §5.1b has the table. The engine
consequence points the same way as the corollaries above: **"keep it under
32 MiB for 3.4x" only holds for a fully-covering access pattern.**

**Two measurement lessons, both now in the code:**
- **A 340 µs batch is not a measurement of a cache-resident kernel.** 16 MiB
  at ~950 GB/s is a 17 µs dispatch, and the suite's default 20 iterations
  average that over 340 µs. Each `stride` case now sizes its own iteration
  count from its probe dispatch to fill 50 ms (`strideTargetBatchNs`), which
  `TimeDispatch` still caps at `maxBatchNs`.
- **Run one benchmark at a time.** An orphaned `exe/bench` from a killed run
  competed for the GPU through several sweeps here and put ~10% of the
  MALL-resident cases at 0.3-0.5x, on a different cell each run, with sclk
  and fclk both pinned — which looked exactly like a memory-system finding
  and was not. Note `pkill -f cmd/bench` does not do it: `go run` execs a
  binary under `$GOCACHE` whose argv no longer contains that string, and a
  pattern that matches your own shell command line kills the shell first.
  Cache-resident cases keep ~10% one-sided scatter even with sole use of the
  GPU (the display shares this iGPU and its MALL), so each case runs three
  timed batches and keeps the fastest (`strideBatches`); DRAM-resident cases
  reproduce to within 2% either way and are unaffected by that choice.

### Thirteen sessions ago: IDEAS §2.3 — the transposed-B contradiction, resolved

`IDEAS.md` §2.3's open question was why the *better* instruction stream is
2.8x slower: storing B as [N,K] and loading it column-major cuts
`wmma_reg64`'s loop from 577 instructions to 389, and loses badly at
N=4096. Its one untested mechanism was channel/bank aliasing from the
power-of-two leading dimension. **It is that, it is periodic in the stride
with a 2 KB period, and it was costing the kernels that already won — so
fixing it raised the suite's best GEMM from 25.2 to 28.4 TFLOP/s, 45% → 51%
of this chip's measured WMMA ceiling, with no change to any kernel's
structure.**

**New code** (small — this session is a measurement, not a kernel):
- `shaders/gemm_wmma.comp` takes both leading dimensions as push constants
  (`pc.ldb`, `pc.lda`) instead of reusing `pc.K`/`pc.N` as extent *and*
  stride. So a padded case reuses its unpadded case's SPIR-V and differs in
  exactly one push-constant word — which is what makes the pair a clean
  test. Verified the codegen didn't regress: `reg64_bt` is still 389
  instructions with 16 `buffer_load_b128` and no `buffer_load_d16_b16`.
- `bench/ops_gemm_wmma.go` gains `padA`/`padB` on `wmmaVariant` (in halves,
  and they must be multiples of 8 or the rows stop being 16-byte aligned and
  `coopMatLoad` falls back to scalar loads, which would measure alignment
  instead of aliasing), `padRows`, and fourteen new cases: a seven-point
  stride sweep on `reg64_bt`, three controls, and four pad-A cases — the
  three §2.1 winners plus a both-operands-padded `reg64_bt`.
  `Detail` now carries `strideA`/`strideB` in bytes. The push-constant size
  is a named constant checked against the built blob, because getting it
  wrong (20 bytes declared against a 24-byte block, which happened here)
  leaves the last field reading out of range and RADV answers that with
  plausible numbers rather than an error.
- `cmd/probe/main.go` pushes 64 bytes of push constants instead of 16, since
  a layout range smaller than the block the shader declares is invalid.

**The stride sweep** (`wmma_reg64_bt`, N=K=4096, as committed in
`results.csv`; five runs agree to ≤2% on every row here, so the ordering is
the finding and the third digit is not):

| pad (halves) | stride | mod 2 KB | GFLOP/s | vs unpadded |
|---|---|---|---|---|
| 0 | 8192 B | **0** | 8618 | — |
| 8 | 8208 B | 16 B | 11713 | 1.36x |
| 64 | 8320 B | 128 B | 13052 | 1.51x |
| **128** | 8448 B | 256 B | **14028** | **1.63x** |
| 256 | 8704 B | 512 B | 13132 | 1.52x |
| 512 | 9216 B | 1 KB | 11804 | 1.37x |
| 1024 | 10240 B | **0** | 9917 | 1.15x |
| 2048 | 12288 B | **0** | 9284 | 1.08x |

**The prefill table, after padding A by 256 B** (`_pada128`; B untouched):

| kernel | N=1024 | N=2048 | N=4096 | best, % of 55.5 T |
|---|---|---|---|---|
| `wmma_reg64` | 23229 | 20216 | 24530 | 44% |
| `wmma_reg64_pada128` | 24541 | **27260** | 23747 | 49% |
| `wmma_reg64x128` | 18759 | 20008 | 25371 | 46% |
| `wmma_reg64x128_pada128` | 19595 | 23862 | 25498 | 46% |
| `wmma_wg128x256` | 18987 | 23363 | 25331 | 46% |
| `wmma_wg128x256_pada128` | 19858 | 24359 | **26008** | 47% |
| `wmma_reg64_bt` | 11002 | 8070 | 8618 | 20% |
| `wmma_reg64_bt_padab128` | **28377** | 19127 | 15669 | **51%** |

These rows, unlike the stride sweep, reproduce only to 2-4% across runs, and
the pad-A effect at N=4096 is inside that (over four runs: +3-5% on
`wg128x256`, +0.5-4.5% on `reg64x128`, −3 to +0.5% on `reg64`). The solid
effects are the 1.2-1.4x at N=2048 and the 1.8-2.6x on transposed-B, both
present in every run.

**Five things worth carrying forward:**

1. **The penalty is periodic in the stride, not monotone in the pad.** The
   three strides that are multiples of 2 KB (8192, 10240, 12288 B) are the
   three slowest rows; the fast ones sit 128-512 B past a multiple. 2 KB is
   one interleave rotation of eight 256 B channel chunks, so a K-strided
   fragment load at a power-of-two stride aims all 16 of its addresses at
   one channel. **+256 B is the measured optimum**, and a pad that lands
   back on a 2 KB multiple buys nothing — which is the difference between
   this being aliasing and it being "any pad helps".
2. **It is the strided gather that is fixed, not addresses in general.**
   Padding a *row-major* B changes nothing (`wmma_reg64_pad64`: 23538 at
   N=4096 against 24530 unpadded, and 0.2-2.7% lower at the other two sizes
   — inside the 23.5-24.9 TFLOP/s that kernel spans across five runs) because
   its B fragments are gathered element by element along N. Padding costs
   nothing either, so the engine should just always do it.
4. **A is K-strided in every variant, so this was taxing the winners.**
   That is the whole reason this item paid: `wmma_reg64` at N=2048 goes
   20216 → 27260 (1.35x), and the N=2048 dip visible in every unpadded row
   was aliasing all along. At N=4096 the same pad is worth 0-5% depending on
   the kernel, i.e. at the edge of the run-to-run spread, so the effect is
   shape-dependent and has to be measured per shape, not assumed.
5. **One prediction here was wrong, instructively.** The LDS path's global
   B reads were expected to be immune, being contiguous 128-bit staging
   loads rather than `coopMatLoad` gathers. They are not:
   `wmma_lds128_db_bt` goes 13286 → **20597** with the same pad, closing
   almost all of its gap to row-major. A staging load is contiguous only
   *within* a row, and consecutive slab rows are `ldb*2` bytes apart, so the
   wave's 64 addresses are the same aliased pattern. **What matters is the
   address stride across concurrent loads, not which instruction issues
   them.**
6. **The residue is still open, and the obvious explanations are already
   refuted.** With both strides padded, transposed-B still loses 1.6x at
   N=4096 (15669 vs 24530) while *winning* at N=1024, so the
   footprint-dependent half of §2.3 survives de-aliasing. It is not request
   count: the two kernels' A loads are identical gathers, and the *faster*
   one's B fragments are 64 scalar 2-byte loads against the slower one's 16
   wide ones. Nor is it cache-line survival above MALL size —
   `wmma_reg64_bt_k64_pad128` (deeper K-slab, so one slab consumes a whole
   128 B line) measures 14026 against 14028 at N=4096, i.e. nothing, on a
   kernel where aliasing can no longer mask it, though the same change does
   help at N=1024. Settling it needs the memory system measured directly
   rather than through a GEMM: a strided-load microbenchmark at fixed
   bytes-touched, sweeping request size, request count and stride
   independently. That is IDEAS §5.1b, and it would also turn the alias
   penalty into a curve the engine can allocate against instead of one
   kernel's anecdote.

### Thirteen sessions ago: IDEAS §2.1, the register-blocked WMMA GEMM

`IDEAS.md`'s largest remaining item is done, and it beat its own forecast.
**A register-blocked cooperative-matrix GEMM reaches 25.2 TFLOP/s at
N=4096 — 6.0x the old coopmat kernel at the same shape and 45% of this
chip's measured 55.5 TFLOP/s of WMMA throughput, up from 8%.** (Against the
old kernel's own best figure anywhere in the sweep, 5.4 TFLOP/s at N=1024,
it is 4.6x.) §2.1 predicted 2-4x and 10-20 TFLOP/s.

**New code:**
- `shaders/gemm_wmma.comp`, compiled into **twelve** variants off one source
  via `glslc -D`. Four independent axes, so every effect is attributable
  rather than bundled: accumulator grid per wave (`WM`×`WN`), waves per
  workgroup (`WAVES_M`×`WAVES_N`), LDS staging (`USE_LDS`) with optional
  double buffering (`DOUBLE_BUFFER`), and B stored [N,K] + loaded
  column-major (`B_COLMAJOR`). Arithmetic intensity climbs 16 → 85 FLOP/byte
  against the old kernel's 8.
- `bench/ops_gemm_wmma.go` — the new `gemm` cases, each verified against a
  CPU reference at a 2×2-workgroup / 4-K-slab shape before timing (small
  enough for a Go reference, large enough that a wrong tile origin or
  wave-to-tile mapping fails). Its `Detail` column carries tile shape, total
  waves and arithmetic intensity, so a row can be placed against the
  bandwidth and MMA ceilings without re-deriving them.
- `cmd/probe/main.go` — dispatches a single `.spv` once so `RADV_DEBUG=asm`
  and `RADV_DEBUG=shaderstats` can be aimed at any shader. This is IDEAS
  §6.1, and it is what found the B-layout effect below.

**The prefill table, after** (N=4096; implied traffic is GFLOP/s ÷
arithmetic intensity, to place each row against the 805 GB/s MALL and
236 GB/s DRAM ceilings):

| kernel | tile | waves/wg | AI | GFLOP/s | % of 55.5 T | implied traffic |
|---|---|---|---|---|---|---|
| `coopmat` (before) | 16×16 | 1 | 8 | 4190 | 7.5% | 524 GB/s |
| `wmma_reg32` | 32×32 | 1 | 16 | 12233 | 22% | **765 GB/s** |
| `wmma_reg64` | 64×64 | 1 | 32 | 24276 | 44% | **759 GB/s** |
| **`wmma_reg64x128`** | 64×128 | 1 | 43 | **24458** | **44%** | 573 GB/s |
| **`wmma_wg128x256`** | 128×256 | 4 | ≤85 | **25245** | **45%** | ≤296 GB/s |
| `wmma_lds128_db` | 128×128 | 4 | 64 | 21223 | 38% | 332 GB/s |
| `wmma_lds128` | 128×128 | 4 | 64 | 17163 | 31% | 268 GB/s |
| `wmma_reg64_bt` | 64×64 | 1 | 32 | 8543 | 15% | 267 GB/s |

The `wg*` row's intensity is an upper bound, not a fact: its four waves
issue overlapping fragment loads and only reach it if L0/L1 dedups them.

**Five things worth carrying forward:**

1. **Arithmetic intensity was the right variable, and it pays linearly
   once the MALL saturates.** AI 8 → 16 → 32 gives 4.19 → 12.2 → 24.3
   TFLOP/s, and the traffic column says why: at AI 16 and 32 the kernel is
   pinned at 765 and 759 GB/s (766-814 across the sizes swept), i.e. at the
   measured 805 GB/s MALL, so the 16 → 32 step is *exactly* 2.0x — a
   doubling of reuse buying a doubling of throughput out of a fixed budget.
   The 8 → 16 step is 2.9x rather than 2x because the old kernel's 524 GB/s
   shows it lost twice over: with one accumulator per wave there is no
   independent work to hide load latency behind, so it was not even
   bandwidth-saturated.
2. **Getting rid of LDS staging is the win; grouping waves is neither here
   nor there.** The top two configurations tie within run-to-run noise —
   one wave per workgroup with no shared memory (`wmma_reg64x128`) and four
   waves with no shared memory and no barriers (`wmma_wg128x256`) — while
   *every* LDS variant loses, the best of them by 1.2x. And the LDS variants
   are nowhere near bandwidth-bound (149-332 of 805 GB/s); they are bound by
   their own shared-memory reads and barriers. **The 32 MiB MALL is already
   supplying the cross-wave reuse that LDS staging exists to provide**, so
   staging it explicitly just adds barriers and `ds_load`s on top. On a
   discrete part with a small L2 this ordering would very likely invert, so
   it is a property of the APU, not of the kernel.
3. **Double buffering works and is worth 1.24x** (17.2 → 21.2 TFLOP/s) — one
   barrier per K-step instead of two, with the next slab's global loads
   issued before the current slab's MMAs so the compiler sinks the
   `s_waitcnt` past them. It just is not enough to make the LDS path
   competitive here.
4. **§2.3's B layout is a sharp, unresolved contradiction, and it is the
   single most useful thing reading the ISA produced.** RADV's `coopMatLoad`
   emits **2 `buffer_load_b128`** for a row-major A or **column-major** B
   operand, and **16 scalar `buffer_load_d16_b16`** for the other two
   combinations — the hardware WMMA operand layout is K-contiguous for both
   operands, so the conventional [K,N] row-major B is gathered element by
   element. Storing B [N,K] takes `wmma_reg64`'s loop from **577
   instructions (232 of them address arithmetic) to 389 (144)** for the same
   16 `v_wmma`… **and is 2.8x slower at N=4096** (8543 vs 24276), across
   every tiling tried. It is *faster* at N=512 (14265 vs 12838) and degrades
   exactly as B stops being cache-resident, so it is a memory-system effect
   on a strided gather, not an instruction-count one. One mechanism was
   tested and refuted (partial cache-line consumption: deepening the K-slab
   to 64 so a whole 128-byte line is consumed per MMA block changes nothing,
   8583 vs 8543). The
   untested one is channel/bank aliasing from the power-of-two leading
   dimension — a column-major fragment load issues 16 addresses `K·2` bytes
   apart within one instruction, and every K swept here is a power of two.
   **[That is what it was; see this session's section above, which also
   found the same tax on the kernels that won.]**
5. **The occupancy floor never bound anything.** §0.1's warning was that
   8-16 accumulators per wave would drop below the 2-waves-per-CU WMMA needs.
   `shaderstats` says **144 VGPRs for 16 accumulators, 252 for 32, no spills
   and no scratch** — 3-5 waves per SIMD. Register blocking here is capped by
   the 256-VGPR wave64 limit, not by occupancy, which also makes §6.2
   (wave32) more interesting than it was: WMMA is natively a wave32 shape.

### Fourteen sessions ago: IDEAS §1.1, the W4A8 GEMV kernel

The highest-value item in `IDEAS.md` is done, and it delivered more than it
promised. **W4A8 GEMV — 4-bit weights against int8 activations, both fed to
the packed-int8 dot instruction — runs at 819 GFLOP/s and 211 GB/s against
DRAM-resident weights: 2.4x the previous best decode kernel, and 89% of
this machine's 236 GB/s of DRAM bandwidth.** Decode has gone from
ALU-bound-on-the-unpack to bandwidth-bound, which is the end state §1 was
aiming for.

**New code:**
- `shaders/gemv_w4a8.comp`, compiled in two variants off one source:
  `VEC=1` loads one `uint` (8 weights) per lane per step, `VEC=4` one
  `uvec4` (32 weights). They differ in nothing else, so the pair also
  measures IDEAS §1.3's "wider loads" hypothesis in isolation.
- `bench/ops_w4a8.go` — the `gemv`-family cases (variants `subgroup` and
  `subgroup_vec4`, format `w4a8`), each verified against a CPU reference
  before timing.
- `bench/quant.go` gains `repackQ4ToW4A8`, `int8BlockSums` and the
  int32/uint32 byte helpers; `bench/ops_gemv_cold.go` gains a `coldKind`
  for the three different shader calling conventions and runs W4A8 in the
  DRAM-resident sweep too.

**The decode table, after:** (`gemv_cold`, 256MB fp16-equivalent footprint,
M=32768, N=4096, block=128 — the regime real decode runs in)

| format | kernel | GFLOP/s | GB/s | % of its own DRAM ceiling |
|---|---|---|---|---|
| fp16 | subgroup | 177 | 177 | 75% |
| q8 | subgroup | 234 | 119 | 50% |
| q4 | subgroup, float unpack | 283 | 73 | 31% |
| w8a8 | packed dot | 339 | 172 | 73% |
| **w4a8** | packed dot, `uint` loads | **698** | 180 | 76% |
| **w4a8** | packed dot, `uvec4` loads | **819** | **211** | **89%** |

Cache-resident (`gemv`, N=1024, block=128) the same kernel does 1326
GFLOP/s against W8A8's 523 and Q4's 378, and 2538 GFLOP/s at 654 GB/s at
the 64MB footprint, where 4-bit weights still fit the 32MB MALL.

**Four things worth carrying forward:**

1. **The obvious de-bias bit trick does not work, and the fix is better
   than the trick.** `(v & 0x0F0F0F0F) - 0x08080808` borrows across byte
   lanes whenever a nibble is below 8, corrupting the neighbouring weight;
   there is no borrow-free per-byte subtract. So the kernel keeps the
   nibbles unsigned and uses the **mixed-signedness** overload
   `dotPacked4x8EXT(int, uint)` (this device reports
   `integerDotProduct4x8BitPackedMixedSignednessAccelerated = true`),
   cancelling the +8 bias afterwards with
   `sum (u-8)x = sum u*x - 8*sum x`. The per-block `sum x` is the same for
   every weight row, so it is precomputed once per GEMV into a fifth
   binding and costs one short loop over `blocksPerRow` — zero extra
   instructions in the inner loop. A real engine gets those sums free out
   of its activation-quantize pass.
2. **The weight layout is a repack, not a re-quantization.** One word holds
   8 consecutive weights with its 4 low nibbles carrying elements n..n+3
   and its high nibbles n+4..n+7, so masking yields two packed-int8 words
   that line up with two consecutive words of a plainly packed int8
   activation vector — no per-token permutation of `x`. It is a pure
   permutation of `quantizeQ4`'s output, so the CPU reference is still
   built from `dequantizeQ4` on the pre-repack array, which checks the
   repack as well as the kernel. Real models would pay this once, offline.
3. **The disassembly confirms all of it** (`RADV_DEBUG=asm` on the gemv
   family; the w4a8 pipelines are the last two compiled). The `VEC=4`
   inner loop is 3 `buffer_load_b128` + one `buffer_load_d16_b16` for the
   scale, 8 and/shift pairs, **8 chained
   `v_dot4_i32_iu8 ... neg_lo:[1,0,0] clamp`** (mixed-signedness hardware
   dot: signed activations, unsigned nibbles), one `v_cvt_f32_i32` and one
   `v_fma_mix_f32`. No integer division, no per-byte extraction. W8A8 emits
   `neg_lo:[1,1,0]`, one dot per iteration, for comparison.
4. **Wide loads are worth 1.17x** (698 → 819 GFLOP/s DRAM-resident,
   1204 → 1326 cache-resident) — the difference between 76% and 89% of the
   DRAM ceiling. That is IDEAS §1.3's load half confirmed, and the fp16 and
   W8A8 GEMV kernels (75% and 73%) are the obvious next places to apply it.
   §1.2's "no integer division" is also folded into this kernel: it takes
   `log2(block)` and shifts.

### Fifteen sessions ago: the §0 "measurement validity" block from IDEAS.md

The session that produced `IDEAS.md` implemented its §0 —
the work that had to happen before optimising against any of the existing
numbers — and it changed several answers. **Every number predating it
should be considered superseded.**

**New code:**
- `shaders/alu_peak.comp` (5 variants: fp32 FMA, packed-fp16 FMA,
  `dotPacked4x8AccSatEXT`, WMMA fp16, WMMA int8) + `bench/ops_peak.go` —
  the `peak` family. Operands are register-resident before the loop and
  each iteration feeds the next accumulator, so the timed loop touches no
  memory and cannot be hoisted; 4-8 independent accumulator chains hide
  instruction latency. This establishes the *measured* ceilings every other
  kernel should be judged against.
- `shaders/empty.comp` + `bench/ops_overhead.go` — the `overhead` family:
  dispatch + pipeline-barrier cost with no work.
- `bench/ops_gemv_cold.go` — the `gemv_cold` family: decode-shape GEMV
  against weight matrices several times the last-level cache, swept by
  fp16-equivalent footprint so every format in a row holds the same number
  of weights. Fills buffers with cheap deterministic patterns rather than
  quantizing float32 weights (which would be gigabytes at these sizes) and
  deliberately skips the CPU reference, which `RunGEMV` already does at
  small sizes — only a cheap output sanity check runs.
- `bench/sysmon.go` — clock/power instrumentation. Every `Result` now
  carries mean/min/max sclk and package power for its timed batch (four new
  CSV columns plus a `detail` column), and `TimeDispatch` drives the GPU
  with ALU-heavy work before each measurement until sclk reaches 97% of the
  advertised maximum. **Closed-loop, not a fixed duration** — a first
  attempt with a fixed 200ms warmup still left expensive cases measured
  mid-ramp at 1068-2400 MHz while cheap ones ran at 2900.
- `cmd/bench`: new families registered, plus `-coldfootprints`, `-coldn`,
  `-warmclock`, `-clocksample`, `-cus`; `-bwsizes` now defaults to 12
  points from a 2MB to a 512MB footprint instead of 4.

**Measured hardware ceilings** (40 CUs at 2900 MHz), the main deliverable:

| ceiling | measured | ops/clk/CU | best real kernel | utilisation |
|---|---|---|---|---|
| DRAM bandwidth | 236 GB/s | — | 236 GB/s | ~100% |
| MALL (32 MiB) | ~805 GB/s | — | ~805 GB/s | ~100% |
| fp32 FMA | 22.9 TFLOP/s | 198 | 2.76 TFLOP/s | ~12% |
| packed fp16 FMA | 25.3 TFLOP/s | 218 | unused | — |
| `dotPacked4x8` | 54.0 TOP/s | 465 | 2.2 TOP/s | ~4% |
| WMMA fp16 | 55.5 TFLOP/s | 479 | 4.9 TFLOP/s | ~9% |
| WMMA int8 | 55.7 TOP/s | 480 | 4.7 TOP/s | ~8% |

**Five findings that change prior conclusions:**

1. **WMMA int8 is exactly as fast as WMMA fp16 here — not 2x.** 479 vs 480
   ops/clk/CU. The discrete-RDNA3 assumption that int8 matrix throughput
   doubles does not hold for RDNA3.5. The apparent 2x in the old data
   (`coopmat,q8` 9.6 TOPS vs `coopmat,fp16` 4.4 TFLOP/s at N=512) was a
   *memory* effect — int8 reads half the bytes and both kernels are
   memory-bound. This removes the main reason to write a W4A8 coopmat
   kernel (IDEAS §2.2, downgraded).
2. **Packed fp16 FMA is only 1.10x scalar fp32 FMA**, not the 2x the packed
   ALU should give. Needs an ISA dump to find out whether `v_pk_fma_f16` is
   even being emitted (IDEAS §6.1).
3. **The DVFS artefacts were real and large.** With warming,
   `gemv,subgroup,w8a8,block=128,N=1024` went 190 → **522** GFLOP/s,
   `gemv,subgroup,q4,block=128,N=2048` went 135 → **401**, and
   `bandwidth,copy` at 1M elements went 262 → **775** GB/s. Both GEMV
   sweeps are now monotonic in size. **The previous session's "the W8A8
   GEMV block-size effect is noisy rather than a clean trend" was this, not
   the kernel** — block size now barely matters anywhere in GEMV.
4. **Per-dispatch overhead is ~300 ns**, plus ~0.7 ns per workgroup
   scheduled — not the tens of microseconds that would have capped decode
   throughput. ~500 dispatches per token is under 1% of a ~20 ms token, so
   kernel fusion is justified by DRAM round-trips, not launch cost. (This
   measures dispatch + barrier *inside* a command buffer; CPU-side
   `vkQueueSubmit` + fence wait is still unmeasured.)
5. **The cache cliff is exactly 32 MiB, and it is a cliff, not a slope**:
   804 GB/s at a 32 MiB footprint, 235 GB/s at 48 MB. The in-place
   `elementwise` sweep agrees to the element — its last fully-cached fp32
   case is 8388608 × 4 B = exactly 32 MiB. So there is a **3.4x** bandwidth
   prize for any working set that can be kept under 32 MiB.

**The decode answer, corrected.** `gemv_cold` at a 256MB fp16-equivalent
footprint (M=32768, N=4096), which is the regime real decode runs in:

| format | bytes/weight | best achieved | DRAM-bound ceiling | % of ceiling |
|---|---|---|---|---|
| fp16 | 2 | 179 GFLOP/s @ 179 GB/s | 236 | 76% |
| q8 | 1 | 232 GFLOP/s @ 122 GB/s | 464 | 50% |
| **q4** | 0.5 | **284 GFLOP/s @ 73 GB/s** | **916** | **31%** |
| w8a8 | 1 | 344 GFLOP/s @ 173 GB/s | 464 | 73% |

- The old square sweep **overstates decode by 3.0x** (W8A8: 1033 vs 344).
- W8A8's lead over Q4 collapses from ~2.6x to **1.2x** once weights come
  from DRAM.
- But the prediction that Q4 would *overtake* W8A8 was wrong, and the
  reason is the useful part: **Q4 is ALU-bound in its nibble unpack even
  against DRAM**, moving only 73 of 236 GB/s. Its own bandwidth ceiling is
  916 GFLOP/s; matching W8A8's 73% efficiency would put it at ~670 —
  **2x the current champion**. That is now the highest-value item in
  `IDEAS.md` (§1.1, W4A8: load Q4 as `uint`, unpack 8 nibbles to two
  packed-int8 words with mask/subtract, feed `dotPacked4x8EXT`, scale once
  per block).

### What got built (earlier sessions)

- **`vk/`** — the engine: optional device features (fp16, int8, integer
  dot-product, cooperative-matrix) negotiated at `NewDevice`, N-buffer
  pipelines with push and specialization constants, GPU-timestamp timing
  (`ComputePipeline.DispatchTimed`), buffer allocation preferring this
  APU's `DEVICE_LOCAL|HOST_VISIBLE` unified memory,
  `PhysicalDevice.CooperativeMatrixShapes()` (this device reports only
  16x16x16, subgroup scope, fp16→fp32 and int8→int32).
- **`shaders/`** — 19 `.comp` sources compiling to 36 SPIR-V variants
  (precision/tile variants via `glslc -D`, not duplicated GLSL): bandwidth
  baseline, elementwise/relu, GEMV (naive/subgroup × fp32/fp16/q8/q4 plus
  subgroup W8A8), GEMM (naive/tiled × fp32/fp16/q8/q4, cooperative-matrix
  fp16/int8/q4, plus naive W8A8), RMSNorm/softmax (shared/subgroup), a
  standalone Q4→fp16 dequant kernel, and this session's ALU-peak and empty
  kernels.
- **`bench/`** — the harness: `bench.go` (timing + adaptive iteration
  capping), `quant.go` (fp16 conversion with round-to-nearest, GGML-style
  Q8/Q4 block quantize/dequantize), `sysmon.go` (clock/power), `ops_*.go`
  (one file per op family).
- **`cmd/bench`** — the CLI (`go run ./cmd/bench -h`).
- Original demo (`main.go`, `./strix-halo-vulkan`) still works unchanged.

### Bugs hit and fixed in earlier sessions (worth remembering)

1. **GPU driver hang watchdog**: naive GEMM at N=4096 batched into 20
   back-to-back dispatches took long enough to trip amdgpu's TDR →
   `VK_ERROR_DEVICE_LOST`, which poisons the device for every case still
   queued. `bench.TimeDispatch` now probes with 1 iteration and caps any
   batch to ~500ms worth.
2. **O(N³) CPU reference GEMM inside the size sweep** sat at 98% CPU for
   18+ minutes at N=4096. **If adding a new op/variant, verify once at a
   small fixed size, never inside the sweep loop.**
3. **`VK_KHR_shader_integer_dot_product` was requested but never enabled**:
   `vk/shim.c`'s `shim_create_device` built the feature struct into the
   `pNext` chain but never added the extension name to the enabled list
   (required since the instance targets Vulkan 1.2, where it is not yet
   core). Fixed.

## Phase 1's backlog, still open

> This is the microbenchmark phase's list, left as it stood when phase 2
> began. **It is not what to pick up next** — the newest session entry above
> carries that, and `PIPELINE.md` carries the plan. Nothing here is on the
> pipeline's critical path; it is where to look when a profile points at one
> of these kernels.

`IDEAS.md` has the full backlog with its "Suggested order of attack"
updated for what §0, §1.1, §2.1, §2.2, §2.3, §5.1b, §5.1b's traversal
follow-up, §2.7, §6.2, §1.7, §3.4, §3.5, §1.8, §1.9, §1.10, §1.11 and §1.12
found. In short:

1. **Filling the GEMV's idle lanes — the one arm none of §1.10, §1.11 or
   §1.12 built, and now the only thing left pointing at `down`'s deficit.**
   At down's K=640 a VEC=4 lane-step is 20 loads, so 44 of a wave's 64 lanes
   sit out the whole kernel, and no knob that exists changes that: `NROWS`
   gives the same 20 lanes more rows. The build that would is disjoint lane
   slices per output row reduced with `subgroupClusteredAdd`. §1.10 finding 4
   had lowered the expected gain to near zero (halving the wave is a wash;
   *filling* the lanes via VEC=1 is 1.05x slower), §1.11 finding 4 raised it
   again (the row block partly substitutes for the load width, so what the
   memory system responds to looks like bytes in flight per wave, and the lane
   map is the third way to buy them), and §1.12 finding 6 now says it is the
   *only* thing left: `down` at t=256 is at 72% of the bus, the best plan's
   own residual over its cost model is 5-13%, its pad is 617 slots of 3177,
   and the plan that removes more pad than that **loses** by up to 1.4x. What
   is left is §1.10's per-output-row cost paid over four times the rows.

2. **§2.2's two leftovers — both attribution, both cheap.** (a) The scale
   plane's layout: `QBLOCK=128` beats `QBLOCK=32` by **1.24x** between two
   binaries that are instruction-for-instruction identical, and the nominal
   byte difference is 4.4% standing behind a 24% one. The candidate is
   locality — at QBLOCK=32 a slab's 128 staging chunks read 64 rows of a
   scale plane whose row stride is 160 B, so each 2-byte scale drags in its
   own line — and the test is a blocked scale layout that makes one slab's
   scales contiguous. §1.8 has since narrowed it by elimination: the same axis
   on the GEMV, which reads its scales the way it reads its weights, costs
   1.045-1.116x — its bytes and nothing more — so the GEMM's extra 1.1x is
   about staging, not traffic. Accuracy wants the small block, so this is worth
   knowing the price of. (b) The dequant's two packed fp16 ops per two
   weights, now that the loop body is what binds; the cheap halving is
   numerically invalid (§2.2's finding 9) and the valid form is an
   fp32-epilogue per-K-block row sum of A.
3. **The selection arm's own two leftovers** (IDEAS §1.12, "still open"), both
   cheap. **A width-1 part is a big part**: at t=256 the winning plan puts 17
   experts in the width-1 bucket and 248 in the width-8 one, and at t=16 it is
   121 of 139 in width-1 — nothing measures whether a part below some group
   count is worth folding into its neighbour rather than dispatched. And **the
   calibration transfers across batches but nothing tests whether it transfers
   across shapes**, which is what an engine that fits once would actually want.

   (The item that used to stand here — "explain the down projection's 198 GB/s
   against gate_up's 235" — is **done**: it is a fixed cost per output row
   inside the wave, 19.8% of down's time against -0.4% of gate_up's, and the
   N block takes down to 98% of gate_up's rate. Dividing the workgroup count
   alone is worth 1.02x, so the grid was never it. See IDEAS §1.10. The item
   *before* that one — "carry §3.5's stride window back to the GEMV kernels" —
   is also done and came out backwards: at constant traffic the *unpadded* bank
   wins at both shapes, including down's gcd-64 row, and every pad costs
   1.12-2.4x. The window is a tiled-read rule. See §1.8 finding 6 and §5.1b
   rule 1's second amendment.)

4. **`VEC=32` is retired.** Left here so it is not re-proposed: the largest
   reduction length any of the five models decodes over is 6144, and at
   6144 every built width already reads the bus. There is no matrix for it.
5. **The two inverted cells in §3.5.** `moe.gate_up` at 8192 tokens in
   `reg32_bt_hkab4` and `reg16x32_bt_hkab4_w32`: the *grouped* dispatch is
   0.74-0.75x the per-expert loop, reproducibly across three runs, where
   every other cell goes the other way and the occupancy story predicts a
   tie. Both are the narrow-BN kernels at the largest dispatch (51k and 102k
   workgroups). Small, and the only cell in `moe` that the attribution gets
   backwards.
6. **Finish §2.7's ladder on the other winners.** It was run on `reg64_bt`
   and `reg32_bt` only. `reg64x128` and `wg128x256` are the AI-43/85 shapes
   and already sit at 252 VGPRs with 32 accumulators, so they cannot hoist as
   they stand — whether the lever survives into them is what says how it
   composes with arithmetic intensity. §6.2 has now closed off the wave32
   route to that: the register ceiling halves with the wave, so a
   32-accumulator variant spills at wave32 rather than gaining headroom.
   Also missing: a row-major `hkb4`, which completes the 2x2 of (operand
   hoisted) x (layout) that §2.7 could only half-fill, and an explanation for
   the one reproducible cell where padding *hurts* (`reg32_bt_hkab4`, 0.88x).
7. **The MALL's own slice structure — still open, and now with one more
   clue.** Under partial channel coverage the MALL keeps a working set only
   while its *span* is also inside 32 MiB, one case lands halfway, and the
   traversal axis has added MALL cells the coverage law fits (2048 B rows at
   a dense 2048 B stride, cross-wave: 484 of 868-942) next to cells it does
   not (the same rows at a 4096 B stride, walk: 774-919 at model 0.50). The
   `stride` family already runs the experiment from flags:
   `-stridefootprints 4,8,16,24,32,48 -striderowbytes 1024 -stridepads
   0,1024,3072` separates effective capacity from bus width. Worth doing
   before any MALL-blocking work in §5.1b/§2.4/§3.3, which needs a number to
   size against.
8. **IDEAS §2.4 — workgroup swizzle.** Still cheap and structural, and the
   traversal axis sharpens what it would be testing: a swizzle changes
   exactly which addresses are in flight together, which is now a measured
   axis with a model behind it. But its bandwidth premise is gone — §2.3
   pushed the AI-32 kernels to 742-887 GB/s of implied MALL traffic and §2.7
   pushed the best of them to **1218 GB/s**, past even the 965 GB/s a pure
   MALL read delivers, so implied traffic no longer bounds these kernels at
   all. Run it as a discriminator (helps AI-32, does nothing for AI-85), not
   as a bandwidth fix.
9. **Pad the strides in the other kernels — answered on the expert bank, open
   everywhere else.** This
   used to be a falsification test that the model predicted would find
   nothing: mechanism 3 looked like it had caught the GEMV kernels red-handed
   (one wave per row, rows a stride apart) until its own ladder measured that
   exact shape at the full bus at every stride. §3.5 changed the prediction —
   the rule is a window, [128, 256] B of gcd, and the real 4-bit weight rows
   sit *below* it — so it became a sweep with an expected win rather than a
   check. §1.8 finding 6 then ran that sweep on the MoE expert bank and it came
   out backwards: at constant traffic the unpadded stride wins at both shapes,
   including down's gcd-64 row, and every pad costs 1.12-2.4x (see item 3's
   note). What is left here is the same sweep on the kernels the `moe` family
   does not cover — the plain GEMV/W8A8/fp16 paths of item 10 — where the
   prediction is now "no effect, and any pad costs its own address span".
10. **IDEAS §1.3/§1.7 — the load-width rule on the other GEMV kernels.**
   fp16 GEMV sits at 75% of its DRAM ceiling and W8A8 at 73%, where W4A8 now
   reaches 101% at every N it was swept at. This is no longer "try wider
   loads and see": §1.7 gives the target width in closed form — make one
   lane-step cover a whole weight row — and it is format-independent, since
   it is about bytes of a row, not weights. A W8A8 row is twice the bytes per
   weight, so its `VEC` is half W4A8's at the same N; an fp16 row is four
   times, so a quarter. §3.4 shrank the expected gain without removing it:
   at the models' real reduction lengths the width that reaches the bus is
   usually the narrowest one, so what is on offer here is these kernels' own
   27% gap, not the 1.43x W4A8 collected at N=8192. §1.8 confirmed that at the
   MoE shapes: across VEC 1-16 the spread is 3-5%.
11. **IDEAS §1.2 — remove the runtime integer divisions** (`pc.N / pc.block`
   and `n / pc.block`) from everything W4A8 did not rewrite: the
   naive/tiled GEMM paths and the old quantized GEMV variants.
   `gemv,naive,q4` is still slower than naive fp32, which is the symptom.
12. **IDEAS §1.6 — the activation-quantize cost W8A8 and W4A8 both hide.**
   Both kernels get `x` quantized on the host, outside the timed loop, and
   W4A8 additionally gets its per-block activation sums for free. A real
   decode step must produce both on-GPU between layers. The sums are one
   extra reduction over N (cheap, and fusable into the quantize pass), but
   it should be measured rather than assumed — it is the one place where
   these numbers are friendlier than production would be.
13. **Still open from §0**: CPU-side submit/fence cost (the half of §4.1 the
   `overhead` family does not measure), and the last ISA question from §6.1
   (is `v_pk_fma_f16` actually emitted, given §0.1's 1.10x packed-fp16
   surprise). `-blocks` is worth re-sweeping only where a block-size effect
   survives warming — in GEMV it does not.
14. **IDEAS §3.7 — rewrite the subgroup reductions**, now that §6.2 has
   measured *why* they lose rather than guessing: bandwidth is linear in
   threads per row (256 → 223.7 GB/s, 64 → 68.5, 32 → 35.7), so the fix is a
   256-thread workgroup doing `uvec4` loads → per-subgroup `subgroupAdd` → a
   small LDS combine, and there is no wave-size knob worth trying first.
   Mechanical, and it deletes a variant the engine should never pick.
15. **Note on a fresh checkout.** `*.spv` is gitignored, so
   **`go generate ./...` is required** before anything runs — the thirteen
   `strided_read` variants and the 30 WMMA ones are each built from a single
   `.comp`. **This session added one**, `shaders/gemm_wmma_q4.comp`, built
   into eight binaries, so a checkout that skips `go generate` will not
   compile `shaders/shaders.go` at all.
   Worth knowing when reading ISA: RADV caches compiled pipelines on disk, so
   `RADV_DEBUG=shaderstats` and `RADV_DEBUG=asm` print **nothing** on a
   second run of the same shader. Set `MESA_SHADER_CACHE_DISABLE=true`
   alongside them.
