<!-- Pipeline stage findings, not an IDEAS section: these come from building
     the z-image slice rather than from a numbered experiment. Referenced
     from PIPELINE.md, which stays short by pointing here. -->

[← PIPELINE.md](zimage-pipeline.md) · [research index](README.md) · pipeline stage 5

# Stage 5 — the tokenizer and the text encoder

`zimage/tokenizer` is Qwen2's byte-level BPE and `zimage/qwen` is Qwen3-4B as
Z-Image runs it, both on the CPU. The tokenizer reproduces HF's fast
tokenizer id for id on all 20 corpus cases, raw and through the chat
template; the encoder matches transformers to **8e-6** end to end over 35
layers. `go run ./cmd/textenc` is the slice: prompt in, the tensor
`cap_embedder` consumes out.

Stage 5c — the Vulkan port — has not been built. The last section is what it
has to decide, and it is not the DiT's problem again.

## The pipeline uses less of this model than it looks

`ZImagePipeline._encode_prompt` takes `hidden_states[-2]`, and that index is
load-bearing three times over:

- **35 of 36 layers run.** A `hidden_states` tuple is the embedding output
  followed by each layer's output, so `[-2]` is layer 34's. The last decoder
  layer, the final norm and the lm_head never execute — 111 M of the 4.02 B
  parameters, and a fifth of a percent of nothing, but it also means **the
  output is un-normalised**, which is where the magnitudes in the next
  section come from.
- **The dump checks it rather than assuming it.** transformers v5 collects
  hidden states with a forward hook on the decoder layer instead of v4's
  explicit list, so the tuple's last entry is no longer the normed one and
  `[-2]` could have shifted. `reference/dump_qwen.py` runs the first 35
  layers by hand and reports the difference from `hidden_states[-2]`:
  **exactly 0**.
- **Padding to 512 is a no-op.** The pipeline pads to `max_sequence_length`,
  runs, and then masks the padding straight back out. The padding is on the
  right and the attention is causal, so it cannot reach a real token — the
  dump measures **exactly 0** between the padded run and an unpadded one.
  `Model.Forward` therefore runs at the prompt's own length, which at 24
  tokens is 21x less work than the pipeline does.

## Three things that differ from the DiT next door

Same company, same year, adjacent files, opposite conventions. Each of these
is a negative control in `zimage/qwen`'s tests, and each is caught by
**6500x-160000x** the tolerance, which is the useful part: none of them is
subtle once you look, and all of them are invisible if you do not.

| | Text encoder | DiT |
|---|---|---|
| RoPE pairing | **NeoX**: component `i` with `i + head_dim/2` | **adjacent**: `2j` with `2j+1` |
| Attention | **causal**, and **grouped** 32q/8kv | bidirectional, 30q/30kv |
| q/k norms | per head, `[128]` | per head, `[128]` — the one that agrees |

The GQA mapping has its own two-way choice: query head `h` is served by kv
head `h / (heads/kv_heads)`, i.e. `repeat_interleave` and not a tile.
Mapping it as `h % kv_heads` produces a tensor of exactly the right shape
whose error is 31x the reference's own RMS.

## Massive activations break an RMS-normalised error bound

The VAE's and the DiT's tests normalise a stage's worst absolute error by the
whole tensor's RMS, because those activations cross zero constantly and a
per-element relative error is dominated by the values nearest zero. That
measure does not survive this model:

| tensor | RMS | absmax | ratio |
|---|---|---|---|
| `hidden_1` | 0.215 | 8.65 | 40x |
| `hidden_2` | 0.550 | 59.8 | **109x** |
| `hidden_18` | 68.8 | 16392 | 238x |
| `final` (`hidden_states[-2]`) | 59.5 | 13753 | 231x |

A float32 summation-order difference lands on one of those outliers, and at
`hidden_2` it is 4.1e-4 absolute: **9.4e-6 of the element it sits on and
7.5e-4 of the tensor's RMS**. Only the second number looks like a bug, and it
failed a 2e-4 bound that every other stage passed.

The fix is a denominator with a floor — `max(|want_i|, rms)` per element,
rather than the RMS alone. It keeps the RMS where it was needed (the
near-zero elements) and uses the element itself where the element is the
scale. Every stage then lands at **2e-6 to 2e-5** with the same tolerance,
and the four negative controls still fail it by four to five orders of
magnitude. This is the third variation on "normalise the error by *what*"
that this pipeline has needed (§ stage 4a's is the second); the general
lesson is that the right denominator is a property of the activation
distribution, so it has to be looked at per model rather than inherited.

Two consequences beyond the test:

- **fp16 can hold these activations, but with 4.8x of headroom**, not the
  usual margin: 13753 against 65504. The DiT's stage-4b habit of a
  per-tensor range check rather than a blanket narrowing is the right one
  here too, and the hidden state handed to `cap_embedder` is the tensor to
  watch.
- The outliers are a fixed set of channels, which is the standard massive-
  activation pattern; 3.3e-5 of `final` is above 10x its RMS. Anything that
  quantizes this model per tensor will be sized by them.

## The tokenizer: RE2 cannot express the pre-tokenizer

`tokenizer.json`'s pre-tokenizer is a regex ending in `\s+(?!\S)`, and Go's
`regexp` is RE2, which has no lookahead. The splitter is therefore
hand-written, and the thing to know is that **the alternation order is the
specification**: HF's engine backtracks and takes the first branch that
matches at a position, not the longest, so ` cat` is one piece via branch 2
(`[^\r\n\p{L}\p{N}]?\p{L}+`) rather than a space and a word. Two branches
need their quantifiers' backtracking reproduced by hand — `\s*[\r\n]+` backs
off to the *last* line break in a whitespace run, and `\s+(?!\S)` gives up
its last character to whatever follows — and both are what put the leading
space inside the token rather than beside it.

- **NFC is a documented gap, not a passing test.** `tokenizer.json` asks for
  NFC normalisation; Go's standard library has no NFC and this repository
  has no third-party modules. `TestNFCIsTheKnownGap` asserts the *shape* of
  the gap instead: decomposed "café" tokenizes differently (6 ids against 5)
  and the composed spelling agrees with the reference. Anything a text
  editor or a browser produces is already NFC.
- **A negative control is only as good as the corpus.** Breaking the merge
  table is caught by 19 of 20 cases and breaking the byte alphabet by 19;
  breaking special-token handling is caught by **1**, the single case with
  `<|im_end|>` written out in it. The mechanism the entire chat template
  rides on is one test case away from being untested.

## The regime is the finding: this model is memory-bound

The DiT is 72% GEMM at 73-76% of the WMMA ceiling because at 4096 tokens its
arithmetic intensity is `4*M` = 16384 flop per byte of weight, against this
device's measured crossover of 235 ([§3.4](3.4-model-shapes.md)). The text
encoder runs the same kind of layer at **T = 8 to 512 tokens**, and its
intensity is `T` — a prompt is one sequence, not a batch. At the 24 tokens a
short prompt costs, **every projection here is 24 flop/byte: an order of
magnitude inside the memory-bound half of the roofline.**

So the two halves of one pipeline want opposite kernels, and `results/shapes.csv`'s
lesson that "kernel choice is per shape only while something else is wrong"
does not extend across this boundary — the boundary is the roofline itself.

Measured on the CPU reference (32 cores, fp32, best of 2):

| prompt | tokens | forward | rate |
|---|---|---|---|
| short | 24 | **2.08 s** | 81.7 GFLOP/s |
| paragraph | 105 | **9.17 s** | 80.9 GFLOP/s |

3.53 B parameters in the 35 layers, 14.13 GB as fp32 and **7.06 GB as fp16**.
At 236 GB/s that is a **30 ms floor** on this device — 69x under the CPU, and
flat in T until the crossover, so a 105-token prompt should cost the same
30 ms as a 24-token one. The port below lands at 57 ms and 71 ms, i.e. at
53% and 42% of that floor, and the prediction that the two lengths cost
nearly the same is the part that held.

## Stage 5c — what the port measured

`zimage/qwen/gpu.go`, 21 dispatches per layer over the DiT's four-arena
binding layout, 7.07 GB of fp16 weights in two storage buffers. Correct on
its first run against the reference dump, stage by stage.

| prompt | CPU fp32 | GPU wall | GPU dispatches | speedup |
|---|---|---|---|---|
| 24 tokens | 2.08 s | **57.1 ms** | 51.6 ms | **36x** |
| 105-128 tokens | 9.17 s | **71.4 ms** | 58.7 ms | **128x** |
| 512 tokens | — | 170 ms | — | |

Two runs agree to 1.004 and 1.000. Wall clock carries the host gather and the
read-back — 1.3 MB at this arena's 0.2 GB/s is 6.5 ms of the 128-token
figure — and stage 6 hands the tensor to `cap_embedder` on the device, so the
dispatch sum is the number the image budget should carry.

### The winning kernel moves with the prompt length

This is the finding. In the DiT, once the weight was fragment-tiled and the
grid swizzled, **one kernel won all three shapes**. Here the same seven rungs
over the same weights cannot agree, because T is not a shape the kernel sees
-- it is the arithmetic intensity itself, so the same bytes are read under
three different amounts of work:

| tokens | winner | runner-up | the DiT's default | the narrowest rung |
|---|---|---|---|---|
| 16 | `reg16x64` | `reg32x128` 1.00x | 1.24x | — |
| 24 | `reg16x64` | `reg32x128` 1.00x | **1.24x** | — |
| 64 | `reg32x128` | `reg16x64` 1.01x | 1.25x | — |
| 128 | `reg64` | `wg128x256` 1.12x | 1.12x | 1.26x |
| 256 | `wg128x256` | `reg64` 1.05x | — | 1.51x |
| 512 | `wg128x256` | `reg64` 1.11x | — | **1.92x** |

`PlanFor` is that table, and `AutoPlan` applies it per run. It is free:
every rung reads the same fragment-tiled weight, so re-planning moves a
pipeline and a tile and restages nothing — `SetPlan` is what makes the ladder
one 7 GB load instead of seven, and what lets a 24-token prompt and a
512-token one share an encoder.

**The best single kernel is `reg64_bt16`**, never worse than 1.11x anywhere in
the range, which is what `DefaultGEMMPlan` is. Taking the DiT's default
instead costs 1.24x at the prompt lengths this pipeline actually sees.

### The roofline, walked

The same run, at six lengths, moving from one half of the roofline to the
other — which is stage 5b's prediction with numbers on it:

| tokens | GFLOP/s | GB/s of weight | % of the 236 GB/s bus |
|---|---|---|---|
| 16 | 2002 | 125.2 | 53% |
| 24 | 2979 | 124.1 | 53% |
| 64 | 7513 | 117.4 | 50% |
| 128 | 12670 | 99.0 | 42% |
| 256 | 16627 | 64.9 | 28% |
| 512 | **21227** | 41.5 | 18% |

The weights are read once whatever T is, so the GB/s column falling is the
same fact as the GFLOP/s column rising: past ~235 flop/byte the run stops
being able to hide its arithmetic behind its loads. At 512 tokens it is a
compute-bound GEMM at 38% of the matrix-core ceiling; at 24 it is a
memory-bound stream at 53% of the bus.

### Attention is 0.4% of it, and that was the right thing to check first

At 24 tokens the profile is **96% GEMM**: `down` 23.6%, `gate` 21.5%, `up`
20.7%, `o` 10.9%, `q` 8.9%, `k` and `v` 5.2% each, SwiGLU 1.0%, attention
**0.4%**, and the eleven other elementwise passes 2.6% between them. At 128
tokens attention is 1.0%. The causal, grouped-query build of
`dit_attention_wmma.comp` was worth making correct and is not worth tuning:
`T*T` against `T*K` at K=2560 is not a competition at any prompt length this
model will see.

### Two hypotheses about the missing bandwidth, both falsified

At 24 tokens the best rung streams a weight it never reuses at **124 GB/s of
a 236 GB/s bus**. The obvious reading is too few loads in flight per wave, and
two rungs built to test it both came out *worse*:

| rung | change | at 24 tokens |
|---|---|---|
| `reg16x128` | eight B fragments per K step instead of four | **1.08x slower** |
| `reg16x64_k8` | an eight-tile K slab instead of four | **1.10x slower** |

So it is not per-wave request count, and it is not the workgroup count either
(608 workgroups for the gate projection at the winning rung). What the
per-dispatch profile does show is a **size** effect: the large projections
reach 155-158 GB/s and the small ones do not.

| dispatch | bytes of weight | GB/s |
|---|---|---|
| `gate`, `up` (9728x2560) | 49.8 MB | 157 |
| `down` (2560x9728) | 49.8 MB | 142 |
| `q` (4096x2560) | 21.0 MB | 155 |
| `k`, `v` (1024x2560) | **5.2 MB** | **64** |

A 5.2 MB dispatch takes 82 us where the bus would take 22. That is the next
lever and it is structural rather than a tuning knob: **concatenate q, k and v
into one [6144, 2560] weight** — they share an A operand, so it is one
dispatch of 31 MB instead of three — which the table prices at ~2.9 ms of the
51.6, or 5.6%. The same argument applies to `gate` and `up`. Both need the
elementwise shaders to take a row stride separate from their width, which
they currently do not.

What none of that explains is the 155 GB/s the *large* dispatches stop at,
which is 66% of the bus for a pure stream. That is stage 5d's question.

### What the port cost, and what it did not

- **Two shaders, two flags, five rungs.** `qwen_rope.comp` (NeoX rotary, its
  own file because the convention is a property of the checkpoint and not a
  knob), and `CAUSAL`/`GQA` on `dit_attention_wmma.comp`, which leaves the
  DiT's own builds instruction for instruction identical. Everything else —
  the GEMM, the norm-into-fp16 pass, SwiGLU, the fragment pack, the residual
  add — is the DiT's, unchanged.
- **The causal mask has to be applied twice**, and that is not an
  optimisation detail: once when the row max is taken, once on P. Masking
  only P would leave `exp2(real - future_max)` underflowing to zero in fp16
  whenever a masked score beats every real one by more than ~24 in log2
  units, and the kernel would then divide by a row sum of zero. The first
  mask is why the key loop's row-max scan breaks at the diagonal.
- **Stage 4c's bank machinery needed nothing.** 7.07 GB is two buffers at
  this device's 4.29 GB limit; four layers in four banks are **bit-identical**
  to four layers in one, and the seven ladder rungs agree with each other to
  the last bit as well.
- **21 dispatches per layer, not 18.** The q/k norms, the rotation and the
  pack are four dispatches here where stage 4b's `qkpack` fuses them into
  one, because each is a tensor the reference dumps and this was the port
  being validated. They are 0.6% of the time, so the fusion is available and
  not yet worth taking.

### The tolerance the fp16 path is held to

Per stage, one layer, against the fp32 reference: 4.8e-4 at the normed input,
6e-4 through the projections, 1.5e-3 after the q norm, 2.8e-3 after
attention, **4.1e-3** at the layer output. Over the whole 35-layer stack the
final hidden state lands at **7.9e-4** — *better* than one layer, because the
massive activations that dominate the floored denominator (see above) are far
larger at the end than at layer 1. The bound is 6e-3, and the five negative
controls miss it by **430x to 135000x**.

## What is left

1. **The 66%.** The large projections stream a never-reused weight at
   155-158 GB/s. Per-wave request count is not it (two rungs, both slower)
   and neither is workgroup count. The instrument that would separate a
   launch-ramp effect from an address-pattern one is the `stride` family
   ([§5.1b](5.1b-mall-cliff-and-stride.md)) run at *this* dispatch size, since
   it already sweeps footprint against row stride from flags.
2. **Concatenate q/k/v, and gate/up.** Priced above at 5.6% of the encoder
   from the small-dispatch table, and it needs the elementwise shaders to
   take a row stride separate from their width.
3. **Quantization has a different answer here than in the DiT.** [§3.4](3.4-model-shapes.md)
   says not to quantize the DiT for speed because it is compute-bound. This
   model is not: at the prompt lengths that matter it is at 53% of the bus
   with its time *made of* weight bytes, so 4-bit weights are worth up to 2x
   of a 57 ms run **and** take the residency from 7.06 GB to 1.77 GB. §1.1's
   W4A8 GEMV reaches 101% of the bus at M=1; what this model needs is that
   argument at M=16-128, which is [§1.9](1.9-m-block.md)'s M block in a
   regime it has not been measured in. The massive activations are what makes
   the activation side of it non-trivial.
4. **Fuse the q/k path** (stage 4b's `qkpack`, NeoX-flavoured): 0.6% of the
   run, so it is on the list only because it is already written next door.
