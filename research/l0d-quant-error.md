<!-- LLM.md L0d. Closes the first three items of IDEAS §7. Cited from
     bench/quanterr.go and cmd/quanterr. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [IDEAS §7](ideas.md) · accuracy

# L0d — what the formats actually cost, on real weights and real activations

**Three results and a bug.**

1. **W4A8 is safe.** int8 activations — the thing that makes decode reach the
   bus — cost **1.13x** the error of keeping them in fp16, averaged over 14
   real projections, and 1.6x at the worst site. The 3x throughput cliff
   between W4A8 and W4A16 is bought very cheaply.
2. **Above ~5 bits/weight there is nothing left to buy, because the
   *activations* are the floor.** int8-per-token activations alone contribute
   2.87e-2 of relative error with the weights exact. An 8-bit weight
   contributes 3.4e-3. So **W8A8 is 99% activation error** and lands within
   1.2x of where W5A8 already is, for twice the bytes and half the tok/s.
3. **Asymmetric Q4 is worth 5% at equal bits, not the 2x §7 expected.**
4. **A defect: `quantizeQ8`'s fp16 block scale goes subnormal**, which is worth
   **14x** on one real tensor in fourteen and inverts the block-size ordering.
   Q4 is immune. Two-line fix, named below.

## The instrument, and why it is not a random matrix

`cmd/quanterr` and `bench/quanterr.go`. Real weights, real activations, from a
real forward pass: Z-Image's **Qwen3-4B text encoder**, which is the Qwen3 this
machine already has locally and whose hidden size is **2560 — the same as
qwen3.8-flash-next** — so its projections are the target's dense tensors at the
right width. (LLM.md's L0d entry named `Qwen3-Embedding-0.6B`; that is the same
family at 1024 and would say less.)

Real activations are the whole point. Outliers are a property of a trained
model, they are the entire risk in int8, and a Gaussian test matrix has none —
stage 6 met the same fact from the other side, with SwiGLU overflowing fp16
"on a real prompt and never on a random one". Seven projections per layer over
two layers, 24 tokens, 55-token prompt. `down_proj`'s input needed a new trace
point in `zimage/qwen`, because it is the SwiGLU output and the existing trace
recorded only the block's ends.

Accumulation is **float64 for the reference and every candidate alike**, so the
only thing differing between rows is the quantization. Two error columns are
carried, and keeping them apart turned out to matter: the **weight
reconstruction** error `‖Ŵ−W‖/‖W‖` is the format's own, and the **output**
error is that seen through one projection's conditioning. They disagree, and
the disagreement is finding 5.

## Finding 1: the bits-vs-error front §7 asked for

Weight format alone, activations exact, mean relative output error over the 14
sites, cheapest first:

| bits/w | format | mean out rel RMS | worst site | mean weight rel RMS |
|---:|---|---:|---:|---:|
| 4.062 | q4sym/256 | 8.187e-02 | 1.556e-01 | 1.314e-01 |
| 4.125 | **q4asym/256** | 7.235e-02 | 1.373e-01 | 1.124e-01 |
| 4.125 | q4sym/128 | 7.546e-02 | 1.440e-01 | 1.205e-01 |
| 4.250 | q4sym/64 | 6.867e-02 | 1.295e-01 | 1.099e-01 |
| 4.250 | **q4asym/128** | 6.550e-02 | 1.244e-01 | 1.018e-01 |
| 4.500 | **q4asym/64** | 5.781e-02 | 1.057e-01 | 9.094e-02 |
| 4.500 | q4sym/32 | 6.089e-02 | 1.106e-01 | 9.853e-02 |
| 5.000 | q4asym/32 | 4.929e-02 | 8.430e-02 | 7.898e-02 |
| 8.500 | q8sym-f32s/64 | 3.877e-03 | 7.183e-03 | 6.063e-03 |
| 9.000 | q8sym-f32s/32 | 3.426e-03 | 6.059e-03 | 5.436e-03 |
| 16.000 | fp16 | 7.630e-05 | 2.867e-04 | 1.042e-04 |

(The `q8sym` rows with the shipped fp16 scale are omitted here and discussed in
finding 5, because they are measuring a bug rather than a format.)

**The equal-bits comparisons, which are the ones that decide anything:**

| bits | symmetric | asymmetric | asym is |
|---:|---:|---:|---:|
| 4.500 | q4sym/32 6.089e-02 | q4asym/64 5.781e-02 | **1.053x better** |
| 4.250 | q4sym/64 6.867e-02 | q4asym/128 6.550e-02 | **1.048x better** |
| 4.125 | q4sym/128 7.546e-02 | q4asym/256 7.235e-02 | **1.043x better** |

Consistent, small, and much less than §7's expectation that a zero-point
"typically halves the error". Against that 5%: asymmetric is genuinely free at
*decode* — §1.1's W4A8 GEMV already computes a per-block activation sum to
cancel the +8 bias, and `sum (q−z)x = sum q·x − z·sum x` reuses exactly that
term — but the GEMM path has to carry the zero-point out to the fp32 epilogue
(§2.2 finding 9 shows what happens if it does not). A 5% error reduction is not
obviously worth a second epilogue term.

Also visible: **the 4-bit and 8-bit families are 14x apart and nothing lives in
between.** That gap is where the next finding bites.

## Finding 2: the activations are the floor, and they cap what more bits can buy

With the weights left in fp16 and only the activations quantized to int8 per
token, the mean output error is **2.865e-02** (range 8.1e-3 to 8.3e-2). That is
the floor any W*A8 scheme sits on, and weight error adds to it in quadrature —
which the data confirms:

    sqrt(0.0609² + 0.0287²) = 0.0673   predicted W4A8
                              0.0691   measured
                                       2.7% apart, so the two errors are
                                       near-independent

Putting the ladder together, with int8-per-token activations throughout:

| weight format | bits/w | weight error | total with int8 acts | of which activation |
|---|---:|---:|---:|---:|
| q4sym/256 | 4.06 | 8.19e-02 | 8.68e-02 | 11% |
| q4sym/32 | 4.50 | 6.09e-02 | **6.91e-02** (measured) | 17% |
| q4asym/32 | 5.00 | 4.93e-02 | 5.70e-02 | 25% |
| q8sym-f32s/32 | 9.00 | 3.43e-03 | 2.89e-02 | **99%** |

**So W8A8 buys 2.4x less error than W4A8 for twice the bytes and half the
tok/s — and 99% of what is left is the activations, not the weights.** Eight-bit
weights behind int8 activations are almost entirely wasted. If accuracy ever
demands better than ~5 bits, the thing to fix first is the activation format,
not the weight format; and fixing the activation format means W4A16, which
§1.1 measured at 31% of the bus.

That is the sharpest statement L0d produces for the vertical: **stay at
4-4.5 bits, because the next rung up is throttled by something else.**

## Finding 3: per-token activation scales, and they are free

| weights | acts exact | acts fp16 | int8/token | int8/tensor |
|---|---:|---:|---:|---:|
| fp16 | 7.63e-05 | 1.52e-04 | 2.87e-02 | 5.64e-02 |
| q4sym/32 | 6.09e-02 | 6.09e-02 | 6.91e-02 | 8.62e-02 |

One scale per row instead of one per tensor is worth **1.25x on the combined
error and 1.97x on the activation error alone**, and it costs nothing: a real
engine computes the row's scale in the RMSNorm epilogue it is already running
(§3.1). Note also that `acts fp16` and `acts exact` are identical to three
digits behind 4-bit weights — **W4A16 and W4A32 are the same measurement**, so
nothing is lost by keeping activations at fp16 rather than fp32.

## Finding 4: `down` is the tensor to watch

Per-site cost of int8-per-token against fp16 activations, at q4sym/32 weights:

| site | int8tok / fp16 | | site | int8tok / fp16 |
|---|---:|---|---|---:|
| L0.q, L0.k, L0.v, L0.o | 1.1x | | L1.q, L1.k, L1.v | 1.1x |
| L0.gate, L0.up | 1.0x | | L1.o, L1.gate | 1.0-1.1x |
| **L0.down** | **1.6x** | | **L1.up** | 1.2x |
| | | | **L1.down** | **1.3x** |

and the outlier statistics that explain it:

| site | max/rms | channel spread | kurtosis |
|---|---:|---:|---:|
| L0.gate, L0.up | 11.5 | 5.3 | 4.8 |
| L0.q/k/v | 47.5 | 32.5 | 178.8 |
| **L0.down** | 132.0 | **99.5** | 3 636.9 |
| **L1.down** | 322.5 | **4 172.1** | **87 187.7** |

Gaussian kurtosis is 3. `down_proj`'s input is the SwiGLU output, it is the
worst-behaved activation in the block by two orders of magnitude, and it gets
worse with depth. It is also the one projection whose input cannot be made
well-behaved by a norm, because there is no norm between the gate and it.

For the target model this is 1/3 of the expert bytes, so ~13% of the decode
budget. Running only `down` at W4A16 would cost roughly +26% of decode time at
§1.1's 31%-of-bus figure, which is far too much for 1.6x on one matmul. The
cheaper mitigations — finer activation groups on that input, or a rotation —
are unmeasured and are the obvious follow-up.

## Finding 5: the bug — `quantizeQ8`'s fp16 scale goes subnormal

This was found by chasing an impossible number: Q8's error got *worse* with
finer blocks, which no quantizer can do. Separating weight reconstruction from
output error localised it to one site, and the mechanism is arithmetic.

`quantizeQ8` stores `scale = maxAbs/127` as fp16. fp16's smallest **normal**
value is 6.1e-5, so any block whose largest weight is below **7.75e-3** gets a
subnormal scale — which keeps only a handful of mantissa bits, at which point
the scale is the error rather than the int8. Q4 never sees it: its scale is
`maxAbs/7`, eighteen times larger.

| site | % of Q8 scales subnormal, block 32 → 256 | % of Q4 scales |
|---|---|---|
| every site but one | 0.00% at every block | 0.00% |
| **L1.up** | **41.51 → 39.20 → 36.64 → 33.96%** | 0.00% |

The subnormal fraction *falls* as blocks get coarser, because a coarser block
has a larger maximum — which is exactly the inverted ordering. And the control
settles it: the same scheme with the scale kept in fp32,

| site, block | fp16 scale | fp32 scale | |
|---|---:|---:|---|
| L1.up, 32 | 7.652e-02 | **5.425e-03** | 14.1x, and monotone again |
| L1.up, 256 | 6.444e-02 | 7.179e-03 | |
| L0.up, 32 | 5.399e-03 | 5.398e-03 | identical — 0% subnormal there |

**The fix is to floor the scale at fp16's smallest normal.** That costs a
little resolution on the affected blocks (fewer than 127 codes used) and keeps
the whole mantissa, and it preserves the shader contract, since both sides read
the same stored fp16 value. It is deliberately **not applied here**: it changes
`quantizeQ8`, which every W8A8 correctness check builds its CPU reference from,
so it wants to be its own change with those checks re-run.

Scope: no published *timing* is affected — these are all CPU-side numbers. What
is affected is any accuracy claim about W8A8, and it matters for the vertical
because **UD-Q4_K_XL puts every dense tensor at Q8_0**, so a phase-2 transcode
through this quantizer would hit it. The format the vertical plans to ship is
Q4, which is immune.

## What this does not say

- **This is one projection's output error, not the model's.** Each of these
  feeds a residual stream that carries a much larger accumulated signal, so a
  6e-2 relative error on a projection output is not a 6e-2 error on a hidden
  state, and errors through 48 layers neither add nor stay put in any way
  measured here. §7's last item — **real perplexity** — is still the acceptance
  criterion, and it needs the model loaded, so it is L9.
- **Two layers of one model.** Layer 0 and layer 1 of a 36-layer Qwen3-4B. The
  outlier statistics already move sharply between them (channel spread 99.5 to
  4172 on `down`), so the deep layers are likely worse and are unmeasured.
- **The MoE experts are not represented at all.** This model is dense. Whether
  a 512-way expert bank quantizes like a dense FFN is an open question, and it
  is 97% of the target's parameters.
- The simulation dequantizes both operands and multiplies in float64. The real
  W4A8 kernel accumulates int32 within a block and fp32 across blocks, which is
  marginally worse; that gap is not what these numbers are about.

## Reproducibility

    go run ./cmd/quanterr -layers 2 -tokens 24 -blocks 32,64,128,256 \
        -sites q,k,v,o,gate,up,down          # ~10 min, CPU only
    go run ./cmd/quanterr -describe -sites up # weight stats + subnormal-scale census

Deterministic — no sampling, no timing — so there is no two-run figure to
quote. The one number that moves between runs is nothing.
