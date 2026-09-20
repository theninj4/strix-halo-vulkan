<!-- LLM.md L5a. The MoE block in Go against the 4k dump: the routing, the
     experts, the shared expert, and a routing distribution nothing had
     measured. Cited from llm/moe.go. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L4a](l4-qsa.md) · [L4b](l4b-qsa-gpu.md) · [L3](l3-deltanet.md) · phase 1

# L5a — the MoE block: every tensor matches, and the routing is nothing like balanced

**Result: all fourteen tensors llama.cpp names inside the FFN half match, the
top-10-of-512 selection is reproduced as a set on all 4096 tokens of the 4 k
dump, and two things fell out that are not about correctness.** The first is
that **modelling the reference's own arithmetic makes the fit worse here** —
`Exact` f32 is 1.4-1.8x nearer llama.cpp than `RefQ8` is, where everywhere
else in this vertical modelling it was worth 35x to 2061x. The second is the
number L5b will be designed around: at a 512-token ubatch **274 of 512
experts are touched and one of them takes 95% of the tokens**.

## The block

`build_layer_ffn` is two halves added together, and the 4 k dump names every
node of both because L4a's filter took layers 0 and 3 — so L5's fixture
existed before L5 started. That matters more here than anywhere else in the
vertical: the routing is a **discrete** function of the logits, and a 7-token
dump would have said nothing at all about whether our top-10 of 512 is the
reference's.

    router (F32 [2560,512])  ->  softmax over 512  ->  argsort desc, take 10
      ->  weights = probs[topk]  ->  sum  ->  clamp  ->  divide
      ->  per selected expert: gate, up  ->  silu(gate)*up  ->  down
      ->  x weights  ->  sum the ten                        = ffn_moe_out
    shared: gate, up -> silu(gate)*up -> down               = ffn_shexp
            x sigmoid(one-column gate)                      = ffn_shexp_gated
    ffn_out = ffn_moe_out + ffn_shexp_gated

Three things are the reference's decision rather than the architecture's, and
each is asserted rather than assumed:

- **`norm_w` is on and `expert_weights_scale` is absent**, so the ten weights
  are normalised and never scaled — and the dump confirms it by having no
  `ffn_moe_weights_scaled` node.
- **The weights are applied after the FFN**, not before. `weight_before_ffn`
  is a llama4 special case.
- **`swiglu_clamp_exp` is absent**, so `ffn_moe_swiglu` and not
  `ffn_moe_swiglu_limited` is what the graph emits.

## The input is `hc_mixed` written the second time

The hyper-connection block runs **twice a layer** — once before attention and
once before the FFN — so `hc_mixed-3` appears twice in the trace, at `0061_`
and `0091_`. The FFN's input is the second. Taking the first would have
produced a plausible tensor of exactly the right shape, off by one whole
sub-layer, and nothing but the occurrence index says which is which.

## L5a-1: the routing is the reference's, set for set on all 4096 tokens

| tensor | rms against llama.cpp | |
|---|---:|---|
| `ffn_moe_logits-3` | 3.606e-05 | on values to 9.92 — the F32 router, fp16 operands and an f32 accumulator |
| `ffn_moe_probs-3` | 2.625e-08 | |
| `ffn_moe_weights-3` | 1.673e-07 | |
| `ffn_moe_weights_sum-3` | 1.144e-06 | |
| `ffn_moe_weights_norm-3` | 5.187e-07 | |

> **The selection: 0 of 40 960 slots differ as a set.** That is the number that
> matters, because a top-k is discontinuous and an rms on the weights would
> hide a swapped expert completely — the tenth and eleventh probabilities
> differ by very little and the weight attached to either is nearly the same
> number. All 4096 rows pick the same ten experts the reference does.
>
> **Two slots differ in order, on one row, and they are a tie.** Token 2318
> ranks experts 177 and 265 the other way round; the reference's *own*
> probabilities for them are 0.00535551179 and 0.00535551412 — **4.35e-07
> apart relative**, against a router whose `maxRel` on `ffn_moe_probs` is
> 3.4e-05. So the test does not demand the order agree; it demands that any
> slot that disagrees be a pair the router's own precision cannot resolve,
> which is a claim about the block rather than about luck.

The router's numeric is L2e-3's and not L4a-5's: an F32 × F32 matmul above the
8-column threshold goes to the fp16 coopmat path, but `.f16acc` is a property
of the *quantised* kernels, and L4a-5's census put `ffn_moe_logits` at **0.0%**
exactly-representable halves where every quantised matmul is at 100.0%. So the
operands are fp16 and the accumulator is f32, which is what `matvecW16` does.
`shared_expert_gate` is 0.0% for the opposite reason — one output column keeps
it on the f32 vector path at any prompt length — and it is correspondingly the
tightest tensor in the block at **4.152e-05 rms on values to 2.753**.

## L5a-2: the block matches, at the tolerance the fp16 accumulator sets

Over 48 tokens, from llama.cpp's own input, every tensor of both halves:

| tensor | rms | ref \|max\| | |
|---|---:|---:|---|
| `ffn_moe_gate-3` | 2.939e-03 | 3.84 | Q4_K weights, int8 activations, fp16 accumulator |
| `ffn_moe_up-3` | 2.679e-03 | 2.54 | |
| `ffn_moe_swiglu-3` | 7.382e-04 | 3.70 | |
| `ffn_moe_down-3` | 3.334e-04 | 0.27 | Q5_1 |
| `ffn_moe_weighted-3` | 4.159e-05 | 0.066 | |
| `ffn_moe_out-3` | 1.319e-04 | 0.104 | the ten summed |
| `ffn_gate-3` / `ffn_up-3` | 2.995e-03 / 2.635e-03 | 2.53 / 2.91 | the shared expert, Q8_0 |
| `ffn_shexp-3` | 2.927e-04 | 0.249 | |
| `shared_expert_gate-3` | 4.152e-05 | 2.75 | the f32 vector path |
| `ffn_shexp_gated-3` | 3.989e-05 | 0.037 | |
| **`ffn_out-3`** | **1.378e-04** | 0.102 | the block |

These are L4a-5's numbers and not L2's. Every one of the six matmuls here is a
quantised weight against an int8 activation, and at 4096 columns the reference
accumulates all of them in fp16 — so ~3e-03 rms on a tensor whose own rms is
order 1 is the *oracle's* precision, not ours. The 1e-6 figures L2b, L2d, L2e
and L3a report are facts about a 7-token dump.

## L5a-3: modelling the reference's arithmetic makes it worse — by 1.4-1.8x

This is L4a-5 tested where it costs most rather than taken on trust, and the
answer is sharper than L4a-5's own "three operand models within 1.06x":

| tensor | `Exact` (pure f32) | `RefQ8` (the reference's int8 activations) | |
|---|---:|---:|---:|
| `ffn_moe_gate-3` | **2.217e-03** | 3.054e-03 | 0.73x |
| `ffn_moe_out-3` | **6.552e-05** | 1.152e-04 | 0.57x |
| `ffn_shexp-3` | **1.640e-04** | 2.824e-04 | 0.58x |

> Everywhere else in this vertical, modelling the reference's int8 activations
> was the single largest correction available: **233x** at L2b-2, **69x** at
> L2e-3, **2072x** at L2e-2, **2061x** at L4a-6. Here it is worth **less than
> nothing.** The reference has already thrown away eleven mantissa bits in the
> accumulator before our quantisation error is added, so the two errors are
> independent and ours adds in quadrature rather than cancelling.

The practical consequence is for **our own kernels**, and it is the same
conclusion L2c-3, L2f-6 and L3b-6 reached from the other direction: an f32
accumulator is *nearer the model and further from the oracle*, and at the MoE
the gap is large enough that a tensor comparison should be read as a bound
rather than as a fit. It also means L5b does not have to reproduce anything
here — there is nothing to reproduce.

## L5a-4: the routing is nothing like balanced, and that is what L5b is sized by

L2a's decode budget said "each expert is read 10 times in 512" — 512 experts,
10 per token, so a mean bucket of 10. The mean is right and nothing else about
that picture is. Measured on llama.cpp's own `ffn_moe_topk`, cross-checked
bucket for bucket against ours:

| ubatch | experts touched | bucket min | mean over touched | max | |
|---:|---:|---:|---:|---:|---|
| 512 | **274 of 512** (54%) | 1 | 18.7 | **484** | 26x the mean, **95% of the ubatch's tokens** |
| 2048 | 399 (78%) | 1 | 51.3 | 1224 | 24x, 60% |
| 4096 | 435 (85%) | 1 | 94.2 | 1873 | 20x, 46% |

> **Expert 454 is a second shared expert in all but name.** At a 512-token
> ubatch it is chosen by **484 of the 512 tokens**, and it is the
> *highest-ranked* expert for 274 of them. Its slot histogram is
> `[274 77 61 33 13 11 5 4 5 1]` — it is not merely present, it is usually
> first.

Four consequences for L5b, all of which change the design rather than merely
decorating it:

- **Only about half the bank is read per ubatch at 512 tokens**, not all of
  it. The arithmetic that says "the expert bank is 64 GB at 2048 tokens, a
  273 ms floor" (L2a's retirement of §3.4) is an upper bound on traffic, and
  the real figure depends on the prompt.
- **A workgroup-per-expert kernel has a 26x load imbalance** and cannot be
  scheduled statically. The permutation buffer has to be sized for a bucket
  of 484 at ubatch 512, not for the mean of 10.
- **The hot expert is effectively dense.** Its 4.9 M weights are read once and
  amortised over 484 tokens, where a cold expert's are read once for one
  token: the arithmetic intensity per expert varies by **484x** inside a single
  dispatch. That is the argument for a grouped GEMM over a gather-GEMV, and it
  says the grouping has to be by expert with a variable row count.
- **At decode it is a caching question.** One token reads 10 experts, but the
  top one is the same expert 95% of the time, so ~0.5 MB of Q4_K is read every
  step and would sit in the 32 MiB MALL if anything kept it there.

None of this is in the architecture, none of it is in llama.cpp's source, and
it is invisible at 7 tokens.

## The controls

Three things in this block have a plausible wrong version that every tolerance
above would pass, so each has a test that demands the wrong version disagree:

- **`TestMoEExpertRowsAreExpertMajor`** — `ffn_gate_exps` is `[2560, 640, 512]`
  and expert *e*'s output row *j* is flat row `e*640 + j`. The interleaved
  reading is equally plausible and produces a tensor of the right shape built
  from 640 rows of other experts. Routing every token one expert along is
  **175x worse**.
- **`TestMoESwigluTakesTheGate`** — `ggml_swiglu_split(gate, up)` is
  `silu(gate) * up`. The transpose has the same shape, magnitude and sign
  pattern nearly everywhere, because silu is near-linear away from zero. It is
  **59x worse**, which is the smallest margin of the three and is why it needs
  a test at all.
- **`TestMoEWeightsSumIsClamped`** — the clamp to fp16's smallest normal
  guards a division by zero that a softmax over 512 logits can never reach.
  The smallest `weights_sum` on this prompt is **0.0554** against the clamp's
  6.1e-05, and the reference's two nodes are bit-identical on all 4096 rows.
  So the clamp is invisible in the data and has to be asserted from the graph:
  the test checks both that we reproduce it and that it is the identity here.

## How to run it

    go test ./llm/ -v -run TestMoE

`TestMoERouting` and `TestMoERoutingBalance` run over all 4096 tokens in under
a second. `TestMoEBlock` runs the expert half over 48 tokens in ~2.2 s; the
knob is `moeTokens`, and it is a knob on runtime rather than on coverage,
because the tensors compared are the reference's own row for row. The cost is
dequantisation: one (token, expert) pair is 4.9 M weights of Q4_K and Q5_1, a
full 4096-token prefill is 40 960 pairs, and the reference walks by **expert**
so that each bank is gathered once for all its tokens rather than once per
token — the same permutation a kernel does, for the same reason.

## What is open for L5b

- **The grouped GEMM's schedule against a 26x imbalance.** §2.2 measured a Q4
  WMMA MoE block at 642 ms against llama.cpp's 1626 (2.53x) — but at what
  bucket distribution? The shapes in `results/shapes.csv` assume a uniform
  split, and the real one has one expert with 484 rows and 273 with fewer than
  19.
- **Whether the router wants a kernel of its own.** It is 5.2 M F32 weights
  read every token, 0.25 GB a graph, and its output is 512 columns wide —
  L2a's "tiny-N F32 matmul" category from the other side. The softmax, the
  top-10 and the normalise are all one workgroup a token, which is
  `llm_attn_select.comp`'s shape exactly.
- **Whether the shared expert fuses with the hot routed one.** They are the
  same shape ([2560, 640] twice and [640, 2560]), the shared one runs for every
  token, and expert 454 runs for 95% of them.
