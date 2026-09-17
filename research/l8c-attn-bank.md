<!-- LLM.md L8c-7. The full-attention layer on the 4.5-bit bank: the fourth
     family, the one that needed no new format at all, and a fifth family that
     took a second context length to grade at all.
     Cited from llm/gpu_attn.go, llm/bank.go, llm/graph.go. -->

[← LLM.md](../LLM.md) · [research index](README.md) · bank

# L8c-7 — the attention layer at 4.5 bits, and a family that took two contexts to grade

**Result: the twelve full-attention layers are on L8c-4's bank, decode is
31.4 tok/s against 30.1 — 1.25x llama.cpp — prefill is faster at every ubatch,
and the family costs +1.65% of perplexity, which is 5.52 percentage points per
GB of decode token and makes it the most expensive of the four by 2.1x. Four
families together are 4.1787, +3.72%, against a sum of their four separate
corpus deltas of 3.71% — additivity to 0.01 pp. And `qsa_indexer` rode along:
bit-identical over 145 chunks at n_ctx 2048 because nothing there reads it,
and +0.005% at n_ctx 2560 where the selection bites.**

Twelve of the 48 layers are full attention with the QSA indexer, and their two
projections are 0.635 GB of a 6.334 GB decode token on the checkpoint's own
widths — the fourth of the five streamed dense families, and the last one with
a kernel question in it. There was none. `llm_gemm.comp -DQ4B` and
`llm_gemv.comp -DQ4B` already had every arm this block needs, including the
fp16 tail arm L8c-5 first exercised, and both of this layer's matrices are a
multiple of 256 wide — so unlike L8c-6 there is no record to invent and unlike
L8c-4 there is no encoder to write. What is here instead is the wiring, one
structural choice about the tail, and the first family in this stage the gate
as written **cannot measure at all**.

Files: `llm/gpu_attn.go`, `llm/bank.go`, `llm/graph.go`, `llm/attn.go`,
`llm/model.go`, `cmd/llm/bench_attn.go`. Results:
`results/l8c_decode_attn.csv`, `results/l8c_decode_attn_idx.csv`,
`results/l8c_graph_attn.csv`,
`results/l8c_ppl_attn_only.csv`, `results/l8c_ppl_attn_q4k.csv`,
`results/l8c_ppl_attn_idx.csv`, `results/l8c_ppl_attn_c2560.csv`,
`results/l8c_ppl_idx_c2560.csv`, `results/l8c_attn_gemv.csv`.

---

## 1. Nothing new, and that is the finding about the kernel

The three families before this one each needed something the bank did not
have. L8c-4 needed the format itself — the nibble tiling, ggml's sixteen-byte
record, the `-DQ4B` arms. L8c-5 needed `tileBQ4K` to write into a plane wider
than its source, and the layer index on `DeltaNetWeights` so each source of a
fused matrix could be calibrated under its own name. L8c-6 needed a record
that is not ggml's at all, because `hc_*_up` is 320 wide.

This family needed none of it:

| | `full_attn` |
|---|---|
| the fused projection | [13952, 2560] — `k % 256 == 0`, eight-group super-blocks |
| the output projection | [2560, 6144] — the same |
| the record | ggml's sixteen bytes, `get_scale_min_k4` |
| the tail | the indexer's two BF16 matrices, exactly where L8a put them |
| the arms | `-DQ4B` MODE 2 with `Q8_SPLIT`, and `llm_gemv.comp -DQ4B` |

so the whole of it is `AttnGPU` learning the three-valued `DenseBank` the
other blocks already speak. Which is also where the last of the two-valued
spelling died: `q8Pipe`, `gemvPipe` and `gemmBuilds` — the `q8 bool`
constructors' half of the bank API — had no callers left once this block
stopped taking a `bool`, and are gone.

`NewAttnGPUBank` takes the bank, the format and one more thing the other
blocks do not need, and §3 is what that is.

## 2. The bank is the format, said twice and then a third time

The stage's one load-bearing property is that what is stored is what L8c-3
measured, and it is checked the same two ways as the two families before it.

**On the device, value for value.** `TestAttnGPUQ4IsTheSim` stages layer 3
twice from the same checkpoint weights — once on the fp16 bank with the format
applied through the same encoder, once on the real 4.5-bit bank — and runs the
whole layer over the trace's seven tokens. The fused projection's 97 664
values and the layer's 17 920 output values are **identical**. The comparison
is taken at the layer's output as well as at the projection, so the per-head
norm, the two ropes, the indexer's pooling, its rectified score, the flash
attention and the output projection have all run on top of it — an address
wrong by one record cannot hide behind a tolerance. Both arrangements of the
tail are checked, because the tail is the one structural choice here.

**Over the corpus, chunk for chunk.** The bank and its simulation at eight
chunks:

    bank   LLM_DENSE_BANK=full_attn=q4_k/32
    sim    LLM_DENSE_SIM=full_attn=q4_k/32
           LLM_DENSE_SIM_SRC=q8 LLM_DENSE_SIM_QUANT=imatrix

agree **chunk for chunk to the last digit the CSV carries** — 1.4301, 1.3942,
1.4615, 1.9409, 2.1402, 2.0518, 1.9405 and a final **2.0405**, with the
`nll` column identical to six decimals on all eight rows.

`SRC=q8` is what makes them the same question: the simulation's default scope
covers the indexer's two BF16 matrices, which the bank leaves in its fp16
tail, so without it the two arms quantise different sets of weights.

**And a third check this family needed and the others did not.** The two
tests above both run the GEMM: the trace is seven tokens and a prefill chunk
is 2048, so neither reaches `llm_gemv.comp`, which exists only at one row —
and the fused projection's tail is exactly the thing a split-K GEMV can get
wrong, because it has to derive the same split from `pc.lowRank` that the
GEMM does and a column read out of the wrong plane is not a tolerance but
somebody else's indexer query. `TestAttnGPUQ4Gemv` is that comparison at
every rung, on both arrangements of the tail, with the tail's columns also
compared on their own — they are 4.6% of the width and would otherwise hide
inside the whole matrix's rms. It is a tolerance and not an equality for
L8c-4's reason: a K-quant group is affine, so the GEMV forms `d*sc*l -
dmin*m` in a register where the GEMM rounds that same value to a half on its
way into LDS, one fewer rounding per weight.

## 3. The tail stopped being cheap, and it is a family of its own

The indexer's `q_proj` and `k_proj` are the only **BF16** weights in this
checkpoint. L8a left them as halves in a tail of their own for two reasons:
int8 would have been a real re-quantisation rather than a repacking, and
L4b-4 had already measured the indexer's score to be the sensitive part of
this layer — 24x further from the reference for an unmodelled activation than
anything else in it. A nibble is not a reason to revisit the second of those.
But the first argument has changed shape, because the *price* of the tail has:

| a full-attention layer, staged | bytes | bits/weight |
|---|---:|---:|
| halves (pre-L8) | 102.89 MB | 16.000 |
| the checkpoint's Q8_0 (L8a) | 57.94 MB | 9.010 |
| **`q4_k/32`, indexer in the fp16 tail** | **32.22 MB** | **5.010** |
| **`q4_k/32`, indexer on the plane** | **28.94 MB** | **4.500** |
| 12 layers | **0.39 GB** / **0.35** against 0.70 and 1.23 | |

640 rows of 13952 — 4.6% of the fused matrix — carried at 16 bits cost
**0.51 bits a weight over the whole layer**, where the gated DeltaNet's 96
F32 rows cost 0.090 and the hyper-connection block's 32 cost less still. On
L8a's bank the same tail cost 0.51 of 9.010, which is 5.7% of the layer; at
4.5 bits it is 11.3%, and it is the single largest thing left between this
block and the format it claims to be.

So `qsa_indexer` is wired as a family of its own rather than folded into
`full_attn` or left alone. Naming it in the plan moves those two matrices
into the quantised plane beside the query, the key and the value — and then
there is no tail at all, `pc.gateOff` is `NO_W`, and the block's GEMM takes
the same branch the output projection always did. Leaving it out keeps
L8a's arrangement exactly. `qkvQuantRows` is the one function that knows,
and everything else follows from it.

The quantised plane covers **all** 13952 rows either way, because the kernel
derives the record plane's base from `pc.gemmN * pc.gemmK` and `gemmN` is the
stride of the output the store writes. With the tail in place that is 0.82 MB
a layer of nibbles and records nothing reads — against the 25.7 MB a layer the
width saves — and it is why the first row above is 5.010 bits and not 4.94:
the tail's 640 rows are carried at 16 bits *and* at 4.5.

## 4. What the width buys, and D14's boundary measured twice inside one block

**At decode the block is 1.52x**, and it is the smallest of the four families
because it is the smallest family: 12 layers against the DeltaNet's 36.

| a decode token | L8c-6 (three families) | **L8c-7 (four)** | **+ `qsa_indexer`** |
|---|---:|---:|---:|
| decode | 30.05 / 30.11 tok/s | **31.41 / 31.57** | **31.51** |
| against llama.cpp's 25.15 | 1.20x | **1.25x** | **1.25x** |
| a token | 33.3 ms | **31.8 ms** | **31.7 ms** |
| the attention block, per token | 3.5 ms | **2.3 ms** | **2.1 ms** |
| the block's staged bank | 0.70 GB | **0.39 GB** | **0.35 GB** |
| residency | 80.20 GB | **79.89 GB** | **79.85 GB** |

On its own two-layer ladder at one token the block is 232.6 us a layer to
**110.0**, 2.11x — the fused projection 176.6 to **64.0** and the output
projection 27.3 to **17.8** — but every rate in that ladder is 1470-1820 GB/s
against a 242 GB/s bus, which is **D16**: a two-layer fixture repeats one
dispatch over a weight that fits the 32 MiB MALL, so what it ranks is kernel
shape against an L3 hit. The whole-model number, 3.5 ms a token to 2.3, is the
one that counts, and it is 1.52x where the fixture says 2.11.

**At prefill it is faster at every ubatch**, which is the third time D14's
hazard has been asked and the second time it has not fired:

| the block, per graph | 128 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|
| q8 (L8c-6) | 8.5 ms | 24.6 | 50.0 | 109.7 |
| **`q4_k`** | **7.2** | **22.6** | **45.6** | **102.1** |
| the whole graph, tok/s | 313.9 | **671.4** | 884.5 | **1071.8** |
| against L8c-6's | 312.7 | 669.7 | 882.1 | 1065.9 |

and the reason is L8c-5's, on the same axis and now measurable **twice inside
one block**, because this layer's two matrices land on opposite sides of it:

| | staged | 32 MiB MALL |
|---|---:|---|
| the fused [13952, 2560] projection, halves | 71.4 MB | no |
| — the checkpoint's Q8_0 | 35.7 MB | no |
| — **`q4_k/32`** | **17.9 MB** | **yes** |
| the output [2560, 6144] projection, halves | 31.5 MB | yes |
| — the checkpoint's Q8_0 | 15.7 MB | yes |
| — **`q4_k/32`** | **7.9 MB** | **yes** |

The fused projection **changes side** going from int8 to nibbles, exactly as
the DeltaNet's did, so its re-reads come off DRAM and it is 1.13x at ubatch
2048 on the fixture's own ladder. The output projection **fits at every
width**, so the narrower bank buys it nothing and adds the unpack's ALU — and
on the fixture it is **1.00x at 128 tokens and 1.04x slower at 512**, which is
D14's hazard firing, in isolation, on one of the two dispatches of a block
that is faster overall. Four blocks, four answers, one rule, and this is the
first block that contains both answers.

## 5. The accuracy, and the family that costs the most per byte

| `full_attn` at `q4_k/32`, imatrix | baseline | measured | delta |
|---|---:|---:|---:|
| the simulation, 8 chunks (L8c-3) | 2.0189 | 2.0402 | **+1.05%** |
| **the bank**, 145 chunks | 4.0289 | **4.0954** | **+1.65%** |

which is a fourth reading of L8c-4's warning and the second in this direction.
The four families now measured both ways:

| family | 8-chunk screen | 145 chunks | error |
|---|---:|---:|---:|
| `lm_head` | +0.40% | +0.82% | +0.42 pp |
| `deltanet` | **−0.30%** | +0.93% | +1.23 pp, and the sign |
| `hyper_conn` | +0.90% | **+0.31%** | −0.59 pp |
| `full_attn` | +1.05% | **+1.65%** | +0.60 pp |

So a screen is wrong by ±0.6 pp on three families and by 1.23 pp and a sign
flip on the fourth, and the ranking it gives survives only at the top: it puts
`full_attn` first and `deltanet` last, where the corpus puts `full_attn`
first and `hyper_conn` last.

**And additivity is now exact enough to be surprising.** Four corpus deltas
summing to 3.71% measure **3.72%**:

| 145 chunks, against 4.0289 | ppl | delta |
|---|---:|---:|
| `lm_head` alone (L8c-4) | 4.0621 | +0.82% |
| `deltanet` alone (L8c-5) | 4.0665 | +0.93% |
| `hyper_conn` alone (L8c-6) | 4.0416 | +0.31% |
| `full_attn` alone | **4.0954** | **+1.65%** |
| **all four** | **4.1787** | **+3.72%** |
| their four deltas summed | — | 3.71% |

0.01 pp over four families, against a corpus standard error of ±0.024 on each
side. L8c-1 measured four *symmetric* families summing to 11.7% where the plan
measured 18.5% — a 1.58x gap — and every asymmetric plan since has been closer;
at four families the compounding on L6b-3's x1.085 a layer is not measurable at
all. That is the property the plan rests on, and it is the one thing an
eight-chunk screen still cannot do for itself: **the sum of corpus deltas
predicts the plan, and the sum of screens does not predict either.**

**The ranking this retires is L8c-1's, for the second time and in a new
shape.** In percentage points per GB of decode token bought back:

| family | GB/token saved | corpus cost | pp per GB |
|---|---:|---:|---:|
| `deltanet` | 1.073 | +0.93% | **0.87** |
| `hyper_conn` | 0.335 | +0.31% | **0.93** |
| `lm_head` | 0.317 | +0.82% | **2.59** |
| `full_attn` | **0.299** | **+1.65%** | **5.52** |

L8c-1 said the cost of 4 bits runs *inverse* to the bytes; L8c-6 retired that
and found the two large families almost exactly equal with the head an
outlier at 2.59. With four measured the shape is neither inverse nor flat but
**bimodal**, and the split is not size: the two families that are 36 layers
and 97 mixers cost ~0.9 pp/GB, and the two that are 12 layers and one matrix
cost 2.59 and 5.52. The full-attention layer is **6.3x** the DeltaNet per byte
bought back and is now, by a wide margin, the family a mixed plan should spend
its spare bits on — which is what L8c-3's mixed plan already proposed for a
different reason, and the first time this stage has measured a reason to
prefer one.

What the two expensive families have in common is not their width and not
their calibration — `output.weight` has no imatrix row at all and
`blk.N.attn_q` has one — so the honest statement is that the ranking is
measured and its mechanism is not. The reading that fits is that error in a
family present at 36 or 97 depths is partly re-absorbed by the residual
stream it is injected into, where the head's error **is** the logit and a
full-attention layer's is the only thing in the model the 36 linear layers
cannot reproduce. That is a hypothesis with four data points under it and it
should be treated as one.

## 6. `qsa_indexer`, and the two contexts it takes to grade it

L8c-1 found that **perplexity at `-c 2048` cannot grade the QSA indexer at
all**, and argued it from the selection's width: `top_k + ratio - 1` is 2051,
so at a 2048-cell cache the indexer names every cell, the attention kernel
takes its dense arm and the score it computed is discarded. That was an
argument. This stage measured it, and it is an **equality**:

| 145 chunks at n_ctx 2048 | ppl |
|---|---:|
| `full_attn` alone | 4.0954 |
| `full_attn` **and `qsa_indexer`** | **4.0954** |

— and not to four decimals. The two runs' per-chunk `nll` columns are
identical to **six** decimals on all 145 rows; the CSVs differ only in the
`seconds` column. Quantising the indexer's two matrices from BF16 to 4.5 bits
changes nothing at this context because nothing at this context reads them.
D13's last open clause — "the indexer's two BF16 projections rode along
untested, because nothing at `-c 2048` reads them" — is now tested and closed
with the answer it predicted.

**So the grading run has to be at a context where the selection bites**, and
the smallest one the corpus can be re-chunked into is where it lands: at
n_ctx 2560, `selWidth` is 2051 of 2560 cells, the selection is dispatched, and
the attention kernel takes its sparse arm. The corpus re-chunks into 116
chunks of 2560 scoring 1279 each, so this is a different measurement from
§5's and its own baseline is the run beside it rather than 4.0289:

| 116 chunks at n_ctx 2560, selection live | ppl | delta |
|---|---:|---:|
| `full_attn` at 4.5 bits | 4.0723 | — |
| **+ `qsa_indexer` at 4.5 bits** | **4.0725** | **+0.0002, +0.005%** |

**+0.005%, against a standard error of ±0.0231 on each side** — one part in
forty of what the corpus can resolve. And it is not that the indexer went
unread: all **116** per-chunk rows differ between the two runs, so the
quantised score really is selecting different cells; the cumulative nll
wanders between −0.0051 and +0.0000 across the corpus and lands at +0.00004.
The selection is live, the narrower indexer changes which cells it names, and
the corpus cannot tell.

So `qsa_indexer` at 4.5 bits is **0.028 GB a token for +0.005%** — by a wide
margin the cheapest thing this stage could buy, where `full_attn` beside it is
0.299 GB for +1.65%. The caution L4b-4 raised is not contradicted: it was
about an *unmodelled activation* changing the score by 24x, and a 4.5-bit
weight changes it by far less than the top-2051 boundary cares about. On the
bytes alone it is also what takes the layer from 5.010 bits a weight to
**4.500** and deletes the tail branch from the fused projection.

**One thing this could not be measured at, and it is worth writing down
because it is not about this stage.** The obvious grading context was 4096,
where the selection discards half the cache rather than a fifth. The
whole-model graph does not complete there:

| `-graph`, 48 layers, one staging for 4096 | |
|---|---|
| 2048 tokens | 1970.8 ms, 1039.2 tok/s |
| 2560 tokens | 2346.7 ms, 1090.9 tok/s |
| **3072 tokens** | **never returns** |
| **4096 tokens** | **never returns** |

with no `LLM_DENSE_BANK` set, on the default int8 bank, and with `-ppl` not
involved — so it is **pre-existing and not L8c-7's**. The 4-layer prefix runs
2560, 3072 and 4096 in 196/228/297 ms; `-ppl -ctx 4096` completes at 4 and 12
layers and hangs at 32 and 48; and the attention block *alone* at 4096 tokens
in a 4096-cell cache, twelve layers with the selection live, is **325 ms**.
So neither the selection nor the block is implicated: it is the whole graph,
above ~2560 rows, at a large layer count. A `SIGQUIT` puts the stalled
goroutine in `Graph.flush → recorder.submit`, in `[syscall]`, one thread at
100% of a core with the GPU at 2-3%, no disk I/O, and the shim's own 20-second
fence timeout never firing — which places it before the wait, inside the
driver's submit. The suspicion, untested, is that 82 GB of pinned buffers plus
the 28.8 GB mmap'd n-gram table plus the larger arenas exceed what the machine
can keep resident, and every submit re-validates. **It is the next thing
somebody should look at, because a long context is what this model is for.**



## 7. What this does not settle

**The text moved, and this is the stage where it was always going to.** At
temperature zero and on the same five-token prompt, four families write a
different completion from three — the divergence is at the eighth token, where
`Paris is a city in France. France is a country in Europe.` becomes
`Paris is a city. Therefore, what can be said about Paris?` — and both
continue coherently for the full 64. Twelve of the 48 layers changed what they
compute, so a re-worded sentence is the expected outcome rather than a
symptom; D17 and L8c's gate are written against §5's perplexity for exactly
this reason, and the text gate retired at L8c-4.

**One family is left, and it is the smallest thing in the model.**
`ple_proj` is the PLE block's fused key/value projection: 0.033 B parameters,
0.035 GB of a 6.334 GB token, run **once**, at layer 1. It is the one
remaining streamed dense family and the one block in this vertical that never
got L8a's int8 bank either — `PLEGPU` still stages halves — so it is two bank
stages in one and worth 0.017 GB a token. It has no obstacle in it: `ple_key`
and `ple_value` are both 2560 wide.

    cmd/gguf -width basis, the checkpoint's own widths
    a token   6.334 GB  as shipped
              6.016     lm_head at 4.5             (L8c-4)
              4.943     + deltanet at 4.5          (L8c-5)
              4.608     + hyper_conn at 4.5        (L8c-6)
              4.309     + full_attn at 4.5         (L8c-7, a 56.2 tok/s ceiling)
              4.281     + qsa_indexer at 4.5        (L8c-7 too, 56.5)
              4.264     + ple_proj at 4.5           (L8c-3's plan, 56.7)

Measured, this stage is 31.4 tok/s of that 56.2-56.5 — **56%**, against L8c-5's 59%
and L8c-6's 57% of their own ceilings — so the bank is not where the remaining
decode time is, and it has not been since L8c-5. What is left on the decode
side is the two things the simulation cannot reach at all: the **F32 router**
at fp16 (+1.5 tok/s of ceiling) and the **512 expert banks** at ~4.25 (+3.0),
which are 1.504 GB of every token and 35% of what a step now reads.
