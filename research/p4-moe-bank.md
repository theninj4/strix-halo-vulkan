# P4 — the router row, the 640 that has no K-quant, and what a transcode can reach

> 2026-09-19. P4 was written as two bullets: **"the F32 router at fp16,
> +1.5 tok/s of ceiling"** and **"the expert banks toward ~4.25 bits, +3.0 —
> transcode `ffn_down_exps`'s Q5_1 and the five Q8_0 layers to Q4_K with the
> imatrix; the existing `llm_moe_gemm/gemv` arms already read Q4_K, so this
> is a transcode and a corpus run, not a kernel."**
>
> **Neither is true, and both fail on a fact about the checkpoint rather than
> on a measurement.** The router has been staged as halves since L5b — the
> item was banked before it was proposed, and four budget tables were quoting
> the GGUF's F32 row instead. And `ffn_down_exps` is `[640, 2560, 512]`: a
> ggml K-quant super-block is 256 elements, 640 is two and a half of them, so
> **there is no Q4_K for that tensor at any accuracy**. That is why unsloth
> shipped it Q5_1 — not because the imatrix called the down projection
> sensitive, which is how `LLM.md`'s inventory read it.
>
> What is left is real and smaller: **three rows sitting at 8.5 bits with a
> format the kernels already read**. Transcoded, they are **0.1288 GB a
> token** of 4.408.

## 1. The router row was already fp16, and it moved four numbers

`ffn_gate_inp` is `[2560, 512]` F32 in the GGUF: 5.24 MB a layer, 0.252 GB
over 48, and `cmd/gguf`'s decode budget has quoted that since the inventory
was written. D3 turned it into a line item — "the routers are F32: 0.25 GB
read every single token, 4% of the decode budget for 0.06 B of parameters" —
and P4 carried it as +1.5 tok/s of ceiling.

But `MoEGPU.stage` has never staged it:

    router := make([]uint16, g.routerN()*c.NEmbd)
    tileB(router, w.Router, c.NExpert, c.NEmbd, ...)
    g.bank.WriteUint16At(int(g.layers[i].router), router)

`g.bank` is `len(layers) * routerN * NEmbd * 2` bytes, and the two router
kernels read no other copy. `routerN` is the 512 experts plus the shared
expert's gate as a 513th column, padded up to the plain GEMM's 64-wide
block — **576 columns, 2.95 MB a layer, 0.1416 GB a token**.
`TestMoERouterBankIsHalves` is the equality, and it logs both numbers so the
next reader does not have to find this twice.

Four consequences, in increasing order of interest.

**The item is void.** There is no +1.5 tok/s; it was spent in L5b.

**L5a's tie analysis is retired rather than owed.** P4 named it as the risk
of the move ("check the top-10 sets over a real prompt, not just ppl"). Every
perplexity number in this log, 4.0289 included, was measured with the router
already at fp16, so the risk was priced 5 000 chunks ago. L5a-1's own result
is the demonstration: 0 of 40 960 top-10 slots differ from llama.cpp's as a
*set*, on the fp16 bank.

**Every ceiling is 0.110 GB a token pessimistic.**

| quoted | row at F32 | row at fp16 |
|---|---:|---:|
| P1's attributed bank | 4.281 GB → 56.5 tok/s | **4.171 → 58.0** |
| D18 (P3) | 4.281 → 56.5 | **4.171 → 58.0** |
| D19 (P3a) | 4.518 → 53.6 | **4.408 → 54.9** |

**And P1's "the router reads above the bus" is an artefact of the same row.**
The attribution put `moe_router` at 0.252 GB in 0.764 ms — 329.8 GB/s — and
*excluded it from the loss column* under D16 on the grounds that no DRAM
measurement can exceed 242. On the right byte count it is **0.1416 GB in
0.764 ms, 185 GB/s** (or 203 against the 0.697 ms the streaming dispatch
alone takes, with `moe.router.sum` counted separately). That is an ordinary
streaming rate, beside `deltanet`'s 201.5 and `moe.up`'s 200.6 — the router
is a normal member of the weight-streaming set and had been filed as an
anomaly.

### 1a. The same error, one row over, and it is the one that cost a stage

P1's table splits the expert traffic as **gate+up 1.003 GB and down 0.501** —
a 2:1 split, which is the ratio of their *parameters*. They do not have the
same width. gate and up are Q4_K (Q5_K on layer 2) at 4.52 bits; `down` is
Q5_1 on 43 layers and Q8_0 on five, **6.26 bits**. The bytes are:

    gate + up   2 x 40.265 B params at 4.52 bits = 45.49 GB  x 10/512 = 0.8886 GB/tok
    down            40.265 B params at 6.26 bits = 31.51 GB  x 10/512 = 0.6154

which sum to the 1.504 the inventory reports, as the 1.003/0.501 pair also
does. Re-read against P1's own milliseconds:

| family | GB/tok | ms | GB/s as printed | GB/s corrected |
|---|---:|---:|---:|---:|
| moe experts, gate+up | 1.003 → **0.889** | 4.998 | 200.6 | **177.8** |
| moe experts, down | 0.501 → **0.615** | 3.399 | 147.5 | **181.0** |
| moe_router | 0.252 → **0.142** | 0.764 | 329.8 | **185.3** |

**`moe.down` was never the outlier.** Its 147.5 GB/s — "60.9% of the bus,
+1.19 ms lost", the largest single item on P1's list after `hyper_conn` — is
arithmetic, not silicon: at the true byte count it reads at 181, *faster*
than the gate/up pair off the same bank. P1b spent a stage on it, re-screened
sixteen cold banks at one iteration, found that **no rung moves on any axis**,
and concluded "the 1.19 ms was the MALL's arithmetic". The null result was
right and its explanation was not: there was no 1.19 ms, because there was no
147.5 GB/s.

The lesson generalises past this table. **D16 says a rate above 242 GB/s was
not measuring DRAM; the rate is a quotient, and the numerator is as capable
of being wrong as the denominator.** Three of the nine rows in P1's table had
a byte count from a different source than the bank they were timing — two
from the checkpoint's widths and one from a parameter split — and each of
them landed on a rate that was read as a finding.

## 2. 640 is not a multiple of 256

The expert half of P4 asked for `ffn_down_exps` at Q4_K. Its dims are
`[640, 2560, 512]`: the reduction axis, the one a scale group and an
importance row both run along, is **640**.

A ggml K-quant's super-block is `QK_K = 256` elements. Q4_K, Q5_K, Q6_K,
IQ4_XS — every format whose name ends in `_K` or begins `IQ` and blocks by
256 — can only store a row that divides 256. 640 does not: it is 2.5
super-blocks. **The format does not exist for this tensor**, and no imatrix,
no kernel and no accuracy argument changes that.

`TestMoEBankTranscodeRefusesAnImpossibleFormat` is the guard, and
`cmd/gguf` now carries the fact in the inventory itself — a `ROW` column that
says `k-quant` or `block-32 (640)` per group, and a note on any re-pricing
that asks a checkpoint-byte group for a width only a K-quant can meet:

    GROUP            PARAMS B  GB     BITS/W  READ       ROW                  QUANT MIX (GB)
    moe_experts      120.796   77.02  5.10    10/512     block-32 (640)       Q4_K:44.4 Q5_1:27.1 Q8_0:4.5 Q5_K:1.2
    ngram_ple_table   51.200   28.80  4.50    gather     block-32 (160)       IQ4_NL:28.8
    ...
    re-priced: experts 4.50 bits
    moe_experts  4.50  67.95  1.327  1.504   <- no such format: rows 640 are not multiples of 256

The note is restricted to `moe_experts` and `moe_shared` on purpose. Those
two are the groups whose device bank is **the checkpoint's own bytes**, read
by a shader that implements a ggml format. Every other streamed family goes
through `bank_q4.go`, which is not ggml's layout — its own fragment tiling,
and a super-block that is the *whole row* for the 320-wide hyper-connection
family — so the same note would be wrong about it. That distinction is the
one P4 needed and no table had.

### What this says about the checkpoint

`LLM.md`'s inventory reads unsloth's choice as a statement about sensitivity:

> **`ffn_down_exps` is Q5_1 on 43 of 48 layers and Q8_0 on the other 5**
> while gate/up are Q4_K — that is unsloth's imatrix telling them the down
> projection is the sensitive one

`llama-quantize` has a function for exactly this, and its table names the
two types this checkpoint's down projection is in.
`src/llama-quant.cpp`'s `tensor_type_fallback` (read at `d1d3c3396`, the
checkout's head) warns `ncols not divisible by 256 (required for type ...)`
and demotes:

    case GGML_TYPE_Q4_K:    return_type = GGML_TYPE_Q5_0;   break;
    case GGML_TYPE_Q5_K:    return_type = GGML_TYPE_Q5_1;   break;
    case GGML_TYPE_Q6_K:    return_type = GGML_TYPE_Q8_0;   break;

**`Q5_K → Q5_1` on 43 layers and `Q6_K → Q8_0` on five.** That is the whole
of unsloth's "choice": the UD-Q4_K_XL recipe asked for a K-quant on the down
projection, the row is 640, and ggml handed back the block-32 type on the
right of that table. So the observation is **a fact about the row length,
not about the imatrix**, and
the two readings predict opposite things: the sensitivity reading says
narrowing `down` will be expensive, the format reading says nobody has
measured it. Nobody has. It is now a kernel question — a block-32 format at
4.5–5 bits that the down mode has no arm for — and it is the largest single
row in the MoE budget at **0.615 GB a token**. Written down in `LLM2.md` as
P4c rather than built here.

## 3. What a transcode can actually reach

Three rows in this block sit at Q8_0 — 8.5 bits — and every one of them has a
ggml format the MoE kernels already build:

| row | reduction axis | ships | to | GB/token saved | arm |
|---|---:|---|---|---:|---|
| `ffn_gate_shexp` | 2560 | Q8_0 | Q4_K | 0.0393 | `up_q4k_*`, exists |
| `ffn_up_shexp` | 2560 | Q8_0 | Q4_K | 0.0393 | `up_q4k_*`, exists |
| `ffn_down_shexp` | 640 | Q8_0 | Q5_1 | 0.0246 | `down_q51_*`, exists |
| `ffn_down_exps`, 5 layers of 48 | 640 | Q8_0 | Q5_1 | 0.0256 | `down_q51_*`, exists |
| | | | **total** | **0.1288** | |

Three of the four come off the **shared expert**, which runs for every token
and which P1 measured at **136.5 GB/s** — the second-slowest family on the
board after `hyper_conn`. So the time those bytes buy is worth more than
their share: 0.1032 GB at 136.5 GB/s is 0.756 ms, and the `down_exps` row is
0.0256 at 181 for another 0.141. **0.90 ms of a 28.9 ms step — a predicted
+1.11 tok/s**, against D19's 34.57.

### The build

`llm/moebank.go`, and it is a transcode and a corpus run.

**The plan** is `LLM_MOE_BANK=gate_shexp=q4_k,...` with
`LLM_MOE_BANK_QUANT` beside it — the same shape, the same two arms and the
same `imatrix` default as `LLM_DENSE_BANK`, and the same refusal to be a
default in `cmd/llm`. One rule is new: **a family names a ceiling**, so a
tensor already at or below that width is left bit-for-bit alone. That is what
makes `down_exps=q5_1` mean "the five Q8_0 layers" without naming them, and
`TestMoEBankPlanIsACeiling` is the check.

**The fit is `asymEnc`** — the same encoder `bank_q4.go` packs into *our*
dense layout, so the levels a transcoded expert carries and the levels L8c-4's
bank carries come from one function and a difference between the two banks
can never be a difference of quantiser. The **packing** is ggml's, inverted
from `gguf.dequantQ4_K`, and the first sixteen bytes of a ggml K-quant block
are `bank_q4.go`'s record exactly, so `packQ4KRecord` is shared too.

**Q5_1 is fitted with the imatrix, which ggml does not do.**
`quantize_row_q5_1_ref` is plain min/max, because the legacy quants predate
the importance matrix and were never wired to it. Both sides here are ours,
`makeQkxQuants` fits a (scale, min) pair against a per-element weight for a
block of any length, and a Q5_1 block is just a 32-element one — so the
calibrated arm uses it and `rtn` is ggml's own path. Both dequantise
identically; only the levels differ. One asymmetry is stated rather than
left to be discovered: `makeQkxQuants` clamps the min to zero, which is the
K-quant convention, where `quantize_row_q5_1_ref` keeps the block's true
min — so on an all-one-sign block the calibrated arm gives up a level of
range that the search then has to earn back.

**The imatrix's expert rows.** `imatrix.go` said expert tensors "are not read
here: L5b stages the expert banks byte for byte out of the checkpoint, so the
simulation never sees them", and refused a `[k, nExpert]` entry rather than
picking a row. `ExpertColumns` is that lookup: `in_sum2` row e over
`counts[e]`, falling back to uniform where the calibration never routed to an
expert, which is what an absent entry already does one level up.

### The gates

**The format is the fit.** Every packer is checked by packing a row, reading
it back through `gguf.Dequantize` — the transcription of `dequantize_row_*`
the shaders were written against — and demanding it equal what the encoder
says it stored. For Q4_K and Q5_K that is an **equality over every element**
(`TestPackKRowIsGgmlsLayout`); for Q5_1 it is the same idea expressed as
level membership, because the dequantiser is the only other implementation
(`TestPackQ51RowIsGgmlsLayout`), plus an equality against
`quantize_row_q5_1_ref`'s own (d, m) on the uncalibrated arm.

**The staging is indistinguishable from a checkpoint that shipped the
narrower format.** `TestMoEBankTranscodeStagesLikeTheCheckpoint` stages layer
3 twice — once letting the plan narrow the shared expert, once with the plan
off over tensors whose bytes were transcoded ahead of time — and the block's
outputs are **bit-identical**: 115 343 360 routed `ffn_moe_weighted` values,
2 621 440 shared `ffn_swiglu` values, 10 485 760 `ffn_out` values, zero
differences. Anything the staging path does differently for a narrowed tensor
— an offset, the sixteen-byte alignment, a `moeFmt`, a pipeline — shows up
there and nowhere else, because a whole-model run would read it as an
accuracy cost.

One thing that is *not* an observable, found while writing that test:
`MoEGPU.Swiglu()` past the shared expert's rows includes tile padding that
no down tile reads, and it differs between two stagings of byte-identical
banks. `Weighted()` and `Out()` are the defined outputs; the padding is
inert, and `Weighted()` being identical over 115 M values is what proves it.

## 4. The measurements

### Perplexity: 0.017 pp/GB, and the instrument cannot see it

145 chunks of `wiki.test.raw` at n_ctx 2048, on D19's dense bank, against the
same corpus and chunking. `results/p4_ppl_all.csv`.

| bank | PPL | vs 4.0289 | GB/token |
|---|---:|---:|---:|
| D19 | 4.0948 ± 0.02322 | +1.63% | 4.408 |
| **D19 + P4b** | **4.0970 ± 0.02325** | **+1.69%** | **4.279** |

**0.0022 points for 0.1288 GB a token — 0.017 pp/GB.** D19 refused to *buy*
`deltanet`'s fifth bit at 1.6 pp/GB and took three families above 5.9, so
selling bytes at 0.017 is two orders of magnitude inside the line already
drawn. There is no judgement in this decision.

The delta is also **not resolvable**, and that is worth saying precisely
rather than hiding behind the ±0.023 standard error, which is the error of
the *absolute* number. Paired chunk for chunk, the same 145 chunks through
both banks:

    mean nll delta  +0.000540 a chunk
    sd               0.007141,  stderr 0.000593
    t                0.91
    sign             worse in 85 chunks, better in 60, identical in 0

So the honest statement is **an upper bound**: whatever this costs, it is
under ~0.002 nll a chunk, and the sign is not established.

P4's gate asked for "145-chunk ppl for each step separately". That was
written expecting a cost worth attributing between the four rows. There is
none to attribute, and P3's rule — **a plan is measured, never composed** —
makes the complete plan the decision regardless, so the three extra corpus
runs were not spent. A per-row run is 9 minutes each if a later stage needs
the split.

### Decode: +0.99 tok/s, three interleaved pairs, one binary, one hour

`-gen -n 64 -prompt 'The capital of France is'`, D19's dense bank on both
arms, `LLM_MOE_BANK` the only thing that changes.

| pass | A: D19 | B: D19 + P4b |
|---|---:|---:|
| 1 | 34.48 | 35.52 |
| 2 | 34.49 | 35.52 |
| 3 | 34.62 | 35.53 |
| **mean** | **34.53** | **35.52** |
| within-arm spread | 0.14 | 0.01 |

**+0.99 tok/s, +2.9%, the same sign in every pair and a separation seven
times the wider arm's spread.** 1.37x llama.cpp's 25.15 becomes **1.41x**.

The prediction was +1.11, from 0.1032 GB at the shared expert's measured
136.5 GB/s plus 0.0256 at `moe.down`'s corrected 181 — 0.898 ms of a 28.9 ms
step. Measured 0.807 ms — **the bytes over-predict by 11%**. The rung
re-screen in §5 accounts for a tenth of that gap and no more, so the rest is
the same shape-not-bytes residue P1 named: a dispatch's time is not a
straight function of what it streams.

### Residency, and what staging costs

The MoE block stages **76.00 GB against 77.41**, and the 1.414 GB is the
arithmetic to three decimals:

    5 x ffn_down_exps   Q8_0 -> Q5_1   5 x 262.2 MB = 1311.0 MB
    48 x gate + up shexp Q8_0 -> Q4_K  48 x 1.638   =   78.6
    48 x down_shexp      Q8_0 -> Q5_1  48 x 0.512   =   24.6
                                                      -------
                                                      1414.2 MB

Whole-model residency is **78.53 GB**. Staging the MoE block goes from ~25 s
to **51.6 s** — the transcode is 4.2 B weights of `ffn_down_exps` through a
32-element min/max fit and 236 M through the K-quant scale search, in
parallel over rows. It is paid once per process and nothing about a served
run touches it again.

## 5. What this leaves

- **P4c**, and it is the whole remaining MoE prize: `ffn_down_exps` at
  0.615 GB a token in a block-32 format at 4.5–5 bits. Q4_1 (5.0 bits,
  −0.084 GB/tok), Q4_0 (4.5, −0.125) or IQ4_NL (4.5, −0.125, and the
  calibrated non-linear form llama.cpp itself reaches for when a row will not
  take a K-quant). All three need a `down_` arm the kernel does not have, so
  this is a kernel stage, not a transcode.
- **The shared expert's rungs, re-screened — and D12's law is confirmed
  rather than broken.** Two of the three shared matrices changed format, so
  D12 ("the rung to pick moves when the weight's width does") obliged a
  re-screen: `-moe -tokens 1 -ladder -layers 16 -iters 1` on the transcoded
  bank, P1b's honest-floor form, twice. Two runs agree to a **median ratio
  of 1.0009 (p10-p90 0.986-1.020) over 993 rows**, and every rate is under
  242 GB/s so D16's screen passes. `results/p4_moe_shexp.csv`.

      shexp rung     up us (r1/r2)    down us (r1/r2)
      v64w4/v16w4    13.88 / 13.88    8.19 / 8.20   <- best
      v64w4/v32w4    13.95 / 13.97    8.46 / 8.49   <- the incumbent
      v64w4/v64w4    14.02 / 14.32    8.92 / 9.24
      v32w4/v32w4    17.84 / 17.77
      v16w4/v32w4    25.29 / 25.53

  **The up rung does not move** — v64w4 by 1.28x over the next rung, on
  Q4_K as it was on Q8_0 — and **the down rung moves to exactly the routed
  down's**, v32w4 → v16w4, which is the law working: `ffn_down_shexp` went
  Q8_0 (170 payload dwords a row) to Q5_1 (120), and the routed down has
  been Q5_1 at 120 all along. Two matrices in the same format want the same
  lanes-per-row.

  **Priced and not taken**: 0.27 us a layer is **0.013 ms a token, +0.016
  tok/s**, an order of magnitude under the instrument. Taking it would make
  `MoESharedPlanFor` — today a function of the token count alone — a
  function of whether the bank was transcoded, which is real complexity for
  a number that cannot be measured end to end.
- **The router's padding**, priced and not taken: 576 columns where 513 are
  real is 12% of what the router streams, **0.0118 GB a token**, about
  +0.09 tok/s. It is `roundUpInt(NExpert+1, attnBN)` and the plain GEMM's
  column block is what sets it.
