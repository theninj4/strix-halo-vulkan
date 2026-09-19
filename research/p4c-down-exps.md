# P4c — the down projection below six bits, and a premise that was wrong twice

> 2026-09-19. `ffn_down_exps` is **0.615 GB a token, 41% of the expert traffic
> and 14% of the whole decode bank**, and it sits at 6.26 bits for one reason:
> its row is 640, a ggml K-quant's super-block is 256, and `llama-quantize`
> had nothing narrower to offer it (P4). So it was the largest single row left
> on the throughput frontier and the only one whose accuracy cost **nobody had
> measured** — the sensitivity reading of unsloth's choice predicts it is
> expensive, the format reading predicts nothing at all.
>
> It is a kernel stage, not a transcode: two block-32 formats the down mode
> had no arm for, in both grouped kernels.

## 1. The format question, and the one that was not built

Three candidates block by 32, so the 640 row divides all of them:

| format | bytes / 32 | bits | GB a token | saved | what it is |
|---|---:|---:|---:|---:|---|
| Q5_1 (D20) | 24 | 6.0 | 0.590 | — | what the row is today, on all 48 layers |
| Q4_1 | 20 | 5.0 | 0.492 | 0.098 | fp16 scale, fp16 min, 16 nibble pairs — `packQ51Row` with the fifth-bit plane deleted |
| IQ4_NL | 18 | 4.5 | 0.442 | **0.148** | fp16 scale, nibbles indexing a **non-linear codebook**; the form llama.cpp itself falls back to |
| Q4_0 | 18 | 4.5 | 0.442 | 0.148 | the same bytes as IQ4_NL with evenly spaced levels |

**Q4_0 was not built**, and that is a decision rather than an omission: it is
IQ4_NL's bytes exactly, and this vertical has measured the symmetric linear
form against an asymmetric one at the same width twice — D4, and L8c-1's
**+18.5%** of perplexity. A row that can be IQ4_NL has no reason to be Q4_0,
so the arm would be a pipeline nothing should ever choose.

## 2. The premise, checked against the code, and wrong again

P4's own lesson was that "the routers are F32" and "the imatrix says the down
projection is sensitive" were both readings of the **checkpoint** carried as
facts about the **bank**. Building P4c turned up a third of the same kind, and
this one is P4b's own:

> **Q5_1 is fitted with the imatrix, which ggml does not do.**
> `quantize_row_q5_1_ref` is plain min/max, because the legacy quants predate
> the importance matrix and were never wired to it.

`quantize_row_q5_1_ref` is indeed plain min/max. It is not what
`llama-quantize` calls. `ggml_quantize_chunk` dispatches `GGML_TYPE_Q5_1` to
`quantize_q5_1`, which with a matrix calls `quantize_row_q5_1_impl` — and that
function fits `make_qkx3_quants(32, 31, ..., -0.9f, 0.05f, 36, false)` against
`qw[j]*sqrtf(sigma2 + x*x)`. Those are **our three constants**, because
`makeQkxQuants` is a port of the same function.

So P4b's arm was never "ours against none". It was ours against ggml's, and
the two differ in exactly two places:

- **`sigma2` is the block's here and the row's there.** `quantize_row_q4_1_impl`
  and `quantize_row_q5_1_impl` both take `sum(x*x)/n_per_row` over the whole
  row; `packQ51Row` took a K-quant's `2*sum/32` over the block.
- **The levels are chosen against the stored halves here** and against the f32
  pair there (D10).

P4b's bytes are left exactly as they were measured — D20 is a shipped bank and
its perplexity is a measurement of those bytes — but P4c's two packers take
ggml's normalisation, which is what makes the gate below possible at all.

## 3. The gate: the packers are ggml's, to the level

`reference/quant_ref.c` is generic over ggml types, so P4c's oracle is the
binary L8c-2 already built, over its own fixture: 64 rows of
`blk.3.ffn_down_exps.weight` and expert 0's row of unsloth's imatrix
(`llm/testdata/moepack_{in,ref}.bin`, `TestMoEPackersAgainstGGML`).

| arm | values differing from ggml's | weighted sq error, ours / ggml's |
|---|---:|---:|
| `iq4_nl` / rtn | **0 of 40 960** | 1.0000 |
| `iq4_nl` / imatrix | 2 of 40 960 | 1.0000 |
| `q4_1` / imatrix | 99 of 40 960 (0.24%) | **0.9982** |
| `q4_1` / rtn | 3074 of 40 960 (7.5%) | 1.0000 |

Two things fall out of that table.

**The IQ4_NL transcription is exact.** Its uncalibrated arm reproduces
`quantize_iq4_nl(..., NULL)` bit for bit over 40 960 values — which is the
statement the codebook needs, because unlike an affine fit there is no
(scale, min) to agree on: the search *is* the format.

**The one departure pays, slightly, and is now measured rather than
asserted.** Choosing levels against the stored half instead of the f32 pair
moves 0.24% of Q4_1's values and lowers the weighted error by 0.18%. The test
asserts the direction rather than the size: if the departure ever stops
paying, it should be dropped and the packer made bit-exact.

**And `rtn` is not the same thing in the two formats.** `quantize_q4_1` falls
back to `quantize_row_q4_1_ref` when there is no matrix — min/max, levels from
`(int8_t)(x + 0.5f)` — so 7.5% of values land one level from ours and the
weighted error is identical to four figures. `quantize_iq4_nl` does **not**
fall back: it runs the same sixteen-scale search with `x*x` weights, which is
why the uncalibrated IQ4_NL arm can be exact where the Q4_1 one cannot.

## 4. The kernels

`llm_moe_gemm.comp` and `llm_moe_gemv.comp` are parameterised by `QFMT`, so
each format is an unpack and a `-D` — but it is still a kernel stage, because
the unpack is where a format lives:

- **Q4_1 asks nothing of the addressing.** Twenty bytes a block keeps every
  block four-aligned under a row that is (640 elements is 400 bytes), and the
  nibble pairs split 0..15 / 16..31 exactly as Q5_1's do. It is the Q5_1 arm
  with the `qh` load and its two shifts deleted.
- **IQ4_NL asks for everything.** Eighteen bytes a block puts every other
  block at an offset of 2 mod 4, so a payload dword comes out of Q8_0's
  two-word window; and every nibble is an index into a sixteen-level table
  rather than a level. Written the obvious way — ggml's record layout and the
  codebook packed into a `uvec4` — **the 4.5-bit kernel was slower than the
  6.0-bit one it replaced.** §7 is what that took to fix, and the answer was
  a **layout of our own**, which is a thing a transcoded tensor may have and
  a staged one may not.

Eleven builds a format: five GEMM rungs (`m1`, `m2`, `m4`, `w2m1`, `w4m1`) and
six GEMV ones (`v16`, `v32`, `v64` and their four-wave twins).

## 5. What the gates pin

Three equalities and one tolerance, in the order they catch things:

- **The staging is indistinguishable from a checkpoint that shipped the
  format.** `TestMoEBankTranscodeStagesLikeTheCheckpoint` now runs P4b's plan
  and both of P4c's; the P4c arms narrow a **routed** tensor — three
  dimensions, 0.5 GB a layer, read by the down mode rather than the shared
  expert's — so the offset, the row stride and the pipeline are a different
  path through the same code. All three arms are bit-identical over
  115 343 360 routed values, 2 621 440 shared ones and 10 485 760 of `ffn_out`.
- **Every rung agrees exactly.** A row block changes how many tiles the
  schedule holds and nothing else, so four rung pairs on one narrowed bank
  produce identical output to the last bit — which is what says the block base
  is computed the same way at every BM.
- **The two unpacks agree.** The GEMM's slab into LDS and the GEMV's dwords
  into registers are independent implementations of the same table, and over
  six decode rungs they agree to **rms 2.6e-06**, which is the fp16 rounding
  the GEMM does on its way into LDS (L8d) and nothing else.
- **The output is a quantisation and not a misread.** Against llama.cpp's own
  `ffn_out-3` over 4096 tokens, the narrowed banks are **5.47% (q4_1)** and
  **6.09% (iq4_nl)** of the tensor's own rms, where the shipped bank is 0.63%
  — that 0.63% being a *kernel* rounding over identical weights, which is the
  wrong scale to bound a quantisation against. A block read at the wrong
  offset does not land at a few percent of the signal.

The two formats being **11% apart** on that measure, for half a bit of width,
is the non-linear codebook earning its name.

## 6. The measurements

### Perplexity, and the format that wins is the narrow one

145 chunks of `wiki.test.raw` at n_ctx 2048, D19's dense bank and D20's MoE
plan on every arm, `down_exps` the only thing that changes.
`results/p4c_ppl_{q4_1,iq4_nl}.csv` against `results/p4_ppl_all.csv`.

| bank | bits on the row | PPL | vs 4.0289 | GB/token | ceiling |
|---|---:|---:|---:|---:|---:|
| D19 + D20 | 6.0 | 4.0970 | +1.69% | 4.279 | 56.6 |
| + `down_exps=q4_1` | 5.0 | 4.1092 | +1.99% | 4.181 | 57.9 |
| + `down_exps=iq4_nl` | **4.5** | **4.0992** | **+1.74%** | **4.132** | **58.6** |

**The 4.5-bit format is better than the 5.0-bit one on both axes at once.** It
frees half again as many bytes — 0.148 GB a token against 0.098 — and costs a
fifth as much accuracy: +0.0022 points against +0.0122. There is no trade to
make between them; Q4_1 is dominated, and the only reason it was built is that
nothing said in advance which of the two the row would prefer.

Paired chunk for chunk against the same control, which is where the two
separate properly:

| arm | mean nll delta a chunk | stderr | t | worse / better |
|---|---:|---:|---:|---:|
| `q4_1` | +0.002978 | 0.000724 | **4.11** | 90 / 55 |
| `iq4_nl` | +0.000536 | 0.000678 | **0.79** | 84 / 61 |

So P4c's expectation — "the first MoE row where the cost could plausibly be
real" — is right about one of the two formats and wrong about the other.
Q4_1's cost **is** resolvable by this instrument, at four standard errors;
IQ4_NL's is the same sub-noise result D20 got (t = 0.91 there), at 4.5 bits
on the largest row in the block.

**And the price is 0.34 pp/GB**, against a frontier that bought three
families at 5.9 and above and refused `deltanet` at 1.6. Q4_1's is 3.05 —
inside the band D19's decisions bracket, which is the second reason not to
take it.

### The instrument that got the order wrong

The two formats were also compared *before* the corpus, two other ways, and
**both ranked them the other way round**:

| measure | q4_1 | iq4_nl | ranks |
|---|---:|---:|---|
| imatrix-weighted squared error, 64 real rows (§3) | 3.50e-04 | 4.33e-04 | q4_1 by 1.24x |
| `ffn_out` against llama.cpp, 4096 tokens (§5) | 5.47% | 6.09% | q4_1 by 1.11x |
| **wikitext-2, 145 chunks** | **+0.0122** | **+0.0022** | **iq4_nl by 5.5x** |

This is D3's lesson a third time, and the sharpest instance of it yet. L8c-1
found that reconstruction error does not carry to perplexity; L8c-3 found it
does not even rank two *forms* correctly, the gap being 1.22-1.30x by
reconstruction where it is 2.06-3.69x by corpus. Here the two instruments do
not merely differ in magnitude — **they disagree about which format is
better**, over the same weights, the same importance matrix and the same
kernel. A reconstruction number is not a weak version of a corpus number.

### Decode: +0.73 tok/s, and prefill +1.5% beside it

`-gen -n 64 -prompt 'The capital of France is'` and `-graph -tokens 512`,
D19's dense bank and D20's plan on both arms, `down_exps` the only thing that
changes, interleaved on one binary within one hour (P1c) with nothing else on
the machine.

| | D20 (`q5_1`) | D21 (`iq4_nl`) |
|---|---:|---:|
| decode, three passes | 35.44 / 35.51 / 35.44 | 36.16 / 36.13 / 36.27 |
| **mean** | **35.46** | **36.19** (+0.73, +2.1%) |
| within-arm spread | 0.07 | 0.14 |
| prefill at 512, two passes | 663.1 / 665.3 | 674.1 / 673.6 |
| **mean** | **664.2** | **673.9** (+1.5%) |

**35.46 → 36.19 is 1.41x → 1.44x llama.cpp's 25.15**, at 4.132 GB a token
against 4.279 and a ceiling of 58.6 tok/s. The ladder predicted +0.71 from
11.8 us a layer over 48 layers; the step measured +0.73. Prefill gains too,
by more than the `down` dispatch alone explains — that dispatch is the same
2679 us a layer as Q5_1's on three quarters of the bytes, so the rest is what
7.1 GB less resident bank does to everything else in the pass.

Two earlier decode blocks are worth keeping, because they are what sent this
stage into the kernel:

| arm | mean | against its own control |
|---|---:|---|
| `q4_1` | 36.05 | 35.45 (+0.60) |
| `iq4_nl`, ggml's record layout | 35.31 | 35.45 (**−0.14**) |
| `iq4_nl`, planar rows | 36.23 | 35.31 † (+0.92) |
| **`iq4_nl`, 36-byte pairs** | **36.19** | **35.46 (+0.73)** |

† one control pass of that block read 33.84 because this session was compiling
on the same machine — the rule P1c wrote down, broken by the person following
it. It is reported rather than dropped, and it is why the final block above
was run with nothing else running.

### The rungs, and D12 discharged

`-moe -tokens 1 -ladder -layers 16 -iters 1` — P1b's honest form, sixteen cold
banks at one iteration, so `ProfileSweep` never re-reads a bank warm and no
rate can come back above the 242 GB/s bus (D16). D12 is why it has to be run
at all: the rung to pick moves when the weight's width does.

**It does not move.** `v16w4` wins the down mode on all three formats, as it
has since D20 put the routed and shared down projections on the same 120
payload dwords a row — and both of P4c's formats keep the same **80 payload
dwords** at 640 elements, which is the quantity the lane group is sized to.
The row's *bytes* change; the thing the ladder is about does not.

`results/p4c_moe_{q5_1,q4_1,iq4_nl}_r{1,2}.csv`. The `iq4_nl` files are the
**final** kernel's — §7's two changes were each screened on the same
instrument and only the last arrangement is committed, since a ladder CSV is
a measurement of the kernel in the tree.

## 7. The kernel that was slower at three quarters of the bytes

The `down` dispatch alone, at its best rung, over sixteen cold banks at one
token — and beside it the same dispatch at 512 tokens, where it is the GEMM
rather than the GEMV and the unpack rather than the bus that binds:

| bank / arrangement | bytes a layer | decode us | decode GB/s | prefill us |
|---|---:|---:|---:|---:|
| Q5_1 (D20) | 12.29 MB | 56.94 / 56.85 | 215.8 | 2678.7 |
| Q4_1 | 10.24 | 48.45 / 48.43 | 211.4 | 2537.0 |
| **IQ4_NL, ggml's records** | **9.22** | **60.36** | **152.7** | — |
| IQ4_NL, codebook in LDS | 9.22 | 51.15 | 180.2 | — |
| IQ4_NL, planar rows | 9.22 | 45.34 / 44.97 | 203.3 | **3636.9** |
| **IQ4_NL, 36-byte pairs** | **9.22** | **45.11 / 45.00** | **204.3** | **2679.3** |

(the decode column is two runs where the kernel survived to be re-measured,
agreeing to 0.2-0.8%; the prefill column is the `down` dispatch at 512 tokens,
two layers, `-moe -tokens 512`)

The third row is the whole point of the section: **10% fewer bytes than Q4_1
and 25% fewer than the format it replaced, and slower than both**, because it
read its bank at 152.7 GB/s where they read theirs at 211-216. The decode
measurement agreed before any of this was known — three interleaved passes put
it at 35.31 tok/s against the incumbent's 35.45, giving back everything
0.147 GB a token had bought.

Two things were wrong, and neither is the format.

**The codebook was ALU in the hottest loop.** A nibble's level is needed eight
times per payload dword and the index is divergent, so reading it out of a
packed `uvec4` costs a `cndmask` chain for the component, a bitfield extract
and a convert — about five instructions where Q4_1 needs one convert. Sixteen
lanes now widen the table into **LDS** once per workgroup, under the barrier
the A staging already does: 64 bytes beside the 5 KB of A. With the block's
scale hoisted out of the eight terms — legal here and not for an affine
format, because there is no per-element min to add — that is
**152.7 → 180.2 GB/s**.

**And ggml's record layout was never a constraint.** The MoE bank is staged as
*the checkpoint's own bytes*, which is what makes a ggml record's shape a fact
the kernel has to live with. A **transcoded** tensor is not the checkpoint's
bytes — we write them — and eighteen is not a multiple of four.

### The layout the two kernels disagreed about

The obvious rearrangement is **planar**: every scale, then every block's
nibbles. It is worth **180.2 → 203.3 GB/s** at decode — and it costs the
*prefill* GEMM **36%**, taking the same dispatch from 2673 to 3637 us a layer
and the whole 512-token graph from 663.6 tok/s to 630.5. Two passes each, and
the regression was found only because the graph was measured at all — a decode
A/B would have shipped it.

The two kernels read the same bytes in different shapes:

- a **GEMV lane group walks one row**, so a plane of scales is read once per
  four payload dwords and stays in cache;
- a **GEMM slab unpack reads one 32-element block of sixty-four different
  rows** per K-step, so a scale 300 bytes from its own nibbles doubles the
  cache lines the step opens.

**36-byte pairs satisfy both.** Two IQ4_NL blocks are 4 bytes of scales and 32
of nibbles — a multiple of four, so every access is aligned, and a block's
scale is in the same 36 bytes as its payload. Decode reads it at **204.3
GB/s** (planar's rate, within the instrument — two runs, 45.11 and 45.00 us)
and prefill at **2679 us —
exactly the Q5_1 bank's 2678.7**, on three quarters of the bytes. Both
arrangements produce **identical output**: the device's `ffn_out` over 4096
tokens is the same to the bit through all three layouts, because a layout is a
permutation.

It stays a permutation and is checked as one: `packIQ4NLRow` still writes
ggml's records and is still bit-exact against `quantize_iq4_nl`;
`pairIQ4NLRow` rearranges them and `TestIQ4NLPairIsAPermutation` inverts it
byte for byte. The device's `ffn_out` over 4096 tokens is the same to the last
bit through all three arrangements, which is what a layout change should be.

**The rules this leaves.** *The checkpoint's layout is a constraint only on
the tensors the bank takes verbatim* — every format P4b transcoded happened to
be word-aligned, so the question never came up, and the first one that is not
was worth 28% of the decode read rate. And *a layout is measured on both
kernels that read it*: one arrangement is 1.13x at decode and 0.74x at
prefill, and the number that decided the format would have been the wrong one
either way round.

## 8. Serving it: a transcode is eight minutes and a pure function

`down_exps=iq4_nl` re-fits **40 billion weights** through a sixteen-scale
search — 9.7 s a layer over 48 layers, about **eight minutes** of a 32-core
machine. A measurement run can pay that. A server that otherwise stages in
34 seconds cannot, and shipping D21 without answering this would have been
shipping a fifteenfold regression in startup to buy 2.6% of decode.

The transcode is a pure function of (tensor, format, arm), so `cmd/serve`
points `LLM_BANK_CACHE` at a `bank-cache/` directory beside the checkpoint and
pays it once ever. Measured, `cmd/serve -llm`:

| start | staging | what happened |
|---|---:|---|
| first | **8m50s** | fits every layer and writes 22 GB |
| second | **1m5.7s** | reads them |
| D20, for comparison | ~34 s | four small rows, no cache |

The cache key is the tensor's name and shape, the type it ships in, the type
it is written as, the arm, and a version constant — everything the bytes
depend on except the source bytes themselves, because hashing 30 GB to save
eight minutes of arithmetic gives most of the saving back. It is scoped to a
checkpoint by living inside its directory; a checkpoint edited in place under
a stable name is the case it cannot see, which is the assumption `mmap`
already makes about that file. `TestMoEBankCacheIsTheTranscode` checks the
property that matters — a hit is byte-for-byte the transcode — with a
negative control (the `rtn` arm's bytes differ, so a stale hit would show) and
a truncated file, which must be a miss rather than a wrong answer.

## 9. What this leaves

- **The last 7 GB/s of the decode kernel.** 204.3 against Q4_1's 211.4, and
  what is left is the codebook's LDS reads — eight per payload dword, on a
  sixteen-entry table that 64 lanes index divergently. Worth ~2 us a layer,
  0.1 ms a token.
- **The prefill GEMM's 140 us.** At 512 tokens the down dispatch is 2679 us
  against Q4_1's 2537 on 10% *more* bytes, so what is left there is the
  unpack — a codebook lookup and a multiply per element against an affine
  format's fused multiply-add. Re-measured under pairs, the LDS table wins
  there too (**2679 us against 2726** with the table in registers), so both
  kernels keep it and what is left is the lookup itself.
- **`down_shexp` could be IQ4_NL too.** It is the same kernel arm, the same
  640 row; at Q5_1 it costs 0.037 GB a token and IQ4_NL would save 0.009 —
  **+0.07 tok/s**, an order of magnitude under the instrument, and it would
  need its own 145-chunk run because a plan is measured and not composed.
  Priced and not taken.
- **Q8_0 is the other misaligned format.** Its 34-byte record takes the same
  two-word window, and §7's pairing argument applies to it verbatim wherever a
  Q8_0 tensor is *transcoded* rather than staged verbatim (two Q8_0 blocks are
  68 bytes, which is also a multiple of four). After D20 and D21
  there are no Q8_0 tensors left in the MoE bank, so there is nothing to fix
  — but the next stage that adds one should not repeat this.
- **The frontier's currency is not GB.** `iq4_nl` sells 0.148 GB a token for
  0.05 perplexity points and 0.73 tok/s; `q4_1` sells 0.098 for 0.30 points
  and 0.60 tok/s. Priced in bytes those are 0.34 and 3.05 pp/GB and the
  ranking is clear, but the *reason* a byte is worth having is the time it
  buys, and two families convert bytes to time at rates that differ by 40%
  (137 GB/s on the shared expert against 211 on the routed down). A future
  knapsack should be run in pp per tok/s, which is the quantity both sides of
  the trade are actually denominated in.
