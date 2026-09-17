<!-- LLM.md L8c-6. The hyper-connection block on the 4.5-bit bank: the third
     family, the one that needed a record packing of its own, and the first
     family whose eight-chunk screen was *pessimistic*.
     Cited from llm/gpu.go, llm/quantk.go, llm/bank_q4.go, shaders/llm_q4k.glsl. -->

[← LLM.md](../LLM.md) · [research index](README.md) · bank

# L8c-6 — the hyper-connection block at 4.5 bits, and a record that is not ggml's

**Result: all 97 mixers are on L8c-4's bank, decode is 30.05-30.19 tok/s against
28.84, the block is 1.37x at one token and 1.50x on its own ladder, and the
family costs +0.31% of perplexity where its own eight-chunk screen said
+0.90%.** Its up projection is the one dense matrix in this model that
`get_scale_min_k4` cannot describe, so it gets a twenty-byte record and a
twelve-bit decode of its own — and the bits do not move: 4.500 either way.

The hyper-connection block is the model's plumbing: 97 mixers a pass, two a
layer plus the head's, 0.695 GB of a 6.334 GB decode token on the
checkpoint's own widths. It went third because its bytes said so, and it went
*after* the head and the DeltaNet because it is the family with the obstacle
in it — `hc_{attn,ffn}_up` and `output_hc_up` read the low-rank space and are
**320 wide**, where ggml's K-quants want `k % 256 == 0`.

| a decode token | L8e (q8) | L8c-4 (+head) | L8c-5 (+dn) | **L8c-6 (+hc)** |
|---|---:|---:|---:|---:|
| decode | 24.66 tok/s | 25.75 | 28.84 | **30.05 / 30.19** |
| against llama.cpp's 25.15 | 0.981x | 1.024x | 1.15x | **1.20x** |
| a token | 40.5 ms | 38.8 ms | 34.7 ms | **33.1 ms** |
| the hyper-connection block, per token | — | — | 5.68 ms | **4.16 ms** |
| a mixer, staged | 8.12 MB | — | 8.12 MB | **4.76 MB** |
| the block, resident | 0.79 GB | — | 0.79 GB | **0.47 GB** |
| residency | 81.89 GB | 81.57 GB | 80.53 GB | **80.20 GB** |
| prefill, ubatch 2048 | 1052.8 tok/s | — | 1069.7 | 1065.9 |
| prefill, ubatch 512 | 655.2 tok/s | — | 667.2 | **669.7** |
| perplexity, 145 chunks | 4.0289 | 4.0621 (+0.82%) | 4.1012 (+1.79%) | **4.1104 (+2.02%)** |

Files: `llm/gpu.go`, `llm/quantk.go`, `llm/bank_q4.go`, `llm/bank.go`,
`llm/hc.go`, `llm/model.go`, `llm/graph.go`, `shaders/llm_q4k.glsl`,
`shaders/llm_gemm.comp`, `shaders/llm_hc_gemv.comp`, `cmd/llm/bench.go`.
Results: `results/l8c_decode_hc.csv`, `results/l8c_graph_hc.csv`,
`results/l8c_ppl_hc_only.csv`, `results/l8c_ppl_hc_q4k.csv`,
`results/l8c_hc_ladder.csv`, `results/l8c_hc_ladder_q8.csv`,
`results/l8c_hc_gemv.csv`, `results/l8c_hc_gemv_q8.csv`.

---

## 1. `get_scale_min_k4` is eight groups, and it is not a length

ggml's Q4_K record is sixteen bytes: an fp16 `d`, an fp16 `dmin`, and twelve
bytes holding eight six-bit (scale, min) pairs. The twelve bytes are not a
loop. Four pairs are carried whole in the low six bits of bytes 0-7, and the
*other* four steal their high two bits from the spare bits of the first four:

    j < 4    sc = b[j] & 63              mn = b[j+4] & 63
    j >= 4   sc = (b[j+4] & 15) | (b[j-4] >> 6) << 4
             mn = (b[j+4] >> 4)  | (b[j]   >> 6) << 4

There is no ninth or tenth pair to put anywhere, and there are no spare bits
left to put it in. So a super-block that is not eight groups is a **different
packing wearing the same name**, and L8c-4 wrote that down as the one thing
its bank did not cover.

The simulation had already crossed this bridge. `asymSubBlocks` — the rule
L8c-3 measured the asymmetric form with — is ggml's eight groups wherever
they fit and the **whole row** where they do not, and it does not care about
bytes because nothing in `make_qkx3_quants` or `make_qp_quants` knows the
count. At 320 that is ten groups, and the number quoted in L8c-3 is **4.475
bits**: 32 bits of fp16 pair and 120 of packed pairs over 320 weights.

A bank does care about bytes, because every kernel reads the record plane
through a `uint` view and an offset has to be a multiple of four. So the
ten-group record is **twenty bytes**, not nineteen:

    word 0    d | dmin, the fp16 pair — the same word as ggml's
    word 1-4  ten twelve-bit fields, group j at bit 12*j of a little-endian
              bit stream, scale in the low six and min in the high six

120 bits of 128 used, and the twentieth byte is the alignment. That is
**4.500 bits a weight** at 320 as well as at 256, which is a tidier number
than the simulation's 4.475 and a hair more expensive — one byte a record,
0.025 bits a weight, 6.5 KB a mixer. It is stated rather than rounded off
(`q4kRecordBytes`), because the whole stage rests on the bank being the
format the simulation measured and this is the one place where it is not
byte for byte the same thing.

The decode is *simpler* than ggml's, which is the small consolation:

```glsl
void q4kScaleMin12(uint j, uvec4 s, out uint sc, out uint mn) {
    uint b = j * 12u;
    uint w = b >> 5, o = b & 31u;
    uint v = s[w] >> o;
    if (o > 20u) v |= s[w + 1u] << (32u - o);
    sc = v & 63u;  mn = (v >> 6) & 63u;
}
```

A shift and a mask against a branch on `j`. `j*12` crosses a word boundary
only at j = 2 and j = 5, and the next word exists in both, so there is no
guard beyond the shift that makes it zero.

**Where the choice is made is nowhere.** `tileBQ4K` reads the super-block
length off `k` through the same `asymSubBlocks` the simulation uses, picks
the record size from it, and picks the packing from that; the kernel is told
once, at compile time, by `-DQ4K_SUB=10` on the four up rungs. Nothing on
either side names 320. `TestQ4KRecordPackings` is the writer against both
readers over every representable pair, 2000 trials a length, plus the
assertion that the padding byte stays zero — there is no ggml oracle for a
packing ggml does not have, so the only thing that can be checked is that the
three spellings agree.

## 2. The fp16 tail stopped being free

L8b's hardest finding was that this block's split cannot fall where the other
blocks' do. `inject` is four **F32** rows at row 320 of a fused matrix whose N
is 336, and the down ladder's BN is 48 because 336 is 21 tiles and 21 is
3 x 7 — so the split is the column block that *contains* inject, 288, and the
32 low-rank rows between 288 and 320 are staged in both planes and read out
of the fp16 one.

On L8a's bank that cost nothing but 0.33 MB a mixer, because those 32 rows
are Q8_0 in the checkpoint: the halves in the tail and the bytes in the main
plane are **the same numbers**, and it does not matter which the kernel
reads. At 4.5 bits it matters. The simulation quantises all 320 rows of
`hc_*_down`, so a bank that staged the tail from the *original* floats would
be 10% of that matrix more accurate than the format it claims to be — and
would still pass every tensor comparison, because nothing would look wrong.

So `stageQ4` runs those rows through the same encoder and *then* narrows them
to halves, which is exactly what `sim.go` does one step later. `inject` is
untouched: four F32 rows of the checkpoint, staged as halves, never
quantised, the way D13 left it and L8a's three families left theirs.

`TestHCGPUQ4IsTheSim` is what makes this checkable rather than argued. It
stages one mixer twice from the same weights — once on the fp16 bank with the
format applied through `sim.go`, once on the real bank — and runs the whole
block on every rung of both GEMM ladders:

| | |
|---|---|
| GEMM pairs compared | 16 (4 down rungs x 4 up rungs) |
| values identical | **2 688 320** |
| tensors | `hc_norm`, `lo`, `hc_inject`, `hc_gate`, `hc_mixed`, `hc_combine` |

Not close — identical, through the collapse, through the scatter, and through
the combine. A tail staged unquantised shows up in `hc_mixed` and not in
`hc_inject`, which is the discrimination the test is shaped for.

The split-K GEMV at one token is the exception, and it is L8c-4's exception
rather than a new one: a K-quant group is affine, so there is no pair of
exact halves to multiply and the register path carries one fewer rounding
than the GEMM's LDS store. All six rungs land at **8.29e-05 rms** on
`hc_mixed` against the GEMM — one rounding over a 10240-long dot product,
where the Q8 pair was bit-exact.

## 3. The screen was pessimistic this time, which is worse than optimistic

L8c-4 found an eight-chunk per-family screen under-reads by 2.05x; L8c-5
found it can have the wrong *sign*. This family is the third, and it breaks
the remaining pattern — the screen is **too harsh**:

| `hyper_conn` at `q4_k/32`, imatrix | baseline | measured | delta |
|---|---:|---:|---:|
| the simulation, 8 chunks (L8c-3) | 2.0189 | 2.0371 | **+0.90%** |
| **the bank**, 145 chunks | 4.0289 | **4.0416** | **+0.31%** |

Three families, three directions: `lm_head` +0.40% → +0.82%, `deltanet`
−0.30% → +0.93%, `hyper_conn` +0.90% → **+0.31%**. The screen is not a biased
estimator with a correction factor; it is eight chunks of one corpus, and
what it gets wrong is not only the magnitude but the **ranking**.

With three families measured over the corpus, the ranking can finally be
stated in the units the plan is built in — percentage points of perplexity
per GB of decode token bought back:

| family | GB a token saved | corpus cost | pp per GB |
|---|---:|---:|---:|
| `deltanet` | 1.073 | +0.93% | **0.87** |
| `hyper_conn` | 0.335 | +0.31% | **0.93** |
| `lm_head` | 0.317 | +0.82% | **2.59** |

**That retires L8c-1's headline shape.** "The cost of 4 bits runs inverse to
the bytes" came from a table in which the DeltaNet was 46% of a dense token
for +1.49%, the lm head 14% for +0.28% and the hyper-connection block 14% for
+5.98% — the biggest family disproportionately cheap and the plumbing
disproportionately expensive. Measured on the asymmetric form over 145
chunks, the two large families cost **almost exactly the same per byte** and
the outlier is the **lm head**, at 3x either, in the opposite corner of that
table from where the screen put it.

For this block specifically, the whole of the movement is D7's reversal.
L8c-3 measured the asymmetric form at 3.2x cheaper on this family and the
imatrix taking it from +9.04% to +0.90% — the single largest movement in that
stage's 2x2 — and over the corpus it is better again. The block L8c-1 called
"the model's plumbing" and priced at +5.98% costs **+0.31%**.

What survives from L8c-5 is **additivity**:

| 145 chunks, against 4.0289 | ppl | delta |
|---|---:|---:|
| `lm_head` alone (L8c-4) | 4.0621 | +0.82% |
| `deltanet` alone (L8c-5) | 4.0665 | +0.93% |
| `hyper_conn` alone | **4.0416** | **+0.31%** |
| the three together | **4.1104** | **+2.02%** |
| their three deltas summed | — | +2.06% |

2.02 against 2.06 is compounding of **−0.04 pp** over three families — the
sum is now a slight *over*-estimate, where over two families it was a 0.04 pp
under-estimate. L8c-1 measured four symmetric families summing to 11.7% where
their plan measured 18.5%, a 1.58x gap, and attributed it to errors injected
at 36 or 48 depths compounding on L6b-3's x1.085 a layer. In the asymmetric
form that effect is not merely milder; at three families it is **not
measurable** — 0.04 pp is a sixth of this corpus's standard error on the
delta. The practical consequence is that the remaining two families can be
budgeted by addition: `full_attn` and `ple_proj` at their own corpus numbers
will land where their sum says, and the plan's own +4.24% (L8c-3, all five at
once) remains the thing they are checked against.

## 4. The bank is the format, said twice again

Beside the device equality in §2, the two arms were run over eight chunks of
the corpus:

    bank   LLM_DENSE_BANK=hyper_conn=q4_k/32
    sim    LLM_DENSE_SIM=hyper_conn=q4_k/32
           LLM_DENSE_SIM_SRC=q8 LLM_DENSE_SIM_QUANT=imatrix

and every per-chunk figure is the same to four decimals — 1.4081, 1.3452,
1.4445, 1.9658, 2.1388, 2.0479, 1.9410 and a final **2.0345** on both sides.
`SRC=q8` is what makes them the same question: the simulation's default scope
includes `hc_*_inject`, which is F32 in the checkpoint and which the bank
leaves in its fp16 tail.

Two of this family's tensors have no entry in unsloth's published matrix —
`output_hc_down` and `output_hc_up`, the head mixer's pair, 0.007 B weights —
and both arms fall back to round-to-nearest the way ggml does
(`bankImatrix`). This is the first family where that fallback actually fires,
and it fires identically on both sides, which is the property it was written
for at L8c-5.

**Each matrix is calibrated under its own name.** `hc_attn_up` and
`hc_ffn_up` are different tensors, and `hc_attn_up` is L8c-2's outlier — the
one whose importance is concentrated in an effective 4.4 of every 32 columns
and which carried that stage's whole regression — so a block that calibrated
a mixer with one row would not be the format that was measured. `HCWeights`
gained a `Name` for this, the way `DeltaNetWeights` gained its layer index.

## 5. The ladder moved in three places and stayed put in one

`-hc -tokens 1 -ladder -mixers 24` and `-hc -tokens 64,128,512,1024,2048
-ladder -mixers 24`, on both banks, same session, block microseconds a mixer
at each length's best pair:

| T | q8 | q4_k | q4_k / q8 |
|---:|---|---|---:|
| 1 | 36.9 `gemv32/up_m1` | **24.6** `gemv32/up_m1` | **1.50x** |
| 64 | 255.1 `m1/up_m4` | **202.4** `m1/up_m4` | 1.26x |
| 128 | 289.3 `m1/up_m2` | **241.9** `m1/up_m8` | 1.20x |
| 512 | 536.9 `m2/up_m4` | **517.1** `m1/up_m8` | 1.04x |
| 1024 | 1629.3 `m2/up_m4` | **1601.1** `m2/up_m8` | 1.02x |
| 2048 | 3303.0 `m4/up_m4` | 3373.3 `m4/up_m4` | **0.98x** |

**The up projection wants a wider row block, again.** L8b found that the Q8
arm wants `up_m4` where the fp16 arm never wanted more than `up_m2`, because
the unpack costs 256/WM element conversions per cooperative-matrix step. A
nibble, a six-bit pair and an affine term is more work per weight than a byte
and a multiply, so the same rule applies once more and in the same direction:
`up_m8` wins from 128 to 1536.

**The down ladder's row block goes the other way at 512**, from `m2` to `m1`.
What a wide BM buys is reuse of an unpack across token rows; this bank has
already halved the bytes that unpack feeds on, so the rung that wins is the
one with more workgroups.

**And the GEMV rung did not move, which corrects §5.1b's rule.** D12 says a
split-K ladder does not carry across a bank because a slab's stride changes:
going from halves to int8 moved the winner from 160 to 32, and going from
int8 to nibbles halves the stride again. It did not move:

| KSLABS | 8 | 16 | **32** | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|
| slab, bytes | 10240 | 5120 | **2560** | 2048 | 1024 | 512 |
| x 4 KB | 2.5 | 1.25 | **0.625** | 0.5 | 0.25 | 0.125 |
| us a mixer | 9.26 | 6.48 | **5.60** | 7.57 | 6.04 | 5.87 |

L7d's rule was "every rung whose slab is a whole multiple of 4 KB is slow",
and it separated its six rungs cleanly: 10, 5, 2 and 1 were slow and 2.5 and
0.5 were fast. L8b's int8 ladder was consistent with it — the two whole
multiples, 5 and 1, were the two slow rungs. **Here nothing is a whole
multiple and the spread is still 1.65x**, so the rule as stated does not
decide this ladder at all.

What the three banks do share is narrower, and this stage does not have the
measurements to promote it to a rule: the winner is the rung whose slab is
**5/8 of a 4 KB multiple** on both quantised banks — 5120 bytes at int8 and
2560 at 4.5 bits — and the rung that is an exact power of two nearest it
(4096 then 2048) is the second-slowest on both. That is why the incumbent
rung did not move, and the whole-model numbers are what it is judged on
either way.

(These rows read 291-489 GB/s against a 242 GB/s bus, so D16 applies: the
down projection's staged weight is 2.9 MB a mixer and a sweep re-reads it out
of the 32 MiB MALL. What the ladder ranks is kernel shape against a cache
hit. The whole-model numbers in §6 are the measurement.)

## 6. D14's hazard fires, and it is the smallest it has been

At ubatch 2048 the narrower bank is **1.02x slower on the block** — the
second time in this vertical, after L8b-5. The mechanism is the one D14
names: at 2048 tokens a workgroup reads all of B `M/BM` times, and this
block's weights are 4.8 MB a mixer and fit the 32 MiB MALL at *every* width,
so the DRAM bytes were already hidden and all the narrower bank adds is the
unpack's ALU. L8c-5's block changed side of the MALL and got faster; this one
cannot, because it was never on the other side.

What is new is how small it is. L8b-5 measured 1.09-1.15x slower going from
halves to int8; this is 1.019x going from int8 to nibbles, and in the graph:

| prefill, tok/s | L8c-5 (two families) | **L8c-6 (three)** |
|---|---:|---:|
| ubatch 128 | 310.6 | **312.7** |
| ubatch 512 | 667.2 | **669.7** |
| ubatch 1024 | 878.2 | **882.1** |
| ubatch 2048 | 1069.7 | 1065.9 |

Three ubatches faster, one 0.996x. Against llama.cpp's 391.42 that is 2.72x
at 2048 and 1.71x at 512, which is where L8c-5 left it. The block's own share
of a 2048-token graph is 329.8 ms against 323.7 — 6 ms of a 1921 ms pass.

Nothing about that changes the direction: at one token, which is what this
stage is for, the same weight is **1.50x** and the block is 5.68 ms a token
to **4.16**.

## 7. What the bank is, per mixer

| a mixer, staged | bytes | bits/weight |
|---|---:|---:|
| halves (pre-L8) | 13.43 MB | 16.298 |
| the checkpoint's Q8_0 (L8b) | 8.12 MB | 9.851 |
| **`q4_k/32` (L8c-6)** | **4.76 MB** | **5.776** |
| 97 mixers | **0.46 GB** against 0.79 and 1.30 | |

5.776 and not 4.500, and the gap is entirely the fp16 tail: 48 rows of 10240
halves a mixer, 0.98 MB, which is 21% of the staged bank at this width where
it was 12% at int8. `inject` is 4 of those 48 rows and the other 44 are the
low-rank rows the ladder's BN stranded plus the tile pad. **The narrower the
main plane gets, the more the tail costs relatively** — at 4.5 bits it is now
the largest single piece of overhead in this block, and a 16-column tail
would need the split branch to be per tile rather than per workgroup, which
is the shape L8b-1 left open and nobody has measured.

The up projection alone, which is the 320-wide one, is exactly 4.500 bits:
1.64 MB of nibbles and 0.20 MB of twenty-byte records over 3 276 800 weights.
`TestHCGPUQ4BankSize` states all of it.

## 8. What this does not settle

Two dense families are left and they are small: `full_attn` (0.635 GB a
token) and `ple_proj` (0.035), with `qsa_indexer` (0.039) riding along.
Neither has an obstacle in it — every matrix in both is a multiple of 256
wide — so what remains of the dense half is wiring rather than format.

Then the two the simulation cannot reach, which are together worth more than
those three: **the F32 router at fp16** (+1.5 tok/s of ceiling, L5a's ties
are the risk) and **the 512 expert banks at ~4.25** (+3.0 tok/s, and they are
already Q4_K and already the calibrated part of this checkpoint).

    cmd/gguf -width basis, the checkpoint's own widths
    a token   6.334 GB  as shipped
              6.016     lm_head at 4.5          (L8c-4)
              4.943     + deltanet at 4.5       (L8c-5, a 49.0 tok/s ceiling)
              4.609     + hyper_conn at 4.5     (L8c-6, a 52.5 tok/s ceiling)
              4.264     + the other two         (L8c-3's plan, 56.7 tok/s)

Measured, this stage is 30.19 tok/s of that 52.5 — **57.5%**, against L8c-5's
59% of its own and L8c-4's 61%. The bank keeps buying what it says it will
and the fraction of its own ceiling keeps drifting down, which is the same
statement twice: **what is left at decode is not the dense bank.** The
ceilings above are the width arithmetic's, and what we actually stage is a
little wider than that — every quantised plane in this model covers rows its
kernel skips, and this block carries 0.98 MB of fp16 tail a mixer on top.
