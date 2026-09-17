<!-- LLM.md L8c-4. The 4.5-bit dense bank: the format L8c-3 chose, stored and
     read by a kernel, and graded on the lm head.
     Cited from llm/bank_q4.go, llm/quantk.go, shaders/llm_gemm.comp -DQ4B,
     shaders/llm_gemv.comp -DQ4B. -->

[← LLM.md](../LLM.md) · [research index](README.md) · bank

# L8c-4 — the 4.5-bit bank, and a simulation it reproduces to the last place

**Result: the format L8c-3 recommended now exists as a bank and two kernels,
and the first thing it had to prove it proved as an equality.** On the lm head
the real 4.5-bit bank's logits are **identical, all 248 320 of them**, to the
same format run through `llm/sim.go` — and over 8184 tokens of `wiki.test.raw`
the real bank's perplexity is **2.0269**, which is `results/l8c_asym.csv`'s
simulated `lm_head q4_k/32` row to four decimal places.

**Over the whole corpus the head at 4.5 bits is 4.0621 against our own 4.0289
— +0.82%**, 145 chunks in 7m44s, which is llama.cpp's own 4.0340 plus 0.70%.
And that is **2.05x what the eight-chunk screen said.** L8c-3's per-family
table puts `lm_head q4_k/32` at +0.40%, and the real bank reproduces *that
screen* to four decimal places — 2.0269 against 2.0269 — so the gap is not
the bank, it is the screen. L8c-0 already showed a short run is not
calibrated for a percent on the whole model; this says the same of a
**family**, and in the direction that matters, because every per-family row
in `results/l8c_asym.csv` is an eight-chunk row.

**And the head is 1.89x at one token.** `output.weight` is 0.68 GB in the
checkpoint's own Q8_0 and **0.358 GB** at 4.500 bits; at one row both banks
run the decode GEMV at **210 GB/s of a 242 GB/s bus**, so the bytes *are* the
time: 3214.4 us becomes **1699.8**. Against the halves this block staged
before L8 it is 3.80x.

| lm head, one row | bank | best rung | us | GB/s |
|---|---:|---|---:|---:|
| halves (pre-L8) | 1.27 GB | `gemm_m2` | 6457.0 | 196.9 |
| the checkpoint's Q8_0 (L8a) | 0.675 GB | `gemv_k1` | 3214.4 | 210.1 |
| **`q4_k/32` (L8c-4)** | **0.358 GB** | `gemv_k1` | **1699.8** | **210.4** |

Files: `llm/quantk.go` (new), `llm/bank_q4.go` (new),
`shaders/llm_q4k.glsl` (new), `shaders/llm_gemm.comp` and
`shaders/llm_gemv.comp` (`-DQ4B` arms), `llm/gpu_head.go`, `llm/sim.go`,
`llm/graph.go`, `cmd/llm/bench_head.go`. Results: `results/l8c_head.csv`,
`results/l8c_ppl_head_q4k.csv`, `results/l8c_decode_q4k.csv`.

---

## In the whole model

`-gen -n 64 -prompt 'The capital of France is'`, which is what every committed
decode CSV in this vertical is — `results/l8e_decode.csv` against
`results/l8c_decode_q4k.csv`, row for row:

| | q8 bank (L8e) | **head at `q4_k/32`** |
|---|---:|---:|
| decode | 24.66 tok/s | **25.75** (1.044x) |
| against llama.cpp's 25.15 | 0.981x | **1.024x** |
| a token | 40.5 ms | **38.8 ms** |
| the head block, per token | 3.5 ms | **1.9 ms** |
| residency | 81.89 GB | **81.57 GB** |

**That is the first time this vertical is ahead of llama.cpp at decode.** It
is also, at 1.044x, almost exactly the 1.055x the bytes predict — on L8b's
basis, our *staged* bank: 6.05 GB a token to 5.733, a ceiling of 40.0 tok/s to
42.2. On `cmd/gguf`'s basis, the checkpoint's own widths, the same 0.318 GB
takes 6.334 to 6.016 and 38.2 tok/s to 40.2. The two bases differ and the
delta does not; LLM.md's table is on the second.

The block ratio in the graph is the bench's ratio — 3621.1 us to 2049.9 on
`gemm_m2`, 1.77x — because `Graph` runs the head on the GEMM and not the GEMV.
That is L8d and L8e's decision and it was right on the Q8 bank, where the GEMM
was already at 189 GB/s; on this one the GEMM is at 174 and the GEMV at 210,
so there is **0.35 ms a token** left in wiring the head's decode kernel up.
It was left out of this stage because it changes the arithmetic (one fewer
rounding per weight, above) and the measurements here were already run.

**Prefill does not move and cannot.** The head projects *one* row at prefill —
llama.cpp's graph ends with `inp_out_ids`, which is why `HeadGPU` exists in
this shape at all — so the 1.5 ms it saves is 0.08% of a 2048-token pass. No
prefill ladder was run for that; the 512-row rows in the table above are the
bench's, and they exist to price the unpack rather than the model.

**And the text is no longer a gate — but the divergence is one near-tie, and
`-top 3` says so.** At temperature zero the completion parts at the **third**
token of the body, and both runs offer the *same two* candidates there:

    q8    [*561 " The" 15.590    11751 " Paris" 15.229]   margin 0.361
    q4_k  [*11751 " Paris" 15.537    561 " The" 15.462]   margin 0.075

The 4.5-bit head moved that pair by 0.36 of a logit and flipped a tie that was
already only 0.361 wide. Downstream the two texts have nothing in common — the
Q8 bank recites capitals and ends at `Lisbon.` in 69 tokens, the 4.5-bit head
writes a syllogism about Paris for the full 128 — but that is a five-token
prompt branching, not 248 320 logits going wrong. This is the first stage in
the vertical where the two were ever allowed to differ, and it is what D17 and
L8c's gate were written for: the grade is the perplexity above, not the
tokens.

## The screen under-reads a family, and it is worth saying twice

| `lm_head` at `q4_k/32` | baseline | measured | delta |
|---|---:|---:|---:|
| the simulation, 8 chunks (L8c-3) | 2.0189 | 2.0269 | +0.40% |
| **the bank**, 8 chunks | 2.0189 | **2.0269** | **+0.40%** |
| **the bank**, 145 chunks | 4.0289 | **4.0621** | **+0.82%** |

The two eight-chunk rows agreeing to the last digit printed is the corpus-scale
version of the logit equality — the bank is the format, not a neighbour of it.
The third row is the warning. L8c-0 measured it on the whole model (a
four-chunk run read +0.32% where the corpus read −0.13%, sign and all) and
this is the same fact one level down: **a family's cost measured on the first
8184 tokens of `wiki.test.raw` is not its cost.** The uniform plan's +4.24%
*is* a corpus number and stands; what does not is the per-family attribution
under it, which is optimistic — 0.40 against 0.82 on the one family that has
now been run both ways.

## The layout, and why it is ggml's record over our tiles

Two things had to be true at once. The *values* had to be the ones L8c-3
measured — otherwise the +4.24% on the table is a number about a different
bank — and the *addressing* had to be §2.8's fragment tiling, because that is
what every dense kernel in this vertical reads and what a cooperative-matrix
load off LDS needs.

So the bank is the tiling with a nibble where L8a puts a byte, plus a second
plane that is ggml's own super-block record:

    tiles    tile (nt, kt) is 128 contiguous bytes, kt fastest inside an
             n-tile, holding element (k, n) at nibble (n%16)*16 + k%16 —
             byte (n%16)*8 + (k%16)/2, low nibble for even k. A `uint` is
             eight consecutive k of one output column, where L8a's is four.

    records  one per (n-tile, super-block, row): sixteen bytes, `d` and
             `dmin` as halves and then the twelve bytes ggml's
             `get_scale_min_k4` reads eight 6-bit (scale, min) pairs out of.
             Indexed ((nt*nsb + sb)*16 + n%16), so the sixteen records a tile
             column needs are 256 contiguous bytes — D8's k-major
             arrangement, for D8's reason.

4 bits of level, 12 bits a group of 32 and 32 bits a super-block of 256 is
**4.500 bits a weight exactly**, which is the number every L8c-3 rung is
quoted at. For the head that is 357 580 800 bytes against the Q8 bank's
675 430 400 and the halves' 1 271 398 400.

The twelve bytes are ggml's packing and not a convenient one of our own, which
costs a four-way branch per group to unpack and buys the thing this stage is
about: the same bytes ggml writes, so the port that produced L8c-3's numbers
(`make_qkx2/3_quants`, `make_qp_quants`, bit-identical to
`ggml_quantize_chunk` over 204 800 values an arm) is the *encoder of the bank*
rather than a model of it.

## One encoder, two callers

`applyAsym` used to do the whole thing inline and throw the levels away — a
simulation only needs the floats. `llm/quantk.go` is that arithmetic as an
object that keeps them:

    sim.go      encode, then write back d*sc*l - dmin*m as floats
    bank_q4.go  encode, then pack (d, dmin, the 6-bit pairs, the nibbles)

which is why "the bank is the format the simulation measured" is true by
construction. `TestBankQ4KIsTheSim` states it on a row permutation rather than
the identity, because the only bug a new bank can have is an address.

## The two arms

**`llm_gemm.comp -DQ4B`** is the Q8 arm plus an affine term, and the one
structural difference is *where the record is read*. A record covers 256 k and
a k-slab is BK = 32, so reading it inside the unpack would fetch it eight
times. Instead each lane holds its own column's record in a register,
refreshes it where the loop crosses a super-block, and turns it into
`(d*sc, dmin*m)` in a BN-long LDS vector the unpack indexes — one sixteen-byte
load per column per 256 k, the format's own 12.5%, and no `get_scale_min_k4`
in the inner loop. It needs BK = 32, which is every build in `shaders.go`, and
the kernel says so with an `#error` rather than assuming it.

**`llm_gemv.comp -DQ4B`** is the same idea with no LDS at all: a lane owns one
output column and keeps its record in a register across the four steps of a
super-block. A tile is 32 words and a lane owns two of them.

**The GEMV is not bit-exact against the GEMM on this bank, and it cannot be.**
L8b-2's argument was that `float16_t(q) * d` is a real f16 instruction over
two exact halves, so the register path forms the same half the LDS store
holds. A K-quant group is affine: `d*sc*l - dmin*m` has a subtraction in it,
the GEMV does it in f32 and the GEMM rounds the result to a half on its way
into LDS, so the GEMV carries one **fewer** rounding per weight. That is
exactly `llm_moe_gemv.comp`'s position against `llm_moe_gemm.comp` over the
same format, and it is measured rather than asserted: on the head's row of
248 320 logits the two are **rms 2.29e-04 relative**, against the Q8 pair's
7.84e-06 — which is reduction order alone, since on that bank the weights
really are identical.

## What the ladder says, and where D14 stops applying

The row-block ladder inverts between the banks at one row, and it inverts
harder than the Q8 one did:

| rows | rung | fp16 us | q8 us | q4_k us |
|---:|---|---:|---:|---:|
| 1 | `gemm_m2` | 6457.0 | 3621.1 | 2049.9 |
| 1 | `gemm_m8` | 6897.6 | 4734.6 | 4354.0 |
| 512 | `gemm_m2` | 98887.5 | 53658.3 | 28969.5 |
| 512 | `gemm_m8` | 26864.6 | 18839.0 | 17461.0 |

**D14 says a narrower bank is a decode decision and costs at prefill**, and it
was measured on the hyper-connection block's two projections: 2.3-2.5x faster
at one token and 1.09-1.15x *slower* at ubatch 2048, because at 2048 each
weight is read 32 times out of a 32 MiB MALL and the DRAM bytes were already
hidden. The head is the counter-example and it names the condition: its B is
357 MB and never fits the MALL at any width, so there is nothing to hide
behind and the bank is the kernel at **every** length — 18839.0 us to 17461.0
at 512 rows, 1.08x, in the direction D14 warns about. The unpack does show up
(q4_k moves 82 GB/s of bank there against q8's 143) — it just does not win.

So D14 gains a clause: it is a statement about a weight that **fits the
MALL**. Where it does not, a narrower bank is faster at both ends.

## What this does not cover

A super-block is ggml's eight groups, so `k % 256 == 0`. Every dense matrix in
this model satisfies that but one family, and it is the family the whole
imatrix result turns on: `hc_{attn,ffn}_up` and `output_hc_up` read the
low-rank space and are **320** wide. The simulation handles that with a
ten-group super-block; `get_scale_min_k4`'s packing does not, because the
scheme *is* four low pairs and four high ones and not a length. That family
needs a packing of its own, and it is L8b's block, so it gets both together.

Five bits is the other gap. L8c-3's mixed plan puts the attention layer and
the hyper-connection block at `q5_k` for +2.70% against the uniform plan's
+4.24%; a fifth bit is a plane of its own the way ggml's `qh` is, a third
stream through the unpack, and a tile that is no longer one byte per two
elements. The uniform plan is what this bank is built for.
