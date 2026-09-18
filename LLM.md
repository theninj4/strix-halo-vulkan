# LLM — the qwen3.8-flash-next vertical

> **Current work from 2026-09-17.** `SPEECH.md` is finished-ish (74.2x real
> time, T7 open); `PIPELINE.md` (z-image) is parked at 14.26 s an image. Same
> rules as both: this file is **rewritten** each session rather than appended
> to, history goes to `TODO.md`, closed findings to `research/`. Stage numbers
> are **L0, L1, …**; `§N.M` still addresses `IDEAS.md`.

**Target**: `Qwen/Qwen3.8-Flash-Next` — 180 B params, 6 B active, "a preview of
the Qwen4 architecture" — generating text end to end in Go on Vulkan. The
third vertical, and 200x the parameters of the other two put together.

**Status: the model generates, it is served over HTTP, phase 2's kernel half
is finished, five of the six dense families are now at a width the checkpoint
does not ship, the one correctness cliff is closed, and a decode token is
attributed to the dispatch.** **P0** found that the
graph's never returning above ~2560 rows was amdgpu's gfx ring watchdog killing
any submit that holds the ring past **2 s** — nothing about this model, this
bank or residency — so the recorder chunks by time and **prefill runs to 8192
rows at 1213.5 tok/s, 3.10x llama.cpp, still climbing**. **P1** then attributed
the decode step: **30.52 ms over 36 dispatch labels and six host phases, a 5 us
residual**, and the review's "~12 ms unexplained" is 5.54 ms of dispatches
below 227 GB/s, 3.46 of dispatches that stream no weight and 3.01 of host. It
found the n-gram gather to be **sixteen serialised major page faults a token**
and made them concurrent, which is **decode 31.51 → 32.72 tok/s, 1.30x
llama.cpp**, and it re-ranked everything below it: the bank has stopped being
where decode time is. **P1a** took the top item off that list: the scatter
that closes a hyper-connection mixer and the norm that opens the next are one
workgroup's work said twice, and fusing them is **1501 dispatches a pass to
1407, decode 32.59 → 32.93 tok/s, bit for bit** — while the block's low GB/s
turns out to be a **grid**, not a fusion, because at one token **168
workgroups is 9.26 us where 672 is 5.54 for identical bytes** and `up` is
pinned at 160 by its collapse's 64-column block. **P1b** then re-screened the
MoE decode rungs against DRAM — sixteen cold banks, `-iters 1` — because
every rate the old two-layer ladder reported was an L3 measurement (all six
GEMV down rungs above the 242 GB/s bus), and **no rung moves**: the plan
survives on every axis, the review's 1.9 ms was the MALL's arithmetic, and
what is real is a **1.33 ms a token environment gap** between a dispatch
alone on a cold bank and the same dispatch inside a step. **P1c** then took
the step's re-recording off the host: the only value that varied anywhere in
a decode step's 1407 dispatches was the position in one push dword, so
`SEQ_PAST` moved to dword 0 of each block's arena, the step became
byte-identical token to token, and it is now **one command buffer recorded
once and replayed with a fence** — `record` 1.127 → 0.16 ms, hand-over
0.82 → 0.25, **−1.90 ms a token of the priced 1.96**, bit-identical logits
against the re-recording arm — measured 25.4 → 26.7 tok/s on a day the
machine itself ran every dense-bank kernel 1.6-1.9x slower than P1b's
(yesterday's binary: 39.20 ms a step where it had measured 30.51), which is
its own finding: whole-model numbers now require a same-hour control.
**L9a** put the generation loop behind
`cmd/serve` — the checkpoint's own chat template transcribed and checked
against Jinja, and three envelopes over one loop (`API.md`). L0, L1, the whole of L2, L3, L4, L5, L6, **L7 — a
prompt in, tokens out, one at a time, over a cache the last step extended —
L8a and L8b, which between them put every dense weight in the model
on the width the checkpoint already ships, L8d and L8e, which give every block
that dominates a decode step a kernel shaped for one row, and now L8c-4,
L8c-5, L8c-6 and L8c-7, which put the lm head, all 36 gated DeltaNet layers,
all 97 hyper-connection mixers and all 12 full-attention layers with their QSA
indexers on 4.5 bits.** The checkpoint is downloaded,
llama.cpp runs it, the Go side reads it, the prefill mystery is solved, and
there is a kernel for every layer and every sub-layer: the hyper-connection
block in four dispatches where the reference has sixteen (4.24x, 14.9% of
llama.cpp's whole prefill graph), the PLE n-gram block beside it with its
trigram hash bit-exact, the full-attention layer with its QSA indexer **and**
its selection in seven dispatches where the reference has twenty-eight (L2f
and L4b, 2.32x), the gated DeltaNet — three quarters of the layers — in five
where the reference has eleven (L3b, 1.37x, the recurrence itself at 1.06x),
the MoE block, 35.7% of the graph and 97% of the parameters, in nine
dispatches against about 845 (L5b), the order that makes them one forward
pass, and now the three histories that make a second pass a continuation of
the first.

**Between them the five blocks have taken 1015.5 ms of llama.cpp's 1164.7 ms
prefill graph down to 714.8** — 87.2% of it replaced by 61.4%, a 25.8% saving
on the whole graph. 114 GB in 18 minutes;
`models/Qwen3.8-Flash-Next-GGUF/` holds the four `UD-Q4_K_XL` shards and the
2.79 GB MTP head.

**L5b is the first kernel here whose weights never become floats.** One
layer's three expert banks are 2.52 billion weights — 5.03 GB as halves and
241 GB across the model, against 1.57 GB and 75 GB as they ship — so the bank
is staged byte for byte out of the mmap'd checkpoint and each workgroup
unpacks its own Q4_K/Q5_K/Q5_1/Q8_0 slab into LDS per K-step. **567.4 ms of
llama.cpp's graph becomes 522.3 — 1.09x at its own best ubatch — and 1758.4
becomes 1131.6 at ubatch 2048, 1.55x**, with `ffn_out` at 8.132e-05 rms over
all 4096 tokens and the top-10-of-512 selection the reference's set for set
*and order for order*.

**L6a put the whole model on the device: 84.20 GB of weights and 0.87 GB of
arenas in 68 buffers, staged in 33 seconds, with 41 GB of this machine left
over.** That needed one structural change and it was not in a kernel: a
layer's expert bank is 1.61 GB against a `maxStorageBufferRange` of 4 GiB - 4,
so binding 5 became an **array of buffers, one a layer**. **Residency turns
out to be free in the bank's size** — 2 to 48 banks span 10.4 to 84.2 GB
resident and 11062-11242 us for the same block, a 1.6% spread with no trend.

**L6b is the order, and it clears the gate.** On a seven-token prompt the last
token's argmax is llama.cpp's (**561**), out of a top ten that is llama.cpp's
ten tokens; the drift is a clean geometric **x1.085 a layer** with no step at
any block, priced against a second oracle pass at six depths. Nothing in a
shader changed to get there. **What did was a memory type**: the activation
arenas were on a write-combined type the host reads at **0.18 GB/s** where a
HOST_CACHED one reads them at **25.06**, and moving them was **5.0x on the
whole graph** for 0.14% on the kernels.

**L6c took the glue off the host, and not one number moved.** Every block owns
its own arenas, so an activation crossing a block boundary was read out of one
mapped arena, narrowed to halves and written into the other, 96 times a pass —
22.7% of the graph. `shaders/llm_move.comp` is that move as one dispatch, and
it is **bit-exact**: `l_last` at all six depths and both of the head's tensors
came back identical to the last place. The moves themselves are 2.0% of the
pass, so the shared arena L6c was scoped as would buy that and nothing else,
and **96.6% of the graph is the five blocks**.

**L7 is decode, and the gate it clears is the text.** Given
`The capital of France is` at temperature zero the model produces llama.cpp's
completion — the same list of capitals, the same `<think>` block, the same
`Lisbon.` — diverging at **exactly one token**, where our top two logits are
0.161 apart against the 3 to 10 that decide the tokens either side of it, and
re-converging within a sentence. Underneath it, two equalities: a 4096-token
prompt through the attention layer in chunks of 512, 64, 7 or **one** is
identical to the last place, and the whole graph over 512 tokens in chunks of
128, of 9, or as 495 then seventeen single ones returns `result_norm`
identical to the last place — on a pinned schedule, which is a distinction
L7d had to introduce and measures. **Prefill is 990.6 tok/s at ubatch 2048
against 391.42 — 2.53x — and 622.5 at 512 against 313.62, 1.98x.**

**Decode is 7.46 tok/s against llama.cpp's 25.15, and the attribution is two
kernels rather than a direction.** One `madvise(MADV_RANDOM)` on the n-gram
table was worth **176x** on the host gather; a command buffer a block rather
than a submit a dispatch was worth 12%; and the ceiling turns out to be **25.0
tok/s and not 38.2**, because our dense half is staged as halves and a token
reads 9.67 GB rather than the checkpoint's 6.334.

**L7d took those two kernels, and they were one finding said twice: at one
token every dispatch in this model is short of workgroups, not of rows.**
Decode is **11.89 tok/s against L7c's 7.46 — 1.59x — with the completion
unchanged**, and prefill comes along at **990.6 tok/s, 2.53x**. The
hyper-connection down projection ran in **seven** workgroups because its fused
N is 336 and BN is 48, so at M=1 the parallelism has to come from **K**:
`llm_hc_gemv.comp` is the split-K GEMV over the weight the GEMM already
staged, and §2.8's fragment tiling turns out to be exactly the layout it wants
— **238.6 us to 30.0, 7.95x, 29 GB/s to 230, 95% of the bus**. Which split is
**§5.1b's 4 KB period and not the workgroup count**: every rung whose slab
stride is a whole multiple of 4 KB is slow and both that are not are fast. The
MoE is the same finding on the other axis — ten experts have one row each, so
every rung makes ten tiles and the padding was never the constraint; cutting
BN from 64 to 16 is **1.28x**. And **one command buffer a pass** rather than
one a block, ~490 submits to two, **1.18x**, with the attribution moved inside
it: a timestamp after every dispatch, which is what llama.cpp's own
`GGML_VK_PERF_LOGGER` reports on the other side of every comparison here.
**A token is 84.1 ms at 122 GB/s, half the bus**, and the MoE is the only
block still far off its bytes.

**L8a stops the dense half being halves, and the arithmetic does not move.**
A dense weight is now staged as the int8 the checkpoint already ships it as,
with one fp16 scale per 32 elements of a row in the same §2.8 fragment tiling,
and `llm_gemm.comp`'s `-DQ8B` arm unpacks a slab into LDS per K-step.
**Decode is 14.07 tok/s against 11.90 — 1.18x — prefill 1044.8 at ubatch 2048
(2.67x), residency 84.20 GB to 82.40, and the completion is the same token
list.** Not within a tolerance: ggml picks `d = amax/127`, so a Q8_0 block's
largest level is always 127 and re-deriving (d, q) from the dequantised floats
returns the checkpoint's own pair — **the halves the matrix cores multiply are
the same halves, tile for tile**, and three tests say so as equalities. The
three families the checkpoint does *not* store as Q8_0 — the DeltaNet's F32
alpha and beta, the attention layer's BF16 indexer projections, the
hyper-connection block's F32 inject — would be a real re-quantisation, and
int8 took `ssm_alpha` from 1.3e-04 rms to 3.0e-03, past its own gate; they
keep their halves in a tail, and for L8a's three the split falls on a column
block of its own accord (L8b's does not). **A token is 6.61 GB rather than 9.67 and this bank's own ceiling is
36.6 tok/s rather than 25.0**, within 4% of what the checkpoint's own width
allows.

**L8b is the last dense family, and the block is 2.14x at one token.** The
hyper-connection block was the one L8a left on halves — 1.31 GB a token, 13.3%
of the step — and it was last because none of its three kernels is the plain
arm. Each reason turned into a finding. **The split is a column block and not
a row**: `inject` is four F32 rows at 320 of a fused N of 336 and the down
ladder's BN is 48, so the tail begins at **288** and the 32 low-rank rows
between it and inject are staged in both planes — 0.33 MB a mixer to keep the
branch per workgroup rather than per tile. **The split-K GEMV has no LDS to
round a weight through**, and `float(float16_t(x))` in a register is folded
away by RADV (D10), so it multiplies **in fp16**: `float16_t(q) * d` is a real
f16 instruction whose correctly-rounded result is the same half the GEMM's LDS
store holds, because q carries at most 8 significant bits and d at most 11.
**And the GEMV ladder slides one rung along §5.1b's 4 KB rotation** exactly as
D12 says it must once a slab is bytes: the split that misses the rotation is
**32** (326 GB/s) and no longer 160, and the two whole multiples of 4 KB are
the two slow rungs. **The up projection then wanted a wider row block than the
fp16 arm ever did** — its unpack costs 256/WM element conversions per matrix
step — so the collapse's scratch went from [BM][BN] to **one m-tile**, 4 KB at
every rung, which is 1.13-1.24x for the fp16 arm too. **Decode is 14.71 tok/s
against 14.07, a token is 6.05 GB against a ceiling of 40.0 tok/s, residency
81.89 GB, prefill unchanged at 1041.1 — and every tensor of the block is
identical to the fp16 bank's on all fifteen rungs.**

**L8d took the kernels before the widths, and it is 1.57x on decode with no
bank change at all.** Phase 2's plan was two bank stages and then the
re-quantisation; L8a and L8b moved the attribution instead, leaving the MoE at
56 GB/s of a 242 GB/s bus and 42% of a token where the lm head holds 194 on
the same kernel family. **Decode is 23.15 tok/s against 14.71, prefill 1049.8
at ubatch 2048 against 1041.1 and 653.9 at 512 against 643.4 — faster on both
— and a token is 43.0 ms against 66.7.** The MoE block is 27.6 ms a token to
**12.05** (2.29x, 58 GB/s of bank to **133**) and the gated DeltaNet 18.9 to
**11.0** (1.72x, 119 to 205).

Five of the six findings are D11 — *read a dispatch's shape off its grid* —
and the sixth is D12. **The MoE's tile grid was a static upper bound of 2052
records where eleven exist**, because `maxRows` lets all 512 experts be padded
and a one-token batch routes ten rows: 82 080 workgroups to run 11, and the
reason the first GEMV ladder read as a catastrophe. **`llm_moe_gemv.comp` is
the expert GEMM at one token** — LPR lanes a column, no LDS slab, no barrier
in the K loop and no fragment, since §2.2's rule is about a cooperative-matrix
fragment and a dot product has none — 2.46-6.17x a dispatch and **3.06x on the
block**. **The router was nine workgroups** and its split-K is 11.8x; **the
combine was one workgroup**, and its new grid needed the column block on the
*x* axis, which is 7.8x at decode and 1.35x at prefill where the obvious order
was 9.2x slower than the single workgroup it replaced. **`llm_gemv.comp` is
the same finding on the dense projections**: the DeltaNet's layer 454.0 us to
242.5, with D12 picking k8 for a K of 2560 and k32 for one of 6144. Nothing
reads a different weight; what changes is the order of a sum, and at
temperature zero it re-words the `<think>` block at **one token whose top two
logits are 0.012 apart**.

**L8e is the same finding a fifth time, and it is the last of phase 1's
kernel work.** The full-attention layer's two projections were still
`llm_gemm.comp` MODE 2 at one token — 218 and **40** workgroups for a
sixteen-row fragment holding one row — and `llm_gemv.comp` was already generic
over both banks and both tail cases, so the work was the wiring and a KSLABS
ladder per projection. **The layer is 414.8 us to 225.9, the block 5.5 ms a
token to 3.5, and the fused projection is at 227 GB/s of a 242 GB/s bus.**
Beside it the MoE's shared expert came off the routed pair's rung — a lane
group is sized to a row's *payload words*, and Q4_K packs 320 into a 2560-long
row where Q8_0 packs 640, so the two ladders invert — and the second of its
two counting-sort passes turned out to be the first one again whenever the two
modes share a row block, which is every decode step and both prefill rungs:
**-48 dispatches a pass at every length.**

**Decode is 24.66 tok/s against 23.15, a token 40.5 ms against 43.2, and
prefill is faster at both ubatches** (1052.8 at 2048, 655.2 at 512). The
completion is **token for token identical to `LLM_DECODE_GEMM=1`'s**, which is
to say L8e lands back on L7c's text where L8d had re-worded one sentence at a
0.012 tie.

**And one of the three things L8e was scoped as was wrong, which is D16.** The
MoE ladder says the routed up mode wants v16w4 — 63.7 us a layer at **288
GB/s**, a rate this machine does not have, because the two-layer fixture's ten
routed experts are 32 MB and the sweep re-reads them out of a 32 MiB MALL. In
the whole model that same change is **42.1 ms worse over 64 tokens**. A
micro-bench rung whose rate is above the bus is not a DRAM measurement.

**L8c starts with the instrument, and the instrument says our model is the
reference's model.** L8c is the first stage here that changes what the model
computes, so a tensor comparison stops being a grade and its gate is written
against a perplexity — a number nothing in this vertical could produce, because
`inp_out_ids` means a prefill computes one row of logits. `Graph.ForwardRows`
is that pass without the move: the final mixer over the whole batch, the head
over slabs of 256 rows, and `TestGraphForwardRows` asserting the thing that
makes a second-half perplexity mean anything — **row t of the batch is
identical, to the last place, to the prompt truncated at t**. `cmd/llm -ppl`
is then llama.cpp's protocol line for line off the oracle's own build, over a
tokenization checked to be **297 193 identical ids**. **Ours is 4.0289 ±
0.02279 against llama.cpp's 4.0340 ± 0.02283 — −0.13%, a fifth of one side's
standard error**, in 7m58s over 145 chunks. Forty-eight layers, twenty-three
kernels and a bank that is not the reference's arrangement of the same bits
move perplexity by less than the corpus can resolve, and the sign is the one
L4a-5 predicts: the reference accumulates in fp16 above 8 output columns where
we accumulate in f32, so ours is nearer the model and, it turns out, nearer
the text. **So L8c's delta is stated against 4.0289, not against 4.0340.**

**L8c-1 grades the widths on it, and D3 does not survive.** `llm/sim.go`
round-trips a streamed dense weight through a candidate format on its way to
the device, and because a quantised kernel multiplies in fp16 (L8b-2) the half
it stages is the half that kernel would form — so this is the format's own
perplexity, not a model of it. It is checked where the answer is known
(`q8sym/32` over a Q8_0 tensor is an identity on **696 M real weights**) and
then through the whole graph, where **the default int8 bank, the fp16 arm and
`q8sym/32` on the Q8_0 tensors are a three-way identity at 2.0189** — L8a's
and L8b's bit-exactness claim measured at the corpus level rather than tensor
by tensor.

**~4.25 bits on everything streamed is +18.5% of perplexity**, and the cost of
4 bits runs **inverse to the bytes**: the gated DeltaNet is 46% of a dense
token and costs +1.49%, the lm head is 14% and costs +0.28%, while the
hyper-connection block is 14% and costs **+5.98%** — it is the model's
plumbing, and it wants **levels rather than a finer scale group** (halving the
group is worth 1.06 points where doubling the levels is worth 1.56). The best
mixed plan — DeltaNet and head at 4.5 bits, attention and hyper-connections at
6.5 — is **4.2699 against 4.0289, +5.98%, for 4.575 GB a token against 6.334
and a 52.9 tok/s ceiling**. And **a plan is not the sum of its families**:
uniform 4.5 bits is 18.5% where its four families sum to 11.7%, because errors
injected at 36 or 48 depths compound on L6b-3's x1.085 a layer rather than
adding. Two smaller results fall out — **ggml's 16-level `q4_0` beats L0d's
15-level `q4sym` by 1.13x at identical bits**, which is the baseline D7 was
decided against; and **D13's fp16 tail can be int8 for +0.01%**, though it is
worth only 0.6% of a token, so the reason to do it is deleting the two-plane
machinery rather than speed.

**L8c-2 pulled the one lever left, and it moves the wrong way.** Unsloth's
`imatrix_unsloth.gguf` is a GGUF our own reader opens, and `make_qx_quants` is
ported from ggml and checked against it — **bit-identical over 819 200 values**
through `reference/quant_ref.c`, once three details matched that are each a
case where the more accurate choice is wrong (`nearest_int` rounds half to
**even**, and both its accumulator and `sigma2`'s are `float`). Three arms,
because ggml does the scale search and the calibration in one function.
**Calibration is worse: 22.4% against round-to-nearest's 18.5%**, and per
family the whole loss is the hyper-connection block (+9.04% against +5.98%)
while the DeltaNet does not move at all.

**The mechanism is a gain error, not noise.** Round-to-nearest shrinks a
matrix by 0.08-0.12%; the search shrinks it by 0.32-0.37% and the imatrix
shrinks `hc_attn_up` by **1.35%** — and a systematic bias compounds on L6b-3's
x1.085 a layer where residual noise averages out. **What predicts it is how
evenly the calibration spreads inside a scale group**: `hc_attn_up` reads the
low-rank space, 1% of its 320 columns carry 90% of the energy, and its scale
is fitted to an effective **4.4 of 32** columns against `attn_q`'s 22.5.
Forcing the scale unbiased recovers a third of the regression, which confirms
the mechanism and does not close the gap. **The rule: an imatrix helps in
proportion to how evenly importance is spread inside a scale group — where it
is concentrated, what goes up is the systematic error.**

**L8c-2 left one caveat it could not test — every rung it measured was Q4_0's
*symmetric* form, where this checkpoint's own experts and unsloth's matrix's
own target are Q4_K — and L8c-3 tested it. It reverses D7 and resurrects D3.**
The symmetric form was chosen on L0d's reconstruction ladder, which L8c-1 had
already retired as a proxy — and the bytes it was chosen to save do not exist:
**ggml nests the min**, a super-block of eight groups of 32 carrying one fp16
pair plus 12 bits a group, which is **0.500 bits a weight, exactly what a
symmetric fp16 scale per 32 costs**. So `q4_k/32` and `q4_0/32` are both 4.500
bits and every comparison is free of a width argument. **Uncalibrated, the
asymmetric form is 2.06x cheaper over the whole corpus — 4.3124 against 4.6127,
+7.04% against +14.49% — and 3.2x on the hyper-connection block that decides
the plan**, against a reconstruction gap of 1.22-1.30x and L0d's claimed
1.043-1.053x; the proxy under-read it twice over. **And the imatrix works on
this form**: 4.5 bits on every streamed dense family is **4.1998 against our
4.0289, +4.24%, where the same matrix on the symmetric form makes it worse
still — 4.6588, +15.63%**. The 2x2 is the stage, and the sign flip is a
corpus-scale result rather than a screen: calibration is worth **−2.80 points**
asymmetric and **+1.14** symmetric, and it now helps every family it covers
(`hyper_conn` +9.04% to **+0.90%**, `deltanet` +1.48% to **−0.30%**).
**The mechanism is the second parameter, and it is measured**: a symmetric
group has one free parameter and it *is* the gain, so a calibrated fit can only
express its preference by shrinking the whole group — the extra shrinkage
calibration costs `hc_attn_up` is **1.24 pp symmetric and 0.47 pp asymmetric**,
and on `attn_qkv` it is gone. L8c-2's rule keeps its clause and gains one: an
imatrix helps in proportion to how evenly importance is spread inside a scale
group **and how many parameters that group has to express a preference with**.
The port is bit-identical to `ggml_quantize_chunk` over 204 800 values an arm
and **in f32**, not merely in the half. One departure from ggml, and it is on
the tensor the question is about: K-quants want `k % 256 == 0` where
`hc_*_up` is 320 wide, so its super-block is the whole row — ten groups,
4.475 bits.

**L8c-4 builds it, and the bank is the format rather than a neighbour of it.**
The 4.5-bit asymmetric form is now §2.8's fragment tiling with a *nibble*
where L8a puts a byte — a `uint` is eight consecutive k of one output column —
plus ggml's own sixteen-byte super-block record per (n-tile, super-block,
row), which is **4.500 bits a weight exactly**. One encoder serves the
simulation and the bank, so on the lm head the real bank's **248 320 logits
are identical** to the same format's through `sim.go`, and its eight-chunk
perplexity is `results/l8c_asym.csv`'s simulated row to four decimal places.
**The head is 0.675 GB to 0.358, 1.89x at one token at 210 GB/s of a 242 GB/s
bus, decode 24.66 tok/s to 25.75 (1.024x llama.cpp, ahead of it for the first
time), and 4.0621 against our own 4.0289 —
+0.82%.** Three things came with it. **D14 gains a clause**: the head's B is
357 MB and fits the MALL at no width, so the narrower bank is 1.08x at *512
rows* too, where the hyper-connection block's was 1.09-1.15x slower. **The
GEMV is not bit-exact against the GEMM on this bank and cannot be** — a
K-quant group is affine, so there is no pair of exact halves to multiply and
the register path carries one fewer rounding, rms 2.29e-04 against the Q8
pair's 7.84e-06. And **the per-family screen under-reads by 2.05x**: `lm_head`
screens at +0.40% over eight chunks and measures +0.82% over 145, which is
L8c-0's warning one level down and applies to every per-family row the plan
was assembled from.

**L8c-5 puts the largest dense family on that bank, and the finding is not
the speed.** The gated DeltaNet is 36 of the 48 layers and 2.247 GB of a
6.334 GB token — 46% of the dense half — and it needed no new kernel and no
new format: `-DQ4B` already had the arms, including the fp16 tail arm nothing
had exercised, and `ssm_alpha` and `ssm_beta` stay exactly where L8a left
them. **Decode is 25.75 tok/s to 28.9 — 1.15x llama.cpp — the block 10.9 ms
a token to 6.8 (1.60x), a layer 62.28 MB to 33.28, residency 81.57 GB to
80.53, and prefill 1070.1 tok/s at ubatch 2048 against 1052.8 and 667.0 at
512 against 655.2.** Two equalities say the bank is the format: a whole
layer's output off the real bank is *identical* to the same format through
`sim.go`, and the bank and the simulation agree **chunk for chunk** over
eight chunks.

**The finding is that L8c-4's screen warning is worse than a factor — it is a
sign.** L8c-3's per-family row puts this family at **−0.30%**, calibrated
4-bit weights beating the checkpoint's own Q8_0, and that was one of that
stage's headline results; measured over 145 chunks it is **+0.93%**. So a
per-family row does not bound its family's corpus cost and does not fix its
sign, and the two families now measured both ways sit +0.42 and +1.23
percentage points above their screens. **What does hold is additivity**:
`lm_head` at +0.82% and `deltanet` at +0.93% together measure **4.1012,
+1.79%**, against a sum of 1.75 — where the symmetric form compounded four
families' 11.7% into 18.5%. **And D14 gains its boundary**: prefill is
1.09-1.11x *faster* rather than slower, because the fused [16512, 2560]
projection is 44.9 MB at int8 and **23.8 at 4.5 bits** — it changes side of
the 32 MiB MALL, so sixteen re-reads a graph come off DRAM. Three blocks,
three answers, one rule.

**L8c-6 is the third family, and the obstacle it was left until last for is
a *record* rather than a format.** The hyper-connection block is the model's
plumbing — 97 mixers a pass, 0.695 GB of a token — and its up projection
reads the low-rank space, so its k is **320**. `asymSubBlocks` has divided
that into ten groups since L8c-3; what does not carry is
`get_scale_min_k4`, which *is* eight groups, four carried whole and four
whose high bits are stolen from the first four's spare ones, and has no
tenth pair to put anywhere. So the ten-group record is **twenty bytes** —
the fp16 pair, then ten twelve-bit fields at bit 12*j of a little-endian bit
stream — and its decode is a shift and a mask where ggml's is a branch.
Word alignment costs one byte a record, 0.025 bits a weight, so the bank is
**4.500 bits** where the simulation quotes 4.475. **And the fp16 tail
stopped being free**: L8b's split strands 32 low-rank rows on the fp16
plane, which on L8a's bank held the same numbers as the bytes behind them
and at 4.5 bits do not — they go through the encoder first, or the bank is
10% of that matrix more accurate than the format it claims to be and nothing
looks wrong. **Decode is 28.84 tok/s to 30.05-30.19 — 1.20x llama.cpp — the block
5.68 ms a token to 4.16 and 1.50x on its own one-token ladder, a mixer 8.12
MB to 4.76, residency 80.53 GB to 80.20, and prefill 669.7 tok/s at ubatch
512 against 667.2 and 1065.9 at 2048 against 1069.7.**

**The finding is that the screen is not biased in one direction.** L8c-4
said an eight-chunk per-family row under-reads by 2.05x; L8c-5 said it can
have the wrong sign. This family's screen is **pessimistic**: +0.90% at
eight chunks against **+0.31%** over 145. Three families now read +0.40 →
+0.82, −0.30 → +0.93 and +0.90 → +0.31, so a screen does not bound a
family's cost, does not fix its sign, and does not rank families against
each other. **With three families measured it also retires L8c-1's headline
shape**: in percentage points per GB of decode token bought back, `deltanet`
costs **0.87**, `hyper_conn` **0.93** and `lm_head` **2.59** — the two large
families almost exactly equal and the outlier is the head, where L8c-1's
"the cost of 4 bits runs inverse to the bytes" had the head cheapest and the
hyper-connection block 21x it. All three together are **4.1104, +2.02%**. Additivity holds
again and then some: three corpus deltas summing to 2.06% measure **2.02%**,
so the sum is now a slight over-estimate where at two families it was a
slight under — 0.04 pp either way, a sixth of the corpus's standard error.
**D14's hazard fires for the second time and is the smallest it has been**:
at ubatch 2048 the block is 1.02x slower, because its weights fit the 32 MiB
MALL at every width so the narrower bank adds only the unpack's ALU — 6 ms
of a 1921 ms pass, and the other three ubatches are faster. And **D12's
ladder did not move**: no rung's slab is a whole multiple of 4 KB on this
bank, so L7d's rule does not decide it, yet the spread is still 1.65x and
the winner is 32 as it was at int8.

**L8c-7 is the fourth family, and it needed nothing new.** The twelve
full-attention layers are 0.635 GB of a token, every matrix in the block is a
multiple of 256 wide, and `-DQ4B` already had every arm including the fp16
tail arm L8c-5 first exercised — so the stage is the wiring, and with it the
last of the two-valued `q8 bool` spelling (`q8Pipe`, `gemvPipe`,
`gemmBuilds`) leaves the package. **Decode is 30.05-30.11 tok/s to
31.41-31.57 — 1.25x llama.cpp — the block 3.5 ms a token to 2.3, a layer
57.94 MB to 32.22, residency 80.20 GB to 79.89, and prefill is faster at
every one of the four ubatches** (671.4 tok/s at 512 against 669.7, 1071.8 at
2048 against 1065.9). The bank is the format twice again — 97 664 projection
values and 17 920 output values identical to the same format through `sim.go`
on both arrangements of the tail, and chunk for chunk with the simulation over
eight chunks — plus a third check this family needed and the others did not,
because the decode GEMV has to derive the fused projection's tail split from
`pc.lowRank` and a column read out of the wrong plane is somebody else's
indexer query.

**D14's boundary is now measurable twice inside one block.** The fused
[13952, 2560] projection is 35.7 MB at int8 and **17.9 at 4.5 bits**, so it
changes side of the 32 MiB MALL exactly as the DeltaNet's did; the output
[2560, 6144] projection is 15.7 and 7.9 and fits at every width, so it pays
only the unpack — and on the block's own ladder it is **1.00x at 128 tokens
and 1.04x slower at 512** while the block is faster overall. Four blocks, four
answers, one rule, and this is the first block that contains both of them.

**The accuracy is the finding, and it is that this is the expensive family.**
`full_attn` alone is **4.0954 against 4.0289, +1.65%** over 145 chunks against
an eight-chunk screen of +1.05% — a fourth reading of L8c-4's warning, and the
second optimistic one. In percentage points per GB of decode token bought
back the four now read `deltanet` **0.87**, `hyper_conn` **0.93**, `lm_head`
**2.59** and `full_attn` **5.52**, so the shape is neither L8c-1's "inverse to
the bytes" nor L8c-6's "the two large ones are equal and the head is the
outlier" but **bimodal**, and the split is not size: the two families present
at 36 layers and 97 mixers cost ~0.9 pp/GB and the two present at 12 layers
and one matrix cost 2.6 and 5.5. **All four together are 4.1787, +3.72%,
against a sum of their four separate corpus deltas of 3.71%** — additivity to
0.01 pp, where the symmetric form compounded 11.7% into 18.5%.

**And the fifth family closes D13's last open clause.** `qsa_indexer` is the
fused projection's fp16 tail — the model's only BF16 weights — and it costs
**0.51 bits a weight over the whole layer**, 11.3% of the block at 4.5 bits
where the same tail was 5.7% at int8. Naming it in the plan puts it on the
plane and the layer becomes **4.500 bits exactly**. Grading it took two
contexts. At n_ctx 2048 the quantised indexer is **bit-identical over all 145
chunks** — the per-chunk `nll` columns match to six decimals — because
`top_k + ratio - 1` is 2051, the selection names every cell and the score is
discarded, which is L8c-1's argument measured as an equality. At **n_ctx
2560**, where `selWidth` is 2051 of 2560 and the selection is dispatched, it
is **4.0723 to 4.0725 over 116 chunks: +0.005%, one part in forty of a
±0.0231 standard error** — with all 116 per-chunk rows differing, so the
narrower indexer really does select different cells and the corpus cannot
tell. **0.028 GB a token for +0.005%**, against `full_attn`'s 0.299 GB for
+1.65%. Decode with it is **31.51 tok/s**, the block 2.1 ms a token and
residency 79.85 GB.

**What L8c-7 could not measure is not L8c-7's.** The obvious grading context
was 4096, and the whole-model graph does not complete above ~2560 rows: at 48
layers, in one staging, 2048 and 2560 tokens run at 1039 and 1091 tok/s and
3072 and 4096 never return — with no `LLM_DENSE_BANK` set and `-ppl` not
involved. The 4-layer prefix runs 4096 in 297 ms, `-ppl -ctx 4096` completes
at 4 and 12 layers, and the attention block alone at 4096 tokens in a
4096-cell cache with the selection live is 325 ms for twelve layers — so
neither the block nor the selection is implicated. A `SIGQUIT` puts the stall
in `Graph.flush → recorder.submit`, in `[syscall]`, one thread at 100% of a
core with the GPU at 2-3% and the shim's own 20-second fence timeout never
firing. **It is the open question this stage hands over, because a long
context is what this model is for.**

**The plan was 4.50 bits at +4.24%, and five families in, the measured
running total is 4.281 GB a token and +3.72%.** L8c-3's uniform asymmetric
plan projected 4.264 GB and a 56.7 tok/s ceiling; `lm_head`, `deltanet`,
`hyper_conn`, `full_attn` and `qsa_indexer` between them are **4.281 GB, a
56.5 tok/s ceiling and +3.72% measured over 145 chunks** — every asymmetric
plan measured still dominates L8c-1's recommendation on both axes at once,
and the four corpus deltas add rather than compound. **One family is left**:
`ple_proj`, the PLE block's fused key/value projection, 0.033 B parameters run
**once** at layer 1 — 0.017 GB a token, and the one block in this vertical
that never got L8a's int8 bank either, so it is two bank stages in one.

**After it, the bank is not where the decode time is, and has not been since
L8c-5**: 31.5 tok/s measured against a 56.5 ceiling is 56%. What is left on
that side is the two things the simulation cannot reach — the F32 router at
fp16 (+1.5 tok/s of ceiling) and the 512 expert banks at ~4.25 (+3.0), which
are 1.504 GB of every token and 35% of what a step now reads, and which are
already Q4_K and already the calibrated part of this checkpoint. §1.1's W4A8
layout stays open beside all of it: it is a *symmetric* weight and an
int8-activation one, where every rung built here multiplies an fp16 A, and an
asymmetric version folds the min into the epilogue as a per-group correction
times the activation's column sum — one extra reduction over A and no change
to the matrix core. And **the graph does not complete above ~2560 rows at 48
layers**, which is the one open item that is not about a width. Past all of it
is **phase 3**, where MTP speculation is worth 1.5-1.8x and batching amortises
the dense half completely.

---

## The priority list  *(set 2026-09-18 — the review is `LLM2.md`)*

**P0, P1, P1a and P1b are done** (2026-09-18) and are struck through below;
**P1c is next**. P1b was the largest item on the list by a factor of four and
it evaporated on measurement: the rungs were chosen against the MALL but they
were chosen *right*, so the honest re-screen changes nothing and the 1.9 ms
it promised does not exist. What it found instead is a **1.33 ms environment
gap** — the same dispatch is 8-22% slower inside a real step than alone on a
cold bank — which is bounded under P1c's 1.96 ms and is a new question, not a
rung.

The 2026-09-18 review re-derived the decode budget and found the bank is no
longer the binding constraint: 31.5 tok/s measured against an honest ~53
ceiling, so **~12 ms of every 31.7 ms token is unexplained by bytes at any
achievable rate** and nothing had attributed a decode step since L8e. **P1
attributed it, to the dispatch**, and the 12 ms is 5.54 ms of weight-streaming
dispatches below 227 GB/s, 3.46 ms of dispatches that stream no weight, and
3.01 ms of host. It also found and fixed the largest host item on the way —
the n-gram gather was **sixteen serialised major page faults a token** — which
is **31.51 → 32.72 tok/s** on its own. **P1a then took the top item off that
table and corrected what the rest of it means**: the hyper-connection
boundary's two weightless dispatches are one (**32.59 → 32.93 tok/s**,
bit-exact), and the block's low GB/s turns out to be a *grid* — a one-token
dispatch here is short of workgroups, and 168 of them is 9.26 us where 672 is
5.54 for identical bytes. The order below follows from that table; each
item's full plan and gate is in `LLM2.md`.

| # | item | why it is where it is |
|---|---|---|
| ~~**P0**~~ | ~~**The >2560-row stall**~~ — **done**, and it was a 2 s ring watchdog rather than anything about this model: the recorder now chunks a submit by *time*. `-graph` returns at 4096 (**1174.6 tok/s, 3.00x**) and 8192 (**1213.5, 3.10x**), and `-ppl -ctx 4096` completes at 48 layers at **PPL 3.9392**. | The blocker is gone, so everything long-context below is now measurable. [Write-up](research/p0-ring-watchdog.md) |
| ~~**P1**~~ | ~~**Re-attribute the decode step**~~ — **done**, and the step now sums: 30.52 ms over 36 dispatch labels and six host phases, residual 5 us. The ~12 ms is **5.54 slow + 3.46 weightless + 3.01 host**. The gather was **16.3 serialised major faults a token** and going parallel is 7.0-7.5x, worth **31.51 → 32.72 tok/s** by itself. The pre-recorded command buffer is priced at **1.96 ms** and is *fourth*, not first; the leaders are `hyper_conn` at 114.6 GB/s (485 dispatches a pass) and `moe.down` at 147.5 against `moe.up`'s 200.6. Idea 9 closed with it. | The blocker on every estimate below. [Write-up](research/p1-decode-attribution.md) |
| ~~**P1a**~~ | ~~**The hyper-connection block's 485 dispatches**~~ — **done**, and it was two answers rather than one. The weightless half fused: the scatter that closes a mixer and the norm that opens the next are the same 2560 values per (token, stream) written and read straight back, so `llm_hc_cn.comp` does both in one pass over registers — **1501 dispatches a pass to 1407, 30.69 ms to 30.37, decode 32.59 → 32.93 tok/s**, and **identical to the last place** on `res`, `xn` and `mixed`. The bank half was not a fusion question: the down projection's decode ladder reads the same 1.94 MB off the same bank at a grid that varies twenty-fold, and **168 workgroups is 9.26 us where 672 is 5.54** — so `up_m1`'s 10.87 us at **160 workgroups** is `down_gemv8`'s number at `down_gemv8`'s width, not MODE 1's M=1 waste. What pins it is the collapse's 64-column block, and unpinning it is priced at **0.22 ms a token** and not taken. | The step's GB/s did not move and the label count did, which is the finding. [Write-up](research/p1a-hyper-connection-shape.md) |
| ~~**P1b**~~ | ~~**`moe.down`'s rung, and the shared expert's**~~ — **done, and no rung moves.** The whole one-token MoE ladder in `l8e_moe.csv` was an L3 measurement — every GEMV down rung read 267-357 GB/s of a 242 GB/s bus — so it was re-run on **sixteen cold banks at `-iters 1`** (25.6 GB, so `ProfileSweep` never re-reads a bank warm). Two runs agree to a median ratio of 0.9991, no rung reads above the bus, and **the ranking is unchanged on every axis**: v64w4/v16w4 routed, v64w4/v32w4 shared, k40 router — the plan the model already runs. The honest floors are up 99.0 us at 186 GB/s, down 59.3 at 207, shexp 22.7 + 10.9, router 13.4; the whole model sits 8-22% above them (down 72.2, the widest), a **1.33 ms a token** environment gap that is not a rung choice. D16's up-mode contradiction also closes: cold, v16w4 loses to v64w4 by 1.15x, the same side as the whole model. | The 1.19 + 0.73 ms the review priced was the MALL's arithmetic, not a mis-chosen row block. [Write-up](research/p1b-moe-decode-rescreen.md) |
| **P1c** | **The pre-recorded decode command buffer** (idea 2) | 1.127 ms of recording plus 0.828 of hand-over, **6.4% of the step, ~+2.2 tok/s**. The decode graph is shape-stable: same 1501 dispatches, same buffers, only the position and the token id change. |
| **P2** | **`ple_proj`, and close L8c** | Half a day, two bank stages in one, and D13's payoff: delete the two-plane fp16-tail machinery now that the exception list is empty. Ends with the complete plan's 145-chunk number. |
| **P3** | **The shipped-widths decision (D18)** | The knapsack: additivity + per-family corpus deltas make plans composable, and uniform 4.5 is provably not optimal — `full_attn` costs 5.52 pp/GB where `deltanet` costs 0.87. Sim-grade `q5_k` on `full_attn` first (8 minutes, no kernel), screen the winner on a second corpus. |
| **P4** | **The router at fp16 and the experts at ~4.25 bits** | +4.5 tok/s of ceiling. The expert half is a transcode, not a kernel — `llm_moe_gemm/gemv` already read Q4_K — and L8c-3 says the calibrated form behaves. Grade each on 145 chunks separately; D4 holds. |
| **P5** | **MTP speculation** | ×1.5-1.8 on everything above, so it loses nothing by going after P0/P1. Needs the rollback design first: a rejected draft rewinds 36 recurrent states, both rings, the KV and the host id list, and verification runs at M = 2-8 where D15 refuses the GEMV — the crossover has never been measured. |
| **P6** | **Batching** | Pending the product question: will the API serve more than one stream? Each sequence owns 113 MB of DeltaNet state. If yes, batching may beat MTP for the same effort. |

Parked, unchanged: W4A8 with the asymmetric epilogue, the unpack prefetch,
the fp16 residual, the hot-expert fast path, the float-atomic combine, the
vision tower.

## The number to beat

`llama-bench`, build `cff184438`, Vulkan on RADV STRIX_HALO, the model as
shipped, `-ngl -1`, mmap on, two repetitions, nothing else on the GPU:

| test | tok/s | against |
|---|---:|---|
| pp512 | 313.62 ± 1.74 | |
| **pp2048** | **388.60 ± 1.89** | L2a re-measures **391.42 ± 1.13** at its best ubatch, and attributes the 5x |
| pp8192 | 392.95 ± 1.04 | prefill plateaus, it does not scale with the chunk |
| tg32 | 25.12 ± 0.03 | |
| **tg128** | **25.15 ± 0.02** | **65.8%** of the 242 GB/s bus |

A real generation agrees: `llama-completion -n 64 --temp 0` reports 24.66
tok/s and coherent text. Reproducibility across two separate runs is 0.5% on
pp512 and 0.8% on tg128. And the accuracy reference, on the same checkpoint:
**PPL = 4.0340 ± 0.02283** over wikitext-2's 145 chunks at n_ctx 2048 — the
figure L8c's re-quantisation has to stay near, at the same corpus, context and
chunking. [Write-up](research/l1-baseline.md)

**So the target moves.** Phase 1's job is the 38.2 tok/s the bytes allow, not
llama.cpp's 25.15 — decode leaves a third of the bus unused in the reference
implementation. Phase 2's ~67 is then **2.7x the reference**, not 1.8x.

**L2a has since re-measured the prefill row**: `-ub 512` is llama.cpp's best
ubatch and it does **391.42 ± 1.13**, 0.7% from L1's separate 388.60 ± 1.89.
256/1024/2048 give 347.71 / 380.91 / 336.72 — bigger is *worse*, for a reason
finding 5 below explains.

## What L0 established

> **L0a: the DRAM bus does not care how big the weight bank is.** 236.4 GB/s
> reading a **64 GiB** bank at random against 237.1 GB/s reading a 1 GiB one,
> `rand/seq` 1.00x in every cell at both of the model's real expert slab
> sizes. That *is* §0.4's 236 GB/s ceiling, not merely near it — no TLB cliff,
> no page-table cost, no penalty for scattering. So decode stays linear in
> bits/weight at the size a 180 B model needs, which is what every number
> below depends on. At `-bankgib 80`: 23 buffers, 85.9 GB, every page written,
> **237 GB/s at every prefix** — past UD-Q4_K_XL's 82.52 GB resident core.
> [Write-up](research/l0a-bank-range.md)
>
> **L0b: neither does the memory type, and the heap sizes are fiction.** All
> eight types that can back a storage buffer read **236.0-237.4 GB/s** — 0.57%
> across 32 cells, heap 0 / heap 1 = **1.0004**. Type 3 reserved **105.0 GiB**
> against a heap RADV calls 83.79. **The ceiling is physical RAM.**
> [Write-up](research/5.1-memory-types.md) · `results/bank.csv`
>
> **L0c: §2.2's unexplained 1.24x was locality in the scale plane, and it is
> gone.** A k-major scale plane takes `gate_up` at QBLOCK=32 from **4.71 ms to
> 3.72, 1.27x**, past row-major QBLOCK=128's 3.79. **The fine scale block
> accuracy wants now costs 1.0% instead of 14.2%.**
> [Write-up](research/l0c-scale-plane.md) · `results/moe.csv`
>
> **L0d: W4A8 is safe, and above ~5 bits/weight the activations are the
> floor.** int8-per-token activations cost **1.13x** the error of fp16 ones,
> so the 3x cliff between W4A8 and W4A16 is bought cheaply — but int8
> activations *alone* contribute 2.87e-2 where an 8-bit weight contributes
> 3.4e-3, so **W8A8 is 99% activation error**. Asymmetric Q4 is worth 5% at
> equal bits, not 2x. And a bug fell out: `quantizeQ8`'s fp16 block scale goes
> subnormal under maxAbs 7.75e-3, worth **14x** on one real tensor in fourteen.
> [Write-up](research/l0d-quant-error.md)

## What L2a established — the 5x, attributed

`GGML_VK_PERF_LOGGER=1` puts a timestamp query around every dispatch in
llama.cpp's Vulkan graph. Both of L1's candidates are **wrong**, and the
answer is better than either. [Write-up](research/l2a-prefill-attribution.md)
· `results/l2a_prefill_ops.csv`

> **L2a-1: the architecture is innocent.** `GATED_DELTA_NET` is **1.6%** of a
> prefill graph — 36 dispatches, 74.1 ms of 4556. The whole DeltaNet core is
> 2.7%, full attention plus the QSA indexer and its top-2048 selection 1.5%.
> **The candidate "this architecture has a great deal of prefill that is not
> matmul" accounts for 4.5% and cannot explain a 5x.** Nor is it host
> overhead: **93% of the un-instrumented wall clock is GPU kernel time**.
>
> **L2a-2: it is the hyper-connection block, and half of that is free.**
> Per graph: MoE experts **35.7%**, dense matmul **26.2%**, elementwise glue
> **30.6%** (2453 dispatches), tiny-N F32 matmul **12.2%** at `-ub 512`. The
> worst single line is `MUL_MAT f32 m=4 n=512 k=10240` — `hc_*_inject`, a
> `[10240, 4]` F32 projection run 95 times a graph — at **31.8 GFLOP/s, 0.06%
> of peak, 10.3% of prefill**. It reads the same `xn` that `hc.down` reads, so
> it is four more output columns on a matrix that already has 320: **fusing it
> deletes a tenth of prefill.** `CONCAT` is the other scandal, at 48 GB/s
> falling to **13 GB/s** as the ubatch grows.
>
> **L2a-3: against our own measured kernels the chunk is 2.25-2.53x.**
> `results/shapes.csv`'s best variant per shape beats llama.cpp by **5.08x at
> `hc.down`** (N=320), 2.96x at `moe.shared` (N=640), 2.15x at `attn.o`, and
> only 1.11-1.15x on the two widest-N shapes — **we win exactly where N is
> small.** §2.2's already-built Q4 grouped MoE block is **642 ms against 1626,
> 2.53x**. Total: **711 tok/s with the glue as it stands, ~1150 with three
> quarters of it fused into epilogues.** So **L6 is an afternoon, and ~1150
> — not ~2000 — is what it should be scoped against.**
>
> **L2a-4: `-ub 512` is a MALL-sized optimum, so do not raise it.** 256 / 512
> / 1024 / 2048 give 347.71 / **391.42** / 380.91 / 336.72. Per token the MoE
> does get cheaper with a bigger ubatch (0.78x) — but the glue gets **1.57x**
> more expensive, because the `[10240, T]` **F32** residual is 20.97 MB at
> T=512, **just inside the 32 MiB MALL**, and 83.9 MB at T=2048. `MUL` falls
> from 362 GB/s to 294, `CONCAT` from 48 to 13. **Keep the residual fp16 and
> fuse the traffic away rather than cache it.**
>
> **L2a-5: decode's gap is dispatches, not GEMV.** llama.cpp's decode matmuls
> are already at **198-230 GB/s** (`lm_head` 675 MB in 2.93 ms = 230). The
> 6.334 GB a token needs 26.2 ms and it spends 41.3; the missing 15.1 ms is
> **2896 dispatches that read no weights, costing 8.9 ms — 22% of the step**,
> an order of magnitude past L1's 0.62 ms estimate for the same graph.
> **L7 is won by not dispatching 3837 kernels, not by a better GEMV.**

## What L2b established — the block runs, and the oracle is not exact

`llm/` is the vertical's package: a `Config` read from the checkpoint, the
hyper-connection block, and `reference/eval_dump.c`, which writes **whole**
tensors of a real llama.cpp pass where `llama-eval-callback` prints three per
axis. [Write-up](research/l2b-hyper-connections.md) · `reference/out/llm/`

> **L2b-1: the block matches, and four of its seven mixers match to f32
> round-off.** `token_embd`'s gather and `hc_init` are **bit-exact**; the
> combine is 1e-09 rms; the F32-weighted `hc_norm` and `hc_inject` are 1e-07
> and **3e-06 relative** on values to \|71.7\|; and `hc_gate` — through two
> Q8_0 matmuls, a SiLU and a sigmoid — is **6.2e-08 rms in four of the seven
> mixers of the four dumped layers**. Nothing wrong about the formula, the
> weight indexing or the layout survives that.
>
> **L2b-2: llama.cpp evaluates a Q8_0 matmul over int8 activations, and
> modelling it is worth 233x.** `quantize_q8_1.comp`: blocks of 32, `d =
> amax/127`, `round(x * 1/d)` **with the f32 reciprocal**, and the scale
> *stored* fp16. `hc_gate-0` goes from **3.143e-03 rms to 1.347e-05**.
> Rounding the activations, the weights or both to fp16 changes **nothing**
> (3.14e-03), so this is not a precision effect — it is a different
> arithmetic. Quantising with an already-fp16 scale, the obvious misreading,
> is 35x worse than the real thing.
>
> **L2b-3: so L2's gate changes.** "Matches `llama-eval-callback` to fp16
> tolerance" is unreachable *and* meaningless downstream of a Q8_0 matmul.
> The criterion is an **rms bound under the reference's own numerics**
> (`llm.Numerics`, `Exact` vs `RefQ8`), and maxAbs is a logged diagnostic
> rather than a bound, because an int8 grid makes the error heavy-tailed:
> one flipped rounding among `lo`'s 320 values moves all 10240 gate outputs.
> Three of the seven mixers keep 6e-06 to 3.4e-04 rms for that reason,
> bounded but not explained.
>
> **L2b-4: and D6 is cheaper than it looked.** The baseline **PPL 4.0340**
> was measured with **W8A8 already running on the dense tensors**. L0d priced
> int8-per-token activations at 1.13x fp16's error and called W8A8 "99%
> activation error"; that is not a projection about our bank, it is what the
> reference implementation does. **D3 and D6 give up no activation axis the
> reference keeps**, and L8c's perplexity comparison is like-for-like on it.

## What L2c established — the block runs on the GPU, and it is 4.24x

`shaders/llm_hc_norm.comp`, `llm_gemm.comp` (two of its three epilogues) and
`llm_hc_combine.comp`, driven by `llm/gpu.go`. The first kernel of the
vertical, and the one L2a put at the front of the queue.
[Write-up](research/l2c-hc-kernel.md) · `results/l2c_hc.csv`

> **L2c-1: four dispatches against sixteen, and 226.6 ms becomes 53.4.** Per
> 512-token graph, against the lines of llama.cpp's own graph that are nameably
> this block: the grouped norm 16.7 → **5.3 ms**, `down` + `inject` 163.3 →
> **24.7**, `up` 30.7 → **16.8**, the combine's `REPEAT` 15.9 → **6.6**.
> **4.24x, and 14.9% of the whole 1164.7 ms graph removed by one block** — and
> the rest of the block's glue (the gamma multiply, the gate's sigmoid, the
> `xn*gate` multiply, the collapse's adds, the combine's four elementwise
> passes) is **absent from our graph rather than faster in it**, so that is the
> conservative reading. If nothing else changed, the reference's 391.4 tok/s
> would become **~460**.
>
> **L2c-2: two tensors never exist, and that is where the time goes.**
> `inject` is four more output columns on a `[336, 10240]` fused weight — L2a's
> worst single line, 10.3% of prefill at 31.8 GFLOP/s, deleted. And the
> `[10240, T]` gate — 21 MB a mixer at 512 tokens, the whole MALL — is consumed
> inside the up projection's epilogue, which applies the sigmoid to its
> accumulators and collapses the four streams against `xn` through LDS. That
> needs the four streams of a feature in one workgroup, which is a **weight
> permutation** (`packUpB`), and `TestHCGPUUnpermutedUpIsWrong` is the negative
> control: unpermuted is a plausible tensor built from the wrong four columns.
>
> **L2c-3: it is 38x nearer the f32 model than the oracle is.** All eight
> mixers of layers 0-3, each from llama.cpp's own residual, agree with L2b's
> CPU reference to **1e-04 rms** — `maxRel` on `hc_norm` is 4.9e-04, which *is*
> fp16's step. Against llama.cpp the residual is 2.3e-03 to 6.5e-03, which is
> **L2b's figure for the CPU reference**, not ours: the reference is the side
> computing in int8. Every rung of both ladders is bit-identical to every
> other, which is what checks the M padding at a 7-token prompt.
>
> **L2c-4: the MALL cliff is ours too, and fp16 residual is now worth 1.5x.**
> Between 512 and 1024 tokens `norm` goes **575 → 167 GB/s** and `combine`
> **695 → 170** — above the DRAM bus on the near side, below it on the far —
> because the fp32 residual is 21.0 MB at T=512 and 41.9 MB at T=1024 against a
> 32 MiB MALL. Per token the block is cheapest at 512 and **1.5x** more
> expensive at 2048. L2a inferred "keep the residual in fp16" from llama.cpp's
> profile; this measures it on our own kernel. Not taken here — the residual
> accumulates across 97 combines, so it is an accuracy decision for L6/L8.

## What L2d established — the n-gram block, and a hash that has no tolerance

`per_layer_token_embd` is a quarter of the checkpoint and it is not a matrix.
[Write-up](research/l2d-ple.md) · `results/l2d_ple.csv`

> **L2d-1: the trigram hash is bit-exact.** Sixteen rows of a **320 001 536-row**
> table per token, chosen by a 64-bit mix of the token and its two
> predecessors against sixteen near-prime vocabularies whose offsets tile the
> table exactly. `ple_embd` matches llama.cpp to **0 ulp** — which is the only
> tolerance a lookup has, since a wrong index returns a real embedding
> belonging to another n-gram. The block on top of it matches to **7.7e-08
> rms** (`ple_gate-1`), 2.1e-09 (`gated`) and 1.3e-08 (`conv_out`).
>
> **L2d-2: it closes L2c's one open comparison.** Layer 1's attn mixer reads a
> residual this block has written into, which is why it could not be checked
> against the trace before. Fed through the block it lands where the other
> seven do — 2.4e-04 on `hc_norm`, 2.9e-03 on the gate — so **the two stages
> compose**.
>
> **L2d-3: three dispatches against about thirty, and it is 0.19% of a graph.**
> The key and value projections read the same gathered embedding, so they are
> one fused `[12800, 2560]` matrix; everything between them and the convolution
> is one workgroup per (token, stream); and the dilated depthwise convolution
> carries its SiLU and the residual add. On the nameable lines we are **0.88x
> — slower** (2234 us against 1974), and the projections alone are **1.06x
> while reading twice the weight bytes**, because llama.cpp's are Q8_0 and ours
> are fp16. This block runs **once**, so none of that matters; it was built to
> be correct and to be on the device.
>
> **L2d-4: two levers, and both were free.** The fused projection's B is
> **65.5 MB, twice the MALL**, so it is read M/BM times: BM 32 → 128 is
> **3.3x** at 2048 tokens for identical arithmetic. And making the *channel*
> the fast grid axis instead of the token — so the resident workgroups cover
> one token's 40 KB row rather than one channel block of forty tokens — is
> **2.9-4.0x** on the convolution and 1.14-1.26x on the gate. Same work, same
> bytes; §5.1b's traversal penalty in a kernel written without thinking about
> it.

## What L2e established — the attention layer, and two more oracle numerics

`llm/attn.go`: the fused query/gate projection, interleaved M-RoPE, the QSA
indexer and causal GQA, checked against layer 3 of the dump.
[Write-up](research/l2e-attention.md)

> **L2e-1: it matches, and the chain composes.** The indexer's pooled keys to
> 1.7e-07 rms, its score to **3.5e-07 relative**, the fused query/gate split to
> 1.2e-06, the attention output to 2.4e-04. And from `l_last-2` through our
> mixer, our indexer, our attention and our combine, **`hc_combine-3` agrees
> to 9.75e-06 rms** — four stages, none of the intermediates llama.cpp's.
>
> **L2e-2: the indexer's key cache is fp16, worth 2072x.** The keys are
> rounded to halves *between* being projected and being pooled, which is why
> `indexer_k_raw` is exact and `indexer_k_pooled` three lines later was
> 3.49e-04 until it was modelled. `llama_memory_hybrid_idx` passes the
> context's `type_k` straight through.
>
> **L2e-3: a matmul between two F32 tensors is an fp16 matmul, worth 69x.**
> The indexer's score is F32 x F32 and the reference computes it on the fp16
> matrix cores: 3.52e-03 rms in f32, **the same 3.52e-03 in float64**, and
> 5.11e-05 with both operands rounded to halves. **So the fp16 kernels L2c and
> L2d run are not giving up anything the reference keeps** — they are the same
> arithmetic.
>
> **L2e-4: the block structure is the cache's, not the prompt's.** Seven tokens
> in a 256-cell cache (flash attention pads it) make **one** real block of 4;
> the other 63 pool cell 0 four times because `blk_cells` is zero-filled, and
> the unpooled tail goes to a spare block carrying a 1e9 "always visible"
> marker — finite, so it can never meet a -inf and make a NaN.
>
> **L2e-5: and the selection cannot be tested yet.** `top_k + ratio - 1` is
> 2051 against 256 cells, so it names every cell; running the attention with
> it is bit-identical to running it without. llama.cpp's `TOP_K` is also a
> *selection and not a sort* — it returns the identity permutation where the
> scores are not monotone — which is unobservable at this width. **L4's 4k
> context is where both become testable.**

## What L2f established — the layer on the device, and a ladder that inverts

`llm/gpu_attn.go`, `shaders/llm_attn_{pack,idx,score,wmma}.comp`: six
dispatches, the plain GEMM arm twice and four kernels of this layer's own.
[Write-up](research/l2f-attention-gpu.md) · `results/l2f_attn.csv`

> **L2f-1: six dispatches against nineteen, and 66.3 ms becomes 27.6.** Per
> 512-token graph, against the lines of llama.cpp's own graph that are nameably
> this layer: the five projections 28.2 → **18.4 ms**, the norms and ropes 3.7
> → **0.8**, the indexer's score 0.7 → **0.8**, flash attention 25.2 → **2.1**,
> the output projection 8.6 → **5.6**. **2.40x, and 5.7% of the whole graph
> becoming 2.4%.** The selection llama.cpp also runs — `TOP_K` and its
> `GET_ROWS`, 24 dispatches and 2.4 ms — is left out of the comparison
> entirely, and so is every elementwise line this block shares with the others.
> **L4b puts the selection back in on both sides**: one dispatch of ours
> against those 24, and the layer 68.8 ms against 29.6, **2.32x**.
>
> **L2f-2: one of those rows is not like for like, and saying so needs a
> number.** llama.cpp's flash kernel issues the whole **[512, 2048]** rectangle
> its cache defines — ggml's own FLOP count says so — where ours walks the
> causal triangle of a 512-token prefill, an eighth of the work. On **rate**
> the two kernels are **18.9 TFLOP/s against 12.3, 1.54x**, and repricing that
> row at our shape makes the layer **1.60x** rather than 2.40x. Both are real:
> 2.40x is what a 2048-token prefill in four chunks costs either side, 1.60x is
> what the kernels are worth at equal work.
>
> **L2f-3: six of the reference's matrices are one.** The query, its gate, the
> key, the value and both of the indexer's projections read the same normalised
> residual, so they are one `[13952, 2560]` weight — L2a's argument for
> `inject` and L2d's for the PLE value, at the largest scale in the model. Then
> the per-head norm, the interleaved M-RoPE and the fragment tiling for q, k
> and v are **one dispatch over a (token tile, plane) grid**, because all three
> are column ranges of that output ending in the same layout. And **the output
> gate folds into the attention epilogue**, which already holds the tile for
> the softmax divide — a `SIGMOID`, a `MUL`, a `CONT` and a `[512, 6144]` round
> trip through DRAM, deleted.
>
> **L2f-4: the two projections want opposite schedules on the same kernel.**
> The fused projection's B is **71.4 MB, 2.2x the MALL**, so it is read M/BM
> times and the widest row block wins everywhere above 64 tokens, by 3.2x at
> 2048. The output projection's B is **31.5 MB and fits**, and on that side of
> the line reuse buys nothing while occupancy does: at 64 tokens BM=128 is the
> **worst** rung by 2.1x, at 512 BM=64 wins, and only at 1024 does BM=128 win
> back. **Same kernel, same arithmetic, inverted schedule, decided by which
> side of one cache the weight falls on** — so there are two schedules, and the
> pair is the exact winner of a 27-plan ladder at five of six lengths.
>
> **L2f-5: and the attention ladder ends one rung narrower than every other in
> the repo.** qt1_kt2 **170 us**, qt1_kt4 223, qt2_kt4 241 — monotonic in how
> much a wave holds live. §3.3's conclusion is unchanged (intensity is inert,
> the register file decides); headDim **256 against the DiT's 128** just makes
> it decide sooner, a wave already carrying 16 query fragments and 16 output
> accumulators across the key loop.
>
> **L2f-6: 5.9e-05 rms, and 19.3x nearer the f32 model than the oracle.** Every
> tensor agrees — the six column ranges to 1.7e-04, the pooled indexer key to
> 3.1e-04, the score to **3.8e-07 relative**, the cell mask **exactly** — and
> the layer's output sits at 5.9e-05 against L2e's f32 reference where
> llama.cpp sits at 1.14e-03. That gap is the reference's int8 activations
> under four Q8_0 projections, not ours. One numeric is now **reproduced rather
> than modelled**: the indexer's fp16 key cache is fp16 here because the arena
> is.

## What L3a established — every layer now has a reference, and the oracle has a bug

`llm/deltanet.go`: the fused qkv projection, the depthwise causal convolution,
the L2 normalisation, the two F32 gate projections and the delta rule itself,
against all eighteen tensors llama.cpp names inside layer 0.
[Write-up](research/l3-deltanet.md)

> **L3a-1: it matches, and the recurrence is at f32 round-off.**
> `attn_output-0` is **3.11e-09 rms**, the 786 432-value `new_state-0`
> **1.91e-08**, and `linear_attn_out-0` **1.31e-07** — with `state_predelta-0`
> checked to be exactly zero, so those are the recurrence's own numbers and
> not the fixture's. Layers 1 and 2, whose input the PLE and hyper-connection
> blocks have already written, agree at 6e-10 on the output; their
> `linear_attn_out` sits at 1.2-1.5e-05 and all of that gap appears at the
> last Q8_0 matmul, which is L2b-3's heavy-tailed int8 grid over a wider input.
>
> **L3a-2: the reference does not chunk, so neither does the gate.**
> `cparams.fused_gdn_ch` sends a multi-token batch to `ggml_gated_delta_net`,
> whose Vulkan kernel walks the tokens **one at a time**, one workgroup per
> head, with the state in registers across the loop.
> `build_delta_net_chunking` — 270 lines of `solve_tri` and cumulative-sum
> decay masks — is in the tree and is not what ran; `q_conv_predelta` being
> dumped with **16 heads and not 48** proves it, because the non-fused path
> repeats q and k first. So the chunked form is L3's *performance* question,
> and what it has to beat is **438 us a layer at ubatch 512** (2058 at 2048).
> *(L3b corrects the shape: the dispatch is `{H, n_seqs, S_v}` with one state
> column per workgroup, which on a 64-wide wave is **6144** workgroups and not
> 48 — the source's "one workgroup per head" is the loop's structure, not the
> grid's.)* **vLLM answers it the other way** — `chunk_gated_delta_rule` at
> chunk size 64 for prefill, a fused recurrent update only for decode — so L3b
> is a real open question that two teams have split on, and
> flash-linear-attention's four stages (`scaled_dot_kkt` → `solve_tril` →
> `delta_h` → `fwd_o`) are its blueprint. **L3b answers it: neither team's
> choice is worth much here, because the recurrence is 13% of the layer.**
>
> **L3a-3: the 48 value heads read the 16 key heads by `h % 16` — because
> this checkpoint's V heads are permuted.** ggml's own two broadcast rules
> disagree here (`ggml_repeat` tiles, `ggml_mul_mat`'s batch broadcast
> divides), they agree on 16 of the 48 heads, and the fused op tiles:
> **3.11e-09 against 2.71e-03, 870 000x.** **vLLM divides** — `i_h // (H //
> Hg)` in flash-linear-attention — and both are right, because llama.cpp's
> converter (`_LinearAttentionVReorderBase`) transposes the (16, 3) head grid
> into tiled order across **seven tensor families** so that ggml's cheap
> broadcast lands on the right head. So `h % 16` is a fact about the **GGUF**,
> not the architecture — a live hazard for L8, below. The state is also
> **stored transposed** (ne[0] is the key axis), which is not a quirk: it
> makes all three of a token's operations walk contiguous columns, and that is
> what lets the state stay in registers.
>
> **L3a-4: the oracle's build has a GDN normalisation bug, and it is worth
> 15x.** `build_gdn_l2_norm` divided by `max(|x|, eps)` until llama.cpp commit
> 5fdfa6282, *"models : fix GDN normalization from `max` to `rsqrt`"*
> (#28068), and this repo's build — **cff184438** — predates it. That build is
> also L1's `pp2048 388.60`, `tg128 25.15` and **`PPL 4.0340`**. Reproducing
> the bug takes `k_conv_predelta` from 9.16e-07 to **6.01e-08** and the layer
> output from 1.91e-06 to 1.31e-07, so `DeltaNetConfig.QKNorm` selects the
> spelling, the model's `rsqrt` is the default and the tests set `max`.
> **vLLM independently computes `1/sqrt(sum + 1e-6)`**, so #28068 is a fix
> rather than a change of convention. This is the first time the oracle has
> been wrong about the **model** rather than merely imprecise about the
> arithmetic.
>
> **L3a-5: L2e-3's fp16 F32 matmul is a dispatch threshold at 8 columns, and
> it changes what every tolerance in L2 means.** `ssm_alpha` and `ssm_beta`
> are F32 weights read by an F32 activation — L2e-3's situation exactly — and
> modelling them in fp16 is **86x worse** (2.15e-06 against 1.84e-04).
> `ggml_vk_mul_mat` sends a matmul to the f32 **vector** path whenever the
> output has at most `mul_mat_vec_max_cols = 8` columns and to the fp16 coopmat
> GEMM otherwise; the dump's prompt is **7 tokens**, and the indexer's score
> has 4x7 = 28 columns. One line explains both. **So at a real 512-token
> ubatch every projection in the model crosses to the fp16 path** — a tolerance
> measured against a 7-token dump is not a tolerance at 512, and that now
> applies to L2b, L2d, L2e and L3 alike.
>
> **L3a-6: and two things cross a batch boundary, not one.** 3 + 4 tokens
> reproduce 7 **bit-identically**, output and state both — but only because
> `DeltaNetState` carries the convolution's three-column window beside the
> [128, 128, 48] state. A kernel that carries the state and forgets the window
> is wrong in a way nothing but that test sees, and at L7 it would be wrong on
> every token. Two smaller things fell out: ggml's `softplus` adds in f32 and
> so flushes to **exactly zero** below -16.6, where `math.Log1p` is more
> accurate and wrong; and nine of the trace's tensors are **non-contiguous
> views** that `eval_dump.c` writes as the parent region they span, so
> `ReadDump` now gathers with the recorded strides (`q_conv-0`: 1.61e-01 rms
> read flat, 1.04e-07 read properly).

## What L3b established — the layer on the device, and a ladder that is 6.3x

`llm/gpu_deltanet.go`, `shaders/llm_dn_{conv,scan,norm}.comp`: five dispatches,
the plain GEMM arm twice and three kernels of this layer's own.
[Write-up](research/l3b-deltanet-gpu.md) · `results/l3b_dn.csv`

> **L3b-1: five dispatches against eleven, and 152.8 ms becomes 111.5.** Per
> 512-token graph, against the lines of llama.cpp's own graph that are nameably
> this layer: the four projections 91.3 → **71.3 ms**, the convolution with its
> SiLU, both L2 norms and the softplus 11.4 → **7.0**, the recurrence 15.8 →
> **14.9**, the gated norm 8.6 → **2.2**, the output projection 25.7 → **16.1**.
> **1.37x, and 13.1% of the whole graph becoming 9.6%** — a second run gives
> 113.2 ms and 1.35x. The SiLU, the sigmoid of z, the multiply against it and
> the CONTs behind all three are left out of the comparison entirely — they live in the shared MUL/SIGMOID/CONT lines and
> are *absent* from our graph rather than faster in it.
>
> **L3b-2: four of the reference's matrices are one, and two of them are the
> F32 scandal in miniature.** `attn_qkv`, `attn_gate` and the two **F32**
> [2560, 48] gate projections read the same block input, so they are one
> [16512, 2560] weight — L2a's argument for `inject` at a fourth site. The
> F32 pair is 72 dispatches a graph, **7.8 ms at 1155 GFLOP/s for 0.25 MB of
> weights**, and here it is 96 more columns on a matrix that already has
> 16384. That puts alpha and beta on the fp16 matrix cores, which is a
> deviation **from the 7-token dump and not from the reference**: L3a-5's
> 8-column threshold means llama.cpp does the same at any real ubatch, and
> **L1's PPL 4.0340 was measured with it**. Priced rather than hidden: 1.23e-03
> rms on the log decay, and at most **1.105%** on the per-token forgetting
> factor.
>
> **L3b-3: the scan's ladder is a re-read-to-ALU crossover, and its ends are
> 6.3x apart.** LPC — how many lanes own one column of the [128, 128] state —
> sets the registers a lane holds (128/LPC), the workgroups the layer
> dispatches, and how many times each head's q and k row is re-read per token,
> **all at once**. llama.cpp's own kernel is the top rung: `{H, n_seqs, S_v}`
> with one column a workgroup is **6144 workgroups and a 128-fold re-read** on
> a 64-wide wave — not the "48 workgroups" L3a inferred from the source — and
> ours at that shape costs 2766 us. Five rungs sit pinned at **1.18-1.31 TB/s
> while their flop rate varies 4x**, and the ablation is direct: **deleting the
> operand loads takes the widest rung from 2766 us to 1013**, where deleting
> both subgroup reductions moves it 5%. The winner is 16 state elements a lane
> and 768 workgroups — **413 us against the reference's 438, 1.06x**, and 1.16x
> at ubatch 2048.
>
> **L3b-4: registers beat LDS for the operands until they do not, and the
> turn is at the same rung.** Keeping q and k beside the state is 1.15x at
> LPC 16 and 1.08x at LPC 8, and **loses by 1.33x at LPC 4**, where 96
> registers a lane costs more occupancy than an LDS round trip costs latency.
> Two things that should have helped and did not: sharing the staged operands
> across four waves of a workgroup cuts the re-read 4x and ties, and a one-deep
> software pipeline is worth nothing at the wide end and 1.4x *against* at the
> winner.
>
> **L3b-5: so the chunk is priced rather than built, and the question was
> mis-scoped.** The recurrence is **13% of this layer**; the fused projection
> is **64%**. flash-linear-attention's four stages come to 3.70 GFLOP at 512
> tokens against the scan's 3.22 — **1.15x the arithmetic, all of it matmul
> shaped** — so at §2.7's best measured 39.0 TFLOP/s it would be 95 us against
> 413, a **4.3x ceiling**, and at an honest 10-15 TFLOP/s for 64x64x128 tiles
> with a serial triangular solve and an 8-step serial carry, **1.1-1.6x**.
> A *perfect* chunked kernel saves 10% of the layer and **1.0% of the prefill
> graph**, and puts a recurrent state through fp16 operands where the scan is
> at 2.4e-06. **Revisit at decode, not at prefill.**
>
> **L3b-6: it matches, and the state carries bit-identically.** `attn_output`
> is **2.4e-06 rms** against both L3a's CPU reference and llama.cpp — 200x
> tighter than the tensors feeding it, because the recurrence has no matmul in
> it — and the fifteen rungs agree with each other to 2.6e-10 to 7.1e-10.
> **3 + 4 tokens reproduce 7 exactly**, output and state, because the
> convolution's window is three rows of *negative token index* in front of the
> projection's own output: no branch, no second tensor. The control that zeroes
> it moves the output by 4.0e-02.
>
> **L3b-7: and L2f-4's MALL rule holds on two more weights.** The fused
> projection's B is **84.5 MB**, 2.6x the MALL, and wants the widest row block
> everywhere — by 2.6x at 512 tokens and 3.3x at 2048. `ssm_out`'s is
> **31.5 MB** and fits, and picks the narrow rung at 512 and the wide one at
> 2048 — the same rungs at the same lengths as `attn_output`, which is the same
> shape. Same kernel, opposite schedules, decided by which side of one cache
> the weight falls on.

## What L4a established — the selection, and an arithmetic that changes at 8 columns

The 4k dump: 4096 tokens of wikitext-2 in a 4096-cell cache, one ubatch, the
same pinned build. 8.0 GB, 116 tensors, ~4 minutes — and the filter takes
layers 0 and 3, so **the whole MoE block comes with it**.
[Write-up](research/l4-qsa.md) · `reference/out/llm4k/`

> **L4a-1: the selection is a radix select, and ours is the reference's.**
> `topk_radix_select.comp` is **not a sort**: four 8-bit radix passes over an
> order-preserving float→uint map find the key of the width-th largest value,
> then it emits every cell **strictly above** it and fills the rest from the
> cells **equal** to it in `atomicAdd` order. Ported pass for pass, `topK`
> reproduces `indexer_top_k-3` on **all 4096 rows** — and, unexpectedly, **set
> for set on the 2045 rows where the selection bites**, because filling the tie
> in ascending cell index is what a wave resolving an atomic in lane order
> does over `ratio` consecutive cells. It is also what makes a 4096-token
> fixture affordable: four passes over a row where a partial selection sort
> costs 2051 of them.
>
> **L4a-2: the order is not reproducible and the selection is.** A second dump
> of the same prompt gives **byte-identical** per-cell scores, a top-k whose
> order differs on **4096 of 4096** rows, whose set differs on **1967**, and
> whose set *restricted to causally visible cells* differs on **none**. All the
> run-to-run difference is among the -inf cells the mask drops anyway. **So a
> kernel may emit the selection in whatever order is cheapest**, and must
> reproduce the set, which it can.
>
> **L4a-3: block granularity is now checked rather than read.** vLLM supplied
> the claim in L3 and llama.cpp only implies it; at 4 k it is a property of the
> reference's own output. Past the biting point every row is `(i+1) mod ratio`
> tail cells + **exactly 512** whole blocks + a split of `(width-tail) mod
> ratio` cells off **the lowest-scoring block in the selection** — 3 cells 512
> times, 2 cells 511, 1 cell 511, and **0 cells 511 times**. The selection
> first drops a visible cell at token **2051**, which is the width.
>
> **L4a-4: and 0.0022% of it survives our own scores.** A top-k is
> discontinuous, so two blocks within our error can swap across the threshold.
> Run end to end from `hc_mixed-3` through our own indexer: **138 of 6 298 621**
> visible cells differ, over **30 of 4096** tokens, worst row 8 cells — down
> from 2354 over 471 tokens before L4a-6.
>
> **L4a-5: every quantised matmul accumulates in fp16 at a real ubatch, and
> that is the tolerance.** At 4096 columns **100.0% of the values the reference
> writes out of a quantised matmul are exactly IEEE halves** — `Qcur_full`,
> `Kcur`, `Vcur`, `attn_output`, the DeltaNet's `z` and `linear_attn_out`, the
> MoE's `ffn_gate`/`ffn_up`, `ffn_shexp`. That is `.f16acc`, an fp16
> *accumulator*, and the exceptions confirm it: `hc_inject`, the F32 router and
> `shared_expert_gate` are **0.0%**. The consequence is that no model of the
> operands can close the gap — exact f32, fp16 operands and the reference's own
> int8 activations land within **1.06x** of each other and all ~**5.4e-03 rms**
> away on a tensor whose own rms is 1.21. **L2c-3, L2f-6 and L3b-6's reading
> holds an order of magnitude larger and from a different cause: at prefill the
> reference is the side losing precision.** A 1e-6 tolerance on a projection is
> a fact about a 7-token dump.
>
> **L4a-6: and the indexer's BF16 weights meet a bf16 activation, worth
> 2061x.** Above the 8-column threshold a BF16 `src0` against an F32 `src1`
> sets `y_non_contig`, so the *activation* is converted to BF16 and a
> BF16 x BF16 kernel runs — eight mantissa bits. `indexer_k_raw-3` goes from
> **9.179e-04 rms to 4.454e-07**, and the control is that **fp16 is 1.02x
> *worse* than f32**: this is a different numeric from L2e-3's, not the same
> one again. With it, the whole indexer chain comes back (24-69x on the pooled
> key, the normed key, the query and the score) — and **L2e-3 is separately
> re-confirmed at 4096 columns**, the F32 x F32 score being 7.2x better in fp16
> than in f32 and 59x better than in bf16.
>
> **L4a-7: two subnormal bugs, both invisible to five stages of comparison.**
> The fp16 census caught `ffn_shexp-3` at 99.9% where everything else was
> 100.0%, and the 11 047 exceptions were all the smallest magnitudes and all
> negative: **`f16Round` returned |x| for every negative subnormal half**,
> building it as `-0.0 + q*2^-24`. Fixing that exposed **`f16` widening every
> subnormal half to *half* its value** — exponent field 113+e where the
> arithmetic gives 114+e. `TestHalfPrecisionRoundTrips` now walks all 65 536
> halves and all 65 536 bfloats.

## What L4b established — the selection on the device, and sparsity priced

`llm/gpu_attn.go`, `shaders/llm_attn_select.comp`: a seventh dispatch that
only exists past 2051 cells, and a bitmask the attention kernel reads beside
the causal mask. [Write-up](research/l4b-qsa-gpu.md) · `results/l4b_qsa.csv`

> **L4b-1: one dispatch against twenty-four, and 2.40 ms becomes 1.24.** Per
> 512-token graph, against the two lines that are the selection: `TOP_K`
> (2.19 ms) and `TOPK_QSA GET_ROWS` (0.21) against our one kernel — **1.94x**,
> and 8.89 ms against 3.42 at ubatch 2048, **2.59x**. The second of the
> reference's ops exists only because the first emits an **index list**: it
> spends a `GET_ROWS` turning that back into an f16 mask for a *dense* flash
> attention. Ours writes the mask, which is 32x smaller than the list. With
> the selection counted on both sides the layer is **68.8 ms of llama.cpp's
> graph against 29.6, 2.32x**; a real run at that shape does not select at
> all, and then it is **2.47x**, because the reference runs an identity here
> and `sparse` declines to. L2f's 2.40x, which excluded it from both sides,
> re-measures at 2.38x and is unchanged.
>
> **L4b-2: the bucket search was the kernel, and the histogram never was.**
> The reference walks 256 radix buckets **serially on lane 0** while the other
> 255 wait, four times a row — 1024 dependent LDS reads on the critical path
> of a kernel whose input is 8 KB. The same answer is a suffix sum plus an
> `atomicMax`, and as a subgroup scan over four pinned waves it is **192.0 →
> 103.3 us, 1.86x**. Per-wave histograms — the standard fix for LDS atomic
> contention, and the obvious suspect — are **1.06x against**. What is left is
> 42-61 GB/s, 4-6x off the bus, and 0.1% of a prefill graph.
>
> **L4b-3: the selection costs the attention kernel 1.21x and saves it
> nothing.** At 4096 tokens in a 4096-cell cache the same kernel on the same
> arithmetic is **9950 us dense and 12046 reading the bitmask** — because
> without a selection only the diagonal key block passes through the P mask
> loop, and with one all 128 do. The first version, bit-testing the arena per
> element instead of staging the block's 16 mask words in LDS, was **18546 us,
> 1.90x**. So "the sparsity is semantics at prefill, not a saving" is now a
> number, and it is negative. Two traps the dense path cannot reach:
> `exp2(-inf - -inf)` is a **NaN** when a key block has nothing selected in
> it, and a pad row has no selection at all.
>
> **L4b-4: exact against the CPU selection, 0.0381% against llama.cpp's — and
> the 17x is upstream.** The bitmask is `topK`'s selection **cell for cell on
> all 4096 rows** over the device's own scores, with no tolerance, because a
> bitmask has no order for the tie fill to get wrong. Against llama.cpp it is
> 2400 of 6 298 621 visible cells where L4a's CPU path was 138 — and the cause
> is **L2f-3's fusion**: the indexer's two BF16 weights are fp16 column ranges
> of the layer's one fused matrix, so the device cannot take L4a-6's bf16
> activation. `indexer_k_raw-3` sits at **9.319e-04 rms, which is L4a-6's own
> *unmodelled* 9.179e-04**, 2093x its modelled 4.454e-07, and 24x on the
> score. One weight instead of six, priced.
>
> **L4b-5: the layer at 4 k, and the control.** `attn_output-3` is **9.872e-04
> rms** against llama.cpp at 4096 tokens — *inside* the 1.14e-03 the same
> comparison gives at 7 tokens, so the 17x does not propagate — and
> `attn_gated-3` 3.475e-04. The same layer with the selection off is
> 1.958e-03, **2.0x further**: the bitmask is read, and reading it is what
> moves the layer towards the reference. A kernel that loaded the mask and
> ignored it would pass every tolerance by being a dense causal attention,
> which at 7 tokens is the right answer and at 4096 is a different model.
>
> **L4b-6: the attention ladder's top two invert with length, and sparsity is
> not why.** At 4096 tokens qt1_kt2 is 9950 us, **qt2_kt4 10770** and qt1_kt4
> 13969, where L2f-5 measured 170/223/241 monotonic at 512. The inversion is
> in the dense column too. The winner does not move at either length or
> either density, which is what `DefaultAttnKernel` rests on.


## What L5a established — the MoE block, and a routing that is not balanced

`llm/moe.go`: the router, the softmax, the top-10 of 512, the normalised
weights, the three expert matmuls, the shared expert and its sigmoid gate —
against all fourteen tensors llama.cpp names inside the FFN half.
[Write-up](research/l5a-moe.md)

> **L5a-1: the selection is the reference's, set for set on all 4096 tokens.**
> 0 of 40 960 slots differ as a set — which is the number that matters,
> because a top-k is discontinuous and an rms on the weights would hide a
> swapped expert entirely. Two slots differ in *order*, on one row, and the
> reference's own probabilities for that pair are **4.35e-07 apart relative**
> against a router whose maxRel is 3.4e-05, so the test demands that any
> disagreement in order be a tie the router's precision cannot resolve. The
> router itself is 3.6e-05 rms, and it is L2e-3's numeric rather than
> L4a-5's: fp16 operands with an **f32** accumulator, because `.f16acc`
> belongs to the quantised kernels and L4a-5 put `ffn_moe_logits` at 0.0%
> exact halves. *(L5b-5 corrects the other half of this finding: the claim
> that `shared_expert_gate`'s single output column keeps it on the f32 vector
> path at any prompt length is wrong — the threshold counts the columns of
> the output, which is the token count. At a real ubatch it is on the matrix
> cores like everything else, and the device reproduces it bit-for-bit.)*
>
> **L5a-2: the block matches at the tolerance the fp16 accumulator sets.**
> `ffn_out-3` is **1.378e-04 rms**, with the three routed matmuls at
> 2.7-2.9e-03 on values of order 1 — the oracle's precision, not ours — and
> `shared_expert_gate` at **4.152e-05** because it is the one tensor here with
> no quantised weight in it — not, as this finding first said, because one
> output column keeps it on the f32 vector path; see L5b-5.
>
> **L5a-3: modelling the reference's arithmetic makes the fit *worse* here,
> by 1.4-1.8x.** `Exact` f32 is 2.217e-03 against llama.cpp where `RefQ8` is
> 3.054e-03 on `ffn_moe_gate`, 0.57-0.73x across three tensors. Everywhere
> else in this vertical modelling the reference's int8 activations was the
> largest correction available — 233x at L2b-2, 69x at L2e-3, 2072x at
> L2e-2, 2061x at L4a-6 — and here it is worth less than nothing, because the
> reference has already lost eleven mantissa bits in the accumulator and our
> quantisation error adds in quadrature rather than cancelling. **So L5b has
> nothing to reproduce**, and a tensor comparison at the MoE is a bound rather
> than a fit.
>
> **L5a-4: the routing is nothing like balanced, and that is what L5b is
> sized by.** L2a's budget assumed "each expert is read 10 times in 512". The
> mean is right and nothing else is: at ubatch 512 only **274 of 512 experts
> are touched**, the widest bucket is **484 tokens — 95% of the ubatch** — and
> **expert 454 is ranked first by 274 of the 512 tokens**. At 2048 and 4096 it
> is 399 and 435 experts touched with the top bucket at 60% and 46%. Every
> bucket is cross-checked against llama.cpp's own `ffn_moe_topk`, so the skew
> is the model's. Four consequences: half the bank is read per ubatch rather
> than all of it, a workgroup-per-expert kernel has a **26x** load imbalance
> that cannot be scheduled statically, arithmetic intensity varies **484x**
> between experts inside one dispatch (which is the argument for a grouped
> GEMM with a variable row count over a gather-GEMV), and at decode the top
> expert is the same 0.5 MB of Q4_K 95% of the time and would sit in the MALL
> if anything kept it there.
>
> **L5a-5: three controls, because three things here have a plausible wrong
> version every tolerance would pass.** Reading the expert bank interleaved
> rather than expert-major is **175x** worse; `silu(up)*gate` instead of
> `silu(gate)*up` is **59x** — the smallest margin of the three, because silu
> is near-linear away from zero; and the `weights_sum` clamp guards a division
> by zero the data never reaches (the smallest sum on this prompt is 0.0554
> against the clamp's 6.1e-05), so it is asserted from the graph and measured
> to be the identity on all 4096 rows.


## What L5b established — the block on the device, and two optimisations that were worth nothing

`llm/gpu_moe.go`, `shaders/llm_moe_{gemm,route,perm,combine}.comp`: nine
dispatches, the plain GEMM arm once and four kernels of this block's own.
[Write-up](research/l5b-moe-gpu.md) · `results/l5b_moe.csv`,
`results/l5b_moe_ladder.csv`

> **L5b-1: nine dispatches against about 845, and 567.4 ms becomes 522.3.**
> Per 512-token graph, against the lines of llama.cpp's own graph that are
> nameably this block: the router with the shared expert's gate 15.2 →
> **4.7 ms** (3.22x), the routing's tail 2.6 → **0.7** (3.70x), the gather
> 2.0 → **1.8**, gate/up with its `GLU` 372.3 → **345.4** (1.08x), down
> 158.9 → **134.0** (1.19x), the shared expert 16.4 → **13.6** (1.21x), and
> a **combine the reference does not have at all**, 22.1 ms, counted in full.
> **1.09x, and 48.7% of the whole graph becoming 44.8%** — a second run gives
> 522.2 ms, 0.3% apart. At `-ub 2048` the same comparison is **1758.4 ms
> against 1131.6, 1.55x**, because the MoE is the one block that gets cheaper
> per token on *both* sides as the chunk grows and the glue that makes 2048 a
> worse ubatch overall (L2a-4) belongs to other blocks. The `MUL` that
> applies the ten weights, the `MUL` and `SIGMOID` of the shared gate and the
> `ADD` that sums the two halves are left out of the comparison entirely —
> they live in the shared MUL/SIGMOID/ADD lines and are *absent* from our
> graph rather than faster in it.
>
> **L5b-2: the bank never becomes floats, and that is what makes the kernel
> possible.** Every other GEMM in this vertical takes a B operand the host has
> dequantised to halves and packed into §2.8 fragment tiles; one layer's three
> routed banks are 2.52 billion weights — **5.03 GB at fp16 and 241 GB across
> the model**, against 1.57 and 75 as they ship — so that cannot happen here.
> The bank is staged byte for byte out of the mmap'd checkpoint and each
> workgroup unpacks its own BN x BK slab into LDS per K-step, from **Q4_K,
> Q5_K, Q5_1 or Q8_0** under a `-DQFMT`, which is §2.2's arrangement for
> §2.2's reason: a cooperative-matrix fragment's lane layout is not exposed,
> so nibbles cannot be unpacked into registers and declared a fragment.
>
> **L5b-3: L5a-4's routing is the schedule, and padding the rows is what buys
> the epilogue.** 274 of 512 experts at ubatch 512 with one taking 95% of the
> tokens makes a grid of (experts x row blocks) almost entirely empty, so the
> tiles are a device-built **list** over a static upper bound with an early
> return — 372 real records against a bound of 1216. And every expert's range
> is padded up to the row block, the slack named by a **sentinel** permutation
> entry pointing at a token one past the batch: that is what lets a
> cooperative-matrix store, which covers sixteen rows and cannot be masked, go
> straight to global rather than through an LDS staging buffer. It executes
> 1.09-3.90x the rows that exist depending on the rung and the chunk, and
> **poisoning the sentinel's activation row with 1000 moves `ffn_out` by
> exactly zero** — which is the control that scheme needs and that no tensor
> comparison provides.
>
> **L5b-4: two optimisations worked, two did not, and the two that did not are
> the finding.** Deleting the LDS epilogue is **1.35x** on the whole block and
> widening the unpack's loads is **3.1x on Q8_0** — both about instruction
> count rather than about bytes or arithmetic. But **sharing the dequantised
> slab across two or four waves of one workgroup is 1.15-1.24x *against***, at
> the same row block, even though it halves the unpack per row and doubles the
> waves a CU holds on a kernel running at about one wave a SIMD: occupancy is
> not what this kernel is short of, which is the answer L3b-4 got for the
> scan's staged operands. And **making the K-step cover a whole Q4_K nibble
> group is 1.16x against** even though it removes a real 2x read
> amplification, because the doubled slab takes MODE 0's LDS from 12.8 KB to
> 23 and the workgroups a CU holds from five to two. A third: the
> conflict-free LDS pad is **5.4x worse** than the four-way-conflicting one,
> because 16-byte row alignment is what the fragment load needs and every
> aligned pad has `gcd >= 4` with the bank count.
>
> **L5b-5: the router is bit-exact, and that corrects L5a-1.**
> `ffn_moe_logits-3` is **0 ulp** on all 2 097 152 values where L5a's CPU
> reference is at 3.606e-05 — same operands, same accumulator, and the same
> 16-wide accumulation order. **`shared_expert_gate-3` is bit-exact too.**
> L5a-1 read the reference as keeping that one on the f32 *vector* path "at
> any prompt length" because it has one output column, and inferred that
> fusing it onto the router's matrix as a 513th column would be a deviation to
> price, in L3b-2's spirit. `mul_mat_vec_max_cols` counts the columns of the
> **output**, which is the token count, so at any real ubatch a [2560, 1] F32
> weight is on the fp16 matrix cores like everything else. The fusion costs
> nothing and it deletes 47 dispatches running at **13.6 GFLOP/s** — L2a's
> `inject` scandal in miniature, 9.1 ms a graph.
>
> **L5b-6: it matches, and the selection now matches in order too.**
> `ffn_out-3` is **8.132e-05 rms** against llama.cpp over all 4096 tokens —
> inside L5a's 1.378e-04 over 48 — with the swiglu at 5.5e-04 on values to
> 8.8, which is L4a-5's fp16 accumulator and not ours. The selection is the
> reference's **set for set on all 4096 tokens and 0 of 40 960 slots differ in
> sequence**, where the CPU reference had two: the tie that separated them is
> 4.35e-07 apart relative, the CPU's logits differ from the reference's by
> more than that and the device's do not differ at all. The permutation has a
> gate of its own with no tolerance in it — a bijection over all 40 960
> (token, slot) pairs, every row in the range of the expert that chose it, the
> histogram the selection's bucket for bucket, and the inverse round-tripping
> on all 45 056 slots.
>
> **L5b-7: and the next 2.5x is the unpack, or a bigger ubatch.** `up` runs at
> **10.8 TFLOP/s and 70 GB/s** of quantised bank against §2.7's 39 TFLOP/s and
> L0a's 236 GB/s. Its own byte floor is 2.9 ms against 7.2 measured, with the
> arithmetic able to hide under it. But the arithmetic intensity is the
> **routing's**, not the kernel's — 274 experts read 505 MB to serve 5120 rows,
> about 17 rows an expert — so the lever that actually moves is the chunk,
> which is why the block is 1.09x at 512 and 1.55x at 2048. L2a-4's "do not
> raise the ubatch" is a statement about the glue, not about this.


## What L6a established — the whole model resident, and a bank that is an array

`cmd/llm/resident.go`, `llm/gpu_moe.go`, `vk/shim.c`,
`shaders/llm_common.glsl`: 84.20 GB in 68 buffers, and binding 5 as an array.
[Write-up](research/l6a-residency.md) · `results/l6a_resident.csv`

> **L6a-1: 84.20 GB of weights and 0.87 GB of arenas in 68 buffers, staged in
> 33 seconds, with 41 GB of the machine left over.** 97 hyper-connection
> mixers (1.31 GB), the PLE block (0.07), 36 DeltaNet layers (4.18), 12
> attention layers (1.24) and 48 expert banks (77.41). Against the
> checkpoint's 82.52 GB resident core it is **1.04x** once the 0.68 GB
> embedding and the 0.68 GB lm head — which nothing stages yet — come off the
> reference figure; the **+3.04 GB** is the dense half staged as *halves*,
> 6.79 GB of fp16 against the 3.67 GB of Q8_0 and F32 those tensors ship as,
> 1.85x. The 28.80 GB n-gram table is still mmap'd (D2) and 41 GB of
> MemAvailable is left for its working set. **The DeltaNet bank is the near
> miss**: 36 fused projections in **4.18 GB in one buffer**, 2.7% under this
> device's cap.
>
> **L6a-2: the expert bank is an array of buffers, one a layer, because the
> arena-with-an-offset arrangement cannot express this model.**
> `maxStorageBufferRange` here is **4 GiB - 4** and a layer's bank is 1.61 GB,
> which is also the most a `uint32` byte offset addresses — the same wall
> twice. So binding 5 and binding 6 became `qb[NBANK]` and `qb4[NBANK]`,
> indexed by a **dynamically uniform** push constant, which is core Vulkan and
> not descriptor indexing: one dispatch is one layer. Three consequences.
> `shim_create_compute_pipeline` gained a per-binding `descriptorCount`, so
> seven bindings are 5 + 48 + 48 = **101 buffers** and `counts == NULL` is
> byte for byte the old path. NBANK is **compiled in at 48** — a spec constant
> cannot portably size a descriptor array and the SPIR-V is pre-compiled — so
> a short stage pads the array with the block's 256-byte placeholder. And the
> layer index rides in the **top sixteen bits of `moeUsed`**, because the push
> block is 64 uints against this device's whole 256-byte range and L5b's own
> comment on it reads *"Five fields, and the block is out of room"*.
>
> **L6a-3: the index is load-bearing, and the control says so by 203x.** Every
> other test here stages layer 3 alone, where the bank index is zero and an
> ignored index is indistinguishable from a read one. `TestMoEGPUBankArray`
> stages layers 0-3 and asks for the last: **8.133e-05 rms** against
> llama.cpp, against **1.652e-02** — 203x — for the same run with the index
> forced to zero. Layer 0's experts against layer 3's routing is a perfectly
> plausible tensor of the right shape and magnitude, which is why the control
> is in the test and not in a comment.
>
> **L6a-4: residency is free in the bank's size.** 2, 4, 12, 24 and 48 banks —
> **10.4 to 84.2 GB resident** — run layer 3's block in **11134, 11062, 11162,
> 11067 and 11242 us**: an eightfold range of bytes, a 1.6% spread, no trend.
> That is L0a (*"the DRAM bus does not care how big the weight bank is"*)
> holding at 68 live buffers and the size the model actually is, rather than
> at a probe's 23. What is left is a **flat ~2.5% step** against L5b's
> isolated 10824.7 us that appears as soon as the other four blocks are staged
> and does not grow with the bank — not the count and not the indirection,
> both constant across that column. At a ~1% noise band it is written down
> rather than chased: L6b runs all five blocks in one graph anyway.
>
> **L6a-5: staging order is worth 8 GB.** The expert bank is `memcpy` out of
> the mmap'd checkpoint at about 2.6 GB/s and needs no host floats at all; the
> dense half is **dequantised to f32 on the host** first, and the 36 DeltaNet
> layers are **8.35 GB of transient floats at once**. So the dense blocks go
> first, each block's floats are returned to the OS before the next asks, and
> the 77 GB goes last — the peak is the resident model rather than the
> resident model plus a dequantisation.

## What L6b established — the model runs, and a memory type was worth 5x

`llm/graph.go`, `llm/gpu_head.go`, `llm/arena.go`, `vk/engine.go`,
`cmd/llm/bench_graph.go`: the five blocks in llama.cpp's own order, plus the
embedding gather and the head. [Write-up](research/l6b-graph.md) ·
`results/l6b_graph.csv` · `results/l6b_arena.csv`

> **L6b-1: it picks llama.cpp's token.** On the seven-token prompt the last
> token's argmax is **561** on both sides, and our top ten is llama.cpp's ten
> tokens — three adjacent pairs inverted, which is reported rather than
> demanded, because an inversion between logits a thousandth apart is a
> different fact from a wrong token. `result_norm` is **0.519% of scale** and
> `result_output` **1.983%**. **Nothing in a shader changed**: L6b is the
> order, the two tensors no block owned, and the plumbing.
>
> **L6b-2: three structural facts, all of them the reference's.** There is
> **no output norm** — `output_norm.weight` is absent and the final
> hyper-connection mixer is it, which makes `llm/gpu_head.go` the only bare
> matmul in the vertical. The **final mixer and the head run on one row**,
> because `inp_out_ids` drops every other token before them and the trace
> agrees: `result_output` is one row of 248320 floats for a seven-token
> prompt. And **the wide residual never leaves the mixer's arena** —
> `llm_hc_combine.comp` updates it in place — so what crosses a block
> boundary is the 2560-wide `hc_mixed` out and the 2560-wide block output
> back, not the 10240-wide stream.
>
> **L6b-3: the drift is a curve, and that is what makes 0.881% a number
> rather than a worry.** A second oracle pass filtered to `l_last` every
> eight layers, and the graph stopped at each depth off one staged model:
> **0.034 / 0.055 / 0.103 / 0.154 / 0.330 / 0.881% of scale at 8 / 16 / 24 /
> 32 / 40 / 48 layers** — a clean geometric **x1.085 a layer**, no step
> anywhere. That is what fp16 operands against the reference's int8 dot
> products (L2b-2) and its fp16 accumulators (L4a-5) compound to over 48
> residual additions. The four-layer prefix test prints the same column at
> depths 1-4 in 7 GB and three seconds.
>
> **L6b-4: the activation arenas were on the wrong memory type, and it was
> worth 5.0x.** The first working graph did **280.3 tok/s, 0.72x llama.cpp**,
> with 45.5% of its wall clock in plumbing that moves 5.24 MB a hop. The rate
> was the reason: the type `NewBuffer` prefers here is write-combined, and
> **the host reads it at 0.18 GB/s** where a HOST_CACHED type reads the same
> buffer at **25.06** and writes it 1.4x faster as well — **139x**, and
> DEVICE_LOCAL is one of the slow ones. **This is the other half of L0b**,
> which measured the same table from the *device* at 236.0-237.4 GB/s across
> all of it: free on the GPU, decisive on the host. The control says so
> directly — the MoE block, 97% of the parameters, runs its layer in
> **10834.5 us on cached arenas against 10850.2 on write-combined ones,
> 0.14%** — and is kept as `LLM_ARENA_UNCACHED=1` rather than as a note.
> Weight banks stay where they were: written once, never read.
>
> **L6b-5: the other half of the plumbing was the narrowing.** With cached
> arenas the glue is `Upload`'s f32→fp16 conversion — 1.31 M values a
> sublayer, 96 sublayers, **126 million a graph** — which was running on one
> core. The rows of a prefill are independent, so parallelising it changes
> the wall clock and not one value: **832 ms to 264** at ubatch 512.
>
> **L6b-6: the ladder inverts against the reference's.** 128 / 512 / 1024 /
> 2048 give **177.3 / 403.7 / 553.9 / 713.9 tok/s** where llama.cpp's own are
> 347.71 / 391.42 / 380.91 / 336.72: it peaks at 512 and **loses 14% by
> 2048**, because its 10240-wide F32 residual is 20.97 MB at 512 — just
> inside the 32 MiB MALL — and 83.9 MB at 2048 (L2a-4). Ours has no such
> cliff, because the residual is never a graph tensor and the elementwise
> glue that read it was fused away at L2c. So **1.82x** against the
> reference's best ubatch and **2.12x** against it at 2048, where a long
> prompt actually is. Staged for 512 alone — a quarter of the arenas — ubatch
> 512 does **430.4, 1.10x**.
>
> **L6b-7: and what is left is the glue, at 22.7%.** Every block owns its own
> arenas, so `hc_mixed` is read out of one and narrowed into another. The
> blocks alone already do **931.6 tok/s at ubatch 2048**; deleting the moves
> is one shared activation buffer across the five blocks, the sublayer's
> input offset pointed at the mixer's output, the combine's output pointed at
> the sublayer's, and one small narrowing kernel. L2a-3's projection for this
> graph with its glue fused was ~1150.

## What L6c established — the glue on the device, and a fixed cost that was hiding

`shaders/llm_move.comp`, `llm/move.go`, `llm/graph.go`, `vk/engine.go`: the
activation that crosses a block boundary, as one dispatch.
[Write-up](research/l6c-moves.md) · `results/l6c_graph.csv`

> **L6c-1: one dispatch, nine pipelines, and a `Port` a block.** The move
> declares three bindings of its own rather than llm_common.glsl's five,
> because its two buffers belong to *different* blocks — and bindings 1 and 2
> are **the same VkBuffer bound twice**, so an fp32 destination and an fp16
> one are one pipeline shape and a `narrow` push constant rather than two
> shaders. A pipeline owns its descriptor set here, so there is one per
> (source, destination) pair: nine for the five blocks and the head. Each
> block exposes a `Port` for the two or three tensors that cross its boundary
> and for nothing else, so what the graph may reach into is a list in the
> block's own file and not an exported arena.
>
> **L6c-2: it is bit-exact.** `l_last` at all six depths, `result_norm` and
> `result_output` came back **identical to the last place** — because
> `float16_t(v)` in SPIR-V and `safetensors.F32ToF16(v)` in Go are the same
> round-to-nearest-even. `TestMoveNarrow` makes that a claim rather than an
> observation: 37 rows of 2560 against the host narrowing value for value,
> over halfway cases, subnormals and magnitudes past fp16's exact-integer
> range, with the pad rows zeroed and the A operand's pad *columns* untouched.
> The pad rows are load-bearing — the GEMM rungs have no bounds check, so the
> rows between the prompt and the consumer's row block have to carry the
> products of zeros and a shorter run must not read what a longer one left.
>
> **L6c-3: and it costs 2.0%, which is what decides the shared arena.** The
> 197 moves of a 2048-token pass are **43.7 ms of 2175.8** against the 22.7%
> of host plumbing they replace; the host is 1.4% more. **One set of arenas —
> what this stage was scoped as — would buy that 2.0% and nothing else**, for
> a two-phase construction through five constructors, because a descriptor
> set needs its buffer before a pipeline exists and a Vulkan buffer cannot
> grow. So it is priced rather than built, the same way L3b-5 answered scan
> versus chunk.
>
> **L6c-4: a fixed host cost was hiding behind fifty times its size.** With
> the moves in, the host column **fell with the prompt length** — 70.5 ms at
> 128 tokens against 15.9 at 2048 — and running the ladder backwards showed
> it follows the length, not the position. That is what a flat per-graph cost
> looks like, and it was `DeltaNetGPU.Reset` spelled `WriteFloat32At(off,
> make([]float32, n))`: the recurrent state is 3.1 MB a layer and a fresh
> sequence resets 36 of them, so **113 MB of Go allocation a prefill**,
> zeroed by the runtime and then copied into a mapping that was about to be
> zeroed anyway. `vk.Buffer.ZeroFloat32At` clears it in place and
> `HCGPU.UploadInit` writes `hc_init` straight into the residual arena rather
> than building its 84 MB on the host. The host column is now ~11 ms at every
> length, and it was worth **1.14x at 128 tokens** and 1.01x at 2048.
>
> **L6c-5: the ladder, and where the pass goes.** 128 / 512 / 1024 / 2048 give
> **259.0 / 570.4 / 761.8 / 941.3 tok/s**, 1.32-1.46x on L6b at every length,
> two separate runs agreeing to 0.54% at 128 and **0.01% at 2048**. Staged for
> 512 alone, ubatch 512 is **572.9** — 1.83x llama-bench's own pp512 of
> 313.62. **96.6% of the graph is the blocks now**: MoE 53.7%, DeltaNet 20.3%,
> hyper-connections 15.2%, attention 6.3%. The short-prompt end is the honest
> weak spot — at 128 tokens the MoE is 73% of the pass, because an expert bank
> costs what it costs to *read* however few tokens go through it, which is the
> same reason llama.cpp's pp512 is below its pp2048. That is a decode problem
> wearing a prefill hat, and it is L7's.

## What L7a established — the KV cache, and a round trip that was dead code

`llm/gpu_attn.go`, `shaders/llm_attn_{pack,idx,score,wmma}.comp`: the
full-attention layer over a cache that outlives the batch.
[Write-up](research/l7a-kv-cache.md)

> **L7a-1: the query is a batch and the key is a cache.** They used to be the
> same object — token `t` was cell `t` — and now the query's geometry is
> `plane` padded token rows while the key's is **nKV cells, one plane a
> layer**. Every causal comparison becomes a cell against a *position*, row
> `i` of query block `q0` being cell `SEQ_PAST + q0 + i`, and the pack
> scatters by cell because a batch boundary may land inside a fragment tile.
> The cache is **2.31 KB a cell a layer** — key, value, the indexer's raw key
> and its pooled block — so twelve layers are 27.7 KB a cell: 57 MB at 2048,
> 906 MB at 32768, and a single buffer caps it near 148k cells.
>
> **L7a-2: the indexer needed two things a KV cache does not suggest.** Its
> *raw* key gets a cell cache of its own, written as **one more plane of the
> pack kernel's grid**, because pooling averages `ratio = 4` cells and at
> decode those arrive in four different batches. And its *pooled* table is
> incremental: a cell never changes, so a block is final once its four cells
> exist and a continuing run rebuilds only `[past/ratio, (past+T)/ratio)` —
> one workgroup instead of 1024 — with a **fresh** sequence the one case that
> writes all of it, because the blocks past the last complete one pool cell 0
> four times and must be filled once before they can be left alone.
>
> **L7a-3: two push fields, and the block has been full since L5b.** 64 uints
> is 256 bytes and that is this device's whole range, so `SEQ_PAST` rides
> `lowRank` and `ATTN_IDXRAW` rides `loOff` under macro names, the same
> arrangement `moeUsed` has carried the bank index in since L6a. The mapping
> lives in `llm_common.glsl` and no kernel spells the borrowed field.
>
> **L7a-4: `float(float16_t(x))` is dead code on this driver, and L2e's fp16
> cache had never run on the GPU.** Three builds of the pooling loop,
> checksummed over the whole pooled tensor at both fixture lengths: reading
> the fp16 cell cache gives one answer, and the inline cast gives one
> **bit-identical to no cast at all**. It is not `spirv-opt` — `spirv-dis`
> finds both `OpFConvert`s in the optimised module — so it is RADV's NIR
> folding `f2f32(f2f16(a))` under the inexact rule a shader without
> `NoContraction` permits. **A value that models a memory format has to go
> through memory.** Reading the cache is what the reference does and is forced
> anyway; it moves `indexer_k` from 3.058e-04 to 3.552e-04 against llama.cpp
> and leaves `attn_output`, `l_last-0..3` and the argmax unchanged, because
> the raw key already arrives at 1.65e-04 from our own dequantisation.
>
> **L7a-5: the gate is an equality.** 4096 tokens in chunks of 512, 64, 7 and
> 4063-then-33-singles: **identical to the last place**, all 10 485 760
> floats. A chunked run recomputes nothing — the same dispatches read the same
> values in the same order — so bit-equality is reachable rather than lucky,
> and a tolerance would have passed a cache off by a position.

---

## What L7b established — five histories, and one shared by 36 layers

`llm/graph.go`, `llm/gpu_ple.go`, `llm/gpu_deltanet.go`,
`shaders/llm_seq_hist.comp`: the model continuing a sequence.
[Write-up](research/l7b-sequence.md)

> **L7b-1: five things carry and only one is a KV cache.** The cache and
> pooled blocks (L7a); the PLE convolution's **9**-row ring; every DeltaNet
> layer's recurrent state; its **3**-row window; and the **n-gram trigram**,
> which is the host's — a decode step's gather is a function of three tokens,
> one of which it has, so `Graph` keeps the sequence rather than the batch and
> slices `PLERows` at `past`. The wide residual is deliberately *not* carried:
> it is a function of the token.
>
> **L7b-2: the DeltaNet's convolution window was shared by all 36 layers, and
> nothing could see it.** It lived as `Conv-1` rows of negative token index in
> front of the fused projection's output — an arena sized for one *batch* —
> and while every pass was a fresh sequence all 36 layers read the zeros the
> graph had just written, which is the right answer. The moment a run leaves
> its tail behind, layer 1 reads layer 0's: `l_last-1` goes from 2.473e-04 to
> **3.062e-03, 12x**, with `l_last-0` exact. The assumption that stopped
> holding is worth naming — *the arena in front of the projection's output
> belongs to whoever ran last*.
>
> **L7b-3: one kernel for both rings, addressed by position, so the store is a
> pure write.** `shaders/llm_seq_hist.comp` is fifteen lines and serves both
> convolutions. The only rows a run must save are its own; every older slot is
> already right, because the run that produced it put it there. So no slot is
> both read and written however short the run is. A *shifted* window needs the
> opposite and at one token would read `hist-1` rows to move them one slot
> along — which is what the host-side `DeltaNetGPU.Carry` did, 198 KB a layer
> through a mapped arena, **and it read in front of the arena for any run
> shorter than the window**, i.e. every decode step. Deleted.
>
> **L7b-4: the gate, and the control that comes free.** 512 tokens in chunks
> of 128, of 9, or as 495 then seventeen single ones: `result_norm`
> **identical to the last place** in every case. The control — every token its
> own *sequence* — is **1.785e+00 rms** on values to 18.5. The one-token
> schedule is both at once: a chunk of one has nothing of its own to read, so
> a history that was not carried cannot produce the right answer by accident.

---

## What L7c established — the loop, the text, and where 134 ms goes

`llm/sample.go`, `cmd/llm/generate.go`, `gguf/gguf.go`: generation.
[Write-up](research/l7c-decode.md) · `results/l7c_decode.csv`

> **L7c-1: it generates llama.cpp's text, and the one disagreement is
> priced.** At the divergence our top two are `"." 25.362` against
> `" for" 25.201` — **0.161 apart**, where the tokens either side are decided
> by 3 to 10 — which is L6b-3's x1.085-a-layer drift landing where the model
> is indifferent; both continuations say the same thing and the completion
> re-converges within a sentence. So the gate reads *llama.cpp's text except
> where the argmax is a near-tie*, which is a claim about two implementations
> rather than about a checkpoint neither computes exactly.
>
> **L7c-2: the stop condition is not `eos_token_id`.** This checkpoint's is
> **248046** and what the model emits to finish is **248044** — which the
> metadata calls `bos_token_id` and `padding_token_id`. llama.cpp flags
> `<|endoftext|>`, `<|im_end|>` and `<|eot_id|>` end-of-generation by *name*
> as well as by key, and a loop that trusts the key runs past the end of the
> text and then repeats one token until `-n` is exhausted.
>
> **L7c-3: one `madvise` was worth 176x.** D2 leaves the 28.80 GB n-gram table
> mmap'd because a token reads 1.41 KB of it — right about bandwidth, silent
> about latency. Sixteen scattered 90-byte reads are sixteen page faults and
> the kernel answers each with a 128 KB readahead window: **2 MB of I/O to
> deliver 1.41 KB.** `MADV_RANDOM` on that tensor's pages alone takes the
> gather from 17.6 ms a token to **0.1** at four layers — 2.07x on the decode
> rate — and from 23.6 to 8.5 at 48, where 85.47 GB resident leaves no page
> cache to hold it. Prefill gains too: the ladder goes from L6c's 941.3 to
> **950.3 tok/s**. `TENSOR_READ_LAZY` is a statement about bytes;
> `MADV_RANDOM` is the statement about pages that has to go with it.
>
> **L7c-4: a command buffer a block, and a submit is 31 us.** Every block was
> submitting each dispatch with its own fence wait — fine at prefill, the
> whole cost at decode, where a token is ~1130 of them. One
> `DispatchMultiTimed` a block takes it to ~490 and the rate from 5.99 to
> 6.68; the MoE column prices the submit exactly, 432 to 48 for 12.1 ms.
> Nothing in the shim prevents one command buffer for the **whole token** —
> it binds each pipeline's own descriptor set — and that is worth ~11% more.
>
> **L7c-5: the ceiling is 25.0 tok/s, not 38.2, because our dense half is
> fp16.** A token reads 8.07 GB of dense weights (1.31 hyper-connection + 4.18
> DeltaNet + 1.24 attention + 1.27 head + 0.07 PLE) plus ~1.60 GB of experts =
> **9.67 GB**, against the budget's 6.334 at the shipped Q8_0. 9.67 GB at 242
> GB/s is 40.0 ms — and **25.0 tok/s is llama.cpp's measured rate**, so the
> reference is at the ceiling of the bank *it* reads rather than leaving a
> third of the bus unused. We are at **30% of our own ceiling**, and D3 is
> worth **3.5x** here rather than 1.7x.
>
> **L7c-6: two kernels shaped for prefill are 55% of the step.** Of 134.1 ms:
> hyper-connection 27.6%, MoE 27.5%, DeltaNet 21.4%, gather 6.3%, attention
> 6.3%, moves 5.3%, head 4.8%. The first two are **6.9x and 5.6x** off their
> own bytes where the rest are 1.2-1.7x. The hyper-connection **down**
> projection is 237.6 us of a 293.5 us mixer at **29 GB/s**, because it writes
> 336 columns and launches about **21 workgroups** on a 40-CU device, where
> `up` writes 10240, launches 640 and does 143 GB/s — at M=1 the parallelism
> has to come from splitting **K**, which is what `gemv_w4a8.comp` does at
> 99-103% of the bus (§1.1) and no rung of a GEMM ladder can supply. The MoE
> pads each expert's rows to its 32-row block (L5b-3), so **one token does 32x
> the arithmetic it needs** and the kernel is unpack-bound at 70 GB/s of bank.

## What L7d established — the decode kernels, and a grid said twice

`shaders/llm_hc_gemv.comp`, `shaders/llm_moe_gemm.comp`, `llm/record.go`,
`vk/shim.c`: decode at 11.89 tok/s.
[Write-up](research/l7d-decode-kernels.md) · `results/l7d_decode.csv` ·
`results/l7d_hc_gemv.csv` · `results/l7d_moe_decode.csv`

> **L7d-1: at one token this model is short of workgroups, not of rows, and
> the down projection is 7.95x.** `llm_gemm.comp` blocks the output columns at
> 48 over a fused N of 336, so at M=1 the whole matrix is **seven** workgroups
> on a 40-CU device whatever BM is — 238.6, 256.4 and 303.7 us for m1, m2 and
> m4, all at 23-29 GB/s. The parallelism has to come from **K**, and the split
> is free of any repacking because §2.8's fragment tiling already puts an
> n-tile's consecutive kt next to each other: a workgroup's whole slab is
> **one contiguous run**, a wave of 64 lanes taking sixteen halves each covers
> 1024 consecutive halves, and the (n%16)*16 + k%16 order hands each lane one
> output column and sixteen k of it — one accumulator, no LDS, two shuffles.
> **30.0 us at 230 GB/s, 95% of the bus**; the mixer goes 293.9 to 79.5.
>
> **L7d-2: which split is §5.1b's 4 KB period.** The ladder is not monotonic —
> 8/16/32/40/80/160 slabs give 181/154/**219**/129/138/**230** GB/s — and what
> separates them is the distance between two workgroups' slabs: 40960, 20480,
> **10240**, 8192, 4096, **2048** bytes. Every rung that is a whole multiple
> of 4 KB is slow and both that are not are fast. That is the rotation §5.1b
> pinned across nine strides and §2.3 found in a leading dimension; this is the
> third kernel it has shaped here and the first where the stride is **the
> distance between two workgroups' addresses** rather than a matrix's.
>
> **L7d-3: the MoE's padding was not the constraint, its grid was.** At one
> token ten experts have one row each, so every rung makes exactly ten tiles
> and a narrower row block removes arithmetic nothing was waiting on. The
> profiler shows a grid instead: `up` is 100 workgroups at 71 GB/s of bank,
> the shared expert's `up` is **ten** at **23**, and `down` — same kernel,
> same format, 400 workgroups — is at **136**. Cutting BN from 64 to 16
> multiplies the grid and divides the slab each workgroup unpacks, for the
> same total unpack: **the block goes 612.3 us to 479.9, 1.28x**, with the
> shared expert's up at 46 GB/s against 23.
>
> **L7d-4: one command buffer a pass, and the attribution had to move with
> it.** Nothing in the shim ever prevented mixing blocks — it binds each
> dispatch's own pipeline, descriptor set and push constants. What stopped it
> was that every number here came from the host wall clock around a block's
> `Run`. So `vk.DispatchMultiMarked` writes a timestamp after **every**
> dispatch and a block's figure becomes GPU time *inside* the command buffer,
> which is the same quantity `GGML_VK_PERF_LOGGER` reports for llama.cpp's
> graph and cannot be inflated by a fence wait. ~490 submits become **two**,
> 1405 dispatches a token; **99.0 ms a token becomes 84.1, 1.18x**, and the
> hand-over left is **0.96 ms**.
>
> **L7d-5: the first rung in this vertical that is not bit-exact against its
> ladder.** Every other rung tiles the same arithmetic differently and never
> re-associates a sum, which is why two ladder tests assert `maxAbs == 0`.
> Splitting a 10240-long dot product 160 ways does re-associate it, and at one
> token it is the rung the schedule picks — exactly the chunk length L7b's
> gate is about. So the gate is pinned for its three equalities, which still
> hold **to the last place**, and the decode schedule is measured beside them
> at **4.43e-04 rms — 0.0024% of scale, 4000x smaller than a dropped
> history**, and the same order as our distance from llama.cpp itself.
>
> **L7d-6: a token is 84.1 ms at 122 GB/s, half the bus.** MoE 33.9%,
> DeltaNet 32.1%, hyper-connection 10.6%, attention 9.3%, head 7.5%, moves
> 0.7%, host 5.4%. The hyper-connection block went from 6.9x off its bytes to
> **1.65x** and is no longer where to look; **the MoE at 4.3x is the only
> block still far off**, and **the DeltaNet is now the largest single block**
> at 1.56x off its own 4.18 GB. Underneath both is L8: 8.07 of those 9.67 GB
> are a dense half staged as halves.

## What L8a established — the dense bank, and a round trip that is an identity

`llm/bank.go`, `shaders/llm_gemm.comp` (`-DQ8B`), `shaders/llm_common.glsl`,
`llm/gpu_head.go`, `llm/gpu_deltanet.go`, `llm/gpu_attn.go`,
`cmd/llm/bench_head.go`: decode at 14.07 tok/s.
[Write-up](research/l8a-dense-bank.md) · `results/l8a_decode.csv` ·
`results/l8a_decode_fp16.csv` · `results/l8a_graph.csv` ·
`results/l8a_head.csv`

> **L8a-1: the round trip is an identity, so the gate is an equality rather
> than a tolerance.** A dense weight is staged as int8 in the same §2.8
> fragment tiling — 256 contiguous *bytes* a tile — with one fp16 scale per 32
> elements of a row in a k-major plane behind the tiles (D8). The scales are
> not new information: ggml's `quantize_row_q8_0` picks `d = amax/127`, so the
> largest level in a block is **always 127**, and re-deriving (d, q) from the
> dequantised floats returns exactly the checkpoint's pair — `127*d` is exact
> in f32, dividing by 127 is correctly rounded onto a representable result,
> and `round(q*d*(1/d))` is `q`. Then `float(q)*float(d)` is exact and
> rounding it to fp16 is what `tileB` wrote. So the kernel puts **the same
> halves** in LDS that the fp16 arm loaded from global.
> `TestBankQ8IsTheHalves` and `TestHeadGPUQ8IsTheHalves` are equalities, and
> on the real `output.weight` **0 of 248320 logits differ**.
>
> **L8a-2: three families are not Q8_0, and they cost a tail rather than a
> tolerance.** `ssm_alpha` and `ssm_beta` are F32, the indexer's two
> projections are BF16, `hc_inject` is F32, and all three are fused into the
> middle of somebody else's matrix. Quantising them with the rest **failed the
> DeltaNet layer's own oracle** — 1.31e-04 rms to **2.99e-03**, past the
> 2.0e-03 asserted since L3b — which is the arithmetic one would predict: a
> 32-element int8 group carries ~7 bits relative to its own maximum, so a
> 2560-long dot product lands ~1% from f32 against fp16's ~0.05%. They begin
> **on a column-block boundary** in every case, so the answer is a split: the
> Q8 arm reads `lowRank` as the first row that is not int8 — which is what it
> already means in MODE 0 — and `gateOff` as where those rows' halves are, and
> a whole 64-column workgroup falls on one side. With the tail back, every
> tensor of both layers is identical to the fp16 bank's.
>
> **L8a-3: a compare in the k-loop cost 1.27x on a matrix with no tail.**
> Branching per tile between an LDS fragment and a global one took `ssm_out` —
> Q8_0 throughout, `tail` false in every workgroup — from **125.9 us to 160.0**,
> which is the fp16 arm's own 163.3. Two `coopMatLoad`s in one loop body are
> two fragments the compiler keeps live, and register pressure is what both
> projection ladders were already bound by (L2f-5, L3b-4). Writing the loop
> twice, hoisted above the k-loop, puts it back at 125.9 exactly.
>
> **L8a-4: the unpack costs 4% of the bus, and a tiny second dispatch is not
> free.** `-head` stages `output.weight` on both banks in one process: at one
> row the bank is **1.87x** smaller and the dispatch **1.80x** faster (6454.4
> to 3585.7 us, 197.0 GB/s to 188.4), so the byte extract, the convert, the
> multiply and the LDS store cost 4% of the bandwidth and nothing else. At 512
> rows it is 1.43x, which is the re-read curve every projection ladder here
> has. The tail was a *dispatch* first, and a [128, 2560] fp16 GEMM at one
> token is two workgroups on a 40-CU device: **70.2 us for 0.8% of the rows**
> against 325.1 for the other 99.2% — D11 again — so folding it into the same
> grid was 1.20x on the fused projection.
>
> **L8a-5: 1.18x at decode, 1.055x at prefill, and 1.8 GB of residency.** The
> DeltaNet layer is 1.50x at one token and 1.21x at 512, the attention layer
> 1.46x and 1.21x, the head 1.80x and 1.43x. End to end: **84.0 ms a token
> becomes 71.1**, DeltaNet 27.1 to 18.8, attention 7.8 to 5.6, head 6.3 to
> 3.5, with the MoE unchanged at 28.5 by construction. Prefill at ubatch 2048
> is 2067.4 ms to **1960.2**. Two runs agree to 0.36% and the fp16 control
> reproduces L7d exactly (11.90 against 11.89).
>
> **L8a-6: the ceiling moved, and the MoE is now the token.** A token reads
> **6.61 GB against 9.67** — dense 5.01 against 8.07 — so this bank's own
> limit is **36.6 tok/s** rather than 25.0, within 4% of the 38.2 the
> checkpoint's width allows, and we are at 38% of it. The step is MoE 40.1%,
> DeltaNet 26.5%, hyper-connection 13.3%, attention 7.8%, head 4.9%. The MoE
> is 4.3x off its own bytes where the head is 1.25x; **the bytes have stopped
> being the thing to fix.**

## What L8b established — the last dense family, and a split that is a column block

`llm/gpu.go`, `shaders/llm_gemm.comp` (MODE 0 and MODE 1 on `-DQ8B`),
`shaders/llm_hc_gemv.comp` (`-DQ8B`), `llm/gpu_test.go`, `cmd/llm/bench.go`:
decode at 14.71 tok/s, the block 2.14x at one token.
[Write-up](research/l8b-hc-bank.md) · `results/l8b_decode.csv` ·
`results/l8b_graph.csv` · `results/l8b_graph_fp16.csv` · `results/l8b_hc.csv` ·
`results/l8b_hc_fp16.csv`

> **L8b-1: the split is a column block, not a row, and 32 rows are staged
> twice.** L8a's tail needs a whole BN block on one side of it, because a
> compare inside the k-loop was 1.27x on a matrix with no tail (L8a-3). The
> hyper-connection block cannot give it one: `inject` is four F32 rows at
> lowRank = 320 of a fused N of 336, and BN is 48 because 336 is 21 tiles and
> 21 is 3 x 7. So the split is the column block that *contains* the first
> non-Q8 row — `pc.lowRank` rounded down to BN, **288** — and the 32 low-rank
> rows between 288 and inject live in both planes: in the int8 one, where
> they are never read, and in the fp16 tail, where they are. **0.33 MB a
> mixer, 32 MB a decode token, 0.5% of a step**, against a branch that stays
> per workgroup. Both kernels derive the split from `pc.lowRank` and their own
> column block, so it costs no push field — and the block has been full since
> L5b. The up projection needs none of it: `hc_up.weight` is Q8_0 for all
> 10240 rows, so MODE 1's tail branch is **compiled out** rather than guarded,
> which matters twice because in that mode `pc.gateOff` is the validation
> gate's arena and not a weight.
>
> **L8b-2: the GEMV has no LDS to round through, so it rounds in fp16.** The
> GEMM arm is bit-exact because it *stores* `float16_t(float(q) * d)` into
> LDS. `llm_hc_gemv.comp` has no LDS stage at all, and `float(float16_t(x))`
> in a register is folded away by RADV (D10) — so its effective weight would
> have been the unrounded `q*d` and the two kernels would disagree by a weight
> ulp over a 10240-long dot product, which on `inject`'s |71.7| is ~1e-2,
> an order of magnitude past L7d's own agreement bound. `float16_t(q) * d` is
> a real f16 multiply of two exact halves: nothing for the compiler to fold,
> and **the same half the LDS store produces**, because q carries at most 8
> significant bits and d at most 11, so their f32 product is exact and the
> conversion to fp16 is the only rounding either path does.
>
> **L8b-3: the ladder slid one rung along the 4 KB rotation, and D12 said it
> would.** A workgroup's slab is now `(gemmK/16/KSLABS) * 256` bytes and not
> 512, so the whole table shifts: at 8/16/32/40/80/160 slabs the strides are
> 5/2.5/1.25/1/0.5/0.25 times 4 KB and the rates are
> **231/269/326/135/193/308 GB/s**. The two whole multiples are 8 and 40 and
> they are the two slow rungs, 40 by 2.4x. On halves the same law picked 160
> and 32; on bytes it picks 32 and 160 — the same two strides, one rung along.
> `PlanFor` now takes the bank as an argument.
>
> **L8b-4: the collapse's scratch was [BM][BN], and the unpack wanted the
> widest row block there is.** MODE 1's unpack costs `BN*BK` conversions per
> slab against `WM*WN*BK_TILES` matrix steps — **256/WM per step, and nothing
> else in the shape changes it**; deeper slabs confirmed it rather than fixed
> it (BK_TILES=4 was 1334 us and BK_TILES=1 was 984, against 1003 at 2). But
> this kernel is latency-bound and gets slower as its LDS grows — on the fp16
> bank BM=64 was 1.38x BM=32 at 2048 tokens — so a rung wide enough to
> amortise the unpack did not fit beside a [BM][BN] gate. Cut to **one
> m-tile** the scratch is 4 KB at every rung, at two barriers a tile instead
> of one, and the arithmetic is untouched. Both banks gained: the fp16 arm's
> own best up projection went 176.3 to 142.0 us at 512 tokens and 720.8 to
> 638.2 at 2048, and on the Q8 bank **up_m4 wins from 64 tokens up** where the
> fp16 arm is on m2 nearly everywhere.
>
> **L8b-5: at one token the bank is the kernel; at 2048 it is nearly
> irrelevant.** Microseconds a mixer, each projection at its own best rung:
> down 30.0 to **12.8** and up 40.0 to **16.1** at one token (2.34x and
> 2.48x); down 254.5 to 253.2 and up 142.0 to 152.8 at 512; down 457.3 to
> 525.6 and up 638.2 to 694.5 at 2048. At 2048 the two weights are read 32
> times each out of a 32 MiB MALL, so halving the DRAM bytes buys what the
> cache had already hidden and the unpack's ALU is left over — **1.15x and
> 1.09x the fp16 arm**. The whole graph absorbs it: prefill is 1041.1 tok/s at
> ubatch 2048 against L8a's 1044.8 and 643.4 at 512 against 644.1, inside the
> 0.36% two runs agree to.
>
> **L8b-6: every dense weight is now the checkpoint's own width.** Decode is
> **14.71 tok/s against 14.07** (two runs, 0.34%), a token 71.1 ms to 68.0,
> the block's share of a step 13.3% to **8.7%**, residency 82.40 GB to
> **81.89**. A token reads **6.05 GB** against 9.67 at L7d and the 6.334 the
> checkpoint holds, so this bank's ceiling is **40.0 tok/s** and we are at
> 37% of it. `TestGraphLogits` returns llama.cpp's own argmax out of its own
> top ten with the drift unchanged, and `-gen -n 128` is L7c's completion
> down to `Lisbon.`

## What L8d established — the decode kernels, and five grids said the same thing

`shaders/llm_moe_gemv.comp`, `shaders/llm_moe_router.comp`,
`shaders/llm_gemv.comp`, `shaders/llm_moe_combine.comp`, `llm/gpu_moe.go`,
`llm/gpu_deltanet.go`, `llm/bank.go`, `llm/graph.go`: decode at 23.15 tok/s,
the MoE block 3.06x and the DeltaNet layer 1.87x at one token.
[Write-up](research/l8d-moe-decode.md) · `results/l8d_decode.csv` ·
`results/l8d_decode_gemm.csv` · `results/l8d_graph.csv` ·
`results/l8d_moe.csv` · `results/l8d_dn.csv`

> **L8d-1: the MoE's tile grid was a static bound of 2052 records where eleven
> exist, and it is a dispatch and not a bookkeeping number.** The grouped GEMM
> dispatches an upper bound of (expert, row block) records because the real
> count is built on the device, and the bound was `maxRows/bm` — which lets
> **every one of the 512 experts** have an alignment of slack, right for the
> row space and wrong for the tile list. A batch of `rows` tokens routes
> `rows * used` rows and so pads at most that many experts: **ten**, at one
> token. At the GEMM's own BN of 64 the down mode was 40 x 2052 = 82 080
> workgroups to run 11, ~33 us a layer at about 0.5 ns an empty launch. The
> new bound is exact at one token and 1.74x tighter at 512. **It is also why
> the first GEMV ladder read as a failure** — the routed down mode at BN=16
> measured 481 us against the GEMM's 84, of which 330 was empty launches, and
> at BN=1 it was 3666 us for 5.25 M of them. A kernel whose whole purpose is a
> narrower column block cannot be measured against a grid bound that
> multiplies with it.
>
> **L8d-2: the combine was one workgroup, and its grid has an axis order worth
> 9.2x.** `ffn_out` at one token was a single 256-lane workgroup walking 2560
> columns — 112 KB of reads on a 40-CU device, 22 us at 6 GB/s. Splitting the
> column axis is one line and is **7.8x**. But the obvious spelling puts the
> token on x, and Vulkan varies x fastest: consecutive workgroups are then
> consecutive *tokens* at the same 256-byte slice, and a token's eleven rows
> are 10 KB apart in a 566 MB arena. At ubatch 2048 that was **815 us to
> 7505**. With the column block on x the forty workgroups of one token walk
> its rows end to end: **1104.9 us at 228 GB/s**, 1.35x faster than the single
> workgroup it replaced, at prefill as well as decode.
>
> **L8d-3: the router was nine workgroups, and D12 does not reach it.**
> `ffn_gate_inp` is [513, 2560] fp16, 2.63 MB — the smallest weight in the
> block by two orders of magnitude and **27.7% of it at decode**, 64 us at 41
> GB/s. 576 padded columns at BN=64 is nine workgroups, so the parallelism has
> to come from K: `llm_moe_router.comp` is L7d's split-K over the same §2.8
> tiling with no epilogue, **4.4 us plus a 1.0 us reduce, 11.8x**. Its KSLABS
> ladder is **flat** — 4.6/4.5/4.9/4.3 us at 8/10/20/40 — where the same
> kernel spans 2.4x on the hyper-connection block, because §5.1b's 4 KB
> rotation is a DRAM channel effect and 2.63 MB never leaves the MALL. The
> logits are checked as numbers *and* the top ten as a sequence, since a
> router's output is discrete: rms 3.4e-05, **0 of 10 slots differ**.
>
> **L8d-4: at one token the expert GEMM is the wrong kernel, and a dot product
> has no fragment.** L5b's slab-into-LDS arrangement is right at prefill,
> where the unpack is amortised over 16 to 64 rows, and at decode there is
> **one** row: fifteen sixteenths of every row block is padding and the K-loop
> is a barrier, a dependent load and a consume with one wave in flight. L7d
> narrowed BN and got 71 to 93 GB/s and no further. `llm_moe_gemv.comp` drops
> the slab, the barrier and the cooperative matrix — §2.2's rule is about a
> *fragment*, and this has none — so a lane unpacks its own dwords into
> registers and a whole row's loads are in flight at once. Microseconds a
> layer: routed up 196.5 to **78.1** (94 GB/s to 236), routed down 84.9 to
> **34.5** (145 to 356), shared up 75.9 to **12.3** (46 to 284), shared down
> 11.8 to **7.3**; the block **464.5 to 152.0, 3.06x**, and 58 GB/s to 133 end
> to end. **The lane count inverts between two dispatches of the same kernel**
> — the routed pair wants LPR=16 and the shared expert LPR=64, because Q4_K
> packs 320 payload dwords into a 2560-long row and Q8_0 packs 640.
>
> **L8d-5: the same finding on the dense projections, and D12 *does* reach
> them.** `llm_gemv.comp` is the split-K with the hyper-connection epilogue
> removed, so it serves any `llm_gemm.comp` MODE 2 caller over either bank,
> tail and all. The gated DeltaNet — 36 of the 48 layers — goes **454.0 us a
> layer to 242.5**: qkv 316.8 to 199.1 and out 124.3 to 26.7, which is 5.4x on
> a projection whose grid was 40 workgroups. A slab is
> `(gemmK/16/KSLABS) * 256` bytes, so **the rung moves with K**: at K = 2560
> the whole multiples of 4 KB are 1 and 2 slabs (275.8 and 308.9 us against
> k8's 196.2) and at K = 6144 they are 1, 4 and 8 (32.8, 25.8, 30.7 against
> k32's 22.0). A ladder measured on one matrix does not carry to another of
> the same kernel.
>
> **L8d-6: nothing reads a different weight, and the text moves at one token
> anyway.** Every byte is staged as L5b and L8a staged it; what changes is the
> order of a sum — a cooperative-matrix accumulator adds sixteen k an
> instruction in an undefined order, a GEMV lane adds them serially, a split-K
> reduce adds the slabs afterwards. Against the GEMM over one token that is
> rms 2.5e-06 on the MoE's `ffn_out` (scale 0.032), 1.75e-05 on the DeltaNet's
> fused projection (31.5) and 3.4e-05 on the router's logits (7.46), and it
> **does not move with the rung** — every split of the qkv lands within 0.2%
> of the same number, where a mis-read scale plane would be different at each.
> `TestGraphLogits` is unchanged (argmax 561, llama.cpp's own top ten, the
> drift the same x1.085 a layer) and `Graph.PinSchedule` now covers four
> blocks, so `TestGraphIsAChunkSplit`'s equalities hold to the last place.
> **At temperature zero the `<think>` block re-words at completion token 47**,
> where the top two logits are **0.012 apart** against the 3-25 that decide
> the tokens either side — the same class of thing L7c's own 0.161 divergence
> from llama.cpp is, an order of magnitude tighter. The capitals list and
> `Lisbon.` are identical, and `LLM_DECODE_GEMM=1` reproduces L7c's completion
> exactly.

## What L8e established — the last projection, and a ladder that lied

**Decode is 24.66 tok/s against L8d's 23.15 — 1.065x — a token 40.5 ms against
43.2, and prefill faster at both ubatches: 1052.8 tok/s at 2048 against 1049.8
and 655.2 at 512 against 653.9.** No shader changed and no bank moved. The
full-attention layer is 5.5 ms a token to **3.5**, the MoE 12.1 to **11.7**,
and a pass is **1501 dispatches at decode and 1261 at prefill**, 48 fewer at
both. [Write-up](research/l8e-attn-decode.md).

> **L8e-1: the full-attention layer is D11 a fifth time, and 1.84x.** Its two
> projections were still `llm_gemm.comp` MODE 2 at one token — 218 and **40**
> workgroups for a sixteen-row fragment holding one real row —
> and `llm_gemv.comp` was already generic over both banks and both tail cases,
> so the work was the wiring and a ladder. The layer is **414.8 us to 225.9**:
> qkv 263.8 to 174.3 and out 124.9 to 24.6. D12's rung is measured rather than
> copied and lands on the DeltaNet's two anyway — **k8** at K = 2560, where
> k2's 20 480-byte slab is five whole 4 KB pages and the worst rung on the
> ladder, and **k32** at K = 6144. **The fused projection reads its 39.5 MB of
> bank at 227 GB/s of a 242 GB/s bus**, which is the first dispatch here to be
> *at* the bus rather than near it. Its last 640 rows are the indexer's BF16
> pair in an fp16 tail, and every rung agrees with the GEMM at rms
> 2.93-2.95e-06 over 13 952 columns — 9.89-9.98e-06 on the tail alone, a 0.4%
> spread across six splits.
>
> **L8e-2: the MoE's shared expert has a lane group of its own, and half of
> what that was scoped as was wrong.** Read one dispatch at a time rather than
> by block total, the routed pair and the shared expert want opposite ends of
> the LPR ladder — Q4_K packs 320 payload dwords into a 2560-long row where
> Q8_0 packs 640 — and one field had to take the better *joint* rung. The
> shared down mode goes v16w4 to **v32w4**; the shared up mode was already on
> v64w4 for the wrong reason. **But carrying the routed up mode to the v16w4
> the ladder names is 42.1 ms worse over 64 tokens**, which is D16.
>
> **D16: a micro-bench rung whose rate is above the bus is not a DRAM
> measurement.** `-moe -tokens 1 -ladder` stages two layers and repeats one
> dispatch; a token's ten routed experts are ~32 MB, **the MALL exactly**, so
> every repetition after the first reads at 805-965 GB/s. v16w4's 63.7 us over
> 18.4 MB is **288 GB/s**, a rate this machine does not have. D12 said a
> ladder does not carry across a bank, a width or a matrix; D16 is that it
> does not carry across a **residency**, and that one is the harness's fault
> rather than the kernel's.
>
> **L8e-3: the second counting sort was the first one again.**
> `llm_moe_perm.comp` ran twice a layer because the two expert modes are cut
> to different row blocks — except that at decode all six GEMV rungs fix BM at
> sixteen, and at prefill `m2/m2` and `m4/m4` are equal too. One pass, the
> down mode reading the up mode's tile list: **-48 dispatches a pass at every
> length**, 2.9 us a layer at one token and 17.8 at ubatch 512. What is left
> alone is `route` — 5.8 us a layer, one workgroup, and the only axis it has
> across is the token.
>
> **L8e-4: the text is tighter than L8d's, by luck.** Over 128 tokens at
> temperature zero the completion is **identical to `LLM_DECODE_GEMM=1`'s,
> token for token** — so L8e lands back on L7c's text, where L8d had re-worded
> one sentence of the `<think>` block at a tie 0.012 apart. The extra
> reassociation flipped the tie back; that is luck and not a property, and the
> honest claim is still "the same text at this precision, with a tie in it".
> `TestGraphLogits` is unchanged (argmax **561**, llama.cpp's own top ten,
> drift 0.881% at 48 layers) and `Graph.PinSchedule` now covers **five**
> blocks.

## What L8c-4 established — the bank is the format, and a screen that under-reads

**The 4.5-bit asymmetric bank L8c-3 recommended now exists, and the first
thing it had to prove it proved as an equality: on the lm head its 248 320
logits are identical to the same format run through `sim.go`.** The head is
0.675 GB in the checkpoint's own Q8_0 and **0.358 GB** at 4.500 bits, **1.89x
at one token** — both banks run the decode GEMV at 210 GB/s of a 242 GB/s bus,
so the bytes *are* the time — and 3.80x against the halves it staged before
L8. In the whole model **decode is 25.75 tok/s against L8e's 24.66 — 1.044x, and **1.024x
llama.cpp**, the first time this vertical is ahead of it at decode — with a
token 40.5 ms to **38.8** and the head block 3.5 ms to **1.9****, and **perplexity is 4.0621
against our own 4.0289 — +0.82%**.
[Write-up](research/l8c-dense-bank.md) · `results/l8c_head.csv` ·
`results/l8c_ppl_head_q4k.csv` · `results/l8c_decode_q4k.csv`.

> **L8c-4-1: one encoder, two callers, and that is the whole correctness
> argument.** `applyAsym` used to do ggml's super-block arithmetic inline and
> throw the levels away, because a simulation only needs the floats.
> `llm/quantk.go` is that arithmetic as an object that keeps them, so the
> simulation writes back `d*sc*l - dmin*m` and the bank packs the levels the
> same value came from — "this is the format L8c-3 measured" is true by
> construction rather than by a tolerance. What is left to test is the
> *addressing*, which is the only bug a new bank can have, and
> `TestBankQ4KIsTheSim` tests it over a row permutation rather than the
> identity.
>
> **L8c-4-2: the layout is our tiling and ggml's record.** §2.8's fragment
> tiling with a nibble where L8a puts a byte — a `uint` is **eight**
> consecutive k of one output column, two to a byte, low nibble first — and a
> second plane of sixteen-byte records, one per (n-tile, super-block, row):
> `d`, `dmin` and the twelve bytes `get_scale_min_k4` reads eight 6-bit
> (scale, min) pairs out of, k-major inside an n-tile for D8's reason. 4 bits
> of level, 12 bits a group of 32 and 32 bits a super-block of 256 is
> **4.500 bits a weight exactly**. The twelve bytes are ggml's packing and not
> a convenient one of our own, which costs a four-way branch per group and
> buys the thing the stage is about: the encoder that produced L8c-3's numbers
> is the encoder of the bank.
>
> **L8c-4-3: a record covers 256 k where a k-slab is 32, so it is held in a
> register.** Read inside the unpack it would be fetched eight times. In the
> GEMM each lane holds *its own column's* record, refreshes it only where the
> loop crosses a super-block, and turns it into `(d*sc, dmin*m)` in a BN-long
> LDS vector the unpack indexes — the format's own 12.5% of record traffic,
> and no `get_scale_min_k4` in the inner loop. It needs BK = 32, which is
> every build in `shaders.go`, and the kernel `#error`s rather than assuming
> it. The GEMV does the same in a register across the four steps of a
> super-block and has no LDS at all.
>
> **L8c-4-4: the GEMV is not bit-exact against the GEMM here, and cannot be.**
> L8b-2's argument was that `float16_t(q) * d` is one f16 instruction over two
> exact halves. A K-quant group is affine — `d*sc*l - dmin*m` has a
> subtraction in it — the GEMV does it in f32 and the GEMM rounds the result
> to a half on its way into LDS, so the GEMV carries one **fewer** rounding
> per weight. That is `llm_moe_gemv.comp`'s position against
> `llm_moe_gemm.comp` over the same format, and it is measured: over the
> head's 248 320 logits the two are **rms 2.29e-04 relative**, against the Q8
> pair's 7.84e-06, which is reduction order alone.
>
> **L8c-4-5: D14 gains a clause, because the head is its counter-example.**
> A narrower bank was a decode decision — L8b-5 measured the
> hyper-connection block 2.3-2.5x faster at one token and **1.09-1.15x
> slower** at ubatch 2048, since at 2048 each weight is read 32 times out of a
> 32 MiB MALL and the DRAM bytes were already hidden. The head's B is 357 MB
> at 4.5 bits and fits the MALL at no width, so there is nothing to hide
> behind: **18839 us to 17461 at 512 rows, 1.08x, the way D14 warns against**.
> The unpack does show up — 82 GB/s of bank against the Q8 arm's 143 — it just
> does not win. So D14 is about a weight that fits the MALL.
>
> **L8c-4-6: the per-family screen under-reads, by 2.05x on the one family now
> run both ways.** L8c-3's table puts `lm_head q4_k/32` at **+0.40%**, and the
> real bank reproduces that eight-chunk screen to four decimal places —
> 2.0269 against 2.0269. Over 145 chunks the same bank is **+0.82%**. L8c-0
> established that a short run is not calibrated for a percent on the whole
> model; this is the same fact one level down, and it matters because every
> per-family row in `results/l8c_asym.csv` is eight-chunk. The uniform plan's
> +4.24% is itself a corpus number and stands; the attribution under it is
> optimistic.
>
> **L8c-4-7: the text is no longer a gate, and the divergence is one
> near-tie.** At temperature zero the completion parts at the third token of
> the body, and `-top 3` shows both runs offering the *same two* candidates
> there: the Q8 bank takes `561 " The"` at 15.590 over `11751 " Paris"` at
> 15.229 — **0.361 apart** — and the 4.5-bit head takes `" Paris"` at 15.537
> over `" The"` at 15.462, **0.075**. The bank moved that pair by 0.36 of a
> logit and flipped a tie that was already thin. Downstream the two texts
> share nothing — capitals ending at `Lisbon.` in 69 tokens against a
> syllogism about Paris for the full 128 — but that is a five-token prompt
> branching, not the head going wrong. This is the first stage in the vertical
> where the two were ever allowed to differ: D17's gate is a perplexity, and
> the text gate retires with L8b.

---

## What L8c-5 established — the largest family, and a screen with the wrong sign

**1. The gated DeltaNet is on the bank, and it needed no new kernel.** 36 of
the 48 layers, 2.087 B parameters, 2.247 GB of a 6.334 GB token — 46% of the
dense half and the largest single dense thing in the model. `llm_gemm.comp
-DQ4B` and `llm_gemv.comp -DQ4B` already had every arm, **including the fp16
tail branch nothing had exercised**: `ssm_alpha` and `ssm_beta` are the
checkpoint's only F32 matrices here and they stay where L8a put them (D13),
at a split of 16384 that is a whole column block by construction — the easy
case L8b's `inject` was not. **A layer is 62.28 MB to 33.28 and 36 of them
2.24 GB to 1.20; residency 81.57 GB to 80.53.** Two host changes made it
possible and both are about names rather than widths: `tileBQ4K`'s
destination row count is the **plane's** and not the source's, so four of
llama.cpp's matrices can be staged into one fused plane, and
`DeltaNetWeights` carries its layer index, so each source is calibrated under
**its own tensor name** — which is what the simulation the format was
measured with did.

**2. The bank is the format, said as two equalities.**
`TestDeltaNetGPUQ4IsTheSim` stages one layer twice from the same weights —
the fp16 bank with `sim.go` applied, and the real 4.5-bit bank — and the
fused projection's 115 360 values *and the whole layer's 17 920 output
values* come back **identical**. Over the corpus, the bank and its simulation
agree **chunk for chunk to four decimals** over eight chunks, with
`LLM_DENSE_SIM_SRC=q8` as the thing that makes them the same question: the
simulation's default scope includes the two F32 matrices the bank leaves in
its tail.

**3. Decode is 28.9 tok/s and the block is 1.60x.** 10.9 ms a token to
**6.8**, 28.1% of a token to 19.8%, a token 38.8 ms to **34.4**, and the
whole model **25.75 tok/s to 28.73 and 29.07 on two runs — 1.15x
llama.cpp's 25.15**. The block moves 176 GB/s of staged bank against the Q8
bank's 205, which is the unpack's price on a bank that halves; netting out
the ~0.5 ms a token of scan, convolution, norm and history that read no bank
puts the two projections at roughly 190 against 215.

**4. Prefill is faster, and D14 gains its boundary rather than another
side.** **1070.1 tok/s at ubatch 2048 against 1052.8 and 667.0 at 512 against
655.2** — the block 350.7 ms to 320.5 and 96.5 to 86.7. D14 said a narrower
bank costs at prefill because the re-reads come out of the MALL; L8c-4 added
that this is about a weight that *fits* it. This block's fused projection is
84.5 MB as halves, **44.9 at int8 and 23.8 at 4.5 bits** — it **changes
side** of the 32 MiB MALL, so sixteen re-reads a graph come off DRAM
entirely. Three blocks, three answers, one rule: the hyper-connection block
fits at every width and was 1.09-1.15x slower, the head fits at none and was
faster, this one crosses and is faster.

**5. The finding is the accuracy, and it is about the instrument.** The
family costs **+0.93% over 145 chunks — 4.0665 against 4.0289 — where
L8c-3's eight-chunk screen said −0.30%.** That screen was not a noisy
estimate: on the first 8184 tokens of `wiki.test.raw` the calibrated 4-bit
DeltaNet is *better than the checkpoint's own Q8_0*, and it was quoted as
such. So L8c-4's "the screen under-reads by 2.05x" is the weaker statement of
the two; **a per-family row does not fix the sign**, and the two families
measured both ways now sit +0.42 and +1.23 percentage points above their
screens. Every remaining width has to be called from a 145-chunk run.

**6. What does hold is adding corpus numbers.** `lm_head` at +0.82% and
`deltanet` at +0.93% together measure **4.1012, +1.79%**, against a sum of
1.75 — 0.04 pp of compounding over two families, where L8c-1's four symmetric
families summed to 11.7% and measured 18.5%. L8c-3 predicted the asymmetric
form would compound mildly; this is the first built confirmation, at 36 and 1
depths against L6b-3's x1.085 a layer.

**7. The decode rungs did not move, and the ladder that says so is D16.**
D12 says a split-K ladder does not survive a change of bank, so it was re-run
on nibbles: k8 for the fused projection and k32 for the output one, which are
L8d's own two. But every rung reads **1700-1800 GB/s**, seven times the bus,
because the bench stages a handful of layers over a bank that fits the MALL —
so what it ranks is kernel shape against an L3 hit, the spread is 1.18x where
L7d-2 saw 2.4x in DRAM, and the rung whose slab stride *is* a whole multiple
of 4 KB is only the second-worst. Nothing moved and nothing could be trusted
to; the incumbents stay because no candidate disagrees.

---

## What L9a established — the chat template, and three envelopes over one loop

> **L9a-1: the prompt is metadata, not a design.** A chat request is a list of
> messages and this model reads one string. What turns one into the other is
> `tokenizer.chat_template` in the GGUF's own metadata — **180 lines of
> Jinja** that merge the leading system turns, add a reasoning-effort
> sentence, write the tool block, keep each assistant turn's `<think>` and
> open the generation prompt. A server that improvises it does not fail; it
> answers slightly wrong, everywhere, with nothing to point at. So
> `llm.RenderChat` is a **transcription**, and the gate is Jinja's own output:
> `go run ./cmd/llm -chat-template` exports the template,
> `reference/dump_chat_template.py` renders **23 cases** with the environment
> transformers builds, and `TestRenderChat` diffs them character for
> character.
>
> **L9a-2: four differences hid in one filter.** The template writes a tool
> with `tool | tojson`, and transformers binds that to
> `json.dumps(ensure_ascii=False)`. Go's `encoding/json` writes no space after
> a separator where Python writes one, escapes `<`, `>` and `&` where Python
> does not, escapes U+2028 where Python does not, and sorts a `map[string]any`
> where Python keeps insertion order. Each is legal JSON and each is a
> different prompt, so `llm.pyJSON` re-encodes Python's way and `api.Tool`
> keeps **the bytes the client sent** rather than round-tripping through a
> struct — whose field order is its own and which would add every field the
> client left out.
>
> **L9a-3: the boundaries are markers, and markers straddle tokens.** The
> generation prompt leaves `<think>` open, so the model's first token is
> reasoning and `</think>` is where the answer starts; the tool block asks for
> `<tool_call>`. Both can be split across two tokens, and a `<` already sent
> to a client as an answer cannot be taken back. `llm.ChatDecoder` holds back
> any tail that could still be a marker's beginning — and any trailing
> whitespace, because the blank line before a call is the format's separator
> and not the answer's — which is what makes the streamed and the buffered
> answer the same string.
>
> **L9a-4: reading a call back needs the schema.** The template writes a
> string argument raw between its tags and everything else as JSON, so `123`
> is the number for an `integer` parameter and the text `"123"` for a `string`
> one. Rebuilding OpenAI's `arguments` therefore reads the request's own tool
> schema; without one, valid JSON is taken at face value and anything else is
> a string. Tool calls are also the one thing not streamed: a call is not a
> call until it has closed.
>
> **L9a-5: three envelopes, one generation.** `api.CompletionBackend` is a
> *streaming* interface — `Complete(ctx, req, emit)` — and the buffered
> response is a string builder over it, because the reverse cannot be written.
> `/v1/chat/completions`, `/v1/responses` and `/v1/messages` are then
> translations in and out of it, each with its own SSE taxonomy and none with
> a second copy of the loop. `go test ./api/` runs all three, streams
> included, against a fake in **5 ms** and needs no checkpoint.
>
> **L9a-6: a second turn is a continuation.** A chat client re-sends the whole
> conversation every turn, so the naive server prefills ten turns of history
> to answer the tenth. The adapter keeps the token sequence the graph is
> holding and `Extend`s the new tokens onto it, which is the same call a
> generated token makes and is covered by L7b's gate. Measured on the second
> turn of a two-turn conversation: **90 prompt tokens, 71 reused, 216 ms
> against 421**. The reuse is all-or-nothing because the **gated DeltaNet has
> no rewind** — the attention cache is masked by position and both convolution
> rings are addressed by it, so those would survive a fork, but a DeltaNet
> layer's recurrent state is a running product with no inverse.
>
> **L9a-7: it answers.** `go run ./cmd/serve -llm` stages the whole model in
> **33.5 s**, and then: "What is the capital of France?" with thinking off is
> `Paris` in **356 ms**; a two-step arithmetic question at effort `low`
> separates 42 tokens into a `reasoning_content` and a `content` of `15:55`;
> and a tool request comes back as `get_weather({"city":"Wimbledon","days":2})`
> — the `days` typed as a number by L9a-4 — in 4.6 s. Decode measures
> **22.71 tok/s** through the server, against L7d's 11.89 on the checkpoint's
> own width.

---

## What P0 established — a two-second ring, and a poll with no timeout

> **P0-1: the hang was never in the submit, and `[syscall]` was never a
> syscall.** L8c-7 read a `SIGQUIT` dump as placing the stalled goroutine
> "inside the driver's submit" and suspected residency. Go writes `[syscall]`
> for *any cgo call*; the thread's own accounting separates the two, and it is
> not close — **`utime=5645`, `stime=5`**, 56.45 s of user time against 0.05 s
> of system time, `wchan` 0, state `R`. Sampled for the length of a hang,
> `stime` never moves and `utime` climbs exactly 100 jiffies a second. **And it
> is not residency**: `mem_info_gtt_used` is flat at 80.198 GB, VRAM at 7.423
> and `VmRSS` at 31.56 for the whole stall — nothing allocates, faults or is
> evicted. Nor is it the staging: 3072 rows hangs in a staging sized for 3072
> exactly as it did in L8c-7's sized for 4096.

> **P0-2: the backtrace, and the six lines everyone had walked past.**
> `ptrace_scope` is 1 here, so gdb has to be the *ancestor* — launch under
> `gdb --args`, `handle SIGUSR1 stop`, signal the inferior. The frame is
> `clock_gettime` under three of `libvulkan_radeon.so` under
> **`shim.c:941`**, which is `vkGetQueryPoolResults` with
> `VK_QUERY_RESULT_WAIT_BIT` — six lines *past* the `vkWaitForFences` that was
> supposed to bound it, and which had already returned `VK_SUCCESS`. On RADV
> that flag is an unbounded **userspace** poll of the mapped slot. So the
> "20-second fence timeout that never fires" was never going to: the wait it
> would have bounded was over before the spin began.

> **P0-3: the kernel had been saying it all along.** `journalctl -k` carries
> `ring gfx_0.0.0 timeout, signaled seq=N, emitted seq=N+3` → `Starting
> gfx_0.0.0 ring reset` → `Ring gfx_0.0.0 reset succeeded` once for every run
> that stalled, L8c-7's sessions included. **A ring reset force-signals the
> fences of the jobs it killed**, which is the whole mechanism: the submit
> returns, the fence returns, and the only trace is the timestamps the killed
> dispatches never wrote. Swapping the wait flag for a deadline turns the hang
> into a sentence — `shim: 2 of 1025 timestamp slots never became ready (first
> 1023, last 1024)`. The first missing slot moves run to run (1023, 980, 977)
> because a reset kills everything in flight, not one nominated dispatch, and
> dumping the recorded sequence shows the grids around it are ordinary and
> scale cleanly with the rows: `moe 10x952` at 2560 is `moe 10x1040` at 3072.
> **No dispatch is malformed and nothing overflows.**

> **P0-4: the cliff is a duration, and it is 2.0 s.** Rows in steps of 128 at
> 48 layers, 1024 dispatches a submit, the first command buffer timed by its
> own GPU timestamps: **2560 rows 1.886 s, runs; 2688 rows 1.964 s, runs; 2816
> rows 2.033 s, reset.** The control separates time from everything it is
> confounded with — same model, same rows, same dispatches, same grids, only
> the split changes: **1273 dispatches in one submit are 787 ms at 512 rows and
> run, and are reset at 2688 rows**, where two submits of 1024 and 249 had just
> run the same work; and 3072 rows in five submits of 256 (510-560 ms each)
> runs at 1143.1 tok/s. The kernel constant behind the 2 s was not identified —
> `lockup_timeout` is unset on the command line and debugfs and `dmesg` both
> want root here — so the number is the measured one, and the budget is half
> of it.

> **P0-5: so chunk by time, not by count.** `maxBatch = 1024` was a bound on
> the shim's query pool and the doc said so — "not a tuning knob" — and it
> happened to be a safe amount of work until 2816 rows. A pass's cost is
> **affine** in the rows, not proportional: fitting 789.5 ms at 512 rows,
> 2342.4 at 2560, 2434.2 at 2688 and 2687.5 at 3072 over 1273 dispatches gives
> **322 us a dispatch plus 582 ns a dispatch-row**, reproduces all four to
> within 1.5%, and puts a 1024-dispatch submit at 2816 rows at **2.008 s** —
> the row count that is reset. `batchFor(rows)` fills half the cliff with that
> model. **At one row it is the full 1024, so decode is untouched**; it starts
> biting at ~1700 rows and is 196 at 8192. The extra fence waits are ~40 us
> each and invisible at every shape the vertical measures: 2048 rows is 1037.2
> tok/s against 1036.3 before, 2560 is 1088.8 against 1090.3, **decode is 24.48
> against L8e's committed 24.66** on the same widths and prompt — 0.7%, inside
> the 0.5-0.8% this vertical reproduces to — same text, same two command buffers
> a token, and `-hc` alone still **4.22x**.
> `TestBatchForIsUnderTheWatchdog` checks the fit
> against the four measurements *and* the bound against the cliff, because the
> constants are a fit in milliseconds feeding a `time.Duration` and a factor of
> a thousand either way still compiles — which it duly did, once.

> **P0-7: `-ppl -ctx 4096` completes, and the selection does not cost
> accuracy.** The run that used to hang at 32 and 48 layers now finishes at 48:
> **PPL = 3.9392 +/- 0.02209** over wikitext-2's 297,193 tokens in **72 chunks
> of 4096**, scoring 2047 each, in 7m24s, on the checkpoint's own widths. Read
> it against **4.0289** — the same bank, the same corpus, at n_ctx 2048 — and
> it is the *context* that moved, not a width: **-2.23%** for twice the
> context, with the QSA selection live throughout (2048 rows is below the
> 2051 at which it is still the identity, 4096 is well past it). That is idea
> 7's first long-context gate, and the instrument prints its 2048-context
> comparison lines beside it, which are **not** a like-for-like delta across
> two chunkings and should be read as the context effect only.

> **P0-6: the gate, and prefill does not plateau.** `-graph` at 48 layers now
> returns at every row count tried, each with its own submit size:
>
> | rows | ms | tok/s | vs 391.42 | dispatches a submit |
> |---:|---:|---:|---:|---:|
> | 2048 | 1974.6 | 1037.2 | 2.65x | 660 |
> | 2560 | 2351.2 | 1088.8 | 2.78x | 551 |
> | 3072 | 2703.2 | 1136.4 | 2.90x | 473 |
> | 3584 | 3092.6 | 1158.9 | 2.96x | 415 |
> | 4096 | 3487.0 | **1174.6** | **3.00x** | 369 |
> | 8192 | 6750.6 | **1213.5** | **3.10x** | 196 |
>
> The 8192 row is at `-ctx 8192`, where **the QSA selection does real work for
> the first time in this vertical**: attention is **14.7%** of the pass against
> 7.2% at 2560, and the MoE falls from 54.4% to 41.8%. llama.cpp's prefill
> plateaus — 388.60 at 2048, 392.95 at 8192 — and **ours is still climbing at
> 8192**. Every figure above 2560 rows is a measurement that could not be taken
> before. [Write-up](research/p0-ring-watchdog.md)

---

## What P1 established — the step, to the dispatch, and a gather that was disk

> **P1-1: the step sums, and the ~12 ms is three things.** Every block's
> `graph` method has always built a `kinds []string` beside its dispatches, for
> its error messages, and nothing read it — so the per-dispatch timestamps L7d
> already collects were folded into six *block* rows, and "the MoE is 11.6 ms"
> could not say which of nine kernels that was. `recorder.add` now takes the
> labels, `GraphStats` carries them, and `-gen -attrib` prints the whole token:
> **30.52 ms over 36 dispatch labels and six host phases, with a 5 us
> residual.** Two runs, 32.67 and 32.76 tok/s, 0.3% apart. Against 4.281 GB at
> the 227 GB/s a dispatch actually reaches — 18.86 ms — the gap is
> **5.54 ms of weight-streaming dispatches below that rate, 3.46 ms of
> dispatches that stream no weight at all, and 3.01 ms of host**, which is
> 12.01 against the review's estimate of ~12.
>
> **P1-2: the n-gram gather was sixteen serialised major page faults, and that
> is 1.2 tok/s.** `PLEGather` reads sixteen 90-byte rows a token at random
> offsets into a 320-million-row, 28.80 GB mmap'd table that D2 keeps off the
> device; beside 84 GB of resident model they are not in the page cache.
> Counting `majflt` around 200 gathers of fresh rows: serial **853.2/819.4 us**
> a step at **16.27/16.29** major faults, sixteen touched concurrently first
> **122.2/108.8 us** at **16.34/16.30**. The fault count is identical to two
> decimal places — nothing extra is read, nothing extra is cached — and the
> only thing that changes is how many are outstanding at once. A bounded worker
> pool over the rows takes the gather **2.244 → 0.902 ms** in the whole model
> and decode **31.51 → 32.72 tok/s, 1.30x llama.cpp**. This is the other half
> of **D9**: `MADV_RANDOM` made each fault small, this stops them queueing.
>
> **P1-3: seven of eight families read at 147-204 GB/s and the
> hyper-connection block reads at 114.6.** It is not short of bandwidth — D14
> established that these weights fit the MALL at every width — it is short of
> work per dispatch: **97 mixers x 5 dispatches = 485 dispatches a pass at
> 1.4-17 us each**. Counting its weightless norms and combines the block is
> **4.15 ms, 13.6% of the step, for 8.4% of the bytes**, the largest single
> misallocation in the token, and the lever is fusion, not a narrower bank.
>
> **P1-4: `moe.down` is 1.19 ms behind `moe.up` off the same bank.** 147.5
> against 200.6 GB/s, same experts, same layer, back to back; the difference is
> [NEmbd, FFNExpert] against [FFNExpert, NEmbd], a shape whose row block was
> chosen for one and never re-screened for the other. The shared expert's pair
> repeats it at **136.5 GB/s**, 0.73 ms. This is D12's operative half applied
> to a kernel instead of a bank, and both are a ladder run from an answer.
>
> **P1-5: the pre-recorded command buffer is priced, and it is fourth.**
> Idea 2 was the review's leading candidate for the whole 12 ms. It is
> **1.127 ms of host recording plus 0.828 of hand-over — 1.96 ms, 6.4%, about
> +2.2 tok/s** — which is worth building and is behind the two kernel-shape
> items above. That re-ranking is what P1 was for.
>
> **P1-6: the 0.38 GB is the router and the shared expert** (idea 9). The
> checkpoint inventory files both under "dense, every token"; L8b's table files
> them with the MoE. Less L8b-1's doubled `inject` rows, `PLEGPU`'s halves and
> rounding, the two partitions reconcile exactly. **Every ceiling is now quoted
> on one basis: 4.281 GB a token**, 56.5 tok/s at 242 GB/s and **53.0 at the
> 227 dispatches reach** — against which the measured step is **62%**.
> [Write-up](research/p1-decode-attribution.md) · `results/p1_decode_attrib.csv`

## What P1a established — a boundary that was two dispatches, and a grid pinned at 160

> **P1a-1: the scatter and the next mixer's norm are one workgroup's work,
> said twice.** Both kernels were already one workgroup per (token, stream)
> over the same 2560 values — the combine writes `res`, the norm reads it
> straight back — so `llm_hc_cn.comp` combines into registers, reduces there
> and writes `xn` without a second visit to the arena. Every layer is mix,
> block, combine, *next* mixer's norm, so the adjacency is structural: **94 of
> the 96 combines fuse**, and the two that do not are the PLE layer, where the
> residual is moved out of this arena and back, and the end of the pass, which
> reaches the final mixer through a row move. **1501 dispatches a pass to
> 1407, 30.69 ms to 30.37, decode 32.59 → 32.93 tok/s**, both arms on one
> build with `LLM_HC_NOFUSE=1` as the control. A boundary was 10.5 us and is
> **7.3**.
>
> **And it is 1.7% of prefill for a different reason, at one ubatch of two.**
> At 2048 rows the wide residual is 84 MB, so what the fusion removes is one
> 84 MB read a boundary — 7.9 GB a pass: the block goes **330.1 ms to 294.6**
> and the pass **1912.6 to 1880.3, 1070.8 → 1089.2 tok/s**. At ubatch 512 it
> is worth nothing (66.0 against 66.8 ms), which is **D14 from the activation
> side**: 512 rows are 21 MB, inside the MALL, so the read it removes was
> never going to DRAM.
>
> **P1a-2: it is identical to the last place, and one ulp had to be chased to
> make it so.** `res`, `xn` and `mixed` are bit for bit the pair's, at the
> kernel and through four layers of the graph. The first version was not: it
> differed by **one ulp on 23% of the residual**, with *identical* SPIR-V —
> `OpFMul` then `OpFAdd` in both. Giving the sum a **second reader** (the
> square) lets ACO contract the pair into a fused multiply-add, which rounds
> once where the scatter rounds twice, and `precise` is the only thing that
> says no to it. Nothing else in this vertical had hit that.
>
> **P1a-3: the block's 114.6 GB/s is a grid, not a fusion, and the down
> projection's own ladder is the ruler.** Its decode rungs differ only in how
> many ways they split K, so they read the same 1.94 MB off the same bank at a
> grid that varies twenty-fold: **168 workgroups is 9.26 us, 336 is 6.47, 672
> is 5.54**, and past that it is flat. `up_m1` reads **11.5 KB a workgroup in
> 160 workgroups and takes 10.87 us** — which is `down_gemv8`'s 11.8 KB in 168
> workgroups taking 9.26, to within the bytes. Two modes, two kernels, one
> number, because they are launched at the same width. So **`up`'s cost is not
> MODE 1's M=1 waste**: its four rungs vary `BM` from 16 to 128 and all four
> launch 160 workgroups, because a grid is `wide / BN` and `BN` is 64 in every
> one of them. This is L7d's finding a second time — at one token the
> parallelism has to come from somewhere other than rows — and the first time
> it has been *measured as a law* rather than inferred from a rung that won.
>
> **P1a-4: what pins `BN` at 64 is the collapse, and unpinning it is worth
> 0.22 ms.** MODE 1's epilogue needs the four streams of one feature in one
> workgroup, and `packUpB` buys that by making a 64-column block 16 features x
> 4 streams. A 4x4 block would satisfy it just as well and would make `BN` 16,
> which is 640 workgroups: `up` would go 16.5 → **~14.2 us** in the model,
> **2.3 us a mixer, 0.22 ms a token, +0.24 tok/s**. **Not taken** — the
> permutation is baked into the staged bank, so it is every `up` rung's
> epilogue on the prefill path, and 0.7% of decode does not buy that. Priced
> so it need not be re-derived.
> [Write-up](research/p1a-hyper-connection-shape.md) · `results/p1a_hc_grid.csv`

---

## The three findings that set the direction

**1. `llama.cpp` already implements this architecture, is built with Vulkan on
this box, and supports `qwen4exp`.** `/home/kube/repos/llama.cpp` has
`src/models/qwen4exp.cpp` (1279 lines: DeltaNet, the QSA indexer, PLE n-gram
hashing, hyper-connections) and 50 built tools. Confirmed at L1 by running
them: it is a reference implementation, an oracle (`llama-eval-callback`), an
accuracy criterion (`llama-perplexity`) and a performance baseline
(`llama-bench`) at once. **This vertical does not need a Python reference
dump**, which is what S1 and T1 spent their first session on.

> **And there is now a second implementation to argue with.** `../vllm` has
> `vllm/models/qwen4_exp/` (NVIDIA and AMD paths), `qwen_gdn_linear_attn.py`
> and a vendored flash-linear-attention. It **cannot** be a second oracle — it
> runs bf16 Triton on CUDA against our Q4_K_XL on Vulkan, so there is no
> tensor-level comparison without running a 180 B model here — but it is an
> **arbiter of formulas**, which is what D5's oracle turned out to need. In
> one pass it confirmed the GDN normalisation bug (L3a-4), explained the head
> map and found a hazard for L8 (L3a-3), disagreed about chunking (L3a-2), and
> settled what the QSA selection *is* without the 4k dump L4 was waiting for.
> The rule it earns: **llama.cpp for numbers, vLLM for formulas.**

**2. A quarter of UD-Q4_K_XL is a lookup table that should never touch the
GPU.** Of its 111.32 GB, **28.80 GB** is `per_layer_token_embd` — a 320 M-row
hash table read 16 rows of 160 values at a time, **1.41 KB per token**. Leave
it mmap'd on the host and the resident core is **82.52 GB**.

> **L1 confirmed llama.cpp does exactly this.** `qwen4exp.cpp` creates the
> table with `TENSOR_READ_LAZY` — *"read rows on demand instead of loading
> whole tensor; requires mmap"* — and `free` shows **77 GiB resident** during
> a run against the inventory's 76.86 GiB core. So D2 is not a deviation from
> the reference; it is what the reference does, and the baseline above is a
> like-for-like comparison.

**3. The stock quant is tuned for accuracy-per-byte, not for tok/s on a
bandwidth-bound APU, and it costs 1.7x.** Unsloth put every dense tensor at
Q8_0 and the routers at F32. On a machine where decode is pinned to the DRAM
bus that is the single most expensive choice available, because **76% of the
bytes read per token are dense** — each expert is read 10 times in 512, every
dense weight is read every time.

| | resident core | GB/token | **tok/s ceiling** | PPL |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL as shipped | 82.52 GB | 6.334 | **38.2** | **4.0289** (L8c-0) |
| ~4.25 bits on everything streamed, **symmetric** — D3 as written | ~67 GB | ~3.9 | **62.2** | **+18.5%** (L8c-1) |
| L8c-1's mixed plan: DeltaNet + head 4.5, attention + hc 6.5 | ~73 GB | 4.575 | **52.9** | **4.2699, +5.98%** |
| **L8c-3: 4.5 bits on the dense half, `q4_k` + the imatrix** | 80.5 GB | **4.264** | **56.7** | **4.1998, +4.24%** |
| **L8c-4: the first family of that, built — `lm_head` alone** | 82.20 GB | **6.016** | **40.2** | **4.0621, +0.82%** |
| **L8c-5: the second and largest — `+ deltanet`** | 81.12 GB | **4.943** | **49.0** | **4.1012, +1.79%** |
| L8c-3's mixed plan: `q5_k` on attention + hc, `q4_k` on the rest | 80.6 GB | 4.419 | **54.8** | **4.1377, +2.70%** |
| the same 4.5 bits in the **symmetric** form — what D3 was retired on | 80.5 GB | 4.264 | 56.7 | 4.6127 rtn / 4.6588 imatrix, **+14.5 / +15.6%** |

> **L8c-5's row is L8c-4's plus the DeltaNet's group**: `cmd/gguf -width`
> puts `deltanet` at 2.247 GB a token as shipped and 1.174 at 4.5 bits, so a
> token is 6.016 to **4.943** and the ceiling 40.2 to **49.0**. Measured
> against *our staged bank* it is 5.733 GB to 4.69 and a 51.6 tok/s ceiling,
> with the measured rate **28.9 tok/s** against 25.75; the resident core is
> **80.53 GB** on that basis. The delta is 1.04-1.07 GB either way.
>
> **L8c-4's row is on this table's basis and not on L8b's.** A token reads
> 6.334 GB of the checkpoint's own widths and the head is 0.676 of them, so
> 4.5 bits there is 6.016 and 40.2 tok/s, and the resident core 82.52 GB to
> 82.20 — the head is 0.318 GB of it and the experts are untouched. Measured against *our staged bank*
> — which L8b quotes at 6.05 GB and a 40.0 tok/s ceiling, since it is not the
> same set of bytes — the same change is 5.733 GB and 42.2, and the
> **measured** rate is 25.75 tok/s against 24.66. The delta is 0.318 GB
> either way; only the denominator moves.
>
> **The residency column is not on one basis and the GB/token column is.**
> Rows two and three are pre-simulation estimates that price the *experts*
> down as well as the dense half; rows four to six are `cmd/gguf -width`'s own
> arithmetic over the dense half alone, which is all L8c-3 measured — the
> experts stay as shipped at 77.02 GB of the 80.5, so re-quantising them is
> where the rest of the residency is (and +3.0 tok/s of ceiling with it).
> **The tok/s and GB/token columns are comparable throughout**, since a token
> reads 10 of 512 experts either way.

**L8c-1 retired row two and L8c-3 brought it back at a different width and in
a different form** — rows four and five are the same "everything streamed"
idea, at 4.5 bits rather than 4.25, asymmetric rather than symmetric, and
calibrated. They beat row three on **both** axes. **Row six is the control that makes the point**: the same 4.5 bits, the same
4.264 GB a token and the same residency in the symmetric form is +14.5%, and
calibrating it makes it +15.6% — so what retired D3 was the form, not the
width.

Row two was an inference from L0d's weight-reconstruction ladder and a
bandwidth argument, with no perplexity beside it, and **L8c-1 retired it**:
measured in the symmetric form, a uniform width is the one arrangement
guaranteed to be wrong at both ends, because the cost of 4 bits runs *inverse*
to the bytes. That made row three the recommendation, and at +5.98% it was not
yet a bank worth building a kernel for — see L8c-2. **L8c-3 changed which
sentence in that was load-bearing**: "uniform is wrong at both ends" was a
statement about a format whose group has one free parameter. In the
asymmetric form the sensitive family's cost falls from +5.98% to +0.90% and a
uniform 4.5 bits is the best row on the table, so the bottom row is both
cheaper and more accurate than the plan it replaces — and **is** worth its
kernel.

---

## Direction

**Phase 1 — make it run on UD-Q4_K_XL.** Consume the GGUF natively. **Done at
L1**: the reader, the five dequant paths and the tokenizer all exist and are
checked against llama.cpp. **L2 has the dense skeleton on the device** — the
hyper-connection block, the PLE n-gram block and the full-attention layer with
its indexer — **L3 has the linear-attention layer**, **L4 has the QSA
selection**, and **L5 has the MoE block**, all checked tensor-for-tensor
against `llama-eval-callback`. **Every block of the model now has a kernel.**
The acceptance criterion is *it generates the same text as llama.cpp*. **L2a
sizes it**: ~1150 tok/s prefill against 391.4, and 38.2 tok/s decode against
25.15, both reachable with kernels that already exist plus the epilogue fusion
the hyper-connection block needs. **L2c, L2f, L3b, L4b and L5b between them
have taken 1015.5 ms of llama.cpp's 1164.7 ms prefill graph down to 714.8 —
87.2% of it replaced by 61.4%, a 25.8% saving on the whole graph.** **L6a has
the arena plan**: 84.20 GB in 68 buffers, all 48 layers of all five blocks
resident at once, with residency measured to cost nothing that scales with the
bank. **L6b has the graph, and it clears the gate**: llama.cpp's own argmax
out of llama.cpp's own top ten. **L6c put the glue on the device** and left
prefill at **941.3 tok/s against 391.42 — 2.40x** — with **96.6% of the pass
inside the five blocks**, which is 82% of the way to the ~1150 above. **L7 is
decode, and L7d closes phase 1's kernel work**: 11.89 tok/s against
llama.cpp's 25.15, prefill at **990.6 tok/s — 2.53x** — and the model's own
text unchanged. What was left at decode was not a kernel shape; it was the
bank. **L8a is the first three quarters of that bank and L8b the
rest**: 14.71 tok/s and 1041.1 tok/s prefill, with the same text and the same
halves.

**Phase 2 — our own bank, and L8a and L8b have taken the first 1.24x of it
without changing a number.** Our dense half *was* staged as halves, so a token read
8.07 GB of it against the 3.67 the shipped Q8_0 would be, and the ceiling for
the bank we ran was **25.0 tok/s** — llama.cpp's measured rate exactly.
**L8a stages the DeltaNet, the attention layer and the head as the
checkpoint's own int8 with an fp16 scale per 32 elements**, which is
bit-identical arithmetic (L8a-1), and the token falls from 9.67 GB to 6.61:
**decode 14.07 tok/s, the ceiling 36.6, prefill 1044.8**. **L8b finishes it
with the hyper-connection block** — the split that had to fall on a column
block, the GEMV that had to round in fp16, and the collapse scratch that had
to shrink before the unpack could be amortised: **decode 14.71 tok/s, a token
6.05 GB, the ceiling 40.0, prefill unchanged**. **Then the bytes stopped being
the thing to fix, so L8d took the kernels before the widths** — five grids and
an unpack that waited on its own loads, no bank change and so no accuracy to
argue about: **decode 23.15 tok/s, the MoE 58 GB/s of bank to 133, the
DeltaNet 119 to 205, a token 66.7 ms to 43.0, and prefill 1049.8 tok/s at
ubatch 2048 and 653.9 at 512 — faster on both.** **And L8e is the last of that**, on the two blocks L8d did not reach: the
full-attention layer's projections on `llm_gemv.comp` (the layer 414.8 us to
225.9, the block 5.5 ms a token to 3.5), the MoE's shared expert on a lane
group of its own, and its second counting-sort pass deleted wherever the two
expert modes share a row block — **decode 24.66 tok/s, a token 40.5 ms,
prefill 1052.8 and 655.2, and 48 fewer dispatches a pass at every length.**
That is **62% of this bank's own 40.0 ceiling**, 65% of the checkpoint's 38.2
and **0.98x llama.cpp's measured 25.15**, and nothing in the model computes
anything different yet. **L8c-0 is the instrument for the stage where that
stops being true**: `-ppl` is llama.cpp's own perplexity protocol over our
graph, and it puts the bank we run at **4.0289 against the reference's 4.0340
— −0.13%, inside both error bars** — which is the left-hand side every width
below is a delta from.
*Then* the re-quantisation — and **L8c-1 measured what it is worth before
building it, which changed the plan.** The stated target was ~4.25 bits on
everything streamed for ~67 tok/s; measured, that is **+18.5% of perplexity**,
and the cost of 4 bits runs *inverse* to the bytes — the gated DeltaNet is 46%
of a dense token for +1.49%, the hyper-connection block 14% for **+5.98%**.
The widths that keep the model together are a **plan** (DeltaNet and head at
4.5 bits, attention and hyper-connections at 6.5), and that plan is
**4.2699 against 4.0289 — +5.98% — for 4.575 GB a token and a 52.9 tok/s
ceiling**, not 67. **So the kernel is not the next thing to build.** Every
rung was round-to-nearest, against a checkpoint whose experts are
imatrix-quantised and whose imatrix unsloth publish free
(`imatrix_unsloth.gguf`, 580 MB); calibration is days where the W4A8 layout is
weeks, and the widths it allows are what that layout would be built for.
**L8c-2 pulled that lever and it moved the wrong way** — calibration cost
another 3.8 points, all of it in the hyper-connection block, through a
systematic gain error that compounds where residual noise averages out — which
left one untested caveat: every rung in both stages was Q4_0's *symmetric*
form, where the checkpoint's own experts and the matrix's own target are Q4_K.

**L8c-3 is that caveat, and it reverses the recommendation.** The min costs
nothing, because ggml nests it — a super-block of eight groups of 32 with one
fp16 pair and 12 bits a group is **0.500 bits a weight, the same as a
symmetric fp16 scale per 32** — so the two forms are compared at 4.500 bits
exactly. Uncalibrated the asymmetric form is **2.1x** cheaper end to end and
**3.2x** on the family that decides the plan; calibrated it is **4.1998
against 4.0289, +4.24%**, where the symmetric form with the same matrix cost
+22.4%. **The mechanism is the second parameter**: a symmetric group's one
free parameter *is* the gain, so a calibrated fit can only express a
preference by shrinking the group, and the extra shrinkage calibration costs
`hc_attn_up` is 1.24 pp symmetric against 0.47 asymmetric. **So the dense
kernel is the next thing to build after all**, at 4.5 bits rather than 5.30,
for 4.264 GB a token and a **56.7 tok/s** ceiling — better than L8c-1's plan
on both axes at once. §1.1's W4A8 is a symmetric layout; the asymmetric one
folds the min into the epilogue as a per-group correction times the
activation's column sum, which is one extra reduction over A and no change to
the matrix core. Beside it sit the two things the dense widths cannot reach
and which are worth more together than the last half-bit — the F32 router at
fp16 (+1.5 tok/s of ceiling) and the 512 expert banks at ~4.25 (+3.0), which
are already Q4_K and already the calibrated part of this checkpoint — and,
free either way, **D13's fp16 tail as int8 for +0.01%**.

**L8c-4 builds the bank and puts the first family on it.** The format is
§2.8's fragment tiling with a nibble where L8a puts a byte plus ggml's own
sixteen-byte super-block record — 4.500 bits a weight exactly — and one
encoder serves it and the simulation, so the lm head's 248 320 logits off the
real bank are **identical** to the same format's through `sim.go`. **The head
is 0.675 GB to 0.358, decode 24.66 tok/s to 25.75 — 1.024x llama.cpp, ahead
of it for the first time — and perplexity 4.0621 against our own 4.0289,
+0.82%.**

**L8c-5 is the second family and the largest**, and it is the same kernel and
the same plumbing: the gated DeltaNet's 36 layers, 46% of a dense token, with
the fp16 tail exactly where L8a left it. **Decode 25.75 tok/s to 28.9 — 1.15x
llama.cpp — the block 10.9 ms a token to 6.8, a layer 62.28 MB to 33.28,
residency 81.57 GB to 80.53, and prefill *faster* at both ubatches (1070.1 at
2048, 667.0 at 512).** Its accuracy is where the stage earns its keep:
**+0.93% over 145 chunks where the eight-chunk screen said −0.30%**, so
L8c-4's "the screen under-reads by 2.05x" is really "the screen does not fix
the sign" — while the two families together measure **+1.79% against a sum of
1.75**, which is the mildest compounding this vertical has seen. Three
families were left at that point, in the order their bytes justify
(`hyper_conn` 0.695 GB a token, `full_attn` 0.635, `ple_proj` 0.035), with
one real obstacle in the first of them: `hc_*_up` is 320 wide and
`get_scale_min_k4`'s twelve-byte scheme *is* eight groups, so the
hyper-connection block needed a packing of its own.

**L8c-6 is the third, and the one the stage saved for last.** The
hyper-connection block's up projection is 320 wide, and ggml's
`get_scale_min_k4` is eight groups rather than a length — so the bank gets a
record packing of its own, twenty bytes and twelve bits a group, at the same
4.500 bits and with a simpler decode. The fp16 tail L8b stranded has to go
through the encoder now that the two planes no longer hold the same numbers.
**Decode 28.84 tok/s to 30.05-30.19 — 1.20x llama.cpp — the block 5.68 ms a token
to 4.16, a mixer 8.12 MB to 4.76, residency 80.20 GB, prefill faster at three
of four ubatches and 0.996x at 2048.** Its accuracy inverts the stage's own
warning one more time: **+0.31% over 145 chunks where the eight-chunk screen
said +0.90%**, the first family whose screen was pessimistic, which settles
that a screen does not bound, does not fix a sign and does not rank. Two
families are left and neither has an obstacle in it — `full_attn` (0.635 GB a
token) and `ple_proj` (0.035), with `qsa_indexer` riding along — so what
remains of the dense half is wiring rather than format.

**Phase 3 — the throughput levers that are worth more than the format.** The
MTP head is downloaded (2.79 GB, `mtp-…-Q4_K_M.gguf`); a draft step is ~12% of
a full one, so self-speculation is worth ~1.5-1.8x. Batching amortises the
dense 76% completely. Both dwarf the difference between 4 and 5 bits.

---

## The machine, measured

| limit | value | where from |
|---|---|---|
| DRAM bandwidth | **236 GB/s** peak; **242 GB/s** best real decode GEMV | §0, §1.7 |
| MALL | 32 MiB, 805 GB/s copy / 965 read | §0.4, §5.1b |
| WMMA fp16 | 55.5 TFLOP/s; best real kernel 39.0 (70%) | §0.1, §2.7 |
| memory type, for a GPU read | **irrelevant**, all 8 within 0.57% | §5.1 |
| the real capacity ceiling | **physical RAM**, 117.7 GiB, less the OS | §5.1 |
| a resident 82.52 GB model | **77 GiB, steady**, with 39 GiB left for page cache | L1 |
| `maxBufferSize` / `maxMemoryAllocationSize` | **4 GiB − 4 B** | `vulkaninfo` |
| `maxDescriptorSetStorageBuffers` | 8 388 606 | `vulkaninfo` |
| GTT / system RAM | 117.7 GiB (`amdgpu.gttsize=126976` already set) | `/proc/cmdline` |
| disk free | 1.1 TB after the download | `df` |

The 4 GiB buffer cap is the one structural consequence: a 60-80 GB bank is
**~100 buffers**, one per layer per projection (a layer's Q4 expert pair is
1.26 GB, comfortably under). Descriptor counts are not a constraint.

---

## The model, from its own tensor table

`go run ./cmd/gguf -tensors` over the real checkpoint — 1224 tensors, not a
config.json transcription. `general.size_label` is `512x56B`;
`general.description` is "A Preview of the Qwen4 Architecture".

```
arch            qwen4exp   (Qwen4ExpForConditionalGeneration, multimodal)
layers          48 = 36 linear_attention + 12 full_attention (interval 4)
hidden          2560       residual stream 10240 (hyper-connections, hc_count 4, low_rank 320)
vocab           248320     context 262144 (1 M with YaRN)   tie_word_embeddings false
MoE             512 experts, 10 active + 1 shared, ffn 640, router in F32
```

A GGUF states a weight as `[in, out]`, so every shape below is `[K, N]`:

| tensor | shape | type | per pass |
|---|---|---|---|
| `attn_q` | [2560, **12288**] | Q8_0 | 12 — query *and* gate, 24 heads x 256 x 2 |
| `attn_k`, `attn_v` | [2560, 512] each | Q8_0 | 12 each — 2 KV heads x 256 |
| `attn_output` | [6144, 2560] | Q8_0 | 12 |
| `indexer.q_proj` | [2560, 512] | **BF16** | 12 — QSA, 4 heads x 128, top_k 2048 |
| `indexer.k_proj` | [2560, 128] | BF16 | 12 |
| `attn_qkv` | [2560, **10240**] | Q8_0 | 36 — DeltaNet's *fused* input projection |
| `attn_gate` | [2560, 6144] | Q8_0 | 36 |
| `ssm_out` | [6144, 2560] | Q8_0 | 36 |
| `ssm_alpha`, `ssm_beta` | [2560, 48] each | F32 | 36 each |
| `ssm_conv1d` | [4, 10240] | F32 | 36 — depthwise, k=4 |
| `ssm_a`, `ssm_dt.bias` | [48] | F32 | 36 |
| `hc_{attn,ffn}_{down,up}` | [10240, 320] / [320, 10240] | Q8_0 | **97 each** incl. `output_hc_*` |
| `hc_{attn,ffn}_inject` | [10240, 4] | F32 | 96 |
| `ple_key` | [2560, 10240] | Q8_0 | 1 — layer 1 only |
| `ple_value` | [2560, 2560] | Q8_0 | 1 |
| `per_layer_token_embd` | [160, **320001536**] | IQ4_NL | gather, 1.41 KB/token |
| `ffn_gate_exps`, `ffn_up_exps` | [2560, 640, 512] | Q4_K (Q5_K on layer 2) | 48 |
| `ffn_down_exps` | [640, 2560, 512] | **Q5_1** (Q8_0 on 5 layers) | 48 |
| `ffn_gate_inp` | [2560, 512] | F32 | 48 — the router |
| `ffn_*_shexp` | [2560, 640] / [640, 2560] | Q8_0 | 48 — the always-on shared expert |
| `token_embd`, `output` | [2560, 248320] | Q8_0 | gather / 1 |

Other metadata worth having to hand: `ssm` state 128, groups 16, dt_rank 48,
inner 6144; rope mrope interleaved, sections [11,11,10], 64 of 256 dims, theta
1e7; PLE ngram_size 3, heads_per_ngram 8 → 16 hash heads with prime vocabs
~20 000 0xx; vision tower 27 layers, hidden 1152, in a separate
`mmproj-*.gguf` (0.9 GB, not downloaded).

> **`bench/modelshapes.go` has been corrected to this table** (L1). It had
> `attn.q` at 6144, one fused 512-wide KV, a 2048/6144 DeltaNet pair that does
> not exist, and no rows at all for the hyper-connections, the indexer or the
> PLE block — **220 matmuls a token** and 13.4% of the weights. It now reads
> **6.665 B weights a token against the checkpoint's 6.671 B**, 0.09% apart,
> and `shapes` re-run over it prices the 2081 dispatches at **0.62 ms of
> launch cost, 5% of a 4-bit decode step**.

### What UD-Q4_K_XL actually contains

1224 tensors, 176.944 B params, **111.32 GB, 5.03 bits/weight** — produced by
`cmd/gguf` in 47 ms and cross-checked tensor-for-tensor against
`reference/gguf_inventory.py` (**0 disagreements**):

| group | params | GB | bits/w | read | quant mix (GB) |
|---|---:|---:|---:|---|---|
| moe_experts | 120.796 B | 77.02 | 5.10 | 10/512 | Q4_K 44.4, **Q5_1 27.1**, Q8_0 4.5, Q5_K 1.2 |
| **ngram_ple_table** | 51.200 B | **28.80** | 4.50 | **gather** | IQ4_NL 28.8 |
| deltanet | 2.087 B | 2.25 | 8.62 | every tok | Q8_0 |
| hyper_conn | 0.641 B | 0.70 | 8.68 | every tok | Q8_0 |
| lm_head | 0.636 B | 0.68 | 8.50 | every tok | Q8_0 |
| embed (lookup) | 0.636 B | 0.68 | 8.50 | gather | Q8_0 |
| full_attn | 0.598 B | 0.64 | 8.50 | every tok | Q8_0 |
| moe_router | 0.063 B | 0.25 | **32.00** | every tok | F32 |
| moe_shared | 0.236 B | 0.25 | 8.51 | every tok | Q8_0 |
| qsa_indexer | 0.020 B | 0.04 | 16.00 | every tok | BF16 |
| ple_proj | 0.033 B | 0.04 | 8.55 | every tok | Q8_0 |

Two things worth reading twice. **`ffn_down_exps` is Q5_1 on 43 of 48 layers
and Q8_0 on the other 5** while gate/up are Q4_K — that is unsloth's imatrix
telling them the down projection is the sensitive one, and it is 27 GB of the
77. And **the routers are F32**: 0.25 GB read every single token, 4% of the
decode budget for 0.06 B of parameters.

### Where the bytes go at decode

    dense, every token      4.830 GB   76%
    experts, 10 of 512      1.504 GB   24%
                            -------
    per token               6.334 GB   -> 38.2 tok/s at 242 GB/s
                                       -> llama.cpp reaches 25.15 (159 GB/s)

Capacity is an expert problem (97% of params). **Speed is a dense problem.**
The usual "keep attention at Q8, it's only 2% of the model" instinct is exactly
backwards on this machine: it costs 2% of memory and **a factor of 1.7 in
tok/s**. Conversely, going below 4 bits on the *experts* buys very little —
they are only a quarter of the traffic.

**That table is the checkpoint's, and L7c-5 measured ours.** The dense blocks
were dequantised on the way to the device and staged as **halves**, which L6a
priced as capacity (6.79 GB of fp16 against 3.67 of Q8_0) and which at decode
is bandwidth. **L8a stopped three of the four families doing it and L8b the
fourth:**

                                     L7d      L8a      L8b
    hyper-connection mixers       1.31 GB   1.31 GB   0.74 GB
    gated DeltaNet, 36 layers     4.18 GB   2.25 GB   2.25 GB
    full attention, 12 layers     1.24 GB   0.70 GB   0.70 GB
    lm head                       1.27 GB   0.68 GB   0.68 GB
    PLE projections               0.07 GB   0.07 GB   0.07 GB   halves, runs once
                                 --------  --------  --------
    dense, every token            8.07 GB   5.01 GB   4.45 GB   83% -> 74%
    experts, 11 of 512           ~1.60 GB  ~1.60 GB  ~1.60 GB
                                 --------  --------  --------
    per token                     9.67 GB   6.61 GB   6.05 GB
    ceiling at 242 GB/s              25.0      36.6      40.0 tok/s
    measured                         11.89     14.07     14.71  (89 GB/s)

So **llama.cpp's 25.15 was the ceiling of the bank it reads**, we were under
our own, and 8.5 bits a weight is the checkpoint's own width — which is why
L8a's 36.6 landed within 4% of the 38.2 in the row above. **L8b's 40.0 goes
past it**, and ~~the 0.38 GB between our 4.45 GB dense half and the budget's
4.830 is not accounted for here~~ — **P1-6 accounts for it.** The two columns
are different partitions of the same tensors: the inventory files the **MoE
router (0.252 GB) and the shared expert (0.251)** under "dense, every token"
where the table above files them with the MoE, and the table's column is
larger than the inventory's on three families — L8b-1's doubled `inject` rows
(0.045), `PLEGPU`'s halves (0.035, the one family D13 never reached and P2's)
and rounding on the attention pair (0.026). 0.503 − 0.106 = 0.397, which is
the 0.38. **Every ceiling in this file is now quoted on the third basis, the
measured one: 4.281 GB a token at the shipped bank.** **The 1.66x that needed no
re-quantisation came to 1.81x of the dense half and 1.24x of the rate**, and
the only halves left anywhere are the PLE block's one projection and the
three families that are not Q8_0 on disk.

### Context is cheap here, which is the good news

12 full-attention layers x 2 KV heads x 256, and QSA caps what a step *reads*
at `top_k = 2048` however long the context is:

| context | KV fp16 | + indexer | DeltaNet state |
|---:|---:|---:|---:|
| 32 768 | 0.81 GB | 0.05 | 113 MB (constant) |
| 131 072 | 3.22 GB | 0.20 | 113 MB |
| 262 144 | 6.44 GB | 0.40 | 113 MB |

82.52 GB of weights plus 6.8 GB of KV is comfortable in 117.7 GiB, and L1
measured 77 GiB resident with 39 GiB of page cache still live beside it.
Capacity has stopped being the binding constraint on this model at any format
below Q8. Bandwidth is the whole story.

---

## What exists to build on

| piece | state |
|---|---|
| `gguf/` | **L1: the GGUF reader.** mmap'd, sharded, the split convention, the value grammar, openable from any shard. 1224/1224 tensors against the Python inventory. |
| `gguf/dequant.go` | **L1: Q4_K, Q5_K, Q5_1, Q8_0, IQ4_NL, F32, F16, BF16** — bit-exact against ggml's own `to_float` via `reference/dequant_ref.c`. |
| `cmd/gguf` | inventory, quant mix, decode budget, `-check` against `tensors.json`. |
| `cmd/llm` | the vertical's driver: `-tokenize`, the five block benchmarks, `-resident`, `-graph`, and **L7c's `-gen`** — prefill, then one token at a time, with both rates and where each went. |
| `llm/sample.go` | **L7c: the sampler.** `top_k -> top_p -> temp -> dist` with temperature zero short-circuiting into `Argmax`, which is the default because the gate is only a check if the sampler is a function of the logits alone. `TopK` is a quickselect: a 248320-wide sort is 30 ms, most of a decode step. |
| `gguf.Tensor.AdviseRandom` | **L7c: `MADV_RANDOM` on one tensor's pages.** For the 28.80 GB n-gram table, whose sixteen scattered reads a token drew sixteen 128 KB readahead windows. 176x on the gather; the rest of the shard keeps the readahead staging wants. |
| `shaders/llm_seq_hist.comp` | **L7b: the ring both convolutions leave behind them.** Fifteen lines for the PLE block's nine normed rows and the DeltaNet's three projection rows, addressed by **position modulo the ring's length** — which is what makes the store a pure write and a one-token run cost one row and no read. |
| `zimage/tokenizer/` | **L1: `FromVocab`** builds the BPE from GGUF metadata, and `split()` now carries both Qwen pre-tokenizers. 213/213 against `llama-tokenize`. |
| `llm/` | **L2b: the vertical's package.** `Config` from the checkpoint's own metadata, on-demand dequantisation, the `token_embd` gather (bit-exact), and the hyper-connection block — `HCInit`, `HCMix`, `HCCombine` — with a `Numerics` switch between the exact model and the reference's int8 arithmetic. |
| `reference/eval_dump.c` | **L2b: the oracle.** Whole tensors of a real llama.cpp pass, where `llama-eval-callback` prints three per axis and a sum. `llm/evaldump.go` reads them back as a `Trace`. **L4a** added `-f` (a prompt from a file) and `-nt` (an exact token count), and pins the build's headers. |
| `reference/out/llm4k/` | **L4a: the second trace**, 4096 tokens in a 4096-cell cache in one ubatch — 8.0 GB, 116 tensors. The only place the QSA selection exists, the only place the reference's real prefill arithmetic is visible, and it carries the whole MoE block for L5. The 7-token trace stays beside it: the empty blocks, the spare block and the f32 vector path are only legible there. |
| `llm/gpu.go` | **L2c: the block on the device.** Four arenas, the fused `[336, 10240]` down/inject weight, the up projection's row permutation, an M ladder per projection with a measured `PlanFor` schedule, and a sweep profiler that reads every staged mixer so the weights are as cold as a real graph's. |
| `shaders/llm_gemm.comp` | **L2c/L2d/L2f: the vertical's GEMM**, one kernel and one epilogue per mode — MODE 0 down+inject+silu, MODE 1 up+sigmoid+collapse, MODE 2 plain — because in this architecture the epilogue is where the time goes. |
| `shaders/llm_hc_*.comp`, `llm_ple_*.comp` | **L2c/L2d: the blocks.** `llm_hc_norm` (grouped RMSNorm → fp16 A operand), `llm_hc_combine`, `llm_ple_gate` (both norms, the signed-sqrt gate, the broadcast and the conv norm in one pass) and `llm_ple_conv` (four dilated taps, the SiLU and the residual add), over `llm_common.glsl`'s binding contract. |
| `llm/attn.go` | **L2e: the full-attention layer.** The fused query/gate projection, interleaved M-RoPE, the QSA indexer's pool/score/select with the reference's own cache block structure, and causal GQA — with the fp16 KV cache and the fp16 F32-matmul the reference turns out to use. **L4a** replaced the selection with a port of the reference's radix select, and made the indexer's two BF16 projections take a bf16 activation above the 8-column threshold. |
| `llm/gpu_attn.go` | **L2f: that layer on the device, in six dispatches.** The fused `[13952, 2560]` projection for six of llama.cpp's matrices, a host-built rotary table, two BM ladders because the two projections fall on opposite sides of the MALL, and a sweep profiler over every staged layer. **L7a gave it a cache**: four planes a layer — key, value, the indexer's raw key per cell and its pooled block — with `SetPast`, an incremental `blockRange`, and the query's plane geometry no longer the key's. |
| `shaders/llm_attn_select.comp` | **L4b: the QSA selection.** One workgroup a token, the row's keys cached in LDS, llama.cpp's four radix passes with its serial bucket walk replaced by a subgroup suffix sum, and a **per-cell bitmask** out instead of an index list — which deletes the reference's `GET_ROWS`. Dispatched only where `top_k + ratio - 1` is fewer cells than the cache holds; `SetSparse` forces it either way, which is how it is priced and how the dense control runs. |
| `shaders/llm_attn_*.comp` | **L2f: the layer's four kernels.** `llm_attn_pack` (per-head norm + interleaved M-RoPE + the fragment tiling for q, k and v in one grid), `llm_attn_idx` (the indexer's pooled key and query over two addressings), `llm_attn_score` (the rectified score, its bias, the cells and the causal mask) and `llm_attn_wmma` (causal GQA at headDim 256, **with the output gate in its epilogue**, and **L4b's bitmask staged a key block at a time** beside the causal mask). |
| `llm/moe.go` | **L5a: the MoE block.** `MoEConfig` from the checkpoint, the F32 router with fp16 operands and an f32 accumulator, the softmax/argsort/top-10/clamp/normalise chain, an `ExpertBank` that dequantises one expert's [640, 2560] matrix on demand out of a 77 GB tensor nothing can hold as floats, the routed half walked **by expert** rather than by token, and the shared expert with its one-column gate on the f32 vector path. |
| `cmd/llm -resident` | **L6a: the whole model on the device.** Stages every mixer, every layer and every expert bank, and prints the plan — bytes, buffers, wall clock — beside the checkpoint's inventory and the machine's memory. `-bank 48,24,12,4,2` restages the bank alone to price residency against itself; `-dense` leaves the 77 GB out. |
| `vk.PipelineSpec.Counts` | **L6a: a binding may be an array of buffers.** `shim_create_compute_pipeline` takes a per-binding `descriptorCount` and the flat buffer list is their concatenation; `nil` is the one-buffer-per-binding arrangement every other kernel in the repo uses, unchanged. |
| `llm/gpu_moe.go` | **L5b: the MoE block on the device, in nine dispatches.** The fused `[nExpert+1, nEmbd]` router with the shared expert's gate as its 513th column, a **quantised** bank staged byte for byte out of the checkpoint (1.57 GB a layer, against 5.03 as halves), a seventh binding that is the same bank read as `uvec4`, the permuted row space with the shared expert's group at its front, the measured `MoEPlanFor` schedule and a sweep profiler. **L6a** made the quantised bank one buffer a layer, bound as one array of `moeMaxBanks`, with the layer index in the top sixteen bits of `moeUsed`. |
| `shaders/llm_moe_gemm.comp` | **L5b: the grouped quantised GEMM.** Two modes — gate/up with `silu(gate)*up` on the accumulators, down in permutation order — crossed with the four formats the bank ships in (Q4_K, Q5_K, Q5_1, Q8_0) and five row-block rungs. Each workgroup unpacks its own BN x BK slab into LDS per K-step, and every store is a whole cooperative-matrix fragment because the permutation pads each expert's rows to the block. |
| `llm/gpu_attn.go`, `llm/gpu_moe.go` (L8e) | **L8e: the last two schedule fields.** `AttnGemvFor` puts the full-attention layer's two projections on `llm_gemv.comp` at one token (k8 and k32) with a partial-sum arena beside them, and `PinGemv` holds them on the GEMM — the fifth block `Graph.PinSchedule` covers. `MoESharedPlanFor` gives the shared expert its own pair of rungs, because a lane group is sized to a row's payload words and Q4_K packs half as many as Q8_0; `pad()` is then the widest of four dispatches, and `downTiles()` lets the down mode read the up mode's tile list so the second counting sort is only dispatched when the two row blocks differ. |
| `shaders/llm_moe_{route,perm,combine}.comp` | **L5b: the routing.** `llm_moe_route` is the softmax over 512, ten workgroup argmaxes reproducing `ggml_argsort`'s DESC comparator without materialising the other 502 ranks, the normalised weights and the shared gate's sigmoid — one workgroup a token, `llm_attn_select.comp`'s shape. `llm_moe_perm` is a counting sort, the padded offsets, the sentinel fill, the inverse permutation and the two tile schedules, in one workgroup. `llm_moe_combine` is the eleven contributions weighted and summed in the reference's order. |
| `llm/graph.go` | **L6b: the model.** The five blocks in `llama_model_qwen4exp::graph::graph`'s order over one staged model, and a `GraphStats` that splits a pass into block and glue. `Layers: N` truncates it to a prefix, which is how the order is checked in 7 GB. **L7b made it a sequence**: `Reset`/`Append`/`Extend` beside `Prefill`/`Hidden`/`Forward`, a `past` every block is told once a run, and the whole id list kept because the n-gram hash is a trigram. |
| `llm/move.go`, `shaders/llm_move.comp` | **L6c: the move between two blocks' arenas.** Three bindings of its own — 1 and 2 the same buffer twice, so an fp32 and an fp16 destination are one pipeline shape — one pipeline per (source, destination) pair, and a `Port` per crossing tensor. Bit-exact against the host narrowing, and **2.0% of the pass** against the 22.7% it replaced. |
| `vk.Buffer.Zero*` | **L6c: clearing a mapping in place.** `WriteFloat32At(off, make([]float32, n))` is two costs and on a hot path the allocation is the larger: 36 DeltaNet states is 113 MB of Go allocation a prefill, flat in the prompt length (L6c-4). |
| `llm/bank.go` | **L8a: the dense bank in the checkpoint's own width.** `tileBQ8` writes a matrix into the §2.8 fragment tiling as int8 with one fp16 scale per 32 elements of a row, k-major inside an n-tile (D8), and the round trip through ggml's `d = amax/127` is an **identity** for a Q8_0 tensor — the halves the kernel forms are the halves `tileB` wrote. `q8Bytes`, `q8Align` and `q8Pipe` are the layout and the pipeline key a block on this bank needs. |
| `shaders/llm_gemm.comp -DQ8B` | **L8a: that bank read.** The slab goes through LDS a K-step at a time, as llm_moe_gemm.comp's does and for §2.2's reason, with the scale constant across a sixteen-wide k-tile so a `uint` of four bytes takes one multiply. `pc.lowRank` and `pc.gateOff` carry the **fp16 tail** the three non-Q8_0 families keep (L8a-2), and the reduction is written twice rather than branched inside, because a compare in the k-loop cost 1.27x (L8a-3). `-DDENSE_Q8` is the sixth buffer: the same bank again, as raw words. |
| `llm/quantk.go` | **L8c-4: ggml's K-quant super-block, as a thing that can be stored.** `asymEnc` is the per-super-block arithmetic `applyAsym` used to do inline and throw the levels away, with `packScaleMinK4` beside it — the inverse of `get_scale_min_k4`. Both callers go through it: the simulation writes back `d*sc*l - dmin*m` as floats, the bank packs the levels, so "the bank is the format L8c-3 measured" is true by construction rather than by a tolerance. |
| `llm/bank_q4.go`, `shaders/llm_q4k.glsl` | **L8c-4: the dense bank at 4.500 bits.** §2.8's fragment tiling with a *nibble* where L8a puts a byte — a `uint` is eight consecutive k of one output column — plus a record plane of sixteen bytes per (n-tile, super-block, row): `d`, `dmin` and the twelve bytes ggml packs eight 6-bit (scale, min) pairs into, k-major inside an n-tile for D8's reason. `q4kFits` refuses a row that is not a whole number of 256-element super-blocks, which is the 320-wide hyper-connection family and nothing else. |
| `shaders/llm_gemm.comp -DQ4B`, `llm_gemv.comp -DQ4B` | **L8c-4: that bank read.** The Q8 arm plus an affine term, and one structural difference: a record covers 256 k where a k-slab is 32, so each lane holds *its own column's* record in a register and refreshes it only where the loop crosses a super-block, turning it into `(d*sc, dmin*m)` in a BN-long LDS vector — the format's own 12.5% of record traffic rather than eight times it. The GEMV keeps the record in a register across the four steps of a super-block and has **no LDS at all**, which is also why it is not bit-exact against the GEMM on this bank (L8c-4's finding 3). |
| `llm/imatrix.go`, `reference/quant_ref.c` | **L8c-2: the published importance matrix, and ggml's quantiser checked against ggml.** The imatrix is a GGUF — `<weight>.in_sum2` per input column, `.counts` beside it — so `gguf/` opened it unchanged; it covers `blk.N.*` only, so the lm head has no calibration data and falls back to rtn the way ggml does. `make_qx_quants` is ported into `sim.go` with three arms (`rtn`, `search`, `imatrix`, and `+gain`) and is **bit-identical to `ggml_quantize_chunk` over 819 200 values**, which is what `reference/quant_ref.c` exists for. |
| `llm/sim.go`, `cmd/gguf -width` | **L8c-1: a width, graded before its kernel exists.** `Model.F32` is the seam every streamed dense weight crosses, so a candidate format is one round trip inserted there — and because a quantised kernel multiplies in fp16 (L8b-2), the half it stages is the half that kernel would form. Not a model of the bank: the bank's own numbers. `LLM_DENSE_SIM` takes a plan (`deltanet=q4_0/32,hyper_conn=q6sym/32`) because the families do not want the same width, `_SRC` splits it on L8a-2's line, and every run prints the weights it actually touched — which is how `lm_head` reading as "free" was caught as a missed staging path rather than believed. `cmd/gguf -width` is the bytes half. |
| `cmd/llm -ppl`, `llm.Graph.ForwardRows` | **L8c-0: perplexity, in llama.cpp's own protocol.** A prefill without `inp_out_ids` — the final mixer over the whole batch, the head over slabs of `GraphOpts.HeadRows` because a row of logits is 0.99 MB — and 145 chunks of `wiki.test.raw` reduced the way `tools/perplexity/perplexity.cpp` reduces them. `HCGPU.MixedRowPort(t)` is the one new port it needed. **4.0289 against the reference's 4.0340**, in 7m58s. |
| `cmd/llm -head` | **L8a: the lm head on both banks in one process**, every rung at 1, 8, 64 and 512 rows, with the logits compared element for element — which is how "0 of 248320 differ" is a measurement rather than a claim. **L8c-4 makes it four arms**: the 4.5-bit bank, and *the simulation of it* staged beside it as halves, so the format's own numbers and the bank's are produced in one process against one activation. Plus the decode GEMV at one row on both quantised banks. |
| `LLM_DENSE_BANK`, `llm.DenseBankPlan` | **L8c-4: which families are on the real 4.5-bit bank**, in `LLM_DENSE_SIM`'s own grammar so an accuracy row and a byte row still name the same thing. It defaults to `imatrix` where a simulation defaults to `rtn`, because a bank is built at the arm L8c-3 recommends; a family the published matrix does not cover falls back to round-to-nearest the way ggml does. Setting it *and* `LLM_DENSE_SIM` is refused rather than averaged — it would quantise the same weight twice. |
| `llm/gpu_head.go` | **L6b: the lm head.** `output.weight`, [2560, 248320] Q8_0 — 675 MB, 1.27 GB as halves — staged a 4096-row slab at a time straight into the fragment tiling, and run as `llm_gemm.comp`'s MODE 2 on **one row**, because `inp_out_ids` is what the reference computes. The only matmul in the vertical with no epilogue: the final hyper-connection mixer is the output norm. |
| `llm/arena.go` | **L6b: where an activation arena's memory comes from.** A HOST_CACHED memory type reads at **25.06 GB/s** against the write-combined default's **0.18**, for 0.14% on the kernels (L6b-4) — and `narrowRows`, the f32→fp16 conversion every block's `Upload` does, parallel over rows. `LLM_ARENA_UNCACHED=1` is the control. |
| `vk.Device.NewHostCachedBuffer` | **L6b: a buffer from a HOST_CACHED type**, falling back to `NewBuffer` where there is none. For arenas the host reads; not for weight banks, which are written once. |
| `cmd/llm -graph` | **L6b: the model, end to end.** Tokens to logits, two timed passes a length, with the wall clock split by block and the glue named rather than averaged in. |
| `llm/deltanet.go` | **L3a: the gated DeltaNet, 36 of the 48 layers.** The fused [2560, 10240] qkv projection, the depthwise causal conv, the L2 norm under either of llama.cpp's two spellings of it (`QKNorm`), the two F32 gate projections and the delta rule itself — with a `DeltaNetState` carrying both the [128, 128, 48] recurrent state *and* the convolution's window, bit-identically across a batch split. |
| `llm/gpu_deltanet.go` | **L3b: that layer on the device, in five dispatches.** The fused `[16512, 2560]` projection for four of llama.cpp's matrices — including both F32 gate projections — a recurrent state per staged layer, the convolution's window as `Conv-1` rows of negative token index in front of the projection's own output, and a sweep profiler over every staged layer. |
| `shaders/llm_dn_*.comp` | **L3b: the layer's three kernels.** `llm_dn_conv` (the depthwise causal convolution, its SiLU, the per-head L2 norm of q and k under either spelling, and the softplus/sigmoid pair, over a (plane, token) grid), `llm_dn_scan` (the delta rule — **a fifteen-rung ladder on LPC and QKREG**, whose ends are 6.3x apart) and `llm_dn_norm` (the gated RMS norm straight into the output matmul's A operand). |
| `llm/ple.go`, `llm/gpu_ple.go` | **L2d: the n-gram block.** The trigram hash and the host-side gather out of the mmap'd 28.80 GB table (D2), the CPU block, and the three-dispatch device path. |
| `zimage/qwen/` | a working Qwen3 transformer on the GPU in Go — RMSNorm, RoPE, GQA, SwiGLU, four shared arenas. The skeleton for L2. |
| `shaders/gemv_w4a8.comp` | decode GEMV at **99-103% of the bus** (§1.1, §1.7), plus 25 grouped / M-blocked / N-blocked MoE builds (§1.8-§1.12). |
| `shaders/gemm_wmma_q4.comp` | Q4 prefill GEMM, **2.10x fp16** on a MoE block (§2.2), with L0c's k-major scale plane. |
| `shaders/moe_route/gather/combine` | top-k routing and the permutation around it. |
| `shaders/qwen_attn_wmma_*`, `qwen_rope.comp` | causal WMMA attention at 38.8 TFLOP/s (§3.3); NeoX rotary. |
| `rmsnorm_shared.comp` | 95% of the bus. |
| `shaders/bank_gather.comp` + the `bank` family | L0a and L0b's probe. |
| `bench/modelshapes.go` | **L1: corrected** to the real tensor table. |

## What has to be built

1. ~~**Hyper-connections**~~ — **done at L2b and L2c**, the block the whole
   prefill question turned on: 4-branch gated residual at width 10240, every
   layer, 194 low-rank projections a token, now four dispatches and 4.24x.
2. ~~**Gated DeltaNet — 36 of the 48 layers**~~ — **done at L3a and L3b.**
   The CPU reference matches all eighteen tensors of layer 0 with the
   recurrence at 3.11e-09 rms, and the kernel runs the whole layer in five
   dispatches at 1.37x the reference, the scan itself at 1.06x. §3.6's
   question — scan or chunk — is **answered by pricing rather than by
   building**: the recurrence is 13% of this layer against the fused
   projection's 64%, and a perfect chunked kernel would save 1.0% of the
   prefill graph (L3b-5).
3. ~~**QSA sparse attention**~~ — **done at L2e, L2f, L4a and L4b.** The CPU
   reference, the layer's four kernels, the selection settled as a radix
   select at 4 k, and the selection as a kernel: a per-cell bitmask that
   `llm_attn_wmma.comp` reads beside the causal mask, **1.94x llama.cpp's
   TOP_K and its GET_ROWS together**. What the architecture is named for turns
   out to **cost** 1.21x at prefill, because the reference runs dense flash
   attention over a mask and so do we (L4b-3); it is an optimisation at
   decode, not here. Still not in `IDEAS.md`; it needs a section.
4. ~~**PLE n-gram**~~ — **done at L2d**: trigram hashing into 16 heads over a
   320 M-row mmap'd table, `layer_multipliers`, conv1d k=4, key/value
   projections, bit-exact and in three dispatches.
5. ~~**The MoE block**~~ — **done at L5a and L5b.** The CPU reference matches
   all fourteen tensors of the FFN half at 4096 tokens, and the kernel runs
   the block in nine dispatches against about 845 — **1.09x at llama.cpp's
   own best ubatch and 1.55x at 2048** — over a bank that stays in the
   checkpoint's own Q4_K, Q5_K, Q5_1 and Q8_0 blocks because 241 GB of halves
   is not a thing this machine holds. L5a-4's 26x imbalance is answered by a
   device-built tile *list* rather than a grid, and by padding each expert's
   rows to the block so the epilogue needs no LDS at all (L5b-3).
6. ~~**Residency**~~ — **done at L6a**: 84.20 GB in 68 buffers, and a bank
   that is an array because a storage buffer here is at most 4 GiB - 4.
7. ~~**The order, the head, and the glue**~~ — **done at L6b and L6c.** The
   model runs: the argmax is llama.cpp's, out of llama.cpp's own top ten, and
   prefill is **2.40x** the reference at ubatch 2048 with **96.6% of the
   graph in the five blocks**. What is left of prefill is inside the kernels,
   not between them — L5b-7's unpack in the MoE (53.7% of the pass) and
   L3b-5's fused projection in the DeltaNet (20.3%).
8. ~~**KV cache** and a decode loop~~ — **done at L7a, L7b and L7c.** The
   layer runs over nKV cells with the indexer's raw key cached per cell and
   its pooled table rebuilt only where a block completed; both convolutions
   read a per-layer ring; the graph keeps the sequence because the n-gram hash
   is a trigram; and `cmd/llm -gen` generates llama.cpp's text. The rotary
   table is still a row per context cell — 64 floats a cell, 2 MB at 32768 —
   which is small enough that it stayed a table.

9. **The decode kernels** (L7d): a K-split GEMV where a 336-column projection
   launches 21 workgroups, a row block of one where the MoE pads to 32, and
   one command buffer a token.
10. ~~**The dense bank in the checkpoint's own width**~~ — **done at L8a for
   three of the four families.** int8 tiles and an fp16 scale plane, unpacked
   into LDS, bit-identical to the halves they replace; 8.07 GB a token becomes
   5.01 and decode 11.90 tok/s becomes 14.07. The hyper-connection block is
   L8b, because none of its three kernels is the plain arm.
11. **The re-quantiser** (L8c) — **done at L8c-4 for the lm head**, which is
   the format, both kernel arms and the plumbing; what is left is the other
   four streamed families on the bank that now exists, plus a packing for the
   320-wide hyper-connection pair. And **MTP speculative decoding** (phase 3).

---

## Task list

### L0 — answer the format question by measurement, not arithmetic

- [x] **L0.0** the inventory without downloading the model —
      `reference/gguf_inventory.py`, 35 MB of range requests.
- [x] **L0a** *Does the bus survive a 64-82 GB working set?* **It does not
      notice.** [Write-up](research/l0a-bank-range.md)
- [x] **L0b** *Is heap 0 as fast as heap 1?* **No type or heap is faster than
      any other, and the heap sizes are not limits.**
      [Write-up](research/5.1-memory-types.md)
- [x] **L0c** *The blocked scale plane.* **1.27x, and it changes the format.**
      [Write-up](research/l0c-scale-plane.md)
- [x] **L0d** *Per-format error metrics* (§7). **W4A8, symmetric, per-token
      activation scales.** [Write-up](research/l0d-quant-error.md)

### L1 — get the checkpoint, and a baseline  *(done)*

- [x] Download `UD-Q4_K_XL` (4 shards, 111.32 GB) and `mtp-…-Q4_K_M.gguf`
      (2.79 GB). **18 minutes**, `reference/fetch_llm_checkpoint.sh`, resumable.
- [x] `llama-bench` and `llama-completion`: **pp2048 388.60, tg128 25.15**.
- [x] Go GGUF reader, checked tensor-for-tensor: **1224/1224, 0
      disagreements**.
- [x] Dequant paths for the five formats, **bit-exact against ggml**.
- [x] Tokenizer: 248320 vocab, the multimodal specials and the `qwen35`
      pre-tokenizer — **213/213 against `llama-tokenize`**.
- [x] Correct `bench/modelshapes.go` to the real inventory.
- [x] `llama-perplexity` on wikitext-2 (145 chunks at n_ctx 2048, 15.2
      minutes): **PPL = 4.0340 ± 0.02283**, the accuracy reference for
      everything phase 2 does.
- [x] Re-run the `shapes` family against the corrected table — 313 rows. It
      prices a decode step at 3.33 GB and **73 tok/s at 4 bits before
      attention**, with **2081 dispatches costing 0.62 ms, 5%** (§4.1). The
      old table had 1861 and read 13.4% too few weights.

### L2 — the dense skeleton  *(done)*

- [x] **L2a — settle the prefill question.** Done by per-op attribution of
      llama.cpp's own graph rather than by pricing one layer: **DeltaNet is
      1.6%, attention+QSA 1.5%, and the 5x is MoE (35.7%), dense matmul
      (26.2%), glue (30.6%) and tiny-N F32 (12.2%)**. Against our own kernels
      the chunk is 2.25-2.53x, so **the target is ~1150 tok/s, not ~2000**.
      [Write-up](research/l2a-prefill-attribution.md) · `results/l2a_prefill_ops.csv`
- [x] **L2b — the hyper-connection block, CPU reference.** `llm/`: the block,
      a `Config` read from the checkpoint, and **`reference/eval_dump.c`**, a
      whole-tensor oracle (143 tensors, 35.7 MB, all of layers 0-3 — enough
      for L2, L3 *and* L4 from one model load). Every mixer and combine of the
      four dumped layers matches; four of seven gates to **6.2e-08 rms**. And
      it found what the oracle is really computing — see L2b-2 above.
      [Write-up](research/l2b-hyper-connections.md)
- [x] **L2c — the fused hyper-connection kernel.** L2a-2's specification,
      built: grouped RMSNorm into the GEMM's fp16 A layout, `hc.down` **with
      `inject`'s four columns fused onto it**, the low-rank gate's sigmoid and
      the 4-branch collapse inside the up projection's epilogue, and the
      combine in one pass. **Four dispatches against sixteen, 226.6 ms of
      llama.cpp's 512-token graph against 53.4 — 4.24x**, and it reproduces
      L2b's CPU reference to 1e-04 rms on all eight mixers of layers 0-3.
      [Write-up](research/l2c-hc-kernel.md) · `results/l2c_hc.csv`
- [x] **L2d — the PLE gather and the n-gram block.** The host-side trigram
      hash (**bit-exact**, 112 rows of a 320 M-row table), the block in Go
      (**7.7e-08 rms**) and on the device in three dispatches against about
      thirty. Layer 1's mixer, the one L2c could not compare, now matches too.
      [Write-up](research/l2d-ple.md) · `results/l2d_ple.csv`
- [x] **L2e — one full-attention layer with mrope, and the QSA indexer beside
      it.** The CPU reference: the fused query/gate projection, IMRoPE (which
      is **NeoX on text**, asserted rather than assumed), the block-pooled
      indexer and causal GQA. Every dumped tensor of layer 3 matches, and the
      four-stage chain from `l_last-2` lands at **9.75e-06 rms**. Two more
      oracle numerics fell out. [Write-up](research/l2e-attention.md)
- [x] **L2f — the GPU port of that layer.** Six dispatches against the
      reference's nineteen: one fused `[13952, 2560]` projection for six of its
      matrices, one pass doing the per-head norm, the M-RoPE and the fragment
      tiling for q, k and v together, the indexer's pool/norm/rotate in one
      kernel and its score/bias/mask in another, and causal GQA on the matrix
      cores **with the output gate in its epilogue**. **66.3 ms of llama.cpp's
      512-token graph becomes 27.6 — 2.40x, 1.60x at equal attention work** —
      and the layer's output is 5.9e-05 rms against the f32 model, 19.3x nearer
      it than the oracle. [Write-up](research/l2f-attention-gpu.md) ·
      `results/l2f_attn.csv`
- [x] Gate: layer 3's output matches the reference. **Not "to fp16
      tolerance"** — L2b-3: an rms bound under `llm.RefQ8`, since the
      reference is computing in int8 and the error is heavy-tailed.
      `hc_combine-3`, the residual after the attention half, is at 9.75e-06 rms
      on the CPU and `attn_output-3` at 1.14e-03 against llama.cpp on the
      device. `l_last-3` needs the MoE half, which is L5.

### L3 — Gated DeltaNet  *(done)*

- [x] **L3a — CPU reference, fp32 state, against the dump.** All eighteen
      tensors of layer 0's graph match: the recurrence **3.11e-09 rms**, its
      786 432-value state **1.91e-08**, the layer's output **1.31e-07**, with
      layers 1 and 2 beside it. Five things the architecture does not settle
      and the reference does — it **does not chunk**, the head map is
      **modulo**, the state is **stored transposed**, `softplus` **flushes to
      zero** in f32, and the build's `build_gdn_l2_norm` **predates #28068's
      fix** and reproducing that is worth 15x. Plus L2e-3 sharpened into a
      threshold: the fp16 F32 matmul starts at **8 output columns**, so it is
      absent from this 7-token dump and present at a 512-token ubatch.
      [Write-up](research/l3-deltanet.md)
- [x] Gate: layer 0 matches; state **bit-identical** in fp32 across a chunk
      boundary — 3 + 4 tokens reproduce 7 exactly, output and state, with the
      convolution's window carried beside the recurrent state and a control
      showing the carry matters (4.5e-02 rms without it).
- [x] **L3b — the GPU kernel.** Five dispatches against eleven: one fused
      [16512, 2560] projection for four of llama.cpp's matrices, one pass doing
      the convolution, its SiLU, both L2 norms and the two per-head scalars,
      the delta rule with the state in registers, the gated norm straight into
      the output matmul's A operand, and the output projection. **152.8 ms of
      llama.cpp's 512-token graph becomes 111.5 — 1.37x** — and the scan itself
      is **413 us against 438**. §3.6's question is answered by a price rather
      than a build: the recurrence is 13% of this layer, the projection 64%,
      and a perfect chunked kernel saves 1.0% of the prefill graph.
      [Write-up](research/l3b-deltanet-gpu.md) · `results/l3b_dn.csv`
- [x] Gate: layers 0, 1 and 2 match — `attn_output` at **2.4e-06 rms** against
      both the CPU reference and llama.cpp — and 3 + 4 tokens reproduce 7
      **bit-identically** on the device, with the control that zeroes the
      convolution's window moving the output by 4.0e-02.

### L4 — QSA

- [x] **L4a — re-dump at 4 k, and settle the selection on the CPU.** 4096
      tokens in a 4096-cell cache, one ubatch, 8.0 GB, the same pinned build
      — and the filter brings the whole MoE block with it, so L5's fixture
      exists already. `eval_dump.c` grew `-f` and `-nt`, and is now built
      against headers extracted from the *build's* commit rather than from a
      working tree that has moved on. The selection is a **radix select**
      reproduced pass for pass (all 4096 rows algorithmically, **set for set
      on the 2045 biting rows**), its block granularity is checked rather than
      read, its order is measured to be irreproducible and its visible set to
      be stable, and **0.0022% of it survives our own scores**. Two findings
      that are not about QSA: **every quantised matmul accumulates in fp16 at
      a real ubatch** (so the prefill tolerance is 5e-3, and the reference is
      the side losing precision), and the indexer's **BF16 weights meet a bf16
      activation**, worth 2061x. Plus two subnormal bugs.
      [Write-up](research/l4-qsa.md)
- [x] Gate: our selection is the reference's, from the reference's own scores
      — `TestQSASelectionIsARadixSelect`, with
      `TestQSASelectionIsStableAcrossRuns` and
      `TestQSASelectionIsBlockGranular` beside it.
- [x] **L4b — the selection on the device, and the gathered attention.**
      `llm_attn_select.comp`: one workgroup a token, the row's keys in LDS,
      four radix passes, a **per-cell bitmask** out — and `llm_attn_wmma.comp`
      reading it beside the causal mask, staged a key block at a time. **2.40
      ms of llama.cpp's 512-token graph becomes 1.24, 1.94x**, and its second
      op (a `GET_ROWS` turning an index list back into a mask) is deleted
      rather than beaten. The reference's serial 256-bucket walk is 1.86x of
      it. And the thing the architecture is named for is now priced: at
      prefill the selection **costs** the attention kernel 1.21x, because
      `build_attn_qsa` runs dense flash attention over a mask and so do we.
      [Write-up](research/l4b-qsa-gpu.md) · `results/l4b_qsa.csv`
- [x] Gate: layer 3's `attn_output` matches at 4 k context, on the device,
      where the selection bites — **9.872e-04 rms** against llama.cpp, inside
      the 7-token figure — with the bitmask **cell for cell** the CPU
      selection's on all 4096 rows and a dense control 2.0x further away.

### L5 — the MoE block

- [x] **L5a — the block in Go, against the 4k dump.** `llm/moe.go`: the F32
      router, the softmax over 512, the argsort top-10, the normalised weights
      with the reference's clamp, the three expert matmuls walked **by expert**
      so each bank is gathered once, the shared expert and its one-column
      sigmoid gate. All fourteen tensors match — `ffn_out-3` at **1.378e-04
      rms** — and the selection is the reference's **set for set on all 4096
      tokens**. Two findings that are not about correctness: modelling the
      reference's int8 activations makes the fit **worse** here (L5a-3), and
      the routing is **not balanced** — 274 of 512 experts at ubatch 512, one
      of them taking 95% of the tokens (L5a-4).
      [Write-up](research/l5a-moe.md)
- [x] Gate: layers 3's whole FFN half matches from llama.cpp's own
      `hc_mixed-3` (the **second** occurrence — the block runs twice a layer),
      with three negative controls: the expert bank's row order (175x), the
      SwiGLU's two halves (59x) and the `weights_sum` clamp, which is invisible
      in the data and asserted from the graph.
- [x] **L5b — the kernel.** Nine dispatches: the router with the shared
      expert's gate as a 513th column, the softmax/top-10/normalise in one
      workgroup a token, a counting sort and its two tile schedules, a
      **grouped GEMM over the checkpoint's own quantised blocks** (Q4_K, Q5_K,
      Q5_1, Q8_0 — because 241 GB of halves is not a thing this machine
      holds), and the weighted combine. L5a-4's 26x imbalance is answered by a
      device-built tile *list* over a static bound, and by padding each
      expert's rows to the row block so a cooperative-matrix store goes
      straight to global. **567.4 ms of llama.cpp's 512-token graph becomes
      522.3 — 1.09x — and 1758.4 becomes 1131.6 at ubatch 2048, 1.55x.**
      Two optimisations that should have worked did not, and are written down
      (L5b-4). [Write-up](research/l5b-moe-gpu.md) · `results/l5b_moe.csv`
- [x] Gate: the block matches on the device at 4096 tokens — **`ffn_out-3` at
      8.132e-05 rms**, inside L5a's figure over 48 tokens — with the selection
      the reference's set for set *and order for order*, the permutation
      checked as an exact bijection, and the padding shown to be inert by
      poisoning the row it reads. Rates reported against L0a: `up` at 70 GB/s
      of quantised bank and 10.8 TFLOP/s, `down` at 121 GB/s. 35.7% of
      llama.cpp's graph in `moe_experts` alone, 48.7% counting everything
      nameably this block, becomes 44.8%.
- [ ] **L5c — decode's shape, when L7 needs it.** The grouped Q4 GEMV
      (§1.8-§1.12) against a one-token routing, where L5a-4's hot expert is
      the same 0.5 MB of Q4_K 95% of the time and would sit in the MALL if
      anything kept it there. Not built at L5b, because at prefill the
      grouped GEMM is the whole block.

### L6 — the whole stack, prefill only

- [x] **L6a — residency.** All 48 layers of all five blocks on the device at
      once: **84.20 GB of weights and 0.87 GB of arenas in 68 buffers, staged
      in 33 s**, the n-gram table still mmap'd, 41 GB of the machine left
      over. The MoE bank is an **array of buffers, one a layer**, because
      `maxStorageBufferRange` is 4 GiB - 4 and a layer is 1.61 GB — a
      per-binding `descriptorCount` in the shim, a block-instance array in the
      GLSL, NBANK compiled in at 48, and the layer index in the top sixteen
      bits of `moeUsed` because the push block was full. **Residency is free
      in the bank's size** (L6a-4). [Write-up](research/l6a-residency.md) ·
      `results/l6a_resident.csv`
- [x] Gate: the bank index is load-bearing — layer 3 read from bank 3 is
      **8.133e-05 rms** against llama.cpp, the same run with the index forced
      to zero is **203x further away**, and every L5b gate still passes with
      the array in place.
- [x] **L6b — the graph.** The five blocks in llama.cpp's own order, plus the
      two tensors no block owned: the embedding gather and the 675 MB lm
      head. **Nothing in a shader changed.** The plumbing did — the
      activation arenas moved to a **HOST_CACHED** memory type, because the
      host reads a write-combined one at **0.18 GB/s** against **25.06**
      (L6b-4, 5.0x on the graph for 0.14% on the kernels), and `Upload`'s
      f32→fp16 narrowing was parallelised over rows (L6b-5).
      [Write-up](research/l6b-graph.md) · `results/l6b_graph.csv` ·
      `results/l6b_arena.csv`
- [x] Gate: **the argmax is llama.cpp's** (561), out of llama.cpp's own top
      ten, with the drift priced as a geometric x1.085 a layer at six depths
      (L6b-3); prefill is **713.9 tok/s at ubatch 2048 against 391.42 —
      1.82x**, and 2.12x against llama.cpp *at the same ubatch*.
- [x] **L6c — the glue on the device.** Not the shared arena this was scoped
      as: `shaders/llm_move.comp` is one dispatch over two blocks' buffers,
      nine pipelines for the five blocks and the head, with a `Port` per
      crossing tensor. **Bit-exact** — every figure of L6b's gate came back
      identical to the last place (L6c-2) — and **2.0% of the pass**, which
      is what a shared arena would now buy and why it is priced rather than
      built (L6c-3). Along the way, 113 MB of Go allocation a prefill in
      `DeltaNetGPU.Reset`, flat in the prompt length and so 12% of a
      128-token pass (L6c-4). [Write-up](research/l6c-moves.md) ·
      `results/l6c_graph.csv`
- [x] Gate: **no host tensor between the embedding gather and the logits**;
      **941.3 tok/s at ubatch 2048, 2.40x llama.cpp's 391.42**, and 572.9 at
      512 against llama-bench's own pp512 of 313.62, 1.83x. 96.6% of the
      graph is the blocks; L2a-3's ~1150 projection is 82% reached and the
      rest of it is inside the kernels.

### L7 — decode  *(done)*

- [x] **L7a — the KV cache.** The full-attention layer over nKV cells, one
      plane a layer, with every causal comparison a cell against a *position*.
      The indexer's **raw** key cached per cell and written as one more plane
      of the pack kernel's grid (a pooled block spans four cells and at decode
      those arrive in four batches), and its **pooled** table incremental —
      `[past/ratio, (past+T)/ratio)`, one workgroup instead of 1024. Two more
      push fields borrowed, because 64 uints is 256 bytes. And **L2e's fp16
      round trip had been folded away by the driver since L2e**: a value that
      models a memory format has to go through memory (L7a-4).
      [Write-up](research/l7a-kv-cache.md)
- [x] Gate: a 4096-token prompt in chunks of 512, 64, 7 and one is
      **identical to the last place** — all 10 485 760 floats — and
      `TestAttnGPUCacheKeepsPooledBlocks` asserts the incremental range
      against an independent predicate.
- [x] **L7b — the sequence.** Five histories carry and one of them was
      **shared by all 36 DeltaNet layers**, invisible while every pass was a
      fresh sequence and 12x wrong at `l_last-1` the moment it was not. Both
      convolutions now read a per-layer ring addressed by absolute position,
      written by one 15-line kernel, which is what makes the store a pure
      write and deletes the host-side `Carry` — which was itself wrong for any
      run shorter than the window. [Write-up](research/l7b-sequence.md)
- [x] Gate: 512 tokens in chunks of 128, of 9, and as 495 then seventeen
      single ones give `result_norm` **identical to the last place**; the
      control, every token its own sequence, is 1.785e+00 rms away.
- [x] **L7c — the loop.** `llm/sample.go` (greedy by default, because the gate
      is only a check if the sampler is a function of the logits alone) and
      `cmd/llm -gen`. The stop condition is **not `eos_token_id`**. One
      `madvise(MADV_RANDOM)` on the n-gram table is worth **176x** on the host
      gather; a command buffer a block rather than a submit a dispatch is
      worth 12% and prices a submit at 31 us.
      [Write-up](research/l7c-decode.md) · `results/l7c_decode.csv`
- [x] Gate: **llama.cpp's text**, diverging at one token where our top two
      logits are 0.161 apart and re-converging within a sentence.
      **7.46 tok/s against 25.15** — 30% of the 25.0 this bank's own 9.67 GB a
      token allow (L7c-5) — and prefill unchanged at **950.3 tok/s, 2.43x**.

### L7d — the decode kernels  *(done)*

- [x] **The hyper-connection down projection as a split-K GEMV**,
      `shaders/llm_hc_gemv.comp`. Seven workgroups became 3360 and **238.6 us
      became 30.0 — 7.95x, 29 GB/s to 230, 95% of the bus** — over the weight
      the GEMM already staged, because §2.8's fragment tiling makes a
      workgroup's whole slab one contiguous run and hands each lane a single
      output column. Which split to take is **§5.1b's 4 KB period**: the two
      rungs whose slab stride misses the rotation are the two fast ones, out
      of six.
- [x] **Narrow column blocks for the MoE**, not a narrower row block. At one
      token ten experts have one row each, so every rung makes ten tiles and
      the padding was never the constraint — `up` at 100 workgroups reads 71
      GB/s and the shared expert's at **ten** reads **23**, where `down` at
      400 reads 136. BN 64 to 16 is **1.28x on the block**.
- [x] **One command buffer a pass**, `llm/record.go`: ~490 submits to two,
      **1.18x**. It needed the attribution to move inside the command buffer —
      `vk.DispatchMultiMarked` writes a timestamp after every dispatch, so a
      block's figure is GPU time rather than host wall clock, which is the
      same quantity `GGML_VK_PERF_LOGGER` reports for llama.cpp's graph.
- [x] **The first rung here that is not bit-exact against its ladder.**
      Splitting K re-associates a 10240-long sum, so `TestGraphIsAChunkSplit`
      pins the schedule for its three equalities and measures the decode
      schedule beside them: **4.43e-04 rms, 4000x smaller than a dropped
      history** and the same order as our distance from llama.cpp itself.
- [x] Gate: **the same text** — llama.cpp re-run beside it, the same single
      divergence, the same `Lisbon.` — at **11.89 tok/s against 7.46**, 47% of
      the reference's 25.15 and **48% of this bank's own 25.0 ceiling**, with
      prefill at **990.6 tok/s, 2.53x**. Two runs agree to 0.08%.
      [Write-up](research/l7d-decode-kernels.md) · `results/l7d_decode.csv` ·
      `results/l7d_graph.csv` · `results/l7d_hc_gemv.csv` ·
      `results/l7d_moe_decode.csv`

### L8 — phase 2, our own bank

> **The letters are allocation order; this list is execution order.** L8d was
> planned after L8c and is being done before it, because L8b's own numbers
> said so: two stages of bank work bought 1.24x at decode and left the MoE
> reading its own bytes at 4.3x off the bus, which is worth more than the
> re-quantisation and costs no accuracy to collect. L8c keeps its name because
> twelve places in this file, a closed write-up and a code comment cite it to
> mean *the re-quantisation*, and renaming that is churn with no reader on the
> other end.

- [x] **Stop expanding the dense half**, which is 1.66x on the decode ceiling
      before any re-quantisation: 8.07 GB a token of fp16 against 3.67 of the
      Q8_0 the checkpoint ships (L7c-5). **L8a does three of the four
      families** — the gated DeltaNet, the full-attention layer and the lm
      head — as the checkpoint's own int8 with an fp16 scale per 32 elements
      of a row, in the same §2.8 fragment tiling, unpacked into LDS by
      `llm_gemm.comp`'s `-DQ8B` arm. **8.07 GB becomes 5.01 and the token
      6.61; decode 11.90 to 14.07, prefill 990.6 to 1044.8, residency 84.20 GB
      to 82.40 — and the halves the matrix cores multiply are the same
      halves.** [Write-up](research/l8a-dense-bank.md) ·
      `results/l8a_decode.csv` · `results/l8a_graph.csv` ·
      `results/l8a_head.csv`

### L8b — the hyper-connection block on the same bank  *(done)*

- [x] **Three more arms of the same kernel**, and none of them the plain one.
      The down projection's `inject` is four F32 rows that do not begin on a
      column block, so the split is the block that *contains* them (288) and
      32 low-rank rows are staged twice; the up projection's collapse scratch
      went from [BM][BN] to one m-tile so that a row block wide enough to
      amortise the unpack would fit beside it; and the split-K GEMV, which has
      no LDS to round a weight through, multiplies in fp16 so that its half is
      the GEMM's half. **Decode 14.07 to 14.71 tok/s, the block 79.2 us a
      mixer to 37.0 at one token, a token 6.61 GB to 6.05, residency 82.40 GB
      to 81.89, prefill 1044.8 to 1041.1 at ubatch 2048.**
      [Write-up](research/l8b-hc-bank.md) · `results/l8b_decode.csv` ·
      `results/l8b_graph.csv` · `results/l8b_hc.csv`
- [x] Gate: **the same text** (`TestGraphLogits`, and `-gen -n 128` down to
      `Lisbon.`), **0.740 GB a token** for the block against the ~0.70
      predicted, and prefill inside the 0.36% two runs agree to. The
      prediction of ~15 tok/s was right to 2%. What it did *not* predict is
      that the block's own projections are 1.09-1.15x **slower** at ubatch
      2048 than on halves — at 2048 tokens the two weights are read 32 times
      each out of a 32 MiB MALL, so the bytes were already hidden and only the
      unpack's ALU is left — and that the m-tile scratch pays for it by making
      the fp16 arm 1.13-1.24x faster as well.
### L8d — the decode kernels  *(done)*

- [x] **Read the grid, then the loads** — and the grid turned out to be five
      dispatches rather than one. `llm_moe_gemv.comp` is the expert GEMM at
      one token (2.46-6.17x a dispatch), `llm_moe_router.comp` the router's
      split-K (11.8x), `llm_moe_combine.comp`'s new grid the combine (7.8x at
      decode, 1.35x at prefill), `llm_gemv.comp` the gated DeltaNet's two
      projections (1.87x on the layer) — and under all of them a tile-grid
      bound that was 2052 records where eleven exist. **Decode 14.71 tok/s to
      23.15, the MoE block 27.6 ms a token to 12.05 and 58 GB/s of bank to
      133, the DeltaNet 18.9 to 11.0 and 119 to 205, a token 66.7 ms to 43.0,
      prefill 1041.1 to 1049.8 at ubatch 2048 and 643.4 to 653.9 at 512.**
      [Write-up](research/l8d-moe-decode.md) · `results/l8d_decode.csv` ·
      `results/l8d_decode_gemm.csv` · `results/l8d_graph.csv` ·
      `results/l8d_moe.csv` · `results/l8d_dn.csv`
- [x] Gate: **no bank change**, the MoE at **133 GB/s** at one token against
      the ≥120 asked for, **23.15 tok/s** against ≥18, and prefill *faster* at
      both ubatches rather than merely no slower — D14's hazard did not fire,
      because the decode kernels are a different pipeline and not a different
      rung of the prefill one. The text is where the gate has to be read
      carefully: the capitals list and `Lisbon.` are identical and
      `TestGraphLogits` is unchanged, but at temperature zero the `<think>`
      block **re-words at one token**, whose top two logits are 0.012 apart
      against the 3-25 either side. Nothing reads a different weight; a
      cooperative-matrix accumulator and a GEMV lane associate the same
      products differently. `LLM_DECODE_GEMM=1` is the control and reproduces
      L7c's completion exactly.

<details><summary>What it was scoped as</summary>

- [ ] **Read the grid, then the loads.** The MoE is **42.1% of a token** and
      moves 1.60 GB in 28.6 ms — **56 GB/s of a 242 GB/s bus, 4.3x off its own
      bytes** — where the lm head, the same kernel family over the same kind
      of bank, holds **194** and is 1.25x off. Nothing here needs a
      re-quantisation: the expert bank has been the checkpoint's own Q4_K /
      Q5_K / Q5_1 / Q8_0 since L5b, so this is entirely a question of how the
      kernel reads a bank that is already narrow. **If every block ran at the
      head's rate a token would be 31 ms — 32 tok/s — against today's 68 ms
      and 14.71.** That is the largest single number left in the vertical and
      it is bigger than L8c's.
- [ ] The three leads, in the order D11 says to take them. **(a) The grid.**
      L7d-3 narrowed BN from 64 to 16 and moved `up` from 71 to 93 GB/s, not
      to `down`'s 136, so the grid was part of it and not all of it. **(b) The
      LDS round trip.** What is left of that difference is that MODE 0 gathers
      its A rows into LDS per K-step and MODE 1 does not — and at decode every
      workgroup gathers the **same** 2560-long row, 400 times. So the question
      is whether a **GEMV per expert** can feed itself from Q4_K without the
      round trip §2.2 called unavoidable for a *fragment*; §1.8-§1.12 have 25
      grouped builds of that shape measured against a repacked bank and none
      against this one, and **L8b-2 is the evidence that it works**: the same
      split over int8 tiles is 326 GB/s against the fp16 build's 230, with the
      arithmetic identical. **(c) The loads.** L5b-4 tried more waves, a wider
      K-step and a conflict-free pad (1.15-1.24x, 1.16x and 5.4x *against*);
      what nobody has tried is issuing the loads a K-step ahead of the unpack
      that consumes them (§2.7). L8b-5 is the second block to be left waiting
      on exactly that instruction sequence.
- [ ] **Then the DeltaNet, which is the same finding on a second block.** It
      is 27.8% of a token at 119 GB/s, 2.0x off its bytes, and its two
      projections are 258 and 218 workgroups where the head's grid is 3880 —
      D11 again, and the split-K GEMV L8b just priced over int8 is the same
      kernel shape. Cheaper than the MoE and worth ~half as much.
- [ ] Gate: **the same text**, no bank change at all, the MoE at **≥120 GB/s**
      at one token (2.1x) and **≥18 tok/s** end to end, with prefill no slower
      at ubatch 512 or 2048 — the hazard being D14, since the MoE's weights at
      prefill are read once each and its decode grid is not its prefill grid.

</details>

### L8e — the projections L8d did not reach  *(done)*

- [x] **The full-attention layer is the same finding a fifth time**, and
      `llm_gemv.comp` needed no change to take it: the wiring, a partial-sum
      arena, and a KSLABS ladder per projection. The fused projection lands on
      **k8** and the output one on **k32** — the DeltaNet's two rungs, measured
      rather than copied — and D12's 4 KB rotation is visible on the fused
      one, where k2's 20 480-byte slab is five whole pages and the worst rung
      on the ladder. **The layer 414.8 us to 225.9, the block 5.5 ms a token
      to 3.5 (354.5 ms of a run to 223.3), the fused projection at 227 GB/s of
      a 242 GB/s bus.**
- [x] **The MoE's shared expert has a rung of its own**, and only half of what
      that was scoped as survived contact with the whole model. Its down mode
      goes v16w4 to **v32w4**; its up mode was already on v64w4 for the wrong
      reason and stays; and carrying the *routed* up mode to the v16w4 the
      ladder names is **42.1 ms worse over 64 tokens** — D16.
- [x] **The second `perm` dispatch was the first one again** whenever the two
      expert modes share a row block, which is every decode step (all six GEMV
      rungs fix BM at 16) and both prefill rungs. One pass, and the down mode
      reads the up mode's tile list: **-48 dispatches a pass at every length**,
      0.14 ms a token and 0.9 ms of the 512-token prefill graph.
- [ ] **`route` is 5.8 us a layer and is left alone.** One workgroup of 256
      lanes, a softmax over 512 experts and ten sequential argmaxes — D11's
      symptom but not its disease, because the only axis it has across is the
      token and at decode there is one. Splitting the top-k needs a second
      dispatch to merge and a dispatch here is 1-2 us of the 5.8. The item's
      own condition was "only worth it if the token gets short enough"; at
      40.5 ms it is 0.7%.
- [x] Gate: **the same text** — token for token identical to
      `LLM_DECODE_GEMM=1`'s over 128 tokens, diffed rather than eyeballed,
      which is L7c's completion and so *tighter* than L8d's — **24.66 tok/s**
      against 23.15, and prefill **faster** at both ubatches (1052.8 at 2048
      against 1049.8, 655.2 at 512 against 653.9) rather than merely no
      slower. `TestGraphLogits` unchanged: argmax **561** out of llama.cpp's
      own top ten, drift 0.881% at 48 layers.
      [Write-up](research/l8e-attn-decode.md) · `results/l8e_decode.csv` ·
      `results/l8e_graph.csv` · `results/l8e_attn.csv` · `results/l8e_moe.csv`

### L8c — the widths that are ours rather than the checkpoint's  *(in progress)*

> **L8c-0 through L8c-5 are done, and the stage's direction turned twice.**
> The instrument came first, and what it measured retired D3: the
> re-quantisation as specified costs far more accuracy than it was assumed to,
> so the order became *calibrate, then decide whether to build the kernel*
> rather than *build the kernel*. L8c-2 calibrated, and the answer was no.
> **L8c-3 found that both of those were statements about a format rather than
> about the model** — in ggml's asymmetric K-quant, at the same bits, 4-bit
> dense weights cost +4.24% rather than +18.5% and calibration *helps* — so
> D3 is back at 4.5 bits, D7 is reversed, and the kernel is the next thing to
> build. **L8c-4 built it and L8c-5 put the largest family on it**, and between
> them they turned the stage's own instrument into a finding: an eight-chunk
> per-family screen does not bound a family's corpus cost and does not even
> fix its sign. The items below are in execution order.

- [x] **The instrument first**, because this is the stage where a tensor
      comparison stops being a grade. `Graph.ForwardRows` is a prefill without
      `inp_out_ids` — the final mixer over the whole batch, the head over
      slabs of `GraphOpts.HeadRows` — and `cmd/llm -ppl` is llama.cpp's
      protocol off the oracle's own build `cff184438`: no BOS (this
      checkpoint's `add_bos_token` is false), 145 chunks of 2048, a fresh
      sequence each, positions `[1024, 2047)` scored against the token to
      their right, over a tokenization checked to be **297 193 identical
      ids**. **Ours is 4.0289 ± 0.02279 against 4.0340 ± 0.02283 — −0.13% —
      in 7m58s**, and `TestGraphForwardRows` says row t of the batch is
      identical to the prompt truncated at t **to the last place**.
      [Write-up](research/l8c-perplexity.md) · `results/l8c_ppl.csv`
- [x] **Choose per-tensor widths — by measuring them, because L0d's ladder is
      a proxy and D3's number came off it.** `llm/sim.go` stages a candidate
      format's own halves through the kernels that exist, checked where the
      answer is known (`q8sym/32` over Q8_0 is an identity on 696 M weights)
      and against a **three-way control** — the int8 bank, the fp16 arm and
      `q8sym/32` on the Q8_0 tensors all at 2.0189. **D3's ~4.25 bits on
      everything is +18.5%**; the cost of 4 bits is **inverse to the bytes**
      (DeltaNet 46% of a token for +1.49%, hyper-connections 14% for +5.98%)
      — ~~the shape the plan was assembled on~~, **retired at L8c-6**, where
      on the asymmetric form over 145 chunks the two large families cost 0.87
      and 0.93 pp per GB of token and the outlier is the lm head at 2.59;
      the sensitive family wants **levels, not a finer group**; **`q4_0`'s
      sixteen levels beat `q4sym`'s fifteen by 1.13x free**; and the mixed
      plan is **4.2699, +5.98%, for a 52.9 tok/s ceiling**. **A plan is not
      the sum of its families.** [Write-up](research/l8c-widths.md) ·
      `results/l8c_widths.csv` · `results/l8c_ppl_mixed.csv`
- [x] **Settle D13's fp16 tail**, which L8a-2 deferred to here: int8 for
      **+0.01%** over the whole corpus, worth 0.6% of a token, so it is a
      simplification rather than a speed-up. `results/l8c_ppl_tail.csv`
- [x] **L8c-2 — the imatrix, and it makes this model worse.** Unsloth's
      published matrix is a GGUF our reader opens; `make_qx_quants` is ported
      and **bit-identical to ggml over 819 200 values** via
      `reference/quant_ref.c`, once `nearest_int`'s round-half-to-**even** and
      two `float` accumulators were matched. Three arms, because ggml searches
      and calibrates in one function. **Calibration is 22.4% against
      round-to-nearest's 18.5%**, all of the loss in the hyper-connection
      block (+9.04% against +5.98%, the DeltaNet unmoved). **The mechanism is
      a gain error**: rtn shrinks a matrix 0.08-0.12%, the imatrix shrinks
      `hc_attn_up` **1.35%**, and bias compounds on L6b-3's x1.085 where noise
      averages out. **What predicts it is the participation ratio inside a
      scale group** — `hc_attn_up`'s scale is fitted to an effective 4.4 of 32
      columns. Forcing the scale unbiased recovers a third.
      [Write-up](research/l8c-imatrix.md) · `results/l8c_imatrix.csv`
- [x] **L8c-3 — the asymmetric form, and it reverses D7 and the
      recommendation.** L8c-2's one untested caveat was that every rung in
      this stage was Q4_0's *symmetric* form. The min turns out to cost
      nothing — ggml nests it, so a super-block of eight groups of 32 with one
      fp16 pair and 12 bits a group is **0.500 bits a weight, exactly a
      symmetric fp16 scale per 32** — and at those identical 4.500 bits
      asymmetric is **2.06x cheaper uncalibrated** — 4.3124 against 4.6127
      over the whole corpus, +7.04% against +14.49% — and 3.2x on
      `hyper_conn`. **And the imatrix works on this form**: 4.5 bits on every
      streamed dense family is **4.1998 against 4.0289, +4.24%**, where the
      same matrix on the symmetric form makes it *worse* (4.6588, +15.63%),
      and calibration now helps every family it covers. A mixed plan at 4.75
      bits is **4.1377, +2.70%**. **The mechanism is the
      second parameter**, measured: a symmetric group's one free parameter
      *is* the gain, so a calibrated fit can only express a preference by
      shrinking the group — the extra shrinkage calibration costs
      `hc_attn_up` is **1.24 pp symmetric against 0.47 asymmetric**, and on
      `attn_qkv` it is gone. The port is bit-identical to
      `ggml_quantize_chunk` over 204 800 values an arm and **in f32**. One
      departure from ggml, on the tensor the question is about: K-quants want
      `k % 256 == 0` where `hc_*_up` is 320 wide, so its super-block is the
      whole row — ten groups, 4.475 bits.
      [Write-up](research/l8c-asymmetric.md) · `results/l8c_asym.csv` ·
      `results/l8c_ppl_asym.csv` · `results/l8c_ppl_asym_mixed.csv`
- [x] **L8c-4 — the dense kernel, and the bank is the format.** §2.8's
      fragment tiling with a *nibble* where L8a puts a byte — a `uint` is
      eight consecutive k of one output column — plus ggml's own sixteen-byte
      super-block record per (n-tile, super-block, row), k-major inside an
      n-tile for D8's reason: **4.500 bits a weight exactly**. One encoder
      (`llm/quantk.go`) serves both the simulation and the bank, so
      "this is the format L8c-3 measured" is true by construction, and the
      head's 248 320 logits off the real bank are **identical** to the same
      format's through `sim.go`. `-DQ4B` on both `llm_gemm.comp` and
      `llm_gemv.comp`; the record is held in a register per lane and refreshed
      only where the loop crosses a super-block, which is the format's own
      12.5% of record traffic rather than eight times it.
      **The head is 0.675 GB to 0.358, 1.89x at one token at 210 GB/s of a
      242 GB/s bus, and 4.0621 against 4.0289 over the corpus — +0.82%.**
      [Write-up](research/l8c-dense-bank.md) · `results/l8c_head.csv` ·
      `results/l8c_ppl_head_q4k.csv`
- [x] **L8c-5 — the gated DeltaNet, the largest of the four.** 36 layers,
      46% of a dense token, and no new kernel: `-DQ4B`'s MODE 2 arm and its
      **fp16 tail branch**, which nothing had exercised, over the split L8a
      chose for `ssm_alpha` and `ssm_beta` (D13). `tileBQ4K` gained one
      property to allow it — the destination's row count is the *plane's*,
      not the source's — and `DeltaNetWeights` gained its layer index,
      because each of the fused matrix's sources is calibrated under its own
      tensor name, which is what the simulation did. **Decode 25.75 tok/s to
      28.73/29.07, the block 10.9 ms a token to 6.8 (1.60x, 205 GB/s of bank
      to 176), a layer 62.28 MB to 33.28 and 36 of them 2.24 GB to 1.20,
      residency 81.57 GB to 80.53, prefill 1052.8 tok/s to 1070.1 at ubatch
      2048 and 655.2 to 667.0 at 512.**
      [Write-up](research/l8c-dn-bank.md) · `results/l8c_decode_dn.csv` ·
      `results/l8c_graph_dn.csv` · `results/l8c_ppl_dn_q4k.csv` ·
      `results/l8c_ppl_dn_only.csv` · `results/l8c_dn_gemv.csv`
- [x] Gate: **the bank is the format**, twice — a whole layer's output off the
      real bank is identical to the same format through `sim.go`
      (`TestDeltaNetGPUQ4IsTheSim`, 115 360 projection values and 17 920
      output values), and the bank and its simulation agree **chunk for
      chunk** over eight chunks. Perplexity **4.1012 against 4.0289, +1.79%**
      for head and DeltaNet together and **4.0665, +0.93%** for the DeltaNet
      alone. Prefill *faster* at both ubatches rather than merely no slower,
      which is D14's hazard not firing for a reason the stage names.
- [x] **L8c-6 — the hyper-connection block, and a record that is not
      ggml's.** The third family and the one with the obstacle in it:
      `hc_{attn,ffn}_up` and `output_hc_up` read the low-rank space and are
      **320 wide**, and `get_scale_min_k4` *is* eight groups — four carried
      whole and four whose high bits are stolen from the first four's spare
      ones — so there is no ten-group spelling of it. The super-block is the
      whole row, exactly as `asymSubBlocks` has measured it since L8c-3, and
      the record becomes **twenty bytes**: the fp16 pair, then ten twelve-bit
      fields at bit 12*j of a little-endian bit stream (`packScaleMin12`,
      `q4kScaleMin12`, `-DQ4K_SUB=10`). Word alignment costs one byte a
      record — 0.025 bits a weight — so the bank is **4.500 bits** where the
      simulation quotes 4.475. **The fp16 tail stopped being free**: L8b's
      split leaves 32 low-rank rows on the fp16 plane, and on L8a's bank they
      were Q8_0 and the two planes held the same numbers, where at 4.5 bits
      they have to go through the encoder first or the bank is more accurate
      than the format it claims to be. **Decode 28.84 tok/s to 30.05-30.19 — 1.20x
      llama.cpp — the block 5.68 ms a token to 4.16 (1.37x) and 1.50x on its
      own one-token ladder, a mixer 8.12 MB to 4.76, the block 0.79 GB to
      0.47, residency 80.53 GB to 80.20, prefill 669.7 tok/s at ubatch 512
      against 667.2 and 1065.9 at 2048 against 1069.7.**
      [Write-up](research/l8c-hc-bank.md) · `results/l8c_decode_hc.csv` ·
      `results/l8c_graph_hc.csv` · `results/l8c_ppl_hc_only.csv` ·
      `results/l8c_ppl_hc_q4k.csv` · `results/l8c_hc_ladder.csv` ·
      `results/l8c_hc_gemv.csv`
- [x] Gate: **the bank is the format**, twice again — one mixer's every
      tensor off the real bank is identical to the same format through
      `sim.go` on all sixteen GEMM rung pairs (`TestHCGPUQ4IsTheSim`,
      2 688 320 values), and the bank and its simulation agree **chunk for
      chunk** over eight chunks. Perplexity **4.1104 against 4.0289, +2.02%**
      for the three families together and **4.0416, +0.31%** for
      `hyper_conn` alone — where its own eight-chunk screen said **+0.90%**.
      **So the screen is not biased in one direction either**: three families
      now read +0.40 → +0.82, −0.30 → +0.93 and +0.90 → **+0.31**. It does
      not bound a family's cost, does not fix its sign, and does not rank
      families against each other.
- [x] **L8c-7 — the full-attention layer, and the indexer riding along.**
      The fourth family and the one that needed nothing new: every matrix in
      the block is a multiple of 256 wide and `-DQ4B` already had every arm,
      including L8c-5's fp16 tail arm, so the stage is the wiring — and the
      last of the two-valued `q8 bool` spelling (`q8Pipe`, `gemvPipe`,
      `gemmBuilds`) leaves the package with it. **Decode 30.05-30.11 tok/s to
      31.41-31.57 — 1.25x llama.cpp — the block 3.5 ms a token to 2.3, a
      layer 57.94 MB to 32.22, residency 80.20 GB to 79.89, and prefill
      faster at all four ubatches** (671.4 at 512 against 669.7, 1071.8 at
      2048 against 1065.9). **D14's boundary is measurable twice inside one
      block**: the fused [13952, 2560] projection is 35.7 MB at int8 and
      **17.9** at 4.5 bits and changes side of the MALL, where the output
      [2560, 6144] projection is 15.7 and 7.9 and fits at every width — so on
      the block's own ladder the second one is 1.00x at 128 tokens and
      **1.04x slower** at 512 while the block is faster overall.
      **`qsa_indexer` is the fifth family and it is nearly free**: the fp16
      tail costs **0.51 bits a weight** over the whole layer (11.3% of the
      block at 4.5 bits, against 5.7% at int8), putting it on the plane makes
      the layer **4.500 bits exactly**, and it costs **+0.005%** — 0.028 GB a
      token, decode 31.51 tok/s, residency 79.85 GB.
      [Write-up](research/l8c-attn-bank.md) · `results/l8c_decode_attn.csv` ·
      `results/l8c_decode_attn_idx.csv` · `results/l8c_graph_attn.csv` ·
      `results/l8c_ppl_attn_only.csv` · `results/l8c_ppl_attn_q4k.csv` ·
      `results/l8c_ppl_attn_idx.csv` · `results/l8c_ppl_attn_c2560.csv` ·
      `results/l8c_ppl_idx_c2560.csv` · `results/l8c_attn_gemv.csv`
- [x] Gate: **the bank is the format**, three ways this time. One layer's
      fused projection (97 664 values) and its output (17 920) are identical
      to the same format through `sim.go` on **both** arrangements of the tail
      (`TestAttnGPUQ4IsTheSim`); the bank and its simulation agree chunk for
      chunk over eight chunks, `nll` to six decimals; and
      `TestAttnGPUQ4Gemv` checks the one path neither of those reaches — the
      split-K GEMV at one row, which has to derive the fused projection's tail
      split from `pc.lowRank`, with the tail's columns compared on their own
      because they are 4.6% of the width and would hide. Perplexity
      **4.0954 against 4.0289, +1.65%** for `full_attn` alone and **4.1787,
      +3.72%** for the four families together — against a sum of four separate
      corpus deltas of **3.71%**, so additivity holds to 0.01 pp.
- [x] **D13's last open clause is closed, and it took two contexts.**
      `qsa_indexer` at n_ctx 2048 is **bit-identical over all 145 chunks** —
      `top_k + ratio - 1` is 2051, the selection names every cell, the score
      is discarded — which is L8c-1's argument measured as an equality. At
      **n_ctx 2560**, where `selWidth` is 2051 of 2560 and the selection is
      dispatched, it is 4.0723 to **4.0725 over 116 chunks, +0.005%**, one
      part in forty of a ±0.0231 standard error, with **all 116** per-chunk
      rows differing — so the narrower indexer selects different cells and the
      corpus cannot tell.
- [ ] **`ple_proj`, the last family** *(P2)*, and the smallest thing in the model:
      0.033 B parameters run **once**, at layer 1, for 0.035 GB of a token.
      `ple_key` and `ple_value` are both 2560 wide, so there is no obstacle —
      but `PLEGPU` never got L8a's int8 bank either, so this is two bank
      stages in one and worth 0.017 GB a token.
      §1.1's W4A8 stays open beside this: it is a *symmetric* layout and an
      int8-activation one, where every rung here multiplies an fp16 A —
      the asymmetric version folds the min into the epilogue as a per-group
      correction times the activation's column sum.
- [x] **The per-family attribution is eight-chunk, and it does not fix the
      sign or the ranking.** `lm_head` screened at +0.40% and measures
      **+0.82%**; `deltanet` screened at **−0.30%** — better than the
      checkpoint — and measures **+0.93%**; `hyper_conn` screened at +0.90%
      and measures **+0.31%**, so the third one is *pessimistic* where the
      first two were optimistic; **L8c-7 adds `full_attn` at +1.05% → +1.65%**,
      optimistic again. Every bank reproduces its own screen to four
      decimal places, so the gap is the screen and not the bank. The plan's
      own +4.24% is a corpus number and stands; the rows it was assembled from
      do not, and **the next family's width has to be called from a 145-chunk
      run**. What *is* safe is adding corpus numbers: 0.82 + 0.93 = 1.75
      against a measured 1.79, the three together sum to 2.06 against a
      measured **2.02**, and the four sum to 3.71 against a measured
      **3.72** — so compounding across four asymmetric families is 0.01 pp,
      well inside a single side's ±0.024. **And the pp-per-GB ranking is
      bimodal rather than monotone in anything**: `deltanet` 0.87,
      `hyper_conn` 0.93, `lm_head` 2.59, `full_attn` **5.52**.
- [ ] Then the two the simulation cannot reach *(P4 — after P1's
      attribution, which is worth more than both together)*: **the router to fp16**
      (+1.5 tok/s of ceiling, D3's own line, and L5a's ties are the risk) and
      **the 512 expert banks at ~4.25** (+3.0 tok/s, D4's floor, and they are
      the *calibrated* part of the checkpoint so naive re-quantisation is the
      likeliest way to lose — though **L8c-3 makes that the reason to
      expect them to behave rather than to fear them**, since Q4_K
      *is* the form calibration works on). Everything
      L8a and L8b stage is still **the checkpoint's arithmetic**: 8.5 bits a
      weight, its own levels, its own scales. D3's ~4.5 bits is where the
      next 1.9x of the dense half is, and it is the first step in this
      vertical that changes what the model computes.
- [ ] Re-quantise (transcode from the GGUF, or from bf16 with unsloth's
      published imatrix) into the §1.1 W4A8 layout with an L0c scale plane
      *(parked at the 2026-09-18 review — the GEMM arm's epilogue cost is
      unmeasured and the bank is no longer where the decode time is)* —
      **asymmetric now, per L8c-3, so the plane carries a min beside the
      scale and the epilogue carries the column-sum correction.**
      **If from bf16: apply llama.cpp's V-head reorder first** (L3a-3) — seven
      tensor families per linear layer, or `kHeadOfV` is wrong on 32 of 48
      heads in 36 of 48 layers. ~~**Gated on L8c-2**: at L8c-1's widths it is
      not worth building.~~ **Ungated at L8c-3**: at the asymmetric form's
      widths it is. Note still that §1.1's W4A8 is an *int8-activation*
      format, where every L8c-1 and L8c-3 rung was measured with an fp16 A
      operand — so its accuracy is not either table's (see the open
      question).
- [ ] Gate: perplexity within a stated delta of **4.0289** — our own number on
      the bank we run, not the reference's 4.0340, since L8c-0 shows the two
      already differ by −0.13% at identical weights — on the same corpus,
      context and chunking, and the same text at temperature zero for as long
      as L7c's is. **A short `-chunks N` run is a screen for broken and not
      for a percent**: four chunks put us +0.32% above the reference where 145
      put us 0.13% below, and L8c-1's fp16-tail result read −0.53% at eight
      chunks and **+0.01%** over the corpus.
      **The tok/s and residency half of this gate is L8c-3's to set, and it
      has.** ~67 tok/s and ≤67 GB came off D3 as written; L8c-1 retired that
      and left 52.9 tok/s and ~73 GB; L8c-3's uniform asymmetric plan is
      **56.7 tok/s of ceiling, 4.264 GB a token and 80.5 GB resident, at
      4.1998 — +4.24% of our own number** (or 54.8, 4.419, 80.6 and +2.70% at
      4.75 bits). That is the delta the kernel is built to reproduce. The
      residency barely moves because the dense half is 5.5 GB of 82.52 and
      the experts are 77.02 of it; **≤67 GB was always an expert number**,
      and it stays with the expert item below.

### L9 — phase 3, and shipping

- [x] **L9a — the HTTP API `GOALS.md` asks for.** The chat template
      transcribed from the checkpoint's own metadata and checked against Jinja
      on 23 cases, the decoder that splits a token stream into reasoning,
      answer and tool calls, and three envelopes over one generation loop:
      `/v1/chat/completions`, `/v1/responses` and `/v1/messages`, buffered and
      streamed, with stop sequences on the same holdback and a prefix reuse
      that makes a second turn a continuation. `go run ./cmd/serve -llm`. See
      `API.md`.
- [ ] MTP speculative decoding (the separate GGUF). Target 1.5-1.8x. *(P5 —
      the rollback design doc comes first: recurrent state checkpointing,
      ring rewind, KV truncate, the host id list, and the M = 2-8
      verification-kernel decision D15 currently refuses. See `LLM2.md`.)*
- [ ] Batching — which is now also "more than one conversation at a time",
      since the served graph is one sequence's. *(P6 — gated on the product
      question; each sequence owns 113 MB of DeltaNet state.)*
- [ ] Vision tower, if wanted.

---

## How to run things

    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
    L=/home/kube/repos/llama.cpp

    # ours
    go run ./cmd/gguf $M                      # the inventory and the decode budget
    go run ./cmd/gguf -tensors -kv $M         # every tensor, every metadata key
    go run ./cmd/llm -tokenize 'hello world'
    go run ./cmd/llm -gen -model $M -n 64 -prompt 'The capital of France is'
    go run ./cmd/llm -gen -model $M -n 64 -top 3      # the alternatives it beat
    go run ./cmd/llm -gen -model $M -n 128 -temp 0.7 -top-p 0.8 -seed 1
    go run ./cmd/llm -gen -model $M -n 16 -layers 4   # the loop, in 7 GB
    go run ./cmd/llm -model $M -chat-template        # the checkpoint's own Jinja

    # the server (API.md): the same loop behind three chat envelopes
    go run ./cmd/serve -llm                          # 33.5 s to stage, ~84 GB
    go run ./cmd/serve -llm -llm-layers 4            # the HTTP side, in 7 GB
    curl -sH 'Authorization: Bearer womblesofwimbledon' -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":48}' \
        http://127.0.0.1:8080/v1/chat/completions
    # P1: where a decode token goes, per dispatch label and per host phase.
    # The labels are the ones each block already builds for its error
    # messages; `-attrib` is what reads them. With `-csv` the file *is* the
    # attribution table rather than the one-row summary, because the two do
    # not fit in one shape.
    go run ./cmd/llm -gen -model $M -n 64 -prompt 'The capital of France is' \
        -attrib -csv results/p1_decode_attrib.csv
    go run ./cmd/llm -gen -model $M -n 16 -layers 4 -attrib   # the table, in 7 GB
    # P1a: the hyper-connection boundary fused. `LLM_HC_NOFUSE=1` is the
    # control — the scatter and the norm as two dispatches — and the two arms
    # differ in 94 dispatches a pass and in nothing else, which is what the
    # two equalities below pin.
    LLM_HC_NOFUSE=1 go run ./cmd/llm -gen -model $M -n 64 \
        -prompt 'The capital of France is' -attrib \
        -csv results/p1a_decode_attrib_nofuse.csv
    go test ./llm/ -run TestHCFusion -v   # the kernel, then four layers of
                                          # the graph, both to the last place
    go test ./llm/ -run TestPLEGatherPoolIsTheSerialGather -v
                                             # P1-2: the gather's worker pool
                                             # against one row at a time, at
                                             # the row counts where a stride
                                             # is wrong
    go run ./cmd/llm -gen -model $M -n 64 -prompt 'The capital of France is' \
        -csv results/l8e_decode.csv          # every committed decode CSV is
                                             # this prompt; the default is
                                             # `graphDefaultPrompt`, 81 tokens,
                                             # on which the model emits EOS at
                                             # the first step
    go run ./cmd/llm -hc -model $M -tokens 512          # L2c's block, against L2a's table
    go run ./cmd/llm -hc -model $M -tokens 1 -ladder    # L7d: the decode rungs,
                                             # and the 4 KB rule they measure
    go run ./cmd/llm -hc -model $M -ladder -csv results/l2c_hc.csv
    go run ./cmd/llm -ple -model $M -csv results/l2d_ple.csv   # L2d's block
    go run ./cmd/llm -head -model $M                   # L8a/L8c-4: all three
                                       # dense banks plus the simulation of
                                       # the third, every rung, the decode
                                       # GEMV, and the logits compared
    go run ./cmd/llm -head -model $M -csv results/l8c_head.csv

    # L8c-4 and L8c-5: the **real** 4.5-bit bank, not a simulation of one. Same grammar
    # as LLM_DENSE_SIM, and the two refuse to run together — a weight
    # quantised twice is neither format. The mode defaults to `imatrix` here
    # where a simulation defaults to `rtn`, because a bank is built at the arm
    # L8c-3 recommends. `lm_head` and `deltanet` have kernels; the other
    # three families are still the checkpoint's own width.
    LLM_DENSE_BANK=lm_head=q4_k/32 go run ./cmd/llm -ppl -model $M -chunks 8
    # L8c-5 adds the second family. `deltanet` is 36 layers and the largest
    # of the five; the two names together are what results/l8c_*_dn*.csv is.
    LLM_DENSE_BANK=lm_head=q4_k/32,deltanet=q4_k/32 go run ./cmd/llm -gen \
        -model $M -n 64 -prompt 'The capital of France is' \
        -csv results/l8c_decode_dn.csv
    LLM_DENSE_BANK=lm_head=q4_k/32,deltanet=q4_k/32 go run ./cmd/llm -graph \
        -model $M -tokens 128,512,1024,2048 -csv results/l8c_graph_dn.csv
    LLM_DENSE_BANK=deltanet=q4_k/32 go run ./cmd/llm -ppl -model $M \
        -csv results/l8c_ppl_dn_only.csv      # the family's own corpus number
    LLM_DENSE_BANK=deltanet=q4_k/32 go run ./cmd/llm -dn -model $M \
        -tokens 1 -gemm-ladder -csv results/l8c_dn_gemv.csv   # D12's re-run,
                                       # which D16 says is an L3 measurement
    # L8c-6 adds the third family. Its up projection is 320 wide, so the bank
    # picks a ten-group super-block and a twenty-byte record off `k` alone —
    # nothing on the command line says so, and `-DQ4K_SUB=10` is what the
    # four up rungs were compiled with.
    LLM_DENSE_BANK=lm_head=q4_k/32,deltanet=q4_k/32,hyper_conn=q4_k/32 \
        go run ./cmd/llm -gen -model $M -n 64 -prompt 'The capital of France is' \
        -csv results/l8c_decode_hc.csv
    LLM_DENSE_BANK=lm_head=q4_k/32,deltanet=q4_k/32,hyper_conn=q4_k/32 \
        go run ./cmd/llm -graph -model $M -tokens 128,512,1024,2048 \
        -csv results/l8c_graph_hc.csv
    LLM_DENSE_BANK=hyper_conn=q4_k/32 go run ./cmd/llm -ppl -model $M \
        -csv results/l8c_ppl_hc_only.csv      # the family's own corpus number
    LLM_DENSE_BANK=hyper_conn=q4_k/32 go run ./cmd/llm -hc -model $M \
        -tokens 1 -ladder -mixers 24 -csv results/l8c_hc_gemv.csv
    LLM_DENSE_BANK=hyper_conn=q4_k/32 go run ./cmd/llm -hc -model $M \
        -tokens 64,128,512,1024,2048 -ladder -mixers 24 \
        -csv results/l8c_hc_ladder.csv        # and the same two without the
                                       # variable for the int8 control
    # L8c-7 adds the fourth family and, beside it, a fifth that is the fused
    # projection's fp16 tail. `full_attn` is the query, the key, the value
    # and the output projection; `qsa_indexer` moves the indexer's two BF16
    # matrices onto the same plane, and it needs `full_attn` there too
    # because the two share it. Everything else is unchanged — every matrix
    # in this block is a multiple of 256 wide.
    LLM_DENSE_BANK=lm_head=q4_k/32,deltanet=q4_k/32,hyper_conn=q4_k/32,full_attn=q4_k/32 \
        go run ./cmd/llm -gen -model $M -n 64 -prompt 'The capital of France is' \
        -csv results/l8c_decode_attn.csv
    LLM_DENSE_BANK=lm_head=q4_k/32,deltanet=q4_k/32,hyper_conn=q4_k/32,full_attn=q4_k/32 \
        go run ./cmd/llm -graph -model $M -tokens 128,512,1024,2048 \
        -csv results/l8c_graph_attn.csv
    LLM_DENSE_BANK=full_attn=q4_k/32 go run ./cmd/llm -ppl -model $M \
        -csv results/l8c_ppl_attn_only.csv    # the family's own corpus number
    LLM_DENSE_BANK=full_attn=q4_k/32 go run ./cmd/llm -attn -model $M \
        -tokens 1 -gemm-ladder -layers 2 -csv results/l8c_attn_gemv.csv
    LLM_DENSE_BANK=full_attn=q4_k/32 go run ./cmd/llm -attn -model $M \
        -tokens 1,64,128,512,1024,2048 -layers 2   # and the same without the
                                       # variable for the int8 control
    # `qsa_indexer` is the one family n_ctx 2048 cannot grade, because
    # `top_k + ratio - 1` is 2051 and the selection is the identity below
    # that (L8c-1). So its two runs are at **4096**, where it bites — a
    # different corpus chunking, so the delta is the pair and not 4.0289.
    LLM_DENSE_BANK=full_attn=q4_k/32 go run ./cmd/llm -ppl -model $M \
        -ctx 4096 -csv results/l8c_ppl_attn_c4096.csv
    LLM_DENSE_BANK=full_attn=q4_k/32,qsa_indexer=q4_k/32 go run ./cmd/llm -ppl \
        -model $M -ctx 4096 -csv results/l8c_ppl_idx_c4096.csv
    LLM_DENSE_SIM=full_attn=q4_k/32 LLM_DENSE_SIM_SRC=q8 \
        LLM_DENSE_SIM_QUANT=imatrix go run ./cmd/llm -ppl -model $M -chunks 8
    go test ./llm/ -run TestAttnGPUQ4 -v         # the layer on both arms and
                                       # both arrangements of the tail, the
                                       # decode GEMV against the GEMM, and
                                       # the bank's size
    # the simulation of exactly that bank. SRC=q8 excludes `hc_*_inject`,
    # which is F32 in the checkpoint and stays in the fp16 tail. Agrees chunk
    # for chunk over eight chunks.
    LLM_DENSE_SIM=hyper_conn=q4_k/32 LLM_DENSE_SIM_SRC=q8 \
        LLM_DENSE_SIM_QUANT=imatrix go run ./cmd/llm -ppl -model $M -chunks 8
    go test ./llm/ -run TestHCGPUQ4 -v           # the mixer on both arms over
                                       # sixteen rung pairs, the GEMV's one
                                       # rounding, and the bank's size
    go test ./llm/ -run TestQ4KRecordPackings -v  # both record packings
                                       # against their own readers
    # the simulation of exactly that bank — SRC=q8 is what excludes the two
    # F32 matrices the bank leaves in its fp16 tail, without which the two
    # arms quantise different sets of weights. Agrees chunk for chunk.
    LLM_DENSE_SIM=lm_head=q4_k/32,deltanet=q4_k/32 LLM_DENSE_SIM_SRC=q8 \
        LLM_DENSE_SIM_QUANT=imatrix go run ./cmd/llm -ppl -model $M -chunks 8
    go test ./llm/ -run TestDeltaNetGPUQ4 -v     # the layer on both arms,
                                       # value for value, and the bank's size
    LLM_DENSE_BANK=lm_head=q4_k/32 go run ./cmd/llm -ppl -model $M \
        -csv results/l8c_ppl_head_q4k.csv
    LLM_DENSE_BANK=lm_head=q4_k/32 go run ./cmd/llm -gen -model $M -n 64 \
        -prompt 'The capital of France is'
    LLM_DENSE_BANK_QUANT=rtn LLM_DENSE_BANK=lm_head=q4_k/32 go run ./cmd/llm -ppl ...
    go test ./llm/ -run TestBankQ4KIsTheSim -v   # the bank against the
                                       # simulation, value for value, over a
                                       # row permutation
    go test ./llm/ -run TestHeadGPUQ4 -v         # the same on the device, the
                                       # GEMV against the GEMM, and the bits

    # L8c-0: perplexity, llama.cpp's protocol over our graph. 145 chunks,
    # 36.8 s of staging and 7m58s; -chunks N screens for broken, not for a
    # percent. -head-rows is the logit arena, 0.99 MB a row.
    go run ./cmd/llm -ppl -model $M -csv results/l8c_ppl.csv
    go run ./cmd/llm -ppl -model $M -chunks 4        # the four-chunk screen
    go run ./cmd/llm -ppl -model $M -chunks 2 -layers 4   # the plumbing, in 8 GB

    # L8c-1: a candidate bank's accuracy, before there is a kernel that reads
    # it. Model.F32 round-trips a streamed dense weight through the candidate
    # format on its way to the device, and the half it stages is the half the
    # kernel would form (llm/sim.go) — so this is the format's own perplexity
    # and not a model of it. A sim forces the fp16 arm, so it says nothing
    # about tok/s; `cmd/gguf -width` is the bytes half.
    LLM_DENSE_SIM=q4_0/32 go run ./cmd/llm -ppl -model $M -chunks 8
    LLM_DENSE_SIM=deltanet=q4_0/32,hyper_conn=q6sym/32 go run ./cmd/llm -ppl -model $M
    LLM_DENSE_SIM_FAMILIES=deltanet go run ./cmd/llm -ppl ...   # one family
    LLM_DENSE_SIM_SRC=tail go run ./cmd/llm -ppl ...            # L8a-2's split
    go test ./llm/ -run TestQuantSim -v        # the simulator against q8sym/32,
                                               # where it has to be an identity

    # L8c-2: how the levels and the scale are chosen. `rtn` is ggml's
    # uncalibrated path and every L8c-1 rung; `search` is make_qx_quants
    # alone; `imatrix` is quantize_row_q4_0_impl; `+gain` replaces the
    # search's scale with the unbiased one.
    LLM_DENSE_SIM_QUANT=imatrix LLM_DENSE_SIM=q4_0/32 go run ./cmd/llm -ppl -chunks 8
    LLM_DENSE_SIM_QUANT=imatrix+gain LLM_DENSE_SIM=hyper_conn=q4_0/32 go run ./cmd/llm -ppl
    go test ./llm/ -run TestImatrixQuant -v    # the three arms on real weights
    go test ./llm/ -run TestQuantSimGain -v    # gain against residual, per arm
                                               # and per *form* since L8c-3
    go test ./llm/ -run TestImatrixSkew -v     # why hc_attn_up is the outlier

    # L8c-3: the same three arms on ggml's asymmetric form, which is the same
    # 4.500 bits as q4_0/32 — a super-block of eight groups of 32, each with a
    # 6-bit scale and min against one fp16 pair. `q5_k/32` is 5.500, beside
    # `q5sym/32`. The 320-wide hyper-connection rows take a ten-group
    # super-block (4.475 bits), because ggml's K-quants want k % 256 == 0 and
    # that is the one family the question is about.
    LLM_DENSE_SIM=q4_k/32 LLM_DENSE_SIM_QUANT=imatrix go run ./cmd/llm -ppl -chunks 8
    LLM_DENSE_SIM=hyper_conn=q5_k/32 go run ./cmd/llm -ppl -chunks 8
    go test ./llm/ -run TestQuantSimAsym -v    # the super-block rule, the
                                               # levels, and the 320-wide row

    # the oracle for the port, on dequant_ref.c's precedent. Both testdata
    # files are committed, so the test needs no toolchain.
    L=/home/kube/repos/llama.cpp
    gcc -O2 -o /tmp/quant_ref reference/quant_ref.c \
        -I$L/ggml/include -L$L/build/bin -lggml-base -Wl,-rpath,$L/build/bin
    go test ./llm/ -run TestQuantSimMatchesGGML -updatequant
    /tmp/quant_ref llm/testdata/quant_in.bin llm/testdata/quant_ref.bin
    go test ./llm/ -run TestQuantSimMatchesGGML -v

    # L8c-1's other half, which is arithmetic rather than a measurement:
    # what a candidate width does to a decode token and to the ceiling.
    go run ./cmd/gguf -width 4.5 $M
    go run ./cmd/gguf -width 4.5 -expert-width 4.25 -router-width 16 $M

    # L8a's control. The dense banks are the checkpoint's int8 by default;
    # this puts every block back on the halves it staged before L8, which is
    # how the two are compared against one dump, one trace and one prompt.
    LLM_DENSE_FP16=1 go run ./cmd/llm -gen -model $M -n 64 -prompt 'The capital of France is'
    LLM_DENSE_FP16=1 go test ./llm/ -run TestDeltaNetGPU -v

    # L8d's control, on the same precedent. Every one-token kernel L8d added —
    # the MoE's two expert GEMVs, its split-K router, the dense GEMV under the
    # DeltaNet's two projections — goes back to the cooperative-matrix GEMM.
    # It is the only way to run the two over one prompt, because a block picks
    # its rung from the batch and not from a flag at the call site, and it is
    # what reproduces L7c's completion token for token.
    # Since L8e the two produce the *same* 128 tokens, which is how the text
    # gate is now read: diff the two bodies rather than eyeball them.
    LLM_DECODE_GEMM=1 go run ./cmd/llm -gen -model $M -n 128 -prompt 'The capital of France is'
    LLM_DECODE_GEMM=1 go run ./cmd/llm -gen -model $M -n 64 -csv results/l8d_decode_gemm.csv
    go run ./cmd/llm -attn -model $M                    # L2f's layer, 64..2048
    go run ./cmd/llm -attn -model $M -tokens 512 -ladder
    go run ./cmd/llm -attn -model $M -ladder -csv results/l2f_attn.csv
    go run ./cmd/llm -attn -model $M -tokens 1 -layers 12 -gemm-ladder \
        -csv results/l8e_attn.csv                       # L8e: the decode GEMV's
                                             # split, per projection, and the
                                             # GEMM control beside it. Twelve
                                             # layers, because two put the
                                             # output projection's 16.7 MB in
                                             # the MALL (D16).
    go run ./cmd/llm -attn -model $M -tokens 512,2048 -sel on   # L4b: price the
    go run ./cmd/llm -attn -model $M -tokens 4096 -ctx 4096     # selection, then
    go run ./cmd/llm -attn -model $M -tokens 4096 -ctx 4096 -sel off  # the control
    go run ./cmd/llm -dn -model $M -tokens 512                 # L3b's layer
    go run ./cmd/llm -dn -model $M -ladder -csv results/l3b_dn.csv
    go run ./cmd/llm -dn -model $M -tokens 512,2048 -gemm-ladder
    go run ./cmd/llm -moe -model $M -tokens 1 -ladder      # L7d's narrow-N rungs,
                                             # L8d's six GEMV rungs for both
                                             # modes, and the router's ladder
    go run ./cmd/llm -moe -model $M -tokens 1 -ladder -csv results/l8e_moe.csv
                                             # L8e adds the shared expert's
                                             # two marginals to that cross
    go run ./cmd/llm -moe -model $M -tokens 1 -ladder -layers 16 -iters 1 \
        -csv results/p1b_moe.csv             # P1b: the same cross against
                                             # DRAM — sixteen cold banks and
                                             # one iteration, so ProfileSweep
                                             # never re-reads a bank warm.
                                             # Rows from the two-layer form
                                             # above read past the bus (D16)
    go run ./cmd/llm -dn -model $M -tokens 1 -gemm-ladder  # L8d: the dense
                                             # GEMV's split, per projection,
                                             # and the GEMM control beside it
    go run ./cmd/llm -moe -model $M -tokens 512,2048              # L5b's block
    go run ./cmd/llm -moe -model $M -tokens 512,2048 -ladder -iters 5 \
        -csv results/l5b_moe_ladder.csv
    go run ./cmd/llm -resident -model $M                      # L6a: the whole model
    go run ./cmd/llm -resident -model $M -dense               # the 6.8 GB half alone
    go run ./cmd/llm -resident -model $M -bank 48,24,12,4,2 \
        -csv results/l6a_resident.csv                         # residency, priced
    go run ./cmd/llm -graph -model $M -tokens 512             # L6b: the model
    go run ./cmd/llm -graph -model $M -tokens 128,512,1024,2048 \
        -csv results/l7d_graph.csv                            # the prefill ladder
    go run ./cmd/llm -graph -model $M -tokens 512 -layers 4    # the shape, in 7 GB
    # P0: the row counts that used to hang, and the two environment variables
    # that found out why. A submit may hold the gfx ring for ~2 s before
    # amdgpu resets it, so the recorder chunks a pass by *time* (batchFor);
    # LLM_MAX_BATCH caps that from above and LLM_BATCH_TIMES prints each
    # submit's own GPU time, which is what the 2 s was measured against.
    go run ./cmd/llm -graph -model $M -tokens 2048,2560,3072,3584,4096 \
        -csv results/p0_graph_4096.csv
    go run ./cmd/llm -graph -model $M -tokens 8192 -ctx 8192 \
        -csv results/p0_graph_8192.csv    # 1213.5 tok/s, and the selection biting
    LLM_BATCH_TIMES=1 go run ./cmd/llm -graph -model $M -tokens 2816  # 2.033 s, reset
    LLM_MAX_BATCH=256 LLM_BATCH_TIMES=1 go run ./cmd/llm -graph -model $M -tokens 3072
    LLM_DISPATCH_DUMP=1010:1030 go run ./cmd/llm -graph -model $M -tokens 3072
                                             # the block and grid of each
                                             # dispatch, printed *before* the
                                             # submit, which is the only moment
                                             # one that kills the ring can
                                             # still be named
    go test ./llm/ -run TestBatchForIsUnderTheWatchdog -v   # the fit, and the bound
    journalctl -k | grep -A4 'ring gfx_0.0.0 timeout'       # what a reset looks like
    LLM_ARENA_UNCACHED=1 go run ./cmd/llm -graph -model $M -tokens 512
                                             # L6b-4's control: the arenas back
                                             # on the write-combined type

    # the reference implementation, the oracle, the baseline
    less $L/src/models/qwen4exp.cpp                  # 1279 lines, the whole architecture
    $L/build/bin/llama-bench -m $M -p 512,2048 -n 128 -r 2
    $L/build/bin/llama-completion -m $M -n 128 --temp 0 -no-cnv \
        -p 'The capital of France is'         # L7c's gate, the other side
    $L/build/bin/llama-eval-callback -m $M -p 'hello' -n 1   # per-tensor dumps
    $L/build/bin/llama-perplexity -m $M -f models/wikitext-2-raw/wiki.test.raw -c 2048 -b 2048
    $L/build/bin/llama-perplexity -m $M -f models/wikitext-2-raw/wiki.test.raw \
        -c 2048 -b 2048 --chunks 4           # L8c-0's per-chunk comparison
    $L/build/bin/llama-tokenize -m $M -f models/wikitext-2-raw/wiki.test.raw \
        --ids --no-parse-special             # 297 193 ids, `cmp`-identical to ours

    # L2a's tool: a timestamp query around every dispatch in the Vulkan graph.
    # Costs 4-7% and prints one per-op-shape table per graph compute.
    GGML_VK_PERF_LOGGER=1 $L/build/bin/llama-bench -m $M -p 2048 -n 16 -r 1 -ub 512

    # L2b's tool: whole intermediate tensors of a real pass, to reference/out/llm.
    # See research/l2b-hyper-connections.md for the build line and the filter.
    /tmp/eval_dump -m $M -o reference/out/llm -c 64 -p '…' -n '<regex>'
    go test ./llm/ -v                        # the block against that trace,
                                             # CPU and GPU, plus the controls
    go test ./llm/ -v -run TestDeltaNet      # L3a: the linear layers
    go test ./llm/ -v -run TestDeltaNetGPU   # L3b: its kernels, the ladder, the carry

    # L4a's fixture: 4096 tokens, 4096 cells, one ubatch, 8.0 GB, ~4 minutes.
    # The headers come from the *build's* commit — the checkout has moved past it.
    H=$(mktemp -d) && (cd $L && git archive cff184438 include ggml/include | tar -x -C $H)
    gcc -O2 -o /tmp/eval_dump reference/eval_dump.c -I$H/include -I$H/ggml/include \
        -L$L/build/bin -lllama -lggml-base -Wl,-rpath,$L/build/bin
    mkdir -p reference/out/llm4k
    head -c 40000 models/wikitext-2-raw/wiki.test.raw | head -n -1 > reference/out/llm4k/prompt.txt
    /tmp/eval_dump -m $M -o reference/out/llm4k -f reference/out/llm4k/prompt.txt \
        -nt 4096 -c 4096 -ub 4096 \
        -n '^(model\.input_embed|ple_embd|hc_init)$|-(0|3)$|^ple_(gate|gated_value|conv_out)-1$'
    go test ./llm/ -v -run TestQSA           # L4a: the selection, at the length it exists
    go test ./llm/ -v -run 'TestAttnGPUSelection|TestAttnGPULayer4k|TestAttnGPUIndexer4k'
                                             # L4b: the same on the device, plus
                                             # the dense control and the bf16 price
    go test ./llm/ -v -run 'TestMatMuls|TestF16Acc|TestIndexerProjection|TestHalfPrecision'
    go test ./llm/ -v -run TestMoE           # L5a: the MoE block, its controls,
                                             # and the routing distribution
    go test ./llm/ -v -run TestMoEGPU        # L5b: the same on the device, the
                                             # permutation, the ladder's
                                             # agreement and the padding control
    go test ./llm/ -v -run TestMoEGPUBankArray  # L6a: the bank index, and the
                                             # control that shows it is read
    go test ./llm/ -v -run TestGraphPrefix   # L6b: four layers against
                                             # `l_last-3`, and the drift at
                                             # depths 1-4, in 7 GB
    go test ./llm/ -v -run TestGraphLogits   # L6b's gate: the whole model,
                                             # the drift every eight layers,
                                             # and llama.cpp's own logits
    go test ./llm/ -v -run TestArenaMemoryTypes  # L6b-4: what a mapped arena
                                             # costs the host, by memory type
    go test ./llm/ -v -run TestAttnGPUCache  # L7a: the layer's cache as a
                                             # chunk split, bit for bit, and
                                             # the incremental block range
    go test ./llm/ -v -run TestGraphIsAChunkSplit
                                             # L7b: the model as a chunk
                                             # split, down to one token, and
                                             # the no-history control
    go test ./llm/ -v -run 'TestArgmax|TestTopN|TestSampler'
                                             # L7c: the sampler
    go test ./llm/ -v -run TestHCGPUGemv     # L7d: the split-K down
                                             # projection against the GEMM
                                             # rung, and the batch it refuses
    go test ./llm/ -v -run TestMoEGPUDecode  # L8d: the expert GEMVs against
                                             # the GEMM and against llama.cpp,
                                             # the router's ladder as a top-k,
                                             # and the one-token refusal
    go test ./llm/ -v -run TestDeltaNetGPUGemvAgrees
                                             # L8d: the dense GEMV against the
                                             # GEMM on both projections, the
                                             # fp16 tail on its own, and the
                                             # rungs that cannot cut K
    go test ./llm/ -v -run TestAttnGPUGemvAgrees
                                             # L8e: the same on the attention
                                             # layer, with the indexer's BF16
                                             # tail compared on its own
    go test ./llm/ -v -run TestMoEGPUSharedPlan
                                             # L8e: the shared expert's own
                                             # two rungs, the row-space
                                             # alignment they widen, and the
                                             # batch they refuse
    go test ./llm/ -v -run TestHCGPUQ8       # L8b: the block on both banks,
                                             # fifteen rungs, every tensor
                                             # identical to the last bit
    go test ./llm/ -v -run TestGraphForwardRows
                                             # L8c-0: row t of a batch against
                                             # the prompt truncated at t, and
                                             # the head's slab loop
    go test ./llm/ -v -run TestMove          # L6c: the arena-to-arena move,
                                             # value for value against the
                                             # host narrowing it replaced

    # L6b's depth trace: `l_last` every eight layers of the same prompt.
    /tmp/eval_dump -m $M -o reference/out/llmdepth -c 64 \
        -p 'The capital of France is Paris.' \
        -n '^l_last-(7|15|23|31|39|47)$|^result_(norm|output)$'

    # re-fetch (resumable, checks sizes)
    reference/fetch_llm_checkpoint.sh

## Decisions

| # | decision | why |
|---|---|---|
| D1 | **Consume UD-Q4_K_XL first, don't quantise from bf16.** | 114 GB instead of 360, an imatrix-calibrated checkpoint, and a bit-exact oracle. Phase order: run it, profile it, then optimise (`TODO.md`). |
| D2 | **The n-gram table lives off-heap, mmap'd.** | 28.80 GB of capacity for 1.41 KB/token of bandwidth. **L1: llama.cpp already does this** — `TENSOR_READ_LAZY` — so it is the reference behaviour, not a deviation. |
| D3 | **Target ~4.5 bits on everything *streamed*, not just the experts.** ~~Retired at L8c-1~~ — **restored at L8c-3, in a different format.** | The premise was always right: 76% of decode bytes are dense. The *width* came off L0d's reconstruction ladder, and measured end to end in **Q4_0's symmetric form** it was +18.5% of perplexity, which is what retired it — and what replaced it was a plan, not a width (L8c-1's mixed table, +5.98% at 5.30 bits). **L8c-3 found the width was not the thing that was wrong; the form was.** At the same 4.500 bits, ggml's asymmetric `q4_k` with unsloth's published imatrix is **4.1998 against our 4.0289, +4.24%**, and a mixed plan at 4.75 bits is **4.1377, +2.70%** — better accuracy than the 5.30-bit symmetric plan at **4.264 GB a token against 4.575** and a **56.7 tok/s ceiling against 52.9**. L0d's finding 2 still stands where it was measured, on weight reconstruction under int8 activations; what L8c-1 established is that it does not carry to perplexity, and what L8c-3 adds is that it does not even rank the *formats* correctly — the reconstruction gap between the two is 1.22-1.30x where the corpus-level gap is **2.06x uncalibrated and 3.69x calibrated**, at identical bits and identical bytes. **A plan is still not the sum of its families** (four families sum to +2.05% where the uniform plan measures +3.92% at eight chunks), but the compounding is much milder than the symmetric form's 11.7% into 18.5%. |
| D4 | **Do not go below 4 bits on the experts.** | They are 24% of the traffic; Q3 buys ~11% for a real accuracy hit. |
| D6 | **W4A8, with per-token activation scales.** | L0d: int8-per-token costs **1.13x** the error of fp16 activations, against a 3x throughput cliff. Per-token is free in the RMSNorm epilogue (§3.1). |
| D7 | ~~**Symmetric Q4, not asymmetric.**~~ **Reversed at L8c-3: a 4-bit rung here is `q4_k`, ggml's asymmetric K-quant, and not `q4_0`.** | L0d decided it on weight reconstruction: asymmetric was 1.043-1.053x at equal bits where §7 predicted 2x, so the min was not worth its bytes. **Two things were wrong with that.** The bytes are not a trade at all — ggml nests the min, a super-block of eight groups of 32 carrying one fp16 pair and 12 bits a group, which is **0.500 bits a weight, exactly what a symmetric fp16 scale per 32 costs** — so `q4_k/32` and `q4_0/32` are both 4.500. And the metric was the proxy L8c-1 had already retired: measured over the whole corpus the asymmetric form is **2.06x cheaper at identical bits and identical bytes** (4.3124 against 4.6127, +7.04% against +14.49%), 3.2x on the family that decides the plan. **The larger half is L8c-2's lever, which works on this form and not on the other**: the imatrix takes 4.5 bits on every streamed dense family to **4.1998, +4.24%**, where the same matrix on the symmetric form makes it *worse* — 4.6588, +15.63% — because a symmetric group has one free parameter and it *is* the gain, so a calibrated fit can only express its preference by shrinking the whole group — measured, the extra shrinkage calibration costs `hc_attn_up` is 1.24 pp symmetric and **0.47 pp** asymmetric. The 1.13x `q4_0` gained over L0d's fifteen-level `q4sym` (L8c-1) still stands and is now a fact about the arm that lost. |
| D8 | **fp16 scales per 32 nibbles, in a k-major plane.** | L0c made the fine block cost 1.0% instead of 14.2% at prefill. The remaining cost is decode bytes — 11.1% of the bank against 3.0% at per-128 — the one axis still worth trading if tok/s falls short. |
| D5 | **`llama.cpp` is the oracle, not a Python dump.** | It is built, it is Vulkan, it supports `qwen4exp`, and L1 got correctness, accuracy and a baseline out of one binary. |
| D9 | **`MADV_RANDOM` on the n-gram table, and on nothing else — and the sixteen reads go in parallel.** | L7c-3: sixteen scattered 90-byte reads a token draw sixteen 128 KB readahead windows — 2 MB to deliver 1.41 KB — and removing it is **176x** at decode. Set on that tensor's pages alone, on the first gather, because the rest of the shard is read once, sequentially, and wants the readahead. **P1-2 is the other half.** With the readahead gone each read is still a *fault* — the table is 28.80 GB and D2 keeps it mmap'd beside 84 GB of resident model — and `PLEGather` served them one at a time: **16.3 major faults a step, 853.2/819.4 us**. Issuing them concurrently leaves the fault count identical to two decimal places (16.34/16.30) at **122.2/108.8 us, 7.0-7.5x**, because what was being measured was sixteen device latencies end to end rather than any amount of data. In the whole model, 2.244 → 0.902 ms a token and **31.51 → 32.72 tok/s**. A random-access mapping wants both halves: small faults, and several of them in flight. |
| D11 | **At one token, read a dispatch's shape off its grid before anything else — including the part of the grid that does nothing.** | L7d: three kernels were 23-71 GB/s for one reason — 7, 10 and 100 workgroups on a 40-CU device — and two of them had been attributed to padding and to arithmetic. Fix it by splitting **K** where N is the output (the down projection, the router, both dense projections) or by narrowing **BN** where N is wide enough to cut (the MoE, the combine); both keep the staged weight exactly as it is. **L8d found it five more times and added the corollary**: a *static upper bound* on a grid is a dispatch too. The MoE's tile grid was 2052 records where eleven exist, 82 080 workgroups to run 11 at about 0.5 ns each, and it multiplies with every narrowing of BN — so it has to be tightened on the same axis the kernel narrows, or the kernel measures as a catastrophe (481 us against a GEMM's 84, of which 330 was empty launches). **P1a measures the law the rule was inferred from.** The hyper-connection down projection's decode rungs split K six ways over identical bytes off an identical bank, so the ladder varies nothing but the grid: **168 workgroups is 9.26 us, 336 is 6.47, 672 is 5.54**, and past 672 it is flat — a 1.67x band between "too few" and "enough" on this 40-CU device, entered somewhere around 300-600. The corollary is a *diagnostic*: `up_m1` reads 11.5 KB a workgroup in **160** workgroups and takes 10.87 us, which is `down_gemv8`'s 11.8 KB in 168 taking 9.26 — a different mode of a different kernel landing on the same number because it is launched at the same width. So when a one-token dispatch measures low, compare it against another dispatch's rung at *its* workgroup count before attributing it to the kernel: P1 had read this one as MODE 1's M=1 waste, and its four rungs vary `BM` from 16 to 128 while all four launch 160 workgroups. |
| D15 | **At one token a block runs a different *kernel*, not a different rung, and the host refuses it above one token.** | L8d: `llm_moe_gemv.comp`, `llm_moe_router.comp` and `llm_gemv.comp` are the decode pipeline — no LDS slab, no barrier in the K loop, no cooperative matrix — and they are 1.87-3.06x on the two blocks that dominate a step while being *slower* at prefill, where the GEMM's fragment is full. The MoE's GEMV reads one row of a tile, which is every real row a tile has at one token and not at two, so `SetPlan`, `Upload` and `Resize` refuse it rather than drop rows; `Graph.PinSchedule` is what lets a test ask for the prefill kernels at any length. **§2.2's rule survives intact and was over-read**: nibbles cannot be unpacked into a cooperative-matrix *fragment*, and a dot product has no fragment. **L8e-1 adds the fifth block**, the full-attention layer, on the same kernel and the same refusal. |
| D12 | **A split's stride must miss the 4 KB rotation, and the rung to pick moves when the weight's width does.** | L7d-2: the split-K ladder is 181/154/219/129/138/230 GB/s and the fast rungs are exactly the two whose slab stride is not a multiple of 4 KB. §5.1b's law applies to the distance between two *workgroups*' addresses, not only to a matrix's leading dimension (§2.3). **L8b-3 re-ran it on the same kernel over int8**, where a slab is half the bytes: the ladder is 231/269/326/135/193/308 and the winner is 32 rather than 160 — the same two strides, one rung along. A ladder measured on one bank does not carry to another. **L8d-5 shows it does not carry to another *matrix* either** — k8 wins at K = 2560 and k32 at K = 6144 on one kernel over one bank — and **L8d-3 shows where it stops applying**: the 2.63 MB router never leaves the MALL, so its KSLABS ladder is flat within 0.6 us. A rule about DRAM channels says nothing about a weight that does not reach DRAM. **L8c-6 re-ran the hyper-connection ladder a third time, on nibbles, and the rung did not move — but the *formulation* does not survive it.** A slab is `(gemmK/16/KSLABS) * 128` bytes there, so **no** rung is a whole multiple of 4 KB (2.5, 1.25, 0.625, 0.5, 0.25, 0.125 of it) and L7d's "whole multiple" test cannot separate them — yet the spread is still 1.65x, the two slowest are 2.5 and 0.5 and the two fastest 0.625 and 0.125. What the two quantised banks share is only that the winner's slab is 5/8 of a 4 KB multiple on both (5120 bytes at int8, 2560 at 4.5 bits), which is why the incumbent rung stayed; that is written down rather than promoted, because six rungs on one kernel cannot distinguish it from magnitude. The operative half of D12 — **re-measure when the width changes** — is unaffected and is what caught it. |
| D16 | **A micro-bench rung whose measured rate is above the 242 GB/s bus is not a DRAM measurement; confirm it in the whole model before believing it.** | L8e-2: `-moe -tokens 1 -ladder` stages two layers and re-runs one dispatch twenty times, and a token's ten routed experts are ~32 MB — **the MALL exactly** — so every repetition after the first reads at 805-965 GB/s. The ladder then measures kernel shape against an L3 hit and picks the rung with the most parallelism rather than the best DRAM locality: v16w4 at **288 GB/s** for the routed up mode, which is **42.1 ms worse over 64 tokens** in the whole model where each expert is read once. This is D12 widened a fourth time — a ladder does not carry across a bank (L7d-2), a width (L8b-3), a matrix (L8d-5) or a **residency** — and the last one is a property of the harness rather than of the kernel. The test is on the face of the number: the fused attention projection's winner reads 39.5 MB in 174.3 us (227 GB/s, sound), its output projection's 16.7 MB in 24.6 (679, not). |
| D13 | **A dense weight is staged in the checkpoint's own width, never expanded.** | L8a: 8.5 bits a weight against 16 is 1.61x of a decode token's dense half and **costs nothing in accuracy**, because ggml's `d = amax/127` makes the round trip an identity. The three families the checkpoint does not ship as Q8_0 keep their halves in a tail rather than being re-quantised early — that is L8c's decision to make, with a perplexity number beside it. **L8b holds the rule where it costs something**: `inject` does not begin on a column block, so keeping it in the tail means staging 32 low-rank rows twice, and the answer is to pay the 0.33 MB a mixer rather than to re-quantise four rows early. **L8c-1 makes the deferred decision and the tail loses**: `ssm_alpha`, `ssm_beta` and `inject` as int8 with a scale per 32 are **4.0294 against 4.0289, +0.01%** over the whole corpus — five times inside the error bar — so the two-plane machinery can go. It is worth **0.030 GB a token, 0.6%**, so the reason is deleting `lowRank`/`gateOff` from two kernels and L8b-1's doubled rows, not speed. ~~The indexer's two BF16 projections rode along untested, because nothing at `-c 2048` reads them (L8c-1's finding 2).~~ **L8c-7 tests them and closes the clause.** At n_ctx 2048 a 4.5-bit indexer is **bit-identical over all 145 chunks** — the per-chunk `nll` columns match to six decimals — which is L8c-1's argument measured as an equality rather than argued; at n_ctx 2560, where `selWidth` is 2051 of 2560 and the selection is dispatched, it is **+0.005%** over 116 chunks with all 116 rows differing. So the last family the checkpoint does not ship as Q8_0 is **0.028 GB a token for one part in forty of a standard error**, and the rule's exception list is empty: nothing in the dense half is staged wider than it has to be. |
| D14 | **At prefill the bank is not the kernel; at decode it is — *where the weight fits the MALL*.** | L8b-5: the hyper-connection block's two projections are 2.3-2.5x faster on int8 at one token and **1.09-1.15x slower at ubatch 2048**, because at 2048 each weight is read 32 times out of a 32 MiB MALL and the DRAM bytes were already hidden — what is left is the unpack's ALU, 256/WM conversions a matrix step. A narrower bank is a decode decision, and a block that is asked to serve both needs a *ladder* per bank rather than a kernel per bank. **L8c-4 found the counter-example and it names the condition**: the lm head's B is 357 MB at 4.5 bits and never fits the MALL at any width, so there is nothing for the re-reads to hide behind and the bank is the kernel at *every* length — 18839 us to **17461** at 512 rows, 1.08x, in the direction this rule warns about. The unpack does show up there (82 GB/s of bank against the Q8 arm's 143); it just does not win. So the rule is about a weight that fits the MALL, and where one does not, a narrower bank is faster at both ends. **L8c-5 is the third case and it names the boundary rather than the two sides of it**: the gated DeltaNet's fused [16512, 2560] projection is 84.5 MB as halves, 44.9 at int8 and **23.8 at 4.5 bits**, so it *changes side* — sixteen re-reads a graph at ubatch 2048 come off DRAM — and the block is 1.09-1.11x **faster** at prefill (350.7 ms to 320.5 at 2048, 96.5 to 86.7 at 512). Three blocks, three answers: the hyper-connection block's weights fit at every width and it was slower, the head's fit at none and it was faster, this one crosses and it is faster. **L8c-6 takes the first block a rung further and the sign holds**: at 4.5 bits the same weights still fit at every width, and the block is **1.02x slower at ubatch 2048** — the hazard fires a second time, and it is the smallest it has been (1.019x against int8's 1.09-1.15x against halves, 6 ms of a 1921 ms pass, with the other three ubatches *faster*). So the rule keeps its condition and gains a magnitude: a bank that never leaves the MALL pays only its own unpack, and each halving of the width costs less of it than the last. **L8c-7 is the fourth block and the first that contains *both* answers**: the full-attention layer's fused [13952, 2560] projection is 35.7 MB at int8 and **17.9** at 4.5 bits, so it crosses and is 1.13x at ubatch 2048, while its output [2560, 6144] projection is 15.7 and 7.9 and fits at every width, so it pays only the unpack and is **1.00x at 128 tokens and 1.04x slower at 512** on the block's own ladder. The block is faster at all four ubatches and the whole graph comes along, so the hazard is real, local and outvoted — which is the shape to expect wherever one block holds a weight on each side. |
| D17 | **An accuracy delta is stated against our own number, not against the oracle's.** | L8c-0: at *identical* weights our perplexity is **4.0289** and llama.cpp's **4.0340** — −0.13%, a fifth of either side's standard error, and in the direction L4a-5 predicts, since the reference accumulates every quantised matmul in fp16 above 8 output columns where ours accumulate in f32. That gap is settled, is not the bank, and would be silently charged to the re-quantisation by a delta measured from 4.0340. The reference's number stays as the sanity check that the two implementations are the same model; the *stage's* gate is a delta from ours. |
| D10 | **A value that models a memory format goes through memory.** | L7a-4: `float(float16_t(x))` in a register is folded to `x` by RADV's NIR, so L2e's fp16 key cache had never run on the GPU. If a kernel is reproducing a *storage* rounding, the value has to be stored. |

## Open questions

- ~~**Why does the whole-model graph never return above ~2560 rows?**~~
  **Answered at P0, and it was none of the things it looked like.** It is not
  residency, not the arenas, not a width and not any dispatch: it is **GPU time
  in one submit**, and amdgpu's gfx ring watchdog kills a command buffer that
  holds the ring past **2.0 s**. `journalctl -k` has carried the evidence all
  along — `ring gfx_0.0.0 timeout` then `Starting gfx_0.0.0 ring reset` —
  once per stalled run, L8c-7's own sessions included. The reset
  **force-signals the fence**, which is why `vkQueueSubmit` returned, why
  `vkWaitForFences` returned `VK_SUCCESS` and why its 20-second timeout never
  fired: the wait it would have bounded was already over. What spun was six
  lines further on, `vkGetQueryPoolResults` with `VK_QUERY_RESULT_WAIT_BIT`,
  which on RADV is an unbounded **userspace** poll of a timestamp the killed
  dispatches never wrote. `[syscall]` in the old goroutine dump was Go's label
  for *a cgo call*, not for a kernel one, and the thread's own accounting says
  so: `utime=5645` against `stime=5`, with GTT, VRAM and RSS flat for the whole
  hang. The cliff is a duration and nothing else — 1024 dispatches is 1.886 s
  at 2560 rows and runs, 1.964 s at 2688 and runs, **2.033 s at 2816 and is
  reset** — and the control is exact: the same 1273 dispatches in *one* submit
  are 787 ms at 512 rows and fine, and reset at 2688 rows, where two submits of
  1024 and 249 had just run the same work. So the recorder chunks by time
  rather than by count (`batchFor`), and **`-graph` returns at every row count
  tried**: 4096 at **1174.6 tok/s (3.00x)** and 8192 at **1213.5 (3.10x)**,
  which is the best prefill in this vertical and says prefill does **not**
  plateau where llama.cpp's does. [Write-up](research/p0-ring-watchdog.md)
- ~~**Why does the MoE's up mode read 93 GB/s of bank where its down mode reads
  136 on the same grid?**~~ **Answered at L8d-4, and the premise was wrong in
  a useful way**: neither number was the kernel's ceiling, because both modes
  were running a GEMM whose fragment holds one real row. A **GEMV per expert**
  does feed itself from the checkpoint's own Q4_K without the LDS round trip —
  §2.2's rule is about a cooperative-matrix *fragment* and a dot product has
  none — and the two modes go to **236 and 356 GB/s** with the asymmetry gone.
  What the MODE 0 / MODE 1 gap turns into is a *lane-count* asymmetry between
  the routed pair and the shared expert, and that one is about format: Q4_K
  packs 320 payload dwords into a 2560-long row where Q8_0 packs 640, so the
  wider read wants the wider lane group.
- ~~**What is the gated DeltaNet at one token?**~~ **Answered at L8d-5: the
  grid, as D11 said.** The layer is 454.0 us to **242.5**, 119 GB/s to 205 and
  18.9 ms a token to 11.0, from a split-K GEMV over exactly the weight the
  GEMM already staged. The output projection was the worse of the two at 40
  workgroups and is **5.4x**; the fused one at 258 is 1.58x. **And the
  full-attention layer was the same two dispatches** — answered at L8e-1: the
  layer 414.8 us to **225.9**, the block 5.5 ms a token to 3.5, and the fused
  projection at **227 GB/s** of a 242 GB/s bus, which is the first dispatch in
  this vertical to be at the bus rather than near it. **Every projection in
  the model now has its one-token kernel.**
- ~~**Does the Q8 arm want a split-K rung at one token?**~~ **Answered at
  L8b-2, on the one kernel that already had one: yes, and by 2.34x.** The
  hyper-connection down projection's split-K GEMV over int8 tiles is
  **326 GB/s against the fp16 build's 230** — a lane's sixteen consecutive k
  are four words and one scale, which is if anything simpler than sixteen
  halves — and the arithmetic is identical because the multiply is done in
  fp16 rather than in f32-then-converted. **L8d-5 gave the other projections
  that rung** — `llm_gemv.comp` is the same kernel with the hyper-connection
  epilogue removed — and the DeltaNet's two are 1.58x and 5.43x. **L8e-1 is
  the last pair**: the attention layer's fused projection is 1.51x and its
  output projection 5.08x, on the same two rungs the DeltaNet's pair took.
- **Does the asymmetric epilogue cost what it looks like it costs?** L8c-3
  settles the *format* and leaves its kernel an open shape. Dequantising a
  K-quant group is `d*sc*l - dmin*m`, and the subtraction is not free in a
  cooperative-matrix GEMM: it is a rank-one correction, `-dmin*m` times the
  column sum of A over the group, so it comes out of the matrix core and into
  an epilogue that needs one reduction over A per group of 32. At decode that
  reduction is over one row and is nothing; at prefill it is over BM rows and
  competes with the unpack that L8b-5 already found is what a cached bank is
  bound by. `llm_moe_gemv.comp` does the affine form per lane today and is at
  236-356 GB/s, which says the *GEMV* arm is fine and says nothing about the
  GEMM arm. Nobody has measured the GEMM one.
- **What does the unpack cost when the bytes are already cached?** L8b-5 is
  the first measurement in this vertical where a *narrower* bank is **slower**:
  at ubatch 2048 the hyper-connection block's two projections are 1.09-1.15x
  the fp16 arm's, because each weight is read 32 times out of a 32 MiB MALL
  and only the unpack's ALU is left — 256/WM conversions a matrix step, of
  which the biggest single component is a `bitfieldExtract`, a convert, a
  multiply and a 16-bit LDS store per weight. It is the same instruction
  sequence L5b-7 found the MoE's grouped GEMM 2.5x off its byte floor on, and
  the same thing neither has tried: **issuing the loads a K-step ahead of the
  unpack that consumes them**. Worth ~1.1x at prefill on three blocks, and
  possibly much more on the MoE — **L8d's third lead**.
- **How much is the fp16 tail costing now that the plane around it is
  nibbles?** L8b-1 settled the *shape* — the split is the column block that
  contains `inject`, 288 of 336, and the 32 low-rank rows in between are
  staged in both planes — and priced it at 0.33 MB a mixer, 0.5% of a step.
  L8c-6 does not change any of that and changes what it is a fraction **of**:
  the tail is 48 rows of 10240 halves, 0.98 MB, which was 12% of a staged
  mixer at int8 and is **21%** at 4.5 bits, and it is now the largest single
  piece of overhead in the block. **L8c-7 answers the same question for the
  other block that has a tail, and there the answer was to delete it**: the
  full-attention layer's 640 BF16 rows of 13952 cost **0.51 bits a weight**
  over the whole layer — 5.7% of it at int8 and **11.3%** at 4.5 bits — and
  putting them on the plane costs **+0.005%** of perplexity and makes the
  layer 4.500 bits exactly. That does not carry to the hyper-connection
  block's tail, whose four rows are the F32 `inject` and whose 32 shared
  low-rank rows are a *layout* problem rather than a width one; what it does
  say is that the question is worth asking of a tail before assuming it is
  cheap. A 16-column tail would need the branch to
  be per tile in one workgroup of seven rather than per workgroup — L8a-3's
  1.27x applied to 14% of the dispatch rather than to all of it — and nobody
  has measured that shape. What has changed since L8b-1 wrote that down is
  that the prize has doubled.
- ~~**What is the hyper-connection block's inject row doing to L8b?**~~
  **Answered at L8b-1, and by none of the three options it listed.** The
  branch stays per workgroup, inject stays out of the int8 plane and stays
  fused into the same dispatch: the split is simply moved to the column block
  that *contains* it — `lowRank` rounded down to BN, 288 — and the 32
  low-rank rows in between are staged in both planes. **0.33 MB a mixer, 32 MB
  a decode token, 0.5% of a step.** What is still open is only whether that
  0.5% is worth a narrower tail: a 16-column tail would need the branch to be
  per tile in one workgroup of seven, which is L8a-3's 1.27x applied to 14% of
  the dispatch rather than to all of it, and nobody has measured that shape.
- **L8c's bf16 path has to replicate a head permutation, or it is a different
  model.** L3a-3 and vLLM between them: llama.cpp's converter reorders the 48
  V heads from HF's grouped order into tiled order — `in_proj_qkv`'s V rows,
  `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, `conv1d`'s V
  channels and `out_proj`'s columns, all seven together — so that `h % 16`
  recovers the right key head. Phase 2's stated alternative is to re-quantise
  "from the 360 GB bf16 (clean — and unsloth publish their imatrix)", and that
  source is in **HF's** order, where `h % 16` is wrong on 32 of the 48 heads.
  It would be wrong in **36 of the 48 layers** with nothing but perplexity to
  notice. Either replicate `_reorder_v_heads` or switch `kHeadOfV` to divide;
  never half of each. Transcoding the GGUF (D1's default) is unaffected.
- ~~**Does the DeltaNet kernel want the scan or the chunk?**~~ **Answered at
  L3b, by pricing rather than by building.** The scan, at the rung this
  hardware wants, is **413 us a layer against the reference's 438** — and it is
  **13% of the layer**, where the fused input projection is 64%. The chunked
  form is 1.15x the arithmetic on the matrix cores: a 4.3x ceiling at §2.7's
  best measured rate, 1.1-1.6x at an honest one for 64x64x128 tiles with a
  serial triangular solve in the middle, and a *perfect* version saves 1.0% of
  the prefill graph while putting a recurrent state through fp16 operands.
  **At decode the question dissolves rather than changing**: a chunk is a
  token, so a chunked kernel *is* the scan, and L7c measured the DeltaNet at
  1.7x its own bytes — the closest of the three big blocks to the bus and not
  where L7d's 4x is.
- **Why is llama.cpp's own scan shape 6.3x better in its kernel than in
  ours?** L3b-3 ran the reference's decomposition — one state column per
  workgroup, 6144 workgroups, q and k re-read 128 times — and got 2766 us
  where llama.cpp's own kernel does 438 at what should be the same work. The
  operand traffic is most of ours (deleting it: 2766 → 1013 us) and the
  subgroup reductions are not (5%), but 1013 is still 2.3x the reference's
  whole dispatch. Ours wins at a different rung and by more than this costs,
  so it is written down rather than chased — but it is the one place in this
  vertical where the reference does something we cannot reproduce.

- ~~**Why is prefill 393 and not ~2000?**~~ **Answered by L2a**: neither
  candidate. It is the hyper-connection block's glue and its `[10240, 4]` F32
  `inject` projection, plus a MoE kernel 2.53x off ours. See L2a above.
- ~~**How much of the 30.6% glue actually fuses?**~~ **L2c answers it for the
  first block** (4.24x on the nameable lines, the optimistic end of L2a's
  711-to-1159 spread), **L2f for the second** (2.40x, 1.60x at equal attention
  work) **and L3b for the linear layers** (1.37x on the layer, 3.90x on the
  gated norm alone). All three dense blocks are in and all three land inside
  that spread. **L5b closes the last of it**: the MoE's own glue — the GLU,
  the two weight multiplies, the shared gate's sigmoid and the two-half add —
  is fused into an accumulator, a store and a read that had to happen anyway,
  and the block is 1.09x at ubatch 512 and 1.55x at 2048. All five blocks are
  in, and the whole nameable graph is 87.2% replaced by 61.4%.
- ~~**How does a grouped MoE kernel schedule a 26x bucket imbalance?**~~
  **Answered at L5b: with a list, not a grid, and by padding.** The tiles are
  one device-built record per (expert, row block) that has rows, over a static
  upper bound with an early return — and every expert's range is rounded up to
  the row block so the epilogue can store whole cooperative-matrix fragments.
  **§2.2's 2.53x does not survive the real distribution**: it was swept at a
  uniform split, and at the measured one the block is **1.09x at ubatch 512
  and 1.55x at 2048**, because 274 experts read 505 MB of bank to serve 5120
  rows and the arithmetic intensity is the routing's rather than the tile's.
  The padding it costs is 2.33x the rows at 512 and 1.90x at 2048, and it is
  nearly free time, because a cold expert's tile is reading 0.9 MB of Q4_K to
  multiply seventeen rows either way.
- **Why is the grouped GEMM 2.5x off its own byte floor?** **L8d's first
  question**, and the largest number left in the vertical: at decode the same
  block is 56 GB/s where the lm head is 194, which is 42% of a token. L5b-7: `up` moves
  686 MB of quantised bank in 7.2 ms, 70 GB/s of a 236 GB/s bus, with 78
  GFLOP of matrix work that would take 2.0 ms at §2.7's best rate and could
  hide entirely under the loads. What sits in between is the unpack — 1.22 G
  weights a dispatch, each a shift, a mask, a convert, an fma and a 16-bit LDS
  store, issued by the same wave that is waiting on the load. Three things
  that should have closed it did not (L5b-4): more waves a workgroup, a wider
  K-step and a conflict-free LDS pad are 1.15-1.24x, 1.16x and **5.4x**
  against respectively. What has not been tried is issuing the loads a K-step
  ahead of the unpack that consumes them.
- **Would a float atomic delete the combine?** L5b's down projection stores
  whole fragments in permutation order because a scattered store would need an
  LDS round trip, so a token's eleven contributions are summed by a ninth
  dispatch — 22.1 ms a graph at ubatch 512 and 66.4 at 2048, reading a 58 MB
  tensor that exists for no other reason. `VK_EXT_shader_atomic_float` would
  delete both, at the cost of a summation order that is no longer the
  reference's — which L5a's tie analysis is the reason to be careful about.
  The extension is not enabled in `vk.DeviceFeatures` today.
- **Does the hot expert want to be treated as a second shared expert?** At
  ubatch 512 expert 454 is chosen by 95% of tokens and ranked *first* by 54%.
  It has the same shape as the shared expert, which runs for 100%. Running the
  two together as one dense pair and routing only the other nine would delete
  a gather for half the block's tokens — and would be wrong on the 5% that do
  not select it, so it is a fast path with a correction rather than a
  simplification. **Still unmeasured after L5b**, which treats it as one more
  group of the grouped GEMM like any other — sixteen tiles of it at ubatch 512
  against one for each of the other 273.
- **Should our kernels follow the reference onto an fp16 accumulator — part
  two.** L5a-3 is the sharpest data point yet: at the MoE, modelling the
  reference's arithmetic makes the fit **worse** (0.57-0.73x), because the
  operand error and the accumulator error are independent. So an f32
  accumulator is strictly nearer the model here and there is nothing to
  reproduce — which leaves the question purely one of throughput, and L2f-5
  and L3b-4 both found register pressure binding.
- **Should the indexer's two projections leave the fused matrix?** They are
  the only BF16 weights in the model, and L2f-3 made them fp16 column ranges
  of the layer's one [13952, 2560] matrix — which is most of why the
  projection is 1.53x. **L4b-4 prices what that costs**: the device cannot
  take L4a-6's bf16 activation, so `indexer_k_raw` sits at L4a-6's own
  *unmodelled* 9.3e-04 rather than its modelled 4.5e-07, the score is 24x
  further from the reference and the selection 17x. The layer's output does
  not currently care (L4b-5, 9.87e-04 rms at 4 k against 1.14e-03 at 7
  tokens); whether the *model* does is L8c's perplexity run. Un-fusing them
  would cost a dispatch — llama.cpp spends 5.03 ms a graph on those two lines.
- **The select kernel is 4-6x off the bus, and it is 0.1% of a graph.** Three
  of its four radix passes walk every cell to test a prefix almost none of
  them match; a compaction after pass 1 would fix that, and so would a
  **block-granular** histogram, since all `ratio` cells of a whole block carry
  one score and a 4096-cell row is 1024 distinct values plus a split tail.
  Written down rather than done, on L2f-5's precedent.
- **QSA becomes an optimisation at decode, and only there.** L4b-3 measured
  the selection *costing* the attention kernel 1.21x at prefill, because
  `build_attn_qsa` runs dense flash attention over a mask and so do we, and a
  32-cell key block is 8 indexer blocks — at ~50% density the chance that all
  eight are unselected is ~0.4%, so nothing can be skipped. At decode one
  token reads 2051 of up to 262 144 cells, the bitmask is 32 KB, and skipping
  unselected key blocks is the whole point. L7's.
- **Do the two projection ladders generalise?** L2f-4 found the same kernel
  wanting opposite BM schedules on two weights that differ only in falling
  either side of the 32 MiB MALL — widest-wins at 71.4 MB, narrowest-wins at
  31.5. **L3b-7 checks the DeltaNet's two and the rule holds**: the 84.5 MB
  fused projection wants the widest rung by 2.6x at 512 tokens and 3.3x at
  2048, and `ssm_out` at 31.5 MB picks the narrow rung at 512 and the wide one
  at 2048 — the same rungs at the same lengths as `attn_output`, which is the
  same shape. Still unchecked: the MoE bank and the lm_head.
- **The indexer's score is the one line we lose on**, 0.8 ms against
  llama.cpp's 0.7 (and 1.4 against 0.7 at a 4096-cell cache, which is twice
  the blocks). It is a scalar workgroup per token at 4.1 TFLOP/s where the
  matrix cores do 39, and it is an obvious cooperative-matrix rewrite — but it
  is 0.07% of a prefill graph, so it is written down rather than done.
- **Should our kernels follow the reference onto an fp16 accumulator?** L4a-5
  says the reference does, everywhere a weight is quantised, and that it costs
  it ~0.44% relative on every projection at prefill. Ours accumulate in f32 and
  are therefore *nearer the model and further from the oracle* — which is
  fine for a tensor comparison and is an open question for **throughput**:
  an fp16 accumulator halves accumulator register pressure, which L2f-5 and
  L3b-4 both found to be the binding constraint on the ladder. Worth a rung on
  the attention and projection ladders before L6 freezes them. The accuracy
  half is L8c's: **`PPL 4.0340` was measured with fp16 accumulation on**, so
  matching it does not require avoiding it.
- **The rotary table wants a row per cache cell**, which is right for prefill
  and wrong for a 262144-cell context (67 MB). L7 either builds it per batch,
  as llama.cpp does, or goes back to computing the angle per lane and pays for
  a float32 `pow` at a large position.
- **Should the residual be fp16?** L2c measured the MALL cliff on our own
  kernel (575 → 167 GB/s on the norm between 512 and 1024 tokens), **L2f found
  it again on a different kernel** (the attention pack, 379 → 172 GB/s across
  the same boundary), and it is worth ~1.5x on this block at a long ubatch. But the residual accumulates
  across 97 combines, so it is L6's decision and L8c's perplexity run, not a
  kernel's.
- ~~**Q8_0 weights in the block's kernels.**~~ **Done at L8b**: a mixer is
  8.12 MB staged and 7.63 MB read against 13.43, the block is 2.14x at one
  token, and every tensor it produces is identical to the fp16 bank's. What is
  left of this line is the PLE block, which L2d raised it for — its fused
  key/value projection is still halves, and it runs **once**, at layer 1, for
  0.6% of a decode step.
- **How many other kernels have the grid the wrong way round?** L2d found
  2.9-4.0x on a convolution and 1.26x on a reduction by swapping which grid
  axis is fast, so that the resident workgroups cover one token's row instead
  of one channel block of forty tokens. Nothing else in this vertical has been
  checked for it, and §5.1b says the penalty is a property of the traversal
  rather than of the kernel.
- ~~**When does the selection start to bite?**~~ **Answered at L4a, by the
  dump vLLM's formula was waiting for.** At token **2051** — the width — and
  the shape is vLLM's: `(i+1) mod ratio` tail cells, **exactly 512** whole
  blocks, and a split of at most `ratio-1` cells off the lowest-scoring block
  in the selection. It is `topk_radix_select.comp` rather than a sort, ours
  reproduces it **set for set on all 2045 biting rows**, the reference's
  *order* is irreproducible run to run while its visible set is stable, and
  **0.0022% of it moves** when the scores are our own. What is left is only
  the kernel, which is **L4b**: 1.94x the reference's two ops, exact against
  the CPU selection on all 4096 rows, 0.0381% off llama.cpp's — and a
  selection that costs the attention kernel 1.21x rather than saving it
  anything. [Write-up](research/l4b-qsa-gpu.md)
- ~~**The recurrent state.**~~ **L3a-6 settles the DeltaNet half and L3b-6
  does it on the device**: both the [128, 128, 48] state and the convolution's
  three-column window carry, and 3 + 4 tokens reproduce 7 bit-identically —
  in Go and in the kernel — with a control showing the carry matters. On the
  device the window is not a second tensor at all: it is `Conv-1` rows of
  *negative token index* in front of the fused projection's own output. L2d's PLE convolution reaches 9 tokens back and is still only zeros
  at position zero; wiring both into a ring buffer is L7.
- ~~**What else is the backend computing in fp16?**~~ **L3a-5 answers it with
  a rule**: `ggml_vk_mul_mat` takes the f32 *vector* path at up to
  `mul_mat_vec_max_cols = 8` output columns and the fp16 coopmat GEMM above
  it. So the answer is "everything wider than 8 columns" — which is every
  projection in the model at a real ubatch, and **none of them in the 7-token
  dump**. ~~What is now open is the consequence.~~ **L4a measured it**, and it
  is worse than "an fp16 operand": on the other side of the threshold every
  quantised matmul accumulates in **fp16**, so the reference sits ~5.4e-03 rms
  from the f32 model and no operand model of ours can get closer. The BF16
  indexer projections take a **bf16** activation there, which is modelled and
  worth 2061x. **Every 1e-6 tolerance in L2b, L2d, L2e and L3a is a fact about
  a 7-token dump**; at prefill the number is 5e-3 and the gap is the oracle's.
  What is still open is only whether the same holds for the *elementwise*
  lines, which have no threshold and were never measured at width.
- **The oracle's build is pinned, the checkout is not, and they have
  diverged.** `/home/kube/repos/llama.cpp` is at **`d1d3c3396`** while
  `build/bin` is still **`cff184438`**, so reading `src/models/qwen4exp.cpp`
  today reads a *different implementation* from the one the oracle runs — it
  already carries #28068's GDN fix and a rms_norm fusion. `include/llama.h`
  and `ggml/include/ggml.h` have moved too, so **L4a builds `eval_dump.c`
  against headers extracted from the build's own commit** (`git archive
  cff184438 include ggml/include`) rather than from the working tree, and
  every source citation should be `git show cff184438:…`. L3a-4 found that
  `cff184438` — the build behind the trace, `pp2048 388.60`, `tg128 25.15` and
  **`PPL 4.0340`** — predates llama.cpp's fix to `build_gdn_l2_norm` (#28068),
  and that reproducing the pre-fix `max(|x|, eps)` is worth 15x on a DeltaNet
  layer. `TestDeltaNetL2NormIsTheBuilds` fails loudly if the build moves. The
  open part: **a rebuild invalidates the trace and the baseline together**, so
  regenerating one means regenerating both, and L8c's perplexity comparison
  has to be against a number from the same binary as the checkpoint it grades.
- **Why is `attn_pregate` 1.03e-04 where everything around it is 1e-06?** Its
  inputs are all modelled and rounding the query to fp16 moves it by 3%. What
  is left is the flash-attention kernel's own accumulation order, which the
  dump enables by default. Bounded, not explained — the same category as
  L2b's three residual mixers.
- **Where does llama.cpp's missing 1.5 s at `-ub 2048` go?** The GPU graph is
  4.556 s and the wall clock 6.083 s, against ≤7% host time at `-ub 512`. It
  does not scale with graph count. Not ours to fix, but it means the honest
  baseline is the `-ub 512` one.
- **Does 111.32 GB — the whole GGUF, n-gram table included — hold live?** 80
  GiB is demonstrably residable (L0a) and 105 GiB reservable (L0b); L1 ran at
  77 GiB with 39 GiB of page cache beside it. If the whole thing fits, D2
  becomes an efficiency choice rather than a necessity and the host-side PLE
  gather can be deferred past L2.
- **Why do three of the seven hyper-connection mixers keep a residual?** Four
  reproduce llama.cpp's gate to 6.2e-08 rms under `RefQ8`; `blk.0.hc_attn` and
  `blk.2.hc_attn` sit at ~2e-05 and `blk.3.hc_ffn` at 3.4e-04, with identical
  code, identical Q8_0 weight types and an `hc_norm` input agreeing to 8e-08.
  An int8 grid amplifying a last-bit accumulation difference fits, but a count
  of near-boundary `lo` values does not track it. Bounded, not explained.
- **What is `down_proj`'s activation error really costing?** L0d settled W4A8
  at 1.13x mean, 1.6x worst, and the worst site is `down`, whose SwiGLU input
  has no norm in front of it. Finer activation groups on that one input, or a
  rotation, are the unmeasured mitigations.
- **Fix `quantizeQ8`'s subnormal scale** — two lines (floor it at fp16's
  smallest normal), but it changes the CPU reference every W8A8 correctness
  check is built from. Becomes urgent if phase 2 transcodes UD-Q4_K_XL's Q8_0
  dense tensors through it.
- **Does the scale plane's layout matter at *decode* too?** L0c is all GEMM.
  §1.8 measured the grouped GEMV paying 1.045-1.116x on the same axis and put
  it down to bytes alone, but it has not been tried with a k-major plane.
- ~~**How much of the Q8_0 dense allocation is actually needed?**~~
  **Answered at L8c-1, and the answer is "most of it".** Unsloth put every
  dense tensor at Q8_0 and the assumption here was that ~4.25 bits would do;
  measured, that is **+18.5%** of perplexity, and the best mixed plan is
  **+5.98%** for 1.38x the bytes. The choice was deliberate, not lazy. What is
  now open is the *next* question rather than this one — **does calibration
  move it** — which is L8c-2.
- **Does the width question change under int8 activations?** L8c-1 measured
  every rung with an **fp16 A operand**, because that is what our kernels
  multiply; L0d's whole ladder assumes **int8 activations** and reports a
  2.865e-02 activation floor that our numbers do not sit on. The two disagree
  about which end is binding: L0d says above ~5 bits the activations are the
  floor and so weight bits are wasted, and L8c-1 says the hyper-connection
  block still wants 6.5 weight bits with no activation error at all. If W4A8
  is ever built, its accuracy is **not** L8c-1's table plus a constant — the
  errors are near-independent (L0d finding 2) and compound in quadrature, over
  48 depths. Nobody has measured a W4A8 rung end to end on this model.
- **`§3.4 finding 4` says prefill is "98% memory-bound by weight bytes";
  §2.2's direct measurement of a Q4 block says 2.2x above its memory floor.**
  L2a settles which is nearer: at 2048 tokens the expert bank is 64 GB, a
  **273 ms** floor, and §2.2's kernel takes 642 — so **2.4x above the bus**,
  and §3.4's "98% memory-bound" is the one to retire. The model-level estimate
  built on it was 5x optimistic; L2a's bottom-up 1768 ms replaces it.
- **How many other rungs in this repo were chosen against the MALL?** D16
  caught one, in the block whose fixture happens to touch exactly 32 MB. Every
  `-ladder` here stages a handful of layers and repeats one dispatch, so the
  same question applies to `l5b_moe_ladder.csv`, `l3b_dn.csv` and
  `l2f_attn.csv` — and the cheap screen is already written down: any row whose
  `gbps` exceeds 242 was not measuring DRAM. The prefill rungs are the ones
  worth screening, because a 2048-token ubatch reads its weights 32 times out
  of the MALL for real (D14) and there the residency is the *truth* rather
  than an artefact. **L8c-5 screened one of them and it failed**: every rung
  of `-dn -tokens 1 -gemm-ladder` on the 4.5-bit bank reads 1700-1800 GB/s, so
  `results/l8d_dn.csv`'s decode rungs were chosen against an L3 hit too. They
  happen to be the rungs the new ladder also names, so nothing was acted on —
  but the question is now "which of these ladders has ever measured DRAM", and
  the answer so far is none of the one-token ones. **P1b re-ran the whole MoE
  decode cross against DRAM** — sixteen cold banks, one iteration — and the
  pattern held a third time: every old rate was L3, no ranking changed. So
  the *screen* stays necessary and the mis-choice has yet to actually occur;
  the one D16 caught (routed up, v16w4) was the whole-model number
  disagreeing, and cold DRAM now agrees with the whole model there too. The
  DeltaNet, attention and hyper-connection one-token ladders could be re-run
  the same way (`-layers` high, `-iters 1`) if a decision ever hangs on one.
- **Why is a dispatch 8-22% slower inside a step than alone on a cold bank?**
  P1b's residue. moe.down is 59.3 us alone-and-cold and 72.2 in the model
  (1.22x, the widest); the five MoE weight-streaming labels sum to a **1.33
  ms a token** gap. Two environments differ at once: alone, a dispatch has
  its own fence and a bank read 16 dispatches ago; in the step it sits behind
  barriers in a 1407-dispatch buffer and its bank was last read 4.3 GB of
  traffic ago (TLB and page-table walks are the byte-side suspect, barrier
  drain the buffer-side one). P1c's pre-recorded command buffer touches the
  second suspect, so measure again after it.
- **Batch and speculation.** At batch 4 the dense 76% amortises completely.
  Worth knowing whether the API will ever serve more than one stream before
  optimising the batch-1 path to death.
