# TODO — the state of play

> **Rewritten 2026-09-21.** The per-vertical progress files were
> consolidated into [`research/`](research/README.md) on 2026-09-20 (`LLM.md`,
> `LLM2.md`, `SPEECH.md`, `TTS.md`, `IMAGE.md`, `PIPELINE.md`,
> `EMBEDDING.md`, `IDEAS.md` and the old session-log `TODO.md`) — see the
> file map at the bottom. The root `IMAGE.md` then came back live for one day
> to carry the Qwen-Image-2.1 replacement, and was frozen in turn on
> **2026-09-21** as
> [`research/qimage-vertical.md`](research/qimage-vertical.md). **No live
> progress file remains at the root**: this one is it — what each vertical
> is, what it measures today, and what is open. It is **rewritten**, not
> appended to; a closed item's write-up goes to `research/` and history
> lives in git.
>
> Addresses: `§N.M` (cited from code) resolves in
> [`research/ideas.md`](research/ideas.md); stage letters (L7d, T9, S10,
> Q9b, E7, P6…) resolve in the vertical archives listed at the bottom;
> decisions **D1–D21** are in
> [`research/llm-vertical.md`](research/llm-vertical.md). A code comment
> citing `IMAGE.md` resolves by what follows it: **Q-stages, numbered
> decisions and Q-o numbers** are `research/qimage-vertical.md`, **I-stages**
> are `research/zimage-vertical.md`. `GOALS.md` is the target; `API.md`
> documents the server as it answers today.

## The five verticals, at a glance

All five of `GOALS.md`'s models run end to end in Go on Vulkan, validated
against a reference, and are served by `cmd/serve`. The competition from
here is our own ceiling, not a reference implementation.

| vertical | model | headline, measured | open |
|---|---|---|---|
| text generation | qwen3.8-flash-next (180 B, 6 B active) | decode **36.0 tok/s through `-gen`, 34.2 through the server at every `-llm-batch`** (P16, against 25.8 at the shipped 4096), at **+1.74%** perplexity; prefill **1403.9 tok/s at 8192 rows, 3.59x** (and **1199 through the server** at 4096), still climbing where llama.cpp plateaus; **128 000 cells prefills at 1011 tok/s at ubatch 2048 and 1176 at the served 4096** (P17, against 834 and 900; falloff 0.88x), decode **32.2 tok/s at 128k** and **34.4 at depth zero** (P16/P17), falloff to 128k **0.94x** | batching (P6); the gathered attention at 34% of matrix-core peak with its four bounds eliminated; `hc.cn` at prefill; decode is now fusion at 1-2% a step, the 196 moves the largest |
| speech → text | parakeet-tdt-0.6b-v3 | an 11 s clip in **43 ms — 257x real time**, whole model resident | S10 front end (48% of the pipeline); S9 long clips |
| text → speech | Kokoro-82M | **31 ms for 3.25 s (105x)**, **162 ms for 19.5 s (120x)** — flat per second of audio; the endpoint answers in 59 ms | the vocoder's 20 ms of arithmetic; three small boundaries |
| image generation + editing | Qwen-Image-2.1 | 1024², 40 steps in **1m28.8s**, 31.5 GB resident, native RGBA; streaming previews cost **0.3%**; the fp32 oracle's picture to mean **3.4e-4**. **Edits answer too**: **1m54.2s** on one reference at 1024², 39.4 GB, the oracle's edit to max abs **0.0014** | **parked 2026-09-21** — Q0–Q12 all closed; the 1184²-area ceiling is the one capability left unbuilt |
| embeddings | Qwen3-Embedding-0.6B | a text in **11.5 ms**, the card's similarity matrix to 1.3e-4 over HTTP | E7 batching, worth up to 10x on short texts |

**The server** (`API.md`): one process, one flag per vertical, OpenAI-shaped
(`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/audio/*`,
`/v1/images/*`, `/v1/embeddings`), plus a Wyoming door for Home Assistant
(`-wyoming`, byte-identical audio to the HTTP door). Everything unimplemented
is refused with a reason, never faked.

**Deployment is two machines**: the language model alone on one (~84 GB
resident), the other four verticals on the other (image ~32 GB, the rest
~3 GB together). `-llm` and `-image` do not fit in one 128 GB process, on
purpose — so no footprint quantisation is planned for the small verticals.

**Where the work goes next.** **P17 (2026-09-22): heads on the fragment's
rows.** A 128 000-cell prefill at ubatch 2048 goes **833.5 → 1011.4 tok/s**
(64 000: 896.4 → 1048.6), and decode at depth gains ~2%, exactly
([`research/p17-heads-on-rows.md`](research/p17-heads-on-rows.md)). The QSA
selection is per token and twelve query heads share each kv head, so the
gathered attention now puts a token's heads on the fragment's M axis and reads
that token's 2 051 cells instead of a sixteen-token union of 7 049; the mask
dispatch is gone. P17-2 faults the next prompt chunk's n-gram pages in while
the current one runs (+3% at depth). At the served ubatch 4096, 128k goes
**899.7 → 1175.7 tok/s**. The attention's depth terms are 0.14 ms of a 0.99 ms
prefill token at 128k; what is left of the falloff is the flat floor. **Open
for decode:** the host n-gram gather is ~1 ms of every token (16 dependent
major faults; residency of the 29 GB table is a deployment decision), and
`attn.select` at 128k is 54 µs a layer, of which the first radix pass is 17
(its top 8 bits are the exponent, so its atomics collide) and each other pass 8.
**Long-context prompt processing is closed at
the target.** **P14 (2026-09-22)** takes a 128 000-cell prefill from P13's
**563.6** to **826.0 tok/s at ubatch 2048** and
**946.1 at 8192**, which is the 900 tok/s `GOALS.md` asked for
([`research/p14-prefill-at-depth.md`](research/p14-prefill-at-depth.md)).
Four changes: the QSA selection runs over **block** scores with a per-block
weight instead of the expanded per-cell tensor, which deletes a dispatch and
1.14 GB of arena and is 13.8x on `attn.select`; the attention kernel runs over
a **per-cell gather** — the union of a query tile's rows' selections, compacted
— instead of over the key axis, which is 2.15x on `attn.attn` and needed the
value plane to become cell-major first; the indexer's score moves onto the
**matrix cores**, 2.3x; and `maxStorageBufferRange` gets an **error** instead of
a comment, because exceeding it makes a run *faster* and wrong. Beyond that,
each vertical's list is in its own rough order of value. The context-depth regression that
stood at the head of this list is **closed** — **P7**, **P8**, **P9** and **P10**, all
2026-09-21, take decode at 64k from **4.05 to 27.69 tok/s (6.8x)** and the
falloff from depth zero from **6.9x down to 1.3x down** (0.14x → 0.79x of the
depth-zero rate). P7 was three kernels that walked the *cache* rather than
the *context* plus a QSA selection that was a mask and never a skip
([`research/p7-context-depth.md`](research/p7-context-depth.md)); **P8, P9 and P10
are one finding**
([`research/p8-decode-attention-split.md`](research/p8-decode-attention-split.md))
— **at decode this model's kernels are single waves that all fit on the
device at once, so a dispatch costs one wave's serial walk and neither its
traffic nor its total work.** The probe: cutting the attention grid from 24
workgroups to 2, a twelfth of both, measured **1.00x at every depth**. P8
splits the attention's key axis across workgroups, P9 unpins the indexer —
which was scoring the whole context on one compute unit of forty — and P10
widens the selection, the one kernel that cannot be split at all because its
radix passes are a reduction, from four waves to sixteen.

**P16 (2026-09-22): the wide batch never cost decode anything.** Decode is
**1.16x at every depth** — 29.73 → **34.4 tok/s** at depth zero and 27.20 →
**31.5** at 128 000 cells, ubatch 2048 — and through the server it is **34.2
tok/s at `-llm-batch` 2048, 4096 and 8192 alike**, where P12-7 measured 28.04
and 25.81 ([`research/p16-decode-arena-width.md`](research/p16-decode-arena-width.md)).
The cost P12-7 and P15 priced for a wide batch, and could not explain, was
**`DeltaNetGPU.InPort` and `AttnGPU.InPort` advertising the arena's rows**:
every decode step zero-filled the whole prefill arena of both blocks' A
operand, once a layer, and the qkv projection after each move stalled behind
the writes draining. `cmd/llm -depth -ubatch` found it by staging wide arenas
and prefilling narrow — the cost followed the arena, not the prefill — and the
per-label diff put all of it on `move`, `dn.qkv` and `attn.qkv`. The port now
pads to the row block, as the MoE's always did, and
`TestInPortPadsTheRunNotTheArena` asserts it, because the old padding was
*correct* and no tolerance could see it. Two smaller exact changes ride along:
**`hc.cn` is a workgroup a (token, stream) again at ≤ 64 rows** (25.0 → 7.0 µs,
1.06x of a token — P11's workgroup a token is one workgroup on forty CUs at
decode), and **`attn.select`'s emit walks blocks rather than cells** (68 → 55
µs at 128k). `-llm-batch` stays 4096; 8192 is now purely a memory decision.

**Decode's side of the depth question is now closed too. P15 (2026-09-22)**
takes a decode step at 128 000 cells from **22.58 to 27.20 tok/s** and the
falloff from depth zero from **0.76x to 0.91x**, with prefill and depth-zero
decode flat as the controls
([`research/p15-decode-at-depth.md`](research/p15-decode-at-depth.md)). Three
changes, and the largest was not on the device: **`PLERows` hashed the whole
sequence on every token** to use sixteen of its rows, which at 128k is 8.2 MB
allocated and 2.05 M rows per step — **5.18 ms of a 44.3 ms token and 39% of
the whole falloff**, deleted exactly by `PLERowsFrom`. Then **`attn.score`
was striped sixteen ways on a forty-CU device**: P9 unpinned it and stopped at
16, the stripe is a grid and not a reduction, and 64 is **2.61x** (211.6 → 80.9
µs) and better at *every* depth. And **the gather and the split compose, which
P14 said they would not** — that argument is right about the gather alone and
is measured (unsplit gather 986.7 µs against the split's 419.8), but the two
cut different things: the split is 7.12x on the *walk* and the gather 2.99x on
the *work*, so `GATHER`+`SPLITK` together is **3.23x** (419.8 → 129.9). A
decode tile is one real row, so the union that costs prefill 4.9x costs decode
nothing. It turns on at `2*selWidth` **live** cells, which makes it the first
decode knob that is a function of the depth — so P1c's prerecorded buffer now
carries an epoch and re-records the one step that crosses. The refusal:
**the arenas' HOST_CACHED memory type costs the kernels nothing** even at the
3.85 GB the KV planes now put in that buffer — every kernel within 1% under
`LLM_ARENA_UNCACHED=1` while host glue moves 29x — which confirms L6b at the
new scale and spends a third hypothesis for `hc.cn`. After that the two capability
gaps are P6 batching (blocked on a product question: will the API serve more
than one stream?) and E7's batched embeddings, worth up to 10x on short
texts; the largest single-vertical percent is S10, the speech front end at
48% of its pipeline.

---

## Text generation (archive: [`research/llm-vertical.md`](research/llm-vertical.md), review: [`research/llm-review.md`](research/llm-review.md))

**Where it stands.** Phases 1 and 2 are done on both axes and every P-stage
through P5 is closed. The shipped configuration is **D19 + D20 + D21**
(4.5-bit dense bank with a fifth bit on three families, the MoE rows
transcoded, `ffn_down_exps` at IQ4_NL): 4.132 GB a token against a 58.6
tok/s ceiling, perplexity 4.0992 (+1.74% of our own 4.0289, which is itself
−0.13% against llama.cpp's at identical weights). Decode 36.19 tok/s,
prefill **1207.7 tok/s at ubatch 2048 and 1403.9 at 8192** (P11, P12). Served with
prefix reuse; a second turn extends the graph's state rather than
re-prefilling.

**Speculation (P5) is built, lossless, and parked at 0.95x.** The rollback
costs nothing when off. What would take it past 1.0, in order: the draft
head's **acceptance on a real workload** — 65.6% on prose against a
break-even of ~0.72; measure chat/code with the observer
(`cmd/llm -mtp`, ~3 min, no machinery) *before* building anything, and only
above a ≈ 0.8 does the rest get interesting — then the recovery round
(~0.95 → 1.05x) and pre-recording the verification pass (~2%). Hard
ceiling: speculation refuses past 2048 cells (`blk.48` has no compress
ratio). Narrowing the trunk makes speculation worse, twice — the draft
stages at checkpoint widths and acceptance falls as the trunk moves away
from what the draft predicts.

**Open, in rough order of value:**

- ~~**Prompt processing.**~~ **P11, closed 2026-09-21**
  ([write-up](research/p11-prefill.md)). Two changes and three measured
  refusals. **The server prefilled in 512-token chunks and 512 is the worst
  rung this graph has**: llama.cpp plateaus at its best ubatch and this one
  does not, because the MoE's arithmetic intensity is the *routing's* — at 512
  tokens 274 of 512 experts are unpacked whole to serve 5120 rows. `-llm-batch`
  is now **2048**: 1.12 GB of arenas for **1.64x** (667.4 → 1093.3 tok/s at 48
  layers, and 667 → **1053 through HTTP** on a 4128-token prompt), where the
  1.49 GB after it buys 13% more. And **the grouped GEMM's gathered A operand
  was loaded one half at a time** — 32 two-byte loads a lane a K-step against
  the eight the B unpack issues for four times the data — so binding the halves
  arena a second time as `uvec4` is **1.25x on `moe.up`**, the largest kernel in
  a prefill, bit for bit the same slab. The graph: **1079.6 → 1165.1 tok/s at
  2048 rows and 1250.1 → 1328.5 at 8192 (3.40x llama.cpp)**, two runs agreeing
  to 0.2%. What did **not** work, all three reverted and all three the same
  answer: a 16-row alignment with a row count per record (13.7 ms against 11.9),
  three sub-lists one dispatch a rung (12.5), and a 128-row block (13.5). **At
  prefill this GEMM is bound by its per-tile slab unpack and the row padding is
  very nearly free** — L5b built that padding as a cost to justify and it is not
  one, so the way in is fewer *tiles*, which is a bigger ubatch.
  **A second round closed `hc.cn`** — the combine fused with the next mixer's
  norm, the second largest kernel in a prefill and one that does no arithmetic:
  the four streams of a token were each reading the *same* block output row, so
  one workgroup a token instead of one a (token, stream) is **1.37-1.44x** on
  the kernel and bit-identical, taking the graph to **1191.8 / 1372.2 tok/s
  (3.51x)** and 64k prefill to **654**. And it measured, for the first time,
  **how much of the key axis the QSA selection lets the attention kernel skip**
  — at 64 000 cells the shipped 16x32 tile keeps 25.9% of pairs live against a
  14.7% floor for a single query row — which priced and then refused a k-tile
  skip inside the kernel (**1.18x slower, and at 64k slow enough to trip P0's
  ring watchdog**). Two kernels now say the same thing: **a branch inside an
  unrolled cooperative-matrix loop costs more than the work it removes**; change
  the loop's granularity, not what happens inside it.
- ~~**The attention row max, `attn.select`, `hc.cn`, the MoE unpack.**~~
  **P12, closed 2026-09-21** ([write-up](research/p12-prefill-round-two.md)).
  Two changes and four measured refusals; prefill **1.012-1.025x on both
  axes** (1191.2 → **1207.7** at 2048 rows, 1370.5 → **1403.9** at 8192 —
  **3.59x** llama.cpp — and 651.3 → **667.7** at 64 000 cells), both passes of
  each arm agreeing to 1370.5/1370.5 and 651.3/651.2. **The row max did run on
  sixteen lanes of sixty-four** and now runs on all of them: the whole key
  block staged at once and RCL = WAVE/BM lanes a row, folded by a clustered
  subgroup max, **bit-identical** because a max has no rounding and the cells
  scanned are the same — 1.16x on the kernel at 512 tokens, +1.6-1.8% of a
  prefill token from 8000 cells on. And **a K-quant block's header is one
  sixteen-byte load, not four dwords**, which is L5b-7's rule for the third
  time: `moe.up` 12 118 → 11 755 µs at 2048 tokens with `moe.down` flat as the
  control. **The four refusals are the valuable half.** Both of `hc.cn`'s
  named suspects cost **nothing** — deleting the 256-way tree outright is
  5331.5 µs against 5325.8, and dropping the `gamma` read is 5332.9 — so its
  57% of copy bandwidth is unexplained *and* out of hypotheses, and the
  non-bit-exact norm rewrite would have bought zero. The **block skip hoisted
  out of the loop is 1.00x at every depth**, and the reason is the exact
  converse of P8: a prefill dispatches 3072 single-wave workgroups where a
  decode step dispatches 24, so *latency that matters at decode does not
  matter at prefill, because prefill has occupancy* — it is still live for
  decode, where it was never built. The selection mask loop is **3.5-8%**, not
  the rest of the gap. And the MoE's **per-element affine is a floor**: the
  whole scale path is 1.21x at 2048 tokens, P12-2 takes its loads and integer
  ops, and the int→float convert, FMA and f16 convert that remain have no
  bit-exact cheaper form.
- ~~**The ubatch is the largest prefill number that is left.**~~ **Decided
  2026-09-21: `-llm-batch` is now 4096** (P12-7), and measuring it *through the
  server* found the half the graph ladder cannot see. Four distinct
  4.1-4.5k-token prompts, two interleaved passes agreeing to 0.5%: prefill
  **1020.9 → 1146.3 tok/s (1.12x)** and **0.46 s off the time to first
  token** — but decode **28.04 → 25.81 (0.92x)**. **The wider arenas cost 8%
  of decode**, across sixteen non-overlapping samples at both 8 and 256
  generated tokens, because a decode step streams 4.1 GB of weights a token
  and the extra 1.5 GB of arenas sits in the same memory. The two rates cross
  at **~150 generated tokens**: 4096 is 1.10x at 8 tokens and 2048 is 1.07x at
  the 1024 `-llm-max-tokens` defaults to. So 4096 ships as the interactive
  default and `API.md` says outright that a batch summariser should set 2048.
  **The lesson is P11-1's, a second time: a ubatch measured on the prefill
  ladder alone is a hypothesis about prefill.** **Superseded by P16**: the decode
  cost was a padding bug, and decode is now 34.2 tok/s at every batch.
- **`attn.select` at decode is 55 µs a dispatch at 128 000 cells** (P16-3,
  from 68), ~0.66 ms of a 31.7 ms token. A probe split it: ~8 µs a radix pass,
  ~12 µs of emit after P16 took it from 25, ~10 fixed. Two guesses are measured
  dead — the passes are **not** a chain of load latencies (four keys in flight a
  lane is 0.96x) and **not** LDS-atomic contention (a per-wave bucket fold is
  0.69x). What is left is P10's reduction, and the two-level arrangement
  (per-stripe histograms and a merge) is the only idea not yet priced; at ~2%
  of a deep token it is low on the list.
- **What is left of decode's falloff after P15 is ~2 ms of 36.8**, and the
  shape of it has changed: the gathered list is 2051 cells at *every* depth, so
  `attn.attn.split` being 1.571 against 1.092 at depth zero is no longer a cell
  count — it is those same 2051 cells scattered over a deeper cache, where each
  128-byte line costs more to reach. That is a locality question and not a work
  one, and nothing in this vertical has asked it yet.
- ~~**Why 1.5 GB of arenas a decode step never reads costs it 8%.**~~ **P16,
  closed 2026-09-22** ([write-up](research/p16-decode-arena-width.md)). It was
  not the memory: the block input ports padded a decode step's one row out to
  the arena, so a wider arena was more zeros written a layer. At an 8192 arena
  decode goes 23.55 → 33.87 tok/s; through the server decode is 34.2 at every
  batch. What remains between an 8192 and a 2048 sweep is depth — the wider
  prefill leaves a deeper cache — not width.
- ~~**Prompt processing at 128k.**~~ **P13, closed 2026-09-22**
  ([write-up](research/p13-long-context-prefill.md)). Two changes and one
  refusal. **128 000 cells completes**: `batchFor(rows)` was P0's fit and
  every measurement behind it was taken from cell zero, so at 64 000 cells a
  2048-row pass already held the ring for 1.53 s of the 2 s cliff while the
  budget believed it was spending 0.86 — the missing term is 10.46 ns per
  (row, cell), it is carried by the attention dispatches alone, and the
  chunker now walks the recorded sequence charging each dispatch its own
  label's cost and corrects itself against what the last submit measured.
  And **the key block was chosen for the dense regime**: L2f picked
  `qt1_kt2` on 512-token prefills from cell zero, where a 32-cell block is
  two cooperative-matrix tiles of reuse against one; at 128 000 cells the
  selection leaves 11.7% of (16-row tile, 16-cell block) pairs live against
  17.4% at 32 cells, and the narrow rung is **1.61x on `attn.attn`**
  (0.9107 → 0.5659 ms a token) for **471.7 → 563.6 tok/s**, with
  `attn.select`, `attn.score` and `attn.expand` identical to three decimals
  as the control. That is P11-7's 1.40x, collected as a *rung* after P11-7
  lost 1.18x trying to collect it as a branch. It is **prefill's rung and
  not decode's** — with it on both, decode at 128k went 22.94 → 22.27,
  because a decode step costs one wave's serial walk and halving the block
  doubles the walk. **The refusal**: compacting the live key blocks into an
  ascending per-query-tile list deletes 83% of the axis walk, costs 0.001 ms
  a token, and is **1.00x at every depth** — P11-4's finding a third time,
  *latency and redundant reads that matter at decode do not matter at
  prefill, because prefill has occupancy*. The machinery ships behind
  `LLM_ATTN_BLOCK_LIST=1`, bit-identical, because the decode split and a
  per-cell gather both want it.
- ~~**`attn.select` and `attn.expand` are 0.284 ms a token at 128 000 cells.**~~
  ~~**The gather.**~~ **Both closed by P14, 2026-09-22**
  ([write-up](research/p14-prefill-at-depth.md)). Four changes and four measured
  refusals; **128 000 cells prefills at 826.0 tok/s at ubatch 2048 and
  946.1 at 8192**, against P13's 563.6, with the falloff from depth
  zero **0.50x → 0.72x** and decode unmoved as the control.

  The selection now runs over the **block** scores with a per-block weight
  instead of the expanded per-cell tensor: `attn.select` **0.281 → 0.020**
  (13.8x, not the 4x the traffic argument predicts — the histogram is bounded
  at the last block a cell can reach and the LDS cache now holds `ratio` times
  as many cells), `attn.expand` deleted, **1.14 GB of arena** freed, and it is
  bit-identical over four chunk schedules. The attention runs over a **per-cell
  gather** — the union of a query tile's rows' selections, compacted ascending,
  with the causal test folded into a per-row mask — for **2.15x on `attn.attn`**;
  that needed the **value plane to become cell-major** first, which costs the
  block kernel 7% and decode nothing. And the indexer's score moved to the
  **matrix cores**, 2.3x.

  **The union was the number that priced all of it and P13's estimate was
  wrong.** `AttnGPU.SelUnion` measures it: at 128 000 cells a 16-row tile reads
  15 145 cells where its rows' union is 7 049 and one row selects 2 051 — so the
  gather is 2.15x, not 3.3x, and the 3.4x between the union and one row is the
  floor a sixteen-row fragment cannot reach.

  **The gather is the first kernel here that is not chunk-invariant and cannot
  be made so** (its list is the union of sixteen rows, so a chunk ending inside
  the tile gathers a different list and the softmax folds the same terms in a
  different order). It is pinned off under `PinSchedule` with the other
  reassociating kernels, and the exact chunk gates pin it;
  `TestAttnGPUGatherSelectsTheSameCells` is exact on the *set* and
  `TestAttnGPUGatherIsTheBlockKernel` is the tolerance (4.1e-06 rms against
  9.9e-04 from llama.cpp).
- **The gathered kernel is at 34% of matrix-core peak and all four bounds are
  eliminated**, which is P14's most reusable finding. Doubling its matrix work
  costs **2.7%**; a subgroup staging barrier instead of a workgroup one is
  1.00x; staging two head-dim groups a barrier is 1.00x and four is 0.75x;
  sharing the staged key and value across two query heads of one kv head —
  which halves both the global gather reads and the LDS writes — is **1.04x**,
  and four heads is 0.75x. So it is not the matrix cores, not latency, not the
  gather's traffic. What is left is the **LDS round-trip for the fragments and
  the per-head softmax scaffolding**, both linear in the union, which is why the
  kernel's cost is linear in the cells it visits. `LLM_ATTN_GATHER_GRP` and
  `LLM_ATTN_GATHER_HEADS` are the ladders and both ship at 1.
- **`maxStorageBufferRange` is silent when it is exceeded and `checkBufferRange`
  now says so.** A VkBuffer over the range is legal to create; binding more of
  it than the range is not, and the driver **clamps rather than failing**, so a
  kernel reading past it gets zeros. A five-depth `-depth` sweep at ubatch 4096
  raises `-ctx` to 148 520 cells, which is a 4.39 GB fp16 arena against a
  4.29 GB descriptor — and the run came back at **1124 tok/s at 128k against a
  true 885**, 1.28x too fast, with a degenerate indexer selecting a contiguous
  window. **The failure mode is a benchmark that gets faster**, which is the one
  direction nobody audits; the rule it leaves is to price a suspicious win in
  FLOP/s or bytes/s against the device's peak before believing it.
- **`hc.cn` is still at 134 GB/s** where a copy gets 236, and there are now
  **three** spent explanations, not two. P12-4 spent the 256-way tree (deleting
  it outright is 5331.5 µs against 5325.8) and the `gamma` read (5332.9). P15
  spent the **memory type**, which was the most plausible of the three and had
  never been tested: the arenas are HOST_CACHED and this kernel is pure arena
  traffic, so it looked like the whole answer — and `LLM_ARENA_UNCACHED=1`
  moves it 1272.2 → 1260.4 µs, inside the noise, while host glue moves 29x as
  the evidence the knob was really thrown. The next probe is the access pattern
  itself: it runs four read-modify-write streams plus an fp16 write a token,
  where a copy runs two.
- **P6 — batching.** Blocked on the product question: will the API serve
  more than one stream? Each sequence owns 113 MB of DeltaNet state plus
  rings and KV. P5b already built the first stage (R-row decode GEMVs,
  shipped at R = 2); raising `GEMVMaxRows` must land in the same commit as
  its rung in `TestGraphIsAChunkSplit` — R rows were correct in a block test
  and wrong in the model three separate ways (P5c).
- ~~**Context depth is the largest measured regression.**~~ **P7, closed
  2026-09-21** ([write-up](research/p7-context-depth.md)). It was two terms
  and neither was the selection's width. **Three kernels walked `nKV`, the
  cells the arenas were *allocated* for, instead of the cells that exist** —
  `llm_attn_score.comp` scoring every pooled block, `llm_attn_select.comp`
  running four radix passes and an emit over every cell — which is **6.94 ns
  per allocated cell a token an attention layer**, charged in full *at depth
  zero*: **10.9 ms of a decode step** at 131k cells before one cell is real,
  and most of why the old sweep began at 24.27 where the headline is 36.19.
  **And QSA was semantics with no saving**: `llm_attn_wmma.comp` read the
  bitmask beside the causal mask and still walked every key block, so a
  decode step that names 2051 cells was reading 64 000 (0.25 µs a live cell
  a token a layer). The fix is three identities — the dead pooled blocks all
  hold cell 0 pooled `ratio` times so they share one score; a cell past the
  live count is an `-inf` that can never be selected; a key block with
  nothing selected in it contributes nothing and is skipped whole — gated on
  **exact** equality by the new `TestAttnGPUCacheSizeDoesNotChangeTheAnswer`
  (the 4k fixture in a cache three times too big, bit for bit) and by 48
  greedy tokens identical across the two binaries. Two passes each, same
  hour: decode **4.05 → 16.27 tok/s at 64k (4.02x)**, **6.78 → 18.34 at
  32k**, **27.96 → 33.18 at depth 0**; prefill **313.1 → 441.2 at 64k**;
  `-gen` at ctx 32768 **32.43 → 35.92 tok/s**, 94% of the byte ceiling.
  Falloff to 64k: decode **0.14x → 0.49x**, prefill **0.51x → 0.64x**.
- **The gather is what is left, and after P13 it is the only route to 900
  tok/s at 128k.** P13's own probe priced the regime it has to beat: fitting
  the `qt2_kt2` control — 1.46x less K/V a query for 1.37x more work, and
  1.02x *worse* on the clock — says `attn.attn` at 128 000 cells is roughly
  **half arithmetic and half K/V traffic**, so fewer cells is the only thing
  that cuts both. A 16-row query tile's union is ~4 574 selected cells and
  the 16-cell rung reads 14 976 of them: **3.3x**, and that is the floor of
  what a cooperative-matrix tile can skip, because the tile is sixteen cells
  and the selection's runs are four. Priced at **~0.20 ms a token against
  0.566**, which with the selection change and a 4096 ubatch is ~1.06 ms a
  token — **~940 tok/s at 128k**, the only arrangement of these numbers that
  reaches the target. The open question is the arena: a per-(query tile, kv
  head) scratch at 8192 gathered cells is 2.1 GB, which fits only once the
  expansion's 1.14 GB is freed, and the alternative is chunking the gathered
  axis and folding the partials through P8's combine, which already exists.
  P13's `llm_attn_blocks.comp` is the compaction it would be built on.
- ~~**What the gather replaces, from P7's side.**~~ **Closed by P15-3,
  2026-09-22.** The skip prunes key blocks; it does not stop the count of them
  growing — so decode now runs the **gathered** kernel with its axis split,
  `GATHER`+`SPLITK`, and `attn.attn` at 128 000 cells is **419.8 → 129.9 µs**
  a dispatch. P14 wrote down that decode should not take the gather and the
  argument is *right about the gather alone*: unsplit, it is 986.7 µs against
  the split's 419.8. What it missed is that the split cuts the **walk** (7.12x)
  and the gather cuts the **work** (2.99x), and neither had the other's factor.
  The bound is `2*selWidth` **live** cells, below which every live cell is
  selected and the compaction is overhead — 1.09 → 1.56 ms a token the wrong
  way at depth zero. That bound is the first decode knob that moves with the
  depth, so the prerecorded buffer carries an epoch now. **P11 measured the
  prefill side of the same question and it is the whole of the depth term
  there too**: at `-pp 2048` every block is flat in depth except attention,
  which is 97% of the falloff and 47% of a prefill token at 64 000 cells
  (`attn.attn` 27x from depth zero, `attn.select` 46x). 64k prefills at
  **654 tok/s**, 0.57x of the depth-zero rate. **And the prefill side of the
  gather is now priced**: the union of sixteen adjacent queries' selections is
  only 1.76x one query's reach, so a per-query gather is worth 1.76x at 64k
  and not the 31x the 3.2% density suggests — L4b's refusal, with a number.
- ~~**128k still does not complete.**~~ **Closed by P13-1, 2026-09-22**, and
  the marks were never the problem — it was a submit budget with no depth
  term. `maxStorageBufferRange` is the next wall: the KV planes share a
  buffer with the arenas, which caps the cache at **~148k cells**. Past that
  wants L6a's array-of-buffers, one a layer.
- **Long-context gates** (idea 7, enabled by P0): perplexity at ctx 4096 is
  done (3.9392, −2.23% against ctx 2048 — the selection helps); a needle
  test through the API is not run.
- **The downstream task eval** — the last unpriced thing about the widths.
  At +1.74% of wikitext perplexity the instrument may no longer separate
  plans; a few hundred multiple-choice items through the HTTP API is an
  evening.
- **`Reset` is owed a test.** A fresh sequence is not independent of the one
  before it (P5c finding 6): three plain greedy runs over one prompt give
  three different texts, deterministically, when the previous run left cells
  past the new prompt's end. Any future token-for-token claim needs the
  plain-against-plain row printed beside it.
- **Priced and not taken** (in the archives, with numbers): the router's
  padding (+0.09 tok/s), re-screening the GEMV rungs at the q5 width, fp16
  DeltaNet state (+~1.3 ceiling), the unpack prefetch (~1.1x prefill), the
  asymmetric epilogue's GEMM arm, W4A8, the hot-expert fast path, the
  float-atomic combine, the vision tower. Also unexplained and written
  down: a dispatch is 8–22% slower inside a step than alone on a cold bank
  (P1b's 1.33 ms environment gap), and dispatch time is not a function of
  its bytes in either direction (P3a/P4b, opposite signs).

**Instrument rules that must survive** (each earned the hard way): a
whole-model number needs a **same-hour control** (P1c — the machine moved
8.7 ms a step overnight); a ladder rate above 242 GB/s is an L3 hit, and
its *byte count* can lie too (D16, P4a); a plan is **measured, never
composed** (additivity leaks both ways through the n-gram block); a control
has to be able to fail (P5b shipped green tests on stale SPIR-V — `.comp`
edits need `go generate` before measuring); benchmarks need
`LLM_BANK_CACHE` set or a shipped-bank run re-fits for 8 minutes
(`cmd/serve` sets it, `cmd/llm` does not).

## Speech → text (archive: [`research/speech-vertical.md`](research/speech-vertical.md))

**Where it stands.** S1–S8 done: front end 21 ms + encoder 17 + decode 5 =
43 ms for jfk.wav, transcript exact, word/segment timings from the model's
own TDT durations, served at `/v1/audio/transcriptions` and over Wyoming
(with resampling at that door only).

**Open:**

- **S10 — the front end is 48% of the pipeline**: a few thousand 512-point
  float64 FFTs on the host. Either a float32 radix-4 on the host or the
  STFT + mel filterbank as two dispatches (the filterbank is a
  `[T, 257] x [257, 128]` GEMM; `kokoro_istft.comp` is the worked inverse).
- **The submit+fence is 38% of an emission** (40 µs of 105 per token). A
  persistent kernel — which is also what streaming transcripts would want —
  and/or speculating on blanks (consecutive-frame joints are independent
  during a blank run: one GEMM at M = 16 for the price of M = 1).
- **S9 — long clips.** Full attention means a chunk boundary changes every
  frame; chunking belongs in the design. The quadratic term arrives around
  1024 frames (~82 s); `-max-audio` refuses past the sizing today.
- Small: cache `UploadMel`'s sinusoidal position rows (1 ms, depends only
  on T); the eight per-head position-score GEMMs are §3.5's grouped shape.

## Text → speech (archive: [`research/speech-vertical.md`](research/speech-vertical.md), recap: [`research/tts-recap.md`](research/tts-recap.md))

**Where it stands.** T1–T10 and W1 done. Every stage on the device, staged
once for the life of the server (T9: the endpoint went 550 → 59 ms with
byte-identical audio); voices blend in upstream's own spelling (T8); T10
put PL-BERT's attention on the matrix cores so synthesis is a straight
8.0 ms per second of audio at every length. The round trip
(`cmd/roundtrip`) closes at 70.1x real time, six prose cases exact.

**Open, in order of what a round trip buys:**

- **The generator's 12 ms and the tail's 8** — the only arithmetic-bound
  parts of the model, 65% of a short utterance, measured since T4.
- **The host embedding stack** (R1's #2): `bert` is 12 ms of a paragraph
  and only 2.5 on the device — the "not worth a dispatch" comment is stale
  the same way the attention kernel's was.
- **Three boundaries, one change each**: the excitation's 0.2 ms is 88%
  submit+readback (move the two noise convolutions onto the device and
  nothing of that stage touches the bus); the phoneme side's remaining 6 ms
  is readbacks and submits (move the vocoder's input boundary); a 16 kHz
  path out of the vocoder would delete the client resample (9.7% of the
  loop — can the iSTFT head be asked for the rate directly?).
- **G2P on unseen text**: designed corpus 24/24; on 400 unseen sentences
  68.8% of sentences, 92.4% of phoneme words agree with misaki. Closing the
  gap is more *measured* rules — the syntactically obvious ones scored
  worse than no tagger.
- **Known, unsettled**: the style row is indexed by the phoneme *character*
  count (upstream's behaviour), so a front end emitting different characters
  picks a different row. And kokoro spells numbers out where parakeet writes
  digits back — the round trip measures those three cases and does not
  count them as failures.

## Image generation — **parked** (archive: [`research/qimage-vertical.md`](research/qimage-vertical.md))

**Where it stands.** Done and **parked 2026-09-21** — parked because the plan
ran out, not because it stalled. (The superseded z-image-turbo vertical is
[`research/zimage-vertical.md`](research/zimage-vertical.md) plus
[`research/zimage-pipeline.md`](research/zimage-pipeline.md); its stages are
**I0–I7** and none of its code survives except `zimage/qwen` and
`zimage/tokenizer`, which `embed`, `llm` and `parakeet` import.) The vertical was replaced 2026-09-20
(Z-Image-Turbo out, `Qwen/Qwen-Image-2.1` in, for native RGBA and
reference-image editing — `GOALS.md` #4), and **stages Q0–Q12 all closed
inside two days**. Both endpoints are served: `POST /v1/images/generations`
answers at **1m28.8s / 1m29.8s for a 1024²/40-step image** (31.5 GB resident,
matching the fp32 oracle's own picture at mean 3.4e-4) and `POST
/v1/images/edits` at **1m54.2s / 1m56.6s for a 1024² edit on one reference**
(39.4 GB, matching the oracle's edit at max abs 0.0014), both with native
RGBA and unconditional in-progress previews (a fitted 64x4 matrix, 159 µs a
frame, three partials for 0.3% of a request — no flag, because there is
nothing to load). Q10 turned the ceiling from a side box into an area, so
16:9 comes back **1344x768** instead of 1024x576; Q11 put the client's
hang-up through to the sampler and the VAE's submit batches; Q12 finished the
Z-Image deletion (**6,839 lines of Go and 895 of GLSL** out, every gate
re-run with no digit changed). The full write-up, every tolerance with its
instrument named, is in the archive.

**What to read before touching this code again** — three precision facts,
each of which has already caught a port:

- **the VAE cannot take fp16 operands anywhere** (Q9b). Its tail norm divides
  a per-pixel L2 out of a residual stream at absmax 2.6e5, so a 5e-4 relative
  perturbation becomes an absolute one: **one** narrowed convolution costs the
  decoded image max abs 0.0885 and the whole 3x3 set costs 0.178, against an
  fp32 port sitting at 7.3e-4. The rule is not "watch the range", it is "do
  not narrow". `TestConvFP16Ladder` is the instrument; re-run it before
  pointing any narrowing kernel at `qimage/vae`. Q12 deleted
  `vae_conv_wmma`/`vae_attention_wmma` outright, so the shortcut is not in
  the tree — resurrect from git history only if that ladder says otherwise.
- **the vision tower amplifies an input perturbation by ~10³**, so a
  condition image must be quantized exactly as the reference's is —
  compositing alpha over white in float rather than on 8-bit levels moves the
  prompt embedding by rel 11 (a firing control). That is why the Lanczos
  resampler is gated on exact 8-bit equality and not a tolerance.
- **on a non-square condition image the fp32 dump is the less accurate
  side** — rel 1.4e-3 from a float64 run where the Go tower sits 2.1e-4 — so
  that stage is gated against dumped float64 rows.

**If it is ever unparked**, in the archive's order:

- **The 1184²-area ceiling** — the only remaining *capability*, not a
  percent. The VAE decoder's activation arena is one storage buffer against a
  4 GiB − 4 device limit at a measured **3060 bytes a pixel for every aspect
  ratio** (`TestArenaShape`), which caps a request at 1,403,584 pixels
  whatever shape they are in, so the model's own 2048² examples do not
  decode. Two routes: tiled decode, or a multi-buffer arena
  (`vk.PipelineSpec.Counts`, with the LLM's 77 GB bank as precedent). Tiling
  is the less attractive of the two against a tail norm that is a per-pixel
  L2 over the whole feature map.
- **conv3x3's last ceiling** — after Q9b's register block it is **85.9% of
  the decode, 3.36 s at 5.0 TFLOP/s**, and its remaining limit is one shared
  read per multiply-add. A pixel block is priced at ~1.5 s and is the only
  port here that would **not** be bit-identical. Low value now: the VAE is
  4.5% of an image, the DiT is 95%.
- **The DiT's remaining percents are fusions** — that is where the image's
  time actually is, post-Q9.
- **Masked edits** (Q-o3) — still a 501, but the reason moved: 2.1 *can* do
  masked and annotated local edits; how a mask is fed is not in the diffusers
  implementation this port follows. External research, not a port.
- **Two watch items**: Q-o2, whether a turbo/distilled 2.1 checkpoint or
  step-distillation LoRA appears (the examples repo and lightx2v); Q-o4,
  whether [taehv](https://github.com/madebyollin/taehv) grows a 2.1 variant,
  which would turn the fitted linear preview from a fallback into an upgrade.

**Settled, so it does not get relitigated**: the step count — 40 stays the
default because it is the only count safe across prompt kinds. A fox
photograph and an impasto harbour are convincing at **12 steps**, at 36% of
the cost; a bicycle drivetrain diagram is coherent at 24 and has
disintegrated by 12 (floating parts, ghosted tubes, contrast collapsing
toward white). **24 is the honest fast setting** — −35%, no visible loss on
any of the three prompt kinds. `steps` is a request field, so a client that
knows its prompt takes the discount itself. Note the sweep's wall clocks
(40: 1m38–1m42, 24: 1m4, 16: 44–45 s, 12: 35–36 s) were measured **before
Q9 and Q9b**, so the seconds are stale by the 6–7% those two took off every
step while the *ratios* stand; re-run with `QI21_SWEEP=1 go test
./qimage/pipeline -run TestStepSweep` (~10 minutes of device time) before
quoting a number from it.

**Not planned**: quantisation (compute-bound at every servable size, §3.4;
int8 WMMA runs at fp16 rate, §0; and the two-machine deployment removes the
footprint argument). Sampling the encoder's posterior (breaks seed
reproducibility). A self-trained tiny decoder for previews — a training
project this repo does not want. No CFG path (`true_cfg_scale` stays a
refusal) — it would double every step.

## Embeddings (archive: [`research/embedding-vertical.md`](research/embedding-vertical.md))

**Where it stands.** E0–E8 done except E7. Same `zimage/qwen` transformer,
third caller; 500 lines of package. A ≤32-token text costs what the weights
cost (11.5 ms, 32% of the bus); cosine 0.999999+ against fp32. Served with
MRL `dimensions` and the non-OpenAI `instruct` field.

**Open: E7, batching — up to 10x on short texts.** The projections already
take M; the blocker is attention (two texts in one causal stream need a
block-diagonal mask). Measure the cheap shape first: **attention per
segment, projections batched** — attention is 1.5% of the graph and the
push block already carries per-text offsets, so no shader changes. The
segmented-mask kernel is the fallback. Quantisation stays unplanned until
a batch makes the weights the limit again.

## The server, cross-vertical (docs: [`API.md`](API.md))

- **Constrained decoding** — the one refusal that is a missing capability:
  would close `response_format`, `text.format` and named `tool_choice`.
- **More than one conversation** — the graph is one sequence's; two clients
  take turns evicting each other's prefix. A cache per conversation is a
  measurement (residency) plus P6's scheduler question.
- **The image adapter holds the device lock for the whole run** (the LLM's
  is per forward pass). Same fix shape; what a waiting speech request
  actually gains is a measurement.
- `n > 1` images are serial (capped at 4); a forced alignment for
  transcript timings if the 80 ms grid ever isn't enough.

---

## Where the old files went (2026-09-20 consolidation, and IMAGE.md again on 2026-09-21)

| was | now |
|---|---|
| `LLM.md` | [`research/llm-vertical.md`](research/llm-vertical.md) — stages, decisions D1–D21, open questions, how-to-run |
| `LLM2.md` | [`research/llm-review.md`](research/llm-review.md) — hypotheses checked, decode budget, P0–P6 as closed |
| `SPEECH.md` | [`research/speech-vertical.md`](research/speech-vertical.md) — S1–S8, T1–T10, R1, W1 write-ups |
| `TTS.md` | [`research/tts-recap.md`](research/tts-recap.md) |
| `IMAGE.md` (z-image, until 2026-09-20) | [`research/zimage-vertical.md`](research/zimage-vertical.md) |
| `IMAGE.md` (Qwen-Image-2.1, 2026-09-20–21) | [`research/qimage-vertical.md`](research/qimage-vertical.md) — frozen 2026-09-21 with the vertical; **Q-stages, decisions 1–7 and Q-o numbers resolve there** |
| `PIPELINE.md` | [`research/zimage-pipeline.md`](research/zimage-pipeline.md) — inventory, validation rules, budget |
| `EMBEDDING.md` | [`research/embedding-vertical.md`](research/embedding-vertical.md) |
| `IDEAS.md` | [`research/ideas.md`](research/ideas.md) — the `§N.M` backlog and the measured roofline |
| `TODO.md` (session log) | distilled into `research/` as each stage closed; the phase-1 tail is [`research/phase1-backlog.md`](research/phase1-backlog.md); the full log is in git history |

Per-experiment findings remain one file each in [`research/`](research/README.md),
indexed there. `§N.M` numbers are permanent addresses — never renumber.
