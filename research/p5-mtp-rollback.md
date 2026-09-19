# P5 — MTP speculation: the rollback design, and the verification pass priced

> **2026-09-19.** The design pass `LLM2.md` idea 3 asked for, plus the one
> measurement it said had never been made: *"verification runs at M = 2-8,
> where D15 refuses the GEMV path by design — the M=2..8 GEMM/GEMV crossover
> has never been measured on this model's shapes."*
>
> **It has now, and it is the whole stage.** A verification pass over M rows
> costs **3.2 to 4.3 decode steps** for M = 2 to 8, so speculation as the
> model stands is a net *loss* at every acceptance rate and every M. The
> rollback — which the review treated as the hard part — turns out to be
> 120.75 MB of device copy and one deferred dispatch, and it is the easy
> part.

Data: `results/p5_graph_m_r1.csv` (the whole graph, 24 layers, per block),
`results/p5_{moe,hc,dn,attn}_m_r{1,2}.csv` (the four blocks, cold banks,
two runs).

---

## 1. The verification pass, measured

`-graph -tokens 1,2,3,4,6,8 -layers 24 -ctx 2048`, D19 dense bank + D20/D21
MoE bank, one binary, one run. This is the real graph with the real host
loop, not a composition: 591 dispatches a pass, the same five blocks, the
same head.

| M rows | ms | vs M=1 | hc | ple | dn | attn | **moe** | head | host |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **16.7** | 1.00x | 2.1 | 0.3 | 4.2 | 1.2 | **5.2** | 2.4 | 1.3 |
| 2 | **54.1** | **3.24x** | 13.9 | 0.3 | 8.3 | 2.7 | **25.5** | 2.4 | 1.2 |
| 3 | 58.3 | 3.49x | 14.4 | 0.3 | 8.5 | 2.7 | 28.4 | 2.4 | 1.2 |
| 4 | 59.6 | 3.57x | 14.0 | 0.3 | 8.5 | 2.7 | 30.7 | 2.4 | 1.1 |
| 6 | 65.4 | 3.92x | 13.8 | 0.3 | 8.8 | 2.7 | 36.4 | 2.4 | 1.1 |
| 8 | 72.1 | 4.32x | 13.8 | 0.3 | 8.5 | 2.7 | 43.5 | 2.4 | 1.3 |

**Every block has the same shape and there are only two shapes.** Four of
them — `hc`, `dn`, `attn`, and the fixed costs (`ple`, `head`, `move`, the
host) — take **one step up at M = 2 and are then flat to M = 8**. Only
`moe` keeps climbing.

That is exactly D15's refusal made visible. At one token these blocks run
`llm_gemv.comp`, `llm_hc_gemv.comp`, `llm_moe_gemv.comp` and
`llm_moe_router.comp`; at two the host refuses those rungs and the
cooperative-matrix GEMM takes over. The four ladders say it a second time,
cold-bank and per layer, and the rung name is printed beside the number:

| block | ×N | M=1 rung | M=1 us | M=2 rung | M=2 us | M=2/M=1 | M=8 us |
|---|---:|---|---:|---|---:|---:|---:|
| hyper_conn | 97 | `down_gemv32/up_m1` | 51.0 | `down_m1/up_m4` | 273.0 | **5.35x** | 275.6 |
| deltanet | 36 | `gemv k8/k32` | 231.4 | `gemm_m4/gemm_m2` | 429.8 | **1.86x** | 433.1 |
| full_attn | 12 | `gemv k8/k32` | 232.9 | `gemm_m4/gemm_m2` | 422.1 | **1.81x** | 421.2 |
| moe | 48 | `v64w4/v16w4/k40` | 196.8 | `m2/m2/gemm` | 1134.8 | **5.77x** | 2152.6 |

(means of two runs; **M ≥ 2 reproduces to 0.6%**, M = 1 does not — the
one-token rungs are the smallest dispatches in the model and the spread
between runs is 18% on `hc` and 30% on `dn`, which is P1b's environment gap
at its worst. The argument below rests on the M ≥ 2 rows and on the
whole-graph table above, both of which are stable.)

### 1.1 Finding 1 — the dense blocks pay a rung, not a byte

`hc`, `dn` and `attn` read **exactly the same weights** at M = 8 as at
M = 1: every one of their matrices is dense and read once a pass. Their
cost is flat from M = 2 to M = 8 — 13.8/13.9, 8.3/8.5, 2.7/2.7 ms — so the
step from M = 1 to M = 2 is not work, it is **shape**. The GEMM's row block
is 16 to 64 rows and two of them are real; the hyper-connection block's is
the worst because its `up` rung collapses to a 64-column block (P1a) and
its 485 dispatches a pass each pay the padding.

**6.6x of the hyper-connection block at two tokens is padding.** It is the
same law D11 states at one token, one step along: a dispatch here is
short of *rows* now rather than short of *workgroups*, and the fix is the
same kind of kernel.

### 1.2 Finding 2 — the MoE pays a rung *and* a real byte, and the byte does not go away

The MoE is the only block whose weight traffic is a function of M, because
a token reads 10 of 512 experts and two tokens do not read the same ten.
The bench prints the count:

| M | experts touched | E(M)/10 | routed bytes a pass |
|---:|---:|---:|---:|
| 1 | 10 | 1.0 | 1.327 GB |
| 2 | 17 | 1.7 | 2.256 |
| 3 | 26 | 2.6 | 3.450 |
| 4 | 36 | 3.6 | 4.777 |
| 6 | 44 | 4.4 | 5.839 |
| 8 | 49 | 4.9 | 6.502 |

(1.327 GB is the routed half of D21's 4.132 GB a token: `gate`+`up` 0.885
at Q4_K and `down` 0.442 at IQ4_NL. The other 2.805 GB — the dense
families, the shared expert, the router, the lm head — is **M-independent**,
which is the entire reason speculation could pay here at all.)

So the MoE has an irreducible floor that grows, and on top of it the same
padding tax as everyone else: at M = 2 the grouped GEMM executes **544
tile-rows to compute 20 real ones (27.2x)**, and the up mode reads its bank
at **47 GB/s against the GEMV's 185**.

### 1.3 Finding 3 — with today's kernels, speculation loses at every M

A round that drafts M tokens emits between 1 and M+1. Against a 27.63 ms
decode step (36.19 tok/s, D21), the measured pass costs 3.24 to 4.32 steps.
Even at **100% acceptance** — every draft correct, M+1 tokens emitted —
M = 2 yields 3/3.24 = 0.93x and M = 8 yields 9/4.32 = 2.08x, and that is
before a single draft step is paid for. At a realistic acceptance the whole
table is under 1.0x.

**P5 cannot be built on the kernels that exist.** That is the finding, and
it re-orders the stage.

---

## 2. What the pass would cost with a decode-shaped verification kernel

The kernel the verification pass wants is not new: it is
`llm_moe_gemv.comp`, `llm_gemv.comp`, `llm_hc_gemv.comp` and
`llm_moe_router.comp` with **R rows instead of one**. Each of them already
unpacks a weight into registers and dots it against one A vector staged in
LDS; an R-row arm stages R A vectors and keeps R accumulators, and the
weight is read once for all of them. At K = 2560 an A vector is 5 KB of
LDS, so R ≤ 8 fits beside the 64 KB budget (R = 4 is 20 KB and comfortable).

With that kernel the dense blocks return to their M = 1 cost — their bytes
never depended on M — and the MoE's routed half scales with E(M)/10 while
its router, shared expert and combine do not. On the measured M = 1 basis
of the table in §1 (24 layers):

| M | projected ms | vs M=1 | measured today | today/projected |
|---:|---:|---:|---:|---:|
| 1 | 16.7 | 1.00x | 16.7 | 1.00x |
| 2 | 19.4 | **1.16x** | 54.1 | 2.79x |
| 3 | 22.9 | 1.37x | 58.3 | 2.55x |
| 4 | 26.7 | 1.60x | 59.6 | 2.23x |
| 6 | 29.8 | 1.78x | 65.4 | 2.19x |
| 8 | 31.7 | 1.90x | 72.1 | 2.27x |

**This is a projection, not a measurement**, and by this repo's own rule
(P3: *a plan is measured, never composed*) it is owed a number. It assumes
only two things, both of which the table in §1 already demonstrates: that a
block whose bytes do not depend on M costs what it costs at M = 1, and that
the MoE's routed cost is proportional to the experts it touches. The second
is visible in the measured row — `moe` 25.5 → 43.5 ms from M = 2 to M = 8
against E(M)/10 going 1.7 → 4.9, which is 1.71x against 2.88x, i.e. today's
kernel is *sublinear* in experts because it is dominated by padding that
does not grow.

### 2.1 What that is worth, end to end

The draft step: the MTP head is **0.538 GB a token** — its lm head at our
q5_k width is 0.437 of that, the `blk.48` layer 0.053, its ten routed
experts 0.036 and the `nextn` tensors 0.012 — against a main step's 4.132,
so **13.0%, about 3.6 ms**. (`LLM.md`'s carried "~12% of a full step" is
right, and the reason is worth saying: **a draft step is four fifths lm
head**. The MTP layer itself is under 0.07 GB.)

> **Superseded by P5a (2026-09-19), which measured the acceptance rate this
> section could only assume, and corrected the row count.** A round of depth
> M verifies in **M+1 rows**, not M — the committed token plus the M
> candidates — and the measured profile is **a₁ = 74.0%** with a chain that
> flattens rather than decays geometrically. On that basis the optimum is
> **depth 1 at ~1.33x**, not depth 2-3 at 1.5-1.8x, and since a depth-1
> verify pass is two rows **P5b shrinks to an `R = 2` arm**. See
> [research/p5a-draft-head.md](p5a-draft-head.md) §3. The assumed table is
> kept below for the record.

With acceptance `a` per draft token and a geometric accept chain, a round
emits `1 + Σ_{i=1..M} aⁱ` tokens and costs `pass(M) + M × 0.130` steps:

| M | round cost (steps) | a=0.6 | a=0.7 | a=0.8 |
|---:|---:|---:|---:|---:|
| 2 | 1.43 | 1.37x | **1.53x** | **1.71x** |
| 3 | 1.77 | 1.23x | 1.43x | **1.67x** |
| 4 | 2.13 | 1.08x | 1.30x | 1.58x |
| 6 | 2.58 | 0.94x | 1.19x | 1.57x |
| 8 | 2.96 | 0.86x | 1.08x | 1.50x |

Three things follow, and they set the stage's shape:

1. **The optimum is M = 2 or 3, not 8.** The MoE's expert growth and the
   draft's lm head both punish depth, and they punish it from opposite
   ends. `LLM.md`'s carried "1.5-1.8x" is reachable, but only at shallow M
   and only with the new kernel.
2. **Acceptance decides the stage.** At a = 0.6 it is 1.37x at best; at
   a = 0.8 it is 1.7x. Nothing else on the list moves the answer as much,
   and it is the one quantity nobody has measured.
3. **The kernel is not optional and it is not only P5's.** An R-row decode
   kernel is exactly what P6 (batching) needs — R concurrent sequences are
   R rows through the same weights. **P5 and P6 share their first stage**,
   which answers `LLM2.md` idea 8's ordering question: build the kernel
   once and both become possible.

---

## 3. The rollback, designed

The review's worry. It is smaller than it looked, and one part of it is
free.

### 3.1 The inventory: what a pass mutates that outlives it

| thing | where | bytes | rewind |
|---|---|---:|---|
| DeltaNet recurrent state | `abuf[layers[l].state]`, 48 heads × 128 × 128 f32 | 3.146 MB × 36 = **113.25 MB** | must be restored |
| DeltaNet convolution ring | `abuf[layers[l].win]`, 3 × 16512 f32 | 198.1 KB × 36 = **7.13 MB** | must be restored |
| PLE convolution ring | `abuf[aHist]`, 9 × 10240 f32 | **368.6 KB** | must be restored |
| KV cache + indexer raw keys | `hbuf[hK,hV,hIdxRaw]`, per layer per cell | 12 layers × ~2.3 KB a token | **free** — `SetPast` |
| pooled indexer blocks | `hbuf[hIdxK]`, one per `Ratio` cells | — | **free** — self-healing |
| the host id list | `Graph.ids` | — | **free** — truncate |
| hyper-connection stream, all arenas | per pass | — | **free** — not carried |
| **total that must be restored** | | **120.75 MB** | |

**The KV cache is free, and `gpu_attn.go:388` already says why**: nothing
is cleared on a reset because "a cell past `past + nTok` is masked out of
every score and every softmax", so a rewound cell is unreadable rather than
stale, and re-running the position overwrites it with the same bits. The
pooled block table is free for a subtler reason: `blockRange` is
`[past/Ratio, (past+rows)/Ratio)` with a *floor* at both ends, so a block is
only written once it is complete, and re-running the positions rebuilds it
from the re-written cells.

### 3.2 Finding 4 — both convolution rings are corrupted by *any* rejection, including one token

This is the part a design that only checkpointed the recurrent state would
get wrong, and nothing but a batch-split test would catch it.

`llm_seq_hist.comp` addresses a ring by **absolute position modulo the ring
length**, and its header is explicit that this makes it a pure write: "any
of those that predate the run are already in the slots they belong in". That
is true going forwards and false going backwards. The DeltaNet's ring is
`hist = Conv - 1 = 3` slots and its taps reach 0, 1, 2 and 3 tokens back
(`llm_dn_conv.comp`), so the tap at −3 reads slot `(p−3) mod 3` — which **is
slot `p mod 3`, the slot position p itself will be written into**. A
rejected draft token at position P+j writes that slot; re-running position
P+j then reads its own leftover as its oldest tap.

The PLE's ring is worse-conditioned and fails the same way: `ConvHist =
(Conv−1) × NGram = 9` slots with dilated taps at 0, 3, 6 and 9 back, so a
rejected draft three positions past the rewind point clobbers the −6 tap
and one at six clobbers the −3.

So: **one rejected token is enough, at M = 2**, and the failure is silent —
wrong activations, no error, a perplexity delta nobody would attribute.

### 3.3 The design: ping-pong the state, defer the ring write

Two mechanisms, and between them the rollback costs nothing per round.

**(a) The recurrent state ping-pongs, and it is free.** `llm_dn_scan.comp`
loads a lane's slice of S into registers once before the token loop and
stores it once after (`s[r] = act[sbase + …]` at the top, `act[sbase + …] =
s[r]` at the bottom). It never reads the state buffer again inside the loop.
So giving the store a *different* base than the load costs **zero extra
traffic** — the same 113 MB read and 113 MB written that the honest decode
budget already charges. Two state slots per layer, a pass reads slot `cur`
and writes slot `1−cur`, and a round that is committed flips `cur` while a
round that is rejected does not. Residency: +113.25 MB on 78.5 GB.

**(b) The ring write is deferred instead of checkpointed.** The ring is
written by one dispatch at the end of each block (`dn.hist`, `ple.hist`),
reading the pass's own activation arena, which still holds every row when
the pass ends. So the verification pass simply **omits** those 37
dispatches, and the commit issues them with `tokens = j`, the accepted
count, instead of M. The rings then hold exactly the positions the
committed prefix ends on. Cost: 37 dispatches of ~1.1 us (P1's attribution:
`dn.hist` 1.1 us, `ple.hist` 1.1 us) — **about 40 us a round**, against 7.5
MB of copying it replaces.

**(c) Partial acceptance re-runs, and §1 says that is the cheap option.**
When j < M the committed state must be the state at P+j, and neither
ping-pong slot holds it — the pass folded all M tokens into one update. The
alternatives are to snapshot S after every token (M−1 extra writes of
113 MB, ~1.4 ms a round at M = 4) or to re-run the accepted prefix as part
of the *next* round's pass. **Re-running wins, and it wins because of the
whole point of speculation**: the next pass is M rows regardless, and the
projected table in §2 says rows 2 and 3 of a pass cost 0.16x and 0.21x of a
step where the first costs 1.00x. Carrying j extra rows into the next pass
is cheaper than 1.4 ms of state writes at every M the stage would run at.

So a round is: `past` rewinds to the last committed position, the next pass
runs `[re-run the accepted prefix] ++ [the new drafts]`, and the state slot
only flips when a pass is accepted whole.

### 3.4 What the state object API has to grow

Small, and named here so the build stage does not re-derive it:

- `DeltaNetGPU`: a second state slot and a `stateSlot` field; `SetPast`
  gains nothing, but the scan's push block gains a destination offset
  (`ssmStateDstOff`) and `llm_dn_scan.comp` one changed base in its final
  store. `Reset` and `SetState`/`State` write the current slot.
- `DeltaNetGPU` and `PLEGPU`: a `SkipHist bool` that drops the `hist`
  dispatch from `graph()`, and a `CommitHist(n int)` that issues it alone.
- `Graph`: `Commit(n int)` / `Rewind(n int)` over `past`, `ids`, the two
  blocks above and `attn.SetPast`. Nothing else in the graph carries state.
- **P1c's pre-recorded buffer is unaffected at M = 1** and does not apply to
  a verification pass, which is a different row count; `TestDecodeDispatchDiff`
  stays the instrument that proves it.

### 3.5 The gate this design owes

`TestSpeculationRewindIsTheSequence`: prefill a prompt, snapshot the text a
plain greedy loop produces, then run the same tokens through
`Extend(M rows) → Rewind(j) → Extend(...)` with j varying 0..M−1, and
require **the same logits to the last place**. The two convolution rings
are what this test exists for: without §3.2's rewind it fails at j = M−1
with a single rejected token, which is the cheapest possible demonstration
that the finding is real.

---

## 4. The draft head, and the oracle that does not exist

`models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`, 34
tensors, 3.879 B parameters, 2.78 GB. `qwen4exp.nextn_predict_layers = 1`
and `block_count = 49`, so the head is **one extra layer, `blk.48`**, and
the config confirms its shape: `mtp: {hybrid: true, layer_types:
["full_attention"], num_hidden_layers: 1}`.

**Three things about it are good news.** It is a *full-attention* layer, so
it carries **no DeltaNet state**; `ple_layer_ids` is `[2]`, so it runs **no
PLE block and has no second ring**; and `mtp_use_dedicated_embeddings` is
`false`, so its bundled `token_embd` (Q4_K) and `output` (Q6_K) are copies of
the trunk's and need not be staged — the real added residency is **1.90 GB**,
almost all of it the one layer's 512 experts. The draft's own rollback is a
one-layer KV truncate, which §3.1 already shows is free.

**And one thing is not.** The `nextn` wiring has no reference implementation
anywhere this repo can reach:

- **llama.cpp does not implement it.** `src/models/qwen4exp.cpp` has no
  `nextn` symbol at all; the tensors are never created and `blk.48` is never
  loaded. (`qwen3next.cpp` *does* have a `graph_mtp`, and it is the nearest
  relative — `eh_proj(concat(enorm(emb), hnorm(h)))` into an ordinary trunk
  block, then `shared_head_norm` and the lm head — but it is a model without
  hyper-connections.)
- **transformers does not either.** `models/qwen4_exp/modeling_qwen4_exp.py`
  carries `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`: the weights are
  discarded on load.

So D5 — *llama.cpp is the oracle* — has no answer here, and this is the first
block in the vertical built without one.

### 4.1 The shape contradiction, and the hypothesis to test first

`blk.48.nextn.eh_proj` is `[5120, 2560]`, i.e. K = 5120 = 2 × n_embd, which
is the same shape every other architecture in llama.cpp creates. But
`blk.48.nextn.hnorm` is `[10240]`, not `[2560]` — because this model's
residual is four hyper-connection streams. Concatenating a 2560-wide
`enorm(emb)` with a 10240-wide `hnorm(h)` gives 12800, not 5120.

The reading that makes every tensor in the shard account for itself:
`hnorm` is `[2560, 4]` — a per-stream gamma, exactly as `qwen4exp.cpp`
creates `hc_head_norm` with `{n_embd, hc}` and `TENSOR_ALLOW_RESHAPE` — and
**`eh_proj` is applied once per stream**, `concat(enorm(e), hnorm(h)_s)` for
s = 0..3, producing a fresh `[2560, 4]` hyper-connection state for the MTP
block. `blk.48`'s own `hc_attn_*` and `hc_ffn_*` then run the layer and
`blk.48.nextn.hc_head_{norm,down,up}` is its output mixer, mirroring the
trunk's `hc_head_*`. Nothing is left over and nothing is missing.

It is a hypothesis. The competing readings — a reduction to 2560 before
`eh_proj`, or `hc_head_*` at the input rather than the output — leave a
tensor unexplained, which is why this one goes first.

### 4.2 Finding 5 — acceptance is measurable **now**, with no rollback and no kernel

The decisive number for the whole stage is the acceptance profile, and it
does not need the verification pass, the rollback, or the R-row kernel. It
needs the draft head and an ordinary greedy decode loop:

> At step t the main model emits `x_{t+1}` and the trunk leaves its hidden
> state `h_t`. Run the draft head on `(h_t, x_{t+1})` and record its
> prediction for `x_{t+2}`; chain it on its own output for `x_{t+3}` and so
> on to depth 8. At step t+1 the main model emits the true `x_{t+2}`.
> Compare.

Over a few hundred tokens that gives `a₁ … a₈` exactly, as a **passive
observer** of a loop that already works. It costs 8 draft steps a token —
slow, and irrelevant, because it is an experiment and not a product path.

It also **gates §4.1's wiring**: a correct wiring gives `a₁` somewhere
around 0.6-0.8, and every wrong one gives approximately zero. No numerical
oracle is needed to tell them apart.

---

## 5. What P5 should do, in order

The review's order was *design, then the draft head, then the loop*. The
measurement in §1 re-orders it, and puts the cheapest decisive thing first.

~~**P5a — the draft head as a passive observer.**~~ **Done, 2026-09-19.**
`a₁ = 74.0%` against a zeroed-hidden control at 9.8%, the gate met with room;
§4.1's wiring hypothesis **confirmed by a five-arm race** (and the
concatenation order found to be load-bearing — flipped scores 0 of 256); and
the stage priced at **~1.33x at depth 1**, a loss at every depth without P5b.
[Write-up](p5a-draft-head.md)

**P5b — the R-row decode kernels.** `llm_moe_gemv.comp`,
`llm_moe_router.comp`, `llm_gemv.comp` and `llm_hc_gemv.comp` with R rows
per tile: R accumulators, R A-vectors in LDS, one weight read. **P5a scopes
it down: R = 2 is enough for P5**, because the optimum depth is 1 and a
depth-1 verify pass is two rows; R general is P6's requirement. **Gate**: the
§2 table measured rather than projected — a `-graph -tokens 1,2,3,4,6,8`
pass at or under 1.2x at M = 2 and 1.9x at M = 8, and bit-comparable output
against the GEMM arm at the rounding `PinSchedule` already documents. **This
stage is P6's first stage too**, and it is worth building even if P5a comes
back with a low acceptance rate.

**P5c — the rollback and the loop.** §3 as designed: the ping-pong state
slot, the deferred `hist` dispatches, `Graph.Commit`/`Rewind`, and
`TestSpeculationRewindIsTheSequence` as the gate. Then the loop itself.
**Gate**: speculation is lossless — token-for-token identical text at
temperature zero against the plain loop — and a measured multiplier at
ctx 512 *and* at a long context. **P5a amends the second half twice**:
acceptance turns out **not** to be context-dependent here (73.4% at ctx 320
against 74.0% at 2048), and a multiplier measured on greedy self-generated
text measures the *sampler* — the trunk loops a paragraph verbatim within a
few dozen tokens at temperature zero, which reads as 92.2% acceptance. Take
the multiplier teacher-forced, or read the text it ran on.

### Carried forward, priced and not taken

- **The draft's lm head is 81% of a draft step** (0.437 GB of 0.538). A
  draft that scored only a candidate subset of the vocabulary would make
  deep M affordable and move §2.1's optimum right; it is also the only
  lever on the draft side worth anything. Not designed here.
- **The router's M = 1 rung is a 5.9x cliff of its own** — `k40` at 15.2 us
  becomes a GEMM at 89.8 — on a 2.63 MB matrix that never leaves the MALL
  (D12's stopping condition). It is small in absolute terms and it comes
  along free with P5b.
- **The 48-layer verification pass is owed.** §1's table is 24 layers, where
  the M-independent fixed costs (`head` 2.4 ms, `ple`, `move`, the host) are
  24% of a step against 13% at 48, so the real 48-layer ratios are **worse**
  than 3.24x/4.32x by roughly the difference between those shares. It was
  measured at 24 because 48 layers plus the arenas is ~80 GB against 80 GB
  of MemAvailable on this machine today.
