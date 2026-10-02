# LLM — the live plan and open items (qwen3.8-flash-next)

> **Opened 2026-10-02, when the root `TODO.md` was retired.** This is the
> text vertical's live state of play: the decode plan (P19–P25), the open
> items, and the instrument rules. It was `TODO.md`'s "Text generation"
> section, moved whole; code comments citing `TODO.md` with a **P-stage**
> (P20f, P21a, P22b, P22c, P21d …) resolve here. The closing record of
> L0–L9a, P0–P5c and decisions **D1–D21** is
> [`llm-vertical.md`](llm-vertical.md); each closed P-stage's write-up is its
> own `p*.md` file. Batched throughput is
> [`concurrency.md`](concurrency.md), image input
> [`llm-vision.md`](llm-vision.md). Rewrite this file rather than append to
> it; a closed stage's write-up goes to its own file.

## At a glance

| | |
|---|---|
| model | qwen3.8-flash-next (180 B, 6 B active) |
| headline, measured | decode **36.0 tok/s through `-gen`, 34.2 through the server at every `-llm-batch`** (P16, against 25.8 at the shipped 4096), at **+1.74%** perplexity; prefill **1403.9 tok/s at 8192 rows, 3.59x** (and **1199 through the server** at 4096), still climbing where llama.cpp plateaus — **and 1.10–1.17x on top of that from KERNELS.md G5a (2026-09-29, §2.11): the MoE up GEMM's K loop as a software pipeline, `cmd/llm -graph` same-hour A/B 1194/1206 → 1407/1408 tok/s at 2048, 1317/1339 → 1494/1495 at 4096, 1306/1345 → 1459/1462 at 8192, and 1422 / 1506 at 2048 / 4096 with the IQ4_NL down pipeline on top (G5b)*, and **1.07x again from G5c (2026-09-30, §2.12): a short last tile an expert, same-hour A/B 1417/1419 → 1514/1513 tok/s at 2048, 1500/1502 → 1549/1543 at 4096, 1461/1467 → 1481/1483 at 8192**; and G5d (2026-09-30, §2.13, the nibble as an f16 denormal, bit-identical) (same-hour A/B 1513/1513 → 1521/1523 tok/s at 2048, 1549/1547 → 1554/1557 at 4096, 1484/1483 → 1491/1488 at 8192: a rounding of the prefill number, recorded so the next session does not re-run it); **P25 (2026-10-01): heads-on-rows by the share of a chunk past the selection width, not by where it starts — 8192 from cell zero 1519/1523 → 1606/1601 tok/s (1.054x), 2048 and 4096 unmoved; the ladder today 1568 / 1599 / 1604 at 2048 / 4096 / 8192**; **128 000 cells prefills at 1011 tok/s at ubatch 2048 and 1176 at the served 4096** (P17, against 834 and 900; falloff 0.88x), decode **32.2 tok/s at 128k** and **34.4 at depth zero** (P16/P17), falloff to 128k **0.94x** |
| open | **token generation is the focus from 2026-09-30: P19–P24 in the text section; P19 and P21a done the same day (the step attributed: 16.4 / 4.3 / 3.0 / 1.1 ms; the head and `hc.up` — the two projections still on the padded GEMM at one row — moved to GEMVs, 37.5 → 38.7 tok/s at 25.8 ms a step; P21b's load hoisting measured dead; P22a the shared expert as the eleventh tile of the routed MoE dispatch, 38.8 → 39.2 tok/s at 25.5 ms; P21c the MoE down GEMV's five serial round trips a lane hoisted and A staged sixteen bytes a load, 39.1 → 39.8 tok/s at 25.13 ms, both down rungs re-laddered and confirmed), next the rest of P22, speculation re-priced at 1.23 steps a two-row pass**; concurrency and batching (P6, now [`research/concurrency.md`](concurrency.md): three full-context slots, a priority scheduler, per-slot prefix checkpoints and batched decode, all done); the gathered attention at 34% of matrix-core peak with its four bounds eliminated; ~~`hc.cn` at prefill~~ (G6, §5.4: the grid's walk, 184 → 200 GB/s, prefill 1520 → 1538 tok/s at 2048 and decode 36.87 → 37.35); decode is now fusion at 1-2% a step; the 196 moves went in P22b (2026-09-30, 196 → 4 and no zero rows at decode, 25.13 → 24.42 ms, 41.0 tok/s); **P22c + P21d (2026-10-01): the router's tail as one bit-exact dispatch and `ple.kv` — the third one-row GEMM — on the GEMV, 24.42 → 24.19 ms, 41.4 tok/s; the weightless list is a floor of launches now and P20 is next**; **P20f (2026-10-01): speculation served with `-llm-draft` — 56.8 tok/s greedy and 50.1 sampled through the server against 41.1 (1.38x / 1.22x, SPEED-Bench), and the sampler's 2 ms a token gone**; **P20g (2026-10-01): speculative sampling — sampled requests (no seed) 49.2 → 56.6 tok/s served, 1.38x, the greedy rate**; **P20h (2026-10-01): depth three built (four-row decode GEMVs, the ring past three rows) — 1.53x / 62.9 tok/s in `cmd/llm` against 1.49x, but 55.5 against 56.9 served (thinking text accepts less), so the default stays 2** |

## State of play

**Where it stands.** Phases 1 and 2 are done on both axes and every P-stage
through P5 is closed. The shipped configuration is **D19 + D20 + D21**
(4.5-bit dense bank with a fifth bit on three families, the MoE rows
transcoded, `ffn_down_exps` at IQ4_NL): 4.132 GB a token against a 58.6
tok/s ceiling, perplexity 4.0992 (+1.74% of our own 4.0289, which is itself
−0.13% against llama.cpp's at identical weights). Decode 36.19 tok/s,
prefill **1207.7 tok/s at ubatch 2048 and 1403.9 at 8192** (P11, P12). Served with
prefix reuse; a second turn extends the graph's state rather than
re-prefilling.

**Speculation (P5) was built, lossless, and parked at 0.95x** (superseded: P20 below takes it to 1.48x and serves it). The rollback
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

**The plan from here: token generation (set 2026-09-30, not started).**
Prefill has had eight stages and the KERNELS.md G-stages (1194 → ~1520
tok/s); decode got nothing from the G-stages, because they are all WMMA
GEMM work and a decode step is GEMVs at 91 W, bound by memory. Decode is
**36.2 tok/s at depth zero** (27.6 ms a step, `cmd/llm -batch 1`, after
C5's quad GEMV), 35.5 served, ~32 at 128k, against a **58.6 tok/s byte
ceiling** (4.132 GB a token): 62% of it. The one outside number
(`halogen-flash-server`, KERNELS.md §0.7) claims 37.6 serial, which we
match, and **46–56 with a draft head plus prompt lookup**, which is the
whole of their lead. KERNELS.md's "decode GEMV: 96–103% of the bus, done"
is a cache-hot ladder figure: in the model the weight streams measured 178
GB/s (P1) and ~170–200 (P16), and C5 found 6% by counting loads in
`llm_moe_gemv.comp` on 2026-09-23. What carries over from the G-stages is
the method: price a kernel's parts by their absence, count loads and
instructions in the ISA (`cmd/probe`), and screen in the model on a cold
bank.

A step's 27.6 ms, **reconstructed from P1, P16 and C5 and not measured as
one table**: ~17 ms of weights at the bus rate (the floor at this format),
~5–6 ms of weight streams running under the bus, ~3.3 ms of dispatches that
stream no weight (898 of 1431 are under 12 µs; the 196 moves are 0.54), and
~1–2 ms of host, mostly the n-gram gather. The stages, in order:

- [x] **P19 — re-attribute the decode step.** **Done 2026-09-30**
  ([write-up](p19-decode-attribution.md)). A step is **26.66 ms**
  (37.5 tok/s, two runs 0.15% apart, every label within 0.014 ms): **16.4
  ms of weights at 242 GB/s, 4.3 ms of streams under the bus (against 227),
  3.0 ms of weightless dispatches, 1.1 ms of host** — the reconstruction
  above was right. The weight streams average 182 GB/s; the four big dense
  GEMVs are at 195–211 (0.8 ms lost between them, and the Q4 arm's loads
  are already vectorised by the compiler — one `b128`, one `b64`, two
  `b128` a step), the MoE at 193/174 (1.3 ms), **and the two projections
  still on the padded GEMM at one row are the largest single losses: the
  lm head at 185 GB/s (2.37 ms, the biggest dispatch of the step) and
  `hc.up` at 125 (1.75 ms over 97 dispatches)**. The small dispatches
  (`hc.up`, `hc.down`, the shared expert, `ple.kv`: 340 a step of 1–2.3 MB)
  run at half the bus, which is P8's latency rule at family scale. Clock
  under decode: 97 W, shader clock p50 795 MHz — memory-bound, no matrix
  clocks to buy. A two-row one-sequence pass is **1.23 steps** (P5c's 1.36),
  three rows 1.37; three conversations 1.39 / 1.67. **First change, same
  hour A/B: the head at one row on its GEMV (`llm_gemv.comp` K1, staged
  since L8c and never wired), 2.37 → 2.06 ms at 212 GB/s, the step 26.66 →
  26.37, 37.5 → 37.9 tok/s (+1.1%)**; `GraphOpts.HeadGEMM` / `LLM_HEAD_GEMM=1`
  is the control, four bit-identity gates pin the GEMM, and
  `TestGraphHeadDecodeGEMV` prices the kernel (rms 7e-6 of the logits' max).
  **At one row only, until the same night's postscript**: a three-slot
  batched step ran the GEMM head and rounded unlike its solo steps
  (`TestGraphImageDecode`); the head now plans the GEMV to GEMVMaxRows
  like every other decode GEMV, and the gate covers three rows.
  **P21a done the same evening: `hc.up` on a GEMV** (`llm_hc_up_gemv.comp`,
  a workgroup a feature block, five waves each a share of the column's
  k-tiles, the four-stream collapse in the wave; `LLM_HC_UP_GEMM=1` the
  control): 17.9 → 13.3 µs a dispatch, 125 → 169 GB/s, the step 26.10 →
  **25.8 ms, 38.7 tok/s** (five interleaved arms 25.78–25.88). **P21b
  measured dead**: the big GEMVs with their slab's loads issued together
  (17–23 in flight, bit-identical) ran 0.7 ms *slower* — the head +0.3,
  `hc.down` +0.23 — for 60–72 VGPRs against 48 and a branch-free record
  decode at 800 MHz; removed. The four big GEMVs are not load-bound.
- [ ] **P20 — speculation, re-planned 2026-10-01 against llama.cpp's MTP**
  ([research/p20-llamacpp-mtp.md](p20-llamacpp-mtp.md)). The only
  lever with a multiple, parked at 0.95x (P5c). llama.cpp PR 29761 (open,
  2026-09-30) implements the same head with **P5a's wiring exactly** and
  reads **1.55x on a DGX Spark at `--spec-draft-n-max 3`**: 28.36 → 43.88
  tok/s over 24 SPEED-Bench prompts, acceptance **0.64 a drafted token**
  (coding 0.61, qa 0.58, rag 0.69, writing 0.67), ≈2.9 tokens a round for
  ≈1.9 steps. Four things in its loop differ from ours, and one of the four
  is a defect here:
  - **Done 2026-10-01: P20a and P20d's first half** ([§6 of the
    note](p20-llamacpp-mtp.md)). The draft is primed over the
    prompt from `Graph.ForwardResidual` and catches up one owed cell a
    round (`Speculator.CatchUp`, `SPEC_CATCHUP=0` the control); `eh_proj`
    runs on the device and the draft step is one submit (`MTPHead.eh`,
    `LLM_MTP_HOST_NEXTN=1` / `LLM_MTP_UNRECORDED=1` the controls,
    `TestMTPSeedDeviceIsTheHost`). Acceptance at 1024 prompt cells 68.2 →
    90.6% (memorised wikitext: an upper bound), unchanged at 34; the draft
    5.6 → 3.5 ms a round. **`-spec` 1.00x → 1.29x long (41.70 → 53.95
    tok/s) and 1.08x short (42.00 → 45.26)**, two passes each, acceptance
    count for count the host path's. Spec leaves plain at one near-tie a
    run (two-row rounding; recorded, not fixed).
  - **Done 2026-10-01: P20c at depth one** ([§7 of the
    note](p20-llamacpp-mtp.md)). Row 0 of a pass is always
    committed, so the pass also folds row 0 alone into the slots it *read*
    (the scan's in-place store of S after token 0, the ring write's row 0
    into `SEQ_HIST_PREV`; flag `SEQ_KEEP_FIRST` = `ldaLo`), and a rejection
    is `Graph.KeepFirst` — at P+1, drafting again, no recovery round, no
    extra memory (`Graph.SpeculateFirst`, `Speculator.Partial`,
    `SPEC_PARTIAL=0` the control). Gate
    `TestSpeculationKeepFirstIsTheSequence` bit-exact, control rms 0.93.
    **`-spec` short 1.08x → 1.19x (42.00 → 49.82 tok/s), long 1.29x →
    1.31x (41.57 → 54.65)**, both arms self-reproducing. Covers keeping
    row 0 of a pass up to three rows; keeping 0–1 of three needs a stored
    plane (built in P20b below). Then P20e (SPEED-Bench, the number to
    quote).
  - ~~**P20a — the draft's KV cache is complete there and not here.**~~
    llama.cpp runs the draft layer over every row the trunk runs — the
    prompt and each verified batch — paired with the trunk's residual one
    position back, no lm head, and trims the rejected rows.
    `Speculator.Start` resets the draft and never primes it over the prompt,
    and `Next` writes a draft cell only on a speculating round, so the draft
    attends over **unwritten cells for the whole prompt and at every other
    position**. P5c's 65.6% / 46.9% against P5a's 74.0% (same head, complete
    cache) is this before it is the narrowed trunk — §4.2 was confounded with
    it. Fix: the catch-up rows in the draft dispatch (the committed rows
    since the last draft plus the new one, logits for the last), the prompt
    primed at `Start` from the prefill's residual arena a ubatch at a time
    **on the device** (at 128k the residual is 5 GB; not through a mapped
    buffer). Gate: the loop's acceptance on a teacher-forced sequence equals
    the observer's count for count.
  - **Done 2026-10-01: P20b — depth two built, measured, not adopted**
    ([§8](p20-llamacpp-mtp.md)). Keep-any-prefix up to three rows
    (`GraphOpts.SpecRows`, mid slots, `Graph.Keep(k)`,
    `Speculator.Depth` / `-spec-depth`, `SPEC_TRACE`), gate bit-exact with
    a mid-slot sabotage that fails. **Depth 2: short 1.11x against depth
    1's 1.19x, long 1.31x = 1.31x**: the 3-row pass is 1.52 steps against
    1.29 (the MoE's distinct experts, 17 → 24, not a kernel) and the second
    draft step +0.15; a₂|a₁ 62.5%. Depth 1 stays default. Next lever: the
    draft's lm head (2 of 3.7 ms) over a trimmed frequent vocabulary
    (FR-Spec) — price its acceptance on `-mtp` first; then P20e.
  - **Done 2026-10-01: the draft's lm head over a vocabulary prefix**
    ([§9](p20-llamacpp-mtp.md)): the draft proposes from the first
    65 536 ids (BPE id order covers 96.6% of wikitext, 98.3% of Go; counted
    wikitext frequencies fail on code), and `HeadGPU.RunCols` runs the
    trunk's own head GEMV over K/16 tiles — no new weights or kernel
    (`TestHeadGPURunCols` bit-exact). Draft 3.71 → 2.22 ms; **`-spec` short
    1.19x → 1.24x (51.93 tok/s), long 1.31x → 1.32x (55.14)**; acceptance
    count for count the host-cut pricing. `SPEC_DRAFT_VOCAB=0` the control.
    Next P20e (SPEED-Bench, the number to quote), and serving the loop.
  - **Done 2026-10-01: P20e — SPEED-Bench** ([§10](p20-llamacpp-mtp.md)):
    20 complete single-turn prompts (coding/multilingual/qa/rag/writing ×
    4; most of the split is placeholder rows), chat template, 1024 tokens,
    two passes (`cmd/llm -spec -spec-set models/speed-bench/qualitative-20.jsonl`).
    **Depth 1: 41.3 → 55.4 tok/s, 1.34x (a₁ 86%); depth 2: → 61.3 tok/s,
    1.48x** (coding 1.58x, multilingual 1.35x); texts coherent, both arms
    self-reproducing on 20/20. Depth 2 is the default now
    (`llm.SpecDepthDefault`); prose at 73% gives back 0.04. The 2051-cell
    draft refusal is gone (the PR writes blk.48 the trunk's ratio). **Next:
    serving the loop** (the server does not speculate; `GraphOpts.Slots`
    and `Speculative` share the slot index), then depth 3 (~+5%, needs a
    four-row pass and a ring past three rows).
  - **Done 2026-10-01: P20f — the loop served** ([§11](p20-llamacpp-mtp.md)):
    `serve -llm-draft <mtp gguf>` (`-llm-spec-depth`, default 2). Sequence
    slots and speculation planes compose — the DeltaNet and PLE blocks index
    (sequence, plane), a batched row reads its sequence's committed plane
    (`TestSpeculationSlotsAreSequences`, sabotage fails) — the draft has a
    cache a slot, and `NewSlotSpeculator` runs the loop in pieces (`Prime` a
    prefill chunk from `ExtendResidual`, `Round` with the request's own
    sampler drawing each verified row in order — lossless at any
    temperature — `Owe` a plain step's row, `Settle`, `Rewind` to a
    checkpoint; `TestServedSpeculationIsPlain`). The scheduler speculates a
    conversation that decodes alone and batches the rest as before
    (`TestLLMSpeculationIsPlain`: with and without the draft, identical
    streams greedy, sampled, two turns, concurrent, restored). **Served,
    SPEED-Bench 20, 512 tokens: greedy 41.06 → 56.83 tok/s (1.38x), default
    sampling 41.06 → 50.08 (1.22x)**; prefill +9% (priming through the
    host). Spec leaves plain greedy text only at measured near-ties (top-2
    margin 0.003–0.033 against medians 2.7–11, `SPEC_MARGIN=1`). **And the
    sampler cost 2 ms a token** (a 1 MB index and a quickselect over the
    row): one-pass top-k, plain sampled 37.93 → 41.06, the same texts.
    ~+5 GB on the deployed LLM line. **Next:** priming on the device, depth
    3, acceptance on thinking text.
  - **Done 2026-10-01: P20g — speculative sampling** ([§12](p20-llamacpp-mtp.md)):
    a sampled request's draft *samples* its proposal from its own row through
    the request's cuts (`Sampler.DraftSampler`/`Propose`) and the trunk keeps
    it with probability min(1, p/q), else draws from max(0, p − q)
    (`Sampler.Verify`, `Speculator.RoundDraw`): the same distribution of
    text, the draft kept Σ min(p, q) of the time instead of p(argmax).
    Only for requests without a seed (a seeded one stays the plain loop's
    stream); `LLM_SPEC_ARGMAX_DRAFT=1` the control;
    `TestSamplerVerifyIsTheDistribution` the gate (sabotage fails).
    **Served, SPEED-Bench 20, default sampling: 41.1 → 49.2 (argmax
    proposal) → 56.6 tok/s, 1.38x**, 2.36 tokens a round, which is greedy's.
    **Next:** priming on the device, depth 3, acceptance on thinking text.
  - **Done 2026-10-01: P20h — depth three** ([§13](p20-llamacpp-mtp.md)):
    priced first with `SPEC_PROBE` (an uncounted probe draft a round, from
    a depth-2 run: a₃|a₂ 0.837). Then built: `GEMVMaxRows`/`MAXROWS` 4 in the
    five decode GEMVs (one row unchanged, 24.17 → 23.93 ms over three pairs),
    and the ring write keeps any prefix of a run longer than the DeltaNet's
    three-row ring (`TestSpeculationKeepFirstIsTheSequence` at four rows,
    bit-exact, sabotage fails). **`cmd/llm -spec-set`: depth 3 1.53x / 62.9
    tok/s against depth 2's 1.49x / 61.1** (coding 1.70x). **Served it
    loses:** 56.88 → 55.54 greedy, level sampled, because the thinking text
    accepts 2.71 tokens a round against a 2.77 break-even. The default stays
    2, and `-llm-spec-depth 3` is there. The confidence stop
    (`SPEC_STOP`, τ 0.6) priced at +5–9% and measured a wash: −2% at depth 2,
    ±0 at depth 3. **Next:** acceptance on thinking text, then depth 3 again.
  - ~~**P20b — depth 2 or 3, not 1.**~~ P5a's optimum was computed on
    2026-09-19's pass costs and wikitext acceptance; at 0.64 a drafted token
    and P19's 1.23 / 1.37-step two- and three-row passes the optimum moves.
    Measure `cmd/llm -graph -tokens 1,2,3,4` on the shipped banks (one
    sequence, the right instrument here); a four-row pass needs
    `GEMVMaxRows` 4 (`MAXROWS` in the three decode GEMVs and the fused
    router, a four-at-a-time rung in `TestGraphIsAChunkSplit`). The MoE's
    expert growth bounds the depth; P20b measures it on loop traffic rather
    than the design pass's repeated-prompt table.
  - ~~**P20c — partial accept by snapshot planes, not a recovery round**~~
    (done at depth one above, by an in-place row-0 store rather than planes)
    (P5c §4 item 1, now with a reference design): llama.cpp creates the
    target context with `n_rs_seq = n_max`, the gated-delta-net op writes
    the state after each of the last K = n_max + 1 rows into K planes (the
    conv ring and the PLE ring likewise, K copies), and a rollback of j sets
    a plane index the next read takes; row 0 of a pass is always committed,
    so no ping-pong. Here: `llm_dn_scan.comp` stores S after each of the
    last K rows ((K − 1) × 113 MB extra a pass, ~0.07 step at K = 4, against
    a recovery pass of ~1.2), K rings in `llm_seq_hist.comp`'s ping-pong
    arm, `Rewind(j)`; residency 241 → 483 MB, still `GraphOpts.Speculative`;
    `TestSpeculationRewindIsTheSequence` gains j = 1..3 of 4.
  - **P20d — the draft step at its byte floor.** 5.71 ms is 0.236 of
    today's 24.19 ms step (P5c §4.2's effect again) against a floor of 0.13;
    three a round is 0.71 steps against 0.39. `nextn` on the device (four
    host matvecs, 2.2 ms) and the draft step recorded (13 submits) — P5a's
    owed items, on the critical path now — and depth as a parameter.
  - **P20e — the measurement, the PR's way**: free-running on SPEED-Bench's
    qualitative prompts (`nvidia/SPEED-Bench`, the categories above) through
    the chat template at temperature zero, the text read for P5a's loop
    artefact, two runs, the plain-against-plain row printed (`Reset`'s owed
    test, P5c finding 6, first), the shipped banks. The converter in the PR
    gives `blk.48` **the trunk's compress ratio**, so the 2051-cell refusal
    in `NewMTPHead` comes out and the long arm runs past 2048; D5 is back
    for the head once a build with the PR exists. Unverified arithmetic at
    24.19 ms, the PR's acceptance and the draft at the floor: depth 1
    **1.26x**, depth 2 **~1.4x**, depth 3 **~1.5x if a four-row pass is ~1.5
    steps** — 52 / 58 / 63 tok/s, inside the 46–56 the outside number
    claims; with P5a's wikitext profile instead, 1.34x / 1.32x at depths 2
    and 3, so the workload's acceptance is the swing.
  - **P20f** prompt-lookup drafting, demoted: a free extra draft to combine
    with the head (llama.cpp runs several types in order), not the first
    thing; the host-only replay observer stands as written before, and the
    draft's confidence stop (`p_min`, unused in the PR's table) is a lever,
    not the result.
- [ ] **P21 — the G-treatment on the decode GEMVs.** P19's `lost` column
  sets the order; the first suspect is retired (the Q4 arm's loads are
  vectorised as compiled) and so is the second (P21b above: issuing a
  slab's loads together is slower on the big grids). ~~P21a `hc.up`~~ done
  (above, +0.44 ms). ~~Left: **P21c `moe.down` (174 GB/s, +0.6 ms) and
  `hc.down` (162, +0.4)**~~ — **P21c done 2026-09-30** ([Finding
  5](p19-decode-attribution.md)): both rungs re-laddered in the
  whole model (`LLM_MOE_DECODE_PLAN`, `LLM_HC_DOWN_SLABS`) and the shipped
  ones confirmed (v16w4; 32 slabs, with 40 D12's whole multiple at 1.9x and
  160 losing on its reduce); the ISA showed `moe.down`'s lane as five serial
  round trips, each two loads and a `vmcnt(0)`, behind a half-a-load A
  staging — `llm_moe_gemv.comp` now issues every trip's words before the
  first unpack (the guards branch-free, or the compiler sinks the loads
  back) and stages A eight halves a load: three interleaved pairs 25.57 /
  25.50 / 25.56 → **25.09 / 25.13 / 25.17 ms, 39.1 → 39.8 tok/s**,
  `moe.down` 53.3 → 49.2 µs, `moe.up` 104.5 → 100.4 from the staging alone.
  The same treatment on the up mode's quad loop measured +1.5 µs and was
  reverted (P21b a third time). `hc.down`'s 0.4 ms has no idea left; the
  shared expert's two dispatches are P22a's eleventh tile. **P21d done
  2026-10-01** ([Finding 8](p19-decode-attribution.md)): `ple.kv`
  was the third projection still on the padded GEMM at one row — the PLE's
  [2560 × 12800] int8 key/value projection, 35 MB at 127 GB/s — and runs on
  `llm_gemv.comp`'s eight-slab rung at one to three rows (`PLEGPU.gemvFor`,
  `LLM_PLE_KV_GEMM=1` the control, `LLM_PLE_KV_SLABS` the screen,
  `TestPLEGPUKVDecodeGEMV` the gate, the schedule pin honoured): 282 → 167
  µs, −0.12 ms a step. Method: knock-out controls, the ISA's load count, the whole model
  and not the block ladder (L8e-2 and C5 both chose a rung in the graph
  that the ladder got wrong), a rate above 242 GB/s is an L3 hit (D16),
  and **VGPRs and VALU count at 800 MHz** (P21b's lesson: a decode kernel
  runs at a third of the clock, so an instruction costs three times what
  the ISA suggests).
- [ ] **P22 — the weightless dispatches and the small ones.** 3.0 ms of
  weightless dispatches (P19's list: `dn.scan` 0.64, the 196 moves 0.54,
  `moe.route` 0.28, `hc.cn` 0.23, twenty more under 0.16), 1–2 recoverable:
  G3's epilogue pattern (a norm, gate or combine folded into the kernel that
  produces its input) and L6c's single arena, which deletes the moves.
  ~~And **the shared expert as an eleventh tile of `moe.up`**~~ — **P22a
  done 2026-09-30** ([Finding 4](p19-decode-attribution.md)):
  `llm_moe_gemv.comp` carries the shared expert as one tile past the routed
  schedule wherever its bank is the routed bank's format (47 of 48 layers'
  up on the shipped plan; layer 2 is Q5_K and the shared down is Q5_1
  against IQ4_NL), `MoEGPU.foldShared`, `LLM_MOE_SHEXP_SPLIT=1` the control,
  `TestMoEGPUSharedFold` the gate (rms 8e-9 against the split, one to three
  rows, arenas dirtied between arms). Same-hour interleaved pairs 25.75 /
  25.74 → **25.56 / 25.51 ms, 38.8 → 39.2 tok/s**: the shared up is 9.2 µs
  inside the routed dispatch against 15.0 on its own, which is the routed
  rate. The down half is priced at ~0.15 ms more (25.31 with
  `down_shexp=iq4_nl`) and waits on P24's eval of that width. Check every
  re-gridded kernel at `rows == 1` (P16's `hc.cn` lesson). **A fold-order
  bug found and fixed the same night** ([postscript](p19-decode-attribution.md)):
  with the up split and the down folded — the checkpoint's own formats,
  never the shipped plan — the folded shared-down tile read the shared
  swiglu rows before the split `shexp.up` had written them (a decode step
  0.137 rms wrong on the default banks). `shexp.up` is recorded before the
  routed down now and `TestMoEGPUSharedFold` runs the mixed arm.
  ~~**The 196 moves**~~ — **P22b done 2026-09-30** ([Finding
  6](p19-decode-attribution.md)): deleted at the boundary rather
  than with L6c's shared arena. The kernel on either side binds the other
  block's arena at binding 11 (`HCLink`, a pipeline per foreign buffer as
  the mover had): `cn`/`combine` read a sublayer's output where it lies at
  any row count, and the decode up GEMV writes the next sublayer's fp16 A
  operand from the register that holds the value. 196 → 4 moves a step
  (the PLE's residual out and back, the final row move, the head's input);
  bit-exact by construction, `TestGraphHCLinkBitExact` the gate,
  `LLM_HC_MOVES=1` / `GraphOpts.HCMoves` the control. Three same-hour pairs
  25.13 / 25.15 / 25.13 → **24.83 / 24.83 / 24.81 ms, 39.8 → 40.3 tok/s**;
  a third of the 0.53 came back as `hc.up` zeroing the consumers' pad rows
  from one wave, so every `InPort` now reports the run itself as its row
  block on the decode plan (its rungs are GEMVs that read ROWS rows) and no
  zero row is written at decode by either arm.
  Three more pairs with both arms so: **24.91 / 24.89 / 24.97 → 24.42 /
  24.39 / 24.43 ms, 40.1 → 41.0 tok/s** — the zero rows had cost the moves
  arm 0.22 ms on their own in the kernels after each move, so P22b is
  **25.13 → 24.42 ms, 39.8 → 41.0 tok/s** in all
  (`results/p22b_link_*.csv`).
  ~~The epilogue fusions~~ — **P22c done 2026-10-01** ([Finding
  7](p19-decode-attribution.md)): which small dispatch folds into a
  neighbour is decided by the neighbour's parallelism, and `hc.cn` (one
  workgroup a stream, 2560 values in registers) would read the combine's
  eleven rows or a reduce's 32 slabs four times over on one CU each, so the
  combine and the split-K reduces stay (priced dead on paper). The router's
  tail folds: `llm_moe_route_decode.comp` is the split-K reduce, the
  softmax and top ten (a wave a token, no barriers), the weights and the
  permutation with its schedule as one workgroup at one to three rows,
  **bit for bit** the three kernels' output (`TestMoEGPURouteFused`,
  `TestGraphRouteFusedBitExact`; `LLM_MOE_ROUTE_SPLIT=1` the control; one
  build a router rung, because the first form's runtime slab bound made the
  reduce forty serial round trips and measured 26.8 µs against 9.9): 0.471
  → 0.383 ms, −0.09 ms. With P21d, three same-hour pairs 24.37 / 24.43 /
  24.43 → **24.16 / 24.19 / 24.19 ms, 41.0 → 41.4 tok/s**
  (`results/p22c_{ctl,new}{A,B,C}.csv`).
  Left of P22: `dn.scan` (0.64 ms) is the fp32 recurrent state read and
  written every layer — bytes, P24's fp16-state item, not scheduling. The
  rest of the weightless list is a floor of launches (8 µs for a dispatch
  that waits on forty loads and ten reductions, 1–3 µs for one that does
  nothing), and only fewer passes — P20 — gets under it.
- [ ] **P23 — the host gather.** ~1 ms a token, 75% of it two steps in 32
  at ~8.7 ms, unexplained. The fix on the table is residency of the 28.8 GB
  table (or of the rows a conversation can reach), a deployment decision
  beside a ~98 GB LLM line; price it before asking.
- [ ] **P24 — fewer bytes.** ~+0.6 tok/s per 0.1 GB at today's efficiency.
  Gated on the downstream task eval below, since wikitext perplexity at
  +1.74% may no longer separate plans; fp16 DeltaNet state (+~1.3 of
  ceiling) is graded by the same instrument.
- [x] **P25 — prefill: which gathered build a chunk takes.** **Done
  2026-10-01** ([write-up](p25-prefill-heads-rule.md)). P17 took the
  heads-on-rows attention only for a chunk that *starts* past the selection
  width (2051), so every first chunk of a prompt ran the token-row union
  kernel, and at the served `-llm-batch 8192` three quarters of its rows are
  past the width: `attn.attn` was 69.5 ms a layer, the largest label of the
  pass. The rule is now the share of the chunk's rows past the width, heads
  from two fifths (`AttnGPU.headRows`): `-attn` layer 118.6 → 79.6 ms at
  8192, whole model **1518.5/1523.1 → 1605.6/1601.4 tok/s at 8192
  (1.054x)**, 4096 and 2048 unmoved, same hour, interleaved twice. Prefill
  was already a reassociation under this kernel (P17); the full `llm` suite
  passes. **What is left at 8192** (labels, 5.10 s a pass): `moe.up` 16.2%,
  `dn.qkv` 12.4% (40 TFLOP/s, at the dense GEMM's rate), `attn.attn` 11.4% —
  the heads build is ~7 TFLOP/s of useful work, S and P through LDS, twelve
  of sixteen rows, so M11c/G4b's transposed, LDS-free shape is the next
  kernel lever (≤ ~5% of a pass); `moe.down` 10.9% (its fp32 output, 840 MB
  a layer, is ~half its traffic and the ordered combine needs it), `hc.cn`
  8.5% (at the bus), `hc.up`/`hc.down` 9.4% together (small-N dequant GEMMs
  at ~19–27 TFLOP/s).

P19, P21a, P22a, P21c and P22b are done (36.2 → 41.0 tok/s on 2026-09-30,
24.4 ms a step), and P22c with P21d (2026-10-01: 41.0 → **41.4 tok/s, 24.2
ms**). P22 is closed but for `dn.scan`, which is P24's; **next is P20**
(P20a/c/d-first-half done 2026-10-01: `-spec` 1.19x short, 1.31x long —
49.8 / 54.7 tok/s; P20b's depth two measured and not adopted, the third
row is MoE bytes; the draft's head over a 64k-id prefix took short to
1.24x / 51.9 tok/s; P20e on SPEED-Bench: **1.48x, 61.3 tok/s at depth 2**;
P20f served it: **56.8 tok/s greedy, 50.1 sampled through `/v1/chat/completions`**
with `-llm-draft`, and the sampler's 2 ms a token removed),
re-planned on 2026-10-01 against llama.cpp's MTP PR (the draft's KV cache
was never primed here — P20a — and depth 3 with snapshot planes reads 1.55x
there), because the step is now 24.2 ms
against ~19.6 of weights at the bus and ~1.1 of host, and the 3.5 between
is launches and small streams no fusion left on the list recovers. P21–P23
together were priced at 38.7 → 42 tok/s and have delivered 41.4; only P20
reaches the 46–56 range, and P19 re-priced its two-row pass at 1.23 steps
(break-even 23% acceptance for a free draft). Batched throughput
(65 tok/s at three rows) keeps its own list in [`research/concurrency.md`](concurrency.md) *Next*. The
instrument rules at the end of this section apply to every number: a
same-hour control, the shipped banks and `LLM_BANK_CACHE`, nothing else on
the machine.

**Open before this plan, in rough order of value** (the decode items here
are folded into P19–P24 above):

- ~~**Prompt processing.**~~ **P11, closed 2026-09-21**
  ([write-up](p11-prefill.md)). Two changes and three measured
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
  **P12, closed 2026-09-21** ([write-up](p12-prefill-round-two.md)).
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
  closed 2026-09-22** ([write-up](p16-decode-arena-width.md)). It was
  not the memory: the block input ports padded a decode step's one row out to
  the arena, so a wider arena was more zeros written a layer. At an 8192 arena
  decode goes 23.55 → 33.87 tok/s; through the server decode is 34.2 at every
  batch. What remains between an 8192 and a 2048 sweep is depth — the wider
  prefill leaves a deeper cache — not width.
- ~~**Prompt processing at 128k.**~~ **P13, closed 2026-09-22**
  ([write-up](p13-long-context-prefill.md)). Two changes and one
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
  ([write-up](p14-prefill-at-depth.md)). Four changes and four measured
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
  where a copy runs two. **Closed by KERNELS.md G6 (2026-09-30,
  research §5.4): the access pattern it was.** The 134 left the residual's
  write-back out (it was 184), and the rest was the grid — a token is ten
  turns of the 4 KB channel rotation, so workgroups a token apart load the
  same DRAM channels; walked stream-fastest `hc.cn` is at 200 GB/s and
  `hc.norm` at 217, bit-identical.
- **P6 — batching. Unblocked 2026-09-23 and moved to
  [`research/concurrency.md`](concurrency.md)**: the answer is three concurrent
  streams with a priority lane for voice. C1 (sequence slots), C2 (the
  scheduler, `-llm-slots`), C0 (`cmd/loadgen`), C3 (chunk curve) and C4
  (per-slot checkpoints: a voice command's TTFT goes 1.83 → 0.19 s at 48
  layers) and C5 (batched decode: three agents at 18 tok/s each instead of
  11, a three-row pass at 1.83 steps) and C6 (every slot at the full 262k)
  are done; `ai.service`'s LLM line is `-llm-slots 3`. The text below is the
  old framing.
  Blocked on the product question: will the API serve
  more than one stream? Each sequence owns 113 MB of DeltaNet state plus
  rings and KV. P5b already built the first stage (R-row decode GEMVs,
  shipped at R = 2); raising `GEMVMaxRows` must land in the same commit as
  its rung in `TestGraphIsAChunkSplit` — R rows were correct in a block test
  and wrong in the model three separate ways (P5c).
- ~~**Context depth is the largest measured regression.**~~ **P7, closed
  2026-09-21** ([write-up](p7-context-depth.md)). It was two terms
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

## How decode and prefill got here (P7–P18)

*Moved here from the root `TODO.md` when it was retired on 2026-10-02.* This was the head of its "where the work goes next" section, kept
because it explains the regime the P19–P25 plan starts from; the full
write-ups are the `p7`–`p18` files.

**Where the work goes next.** **P18 (2026-09-22): the full trained context.**
The server now holds all **262 144** cells the model was trained for (it capped
at ~148k: the KV cache shared a 4 GiB descriptor range with the arena). The
cache has a buffer a plane. Perplexity is 4.0344 over positions 131k-262k, and
a 237k-token prompt through HTTP recalls its first line at 1022 tok/s prefill
and 29.9 tok/s decode. `ai.service`'s LLM line is `-llm-ctx 262144 -llm-batch 8192`
([`research/p18-full-context.md`](p18-full-context.md)).
**P17 (2026-09-22): heads on the fragment's
rows.** A 128 000-cell prefill at ubatch 2048 goes **833.5 → 1011.4 tok/s**
(64 000: 896.4 → 1048.6), and decode at depth gains ~2%, exactly
([`research/p17-heads-on-rows.md`](p17-heads-on-rows.md)). The QSA
selection is per token and twelve query heads share each kv head, so the
gathered attention now puts a token's heads on the fragment's M axis and reads
that token's 2 051 cells instead of a sixteen-token union of 7 049; the mask
dispatch is gone. P17-2 faults the next prompt chunk's n-gram pages in while
the current one runs (+3% at depth). At the served ubatch 4096, 128k goes
**899.7 → 1175.7 tok/s**. The attention's depth terms are 0.14 ms of a 0.99 ms
prefill token at 128k; what is left of the falloff is the flat floor. **Open
for decode:** the host n-gram gather is ~1 ms of every token (16 dependent
major faults; residency of the 29 GB table is a deployment decision), and
its median step is 236 µs and two steps in 32 take ~8.7 ms at fixed token
positions with ordinary fault counts, which is 75% of the line and needs a
kernel-side trace to explain. **P17-3** takes decode's `attn.select` at 128k
from 54.0 to 43.8 µs a layer (keys in registers, 10-bit digit, exact): decode
at 128k 32.93 → 33.16 tok/s.
**Long-context prompt processing is closed at
the target.** **P14 (2026-09-22)** takes a 128 000-cell prefill from P13's
**563.6** to **826.0 tok/s at ubatch 2048** and
**946.1 at 8192**, which is the 900 tok/s `GOALS.md` asked for
([`research/p14-prefill-at-depth.md`](p14-prefill-at-depth.md)).
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
([`research/p7-context-depth.md`](p7-context-depth.md)); **P8, P9 and P10
are one finding**
([`research/p8-decode-attention-split.md`](p8-decode-attention-split.md))
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
and 25.81 ([`research/p16-decode-arena-width.md`](p16-decode-arena-width.md)).
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
([`research/p15-decode-at-depth.md`](p15-decode-at-depth.md)). Three
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
