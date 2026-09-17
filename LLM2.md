# LLM2 — the review, the hypotheses checked, and the next priority list

> **Written 2026-09-17**, as a review of `LLM.md` at L8c-7 / L9a. `LLM.md`
> stays the stage log; this file is the forward-looking list. Same
> conventions: history to `TODO.md`, closed findings to `research/`.

## Where this stands, in five lines

- **Decode 31.51 tok/s** against llama.cpp's 25.15 — **1.25x** — at a
  measured bank of 4.281 GB a token and a quoted ceiling of 56.5.
- **Prefill 1071.8 tok/s** at ubatch 2048 against 391.4 — **2.7x** — and
  671.4 at 512. The original L2a target of ~1150 is 93% reached.
- **Perplexity 4.1787 against our 4.0289 — +3.72%** (and +3.59% against the
  reference's own 4.0340), with five of six dense families at 4.5 bits and
  additivity holding to 0.01 pp.
- **Served**: `cmd/serve -llm`, three envelopes over one loop, prefix reuse.
- **One correctness cliff**: the whole-model graph never returns above
  ~2560 rows at 48 layers, which blocks every long-context claim this model
  exists to make.

Phase 1 is done, phase 2's kernel half is done, and phase 2's width half is
one trivial family (`ple_proj`) from done. The plan as originally drawn —
run it, profile it, then optimise — executed, and the reference is beaten on
both axes. **The competition from here is our own ceiling, not llama.cpp.**

---

## The hypotheses, checked

| hypothesis | verdict |
|---|---|
| **"Decode is DRAM-bound by weight bytes"** (the premise of D3, L8a-L8c) | **Was true, is now only 56% true.** Through L8b, measured ≈ ceiling: 14.71 tok/s against 15.4 GB-derived numbers. Today 31.5 measured against 56.5 ceiling is **56%**, and the doc itself says the bank has not been where the time is since L8c-5. The binding constraint has moved and **nothing has re-measured where to**. See P1. |
| **"The cost of 4 bits runs inverse to the bytes"** (L8c-1's headline) | **Retired by its own successors.** The pp-per-GB ranking over 145 chunks is bimodal: `deltanet` 0.87, `hyper_conn` 0.93, `lm_head` 2.59, `full_attn` 5.52 — and the split is presence (36 layers / 97 mixers vs 12 layers / 1 matrix), not size. |
| **"An 8-chunk screen predicts a family's corpus cost"** | **Retired, four ways.** +0.40→+0.82, −0.30→+0.93, +0.90→+0.31, +1.05→+1.65. A screen does not bound, does not fix the sign, does not rank. Every future width call needs a 145-chunk run (8 min — affordable). |
| **"Corpus deltas add"** (the asymmetric form's additivity) | **Confirmed to 0.01 pp across four families** — where the symmetric form compounded 11.7% into 18.5%. This is now a *tool*: plans are composable from measured per-family deltas (see idea 4). Caveat: only demonstrated on the asymmetric form at 4.5 bits. |
| **D14: "a bank that fits the MALL pays only its unpack"** | **Confirmed with its boundary, four blocks, both signs, one rule.** Closed as a hypothesis; it is now a design fact. |
| **D12's 4 KB formulation** | **Weakened at L8c-6** (no rung is a whole multiple and the spread is still 1.65x); the operative half — *re-measure when the width changes* — stands and is what caught it. |
| **"Residency is free"** (L6a-4) | **Confirmed at ≤2560 rows — and the >2560 stall is plausibly its boundary.** The untested suspicion in the open question is exactly residency (~80 GB pinned + 28.8 GB mmap + arenas that grow with rows). If P0 confirms it, L6a-4 gains a cliff. |
| **"Context is cheap here"** (the QSA table in `LLM.md`) | **Priced, never demonstrated.** The arithmetic says 6.8 GB of KV at 262k and a capped 2051-cell read per step; the graph has never completed past 2560 rows. Currently an *inference*, not a measurement — P0 gates it. |
| **D16: a rung above 242 GB/s is not a DRAM measurement** | **Confirmed and generalising**: every one-token ladder screened so far was choosing rungs against an L3 hit. Standing rule: re-screen a ladder before acting on it. |

---

## The decode budget, re-derived honestly

The 56.5 tok/s ceiling counts streamed weights only. A token also moves:

    DeltaNet state, 36 x 3.1 MB, read + write     ~0.22 GB
    KV + indexer at 2k context                    ~0.05 GB
    activations, rings, logits row                ~0.01 GB
                                                  --------
    honest per-token traffic                      ~4.56 GB  -> ~53 tok/s

Measured 31.5 tok/s is 31.7 ms a token; 4.56 GB at the best measured
dispatch rate (227 GB/s) is ~20 ms. **~12 ms a token — 37% of the step — is
not accounted for by streaming bytes at any achievable rate**, and no
attribution has been run since L8e (two families and two kernel generations
ago). Candidates, none priced: host record/submit per token, sampler +
detokenize, the PLE host gather, dispatches still far off the bus, block
boundaries. That 12 ms is worth more than the router and the experts
combined (+4.5 tok/s of *ceiling*), and it is a day of measurement with
tooling that already exists (L8d's per-dispatch timestamps).

---

## Ideas that are not in the current plan

1. **Re-attribute the decode step** (above). The single highest
   information-per-hour item on the list.
2. **A pre-recorded, reusable decode command buffer.** IDEAS §4.2 was never
   carried into this vertical. The decode graph is shape-stable token to
   token — same ~490 dispatches, same buffers; only the position and the
   token id change. Record once, feed the varying scalars through a small
   uniform buffer (or indirect dispatch) instead of re-recording. Only worth
   building if P1's attribution names host record time — but if it does,
   this is the fix, and nothing in `LLM.md` mentions it.
3. **MTP needs a rollback story before it needs a kernel.** Speculation on a
   *recurrent* model is not the usual KV-truncate: a rejected draft must
   rewind (a) 36 DeltaNet states — checkpoint/restore of 113 MB, ~0.3 ms as
   a device copy, fine but must be designed in; (b) both convolution rings,
   whose stores are currently pure writes addressed by position; (c) the KV
   cache and pooled indexer blocks (easy — truncate); (d) the host-side id
   list the trigram hash reads. And verification runs at M = 2-8, where D15
   *refuses* the GEMV path by design — the M=2..8 GEMM/GEMV crossover
   (IDEAS §1.5) has never been measured on this model's shapes. A half-page
   design doc first; it changes the state-object API (P5).
4. **The accuracy knapsack.** Additivity + per-family corpus deltas mean the
   shipped plan is now a solvable trade, and the uniform plan is provably
   not optimal in pp-per-GB: `full_attn` buys 0.299 GB a token for +1.65%
   while `deltanet` buys 1.07 GB for +0.93%. Two composable options, both
   computable *today* from numbers already in `results/`:
   - `full_attn` back to int8: ~4.58 GB a token, ~52.8 ceiling, **~+2.07%**.
   - `full_attn` at 5.5 bits: `sim.go` can grade a q5_k arm in one 8-min
     run **before any kernel exists** — likely most of the bytes back for a
     fraction of the 1.65.
   Nothing forces uniform 4.5. Decide with numbers, record it as D18.
5. **A second corpus, and one downstream eval.** +3.72% rests entirely on
   wikitext-2. The instrument is corpus-agnostic — one run on code or on
   chat-formatted text is 8 minutes — and the HTTP API makes a small
   task-level screen (a few hundred multiple-choice items) an evening. Cheap
   insurance before the widths are declared shipped.
6. **The DeltaNet state's width.** 0.22 GB a token of the honest budget is
   f32 state traffic. fp16 state halves it (+~1.3 tok/s of ceiling) — but
   the recurrence accumulates, which is exactly where fp16 goes wrong.
   Gradeable end-to-end with the ppl instrument; low priority, but it is
   the only remaining per-token traffic nobody has named as a lever.
7. **Long-context gates, the day P0 lands**: perplexity at ctx 8k with the
   selection biting hard, decode tok/s flatness against context (QSA's
   whole promise), and a needle test through the API. None exist because
   none *can* yet.
8. **Batching × recurrence.** "Batching amortises the dense 76%" is carried
   from the phase-3 sketch, but each concurrent sequence owns 113 MB of
   DeltaNet state, its own rings and its own KV — the served graph is one
   sequence's today. The open product question ("will the API ever serve
   more than one stream?") should be answered before P5/P6 ordering is
   final, because if yes, batching may beat MTP for the same effort.
9. **Reconcile the 0.38 GB** between the graph's 4.45 GB dense count and the
   checkpoint's 4.830 inventory (flagged in `LLM.md`, never chased). Fold
   into P1's attribution — the ceilings quoted everywhere inherit it.

---

## The priority list

### P0 — the >2560-row stall  *(blocker; nothing long-context ships past it)*

A server that hangs on a 3,000-token prompt is not a server, and "a long
context is what this model is for." Known: `Graph.flush → recorder.submit`
spins in `[syscall]` at 100% of one core, GPU at 2-3%, fence timeout never
fires; 4- and 12-layer prefixes are fine at 4096, 32 and 48 layers hang;
pre-existing, bank-independent. The plan, cheapest observation first:

- [ ] Reproduce and **look at the kernel side**: `/proc/<pid>/stack` of the
      spinning thread (names the ioctl loop), `dmesg`, and amdgpu's debugfs
      GTT/VRAM counters during the hang. One session, likely decisive.
- [ ] **Bisect the cliff's shape.** Rows in steps of 128 at 48 layers;
      layers at fixed 3072 rows; and resident bytes at fixed shape (the
      4.5-bit vs int8 bank is a free 1.7 GB lever). If the cliff tracks
      (layers × rows)-ish totals it is memory pressure and L6a-4 gains its
      boundary; if it is pinned near 2560 rows regardless, it is a
      size/offset bug.
- [ ] Audit the shim's C ABI for 32-bit offsets/sizes on the paths only a
      big pass exercises (barriers, copies, arena offsets).
- [ ] Control: one command buffer per *block* at 3072 rows (L7c's shape) —
      if that completes, the variable is the submit's size, not the bytes.
- [ ] Gate: `-graph` returns at 4096 and 8192 rows at 48 layers, `-ppl -ctx
      4096` completes — then run idea 7's long-context gates and put the
      numbers in `LLM.md`.

### P1 — re-attribute the decode step  *(one day; prices everything below)*

- [ ] Per-dispatch timestamps over 64 decode steps on today's bank, plus
      the host side split (record, submit, sampler, PLE gather, detokenize)
      — a table that sums to 31.7 ms within 5%.
- [ ] Reconcile the 0.38 GB (idea 9) so the ceilings are on one basis.
- [ ] Act on anything ≥1 ms with a known fix (idea 2's pre-recorded command
      buffer is the likely first); write the rest down.
- [ ] Gate: the gap between measured and honest ceiling is *named*, and the
      list below is re-ranked with those numbers rather than these
      estimates.

### P2 — `ple_proj`, and close L8c  *(half a day; closes the stage)*

- [ ] The last family: two 2560-wide matrices, no obstacle, two bank stages
      in one since `PLEGPU` never got the int8 bank either. 0.017 GB a
      token.
- [ ] Take D13's payoff: delete the two-plane fp16-tail machinery
      (`lowRank`/`gateOff` in the kernels, L8b-1's doubled rows) now that
      every tail is priced and the exception list is empty.
- [ ] Gate: bank-is-the-format equalities as at L8c-5/6/7, a 145-chunk
      corpus number for the **complete** uniform plan, and L8c closed in
      `LLM.md` with that number as the stage's result.

### P3 — the shipped-widths decision  *(the knapsack, then D18)*

- [ ] Sim-grade `q5_k` on `full_attn` (and, if cheap, on `lm_head` — the
      other expensive family) — no kernel needed to get the number.
- [ ] Pick the plan from measured corpus deltas: uniform 4.5 (+3.72%),
      `full_attn` at int8 (~+2.07%, −3.7 tok/s of ceiling), or `full_attn`
      at 5.5 bits if the sim says it earns its arm.
- [ ] Screen the winner on a second corpus and one small downstream eval
      (idea 5).
- [ ] Gate: a decision row **D18** with a 145-chunk number, and the served
      default set to it.

### P4 — the last bytes: the router and the experts  *(+4.5 tok/s of ceiling)*

- [ ] The F32 router at fp16 (+1.5 of ceiling; L5a's tie analysis is the
      named risk — check the top-10 sets over a real prompt, not just ppl).
- [ ] The expert banks toward ~4.25 bits (+3.0): transcode `ffn_down_exps`'s
      Q5_1 (27.1 GB) and the five Q8_0 layers to Q4_K with the imatrix —
      **the existing `llm_moe_gemm/gemv` arms already read Q4_K**, so this
      is a transcode and a corpus run, not a kernel. L8c-3 is the reason to
      expect the calibrated form to behave; D4 (nothing below 4 bits) holds.
- [ ] Gate: 145-chunk ppl for each step separately (additivity says they
      can be graded independently), decode measured in the whole model
      (D16: no ladder numbers), same text at temperature zero.

### P5 — MTP speculation  *(×1.5-1.8 on everything above; after P0, with idea 3's design)*

- [ ] The rollback design doc first: state checkpointing, ring rewind, KV
      truncate, id-list rewind, and the M=2..8 verification-kernel
      decision.
- [ ] Then the draft head (2.79 GB GGUF, already downloaded) and the loop.
- [ ] Gate: **speculation is lossless** — token-for-token identical text at
      temperature zero — and a measured multiplier at ctx 512 *and* at a
      long context, because acceptance rates are context-dependent.

### P6 — batching  *(pending the product question in idea 8)*

- [ ] Decide whether the API serves concurrent streams; if yes, size the
      per-sequence state cost and the scheduler before optimising the
      batch-1 path further.

### Parked (unchanged from `LLM.md`, in one place)

W4A8 with the asymmetric epilogue (the GEMM arm is unmeasured); the unpack
prefetch (prefill lead, ~1.1x, "L8d's third lead"); the fp16 residual; the
hot-expert fast path; the float-atomic combine; the vision tower; the
llama.cpp rebuild question (a rebuild invalidates trace and baseline
together).

---

## Why this order

P0 blocks the product and every long-context claim. P1 is a day and
re-prices everything after it — acting on P4 or P5 before P1 risks
optimising bytes while 12 ms a token sits in something that is not bytes.
P2 and P3 close the accuracy story while additivity and the instruments are
warm, and they are small. P4 is known-value ceiling work. P5 is the largest
single multiplier on the list but wants P0 (long prompts are where serving
happens), P1 (its verification step inherits whatever the attribution
finds), and its own design pass first — and its multiplier applies on top
of whatever P4 buys, so it loses nothing by going after.

**The numbers to beat from here: 31.5 tok/s measured, ~53 honest ceiling on
today's bank, and llama.cpp at 25.15 already behind at every ubatch.**
