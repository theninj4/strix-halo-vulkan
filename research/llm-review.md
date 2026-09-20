# LLM2 — the review, the hypotheses checked, and the next priority list

> **ARCHIVED 2026-09-20** — frozen as the closing record of the LLM vertical's 2026-09-17 review: the hypotheses checked, the decode budget, and the P0–P6 priority list as it closed. It was
> `LLM2.md` at the repo root; the live state of play is now [`../TODO.md`](../TODO.md).

> **Written 2026-09-17**, as a review of `llm-vertical.md` at L8c-7 / L9a. `llm-vertical.md`
> stays the stage log; this file is the forward-looking list. Same
> conventions: history to `../TODO.md`, closed findings to `research/`.
>
> **Every GB-a-token and tok/s-ceiling figure below is P4a-corrected**
> (2026-09-19): the MoE router has been staged as halves since L5b, so the
> row is 0.142 GB a token and not the checkpoint's 0.252, and the expert
> bank splits 0.889 gate+up / 0.615 down by *bytes* rather than 1.003/0.501
> by parameters. Numbers quoted before that date are 0.110 GB a token high.

## Where this stands, in five lines

- **Decode 36.19 tok/s on D19's widths plus P4b's and P4c's MoE bank** —
  **1.44x** llama.cpp's 25.15, three interleaved passes against a same-hour
  control at 35.46, and prefill 1.5% up beside it (P4c). Before it, on D20 alone:
- **35.52 tok/s on D19's widths plus P4b's MoE bank** — **1.41x**
  llama.cpp's 25.15, three interleaved pairs against a same-hour D19 control
  at 34.53 (within-arm spread 0.14 and 0.01). D19 alone is 34.57–34.53,
  1.37x, against a D18 control at 35.70: the fifth bit's 0.237 GB a token is
  the price of taking the quantisation damage from +3.87% to +1.63%, and
  **P4b gives 0.129 GB of it back for nothing measurable**. P1 attributed
  the step to the dispatch and took the n-gram gather's sixteen serialised
  page faults out of it; P1a fused the hyper-connection boundary, 1501
  dispatches a pass to 1407; P1c put the step in one pre-recorded command
  buffer, −1.90 ms a token; P2 took the last fp16 matmul and the fp16
  tails out of the bank, −0.47 ms. Every whole-model number needs a
  same-hour control — see P1c's environment finding before comparing
  across days.
- **Prefill 1089.2 tok/s** at ubatch 2048 against 391.4 — **2.78x** — and
  668.1 at 512. The original L2a target of ~1150 is 95% reached. P1a's fusion
  is +1.7% of it at 2048 and nothing at 512, because the residual crosses the
  MALL in between.
- **Perplexity 4.0970 against our 4.0289 — +1.69%**, which is D19 (4.0948,
  +1.63%) plus P4b's MoE transcode at a cost the instrument cannot resolve.
  P3a built the fifth bit and set **D19**: D18's plan (uniform 4.5-bit with
  `ple_proj` on int8, 4.1850, +3.87%) plus ggml's `qh` plane on `full_attn`,
  `qsa_indexer`, `lm_head` and `hyper_conn` — **38% of D18's accuracy cost
  for 1.13 tok/s**, 35.70 → 34.57 over two interleaved passes. `deltanet`'s
  fifth bit is refused at 1.6 pp/GB against the 4.6 D18 already declined.
  **D20** then sold 0.1288 GB a token back at **0.017 pp/GB**. 4.279 GB a
  token, a 56.6 tok/s ceiling. **On a second corpus (Go stdlib) the plans
  rank identically at 2.5x smaller magnitudes**, so the wikitext figure is
  the pessimistic end. Additivity leaks **both ways** and both leaks are the
  n-gram block, so a plan is measured and never composed.
- **Served**: `cmd/serve -llm`, three envelopes over one loop, prefix reuse —
  and since P3a it stages **D19** by default (`llm.ShippedDenseBank`), since
  P4b **D20** beside it (`llm.ShippedMoEBank`).
- **Speculation is built, lossless, and 0.95x** (P5c, 2026-09-19). The draft
  head names the trunk's next token 65.6% of the time on its own generated
  prose and 46.9% at a long context, against a break-even of ~0.72; the rollback restores `result_norm` to the last place through
  rejected passes, and the loop emits the plain loop's ids token for token at
  both contexts — and it is still a **net loss**, because a two-row
  verification pass is 1.36 decode steps rather than P5b's 24-layer 1.17 and
  because at R = 2 a rejection costs a whole extra pass. **Narrowing the trunk
  makes it worse, twice**: the draft head stages at the checkpoint's own widths
  so it does not shrink with D19/D20/D21 (0.176 of a step → 0.208), and the
  acceptance rate falls, because the draft predicts the checkpoint's next token
  and the shipped banks move the trunk away from it. It found **three
  defects in P5b** on the way, two of which made every two-row pass silently
  wrong, and **453 MB of host memset a pass** in `DeltaNetGPU.Resize` that
  only a short re-recorded pass could see. The three things that would take it
  past 1.0 are priced in the write-up; the largest is not engineering.
- ~~**One correctness cliff**~~: **closed at P0 (2026-09-18)**. It was
  amdgpu's gfx ring watchdog killing any submit that holds the ring past
  **2 s** — not rows, layers, bytes or residency. The recorder chunks by time
  now, and prefill runs to **8192 rows at 1213.5 tok/s, 3.10x**, still
  climbing where llama.cpp's plateaus.

Phase 1 is done and phase 2 is done on both halves: the kernels, and the
widths — **L8c is closed at P2**, all six streamed families on the 4.5-bit
bank and the fp16-tail machinery deleted. The plan as originally drawn —
run it, profile it, then optimise — executed, and the reference is beaten on
both axes. **The competition from here is our own ceiling, not llama.cpp.**

---

## The hypotheses, checked

| hypothesis | verdict |
|---|---|
| **"Decode is DRAM-bound by weight bytes"** (the premise of D3, L8a-L8c) | **Was true, is now 79% true, and P1 says where the rest is.** Through L8b, measured ≈ ceiling: 14.71 tok/s against 15.4 GB-derived numbers. Today the weight-streaming dispatches are **24.05 ms of a 30.52 ms step, 79%** — but they run at **178 GB/s against the 227 a dispatch reaches**, so 5.54 ms of that is shape rather than bytes, and another 3.46 ms of dispatches stream no weight at all. The binding constraint has moved from *how many bytes* to *how the dispatches are shaped*: `hyper_conn` at **114.6 GB/s** and the shared expert at **136.5** against the routed experts' 178-181 off the same bank. (**P4a corrects this row**: `moe.down`'s 147.5 and `moe.up`'s 200.6 were a parameter-ratio byte split — at the true widths both are ~180, and `moe.down` was never an outlier. The two real ones are `hyper_conn` and the shared expert.) |
| **"The cost of 4 bits runs inverse to the bytes"** (L8c-1's headline) | **Retired by its own successors.** The pp-per-GB ranking over 145 chunks is bimodal: `deltanet` 0.87, `hyper_conn` 0.93, `lm_head` 2.59, `full_attn` 5.52 — and the split is presence (36 layers / 97 mixers vs 12 layers / 1 matrix), not size. |
| **"An 8-chunk screen predicts a family's corpus cost"** | **Retired, four ways.** +0.40→+0.82, −0.30→+0.93, +0.90→+0.31, +1.05→+1.65. A screen does not bound, does not fix the sign, does not rank. Every future width call needs a 145-chunk run (8 min — affordable). |
| **"Corpus deltas add"** (the asymmetric form's additivity) | **Confirmed to 0.01 pp across four families** — where the symmetric form compounded 11.7% into 18.5%. This is now a *tool*: plans are composable from measured per-family deltas (see idea 4). Caveat: only demonstrated on the asymmetric form at 4.5 bits. |
| **D14: "a bank that fits the MALL pays only its unpack"** | **Confirmed with its boundary, four blocks, both signs, one rule.** Closed as a hypothesis; it is now a design fact. |
| **D12's 4 KB formulation** | **Weakened at L8c-6** (no rung is a whole multiple and the spread is still 1.65x); the operative half — *re-measure when the width changes* — stands and is what caught it. |
| **"Residency is free"** (L6a-4) | **Confirmed at ≤2560 rows — and the >2560 stall is plausibly its boundary.** The untested suspicion in the open question is exactly residency (~80 GB pinned + 28.8 GB mmap + arenas that grow with rows). If P0 confirms it, L6a-4 gains a cliff. |
| **"Context is cheap here"** (the QSA table in `llm-vertical.md`) | **Priced, never demonstrated.** The arithmetic says 6.8 GB of KV at 262k and a capped 2051-cell read per step; the graph has never completed past 2560 rows. Currently an *inference*, not a measurement — P0 gates it. |
| **D16: a rung above 242 GB/s is not a DRAM measurement** | **Confirmed and generalising**: every one-token ladder screened so far was choosing rungs against an L3 hit. Standing rule: re-screen a ladder before acting on it. **P4a adds the other half**: a rate is a *quotient*, and the numerator is as capable of being wrong as the denominator. Three rows of P1's attribution took their bytes from a different source than the bank they were timing, and each landed on a rate that was read as a finding — one of them (`moe.down` at 147.5) cost a whole stage. Screen the byte count the way D16 screens the rate. |

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
attribution has been run since L8e.

**P1 ran it, and the estimate was right to 0.01 ms.** The step is 30.52 ms
after the gather fix, and it sums:

    weight-streaming dispatches, 4.281 GB at 178 GB/s      24.05 ms
      of which, over the 18.86 ms the same bytes take at 227   5.54
    dispatches that stream no weight (scan, moves, norms,
      permutations, combines)                               3.46 ms
    host: record 1.127, gather 0.902, hand-over 0.828,
      sample 0.142, detokenize + emit 0.007                 3.01 ms
    unattributed                                            0.005 ms

**5.54 + 3.46 + 3.01 = 12.01.** The candidates list above was almost exactly
wrong in its ordering: the sampler and detokenizer are **0.15 ms together**,
host record and submit are 1.96, and the two real items are dispatches off
the bus and the 3.46 ms of dispatches nobody had counted because they read no
weights at all. The one candidate that was under-rated is the PLE host
gather, which was 2.24 ms and is now 0.90 — see P1's finding 1.

**P4a re-reads three rows of that table** and the total becomes **4.171 GB at
173.4 GB/s**, with 5.68 ms below 227 rather than 5.54. `moe_router` is
0.1416 GB rather than 0.252 (185 GB/s, and back in the loss column that D16
had excluded it from); the experts split **0.889 gate+up / 0.615 down** by
bytes rather than 1.003/0.501 by parameters, which makes them **177.8 and
181.0 GB/s** — the same rate — where the table said 200.6 and 147.5. **The
dispatches that are off the bus are `hyper_conn` at 114.6 and the shared
expert at 136.5, and that is the whole list.**

---

## Ideas that are not in the current plan

1. ~~**Re-attribute the decode step**~~ **— done (P1), and it was the single
   highest information-per-hour item on the list**: it paid for itself with
   the gather fix before the analysis was written, and it demoted idea 2
   from first to fourth.
2. **A pre-recorded, reusable decode command buffer.** IDEAS §4.2 was never
   carried into this vertical. The decode graph is shape-stable token to
   token — **1501 dispatches** (P1 counted them; the ~490 here was the
   pre-L7d submit count), same buffers; only the position and the token id
   change. Record once, feed the varying scalars through a small uniform
   buffer (or indirect dispatch) instead of re-recording. **P1 priced it:
   1.127 ms of recording plus 0.828 of hand-over, 6.4% of the step, about
   +2.2 tok/s** — real, and, with P1a done and P1b measured empty, now the
   front of the list.
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
4. ~~**The accuracy knapsack.**~~ **— solved at P3, and both of its options
   lost.** `full_attn` back to int8 measures **+2.72%**, not the ~+2.07% the
   additivity estimate gave, and it is **strictly dominated** by L8c-3's
   `hc + attn` at `q5_k` (+2.70% at 4.419 GB against +2.72% at 4.592).
   `full_attn` at 5.5 bits *is* the right answer and the sim priced it
   without a kernel — **4.1560, +3.15%, 14.1 pp/GB** — but `bank_q4.go` is
   nibbles by construction, so it is a kernel stage and became **P3a**. What
   shipped instead is the row nobody had costed: **`ple_proj` back to int8**,
   22 pp/GB, which is D18. The framing error worth remembering is that the
   list quoted `ple_proj` at ~12 pp/GB *from halves* where every other family
   was quoted from int8; on the consistent basis it was ~35, the worst trade
   in the plan by three times the stated margin.
5. **A second corpus** — **done at P3**; the downstream eval is not.
   The Go standard library, cut to wiki.test.raw's byte count: the plans
   **rank identically** and the magnitudes are **2.5x smaller** (D18 +1.34%
   against +3.87%), so wikitext is the pessimistic end rather than a
   universal number, and both of P3's decisions strengthen off it. The
   task-level screen still has no dataset on this machine — a few hundred
   multiple-choice items through the HTTP API remains an evening's work and
   is the last unpriced thing about the widths.
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
9. ~~**Reconcile the 0.38 GB**~~ **— closed at P1.** It is the MoE router
   (0.252 GB) and the shared expert (0.251), which the checkpoint inventory
   files under "dense, every token" and L8b's table files with the MoE, less
   L8b-1's doubled `inject` rows, `PLEGPU`'s halves and rounding. Both
   counts were right; they partitioned the same tensors differently. Every
   ceiling is now quoted on **4.281 GB a token**.

---

## The priority list

### ~~P0 — the >2560-row stall~~  *(**done**, 2026-09-18)*

**It was amdgpu's gfx ring watchdog, and none of the four hypotheses above.**
A submit that holds the gfx ring past **2.0 s** is killed and the ring reset;
the reset *force-signals the fence*, so `vkQueueSubmit` and `vkWaitForFences`
both return success and the spin is six lines later, in
`vkGetQueryPoolResults(VK_QUERY_RESULT_WAIT_BIT)` — an unbounded **userspace**
poll on RADV, of a timestamp the killed dispatches never wrote. `[syscall]` in
the old dump was Go's label for a cgo call: `utime=5645` against `stime=5`.
Full write-up: [p0-ring-watchdog.md](p0-ring-watchdog.md).

- [x] Reproduce and look at the kernel side. `/proc/<pid>/stack` and debugfs
      both want root here; `/proc/<tid>/stat` and `journalctl -k` did the job
      instead, and a native backtrace needed gdb to be the *ancestor*
      (`ptrace_scope` 1).
- [x] Bisect the cliff's shape. It tracks **neither** rows nor layers nor
      bytes: 1024 dispatches is 1.886 s at 2560 rows and runs, 1.964 s at 2688
      and runs, **2.033 s at 2816 and is reset**. GTT, VRAM and RSS are flat
      through the whole stall, so this is **not** L6a-4's boundary and
      "residency is free" comes out unmarked rather than confirmed.
- [x] The C ABI audit was not needed: the control settles it. The same 1273
      dispatches in one submit are 787 ms at 512 rows and fine, and are reset
      at 2688 rows where two submits of the same work had just run.
- [x] Control: smaller command buffers complete. 3072 rows in five submits of
      256 runs at 1143.1 tok/s.
- [x] **Fix**: the recorder chunks by *time* (`batchFor`), from an affine fit
      of 322 us a dispatch plus 582 ns a dispatch-row, budgeted at half the
      cliff. Decode is untouched (1 row still batches at 1024). The shim's
      query reads all go through a deadline that names the missing slots
      instead of spinning.
- [x] **Gate**: `-graph` returns at 4096 (**1174.6 tok/s, 3.00x**) and 8192
      (**1213.5, 3.10x**, at `-ctx 8192`); `-ppl -ctx 4096` completes at 48
      layers at **PPL 3.9392 +/- 0.02209** over 72 chunks of 4096, against
      4.0289 for the same bank at n_ctx 2048 — twice the context is **-2.23%**
      with the selection live, which is idea 7's first gate. Prefill **does not
      plateau** where llama.cpp's does.
- [ ] Carried forward: idea 7's remaining long-context gates — perplexity at
      ctx 8192, decode tok/s against context, and a needle test through the
      API — are now possible and are not yet run.

### ~~P1 — re-attribute the decode step~~  *(**done**, 2026-09-18)*

- [x] Per-dispatch timestamps over 64 decode steps on today's bank, plus
      the host side split — **the table sums to 30.52 ms with a 5 us
      residual**, over 36 dispatch labels and six host phases.
      `-gen -attrib`, `results/p1_decode_attrib.csv`.
- [x] Reconcile the 0.38 GB (idea 9): **it is the MoE router (0.252) and the
      shared expert (0.251)**, which the checkpoint inventory files under
      "dense, every token" and L8b's table files with the MoE, less three
      doublings in the other column. Every ceiling is now quoted on
      **4.281 GB a token**.
- [x] Act on anything ≥1 ms with a known fix: **the PLE gather, 2.244 ms,
      was sixteen serialised major page faults a token** — not compute, not
      bytes. Concurrent: identical fault count (16.3), **7.0-7.5x**,
      2.244 → 0.902 ms, **decode 31.51 → 32.72 tok/s**. The rest is written
      down as P1a-P1c below.
- [x] Gate: the gap is named. **12.01 ms = 5.54 (weight-streaming dispatches
      below 227 GB/s) + 3.46 (dispatches that stream no weight) + 3.01
      (host)**, and idea 2 — the review's favourite for the whole of it — is
      **1.96 ms** and now ranks fourth. [Write-up](p1-decode-attribution.md)

### ~~P1a — the hyper-connection block's 485 dispatches~~  *(**done**, 2026-09-18)*

**The weightless half is fused away bit-exactly; the other half is a grid,
and the grid is priced rather than rebuilt.**

- [x] The scatter that closes one mixer and the norm that opens the next are
      the same 2560 values per (token, stream) written and read straight
      back, and both were already one workgroup per (token, stream).
      `llm_hc_cn.comp` does both in one pass over registers: **1501
      dispatches a pass to 1407, 30.69 ms to 30.37, decode 32.59 → 32.93
      tok/s**, both arms on one build (`LLM_HC_NOFUSE=1` is the control).
      **Prefill comes along at ubatch 2048 — 1070.8 → 1089.2 tok/s, +1.7%,
      the block 330.1 → 294.6 ms** — where the fusion removes an 84 MB read a
      boundary; at 512 it is worth nothing, because 21 MB fits the MALL.
- [x] **Identical to the last place** on `res`, `xn` and `mixed`
      (`TestHCFusionIsTheCombineThenTheNorm`, `TestHCFusionRunsThePass`).
      The ulp chased first: giving the sum a *second reader* lets ACO
      contract it into an FMA where the scatter rounds twice, and `precise`
      is the only thing that says no.
- [x] Gate: the label count moved and **the GB/s did not** — 114.9 against
      114.6 — and the ladder says why. The down projection's decode rungs
      read the same 1.94 MB off the same bank with a grid that varies
      twenty-fold, and **168 workgroups is 9.26 us where 672 is 5.54**. So
      `up_m1`'s 10.87 us at **160 workgroups** is not MODE 1's M=1 waste; it
      is `down_gemv8`'s number at `down_gemv8`'s width.
- [ ] Carried forward, priced and not taken: `BN` cannot fall below 64 in
      MODE 1 because `packUpB` makes a 64-column block 16 features x 4
      streams. A 4x4 block would be 640 workgroups, worth **2.3 us a mixer,
      0.22 ms a token, +0.24 tok/s** — and it is every `up` rung's epilogue
      on the prefill path, so 0.7% of decode does not buy it.
      [Write-up](p1a-hyper-connection-shape.md)

### ~~P1b — `moe.down`'s rung, and the shared expert's~~  *(**done**, 2026-09-18 — a ladder, and no rung moves)*

**The 1.19 + 0.73 ms was the MALL's arithmetic. Re-screened against DRAM the
plan survives on every axis, and what is real is a 1.33 ms environment gap
that is not a rung choice.**

- [x] The ladder, honest: **sixteen cold banks at `-iters 1`** (25.6 GB, so
      `ProfileSweep` walks a dispatch across layers and never re-reads a bank
      warm), against l8e's two-layer, 20-iteration fixture whose every GEMV
      down rung read **267-357 GB/s of a 242 GB/s bus**. Two runs agree to a
      median ratio of 0.9991 (p10-p90 0.982-1.012).
      `results/p1b_moe.csv`.
- [x] The ranking is unchanged: routed **v64w4/v16w4** (99.0 us at 186 GB/s,
      59.3 at 207), shared **v64w4/v32w4** (22.7, 10.9), router **k40**
      (13.4) win every marginal cold — the plan the model already runs. The
      MALL inflated magnitudes (down 34.4 → 59.3 us), not the order. D16's
      up-mode contradiction closes: cold, v16w4 loses to v64w4 by 1.15x, the
      same side as the whole model's 42.1 ms. And the router was never 3.5x
      off — its 4.4 us ladder rate was two fp16 router matrices in the MALL;
      cold it is 13.4-14.8 against 15.1 in the model.
- [x] Gate: D16's test passes on the face of every rung — no row above 242
      GB/s — and the whole-model number is unchanged, as it must be when no
      rung changes: **32.78 tok/s, step 30.51 ms** (P1a measured 32.93),
      `results/p1b_decode_attrib.csv`. What the honest floors expose is a
      **1.33 ms a token environment gap** — moe.down 72.2 us in the model
      against 59.3 alone-and-cold (1.22x), the other four labels 1.08-1.16x —
      between a dispatch alone with a fence on a bank last read microseconds
      ago and the same dispatch in a 1407-dispatch buffer against a bank last
      read 4.3 GB ago. Bounded under P1c's 1.96 ms; written down, not chased.
      [Write-up](p1b-moe-decode-rescreen.md)

### ~~P1c — the pre-recorded decode command buffer~~  *(**done**, 2026-09-18 — 1.90 ms of the priced 1.96)*

**The whole varying state of a decode step turned out to be one uint, and
with it in a buffer the step is one command buffer recorded once.**

- [x] The instrument first: `TestDecodeDispatchDiff` diffs consecutive
      one-token passes dispatch for dispatch. Over 40 steps, every pipeline,
      grid and label is identical, and the only differing push dword in the
      step is `lowRank` carrying the position — the token id never touches a
      push constant or a grid.
- [x] So the "uniform buffer for the varying scalars" is smaller than the
      idea: **`SEQ_PAST` is now dword 0 of each block's own fp32 arena**
      (`actu[0]`, first allocation, written by `SetPast` as a 4-byte mapped
      write; one macro in `llm_common.glsl`, nine `.spv`, no layout change),
      after which **131 of 131 dispatches are byte-identical** step to step.
- [x] `vk.Prerecorded` records all 1407 dispatches in **one** buffer (the
      live path chunks into two), with the query-pool reset inside it so the
      per-dispatch marks re-arm every submit — `-gen -attrib` works
      unchanged on the replay. The first one-token `Extend` with `past > 0`
      captures; every later one uploads, sets the position and submits
      (`extendPrerecorded`). `LLM_NO_PRERECORD=1` is the control.
- [x] Gate, bit-level: 48 greedy tokens *and all 48 full logit rows*
      identical to the re-recording arm off one staged graph; whole-model
      greedy text identical to EOG; prefill at 2048 unregressed against the
      baseline binary the same hour (1070.3 vs 1061.5 tok/s).
- [x] The two host rows: `record` **1.127 → 0.16** (what remains is the real
      per-token uploads), hand-over **0.82 → 0.25** (one submit + fence is
      the floor). On matched machine state **39.39 → 37.50 ms a step,
      25.4 → 26.7 tok/s** — see the caveat below, which is its own finding.
- [x] **The machine moved more than the fix**: yesterday's binary, byte for
      byte, measures 39.20 ms a step where P1b measured 30.51 — every
      dense-bank kernel 1.6-1.9x slower (dn.qkv 115 → 206 us, head
      1964 → 3531), every MoE expert rung unchanged (moe.up 1.02x). 32-day
      uptime, swap full; suspicion is GTT page fragmentation of the
      float-staged dense banks, unprovable without root. Standing rule, D16
      one level up: **a whole-model number is only comparable against a
      control staged the same hour.** On P1b's machine state the arithmetic
      says ~28.6 ms, ~34.9 tok/s — the priced +2.2.
      [Write-up](p1c-prerecorded-decode.md)

### ~~P2 — `ple_proj`, and close L8c~~  *(**done**, 2026-09-18 — and the family is the plan's worst trade)*

**Both halves landed, and the stage closes on 4.2010 (+4.27%) — beside the
4.1998 (+4.24%) L8c-3's simulation projected before any kernel existed,
because after the deletion the bank *is* the simulation, no carve-outs.**

- [x] The last family, two bank stages in one: int8 (bit-identical, both
      tensors ship Q8_0) and q4_k — 65.8 → 35.1 → 18.7 MB, the same MODE 2
      builds the attention block runs, no new kernel. Bank-is-the-format
      three ways: `TestPLEGPUQ8IsTheHalves`, `TestPLEGPUQ4IsTheSim`
      (identical through the whole block), and chunk-for-chunk with the sim
      over eight chunks.
- [x] **The accuracy is the finding**: `ple_proj` alone is **+0.59%** over
      145 chunks for 0.047 GB a token — the worst pp-per-GB of the six
      families by an order of magnitude (~12, against `full_attn`'s 5.5 and
      `deltanet`'s 0.87), and the 8-chunk screen said **−0.11%**: a fifth
      reading of the screen warning and a second sign flip. And the six
      deltas **no longer add**: separate deltas sum to 4.31% where the
      complete plan measures 4.09% (tails still fp16) — the first material
      deviation from additivity (0.22 pp), traceable to the n-gram block.
- [x] D13's payoff: the `Q8_TAIL` arm left `llm_gemm.comp`, the tail arms
      left `llm_gemv.comp` and `llm_hc_gemv.comp`, and `qkvTail`/`downTail`/
      `idxQuant`/L8b-1's doubled rows left the three blocks. On a quantised
      bank alpha, beta, inject and the indexer are quantised at the family
      width under their own names; `qsa_indexer` must equal `full_attn`'s
      format (one plane). Cost of the deletion, measured: **+0.18 pp**
      (4.1939 → 4.2010) for ~0.085 GB a token back and the code gone. The
      int8 bank is no longer bit-identical on those four tensors — the
      trace tests carry that as `bankTol` at D13's measured prices, and
      `TestGraphLogits` accepts an argmax flip only on a measured near-tie.
- [x] Gate: **complete-plan bank ≡ full-scope sim chunk for chunk** (no
      `SRC=q8`), 145-chunk number **4.2010 (+4.27%)** in
      `results/l8c_ppl_uniform_notail.csv`, decode the same hour
      28.400 → 28.336 → **27.865 ms, 35.89 tok/s, 1.43x** llama.cpp, and
      L8c closed in `llm-vertical.md`. Streamed bank ~4.15 GB a token by subtraction;
      the fresh inventory is P3's.
      [Write-up](p2-ple-proj.md)

### ~~P3 — the shipped-widths decision~~  *(**done**, 2026-09-18 — D18, and the answer is the small one)*

**The knapsack was solved by measuring six complete plans, not by composing
per-family deltas — and the only upgrade the current kernel can stage is
worth taking, while everything better needs a fifth bit.**

- [x] Sim-grade `q5_k`: **`full_attn` 4.1560 (+3.15%)**, `lm_head` 4.1743
      (+3.61%), `ple_proj` 4.1919 (+4.05%), against the uniform control's
      4.1998. The fifth bit recovers **66-77% of a family's whole 4.5-bit
      cost for a quarter of int8's bytes**; `lm_head` recovers most because
      unsloth's matrix has **no row for `output.weight`** and the family is
      round-to-nearest.
- [x] Two arms the list did not have, and they are the ones that decided it.
      `bank_q4.go:144` is **nibbles by construction**, so `q5_k` is a kernel
      stage while int8 is free — a family left off the plan stages on L8a's
      bank. **`ple_proj` → int8 is 0.367 pp for 0.0165 GB (22 pp/GB)**;
      `full_attn` → int8 is 1.522 pp for 0.328 GB (4.6), measures **+2.72%**
      where idea 4 priced it at +2.07%, and is **strictly dominated** by
      L8c-3's `hc + attn` at `q5_k` (+2.70% at 4.419 GB against +2.72% at
      4.592). Idea 4 is closed: no int8 arm belongs in the plan but this one.
- [x] **D18 = uniform `q4_k` with `ple_proj` on int8.** 145 chunks:
      **4.1850 ± 0.02389, +3.87%**, the bank equal to the simulation to four
      decimals *and to the same standard error*. Decode over **three
      interleaved pairs**: means 35.74 (uniform) against 35.79 (D18), the
      sign flipping pair to pair against a **within-arm spread of 0.31
      tok/s** — the bytes predict 0.12 tok/s and the instrument cannot
      resolve it. Fresh inventory **4.281 GB a token, 56.5 tok/s ceiling**
      (and note the collision: that 4.281 is *arithmetic*, P1's was
      *measured* at a different bank state).
- [x] Second corpus (idea 5, first half): the Go standard library, cut to
      wiki.test.raw's byte count, 145 chunks. **The ranking is invariant** —
      1.6288 baseline, 1.6560 uniform (+1.67%), 1.6506 D18 (+1.34%), 1.6450
      `full_attn` at `q5_k` (+0.99%) — while the **magnitudes are 2.5x
      smaller**, so **+4.27% is a wikitext figure and the pessimistic end of
      the range**. `ple_proj` is a *larger* share of the damage on code (20%
      against 8.7%) and `full_attn`'s fifth bit recovers 41% against 26%:
      both decisions strengthen off wikitext.
- [x] Gate: **D18** recorded in `llm-vertical.md` with its 145-chunk number, and
      `cmd/serve -llm` stages it by default (`llm.ShippedDenseBank`;
      `LLM_DENSE_BANK` overrides, `off` restores the int8 bank). `cmd/llm`
      deliberately keeps no default — a measurement tool that staged a plan
      nobody named would make every CSV in `results/` ambiguous.
- [ ] Carried forward: idea 5's **downstream task eval**. There is no
      multiple-choice set on this machine and fetching one was out of scope;
      the cross-corpus half is done. [Write-up](p3-widths.md)

### ~~P3a — the fifth bit: a `qh` plane for the dense bank~~  *(**done**, 2026-09-18 — and it is D19)*

**The plane is exact and three of the four families are worth it.** P3's
estimates held to 6% on the three it simulated and were **1.75x optimistic on
the one it inferred** — which is `deltanet`, the only one refused.

- [x] The format: ggml's `Q5_K` is `Q4_K` plus a `qh` bit-plane and **the
      same record**, so nothing L8c-4 or L8c-6 built for the record moves.
      The plane is one 32-byte tile per 128-byte nibble tile, in the same
      order, and **byte i of it is the top bit of each of the eight levels in
      word i of the nibble tile** — so the plane byte index *is* the nibble
      word index, which every kernel already computes (`u` in the GEMM,
      `col*2 + u` in the two GEMVs). The unpack gains a load and two bit
      tests and no second addressing scheme; the plane sits *between* the
      tiles and the records, so both derived bases survive and no push field
      is needed. `-DQ5B` beside `-DQ4B`, 25 new `.spv`, and the four-bit arm
      disassembles unchanged.
- [x] `bank_q4.go:144`'s four-bit refusal is `qkBits` (4 or 5) and the
      two-plane staging is `qkStage` — three planes, their order stated once
      rather than at each of the six blocks. `BankQ5K` is a fourth
      `DenseBank`, because the bank value is what names a pipeline and sizes
      a buffer: staging one width and building another's SPIR-V would be a
      wrong answer, not a slow one.
- [x] Order the families by measured return, each a **complete plan** over
      145 chunks on D18's basis. `full_attn` + `qsa_indexer` **14.9 pp/GB**
      (4.1386, +2.72%), `lm_head` **7.8** (4.1136, +2.10%), `hyper_conn`
      **5.9** (4.0948, +1.63%), `deltanet` **1.6** (4.0780, +1.22% — 0.260 GB
      a token, five times any other family, for the smallest gain). D18
      refused an arm at 4.6 pp/GB, so the line is already drawn and the first
      three are above it.
- [x] Gate: **0 of 145 chunks differ** between the bank and P3's simulation
      of the same complete plan at `full_attn=q5_k` — 4.1560, equal in nll to
      six decimals (`results/p3a_ppl_attn_q5k.csv` against
      `p3_ppl_attn_q5k.csv`). Every block's bank-against-simulation test now
      runs both widths and all five are identical on the GEMM path and one
      rounding apart on the GEMV. Decode: five plans, two interleaved passes,
      one binary, one hour — **35.70 / 35.49 / 35.11 / 34.57 / 33.13**,
      within-arm spread 0.06-0.28. **D18 amended to D19**, `ple_proj`'s int8
      row untouched. [Write-up](p3a-fifth-bit.md)
- [ ] Carried forward, priced and not taken: **the GEMV rungs were not
      re-screened at the new width.** D12 says a split's stride must miss the
      4 KB rotation and that the rung moves when the weight's width does — a
      q5 slab is 1.22x a q4 slab plus a second stream. Every rung measured
      *correct*, and the decode numbers above are on the q4 bank's rungs, so
      this is tok/s possibly left on the table rather than a risk.
      `-attn -tokens 1 -gemm-ladder` with `-layers` high and `-iters 1`
      (P1b's honest floors) is the run.
- [ ] Also carried forward: **the fifth bit costs less decode than its bytes
      predict** — 0.078 GB is −0.55 tok/s at P1's measured 178 GB/s and
      measures −0.21; the whole ladder's 0.497 GB prices at −3.1 and measures
      −2.57. The plane is a second sequential stream at a quarter of the
      first's width out of the same loop, so it plausibly lands nearer the
      227 GB/s a dispatch can reach. Written down; nothing on the list turns
      on it.

### ~~P4 — the last bytes: the router and the experts~~  *(**done**, 2026-09-19 — and both bullets were wrong)*

**Neither half of this item was what it said, and both failed on a fact about
the checkpoint rather than on a measurement.** What replaced them is smaller,
buildable and free: **+0.99 tok/s for a perplexity delta the instrument
cannot resolve.** [Write-up](p4-moe-bank.md)

#### P4a — the router row was already fp16, and it moved four numbers

- [x] `MoEGPU.stage` has narrowed `ffn_gate_inp` to halves since **L5b** —
      `g.bank` is `routerN*NEmbd*2` a layer and the two router kernels read
      no other copy. The row is **0.1416 GB a token** (576 padded columns),
      not the checkpoint's 0.252. `TestMoERouterBankIsHalves` is the
      equality. **The +1.5 tok/s was spent before it was proposed**, and
      L5a's tie analysis is *retired* rather than owed: every perplexity
      number in this log, 4.0289 included, was measured on the fp16 router.
- [x] Every ceiling was 0.110 GB a token pessimistic: **D18 4.281 → 4.171
      (56.5 → 58.0 tok/s), D19 4.518 → 4.408 (53.6 → 54.9)**. `cmd/gguf`
      now prints the budget "as this repo stages it" beside the
      checkpoint's, so the next one does not have to be found.
- [x] **And P1's "the router reads above the bus" was the same row.** At
      0.252 GB it is 329.8 GB/s and was excluded from the loss column under
      D16; at 0.1416 it is **185-203 GB/s**, an ordinary streaming rate
      beside `deltanet`'s 201.5.
- [x] **The same error one row over cost a whole stage.** P1 split the
      expert traffic 1.003/0.501 — the ratio of their *parameters*, where
      gate/up are 4.52 bits and `down` is 6.26. The bytes are **0.889 and
      0.615**, so `moe.down`'s "147.5 GB/s, +1.19 ms lost" — the largest
      item on P1's list after `hyper_conn` — is **181 GB/s, faster than the
      pair off the same bank**. P1b spent a stage re-screening sixteen cold
      banks for it and correctly found nothing; its explanation ("the MALL's
      arithmetic") is wrong and the real one is that there was never a
      1.19 ms. **D16 one level up: a rate is a quotient, and the numerator
      is as capable of being wrong as the denominator.**

#### P4b — the transcode that is actually available  *(D20)*

- [x] **`ffn_down_exps` at Q4_K cannot exist.** Its rows are **640** and a
      ggml K-quant super-block is 256. `llama-quantize`'s
      `tensor_type_fallback` demotes `Q5_K → Q5_1` and `Q6_K → Q8_0` for
      exactly this — which is the 43 and the 5 layers. So `llm-vertical.md`'s reading
      ("unsloth's imatrix telling them the down projection is the sensitive
      one") is a fact about the row length, and **nobody has measured what
      narrowing `down` costs.** `cmd/gguf` now carries the row length per
      group and refuses to price a checkpoint-byte group at a width its rows
      cannot hold.
- [x] What *is* a transcode: the four rows at 8.5 bits with a format the
      kernels already build — `gate_shexp` and `up_shexp` (row 2560) to
      **Q4_K**, `down_shexp` and the five Q8_0 layers of `down_exps` (row
      640) to **Q5_1**. `llm/moebank.go`, `LLM_MOE_BANK` with the same
      grammar and `imatrix` default as `LLM_DENSE_BANK`, plus one rule: a
      family names a **ceiling**, so a tensor already at or below it is left
      bit-for-bit alone. `Imatrix.ExpertColumns` reads the published
      per-expert rows, which this repo had never needed.
- [x] Gate, format: **the bytes read back equal what the fit stored**,
      element for element, through `gguf.Dequantize` (`TestPackKRowIsGgmlsLayout`),
      and `rtn` Q5_1 equals `quantize_row_q5_1_ref`'s own (d, m).
- [x] Gate, staging: layer 3 staged through the plan against the same layer
      staged from pre-transcoded bytes with the plan off is **bit-identical**
      — 115 343 360 routed values, 2 621 440 shared, 10 485 760 `ffn_out`,
      zero differences.
- [x] Gate, corpus: **4.0970 ± 0.02325, +1.69%** over 145 chunks against
      D19's 4.0948 (+1.63%) — **0.0022 points for 0.1288 GB a token, 0.017
      pp/GB**, where D19 refused to *buy* at 1.6 and took three families
      above 5.9. Paired, the delta is +0.00054 nll a chunk against a
      standard error of 0.00059 (**t = 0.91**, worse in 85 and better in
      60), so it is **not resolvable** and the claim is an upper bound.
      `results/p4_ppl_all.csv`.
- [x] Gate, decode: three interleaved pairs, one binary, one hour —
      **34.48/34.49/34.62 against 35.52/35.52/35.53**, means **34.53 →
      35.52, +0.99 tok/s, 1.37x → 1.41x llama.cpp**, within-arm spread 0.14
      and 0.01. The bytes predicted +1.11. Residency **77.41 → 76.00 GB** in
      the MoE block, the arithmetic to three decimals.
- [x] **D20**, and `cmd/serve -llm` stages it by default
      (`llm.ShippedMoEBank`); `cmd/llm` does not, for the reason D18 gave.
- [x] D12's obligation, discharged: the shared expert's rungs re-screened on
      the new widths, twice (median ratio 1.0009 over 993 rows). **The up
      rung does not move** (v64w4 by 1.28x, on Q4_K as on Q8_0) and **the
      down rung moves to exactly the routed down's** (v32w4 → v16w4), which
      is the payload-words law working — `down_shexp` and the routed down
      are both Q5_1 now. Priced at **0.013 ms a token, +0.016 tok/s**, and
      not taken: it would make `MoESharedPlanFor` a function of the bank
      plan for a number under the instrument.

### P4c — `ffn_down_exps` below six bits  *(done 2026-09-19 — D21, and the 4.5-bit format beat the 5.0-bit one on both axes)*

**Built and measured.** Both candidates got an arm in both grouped kernels
and a complete 145-chunk plan; the write-up is
[`p4c-down-exps.md`](p4c-down-exps.md).

| bank | row | PPL | vs 4.0289 | GB/token |
|---|---:|---:|---:|---:|
| D19 + D20 | 6.0 bits | 4.0970 | +1.69% | 4.279 |
| + `q4_1` | 5.0 | 4.1092 | +1.99% | 4.181 |
| + `iq4_nl` | **4.5** | **4.0992** | **+1.74%** | **4.132** |

**IQ4_NL dominates Q4_1**: half a bit narrower *and* a fifth of the accuracy
cost (+0.0022 points against +0.0122), so there was no trade to make. Paired
over the 145 chunks, Q4_1's cost is resolvable (t = 4.11, worse in 90 of 145)
and IQ4_NL's is not (t = 0.79) — which answers the one question the stage
existed to ask, since the sensitivity reading of unsloth's Q5_1 predicted a
real cost at 4.5 bits and the format reading predicted nothing.

Two things fell out of it. **Two other instruments ranked the two formats the
other way round** — the imatrix-weighted reconstruction error by 1.24x and
`ffn_out` against llama.cpp by 1.11x, both favouring Q4_1 — which is D3's
lesson at its sharpest: a reconstruction number is not a weak corpus number,
it is a different ordering. And **P4b's "ggml does not fit Q5_1 with an
imatrix" was wrong**: `quantize_q5_1` does, with our own three constants; the
two arms differ in `sigma2`'s scope and in D10's stored-half assignment, and
P4c's packers take ggml's normalisation so the C oracle can gate them.

*The original plan, for the record:*

**0.615 GB a token, 41% of the expert traffic and 14% of the whole decode
bank** — and the only reason it sits at 6.26 bits is that ggml had no
K-quant to offer a 640-wide row. Nobody has measured what narrowing it
costs, because until P4 the question was thought to be settled by unsloth's
choice.

- [x] Pick the format. All three candidates block by 32, so the row divides
      them: **IQ4_NL** (4.5 bits, −0.125 GB/tok, the calibrated non-linear
      form llama.cpp itself falls back to, and `gguf.dequantIQ4NL` already
      reads it for the n-gram table), **Q4_1** (5.0, −0.084, asymmetric and
      trivial to unpack), **Q4_0** (4.5, −0.125, symmetric — and D4 plus
      L8c-1's +18.5% are the reasons to expect the symmetric form to lose).
- [x] The kernel: a `down_` arm at a new `QFMT`, five GEMM rungs and six
      GEMV. `llm_moe_gemm.comp`/`gemv.comp` are already parameterised by
      `QFMT`, so this is an unpack and a `-D`, not a new kernel — but it
      *is* a kernel stage, which is what P4 got wrong.
- [x] Gate: the P4b shape. Format-is-the-fit round trip, a bit-identical
      staging control, one 145-chunk plan, three interleaved decode pairs.
      Expect **−0.125 GB a token, about +1.0 tok/s**, against an accuracy
      cost nobody has bounded — this is the first MoE row where the cost
      could plausibly be real.
- [ ] Carried forward from P4a, priced and not taken: **the router's
      padding**. 576 columns where 513 are real is **0.0118 GB a token,
      about +0.09 tok/s**, and it is `roundUpInt(NExpert+1, attnBN)`.

### P5 — MTP speculation  *(done, 2026-09-19 — built, lossless, and **0.95x**)*

**The one measurement the design was blocked on says speculation cannot be
built on the kernels that exist.** A verification pass over M rows costs
**3.24x a decode step at M = 2 and 4.32x at M = 8** (`-graph -tokens
1,2,3,4,6,8 -layers 24`, `results/p5_graph_m_r1.csv`), so even at *100%*
acceptance M = 2 yields 0.93x and the whole table is under 1.0x at any
realistic acceptance. Full write-up:
[p5-mtp-rollback.md](p5-mtp-rollback.md).

- [x] The rollback design doc. **The rollback is the easy part**: 120.75 MB
      — 36 DeltaNet states (113.25 MB), their convolution rings (7.13) and
      the PLE ring (0.37) — and both of the *free* mechanisms are already in
      the code. The state ping-pongs for **nothing**, because
      `llm_dn_scan.comp` loads S into registers once and stores it once, so
      a second destination adds no traffic; the rings are **deferred** rather
      than copied, because `llm_seq_hist.comp` is one dispatch at the end of
      a block reading the pass's own arena, so a commit re-issues it with the
      accepted count (37 dispatches, ~40 us a round). The KV cache, the
      pooled indexer table and the id list are free — a rewound cell is
      masked rather than stale, a pooled block is only written once complete,
      and the list is a slice.
- [x] **And a correctness finding the review's inventory would have missed**:
      *both* convolution rings are corrupted by a **single** rejected token,
      not only by a deep one. A ring is addressed by absolute position modulo
      its length, so the DeltaNet's −3 tap reads the slot position p writes
      (`hist` = 3) and the PLE's dilated −6 and −3 taps alias at depth 3 and 6
      of a 9-slot ring. Silent — wrong activations, no error.
- [x] The M = 2..8 verification-kernel decision, measured four ways. **Every
      block but the MoE takes one step up at M = 2 and is then flat to
      M = 8** — `hc` 2.1 → 13.9 → 13.8 ms, `dn` 4.2 → 8.3 → 8.5, `attn`
      1.2 → 2.7 → 2.7 — on *identical* weight bytes, so it is D15's refusal
      and nothing else. Only the MoE keeps climbing, and that half is real:
      **10 experts at M = 1, 17 at 2, 36 at 4, 49 at 8**, against a routed
      row of 1.327 GB of D21's 4.132. The four cold-bank ladders reproduce to
      **0.6% at M ≥ 2** (`results/p5_{moe,hc,dn,attn}_m_r{1,2}.csv`).
- [x] **P5a — the draft head as a passive observer** *(done 2026-09-19 —
      `a₁ = 74.0%`, and the wiring is settled by measurement)*. The head runs
      (`llm/mtp.go`, `go run ./cmd/llm -mtp`): `blk.48` staged as one
      `HCGPU`/`AttnGPU`/`MoEGPU` layer, the trunk's lm head **borrowed**, and
      the four `nextn` tensors on the host. **1024 rounds at n_ctx 2048,
      teacher-forced wikitext, against the trunk's own argmax: a₁ 74.0%,
      chained 65-72% to depth 6, E[tokens] 1.74 at depth 1 rising to 3.02 at
      6**, against a **zeroed-hidden negative control at 9.8%**. Acceptance is
      **flat in context** (73.4% at ctx 320) and the five-arm screen is
      identical count-for-count over two runs.
- [x] **The wiring had no oracle and now has a table.** Five readings
      measured against each other from one recorded trunk walk: the wide
      10240 residual with `hnorm` as `[2560, 4]` and **`eh_proj` once per
      stream** wins at 73.4%; the collapsed `result_norm` broadcast into four
      streams is 55.5%; and **the concatenation order is load-bearing** —
      flipped is **0 of 256**, so llama.cpp's `ggml_concat(e_norm, h_norm)`
      describes this checkpoint too. Two gates came with it: the draft's
      bundled `output`/`token_embd` **are** the trunk's (cosine 0.999792 and
      0.997280 — their own widths' cost and nothing else), and **Q5_0 and
      Q6_K** had to be added to `gguf` for a `Q4_K_M` shard, both exact
      against llama.cpp's `to_float`.
- [x] **And the free-running arm is an artefact.** Advancing on the trunk's
      own argmax reads **92.2%** and E[tokens] 5.56 — but printing the text
      shows greedy decoding at temperature zero in a **verbatim loop**, the
      same paragraph three times in 423 tokens, priming notwithstanding. A
      looping trunk is trivially predictable, so the arm is recorded and
      discarded, and P5c's gate inherits the rule: **a speculation multiplier
      measured on greedy self-generated text is measuring the sampler.**
      [Write-up](p5a-draft-head.md)
- [x] **P5b — the R-row decode kernels** *(done 2026-09-19 — 1.17x measured
      against 1.16x projected)*. All four decode GEMVs — `llm_gemv.comp`,
      `llm_hc_gemv.comp`, `llm_moe_gemv.comp`, `llm_moe_router.comp` — take
      **R rows per tile**: the weight is unpacked once and multiplied into R
      accumulators. **A two-row whole-graph pass is 19.5 ms against 16.6 at
      one row, where it was 54.1**, and one row is unregressed at 16.6
      against 16.7. Per block at two rows: `hc` 13.9 → **2.0** ms, `dn`
      8.3 → **4.1**, `attn` 2.7 → **1.2**, `moe` 25.5 → **8.2**.
      `results/p5b_graph_m.csv`.
- [x] **Three of the four blocks are *faster* at two rows than at one** —
      `hc` 0.51x, `attn` 0.82x, `dn` 0.97x — on identical weight bytes,
      because the second row gives a latency-bound dispatch a second stream
      of A loads to overlap. Only the MoE climbs, at 1.61x, and that is the
      routing: 10 experts at one token, 17 at two.
- [x] **`ROWS` is a specialization constant**, so the ~130 `.spv` of those
      four families do not double; one module serves both row counts and the
      guards fold at pipeline build. The MoE's ragged tail **needed no
      guard**: `llm_moe_perm.comp` already fills every routed row with a
      zero-pad sentinel before the scatter, so reading past an expert's real
      rows multiplies by zeros into slots the combine does not read.
      **D15's refusal moved rather than went away** — the bound is
      `GEMVMaxRows` in five places, and three tests now assert *both* halves
      of it. Residency cost: ~10 MB of partial arenas.
- [x] **And a false result that nearly shipped.** The `.spv` are generated
      and gitignored; the first round edited four `.comp` files without
      `go generate`, so every measurement ran the **previous** module. It
      looked like a triumph — the DeltaNet block 428.5 → 191.8 us, all three
      new tests green — because the old kernel computed **one** row and was
      handed a two-row grid. Comparing the GEMV's row 1 against the GEMM's
      row 1 cannot catch that: the arena still holds what the GEMM wrote.
      The control that can is `rowMoved` — change the second token, require
      the second output row to move — and it is now in all four gates.
      **A control has to be able to fail.** [Write-up](p5b-r-row-decode.md)
- [x] **P5c — the rollback and the loop** *(done 2026-09-19 — **lossless,
      and 0.95x**)*. Both gates met: `TestSpeculationRewindIsTheSequence`
      reproduces `result_norm` **to the last place** through rejected passes
      against a control that moves it by rms 8.5e-01, and the loop emits the
      plain loop's ids **token for token** at ctx 512 and at 2048 cells with a
      1024-token prompt. On the **shipped banks** the multiplier is **0.95x**
      short and **0.89x** long, over two runs each — 34.65 against 36.38 tok/s
      and 28.20 against 31.56, with the plain column agreeing with the carried
      36.19 to 0.6%. On the checkpoint's own widths it reads 0.98x at both, and
      that gap is §4.2 of the write-up: **narrowing the trunk makes speculation
      worse**, because the draft head stages at the checkpoint's own widths and
      does not shrink with it (0.176 of a step → 0.208), and because acceptance
      falls as the trunk moves away from the checkpoint the draft predicts.
      [Write-up](p5c-speculative-loop.md)
- [x] **Two of the design's five rollback entries were wrong, and both in the
      cheap direction.** The **deferred ring write is impossible**: `aQKV` is
      one arena shared by all 36 DeltaNet layers, so when a pass ends it holds
      layer 47's projection and there is nothing to re-issue the other 35
      from. The rings ping-pong instead, which is a second arm in
      `llm_seq_hist.comp` — every slot written, the ones this run did not
      produce carried over — and the one-row path keeps the pure write. And
      the **pooled indexer table does not self-heal**: a block is written the
      one time a run completes its cells and never again, so a block completed
      out of a rejected token is stale until every one of its cells is
      rewritten. `Rewind` snapshots it (a few kilobytes). It cannot bite below
      2051 cells, where the selection is the identity — which is where it
      would first have been turned on.
- [x] **Three defects in P5b, two of them silently wrong.** A two-row schedule
      was exercised by no gate in the suite, and the whole graph run two
      tokens at a time did not reproduce the prompt. (a) **`PinSchedule`
      stopped pinning at two rows** — its MoE branch asked `MoEPlanFor`, which
      P5b taught to name the GEMV up to `GEMVMaxRows`, so every bit-exactness
      gate in this vertical had a hole. (b) **The four MoE expert dispatches
      never named their R-row pipeline**, so every two-row batch ran the
      one-row kernel and each tile's second row kept what the arena held: a
      45x error on the residual, invisible to `TestMoEGPUDecodeTwoRows`
      because its `rowMoved` control is a control on the *combine*. (c) The
      R-row GEMV **re-read the bank per row** — `moe.up` at 101 GB/s against
      141 at one row — because "it is in the L1 after the first pass" is true
      on a two-layer fixture and false across 48 layers of expert bank.
      Moving the row loop inside the dword loop is **0.91x → 0.98x** on the
      checkpoint's own widths, one row unregressed.
- [x] **And 453 MB of host memset a pass.** `DeltaNetGPU.Resize` cleared the
      output projection's pad rows on every call and `sublayer` calls it once
      a layer — 12.6 MB × 36 at a 1024-token prompt. A prefill never saw it
      (`nTok == arenaRows`) and P1c's pre-recorded step never saw it (one
      Resize, not 36); a short *re-recorded* pass is the first thing in this
      vertical that is both. Clearing only what a longer pass dirtied is the
      long context's **0.91x → 0.98x** on the checkpoint's own widths.
- [x] **Parked, and it costs the product path nothing.** The rollback's
      120.75 MB is `GraphOpts.Speculative` and nothing else stages it; a decode
      graph has one slot per carried tensor and `Speculate(true)` refuses
      (`TestSpeculationNeedsItsSlots`). Measured against a pre-stage control on
      the shipped banks: `-gen -n 64` is **36.09 / 36.04 tok/s against 35.90 /
      35.93 on identical 795.6 MB arenas**, so the whole of P5c is free when it
      is off — and §2.4's memset fix is why the after column is the larger one.
- [ ] **What would make it pay, priced from the measurement.** The **recovery
      round** is worth 0.95x → ~1.05x: at R = 2 a rejection's re-run prefix
      fills the pass and displaces the draft, 25 rounds in 89, and removing it
      means committing a partial accept — either a store in
      `llm_dn_scan.comp`'s carried loop or (host-side only) **splitting the
      scan into two one-row dispatches**, since every row offset it reads is
      already a push field. **Pre-recording the verification pass** is ~2%
      (955 dispatches re-encoded a round against P1c's one replay; it is two
      buffers rather than one, because the state destination flips).
      **Acceptance** is the rest and it is now the *first* thing rather than
      the last: 65.6% on generated prose and 46.9% at a long context against
      P5a's 74.0% teacher-forced, where break-even is ~0.72. The whole
      remaining programme is worth **~1.08x at a = 0.66** and only becomes
      interesting above **a ≈ 0.8** — so measure the draft head on chat or code
      with P5a's observer (`cmd/llm -mtp`, no machinery, ~3 min) before
      building any of it.
- [ ] **A hard ceiling nobody had priced: speculation stops at 2048 cells.**
      `blk.48`'s `compress_ratios` entry is 0, so `NewMTPHead` refuses any
      context where the QSA selection is not provably the identity — against a
      trunk that prefills 8192 rows and prices 262k of KV.
- [ ] **Finding 6, and it is not speculation's: a fresh sequence is not
      independent of the one before it.** Three plain greedy runs over the
      same prompt in one process give three different texts, at token 24 and
      98 — *deterministically*, the same three every invocation, so decode is
      not racy and something survives `Reset`. It only shows where the
      previous run left cells past the next prompt's end (34 tokens in 512
      cells; at 1024 in 2048 every arm agrees). Not the recurrent state, not
      the rings, not the cache or the pooled table. **Any future "token for
      token" claim needs the plain-against-plain row printed beside it**, and
      `Reset` is owed a test that runs one prompt after different
      predecessors.

**What it is worth, now measured rather than assumed.** A round of depth M
verifies in a pass of **M+1 rows** and emits `j+1` tokens. The draft step is
6.99 ms in P5a's harness — **2.21 host `nextn` + 4.63 the layer's thirteen
submits**, against a byte floor of 0.130 of a step — so `c_d` is 0.130-0.166.
On the measured ctx-2048 profile:

| depth | rows | E[tokens] | before P5b | **with P5b, measured** |
|---:|---:|---:|---:|---:|
| **1** | 2 | 1.740 | 0.52x | **1.34x / 1.30x** |
| 2 | 3 | 2.222 | 0.59x | 0.59x (still the GEMM) |

So **`llm-vertical.md`'s carried 1.5-1.8x is not reachable on wikitext**: the honest
number is **~1.33x at depth 1**, and every depth was a loss before P5b. The
MoE's expert growth and the draft's lm head punish depth from opposite ends,
and a 74% draft is not enough to pay for either. **Raising the bound to three
rows buys nothing** — depth 2 would project to 2.222/(1.42+0.13) = 1.32x, the
same number — so `R = 2` is matched to the optimum rather than merely
convenient. What could still move it is the one thing nobody has measured:
**acceptance on a real workload**, where chat or code is more templated than
wikitext continuation.

> **P5c measured the whole of that row and it is 0.95x, not 1.33x.** Three
> numbers moved and none of them is the rollback. The verification pass is
> **1.36 steps, not 1.17** — P5b's figure is 24 layers, where the M-independent
> head, PLE and host are 24% of a step against 13% at 48. Acceptance on the
> loop's own traffic is **65.6%** on generated prose and **46.9%** at a long
> context, not 74.0%, against a break-even of ~0.72. And the round is a **two-state chain** rather than one
> round type: at R = 2 a rejection's re-run prefix fills the pass and
> displaces the draft, so a super-round is `(pass + draft) + (1−a)·pass` for
> exactly two tokens, which the design pass priced as free at general M and is
> not here. The measured chain, `(1.36 + 0.21) + 0.34 × 1.36 = 2.04` steps for
> two tokens, is 0.98x against a measured 0.95x. And **the width work makes it
> worse**: the draft head stages at the checkpoint's own widths, so narrowing
> the trunk took it from 0.176 of a step to 0.208 without changing it. See
> [p5c-speculative-loop.md](p5c-speculative-loop.md) §3.

### P6 — batching  *(pending the product question in idea 8 — and P5b built its first stage, which P5c then had to fix three times)*

- [ ] Decide whether the API serves concurrent streams; if yes, size the
      per-sequence state cost and the scheduler before optimising the
      batch-1 path further.
- [x] **P5's design answered half of idea 8's ordering question and P5b
      built it**: R concurrent sequences are R rows through the same weights,
      which is exactly P5b's kernel. It ships at **R = 2** because that is
      P5's optimum; the shaders are written as a `MAXROWS` bound, so raising
      it for batching is that constant in four shaders plus `GEMVMaxRows`.
      What P6 still owes is the per-sequence state cost and a scheduler.
      **P5c is the reason to trust it**: R rows were correct in a block test
      and wrong in the model three separate ways, and the gate that catches
      that is `TestGraphIsAChunkSplit` at two and three tokens, which is now
      in the suite. P6 should raise `GEMVMaxRows` and add the rung it wants to
      that test in the same commit.

### Parked (unchanged from `llm-vertical.md`, in one place)

W4A8 with the asymmetric epilogue (the GEMM arm is unmeasured); the unpack
prefetch (prefill lead, ~1.1x, "L8d's third lead"); the fp16 residual; the
hot-expert fast path; the float-atomic combine; the vision tower; the
llama.cpp rebuild question (a rebuild invalidates trace and baseline
together).

---

## Why this order

**P0, P1, P1a and P1b are done, and each of them moved the list.** P1's suspicion —
that the bank had stopped being the binding constraint — is confirmed, and
the replacement is **kernel shape**: four of the five leading items are
dispatches that are too small or the wrong way round, not bytes.

**P1a then sharpened what "too small" means, and it is not what the phrase
suggested.** Fusing the weightless half of the hyper-connection boundary was
worth its estimate (+0.34 tok/s, 94 dispatches, bit-exact); the *bank* half
was not a fusion question at all. A one-token dispatch on this device is
short of **workgroups**, and the down projection's own decode ladder measures
the law on identical bytes: **168 workgroups is 9.26 us where 672 is 5.54**.
`up` sits at 160 workgroups because MODE 1's collapse pins `BN` at 64, and
unpinning it is priced at 0.22 ms a token — real, and not worth every rung's
epilogue on the prefill path.

**P1b looked like the largest item on the list by a factor of four, and it
was the MALL's arithmetic.** The one-token MoE ladder those rungs were chosen
from was an L3 measurement end to end (every GEMV down rung above the bus);
re-run on sixteen cold banks the ranking does not move on any axis, so the
1.9 ms was never there. What is there is a **1.33 ms environment gap** — the
same dispatch is 8-22% slower inside a real step than alone on a cold bank —
which promoted P1c to the front of the list.

**P1c is done and worth its price — 1.90 ms of the estimated 1.96 — and its
lasting lesson is about the instrument, twice over.** Once small: the decode
step's entire varying state was a single uint, so "feed the varying scalars
through a uniform buffer" collapsed to one arena dword and a one-line macro.
Once large: the *machine* moved 8.7 ms a step overnight with yesterday's
binary — every dense-bank kernel 1.6-1.9x slower, every MoE expert rung
unchanged — so P1b's environment gap has a much bigger sibling that lives
across days rather than within a step, and every whole-model number now
needs a same-hour control the way D16 made every ladder need a cold bank.
P2 and P3 still close the accuracy story while additivity and the
instruments are warm, and they are small. P5 is the largest single
multiplier on the list but wants its own design pass, and its multiplier
applies on top of whatever the rest buys, so it loses nothing by going
after.

**P3 closes the accuracy story, and its lesson is about which question was
open.** The review framed the knapsack as a trade between `full_attn`'s
bytes and its perplexity, and both of the options it listed lost — the int8
arm to a plan L8c-3 had already measured, the `q5_k` arm to `bank_q4.go`
being nibbles by construction. The row that won was the one the list had
mis-costed: `ple_proj`, quoted at ~12 pp/GB from *halves* where every other
family was quoted from int8, and worth ~35 on the consistent basis. Put back
on int8 it is **0.367 points for 0.0165 GB and no measurable tok/s**, which
is D18. **What the stage really bought is the case for P3a**: `full_attn` at
`q5_k` is 14.1 pp/GB against the best buildable arm's 4.6 — and P3a built it
and measured 14.9, so the case was right and slightly under-stated.

Three instrument rules came out of it. **A plan is measured, never
composed** — additivity leaks 0.22 pp sub- and ~0.15 pp super-additively and
both leaks are the n-gram block. **An A/B on a difference this small is
interleaved** — one pair would have read −0.06 and another +0.14 on a real
difference of ~0.04. And **a perplexity delta names its corpus**: the same
plans rank identically on code at 2.5x smaller magnitudes, so +3.87% is a
wikitext figure.

**P3a built the plane and the accuracy frontier is now flat.** The fifth bit
behaved: the bank is the simulation to six decimals of nll over all 145
chunks, and P3's pre-kernel estimates landed within 6% on the three families
`sim.go` actually simulated. The one that missed — `deltanet`, 1.6 pp/GB
against an estimated ~2.8 — is the one P3 marked *inferred*, which is P3's
own rule ("a plan is measured, never composed") coming back one level up: an
estimate that was never simulated is a composition wearing a number.

The cut needed no new judgement. D18 had already refused a trade at 4.6
pp/GB and taken one at 22; the fifth bit offers 14.9, 7.8, 5.9 and 1.6, so
**D19 is D18's own line applied unmoved** — three families in, `deltanet`
out. What is left on the frontier is a sixth bit, which nobody expects to
rank differently, and the one item P3 carried forward: **the downstream task
eval**, now more valuable than it was, because a plan at +1.63% of wikitext
perplexity may be close enough to the unquantised model that perplexity no
longer separates the arms at all.

**P4 is the stage where two items dissolved on inspection and the
replacement was better than either.** The router had been fp16 since L5b and
`ffn_down_exps` has no K-quant at any accuracy, so the +4.5 tok/s of ceiling
P4 promised was 1.5 already spent and 3.0 that cannot be built. What was
there instead — four rows at 8.5 bits with a format the kernels already
read — is **+0.99 tok/s for 0.0022 points of perplexity**, which is 0.017
pp/GB against a frontier that refused to buy at 1.6.

The lasting part is not the tok/s. **Two of P1's nine attribution rows had
byte counts from a different source than the bank they were timing, and one
of them sent P1b to spend a stage re-screening sixteen cold banks for a
1.19 ms that did not exist.** D16 made every *rate* suspect above 242 GB/s;
P4a makes every rate suspect in its numerator too. And the same confusion
produced P4 itself: "the routers are F32" and "unsloth's imatrix says the
down projection is sensitive" are both readings of the **checkpoint** that
were carried as facts about the **bank** and about the **model**. Checking a
premise against the code cost an afternoon and retired two stages.

**The numbers to beat from here, each on its own day's control: 36.19 tok/s
on D19 + D20 + D21 (1.44x llama.cpp's 25.15, against a same-hour D20 control
at 35.46), perplexity 4.0992 (+1.74%) on 4.132 GB a token with a 58.6 tok/s
ceiling — and llama.cpp behind at every ubatch. The accuracy frontier is
closed; the throughput frontier had P5 and P6, and **P5 is now closed too and
it did not move the rate**: speculation is built, lossless and 0.95x, and what
would take it past 1.0 is priced in P5c's write-up rather than assumed. So the
throughput frontier is **P6 and P5c's three carried items**, and the largest
of those is not engineering — it is the draft head's acceptance on a workload
that is not prose.**

**And one instrument note out of P4c**, because it changes how the next
knapsack should be run: the frontier has been priced in *bytes* (pp/GB), and
bytes are not what the trade is about. Two families convert bytes to time at
rates that differ by 40% — 137 GB/s on the shared expert against 211 on the
routed down — so `iq4_nl`'s 0.34 pp/GB and `q4_1`'s 3.05 rank them correctly
only by accident of being on the same row. Price the next one in **pp per
tok/s**, which is the quantity both sides are denominated in.
