# LLM2 — the review, the hypotheses checked, and the next priority list

> **Written 2026-09-17**, as a review of `LLM.md` at L8c-7 / L9a. `LLM.md`
> stays the stage log; this file is the forward-looking list. Same
> conventions: history to `TODO.md`, closed findings to `research/`.

## Where this stands, in five lines

- **Decode 35.89 tok/s** against llama.cpp's 25.15 — **1.43x**, measured
  the day P2 landed (27.87 ms a step; same-hour ladder 28.40 → 28.34 →
  27.87) — at a streamed bank of ~4.15 GB a token after P2. P1 attributed
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
- **Perplexity 4.2010 against our 4.0289 — +4.27%** (and +4.14% against the
  reference's own 4.0340), with **all six** dense families at 4.5 bits and
  no fp16 tail anywhere — the bank is L8c-3's simulation (which projected
  +4.24%) with no scope carve-out. Additivity broke at the sixth family:
  the separate deltas sum to 4.31% where five-families-plus-`ple_proj`
  measured 4.09% (P2).
- **Served**: `cmd/serve -llm`, three envelopes over one loop, prefix reuse.
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
| **"Decode is DRAM-bound by weight bytes"** (the premise of D3, L8a-L8c) | **Was true, is now 79% true, and P1 says where the rest is.** Through L8b, measured ≈ ceiling: 14.71 tok/s against 15.4 GB-derived numbers. Today the weight-streaming dispatches are **24.05 ms of a 30.52 ms step, 79%** — but they run at **178 GB/s against the 227 a dispatch reaches**, so 5.54 ms of that is shape rather than bytes, and another 3.46 ms of dispatches stream no weight at all. The binding constraint has moved from *how many bytes* to *how the dispatches are shaped*: `hyper_conn` at 114.6 GB/s and `moe.down` at 147.5 against `moe.up`'s 200.6 off the same bank. |
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
the bus (`hyper_conn` at 114.6 GB/s, `moe.down` at 147.5) and the 3.46 ms of
dispatches nobody had counted because they read no weights at all. The one
candidate that was under-rated is the PLE host gather, which was 2.24 ms and
is now 0.90 — see P1's finding 1.

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
Full write-up: [research/p0-ring-watchdog.md](research/p0-ring-watchdog.md).

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
      **1.96 ms** and now ranks fourth. [Write-up](research/p1-decode-attribution.md)

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
      [Write-up](research/p1a-hyper-connection-shape.md)

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
      [Write-up](research/p1b-moe-decode-rescreen.md)

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
      [Write-up](research/p1c-prerecorded-decode.md)

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
      L8c closed in `LLM.md`. Streamed bank ~4.15 GB a token by subtraction;
      the fresh inventory is P3's.
      [Write-up](research/p2-ple-proj.md)

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

**The numbers to beat from here, each on its own day's control: 35.89 tok/s
(27.87 ms a step, P2's day, 1.43x llama.cpp's 25.15), ~54.5 honest ceiling
on the ~4.15 GB bank at the 227 GB/s dispatches reach — and llama.cpp
behind at every ubatch. P3 (the shipped-widths decision) now owns the
accuracy story: its knapsack gained a sixth row (`ple_proj`, the worst
pp-per-GB of the six) and lost additivity as a free tool.**
