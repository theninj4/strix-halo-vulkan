# P5b — the R-row decode kernels: a two-row pass at 1.17x instead of 3.24x

> **2026-09-19.** P5's design pass found that a verification pass over M rows
> costs **3.24 decode steps at M = 2**, so speculation was a net loss at every
> depth and every acceptance rate, and named the fix: the four decode GEMVs
> with R rows per tile instead of one. P5a then measured the acceptance rate
> at **74.0%** and found the optimum depth to be **1** — a two-row
> verification pass — which scoped this stage from "R general" to **R = 2**.
>
> **Built and measured. A two-row pass is 19.5 ms against 16.6 at one row —
> 1.17x, against the 1.16x the design pass projected — where it was 54.1.**
> One row is unregressed at 16.6 against 16.7, which matters more than the
> headline: that is the product decode path.

Data: `results/p5b_graph_m.csv` against `results/p5_graph_m_r1.csv`.
Code: `shaders/llm_gemv.comp`, `llm_hc_gemv.comp`, `llm_moe_gemv.comp`,
`llm_moe_router.comp`, and `GEMVMaxRows` in `llm/bank.go`.

---

## 1. The measurement

`-graph -tokens 1,2,3,4,6,8 -layers 24 -ctx 2048`, D19 dense bank + D20/D21
MoE bank, the same configuration P5's baseline was taken on:

| M rows | before | after | | hc | ple | dn | attn | **moe** | head |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 16.7 | **16.6** | 1.00x | 2.1 | 0.3 | 4.0 | 1.3 | 5.3 | 2.4 |
| **2** | **54.1** | **19.5** | **1.17x** | **2.0** | 0.3 | **4.1** | **1.2** | **8.2** | 2.4 |
| 3 | 58.3 | 57.6 | 3.47x | 14.0 | 0.3 | 8.7 | 2.7 | 28.5 | 2.4 |
| 4 | 59.6 | 60.1 | 3.62x | 14.1 | 0.3 | 9.0 | 2.7 | 30.7 | 2.4 |
| 6 | 65.4 | 65.2 | 3.93x | 13.9 | 0.3 | 8.5 | 2.7 | 36.4 | 2.4 |
| 8 | 72.1 | 72.7 | 4.38x | 14.0 | 0.3 | 8.7 | 2.7 | 43.5 | 2.4 |

Rows 3 and up are **unchanged by construction**: `GEMVMaxRows` is 2, so the
host still falls back to the GEMM there, and the fact that those rows land
within 1% of the baseline is the cheapest available check that nothing else
moved.

Per block, cold-bank ladders at the rung the model runs:

| block | ×N | M=1 before → after | M=2 before → after | M=2/M=1 now |
|---|---:|---|---|---:|
| hyper_conn | 97 | 55.6 → 78.9 us | 273.1 → **40.3** | 0.51x |
| deltanet | 36 | 201.1 → 197.8 | 428.5 → **191.8** | 0.97x |
| full_attn | 12 | 230.4 → 239.4 | 420.9 → **197.3** | 0.82x |
| moe | 48 | 196.2 → 204.7 | 1149.3 → **329.2** | 1.61x |

(the M = 1 column moves by up to 1.4x between runs on `hc` and `dn` — these
are the smallest dispatches in the model and P1b's environment gap is at its
widest here. The M = 2 column and the whole-graph table are what the argument
rests on. Only one whole-graph run was taken rather than the usual two: the
effect is 2.8x and run-to-run spread on this instrument is a few per cent.)

### Finding 1 — three of the four blocks are *faster* at two rows than at one

`hc` 0.51x, `dn` 0.97x, `attn` 0.82x. Not "flat" — **faster**. The weight
traffic is identical, and the second row gives the memory system a second
independent stream of A loads to overlap against the same bank read, on
dispatches that were latency-bound rather than bandwidth-bound at one row.
The MoE is the only block that climbs, at 1.61x, and that half is real: it is
the routing, 10 experts at one token and 17 at two.

This is D11's law one step further along. At one token a dispatch here is
short of *workgroups*; at two rows on the old kernel it was short of *rows*,
sixteen to sixty-four of them with two real. The R-row arm has neither
problem, and the spare row turns out to be worth something rather than
nothing.

---

## 2. What was built

One change, four times: the weight is unpacked once and multiplied into R
accumulators against R rows of A staged from R places.

| kernel | dispatches it serves | what R rows needed |
|---|---|---|
| `llm_gemv.comp` | `dn.qkv`, `dn.out`, `attn.qkv`, `attn.out` | R accumulators, A at `+ r*lda`, partials `[r][slab][n]`, the reduce's grid gains a row axis |
| `llm_hc_gemv.comp` | `hc.down` (97 a pass) | the same, plus an epilogue that writes each row's gate at `ldaLo` and its `inject` tile at `gemmN - lowRank` |
| `llm_moe_gemv.comp` | `moe.up`, `moe.down`, `shexp.up`, `shexp.down` | R A-vectors in LDS (10 KB at K = 2560), R rows read from the tile's base |
| `llm_moe_router.comp` | `moe.router` | R accumulators over an fp16 matrix that never leaves the MALL |

**`ROWS` is a specialization constant, not a `-D`.** These four families are
about 130 `.spv` between them — every split-K rung over every bank, every
lane-count over every expert format — and a compile-time define would have
doubled all of them. One module serves both row counts, the `r < ROWS` guards
fold when the pipeline is specialized, and the cost is one extra pipeline per
rung. `MAXROWS` is the compile-time bound the accumulators and the LDS are
sized by; raising it is that constant in four shaders plus `GEMVMaxRows`.

### 2.1 The ragged tail needed no guard, because the permutation already had one

The MoE was the one that looked like it needed bookkeeping. A tile is
(expert, row block) and at two tokens an expert may hold one real row or two,
so an R-row kernel reading past the real rows would read a neighbour's
activation.

It does not, and `llm_moe_perm.comp` is why: *before* the scatter, every
routed row is filled with a sentinel — a `(token, slot)` pair one token past
the batch, whose activation row is a zero pad — so what the scatter does not
overwrite multiplies by zeros and lands in output slots the combine never
reads. That is exactly what the grouped GEMM already does with the same rows.
The R-row arm inherits it for free, which is the whole reason this stage was
four kernels and not four kernels plus a schedule change.

### 2.2 D15's refusal moved rather than went away

"At one token a block runs a different *kernel*, not a different rung, and the
host refuses it above one token." The refusal was about the GEMM's shape — a
sixteen-row fragment holding one row — and not about a dot product's, so what
it bounds is now `GEMVMaxRows` rather than 1, in five places: `SetGemv` on the
gated DeltaNet and the full-attention layer, `SetPlan` and `moeCheckGemv` on
the MoE, `SetRouter`, and `HCGPU.SetPlan`. **Past the bound it still refuses**
— a GEMV there would silently drop rows, which is the hazard D15 exists for —
and three tests now assert *both* halves, because a refusal that refused
everything would pass the old assertion.

Residency: the partial-sum arenas are sized by `GEMVMaxRows` and cost about
**10 MB** across the four blocks, against 78.5 GB.

---

## 3. The gates

Four new tests, one per block, each comparing the R-row GEMV against the GEMM
over two rows and each checking the second row on its own:

| test | arms | agreement |
|---|---|---|
| `TestDeltaNetGPUGemvTwoRows` | every rung of `qkv` and `out` | rms 1.2e-5 |
| `TestAttnGPUGemvTwoRows` | every rung of both projections | rms ≤ 1e-4 |
| `TestHCGPUGemvTwoRows` | every down rung × `lo`, `inject`, `mixed` | maxAbs ≤ 1e-3 |
| `TestMoEGPUDecodeTwoRows` | four expert/router plans | rms 2.6e-6 |

The tolerances are the ones the one-row tests already use, and for the reason
`Graph.PinSchedule` gives: the two paths associate a 2560- or 6144-long sum
differently, they do not read different weights.

### Finding 2 — the row-against-row comparison is not a control, and it nearly shipped a false result

Comparing the GEMV's second output row against the GEMM's second output row
**passes trivially when the GEMV never writes that row**: the arena still
holds what the GEMM put there on the reference pass, which is exactly the
value being compared against.

That is not hypothetical. The `.spv` in this repo are generated and
gitignored, and the first round of this stage edited four `.comp` files
without running `go generate` — so every measurement ran the **previous**
module. The result looked like a triumph: the DeltaNet block went from 428.5
us at two rows to 191.8, the attention layer from 420.9 to 197.3, and all
three new tests passed. The old kernel was computing one row and being handed
a two-row grid, so it was genuinely twice as fast per row — at half the work.

What caught it was asking a question the comparison could not answer: *is row
1 being written at all?* Feed a different second token and require the second
output row to move. It did not. `rowMoved` is now that assertion, in all four
tests, with the reason in its comment.

Two rules out of it, and the second is the general one:

- **`go generate` is not optional after a `.comp` edit**, and a hand-compile
  to `/tmp` to check syntax is not it. `strix-halo-bench-runs` already said
  so; this is what ignoring it looks like.
- **A control has to be able to fail.** Comparing a new path against an old
  one over a buffer the old one wrote is a comparison, not a control. The
  control is the one that changes an input and requires the output to change.

---

## 4. What it is worth

A round of speculation depth M verifies in a pass of **M+1 rows** and emits
`1 + Σ survival` tokens. With P5a's measured profile (ctx 2048, teacher-forced
wikitext, `a₁ = 74.0%`) and the draft step at 0.130-0.166 of a decode step:

| depth | rows | E[tokens] | pass cost | before | **after** |
|---:|---:|---:|---:|---:|---:|
| **1** | 2 | 1.740 | **1.17** | 0.52x | **1.34x / 1.30x** |
| 2 | 3 | 2.222 | 3.47 (GEMM) | 0.59x | 0.59x |

**Depth 1 is where the multiplier is, and the R = 2 bound is matched to it
rather than merely convenient.** Raising `GEMVMaxRows` to 3 would put depth 2
on the R-row path at a projected pass cost of 1.42 steps, which is
2.222/(1.42+0.13) = **1.32x** — the same number depth 1 already gets. The MoE's
expert growth eats the extra acceptance exactly as fast as the acceptance
arrives, so there is no third row worth having for P5.

**P6 is the reason to raise it anyway.** R concurrent sequences are R rows
through the same weights, and there the expert-set growth is the price of
serving more streams rather than a tax on one. The shaders are written as a
bound rather than as a second accumulator for that reason; `MAXROWS` and
`GEMVMaxRows` are the whole change.

---

## 5. What is owed

- **A whole-model decode control.** The no-regression evidence here is the
  24-layer M = 1 column — 16.6 ms against 16.7 — and the one-row tests, which
  are bit-comparable. A 48-layer `-gen` against a same-hour control is the
  gate this stage should have before the result is quoted as tok/s, on P1c's
  rule.
- **A second whole-graph run.** One was taken; the convention is two.
- **P5c**, which is now the last item: the rollback as designed
  (research/p5-mtp-rollback.md §3 — the ping-pong state slot, the deferred
  `hist` dispatches, `Graph.Commit`/`Rewind`), the loop, and
  `TestSpeculationRewindIsTheSequence`. With P5b landed it is the only thing
  between here and the 1.34x.
- Carried from P5a and unchanged: acceptance **on a real workload**, `nextn`
  on the device, and a recorded draft step.
