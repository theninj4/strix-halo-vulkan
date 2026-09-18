# P2 — `ple_proj` on the bank, and the fp16 tail deleted

> 2026-09-18. The last streamed dense family staged at 4.5 bits — two bank
> stages in one, since the PLE block never got L8a's int8 stage either — and
> then D13's payoff taken: the two-plane fp16-tail machinery deleted from
> three kernels and three blocks, so every row of a quantised bank is on the
> plane and **the bank is the simulation with no scope carve-out**. This
> closes L8c: the complete uniform plan, measured as built, is
> **4.2010 (+4.27%) over 145 chunks**, beside the 4.1998 (+4.24%) L8c-3's
> simulation projected before any kernel existed. Decode the same hour:
> **27.87 ms a step, 35.89 tok/s, 1.43x llama.cpp's 25.15**.

## 1. The family: two matrices, one dispatch, three banks

`ple_key` [10240, 2560] and `ple_value` [2560, 2560] ship as Q8_0 and are
staged fused, as one [12800, 2560] B whose kernel is MODE 2 of
`llm_gemm.comp` — the mode that exists *for* this block. Both dimensions are
multiples of 256, so there is no tail, no pad row that carries data, and no
new kernel: the same `-DQ4B` builds the attention block runs (`LLMGEMMQ4M*`)
and the same int8 builds (`LLMGEMMQ8M*`) bind the bank at binding 5.

    bank        staged      the fused projection at one token
    fp16        65.8 MB     428 us on P1b's day (153 GB/s)
    int8+scale  35.1 MB     263 us today
    q4_k/32     18.7 MB     217 us today

The int8 stage is bit-identical arithmetic (both tensors ship Q8_0;
`TestPLEGPUQ8IsTheHalves` — key, value and the whole residual identical to
the fp16 arm). The 4.5-bit stage is the format the simulation measured
(`TestPLEGPUQ4IsTheSim` — identical to `sim.go`'s round trip through the
same encoder), each tensor calibrated under its own name; unsloth's imatrix
has rows for both. At the graph level the bank agrees with
`LLM_DENSE_SIM=ple_proj=q4_k/32` **chunk for chunk over eight chunks**, the
`nll` column identical to six decimals.

The dispatch is not why this family was staged — it runs once, at layer 1 —
but the ladder above says the same thing P1a's did: 217 us for 18.4 MB is
85 GB/s, because 200 workgroups is not enough grid for this device, so the
bytes saved return less time than the bus would predict.

## 2. The accuracy: the most expensive bytes in the plan

`ple_proj` alone over the corpus: **4.0528, +0.59%**, for 0.047 GB a token
back from halves (0.017 from int8). That is the worst pp-per-GB of the six
families by an order of magnitude — the bimodal split of L8c-7 (families at
36 layers ~0.9 pp/GB, families at one matrix 2.6-5.5) gains a third rung:
a family present at **one layer, run once**, costs ~12 pp/GB from halves.

And the eight-chunk screen read **−0.11%** (2.0246 against the baseline's
2.0269) where the corpus says +0.59% — a fifth reading of L8c-4's warning
and the second sign flip. The screen still does not bound, fix the sign, or
rank.

The complete plan with the tails still fp16 measured **4.1939, +4.09%** —
against a sum of the six families' separate deltas of 4.31%. The first
material deviation from additivity (0.22 pp where four families summed to
0.01): `ple_proj` interacts where the first five did not. Both facts go to
P3's knapsack: this family is the plan's worst trade, and additivity is now
known to hold only approximately once the n-gram block is in play.

## 3. The deletion: D13's payoff, priced

Every tail was priced — alpha, beta and inject at int8 for **+0.01%**
(L8c-1), the indexer at 4.5 bits for **+0.005%** where the selection bites
(L8c-7) — so the machinery went:

- **Kernels**: the `Q8_TAIL` branch and its `qSplit` left `llm_gemm.comp`;
  both split-K tail arms left `llm_gemv.comp`; the split, `DOWN_BN` and the
  tail loop left `llm_hc_gemv.comp`. `pc.gateOff` means a weight nowhere;
  `pc.lowRank` survives only as MODE 0's epilogue split (inject's columns)
  and MODE 1's K.
- **Blocks**: `qkvTail`/`downTail` offsets, the tail staging, the doubled
  low-rank rows of L8b-1 and the `idxQuant` flag are gone. On a quantised
  bank alpha, beta, inject and the indexer's two projections are quantised
  onto the plane at their family's width, each under its own imatrix name.
  `qsa_indexer` must match `full_attn`'s format if named — they share one
  plane — and the graph refuses a mismatch.
- **The int8 bank is no longer bit-identical arithmetic** on those four
  tensors: F32 alpha/beta/inject and BF16 indexer through int8 is a real
  re-quantisation. The trace tests carry that as `bankTol` — the tight
  fp16-bank bound kept under `LLM_DENSE_FP16=1`, a derated bound on the
  quantised bank at D13's measured prices (alpha/beta rms 3.0e-3, exactly
  L8a-2's number; inject 2.7e-2; indexer 3.7e-3) — and
  `TestGraphLogits` tolerates an argmax flip only where the reference's own
  gap between the two tokens is within the measured logit drift (here
  0.107 of a 15.86 logit against rms 0.33: a near-tie resolved the other
  way).

What the deletion is *for* is the sentence at the top: with no tail, the
bank quantises exactly the set of weights the simulation always did, and
the two agree **chunk for chunk over the complete plan** with no `SRC=q8`
qualifier. Every number L8c-3 simulated is now a number about the shipped
bank.

## 4. The closing numbers

Perplexity, 145 chunks of 2048, against 4.0289:

| plan | ppl | delta |
|---|---:|---:|
| five families (P1c's state) | 4.1787 | +3.72% |
| + `ple_proj` at q4_k | 4.1939 | +4.09% |
| + the tails on the plane — **the complete plan as shipped** | **4.2010** | **+4.27%** |
| L8c-3's simulation of that plan, before any kernel | 4.1998 | +4.24% |

Decode, all in one hour on one machine state (P1c's rule):

| arm | step | tok/s |
|---|---:|---:|
| five families, ple fp16-staged on the int8 default | 28.400 ms | 35.2 |
| + `ple_proj` on the bank (`ple.kv` 263 → 217 us) | 28.336 ms | 35.3 |
| + the tails deleted | **27.865 ms** | **35.89** |

35.89 is **1.43x** llama.cpp's 25.15. The deletion returned 0.47 ms a step —
the tails' separate reads across 36 layers, 97 mixers and 12 layers — which
at the measured streaming rate is ~0.085 GB a token, and with `ple_proj`'s
0.047 the streamed bank is ~**4.15 GB a token** by subtraction from P1's
4.281 (a fresh inventory is P3's to make when it sets D18). Prefill at 2048
is unregressed (35.0 tok/s at the 5-token smoke prompt's prefill; the
corpus's 2048-token chunks ran at the usual 2.4-2.5 s).

## 5. What this hands P3

- The knapsack has a sixth row, and it is the worst one: `ple_proj` buys
  0.047 GB for +0.59% where `deltanet` buys 1.07 GB for +0.93%. An int8
  `ple_proj` (bit-identical, 0.017 GB dearer than q4_k) is the obvious arm
  to price against `full_attn` at q5_k.
- Additivity now has a measured exception (−0.22 pp), so D18's winning plan
  should be corpus-run as a whole, not composed.
- The tails are on the plane at the family width. If D18 moves a family's
  width, the tails move with it — there is no second plane to decide about.

Files: `llm/gpu_ple.go` (the bank stages), `llm/gpu_attn.go`,
`llm/gpu_deltanet.go`, `llm/gpu.go` (tail deletion), `shaders/llm_gemm.comp`,
`shaders/llm_gemv.comp`, `shaders/llm_hc_gemv.comp` (the branch deletion),
`llm/graph.go` (the plan wiring), tests in `llm/ple_test.go` and the three
block test files. Results: `results/l8c_ppl_ple_only.csv`,
`results/l8c_ppl_uniform.csv` (tails fp16),
`results/l8c_ppl_uniform_notail.csv` (as shipped).
