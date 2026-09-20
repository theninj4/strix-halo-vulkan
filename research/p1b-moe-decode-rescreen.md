<!-- LLM.md P1b. The MoE decode ladder re-screened against DRAM: sixteen
     cold banks and one iteration, because D16 says the two-layer fixture's
     rungs were MALL measurements. Cited from llm/gpu_moe.go (ProfileSweep),
     cmd/llm/bench_moe.go and results/p1b_moe.csv. -->

[← LLM.md](llm-vertical.md) · [← LLM2.md](llm-review.md) · [research index](README.md) · [P1](p1-decode-attribution.md) · [P1a](p1a-hyper-connection-shape.md) · phase 2

# P1b — the MoE decode rungs re-screened against DRAM, and a plan that survives its own bad evidence

**Result: no rung changes.** Every decode rate in `results/l8e_moe.csv` was a
MALL measurement — all six GEMV down rungs read 267–357 GB/s on a 242 GB/s
bus — but re-screened on sixteen cold banks the **ranking is unchanged**:
`v64w4/v16w4` routed, `v64w4/v32w4` shared and `k40` on the router win every
marginal, which is the plan the model already runs. The MALL inflated the
magnitudes, not the order. What P1b actually bought is the honest per-dispatch
floor, and against it **the whole-model gap is 8–22% and is not a rung
choice** — the largest single item LLM2's review named, "1.19 + 0.73 ms off a
mis-chosen row block", does not exist. What does exist is a **1.33 ms
environment gap** between a dispatch alone on a cold bank and the same
dispatch inside a real step, and it is a new question rather than a rung.

## The measurement D16 asked for

The l8e fixture staged **two** layers and timed each dispatch over 20
back-to-back iterations, so a one-token step's ten routed experts — 32 MB, the
MALL exactly — were re-read out of L3 nineteen times in twenty. P1b stages
**sixteen** layers (25 GB of bank) at `-iters 1`: `ProfileSweep` walks a
dispatch kind across the layers, so consecutive timings read different 1.57 GB
banks and every number is a cold-DRAM one.

    go run ./cmd/llm -moe -model $M -tokens 1 -ladder -layers 16 -iters 1 \
        -csv results/p1b_moe.csv

Run twice; over 679 common (plan, dispatch) rows above 5 us the median ratio
is **0.9991** with p10–p90 at 0.982–1.012, and both runs pick the same winner
on every axis.

## What the honest ladder says, per axis

us a layer at one token, best three rungs per marginal; l8e's contaminated
number for the same rung in the last column.

| axis | winner | 2nd | 3rd | winner's rate | l8e said |
|---|---|---|---|---:|---:|
| routed up | **v64w4** 99.0 | v32w4 102.2 | v16w4 114.3 | 186 GB/s | v16w4, 63.7 us "288 GB/s" |
| routed down | **v16w4** 59.3 | v32w4 61.0 | v64w4 61.7 | 207 GB/s | v16w4, 34.4 us "357 GB/s" |
| shared up | **v64w4** 22.7 | v32w4 34.4 | v32 38.8 | 150 GB/s | v64w4, 12.0 us |
| shared down | **v32w4** 10.9 | v16 11.4 | v16w4 13.3 | 151 GB/s | v32w4, 5.7 us |
| router | **k40** 13.4 | k8 15.3 | k10 20.0 | 195 GB/s | k40, 4.4 us |

The GEMM rungs stay far behind on both modes — the best GEMM down is `m1` at
90.8-93.0 us (135 GB/s) against the GEMV's 59.3, and the best GEMM up 230.5
against 99.0 — so L8d's kernel decision is also re-confirmed on cold banks.

Not one rung reads above 242 GB/s, which is the gate. Three corrections fall
out of the table:

1. **The up mode's D16 anomaly closes.** l8e's ladder said the routed up mode
   wants v16w4 and the whole model said that change was 42.1 ms *worse* over
   64 tokens — the contradiction that named D16. On cold banks v16w4 is
   **1.15x slower** than v64w4 (114.3 against 99.0), the same side as the
   whole model. The instrument now agrees with the truth it was checked
   against.
2. **The routed down rung was right by luck.** v16w4 wins cold at 59.3 us
   where the MALL said 34.4; the eight-rung spread is only 59.3–74.7, so
   there was never 1.19 ms to win here.
3. **The router was never 3.5x off.** P1's table put `moe.router` at 15.3 us
   a layer against a 4.4 us ladder rate and ranked it a leader; the 4.4 was
   two layers' fp16 router matrices (5.3 MB) sitting in the MALL. Cold, the
   k40 split reads 13.4–14.8 us — the whole model is within 10% of its own
   dispatch's floor.

## The gap that remains, and where it is not

Whole-model `-gen -attrib` on this build — 32.78 tok/s, step 30.51 ms, the
five 4.5-bit families staged, `results/p1b_decode_attrib.csv`, agreeing with
P1a's 32.93 within run-to-run noise — against the cold ladder, us a layer:

| label | in the model | alone, cold | gap |
|---|---:|---:|---:|
| moe.up | 106.9 | 99.0 | 1.08x |
| moe.down | 72.2 | 59.3 | 1.22x |
| moe.shexp.up | 26.4 | 22.7 | 1.16x |
| moe.shexp.down | 12.5 | 10.9 | 1.15x |
| moe.router | 15.1 | 13.4 | 1.13x |

The rung is the same dispatch over the same bytes in both columns, so the gap
is the *environment*: in a real step the bank a layer reads was last touched
a full pass — a token is 4.28 GB of reads between two visits to the same
layer — where the ladder revisits a layer after sixteen dispatches, and the
step's dispatch sits in a 1407-dispatch command buffer behind barriers rather
than alone with a fence. **1.33 ms of a 30.5 ms token** sits in this
environment gap across the five labels; attributing it (TLB and page-table
walks against command-buffer effects) is a new question, not P1b's, and it is
bounded under P1c's 1.96 ms.

## What this stage is, methodologically

L8e-3 (D16) caught one rung being chosen against the MALL and corrected it by
whole-model measurement; L8c-5 screened `l8d_dn.csv` and found every
one-token rung above the bus. P1b is the third reading and the general one:
**the entire one-token MoE ladder was an L3 measurement, and re-run against
DRAM it endorses the same plan**. A fixture whose working set fits 32 MiB
cannot rank kernels by DRAM behaviour except by luck — here the luck held,
because the rungs' relative order is set by grid width and lane-group shape,
which the MALL flatters uniformly at these sizes. The cheap screen stands:
any ladder row above 242 GB/s was not measuring DRAM, and now there is a
ladder invocation (`-layers 16 -iters 1`) that produces rows that pass it.
