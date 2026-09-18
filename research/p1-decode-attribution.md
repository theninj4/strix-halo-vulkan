<!-- LLM.md P1. Where a decode token's 30.5 ms goes, at the resolution of a
     dispatch label. Cited from llm/record.go, llm/graph.go, llm/model.go and
     cmd/llm/generate.go. -->

[← LLM.md](../LLM.md) · [← LLM2.md](../LLM2.md) · [research index](README.md) · [P0](p0-ring-watchdog.md) · [L8d](l8d-moe-decode.md) · [L8e](l8e-attn-decode.md) · phase 2

# P1 — the decode step, attributed: the 12 ms is three things, and one of them was disk

**Result: the step now sums.** A decode token is **30.52 ms** and every
microsecond of it has a name — 36 dispatch labels, six host phases, a residual
of 5 us. LLM2's "~12 ms a token unexplained by bytes at any achievable rate"
was real and is now **12.01 ms in three parts**:

    weight-streaming dispatches, below 227 GB/s     5.54 ms
    dispatches that stream no weight at all         3.46 ms
    the host, outside the command buffer            3.01 ms
                                                   -------
                                                   12.01 ms

**And the measurement paid for itself before the analysis did.** The host
third was dominated by the PLE n-gram gather at 2.24 ms a token, which turned
out not to be compute: sixteen random rows of a 28.80 GB mmap'd table are
**16.3 major page faults a step**, served one at a time. Issuing the same
sixteen concurrently — identical fault count, identical bytes — is **7.0-7.5x**
on the micro-benchmark and takes the gather from 2.244 to 0.902 ms in the whole
model. **Decode goes 31.51 → 32.72 tok/s, 1.30x llama.cpp.**

## The instrument

Every block's `graph` method has always built a `kinds []string` beside its
dispatches, for its error messages. Nothing read it: `recorder.add` took the
*block's* name and the per-dispatch timestamps L7d already collects were folded
into six rows. So the block table said the MoE is 11.6 ms of the token and
could not say which of its nine kernels that was.

`add` now takes the labels too, `GraphStats` carries a `map[string]DispatchStat`
beside the six block rows, and `-gen -attrib` prints it with the host side:

    go run ./cmd/llm -gen -model $M -n 64 -prompt 'The capital of France is' \
        -attrib -csv results/p1_decode_attrib.csv

A mark-to-mark interval is the dispatch **and** the barrier in front of it, so
the labels sum to the whole command buffer by construction and there is no
barrier residual to argue about. The host rows are wall clock around the loop's
own calls; `submit+fence` contains the dispatches, so the step is
`dispatches + (submit - GPU) + gather + record + sample + detokenize + emit`.
Two runs: **32.67 and 32.76 tok/s**, 0.3% apart.

## Where the token goes

30.52 ms a step, 1501 dispatches a pass in 2 command buffers, at the shipped
bank (five dense families at 4.5 bits, 4.281 GB a token). `GB/tok` is
`cmd/gguf -width`'s inventory for that family; `ms@227` is what those bytes
cost at the best dispatch rate ever measured on this machine (L8c-7's fused
attention projection, 39.5 MB in 174.3 us).

| family | GB/tok | ms | GB/s | % of 242 | ms@227 | lost |
|---|---:|---:|---:|---:|---:|---:|
| moe experts, gate+up | 1.003 | 4.998 | 200.6 | 82.9% | 4.417 | +0.58 |
| **moe experts, down** | 0.501 | 3.399 | **147.5** | 60.9% | 2.209 | **+1.19** |
| deltanet | 1.174 | 5.827 | 201.5 | 83.3% | 5.172 | +0.66 |
| **hyper_conn** | 0.360 | 3.142 | **114.6** | 47.3% | 1.586 | **+1.56** |
| lm_head | 0.358 | 1.936 | 184.9 | 76.4% | 1.577 | +0.36 |
| full_attn + indexer | 0.347 | 1.704 | 203.6 | 84.1% | 1.529 | +0.18 |
| moe_router | 0.252 | 0.764 | *329.8* | *136.3%* | 1.110 | −0.35 |
| **moe shared expert** | 0.251 | 1.839 | **136.5** | 56.4% | 1.106 | **+0.73** |
| ple_proj | 0.035 | 0.441 | 79.4 | 32.8% | 0.154 | +0.29 |
| **total** | **4.281** | **24.050** | **178.0** | **73.6%** | 18.859 | **+5.54** |

`moe_router` reads above the bus, which by **D16** is not a DRAM measurement:
its 2.63 MB a layer never leaves the MALL (D12), so its 0.252 GB is in the
*inventory* and not in the traffic. It is excluded from the loss column.

Beside those, **3.46 ms of dispatches that stream no weight at all** (11.3% of
the step) — the recurrence, the moves, the permutations and the norms:

    dn.scan 0.638   hc.norm 0.610   move 0.563   hc.combine 0.359
    moe.route 0.285  attn.attn 0.194  moe.combine 0.160  dn.conv 0.141
    moe.perm.up 0.139  dn.norm 0.125  attn.score 0.114  attn.pack 0.091
    dn.hist 0.036  ple.conv 0.004  ple.hist 0.001

And **3.01 ms of host** (9.9%):

    record          1.127     building push constants for 1501 dispatches,
                              plus the 0.99 MB logits read-back
    gather (ple)    0.902     the n-gram table, below
    hand-over       0.828     submit + fence, minus the GPU time inside it
    sample          0.142     argmax over 151 936 logits
    detokenize      0.002
    emit            0.005
    unattributed    0.005

## Finding 1 — the n-gram gather was disk latency, and it was serial

`Model.PLEGather` reads sixteen rows of `per_layer_token_embd` a token: 90
bytes each, 1.41 KB in total, at sixteen *random* offsets into a 320-million-row,
**28.80 GB** mmap'd table that D2 deliberately keeps off the device. On a
machine holding 84 GB of resident model those rows are not in the page cache.
Counting `majflt` around 200 gathers of fresh random rows:

| arm | us a step | minflt a step | **majflt a step** |
|---|---:|---:|---:|
| serial (as written) | 853.2 / 819.4 | 0.6 / 0.1 | 16.27 / 16.29 |
| sixteen touched concurrently, then gathered | **122.2 / 108.8** | 0.3 / 0.1 | 16.34 / 16.30 |

**The fault count is identical to two decimal places.** Nothing is read that
was not read before, no readahead is re-enabled, no page is cached that was not
cached. What changes is that sixteen faults are outstanding at once instead of
one at a time — 7.0-7.5x, and the whole of it is device latency.

So the gather goes parallel: a bounded worker pool over the rows, destinations
disjoint, one scratch buffer a worker. Value for value it is the same function.
In the whole model the gather falls **2.244 → 0.902 ms a token**, and the step
**31.84 → 30.52 ms**.

This is the other half of **D9**. `MADV_RANDOM` made each fault small — sixteen
128 KB readahead windows to deliver 1.41 KB was 176x at L7c — and what was left
was that they were *serialised*. A random-access mapping wants both: small
faults, and several of them in flight.

The residual 0.902 ms is still ~16 faults a token and still the third-largest
host row. It is larger than the micro-benchmark's 0.11-0.12 because in the whole
model the page cache is competing with 84 GB of resident weights. Nothing
cheaper is available while D2 holds: the table is 28.80 GB and cannot be
resident beside the model.

## Finding 2 — the hyper-connection block is the worst-spent time in the step

Seven of eight weight-reading families are between 147 and 204 GB/s. The
hyper-connection mixers are **114.6**, and they are not short of bandwidth —
**D14** established that these weights fit the 32 MiB MALL at every width, so
they pay only their own unpack. What they are short of is work per dispatch:
97 mixers × 5 dispatches = **485 dispatches a pass at 1.4-17 us each**.

Counting everything the block owns — `hc.up` 1.616, `hc.down` 1.435,
`hc.norm` 0.610, `hc.combine` 0.359, `hc.down_reduce` 0.130 — it is **4.15 ms,
13.6% of the step, for 8.4% of the bytes**. It is the largest single
misallocation in the token, and the lever is fusion or batching across mixers,
not a narrower bank: L8c-6 already took this family to 4.5 bits and D14 warned
the hazard would fire here.

## Finding 3 — the MoE's down projection is 1.19 ms behind its own up

`moe.up` and `moe.down` read the same bank, the same experts, the same layer,
back to back — and they measure **200.6 against 147.5 GB/s**. The difference is
the shape: up is [FFNExpert, NEmbd] and down is [NEmbd, FFNExpert], so the
split and the row block that D11/D12 chose for one were never re-screened for
the other. The shared expert's pair repeats it at a smaller size
(**136.5 GB/s**, 0.73 ms behind).

Both are D12's operative half — *re-measure when the width changes* — applied
to a kernel rather than to a bank, and both are a ladder run away from an
answer. Together they are **1.92 ms, 6.3% of the step**.

## Finding 4 — record + hand-over is 1.96 ms, and now it is priced

Idea 2's pre-recorded decode command buffer has never had a number. It has one:
**1.127 ms of host recording plus 0.828 ms of hand-over is 6.4% of the step**,
about **+2.2 tok/s**, and the decode graph is shape-stable token to token — same
1501 dispatches, same buffers, only the position and the token id change. That
is worth building, and it is the *fourth* item rather than the first, which is
what P1 was for: before the measurement it was the leading candidate for the
whole 12 ms.

## Finding 5 — the 0.38 GB, reconciled (idea 9)

`LLM.md`'s budget said 4.830 GB of dense traffic a token; L8b's own table said
4.45, and "nothing has reconciled the two tensor by tensor". Group by group:

| group | inventory | graph's table | why they differ |
|---|---:|---:|---|
| deltanet | 2.247 | 2.25 | — |
| hyper_conn | 0.695 | 0.74 | L8b-1's doubled `inject` rows, 0.33 MB × 97 mixers (D13) |
| lm_head | 0.675 | 0.68 | — |
| full_attn + qsa_indexer | 0.674 | 0.70 | rounding |
| ple_proj | 0.035 | 0.07 | `PLEGPU` staged halves — the one family D13 never reached, and P2's |
| **moe_router** | **0.252** | — | filed under the MoE block, not under "dense" |
| **moe shared expert** | **0.251** | — | same: counted in L8b's "11 of 512" expert row |
| total | 4.830 | 4.45 | |

**The 0.38 GB is the router and the shared expert**, which the checkpoint
inventory files as dense-every-token and the graph's table files with the MoE,
less the three doublings above. Neither count was wrong; they were different
partitions of the same tensors.

The attribution table settles it, because it is a *third* basis that names both
explicitly: `moe.router` and `moe.shexp.*` are rows in it. **Every ceiling in
`LLM.md` and `LLM2.md` is now quoted on one basis: 4.281 GB a token at the
shipped bank**, which is 56.5 tok/s at 242 GB/s and **53.0 at the 227 GB/s
that dispatches actually reach**.

## What this re-ranks

The honest ceiling is not 53 tok/s of weights alone; it is 53 minus whatever of
the 3.46 ms of weightless dispatches and the 3.01 ms of host cannot be removed.
Measured 30.52 ms against 18.86 of weights at 227 GB/s is **62%**, and the
recoverable items, in order:

| ms | item | lever |
|---:|---|---|
| **1.56** | `hyper_conn` at 114.6 GB/s | fuse or batch across 97 mixers; not a bank change (D14) |
| **1.19** | `moe.down` at 147.5 against `moe.up`'s 200.6 | re-screen the down rung (D11/D12) |
| **1.96** | record + hand-over | idea 2's pre-recorded command buffer |
| **0.97** | `hc.norm` + `hc.combine`, no weights | same fusion as the first row |
| **0.73** | the shared expert's pair at 136.5 | the same ladder as `moe.down`; P4 narrows it too |
| **0.29** | `ple_proj` at 79.4 — one 411 us dispatch a pass | P2, and check its grid against D11 |
| 0.64 | `dn.scan`, the recurrence | idea 6 (fp16 state) is the only named lever |
| 0.56 | `move` | the shared-arena change `GraphStats` has warned about since L6b |

Four of the first five are kernel shape, not bytes. **The bank has stopped
being where decode time is**, which is what the 2026-09-18 review suspected and
could not show; P3 and P4 are still worth doing for the *ceiling*, but the
30.52 ms step is 62% efficient against that ceiling and the gap is now itemised.

## Files

- `llm/record.go` — `add` takes the labels; `submit` returns `map[string]DispatchStat`
- `llm/graph.go` — `GraphStats.Kinds`, `GraphStats.Submit`
- `llm/model.go` — `PLEGather`, parallel
- `cmd/llm/generate.go` — `-attrib`, the host phases, `writeAttribCSV`
- `results/p1_decode_attrib.csv` — the table above, as measured
