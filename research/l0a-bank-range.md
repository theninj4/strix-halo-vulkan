<!-- LLM.md L0a. Cited from shaders/bank_gather.comp and bench/ops_bank.go.
     This is a vertical's stage finding, not a numbered IDEAS experiment, so
     it carries a name; it does however close most of IDEAS §5.2. -->

[← LLM.md](../LLM.md) · [research index](README.md) · [IDEAS §5.2](../IDEAS.md) · decode path

# L0a — the DRAM bus does not care how big the weight bank is

**Result: flat. 236.4 GB/s reading a 64 GiB bank at random, against 237.1
GB/s reading a 1 GiB one — a 0.3% difference, inside the run-to-run noise.**
There is no TLB cliff, no page-table cost and no penalty for scattering the
reads, so `LLM.md`'s whole decode argument — that tok/s is linear in
bits/weight and nothing else — holds at the size a 180 B-parameter model
actually needs.

## Why this had to be measured first

Every bandwidth number this repo quotes was taken over a small range.
`bandwidth` (copy.comp) tops out at a 512 MB footprint; `gemv_cold`'s largest
DRAM-resident case reads **64 MB** of 4-bit weights per dispatch; `moe`'s
expert banks reach 4 GB. qwen3.8-flash-next's resident bank is **64–82 GB**,
read as ~500 scattered expert slabs per token — 20x the largest range anything
here had probed, and §5.2 ("heap topology, carveout size and page size …
informs how to lay out a big model") had never been run.

That is not a pedantic gap. §1.1 and §1.7 are the foundation of the vertical:
the W4A8 GEMV reaches 99–103% of the bus, so decode is bandwidth-bound, so
bits/weight *is* tok/s. If a 64 GB working set had cost even 20%, the right
answer would have flipped from "use ~4.25 bits and keep the model resident" to
"shrink the bank", and it would have flipped **before** a single layer of the
model was written.

## The instrument

`shaders/bank_gather.comp` and the `bank` family (`bench/ops_bank.go`). It
deliberately does no arithmetic worth the name: it reads slabs of a bank with
the same 128-bit `buffer_load_b128` that gemv_w4a8.comp's inner loop issues
(§1.1 note 3), sums them so nothing is eliminated, and writes one dword per
workgroup. Standing a tuned GEMV in the way of a memory-system question would
only have added a variable — the kernel is already at the bus, and what was in
doubt was whether the bus would still be there.

Held fixed: **1024 MiB read per timed step**, the access shape (consecutive
lanes take consecutive 16-byte chunks, so one step covers 4 KB in one run and
§5.1b's coverage law is satisfied by construction), and the instruction count.
Varied: **the size of the range those bytes are drawn from**, 1 GiB to 64 GiB,
and the order they are drawn in.

Three orders, which separate two things a naive "read a big bank" test
conflates:

| order | selection | what it isolates |
|---|---|---|
| `seq` | the first K slabs of the prefix | the control — nominally large range, small working set |
| `spread` | K slabs evenly spaced across the prefix, ascending | real range, ordered |
| `rand` | K slabs drawn at random from the prefix | real range, unordered — what MoE decode does |

At the 1 GiB prefix all three select the same set, which is a free consistency
check: they agree to 0.1%.

Three slab sizes: the model's two real 4-bit expert slabs — `down` is
2560×640 nibbles (800 KB) and `gate_up` is 1280×2560 (1600 KB) — plus 4 MiB as
a control an order of magnitude up.

**The bank is 19 buffers**, because `maxMemoryAllocationSize` here is
0xfffffffc, 4 GiB minus 4; 64 GiB is 18 allocations of 3.5 GiB and a
remainder. Every page of it is written once before measuring: an allocation
whose pages have never been dirtied need not have distinct physical pages
behind it, and a bank that is really one zero page repeated would read at
cache speed and answer the wrong question.

## Finding 1: flat, at every size, in every order

GB/s, mean of two runs, `rand` (the MoE-decode pattern):

| bank | 1 GiB | 2 | 4 | 8 | 16 | 32 | 48 | 64 GiB |
|---|---|---|---|---|---|---|---|---|
| slab 800 KB | 237.1 | 237.2 | 236.7 | 236.5 | 236.1 | 236.9 | 236.4 | **236.4** |
| slab 1600 KB | 236.9 | 237.1 | 237.2 | 236.1 | 236.5 | 236.6 | 236.8 | **236.5** |

`rand/seq` is **1.00x in every one of the 48 cells** at the two real slab
sizes. Across all 144 cells of both runs the mean is 235.5 GB/s and the range
is 225.5–237.4.

**And at 80 GiB.** A later single run with `-bankgib 80` — 23 buffers, 85.9 GB,
every page written — reads **237 GB/s at every prefix from 1 to 80 GiB,
`rand/seq` 1.00x**. That is past UD-Q4_K_XL's 82.52 GB resident core, so the
largest bank this vertical would ever hold has now been measured live rather
than extrapolated. It is one run rather than two, but it agrees cell for cell
with the two-run 64 GiB sweep above over their common prefixes.

Two reference points make that number readable:

- §0.4's contiguous copy benchmark measures the DRAM ceiling at **236 GB/s**.
  This probe reads 236–237 at 64 GiB *at random*. It is not near the ceiling;
  it **is** the ceiling.
- §1.7's best W4A8 GEMV reads 242 GB/s. The 242 is above 236 because it counts
  weight bytes against a nominal peak — the two figures agree to their
  respective conventions, and neither leaves room for a big-bank penalty.

## Finding 2: the 4 MiB slab is 1.7% slower, and it is occupancy, not memory

The control slab lands at 232.9 GB/s mean against 236.7–236.8 for the two real
ones. At a fixed 1 GiB read, a 4 MiB slab means 256 slabs where a 1600 KB slab
means 640, and the grid is `slabs × parts`: 4096 workgroups of 256 KB each
against 10240 of 100 KB. Fewer, longer workgroups leave a larger tail when the
last wave of them drains. It is not a memory effect — it does not grow with
the bank, which is the whole point of the sweep — and at 1.7% it is not worth
chasing. It is recorded because the real slabs are *below* it and a reader
should know which side of the knee the model sits on.

## Finding 3: a bandwidth-bound read costs 85 W

The bandwidth-bound cells run at **2892 MHz and 85 W** (80–102 across the
sweep), against the 118–142 W the same machine draws doing ALU-heavy work
(§0.1 measured fp32 FMA *power-limited* at 138 W, and this session's own clock
warmup shows 127→81 W as it hands over to the probe). The clock never drops
below 2890 once warm.

So a decode step that is pure weight streaming leaves **~50 W of package
budget on the table**. That is not a result about bandwidth, but it is the
reason speculative decoding and batching — which convert idle bus time into
arithmetic — are unlikely to be throttled into giving back what they win.

## Reproducibility

Two runs of the family, 72 common rows: **median ratio 1.0002, p10 0.9970,
p90 1.0036**, worst single cell 0.9655 (a 4 MiB `spread` cell). Results in
[`results/bank.csv`](../results/bank.csv).

    go run ./cmd/bench -bankgib 64 -bankreadmib 1024 bank

Allocation and page-dirtying of 64 GiB dominates the wall clock; the 72 timed
cells are seconds.

## What this closes, and what it does not

**Closes**, for the purposes of the LLM vertical:

- The bus survives a 64 GiB working set. `LLM.md`'s decode arithmetic stands:
  UD-Q4_K_XL's 6.334 GB/token is a ~38 tok/s ceiling and a ~4.25-bit bank of
  our own is ~67, because bytes are the only currency.
- **§5.2's "informs how to lay out a big model" is answered in the negative** —
  there is nothing to lay out for. Page size, carveout and heap topology do not
  show up in achieved bandwidth at 64 GiB, so a weight bank can be allocated in
  whatever shape is convenient. The 4 GiB allocation cap forces ~19 buffers and
  that costs nothing; stage 4c already found the matching result on the DiT
  side (six banks bit-identical to one, one pipeline per bank and nothing per
  dispatch).

**Does not close:**

- ~~**Heap 0.**~~ Answered by **L0b**, [§5.1](5.1-memory-types.md): all eight
  buffer-compatible memory types read within 0.57% of each other, heap 0 and
  heap 1 alike. There is no faster heap and no slower one.
- ~~**The other 20 GB.**~~ Closed by the 80 GiB run above and by
  [§5.1](5.1-memory-types.md), which found the 83.79 GiB heap figure is not a
  limit at all (105.0 GiB reserved). What is genuinely unmeasured now is only
  the band between 80 GiB touched and 105 GiB reserved.
- **Concurrent pressure.** One process, nothing else on the GPU. A serving
  engine holding 64 GB while the host also holds a KV cache and a 28.8 GB
  mmap'd n-gram table is a different memory-pressure regime.
