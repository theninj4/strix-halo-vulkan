<!-- LLM.md L0c. Closes the first of IDEAS §2.2's three open items. Cited from
     shaders/gemm_wmma_q4.comp and bench/ops_moe_q4.go. -->

[← LLM.md](../LLM.md) · [research index](README.md) · [IDEAS §2.2](2.2-q4-coopmat.md) · prefill path

# L0c — the scale plane's layout, and §2.2's unexplained 1.24x

**Result: it was locality, and transposing the scale plane removes it.**
On `gate_up`, k-major at QBLOCK=32 is **3.72 ms against row-major's 4.71 —
1.27x** — and is *faster* than row-major at QBLOCK=128 (3.79). The fine scale
block, which accuracy wants, now costs **1.0%** of the two matmuls instead of
14.2%.

## The question

§2.2 finding 4 is the largest number in the Q4 path with no mechanism attached.
QBLOCK=128 beats QBLOCK=32 by 1.24x between two binaries `RADV_DEBUG=shaderstats`
prices identically — same 252 VGPRs, same 5064 bytes of code, same 802
instructions, same 513 VALU, same 100 VMEM — with one shift in the scale index
as the only difference in the source, standing behind a 4.4-point difference in
nominal traffic. §1.8 then narrowed it: the same axis on the grouped *GEMV*,
which reads its scales exactly the way it reads its weights, costs only
1.045-1.116x, so whatever it is belongs to how the **GEMM stages** them.

The candidate §2.2 named was where the loads land. One staging step reads BN
rows of the scale plane at one k-block; at layout `[rows][ldb/QBLOCK]` those BN
scales are a row stride apart, so each 2-byte scale arrives in its own cache
line and a step is a 64-address gather — §5.1b's worst request shape. The test
it asked for was "one slab's 64 scales contiguous instead of 160 B apart".

## What was built

`gemm_wmma_q4.comp` gains `SCALE_LAYOUT`, and `bench/ops_moe_q4.go` a matching
host packer (`packScalePlane`), so the two sides can only agree by construction
— a layout they disagree about produces *wrong* numbers, not slow ones, and the
existing Q4 correctness check catches it. All four new builds pass it.

| | layout | one staging step's 64 scales |
|---|---|---|
| 0 | `[rows][ldb/QBLOCK]` — what §2.2 measured | a row stride apart |
| 1 | `[ldb/QBLOCK][snrows]` — a transpose | 64 contiguous halves = 128 B |
| 2 | `[rows/BN][ldb/QBLOCK][BN]` — tile-blocked | the same, and the tile's whole K sweep contiguous too |

The dead `block` push-constant slot carries the plane's leading dimension for
layout 1 rather than growing the struct — `TODO.md` records a bug where adding
a push constant to a shared shader silently zeroed another caller.

## Finding 1: 1.27x, and the fine scale block becomes free

2048 tokens, 64x64x64 tile, `LDS_PAD=8`, ms per dispatch:

| scale layout | gate_up qb32 | gate_up qb128 | down qb32 | down qb128 | block (qb32) |
|---|---:|---:|---:|---:|---:|
| row-major | 4.71 | 3.79 | 4.54 | 4.30 | **9.25** |
| k-major | **3.72** | 3.63 | 4.44 | 4.29 | **8.15** |
| tile-blocked | 3.72 | 3.62 | 4.43 | 4.29 | **8.15** |

The comparison that settles it is **k-major qb32 (3.72) against row-major qb128
(3.79)**: the fine scale block, laid out for the staging step, is *better* than
the coarse one laid out badly. So §2.2's 1.24x — reproduced here as 1.243x on
`gate_up`, 4.71 against 3.79 — was never about the scale block size.

For phase 2 that is the whole point of the item: **block-32 granularity now
costs 8.15 ms against 8.09, 1.0%**, where before it cost 14.2%.

## Finding 2: the predictor is the plane's row stride, and the two layers prove it

The two layers carry **identical nominal scale overhead** — 11.1% of the bank
at qb32, 3.0% at qb128, because both are 2 bytes per 32 nibbles. Bytes
therefore cannot explain why one layer recovers 1.27x and the other 1.02x. The
stride can:

| layer | K | scales/row at qb32 | plane row stride | rows per 64 B line | recovered by transposing |
|---|---:|---:|---:|---:|---:|
| gate_up | 2560 | 80 | **160 B** | 0.4 — every scale alone in its line | **1.266x** |
| down | 640 | 20 | **40 B** | 1.6 — already sharing lines | 1.023x |

`down`'s plane was never pathological, so there was never much to win there,
and what remains of its qb32/qb128 gap (4.44 against 4.29, 1.035x) is the
scale bytes themselves. This is a within-experiment control rather than an
argument: same kernel, same tile, same nominal traffic, opposite strides,
opposite outcomes.

The achieved-bandwidth column says the same thing more directly. On `gate_up`
at qb32 the kernel reads **100 GB/s of bank row-major and 127 k-major** — the
same bytes, 27% more of them per second.

## Finding 3: tile-blocking buys 0.1%, so do not couple the layout to the tile

Layout 2 goes further than layout 1: it makes a whole N-tile's K sweep
contiguous, not just one staging step's 64 scales. It is worth **8.15 against
8.15 ms at qb32 and 7.91 against 7.92 at qb128** — nothing.

That is itself evidence for the cache-line reading: once the 64 scales a step
needs are in two lines, further contiguity has nothing left to remove. And it
settles the design question, because the two layouts are not equally cheap to
adopt. Layout 2's packing depends on `BN`, so changing the tile geometry would
mean repacking the whole bank — 60 GB for a real model. Layout 1 is a property
of the weights alone. **Take layout 1.**

It also disposes of an arithmetic explanation: layout 2 does two runtime
multiplies in the index where layout 1 does one, and they measure the same, so
the index arithmetic is not on the critical path in either.

## Finding 4: the layout helps the coarse block too

Row-major qb128 is 8.09 ms; tile-blocked qb128 is 7.91. So even at QBLOCK=128,
where a row is only 40 B, transposing is worth **1.02x**, and the best cell in
the table is **1.169x over the configuration §2.2 shipped**.

## What this changes

- **§2.2's first open item is closed.** "A 1.24x with no mechanism attached,
  from two identical binaries" now has one, and the fix is a packing change
  rather than a kernel change.
- **LLM.md's phase-2 format simplifies.** The two-level scale (int4 values,
  int8 per-32 scale, fp16 per-256 super-scale) was motivated half by accuracy
  and half by dodging this tax. The tax is gone, so plain **fp16 scales per 32
  in a k-major plane** is the design, and the only remaining argument for int8
  sub-scales is the 8.1 points of bank bytes they would save — worth having at
  decode, where bytes *are* the clock, and worth nothing at prefill.
- **A decode question opens.** All of this is the GEMM. §1.8 measured the
  grouped GEMV paying 1.045-1.116x for the same axis and attributed it to bytes
  alone; that kernel reads its scales the way it reads its weights, so it
  should not care about the plane's layout — but it has not been tried, and
  decode is the phase where scale bytes are charged against the bus.

## Reproducibility

Two full `moe` runs, 1024 common rows: **median ratio 0.9998, p10 0.9923, p90
1.0039**. The twelve L0c cells: **median 1.0001, range 0.9885-1.0035**. Results
in [`results/moe.csv`](../results/moe.csv); the table above is run 2.

    go run ./cmd/bench -blocks 32,64,128,256,512,1024 moe     # ~13 min

**Not done:** the ISA was not re-dumped. §2.2's "instruction-for-instruction
identical" claim was checked for the QBLOCK axis, not for this one; the
attribution here rests on the two-layer control in finding 2 and on finding 3's
arithmetic argument instead.
