# P16 — the wide batch never cost decode anything: two moves padded to the arena

**2026-09-22.** Three changes, all exact, and two measured refusals. Decode
goes **1.16x at every depth** at ubatch 2048 — **29.73 → 34.4 tok/s at depth
zero and 27.20 → 31.5 at 128 000 cells** — and the cost of a wide prefill
batch on decode, which P12-7 priced at 8% and P15 at 11-22% and nobody could
explain, **was a bug and is gone**: at an 8192-row arena decode goes **23.55 →
33.87 tok/s (1.44x)**. Prefill is flat as the control.

| depth | P15 pp | P16 pp | P15 tg | P16 tg | |
|------:|-------:|-------:|-------:|-------:|---:|
| 0 | 1150.7 | 1155.8 | 29.73 | **34.39** | 1.16x |
| 8 000 | 969.6 | 973.9 | 28.78 | **33.54** | 1.17x |
| 32 000 | 915.3 | 919.7 | 28.72 | **33.20** | 1.16x |
| 64 000 | 892.8 | 896.3 | 28.14 | **32.41** | 1.15x |
| 128 000 | 829.4 | 834.4 | 27.20 | **31.53** | 1.16x |
| falloff | 0.72x | 0.72x | 0.91x | 0.92x | |

48 layers, shipped banks, `-pp 2048 -tg 16 -ctx 139000`, `wiki.test.raw`.
Two passes of the P16 arm, agreeing to 0.2% on tg and 0.4% on pp; the table
is their mean. The P15 columns are P15's own table, not a same-hour control —
the same-hour controls are the arm tables below, which are.

## P16-1: the arena width, and the probe that settled it

P15 left the question narrowed to one probe: *stage the wide arenas and
prefill in narrow chunks anyway, and see whether the cost follows the arena's
width or the prefill's footprint.* `cmd/llm -depth` now takes `-ubatch`, the
arena rows, separately from `-pp`, so that is one flag. Depth zero, `-ctx
16384`, `-tg 64`, two interleaved passes:

| arenas | prefill batch | tg tok/s |
|---:|---:|---:|
| 2048 | 2048 | 30.17 / 30.59 |
| 8192 | **2048** | 23.97 / 24.16 |
| 8192 | 8192 | 23.52 / 23.57 |

**It follows the arena.** A wide arena that a narrow prefill never filled
costs decode the same 21% as one a wide prefill did. And P15 had already shown
it is not *volume* — 3.85 GB of extra `-ctx` costs 0.2% — so it had to be
something a decode step *does* in proportion to the arena's rows. The per-label
diff finds it in one line: everything is flat except three labels.

| label | 2048 arena | 8192 arena | Δ ms a token |
|---|---:|---:|---:|
| `dn.qkv` | 120.3 µs | 233.1 µs | +4.06 |
| `move` | 5.5 µs | 19.7 µs | +2.78 |
| `attn.qkv` | 128.0 µs | 247.1 µs | +1.43 |
| everything else | | | +0.3 |

`DeltaNetGPU.InPort` and `AttnGPU.InPort` advertised `Rows: g.arenaRows`, and
`mover.Move` writes zeros from the run's rows up to the destination's. So
**every decode step zero-filled the whole prefill arena of both blocks' A
operand, once a layer** — at 8192 rows, 8192 × 2560 halves = 42 MB a move, 48
moves a token. The arithmetic closes: the 48 padded moves are ~63 µs each
(the other 148 moves stay at 5.5), which is too fast to have written 42 MB, and
the qkv projection that runs next is ~114 µs slower — together ~177 µs a layer,
**237 GB/s for 42 MB, which is this device's copy rate**. The move's writes are
posted, and the next kernel pays for them draining. The
comment on the port said why it was the arena: *the GEMM rungs have no bounds
check*. That is true and it asks for `roundUp(rows, bm)`, not the arena — the
widest row block any rung reads is what the arena is itself rounded to, so the
port now pads to `roundUpInt(g.rows, g.rowAlign)`, which is what the MoE's port
(`roundUpInt(g.rows, moeBMMax)`) had done all along.

It is not only the wide arenas' bug. At 2048 rows it was 4 MB a move and
**9% of decode** (30.38 → 33.13 tok/s with only this change, the `hc.cn`
control arm below), and the P12-7 server measurement that made 4096 a trade
was this line and nothing else.

The mechanism is asserted by `TestInPortPadsTheRunNotTheArena` — one row pads
to one row block, a full batch to the arena — because no tolerance can see
it: the old padding is *correct*. `TestGraphIsAChunkSplit`,
`TestPrerecordedDecodeIsTheRecordedDecode` and 96 greedy tokens on a
597-token prompt, identical across the two binaries, hold the answer. And all three
changes together are **bit-exact through the whole model**: `TestGraphLogits`
at 48 layers prints the same per-depth rms, `result_norm`, logits and top ten
to the last digit as the HEAD binary run beside it.

## P16-2: `hc.cn` goes back to a stream a workgroup at decode

P11 made `hc.cn` one workgroup a **token** instead of one a (token, stream),
because the four streams read the same block output row and at prefill the
kernel is short of bandwidth: 1.37-1.44x there. At decode it is the P8 rule
the other way round: one token is **one workgroup on a forty-CU device**,
walking four streams, each through the 256-way tree, in series — **25.0 µs a
dispatch, 94 a token, 2.35 ms of a 33.1 ms step**. P1a had priced the fused
kernel at ~6-8 µs at one token, and that was before P11.

So the grid is chosen by the width: at `rows <= 64` (`hcCNSplitRows`,
`LLM_HC_CN_SPLIT_ROWS`, 0 is P11's grid everywhere) it is `(rows, hc)` and a
workgroup walks only its own stream. Every stream still reduces the same
per-thread partials through the same tree, so the two grids are the same bits;
`TestHCFusionIsTheCombineThenTheNorm` now asserts the pair against both.

**25.0 → 7.0 µs a dispatch, 33.13 → 35.14 tok/s (1.06x)** at depth zero, two
passes each (33.19/33.07 against 35.16/35.12). Prefill is P11's grid and flat.

## P16-3: `attn.select`'s emit walks blocks, not cells

TODO named `attn.select` the last unstriped decode kernel. Two guesses about
what bounds it were measured and are wrong (below); a probe then said where
its 68 µs at 128 000 cells go: dropping the emit leaves **43**, and dropping
three of the four radix passes also leaves **43** — so ~8 µs a pass, **~25 µs
of emit**, and ~10 fixed.

The emit walked 32 cells a lane, dividing by the runtime `ratio` a cell and
fetching each block's key behind a branch on the last. A block's `ratio` cells
all carry its key, so the walk is now a block at a time — one compare and a
run of bits — with `EMIT_BATCH` keys fetched before any is compared, and the
incomplete tail block handled as its own span. Same bits by construction;
`TestAttnGPUSelectBlocksIsTheCellSelect`, `TestAttnGPUSelectionIsTheCPUs`,
`TestAttnGPUGatherSelectsTheSameCells` and
`TestAttnGPUCacheSizeDoesNotChangeTheAnswer` all pass.

**68.1 → 54.9 µs at decode (1.24x) and 3459 → 2876 µs a prefill dispatch
(1.20x)** at 128 000 cells, two passes each on the 4-layer probe. It is ~0.4%
of a deep token either way — worth having because it is free, and not worth
more work: what is left is four ~8 µs passes that are a reduction.

## The refusals

- **Batching the histogram's loads is 0.96x** (67.0 → 69.6 µs). Four keys in
  flight a lane before the first atomic: the pass is not a chain of load
  latencies.
- **Folding same-bucket lanes a wave before the atomic is 0.69x** (68 →
  98 µs). The first pass buckets on sign and exponent, so it looked like ~1024
  lanes queuing on two or three LDS words; if they do, it costs less than a
  `subgroupMin` + `subgroupAdd` round a trip does.

## What changes for the server: `-llm-batch` is no longer a trade

P12-7's method repeated: `cmd/serve -llm -llm-ctx 8192`, four disjoint
wikitext prompts of 3918-4346 tokens, 256 generated tokens each, greedy, a
fresh server per arm, two interleaved passes. The passes agree to 0.1%.

| `-llm-batch` | prefill | P12-7 | decode | P12-7 |
|---:|---:|---:|---:|---:|
| 2048 | 1052.2 | 1020.9 | **34.19** | 28.04 |
| 4096 | 1198.5 | 1146.3 | **34.20** | 25.81 |
| 8192 | 1232.5 | — | **34.16** | — |

**Decode is flat across the batch** (34.02-34.44 in every arm), so the
break-even P12-7 drew at ~150 generated tokens is gone and 4096 is the faster
answer at every length. Against P12-7's shipped 4096, decode through the
server is **1.33x**. 8192 only differs on the one prompt longer than 4096
tokens (1111 → 1251 tok/s), which is the depth sweep's story: at 128 000
cells P14 measured 886 at 4096 and 946 at 8192. It costs ~3 GB of arenas
more than 4096 and lowers the `maxStorageBufferRange` cap on `-llm-ctx`, so
**4096 stays the default** and 8192 is the setting for a deployment whose
prompts are long; that is a memory question now, not a latency one.

The 4096 depth sweep at 48 layers, `-ctx 141000` (three depths — five at 4096
rows push the cache past the ~146.7k cells the descriptor reaches, and
`checkBufferRange` refuses it, as it should):

| depth | pp tok/s | tg tok/s |
|------:|---------:|---------:|
| 0 | 1292.5 | 33.01 |
| 64 000 | 979.4 | 33.00 |
| 128 000 | **904.6** | **31.63** |

## Where a decode step goes now

Depth zero, 2048 arena, 28.5 ms a token on the device, 1431 dispatches: the
four weight-streaming kernels (`moe.up` 5.25 ms, `dn.qkv` 4.07, `moe.down`
2.51, the head 2.38) run at roughly 170-200 GB/s, near this device's copy
rate. 898 of the 1431 dispatches are under 12 µs and add up to 3.3 ms; the
196 moves are 0.54 ms of it. What is left of decode is fusion at a percent or
two a step, and L6c's single arena (which would delete the moves) is the
largest of those.
