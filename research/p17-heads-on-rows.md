# P17 — heads on the fragment's rows: 128k prefills past 1000 tok/s

**2026-09-22.** Three changes: P17-1, the kernel below; P17-2, a prefetch
worth another 3% of prefill at depth; and P17-3, the decode select, +0.7% of
decode at 128k. A 128 000-cell prefill at ubatch 2048 goes
**833.5 → 1011.4 tok/s (1.21x)**, 64 000 cells **896.4 → 1048.6 (1.17x)**, and
the prefill falloff from depth zero goes from **0.72x to 0.88x**. Decode at depth gains
**~2%** because the mask pass is gone (31.62 → 32.21 tok/s at 128k), and a
decode step's answer is **bit-identical**.

| depth | pp before | pp after | | tg before | tg after |
|---:|---:|---:|---:|---:|---:|
| 0 | 1156.6 | 1156.3 ¹ | 1.00x | 34.39 | 34.34 |
| 4 096 | 1015.4 | **1049.1** | 1.03x | 33.33 | **34.10** |
| 64 000 | 896.4 | **1048.6** | **1.17x** | 32.17 | **32.97** |
| 128 000 | 833.5 | **1011.4** | **1.21x** | 31.62 | **32.21** |

48 layers, shipped banks, `-pp 2048 -tg 16 -ctx 139000`, `wiki.test.raw`,
same binary, arms selected by `LLM_ATTN_GATHER_HROWS`. Two interleaved runs a
arm at 0/64k/128k (agreeing to 0.2%, the table is their mean:
`results/p17_ab48_h{0,1}_r{1,2}.csv`), one at 0/4096 with the default rule
against the control (`results/p17_rule_{def,0}.csv`).
¹ The default rule, which takes the token-row kernel at depth 0. Forcing the
heads build there (`LLM_ATTN_GATHER_HROWS=1`) is 1144.3, −1.1% — see below.

**At the served `-llm-batch` of 4096, both changes together take 128 000
cells from 899.7 to 1175.7 tok/s (1.31x)**, decode 31.34 → 32.18 (one run a
arm, `results/p17_ub4096_{0,1}.csv`; the control arm is
`LLM_ATTN_GATHER_HROWS=0 LLM_PLE_PREFETCH=0`).

## The idea

P14's gathered attention puts sixteen consecutive **tokens** of one query head
on a cooperative-matrix fragment's M axis, and reads the **union** of their
QSA selections. At 128 000 cells that is 7 049 cells a tile where one token
selects 2 051. P14 called that 3.4x "the shape of the hardware against the
shape of QSA", something nothing in this vertical closes.

But the selection is per **token**, not per head, and this model has 24 query
heads over 2 kv heads. The twelve heads of one token are twelve rows that attend
over **exactly the same cells** of **exactly the same key and value**. Put them
on M instead:

- **The list is one token's selection**, 2 051 cells, not a sixteen-token
  union. The gather's existing `BM=1` build writes it (pass 1 only).
- **There is no per-row mask.** Every row wants every gathered cell, so the
  `attn.gathmask` dispatch is gone. The kernel's only predicate is `col < cnt`
  on a short last chunk.
- **The kv plane is staged once for twelve heads**, not once a head.
  That is the sharing P14's `HPW` knob tried, at 12x instead of 2x.
- **The Q fragment needs no repack.** The plane is head-major, tiled 16 tokens
  by 16 dims, so row `r` of (token t, dim group kt) is sixteen contiguous halves
  one head's plane apart: a RowMajor `coopMatLoad` with a stride of one head.

Rows are heads `h0 .. h0+15` with `h0 = min(g*rep, heads-16)`. The four rows
belonging to the other kv head are computed against the wrong key and never
stored, which is cheaper than any test inside the fragment and keeps every Q
load inside the plane. 12 of 16 rows are useful. Against the 2 051/7 049 the
token rows got at 128k, that is the whole win. The grid is (token, kv head),
or (token·splits + slice, kv head) split.

`attn.attn` at 128k, 48 layers: **51.6 → 15.9 ms a pass, 3.25x** (0.302 →
0.093 ms a token). On the 4-layer probe, whose one attention layer is denser
than the mean, it is 77.2 → 16.8 ms, 4.6x.

## Where it loses, and the rule

From a cold cache the selection names every live cell, so a token tile's union
*is* one row's, and heads-on-rows only wastes a quarter of the fragment. On the
4-layer probe (tok/s, two runs each, `results/p17_cross_h{0,1}.csv`):

	past        0     2048    4096    8192   16384  128000
	tokens  11109   10406   10162    9818    9424    7817
	heads   11076   10504   10590   10529   10500   10271

At 48 layers and ctx 139 000 the depth-0 loss is 1.1% (5.40 → 6.58 ms of
`attn.attn` a pass). So prefill takes heads-on-rows only when
`past >= selWidth` (`LLM_ATTN_GATHER_HROWS_MIN` overrides). Decode always
takes it once it gathers, because there the two builds are the same bits.

## Exactness

**Decode is bit-identical**, and the gate asserts a zero. At one row, the
token fragment's union *is* the token's selection, so both kernels fold the
same cells in the same chunks in the same order. A matrix product's rows are
independent, so which fragment row a head lands in does not change its sums. What
decode gains is the deleted mask dispatch (6.6 µs a layer) and a gather that
no longer ORs sixteen rows.

**Prefill is a reassociation**, like P14-2: the same set, folded in different
chunks. 3.6e-05 relative rms from the token-row kernel against 1.2e-02 of
either from llama.cpp, and against the block kernel 4.31e-06 where the token-row
gather is 4.10e-06. It falls under `PinSchedule` with the gather it rides on,
so nothing new there.

Gates:

	go test ./llm/ -run TestAttnGPUHeadRowsIsTheGather   # prompt tolerance, decode exact, dispatch list
	go test ./llm/ -run 'TestAttnGPU|TestGraphIsAChunkSplit|TestPrerecordedDecodeCrossesTheGatherBound'

**The engagement check had to be the dispatch list and not the answer.** The
first draft asserted "decode differs from the token rows, else it never
engaged", and it failed on a zero. That zero was the arithmetic, not a bug. The
split's sibling test (`TestAttnGPUGatherSplitIsTheGather`) cannot see this arm
at all: its 1e-4 is the fp16 store of the unsplit output, identical in both
arms to six places. So the gate asserts the zero, then reads `graph(0)`'s
kinds for one gather, no mask and one split.

`TestAttnGPUGatherSelectsTheSameCells` reads the token-tile layout through
`Gathered`, and now pins `SetHeadRows(false)`: its second chunk starts past the
bound. The arena is sized for whichever layout is larger.

## Where 128k prefill stands

ms a prefill token at 128 000 cells, ubatch 2048, from the labels:

	                 P14/P16   P17
	floor              0.818   0.818   everything flat in depth
	attn.attn          0.302   0.093
	attn.score         0.027   0.027
	attn.select        0.017   0.017
	gather + mask      0.010   0.002
	host gather        0.036   0.012   P17-2

The attention's depth terms are now **0.14 ms of a 0.99 ms token**. What is
left of the falloff is mostly the floor: `moe.up` alone is a quarter of the
token.

## P17-2: the next chunk's n-gram pages, faulted in while this one runs

The host `gather` line of a prefill pass **grows with depth, and that is not
depth**: 16-18 ms a 2048-token pass at depth 0 and **85 ms at 64 000 cells**.
It is the PLE n-gram gather (`Model.PLEGather`): 16 rows a token at random
offsets into the 28.80 GB mapping that D2 keeps on the host, ~32 000 major
faults a fresh chunk. At depth 0 the bench's warm-up pass has already touched
the same text; at depth it has not, and neither has a server on a new prompt.

A prompt's tokens are all known before its first chunk runs. So
`Graph.PrefetchPLE(prev, next)` runs the *next* chunk's gather on a goroutine
while this chunk is on the device, when the host is otherwise waiting on a
fence, and discards the result. What it leaves behind is warm pages. The real
gather waits for it and then reads warm. It is called by
`backend.LLM.chunks` (the server's chunk loop) and by the depth bench's fill,
whose last batch prefetches the timed one.

48 layers, 64 000 cells, two interleaved runs each (`results/p17_pf{0,1}_r*.csv`):

	            pp tok/s          host gather a pass
	off      1049.5  1048.4        85.4  85.6 ms
	on       1081.2  1081.2        26.2  24.3 ms       1.031x

Depth 0 and decode are unmoved, as the control. It changes no value: the rows
are the table's whoever faulted them. The one thing it had to get right is
concurrency with the mapping, so the first-gather `madvise` is a `sync.Once` (it was a
plain flag), and `Graph.Destroy` waits for an in-flight prefetch.
`LLM_PLE_PREFETCH=0` is the control arm.

**Refused: `MADV_WILLNEED` over the table.** 1.00x: the page cache grew 4 GB
of the 28.8, and the gather line did not move. **And a fully pre-read table
does not survive staging**: reading all 28.8 GB into the cache (6.1 s) and
then staging the model left 43 GB of cache where there had been 68. Staging
streams ~71 GB of weights through it.

**Open: decode's host gather, and it is a tail.** Traced per step at 64 000
cells (an env-gated trace in a scratch build, not committed): the median step
is **236 µs** with 7-16 major faults, but **two of 32 steps take ~8.7 ms**, and
those two are **75%** of the line. They fall at the *same* token positions in two
separate runs (the 10th and 27th of the timed decode). Their fault counts
(14, 16) are the same as the fast steps', and no GC cycle falls anywhere in the
decode run (`GODEBUG=gctrace=1`). So it is one fault taking milliseconds, not
more faults. An isolated replay of random tokens has no tail (0 of 200 steps
over 2 ms). Replaying the bench's exact tokens reads warm, because the bench
has just faulted them, so it cannot reproduce the cold case. Pinning it down
needs a trace inside the kernel's fault path, which this account cannot take.
It is worth ~0.5 ms a token (~1.8% of decode). Residency of the 29 GB table would
remove it along with the median, and that is a deployment decision.

## P17-3: the decode select, keys in registers and a 10-bit digit

`attn.select` at decode is one 1024-lane workgroup a token. At 128 000 cells
it was **54.0 µs a layer**, priced by compiling scratch copies with the
pass loop cut (a timing, not an answer): ~13 µs of emit and setup, **~17 µs
for the first radix pass and ~8 µs for each of the other three**. Two causes,
two changes, and both leave the bitmask exact:

- **Past `SEL_LDS` (4096 blocks) every pass re-read every score from global
  memory.** Now each lane holds its blocks' keys in registers, `KREG = 40` a
  lane, 40 960 blocks, past the cache's buffer-range cap. The row is read once.
- **The first digit was the exponent.** An `f2ui` key's top 8 bits are sign
  plus seven exponent bits, and a row of block scores shares one or two of
  them, so pass 1's atomics collided. The 1024-lane build now takes a 10-bit
  digit (10+10+10+2), which adds three mantissa bits to the first. The P10
  bucket search is one lane a bucket, and still fits.

4-layer probe, 128 000 cells, one layer, µs, two runs each:

	            decode        prefill (2048 rows)
	8-bit      53.9  54.0     2935  2944
	10-bit     48.7  47.8     2923  2921
	regs       54.7  49.4     3212  3077
	regs+10    43.6  43.9     3701  2774

Neither half pays alone and together they are 1.23x. Prefill is noisy with
the registers and does not need them (2048 workgroups have occupancy to hide
the loads), so **decode takes regs+10 and prefill takes 10-bit**
(`AttnGPU.selectPipe`).

48 layers, shipped banks, two interleaved runs each
(`results/p17_select_{base,def}_r{1,2}.csv`):

	            128k select (ms a token)   128k decode tok/s
	base         0.687  0.687               32.90  32.96
	P17-3        0.544  0.544               33.19  33.12      1.007x

Depth 0 is +0.004 ms a token of select (the 10-bit digit where the LDS
cache already holds the row), 0.01%, and prefill is flat.

**Exactness** is the existing gate, `TestAttnGPUSelectBlocksIsTheCellSelect`,
the block select's bitmask against the cell select's, bit for bit. Its 4k
fixture is 1024 blocks and would never leave the LDS cache, so a test-only
build with `SEL_LDS=64` (`LLM_ATTN_SELECT_VARIANT=kreg_r10_lds64`) sends it down
the register path. A copy of that build with every register key zeroed
**fails** the gate, so the gate sees the path.

## Control arms

	LLM_ATTN_GATHER_HROWS=0        # P14/P15's token-row gathered kernel
	LLM_ATTN_GATHER_HROWS=1        # forced, at every depth (bypasses the rule)
	LLM_ATTN_GATHER_HROWS_MIN=<n>  # the prefill past bound, default selWidth
	LLM_PLE_PREFETCH=0             # P17-2's control
	LLM_ATTN_SELECT_VARIANT=base   # P17-3's control (P14-1's 8-bit select)
	LLM_ATTN_SELECT_VARIANT=r10|kreg|kreg_r10|kreg_r10_lds64
