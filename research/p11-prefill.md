<!-- P11. Prompt processing, short and long: where a prefill token goes after
     P7-P10, the two changes that moved it, and the three that did not.
     Cited from TODO.md, backend/llm.go, cmd/serve/main.go,
     shaders/llm_moe_gemm.comp and llm/gpu_moe.go. -->

[← TODO.md](../TODO.md) · [research index](README.md) · [L2a](l2a-prefill-attribution.md) · [L5b](l5b-moe-gpu.md) · [P0](p0-ring-watchdog.md) · [P7](p7-context-depth.md)

# P11 — prefill: the batch was the biggest number in the file, and the padding was not

> **Second round below** (P11-6, P11-7): `hc.cn` one workgroup a token is
> another **1.37-1.44x on the second largest kernel in a prefill**, and the
> QSA selection's skip rate at depth is measured for the first time — which
> priced, and then refused, a finer skip inside the attention kernel.
> The ladder after both rounds: **681.7 / 925.1 / 1191.8 / 1340.7 / 1372.2
> tok/s** at 512 / 1024 / 2048 / 4096 / 8192 rows, **3.51x llama.cpp**.

**Result. Served prompt processing is 1.64x for a gigabyte of arenas, and the
graph itself is 1.06-1.08x on one kernel's load width.** The two are
independent and they compose: `cmd/serve` prefilled in 512-token chunks
because that is llama.cpp's best ubatch and what every number in this vertical
is quoted against — and **512 is the worst rung this graph has**. The default
is now 2048. Separately, the grouped MoE GEMM's gathered A operand was being
staged one half at a time; eight at a time is **1.25x on `moe.up`**, the
largest kernel in a prefill. Together, at 48 layers and the shipped banks:

| ubatch | before | after | | against llama.cpp |
|---:|---:|---:|---:|---:|
| 512 | 660.8 | **680.7** | 1.030x | 1.74x |
| 1024 | 886.5 | **923.1** | 1.041x | 2.36x |
| 2048 | 1079.6 | **1165.1** | 1.079x | 2.98x |
| 4096 | 1214.5 | **1301.2** | 1.071x | 3.32x |
| 8192 | 1250.1 | **1328.5** | 1.063x | **3.40x** |

Two runs each, agreeing to 0.02-0.2%. `results/p11_pp_{base,gather}[_run2].csv`
and the per-label tables beside them. **A request through the server was 667
tok/s and is now 1093** — the same graph, three-quarters of it the batch.

**And three changes that looked right and are not**, all measured on the same
instrument, all reverted: a per-fragment row bound, a three-rung schedule, and
a 128-row block. They share one answer, which is the finding worth keeping:
**at prefill the MoE GEMM is bound by its slab unpack, which is per *tile*,
and the row padding the schedule executes is very nearly free.** L5b built the
padding as a cost to be justified; it is not a cost.

## P11-1: the ubatch, which was the largest number and the cheapest

`backend.LLMOptions.Batch` sized the prefill arenas and so the chunk a prompt
is split into, and it was 512 with the comment "llama.cpp's own best ubatch on
this part and what every measurement in LLM.md is against". Both halves are
true and the conclusion does not follow. llama.cpp **plateaus** at its ubatch
(391.42 ± 1.13 at `-ub 512`, L2a-4) and this graph does not — P0 already
measured it still climbing at 8192 — because the MoE's arithmetic intensity is
the **routing's** and not the tile's: at 512 tokens 274 of 512 experts are
touched to serve 5120 rows, every one of them unpacked whole, and the schedule
pads those 5120 rows out to 11 904. Widening the chunk spreads the same unpack
over more rows and costs only arenas.

At 48 layers, `-ctx 4096`, shipped banks:

| `-llm-batch` | arenas | prefill | |
|---:|---:|---:|---:|
| 512 | 1.16 GB | 667.4 tok/s | |
| **2048** | **2.28 GB** | **1093.3** | **1.64x** |
| 4096 | 3.78 GB | 1237.0 | 1.85x |

**2048 is the default** because 1.12 GB buys 64% and the 1.49 GB after it buys
13% more; a deployment with memory to spare should say 4096. Nothing else
changes: a prompt shorter than the batch is still one chunk, so first-token
latency for short prompts is untouched, and `chunks` is the same code path
with fewer trips through it.

> This is the second time in this vertical that a constant inherited from the
> reference has been the largest single number on the board, and both times it
> was inherited *with* a measurement that justified it — for the reference.
> The rule it suggests: a parameter copied from llama.cpp is a hypothesis
> about llama.cpp, and this model's curve has to be measured before it is one
> about this model.

## P11-2: where a prefill token actually goes

`cmd/llm -graph` now writes a per-label table beside its block one
(`graphLabelRows`), which is the resolution `-depth` already gave a decode
step. 48 layers, shipped banks, after the change:

| | 512 | 2048 | 8192 |
|---|---:|---:|---:|
| `moe.up` | **49.0%** | **33.0%** | **22.3%** |
| `moe.down` | 19.4 | 15.3 | 10.0 |
| `hc.cn` | 2.9 | 9.3 | **10.7** |
| `dn.qkv` | 6.3 | 9.0 | 10.2 |
| `attn.attn` | 0.4 | 2.2 | 9.6 |
| `moe.combine` | 2.0 | 3.3 | 3.6 |
| `move` | 1.2 | 1.9 | 2.3 |

Three things this says. **`moe.up` is the kernel** at every length and it is
half the graph at the ubatch the server used to run. **`hc.cn` is next**, and
it is elementwise: the combine and the next mixer's norm over a four-stream
f32 residual, 901 MB a dispatch at 8192 rows in 7.0 ms — **128 GB/s against
the 236 a copy gets** (`results/bandwidth.csv`), so there is about 1.8x lying
in a kernel that does no arithmetic. **And attention is the whole of the depth
term** — see P11-5.

## P11-3: the gathered operand, eight halves a load

`llm_moe_gemm.comp`'s MODE 0 stages BM scattered rows of the block input into
LDS per K-step — the A operand is a *gather*, because a routed row is some
token's row of the input named by the permutation, where MODE 1's rows are
permutation positions and so contiguous. It was doing it one half at a time:
`BM * BK / LANES` iterations, **32 loads a lane a K-step at BM 64, each
fetching two bytes**, against the eight the B unpack issues for four times the
data.

This is L5b-7's finding on the operand instead of the weight — *the kernel is
bound by how many load instructions its unpack issues, not by the bytes they
fetch* — and the fix is the same one: a second view of the same buffer as
sixteen-byte words (binding 7, `hact4`), four `uvec4` loads a row a K-step
instead of 32 scalar ones, and the permutation entry behind them read once per
load rather than once per half. `unpackFloat2x16` is a reinterpretation, the
LDS layout does not move and the loop covers the same (r, c) in the same
order, so the slab is **bit for bit the one it was**.

One layer, real routing (`cmd/llm -moe`), µs a layer:

| tokens | `moe.up` before | after | | `moe.down` |
|---:|---:|---:|---:|---:|
| 512 | 7375.6 | **6781.0** | 1.09x | unchanged |
| 2048 | 14918.6 | **11935.8** | **1.25x** | unchanged |
| 4096 | 22656.9 | **17929.4** | **1.26x** | unchanged |

MODE 1 is untouched and is the control: it has no gather. The host's half of
the contract is that every offset the shader divides by eight is a whole load
— `lda` is `NEmbd + gemmPad` and every `halloc` rounds to 64 elements, so they
are, and `MoEGPU.build` says so rather than trusting it.

**The ladder boundary was re-screened after the change and does not move**:
at 2048 tokens `m4` is still the winner (12.1 ms against `m2`'s 17.5 and
`m1`'s 25.0) and the row padding is the same at every rung, so `MoEPlanFor`
stands.

## P11-4: three ways to spend the padding, and none of them pays

The schedule rounds every expert's row range up to the widest row block any
dispatch runs, so that a tile's rows are always that expert's and a
cooperative-matrix store — which covers sixteen rows and cannot be masked —
goes straight to global. At a 2048-token ubatch that executes **38 912 rows
for 20 480 real ones, 1.90x**; at 512 it is 2.33x. It reads like the biggest
waste in the block.

The constraint is the **fragment**, sixteen rows, not the row block, so all
three attempts pad to sixteen and differ only in how the GEMM is told:

| | tiles | rows | `moe.up` at 2048 |
|---|---:|---:|---:|
| as shipped, pad 64 | 608 | 38 912 | **11 936 µs** |
| a row count per record, bound in the unrolled loops | 608 | 24 304 | 13 662 |
| the same, with the gather's trip count left compile-time | 608 | 24 304 | 12 520 |
| three sub-lists, one dispatch a rung, no bounds at all | 608 | 25 184 | 12 532 |
| BM 128 (`w2m4`/`w4m2`), pad 128 | 477 | 61 056 | 13 509 |

**35% fewer rows is worth nothing, and paying anything for it loses.**

- **A bound inside the loops costs more than the rows.** At BM 64 the
  per-fragment branch sits inside the doubly-unrolled accumulator loops and
  the kernel goes to 256 VGPRs; making the gather's trip count dynamic on top
  of that is another 1.1 ms, because four `uvec4` loads stop being one clause.
- **Three dispatches cost what the rows saved.** One sub-list per rung — 64,
  32 and 16 — needs no guard anywhere and every kernel stays exactly the code
  it was. It came out 5% *slower* than one dispatch of 608 full tiles: the
  small lists do not fill the machine, and three grids have three ramps.
  (It also found a real bug on the way in, which is the one part worth
  keeping: the two modes **share one permutation** and may be cut to different
  row blocks, so a row range that depends on the pass's `bm` puts the up
  mode's expert 7 where the down mode does not look. `m1/m4` is a rung of the
  ladder and `TestMoEGPUDownNarrowed` caught it.)
- **A wider block trades the wrong way.** BM 128 cuts the tile count 608 → 477
  but takes the padding to 2.98x, and loses by 1.13x. The tile count does not
  fall as far as it looks like it should, because at 2048 tokens 399 experts
  are touched and **the floor is one tile an expert** however wide the block.

Fitting a per-tile and a per-row cost to the rung ladder at fixed rows (`m1`
24 961 µs at 2432 tiles, `m2` 17 457 at 1216, `m4` 11 936 at 608) puts ~9 µs
on a tile and ~0.17 µs on a row — so at BM 64 the **tiles are 45% of the
dispatch and the rows 55%** on paper, and in practice the rows behave as if
they were free. The reading that survives all four measurements: a tile's cost
is dominated by unpacking a `BN x BK` slab of a 4.5-bit bank per K-step, that
work is the same whether the tile holds 16 rows or 64, and the matrix cores
absorb the rows it is multiplied into.

**So the way to make this kernel faster is fewer tiles at the same block
width** — which means fewer *experts touched*, which means a bigger ubatch.
Which is P11-1, and is why P11-1 is the whole of the result.

## P11-5: long context, and it is all one kernel

`cmd/llm -depth -pp 2048`, 48 layers, shipped banks, `wiki.test.raw`, one
sequence walked forward (`results/p11_depth.csv`):

| depth | pp tok/s | vs 0 | ms a token | hc | dn | moe | **attn** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **1146.6** | 1.00x | 0.872 | 0.121 | 0.156 | 0.478 | 0.083 |
| 8 000 | 948.6 | 0.83x | 1.054 | 0.121 | 0.156 | 0.486 | 0.217 |
| 16 000 | 883.1 | 0.77x | 1.132 | 0.121 | 0.157 | 0.462 | 0.319 |
| 32 000 | 790.4 | 0.69x | 1.265 | 0.121 | 0.157 | 0.455 | 0.462 |
| 64 000 | **654.0** | 0.57x | 1.529 | 0.121 | 0.157 | 0.459 | **0.726** |

**Every block is flat in depth except attention, and attention is 97% of the
falloff** — 0.644 ms a token of the 0.663 a token a prefill gains between
depth zero and 64k. At 64 000 cells it is 47% of a prefill token. Inside it,
by label (ms a token, depth 0 → 64 000):

	attn.attn     0.0189 -> 0.5122   27x
	attn.select   0.0025 -> 0.1152   46x
	attn.score    0.0080 -> 0.0391    5x
	attn.expand   0.0180 -> 0.0256

`attn.attn` and `attn.select` are 0.627 of the 0.644. This is the quadratic
term any dense attention has and llama.cpp has it too; what is *ours* is how
far the QSA selection gets to bite here, which P11-7 measures for the first
time. **For a client, 64k of context now prefills at 654 tok/s where it was
440 at the old batch**, and depth zero at 1146.6 where it was 691.

## P11-6: `hc.cn`, one workgroup a token

The hyper-connection combine fused with the next mixer's norm is the **second
largest kernel in a prefill** — 10.7% of an 8192-row pass — and it does no
arithmetic: it adds the block output into a four-stream f32 residual and norms
each stream out to fp16. 901 MB a dispatch in 7.0 ms is **128 GB/s where a
copy on this machine gets 236** (`results/bandwidth.csv`).

It was one workgroup per **(token, stream)**, and the four streams of a token
combine *the same* block output row — `out[t]` has no stream index. So those
2560 floats were read four times: 110 KB a token where 80 would do. One
workgroup a token, the output row held in the ten registers a thread already
had spare, and the streams walked one at a time:

| | before | after | |
|---|---:|---:|---:|
| `hc.cn` at 2048 rows | 163.7 ms | **119.4** | 1.37x |
| `hc.cn` at 8192 rows | 661.1 ms | **459.4** | **1.44x** |
| the graph at 2048 | 1165.1 tok/s | **1191.8** | 1.023x |
| the graph at 8192 | 1328.5 tok/s | **1372.2** | 1.033x |

**Bit-identical**, and deliberately so: each stream still reduces through the
same 256-way tree over the same per-thread partials, which is the thing L2c
and P1a built this kernel's width around. The one new line is a leading
`barrier()` so `partials` can be reused across the four streams.

What it does *not* fix is the rate: 655 MB in 4.89 ms is **134 GB/s**, so the
kernel is still at 57% of what a copy gets and the redundancy was not why.
That is now the cleanest unexplained number in the prefill graph.

## P11-7: how much the QSA selection can actually skip, and why a finer skip loses

Every block of the model is flat in depth except attention (P11-5), so the
question is what decides *attention's* slope. It is not the selection's
density — 2051 cells however deep the cache, which at 64k is 3.2% — because
`llm_attn_wmma.comp` skips at the granularity of a **(query tile, key block)**
pair, and a cooperative-matrix fragment is sixteen rows. What matters is the
density of the **union** over the queries a tile covers, and that is a fact
about the text.

`AttnGPU.SelSkip` measures it on the bitmask a real 2048-token batch leaves
behind; `cmd/llm -depth` prints it per depth. Live (query tile) x (key block)
pairs, and the floor a single query row would reach:

| depth | 16x16 | 16x32 | 16x64 | 32x64 | 64x64 | one row |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100% | 100% | 100% | 100% | 100% | 100% |
| 8 000 | 75.5 | 86.7 | 94.8 | 98.0 | 99.3 | 70.3 |
| 32 000 | 39.1 | 51.5 | 65.1 | 75.8 | 85.2 | 30.1 |
| 64 000 | **18.5** | **25.9** | 35.6 | 45.4 | 56.3 | **14.7** |

Three things fall out. **The shipped rung is already the good one**:
`DefaultAttnKernel` is `qt1_kt2`, a 16x32 tile, which at 64k keeps 25.9% live
— the ladder that L2f chose on 512-token prefills happens to be the right
choice at depth for a different reason. **A per-query gather is worth 1.76x
and not the 31x the density suggests**, which is L4b's "at prefill the density
makes it pointless" restated with a number: adjacent queries select nearly the
same cells, so the union over sixteen of them is only 1.76x their individual
reach. And **the whole remaining headroom is the 1.40x between a 32-cell block
and a 16-cell one**.

That last one is a k-tile: a block is `KTIL` cooperative-matrix tiles and each
has its own K and V fragment and its own `coopMatMulAdd`, so a tile with
nothing selected can be dropped on its own — bit-identically, by exactly the
identity the block skip already rests on. It was built: a per-word OR of the
staged mask down the query tile, one scalar `kLive[kk]` per tile, and a
`continue` on it in the QK loop, the row max, the exponential, the PV loop and
the row sum.

**It is 1.18x slower, and at 64k it does not finish.** 866.1 → 737.2 tok/s at
16 000 and 775.4 → 636.2 at 32 000; at 64 000 the pass got slow enough to hold
the graphics ring past P0's two-second watchdog and was reset mid-submit.

This is the **same answer P11-4 got in a different kernel**, and two
independent measurements make it a rule rather than an anecdote:

> **Work removed from inside an unrolled cooperative-matrix loop by a branch
> costs more than it saves.** The MoE GEMM lost 1.15x to a per-fragment row
> bound that removed 35% of its rows; the attention kernel lost 1.18x to a
> per-tile bound that removed 29% of its multiplies. Neither kernel is
> arithmetic-bound — they are bound by issue and latency — so a branch adds to
> the term that dominates and subtracts from the one that does not. The way to
> skip more is to change the **loop's granularity** (a rung), not to add a
> test inside it.

Which leaves, for the depth term: `attn.attn` runs at **9.8 TFLOP/s of useful
work at 64 000 cells against 16.0 at depth zero** on the same kernel, so
there is ~1.6x in it that is not about skipping anything at all. The two
candidates are the parts that only run when the selection is on — the row
max's `selected()` scan, which is a serial loop over sixteen cells on
**sixteen of the wave's sixty-four lanes**, and the mask loop that every live
block now passes through. The first is a max-reduction and so is exactly
bit-identical under any reassociation; nobody has tried widening it.

## What is open, in order

- **The attention kernel's row-max scan uses sixteen of sixty-four lanes.**
  `for (i = lane; i < BM; i += WAVE)` with BM 16 and WAVE 64, then a serial
  walk of TILE cells per row per k-tile with a `selected()` bit test on each —
  and it only runs at all when the selection is on, which is exactly at depth.
  A max is associative and exact under any reassociation, so spreading it over
  all 64 lanes with a clustered subgroup reduce is **bit-identical**. It is the
  one obviously under-parallelised loop left in the block body, and it is the
  first thing to try against the 1.6x gap between `attn.attn`'s 9.8 TFLOP/s at
  64k and its 16.0 at depth zero.
- **`hc.cn` is still at 134 GB/s** after P11-6 removed the redundancy, where a
  copy gets 236. The other candidate is the norm's tail: an eight-barrier
  256-way tree where a subgroup reduction is one barrier. That one is **not**
  bit-exact and `llm_hc_norm.comp` would have to move with it.
- **`attn.select` is 7.4% of a prefill token at 64 000 cells** and 46x its
  depth-zero cost — four radix passes and an emit, each streaming the row out
  of DRAM. The row's keys are *block* scores repeated `ratio` times, so a
  select over `nKV/ratio` entries followed by one expanding emit is ~2.5x the
  traffic back; the risk is the causal boundary, where a block is partly
  masked and the reference's tie fill is a block split.
- **`moe.up` is still a third of a 2048-row prefill** and P11-4 says the way
  in is fewer tiles, not fewer rows. The unmeasured one is **the unpack
  prefetch** — issuing a K-step's bank loads before the unpack that consumes
  them, L8d's third lead — which is about the tile's cost rather than the
  tile count, and is the only idea left that attacks the term that dominates.
- **`attn.select` at prefill**: 46x from depth 0 to 64k, and P10 widened it
  for *decode*, where it is one row. Nobody has looked at its prefill grid.
- **The batch could be 4096** and is 1.13x more there; it needs a memory
  budget for the deployment rather than a measurement.

## Reproducing

	go build -o /tmp/pp ./cmd/llm        # once, never `go run` per arm
	export LLM_BANK_CACHE=models/Qwen3.8-Flash-Next-GGUF/bank-cache
	export LLM_DENSE_BANK=deltanet=q4_k/32,hyper_conn=q5_k/32,lm_head=q5_k/32,full_attn=q5_k/32,qsa_indexer=q5_k/32
	export LLM_MOE_BANK=gate_shexp=q4_k,up_shexp=q4_k,down_shexp=q5_1,down_exps=iq4_nl
	M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

	/tmp/pp -model $M -graph -tokens 512,1024,2048,4096,8192 -ctx 8192 -csv results/p11_pp.csv
	/tmp/pp -model $M -moe -tokens 512,2048,4096 -iters 3 -layers 1      # one layer, 228 ms to stage
	/tmp/pp -model $M -depth -depths 0,8000,16000,32000,64000 -pp 2048 -tg 8 -ctx 76000

`-depth` prints the selection's live-pair table (P11-7) after every timed
prompt batch, which is the instrument that says what a change to the attention
kernel's granularity could possibly be worth before anyone writes one.

The `-moe` bench is the loop to work in: it stages one layer in **228 ms**,
runs the real 4k routing trace, and reports per-mode GFLOP/s and GB/s of bank,
so a kernel idea is five minutes rather than the seven a 48-layer ladder takes.
The gates after touching any of this:

	go test ./llm/ -run TestMoEGPU              # the block, the ladder, the padding's inertness
	go test ./llm/ -run TestGraphIsAChunkSplit  # the graph
