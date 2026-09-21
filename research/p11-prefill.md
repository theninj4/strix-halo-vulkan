<!-- P11. Prompt processing, short and long: where a prefill token goes after
     P7-P10, the two changes that moved it, and the three that did not.
     Cited from TODO.md, backend/llm.go, cmd/serve/main.go,
     shaders/llm_moe_gemm.comp and llm/gpu_moe.go. -->

[← TODO.md](../TODO.md) · [research index](README.md) · [L2a](l2a-prefill-attribution.md) · [L5b](l5b-moe-gpu.md) · [P0](p0-ring-watchdog.md) · [P7](p7-context-depth.md)

# P11 — prefill: the batch was the biggest number in the file, and the padding was not

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
| 0 | **1119.1** | 1.00x | 0.894 | 0.142 | 0.156 | 0.479 | 0.083 |
| 8 000 | 930.3 | 0.83x | 1.075 | 0.142 | 0.156 | 0.488 | 0.216 |
| 16 000 | 866.1 | 0.77x | 1.155 | 0.142 | 0.156 | 0.464 | 0.318 |
| 32 000 | 775.4 | 0.69x | 1.290 | 0.142 | 0.157 | 0.458 | 0.462 |
| 64 000 | **642.2** | 0.57x | 1.557 | 0.142 | 0.157 | 0.461 | **0.727** |

**Every block is flat in depth except attention, and attention is 97% of the
falloff** — 0.644 ms a token of the 0.663 a token a prefill gains between
depth zero and 64k. At 64 000 cells it is 47% of a prefill token. Inside it,
by label (ms a token, depth 0 → 64 000):

	attn.attn     0.0189 -> 0.5122   27x
	attn.select   0.0025 -> 0.1152   46x
	attn.score    0.0080 -> 0.0391    5x
	attn.expand   0.0180 -> 0.0256

`attn.attn` and `attn.select` are 0.627 of the 0.644. This is the quadratic
term any dense attention has and llama.cpp has it too; what is *ours* is that
the QSA selection does not help here the way P7 made it help at decode — at
2048 queries a batch the union of selected key blocks is very nearly
everything, which is L4b's reason for declining the gather at prefill, still
standing. **For a client, 64k of context now prefills at 642 tok/s where it
was 440**, and the two changes are why.

## What is open, in order

- **`hc.cn` at 128 GB/s** — 10.7% of an 8192-row prefill and 9.3% of a 2048
  one, elementwise, against 236 GB/s for a copy. Two candidates, both cheap:
  the block output `act[src]` is read once per *stream*, so four times a
  token, where one workgroup a token would read it once; and the norm's tail
  is an eight-barrier 256-way tree where a subgroup reduction is one barrier
  (that one is not bit-exact and would have to move `llm_hc_norm.comp` with
  it).
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

The `-moe` bench is the loop to work in: it stages one layer in **228 ms**,
runs the real 4k routing trace, and reports per-mode GFLOP/s and GB/s of bank,
so a kernel idea is five minutes rather than the seven a 48-layer ladder takes.
The gates after touching any of this:

	go test ./llm/ -run TestMoEGPU              # the block, the ladder, the padding's inertness
	go test ./llm/ -run TestGraphIsAChunkSplit  # the graph
