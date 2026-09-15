<!-- LLM.md L2c. The fused hyper-connection kernel: four dispatches where
     llama.cpp has sixteen, checked against L2b's CPU reference and the
     reference's own tensors. Cited from llm/gpu.go and shaders/llm_hc_*. -->

[← LLM.md](../LLM.md) · [research index](README.md) · [L2a](l2a-prefill-attribution.md) · [L2b](l2b-hyper-connections.md) · phase 1

# L2c — the fused hyper-connection kernel

**Result: the block L2a costed at over half of llama.cpp's prefill graph runs
on the device in four dispatches instead of sixteen, and the lines of the
reference's graph it replaces go from 226.6 ms to 53.4 ms — 4.24x, and 14.9%
of the whole 1165 ms prefill graph removed by one block.** It reproduces
L2b's CPU reference to **1e-04 rms** across all eight mixers of the four
dumped layers, which is fp16's own rounding, and it is **38x nearer the f32
model than llama.cpp's own answer is**.

| at 512 tokens, per graph | llama.cpp | ours | |
|---|---:|---:|---:|
| `RMS_NORM(2560,4,512)` x98 | 16.7 ms | **5.3 ms** | 3.15x |
| `MUL_MAT q8_0 m=320 k=10240` + `MUL_MAT f32 m=4 k=10240` x95 | 163.3 ms | **24.7 ms** | 6.61x |
| `MUL_MAT q8_0 m=10240 k=320` x95 | 30.7 ms | **16.8 ms** | 1.83x |
| `REPEAT` x98 (the combine's broadcast) | 15.9 ms | **6.6 ms** | 2.42x |
| **block** | **226.6 ms** | **53.4 ms** | **4.24x** |

Both sides are one prefill graph of 512 tokens, llama.cpp's best ubatch.
Theirs is `results/l2a_prefill_ops.csv`; ours is `results/l2c_hc.csv`, at 97
mixers a graph. **And the rest of the block's glue is not in that table at
all** — the gamma multiply, the gate's sigmoid, the `xn*gate` multiply, the
collapse's three adds, the combine's scale/sigmoid/mul/add — because those
dispatches are inside the `MUL` (479), `ADD` (231), `SIGMOID` (326) and
`MULTI_ADD` (170) lines the whole model shares. They are **absent from our
graph rather than faster in it**, so 4.24x is the conservative reading.

## The four dispatches

`shaders/llm_hc_norm.comp`, `llm_gemm.comp` (two of its three epilogues) and
`llm_hc_combine.comp`,
driven by `llm/gpu.go`:

| ours | what llama.cpp spends on it |
|---|---|
| **norm** — grouped RMSNorm, gamma, narrowed straight into the GEMM's fp16 A layout | `RMS_NORM` + `MUL` + `CONT` |
| **down** — `[lowRank + hc, wide] * xn`, silu on the accumulator, fp16 `lo` out, `inject` out | `MUL_MAT` x2 + `SCALE` + `SILU` |
| **up** — `[wide, lowRank] * lo`, sigmoid on the accumulator, 4-branch collapse in LDS, `mixed` out | `MUL_MAT` + `SIGMOID` + `MUL` + 3 `ADD` |
| **combine** — `res += out * 2*sigmoid(inject/hc)` | `SCALE` + `SIGMOID` + `REPEAT` + `MUL` + `ADD` |

Two of the block's tensors never exist, and that is where the time goes.

**1. `inject` is four more output columns on `down`.** L2a's finding 3: the
reference projects `xn` down to four scatter weights per token with an F32
`[10240, 4]` matrix, 95 times a graph, at **31.8 GFLOP/s — 0.06% of this
part's peak and 10.3% of prefill**. It reads the *same* `xn` that `w_down`
reads. The host packs them as one `[336, 10240]` fp16 weight (320 is 20
fragment tiles, so the split is on a tile boundary and costs a compare per
tile, not a branch per element), and the epilogue writes `silu(lo/hc)` into
the fp16 arena and `inject` into the fp32 one. **163.3 ms becomes 24.7.**

**2. The gate is consumed where it is computed.** `gate` is `[10240, T]` —
**21 MB at 512 tokens, i.e. the whole 32 MiB MALL**, per mixer — and nothing
downstream wants it: `mixed[t][i] = mean_c(xn[t][c][i] * gate[t][c][i])`. So
the up projection's epilogue applies the sigmoid to its accumulators, stores
them to LDS, and collapses the four streams against `xn` there, writing the
2560-wide `mixed` instead. The kernel is the only place that tensor is ever
whole.

That needs the four streams of one feature in one workgroup, which is a
**layout, not a loop**: `packUpB` permutes the projection's output rows so
that row `c*nEmbd + i` is packed at `(i/16)*64 + c*16 + i%16`, making a
64-column block 16 features x 4 streams. `TestHCGPUUnpermutedUpIsWrong` is the
negative control — staging the checkpoint's own row order is not a crash and
not a NaN, it is a plausible tensor built from the wrong four columns, so the
only thing that distinguishes them is a test that demands they disagree.

The LDS round trip is there to avoid depending on something the cooperative
matrix extension does not define: which lane holds which element of a tile.
Storing the gate tiles and reading them back with ordinary indexing pairs each
gate element with its `xn` element portably, for 16 KB and one barrier.

## Against both references

`go test ./llm/` runs all eight mixers of layers 0-3, each from llama.cpp's
own residual (`hc_init`, then `l_last-(L-1)` or `hc_combine-L`), with two
bounds per tensor — and the gap between them is the finding:

| worst of 8 mixers | vs the f32 model (`llm.HCMix`) | vs llama.cpp |
|---|---:|---:|
| `hc_norm` | 2.74e-04 rms, maxRel **4.9e-04** | 2.74e-04 |
| `hc_gate` | **8e-05 to 1.24e-04** | 2.25e-03 to 4.67e-03 |
| `hc_mixed` | 1.15e-04 | 1.85e-03 to 6.53e-03 |
| `hc_inject` | 1.13e-03 on values to \|71.7\| | 1.13e-03 |

The left column is fp16's own step and nothing else: `maxRel` on `hc_norm` is
4.9e-04, which *is* 2^-11. The right column is **L2b's number, not ours** —
3.14e-03 on `hc_gate-0` is exactly what L2b measured for the *CPU* reference
against the same tensor, because llama.cpp's Vulkan backend evaluates these
Q8_0 matmuls over int8 activations. `TestHCGPUIsNearerTheModelThanTheReference`
states it as a checked claim: against the same f32 model, on the same input,
**the fused kernel is 38x closer than the oracle it is validated against**.

`hc_inject` is the tensor where the two columns agree, and that is the same
asymmetry that located the difference in the first place: its weights are F32
in the checkpoint, so the reference is not quantising anything on that path
and our fp16 narrow is the only error either side carries.

Two more checks worth naming. Every rung of both ladders produces
**bit-identical** output at a 7-token prompt, which is what says the M padding
and the tile indexing are right where a BM of 64 covers nine times the tokens
that exist. Layer 1's attn mixer was the one exception when this was written —
the PLE n-gram block runs between `l_last-0` and it, so the residual it reads
is not in the trace — and **L2d closes that**: the test now builds that input
by running the block, which is also the check that the two stages compose.

## What it costs, and where it stops scaling

`results/l2c_hc.csv`, per mixer, at the scheduled kernel pair:

| T | norm | down | up | combine | block | block/token |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 9.4 us | 252.8 | 40.2 | 9.8 | **312 us** | 4.88 us |
| 128 | 16.5 | 251.4 | 59.9 | 21.2 | **349** | 2.73 |
| 256 | 28.7 | 253.0 | 103.0 | 34.2 | **419** | 1.64 |
| 512 | 54.7 | 254.6 | 173.2 | 67.9 | **550** | 1.07 |
| 1024 | 376.1 | 271.6 | 390.3 | 554.8 | **1593** | 1.56 |
| 2048 | 720.1 | 475.8 | 721.9 | 1332.6 | **3250** | 1.59 |

Three things in that table.

**The down projection barely depends on the token count** — 253 us at 64
tokens and 255 at 512 — because what it reads is 6.9 MB of weight per M block
and not the activation. It is at 1.7 TFLOP/s at 64 tokens and 29.6 at 2048,
which is the same arithmetic taking the same time. That is also why its ladder
moves: the rung that wins at a short prompt is the one that fills the machine
(BM=16 gives 224 workgroups for 40 CUs; BM=64 gives 56 and a ruinous tail),
and the rung that wins at a long one is the one that reads the weight fewer
times.

**And there is a MALL cliff between 512 and 1024 tokens, which is L2a's
finding 5 again, on our own kernel.** `norm` goes from **575 GB/s to 167** and
`combine` from **695 to 170**; both are above the 236 GB/s DRAM bus on the
near side, i.e. reading out of L3, and below it on the far side. The
mechanism is exactly the one L2a identified in llama.cpp: the wide residual is
`[10240, T]` **fp32**, 21.0 MB at T=512 and 41.9 MB at T=1024, and the MALL is
32 MiB. Per token the block is cheapest at 512 tokens and 1.5x more expensive
at 2048.

**So L2a's recommendation "keep the residual in fp16" now has a measurement
behind it rather than an inference from someone else's profile**, and it is
worth 1.5x on this block at a long ubatch. It is not taken here: the residual
is accumulated across 97 combines, so narrowing it is an accuracy decision
that belongs with L6's stack and L8's perplexity run, not with a kernel.

## What this does to the prefill estimate

L2a's bottom-up total was 2879 ms with the glue unfused and 1768 ms with three
quarters of it fused — 711 and 1159 tok/s — and it named that spread as "the
widest uncertainty left in the prefill estimate". This is the first of the
fusions actually built, and on the lines that can be named it is **4.24x, at
the optimistic end of that range**.

Against llama.cpp's own graph, if nothing else changed at all: 1164.7 ms
becomes **992 ms**, so the reference's 391.4 tok/s at `-ub 512` would become
**~460**. That is not our target — it is the size of the hole this one block
was.

## How to reproduce

    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

    go test ./llm/ -v -run TestHCGPU        # against the trace and the CPU reference
    go run ./cmd/llm -hc -model $M -tokens 512
    go run ./cmd/llm -hc -model $M -ladder -csv results/l2c_hc.csv

The bench times GPU dispatches swept across **every staged mixer** rather than
repeating one: a mixer is 13.4 MB of fp16 weights, which fits the MALL, so
timing one dispatch in a loop would measure a kernel reading L3 where the
graph it models reads 97 cold ones. Eight mixers, 107.8 MB, is the default.
Two full ladder runs agree to **1.4% on the mean cell and 0.6% on the block
totals at the scheduled pair**; the dispersion is on dispatches under 20 us
and on the `up_m1` rung, where it reaches 30%.

## What is not built

The kernel reads **fp16 weights**, dequantised from the checkpoint's Q8_0 at
upload, where llama.cpp reads Q8_0 directly. That costs 2 bytes a weight
against 1.06, i.e. 13.4 MB a mixer against 7.1 — and the down projection is
weight-read-bound, so a Q8_0 path is the obvious next lever on it. It is also
what phase 2 will want anyway: L8's bank is W4A8, and this block's two
matrices are 0.70 GB of the 82.52 GB resident core.

The `hc.up` rung is the weaker half at 19.4 TFLOP/s against the 34-40 the
`shapes.csv` kernels hold on similar shapes, and it has not been tuned at all
beyond the M ladder — no K-slab depth, no hoist, no swizzle, all of which
§2.7 and §2.4 measured as worth 1.5-2.1x on the DiT's GEMM. There is more
here, and it was not the point: the point was the dispatch count.
