<!-- LLM.md L6b. The five blocks in llama.cpp's own order, from tokens to
     logits: the order, the drift with depth, the memory type that turned out
     to be worth 5x, and the prefill ladder. Cited from llm/graph.go,
     llm/gpu_head.go, llm/arena.go, vk/engine.go and cmd/llm/bench_graph.go. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L6a](l6a-residency.md) · [L2a](l2a-prefill-attribution.md) · phase 1

# L6b — the graph: llama.cpp's own logits, and 1.82x its prefill

**Result: qwen3.8-flash-next runs end to end in Go on Vulkan. The last token
of a seven-token prompt picks llama.cpp's token, out of a top ten that is
llama.cpp's top ten, and prefill is 713.9 tok/s at ubatch 2048 against the
reference's best-of-any-ubatch 391.42 — 1.82x.** At llama.cpp's own best
ubatch of 512 it is 430.4, 1.10x, and at 128 it is 177.3 and loses.

Two things made that number, and only one of them was a kernel.

**The kernels were already there.** L2 through L5 built a block at a time
against llama.cpp's own tensors, and L6a put all 48 layers of all five of them
on the device at once. L6b is the order, the two tensors no block owned — the
embedding gather and the 675 MB language-model head — and the plumbing
between them. Nothing in a shader changed.

**The plumbing is where the time went, and the reason was a memory type.**
Every block here owns four buffers, and the two activation arenas had never
been read on a hot path: the block tests read one tensor at the end, and the
profilers time the GPU. The graph reads them twice a sublayer, 96 times a
pass. On this device the memory type `vkAllocateMemory` was being asked for is
write-combined, and **the host reads it at 0.18 GB/s** — where a HOST_CACHED
type reads the same buffer at **25.06**, and writes it faster too. One line
choosing a different `memoryTypeIndex` was **5.0x on the whole graph**, and
the control says it costs the kernels 0.14%.

## L6b-1: the order, and the two tensors nothing owned

`llama_model_qwen4exp::graph::graph`, which is not the obvious order:

```
res = repeat(embed(ids), hc)            "hc_init"
for each layer:
    if it is the PLE layer: res += ple(res, ngram_embd)
    mixed, inject = hc_mix(res, attn)   the layer's *attention* mixer
    out = deltanet(mixed) | attention(mixed)
    res = hc_combine(res, out, inject)  "hc_combine-L"
    mixed, inject = hc_mix(res, ffn)
    out = moe(mixed)                    "ffn_out-L"
    res = hc_combine(res, out, inject)  "l_last-L"
norm = hc_mix(res[last], head)          "result_norm"
logits = output * norm                  "result_output"
```

Three things about it are worth saying out loud.

**There is no output norm.** `output_norm.weight` is absent from this
checkpoint and that is not an omission: the final hyper-connection mixer *is*
the norm, which is why `llm/gpu_head.go` is a bare matmul with no epilogue at
all — the only one in the vertical.

**The final mixer and the head run on one row.** `inp_out_ids` drops every
token but the last before them, and the trace agrees: `result_output` is one
row of 248320 floats for a seven-token prompt, not seven. Doing anything else
would be 508 MB of logits a graph at ubatch 512 for 511 rows nothing reads,
and it would not be the reference's arithmetic to compare against.

**The wide residual never leaves the hyper-connection block's arena.**
`llm_hc_combine.comp` updates it in place, so what crosses a block boundary is
the 2560-wide `hc_mixed` going out and the 2560-wide block output coming back
— not the 10240-wide stream. The PLE block is the one exception, once a graph.

The head itself is `output.weight`, [2560, 248320] Q8_0: 675 MB in the
checkpoint and 1.27 GB as the halves the matrix cores want. It is staged a
slab of 4096 rows at a time, dequantised and tiled straight into the bank,
because the fragment tiling's row blocks are sixteen rows wide and a slab
boundary that is a multiple of sixteen maps to a contiguous slab of the bank —
which is what keeps a 2.54 GB f32 copy of the head from ever existing.

## L6b-2: it picks llama.cpp's token

`go test ./llm/ -run TestGraphLogits`, the seven-token prompt *"The capital of
France is Paris."*:

| | ours | llama.cpp |
|---|---|---|
| argmax | **561** (15.5956) | **561** (15.8578) |
| top 10 | 561 11751 271 1049 1061 248046 198 9338 19592 15767 | 561 11751 271 1061 1049 19592 198 248046 15767 9338 |
| `result_norm` | rms 5.480e-01 on \|105.6\| | **0.519% of scale** |
| `result_output` | rms 3.144e-01 on \|15.86\| | **1.983% of scale** |

The argmax is the reference's and the top ten are the reference's ten tokens.
Their *order* differs in three adjacent pairs, and that is reported rather
than demanded: an inversion between logits a thousandth apart is a different
fact from a wrong token, and no logit tolerance establishes the first without
also forbidding the second.

## L6b-3: the drift is a curve, not a defect

0.5% at the bottom of 48 layers is either accumulation or a bug, and one
number cannot tell them apart. So the oracle was run a second time over the
same prompt, filtered to `l_last` every eight layers
(`reference/out/llmdepth/`), and the graph was stopped at each of those depths
off the same staged model:

| layers | rms | \|ref\|max | of scale |
|---:|---:|---:|---:|
| 8 | 1.026e-03 | 3.025 | 0.034% |
| 16 | 1.613e-03 | 2.929 | 0.055% |
| 24 | 3.053e-03 | 2.954 | 0.103% |
| 32 | 6.574e-03 | 4.277 | 0.154% |
| 40 | 2.347e-02 | 7.118 | 0.330% |
| 48 | 7.228e-02 | 8.201 | **0.881%** |

That is a clean geometric curve — **×1.085 a layer**, no step at any depth and
no block that breaks it. Every block in this model is a Q8_0 or Q4_K matmul
with fp16 operands where llama.cpp evaluates an integer dot product over int8
activations (L2b-2) and accumulates in fp16 (L4a-5), so a per-layer relative
error of a tenth of a percent compounding through 48 residual additions is
exactly this shape. The four-layer prefix test prints the same column at
depths 1-4 (0.070%, 0.069%, 0.091%, 0.108%) in 7 GB and three seconds, which
is the one to run while the order is being changed.

## L6b-4: the arenas were on the wrong memory type, and it was worth 5.0x

The first working graph did **280.3 tok/s at ubatch 512, 0.72x llama.cpp**,
with 45.5% of its wall clock in the host plumbing. The plumbing is not much
data — 5.24 MB a hop — so the rate was the question, and the answer was not
what the memcpy rate suggested:

| | 5.24 MB |
|---|---|
| write into a mapped arena | 50.0 GB/s |
| host to host, for scale | 70.7 GB/s |
| **read out of a mapped arena** | **0.18 GB/s** |

A write-combined mapping is fast to write and catastrophic to read: every load
misses to DRAM a line at a time. `vkGetPhysicalDeviceMemoryProperties` offers
six host-visible types here that can back a storage buffer
(`results/l6b_arena.csv`):

| type | heap | flags | host read | host write |
|---:|---:|---|---:|---:|
| 2 | 0 | HOST_VISIBLE\|HOST_COHERENT | 0.18 GB/s | 50.51 |
| 3 | 1 | DEVICE_LOCAL\|HOST_VISIBLE\|HOST_COHERENT | 0.18 GB/s | 51.62 |
| **5** | **0** | **HOST_VISIBLE\|HOST_COHERENT\|HOST_CACHED** | **25.06 GB/s** | **73.78** |
| 8 | 0 | ...\|DEVICE_COHERENT\|DEVICE_UNCACHED | 0.18 GB/s | 51.20 |
| 9 | 1 | DEVICE_LOCAL\|...\|DEVICE_UNCACHED | 0.18 GB/s | 50.99 |
| **10** | **0** | **...\|HOST_CACHED\|DEVICE_COHERENT\|DEVICE_UNCACHED** | **24.07 GB/s** | **68.02** |

**139x on the read**, and the write is 1.4x faster as well. Type 3 —
DEVICE_LOCAL, which sounds like the right answer — is one of the slow ones.

This is the other half of **L0b**, which measured the same table from the
*device* and found 236.0-237.4 GB/s across all of it, 0.57% over 32 cells. So
the choice is free on the GPU and decisive on the host, and the control says
so directly: the MoE block, 97% of the parameters, runs its layer in
**10834.5 us on cached arenas and 10850.2 on write-combined ones — 0.14%**, on
GPU timestamps, reproducing L5b's isolated 10824.7 to 0.09%.
`LLM_ARENA_UNCACHED=1` is that control, kept as a flag rather than as a note.

Weight banks stay on the type they were on. A bank is written once and never
read, so the cached type buys only its faster write, and leaving it alone
keeps L0a's and L6a's residency measurements measuring the same allocation.

**The second half of the plumbing was the narrowing, not the memory.** With
cached arenas the remaining glue was `Upload`'s f32→fp16 conversion: 1.31
million values a sublayer at ubatch 512, 96 sublayers, 126 million a graph, on
one core. Parallelising it over the rows — they are independent inside a
prefill, so it changes the wall clock and not one value — took the glue from
832 ms to 264 at ubatch 512.

## L6b-5: the ladder

`go run ./cmd/llm -graph -tokens 128,512,1024,2048` (`results/l6b_graph.csv`),
two timed passes after one thrown away, 48 layers, 85.47 GB resident:

| tokens | ms | tok/s | vs llama.cpp's 391.42 | blocks alone | glue |
|---:|---:|---:|---:|---:|---:|
| 128 | 722.0 | 177.3 | 0.45x | 247.4 tok/s | 28.0% |
| 512 | 1268.2 | 403.7 | 1.03x | 545.6 tok/s | 25.4% |
| 1024 | 1848.8 | 553.9 | 1.42x | 740.1 tok/s | 24.6% |
| **2048** | **2868.7** | **713.9** | **1.82x** | **931.6 tok/s** | 22.7% |

Staged for 512 alone — a quarter of the arenas, and so a different cache
story — ubatch 512 does **430.4 tok/s, 1.10x**.

**The schedule inverts against the reference's.** llama.cpp's own ladder is
347.71 / 391.42 / 380.91 / 336.72 at 256 / 512 / 1024 / 2048: it peaks at 512
and *loses* 14% by 2048, because its 10240-wide F32 residual is 20.97 MB at
512 — just inside the 32 MiB MALL — and 83.9 MB at 2048 (L2a-4). Ours has no
such cliff, because the residual is never materialised as a graph tensor and
the elementwise glue that reads it was fused away at L2c. So against
llama.cpp *at 2048*, where a long prompt actually is, the graph is **2.12x**.

Where the 2048 pass goes, by phase:

| | ms | % |
|---|---:|---:|
| moe | 1225.9 | 42.7 |
| deltanet | 469.3 | 16.4 |
| hyper-conn | 338.4 | 11.8 |
| attention | 143.4 | 5.0 |
| ple n-gram | 14.9 | 0.5 |
| lm head | 6.5 | 0.2 |
| gather (host) | 20.4 | 0.7 |
| **glue (host)** | **649.8** | **22.7** |

The MoE is 42.7% here against L2a-2's 35.7% of llama.cpp's graph, which is
what it looks like when everything around it got faster and it got 1.55x.

## What is left

> **Since: [L6c](l6c-moves.md) did it, and not with a shared arena.** The move
> became one dispatch over two blocks' buffers rather than one arena across
> five — bit-exact, and **2.0% of the pass** against the 22.7% below. Prefill
> is now **941.3 tok/s at ubatch 2048, 2.40x**. The paragraph below is what it
> was scoped against, and L6c-3 is why the shared arena was priced rather than
> built.

**The glue is 22.7%, and it is not arithmetic.** Every block owns its own
arenas, so `hc_mixed` is read out of one fp32 arena, narrowed to halves and
written into another. At 25 GB/s and on sixteen cores that is now 650 ms at
ubatch 2048 rather than 5 s, but the reference does none of it. Deleting it is
a shared-arena change: one activation buffer across the five blocks, the
sublayer's input offset pointed at the mixer's output, the combine's block
output pointed at the sublayer's, and one small narrowing kernel where a f32
tensor has to become an fp16 A operand. **931.6 tok/s is what that is worth at
ubatch 2048**, and L2a-3's projection for the same graph with its glue fused
was ~1150.

**Decode is untouched.** The KV cache is written from cell zero every pass and
every DeltaNet state is reset; continuing a sequence is L7's, and L2a-5 says
that one is won by not dispatching 3837 kernels rather than by a better GEMV.
