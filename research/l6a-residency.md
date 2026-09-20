<!-- LLM.md L6a. The whole model resident: the bank as an array of buffers,
     what 48 layers of five block types cost in bytes and in buffers, and the
     measurement that says residency is free. Cited from cmd/llm/resident.go,
     llm/gpu_moe.go, vk/shim.c and shaders/llm_common.glsl. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L5b](l5b-moe-gpu.md) · [L0a](l0a-bank-range.md) · phase 1

# L6a — the whole model on the device: 84.20 GB in 68 buffers, and a bank that is an array

**Result: every layer of every block of qwen3.8-flash-next is resident at
once — 84.20 GB of weights and 0.87 GB of arenas in 68 buffers, staged in 33
seconds, leaving 41 GB of this machine's memory free.** The MoE block runs at
the same speed with 48 expert banks live as with two, so residency costs
nothing that scales; L6b's graph can assume the weights are simply there.

Getting there needed one structural change, and it was not in a kernel. Every
block in this vertical lays its weights out as **one arena with a per-layer
offset** — `bank[layer*perLayer + ...]` — and that arrangement cannot express
this model. `maxStorageBufferRange` on this device is **4 GiB − 4**, a layer's
expert bank is 1.61 GB, and there are 48 of them. So binding 5 became an
**array of buffers, one a layer**, which meant a descriptor array in the
shim, a block-instance array in the GLSL, and a layer index in a push block
that was already full.

## L6a-1: the plan, measured

`go run ./cmd/llm -resident`, arenas for 512 tokens in a 2048-cell cache:

| block | layers | buffers | weights | arenas | staged in |
|---|---:|---:|---:|---:|---:|
| hyper-conn | 97 | 4 | 1.31 GB | 42.6 MB | 1.70 s |
| ple n-gram | 1 | 4 | 0.07 GB | 91.9 MB | 0.11 s |
| deltanet | 36 | 4 | 4.18 GB | 216.4 MB | 4.40 s |
| attention | 12 | 4 | 1.24 GB | 56.4 MB | 1.39 s |
| moe | 48 | 52 | 77.41 GB | 462.0 MB | 29.26 s |
| **total** | | **68** | **84.20 GB** | **869.2 MB** | **36.9 s** |

Three numbers in that table are the answer to a question L5b left open.

**68 buffers**, where LLM.md's estimate was "~100". Only the MoE needs more
than the four every block here has, and it needs one a layer plus its four.

**84.20 GB against the checkpoint's 82.52 GB resident core**, which is 1.04x
once the 0.68 GB embedding table and the 0.68 GB lm head — neither of which
any block stages yet — are taken off the reference figure. The +3.04 GB is
the dense half being staged as **halves**: 6.79 GB of fp16 against the 3.67 GB
of Q8_0 and F32 those tensors ship as, 1.85x. It is the price of §2.8's
fragment tiling, it is paid once at prefill, and it is 3.6% of the model.

**41 GB of MemAvailable left**, against a 28.80 GB n-gram table that stays
mmap'd (D2) and is read 1.41 KB a token. The capacity question that has been
open since L0 — *does a 180 B model fit on a 117.7 GiB machine with our
layout rather than llama.cpp's?* — is answered yes, with room for the table's
whole working set and 12 GB besides.

**The DeltaNet bank is the near miss.** Its 36 fused [16512, 2560]
projections plus their output matrices are **4.18 GB in one buffer**, which
clears this device's 4 GiB − 4 by 2.7%. A checkpoint with two more linear
layers, or a dense half staged in anything wider than halves, would have
needed the same treatment the MoE got.

## L6a-2: the bank is an array of buffers, one a layer

The MoE GEMM reads its B operand as raw bytes out of the checkpoint's own
Q4_K/Q5_K/Q5_1/Q8_0 blocks (L5b-2). At L5b that bank was a single buffer with
a byte offset per tensor, and the constructor said so: *"one layer costs
1.57 GB of device memory and two is the most a 4 GiB buffer holds."* Two is
also the most a `uint32` byte offset addresses. Both walls are the same wall.

So `qbuf` became `qbufs`, one allocation a layer, and binding 5 became an
array:

```glsl
layout(binding = 5) readonly buffer QBank  { uint  w[]; } qb [NBANK];
layout(binding = 6) readonly buffer QBank4 { uvec4 w[]; } qb4[NBANK];
#define qbank  qb[MOE_BANK].w
#define qbank4 qb4[MOE_BANK].w
```

Three things about that are worth stating, because each was a choice with an
alternative.

**It is core Vulkan, not descriptor indexing.** One dispatch is one layer, so
the index is dynamically uniform, and an array of block instances indexed by a
push constant needs no extension and no `nonuniformEXT`. The two macros mean
the kernel goes on spelling its thirty-odd loads `qbank[...]` — which buffer a
dispatch reads is the same fact for every one of them, so it belongs at the
declaration.

**NBANK is compiled in at 48 and short stages pad the array.** A
specialization constant cannot size a descriptor array portably, and the
SPIR-V here is pre-compiled by `go generate`, so the width is fixed and
`moeMaxBanks` in Go has to agree with `-DNBANK=48` in `shaders.go`. Every
descriptor of an array must be written whether or not a shader indexes it, so
a stage of four layers binds four real banks and 44 copies of the block's
256-byte placeholder. `TestMoEGPUBankArray` fails loudly if the checkpoint's
layer count ever stops being 48.

**The layer index rides in the top sixteen bits of `moeUsed`.** The push block
is 64 uints, which is this device's entire `maxPushConstantsSize` of 256
bytes, and L5b's own comment on it reads *"Five fields, and the block is out
of room"*. There is no 65th field to put a bank index in. The used count is at
most sixteen, so the two share a uint behind `MOE_USED` and `MOE_BANK` and no
kernel spells `pc.moeUsed` any more.

The shim change is the other half: `shim_create_compute_pipeline` gained a
per-binding `counts` array, so a binding's `descriptorCount` can be more than
one and the flat buffer list is its concatenation. Seven bindings, 5 + 48 + 48
= 101 buffers. `counts == NULL` is the arrangement every other kernel in the
repo uses and is byte for byte what it was.

## L6a-3: the bank index is load-bearing, and the control says so by 203x

Every other test in this vertical stages layer 3 alone, where the bank index
is zero — and an index that was silently ignored would look exactly like an
index that was read. So `TestMoEGPUBankArray` stages layers 0 through 3, 6.80
GB, and asks for the last:

| | `ffn_out-3` against llama.cpp |
|---|---|
| layer 3 from bank 3 | **8.133e-05 rms** |
| the same run, bank index forced to 0 | 1.652e-02 rms — **203x further away** |

Layer 0's experts against layer 3's activations and layer 3's routing is a
perfectly plausible tensor: right shape, right magnitude, `reference |max|`
unchanged. Nothing but the comparison tells the two apart, which is why the
control is in the test rather than in a comment.

## L6a-4: residency is free in the bank's size

The kernel is unchanged, the descriptor array is 48 wide either way, and a
dispatch reads the same 1.6 GB whatever else is resident. So sweeping how many
banks are live, with the four dense blocks staged and held still, prices the
one thing L0a could only price in isolation — whether a real 84 GB working set
behaves like a 10 GB one:

| banks | resident | layer 3's block | a graph |
|---:|---:|---:|---:|
| 2 | 10.4 GB | 11134 us | — |
| 4 | 13.6 GB | 11062 us | — |
| 12 | 26.5 GB | 11162 us | — |
| 24 | 45.5 GB | 11067 us | — |
| 48 | **84.2 GB** | **11242 us** | **539.6 ms** |

**An eightfold range of resident bytes, and a 1.6% spread with no trend** —
which is L0a's finding (*"the DRAM bus does not care how big the weight bank
is"*) holding at the size the model actually is, with 68 live buffers rather
than one probe's 23. `results/l6a_resident.csv`.

What the sweep does show is a **flat ~2.5% step** against L5b's own figure:
10824.7 us for layer 3 with the MoE block alone on the device, 11062-11242 us
with the rest of the model beside it, at every width including two banks. It
is not the bank count and it is not the indirection — both are constant across
that column — so it is the presence of the other four blocks' 6.79 GB and
their arenas. At ~2.5% against a ~1% run-to-run noise band it is written down
rather than chased; L6b runs all five blocks in one graph anyway, and that is
where it will either matter or disappear.

## L6a-5: staging is 33 seconds, and the order is what makes it fit

77 GB of expert bank is `memcpy` out of the mmap'd checkpoint and takes 29 s,
about 2.6 GB/s. The dense half takes 7.6 s, and it is the more delicate half:
a layer's weights are **dequantised to f32 on the host** before they are
packed into fp16 fragment tiles, and the 36 DeltaNet layers are **8.35 GB of
transient floats held at once**.

So the tool stages the dense blocks first, drops each block's host floats
before the next asks for its own (`runtime.GC` + `debug.FreeOSMemory`), and
stages the MoE bank last. Peak is then the resident model, not the resident
model plus a dequantisation — the ordering is worth 8 GB on a machine that
ends the run with 41.

## How to run it

    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

    go run ./cmd/llm -resident -model $M                  # the whole model
    go run ./cmd/llm -resident -model $M -dense           # the 6.8 GB half alone
    go run ./cmd/llm -resident -model $M -bank 48,24,12,4,2 \
        -csv results/l6a_resident.csv                     # L6a-4's sweep

    go test ./llm/ -v -run TestMoEGPUBankArray            # the index, and its control
    go test ./llm/ -v -run TestMoEGPU                     # L5b's gates, unchanged
