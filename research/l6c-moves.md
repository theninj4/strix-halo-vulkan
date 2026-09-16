<!-- LLM.md L6c. The glue between blocks moved from the host to the device:
     the move kernel, why it is not a shared arena, the fixed host cost that
     was hiding behind it, and the prefill ladder. Cited from llm/move.go,
     shaders/llm_move.comp, llm/graph.go and vk/engine.go. -->

[← LLM.md](../LLM.md) · [research index](README.md) · [L6b](l6b-graph.md) · [L2a](l2a-prefill-attribution.md) · phase 1

# L6c — the glue on the device: 941.3 tok/s, 2.40x llama.cpp

**Result: prefill is 941.3 tok/s at ubatch 2048 against llama.cpp's
best-of-any-ubatch 391.42 — 2.40x — and 572.9 at ubatch 512 against its 313.62
for the same 512-token prompt, 1.83x. Nothing crosses a block boundary through
the host any more, and not one number in the graph changed: `l_last` at all
six depths, `result_norm` and `result_output` are identical to the last digit.**

L6b left 22.7% of the graph in host plumbing. Every block here owns its own
four buffers, so an activation leaving one block and entering the next had to
cross a buffer boundary, and it crossed it by being read out of one mapped
arena, narrowed to halves on sixteen cores, and written into the other — 96
round trips a pass. This is that move as one dispatch.

| ubatch | L6b | L6c | | vs llama.cpp's 391.42 |
|---:|---:|---:|---|---:|
| 128 | 177.3 | **259.0** | 1.46x | 0.66x |
| 512 | 403.7 | **570.4** | 1.41x | 1.46x |
| 1024 | 553.9 | **761.8** | 1.38x | 1.95x |
| **2048** | **713.9** | **941.3** | **1.32x** | **2.40x** |

Staged for 512 alone — a quarter of the arenas — ubatch 512 goes from 430.4 to
**572.9, 1.33x**. Two separate runs of the ladder agree to 0.54% at 128 tokens
and **0.01% at 2048**.

## L6c-1: the move, and why it is not a shared arena

`shaders/llm_move.comp` is three bindings and one loop: read floats from one
block's fp32 arena, write them into another block's arena as floats or as
halves. Bindings 1 and 2 are the *same* `VkBuffer` bound twice — a destination
is either an fp32 arena or an fp16 one, and a `narrow` push constant decides
which declaration the kernel writes through — so the two cases are one
pipeline shape rather than two shaders. A compute pipeline here owns its
descriptor set, so the host builds one per (source, destination) pair: **nine,
for the five blocks and the head**, on first use.

Each block exposes a `Port` for the two or three tensors that cross its
boundary and for nothing else, so what the graph may reach into is a short
list in each block's own file rather than an exported arena. A layer is then:

```
hc.Run(attn mixer)                          mixed, inject in the mixer's arena
move(sublayer.InPort()  <- hc.MixedPort())  f32 -> fp16, narrowed
sublayer.Run()
move(hc.BlockOutPort()  <- sublayer.OutPort())
hc.RunCombine(attn mixer)                   the residual, updated in place
```

**LLM.md's L6c said "one set of arenas", and this is not that.** Making the
five blocks bump-allocate out of one pair of buffers means splitting every
block's `alloc` into a pure layout pass and a materialisation — a descriptor
set needs its buffer before a pipeline exists, and a Vulkan buffer cannot grow
— which is a two-phase construction through five constructors and every caller
of them. The move costs what it costs instead, and L6c-2 says what that is.

## L6c-2: it is bit-exact, and it costs 2%

**Bit-exact.** Every figure L6b's gate prints came back identical to the last
place: `l_last` at 0.034 / 0.055 / 0.103 / 0.154 / 0.330 / 0.881% of scale at
8 through 48 layers, `result_norm` at 5.480e-01 rms, `result_output` at
3.144e-01, the same argmax out of the same top ten. That is the right outcome
and not an accident — `float16_t(v)` in SPIR-V and `safetensors.F32ToF16(v)`
in Go are the same round-to-nearest-even — and `TestMoveNarrow` is that claim
on its own: 37 rows of 2560 against the host narrowing **value for value**,
over halfway cases, subnormals and magnitudes past fp16's exact-integer range,
with the 27 pad rows zeroed and the A operand's pad *columns* untouched.

Those pad rows are not tidiness. The GEMM rungs have no bounds check, so the
rows between the prompt and the consumer's row block have to carry the
products of zeros, and a shorter run must not read what a longer one left
behind — which is why a destination `Port` states the row count it needs and
the move writes zeros up to it.

**And it costs 2%.** At ubatch 2048 the 197 moves a pass are **43.7 ms of
2175.8, 2.0%**, against the 22.7% of host plumbing they replace; at 128 tokens
they are 10.2 ms of 494.2, 2.1%. What is left on the host is 1.4-2.7%: the
embedding gather, the n-gram gather that D2 keeps off the device entirely, and
reading the logits back. **So a shared arena would buy that last 2% and
nothing else** — which is the measurement that turns "not built" into a
decision rather than a shortcut.

## L6c-3: the fixed host cost that was hiding behind the glue

With the moves in, the host column did something it should not: it **fell**
with the prompt length — 70.5 ms at 128 tokens, 61.3 at 512, 53.1 at 1024,
15.9 at 2048. Run the ladder backwards and the pattern follows the length and
not the position, so it is not a warm-up.

A cost that is flat per graph looks exactly like that. It was
`DeltaNetGPU.Reset`: the recurrent state is [128, 128, 48] f32 — **3.1 MB a
layer** — and a fresh sequence resets all 36 of them, spelled
`WriteFloat32At(off, make([]float32, StateSize()))`. That is **113 MB of Go
allocation a prefill**, zeroed by the runtime and then copied into a mapped
buffer that was about to be zeroed anyway. `vk.Buffer.ZeroFloat32At` clears the
mapping in place instead, and `HCGPU.UploadInit` writes `hc_init` — the
embedding repeated into four streams, 84 MB at ubatch 2048 — straight into the
residual arena rather than building the wide tensor on the host first.

| | 128 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|
| host, before | 70.5 ms | 61.3 | 53.1 | 15.9 |
| host, after | **11.1 ms** | **10.8** | **11.3** | **10.0** |
| tok/s | 226.8 → **259.0** | 533.8 → **570.4** | 731.1 → **761.8** | 932.2 → **941.3** |

Flat, as a fixed cost should be once it is only the logit read-back and the
`Resize` calls. It is worth **1.14x at 128 tokens and 1.01x at 2048**, and it
was invisible for as long as the glue around it was fifty times larger. The
same `WriteFloat32(make([]float32, n))` spelling appears in `zimage/dit`,
`zimage/vae`, `zimage/qwen` and `kokoro` — all of them at *allocation* time,
where it costs once and does not matter.

## L6c-4: where the 2048-token pass goes now

| | ms | % |
|---|---:|---:|
| moe | 1179.6 | 53.7 |
| deltanet | 446.4 | 20.3 |
| hyper-conn | 333.7 | 15.2 |
| attention | 137.3 | 6.3 |
| ple n-gram | 8.9 | 0.4 |
| lm head | 6.7 | 0.3 |
| move (device) | 43.7 | 2.0 |
| gather + host | 31.5 | 1.4 |

**96.6% of the graph is now the blocks** — 941.3 tok/s against the 974.9 they
would do with no plumbing at all. Against L2a-3's projection for this graph
with its glue fused (~1150) it is 82%, and the rest is inside the kernels:
L5b-7 already names the next 2.5x in the MoE as the unpack, and the DeltaNet's
fused input projection is 64% of that layer against the recurrence's 13%
(L3b-5).

**The short-prompt end is the honest weak spot.** At 128 tokens the MoE is
73% of the pass, because a routed expert bank costs what it costs to *read*
however few tokens are routed through it — the same reason llama.cpp's own
pp512 (313.62) is below its pp2048 (388.60). That is a decode problem wearing a
prefill hat, and it is L7's.
