<!-- LLM.md L2d. The PLE n-gram block: the trigram hash over a 320 M-row table,
     the block it feeds, and both on the device. Cited from llm/ple.go,
     llm/gpu_ple.go and shaders/llm_ple_*. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L2b](l2b-hyper-connections.md) · [L2c](l2c-hc-kernel.md) · phase 1

# L2d — the PLE n-gram block, and the quarter of the checkpoint that is a hash table

**Result: the trigram hash reproduces llama.cpp's gather bit-exactly — 112
rows of a 320 001 536-row table, chosen by a 64-bit mix of a token and its two
predecessors — and the block it feeds matches to 7.7e-08 rms on the CPU and
runs on the device in three dispatches.** It also closes the one hole L2c left
in the GPU tests: layer 1's hyper-connection mixer reads a residual this block
has written into, which is why that one mixer could not be compared against
the trace before.

| | vs llama.cpp |
|---|---:|
| `ple_embd` — the gather | **bit-exact, 0 ulp** |
| `ple_gate-1` | **7.7e-08** rms (CPU, `RefQ8`) |
| `ple_gated_value-1` | **2.1e-09** |
| `ple_conv_out-1` | **1.3e-08** |
| `hc_norm-1` through the block | **8.1e-07** on values to \|41.0\| |

## What it is

`per_layer_token_embd` is **28.80 GB of the 111.32 GB checkpoint** — a quarter
of it — and it is not a matrix. It is 320 001 536 rows of 160 IQ4_NL values,
and a token reads **sixteen** of them: 1.41 KB. That is D2's whole argument,
and llama.cpp agrees, creating the tensor `TENSOR_READ_LAZY` and gathering
from the mapping.

Which sixteen is a hash, computed on the host. For n = 2 and n = 3 the n-gram
is folded into one 64-bit value —

    mixed = ctx[0]*mult[0];  then  mixed ^= ctx[j]*mult[j]  for j = 1..n-1

— and each of the 8 heads of that group takes `mixed % vocab[h] + offset[h]`.
The sixteen vocabularies are near-primes around 20 000 0xx and their offsets
tile the table exactly (`TestPLEConfig` checks that they do), so one mixed
value becomes sixteen independent lookups in sixteen disjoint ranges. An EOS
anywhere in the window, or running off the start of the sequence, resets every
earlier position to EOS; a token's own EOS does not cut its own context.

Then the block, which adds into the wide residual twice:

    key    = W_key  * emb          [hc*nEmbd]   grouped-normed
    value  = W_val  * emb          [nEmbd]
    query  = grouped_norm(res)
    s      = sum_i(key*query) / sqrt(nEmbd)              per stream
    gate   = sigmoid(sgn(s) * sqrt(clamp(|s|, 1e-6, inf)))
    gated  = value (broadcast over the streams) * gate
    conv   = silu(depthwise causal conv of grouped_norm(gated))
    res   += gated + conv

Two details are worth naming because they are silent when wrong. The **signed
square root** before the sigmoid is not a normalisation: a dot product of two
2560-wide normalised vectors is too wide for a sigmoid to resolve, and `sqrt`
compresses it while `sgn` keeps the sign it would otherwise lose. And the
**convolution is dilated by the n-gram size** — four taps reaching 0, 3, 6 and
9 tokens back — which makes it the second recurrent thing in this model after
DeltaNet.

`TestPLEConvIsDilatedAndOrdered` is the control for both of the ways to
transcribe that convolution wrong, which llama.cpp's own comment warns about
(`ggml_conv_1d_dw` "is documented as unreliable", so it spells the operation
out as a sum of shifted copies). Losing the dilation is **1.2e5x** worse than
the real thing and reversing the tap order — the difference between a
convolution and a correlation — is **8.5e5x**. Both produce a tensor of the
right shape and the right magnitude.

## Bit-exact is the only tolerance a hash has

`ple_embd` is a lookup, so there is nothing to be nearly right about: if the
sixteen row indices are right the 2560 values are exact, and if one of them is
wrong the row is a real embedding belonging to some other n-gram. It came out
bit-exact once one bug was out of the way, and the bug is worth recording
because it is an API edge this repo will meet again: **`gguf.Dequantize`
appends**, so a scratch row has to be handed over as `buf[:0]`. Handed a
full-length buffer it silently returns the *first* row every time — 112
identical rows, all plausible, no error.

## On the device: three dispatches

`shaders/llm_ple_gate.comp`, `llm_ple_conv.comp`, and the plain arm
(`MODE 2`) of `llm_gemm.comp`, driven by `llm/gpu_ple.go`. llama.cpp spends
about thirty dispatches on the same block.

| ours | what llama.cpp spends on it |
|---|---|
| **kv** — one fused `[hc*nEmbd + nEmbd, nEmbd]` projection | 2 `MUL_MAT` |
| **gate** — both norms, the dot product, the signed sqrt, the sigmoid, the broadcast multiply and the conv norm | 3 `RMS_NORM`, 3 `MUL`, `SUM_ROWS`, `SCALE`, `ABS`, `CLAMP`, `SQRT`, `SGN`, `SIGMOID`, `REPEAT`, `MUL` |
| **conv** — four dilated taps, the SiLU and the residual add | 4 `CONT` + 4 `MUL` + 3 `ADD`, `SILU`, 2 `ADD` |

The key and value projections read the *same* gathered embedding, so they are
one matrix — the same argument that puts `inject` on the hyper-connection
block's down projection (L2c). The gate kernel is one workgroup per (token,
stream), which makes all three of its reductions workgroup-local.

Accuracy on the device, against the CPU reference in `Exact` mode, where the
only difference is the fp16 projection: `ple_gate` **5.1e-06** rms,
`ple_gated_value` 4.7e-07, `ple_conv_out` 3.0e-06, and the residual it leaves
behind 3.1e-06. Against llama.cpp all three are 1.5e-05, 1.4e-06 and 7.2e-06 —
i.e. still dominated by the reference's own int8 activations, as L2b found
everywhere a Q8_0 weight is involved.

## What it costs, and the two things that moved it

`results/l2d_ple.csv`, microseconds per dispatch, one block per forward pass:

| T | kv (BM=32) | kv (BM=64) | kv (BM=128) | gate | conv | block |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 423.8 | **390.1** | 437.2 | 26.3 | 14.1 | **478** |
| 128 | 755.4 | **419.6** | 436.6 | 42.8 | 24.5 | **504** |
| 256 | 1214.8 | 790.0 | **714.4** | 227.3 | 48.8 | **991** |
| 512 | 2540.6 | 1591.0 | **1379.4** | 487.4 | 367.0 | **2234** |
| 1024 | 7327.1 | 3304.2 | **2607.1** | 930.6 | 731.7 | **4269** |
| 2048 | 17209.7 | 7625.1 | **5198.5** | 1793.5 | 1464.8 | **8457** |

**1. The projection is bound by the weight, not the arithmetic, and BM is
worth 3.3x.** Its fused B is **65.5 MB of fp16 — twice the 32 MiB MALL** — so
a workgroup carrying BM token rows reads all of it M/BM times: 1.05 GB at 512
tokens with BM=32 and 262 MB with BM=128. The measured ratio at 2048 tokens is
17209/5198 = **3.3x** for a kernel doing identical arithmetic. At BM=128 and
512 tokens it is 24.3 TFLOP/s and ~190 GB/s of weight traffic, which is 80% of
the bus.

**2. Which grid axis is fast is worth 2.9-4.0x, for free.** Both elementwise
kernels were launched with the token as the fast axis, so the workgroups
resident at one moment covered one channel block of forty different tokens —
addresses 40 KB apart. Making the *channel block* the fast axis makes them
cover one token's whole 40 KB row instead: the convolution went from 1077 us
to 367 at 512 tokens (**2.9x**) and 5902 to 1465 at 2048 (**4.0x**), and the
gate from 550 to 483 and 2256 to 1793 (1.14-1.26x). Same work, same bytes,
same instruction count — only which addresses are outstanding together
changes. That is §5.1b's cross-wave traversal penalty, showing up in a kernel
that was written without thinking about it.

## Against llama.cpp, honestly

**On the lines that can be named we are slower: 1974 us against 2234, 0.88x.**
Those are the two projections (1346.8 + 115.8 us) and three `RMS_NORM(2560,4,512)`
at 170.5 each, from `results/l2a_prefill_ops.csv`. Split further, the
comparison goes both ways:

- **the projections: 1462.6 us against 1379.4, 1.06x for us** — while reading
  **twice the weight bytes**, because llama.cpp's are Q8_0 at 1.06 bytes a
  weight and ours are fp16 at 2. That is the clearest case yet for a Q8_0 B
  path: it would halve the traffic in a dispatch that is 80% weight-bound.
- **everything else: 854 us against 511** for the three norms alone — but our
  two remaining dispatches also absorb roughly *twenty-five* others, whose
  cost is inside the pooled `MUL` (479 dispatches at 173.8 us each), `ADD`
  (231 at 109.6), `REPEAT` (98 at 162.5) and `SIGMOID` (326 at 41.5) rows that
  the whole model shares.

A bottom-up estimate from those per-dispatch means — nine 10240-wide `MUL`s,
five `ADD`s, a `REPEAT`, five `CONT`s and the small scalar ops — puts
llama.cpp's *whole* PLE block near **4.5 ms** against our 2.2. That is an
estimate built on pooled means, not a measurement, and it is stated as one.

**None of it matters much, which is the real finding.** This block runs once,
at layer 1: 2.2 ms of a 1165 ms graph is **0.19%**, against llama.cpp's
nameable 0.17%. It was built to be *correct* and to be *on the device* —
because the alternative is a 21 MB round trip of the residual to the host in
the middle of the stack, out of an arena that reads at 0.2 GB/s — and the two
optimisations above were taken only because they were a compile flag and a
grid argument.

## How to reproduce

    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

    go test ./llm/ -v -run TestPLE          # the hash, the block, the controls, the GPU
    go run ./cmd/llm -ple -model $M -tokens 512
    go run ./cmd/llm -ple -model $M -csv results/l2d_ple.csv

Two full runs agree to **1.2% on every dispatch of 100 us or more** and 1.4%
on the block totals; the worst cell is 6.5% on a 23 us dispatch.

## What is not built

The **decode path for the convolution**. Its four taps reach 9 tokens back, so
a decode step needs a 9-token history per sequence — llama.cpp keeps one in a
row of its recurrent cache. Everything here is prefill from position zero,
where that history is zeros. L7 is where it has to exist, and DeltaNet (L3)
needs the same machinery for its own conv and its recurrent state, so it is
one piece of work rather than two.

The **image token path**: a batch of image embeddings has no token ids, and
llama.cpp hashes `ple.image_token_id` in their place. `PLEConfig` reads the
key; nothing uses it until the vision tower exists.
