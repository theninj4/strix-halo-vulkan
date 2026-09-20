<!-- LLM.md L7a. The attention layer over a persistent KV cache: the cell
     geometry, the incremental indexer, the two borrowed push fields, the
     chunk-split gate, and the fp16 round trip that had been dead code. Cited
     from llm/gpu_attn.go, shaders/llm_attn_*.comp and
     llm/gpu_attn_cache_test.go. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L2e](l2e-attention.md) · [L2f](l2f-attention-gpu.md) · [L4b](l4b-qsa-gpu.md) · [L7b](l7b-sequence.md) · phase 1

# L7a — the KV cache: a chunk split, bit for bit

**Result: the full-attention layer runs over a persistent cache, and splitting
a 4096-token prompt into chunks — 512, 64, 7, or a prompt and then one token
at a time — produces output identical to the last place. Not to a tolerance:
every one of 10 485 760 floats.**

Everything above this stage ran a layer as a *fresh sequence*. Token `t` was
cell `t`, its position was `t`, and the key and value planes were the batch's
own padded token rows — so the batch and the cache were the same object and
the distinction never had to be made. Decode is the opposite: one token at a
time, at cell `past`, attending over every cell before it. The only way that
is the same model is if the split changes nothing, and the only honest test of
that is an equality.

## L7a-1: what is a batch and what is a cache

The layer's tensors divide cleanly once the question is asked, and the
division is not the one the arenas had:

| tensor | was | is |
|---|---|---|
| the fused projection, the query plane, the context | this batch's padded token rows | unchanged |
| the **key** and **value** planes | this batch's padded token rows | **nKV cells, one plane a layer** |
| the indexer's **pooled** key | nBlocks rows, shared by every layer | **nBlocks rows, one plane a layer** |
| the indexer's **raw** key | read out of the projection's own output | **nKV cells, one plane a layer** |

So the query's geometry is `plane` — the batch padded up to the widest tile —
and the key's is `nKV`, and the two no longer coincide. `llm_attn_wmma.comp`
is told both: `qTiles = plane/16` for the query fragments it holds in
registers, `kvTiles = nKV/16` for the key and value fragments it streams. The
key loop then runs over `SEQ_PAST + tokens` cells and every causal comparison
is between a **cell** and a **position** — row `i` of query block `q0` is cell
`SEQ_PAST + q0 + i` — where before both sides of the comparison were batch
indices that happened to be positions too.

`llm_attn_pack.comp` scatters by cell rather than copying whole tiles, because
`SEQ_PAST` need not be a multiple of sixteen and a batch boundary is allowed
to land inside a fragment. For two kv heads that turns a 512-byte contiguous
store into sixteen-wide runs, which is not where this kernel's time is.

**The cache is 2.31 KB a cell a layer** — 1 KB of key, 1 KB of value, 256 B of
raw indexer key, 64 B of pooled block — so twelve layers are 27.7 KB a cell:
57 MB at 2048 cells, 906 MB at 32768. It lives in the same fp16 activation
arena as the batch tensors, which costs nothing and has one structural
consequence: `maxStorageBufferRange` is 4 GiB - 4 here, so a single buffer
caps this at about 148k cells. Past that the cache wants L6a's
array-of-buffers, one a layer.

## L7a-2: the indexer's raw key needs a cache, and it is a plane of the pack

The pooled indexer key averages `ratio = 4` consecutive cells. At decode those
four cells arrive in four *different* batches, so pooling cannot read the
current projection's output the way L2e's kernel did — three quarters of what
it needs is not there.

The raw key therefore gets a cache of its own, 128 halves a cell, and it is
written by **`llm_attn_pack.comp` as one more plane of its grid** rather than
by a dispatch of its own. The pack's grid is already one workgroup per (token
tile, plane) with the plane saying which head this is; plane
`heads + 2*kvHeads` is not a head at all and copies the indexer's raw key
straight out of the fused projection's fp32 row. It has to be the pack and not
the indexer kernel because `llm_attn_idx.comp` *reads* it, and two dispatches
are how that ordering is guaranteed.

That move is also what made L2e's fp16 round trip real for the first time —
see L7a-5.

## L7a-3: a complete block is never pooled twice

A cell's contents never change, so pooled block `b` is final the moment all
`ratio` of its cells exist. A continuing run rebuilds only the blocks its own
tokens completed:

```
blockLo = past == 0 ? 0        : past / ratio
blockHi = past == 0 ? nBlocks  : (past + T) / ratio
```

and the host sizes the grid from the same two expressions the shader derives,
so there is one definition and not two. The arithmetic is exact rather than
conservative: block `b` is incomplete before the run iff `(b+1)*ratio > past`,
which is `b >= floor(past/ratio)` whether or not `past` divides `ratio`, and
complete after it iff `(b+1)*ratio <= past+T`, which is `b < floor((past+T)/ratio)`.

A **fresh** sequence is the one case that writes the whole table, because the
blocks past the last complete one pool cell 0 `ratio` times (L2e's
`blk_cells` is zero-filled) and have to be filled once before they can be left
alone. After that they carry over untouched, and when they become real the
range above catches them.

At a 4096-cell cache that is 1024 workgroups on the first run and **one** on a
decode step that happens to complete a block, against 1024 every step if the
table were rebuilt.

`TestAttnGPUCacheKeepsPooledBlocks` asserts the range directly at six
(past, tokens) pairs, and then checks every block of the table against the
independent predicate "completed by this run and not before it" — so the range
is a claim about the model rather than a transcription of the code.

## L7a-4: two fields the push block did not have

The push block has been full since L5b: 64 uints is 256 bytes and that is this
device's whole `maxPushConstantsSize`. A continuing layer knows two things a
fresh one does not — where it starts, and where the raw indexer key cache is —
and there is no 65th field.

So they ride fields this layer does not use, the same arrangement `moeUsed`
has documented since L6a, with the mapping in `llm_common.glsl` and the
kernels spelling only the macro:

```
lowRank   SEQ_PAST     cells already in the cache
loOff     ATTN_IDXRAW  fp16 [nKV][idxDim], the raw indexer key per cell
```

`kOff`, `vOff` and `idxKOff` are unchanged as *fields* and changed as
*tensors*: they now address one staged layer's cache rather than a shared
arena.

## L7a-5: `float(float16_t(x))` is dead code on this driver, and L2e's round trip was never running

L2e established that the reference's indexer key cache is fp16 — the keys are
rounded to halves between being projected and being pooled — and priced it at
**2072x** on the pooled key, 3.49e-04 rms to 1.69e-07, on the **CPU**
reference. The GPU kernel expressed the same thing as
`float(float16_t(act[...]))` inside the pooling loop.

Moving the pooling to read halves out of memory changed the answer, which it
should not have if the cast was doing anything. Three builds, checksummed over
the whole pooled tensor at both fixture lengths:

| pooling reads | 4096-token crc | 7-token crc |
|---|---|---|
| **A** the fp16 cell cache | `576a9cf0` | `6fa5fcdd` |
| **B** `float(float16_t(f32))`, inline — what L2e shipped | `e9bc447c` | `32865a84` |
| **C** the f32 value, no cast at all | `e9bc447c` | `32865a84` |

**B and C are bit-identical at both lengths.** The inline cast was not
rounding anything. It is not `spirv-opt` — `spirv-dis` finds
`OpFConvert %half` immediately followed by `OpFConvert %float` in the
optimised module — so it is RADV's NIR, whose algebraic table folds
`f2f32(f2f16(a))` to `a` under the inexact rule that a compute shader without
`NoContraction` permits.

The rule this earns: **a value that models a memory format has to go through
memory.** A register round trip through a narrower type is a hint, and this
driver is entitled to ignore it.

Keeping the cache (A) is the faithful choice — it is what the reference does,
and L2e's CPU measurement is what says so — and it is the only option once the
cells span batches anyway. It moves two intermediates and no conclusion:

| tensor, vs llama.cpp | B (register) | A (memory) |
|---|---:|---:|
| `indexer_k-3`, 7 tokens | 3.058e-04 | 3.552e-04 |
| `indexer_k-3`, 4096 tokens | 6.122e-04 | 6.227e-04 |
| `indexer_score-3`, 7 tokens | 5.569e-03 | 1.110e-02 |
| `indexer_score-3`, 4096 tokens | 1.929e-02 | 1.947e-02 |
| **`attn_output-3`, 7 tokens** | **1.144e-03** | **1.144e-03** |
| **`attn_output-3`, 4096 tokens** | **9.872e-04** | **9.872e-04** |

The layer's output does not move at either length, `l_last-0` through
`l_last-3` in `TestGraphPrefix` are unchanged to four digits, and
`TestGraphLogits` still returns llama.cpp's argmax out of llama.cpp's ten. The
pooled key sits at 3-6e-04 because the *raw* key already does — it is 1.65e-04
rms out of our own Q8_0 dequantisation — so which way the pooling rounds is
below the noise the operands arrive with. On the CPU, where the raw key is
exact, it was worth 2072x.

## L7a-6: the gate

`TestAttnGPUCacheIsAChunkSplit`, on the 4k fixture — 4096 tokens, a 4096-cell
cache, the QSA selection engaged — with the kernel rungs pinned so that what
is being compared is one kernel against itself and not two tilings:

| schedule | chunks | `attn_output` against the single pass |
|---|---:|---|
| 512 at a time | 8 | identical to the last place |
| 64 at a time, inside a fragment tile | 64 | identical |
| 7 at a time, crossing every block boundary | 586 | identical |
| 4063, then one token at a time | 34 | identical |

Bit-equality is the right criterion here rather than a tolerance, and it is
reachable rather than lucky: a chunked run recomputes nothing. The cells a
previous chunk wrote are read back as they were written, the pooled blocks are
carried, and the flash-attention key loop walks the same absolute blocks in
the same order — so every dispatch is the same arithmetic on the same values.
A tolerance would have passed a cache that was off by a position.

The last schedule is the one that matters, and it is a control as well as a
gate: a chunk of one has nothing of its own to read. Its query attends to a
cache it did not write, its indexer scores blocks it did not pool, and the 33
of them at the end of a 4096-token prompt are exactly what decode does.

## What this leaves

The layer continues a sequence. The *model* does not yet: the PLE block's
convolution and every DeltaNet layer's are the other two things in this
architecture that read behind the run, and the DeltaNet's window turned out to
be shared across all 36 layers. That is [L7b](l7b-sequence.md).
