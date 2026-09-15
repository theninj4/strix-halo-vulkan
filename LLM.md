# LLM — the qwen3.8-flash-next vertical

> **Current work from 2026-09-15.** `SPEECH.md` is finished-ish (74.2x real
> time, T7 open); `PIPELINE.md` (z-image) is parked at 14.26 s an image. Same
> rules as both: this file is **rewritten** each session rather than appended
> to, history goes to `TODO.md`, closed findings to `research/`. Stage numbers
> are **L0, L1, …**; `§N.M` still addresses `IDEAS.md`.

**Target**: `Qwen/Qwen3.8-Flash-Next` — 180 B params, 6 B active, "a preview of
the Qwen4 architecture" — generating text end to end in Go on Vulkan. The
third vertical, and 200x the parameters of the other two put together.

**Status: scoped; L0 complete — L0a, L0b, L0c and L0d all done.** The checkpoint is not downloaded. What
exists is the arithmetic below — all of it from the real `config.json` and
from the **actual tensor tables of `unsloth/…-UD-Q4_K_XL`**, read out of
35 MB of HTTP range requests rather than 111 GB of download
(`reference/gguf_inventory.py`) — plus the two measurements it was resting on.
Both came back negative, which is the useful kind: they removed constraints
rather than adding them.

> **L0a: the DRAM bus does not care how big the weight bank is.** 236.4 GB/s
> reading a **64 GiB** bank at random against 237.1 GB/s reading a 1 GiB one,
> `rand/seq` 1.00x in every cell at both of the model's real expert slab
> sizes. That *is* §0.4's 236 GB/s ceiling, not merely near it — no TLB cliff,
> no page-table cost, no penalty for scattering. So decode stays linear in
> bits/weight at the size a 180 B model needs, which is what every number
> below depends on. [Write-up](research/l0a-bank-range.md)
>
> **L0a again, at 80 GiB.** `-bankgib 80`: 23 buffers, 85.9 GB, every page
> written, **237 GB/s at every prefix** — past UD-Q4_K_XL's 82.52 GB resident
> core, so the largest bank this vertical would hold is now measured live.
>
> **L0b: neither does the memory type, and the heap sizes are fiction.** All
> eight types that can back a storage buffer read **236.0-237.4 GB/s** — 0.57%
> across 32 cells, heap 0 / heap 1 = **1.0004**. And the advertised heap sizes
> are not limits: type 3 reserved **105.0 GiB** against a heap RADV calls
> 83.79, type 2 **101.5 GiB** against 41.89, neither refused by the driver.
> **The ceiling is physical RAM.** [Write-up](research/5.1-memory-types.md) ·
> `results/bank.csv`

> **L0c: §2.2's unexplained 1.24x was locality in the scale plane, and it is
> gone.** One staging step reads 64 rows of the fp16 scale plane, and laid out
> `[rows][ldb/QBLOCK]` those scales are a row stride apart — 160 B on
> `gate_up` — so each 2-byte scale arrives alone in its cache line. A k-major
> plane makes them contiguous and takes `gate_up` at QBLOCK=32 from **4.71 ms
> to 3.72, 1.27x**, past row-major QBLOCK=128's 3.79. **The fine scale block
> accuracy wants now costs 1.0% instead of 14.2%.**
> [Write-up](research/l0c-scale-plane.md) · `results/moe.csv`

> **L0d: W4A8 is safe, and above ~5 bits/weight the activations are the
> floor.** Real weights and real activations from a Qwen3-4B forward pass, 14
> projections, float64 reference. int8-per-token activations cost **1.13x** the
> error of fp16 ones, so the 3x cliff between W4A8 and W4A16 is bought
> cheaply — but int8 activations *alone* contribute 2.87e-2 where an 8-bit
> weight contributes 3.4e-3, so **W8A8 is 99% activation error** and lands
> within 1.2x of W5A8 for twice the bytes and half the tok/s. Asymmetric Q4 is
> worth 5% at equal bits, not the 2x §7 predicted. And a bug fell out:
> `quantizeQ8`'s fp16 block scale goes subnormal under maxAbs 7.75e-3, worth
> **14x** on one real tensor in fourteen — Q4 is immune.
> [Write-up](research/l0d-quant-error.md)

> **What L0b changed.** This file previously ruled Q5 and Q6 out because they
> did not fit the 83.79 GiB device-local heap. **That argument was wrong** and
> is withdrawn. They are ruled out now on *throughput* — Q5 costs 20% of tok/s
> and Q6 33% — which is the better argument anyway, because it is the one that
> would still hold on a machine with more memory.

---

## The three findings that set the direction

**1. `llama.cpp` already implements this architecture, is built with Vulkan on
this box, and supports `qwen4exp`.** `/home/kube/repos/llama.cpp` at `cff184438`
has `src/models/qwen4exp.cpp` (1279 lines: DeltaNet, the QSA indexer, PLE
n-gram hashing, hyper-connections) and 50 built tools. That is a reference
implementation, an oracle (`llama-eval-callback` dumps every tensor), an
accuracy criterion (`llama-perplexity` — §7's "real perplexity", free) and a
performance baseline (`llama-bench`) all at once. **This vertical does not need
a Python reference dump**, which is what S1 and T1 spent their first session on.

**2. A quarter of UD-Q4_K_XL is a lookup table that should never touch the
GPU.** Of its 111.32 GB, **28.80 GB** is `per_layer_token_embd` — a 320 M-row
hash table read 16 rows of 160 values at a time, **1.41 KB per token**. Leave
it mmap'd on the host and the resident core is **82.52 GB**. L0b since
established that 111.32 GB would *fit* if we insisted, so this is no longer a
necessity — it is 26% of the machine's memory bought for 0.02% of its
bandwidth, which is reason enough.

**3. The stock quant is tuned for accuracy-per-byte, not for tok/s on a
bandwidth-bound APU, and it costs 1.7x.** Unsloth put every dense tensor at
Q8_0 and the routers at F32. On a machine where decode is pinned to the DRAM
bus that is the single most expensive choice available, because **76% of the
bytes read per token are dense** — each expert is read 10 times in 512, every
dense weight is read every time.

| | resident core | GB/token | **tok/s ceiling** |
|---|---:|---:|---:|
| UD-Q4_K_XL as shipped | 82.52 GB | 6.334 | **38.2** |
| our own bank, same inventory, ~4.25 bits on everything streamed | ~67 GB | ~3.6 | **~67** |

So: **run theirs first, then build ours.** Which is `TODO.md`'s phase order
anyway — a thing that runs, then a profile, then the optimisation.

---

## Direction

**Phase 1 — make it run on UD-Q4_K_XL.** Consume the GGUF natively: a GGUF
reader in Go, and dequant paths for the five formats it actually uses (Q4_K,
Q5_1, Q8_0, Q5_K, IQ4_NL). Correctness is checked tensor-for-tensor against
`llama-eval-callback`. The acceptance criterion is *it generates the same text
as llama.cpp*, and the number to beat is whatever `llama-bench` reports on the
same GPU.

**Phase 2 — our own bank.** Re-quantise to the repo's W4A8 layout (§1.1's
repack) at widths chosen for this bus rather than for a generic machine:
~4.25 bits on everything streamed, fp16 routers, the n-gram table left as
IQ4_NL on disk, and — L0c — **fp16 scales per 32 nibbles in a k-major plane**.
The two-level scale this file used to recommend (int8 per 32 under an fp16
per-256 super-scale) was half an accuracy argument and half a way to dodge
§2.2's 1.24x tax on fine scale blocks; L0c removed the tax, so the fine block
is essentially free at prefill and the extra machinery buys nothing there.
What survives of the int8-sub-scale argument is **bytes** — fp16-per-32 is
11.1% of the bank against 3.0% at per-128, and at *decode* bytes are the clock
— so it is a question for L0d, not a decision already taken. Target ~67 tok/s against phase 1's ~38. Source it either by
transcoding the GGUF (cheap, double-quantises) or from the 360 GB bf16
(clean — and **unsloth publish their imatrix**, `imatrix_unsloth.gguf`, 580 MB,
so the calibration is free).

**Phase 3 — the throughput levers that are worth more than the format.** The
MTP head ships as a separate 2.79 GB file; a draft step is ~12% of a full one,
so self-speculation is worth ~1.5-1.8x. Batching amortises the dense 76%
completely. Both dwarf the difference between 4 and 5 bits.

---

## The machine, measured

| limit | value | where from |
|---|---|---|
| DRAM bandwidth | **236 GB/s** peak; **242 GB/s** best real decode GEMV | §0, §1.7 |
| MALL | 32 MiB, 805 GB/s copy / 965 read | §0.4, §5.1b |
| WMMA fp16 | 55.5 TFLOP/s; best real kernel 39.0 (70%) | §0.1, §2.7 |
| Vulkan heap 1 (`DEVICE_LOCAL`) | advertised 83.79 GiB — **not a limit**: 105.0 GiB reserved | §5.1 |
| Vulkan heap 0 (host-visible) | advertised 41.89 GiB — 101.5 GiB reserved | §5.1 |
| memory type, for a GPU read | **irrelevant**, all 8 within 0.57% | §5.1 |
| the real capacity ceiling | **physical RAM**, 117.7 GiB, less the OS | §5.1 |
| visible VRAM carveout | 8 GiB — a *host*-read boundary only | `cmd/bus`, §5.1 |
| `maxBufferSize` / `maxMemoryAllocationSize` | **4 GiB − 4 B** | `vulkaninfo` |
| `maxDescriptorSetStorageBuffers` | 8 388 606 | `vulkaninfo` |
| GTT / system RAM | 117.7 GiB (`amdgpu.gttsize=126976` already set) | `/proc/cmdline` |
| disk free | 1.2 TB | `df` |

The 4 GiB buffer cap is the one structural consequence: a 60-80 GB bank is
**~100 buffers**, one per layer per projection (a layer's Q4 expert pair is
1.26 GB, comfortably under). Descriptor counts are not a constraint.

---

## The model

From `config.json` and the GGUF metadata, not assumed. `general.size_label` is
`512x56B`; `general.description` is "A Preview of the Qwen4 Architecture".

```
arch            qwen4exp   (Qwen4ExpForConditionalGeneration, multimodal)
layers          48 = 36 linear_attention + 12 full_attention (interval 4)
hidden          2560       residual stream 10240 (hyper-connections, hc_count 4, low_rank 320)
vocab           248320     context 262144 (1 M with YaRN)   tie_word_embeddings false
MoE             512 experts, 10 active + 1 shared, ffn 640, router in F32
full attn       24 Q heads x 256, 2 KV heads, q_proj is 12288 wide (query + gate)
  QSA indexer   4 heads x 128, 1 KV head, top_k 2048, compress_ratio 4
  rope          mrope interleaved, sections [11,11,10], dim_count 64 of 256, theta 1e7
linear attn     ssm inner 6144 (48 v-heads x 128), state 128, groups 16, dt_rank 48, conv k=4
  state dtype   float32 (mandated) -> 113 MB per sequence, context-independent
PLE n-gram      one table at layer 1, ngram_size 3, heads_per_ngram 8 -> 16 hash heads
                320 001 536 rows x 160, head vocabs ~20 000 0xx (prime, for hashing)
MTP             1 full-attention layer, shipped as a separate GGUF
vision          27 layers, hidden 1152 — a separate mmproj-*.gguf, 0.9 GB. Defer.
```

**`bench/modelshapes.go` is incomplete for this model** and should be corrected
at L1: it has no DeltaNet input projections at their real width, no
hyper-connections, no PLE, no indexer, and it gives `attn.q` as 6144 where the
checkpoint says 12288.

### What UD-Q4_K_XL actually contains

`reference/gguf_inventory.py s1.gguf s2.hdr s3.hdr s4.hdr` — 1224 tensors,
176.944 B params, **111.32 GB, 5.03 bits/weight**:

| group | params | GB | bits/w | read | quant mix (GB) |
|---|---:|---:|---:|---|---|
| moe_experts | 120.796 B | 77.02 | 5.10 | 10/512 | Q4_K 44.4, **Q5_1 27.1**, Q8_0 4.5, Q5_K 1.2 |
| **ngram_ple_table** | 51.200 B | **28.80** | 4.50 | **gather** | IQ4_NL 28.8 |
| deltanet | 2.087 B | 2.25 | 8.62 | every tok | Q8_0 |
| hyper_conn | 0.641 B | 0.70 | 8.68 | every tok | Q8_0 |
| lm_head | 0.636 B | 0.68 | 8.50 | every tok | Q8_0 |
| embed (lookup) | 0.636 B | 0.68 | 8.50 | gather | Q8_0 |
| full_attn | 0.598 B | 0.64 | 8.50 | every tok | Q8_0 |
| moe_router | 0.063 B | 0.25 | **32.00** | every tok | F32 |
| moe_shared | 0.236 B | 0.25 | 8.51 | every tok | Q8_0 |
| qsa_indexer | 0.020 B | 0.04 | 16.00 | every tok | BF16 |
| ple_proj | 0.033 B | 0.04 | 8.55 | every tok | Q8_0 |

Two things worth reading twice. **`ffn_down_exps` is Q5_1 on 43 of 48 layers
and Q8_0 on the other 5** while gate/up are Q4_K — that is unsloth's imatrix
telling them the down projection is the sensitive one, and it is 27 GB of the
77. And **the routers are F32**: 0.25 GB read every single token, 4% of the
decode budget for 0.06 B of parameters. Defensible (a mis-routed token is not
a rounded token) and worth re-pricing at fp16.

### Where the bytes go at decode

    dense, every token      4.830 GB   76%
    experts, 10 of 512      1.504 GB   24%
                            -------
    per token               6.334 GB   -> 38.2 tok/s at 242 GB/s

Capacity is an expert problem (97% of params). **Speed is a dense problem.**
The usual "keep attention at Q8, it's only 2% of the model" instinct is exactly
backwards on this machine: it costs 2% of memory and **a factor of 1.7 in
tok/s**. Conversely, going below 4 bits on the *experts* buys very little —
they are only a quarter of the traffic.

### Context is cheap here, which is the good news

12 full-attention layers x 2 KV heads x 256, and QSA caps what a step *reads*
at `top_k = 2048` however long the context is:

| context | KV fp16 | + indexer | DeltaNet state |
|---:|---:|---:|---:|
| 32 768 | 0.81 GB | 0.05 | 113 MB (constant) |
| 131 072 | 3.22 GB | 0.20 | 113 MB |
| 262 144 | 6.44 GB | 0.40 | 113 MB |

Before L0b this section ended by saying UD-Q4_K_XL could not hold full context
because 82.52 GB left only 7.4 GB of an 89.97 GB heap. **That was wrong** — the
heap figure is advisory and 105 GiB is reservable — so 82.52 GB of weights plus
6.8 GB of KV is comfortable, and even keeping the n-gram table resident
(111.32 GB total) is within reach. Capacity has stopped being the binding
constraint on this model at any format below Q8. Bandwidth is the whole story.

### Prefill behaves differently from decode

Decode is pinned to the bus; prefill is not. §2.2 measured a Q4 MoE block at
2048 tokens at **13.37 ms against a 5.5-6.0 ms memory floor** — 2.2x above it,
bound by the dequant path and by the 40-rows-per-expert tile (§3.4 finding 3:
**14% of the WMMA ceiling**). So:

- Wider formats cost prefill **sub-linearly** — roughly 1.3x for Q8 where
  decode pays 2x. The phase-1 Q8_0 dense tensors hurt decode far more than they
  hurt prefill.
- **Chunk much larger than 2048.** The expert bank is read once per chunk
  either way; at 8192 tokens each expert sees 160 rows instead of 40.
- Adding the measured pieces up: ~2000 tok/s prefill at a 2048 chunk, about 4x
  above what the bytes alone imply.

---

## What exists to build on

| piece | state |
|---|---|
| `zimage/qwen/` | **a working Qwen3 transformer on the GPU in Go** — RMSNorm, RoPE, GQA, SwiGLU, four shared arenas. The skeleton for L2. |
| `shaders/gemv_w4a8.comp` | decode GEMV at **99-103% of the bus** (§1.1, §1.7), plus 25 grouped / M-blocked / N-blocked MoE builds (§1.8-§1.12). |
| `shaders/gemm_wmma_q4.comp` | Q4 prefill GEMM, **2.10x fp16** on a MoE block (§2.2). |
| `shaders/moe_route/gather/combine` | top-k routing and the permutation around it. |
| `shaders/qwen_attn_wmma_*`, `qwen_rope.comp` | causal WMMA attention at 38.8 TFLOP/s (§3.3); NeoX rotary. |
| `rmsnorm_shared.comp` | 95% of the bus. |
| `shaders/bank_gather.comp` + the `bank` family | L0a's probe: fixed bytes read from a bank of swept size. The instrument for L0b too. |
| `safetensors/` | mmap reader, sharded — the model for a GGUF reader. |
| `zimage/tokenizer/` | Qwen BPE. New vocab and multimodal specials, same machinery. |
| `vk/engine.go` | dispatch, arenas, per-pipeline wave size. |
| `bench/` `moe`, `shapes` families | already sweep this model's expert bank and decode shapes. |

## What has to be built

1. **Gated DeltaNet — 36 of the 48 layers.** §3.6, unbuilt, "effort: high".
   Chunked linear attention, fp32 recurrent state, depthwise conv1d k=4 over
   10240 channels, `A_log`/`dt_bias` gating, sigmoid output gate. The largest
   new kernel in the project's history, and three quarters of the model.
2. **QSA sparse attention** — indexer, top-2048 selection, gathered attention.
   Not in `IDEAS.md` at all; it needs a new section.
3. **A GGUF reader + K-quant dequant** (Q4_K, Q5_1, Q8_0, Q5_K, IQ4_NL).
   llama.cpp's Vulkan shaders are the crib.
4. **Hyper-connections** — 4-branch gated residual at width 10240, every layer.
5. **PLE n-gram** — trigram hashing into 16 heads over a 320 M-row mmap'd
   table, `layer_multipliers`, conv1d k=4, key/query/value projections.
6. **KV cache**, mrope (interleaved, sections [11,11,10], 64 of 256 dims), and
   a decode loop.
7. **The re-quantiser** (phase 2) and **MTP speculative decoding** (phase 3).

---

## Task list

### L0 — answer the format question by measurement, not arithmetic

- [x] **L0.0** Read the real inventory without downloading the model.
      `reference/gguf_inventory.py`, 35 MB of range requests. **Done** — the
      tables above.
- [x] **L0a** *Does the bus survive a 64-82 GB working set?* **Done — it does
      not notice.** Every bandwidth number in this repo had been measured at
      ≤256 MB (`gemv_cold`) or ≤4 GB (`moe`); this model is ~19 buffers with a
      10-of-512 random gather on top, a TLB and page-table question the suite
      had never asked (§5.2). New `bank` family and
      `shaders/bank_gather.comp`: 1 GiB read per step, held constant, drawn
      from a bank swept 1 → 64 GiB in three orders (`seq` control, `spread`,
      `rand`) at the model's real expert slab sizes. **Flat at 236-237 GB/s
      everywhere, `rand/seq` 1.00x**, two-run median ratio 1.0002 (p10-p90
      0.997-1.004). Two side results: the 4 MiB control slab's 1.7% deficit is
      workgroup tail rather than memory, and a bandwidth-bound read draws
      **85 W against ALU work's 118-142**, so speculation and batching have
      ~50 W of package budget to spend.
      [Write-up](research/l0a-bank-range.md).
- [x] **L0b** *Is heap 0 as fast as heap 1 for GPU reads?* **Done — no type
      or heap is faster than any other, and the heap sizes turned out not to
      be limits.** §5.1, never run before. New `vk.Device.MemoryTypes()` /
      `NewBufferOfType()` over two shim entry points, and two more arms on the
      `bank` family. All eight buffer-compatible types read **236.0-237.4
      GB/s** (0.57% spread, heap0/heap1 = 1.0004); `HOST_CACHED` and
      `DEVICE_UNCACHED` are worth nothing either way, because the property
      bits describe how the *host* sees the pages and a GPU read never goes
      through the CPU's caches. The capacity probe then reserved **105.0 GiB
      from a heap advertised as 83.79** and 101.5 from one advertised as
      41.89, stopped only by its own host-memory floor. Two side findings: the
      8 GiB carveout behind `cmd/bus`'s "the first heap runs out at 8 GB" is a
      host-read boundary the GPU does not notice, and **freed GPU memory
      returns to the driver's TTM pool, not to `MemAvailable`**, so a capacity
      probe gets one memory type per process.
      [Write-up](research/5.1-memory-types.md).
- [x] **L0c** *The blocked scale plane.* **Done, and unlike L0a and L0b it is
      a positive: 1.27x, and it changes the format.** `SCALE_LAYOUT` on
      `gemm_wmma_q4.comp` with a matching host packer, three layouts x two
      scale-block sizes at §2.2's winning geometry. k-major qb32 beats
      row-major qb128 on `gate_up` (3.72 vs 3.79 ms), so **block-32
      granularity costs 1.0% of the two matmuls instead of 14.2%**. The
      mechanism is the plane's row stride and the two layers are the control:
      identical nominal scale overhead (11.1% of the bank), strides of 160 B
      and 40 B, recoveries of 1.27x and 1.02x. Tile-blocking beyond k-major is
      worth 0.1%, so the layout stays a property of the weights rather than of
      the tile geometry — which matters, because layout 2 would mean repacking
      60 GB whenever `BN` changed. Best cell is **1.169x** over what §2.2
      shipped. [Write-up](research/l0c-scale-plane.md).
- [x] **L0d** *Per-format error metrics* (§7). **Done, and it settles the
      format.** New `cmd/quanterr` + `bench/quanterr.go`: 14 real projections
      from a Qwen3-4B forward pass against a float64 reference, weight
      reconstruction error carried separately from output error. Not
      `Qwen3-Embedding-0.6B` as this entry originally said — Z-Image's Qwen3-4B
      text encoder is hidden **2560**, the target's own width, where the
      embedding model is 1024. **W4A8 costs 1.13x W4A16** (worst site 1.6x, on
      `down`); **the activation error is the floor above ~5 bits**; asymmetric
      is worth 5% at equal bits; per-token activation scales beat per-tensor by
      1.97x and are free. Outliers are severe — channel spread 4172 and
      kurtosis 87 187 at layer 1's `down` — and survivable with a per-row
      scale. Found a defect in `quantizeQ8` on the way (see below).
      [Write-up](research/l0d-quant-error.md).

### L1 — get the checkpoint, and a baseline

- [ ] Download `UD-Q4_K_XL` (4 shards, 111.32 GB) and
      `mtp-…-Q4_K_M.gguf` (2.79 GB). ~114 GB of 1.2 TB free.
- [ ] `llama-bench` and `llama-cli` on it: **the number this vertical has to
      beat**, prefill and decode, on the same Vulkan device.
- [ ] `llama-perplexity` on a fixed corpus: the accuracy reference for
      everything phase 2 does.
- [ ] Go GGUF reader (mmap, sharded, the split KV convention), checked
      tensor-for-tensor against `reference/gguf_inventory.py`.
- [ ] Tokenizer: the 248320 vocab and the multimodal specials, exact against
      `llama-tokenize`.
- [ ] Correct `bench/modelshapes.go` to the real inventory and re-run `shapes`.

### L2 — the dense skeleton

- [ ] Embeddings, PLE gather (host, mmap'd), hyper-connection mix/combine,
      RMSNorm, one full-attention layer with mrope.
- [ ] Gate: layer 3's output matches `llama-eval-callback` to fp16 tolerance.

### L3 — Gated DeltaNet  *(the big one)*

- [ ] CPU reference first, fp32 state, against the dump.
- [ ] The chunked GPU kernel (§3.6). Find out whether it is matmul-bound
      (coopmat) or serialised on the state update (careful chunking).
- [ ] Gate: layer 0 matches; state exact in fp32 across a chunk boundary.

### L4 — QSA

- [ ] Indexer, top-2048 selection, gathered attention.
- [ ] Gate: layer 3 matches at 4 k context, where selection actually bites.

### L5 — the MoE block

- [ ] Router (F32), grouped Q4 GEMV for decode (§1.8-§1.12), Q4 WMMA GEMM for
      prefill (§2.2), gather/combine.
- [ ] Gate: one block matches; GB/s reported against L0a.

### L6 — the whole stack, prefill only

- [ ] All 48 layers resident, the n-gram table mmap'd, ~100 buffers.
- [ ] Gate: logits match llama.cpp; prefill tok/s at chunk 2048 and 8192.

### L7 — decode

- [ ] KV cache, ring buffer, sampling, the generation loop.
- [ ] Gate: it generates the same text as llama.cpp; tok/s against the 38.2
      ceiling and against L1's baseline.

### L8 — phase 2, our own bank

- [ ] Choose per-tensor widths from L0d and L1's perplexity.
- [ ] Re-quantise (transcode from the GGUF, or from bf16 with unsloth's
      published imatrix) into the §1.1 W4A8 layout with an L0c scale plane.
- [ ] Gate: ~67 tok/s, ≤67 GB resident, perplexity within a stated delta of
      phase 1.

### L9 — phase 3, and shipping

- [ ] MTP speculative decoding (the separate GGUF). Target 1.5-1.8x.
- [ ] Batching, then the HTTP API `GOALS.md` asks for.
- [ ] Vision tower, if wanted.

---

## How to run things

    # the inventory, without downloading the model (35 MB)
    B=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-Q4_K_XL
    N=Qwen3.8-Flash-Next-UD-Q4_K_XL
    curl -sL -o s1.gguf "$B/$N-00001-of-00004.gguf"
    for i in 2 3 4; do curl -sL -r 0-8388607 -o s$i.hdr "$B/$N-0000$i-of-00004.gguf"; done
    python3 reference/gguf_inventory.py s1.gguf s2.hdr s3.hdr s4.hdr

    # the reference implementation, the oracle, the baseline
    L=/home/kube/repos/llama.cpp
    less $L/src/models/qwen4exp.cpp          # 1279 lines, the whole architecture
    $L/build/bin/llama-bench -m …UD-Q4_K_XL-00001-of-00004.gguf
    $L/build/bin/llama-eval-callback -m … -p 'hello' -n 1   # per-tensor dumps
    $L/build/bin/llama-perplexity -m … -f wiki.test.raw

    # ours, once it exists
    go run ./cmd/llm -gpu -prompt '…'

---

## Decisions

| # | decision | why |
|---|---|---|
| D1 | **Consume UD-Q4_K_XL first, don't quantise from bf16.** | 114 GB instead of 360, an imatrix-calibrated checkpoint, and a bit-exact oracle. Phase order: run it, profile it, then optimise (`TODO.md`). |
| D2 | **The n-gram table lives off-heap, mmap'd.** | 28.80 GB of capacity for 1.41 KB/token of bandwidth — 26% of the machine for 0.02% of its bus. L0b showed it would *fit* on the GPU; that does not make it worth a quarter of the memory. |
| D3 | **Target ~4.25 bits on everything *streamed*, not just the experts.** | 76% of decode bytes are dense. Bits spent there cost ~nothing in memory and everything in tok/s. L0d adds the accuracy half: above ~5 bits the int8 activations are the floor anyway, so the bits would not buy what they appear to. |
| D4 | **Do not go below 4 bits on the experts.** | They are 24% of the traffic; Q3 buys ~11% for a real accuracy hit. |
| D6 | **W4A8, with per-token activation scales.** | L0d: int8-per-token costs **1.13x** the error of fp16 activations, against a 3x throughput cliff (§1.1's float-unpack Q4 GEMV reaches 31% of the bus). Per-*tensor* is 1.97x worse on the activation error and per-token is free in the RMSNorm epilogue (§3.1). |
| D7 | **Symmetric Q4, not asymmetric.** | L0d: 1.043-1.053x at equal bits per weight, where §7 predicted 2x. Free at decode (§1.1's block sum already exists) but an extra fp32-epilogue term in the GEMM path (§2.2 finding 9). Revisit only if L9's perplexity asks for it. |
| D8 | **fp16 scales per 32 nibbles, in a k-major plane.** | L0c made the fine block cost 1.0% instead of 14.2% at prefill, so accuracy gets the granularity it wants for nothing. The remaining cost is decode bytes — 11.1% of the bank against 3.0% at per-128 — which is the one axis still worth trading if tok/s falls short. |
| D5 | **`llama.cpp` is the oracle, not a Python dump.** | It is built, it is Vulkan, it supports `qwen4exp`, and it gives correctness, accuracy and a baseline from one binary. |

## Open questions

- ~~**L0a's answer.**~~ ~~**L0b's.**~~ Both settled, both negative, and the
  residency gap between them is closed to 80 GiB: `-bankgib 80` holds **23
  buffers, 85.9 GB, every page written, 237 GB/s at random**, past
  UD-Q4_K_XL's 82.52 GB core. What remains taken on trust is only the band
  between 80 GiB touched and 105 GiB reserved.
- **Does 111.32 GB — the whole GGUF, n-gram table included — hold live?** It
  is reservable and 80 GiB of it is demonstrably residable. If the whole thing
  is, D2 becomes purely an efficiency choice and the host-side PLE gather can
  be deferred past L2, which would simplify the first working version.
- ~~**W4A8 vs W4A16**~~ Settled by L0d: 1.13x mean, 1.6x worst. Take W4A8.
  What is left of it is `down_proj` specifically, whose SwiGLU input has no
  norm in front of it and carries the outliers. Finer activation groups on
  that one input, or a rotation, are the unmeasured mitigations; running it at
  W4A16 is not one, because 13% of the decode budget at 31% of the bus is
  +26% of decode time.
- **Fix `quantizeQ8`'s subnormal scale** — two lines (floor it at fp16's
  smallest normal), but it changes the CPU reference every W8A8 correctness
  check is built from, so it wants its own change with those re-run. Not
  urgent for a Q4 engine; it becomes urgent if phase 2 ever transcodes
  UD-Q4_K_XL's Q8_0 dense tensors through it.
- **Does the scale plane's layout matter at *decode* too?** L0c is all GEMM.
  §1.8 measured the grouped GEMV paying 1.045-1.116x on the same axis and put
  it down to bytes alone — that kernel reads its scales the way it reads its
  weights, so it should not care about the plane — but it has not been tried
  with a k-major plane, and decode is where scale bytes are charged against
  the bus.
- **How much of the Q8_0 dense allocation is actually needed?** Unsloth chose
  it with an imatrix and 45 calibration chunks. D3 assumes ~4.25 bits is
  enough; L1's perplexity plus L8's is the only honest test.
- **`§3.4 finding 4` says prefill is "98% memory-bound by weight bytes";
  §2.2's direct measurement of a Q4 block says 2.2x above its memory floor.**
  They disagree. §2.2 measured a real block and §3.4 modelled one, so lean on
  §2.2 — but reconcile before sizing prefill.
- **Does `llama.cpp` actually keep 82 GB device-local, or is it mmap-and-page?**
  It changes what L1's baseline means. Check with `radeontop`/heap budget while
  `llama-bench` runs.
- **How many GGUF formats does phase 1 really need?** Five appear. Q5_1 alone
  is 27 GB of the experts; if transcoding just that one to Q4_K-shaped data at
  load time is acceptable, phase 1 gets simpler and smaller.
- **Batch and speculation.** At batch 4 the dense 76% amortises completely.
  Worth knowing whether the API will ever serve more than one stream before
  optimising the batch-1 path to death.
