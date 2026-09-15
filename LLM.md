# LLM — the qwen3.8-flash-next vertical

> **Current work from 2026-09-15.** `SPEECH.md` is finished-ish (74.2x real
> time, T7 open); `PIPELINE.md` (z-image) is parked at 14.26 s an image. Same
> rules as both: this file is **rewritten** each session rather than appended
> to, history goes to `TODO.md`, closed findings to `research/`. Stage numbers
> are **L0, L1, …**; `§N.M` still addresses `IDEAS.md`.

**Target**: `Qwen/Qwen3.8-Flash-Next` — 180 B params, 6 B active, "a preview of
the Qwen4 architecture" — generating text end to end in Go on Vulkan. The
third vertical, and 200x the parameters of the other two put together.

**Status: L0 and L1 complete. The checkpoint is downloaded, llama.cpp runs it,
and the Go side reads it.** 114 GB in 18 minutes;
`models/Qwen3.8-Flash-Next-GGUF/` holds the four `UD-Q4_K_XL` shards and the
2.79 GB MTP head. **L2 — the dense skeleton — is next.**

## The number to beat

`llama-bench`, build `cff184438`, Vulkan on RADV STRIX_HALO, the model as
shipped, `-ngl -1`, mmap on, two repetitions, nothing else on the GPU:

| test | tok/s | against |
|---|---:|---|
| pp512 | 313.62 ± 1.74 | |
| **pp2048** | **388.60 ± 1.89** | the ~2000 this file predicted — **5x out** |
| pp8192 | 392.95 ± 1.04 | prefill plateaus, it does not scale with the chunk |
| tg32 | 25.12 ± 0.03 | |
| **tg128** | **25.15 ± 0.02** | **65.8%** of the 242 GB/s bus |

A real generation agrees: `llama-completion -n 64 --temp 0` reports 24.66
tok/s and coherent text. Reproducibility across two separate runs is 0.5% on
pp512 and 0.8% on tg128. And the accuracy reference, on the same checkpoint:
**PPL = 4.0340 ± 0.02283** over wikitext-2's 145 chunks at n_ctx 2048 — the
figure L8's re-quantisation has to stay near, at the same corpus, context and
chunking. [Write-up](research/l1-baseline.md)

**So the target moves.** Phase 1's job is the 38.2 tok/s the bytes allow, not
llama.cpp's 25.15 — decode leaves a third of the bus unused in the reference
implementation. Phase 2's ~67 is then **2.7x the reference**, not 1.8x.

And **prefill is the surprise**. §3.4 and §2.2 were added up into "~2000 tok/s
at a 2048 chunk"; the reference gets 388.60 and does not improve with a bigger
chunk. Either this architecture has a great deal of prefill that is not matmul
— 36 DeltaNet layers, the QSA indexer, 194 hyper-connection projections a
token — or llama.cpp's Vulkan path leaves 5x on the table. Which one it is
decides whether L6 is an afternoon or a fortnight, and **it is the first thing
L2 should find out**, because both candidates are measurable before the whole
model runs.

## What L0 established

> **L0a: the DRAM bus does not care how big the weight bank is.** 236.4 GB/s
> reading a **64 GiB** bank at random against 237.1 GB/s reading a 1 GiB one,
> `rand/seq` 1.00x in every cell at both of the model's real expert slab
> sizes. That *is* §0.4's 236 GB/s ceiling, not merely near it — no TLB cliff,
> no page-table cost, no penalty for scattering. So decode stays linear in
> bits/weight at the size a 180 B model needs, which is what every number
> below depends on. At `-bankgib 80`: 23 buffers, 85.9 GB, every page written,
> **237 GB/s at every prefix** — past UD-Q4_K_XL's 82.52 GB resident core.
> [Write-up](research/l0a-bank-range.md)
>
> **L0b: neither does the memory type, and the heap sizes are fiction.** All
> eight types that can back a storage buffer read **236.0-237.4 GB/s** — 0.57%
> across 32 cells, heap 0 / heap 1 = **1.0004**. Type 3 reserved **105.0 GiB**
> against a heap RADV calls 83.79. **The ceiling is physical RAM.**
> [Write-up](research/5.1-memory-types.md) · `results/bank.csv`
>
> **L0c: §2.2's unexplained 1.24x was locality in the scale plane, and it is
> gone.** A k-major scale plane takes `gate_up` at QBLOCK=32 from **4.71 ms to
> 3.72, 1.27x**, past row-major QBLOCK=128's 3.79. **The fine scale block
> accuracy wants now costs 1.0% instead of 14.2%.**
> [Write-up](research/l0c-scale-plane.md) · `results/moe.csv`
>
> **L0d: W4A8 is safe, and above ~5 bits/weight the activations are the
> floor.** int8-per-token activations cost **1.13x** the error of fp16 ones,
> so the 3x cliff between W4A8 and W4A16 is bought cheaply — but int8
> activations *alone* contribute 2.87e-2 where an 8-bit weight contributes
> 3.4e-3, so **W8A8 is 99% activation error**. Asymmetric Q4 is worth 5% at
> equal bits, not 2x. And a bug fell out: `quantizeQ8`'s fp16 block scale goes
> subnormal under maxAbs 7.75e-3, worth **14x** on one real tensor in fourteen.
> [Write-up](research/l0d-quant-error.md)

---

## The three findings that set the direction

**1. `llama.cpp` already implements this architecture, is built with Vulkan on
this box, and supports `qwen4exp`.** `/home/kube/repos/llama.cpp` has
`src/models/qwen4exp.cpp` (1279 lines: DeltaNet, the QSA indexer, PLE n-gram
hashing, hyper-connections) and 50 built tools. Confirmed at L1 by running
them: it is a reference implementation, an oracle (`llama-eval-callback`), an
accuracy criterion (`llama-perplexity`) and a performance baseline
(`llama-bench`) at once. **This vertical does not need a Python reference
dump**, which is what S1 and T1 spent their first session on.

**2. A quarter of UD-Q4_K_XL is a lookup table that should never touch the
GPU.** Of its 111.32 GB, **28.80 GB** is `per_layer_token_embd` — a 320 M-row
hash table read 16 rows of 160 values at a time, **1.41 KB per token**. Leave
it mmap'd on the host and the resident core is **82.52 GB**.

> **L1 confirmed llama.cpp does exactly this.** `qwen4exp.cpp` creates the
> table with `TENSOR_READ_LAZY` — *"read rows on demand instead of loading
> whole tensor; requires mmap"* — and `free` shows **77 GiB resident** during
> a run against the inventory's 76.86 GiB core. So D2 is not a deviation from
> the reference; it is what the reference does, and the baseline above is a
> like-for-like comparison.

**3. The stock quant is tuned for accuracy-per-byte, not for tok/s on a
bandwidth-bound APU, and it costs 1.7x.** Unsloth put every dense tensor at
Q8_0 and the routers at F32. On a machine where decode is pinned to the DRAM
bus that is the single most expensive choice available, because **76% of the
bytes read per token are dense** — each expert is read 10 times in 512, every
dense weight is read every time.

| | resident core | GB/token | **tok/s ceiling** | measured |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL as shipped | 82.52 GB | 6.334 | **38.2** | llama.cpp: **25.15** |
| our own bank, ~4.25 bits on everything streamed | ~67 GB | ~3.6 | **~67** | — |

---

## Direction

**Phase 1 — make it run on UD-Q4_K_XL.** Consume the GGUF natively. **Done at
L1: the reader, the five dequant paths and the tokenizer all exist and are
checked against llama.cpp.** What is left is the model itself — L2 to L7.
Correctness is checked tensor-for-tensor against `llama-eval-callback`; the
acceptance criterion is *it generates the same text as llama.cpp*.

**Phase 2 — our own bank.** Re-quantise to the repo's W4A8 layout (§1.1's
repack) at widths chosen for this bus rather than for a generic machine:
~4.25 bits on everything streamed, fp16 routers, the n-gram table left as
IQ4_NL on disk, and — L0c — **fp16 scales per 32 nibbles in a k-major plane**.
Target ~67 tok/s against phase 1's 38.2 ceiling and the reference's 25.15.
Source it either by transcoding the GGUF (cheap, double-quantises) or from
the 360 GB bf16 (clean — and **unsloth publish their imatrix**,
`imatrix_unsloth.gguf`, 580 MB, so the calibration is free).

**Phase 3 — the throughput levers that are worth more than the format.** The
MTP head is downloaded (2.79 GB, `mtp-…-Q4_K_M.gguf`); a draft step is ~12% of
a full one, so self-speculation is worth ~1.5-1.8x. Batching amortises the
dense 76% completely. Both dwarf the difference between 4 and 5 bits.

---

## The machine, measured

| limit | value | where from |
|---|---|---|
| DRAM bandwidth | **236 GB/s** peak; **242 GB/s** best real decode GEMV | §0, §1.7 |
| MALL | 32 MiB, 805 GB/s copy / 965 read | §0.4, §5.1b |
| WMMA fp16 | 55.5 TFLOP/s; best real kernel 39.0 (70%) | §0.1, §2.7 |
| memory type, for a GPU read | **irrelevant**, all 8 within 0.57% | §5.1 |
| the real capacity ceiling | **physical RAM**, 117.7 GiB, less the OS | §5.1 |
| a resident 82.52 GB model | **77 GiB, steady**, with 39 GiB left for page cache | L1 |
| `maxBufferSize` / `maxMemoryAllocationSize` | **4 GiB − 4 B** | `vulkaninfo` |
| `maxDescriptorSetStorageBuffers` | 8 388 606 | `vulkaninfo` |
| GTT / system RAM | 117.7 GiB (`amdgpu.gttsize=126976` already set) | `/proc/cmdline` |
| disk free | 1.1 TB after the download | `df` |

The 4 GiB buffer cap is the one structural consequence: a 60-80 GB bank is
**~100 buffers**, one per layer per projection (a layer's Q4 expert pair is
1.26 GB, comfortably under). Descriptor counts are not a constraint.

---

## The model, from its own tensor table

`go run ./cmd/gguf -tensors` over the real checkpoint — 1224 tensors, not a
config.json transcription. `general.size_label` is `512x56B`;
`general.description` is "A Preview of the Qwen4 Architecture".

```
arch            qwen4exp   (Qwen4ExpForConditionalGeneration, multimodal)
layers          48 = 36 linear_attention + 12 full_attention (interval 4)
hidden          2560       residual stream 10240 (hyper-connections, hc_count 4, low_rank 320)
vocab           248320     context 262144 (1 M with YaRN)   tie_word_embeddings false
MoE             512 experts, 10 active + 1 shared, ffn 640, router in F32
```

A GGUF states a weight as `[in, out]`, so every shape below is `[K, N]`:

| tensor | shape | type | per pass |
|---|---|---|---|
| `attn_q` | [2560, **12288**] | Q8_0 | 12 — query *and* gate, 24 heads x 256 x 2 |
| `attn_k`, `attn_v` | [2560, 512] each | Q8_0 | 12 each — 2 KV heads x 256 |
| `attn_output` | [6144, 2560] | Q8_0 | 12 |
| `indexer.q_proj` | [2560, 512] | **BF16** | 12 — QSA, 4 heads x 128, top_k 2048 |
| `indexer.k_proj` | [2560, 128] | BF16 | 12 |
| `attn_qkv` | [2560, **10240**] | Q8_0 | 36 — DeltaNet's *fused* input projection |
| `attn_gate` | [2560, 6144] | Q8_0 | 36 |
| `ssm_out` | [6144, 2560] | Q8_0 | 36 |
| `ssm_alpha`, `ssm_beta` | [2560, 48] each | F32 | 36 each |
| `ssm_conv1d` | [4, 10240] | F32 | 36 — depthwise, k=4 |
| `ssm_a`, `ssm_dt.bias` | [48] | F32 | 36 |
| `hc_{attn,ffn}_{down,up}` | [10240, 320] / [320, 10240] | Q8_0 | **97 each** incl. `output_hc_*` |
| `hc_{attn,ffn}_inject` | [10240, 4] | F32 | 96 |
| `ple_key` | [2560, 10240] | Q8_0 | 1 — layer 1 only |
| `ple_value` | [2560, 2560] | Q8_0 | 1 |
| `per_layer_token_embd` | [160, **320001536**] | IQ4_NL | gather, 1.41 KB/token |
| `ffn_gate_exps`, `ffn_up_exps` | [2560, 640, 512] | Q4_K (Q5_K on layer 2) | 48 |
| `ffn_down_exps` | [640, 2560, 512] | **Q5_1** (Q8_0 on 5 layers) | 48 |
| `ffn_gate_inp` | [2560, 512] | F32 | 48 — the router |
| `ffn_*_shexp` | [2560, 640] / [640, 2560] | Q8_0 | 48 — the always-on shared expert |
| `token_embd`, `output` | [2560, 248320] | Q8_0 | gather / 1 |

Other metadata worth having to hand: `ssm` state 128, groups 16, dt_rank 48,
inner 6144; rope mrope interleaved, sections [11,11,10], 64 of 256 dims, theta
1e7; PLE ngram_size 3, heads_per_ngram 8 → 16 hash heads with prime vocabs
~20 000 0xx; vision tower 27 layers, hidden 1152, in a separate
`mmproj-*.gguf` (0.9 GB, not downloaded).

> **`bench/modelshapes.go` has been corrected to this table** (L1). It had
> `attn.q` at 6144, one fused 512-wide KV, a 2048/6144 DeltaNet pair that does
> not exist, and no rows at all for the hyper-connections, the indexer or the
> PLE block — **220 matmuls a token** and 13.4% of the weights. It now reads
> **6.665 B weights a token against the checkpoint's 6.671 B**, 0.09% apart,
> and `shapes` re-run over it prices the 2081 dispatches at **0.62 ms of
> launch cost, 5% of a 4-bit decode step**.

### What UD-Q4_K_XL actually contains

1224 tensors, 176.944 B params, **111.32 GB, 5.03 bits/weight** — produced by
`cmd/gguf` in 47 ms and cross-checked tensor-for-tensor against
`reference/gguf_inventory.py` (**0 disagreements**):

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
decode budget for 0.06 B of parameters.

### Where the bytes go at decode

    dense, every token      4.830 GB   76%
    experts, 10 of 512      1.504 GB   24%
                            -------
    per token               6.334 GB   -> 38.2 tok/s at 242 GB/s
                                       -> llama.cpp reaches 25.15 (159 GB/s)

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

82.52 GB of weights plus 6.8 GB of KV is comfortable in 117.7 GiB, and L1
measured 77 GiB resident with 39 GiB of page cache still live beside it.
Capacity has stopped being the binding constraint on this model at any format
below Q8. Bandwidth is the whole story.

---

## What exists to build on

| piece | state |
|---|---|
| `gguf/` | **L1: the GGUF reader.** mmap'd, sharded, the split convention, the value grammar, openable from any shard. 1224/1224 tensors against the Python inventory. |
| `gguf/dequant.go` | **L1: Q4_K, Q5_K, Q5_1, Q8_0, IQ4_NL, F32, F16, BF16** — bit-exact against ggml's own `to_float` via `reference/dequant_ref.c`. |
| `cmd/gguf` | inventory, quant mix, decode budget, `-check` against `tensors.json`. |
| `cmd/llm` | the vertical's driver. `-tokenize` today; generation at L7. |
| `zimage/tokenizer/` | **L1: `FromVocab`** builds the BPE from GGUF metadata, and `split()` now carries both Qwen pre-tokenizers. 213/213 against `llama-tokenize`. |
| `zimage/qwen/` | a working Qwen3 transformer on the GPU in Go — RMSNorm, RoPE, GQA, SwiGLU, four shared arenas. The skeleton for L2. |
| `shaders/gemv_w4a8.comp` | decode GEMV at **99-103% of the bus** (§1.1, §1.7), plus 25 grouped / M-blocked / N-blocked MoE builds (§1.8-§1.12). |
| `shaders/gemm_wmma_q4.comp` | Q4 prefill GEMM, **2.10x fp16** on a MoE block (§2.2), with L0c's k-major scale plane. |
| `shaders/moe_route/gather/combine` | top-k routing and the permutation around it. |
| `shaders/qwen_attn_wmma_*`, `qwen_rope.comp` | causal WMMA attention at 38.8 TFLOP/s (§3.3); NeoX rotary. |
| `rmsnorm_shared.comp` | 95% of the bus. |
| `shaders/bank_gather.comp` + the `bank` family | L0a and L0b's probe. |
| `bench/modelshapes.go` | **L1: corrected** to the real tensor table. |

## What has to be built

1. **Gated DeltaNet — 36 of the 48 layers.** §3.6, unbuilt, "effort: high".
   Chunked linear attention, fp32 recurrent state, depthwise conv1d k=4 over
   10240 channels, `A_log`/`dt_bias` gating, sigmoid output gate. The largest
   new kernel in the project's history, and three quarters of the model.
2. **QSA sparse attention** — indexer, top-2048 selection, gathered attention.
   Not in `IDEAS.md` at all; it needs a new section.
3. **Hyper-connections** — 4-branch gated residual at width 10240, every layer,
   and 194 low-rank projections a token.
4. **PLE n-gram** — trigram hashing into 16 heads over a 320 M-row mmap'd
   table, `layer_multipliers`, conv1d k=4, key/value projections.
5. **KV cache**, mrope (interleaved, sections [11,11,10], 64 of 256 dims), and
   a decode loop.
6. **The re-quantiser** (phase 2) and **MTP speculative decoding** (phase 3).

---

## Task list

### L0 — answer the format question by measurement, not arithmetic

- [x] **L0.0** the inventory without downloading the model —
      `reference/gguf_inventory.py`, 35 MB of range requests.
- [x] **L0a** *Does the bus survive a 64-82 GB working set?* **It does not
      notice.** [Write-up](research/l0a-bank-range.md)
- [x] **L0b** *Is heap 0 as fast as heap 1?* **No type or heap is faster than
      any other, and the heap sizes are not limits.**
      [Write-up](research/5.1-memory-types.md)
- [x] **L0c** *The blocked scale plane.* **1.27x, and it changes the format.**
      [Write-up](research/l0c-scale-plane.md)
- [x] **L0d** *Per-format error metrics* (§7). **W4A8, symmetric, per-token
      activation scales.** [Write-up](research/l0d-quant-error.md)

### L1 — get the checkpoint, and a baseline  *(done)*

- [x] Download `UD-Q4_K_XL` (4 shards, 111.32 GB) and `mtp-…-Q4_K_M.gguf`
      (2.79 GB). **18 minutes**, `reference/fetch_llm_checkpoint.sh`, resumable.
- [x] `llama-bench` and `llama-completion`: **pp2048 388.60, tg128 25.15**.
- [x] Go GGUF reader, checked tensor-for-tensor: **1224/1224, 0
      disagreements**.
- [x] Dequant paths for the five formats, **bit-exact against ggml**.
- [x] Tokenizer: 248320 vocab, the multimodal specials and the `qwen35`
      pre-tokenizer — **213/213 against `llama-tokenize`**.
- [x] Correct `bench/modelshapes.go` to the real inventory.
- [x] `llama-perplexity` on wikitext-2 (145 chunks at n_ctx 2048, 15.2
      minutes): **PPL = 4.0340 ± 0.02283**, the accuracy reference for
      everything phase 2 does.
- [x] Re-run the `shapes` family against the corrected table — 313 rows. It
      prices a decode step at 3.33 GB and **73 tok/s at 4 bits before
      attention**, with **2081 dispatches costing 0.62 ms, 5%** (§4.1). The
      old table had 1861 and read 13.4% too few weights.

### L2 — the dense skeleton

- [ ] **First, settle the prefill question above**: price one full-attention
      layer and one DeltaNet layer against §2.2's kernels before building the
      stack, so the 5x is attributed before it is designed around.
- [ ] Embeddings, PLE gather (host, mmap'd), hyper-connection mix/combine,
      RMSNorm, one full-attention layer with mrope.
- [ ] Gate: layer 3's output matches `llama-eval-callback` to fp16 tolerance.

### L3 — Gated DeltaNet  *(the big one)*

- [ ] CPU reference first, fp32 state, against the dump.
- [ ] The chunked GPU kernel (§3.6). Find out whether it is matmul-bound
      (coopmat) or serialised on the state update.
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
- [ ] Gate: logits match llama.cpp; prefill tok/s against 388.60.

### L7 — decode

- [ ] KV cache, ring buffer, sampling, the generation loop.
- [ ] Gate: it generates the same text as llama.cpp; tok/s against the 38.2
      ceiling and against 25.15.

### L8 — phase 2, our own bank

- [ ] Choose per-tensor widths from L0d and L1's **PPL 4.0340**.
- [ ] Re-quantise (transcode from the GGUF, or from bf16 with unsloth's
      published imatrix) into the §1.1 W4A8 layout with an L0c scale plane.
- [ ] Gate: ~67 tok/s, ≤67 GB resident, perplexity within a stated delta of
      **4.0340** on the same corpus, context and chunking.

### L9 — phase 3, and shipping

- [ ] MTP speculative decoding (the separate GGUF). Target 1.5-1.8x.
- [ ] Batching, then the HTTP API `GOALS.md` asks for.
- [ ] Vision tower, if wanted.

---

## How to run things

    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
    L=/home/kube/repos/llama.cpp

    # ours
    go run ./cmd/gguf $M                      # the inventory and the decode budget
    go run ./cmd/gguf -tensors -kv $M         # every tensor, every metadata key
    go run ./cmd/llm -tokenize 'hello world'  # generation arrives at L7

    # the reference implementation, the oracle, the baseline
    less $L/src/models/qwen4exp.cpp                  # 1279 lines, the whole architecture
    $L/build/bin/llama-bench -m $M -p 512,2048 -n 128 -r 2
    $L/build/bin/llama-eval-callback -m $M -p 'hello' -n 1   # per-tensor dumps
    $L/build/bin/llama-perplexity -m $M -f models/wikitext-2-raw/wiki.test.raw -c 2048 -b 2048

    # re-fetch (resumable, checks sizes)
    reference/fetch_llm_checkpoint.sh

## Decisions

| # | decision | why |
|---|---|---|
| D1 | **Consume UD-Q4_K_XL first, don't quantise from bf16.** | 114 GB instead of 360, an imatrix-calibrated checkpoint, and a bit-exact oracle. Phase order: run it, profile it, then optimise (`TODO.md`). |
| D2 | **The n-gram table lives off-heap, mmap'd.** | 28.80 GB of capacity for 1.41 KB/token of bandwidth. **L1: llama.cpp already does this** — `TENSOR_READ_LAZY` — so it is the reference behaviour, not a deviation. |
| D3 | **Target ~4.25 bits on everything *streamed*, not just the experts.** | 76% of decode bytes are dense. L0d adds the accuracy half: above ~5 bits the int8 activations are the floor anyway. |
| D4 | **Do not go below 4 bits on the experts.** | They are 24% of the traffic; Q3 buys ~11% for a real accuracy hit. |
| D6 | **W4A8, with per-token activation scales.** | L0d: int8-per-token costs **1.13x** the error of fp16 activations, against a 3x throughput cliff. Per-token is free in the RMSNorm epilogue (§3.1). |
| D7 | **Symmetric Q4, not asymmetric.** | L0d: 1.043-1.053x at equal bits, where §7 predicted 2x. Revisit only if L9's perplexity asks for it. |
| D8 | **fp16 scales per 32 nibbles, in a k-major plane.** | L0c made the fine block cost 1.0% instead of 14.2% at prefill. The remaining cost is decode bytes — 11.1% of the bank against 3.0% at per-128 — the one axis still worth trading if tok/s falls short. |
| D5 | **`llama.cpp` is the oracle, not a Python dump.** | It is built, it is Vulkan, it supports `qwen4exp`, and L1 got correctness, accuracy and a baseline out of one binary. |

## Open questions

- **Why is prefill 393 and not ~2000?** The largest open number in this file,
  and it is new. Candidates: the 36 DeltaNet layers and the QSA indexer are
  not matmuls and were never in the estimate; the 194 hyper-connection
  projections a token are narrow (320 wide) and were missing from the shape
  table entirely; or llama.cpp's Vulkan path is simply well short of this
  machine. Measurable at L2 with one layer of each kind, before the stack
  exists.
- **Does 111.32 GB — the whole GGUF, n-gram table included — hold live?** 80
  GiB is demonstrably residable (L0a) and 105 GiB reservable (L0b); L1 ran at
  77 GiB with 39 GiB of page cache beside it. If the whole thing fits, D2
  becomes an efficiency choice rather than a necessity and the host-side PLE
  gather can be deferred past L2.
- **What is `down_proj`'s activation error really costing?** L0d settled W4A8
  at 1.13x mean, 1.6x worst, and the worst site is `down`, whose SwiGLU input
  has no norm in front of it. Finer activation groups on that one input, or a
  rotation, are the unmeasured mitigations.
- **Fix `quantizeQ8`'s subnormal scale** — two lines (floor it at fp16's
  smallest normal), but it changes the CPU reference every W8A8 correctness
  check is built from. Becomes urgent if phase 2 transcodes UD-Q4_K_XL's Q8_0
  dense tensors through it.
- **Does the scale plane's layout matter at *decode* too?** L0c is all GEMM.
  §1.8 measured the grouped GEMV paying 1.045-1.116x on the same axis and put
  it down to bytes alone, but it has not been tried with a k-major plane.
- **How much of the Q8_0 dense allocation is actually needed?** D3 assumes
  ~4.25 bits is enough; **4.0340** against L8's re-measurement is the only
  honest test.
- **`§3.4 finding 4` says prefill is "98% memory-bound by weight bytes";
  §2.2's direct measurement of a Q4 block says 2.2x above its memory floor.**
  They disagree, and L1's 393 tok/s says the model-level estimate built on
  either is 5x optimistic. Reconcile before sizing prefill.
- **Batch and speculation.** At batch 4 the dense 76% amortises completely.
  Worth knowing whether the API will ever serve more than one stream before
  optimising the batch-1 path to death.
