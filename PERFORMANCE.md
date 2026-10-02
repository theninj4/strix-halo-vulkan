# PERFORMANCE — headline numbers

> **Written 2026-10-02.** One page of what this tree achieves on one AMD
> Ryzen AI MAX+ 395 "Strix Halo" (gfx1151, Radeon 8060S, 40 CUs, RDNA3.5,
> 128 GB unified LPDDR5X, RADV / Mesa 26.2.2, ~150 W package cap). Every
> number here is measured and taken from a live doc or archive, which is
> linked. When a number changes, update it here and in its source. History
> stays in the source docs, not here.

## 1. The hardware, and how much of it we reach

### The ceilings, measured on this chip

| resource | spec | measured ceiling | source |
|---|---|---|---|
| matrix cores, fp16 WMMA | 59.4 TFLOP/s (512 FLOP/clk/CU × 40 × 2.9 GHz) | **55.5 TFLOP/s** (478 FLOP/clk/CU at 2899 MHz, register-only `peak` probe) | [`research/kernels-vertical.md`](research/kernels-vertical.md), `results/peak.csv` |
| int8 dot4 / packed fp16 FMA / fp32 FMA | — | 54 / 25 / 23 TOP/s | [`research/kernels-vertical.md`](research/kernels-vertical.md) §0.7 |
| DRAM bandwidth | 256 GB/s | **236 GB/s** (stream, ≥ 50 MB footprint) | [`research/0-measurement-validity.md`](research/0-measurement-validity.md) |
| MALL (32 MiB last-level cache) | — | **~805 GB/s** copy, 940–965 GB/s read-only | [`research/5.1b-mall-cliff-and-stride.md`](research/5.1b-mall-cliff-and-stride.md) |

The clock moves under load. A real GEMM runs at ~2640–2810 MHz and
137–154 W, against the probe's 2899 MHz at ~110 W. So we report kernel
efficiency **per clock**, against 478 FLOP/clk/CU.

### What our kernels achieve

| kernel class | achieved | of ceiling | source |
|---|---|---|---|
| **fp16 WMMA GEMM**, shipped (`dit_gemm_wg128x256_lds_w32`, LDS-staged, wave32) | **42.8–43.5 TFLOP/s** on H3's real projections at ~2650 MHz | **85% per clock** (406 FLOP/clk/CU) | KERNELS.md G2, G-o8, §2.9–2.10 |
| **WMMA flash attention**, shipped (`h3_attn_t_kvlds_w8_kt4`, transposed, K/V through LDS) | **40.8–40.9 TFLOP/s** in a 480p video forward | **78% per clock** | KERNELS.md G4b, §3.9 |
| **Q4-weight MoE GEMM** (LLM prefill, up projection) | **37 TFLOP/s executed** at 2048 tokens | ~75% per clock | KERNELS.md G5a, §2.11 |
| conv3x3 as implicit GEMM | 41–42 TFLOP/s | ~74% | `research/stage-8-vae-conv.md` |
| **decode GEMV** (W4A8, 4-bit weights × int8 activations) | **235–243 GB/s** from DRAM | **99–103% of the bus** at every N | README, §1.1, §1.7–§1.12 |
| streaming kernels (norms, packs, gates) | 173–217 GB/s | 73–92% of the bus | KERNELS.md G6, §5.4 |
| a whole 480p video forward, every kernel | 33.5 TFLOP/s sustained | 60% of spec, 80% of our best kernel | VIDEO.md M11h |

**Outside calibration.** The best public number on this chip is 36.9
TFLOP/s (PyTorch bf16 on ROCm; rocBLAS 6–20). Our shipped GEMM beats it by
1.18x. A mature library on RDNA3 silicon (rocBLAS on a 7900 XTX) reaches
~92% of peak, so ~7 points of GEMM headroom remain, mostly the barrier's
lockstep.

**Energy.** The matrix pipe at full rate costs 1.4 pJ a FLOP (567 GFLOP/J).
A DRAM byte costs ~490 pJ, about the energy of 340 matrix FLOPs. The best
GEMM reaches 285 GFLOP/J and a whole video forward 218 GFLOP/J. LLM decode
draws 91 W and prefill 142 W (KERNELS.md §0.7).

## 2. Headline numbers by vertical

Every vertical runs end to end in Go on Vulkan. Each one is gated against
a reference implementation and served by `cmd/serve` behind an
OpenAI-shaped API ([`research/api-server.md`](research/api-server.md)).

### Text generation: Qwen3.8-Flash-Next (180 B MoE, 6 B active, 4-bit banks)

| metric | number | source |
|---|---|---|
| decode, depth 0, serial | **41.4 tok/s** (24.19 ms a step) | P22c + P21d, [`research/p19-decode-attribution.md`](research/p19-decode-attribution.md) |
| decode, speculative (MTP draft head, depth 2), served | **56.8 tok/s greedy, 56.6 sampled** (1.38x), SPEED-Bench | P20f/P20g, [`research/p20-llamacpp-mtp.md`](research/p20-llamacpp-mtp.md) |
| decode, speculative, `cmd/llm` | 61.3 tok/s (1.48x) at depth 2, 62.9 (1.53x) at depth 3 | P20e/P20h |
| prefill | **1568 / 1599 / 1604 tok/s** at ubatch 2048 / 4096 / 8192 (3.59x llama.cpp at 8192) | P25, [`research/p25-prefill-heads-rule.md`](research/p25-prefill-heads-rule.md) |
| prefill at 128k context | 1011 tok/s at ubatch 2048, **1176 at the served 4096** | P17 |
| decode at 128k context | 31.5–32.2 tok/s (0.94x of depth 0)¹ | P16/P17 |
| full trained context | **262 144 cells** served; a 237k-token prompt prefills at 1022 tok/s and decodes at 29.9 tok/s | P18 |
| concurrency | 3 full-context slots; batched decode of 3 conversations **56.2 tok/s aggregate** (1.64x one slot) | [`CONCURRENCY.md`](CONCURRENCY.md) C5 |
| quality | wikitext perplexity 4.0992, **+1.74%** of our own unquantised 4.0289; 4.0344 over positions 131k–262k | P4c, P18 |
| footprint | ~84–98 GB resident (whole machine A) | |

¹ Measured before P19–P22. Those took depth-0 decode from 34.4 to 41.4
tok/s, and the depth figures have not been re-run since.

**Vision** (`-llm-mmproj`, 27-layer tower). Images go through all three chat
doors. The tower takes 28 ms (a QR code) to 2.1 s (a photo at the size
cap). The V10 eval scores **17/18, the same as llama.cpp** on the same
weights. The tower yields the device every 50 ms, so a voice turn still
decodes at ~10 tok/s beside it ([`research/llm-vision.md`](research/llm-vision.md)).

### Speech → text: parakeet-tdt-0.6b-v3

An 11 s clip takes **43 ms, 257x real time** (front end 21 + encoder 17 +
decode 5 ms). The transcript is exact, with word timings. Served on
`/v1/audio/transcriptions` and over Wyoming.
([`research/speech-vertical.md`](research/speech-vertical.md))

### Text → speech: Kokoro-82M

**31 ms for 3.25 s of audio (105x)**, **162 ms for 19.5 s (120x)**: a flat
8.0 ms per second of audio. The endpoint answers in 59 ms. The STT→TTS round
trip runs at 70.1x real time. ([`research/tts-recap.md`](research/tts-recap.md))

### Image generation and editing: Qwen-Image-2.1

| metric | number |
|---|---|
| 1024², 40 steps, end to end | **1m28.8s** (fp16, before G3/G6) |
| a 1024² step, int8 bank, today | **1.886 s** (after G3's SwiGLU epilogue and G6) |
| edit, 1024², one reference | **1m54.2s** |
| resident | **20.4 GB** on int8 banks (31.5 in fp16); edits 39.4 → ~28 GB |
| fidelity | mean abs **3.4e-4** against the fp32 oracle's picture; edit max abs 0.0014 |
| previews | streamed for 0.3% of a request |

([`research/qimage-vertical.md`](research/qimage-vertical.md))

### Video with sound: MiniMax-H3

| metric | number |
|---|---|
| a 480p forward | **24.3 s** (35.6 at first serve; M11c, M11d, G2, G-o8, G3, G6, G4b) |
| a 768p forward | **92.5 s** (143 s at first serve) |
| a 480p VAE decode | **27.5 s** (38.4 before M11e, G3, G6) |
| served, the README t2va prompt (448×256, 124 frames, 8 steps) | **80 s** (199 s at M9) |
| request peak memory | 31–33 GB on int8 banks (57 GB fp16) |
| stagings from `bank-cache/` | ~12 s, down from ~94 s |

([`research/video-vertical.md`](research/video-vertical.md))

### Music: ACE-Step 1.5 XL turbo + 5 Hz LM 4B

A **60 s song with thinking takes 14.4 s**, most of it the int8 LM at the
memory bus. A 4-minute song DiT-only takes 9.6 s, and a 30 s cover or
repaint ~2 s. The DiT forward runs at 111 ms for 30 s of audio and 2050 ms
for 10 minutes (after G3/G6). Our latents are **2.8–39x closer to fp32 than
upstream's own bf16**. ~18 GB resident.
([`research/music-vertical.md`](research/music-vertical.md))

### Embeddings: Qwen3-Embedding-0.6B

One text takes **10.4 ms** served (9.6 ms on the device). Batched (E7), 32
short texts take **46 ms** (8.3x) and 128 texts of ~36 tokens take
**305 ms** (5.0x). Batched vectors are bit-exact against a lone run, and
cosine is 0.999999+ against fp32. The batched path is compute-bound past
~450 rows at 15.6 TFLOP/s. ([`research/embedding-vertical.md`](research/embedding-vertical.md))

### Classification: Kev-4B (Qwen3.5-4B + LoRA + pointer head)

The README ticket takes **52 ms** on the device and 54 ms over HTTP, on an
int8 bank. A 2,269-token text takes **761 ms new and 87 ms again** from the
prefix cache, bit-identically. Concurrent requests share passes, at ~29
req/s against 19.5. Results are within 4e-4 of Kev's fp32 probabilities.
**Kev's published accuracy is reproduced**: fp16 is within a question of
the card on every suite, and int8 costs −0.19 pp.
([`CLASSIFICATION.md`](CLASSIFICATION.md))

### OCR: PaddleOCR-VL-1.6 + PP-DocLayoutV3

On a 331-page OmniDocBench v1.6 subset, the Go pipeline scores **96.13**.
PaddleX's own pipeline on our engine scores 96.14, and the model card
gives 96.34 on the full set. Batched decode over a paged KV cache (O11a)
runs the subset in **18 min instead of 66**: 3.0 s a page mean, with the
worst page 143 → 18 s. ([`research/ocr-vertical.md`](research/ocr-vertical.md))

### Object detection: RF-DETR-L

Planned only (B-stages, [`OBJ-DETECTION.md`](OBJ-DETECTION.md)). No
numbers yet.

## 3. Serving the whole set

Deployment uses two machines: the LLM alone on one, everything else on the
other. Image, music and video share one swap slot that sits at **9.4 GB
at rest and peaks at 60 GB**, against ~112 GB with all three resident. A
switch costs ~19–30 s, and speech is answered through every load
([`TODO.md`](TODO.md), [`research/api-residency.md`](research/api-residency.md)).
