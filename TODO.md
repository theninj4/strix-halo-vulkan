# TODO — the state of play

> **Rewritten 2026-09-21.** The per-vertical progress files were
> consolidated into [`research/`](research/README.md) on 2026-09-20 (`LLM.md`,
> `LLM2.md`, `SPEECH.md`, `TTS.md`, `IMAGE.md`, `PIPELINE.md`,
> `EMBEDDING.md`, `IDEAS.md` and the old session-log `TODO.md`) — see the
> file map at the bottom. The root `IMAGE.md` then came back live for one day
> to carry the Qwen-Image-2.1 replacement, and was frozen in turn on
> **2026-09-21** as
> [`research/qimage-vertical.md`](research/qimage-vertical.md). **No live
> progress file remains at the root**: this one is it — what each vertical
> is, what it measures today, and what is open. It is **rewritten**, not
> appended to; a closed item's write-up goes to `research/` and history
> lives in git.
>
> Addresses: `§N.M` (cited from code) resolves in
> [`research/ideas.md`](research/ideas.md); stage letters (L7d, T9, S10,
> Q9b, E7, P6…) resolve in the vertical archives listed at the bottom;
> decisions **D1–D21** are in
> [`research/llm-vertical.md`](research/llm-vertical.md). A code comment
> citing `MUSIC.md` (A-stages) resolves in
> [`research/music-vertical.md`](research/music-vertical.md). A code comment
> citing `IMAGE.md` resolves by what follows it: **Q-stages, numbered
> decisions and Q-o numbers** are `research/qimage-vertical.md`, **I-stages**
> are `research/zimage-vertical.md`. `GOALS.md` is the target; `API.md`
> documents the server as it answers today.

## The five verticals, at a glance

All five of `GOALS.md`'s models run end to end in Go on Vulkan, validated
against a reference, and are served by `cmd/serve`. The competition from
here is our own ceiling, not a reference implementation.

| vertical | model | headline, measured | open |
|---|---|---|---|
| text generation | qwen3.8-flash-next (180 B, 6 B active) | decode **36.0 tok/s through `-gen`, 34.2 through the server at every `-llm-batch`** (P16, against 25.8 at the shipped 4096), at **+1.74%** perplexity; prefill **1403.9 tok/s at 8192 rows, 3.59x** (and **1199 through the server** at 4096), still climbing where llama.cpp plateaus — **and 1.10–1.17x on top of that from KERNELS.md G5a (2026-09-29, §2.11): the MoE up GEMM's K loop as a software pipeline, `cmd/llm -graph` same-hour A/B 1194/1206 → 1407/1408 tok/s at 2048, 1317/1339 → 1494/1495 at 4096, 1306/1345 → 1459/1462 at 8192, and 1422 / 1506 at 2048 / 4096 with the IQ4_NL down pipeline on top (G5b)*, and **1.07x again from G5c (2026-09-30, §2.12): a short last tile an expert, same-hour A/B 1417/1419 → 1514/1513 tok/s at 2048, 1500/1502 → 1549/1543 at 4096, 1461/1467 → 1481/1483 at 8192**; and G5d (2026-09-30, §2.13, the nibble as an f16 denormal, bit-identical) (same-hour A/B 1513/1513 → 1521/1523 tok/s at 2048, 1549/1547 → 1554/1557 at 4096, 1484/1483 → 1491/1488 at 8192: a rounding of the prefill number, recorded so the next session does not re-run it); **128 000 cells prefills at 1011 tok/s at ubatch 2048 and 1176 at the served 4096** (P17, against 834 and 900; falloff 0.88x), decode **32.2 tok/s at 128k** and **34.4 at depth zero** (P16/P17), falloff to 128k **0.94x** | **token generation is the focus from 2026-09-30: P19–P24 in the text section; P19 and P21a done the same day (the step attributed: 16.4 / 4.3 / 3.0 / 1.1 ms; the head and `hc.up` — the two projections still on the padded GEMM at one row — moved to GEMVs, 37.5 → 38.7 tok/s at 25.8 ms a step; P21b's load hoisting measured dead; P22a the shared expert as the eleventh tile of the routed MoE dispatch, 38.8 → 39.2 tok/s at 25.5 ms; P21c the MoE down GEMV's five serial round trips a lane hoisted and A staged sixteen bytes a load, 39.1 → 39.8 tok/s at 25.13 ms, both down rungs re-laddered and confirmed), next the rest of P22, speculation re-priced at 1.23 steps a two-row pass**; concurrency and batching (P6, now [`CONCURRENCY.md`](CONCURRENCY.md): three full-context slots, a priority scheduler, per-slot prefix checkpoints and batched decode, all done); the gathered attention at 34% of matrix-core peak with its four bounds eliminated; ~~`hc.cn` at prefill~~ (G6, §5.4: the grid's walk, 184 → 200 GB/s, prefill 1520 → 1538 tok/s at 2048 and decode 36.87 → 37.35); decode is now fusion at 1-2% a step; the 196 moves went in P22b (2026-09-30, 196 → 4 and no zero rows at decode, 25.13 → 24.42 ms, 41.0 tok/s); **P22c + P21d (2026-10-01): the router's tail as one bit-exact dispatch and `ple.kv` — the third one-row GEMM — on the GEMV, 24.42 → 24.19 ms, 41.4 tok/s; the weightless list is a floor of launches now and P20 is next** |
| speech → text | parakeet-tdt-0.6b-v3 | an 11 s clip in **43 ms — 257x real time**, whole model resident | S10 front end (48% of the pipeline); S9 long clips |
| text → speech | Kokoro-82M | **31 ms for 3.25 s (105x)**, **162 ms for 19.5 s (120x)** — flat per second of audio; the endpoint answers in 59 ms | the vocoder's 20 ms of arithmetic; three small boundaries |
| image generation + editing | Qwen-Image-2.1 | 1024², 40 steps in **1m28.8s** (fp16; int8 +4% a step), **20.4 GB resident since Q13's int8 banks** (31.5 in fp16), native RGBA; streaming previews cost **0.3%**; the fp32 oracle's picture to mean **3.4e-4**. **Edits answer too**: **1m54.2s** on one reference at 1024², 39.4 GB, the oracle's edit to max abs **0.0014** | **parked 2026-09-21** — Q0–Q12 all closed; the 1184²-area ceiling is the one capability left unbuilt; KERNELS.md G4 (2026-09-30, research §3.8): the transposed attention (`h3_attn_t.comp`) on this DiT is 1.07x on the attention and 1.7% of a step, **opt-in** (`QIMAGE_ATTN_T=1`) because the edit oracle's teacher-forced prefill amplifies fp16-level differences in the prefix rows ~100x at three target rows — the kernel is exact on identical inputs at every block, a plain-family perturbation reads 1.73e-2 against the 2e-2 bound, and the call on that bound is this vertical's. **KERNELS.md G3 (2026-09-30, §2.14): w1|w3 as one GEMM whose epilogue is the SwiGLU**, bit-identical, a 1024² cached step **2044 → 1929–1968 ms** on the int8 bank (1.04–1.06x); the served binary needs a redeploy |
| music | ACE-Step 1.5 XL turbo + 5 Hz LM 4B | a **60 s song with thinking in 14.4 s** (the int8 LM at the bus, ~80% of it), 4 min DiT-only in 9.6 s; a **30 s cover or repaint in ~2 s**; latents **2.8–39× closer to fp32 than upstream's own bf16** across text2music, cover and repaint; served as `/v1/music` jobs, ~18 GB resident in the swap slot | **closed 2026-09-27** (A0–A12); listening to the A10/A11 songs; not in `ai.service` |
| embeddings | Qwen3-Embedding-0.6B | a text in **10.4 ms** served (9.6 on the device), the card's similarity matrix to 1.3e-4 over HTTP; **E7 batching: 32 short texts in 46 ms (8.3x), 128 of ~36 tokens in 305 ms (5.0x)**, bit-exact against a lone run, requests share passes | not deployed yet (`ai.service` restart); latency levers: fused qkv / gate+up, split-K on o/down, an int8 bank on the small-M kernels |
| classification | Kev-4B (Qwen3.5-4B-Base + LoRA + pointer head) | TypeSafe's `/v1/systemone` via `-kev`, **within 4e-4 of Kev's fp32 probabilities**, questions isolated bit-exactly, the TypeSafe SDK unchanged; the README ticket in **52 ms** on an int8 bank (K7.1, K7.6); **Kev's published accuracy reproduced (K8): fp16 on every suite within a question of the card, fp16 agrees with Kev's fp32 except on exact ties, int8 −0.19 pp**; a repeated text from the prefix cache, bit-identically (K7.2), attention on the matrix cores (K7.3): the GDN scan in the LLM's l8 shape (K7.4), SwiGLU fused into the gate+up GEMM and a GEMM rung per projection (K7.6): a 2,269-token text 761 ms new, 87 ms again ; concurrent requests share passes (K7.5, ~29 req/s against 19.5) | live plan in root [`CLASSIFICATION.md`](CLASSIFICATION.md) (K-stages): bit-exact across batches, chunks and cache hits (K7.7: the WMMA residue was stale V padding); deployed in `ai.service` (2026-09-25); open: int8's −0.19 pp, int8 GEMM speed |

**Vision (2026-09-24): the text vertical reads images.** `-llm-mmproj`
stages the checkpoint's 27-layer vision tower beside the LLM, and all three
chat doors take images, gated against HF (processor bit-exact, positions,
PLE) and llama.cpp (the whole model's argmax). Live record and handoff:
root [`LLM-VISION.md`](LLM-VISION.md) (V-stages). V10's eval and video are
open.

**Video (opened 2026-09-26): MiniMax-H3, text/keyframes to video with
stereo sound** (`GOALS.md` item 7). The live plan and handoff is the root
[`VIDEO.md`](VIDEO.md) (M-stages). M0–M10 and M11a are done: a prompt, and
optionally a first and/or last keyframe (`fl2va`, M10), becomes an mp4 with
sound (`cmd/h3`), and **`serve -video` answers `/v1/videos`** as
OpenAI's async jobs (SGLang's H3 envelope too), sharing the device while it
runs (speech beside it ≤ 0.27 s). It takes **35.6 s a forward at the served
480p** (~14 min a request) and 143 s at the trained 768p (~2 h for 50 steps).
Deployed in `ai.service` beside the image model (2026-09-26). At rest the
service leaves ~35 GB available (image is 49.7 GB with `-edits 3`), and
since **M11a** a 480p request runs through the service with 15.4 GB still
free (it was OOM-killed before): int8 banks for the encoder and
transformer, and two host-memory fixes in `safetensors` and `serve`. Since
**M11b** (2026-09-29) the int8 banks and AdaLN tables are
cached in `bank-cache/` beside the checkpoint (47 GB of disk): a request's
stagings take ~12 s instead of ~94, output byte-identical. Since **M11c**
(2026-09-29) the attention runs transposed so P never
goes through LDS: 1.55–1.61x on the kernel, a 480p forward 35.6 → 30.6 s,
768p 144 → 111 s. **M11d** runs the down projection as two K passes,
bit-identical: a 480p forward is ~29.9 s, **27.1 s** since KERNELS.md
G2 (2026-09-29) put the GEMM on LDS-staged slabs at wave32, and **26.7 s**
since G-o8 the same day kept that build's K tiles as a loop (research
§2.10), both bit-identical. **M11e** runs the video VAE's
attention on the same transposed kernel at head 64: a 480p decode 38.4 →
37.1 s, PSNR unchanged. **M11f** priced the audio decode on the device at
≤ 2.4 s a request (it already overlaps the video decode) and left it on
the CPU. Open: the rest of performance
(M11). The Context-IR stand-in (M12) is a client concern: the
front-end rewrites prompts before it submits. M11b–M11e deployed 2026-09-29: the
README prompt at 448×256 × 8 steps serves in 80 s (M9: 199 s). **M11g** timed
the last stagings: the embedding table is under a second; the video VAE's
is 8.5 s in a request (4.6 s alone, the audio decode contends). **M11h**
(2026-09-29) assessed the headroom: a 480p forward runs at 33.5 TFLOP/s,
80% of this machine's best kernel, so exact work has ≤ 1.25x left; the
row chunk at 4096 is 1.04x measured (to ship); the multiples are lossy
(step caching, block-sparse attention, a 4-step LoRA) and a product call.

- [x] **Video in int8, to fit beside the image model** (2026-09-26,
  VIDEO.md M11a). Encoder and transformer as int8 banks by default, every
  teacher-forced step inside the released bf16 pipeline's error, no speed
  cost. A 480p request through `ai.service` beside the image model
  completed in 834 s with 15.4 GB free, after fixing what its first
  attempt's OOM kill exposed: `safetensors` F32/F16 grew by `append` (5×
  churn), and `serve` kept ~11 GB of staging garbage resident. That gave
  every vertical ~10 GB more room at rest (36.9 → 47.5 GB).

**Kernels (reopened 2026-09-29): the matrix cores to their ceiling.** The
project's first vertical, given its own live plan in root
[`KERNELS.md`](KERNELS.md) (G-stages). The fp16 GEMM has sat at **42.0 of
55.5 TFLOP/s** since stage 4, the attention at 38, and every vertical since
has ended on "the GEMMs are at their ceiling". Restated per clock (the GEMM
runs at 2813 MHz and 137 W where the peak probe ran 2899 and 110) that is
72–78%, against ~92% a mature library reaches on RDNA3 silicon. The shipped
GEMM is 240 VGPRs, no spill, one K tile a loop with no prefetch; both
head-128 attentions spill 112–126 VGPRs. The plan: pin the ceiling (G0),
read the ISA (G1), a register-prefetch GEMM (G2), attention without spills
and with lazy rescaling (G4), int8/Q4 fragments built in registers from the
probed lane layout instead of through LDS (G5). `cmd/probe` (the ISA tool,
deleted by accident 2026-09-22) is restored. **G0 and G1 done
2026-09-29** (research §0.6, §6.5): the ceiling is the hardware's (480
FLOP/clk/CU, flat across chains and wave size); **a video forward runs at
2600 MHz and ~149 W**, so its ceiling is 49.9 TFLOP/s and the shipped
GEMM is at 74% of it; **32 busy CPU threads halve the GPU clock** (a
forward 1.67x slower; VIDEO.md M11f's mechanism). The GEMM's loop has no
prefetch across K tiles, its rate is flat from 1 to 3+ workgroups a CU,
and with every load an L0 hit it reaches 82–85% per clock: ~10 points
memory, ~16 in-wave schedule. **G2 done 2026-09-29** (§2.9): the prefetch
that fits is the K slab staged through LDS at wave32
(`dit_gemm_wg128x256_lds_w32`), bit-identical, 74 → 80% per clock, and
**G8 carried it to all nine hosts** the same day (it loses under ~40
workgroups; `ace/dit` keeps the wave64 build up to 1024 rows). **G-o8
done 2026-09-29** (§2.10): every VALU instruction costs the matrix pipe
a clock on this part, the allocator's ~40 moves a tile were 7% of it, and
the K tiles kept as a loop take the GEMM to **85% per clock** (H3's
projections 43.5 TFLOP/s, a 480p forward 26.7 s), bit-identical again.
Left in the GEMM: the barrier's lockstep, ≤ 9 points. **G3 done
2026-09-30** (§2.14): the fp32 C store costs the matrix pipe nothing (a
store-free control runs the same per-clock rate; its +9% is clock), the
grid tail swings ≤ 5% past ~1300 rows, and the two H3 passes that read
the fp32 C are the GEMM's epilogue now — v stored as the attention's
fragment tiles, gate|up as one GEMM ending in the SwiGLU — bit-identical,
**a 480p forward 26.2–26.3 → 25.6 s**; the served binary needs a redeploy
(the int8 cache restages once). **Carried (G8) the same day**, every one
bit-identical to what it replaces: the image DiT's SwiGLU (a 1024² step
1.04–1.06x), the music DiT (a forward 1.07–1.10x: 30 s 120 → 111 ms,
4 min 803 → 745, 10 min 2260 → 2124) and the video VAE (**a 480p decode
34.4 → 30.0 s, 768p 64.3 → 56.0 s, 1.15x**); the speech encoders priced
and left (2.3% of a 13.5 ms encoder). **G6 done 2026-09-30** (§5.4): the
streaming kernels' shortfall was the dispatch grid's fast axis — the
workgroups in flight together are neighbours along X, and where a step
along X is a whole number of the 4 KB DRAM channel rotation they all load
the same channels. The LLM's hyper-connection kernels walked
stream-fastest and the DiT family's q/k packs head-fastest, bit-identical:
`hc.norm` 183 → 217 GB/s, the packs 93–151 → 173–209, **LLM prefill 1520 →
1538 tok/s at 2048 and decode 36.87 → 37.35 tok/s, a 480p VAE decode 29.95
→ 27.5 s, a video forward 25.5 → 25.2 s, a 10-minute music forward 2113 →
2050 ms, an image step 1.895 → 1.886 s**. **G4b done 2026-09-30** (§3.9):
the transposed attention's K and V blocks fetched once by a workgroup of
eight waves and shared through LDS, bit-identical, the default in
`h3/dit`: the kernel 1.10x at 480p and 1.115x at 768p, **a 480p video
forward 25.2 → 24.3 s, 768p 95.9 → 92.5 s**. Next: G7, or the rest of
G6 (gate+norm fusion; kev, ocr and zimage unprofiled).

**Music (2026-09-27): ACE-Step 1.5 XL turbo — closed the same day.** A
caption and lyrics (or an uploaded song) become 48 kHz stereo; see the
music section below and the archive
[`research/music-vertical.md`](research/music-vertical.md).

**OCR (opened 2026-09-27): PaddleOCR-VL-1.6** (`GOALS.md` item 10). The
live plan and handoff is the root [`OCR.md`](OCR.md) (O-stages): element
recognition first (the 0.9 B VLM behind `/v1/chat/completions`, which
PaddleOCR's own pipeline can drive unchanged), then page parsing with
PP-DocLayoutV3 and the glue in Go (`/v1/ocr`, Mistral's shape). O0–O10
are done: every element case token-identical to fp32 HF, the glue PaddleX's
byte for byte, and **OmniDocBench v1.6 on a 331-page subset: the Go
pipeline 96.13, PaddleX's own pipeline on our engine 96.14 with the card's
text edit (0.0326)** (card 96.34 on the full set). O11a batches a page's
regions over a paged KV cache: the subset at the same score in 18 min
instead of 66 (3.0 s a page mean, worst 143 → 18 s). Open: O11b (towers,
the long tail), reading order through HF's layout port (O-o5).

**Image, music and video share one swap slot (2026-09-27, `backend.Swap`,
`API.md` *Residency*).** A request loads its vertical and unloads the
other; `-swap-idle` (10 min) empties the slot. Served with the unit's full
flags through image → music → video → image: 9.4 GB at rest with nothing
staged, 60 GB at the worst moment (all three resident was ~112 GB), a switch
costs ~25-30 s before an image and ~19 s before a song, and speech is
answered through every load (loads take no device lock). This is what lets
`-image -edits 3 -video -music` go back into the unit; deployed (the unit
runs it, and the 2026-09-29 18:40 start logs the swaps).

**The server** (`API.md`): one process, one flag per vertical, OpenAI-shaped
(`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/audio/*`,
`/v1/images/*`, `/v1/embeddings`), plus a Wyoming door for Home Assistant
(`-wyoming`, byte-identical audio to the HTTP door). Everything unimplemented
is refused with a reason, never faked.

**Deployment is two machines**: the language model alone on one (~84 GB
resident), the other four verticals on the other (image ~32 GB, the rest
~3 GB together). `-llm` and `-image` do not fit in one 128 GB process, on
purpose — so no footprint quantisation is planned for the small verticals.

**Where the work goes next.** **P18 (2026-09-22): the full trained context.**
The server now holds all **262 144** cells the model was trained for (it capped
at ~148k: the KV cache shared a 4 GiB descriptor range with the arena). The
cache has a buffer a plane. Perplexity is 4.0344 over positions 131k-262k, and
a 237k-token prompt through HTTP recalls its first line at 1022 tok/s prefill
and 29.9 tok/s decode. `ai.service`'s LLM line is `-llm-ctx 262144 -llm-batch 8192`
([`research/p18-full-context.md`](research/p18-full-context.md)).
**P17 (2026-09-22): heads on the fragment's
rows.** A 128 000-cell prefill at ubatch 2048 goes **833.5 → 1011.4 tok/s**
(64 000: 896.4 → 1048.6), and decode at depth gains ~2%, exactly
([`research/p17-heads-on-rows.md`](research/p17-heads-on-rows.md)). The QSA
selection is per token and twelve query heads share each kv head, so the
gathered attention now puts a token's heads on the fragment's M axis and reads
that token's 2 051 cells instead of a sixteen-token union of 7 049; the mask
dispatch is gone. P17-2 faults the next prompt chunk's n-gram pages in while
the current one runs (+3% at depth). At the served ubatch 4096, 128k goes
**899.7 → 1175.7 tok/s**. The attention's depth terms are 0.14 ms of a 0.99 ms
prefill token at 128k; what is left of the falloff is the flat floor. **Open
for decode:** the host n-gram gather is ~1 ms of every token (16 dependent
major faults; residency of the 29 GB table is a deployment decision), and
its median step is 236 µs and two steps in 32 take ~8.7 ms at fixed token
positions with ordinary fault counts, which is 75% of the line and needs a
kernel-side trace to explain. **P17-3** takes decode's `attn.select` at 128k
from 54.0 to 43.8 µs a layer (keys in registers, 10-bit digit, exact): decode
at 128k 32.93 → 33.16 tok/s.
**Long-context prompt processing is closed at
the target.** **P14 (2026-09-22)** takes a 128 000-cell prefill from P13's
**563.6** to **826.0 tok/s at ubatch 2048** and
**946.1 at 8192**, which is the 900 tok/s `GOALS.md` asked for
([`research/p14-prefill-at-depth.md`](research/p14-prefill-at-depth.md)).
Four changes: the QSA selection runs over **block** scores with a per-block
weight instead of the expanded per-cell tensor, which deletes a dispatch and
1.14 GB of arena and is 13.8x on `attn.select`; the attention kernel runs over
a **per-cell gather** — the union of a query tile's rows' selections, compacted
— instead of over the key axis, which is 2.15x on `attn.attn` and needed the
value plane to become cell-major first; the indexer's score moves onto the
**matrix cores**, 2.3x; and `maxStorageBufferRange` gets an **error** instead of
a comment, because exceeding it makes a run *faster* and wrong. Beyond that,
each vertical's list is in its own rough order of value. The context-depth regression that
stood at the head of this list is **closed** — **P7**, **P8**, **P9** and **P10**, all
2026-09-21, take decode at 64k from **4.05 to 27.69 tok/s (6.8x)** and the
falloff from depth zero from **6.9x down to 1.3x down** (0.14x → 0.79x of the
depth-zero rate). P7 was three kernels that walked the *cache* rather than
the *context* plus a QSA selection that was a mask and never a skip
([`research/p7-context-depth.md`](research/p7-context-depth.md)); **P8, P9 and P10
are one finding**
([`research/p8-decode-attention-split.md`](research/p8-decode-attention-split.md))
— **at decode this model's kernels are single waves that all fit on the
device at once, so a dispatch costs one wave's serial walk and neither its
traffic nor its total work.** The probe: cutting the attention grid from 24
workgroups to 2, a twelfth of both, measured **1.00x at every depth**. P8
splits the attention's key axis across workgroups, P9 unpins the indexer —
which was scoring the whole context on one compute unit of forty — and P10
widens the selection, the one kernel that cannot be split at all because its
radix passes are a reduction, from four waves to sixteen.

**P16 (2026-09-22): the wide batch never cost decode anything.** Decode is
**1.16x at every depth** — 29.73 → **34.4 tok/s** at depth zero and 27.20 →
**31.5** at 128 000 cells, ubatch 2048 — and through the server it is **34.2
tok/s at `-llm-batch` 2048, 4096 and 8192 alike**, where P12-7 measured 28.04
and 25.81 ([`research/p16-decode-arena-width.md`](research/p16-decode-arena-width.md)).
The cost P12-7 and P15 priced for a wide batch, and could not explain, was
**`DeltaNetGPU.InPort` and `AttnGPU.InPort` advertising the arena's rows**:
every decode step zero-filled the whole prefill arena of both blocks' A
operand, once a layer, and the qkv projection after each move stalled behind
the writes draining. `cmd/llm -depth -ubatch` found it by staging wide arenas
and prefilling narrow — the cost followed the arena, not the prefill — and the
per-label diff put all of it on `move`, `dn.qkv` and `attn.qkv`. The port now
pads to the row block, as the MoE's always did, and
`TestInPortPadsTheRunNotTheArena` asserts it, because the old padding was
*correct* and no tolerance could see it. Two smaller exact changes ride along:
**`hc.cn` is a workgroup a (token, stream) again at ≤ 64 rows** (25.0 → 7.0 µs,
1.06x of a token — P11's workgroup a token is one workgroup on forty CUs at
decode), and **`attn.select`'s emit walks blocks rather than cells** (68 → 55
µs at 128k). `-llm-batch` stays 4096; 8192 is now purely a memory decision.

**Decode's side of the depth question is now closed too. P15 (2026-09-22)**
takes a decode step at 128 000 cells from **22.58 to 27.20 tok/s** and the
falloff from depth zero from **0.76x to 0.91x**, with prefill and depth-zero
decode flat as the controls
([`research/p15-decode-at-depth.md`](research/p15-decode-at-depth.md)). Three
changes, and the largest was not on the device: **`PLERows` hashed the whole
sequence on every token** to use sixteen of its rows, which at 128k is 8.2 MB
allocated and 2.05 M rows per step — **5.18 ms of a 44.3 ms token and 39% of
the whole falloff**, deleted exactly by `PLERowsFrom`. Then **`attn.score`
was striped sixteen ways on a forty-CU device**: P9 unpinned it and stopped at
16, the stripe is a grid and not a reduction, and 64 is **2.61x** (211.6 → 80.9
µs) and better at *every* depth. And **the gather and the split compose, which
P14 said they would not** — that argument is right about the gather alone and
is measured (unsplit gather 986.7 µs against the split's 419.8), but the two
cut different things: the split is 7.12x on the *walk* and the gather 2.99x on
the *work*, so `GATHER`+`SPLITK` together is **3.23x** (419.8 → 129.9). A
decode tile is one real row, so the union that costs prefill 4.9x costs decode
nothing. It turns on at `2*selWidth` **live** cells, which makes it the first
decode knob that is a function of the depth — so P1c's prerecorded buffer now
carries an epoch and re-records the one step that crosses. The refusal:
**the arenas' HOST_CACHED memory type costs the kernels nothing** even at the
3.85 GB the KV planes now put in that buffer — every kernel within 1% under
`LLM_ARENA_UNCACHED=1` while host glue moves 29x — which confirms L6b at the
new scale and spends a third hypothesis for `hc.cn`. After that the two capability
gaps are P6 batching (blocked on a product question: will the API serve more
than one stream?) and E7's batched embeddings, worth up to 10x on short
texts; the largest single-vertical percent is S10, the speech front end at
48% of its pipeline.

---

## Text generation (archive: [`research/llm-vertical.md`](research/llm-vertical.md), review: [`research/llm-review.md`](research/llm-review.md))

**Where it stands.** Phases 1 and 2 are done on both axes and every P-stage
through P5 is closed. The shipped configuration is **D19 + D20 + D21**
(4.5-bit dense bank with a fifth bit on three families, the MoE rows
transcoded, `ffn_down_exps` at IQ4_NL): 4.132 GB a token against a 58.6
tok/s ceiling, perplexity 4.0992 (+1.74% of our own 4.0289, which is itself
−0.13% against llama.cpp's at identical weights). Decode 36.19 tok/s,
prefill **1207.7 tok/s at ubatch 2048 and 1403.9 at 8192** (P11, P12). Served with
prefix reuse; a second turn extends the graph's state rather than
re-prefilling.

**Speculation (P5) is built, lossless, and parked at 0.95x.** The rollback
costs nothing when off. What would take it past 1.0, in order: the draft
head's **acceptance on a real workload** — 65.6% on prose against a
break-even of ~0.72; measure chat/code with the observer
(`cmd/llm -mtp`, ~3 min, no machinery) *before* building anything, and only
above a ≈ 0.8 does the rest get interesting — then the recovery round
(~0.95 → 1.05x) and pre-recording the verification pass (~2%). Hard
ceiling: speculation refuses past 2048 cells (`blk.48` has no compress
ratio). Narrowing the trunk makes speculation worse, twice — the draft
stages at checkpoint widths and acceptance falls as the trunk moves away
from what the draft predicts.

**The plan from here: token generation (set 2026-09-30, not started).**
Prefill has had eight stages and the KERNELS.md G-stages (1194 → ~1520
tok/s); decode got nothing from the G-stages, because they are all WMMA
GEMM work and a decode step is GEMVs at 91 W, bound by memory. Decode is
**36.2 tok/s at depth zero** (27.6 ms a step, `cmd/llm -batch 1`, after
C5's quad GEMV), 35.5 served, ~32 at 128k, against a **58.6 tok/s byte
ceiling** (4.132 GB a token): 62% of it. The one outside number
(`halogen-flash-server`, KERNELS.md §0.7) claims 37.6 serial, which we
match, and **46–56 with a draft head plus prompt lookup**, which is the
whole of their lead. KERNELS.md's "decode GEMV: 96–103% of the bus, done"
is a cache-hot ladder figure: in the model the weight streams measured 178
GB/s (P1) and ~170–200 (P16), and C5 found 6% by counting loads in
`llm_moe_gemv.comp` on 2026-09-23. What carries over from the G-stages is
the method: price a kernel's parts by their absence, count loads and
instructions in the ISA (`cmd/probe`), and screen in the model on a cold
bank.

A step's 27.6 ms, **reconstructed from P1, P16 and C5 and not measured as
one table**: ~17 ms of weights at the bus rate (the floor at this format),
~5–6 ms of weight streams running under the bus, ~3.3 ms of dispatches that
stream no weight (898 of 1431 are under 12 µs; the 196 moves are 0.54), and
~1–2 ms of host, mostly the n-gram gather. The stages, in order:

- [x] **P19 — re-attribute the decode step.** **Done 2026-09-30**
  ([write-up](research/p19-decode-attribution.md)). A step is **26.66 ms**
  (37.5 tok/s, two runs 0.15% apart, every label within 0.014 ms): **16.4
  ms of weights at 242 GB/s, 4.3 ms of streams under the bus (against 227),
  3.0 ms of weightless dispatches, 1.1 ms of host** — the reconstruction
  above was right. The weight streams average 182 GB/s; the four big dense
  GEMVs are at 195–211 (0.8 ms lost between them, and the Q4 arm's loads
  are already vectorised by the compiler — one `b128`, one `b64`, two
  `b128` a step), the MoE at 193/174 (1.3 ms), **and the two projections
  still on the padded GEMM at one row are the largest single losses: the
  lm head at 185 GB/s (2.37 ms, the biggest dispatch of the step) and
  `hc.up` at 125 (1.75 ms over 97 dispatches)**. The small dispatches
  (`hc.up`, `hc.down`, the shared expert, `ple.kv`: 340 a step of 1–2.3 MB)
  run at half the bus, which is P8's latency rule at family scale. Clock
  under decode: 97 W, shader clock p50 795 MHz — memory-bound, no matrix
  clocks to buy. A two-row one-sequence pass is **1.23 steps** (P5c's 1.36),
  three rows 1.37; three conversations 1.39 / 1.67. **First change, same
  hour A/B: the head at one row on its GEMV (`llm_gemv.comp` K1, staged
  since L8c and never wired), 2.37 → 2.06 ms at 212 GB/s, the step 26.66 →
  26.37, 37.5 → 37.9 tok/s (+1.1%)**; `GraphOpts.HeadGEMM` / `LLM_HEAD_GEMM=1`
  is the control, four bit-identity gates pin the GEMM, and
  `TestGraphHeadDecodeGEMV` prices the kernel (rms 7e-6 of the logits' max).
  **At one row only, until the same night's postscript**: a three-slot
  batched step ran the GEMM head and rounded unlike its solo steps
  (`TestGraphImageDecode`); the head now plans the GEMV to GEMVMaxRows
  like every other decode GEMV, and the gate covers three rows.
  **P21a done the same evening: `hc.up` on a GEMV** (`llm_hc_up_gemv.comp`,
  a workgroup a feature block, five waves each a share of the column's
  k-tiles, the four-stream collapse in the wave; `LLM_HC_UP_GEMM=1` the
  control): 17.9 → 13.3 µs a dispatch, 125 → 169 GB/s, the step 26.10 →
  **25.8 ms, 38.7 tok/s** (five interleaved arms 25.78–25.88). **P21b
  measured dead**: the big GEMVs with their slab's loads issued together
  (17–23 in flight, bit-identical) ran 0.7 ms *slower* — the head +0.3,
  `hc.down` +0.23 — for 60–72 VGPRs against 48 and a branch-free record
  decode at 800 MHz; removed. The four big GEMVs are not load-bound.
- [ ] **P20 — speculation, re-planned 2026-10-01 against llama.cpp's MTP**
  ([research/p20-llamacpp-mtp.md](research/p20-llamacpp-mtp.md)). The only
  lever with a multiple, parked at 0.95x (P5c). llama.cpp PR 29761 (open,
  2026-09-30) implements the same head with **P5a's wiring exactly** and
  reads **1.55x on a DGX Spark at `--spec-draft-n-max 3`**: 28.36 → 43.88
  tok/s over 24 SPEED-Bench prompts, acceptance **0.64 a drafted token**
  (coding 0.61, qa 0.58, rag 0.69, writing 0.67), ≈2.9 tokens a round for
  ≈1.9 steps. Four things in its loop differ from ours, and one of the four
  is a defect here:
  - **Done 2026-10-01: P20a and P20d's first half** ([§6 of the
    note](research/p20-llamacpp-mtp.md)). The draft is primed over the
    prompt from `Graph.ForwardResidual` and catches up one owed cell a
    round (`Speculator.CatchUp`, `SPEC_CATCHUP=0` the control); `eh_proj`
    runs on the device and the draft step is one submit (`MTPHead.eh`,
    `LLM_MTP_HOST_NEXTN=1` / `LLM_MTP_UNRECORDED=1` the controls,
    `TestMTPSeedDeviceIsTheHost`). Acceptance at 1024 prompt cells 68.2 →
    90.6% (memorised wikitext: an upper bound), unchanged at 34; the draft
    5.6 → 3.5 ms a round. **`-spec` 1.00x → 1.29x long (41.70 → 53.95
    tok/s) and 1.08x short (42.00 → 45.26)**, two passes each, acceptance
    count for count the host path's. Spec leaves plain at one near-tie a
    run (two-row rounding; recorded, not fixed).
  - **Done 2026-10-01: P20c at depth one** ([§7 of the
    note](research/p20-llamacpp-mtp.md)). Row 0 of a pass is always
    committed, so the pass also folds row 0 alone into the slots it *read*
    (the scan's in-place store of S after token 0, the ring write's row 0
    into `SEQ_HIST_PREV`; flag `SEQ_KEEP_FIRST` = `ldaLo`), and a rejection
    is `Graph.KeepFirst` — at P+1, drafting again, no recovery round, no
    extra memory (`Graph.SpeculateFirst`, `Speculator.Partial`,
    `SPEC_PARTIAL=0` the control). Gate
    `TestSpeculationKeepFirstIsTheSequence` bit-exact, control rms 0.93.
    **`-spec` short 1.08x → 1.19x (42.00 → 49.82 tok/s), long 1.29x →
    1.31x (41.57 → 54.65)**, both arms self-reproducing. Covers keeping
    row 0 of a pass up to three rows; keeping 0–1 of three needs a stored
    plane (built in P20b below). Then P20e (SPEED-Bench, the number to
    quote).
  - ~~**P20a — the draft's KV cache is complete there and not here.**~~
    llama.cpp runs the draft layer over every row the trunk runs — the
    prompt and each verified batch — paired with the trunk's residual one
    position back, no lm head, and trims the rejected rows.
    `Speculator.Start` resets the draft and never primes it over the prompt,
    and `Next` writes a draft cell only on a speculating round, so the draft
    attends over **unwritten cells for the whole prompt and at every other
    position**. P5c's 65.6% / 46.9% against P5a's 74.0% (same head, complete
    cache) is this before it is the narrowed trunk — §4.2 was confounded with
    it. Fix: the catch-up rows in the draft dispatch (the committed rows
    since the last draft plus the new one, logits for the last), the prompt
    primed at `Start` from the prefill's residual arena a ubatch at a time
    **on the device** (at 128k the residual is 5 GB; not through a mapped
    buffer). Gate: the loop's acceptance on a teacher-forced sequence equals
    the observer's count for count.
  - **Done 2026-10-01: P20b — depth two built, measured, not adopted**
    ([§8](research/p20-llamacpp-mtp.md)). Keep-any-prefix up to three rows
    (`GraphOpts.SpecRows`, mid slots, `Graph.Keep(k)`,
    `Speculator.Depth` / `-spec-depth`, `SPEC_TRACE`), gate bit-exact with
    a mid-slot sabotage that fails. **Depth 2: short 1.11x against depth
    1's 1.19x, long 1.31x = 1.31x**: the 3-row pass is 1.52 steps against
    1.29 (the MoE's distinct experts, 17 → 24, not a kernel) and the second
    draft step +0.15; a₂|a₁ 62.5%. Depth 1 stays default. Next lever: the
    draft's lm head (2 of 3.7 ms) over a trimmed frequent vocabulary
    (FR-Spec) — price its acceptance on `-mtp` first; then P20e.
  - **Done 2026-10-01: the draft's lm head over a vocabulary prefix**
    ([§9](research/p20-llamacpp-mtp.md)): the draft proposes from the first
    65 536 ids (BPE id order covers 96.6% of wikitext, 98.3% of Go; counted
    wikitext frequencies fail on code), and `HeadGPU.RunCols` runs the
    trunk's own head GEMV over K/16 tiles — no new weights or kernel
    (`TestHeadGPURunCols` bit-exact). Draft 3.71 → 2.22 ms; **`-spec` short
    1.19x → 1.24x (51.93 tok/s), long 1.31x → 1.32x (55.14)**; acceptance
    count for count the host-cut pricing. `SPEC_DRAFT_VOCAB=0` the control.
    Next P20e (SPEED-Bench, the number to quote), and serving the loop.
  - ~~**P20b — depth 2 or 3, not 1.**~~ P5a's optimum was computed on
    2026-09-19's pass costs and wikitext acceptance; at 0.64 a drafted token
    and P19's 1.23 / 1.37-step two- and three-row passes the optimum moves.
    Measure `cmd/llm -graph -tokens 1,2,3,4` on the shipped banks (one
    sequence, the right instrument here); a four-row pass needs
    `GEMVMaxRows` 4 (`MAXROWS` in the three decode GEMVs and the fused
    router, a four-at-a-time rung in `TestGraphIsAChunkSplit`). The MoE's
    expert growth bounds the depth; P20b measures it on loop traffic rather
    than the design pass's repeated-prompt table.
  - ~~**P20c — partial accept by snapshot planes, not a recovery round**~~
    (done at depth one above, by an in-place row-0 store rather than planes)
    (P5c §4 item 1, now with a reference design): llama.cpp creates the
    target context with `n_rs_seq = n_max`, the gated-delta-net op writes
    the state after each of the last K = n_max + 1 rows into K planes (the
    conv ring and the PLE ring likewise, K copies), and a rollback of j sets
    a plane index the next read takes; row 0 of a pass is always committed,
    so no ping-pong. Here: `llm_dn_scan.comp` stores S after each of the
    last K rows ((K − 1) × 113 MB extra a pass, ~0.07 step at K = 4, against
    a recovery pass of ~1.2), K rings in `llm_seq_hist.comp`'s ping-pong
    arm, `Rewind(j)`; residency 241 → 483 MB, still `GraphOpts.Speculative`;
    `TestSpeculationRewindIsTheSequence` gains j = 1..3 of 4.
  - **P20d — the draft step at its byte floor.** 5.71 ms is 0.236 of
    today's 24.19 ms step (P5c §4.2's effect again) against a floor of 0.13;
    three a round is 0.71 steps against 0.39. `nextn` on the device (four
    host matvecs, 2.2 ms) and the draft step recorded (13 submits) — P5a's
    owed items, on the critical path now — and depth as a parameter.
  - **P20e — the measurement, the PR's way**: free-running on SPEED-Bench's
    qualitative prompts (`nvidia/SPEED-Bench`, the categories above) through
    the chat template at temperature zero, the text read for P5a's loop
    artefact, two runs, the plain-against-plain row printed (`Reset`'s owed
    test, P5c finding 6, first), the shipped banks. The converter in the PR
    gives `blk.48` **the trunk's compress ratio**, so the 2051-cell refusal
    in `NewMTPHead` comes out and the long arm runs past 2048; D5 is back
    for the head once a build with the PR exists. Unverified arithmetic at
    24.19 ms, the PR's acceptance and the draft at the floor: depth 1
    **1.26x**, depth 2 **~1.4x**, depth 3 **~1.5x if a four-row pass is ~1.5
    steps** — 52 / 58 / 63 tok/s, inside the 46–56 the outside number
    claims; with P5a's wikitext profile instead, 1.34x / 1.32x at depths 2
    and 3, so the workload's acceptance is the swing.
  - **P20f** prompt-lookup drafting, demoted: a free extra draft to combine
    with the head (llama.cpp runs several types in order), not the first
    thing; the host-only replay observer stands as written before, and the
    draft's confidence stop (`p_min`, unused in the PR's table) is a lever,
    not the result.
- [ ] **P21 — the G-treatment on the decode GEMVs.** P19's `lost` column
  sets the order; the first suspect is retired (the Q4 arm's loads are
  vectorised as compiled) and so is the second (P21b above: issuing a
  slab's loads together is slower on the big grids). ~~P21a `hc.up`~~ done
  (above, +0.44 ms). ~~Left: **P21c `moe.down` (174 GB/s, +0.6 ms) and
  `hc.down` (162, +0.4)**~~ — **P21c done 2026-09-30** ([Finding
  5](research/p19-decode-attribution.md)): both rungs re-laddered in the
  whole model (`LLM_MOE_DECODE_PLAN`, `LLM_HC_DOWN_SLABS`) and the shipped
  ones confirmed (v16w4; 32 slabs, with 40 D12's whole multiple at 1.9x and
  160 losing on its reduce); the ISA showed `moe.down`'s lane as five serial
  round trips, each two loads and a `vmcnt(0)`, behind a half-a-load A
  staging — `llm_moe_gemv.comp` now issues every trip's words before the
  first unpack (the guards branch-free, or the compiler sinks the loads
  back) and stages A eight halves a load: three interleaved pairs 25.57 /
  25.50 / 25.56 → **25.09 / 25.13 / 25.17 ms, 39.1 → 39.8 tok/s**,
  `moe.down` 53.3 → 49.2 µs, `moe.up` 104.5 → 100.4 from the staging alone.
  The same treatment on the up mode's quad loop measured +1.5 µs and was
  reverted (P21b a third time). `hc.down`'s 0.4 ms has no idea left; the
  shared expert's two dispatches are P22a's eleventh tile. **P21d done
  2026-10-01** ([Finding 8](research/p19-decode-attribution.md)): `ple.kv`
  was the third projection still on the padded GEMM at one row — the PLE's
  [2560 × 12800] int8 key/value projection, 35 MB at 127 GB/s — and runs on
  `llm_gemv.comp`'s eight-slab rung at one to three rows (`PLEGPU.gemvFor`,
  `LLM_PLE_KV_GEMM=1` the control, `LLM_PLE_KV_SLABS` the screen,
  `TestPLEGPUKVDecodeGEMV` the gate, the schedule pin honoured): 282 → 167
  µs, −0.12 ms a step. Method: knock-out controls, the ISA's load count, the whole model
  and not the block ladder (L8e-2 and C5 both chose a rung in the graph
  that the ladder got wrong), a rate above 242 GB/s is an L3 hit (D16),
  and **VGPRs and VALU count at 800 MHz** (P21b's lesson: a decode kernel
  runs at a third of the clock, so an instruction costs three times what
  the ISA suggests).
- [ ] **P22 — the weightless dispatches and the small ones.** 3.0 ms of
  weightless dispatches (P19's list: `dn.scan` 0.64, the 196 moves 0.54,
  `moe.route` 0.28, `hc.cn` 0.23, twenty more under 0.16), 1–2 recoverable:
  G3's epilogue pattern (a norm, gate or combine folded into the kernel that
  produces its input) and L6c's single arena, which deletes the moves.
  ~~And **the shared expert as an eleventh tile of `moe.up`**~~ — **P22a
  done 2026-09-30** ([Finding 4](research/p19-decode-attribution.md)):
  `llm_moe_gemv.comp` carries the shared expert as one tile past the routed
  schedule wherever its bank is the routed bank's format (47 of 48 layers'
  up on the shipped plan; layer 2 is Q5_K and the shared down is Q5_1
  against IQ4_NL), `MoEGPU.foldShared`, `LLM_MOE_SHEXP_SPLIT=1` the control,
  `TestMoEGPUSharedFold` the gate (rms 8e-9 against the split, one to three
  rows, arenas dirtied between arms). Same-hour interleaved pairs 25.75 /
  25.74 → **25.56 / 25.51 ms, 38.8 → 39.2 tok/s**: the shared up is 9.2 µs
  inside the routed dispatch against 15.0 on its own, which is the routed
  rate. The down half is priced at ~0.15 ms more (25.31 with
  `down_shexp=iq4_nl`) and waits on P24's eval of that width. Check every
  re-gridded kernel at `rows == 1` (P16's `hc.cn` lesson). **A fold-order
  bug found and fixed the same night** ([postscript](research/p19-decode-attribution.md)):
  with the up split and the down folded — the checkpoint's own formats,
  never the shipped plan — the folded shared-down tile read the shared
  swiglu rows before the split `shexp.up` had written them (a decode step
  0.137 rms wrong on the default banks). `shexp.up` is recorded before the
  routed down now and `TestMoEGPUSharedFold` runs the mixed arm.
  ~~**The 196 moves**~~ — **P22b done 2026-09-30** ([Finding
  6](research/p19-decode-attribution.md)): deleted at the boundary rather
  than with L6c's shared arena. The kernel on either side binds the other
  block's arena at binding 11 (`HCLink`, a pipeline per foreign buffer as
  the mover had): `cn`/`combine` read a sublayer's output where it lies at
  any row count, and the decode up GEMV writes the next sublayer's fp16 A
  operand from the register that holds the value. 196 → 4 moves a step
  (the PLE's residual out and back, the final row move, the head's input);
  bit-exact by construction, `TestGraphHCLinkBitExact` the gate,
  `LLM_HC_MOVES=1` / `GraphOpts.HCMoves` the control. Three same-hour pairs
  25.13 / 25.15 / 25.13 → **24.83 / 24.83 / 24.81 ms, 39.8 → 40.3 tok/s**;
  a third of the 0.53 came back as `hc.up` zeroing the consumers' pad rows
  from one wave, so every `InPort` now reports the run itself as its row
  block on the decode plan (its rungs are GEMVs that read ROWS rows) and no
  zero row is written at decode by either arm.
  Three more pairs with both arms so: **24.91 / 24.89 / 24.97 → 24.42 /
  24.39 / 24.43 ms, 40.1 → 41.0 tok/s** — the zero rows had cost the moves
  arm 0.22 ms on their own in the kernels after each move, so P22b is
  **25.13 → 24.42 ms, 39.8 → 41.0 tok/s** in all
  (`results/p22b_link_*.csv`).
  ~~The epilogue fusions~~ — **P22c done 2026-10-01** ([Finding
  7](research/p19-decode-attribution.md)): which small dispatch folds into a
  neighbour is decided by the neighbour's parallelism, and `hc.cn` (one
  workgroup a stream, 2560 values in registers) would read the combine's
  eleven rows or a reduce's 32 slabs four times over on one CU each, so the
  combine and the split-K reduces stay (priced dead on paper). The router's
  tail folds: `llm_moe_route_decode.comp` is the split-K reduce, the
  softmax and top ten (a wave a token, no barriers), the weights and the
  permutation with its schedule as one workgroup at one to three rows,
  **bit for bit** the three kernels' output (`TestMoEGPURouteFused`,
  `TestGraphRouteFusedBitExact`; `LLM_MOE_ROUTE_SPLIT=1` the control; one
  build a router rung, because the first form's runtime slab bound made the
  reduce forty serial round trips and measured 26.8 µs against 9.9): 0.471
  → 0.383 ms, −0.09 ms. With P21d, three same-hour pairs 24.37 / 24.43 /
  24.43 → **24.16 / 24.19 / 24.19 ms, 41.0 → 41.4 tok/s**
  (`results/p22c_{ctl,new}{A,B,C}.csv`).
  Left of P22: `dn.scan` (0.64 ms) is the fp32 recurrent state read and
  written every layer — bytes, P24's fp16-state item, not scheduling. The
  rest of the weightless list is a floor of launches (8 µs for a dispatch
  that waits on forty loads and ten reductions, 1–3 µs for one that does
  nothing), and only fewer passes — P20 — gets under it.
- [ ] **P23 — the host gather.** ~1 ms a token, 75% of it two steps in 32
  at ~8.7 ms, unexplained. The fix on the table is residency of the 28.8 GB
  table (or of the rows a conversation can reach), a deployment decision
  beside a ~98 GB LLM line; price it before asking.
- [ ] **P24 — fewer bytes.** ~+0.6 tok/s per 0.1 GB at today's efficiency.
  Gated on the downstream task eval below, since wikitext perplexity at
  +1.74% may no longer separate plans; fp16 DeltaNet state (+~1.3 of
  ceiling) is graded by the same instrument.

P19, P21a, P22a, P21c and P22b are done (36.2 → 41.0 tok/s on 2026-09-30,
24.4 ms a step), and P22c with P21d (2026-10-01: 41.0 → **41.4 tok/s, 24.2
ms**). P22 is closed but for `dn.scan`, which is P24's; **next is P20**
(P20a/c/d-first-half done 2026-10-01: `-spec` 1.19x short, 1.31x long —
49.8 / 54.7 tok/s; P20b's depth two measured and not adopted, the third
row is MoE bytes; the draft's head over a 64k-id prefix took short to
1.24x / 51.9 tok/s; next P20e, then serving it),
re-planned on 2026-10-01 against llama.cpp's MTP PR (the draft's KV cache
was never primed here — P20a — and depth 3 with snapshot planes reads 1.55x
there), because the step is now 24.2 ms
against ~19.6 of weights at the bus and ~1.1 of host, and the 3.5 between
is launches and small streams no fusion left on the list recovers. P21–P23
together were priced at 38.7 → 42 tok/s and have delivered 41.4; only P20
reaches the 46–56 range, and P19 re-priced its two-row pass at 1.23 steps
(break-even 23% acceptance for a free draft). Batched throughput
(65 tok/s at three rows) keeps its own list in CONCURRENCY.md *Next*. The
instrument rules at the end of this section apply to every number: a
same-hour control, the shipped banks and `LLM_BANK_CACHE`, nothing else on
the machine.

**Open before this plan, in rough order of value** (the decode items here
are folded into P19–P24 above):

- ~~**Prompt processing.**~~ **P11, closed 2026-09-21**
  ([write-up](research/p11-prefill.md)). Two changes and three measured
  refusals. **The server prefilled in 512-token chunks and 512 is the worst
  rung this graph has**: llama.cpp plateaus at its best ubatch and this one
  does not, because the MoE's arithmetic intensity is the *routing's* — at 512
  tokens 274 of 512 experts are unpacked whole to serve 5120 rows. `-llm-batch`
  is now **2048**: 1.12 GB of arenas for **1.64x** (667.4 → 1093.3 tok/s at 48
  layers, and 667 → **1053 through HTTP** on a 4128-token prompt), where the
  1.49 GB after it buys 13% more. And **the grouped GEMM's gathered A operand
  was loaded one half at a time** — 32 two-byte loads a lane a K-step against
  the eight the B unpack issues for four times the data — so binding the halves
  arena a second time as `uvec4` is **1.25x on `moe.up`**, the largest kernel in
  a prefill, bit for bit the same slab. The graph: **1079.6 → 1165.1 tok/s at
  2048 rows and 1250.1 → 1328.5 at 8192 (3.40x llama.cpp)**, two runs agreeing
  to 0.2%. What did **not** work, all three reverted and all three the same
  answer: a 16-row alignment with a row count per record (13.7 ms against 11.9),
  three sub-lists one dispatch a rung (12.5), and a 128-row block (13.5). **At
  prefill this GEMM is bound by its per-tile slab unpack and the row padding is
  very nearly free** — L5b built that padding as a cost to justify and it is not
  one, so the way in is fewer *tiles*, which is a bigger ubatch.
  **A second round closed `hc.cn`** — the combine fused with the next mixer's
  norm, the second largest kernel in a prefill and one that does no arithmetic:
  the four streams of a token were each reading the *same* block output row, so
  one workgroup a token instead of one a (token, stream) is **1.37-1.44x** on
  the kernel and bit-identical, taking the graph to **1191.8 / 1372.2 tok/s
  (3.51x)** and 64k prefill to **654**. And it measured, for the first time,
  **how much of the key axis the QSA selection lets the attention kernel skip**
  — at 64 000 cells the shipped 16x32 tile keeps 25.9% of pairs live against a
  14.7% floor for a single query row — which priced and then refused a k-tile
  skip inside the kernel (**1.18x slower, and at 64k slow enough to trip P0's
  ring watchdog**). Two kernels now say the same thing: **a branch inside an
  unrolled cooperative-matrix loop costs more than the work it removes**; change
  the loop's granularity, not what happens inside it.
- ~~**The attention row max, `attn.select`, `hc.cn`, the MoE unpack.**~~
  **P12, closed 2026-09-21** ([write-up](research/p12-prefill-round-two.md)).
  Two changes and four measured refusals; prefill **1.012-1.025x on both
  axes** (1191.2 → **1207.7** at 2048 rows, 1370.5 → **1403.9** at 8192 —
  **3.59x** llama.cpp — and 651.3 → **667.7** at 64 000 cells), both passes of
  each arm agreeing to 1370.5/1370.5 and 651.3/651.2. **The row max did run on
  sixteen lanes of sixty-four** and now runs on all of them: the whole key
  block staged at once and RCL = WAVE/BM lanes a row, folded by a clustered
  subgroup max, **bit-identical** because a max has no rounding and the cells
  scanned are the same — 1.16x on the kernel at 512 tokens, +1.6-1.8% of a
  prefill token from 8000 cells on. And **a K-quant block's header is one
  sixteen-byte load, not four dwords**, which is L5b-7's rule for the third
  time: `moe.up` 12 118 → 11 755 µs at 2048 tokens with `moe.down` flat as the
  control. **The four refusals are the valuable half.** Both of `hc.cn`'s
  named suspects cost **nothing** — deleting the 256-way tree outright is
  5331.5 µs against 5325.8, and dropping the `gamma` read is 5332.9 — so its
  57% of copy bandwidth is unexplained *and* out of hypotheses, and the
  non-bit-exact norm rewrite would have bought zero. The **block skip hoisted
  out of the loop is 1.00x at every depth**, and the reason is the exact
  converse of P8: a prefill dispatches 3072 single-wave workgroups where a
  decode step dispatches 24, so *latency that matters at decode does not
  matter at prefill, because prefill has occupancy* — it is still live for
  decode, where it was never built. The selection mask loop is **3.5-8%**, not
  the rest of the gap. And the MoE's **per-element affine is a floor**: the
  whole scale path is 1.21x at 2048 tokens, P12-2 takes its loads and integer
  ops, and the int→float convert, FMA and f16 convert that remain have no
  bit-exact cheaper form.
- ~~**The ubatch is the largest prefill number that is left.**~~ **Decided
  2026-09-21: `-llm-batch` is now 4096** (P12-7), and measuring it *through the
  server* found the half the graph ladder cannot see. Four distinct
  4.1-4.5k-token prompts, two interleaved passes agreeing to 0.5%: prefill
  **1020.9 → 1146.3 tok/s (1.12x)** and **0.46 s off the time to first
  token** — but decode **28.04 → 25.81 (0.92x)**. **The wider arenas cost 8%
  of decode**, across sixteen non-overlapping samples at both 8 and 256
  generated tokens, because a decode step streams 4.1 GB of weights a token
  and the extra 1.5 GB of arenas sits in the same memory. The two rates cross
  at **~150 generated tokens**: 4096 is 1.10x at 8 tokens and 2048 is 1.07x at
  the 1024 `-llm-max-tokens` defaults to. So 4096 ships as the interactive
  default and `API.md` says outright that a batch summariser should set 2048.
  **The lesson is P11-1's, a second time: a ubatch measured on the prefill
  ladder alone is a hypothesis about prefill.** **Superseded by P16**: the decode
  cost was a padding bug, and decode is now 34.2 tok/s at every batch.
- **`attn.select` at decode is 55 µs a dispatch at 128 000 cells** (P16-3,
  from 68), ~0.66 ms of a 31.7 ms token. A probe split it: ~8 µs a radix pass,
  ~12 µs of emit after P16 took it from 25, ~10 fixed. Two guesses are measured
  dead — the passes are **not** a chain of load latencies (four keys in flight a
  lane is 0.96x) and **not** LDS-atomic contention (a per-wave bucket fold is
  0.69x). What is left is P10's reduction, and the two-level arrangement
  (per-stripe histograms and a merge) is the only idea not yet priced; at ~2%
  of a deep token it is low on the list.
- **What is left of decode's falloff after P15 is ~2 ms of 36.8**, and the
  shape of it has changed: the gathered list is 2051 cells at *every* depth, so
  `attn.attn.split` being 1.571 against 1.092 at depth zero is no longer a cell
  count — it is those same 2051 cells scattered over a deeper cache, where each
  128-byte line costs more to reach. That is a locality question and not a work
  one, and nothing in this vertical has asked it yet.
- ~~**Why 1.5 GB of arenas a decode step never reads costs it 8%.**~~ **P16,
  closed 2026-09-22** ([write-up](research/p16-decode-arena-width.md)). It was
  not the memory: the block input ports padded a decode step's one row out to
  the arena, so a wider arena was more zeros written a layer. At an 8192 arena
  decode goes 23.55 → 33.87 tok/s; through the server decode is 34.2 at every
  batch. What remains between an 8192 and a 2048 sweep is depth — the wider
  prefill leaves a deeper cache — not width.
- ~~**Prompt processing at 128k.**~~ **P13, closed 2026-09-22**
  ([write-up](research/p13-long-context-prefill.md)). Two changes and one
  refusal. **128 000 cells completes**: `batchFor(rows)` was P0's fit and
  every measurement behind it was taken from cell zero, so at 64 000 cells a
  2048-row pass already held the ring for 1.53 s of the 2 s cliff while the
  budget believed it was spending 0.86 — the missing term is 10.46 ns per
  (row, cell), it is carried by the attention dispatches alone, and the
  chunker now walks the recorded sequence charging each dispatch its own
  label's cost and corrects itself against what the last submit measured.
  And **the key block was chosen for the dense regime**: L2f picked
  `qt1_kt2` on 512-token prefills from cell zero, where a 32-cell block is
  two cooperative-matrix tiles of reuse against one; at 128 000 cells the
  selection leaves 11.7% of (16-row tile, 16-cell block) pairs live against
  17.4% at 32 cells, and the narrow rung is **1.61x on `attn.attn`**
  (0.9107 → 0.5659 ms a token) for **471.7 → 563.6 tok/s**, with
  `attn.select`, `attn.score` and `attn.expand` identical to three decimals
  as the control. That is P11-7's 1.40x, collected as a *rung* after P11-7
  lost 1.18x trying to collect it as a branch. It is **prefill's rung and
  not decode's** — with it on both, decode at 128k went 22.94 → 22.27,
  because a decode step costs one wave's serial walk and halving the block
  doubles the walk. **The refusal**: compacting the live key blocks into an
  ascending per-query-tile list deletes 83% of the axis walk, costs 0.001 ms
  a token, and is **1.00x at every depth** — P11-4's finding a third time,
  *latency and redundant reads that matter at decode do not matter at
  prefill, because prefill has occupancy*. The machinery ships behind
  `LLM_ATTN_BLOCK_LIST=1`, bit-identical, because the decode split and a
  per-cell gather both want it.
- ~~**`attn.select` and `attn.expand` are 0.284 ms a token at 128 000 cells.**~~
  ~~**The gather.**~~ **Both closed by P14, 2026-09-22**
  ([write-up](research/p14-prefill-at-depth.md)). Four changes and four measured
  refusals; **128 000 cells prefills at 826.0 tok/s at ubatch 2048 and
  946.1 at 8192**, against P13's 563.6, with the falloff from depth
  zero **0.50x → 0.72x** and decode unmoved as the control.

  The selection now runs over the **block** scores with a per-block weight
  instead of the expanded per-cell tensor: `attn.select` **0.281 → 0.020**
  (13.8x, not the 4x the traffic argument predicts — the histogram is bounded
  at the last block a cell can reach and the LDS cache now holds `ratio` times
  as many cells), `attn.expand` deleted, **1.14 GB of arena** freed, and it is
  bit-identical over four chunk schedules. The attention runs over a **per-cell
  gather** — the union of a query tile's rows' selections, compacted ascending,
  with the causal test folded into a per-row mask — for **2.15x on `attn.attn`**;
  that needed the **value plane to become cell-major** first, which costs the
  block kernel 7% and decode nothing. And the indexer's score moved to the
  **matrix cores**, 2.3x.

  **The union was the number that priced all of it and P13's estimate was
  wrong.** `AttnGPU.SelUnion` measures it: at 128 000 cells a 16-row tile reads
  15 145 cells where its rows' union is 7 049 and one row selects 2 051 — so the
  gather is 2.15x, not 3.3x, and the 3.4x between the union and one row is the
  floor a sixteen-row fragment cannot reach.

  **The gather is the first kernel here that is not chunk-invariant and cannot
  be made so** (its list is the union of sixteen rows, so a chunk ending inside
  the tile gathers a different list and the softmax folds the same terms in a
  different order). It is pinned off under `PinSchedule` with the other
  reassociating kernels, and the exact chunk gates pin it;
  `TestAttnGPUGatherSelectsTheSameCells` is exact on the *set* and
  `TestAttnGPUGatherIsTheBlockKernel` is the tolerance (4.1e-06 rms against
  9.9e-04 from llama.cpp).
- **The gathered kernel is at 34% of matrix-core peak and all four bounds are
  eliminated**, which is P14's most reusable finding. Doubling its matrix work
  costs **2.7%**; a subgroup staging barrier instead of a workgroup one is
  1.00x; staging two head-dim groups a barrier is 1.00x and four is 0.75x;
  sharing the staged key and value across two query heads of one kv head —
  which halves both the global gather reads and the LDS writes — is **1.04x**,
  and four heads is 0.75x. So it is not the matrix cores, not latency, not the
  gather's traffic. What is left is the **LDS round-trip for the fragments and
  the per-head softmax scaffolding**, both linear in the union, which is why the
  kernel's cost is linear in the cells it visits. `LLM_ATTN_GATHER_GRP` and
  `LLM_ATTN_GATHER_HEADS` are the ladders and both ship at 1.
- **`maxStorageBufferRange` is silent when it is exceeded and `checkBufferRange`
  now says so.** A VkBuffer over the range is legal to create; binding more of
  it than the range is not, and the driver **clamps rather than failing**, so a
  kernel reading past it gets zeros. A five-depth `-depth` sweep at ubatch 4096
  raises `-ctx` to 148 520 cells, which is a 4.39 GB fp16 arena against a
  4.29 GB descriptor — and the run came back at **1124 tok/s at 128k against a
  true 885**, 1.28x too fast, with a degenerate indexer selecting a contiguous
  window. **The failure mode is a benchmark that gets faster**, which is the one
  direction nobody audits; the rule it leaves is to price a suspicious win in
  FLOP/s or bytes/s against the device's peak before believing it.
- **`hc.cn` is still at 134 GB/s** where a copy gets 236, and there are now
  **three** spent explanations, not two. P12-4 spent the 256-way tree (deleting
  it outright is 5331.5 µs against 5325.8) and the `gamma` read (5332.9). P15
  spent the **memory type**, which was the most plausible of the three and had
  never been tested: the arenas are HOST_CACHED and this kernel is pure arena
  traffic, so it looked like the whole answer — and `LLM_ARENA_UNCACHED=1`
  moves it 1272.2 → 1260.4 µs, inside the noise, while host glue moves 29x as
  the evidence the knob was really thrown. The next probe is the access pattern
  itself: it runs four read-modify-write streams plus an fp16 write a token,
  where a copy runs two. **Closed by KERNELS.md G6 (2026-09-30,
  research §5.4): the access pattern it was.** The 134 left the residual's
  write-back out (it was 184), and the rest was the grid — a token is ten
  turns of the 4 KB channel rotation, so workgroups a token apart load the
  same DRAM channels; walked stream-fastest `hc.cn` is at 200 GB/s and
  `hc.norm` at 217, bit-identical.
- **P6 — batching. Unblocked 2026-09-23 and moved to
  [`CONCURRENCY.md`](CONCURRENCY.md)**: the answer is three concurrent
  streams with a priority lane for voice. C1 (sequence slots), C2 (the
  scheduler, `-llm-slots`), C0 (`cmd/loadgen`), C3 (chunk curve) and C4
  (per-slot checkpoints: a voice command's TTFT goes 1.83 → 0.19 s at 48
  layers) and C5 (batched decode: three agents at 18 tok/s each instead of
  11, a three-row pass at 1.83 steps) and C6 (every slot at the full 262k)
  are done; `ai.service`'s LLM line is `-llm-slots 3`. The text below is the
  old framing.
  Blocked on the product question: will the API serve
  more than one stream? Each sequence owns 113 MB of DeltaNet state plus
  rings and KV. P5b already built the first stage (R-row decode GEMVs,
  shipped at R = 2); raising `GEMVMaxRows` must land in the same commit as
  its rung in `TestGraphIsAChunkSplit` — R rows were correct in a block test
  and wrong in the model three separate ways (P5c).
- ~~**Context depth is the largest measured regression.**~~ **P7, closed
  2026-09-21** ([write-up](research/p7-context-depth.md)). It was two terms
  and neither was the selection's width. **Three kernels walked `nKV`, the
  cells the arenas were *allocated* for, instead of the cells that exist** —
  `llm_attn_score.comp` scoring every pooled block, `llm_attn_select.comp`
  running four radix passes and an emit over every cell — which is **6.94 ns
  per allocated cell a token an attention layer**, charged in full *at depth
  zero*: **10.9 ms of a decode step** at 131k cells before one cell is real,
  and most of why the old sweep began at 24.27 where the headline is 36.19.
  **And QSA was semantics with no saving**: `llm_attn_wmma.comp` read the
  bitmask beside the causal mask and still walked every key block, so a
  decode step that names 2051 cells was reading 64 000 (0.25 µs a live cell
  a token a layer). The fix is three identities — the dead pooled blocks all
  hold cell 0 pooled `ratio` times so they share one score; a cell past the
  live count is an `-inf` that can never be selected; a key block with
  nothing selected in it contributes nothing and is skipped whole — gated on
  **exact** equality by the new `TestAttnGPUCacheSizeDoesNotChangeTheAnswer`
  (the 4k fixture in a cache three times too big, bit for bit) and by 48
  greedy tokens identical across the two binaries. Two passes each, same
  hour: decode **4.05 → 16.27 tok/s at 64k (4.02x)**, **6.78 → 18.34 at
  32k**, **27.96 → 33.18 at depth 0**; prefill **313.1 → 441.2 at 64k**;
  `-gen` at ctx 32768 **32.43 → 35.92 tok/s**, 94% of the byte ceiling.
  Falloff to 64k: decode **0.14x → 0.49x**, prefill **0.51x → 0.64x**.
- **The gather is what is left, and after P13 it is the only route to 900
  tok/s at 128k.** P13's own probe priced the regime it has to beat: fitting
  the `qt2_kt2` control — 1.46x less K/V a query for 1.37x more work, and
  1.02x *worse* on the clock — says `attn.attn` at 128 000 cells is roughly
  **half arithmetic and half K/V traffic**, so fewer cells is the only thing
  that cuts both. A 16-row query tile's union is ~4 574 selected cells and
  the 16-cell rung reads 14 976 of them: **3.3x**, and that is the floor of
  what a cooperative-matrix tile can skip, because the tile is sixteen cells
  and the selection's runs are four. Priced at **~0.20 ms a token against
  0.566**, which with the selection change and a 4096 ubatch is ~1.06 ms a
  token — **~940 tok/s at 128k**, the only arrangement of these numbers that
  reaches the target. The open question is the arena: a per-(query tile, kv
  head) scratch at 8192 gathered cells is 2.1 GB, which fits only once the
  expansion's 1.14 GB is freed, and the alternative is chunking the gathered
  axis and folding the partials through P8's combine, which already exists.
  P13's `llm_attn_blocks.comp` is the compaction it would be built on.
- ~~**What the gather replaces, from P7's side.**~~ **Closed by P15-3,
  2026-09-22.** The skip prunes key blocks; it does not stop the count of them
  growing — so decode now runs the **gathered** kernel with its axis split,
  `GATHER`+`SPLITK`, and `attn.attn` at 128 000 cells is **419.8 → 129.9 µs**
  a dispatch. P14 wrote down that decode should not take the gather and the
  argument is *right about the gather alone*: unsplit, it is 986.7 µs against
  the split's 419.8. What it missed is that the split cuts the **walk** (7.12x)
  and the gather cuts the **work** (2.99x), and neither had the other's factor.
  The bound is `2*selWidth` **live** cells, below which every live cell is
  selected and the compaction is overhead — 1.09 → 1.56 ms a token the wrong
  way at depth zero. That bound is the first decode knob that moves with the
  depth, so the prerecorded buffer carries an epoch now. **P11 measured the
  prefill side of the same question and it is the whole of the depth term
  there too**: at `-pp 2048` every block is flat in depth except attention,
  which is 97% of the falloff and 47% of a prefill token at 64 000 cells
  (`attn.attn` 27x from depth zero, `attn.select` 46x). 64k prefills at
  **654 tok/s**, 0.57x of the depth-zero rate. **And the prefill side of the
  gather is now priced**: the union of sixteen adjacent queries' selections is
  only 1.76x one query's reach, so a per-query gather is worth 1.76x at 64k
  and not the 31x the 3.2% density suggests — L4b's refusal, with a number.
- ~~**128k still does not complete.**~~ **Closed by P13-1, 2026-09-22**, and
  the marks were never the problem — it was a submit budget with no depth
  term. `maxStorageBufferRange` is the next wall: the KV planes share a
  buffer with the arenas, which caps the cache at **~148k cells**. Past that
  wants L6a's array-of-buffers, one a layer.
- **Long-context gates** (idea 7, enabled by P0): perplexity at ctx 4096 is
  done (3.9392, −2.23% against ctx 2048 — the selection helps); a needle
  test through the API is not run.
- **The downstream task eval** — the last unpriced thing about the widths.
  At +1.74% of wikitext perplexity the instrument may no longer separate
  plans; a few hundred multiple-choice items through the HTTP API is an
  evening.
- **`Reset` is owed a test.** A fresh sequence is not independent of the one
  before it (P5c finding 6): three plain greedy runs over one prompt give
  three different texts, deterministically, when the previous run left cells
  past the new prompt's end. Any future token-for-token claim needs the
  plain-against-plain row printed beside it.
- **Priced and not taken** (in the archives, with numbers): the router's
  padding (+0.09 tok/s), re-screening the GEMV rungs at the q5 width, fp16
  DeltaNet state (+~1.3 ceiling), the unpack prefetch (~1.1x prefill), the
  asymmetric epilogue's GEMM arm, W4A8, the hot-expert fast path, the
  float-atomic combine, the vision tower. Also unexplained and written
  down: a dispatch is 8–22% slower inside a step than alone on a cold bank
  (P1b's 1.33 ms environment gap), and dispatch time is not a function of
  its bytes in either direction (P3a/P4b, opposite signs).

**Instrument rules that must survive** (each earned the hard way): a
whole-model number needs a **same-hour control** (P1c — the machine moved
8.7 ms a step overnight); a ladder rate above 242 GB/s is an L3 hit, and
its *byte count* can lie too (D16, P4a); a plan is **measured, never
composed** (additivity leaks both ways through the n-gram block); a control
has to be able to fail (P5b shipped green tests on stale SPIR-V — `.comp`
edits need `go generate` before measuring); benchmarks need
`LLM_BANK_CACHE` set or a shipped-bank run re-fits for 8 minutes
(`cmd/serve` sets it, `cmd/llm` does not).

## Speech → text (archive: [`research/speech-vertical.md`](research/speech-vertical.md))

**Where it stands.** S1–S8 done: front end 21 ms + encoder 17 + decode 5 =
43 ms for jfk.wav, transcript exact, word/segment timings from the model's
own TDT durations, served at `/v1/audio/transcriptions` and over Wyoming
(with resampling at that door only).

**Open:**

- **S10 — the front end is 48% of the pipeline**: a few thousand 512-point
  float64 FFTs on the host. Either a float32 radix-4 on the host or the
  STFT + mel filterbank as two dispatches (the filterbank is a
  `[T, 257] x [257, 128]` GEMM; `kokoro_istft.comp` is the worked inverse).
- **The submit+fence is 38% of an emission** (40 µs of 105 per token). A
  persistent kernel — which is also what streaming transcripts would want —
  and/or speculating on blanks (consecutive-frame joints are independent
  during a blank run: one GEMM at M = 16 for the price of M = 1).
- **S9 — long clips.** Full attention means a chunk boundary changes every
  frame; chunking belongs in the design. The quadratic term arrives around
  1024 frames (~82 s); `-max-audio` refuses past the sizing today.
- Small: cache `UploadMel`'s sinusoidal position rows (1 ms, depends only
  on T); the eight per-head position-score GEMMs are §3.5's grouped shape.

## Text → speech (archive: [`research/speech-vertical.md`](research/speech-vertical.md), recap: [`research/tts-recap.md`](research/tts-recap.md))

**Where it stands.** T1–T10 and W1 done. Every stage on the device, staged
once for the life of the server (T9: the endpoint went 550 → 59 ms with
byte-identical audio); voices blend in upstream's own spelling (T8); T10
put PL-BERT's attention on the matrix cores so synthesis is a straight
8.0 ms per second of audio at every length. The round trip
(`cmd/roundtrip`) closes at 70.1x real time, six prose cases exact.

**Open, in order of what a round trip buys:**

- **The generator's 12 ms and the tail's 8** — the only arithmetic-bound
  parts of the model, 65% of a short utterance, measured since T4.
- **The host embedding stack** (R1's #2): `bert` is 12 ms of a paragraph
  and only 2.5 on the device — the "not worth a dispatch" comment is stale
  the same way the attention kernel's was.
- **Three boundaries, one change each**: the excitation's 0.2 ms is 88%
  submit+readback (move the two noise convolutions onto the device and
  nothing of that stage touches the bus); the phoneme side's remaining 6 ms
  is readbacks and submits (move the vocoder's input boundary); a 16 kHz
  path out of the vocoder would delete the client resample (9.7% of the
  loop — can the iSTFT head be asked for the rate directly?).
- **G2P on unseen text**: designed corpus 24/24; on 400 unseen sentences
  68.8% of sentences, 92.4% of phoneme words agree with misaki. Closing the
  gap is more *measured* rules — the syntactically obvious ones scored
  worse than no tagger.
- **Known, unsettled**: the style row is indexed by the phoneme *character*
  count (upstream's behaviour), so a front end emitting different characters
  picks a different row. And kokoro spells numbers out where parakeet writes
  digits back — the round trip measures those three cases and does not
  count them as failures.

## Image generation — **parked** (archive: [`research/qimage-vertical.md`](research/qimage-vertical.md))

**Where it stands.** Done and **parked 2026-09-21** — parked because the plan
ran out, not because it stalled. (The superseded z-image-turbo vertical is
[`research/zimage-vertical.md`](research/zimage-vertical.md) plus
[`research/zimage-pipeline.md`](research/zimage-pipeline.md); its stages are
**I0–I7** and none of its code survives except `zimage/qwen` and
`zimage/tokenizer`, which `embed`, `llm` and `parakeet` import.) The vertical was replaced 2026-09-20
(Z-Image-Turbo out, `Qwen/Qwen-Image-2.1` in, for native RGBA and
reference-image editing — `GOALS.md` #4), and **stages Q0–Q12 all closed
inside two days**. Both endpoints are served: `POST /v1/images/generations`
answers at **1m28.8s / 1m29.8s for a 1024²/40-step image** (31.5 GB resident,
matching the fp32 oracle's own picture at mean 3.4e-4) and `POST
/v1/images/edits` at **1m54.2s / 1m56.6s for a 1024² edit on one reference**
(39.4 GB, matching the oracle's edit at max abs 0.0014), both with native
RGBA and unconditional in-progress previews (a fitted 64x4 matrix, 159 µs a
frame, three partials for 0.3% of a request — no flag, because there is
nothing to load). Q10 turned the ceiling from a side box into an area, so
16:9 comes back **1344x768** instead of 1024x576; Q11 put the client's
hang-up through to the sampler and the VAE's submit batches; Q12 finished the
Z-Image deletion (**6,839 lines of Go and 895 of GLSL** out, every gate
re-run with no digit changed). The full write-up, every tolerance with its
instrument named, is in the archive.

**Int8 banks (Q13, 2026-09-26, after parking)**: the text encoder and the
DiT now stage as int8 by default (`serve -image-fp16` is the control):
resident **31.5 → 20.4 GB**, edits the same 11 GB less, ~4% a step. Priced
against the released bf16 pipeline: DiT teacher-forced steps 0.03–0.10x its
error at 1024², the free run 0.81x; the encoder keeps layers 6 and 16 in
fp16 (the massive-activation channel is written there) and lands 3x inside
bf16 on real prompts. `ai.service` picks it up on its next deploy. Still
open: the served peak measured inside the service, and the prefix KV cache.

**What to read before touching this code again** — three precision facts,
each of which has already caught a port:

- **the VAE cannot take fp16 operands anywhere** (Q9b). Its tail norm divides
  a per-pixel L2 out of a residual stream at absmax 2.6e5, so a 5e-4 relative
  perturbation becomes an absolute one: **one** narrowed convolution costs the
  decoded image max abs 0.0885 and the whole 3x3 set costs 0.178, against an
  fp32 port sitting at 7.3e-4. The rule is not "watch the range", it is "do
  not narrow". `TestConvFP16Ladder` is the instrument; re-run it before
  pointing any narrowing kernel at `qimage/vae`. Q12 deleted
  `vae_conv_wmma`/`vae_attention_wmma` outright, so the shortcut is not in
  the tree — resurrect from git history only if that ladder says otherwise.
- **the vision tower amplifies an input perturbation by ~10³**, so a
  condition image must be quantized exactly as the reference's is —
  compositing alpha over white in float rather than on 8-bit levels moves the
  prompt embedding by rel 11 (a firing control). That is why the Lanczos
  resampler is gated on exact 8-bit equality and not a tolerance.
- **on a non-square condition image the fp32 dump is the less accurate
  side** — rel 1.4e-3 from a float64 run where the Go tower sits 2.1e-4 — so
  that stage is gated against dumped float64 rows.

**If it is ever unparked**, in the archive's order:

- **The 1184²-area ceiling** — the only remaining *capability*, not a
  percent. The VAE decoder's activation arena is one storage buffer against a
  4 GiB − 4 device limit at a measured **3060 bytes a pixel for every aspect
  ratio** (`TestArenaShape`), which caps a request at 1,403,584 pixels
  whatever shape they are in, so the model's own 2048² examples do not
  decode. Two routes: tiled decode, or a multi-buffer arena
  (`vk.PipelineSpec.Counts`, with the LLM's 77 GB bank as precedent). Tiling
  is the less attractive of the two against a tail norm that is a per-pixel
  L2 over the whole feature map.
- **conv3x3's last ceiling** — after Q9b's register block it is **85.9% of
  the decode, 3.36 s at 5.0 TFLOP/s**, and its remaining limit is one shared
  read per multiply-add. A pixel block is priced at ~1.5 s and is the only
  port here that would **not** be bit-identical. Low value now: the VAE is
  4.5% of an image, the DiT is 95%.
- **The DiT's remaining percents are fusions** — that is where the image's
  time actually is, post-Q9.
- **Masked edits** (Q-o3) — still a 501, but the reason moved: 2.1 *can* do
  masked and annotated local edits; how a mask is fed is not in the diffusers
  implementation this port follows. External research, not a port.
- **Two watch items**: Q-o2, whether a turbo/distilled 2.1 checkpoint or
  step-distillation LoRA appears (the examples repo and lightx2v); Q-o4,
  whether [taehv](https://github.com/madebyollin/taehv) grows a 2.1 variant,
  which would turn the fitted linear preview from a fallback into an upgrade.

**Settled, so it does not get relitigated**: the step count — 40 stays the
default because it is the only count safe across prompt kinds. A fox
photograph and an impasto harbour are convincing at **12 steps**, at 36% of
the cost; a bicycle drivetrain diagram is coherent at 24 and has
disintegrated by 12 (floating parts, ghosted tubes, contrast collapsing
toward white). **24 is the honest fast setting** — −35%, no visible loss on
any of the three prompt kinds. `steps` is a request field, so a client that
knows its prompt takes the discount itself. Note the sweep's wall clocks
(40: 1m38–1m42, 24: 1m4, 16: 44–45 s, 12: 35–36 s) were measured **before
Q9 and Q9b**, so the seconds are stale by the 6–7% those two took off every
step while the *ratios* stand; re-run with `QI21_SWEEP=1 go test
./qimage/pipeline -run TestStepSweep` (~10 minutes of device time) before
quoting a number from it.

**Not planned**: quantisation (compute-bound at every servable size, §3.4;
int8 WMMA runs at fp16 rate, §0; and the two-machine deployment removes the
footprint argument). Sampling the encoder's posterior (breaks seed
reproducibility). A self-trained tiny decoder for previews — a training
project this repo does not want. No CFG path (`true_cfg_scale` stays a
refusal) — it would double every step.

## Music generation — **closed** (archive: [`research/music-vertical.md`](research/music-vertical.md))

**Where it stands.** Closed 2026-09-27: A0–A12 are all done, and the plan
ran out. ACE-Step 1.5's three generators (the 5 Hz LM, the 4 B DiT, the
Oobleck VAE) run upstream's default thinking path, sample mode, and the
turbo model's audio-in tasks (cover, cover-nofsq, repaint, a reference's
timbre). Every gate is against upstream's handler in fp32, with the drift
priced against its bf16. `serve -music` answers `/v1/music` as
submit-then-poll jobs (multipart audio in), sharing the swap slot with
image and video. Code: `ace/` (`plan`, `dit`, `vae`, `lm`, `pipeline`),
`api/music.go`, `backend/music.go`, `cmd/ace`.

**Open, none of it planned:**

- **Listen** to `out/ace-a10-int8-*` (the int8 LM) and `out/ace-a11-*`
  (a cover, a repaint, a referenced song). The user listened after A8
  ("sounds great") but not since.
- **Deployment**: `-music` is not in `ai.service` (the user's call; it
  belongs on the non-LLM machine).
- **Speed past the LM**: the DiT (1.8 s of a 60 s song, 6.9 s of 4 min) and
  the VAE are ≤ 20% of a request, and the LM's step is at the bus. A 4-bit
  bank would be the next byte cut, and phase 1 has no headroom under bf16
  for it.
- Upstream's retake and flow-edit, and the base model's tasks (lego,
  extract, complete), stay refusals.

## Embeddings (archive: [`research/embedding-vertical.md`](research/embedding-vertical.md))

**Where it stands.** E0–E8 and E7 done. Same `zimage/qwen` transformer,
third caller. Cosine 0.999999+ against fp32. Served with MRL `dimensions` and
the non-OpenAI `instruct` field.

**E7, batching (done 2026-09-28).** The shape the archive proposed:
projections batched, attention per text, with no shader changes.
- **`qwen.GPUEncoder.RunBatch`.** The texts sit back to back in the
  residual stream with no gap rows, so the GEMMs pay nothing for the
  batching. Each text gets its own key-block (64-row) aligned region of the
  packed q/k/v planes, written by a per-text pack that zero-fills its tail.
  Attention is one dispatch per text. Positions restart per text.
- **Bit-exact.** Because of that layout, a text's vector out of a batch is
  **bit-identical** to its lone run under the same plan
  (`TestGPUBatchMatchesSingle`: 10 lengths including 63/64/65, both orders,
  two plans). A control that lets text 1 see text 0 moves it to cosine 0.896.
- **Measured on the device** (`TestGPUBatchThroughput`, ms a text):

  | tokens a text | alone | 8 texts | 32 | 64 |
  |---|---|---|---|---|
  | ~14 | 9.6 | 1.76 | **1.14** | 1.12 |
  | ~38 | 9.8 | 2.80 | 2.24 | 2.28 |
  | ~108 | 11.4 | 5.57 | 5.89 | – |

- **Compute-bound past ~450 rows**, at 0.056 ms a row: 15.6 TFLOP/s, 28% of
  peak. `reg64` stays the best rung up to 1,796 rows
  (`TestGPUBatchLadder`), so `embed.PlanFor` is unchanged.
- **Free latency on the side.** `PerSubmit`: a layer per command buffer
  instead of qwen's 8 dispatches takes a lone 27-token text from
  **11.9 → 9.6 ms** (the ~40 µs submit+fence, 84 times). Qwen's other
  callers keep 8.
- **Serving** (`backend/embed.go`). Inputs are tokenized in the handler and
  queued. One worker runs passes of at most `-embed-batch-tokens` (1024)
  rows, taken **round-robin across requests**, with one `Device.Do` a pass.
  The arenas are 2x the pass, because 64-row plane regions make 32 short
  queries want 2048 plane rows for 448 stream rows.
- **Served, private `-embed` server** (production numbers in brackets):

  | | new | production |
  |---|---|---|
  | 1 text | 10.4 ms | [12.2] |
  | 32 short | **46 ms** | [383] |
  | 32 × 36 tok | 82 ms | [385] |
  | 128 × 36 tok | **305 ms** | [1535] |
  | lone query behind a 128-text job | **94 ms** | [1523] |

- **Trap found on the way.** The request path first took the model lock
  that a pass holds, so a lone query could not even queue until the passes
  ahead of it had drained, and it came back *with* the job at 300 ms.
  `TestEmbedQueryOvertakesJob` pins it.
- **Deployed** by the 2026-09-29 18:40 restart (VIDEO.md session 10).

**Next, for single-query latency** (the GEMMs run at ~70 GB/s against a
236 GB/s bus; small-M grids underfill at hidden 1024):
- q/k/v as one projection: k and v cost nearly what q does at half its size.
- gate+up with SwiGLU fused in: Kev's K7.6 kernel, and its lesson about grid
  order.
- split-K on o and down.
- pricing an int8 bank on the LLM's small-M kernels.

Also unexplained: the Sep 27 production requests ran at 47 ms a text, 4x
the model's own time, probably another vertical holding the device.

## The server, cross-vertical (docs: [`API.md`](API.md))

- **Constrained decoding** — the one refusal that is a missing capability:
  would close `response_format`, `text.format` and named `tool_choice`.
- **More than one conversation** — the graph is one sequence's; two clients
  take turns evicting each other's prefix. A cache per conversation is a
  measurement (residency) plus P6's scheduler question.
- **The image adapter holds the device lock for the whole run** (the LLM's
  is per forward pass). Same fix shape; what a waiting speech request
  actually gains is a measurement.
- `n > 1` images are serial (capped at 4); a forced alignment for
  transcript timings if the 80 ms grid ever isn't enough.

---

## Where the old files went (2026-09-20 consolidation, IMAGE.md again on 2026-09-21, MUSIC.md on 2026-09-27)

| was | now |
|---|---|
| `LLM.md` | [`research/llm-vertical.md`](research/llm-vertical.md) — stages, decisions D1–D21, open questions, how-to-run |
| `LLM2.md` | [`research/llm-review.md`](research/llm-review.md) — hypotheses checked, decode budget, P0–P6 as closed |
| `SPEECH.md` | [`research/speech-vertical.md`](research/speech-vertical.md) — S1–S8, T1–T10, R1, W1 write-ups |
| `TTS.md` | [`research/tts-recap.md`](research/tts-recap.md) |
| `IMAGE.md` (z-image, until 2026-09-20) | [`research/zimage-vertical.md`](research/zimage-vertical.md) |
| `IMAGE.md` (Qwen-Image-2.1, 2026-09-20–21) | [`research/qimage-vertical.md`](research/qimage-vertical.md) — frozen 2026-09-21 with the vertical; **Q-stages, decisions 1–7 and Q-o numbers resolve there** |
| `PIPELINE.md` | [`research/zimage-pipeline.md`](research/zimage-pipeline.md) — inventory, validation rules, budget |
| `EMBEDDING.md` | [`research/embedding-vertical.md`](research/embedding-vertical.md) |
| `MUSIC.md` (2026-09-27) | [`research/music-vertical.md`](research/music-vertical.md) — frozen 2026-09-27 with the vertical; **A-stages, decisions 1–7 and A-o numbers resolve there** |
| `IDEAS.md` | [`research/ideas.md`](research/ideas.md) — the `§N.M` backlog and the measured roofline |
| `TODO.md` (session log) | distilled into `research/` as each stage closed; the phase-1 tail is [`research/phase1-backlog.md`](research/phase1-backlog.md); the full log is in git history |

Per-experiment findings remain one file each in [`research/`](research/README.md),
indexed there. `§N.M` numbers are permanent addresses — never renumber.
