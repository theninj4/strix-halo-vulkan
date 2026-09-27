# OCR — document parsing (PaddleOCR-VL-1.6)

> **Live tracking and session handoff doc, opened 2026-09-27.** Stage letters
> are **O**. When the vertical closes, this file is frozen to
> `research/ocr-vertical.md` like the others and `TODO.md` gets the one-line
> summary. Until then: tick a stage when its gate passes, put its measured
> numbers under it, and keep **§ Handoff** at the bottom current. A new
> session should be able to start from there.

## What we are building

`GOALS.md` item 10: OCR via
[`PaddlePaddle/PaddleOCR-VL-1.6`](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
(read at `c5630aba`, 2026-08-08; Apache-2.0; technical report
[arXiv 2606.03264](https://arxiv.org/pdf/2606.03264)). The card claims
96.33% on OmniDocBench v1.6 and 93.19 on Real5-OmniDocBench, from a 0.9 B
model.

**The checkpoint on the Hub is half of the product.** The report (§2) and
PaddleX's pipeline config (`paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml`)
agree that document parsing is **two models**:

| stage | model | here |
|---|---|---|
| layout | **PP-DocLayoutV3**: RT-DETR (HGNetV2-L backbone, deformable decoder) with a reading-order pointer head and a polygon mask head; 25 region classes. "Kept unchanged" from 1.5 | `PaddlePaddle/PP-DocLayoutV3_safetensors@97d101e6`, 33.3 M fp32, in `models/PP-DocLayoutV3/` |
| recognition | **PaddleOCR-VL-1.6-0.9B**: a NaViT vision tower, a 2×2-merge projector and ERNIE-4.5-0.3B, prompted per region (`OCR:`, `Table Recognition:`, `Formula Recognition:`, `Chart Recognition:`, `Seal Recognition:`, `Spotting:`) | `models/PaddleOCR-VL-1.6/`, 0.96 B bf16 |

A page goes: layout → filter/merge boxes → crop each region → one VLM
generation per region, prompt chosen by label → post-process (OTSL tables to
HTML, formula delimiters, repetition truncation) → markdown in reading order.
Doc orientation and unwarping (PP-LCNet, UVDoc) are **off** in the 1.6
default (`use_doc_preprocessor: False`), and so is chart and seal
recognition. We do not build them.

So there are two products, and they are built in that order:

1. **Element-level recognition**: an image and a task in, text out. This is
   the HF checkpoint alone, and it is what vLLM serves as an OpenAI chat
   model. **PaddleOCR's own pipeline talks to a VLM server over exactly that
   door** (`vl_rec_backend="vllm-server"`), so a faithful
   `/v1/chat/completions` makes the whole official pipeline run against us
   unchanged. That is the O6 gate, as the unchanged TypeSafe SDK was Kev's.
2. **Page-level parsing**: a page (image or PDF) in, markdown and blocks
   out, with the layout model and the glue in Go.

### Sources, and who is ground truth

- **The checkpoint's own remote code** (`models/PaddleOCR-VL-1.6/modeling_paddleocr_vl.py`,
  the authors') and **PaddleX** (`~/repos/PaddleX` at `ffb6490`, sparse:
  `inference/models/doc_vlm`, `inference/pipelines/paddleocr_vl`,
  `inference/genai`) are the authors' two implementations.
- **HF transformers 5.17.0** has a native port (`models/paddleocr_vl`,
  `models/pp_doclayout_v3`) and it is what runs in `.venv`. It is the
  **oracle we dump from**, with one override (decision 3).
- **vLLM** (`~/repos/vllm`, `model_executor/models/paddleocr_vl.py`) and
  **llama.cpp** (`tools/mtmd/models/paddleocr.cpp`) are the third and fourth
  opinions. llama.cpp also runs the model on this machine's Vulkan, so it is
  the **speed baseline** (O0).

## The model, from config.json and the safetensors header

620 tensors, all bf16. What a request touches:

| part | params | fp16 | notes |
|---|---|---|---|
| vision tower (27 blocks) | 413.0 M | 0.83 GB | SigLIP-so400m widths |
| projector (`mlp_AR`) | 26.0 M | 0.05 GB | |
| ERNIE layers (18) + norm | 254.8 M | 0.51 GB | |
| `embed_tokens` | 105.9 M | 0.21 GB | a host lookup |
| `lm_head` | 105.9 M | 0.21 GB | **untied** (below) |
| unused: `packing_position_embedding`, `vision_model.head` | 52.9 M | — | never loaded (HF ignores them too) |

About 1.6 GB on the device in fp16. The layout model is 0.13 GB in fp32.

### The vision tower (`visual.vision_model`)

- **Patch 14**, Conv2d `[1152, 3, 14, 14]` with bias (no temporal axis:
  `temporal_patch_size` is 1 in the processor, 588 values a patch).
- hidden 1152, 16 heads × 72, FFN 4304 `gelu_pytorch_tanh`, LayerNorm eps
  1e-6, **separate q/k/v projections with biases** (fuse at load).
- **Learned positions: a 27×27 grid, bilinearly resized** to the image's
  patch grid. Nothing about the grid is stored; it is recomputed per size.
- **2-D axial rope** over the 72-wide head, theta 10000: 18 inverse
  frequencies per axis, cos/sin built `[h(18), w(18)]` and duplicated to 72,
  NeoX halves (`rotate_half`). Same scheme as `qimage/vision` (trap 5 there).
- **Full bidirectional attention within one image**, no windows. Several
  images in one pass are block-diagonal (`cu_seqlens`).
- **Raster patch order.** Unlike Qwen-VL's towers, the processor emits patches
  row-major and the tower keeps them that way. The merge reorders later.
- `post_layernorm` (eps 1e-6) closes the tower.

That is `qimage/vision`'s tower with four differences: patch 14 not 16,
raster order not 2×2-block-major, a 27² grid not 48², and a
`post_layernorm` inside the tower. The GPU path in `qimage/vision/gpu.go`
is the starting point (O3), not a rewrite.

### The projector (`mlp_AR`)

`pre_norm` LayerNorm(1152, **eps 1e-5**) per patch row → gather each 2×2
block (`(t, h/2, 2, w/2, 2, d) → transpose → (h/2·w/2, 4·1152)`, so the
four rows are `(r,c), (r,c+1), (r+1,c), (r+1,c+1)` in that order) →
`linear_1` 4608→4608 + bias → **exact erf GELU** (`GELUActivation`) →
`linear_2` 4608→1024 + bias. **Two LayerNorms in a row** with different
epsilons (tower's post-LN, then the projector's pre-norm): skipping either
gives numbers, not an error.

### The language model (ERNIE-4.5-0.3B)

18 layers, hidden 1024, **16 query heads and 2 KV heads** × 128 (GQA 8:1),
SwiGLU 3072, RMSNorm eps 1e-5, no biases, **no q/k norm**, vocab 103 424.
Rope theta 500 000, NeoX halves, **3-D M-RoPE in sections `[16, 24, 24]`**
over the 64 rotary pairs, *chunked, not interleaved*: pairs 0–15 read t,
16–39 read h, 40–63 read w (`recomposition_frequencies`). That is the other
convention from the one `llm.RoPEMulti` implements for Qwen3.8 (interleaved
`j%3`), so it is a new table, not a reuse.

Against `ace/lm` (a dense Qwen3-4B with a KV cache, the repo's only dense
decoder): a quarter of the width, 18 layers not 36, no q/k norm, GQA 8:1 not
4:1, and positions are three numbers a row. The design carries over whole
(O4).

### The prompt, positions and tokenizer

- Template (`chat_template.jinja`):
  `<|begin_of_sentence|>User: <|IMAGE_START|>` + N ×
  `<|IMAGE_PLACEHOLDER|>` + `<|IMAGE_END|>` + task + `\nAssistant:\n`.
  Ids: begin 100273, IMAGE_START 101305, **image token 100295 =
  `<|IMAGE_PLACEHOLDER|>`** (not `<|image_pad|>` 101304), IMAGE_END 101306,
  eos `</s>` = 2. `N = (gh·gw)/4`. The `OCR:` prompt on a 668×70 line is 165
  tokens; the image is 152 of them.
- Positions (`get_rope_index`): text runs `p, p+1, …` on all three axes. An
  image starting at `p` with merged grid `mh × mw` gives token `(r, c)` the
  position `(p, p+r, p+c)`, and the next text token is at `p + max(mh, mw)`,
  **not** `p + mh·mw`. The causal mask stays by cache index.
- Tokenizer: Llama-style byte-fallback BPE (`legacy: true`), 100 295 vocab +
  1 041 added tokens, normaliser `' ' → '▁'` with **no prefix space**, no BOS
  added. We only ever *encode* six fixed prompts, so encoding is a table
  checked against HF; **decoding** is the real port: `▁` → space, `<0xNN>`
  byte fallback fused into UTF-8, specials skipped.

### Preprocessing

`smart_resize(h, w, factor 28, min_pixels 112 896, max_pixels 1 003 520)`:
round each side to a multiple of 28, then scale down (floor) or up (ceil) to
the pixel bounds; a side under 28 is raised to 28 with the other scaled.
Max 1 003 520 px = **5 120 patches = 1 280 image tokens**. Spotting raises
the bound to 1 605 632 and upscales small images 2× (Lanczos) first; the
page pipeline can set per-label bounds. Then rescale 1/255, normalise with
mean = std = 0.5, patchify raster.

### PP-DocLayoutV3 (read from config.json; O7 reads the code)

800×800 input, rescale only (mean 0, std 1), bicubic. HGNetV2-L backbone
(stages 1–4, BN folded), hybrid encoder (one AIFI transformer layer on the
stride-32 map, CCFM fusion across strides 8/16/32), 300 queries, 6
deformable-attention decoder layers (8 heads, 4 points, 3 levels), a
**reading-order head** (`decoder_order_head`, `decoder_global_pointer`) and
a **mask head** for polygon outlines. 858 fp32 tensors. The new kernel kinds
are the deformable sampling (bilinear gathers at predicted offsets) and the
small convs; everything else is GEMM and LayerNorm we already run.

## Decisions (so future sessions don't relitigate)

1. **Element level first, page level second.** Element-level is the HF
   checkpoint alone, it is a complete product on its own (every region
   prompt, served the way vLLM serves it), and PaddleOCR's pipeline can
   already drive it. The layout model and the glue come after (O7–O9).
2. **Doors.** Element level is `/v1/chat/completions` with model id
   `PaddleOCR-VL-1.6-0.9B`, including the vLLM extras PaddleX sends
   (`extra_body.mm_processor_kwargs.{min_pixels,max_pixels}`,
   `skip_special_tokens`, `repetition_penalty`, `max_completion_tokens`).
   When `-llm` is also loaded (not the deployment, see 8) the chat door
   routes by `model`. Page level is **`POST /v1/ocr` in Mistral's OCR API
   shape** (`document_url` / `image_url`, `pages[].markdown`, images), the
   closest thing to an industry standard, plus **PaddleX's
   `/layout-parsing`** envelope so PaddleOCR serving clients work: one
   engine, two envelopes, as `/v1/videos` carries SGLang's.
3. **Position-grid interpolation is `align_corners=False`.** The authors'
   remote code, vLLM and PaddleX all use False; HF's native port uses True.
   Three to one, and the three include both of the authors'. The dumps
   override HF to False. Measured on the 668×70 line: step-0 KL 5.4e-6 and
   the same text either way. O0 measures it on a full page.
4. **Resize: Pillow's BICUBIC**, which is what PaddleX's native processor
   calls (`image.resize(..., resample=3)`). HF 5.17 defaults to torchvision's
   antialiased bicubic, which `llm/pixels` already ports (V3). O1 measures
   the gap between them; if it moves text, PIL wins, because the authors
   evaluate with their own processor.
5. **lm_head is untied.** config.json says `tie_word_embeddings: false` and
   the two tensors differ (max 0.20). HF's native config *defaults* to tied
   and only refuses because both tensors are present with different values:
   a converter that drops `lm_head` would silently tie them. Load both.
6. **Greedy decoding, 8 192 new tokens by default** (PaddleX's
   `PADDLEOCR_VL_MAX_NEW_TOKENS`; the pipeline asks `temperature: 0`). Every
   gate is greedy token-for-token against HF.
7. **fp16 on the device, fp32 residual, like every other vertical,** unless
   O0's range audit finds ERNIE's activations near fp16's ceiling (ace/lm's
   Qwen3 reached 6.4e4). A 0.9 B model at 1.6 GB needs no int8 for
   footprint; int8 is only a speed question (O11).
8. **Deployment: machine B, resident.** It sits with speech, TTS, embeddings
   and Kev, not in the image/music/video swap slot: 1.8 GB is less than a
   swap costs to reload, and OCR is interactive.
9. **PDF pages are rasterised by `pdftoppm`** (poppler) as ffmpeg muxes video:
   a subprocess, not a Go PDF renderer. DPI is PaddleX's (O9 reads it).
10. **Batching is the performance design, not an afterthought.** A page is
    10–40 regions, each an independent generation. Decode at this size is
    latency-bound (a step reads ~0.7 GB), so N regions decoding together cost
    about one. The engine is ace/lm's (token, slot, position) rows from the
    start, and a page's regions are slots.

## Budget

- **Device memory**: ~1.6 GB weights fp16 + KV cache (18 layers × 2 heads ×
  128 × 2 × fp16 = **18 KB a token**, so 64 slots × 8 k tokens = 9.4 GB at
  the extreme; 16 × 4 k = 1.2 GB is plenty) + tower activations at 5 120
  patches (~0.3 GB) + layout (<0.5 GB).
- **Tower**: 5 120 patches × 27 layers ≈ 4.2 TFLOP of GEMM + 3.3 of
  attention. At the ~25–35 TFLOP/s our WMMA paths reach, **~0.3 s a
  full-bound crop**; most regions are far smaller.
- **Decode**: ~0.5 GB of layers + 0.2 GB head a step. The floor is latency,
  not bandwidth (memory: *Decode is latency, not bandwidth*); estimate
  3–5 ms a step for one row, about the same for 16. A dense page of Chinese
  text is ~2–4 k output tokens across its regions: **~1–2 s a page batched,
  10–20 s serial**.
- **Layout**: an 800² RT-DETR is tens of GFLOP: tens of ms.

The HF oracle on the CPU (32 threads, fp32) took 6.5 s for the one-line
demo, so a page on the CPU is minutes: fine for dumps, useless as a server.

## Stages

| # | Stage | State |
|---|---|---|
| O0 | Weights, oracle scripts, fp16 range audit, decision 3 on a page, llama.cpp baseline on this machine, the official pipeline in a Paddle CPU venv as the page-level oracle | **done 2026-09-27 except the Paddle venv** (moved to O8): `dump_ocr.py` over 6 cases, byte-identical twice; **fp16 audit: largest activation 2,725, 24x of headroom**; decision 3 moves the page's step-0 KL by 1.0e-5 and none of 300 tokens; **llama.cpp: 243 tok/s decode, the page in 7.4 s** |
| O1 | Host side in Go (`ocr/`): smart_resize, PIL bicubic, patchify, template ids, 3-D positions, tokenizer both ways | **done 2026-09-27**: tokenizer 229/229 encodes and 880/880 decodes exact; all 6 prompts, 3-D positions, rope deltas and decodes exact; PNG pixels bit-exact (RGB, RGBA, grayscale), JPEG ≤ 3 levels on 2.8% (Go's IDCT) |
| O2 | Tower + projector on the CPU in fp32 against the dumps | **done 2026-09-27** (`ocr/vision.go`, on qimage/vision's blocks): embedding with the resized grid 2.6e-7, blocks 0–1 ≤ 4.9e-7, projector 3.5e-7 from the dump's post-LN; the whole stack is O3's gate (scalar attention is minutes) |
| O3 | Tower + projector on the GPU (from `qimage/vision/gpu.go`) | **done 2026-09-27** (`ocr/gpu.go`): qimage's device tower through five hooks, patches run 2x2-block-major; all six cases, every dumped stage: **proj ≤ 2.8e-3, post_ln ≤ 1.8e-3** of fp32; raster-order control 1.12; **the page's 4,884 patches in 389 ms**, a line in 33 ms |
| O4 | ERNIE on the GPU (from `ace/lm`): prefill, KV decode, M-RoPE sections, head; teacher-forced logits, then greedy equality | **done 2026-09-27** (`ocr/lm.go`): all six cases greedy-equal to fp32 HF, **the page's 768 tokens included**; layers ≤ 7.5e-4, logits KL ≤ 5.3e-6; 1-D rope control rel 0.24; prefill of the page's 1,234 rows 100 ms; **decode 4.06 ms a step on the device, 4.4 ms a token end to end** (llama.cpp 4.1) |
| O5 | Element level end to end, `cmd/ocr`, all six tasks against HF greedy | **done 2026-09-27** (`ocr/engine.go`, `cmd/ocr`): image file → text, **all six cases token- and text-identical to fp32 HF** (the page's 768 and both JPEGs included); 2.33 GB on the device; every element case faster than llama.cpp end to end (line 83 vs ~140 ms, seal 293 vs ~450); **the full page 8.1 s to `</s>` at 212 tok/s vs llama.cpp's 7.4 s at 233**: prefill wins 3x, decode at depth loses |
| O6 | Serve element level (`-ocr`, chat door, vLLM extras); **gate: PaddleOCR's `doc_parser --vl_rec_backend vllm-server` against us, unchanged** | **done 2026-09-27** (`backend/ocr.go`, `api/route.go`): `serve -ocr` answers `PaddleOCR-VL-1.6-0.9B` on `/v1/chat/completions`, routed beside `-llm`; **PaddleOCR 3.7.0's `doc_parser` runs unchanged against it**: the demo page's 27 regions all 200, markdown byte-identical across runs; table, formula and chart pages too; streamed = buffered = fp32 text on all six cases |
| O7 | PP-DocLayoutV3 on the CPU then the GPU, against HF `pp_doclayout_v3` | **done 2026-09-27** (`ocr/layout`): CPU oracle ≤ 2.5e-5 of HF fp32 at every stage, regions identical; preprocessing bit-exact on PNG (`pixels.ResizeNoAA`); **device trunk** (fp16 GEMMs, 35 ms) + host AIFI and decoder: **~225 ms a page against the CPU's 2.2 s**, every region, label and reading order of the five cases HF's, scores ≤ 6.6e-4 |
| O8 | Page glue in Go: box post-processing, crops, merges, per-label prompts, OTSL→HTML, markdown; against PaddleX's functions | **O8a done 2026-09-27** (`ocr/page`, `cmd/ocr -page`): PaddleX's glue in rect mode, **every step identical to PaddleX 3.7.2's own functions** on five pages (layout boxes, crops by hash, merged images, prompts, block list, markdown byte for byte), 204 text-function edge cases exact; the Go pipeline end to end = PaddleX's markdown, the demo page in 8.3 s. **O8b left**: polygons (`layout_shape_mode` auto), figures inside tables |
| O9 | Serve page level: `/v1/ocr`, `/layout-parsing`, PDFs; a page's regions batched | **done 2026-09-27 except the batching** (moved to O11): `serve -ocr` answers Mistral's `/v1/ocr` (**Mistral's own SDK works unchanged**) and PaddleX's `/layout-parsing` (PaddleX's markdown and prunedResult for the page); PDFs by `pdftoppm` at PaddleX's 144 dpi; 3.35 GB resident with the layout |
| O10 | Accuracy: an OmniDocBench v1.6 subset against the card, as K8 reproduced Kev's | |
| O11 | Performance: multi-row decode across regions, tower batching, int8 if decode pays for it | |

### O0 — weights, environment, oracles

In (2026-09-27, curl from `resolve/<sha>`):

- `models/PaddleOCR-VL-1.6/` (1.8 GB) at `c5630aba`;
- `models/PP-DocLayoutV3/` (133 MB) from `PP-DocLayoutV3_safetensors@97d101e6`;
- `testdata/ocr/`: PaddleX's two demo images, a 1524×1368 newspaper page
  (`paddleocr_vl_demo.png`, RGBA) and a 668×70 line (`ocr_demo.jpg`).
- `~/repos/PaddleX` (sparse, `ffb6490`) for the pipeline glue.

The HF oracle needs nothing new in `.venv` (transformers 5.17.0, torch 2.14
CPU): the smoke run reads the line as `相思那得夢魂来。</s>` in 6.5 s.

**The oracle** is `reference/dump_ocr.py` (~3.5 min on 32 cores): six cases,
`line` (668×70 JPEG, `OCR:`), `formula` (448×64 grayscale PNG), `table`
(551×132 JPEG), `seal` (640² PNG), `chart` (625×570 RGBA PNG) and `page`
(the 1524×1368 RGBA newspaper at max_pixels: 66×74 patches, 1,234 prompt
tokens, capped at 768 new). Per case, `record.json` and a
`tensors.safetensors` of pixels (PIL and torchvision backends), the tower's
embedding, blocks (all 27 when ≤1,024 patches, else 0/1/13/26), post-LN,
projector, `inputs_embeds`, all 18 LM layers, the final norm, the prompt's
logits and 32 teacher-forced steps of logits. Every case's teacher-forced
argmax agrees with its greedy run. `reference/dump_ocr_tokens.py` writes the
tokenizer corpus (229 encodes, 880 decodes). ~800 MB in
`reference/out/ocr/`, gitignored.

**Two things the first run found:**

- **`generation_config.json` ships `use_cache: false`.** HF then recomputes
  the whole sequence every step: 80 tokens of formula took 66 s instead of
  9 s. The dump passes `use_cache=True`; it changes speed, not the answer.
- **The PIL and torchvision processors differ by one level** (max 0.0078,
  0.0157 on the seal) on every case (O-o1 is how much that moves the text).

**fp16 audit** (`ranges.json`, every Linear's output and every
`down_proj`/`fc2`/`linear_2` input over all six cases): the largest value is
**2,725**, the ERNIE residual after layer 1's `down_proj` (its SwiGLU product
is 2,162; layer 17's 2,035). The tower peaks at 2,576 in its last block's
`fc2`, the projector at 17.9, the logits at 42.3. **24x under fp16's 65,504**:
no narrowing factor like ace/lm's 1/16 is needed (decision 7 holds).

**Reproducible**: a second run into another directory is byte-identical on
all six cases and `ranges.json`.

**Decision 3 on the page** (1524×1368 at 66×74 patches): step-0 KL between
`align_corners` False and True is **1.0e-5**, and the first 300 greedy tokens
are identical. It barely matters; we follow the authors anyway.

**llama.cpp baseline** (`~/repos/llama.cpp` at `d1d3c3396`, Vulkan, the
GGUFs converted by its `convert_hf_to_gguf.py` into
`models/PaddleOCR-VL-1.6-GGUF/`, 0.93 + 0.88 GB f16; `sentencepiece` was
added to `.venv` for it). Through `llama-server --jinja`, greedy, first
request per image (the second hits its image cache), with `ai.service` idle
on the machine:

| case | prompt tokens | prefill (tower + LM) | new tokens | decode |
|---|---|---|---|---|
| line 668×70 | 165 | 99 ms | 10 | 243 tok/s |
| formula 448×64 | 173 | 106 ms | 80 | 245 tok/s |
| table 551×132 | 163 | 96 ms | 71 | 243 tok/s |
| seal 640² | 543 | 330 ms | 29 | 242 tok/s |
| chart 625×570 | 453 | 250 ms | 137 | 244 tok/s |
| page 1524×1368 | 1,234 | 1,045 ms | 1,453 | 233 tok/s (7.4 s wall) |

**4.1 ms a token** is the number to beat for one stream, and one stream is
all llama.cpp does; decision 10's batching is where the page-level win is.
Every case's text is HF's, except that **on the page llama.cpp writes a
full-width `（` at character 108 where fp32 greedy writes `(`**: a near-tie
that f16 tips. So an fp16 port's page-length gate cannot be greedy equality;
it is teacher-forced logits plus agreement up to the first near-tie
(LLM-VISION's V10 met the same thing as MoE routing ties).

Moved to O8, where it is needed: `.venv-paddle` with paddlepaddle CPU and
`paddleocr[doc-parser]`, running the official page pipeline against the HF
oracle and later against our server.

### O3 — the tower on the device

`ocr/gpu.go` is `qimage/vision`'s `GPU`, not a copy. That package gained five
device-path hooks, each nil or a no-op for its two existing users (their
gates, `TestGPUTower` and `TestLLMGPUTower`, read the same numbers after):

- `Model.PostNorm`/`PostNormEps`: an affine LayerNorm in place after the last
  block, on `parakeet_layernorm` (fp32 out, the only affine-LN-to-fp32 kernel
  in the tree). The projector's pre_norm cannot fold into it: two norms in a
  row.
- `Merger.NormEps`: the projector's 1e-5.
- `Model.Geometry`: the position grid and rope table, supplied by the
  caller.
- the patch reduction is padded to the GEMM's 64-wide K slab (588 → 640,
  weight and pixels zero-extended; qimage's 1536 already is one);
- `GPU.Tap`: the residual stream after the embedding, each block and the
  post-norm, splitting submits there. For gates only.

**The one idea: run the tower 2x2-block-major although the checkpoint is
raster.** Full attention does not care about row order when every row
carries its own grid position and rope angles, so the pixels, the grid and
the rope table are permuted on the host, and the projector's 2x2 gather
becomes the free four-row concatenation qimage's merger already does. The
merged rows come out in merged-grid raster order, the image tokens' order.
Taps are permuted back. `TestGPUTowerRasterControl` feeds raster rows
straight in: proj rel **1.12**.

Measured (`go test ./ocr -run GPUTower`, 5 s, ai.service up and idle; two
runs, same numbers):

| case | patches | embed | block 26 | post_ln | proj | tower + projector |
|---|---|---|---|---|---|---|
| line | 76×8 | 3.8e-5 | 7.1e-3 | 5.5e-4 | 8.4e-4 | 33 ms |
| formula | 64×10 | 1.8e-5 | 1.2e-3 | 5.1e-4 | 9.9e-4 | 33 ms |
| table | 50×12 | 3.4e-5 | 2.2e-3 | 5.2e-4 | 8.6e-4 | 33 ms |
| seal | 46×46 | 7.6e-5 | 3.6e-3 | 1.0e-3 | 2.0e-3 | 136 ms |
| chart | 44×40 | 4.6e-5 | 3.5e-3 | 9.3e-4 | 1.7e-3 | 105 ms |
| page | 74×66 | 3.8e-5 | 4.5e-3 | 1.8e-3 | 2.8e-3 | **389 ms** |

Bounds: embed 3e-4, blocks 2e-2, post_ln 6e-3, proj 1e-2. This tower is
**fifty times kinder to fp16 than Qwen-Image's** (0.14–0.56 on the same
kernels, qimage/vision's gate): its residual peaks at 2.6k, not 1.4e4.
Device cost 0.89 GB weights + 0.36 GB activations at the 5,120-patch budget.

Speed, for O11: llama.cpp's page *prefill* (tower + 1,234 LM tokens) is
1,045 ms, so the tower is already well under it. The small crops sit on a
**33 ms floor**: ~600 dispatches at 8 a submit is ~75 synchronous submits,
which is qimage's watchdog bargain for a 4k-patch image and pure overhead
for a 600-patch crop. A page is 10–40 crops, so that floor is the first
thing O11 takes (a submit size by patch count, or several crops a pass).

### O4 — ERNIE on the device

`ocr/lm.go` is ace/lm's decoder at ERNIE's shape, fp16 only (no int8 bank;
decision 7): the projections in the fragment tiling, read by the DiT GEMM
rungs for a pass of many rows and by llm_gemv's split-K GEMV at 1–3 rows; a
KV cache a slot; attention over the cache only, so a prefill and a decode
step are the same dispatches. What changed:

- **Shaders**: `ace_lm_prep.comp` and `ace_lm_attn.comp` take `NQ`/`NKV` as
  build switches, and prep gains `QK_NORM=0` and `MROPE=1`: a row's rope
  positions are three numbers apart from its cache index (at `pc.dim`,
  3 a row), and pair i reads t for i < 16, h for i < 40, w after (HF's
  `recomposition_frequencies`, chunked). One `[pos][64]` table still serves:
  only the position a pair reads changes. The OCR builds are
  `ocr_lm_{prep,attn,attn_combine}.spv`; ACE's three binaries are
  byte-identical (prep was rebuilt and differed only in SPIR-V ids, so the
  committed one stays).
- **Rows** (`Row`) are a token id or an embedding (the image rows), a slot,
  a cache position and a rope (t, h, w). Generated token j sits at cache
  L+j and rope L+j+`rope_delta` on all three axes.
- **The head is untied** (`lm_head.weight`, own bank); logits come for the
  last 1–3 rows of any pass, so a prefill needs no separate last-token pass.
- **No SwiGLU narrowing** (ace/lm's 1/16): O0's audit found 24x headroom.

Gate (`go test ./ocr -run TestLM`, 9 s): prefill from the dump's own inputs
(text rows by id through the host's bf16 lookup, image rows from
`lm.embed`), every layer, the prompt logits, every teacher-forced step
against `logits.tf`, then greedy against the reference generation:

| case | prompt | worst layer rel | prompt logits rel / KL | tf worst rel / KL | greedy |
|---|---|---|---|---|---|
| line | 165 | 5.0e-4 | 1.2e-3 / 4e-9 | 9.2e-4 / 8e-7 | 10/10 |
| formula | 173 | 7.0e-4 | 2.2e-3 / 4e-10 | 2.0e-3 / 2e-7 | 80/80 |
| table | 163 | 4.0e-4 | 1.1e-3 / 8e-10 | 2.3e-3 / 5e-7 | 71/71 |
| seal | 543 | 5.1e-4 | 3.2e-3 / 2e-10 | 2.5e-3 / 7e-7 | 29/29 |
| chart | 453 | 7.5e-4 | 1.6e-3 / 4e-7 | 6.3e-3 / 5e-6 | 137/137 |
| page | 1,234 | 6.6e-4 | 1.3e-3 / 6e-9 | 1.9e-3 / 3e-6 | **768/768** |

Bounds: layers 5e-3, logits rel 2e-2, KL 1e-4 a step. **The page is
greedy-equal for all 768 tokens**, where llama.cpp's f16 flips a `(` at
character 108 (O0): the near-tie did not tip here, but it can on another
page, so O5's page gate stays teacher-forced plus first-divergence.
`TestLMRopeControl` gives every row a plain 1-D rope at its cache index
(what a Qwen3 port does): prompt logits rel **0.24**, KL 6.7e-3.

**Speed.** Prefill: 10 ms for ~165 rows, 28 ms for 543, **100 ms for the
page's 1,234** (llama.cpp's tower + LM prefill for the page was 1,045 ms;
ours is 389 + 100). Decode, one row at depth 200
(`TestLMStepLadder`, median of 64, ai.service up and idle):

| split-K (q k v o gate up down) | device | wall |
|---|---|---|
| ace/lm's, scaled: 8 16 16 16 4 4 8 | 5.64 ms | 5.89 ms |
| all 1 | 4.56 | 4.76 |
| all 4 / all 8 | 5.40 / 5.58 | 5.66 / 5.81 |
| all 16 | 4.31 | 4.60 |
| **16 16 16 1 16 16 16** | **4.06** | **4.30** |

Greedy end to end (host argmax over 103k logits included) is **4.35–4.62
ms a token** against llama.cpp's 4.1. The ladder is not monotone: gate/up at
8 slabs cost 0.7 ms over 16, and every middle table is slower than both
ends; O11 should find out why before tuning further. A step is ~340
dispatches at ~12 µs each: fusing q/k/v and gate/up (adjacent in the bank
already) and batching slots (decision 10) are the obvious next moves, and
both belong to O11.

### O5 — element level end to end

`ocr.Engine` (`ocr/engine.go`) is the unit O6's chat door and O8's page
glue both call: `Recognize(ctx, Request{Image, Prompt, MinPixels,
MaxPixels, MaxTokens, OnToken})` → `Result{IDs, Text, Finish, timings}`.
Process (Go, PIL bicubic) → `BuildPrompt` → the device tower → the image
rows spliced over the placeholders → one prefill pass (chunked if a prompt
outgrows `LMOptions.Rows`) with the last row's logits → greedy, first-max
ties as torch.argmax, until `</s>` or the cap (lowered to what the cache has
left). One request holds the device at a time (a mutex); batching is O11.
**Not in the engine, because vLLM's server does not do it either:**
Spotting's 2x Lanczos upscale (`pre_process_for_spotting`), per-label pixel
bounds and the text post-processing. Those are PaddleX's pipeline: O8.

`cmd/ocr -image F [-task ocr|table|formula|chart|seal|spotting] [-prompt
T] [-max-tokens N] [-min-pixels] [-max-pixels]`: the text on stdout,
timings on stderr.

Gate (`go test ./ocr -run TestEngine`, 8 s): every case from its **image
file**, not the dump's pixels, so Go's JPEG decode (≤3 levels off on 2.8%,
O1) and the fp16 tower are both in the loop. **All six are token-identical
and text-identical to the fp32 reference**, the page through its 768-token
cap included. The test device now opens with subgroup size control, as the
server does (the O3/O4 numbers are unchanged).

End to end, ai.service up and idle, against llama.cpp's O0 table (its
total is prefill + tokens at its decode rate):

| case | prompt + new | tower | prefill | decode | ms/token | total | llama.cpp |
|---|---|---|---|---|---|---|---|
| line | 165 + 10 | 33 ms | 10 ms | 39 ms | 3.89 | **83 ms** | ~140 ms |
| formula | 173 + 80 | 33 | 11 | 347 | 4.34 | **393** | ~432 |
| table | 163 + 71 | 33 | 11 | 309 | 4.35 | **354** | ~388 |
| seal | 543 + 29 | 130 | 29 | 126 | 4.36 | **293** | ~450 |
| chart | 453 + 137 | 103 | 25 | 605 | 4.42 | **740** | ~811 |
| page, to `</s>` | 1,234 + 1,621 | 354 | 100 | 7,656 | 4.72 | **8.14 s** | 7.4 s (1,453 tokens) |

(`cmd/ocr` on the page twice: identical text, 8.143 and 8.141 s. Load is
2.3 s; 2.33 GB on the device, of which the KV cache for 10,240 positions is
0.19 GB.) Prefill is where we win, 3x on the page; decode is ~5% behind
llama.cpp at shallow depth and ~10% at the page's depth of 2,855, where
the scalar attention over the cache grows. The two runs of the page stop
at different lengths (1,621 against 1,453) because llama.cpp's f16 took
another branch at character 108 (O0); both are the demo's article.

### O6 — the chat door

`serve -ocr [-ocr-model DIR]` loads `backend.OCR` (an `ocr.Engine`, 2.33 GB,
2.3 s) as a chat backend with the id **`PaddleOCR-VL-1.6-0.9B`**, the name
PaddleOCR's own `genai_server` serves it under. With `-llm` too, the door is
`api.CompletionRoute{llm, ocr}`: a request naming the OCR id goes to OCR,
every other name to the LLM as before, and `/v1/models` lists the LLM
first. Responses name the model that ran (`modelID` now returns the
requested id when the backend owns it). The machine-B line of `ai.service`
carries `-ocr` (decision 8); **it is not deployed yet** (`deploy.sh`
restarts the live server).

**What PaddleOCR sends**, captured by pointing `doc_parser` at a stub that
records bodies (27 requests from the demo page, all identical but the
image): `model`, one user message `[image_url (JPEG data URL), text
"OCR:"]`, `max_completion_tokens: 4096`, `temperature: 0`,
`skip_special_tokens: true`, `mm_processor_kwargs: {min_pixels: 112896,
max_pixels: 1003520}`. It picks the model as `/v1/models`' first entry
unless `--vl_rec_api_model_name` says otherwise, and keeps up to 200
requests in flight (they queue on the engine's mutex; batching is O11).

The api layer gained vLLM's three extras as request fields
(`repetition_penalty`, `skip_special_tokens`, `mm_processor_kwargs`);
**the LLM backend refuses all three** rather than ignoring them. The OCR
backend (`ocrRequest`) accepts one user message with one image and any text
parts (joined, after the image, as the template does), greedy only:
temperature ≠ 0, top_k, min_p, `repeat_penalty` (llama-server's), presence
penalty, stop sequences, tools, template kwargs and a `max_pixels` over the
staged budget are 400s; `top_p` is accepted because it is a no-op under
greedy. `repetition_penalty` is vLLM's over prompt + output ids, in the
engine. Each device pass (tower, prefill, every decode step) takes
`Device.Do` on its own, so speech interleaves with a page token by token.
Streaming decodes the whole prefix per token and sends the new tail,
holding back while it ends in U+FFFD (an incomplete byte-fallback
character), so the deltas concatenate to exactly the buffered text.

Gates:

- `go test ./backend -run TestOCR` (8 s): the refusals; every dumped case
  through `Complete` from a data URL, streamed deltas joined = the fp32
  reference text, no delta holding a partial character.
- `go test ./api -run TestCompletionRoute`: routing, reported model, the
  extras reaching the backend, `/v1/models` order.
- **PaddleOCR's pipeline, unchanged** (`.venv-paddle`: Python 3.12,
  paddlepaddle 3.3.1 CPU, paddleocr 3.7.0, paddlex 3.7.2, python-docx):

      .venv-paddle/bin/paddleocr doc_parser -i testdata/ocr/paddleocr_vl_demo.png \
          --pipeline_version v1.6 --vl_rec_backend vllm-server \
          --vl_rec_server_url http://127.0.0.1:18080/v1 --save_path OUT

  against `ai -addr 127.0.0.1:18080 -token= -ocr`: exit 0, markdown, JSON
  and docx written; the page's **27 region requests all 200**, 8.1 s of
  recognition first request to last (1,770 tokens, serial), and **the
  markdown is byte-identical across two runs**. The table image comes back
  as an HTML table (1 `Table Recognition:` request), the formula as `$$ …
  $$` LaTeX (1 `Formula Recognition:`), the chart page's three text crops
  as text; chart and seal recognition are off in 1.6's default config, so
  those regions send nothing. Three of the page's text blocks come back
  empty in the markdown: that is PaddleX merging a paragraph that runs
  across columns into its first block, not a failed request (every
  request returned text); O8 compares the glue against PaddleX's own.

### O7 — PP-DocLayoutV3

**The model** (HF `modeling_pp_doclayout_v3.py`, read whole): RT-DETR with
two heads added. HGNetV2-L backbone (stem 32/48 with a 2x2 branch and a
padded max-pool; stages 128@stride 4, 512@8, 1024@16, 2048@32; stages 3-4
are "light": 1x1 then depthwise 5x5), a hybrid encoder (one post-norm
transformer layer, GELU, over the 25x25 map with 2-D sine positions; a
top-down FPN and bottom-up PAN of CSP-RepVGG blocks, SiLU), 300 queries by
top-k of the encoder's class logits over 13,125 memory rows (anchors
outside (0.01, 0.99) zero their row), and 6 decoder layers (self-attention,
deformable cross-attention: 8 heads x 3 levels x 4 points, bilinear
`grid_sample` with zero padding, then a ReLU MLP; boxes refined through
`inverse_sigmoid` every layer). PP-DocLayoutV3's own parts:

- **a mask head**: 32 prototype maps at stride 4 (200x200) built from the
  PAN and the stride-4 backbone map; a query's mask is its 32-wide
  embedding times them. **The decoder's initial boxes are the bounding
  boxes of the selected queries' masks > 0** (`mask_enhanced`), not the
  encoder's box head, whose output is computed and never used;
- **a reading-order head**: the last layer's queries through a linear and a
  global pointer (64-wide q and k, lower triangle masked to -1e4); a
  query's order is its rank by votes;
- **tied heads**: the decoder's class and box heads are the encoder's
  (`enc_score_head`, `enc_bbox_head`), stored once.

The checkpoint's names are the original's (`self_attn.out_proj`,
`encoder.encoder.0.layers.0`, `layers.N.fc1`), which HF renames on load;
the Go loader reads them as stored and folds every batch norm (eps 1e-5)
into its conv.

**Labels: use PaddleX's `label_list`, not HF's `id2label`.** HF's config
folds `display_formula`/`inline_formula` into "formula", and
`footer_image`, `header_image` and `vertical_text` into their plain
names; PaddleX's pipeline picks prompts and merges by the distinct names.
`layout.Labels` is inference.yml's list.

**The oracle agrees with Paddle's own model.** Paddle's `PP-DocLayoutV3`
(paddlex 3.7.2, threshold 0, `.venv-paddle`) against HF's queries on the
page: all 48 of its boxes are HF queries with the same label, **scores
within 0.001**; its boxes are clipped to the image where HF's are not (the
formula's HF box starts at x = -4.3), up to 8 px. On the formula the score
moves 0.02, which is the cv2 vs torchvision resize (decision-4-shaped;
O8/O10 decide if it matters). The page's `doc_parser` output keeps 33
regions against HF's 31 at 0.5: two `vision_footnote` boxes at 0.43/0.35
where HF's threshold keeps one at 0.64. That is PaddleX's post-processing
(per-class thresholds, NMS, merging), O8's job.

**The oracle**: `reference/dump_doclayout.py` (0.7 s an image, five
images: the page, table, formula, chart, seal) writes the pixels, the stem,
four stages, the three projections, AIFI, the PAN, `mask_feat`, the
decoder memory, the encoder's class and box logits, the top-k indices, the
initial (mask) boxes, all six layers, every layer's boxes and logits, the
order logits and the masks, plus the detections at 0.5.

**The CPU port** (`ocr/layout`: `ops.go` convs by im2col + parallel
products, `model.go` loader, `forward.go` graph, `post.go` preprocessing
and post-processing): `go test ./ocr/layout` (~35 s).

| | worst over five cases |
|---|---|
| any stage, rel to HF fp32 | 2.5e-5 (the formula's masks) |
| top-300 selection | identical on all five |
| initial boxes (mask > 0) | 2.6e-8 |
| regions at 0.5 | identical: query, label, reading order; scores ≤ 1.1e-6, boxes ≤ 0.001 px |
| from the image files (Go decode) | identical; the JPEG's scores ≤ 7.7e-5 |

**Preprocessing, exact**: HF's processor is torchvision's `resize(800x800,
bicubic, antialias=False)` on uint8, which on an AVX2 CPU is torch's native
uint8 kernel, not the float one: the separable int16 passes of the
antialiased kernel (horizontal first, uint8 intermediate), with the float
kernel's four taps (a = -0.75) **clamped and folded at the edges**, not
Pillow's truncate-and-renormalise (that reading was 13 levels off on the
formula's 12.5x upscale; a float bicubic + rounding is 9 levels off). Now
`pixels.ResizeNoAA`, **bit-exact on all four PNGs**, then `x / 255` as a
division (HF's fused rescale; the product by 1/255 is an ulp off).

**Time on the CPU**, the page, 32 cores: 2.2 s. Backbone 0.87 s, FPN/PAN
0.94 s, mask head 0.18 s, the selection 0.13 s, six decoder layers 0.22 s:
the convolutions are 90% of it.

**The fp16 ladder** (`TestFP16Ladder`: every conv's and linear's operands
rounded to half, fp32 sums, what a matrix core computes): **the largest
activation anywhere is 204**, so half's range is no question. The trunk
costs ~1e-3 (stage 4 ≤ 2.9e-3, PAN ≤ 2.3e-3, `mask_feat` ≤ 1.6e-3); the
selection keeps 299 or 300 of the 300 queries. The decoder's state moves by
0.2-0.4 and the initial boxes by up to 23 in logit space, **from the convs
alone** (fp32 linears change nothing): a junk query's mask flips between
empty and a pixel at the mask > 0 step. **No region can see it**: all 37
regions of the five cases kept, scores within 0.001, boxes within 0.1 px.
So the device port is fp16 with fp32 sums throughout, and its gate is the
regions (and the trunk's tensors), not the decoder's raw state.

**The device trunk** (`ocr/layout/gpu.go`, `shaders/layout_ops.comp`):
the backbone, the FPN/PAN, the mask prototypes, the decoder-input
projections and the head's two big products (the encoder output's linear
and the six layers' value projections over all 13,125 memory rows) run on
the device; the AIFI layer (625 tokens) and everything from the query
selection on run on the host (`head`, shared with the CPU path). Maps are
fp32 channel-last with the row stride rounded to 64 and the pad channels
zero; a conv is an im2col (or, for a 1x1, the map narrowed straight) into
the fp16 A operand, `dit_gemm` (reg64, fragment-tiled B) and
`qvit_bias_act` (which gained ReLU = 3 and SiLU = 4). A channel
concatenation feeding a 1x1 is never built: each part is narrowed into A at
its column offset and the weight's columns placed to match (HGNetV2's
aggregations, the CSP inputs). **RepVGG is reparameterised at load** (the
1x1 branch added into the 3x3's centre tap). `layout_ops.comp` is the rest,
one build a mode: im2col, depthwise conv, the stem's padded max-pool,
nearest and bilinear 2x upsamples, add, channel copy. The graph is static
(800x800 always), recorded once, every map its own arena slot: **1.02 GB
on the device**, which arena reuse can cut (O11). Staging is 0.29 s.

Gate (`go test ./ocr/layout -run TestGPU`, 3.4 s): every trunk stage
against HF fp32 is **0.65e-3 to 3.2e-3** (the ladder predicted 2.3e-3 at
stage 4), and **every region of the five cases is HF's: same query, same
label, same sequence**; scores within **6.6e-4**, boxes within **0.02 px**
(bounds 3e-3 and 0.5 px). The absolute rank among all 300 queries can move
by one (a junk query's vote), so the device gate compares the sequence and
only the fp32 gate the rank.

Time a page (the device forward, `Timings`): upload 1 ms, **device trunk
23 + 12 ms**, host AIFI 29 ms, readback 22 ms (94 MB: the memory, the
prototypes, seven 13,125x256 products), host head ~135 ms (six decoder
layers ~15 ms each, the mask boxes 20 ms, the final masks 14 ms, all
scalar Go over 300 rows). **~225 ms against the CPU oracle's 2.2 s.** The
host head and AIFI are the next 160 ms, and O11's: a decoder on the device
is small matrices, a deformable gather and a 300x300 pointer.

### O8 — the page glue

**What PaddleX does between the two models**, read whole
(`paddlex/inference/models/layout_analysis/processors.py`,
`pipelines/paddleocr_vl/{pipeline,uilts,result}.py`,
`common/result/converter/markdown_*`):

1. **Layout post-processing** (`LayoutAnalysisProcess`). Its transformers
   engine for this model is HF's `post_process_object_detection` at the
   layout threshold (0.3), each box a row [label, score, x0, y0, x1, y1,
   order_seq]. Then: coordinates rounded (half-even); score **> 0.3**;
   NMS (same class IoU 0.6, different 0.98, "+1" areas; which also removes
   a query's second label); an `image` box over 82%/93% of the page
   dropped; the "large" merge mode for chart, display_formula, doc_title,
   inline_formula and paragraph_title (a box 90% inside one of those goes);
   sorted by the order head; clipped to the page as ints; `order` numbered
   over all but SKIP_ORDER_LABELS. numpy 2's scalar rules keep all of it in
   float32.
2. **The pipeline**: `filter_overlap_boxes` (drop `reference`, < 6 px, and
   the smaller of two overlapping > 0.7, inline formulas at 0.5, visual
   pairs exempt unless against a table); crops; `merge_blocks` (consecutive
   text blocks continuing across a column gap or down an edge-aligned
   column beside a non-merge block are stacked on a white canvas, centred
   or edge-aligned, unless the stack is 3x taller than wide); prompts:
   `OCR:`, `Table Recognition:`, `Formula Recognition:` on a
   margin-trimmed crop (cv2's fixed-point grey, levels stretched, <= 200 is
   ink); image, chart and seal regions are never recognised at the defaults
   (chart/seal recognition off); every block 112,896-1,003,520 px, 4,096
   new tokens.
3. **Assembly**: repetition truncation (at 50 chars, 5,000 for tables),
   `\(`/`\[` → `$`/`$$`, OTSL → HTML for tables, image paths
   `imgs/img_in_<label>_box_x0_y0_x1_y1.jpg`, figures inside tables dropped.
4. **Markdown** (`pretty`): per-label handlers (titles renumbered, text
   paragraphs double-spaced, images as centred `<img width=N%>`, tables
   styled), `markdown_ignore_labels` (number, footnote, header(_image),
   footer(_image), aside_text) left out, joined by blank lines.

**The oracle** is PaddleX itself: `reference/dump_ocr_page.py` (in
`.venv-paddle`) runs those functions, unmodified, on HF's detections at 0.3
(`dump_doclayout.py` now writes them), with each recognition from our
`serve -ocr` over PNG, and records every step. Its page markdown is
**byte-identical to the real `doc_parser` run of O6** (Paddle's own layout
model, JPEG crops): on this page neither difference moves a byte.
`reference/dump_ocr_textfns.py` feeds PaddleX's text functions 204 edge
cases (truncation, OTSL, delimiters, titles); 3 malformed tables make
PaddleX's OTSL parser raise `IndexError` (which fails its page), recorded;
Go counts within bounds instead.

**Two options fixed**, both PaddleX's own: `layout_shape_mode="rect"`
(polygons are O8b: 31 of the page's 33 regions are rectangles either way,
the other two are the image caption's) and the native backend's raw crops
(its vllm-server client JPEG-encodes them).

**The port** (`ocr/page`: `layout.go` the post-processing, `blocks.go`
filtering, crops, merging, prompts and `crop_margin`, `post.go` the text
functions and markdown, `page.go` assembly, the JSON block list and
`Parser`). Gates (`go test ./ocr/page`, 13 s with the device):

- `TestTextFunctions`: 204 cases, **0 wrong**;
- `TestGlue`, from HF's detections: **layout boxes, blocks, group ids and
  aligns, every crop and recognition input by SHA-256 (the PNG pages), the
  prompts and bounds, the block list and the markdown all PaddleX's**, on
  all five pages (the JPEG's crops by size: Go's decoder);
- `TestParse`, the Go pipeline end to end (device layout, glue, device
  engine) from the image files: **the markdown is PaddleX's byte for byte on
  the four PNG pages; all 32 recognitions are identical**, the JPEG's
  included.

`cmd/ocr -page -image F` prints the markdown. The demo page: layout 0.22
s, glue 1 ms, **27 recognitions 8.1 s serially**: the page is now the
engine's single-stream decode, which O11's batching (decision 10) is for.

**O8b, not done**: `layout_shape_mode="auto"` (the default: masks traced
with OpenCV's `findContours`/`approxPolyDP`/`minAreaRect`, polygons
compared with shapely, the crop whitened outside the polygon);
`tokenize_figure_of_table` (a figure >= 25 px inside a table is painted
over with `[F<n>]` in OpenCV's Hershey font; `Parse` refuses such a page
rather than guess); Spotting's pre- and post-processing (only reachable
without layout).

### O9 — the page doors

`serve -ocr` now also loads PP-DocLayoutV3 (`-ocr-layout-model`, default
`models/PP-DocLayoutV3`; empty serves the chat door alone) and answers two
envelopes over one `api.DocumentBackend` (`api/document.go`,
`backend/ocr_document.go`):

- **`POST /v1/ocr`, Mistral's OCR API**: `document` as `document_url` or
  `image_url` (a string or `{url}`; data URLs or fetched,
  `util.FetchDocument`), `pages`, `include_image_base64`, `table_format`
  (`"html"` takes tables out as `[tbl-N.html](tbl-N.html)` + `tables[]`;
  absent leaves them inline as HTML; `"markdown"` is refused, the model
  writes HTML with spans), `extract_header`/`extract_footer` (the header
  and footer blocks, which the markdown never carries). The markdown is
  PaddleX's plain one with Mistral's picture references
  `![img-N.jpeg](img-N.jpeg)`, numbered across the document, and
  `images[]` their boxes (and JPEG base64); `dimensions.dpi` is 144 for a
  PDF page, 0 for an image. Refused by name: `file` ids, annotation
  formats.
- **`POST /layout-parsing`, PaddleX's serving envelope** (outside `/v1`,
  where its clients call): `file` base64 or URL, `fileType`,
  `minPixels`/`maxPixels`/`maxNewTokens`/`repetitionPenalty`/`temperature
  0`/`topP`, `returnMarkdownImages`; `{logId, errorCode, errorMsg,
  result: {layoutParsingResults: [{prunedResult, markdown: {text,
  images}}], dataInfo}}`, and PaddleX's 422 envelope for a refusal. Every
  other field is accepted at its default or null and refused by name
  otherwise (`layoutShapeMode` other than `"rect"`, chart/seal recognition,
  preprocessing, `visualize`, `restructurePages`, `outputFormats`,
  `prettifyMarkdown: false`, custom thresholds and merge modes).

**PDFs** (decision 9): `pdfinfo` for the page sizes, then `pdftoppm -png
-singlefile -scale-to-x/-y` at **ceil(points x 2)** a side: PaddleX's
serving renders with pypdfium2 at `PDF_RENDER_SCALE` 2.0, and the sizes
agree exactly (the anti-aliasing of the two renderers does not have to).
`-ocr-max-pages` (100) bounds a request; PaddleX's serving stops at 10.

The layout forward runs under `Device.Do` whole (~225 ms, of which the
trunk is 35 ms); the recognitions take it a pass at a time as before.

Gates:

- `go test ./api -run 'TestOCREndpoint|TestLayoutParsing'` (fake backend):
  both envelopes' shapes, `image_url` in both forms, the refusals, the PDF
  `dataInfo`.
- `go test ./backend -run TestOCRDocument` (device, 23 s): every PNG page
  oracle through the document path: **markdown = PaddleX's**, prunedResult
  = PaddleX's within the device layout's fp16 (scores 3e-3, and one box
  edge of the demo page a pixel over: HF's 656.5 against the device's
  656.52 rounds the other way; the text of that block is unchanged); the
  two-page PDF (`testdata/ocr/two_pages.pdf`: the demo page and the table
  image at 144 dpi) at the right sizes, `page_count` 2, page selection.
- **Over HTTP, the real clients** (`reference/gate_ocr_http.py`, in
  `.venv-paddle`, against `ai -ocr`): PaddleX's documented request to
  `/layout-parsing` returns **PaddleX's markdown and markdown images for the
  demo page** (8.5 s) and a two-page `dataInfo` for the PDF (10.0 s); and
  **Mistral's own SDK** (`mistralai` 2.10.1, `client.ocr.process` with
  `server_url` pointed here) parses the PDF unchanged: 144 dpi dimensions,
  `img-0.jpeg` with its box and base64, the table as `tbl-0.html`.

**Not done here, moved to O11**: a page's regions decoding together. A page
is still one recognition after another (27 of them, 8.1 s, on the demo
page).

## Open questions

- O-o1: how far does PIL vs torch bicubic move a crop's text? (decision 4)
- ~~O-o2: do ERNIE's activations fit fp16?~~ Yes, with 24x to spare (O0).
- O-o3: ~~HF's `pp_doclayout_v3` vs Paddle's model~~ agree (scores within
  0.001, O7); what remains is PaddleX's post-processing (thresholds, NMS,
  merges, box clipping, polygons), which O8 ports and gates.
- O-o4: the report's own throughput numbers, for a reference to beat.

## Handoff

**2026-09-27, session 2.** O3 and O4 done: `ocr/gpu.go` (`NewGPUTower`,
`Forward(ctx, *Image)` → the image-token rows) and `ocr/lm.go` (`LoadLM`,
`Pass([]Row, logits)`), gated by `go test ./ocr` (28 s, all gates).
`qimage/vision` gained the hooks listed under O3; the ace_lm shaders gained
build switches (O4).

- O5 done the same day: `ocr.Engine` + `cmd/ocr`, gated by
  `go test ./ocr -run TestEngine`.
- O6 done the same day: `serve -ocr`, PaddleOCR's pipeline runs unchanged
  against it (section O6). `ai.service`'s machine-B line has `-ocr` but is
  **not deployed**: run `./deploy.sh` when the live server may restart.
- `.venv-paddle` exists now (O6's gate command is in its section).
- O7 done the same day: `ocr/layout` (CPU oracle + device trunk),
  `reference/dump_doclayout.py`; `go test ./ocr/layout` is ~85 s with the
  fp16 ladder, `-short` skips the ladder and the timings.
- O8a done the same day: `ocr/page` + `cmd/ocr -page`, oracles
  `reference/dump_ocr_page.py` (needs `serve -ocr` on :18080, see its
  docstring) and `reference/dump_ocr_textfns.py`, both in `.venv-paddle`.
- O9 done the same day (section O9): `serve -ocr` serves `/v1/ocr` and
  `/layout-parsing`, the HTTP gate is `reference/gate_ocr_http.py`.
  `ai.service` carries `-ocr`, which now means 3.35 GB with the layout;
  **still not deployed**.
- Next: **O10**, accuracy on an OmniDocBench v1.6 subset against the card
  (96.33 overall), the way K8 reproduced Kev's card: fetch the benchmark
  and its evaluation code, run a few hundred pages through the Go page
  pipeline (rect mode) and, as the control, through PaddleX's own pipeline
  (`doc_parser --layout_shape_mode rect` against `serve -ocr`), and score
  both. That also prices O8b (rect against auto) and O11's int8/batching
  later. Or **O11** first if speed matters more: batched decode across a
  page's regions (decision 10), the scalar attention at depth, the host
  decoder of the layout.

**2026-09-27, session 1.** Plan written; O0, O1 and O2 done.

- Weights: `models/PaddleOCR-VL-1.6/`, `models/PP-DocLayoutV3/`; test
  images in `testdata/ocr/`; PaddleX sparse in `~/repos/PaddleX`.
- Oracle: `.venv/bin/python reference/dump_ocr.py` then
  `reference/dump_ocr_tokens.py` (it reads the dump's generations). No new
  packages were needed for the HF oracle; `sentencepiece` was added to
  `.venv` for llama.cpp's converter.
- Go: `ocr/` holds the host side (tokenizer, prompt + positions, processor);
  `go test ./ocr` runs its gates in 0.3 s. `llm/pixels` gained
  `ResizePillow` (Pillow's fixed 22-bit coefficients) beside torch's
  `Resize`; its default is unchanged.
- `ocr/vision.go` is the CPU tower + projector on `qimage/vision`'s
  exported blocks (`go test ./ocr` without `-short` adds its ~15 s gate).
- llama.cpp's GGUFs are in `models/PaddleOCR-VL-1.6-GGUF/`; the baseline
  table is in O0.
- (O3/O4's plans, done, are in their sections above.)
