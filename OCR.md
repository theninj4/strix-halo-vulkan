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
| O3 | Tower + projector on the GPU (from `qimage/vision/gpu.go`) | |
| O4 | ERNIE on the GPU (from `ace/lm`): prefill, KV decode, M-RoPE sections, head; teacher-forced logits, then greedy equality | |
| O5 | Element level end to end, `cmd/ocr`, all six tasks against HF greedy | |
| O6 | Serve element level (`-ocr`, chat door, vLLM extras); **gate: PaddleOCR's `doc_parser --vl_rec_backend vllm-server` against us, unchanged** | |
| O7 | PP-DocLayoutV3 on the CPU then the GPU, against HF `pp_doclayout_v3` | |
| O8 | Page glue in Go: box post-processing, crops, merges, per-label prompts, OTSL→HTML, markdown; against PaddleX's functions | |
| O9 | Serve page level: `/v1/ocr`, `/layout-parsing`, PDFs; a page's regions batched | |
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

## Open questions

- O-o1: how far does PIL vs torch bicubic move a crop's text? (decision 4)
- ~~O-o2: do ERNIE's activations fit fp16?~~ Yes, with 24x to spare (O0).
- O-o3: what are HF's `pp_doclayout_v3` and PaddleX's post-processing
  disagreements, if any? (O7/O8 will find out the same way decision 3 was.)
- O-o4: the report's own throughput numbers, for a reference to beat.

## Handoff

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
- Next: **O3**, the tower on the GPU. Start from `qimage/vision/gpu.go` and
  change what O2 changed: patch 14 (588 values, pad to the GEMM's K), raster
  order for the rope table and positions, the 27² grid with align_corners
  False, the post-LN, then the projector (pre-norm eps 1e-5, the 2x2
  gather, erf GELU). Gate every `vis.*` and `proj` of all six cases,
  including the page's 5,120 patches, against llama.cpp's 1,045 ms prefill.
  Then **O4**, ERNIE from `ace/lm`: its constants become ERNIE's (1024, 16/2
  heads, 3072, 18 layers), `ace_lm_prep.comp` loses the q/k norm and gains
  the chunked `[16, 24, 24]` 3-position rope.
