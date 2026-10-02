# OBJ-DETECTION — boxes, labels and scores (RF-DETR-L)

> **Moved to `research/` on 2026-10-02** — this was `OBJ-DETECTION.md` at the
> repo root, and anything citing `OBJ-DETECTION.md` resolves here.
>
> **Live tracking and session handoff doc, opened 2026-09-29.** Stage letters
> are **B** (boxes; D and R are taken by the LLM and speech archives). It is
> updated here in place; when the vertical closes, it is frozen in place and
> [`README.md`](README.md)'s index marks it closed. Long stage write-ups go in `b*.md` files beside it. Until then: tick a
> stage when its gate passes, put its measured numbers under it, and keep
> **§ Handoff** at the bottom current. A new session should be able to start
> from there.

## What we are building

`GOALS.md` item 11: object detection via
[`Roboflow/rf-detr-large`](https://huggingface.co/Roboflow/rf-detr-large)
(read at `f62f7dd5`, 2026-05-20; Apache-2.0; paper
[arXiv 2511.09554](https://arxiv.org/abs/2511.09554); code
[`roboflow/rf-detr`](https://github.com/roboflow/rf-detr) read at `b7d79c4`,
2026-09-29). The upstream README gives RF-DETR-L **56.5 AP50:95 / 75.1 AP50
on COCO val2017** (all 5,000 images, pycocotools), 33.9 M parameters, 704×704
input, and **6.8 ms on a T4** (TensorRT fp16, batch 1).

The product: **an image goes in, and a list of `(box, label, score)` comes
out over the 80 COCO classes**. Optionally the same image comes back with
the boxes and labels drawn on it. There is no NMS and no prompt. The
model is end to end: 300 queries in, each query's box and 91 sigmoid scores
out, then a top-300 over queries × classes and a threshold.

It is the smallest vertical so far (~61 MB in fp16) and the most
familiar. **Almost every part has already been built somewhere in this
repo:**

| part | nearest thing we have |
|---|---|
| DINOv2-S ViT, windowed and global attention | `qimage/vision` device tower (Qwen2.5-VL's windowed + full-attention blocks), driven through hooks as `ocr/gpu.go` does |
| C2f projector (1x1 and 3x3 convs, SiLU) | `ocr/layout` device trunk: im2col + `dit_gemm` + `qvit_bias_act` (SiLU = 4), `layout_ops.comp` |
| two-stage query selection, deformable decoder | `ocr/layout` host head (PP-DocLayoutV3 is an RT-DETR): anchors, top-k, deformable cross-attention with `grid_sample` bilinear/zeros |
| image decode | `llm/pixels` (PNG, JPEG, EXIF-free RGB) |
| drawing boxes and text | `ocr/page/figtoken.go`: a port of OpenCV 4.10's `putText` (Hershey simplex), anti-aliased lines and filled rectangles. Roboflow's `supervision` annotators draw with exactly these cv2 calls |

So this vertical mostly **assembles existing parts**. The new work is the glue and a
few pieces RF-DETR does differently (§ Traps).

### Sources, and who is ground truth

- **Upstream `rfdetr`** (the authors' package, `src/rfdetr/`, Apache-2.0)
  is ground truth for preprocessing, post-processing and the COCO number.
  B0 installs it into its own venv (`.venv-rfdetr`) so its torch pin can't
  disturb `.venv`.
- **HF transformers 5.17.0** (`models/rf_detr`, already in `.venv`) has a
  native port that loads the Hub checkpoint. It is the **oracle we dump
  from**, because its intermediate tensors are easy to reach (the same arrangement
  as OCR). Where it disagrees with upstream, upstream wins and the dump is
  overridden (decision 3).
- The Hub checkpoint is the **upstream state dict** (`backbone.0.encoder…`,
  `transformer.decoder…`, `refpoint_embed`), which HF renames on load. The
  Go loader reads the names as stored.

## The model, from config.json and the safetensors header

509 tensors, all fp32, 33.93 M parameters (136 MB).

| part | params used | notes |
|---|---|---|
| backbone: DINOv2-S, 12 layers | ~22.0 M | hidden 384, 6 heads × 64, MLP 1536 GELU, LN eps 1e-6, q/k/v separate with biases, LayerScale |
| patch embed + positions | 1.04 M | Conv 16×16 stride 16; positions `[1, 1937, 384]` = CLS + **44×44**, so at 704² they need **no interpolation** |
| projector (C2f, one level P4) | ~1.4 M | 1536 → 256, 3 bottlenecks at 128 |
| two-stage head, group 0 | ~0.26 M | `enc_output.0` + LN, `enc_out_class_embed.0` (91), `enc_out_bbox_embed.0` (3-layer MLP) |
| decoder: 4 layers | ~10 M | d 256, self-attn 8 heads, deformable cross-attn **16 heads × 1 level × 2 points**, ReLU FFN 2048, post-norm |
| queries | 0.08 M | `query_feat[:300]`, `refpoint_embed[:300]` |
| final heads | 0.22 M | `class_embed` 256→91, `bbox_embed` 3-layer MLP → 4 |
| **unused at inference** | ~3.6 M | Group-DETR groups 1–12 (`enc_output.{1..12}`, their heads, `query_feat[300:]`, `refpoint_embed[300:]`), `mask_token` |

About **30 M used, ~61 MB in fp16**.

### How an image runs

1. **Preprocess.** RGB → float in [0, 1] → **bilinear resize to 704×704,
   `antialias=False`, aspect ratio not kept** → ImageNet mean/std. The
   original size is kept for the boxes.
2. **Backbone.** Patchify to 44×44 = 1,936 tokens, add positions, prepend
   CLS. **Window partition 2×2**: four windows of 22×22 = 484 tokens, the
   CLS token **copied into each window** (4 × 485 = 1,940 rows). Layers
   0,1,2,4,5,7,8,10,11 attend within a window. Layers **3, 6, 9 are
   global**: the four windows are joined into one sequence of 1,940 tokens
   (four CLS copies included) and split again afterwards.
3. **Feature taps.** `stage3/6/9/12` = the hidden states after layers 3, 6,
   9 and 12 (1-based), each through the backbone's final LayerNorm
   (`apply_layernorm`), CLS dropped, windows unpartitioned to 44×44×384.
4. **Projector.** Concatenate the four taps on channels (1536) → C2f:
   `cv1` 1x1 → 256, split 128 | 128, three bottlenecks on the last half (each
   two 3x3 conv → **LayerNorm over channels** → SiLU), concatenate the 5 × 128 halves
   (640) → `cv2` 1x1 → 256 → a final channel LayerNorm. This is the **memory**: 1,936
   rows × 256, one level at stride 16.
5. **Two-stage selection.** A proposal per memory row: centre
   `((x+0.5)/44, (y+0.5)/44)`, size 0.05 (all 1,936 are inside (0.01, 0.99),
   so none is masked at 704²). `enc_output.0` + LN → class logits (91) and box
   deltas → refined proposal. **Top 300 rows by max class logit.**
6. **Queries.** Reference boxes = the 300 selected proposals **refined again
   by the learned `refpoint_embed[:300]` deltas**. The decoder's input
   content is the **learned `query_feat[:300]`**, not the selected
   encoder features.
7. **Decoder, 4 layers.** `query_pos` = MLP(sine embedding of the 4-d
   reference box), computed **once**. Each layer: self-attention (q = k =
   x + pos, v = x) → add & LN → deformable cross-attention (query x + pos,
   16 heads × 2 points around the reference box, `value_proj` over the
   memory) → add & LN → FFN → add & LN. The decoder's `layernorm` is
   applied to the last layer's output only for the head. The next layer
   reads the un-normed state. **Reference boxes do not move between layers.**
8. **Head.** `class_embed` → 91 logits; `bbox_embed` → deltas → refined
   against the reference box.
9. **Post-process.** Sigmoid, **top 300 over the 300 × 91 flattened
   scores** (so a query can yield two labels), cxcywh → xyxy, scaled to the
   original image size, then the threshold (upstream `predict` default 0.5,
   the model card uses 0.35). The label is the **COCO category id**
   (1–90, with gaps). Names come from upstream `assets/coco_classes.py`.

## The details that decide everything (traps)

Each of these produces plausible numbers when read wrong. None of them
raises an error.

1. **Boxes are `bbox_reparam`, not inverse-sigmoid.** `cxcy' = d_xy · wh +
   cxcy`, `wh' = exp(d_wh) · wh`. `ocr/layout`'s RT-DETR head uses
   `inverse_sigmoid` refinement. **Do not copy that part.**
2. **Group DETR: take group 0 and the first 300 rows.** Thirteen copies of
   the two-stage head and 3,900 query rows are stored. Any other group
   loads fine and detects worse.
3. **The decoder input is `query_feat`, not the gathered encoder rows.**
   The gathered rows (`enc_outputs_class`) exist only for the training
   loss.
4. **Reference boxes are fixed across decoder layers** (LW-DETR's "lite"
   refinement), so `query_pos` is computed once. Only the final head
   refines. Deformable DETR's per-layer refinement is the wrong model.
5. **LayerNorm, not BatchNorm, in the projector.** It cannot be folded into
   the convs as `ocr/layout` folds its BNs. The epsilons differ: the
   conv-norm layers use `layer_norm_eps` **1e-5** and the projector's
   closing LN defaults to **1e-6**. The backbone is 1e-6. B0 prints every
   module's eps.
6. **The feature taps go through the final LayerNorm**, all four of them,
   not just the last. Taps are in windowed layout and are unpartitioned
   before the projector.
7. **The CLS token is copied into each window**, and in the global layers
   all four copies take part in attention. A port that keeps one CLS token gives
   slightly different features everywhere.
8. **The window partition's axis order.** HF views the grid as
   `(nw, w/nw, nw, h/nw)` and transposes. At 44×44 the grid is square, so a
   swapped h/w is silent. B2's gate compares windowed tokens element for element.
9. **LayerScale** (`layer_scale1.lambda1`, `layer_scale2.lambda1`, 384
   each a layer, DINOv2's per-channel γ). It is folded into
   `attention.output.dense` and `mlp.fc2` (weights and biases) at load.
   Leaving it out gives a working ViT with the wrong residual mix.
10. **91 logits, including `N/A` and the unused ids.** The top-k runs over
    all 91, as upstream's does. We keep it that way and don't filter.
11. **DINOv2 without registers has high-norm "artifact" tokens.** The model
    card says "with-registers style", but the header has no register tokens.
    fp16 range is a real question here, unlike OCR's 24× headroom. B0
    audits it.

## Decisions (so future sessions don't relitigate)

1. **Detection only.** The segmentation, keypoint and XL/2XL (PML-licensed)
   variants are out of scope. Other Apache sizes (N/S/M) share the
   architecture and differ only in config, so the loader reads `config.json`
   rather than hard-coding L. That keeps them a cheap follow-up, but they are
   not a goal.
2. **Fixed 704×704.** Upstream `predict` allows another size, but the
   windows need the grid divisible by 2 and the positions would need
   interpolating. One static graph at 704², recorded once, as
   PP-DocLayoutV3's 800² is.
3. **Preprocessing is upstream's**: float bilinear with `antialias=False`
   after the /255 (upstream's comment says this matches `cv2.INTER_LINEAR`).
   HF's fast processor resizes differently. B0 measures how many detections
   the difference moves, and the dumps use upstream's resize whatever the
   answer is. The same rule as OCR decision 4: the authors' own processor
   wins.
4. **fp16 on the device with fp32 sums**, like every other vertical, **unless
   B0's range audit says otherwise** (trap 11). If a few tokens overflow,
   the fallback is fp32 residuals with fp16 GEMM operands, which is what the
   `qimage/vision` tower already does. No int8: at 61 MB there is nothing to
   save.
5. **The door.** OpenAI has no detection endpoint, so there is no OpenAI
   shape to follow (memory: *API follows OpenAI's standard*, "check the
   reference before inventing"). The closest thing to a standard for this
   model is **Roboflow Inference's `POST /infer/object_detection`**. Roboflow
   ships the model and its `inference_sdk` client. The response is
   `predictions[]` of `{x, y, width, height, confidence, class, class_id,
   detection_id}` (centre-based boxes in pixels) plus `image {width,
   height}`, and with `visualize_predictions: true` an annotated image comes back
   in `visualization`. That field also covers the "draw the boxes" half of
   the request, and **the gate is `inference_sdk` working unchanged against
   us** (as the Mistral SDK was for OCR). One door only. HF's
   `object-detection` task shape is the alternative, and it is listed under
   § Open questions, not built.
6. **Drawing matches `supervision`'s** `BoxAnnotator` + `LabelAnnotator`
   defaults. This is what Roboflow's server draws, and it is all cv2
   (`rectangle`, `putText` Hershey simplex). `ocr/page/figtoken.go` already
   ports those cv2 calls exactly. B5 lifts them into a shared package,
   adds colour and the printable-ASCII glyphs, and gates pixel-identical
   against `supervision` on PNG inputs.
7. **Deployment: machine B, resident.** It sits with speech, TTS, embeddings,
   Kev and OCR. At ~61 MB plus < 0.5 GB of activations, loading it from the swap slot
   would cost more than keeping it.
8. **A device graph first, then a device decoder.** O7's order: the trunk
   (backbone + projector) goes on the device while the head stays in host Go,
   shared with the CPU path. The head moves to the device in B8 only if
   timings show it matters.

## Budget

- **Device memory**: ~61 MB of weights; activations at 1,940 × 1,536 × 4 B
  are a few MB per map. The whole graph with every map in its own
  arena slot is well under 0.5 GB.
- **Compute**: backbone linears ≈ 12 × 1,940 × 2 × 1.77 M ≈ **82 GFLOP**;
  attention ≈ 9 windowed × 1.4 + 3 global × 5.8 ≈ **30 GFLOP**; projector
  ≈ **6 GFLOP**; two-stage head over 1,936 rows ≈ 1 GFLOP; decoder ≈ 3
  GFLOP. **~120 GFLOP an image.** At the 25–35 TFLOP/s our WMMA paths
  reach, that is **~4–5 ms** of arithmetic. A realistic device forward is
  10–20 ms, against the T4's 6.8 ms under TensorRT.
- **Host**: the decoder in scalar Go took ~15 ms a layer in O7 at 8 heads × 3
  levels × 4 points. RF-DETR has 4 layers at 16 × 1 × 2, so ~30–60 ms,
  hence decision 8. **JPEG decode of a 12 MP photo** (Go's `image/jpeg`)
  may cost more than the model. B1 measures it.
- **CPU oracle**: tens of GFLOP in PyTorch fp32 on 32 cores is ~0.2–0.5 s an
  image, so dumps are cheap and a CPU run over all 5,000 val2017 images
  takes about half an hour.

## Stages

| # | Stage | State |
|---|---|---|
| B0 | Weights, `.venv-rfdetr` (upstream), HF dump script, decision 3 measured, fp16 range audit, COCO val2017 + upstream's own AP on this machine, CPU speed baseline | not started |
| B1 | Host side in Go (`detect/`): decode, float bilinear no-AA resize, normalize, COCO label table, post-process (top-300, xyxy, scale, threshold) | not started |
| B2 | The whole model on the CPU in fp32 against the dumps, every stage | not started |
| B3 | Backbone + projector on the device | not started |
| B4 | Two-stage head + decoder wired to the device trunk: detections equal to the fp32 oracle | not started |
| B5 | `cmd/detect`: image → JSON and an annotated PNG, drawing gated against `supervision` | not started |
| B6 | Serve: `serve -detect`, `POST /infer/object_detection`; **gate: `inference_sdk` unchanged against us** | not started |
| B7 | Accuracy: all of COCO val2017 through the Go engine, pycocotools, against upstream's AP and the README's 56.5 | not started |
| B8 | Performance: the decoder on the device, several images a pass, arena reuse | not started |

### B0 — weights, environment, oracles

- `models/rf-detr-large/` from `resolve/f62f7dd5…` (config,
  preprocessor_config, model.safetensors, 136 MB).
- `.venv-rfdetr`: `uv pip install rfdetr` at the `b7d79c4` tag or release
  nearest to it, plus `supervision` (it comes as a dependency). Record the
  versions here. Upstream fetches `rf-detr-large-2026.pth`; **check that its
  state dict equals the Hub safetensors** tensor for tensor. If it doesn't,
  the Hub copy is not the model the README measured, and that has to be known before B7.
- `reference/dump_rfdetr.py`, in the pattern of `dump_doclayout.py`: HF model,
  **upstream's preprocessing** (decision 3), dumping pixels, patch
  embeddings, windowed tokens after layer 0, the hidden states after each
  layer, the four taps after the LN, each projector stage, the memory, the
  proposals, enc class/box, the top-300 indices, reference boxes,
  `query_pos`, each decoder layer, the logits, the boxes, and the detections
  at 0 and 0.5.
- **Upstream vs HF on the same pixels**: logits and boxes should agree to
  ~1e-5. Then **upstream preprocessing vs HF's processor**: count detections
  at 0.5 that change. Record both.
- `testdata/detect/`: COCO `000000039769.jpg` (the cats and remotes of the
  card) and a handful more: a crowd, small objects, a PNG with alpha, a
  grayscale image, a portrait-aspect photo, a large 12 MP JPEG.
- **fp16 range audit** (as O0 and O7's ladder): the largest activation per
  stage, and the tokens over 1e4 if there are any. Trap 11 is the reason.
- **COCO val2017** (5,000 images + `instances_val2017.json`) into
  `~/repos/coco` and upstream's own evaluation on this machine's CPU.
  **That number, not the README's, is B7's target**. The README's runs
  TensorRT fp16 on a T4.

### B1 — host side

`detect/` package: `pixels` decode to RGB (EXIF orientation: check what
upstream does, which is PIL without `exif_transpose` unless the caller
applies it), float bilinear resize without antialiasing (new; `pixels`
has the uint8 bicubic `ResizeNoAA` from O7 and the antialiased bicubic
from V3, not this one), normalize, CHW. Post-process as upstream's
`PostProcess`. Labels from `coco_classes.py`.

Gate: pixels against the dump **bit-exact on PNG** (float bilinear is
deterministic; if torch's CPU kernel accumulates in a different order, an
ulp is the bound), JPEG within Go's IDCT differences as in O1. Post-process
on the dumped logits: identical detections.

### B2 — the CPU port

`detect/` model loader (upstream names, group 0, LayerScale folded) and a
scalar/parallel Go forward. `ocr/layout/ops.go` has the convs, and
`qimage/vision/vision.go` has a ViT block on the CPU to crib from. Traps 1–9
are all checked here, against the dumps.

Gate (`go test ./detect`): every dumped stage ≤ 1e-4 rel to HF fp32, the
top-300 selection identical, detections at 0 identical (label, order),
scores ≤ 1e-5, boxes ≤ 0.01 px on every test image.

### B3 — backbone + projector on the device

The ViT runs through `qimage/vision`'s device tower via hooks, as
`ocr/gpu.go` did for PaddleOCR. The differences: learned positions and no
rope, patch 16 at 44×44, a CLS row per window, 2×2 windows with every
fourth layer global, LayerScale folded, taps after layers 3/6/9/12 through
the final LN. The projector uses `ocr/layout/gpu.go`'s machinery: 1x1 convs as a
narrowed A operand, 3x3 by im2col, `dit_gemm`, and a channel LayerNorm,
which in channel-last layout is a row LN over 128 or 256 columns.
Concatenation is never built (O7's column-offset trick). Static graph,
recorded once.

Gate: the four taps and the memory against HF fp32 within the fp16
ladder's prediction (B0), and B2's head on the device memory gives the
same detections as fp32.

### B4 — the head, end to end

The two-stage head's GEMMs over 1,936 rows and `value_proj` for each of
the 4 layers run on the device, like O7's seven 13,125-row products. The top-k,
the decoder and the final head run on the host, shared with B2.

Gate: on every test image, every detection at 0.35 has the fp32 oracle's
label and rank, scores ≤ 1e-3, boxes ≤ 0.5 px (O7's bounds). Report
the time split (upload, trunk, readback, host head).

### B5 — `cmd/detect` and the drawing

`go run ./cmd/detect -img x.jpg [-threshold 0.5] [-out x.boxes.png]
[-json]`. The cv2 port in `ocr/page/figtoken.go` moves to a shared package
(`util/draw` or `llm/pixels/draw`) with colour, the full printable-ASCII
Hershey simplex table, and `supervision`'s default palette, box thickness,
label placement and text scale.

Gate: on the PNG test images, **our annotated image is byte-identical to
`sv.BoxAnnotator().annotate` + `sv.LabelAnnotator().annotate`** on the same
detections (the detections come from the dump, so the drawing is gated
alone).

### B6 — serve

`backend/detect.go`, `api/` route, flag `-detect <dir>`. Roboflow
Inference's request: `{model_id, image: {type: base64|url, value},
confidence, visualize_predictions, visualization_labels, ...}`. **B6
reads `inference_sdk`'s client and the server's request models whole**,
decides the `model_id` alias (upstream uses `rfdetr-large`), and lists the fields we
refuse rather than fake, in API.md's pattern. Image URLs are fetched, as
the chat door fetches them (memory: *API follows OpenAI's standard*,
behaviour not just shape). Update `research/api-server.md` and `deploy.sh`'s machine-B line.

Gate: `InferenceHTTPClient(api_url="http://…").infer(img,
model_id=…)` and the visualisation path work unchanged, and the numbers
equal `cmd/detect`'s.

### B7 — accuracy

The Go engine over all 5,000 val2017 images, output as COCO results JSON,
scored with pycocotools as upstream's `coco_eval.py` does (300 detections an
image, no threshold). Also report AP50 and the small/medium/large split.

Gate: **within 0.1 AP of upstream's own number on this machine (B0)**,
and within reach of the README's 56.5. A gap with the README that
upstream shares is not ours.

### B8 — performance

Only what B4's timings justify. The candidates are: the decoder on the device
(small GEMMs, a 300×300 self-attention, a deformable gather shader, 1
level × 2 points so the shader is simple), several images in one pass (the
windows are already a batch axis, so N images are 4N windows plus N-way
block-diagonal global layers), arena reuse, and overlapping JPEG decode
with the previous forward.

## Open questions

- **B-o1** Does the Hub safetensors equal upstream's `rf-detr-large-2026.pth`? (B0)
- **B-o2** How much does HF's processor vs upstream's resize move detections? (B0; decision 3 holds regardless)
- **B-o3** fp16 headroom with no registers (trap 11). (B0)
- **B-o4** A second envelope in HF's `object-detection` task shape
  (`[{score, label, box: {xmin, ymin, xmax, ymax}}]`, what
  `huggingface_hub.InferenceClient.object_detection` expects)? Cheap to add, but
  OCR removed its second envelope for costing upkeep (OCR decision 2). Only if a
  client needs it.
- **B-o5** Other RF-DETR sizes or fine-tuned checkpoints through the same
  loader (decision 1). Not a goal.
- **B-o6** Letterboxing instead of stretching? No: the model was trained on
  stretched squares, and upstream stretches. Listed so no one tries it.

## Handoff

Nothing built yet. Plan written 2026-09-29 from the HF config, the
safetensors header (509 tensors, upstream names), HF's `modeling_rf_detr.py`
and upstream's `config.py` / `detr.py` `predict`. **Next: B0.** Download
the weights, set up `.venv-rfdetr`, write `reference/dump_rfdetr.py`, and
answer B-o1 to B-o3 before any Go is written.
