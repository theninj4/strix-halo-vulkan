# O7 — PP-DocLayoutV3

*The O7 stage, broken out of `OCR.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`ocr-vertical.md`](ocr-vertical.md).*


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
