# O10 — accuracy on OmniDocBench

*The O10 stage, broken out of `OCR.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`ocr-vertical.md`](ocr-vertical.md).*


**The benchmark**: OmniDocBench v1.6 (`opendatalab/OmniDocBench` on HF at
`aa1ee96d`, 1,651 pages, in `~/repos/omnidocbench-data/ds`) and its
evaluator (`~/repos/OmniDocBench` at `f133a71`), run in the authors' Docker
image (`ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204`: TeX Live,
ImageMagick 7 and Ghostscript for CDM) with the README's config.
Overall = ((1 − text edit) × 100 + formula CDM + table TEDS) / 3, page
averages, as the leaderboard and the evaluator's own `parse_results.py`
compute it. The card's row (v1.6_full): **96.34**, text edit 0.0326, CDM
97.53, TEDS 94.76, TEDS-S 97.10, reading order 0.1278.

**The subset**: every fifth page per data source
(`reference/omnidocbench_subset.py --every 5`), 331 pages, all ten sources
in proportion. A prediction is `<stem>.md`, PaddleX's plain markdown
(`pretty=False`, as the benchmark's `PaddleOCR_img2md.py` saves it).

**Three arms, one engine** (every recognition is our `serve -ocr`, fp16):

- `go_rect`: the Go pipeline, `cmd/ocr -page -dir IN -list L -out OUT`
  (new: a directory mode that resumes and logs timings.tsv / failed.tsv);
  device layout, rect mode.
- `px_rect`: PaddleOCR 3.7.0's own `PaddleOCRVL` pipeline v1.6 (Paddle's
  layout on the CPU, PaddleX's glue, crops JPEG-encoded by its vllm-server
  client) against `serve -ocr`, `layout_shape_mode="rect"`
  (`reference/omnidocbench_paddlex.py`).
- `px_auto`: the same at PaddleX's default, polygons.

Scored by `reference/omnidocbench_eval.sh` and
`reference/omnidocbench_score.py`:

| arm | Overall | Text Edit | Formula CDM | Table TEDS | TEDS-S | Read Order |
|---|---|---|---|---|---|---|
| card (all 1,651 pages) | 96.34 | 0.0326 | 97.53 | 94.76 | 97.10 | 0.1278 |
| go_rect, first run | 95.61 | 0.0397 | 98.08 | 92.73 | 94.87 | 0.1392 |
| **go_rect** | **96.13** | 0.0333 | 98.08 | 93.65 | 96.01 | 0.1333 |
| px_rect | 96.14 | 0.0326 | 98.26 | 93.42 | 95.84 | 0.1226 |
| px_auto | 95.67 | 0.0328 | 96.88 | 93.42 | 95.84 | 0.1226 |

CDM: 449 formulas, TEDS: 136 tables, no timeouts or errors in either.

**What it says.**

1. **The engine reproduces the card.** PaddleX's pipeline around it lands
   the card's text edit to four places; Overall is 0.2 under the card on a
   subset whose mix is not the full set's (CDM is 0.7 over, TEDS 1.3
   under), so the subset is not the place to split hairs finer.
2. **The first Go run lost 0.52, and 90% of the text-edit gap was two
   pages**: `Parse` refused a page with a figure inside a table
   (O8b's `tokenize_figure_of_table`), which scores as an empty page.
   Ported (`ocr/page/figtoken.go`): CPython's Mersenne Twister for
   `random.seed(1024)`/`shuffle`, and OpenCV 4.10's `getTextSize` and
   `putText` in Hershey simplex with LINE_AA (ThickLine, LineAA,
   FillConvexPoly, EllipseEx from `drawing.cpp`), then untokenize in
   assembly. **Byte for byte**: `TestFigureTokens` holds 120 generated
   tables (284 tokens, figures 5–810 px, noise and flat crops) to
   PaddleX's function under cv2 (`reference/dump_ocr_figtoken.py`), 0
   bytes off; `TestGlue` gained the OmniDocBench page
   (`dump_doclayout.py --case figtab1=…`, `dump_ocr_page.py --case
   figtab1`) with its painted table crop identical by SHA-256 and the
   model writing the four tokens back into `<img>`s. The second run: 96.13.
3. **What remains is reading order** (0.133 against 0.123; 37 pages worse,
   7 better, 281 identical) **and it is the layout model's runtime, not the
   glue**: `TestGlue` shows the glue is PaddleX's given the same
   detections, and on the worst page HF's model (which ours matches) finds
   a 0.54 `paragraph_title` that Paddle's does not report at all. Not the
   resize: torch's no-AA bicubic and cv2's (whose 8-bit cubic is Intel
   IPP's, closed; its output is cv2's float path rounded, to 3 values in
   1.9 M) differ by 1.5e-4 on average on that page's input. O-o5.
4. **PaddleX's default polygons cost formulas here**: `auto` = `rect` on
   text, tables and order, and 1.4 points of CDM lower. So O8b (polygons)
   is not an accuracy item on this evidence; rect stays the Go default.

**Timings** (not a speed measurement: three clients shared the GPU, and
the live `ai` service ran beside them): the Go arm 11.8 s a page mean,
8.2 median, 143 s worst (a newspaper, 194 regions), 19 recognitions a
page on average; layout 0.31 s. The recognitions are one after another,
which is O11's batching.

**Layout gates, widened by the new page** (every region, label and order
still HF's): at fp32 a region's rank among all 300 queries may move by one
(HF sums votes in float32 and argsorts unstably; figtab1 has such a
near-tie with an undetected query); the device's fp16 score bound is
1e-2 (figtab1 moves one score 4.6e-3 from a trunk no worse than the other
pages', 2.8e-3). The JPEG page with a figure in a table stayed out of the
layout dumps: its differences are Go's JPEG decoder, which those gates
already exclude.
