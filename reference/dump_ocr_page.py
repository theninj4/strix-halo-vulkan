"""PaddleX's page glue on HF's layout, for OCR.md stage O8 (the page oracle).

Runs in .venv-paddle (paddlex 3.7.2), not .venv: it calls PaddleX's own
functions, unmodified, on the layout detections reference/dump_doclayout.py
wrote at the pipeline's threshold (0.3). That is exactly what PaddleX's
transformers engine for PP-DocLayoutV3 computes (layout_analysis/predictor.py
runs HF's post_process_object_detection and hands its boxes, scores, labels
and order_seq to the same LayoutAnalysisProcess).

Two choices, both PaddleX options rather than changes:

  - layout_shape_mode="rect": no polygons (they need OpenCV's contour
    tracing and shapely; O8b);
  - the VLM is the "native" semantics (raw crop pixels), served by our own
    `serve -ocr` over OpenAI chat with each crop as a lossless PNG, where
    PaddleX's vllm-server client would JPEG-encode it.

Per page, reference/out/ocr_page/<name>/ gets:

  record.json   layout_det_res (after PaddleX's layout post-processing), the
                blocks after filtering, cropping and merging (label, box,
                group id, crop size and sha256 of its RGB bytes), each VLM
                entry (block index, prompt, pixel bounds, raw text), the
                parsing_res_list JSON and the markdown
  block_<j>.png every VLM input image, RGB

    .venv-paddle/bin/python reference/dump_ocr_page.py --server http://127.0.0.1:18080/v1
"""

import argparse
import base64
import hashlib
import io
import json
import os

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

from paddlex.inference.models.layout_analysis.processors import LayoutAnalysisProcess
from paddlex.inference.pipelines.components import CropByBoxes
from paddlex.inference.pipelines.layout_parsing.utils import gather_imgs
from paddlex.inference.pipelines.paddleocr_vl.pipeline import IMAGE_LABELS
from paddlex.inference.pipelines.paddleocr_vl.pipeline import _PaddleOCRVLPipeline as PaddleOCRVLPipeline
from paddlex.inference.pipelines.paddleocr_vl.result import PaddleOCRVLResult
from paddlex.inference.pipelines.paddleocr_vl.uilts import filter_overlap_boxes, merge_blocks

PAGES = ["page", "table", "formula", "chart", "seal"]

# PaddleOCR-VL-1.6.yaml
LAYOUT = dict(
    threshold=0.3,
    layout_nms=True,
    layout_unclip_ratio=[1.0, 1.0],
    layout_merge_bboxes_mode={0: "union", 1: "union", 2: "union", 3: "large", 4: "union", 5: "large",
                              6: "large", 7: "union", 8: "union", 9: "union", 10: "union", 11: "union",
                              12: "union", 13: "union", 14: "union", 15: "large", 16: "union", 17: "large",
                              18: "union", 19: "union", 20: "union", 21: "union", 22: "union", 23: "union",
                              24: "union"},
)
MARKDOWN_IGNORE = ["number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text"]
LABELS = ["abstract", "algorithm", "aside_text", "chart", "content", "display_formula", "doc_title",
          "figure_title", "footer", "footer_image", "footnote", "formula_number", "header", "header_image",
          "image", "inline_formula", "number", "paragraph_title", "reference", "reference_content", "seal",
          "table", "text", "vertical_text", "vision_footnote"]


def sha(img_rgb):
    return hashlib.sha256(np.ascontiguousarray(img_rgb).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="reference/out/doclayout")
    ap.add_argument("--out", default="reference/out/ocr_page")
    ap.add_argument("--server", default="http://127.0.0.1:18080/v1")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    client = OpenAI(base_url=args.server, api_key="null")
    only = set(filter(None, args.only.split(",")))

    for name in PAGES:
        if only and name not in only:
            continue
        rec = json.load(open(os.path.join(args.layout, name, "record.json")))
        img = cv2.imread(rec["image"], cv2.IMREAD_COLOR)  # BGR, as PaddleX's reader
        h, w = img.shape[:2]

        # The transformers engine's boxes: [label, score, x0, y0, x1, y1, order].
        dets = rec["detections_03"]
        arr = np.array([[d["label_id"], d["score"], *d["box"], d["order"]] for d in dets], dtype=np.float32)
        proc = LayoutAnalysisProcess(labels=LABELS, scale_size=[800, 800])
        boxes = proc([{"boxes": arr}], [{"ori_img_size": (w, h)}], layout_shape_mode="rect",
                     filter_overlap_boxes=False, **LAYOUT)[0]
        layout_det_res = {"boxes": boxes}

        # PaddleOCRVLPipeline.get_layout_parsing_results, defaults: no chart
        # or seal recognition, no OCR for image blocks, merged blocks.
        image_labels = IMAGE_LABELS.copy() + ["chart", "seal"]
        vis_image_labels = IMAGE_LABELS + ["seal", "chart"]
        cfg = {
            "layout_shape_mode": "rect", "merge_layout_blocks": True, "image_labels": image_labels,
            "use_chart_recognition": False, "use_seal_recognition": False,
            **{f"{k}_{m}_pixels": v for k in ("ocr", "table", "chart", "formula", "seal")
               for m, v in (("min", 112896), ("max", 1003520))},
        }
        imgs_in_doc = gather_imgs(img, boxes)
        filtered = filter_overlap_boxes(layout_det_res, "rect")
        blocks = CropByBoxes()(img, filtered["boxes"], "rect")
        blocks = merge_blocks(blocks, non_merge_labels=image_labels + ["table"])
        entries, has_spotting, drop = PaddleOCRVLPipeline._paddleocr_vl_collect_page_vlm_entries_core(
            None, 0, blocks, imgs_in_doc, cfg)

        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        batch, id2key, vlm = {}, {}, []
        for (i, j, block_img, prompt, px, fmap) in entries:
            rgb = cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(os.path.join(out_dir, f"block_{j}.png"))
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            resp = client.chat.completions.create(
                model="PaddleOCR-VL-1.6-0.9B", temperature=0, max_completion_tokens=4096,
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                       {"type": "text", "text": prompt}]}],
                extra_body={"skip_special_tokens": not has_spotting,
                            "mm_processor_kwargs": {"min_pixels": px[0], "max_pixels": px[1]}})
            text = resp.choices[0].message.content
            b = batch.setdefault(px, {"images": [], "queries": [], "figure_token_maps": [], "vlm_block_ids": [],
                                      "curr_vlm_block_idx": 0, "vlm_results": []})
            b["images"].append(block_img)
            b["queries"].append(prompt)
            b["figure_token_maps"].append(fmap)
            b["vlm_block_ids"].append((i, j))
            b["vlm_results"].append({"result": text})
            id2key[(i, j)] = px
            vlm.append({"block": j, "prompt": prompt, "min_pixels": px[0], "max_pixels": px[1],
                        "size": [rgb.shape[1], rgb.shape[0]], "sha256": sha(rgb), "raw": text})

        parsing, tables, spotting = PaddleOCRVLPipeline._paddleocr_vl_assemble_parsing_results(
            None, [blocks], batch, id2key, drop, vis_image_labels)
        settings = {"use_doc_preprocessor": False, "use_layout_detection": True, "use_chart_recognition": False,
                    "use_seal_recognition": False, "use_ocr_for_image_block": False, "format_block_content": False,
                    "merge_layout_blocks": True, "markdown_ignore_labels": MARKDOWN_IGNORE,
                    "return_layout_polygon_points": False}
        res = PaddleOCRVLResult({
            "input_path": rec["image"], "page_index": None, "page_count": None, "width": w, "height": h,
            "doc_preprocessor_res": None, "layout_det_res": layout_det_res, "table_res_list": tables[0],
            "parsing_res_list": parsing[0], "spotting_res": spotting[0], "imgs_in_doc": imgs_in_doc,
            "model_settings": settings,
        })
        md = res.markdown["markdown_texts"]
        md_plain = res._to_markdown(pretty=False)["markdown_texts"]
        js = res.json["res"]

        def plain(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            raise TypeError(type(v))

        out = {
            "name": name, "image": rec["image"], "size": [h, w],
            "layout_det_res": boxes,
            "imgs_in_doc": [d["path"] for d in imgs_in_doc],
            "blocks": [{"label": b["label"], "box": [int(v) for v in b["box"]], "group_id": b.get("group_id"),
                        "merge_aligns": b.get("merge_aligns"),
                        "img": None if b["img"] is None else {
                            "size": [b["img"].shape[1], b["img"].shape[0]],
                            "sha256": sha(cv2.cvtColor(b["img"], cv2.COLOR_BGR2RGB))}}
                       for b in blocks],
            "vlm": vlm,
            "parsing_res_list": js["parsing_res_list"],
            "markdown": md,
            "markdown_plain": md_plain,
        }
        with open(os.path.join(out_dir, "record.json"), "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False, default=plain)
        print(f"{name}: {len(boxes)} layout boxes, {len(blocks)} blocks, {len(vlm)} VLM calls, "
              f"{len(md)} markdown chars")


if __name__ == "__main__":
    main()
