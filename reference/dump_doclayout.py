"""Dump PP-DocLayoutV3's fp32 reference for OCR.md stage O7.

The oracle is HF transformers' native port (5.17.0, `pp_doclayout_v3`), in
fp32 on the CPU, from models/PP-DocLayoutV3 (PaddlePaddle's safetensors
conversion). The image processor is HF's: an 800x800 bicubic resize without
antialiasing (torchvision; it stands in for PaddleX's cv2.INTER_CUBIC), then
x/255. The pixels are saved, so the Go model is gated on the same input and
the resize is gated on its own.

For every image in CASES this writes, under reference/out/doclayout/<name>/:

  record.json   image, size, the post-processed detections at threshold 0.5
                (label, score, box in image pixels, reading order), and each
                tensor's shape
  tensors.safetensors
                pixels          [3, 800, 800]
                bb.stem         backbone stem, [48, 200, 200]
                bb.stage{1..4}  backbone stages: 128@200, 512@100, 1024@50, 2048@25
                enc.proj{0..2}  the 1x1 conv+BN projections of stages 2-4 to 256
                enc.aifi        the AIFI layer over the stride-32 map
                enc.pan{0..2}   the hybrid encoder's outputs (strides 8, 16, 32)
                enc.mask_feat   the mask prototypes, [32, 200, 200]
                dec.memory      the flattened decoder-input projections, [13125, 256]
                enc.class       enc_score_head over the memory, [13125, 25]
                enc.coord       enc_bbox_head + anchors (logits), [13125, 4]
                enc.topk        the 300 selected memory rows (as float)
                dec.init_ref    the decoder's initial reference (logits, after
                                the mask-enhanced boxes), [300, 4]
                dec.layer{i}    each decoder layer's hidden state, [300, 256]
                dec.ref         each layer's refined boxes (cxcywh, sigmoid), [6, 300, 4]
                dec.logits      each layer's class logits, [6, 300, 25]
                order_logits    the last layer's order logits, [300, 300]
                masks           the last layer's mask logits, [300, 200, 200]

    .venv/bin/python reference/dump_doclayout.py [--only page]
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoImageProcessor, AutoModelForObjectDetection

CASES = [
    ("page", "testdata/ocr/paddleocr_vl_demo.png"),
    ("table", "testdata/ocr/table_recognition.jpg"),
    ("formula", "testdata/ocr/general_formula_rec_001.png"),
    ("chart", "testdata/ocr/chart_parsing_02.png"),
    ("seal", "testdata/ocr/seal_text_det.png"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/PP-DocLayoutV3")
    ap.add_argument("--out", default="reference/out/doclayout")
    ap.add_argument("--only", default="")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForObjectDetection.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    m = model.model
    labels = model.config.id2label

    only = set(filter(None, args.only.split(",")))
    for name, path in CASES:
        if only and name not in only:
            continue
        t0 = time.time()
        img = Image.open(path).convert("RGB")
        inputs = proc(images=[img], return_tensors="pt")
        px = inputs["pixel_values"]

        tensors = {"pixels": px[0]}
        hooks = []

        def keep(key, pick=None):
            def hook(_mod, _inp, out):
                v = pick(out) if pick else out
                tensors[key] = v.detach().float()[0].clone()
            return hook

        bb = m.backbone.model
        hooks.append(bb.embedder.register_forward_hook(keep("bb.stem")))
        for i, st in enumerate(bb.encoder.stages):
            hooks.append(st.register_forward_hook(keep(f"bb.stage{i + 1}")))
        for i, p in enumerate(m.encoder_input_proj):
            hooks.append(p.register_forward_hook(keep(f"enc.proj{i}")))
        hooks.append(m.encoder.aifi[0].register_forward_hook(keep("enc.aifi")))
        for i, layer in enumerate(m.decoder.layers):
            hooks.append(layer.register_forward_hook(keep(f"dec.layer{i}")))
        # The memory the decoder reads: every level's projection, flattened.
        srcs = []
        for p in m.decoder_input_proj:
            hooks.append(p.register_forward_hook(lambda _m, _i, o: srcs.append(o.detach().float()[0].clone())))

        out = model(pixel_values=px, output_hidden_states=False)
        for h in hooks:
            h.remove()


        full = model.model(pixel_values=px)
        for i, t in enumerate(full.encoder_last_hidden_state):
            tensors[f"enc.pan{i}"] = t.detach().float()[0].clone()
        # mask_feat is not on the output object; recompute it the way the model does.
        feats = m.backbone(px, torch.ones(1, 800, 800, dtype=torch.long))
        x4 = feats.pop(0)
        proj = [m.encoder_input_proj[level](src) for level, (src, _mask) in enumerate(feats)]
        encout = m.encoder(proj, x4)
        tensors["enc.mask_feat"] = encout.mask_feat.detach().float()[0].clone()

        mem = torch.cat([s.flatten(1).transpose(0, 1) for s in srcs[:3]], 0)
        tensors["dec.memory"] = mem
        tensors["enc.class"] = full.enc_outputs_class.detach().float()[0].clone()
        tensors["enc.coord"] = full.enc_outputs_coord_logits.detach().float()[0].clone()
        _, topk = torch.topk(full.enc_outputs_class.max(-1).values, model.config.num_queries, dim=1)
        tensors["enc.topk"] = topk[0].float()
        tensors["dec.init_ref"] = full.init_reference_points.detach().float()[0].clone()
        tensors["dec.ref"] = out.intermediate_reference_points.detach().float()[0].clone()
        tensors["dec.logits"] = out.intermediate_logits.detach().float()[0].clone()
        tensors["order_logits"] = out.order_logits.detach().float()[0].clone()
        tensors["masks"] = out.out_masks.detach().float()[0].clone()

        # Post-processing, the box half of HF's post_process_object_detection
        # (the polygons need cv2 and are O8's).
        h, w = img.size[1], img.size[0]
        order_seq = proc._get_order_seqs(out.order_logits)[0]
        boxes = out.pred_boxes[0]
        xyxy = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], -1)
        xyxy = xyxy * torch.tensor([w, h, w, h], dtype=xyxy.dtype)
        logits = out.logits[0]
        scores = logits.sigmoid()
        nq, nc = scores.shape
        sc, idx = torch.topk(scores.flatten(), nq)
        lab = idx % nc
        q = idx // nc
        keepm = sc >= args.threshold
        dets = []
        for s, l, qi in zip(sc[keepm], lab[keepm], q[keepm]):
            dets.append({"query": int(qi), "label": labels[int(l)], "label_id": int(l), "score": float(s),
                         "box": [float(v) for v in xyxy[qi]], "order": int(order_seq[qi])})
        dets.sort(key=lambda d: d["order"])
        # PaddleX's transformers path (layout_analysis/predictor.py) runs HF's
        # post-processing at the pipeline's layout threshold, 0.3 for
        # PaddleOCR-VL-1.6: the input the page glue (O8) starts from.
        keep3 = sc >= 0.3
        dets03 = [{"query": int(qi), "label_id": int(l), "score": float(s),
                   "box": [float(v) for v in xyxy[qi]], "order": int(order_seq[qi])}
                  for s, l, qi in zip(sc[keep3], lab[keep3], q[keep3])]
        dets03.sort(key=lambda d: d["order"])

        os.makedirs(os.path.join(args.out, name), exist_ok=True)
        save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(args.out, name, "tensors.safetensors"))
        rec = {"name": name, "image": path, "size": [h, w], "threshold": args.threshold, "detections": dets, "detections_03": dets03,
               "shapes": {k: list(v.shape) for k, v in tensors.items()}}
        with open(os.path.join(args.out, name, "record.json"), "w") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        print(f"{name}: {w}x{h}, {len(dets)} detections, {time.time() - t0:.1f}s")
        for d in dets:
            print(f"   {d['order']:3d} {d['label']:16s} {d['score']:.3f} {[round(v, 1) for v in d['box']]}")


if __name__ == "__main__":
    main()
