"""Dump PaddleOCR-VL-1.6's fp32 reference for OCR.md stages O0-O5.

The oracle is HF transformers' native port (5.17.0), in fp32 on the CPU,
with two things set on purpose (OCR.md decisions 3 and 4):

  - the position grid is resized with align_corners=False, the authors' and
    vLLM's convention; HF's port says True (`--align-corners` restores it);
  - the image processor is the PIL backend (Pillow BICUBIC), which is what
    PaddleX's own processor calls. The torchvision backend's pixels are saved
    beside them so O1 can price the difference.

For every case in CASES this writes, under reference/out/ocr/<name>/:

  record.json   image, task, sizes, grid, input ids, 3-D positions, the
                greedy generation (ids, text with and without specials),
                the rope delta, and each hooked tensor's shape
  tensors.safetensors
                pixels            [P, 3, 14, 14], PIL backend
                pixels_tv         the same from the torchvision backend
                vis.embed         patch embedding + interpolated positions
                vis.block{i}      tower block outputs (every block for small
                                  images, 0/1/13/26 for large ones)
                vis.post_ln       after the tower's post_layernorm
                proj              projector output, [P/4, 1024]
                lm.embed          inputs_embeds after the image rows land
                lm.layer{i}       every decoder layer's output
                lm.final          after the final norm
                logits.prompt     the last prompt position's logits
                logits.tf         teacher-forced logits over the first
                                  --tf-steps generated tokens, [S, V]

and reference/out/ocr/ranges.json: every Linear's output absmax and every
down_proj's input absmax (the SwiGLU product) over all cases, against
fp16's 65504 (decision 7).

    .venv/bin/python reference/dump_ocr.py [--only line,table]
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.models.paddleocr_vl.image_processing_paddleocr_vl import PaddleOCRVLImageProcessor
from transformers.models.paddleocr_vl.image_processing_pil_paddleocr_vl import PaddleOCRVLImageProcessorPil

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "seal": "Seal Recognition:",
    "spotting": "Spotting:",
}

# name, image, task, max new tokens
CASES = [
    ("line", "testdata/ocr/ocr_demo.jpg", "ocr", 64),
    ("formula", "testdata/ocr/general_formula_rec_001.png", "formula", 256),
    ("table", "testdata/ocr/table_recognition.jpg", "table", 512),
    ("seal", "testdata/ocr/seal_text_det.png", "seal", 256),
    ("chart", "testdata/ocr/chart_parsing_02.png", "chart", 512),
    ("page", "testdata/ocr/paddleocr_vl_demo.png", "ocr", 768),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/PaddleOCR-VL-1.6")
    ap.add_argument("--out", default="reference/out/ocr")
    ap.add_argument("--only", default="")
    ap.add_argument("--tf-steps", type=int, default=32)
    ap.add_argument("--align-corners", action="store_true", help="HF's convention instead of the authors'")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.float32).eval()
    lm_head_ckpt = model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr()
    if not lm_head_ckpt:
        raise SystemExit("lm_head got tied to embed_tokens; the checkpoint's are different tensors (decision 5)")
    vis = model.model.visual.vision_model
    vis.embeddings.interpolation_align_corners = a.align_corners
    proc = AutoProcessor.from_pretrained(a.model)
    pil_ip = PaddleOCRVLImageProcessorPil.from_pretrained(a.model)
    tv_ip = PaddleOCRVLImageProcessor.from_pretrained(a.model)
    proc.image_processor = pil_ip

    ranges = {}

    def track(name, t):
        m = t.detach().abs().max().item()
        if m > ranges.get(name, 0.0):
            ranges[name] = m

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.register_forward_hook(lambda _m, _i, o, n=name: track(n, o))
            if name.endswith("down_proj") or name.endswith("fc2") or name.endswith("linear_2"):
                mod.register_forward_pre_hook(lambda _m, i, n=name + ".input": track(n, i[0]))
    for i, layer in enumerate(model.model.language_model.layers):
        layer.register_forward_hook(lambda _m, _i, o, n=f"lm.residual{i}": track(n, o[0] if isinstance(o, tuple) else o))
    for i, layer in enumerate(vis.encoder.layers):
        layer.register_forward_hook(lambda _m, _i, o, n=f"vis.residual{i}": track(n, o[0] if isinstance(o, tuple) else o))

    only = set(filter(None, a.only.split(",")))
    for name, path, task, max_new in CASES:
        if only and name not in only:
            continue
        t0 = time.time()
        out = os.path.join(a.out, name)
        os.makedirs(out, exist_ok=True)
        img = Image.open(path)
        orig_mode, orig_size = img.mode, img.size
        img = img.convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPTS[task]}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        ids = inp["input_ids"]
        grid = inp["image_grid_thw"]
        pixels_tv = tv_ip(images=[img], return_tensors="pt")["pixel_values"]

        tensors = {"pixels": inp["pixel_values"].float().contiguous(), "pixels_tv": pixels_tv.float().contiguous()}
        npatch = int(grid.prod())
        blocks = range(len(vis.encoder.layers)) if npatch <= 1024 else [0, 1, 13, len(vis.encoder.layers) - 1]
        hooks = []

        def keep(key):
            def f(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                tensors[key] = t.detach().float().reshape(-1, t.shape[-1]).clone()
            return f

        hooks.append(vis.embeddings.register_forward_hook(keep("vis.embed")))
        for i in blocks:
            hooks.append(vis.encoder.layers[i].register_forward_hook(keep(f"vis.block{i}")))
        hooks.append(vis.post_layernorm.register_forward_hook(keep("vis.post_ln")))
        hooks.append(model.model.projector.register_forward_hook(keep("proj")))
        lmm = model.model.language_model
        for i, layer in enumerate(lmm.layers):
            hooks.append(layer.register_forward_hook(keep(f"lm.layer{i}")))
        hooks.append(lmm.norm.register_forward_hook(keep("lm.final")))
        hooks.append(lmm.register_forward_pre_hook(
            lambda _m, args, kw: tensors.__setitem__("lm.embed", kw["inputs_embeds"].detach().float()[0].clone()),
            with_kwargs=True))

        position_ids, rope_delta = model.model.get_rope_index(ids, inp["mm_token_type_ids"], grid)
        res = model(**inp)
        tensors["logits.prompt"] = res.logits[0, -1].float().clone()
        for h in hooks:
            h.remove()

        # generation_config.json ships use_cache: false, which recomputes the
        # whole sequence every step; the cache changes speed, not the answer.
        gen = model.generate(**inp, max_new_tokens=max_new, do_sample=False, use_cache=True)
        new = gen[0, ids.shape[1]:]
        # Teacher-forced logits over the first steps: one forward over prompt + generated.
        s = min(a.tf_steps, new.shape[0])
        full = torch.cat([ids, new[None, : s]], dim=1)
        tf_inp = dict(inp)
        tf_inp["input_ids"] = full
        tf_inp["attention_mask"] = torch.ones_like(full)
        tf_inp["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros(1, s, dtype=inp["mm_token_type_ids"].dtype)], dim=1)
        tf = model(**tf_inp, logits_to_keep=s + 1).logits[0, :-1] if s > 0 else torch.zeros(0)
        tensors["logits.tf"] = tf.float().contiguous()
        tf_argmax_agrees = bool((tf.argmax(-1) == new[:s]).all()) if s > 0 else True

        rec = {
            "name": name,
            "image": path,
            "mode": orig_mode,
            "size": list(orig_size),
            "task": task,
            "prompt": PROMPTS[task],
            "align_corners": a.align_corners,
            "grid_thw": grid[0].tolist(),
            "resized_hw": [int(grid[0, 1]) * 14, int(grid[0, 2]) * 14],
            "input_ids": ids[0].tolist(),
            "position_ids": position_ids[:, 0].tolist(),
            "rope_delta": int(rope_delta[0, 0]),
            "generated_ids": new.tolist(),
            "text": proc.decode(new, skip_special_tokens=True),
            "text_raw": proc.decode(new, skip_special_tokens=False),
            "max_new_tokens": max_new,
            "stopped": bool(new.shape[0] < max_new),
            "tf_steps": s,
            "tf_argmax_agrees": tf_argmax_agrees,
            "pixels_pil_vs_tv_maxabs": float((tensors["pixels"] - tensors["pixels_tv"]).abs().max()),
            "shapes": {k: list(v.shape) for k, v in tensors.items()},
        }
        with open(os.path.join(out, "record.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        save_file(tensors, os.path.join(out, "tensors.safetensors"))
        print(f"{name}: grid {rec['grid_thw']}, {ids.shape[1]} prompt + {new.shape[0]} new tokens, "
              f"tf agrees {tf_argmax_agrees}, pil-vs-tv {rec['pixels_pil_vs_tv_maxabs']:.4f}, "
              f"{time.time() - t0:.1f}s: {rec['text'][:120]!r}", flush=True)

    worst = sorted(ranges.items(), key=lambda kv: -kv[1])
    with open(os.path.join(a.out, "ranges.json"), "w") as f:
        json.dump({"fp16_max": 65504.0, "absmax": dict(worst)}, f, indent=1)
    print("largest activations:")
    for k, v in worst[:12]:
        print(f"  {v:12.1f}  {k}")


if __name__ == "__main__":
    main()
