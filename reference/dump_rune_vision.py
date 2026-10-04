#!/usr/bin/env python3
"""Rune's image input, the reference for research/rune-vertical.md stage R10 (images).

transformers 5.17.0's Gemma4ForConditionalGeneration over models/rune-26b-a4b, as dump_rune.py
runs the text tower (weights stored bf16, each decoder layer upcast to fp32 for its forward),
plus the vision tower and its projector (`embed_vision`) in fp32 throughout (0.5B parameters).

The cases are gemma4/testdata/vision_cases.json: decisions questions with images attached to the
user turn ahead of the text, as surogate's decisions v1 renders them (`images` field). Every
image is written out as PNG first and read back, so the Go side decodes the same pixels (JPEG
decoders differ; PNG does not). For each case, under reference/out/rune/vision/<name>/:

  meta.json         the prompt text, the ids (the processor's: <|image> + n x <|image|> + <image|>
                    per image), the label ids, the label logits after the softcap, p at T=1 and
                    T=2, the images' files and soft-token counts
  rows.safetensors  img{k}.pixels [N, 768] (the processor's patches, unpadded), img{k}.pos [N, 2],
                    img{k}.embed [N, 1152] (patch embedder output), img{k}.layer{i} [N, 1152]
                    (vision encoder layer i), img{k}.pooled [n, 1152] (after pool, sqrt(1152)
                    scale and standardize), img{k}.features [n, 2816] (embed_vision's output: what
                    replaces the image tokens), and text embed / layer{i} [L, 2816] / final

    .venv/bin/python reference/dump_rune_vision.py [--only name,name] [--bf16]

--bf16 runs everything in bf16, as Rune is served upstream: the precision baseline Q8 is judged
against (R7's rule).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration, create_masks_for_vision_model, get_block_sequence_ids_for_mask)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_image(spec, out_dir):
    """An image as PNG in out_dir, from a file or a synthetic pattern; returns its path."""
    path = os.path.join(out_dir, spec["name"] + ".png")
    if "file" in spec:
        im = Image.open(os.path.join(ROOT, spec["file"]))
        im.load()
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
    else:
        w, h = spec["size"]
        y, x = np.mgrid[0:h, 0:w]
        r = (x * 255 // max(w - 1, 1)).astype(np.uint8)
        g = (y * 255 // max(h - 1, 1)).astype(np.uint8)
        b = (((x // 7 + y // 5) % 2) * 200 + 30).astype(np.uint8)
        im = Image.fromarray(np.stack([r, g, b], -1), "RGB")
    im.save(path)
    return path


def render_user(case, base):
    if "base" in case:
        return base[case["base"]]["system"], base[case["base"]]["user"]
    q = case["question"]
    state = "SHARED STATE (JSON string):\n" + json.dumps(case["state"], ensure_ascii=False) + "\n\n"
    lines = "\n".join(f"{chr(65 + i)}: {t}" for i, t in enumerate(q["options"]))
    user = state + "QUESTION:\n" + q["instructions"] + "\nOPTIONS:\n" + lines + "\nAnswer with one option letter only."
    return base["ticket/sentiment"]["system"], user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/rune-26b-a4b"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "gemma4/testdata/vision_cases.json"))
    ap.add_argument("--prompts", default=os.path.join(ROOT, "gemma4/testdata/prompts.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reference/out/rune/vision"))
    ap.add_argument("--only", default="")
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count())
    out_root = a.out + ("-bf16" if a.bf16 else "")
    os.makedirs(out_root, exist_ok=True)

    cases = json.load(open(a.cases))
    base = {p["name"]: p for p in json.load(open(a.prompts))["prompts"]}
    img_dir = os.path.join(out_root, "images")
    os.makedirs(img_dir, exist_ok=True)
    images = {s["name"]: make_image(s, img_dir) for s in cases["images"]}

    proc = AutoProcessor.from_pretrained(a.model)
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True)
    model.eval()
    mm = model.model
    lm = mm.language_model
    cfg = lm.config
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    if not a.bf16:
        mm.vision_tower.float()
        mm.embed_vision.float()
        lm.norm.float()
        for layer in lm.layers:
            layer.register_forward_pre_hook(lambda m, args, kwargs: (m.float(), None)[1], with_kwargs=True)
            layer.register_forward_hook(lambda m, args, out: (m.bfloat16(), None)[1])
    dt = torch.bfloat16 if a.bf16 else torch.float32
    tower = mm.vision_tower

    only = [n for n in a.only.split(",") if n]
    for case in cases["cases"]:
        name = case["name"]
        if only and name not in only:
            continue
        t0 = time.time()
        system, user = render_user(case, base)
        content = [{"type": "image"} for _ in case["images"]] + [{"type": "text", "text": user}]
        msgs = [{"role": "system", "content": system},  # a string, as decisions renders it (a list adds a space)
                {"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pics = [Image.open(images[n]) for n in case["images"]]
        inputs = proc(text=[text], images=[pics], return_tensors="pt", add_special_tokens=False)
        ids = inputs["input_ids"]
        labels = [chr(65 + i) for i in range(len(case["question"]["options"]))] if "question" in case \
            else base[case["base"]]["labels"]
        label_ids = [proc.tokenizer.encode(text + l, add_special_tokens=False)[-1] for l in labels]

        rows = {}
        pv, pos = inputs["pixel_values"].to(dt), inputs["image_position_ids"]
        feats = []
        for k in range(pv.shape[0]):
            valid = (pos[k] != -1).all(-1)
            n = int(valid.sum())
            rows[f"img{k}.pixels"] = inputs["pixel_values"][k, :n].float().clone()
            rows[f"img{k}.pos"] = pos[k, :n].to(torch.int64).clone()
            hooks = []

            def keep(key):
                def hook(m, args, out):
                    o = out[0] if isinstance(out, tuple) else out
                    rows[key] = o.detach()[0, :n].float().clone()
                return hook
            hooks.append(tower.patch_embedder.register_forward_hook(keep(f"img{k}.embed")))
            for i, layer in enumerate(tower.encoder.layers):
                hooks.append(layer.register_forward_hook(keep(f"img{k}.layer{i}")))
            vo = tower(pixel_values=pv[k:k + 1], pixel_position_ids=pos[k:k + 1])
            for h in hooks:
                h.remove()
            rows[f"img{k}.pooled"] = vo.last_hidden_state.float().clone()
            f = mm.embed_vision(inputs_embeds=vo.last_hidden_state)
            rows[f"img{k}.features"] = f.float().clone()
            feats.append(f)

        image_mask = ids[0] == model.config.image_token_id
        safe = torch.where(ids == model.config.image_token_id, cfg.pad_token_id, ids)
        embeds = lm.embed_tokens(safe).to(dt)  # HF's bf16 lookup and bf16 scale
        embeds[0, image_mask] = torch.cat(feats).to(dt)
        rows["embed"] = embeds[0].float().clone()
        L = ids.shape[1]
        position_ids = torch.arange(L).unsqueeze(0)
        block = get_block_sequence_ids_for_mask(inputs["mm_token_type_ids"], device=embeds.device)
        masks = create_masks_for_vision_model(config=cfg, inputs_embeds=embeds, attention_mask=None,
                                              past_key_values=None, position_ids=position_ids,
                                              block_sequence_ids=block)
        hooks = []
        for i, layer in enumerate(lm.layers):
            def keep_layer(m, args, out, i=i):
                o = out[0] if isinstance(out, tuple) else out
                rows[f"layer{i}"] = o.detach()[0].float().clone()
            hooks.append(layer.register_forward_hook(keep_layer))
        try:
            out = lm(inputs_embeds=embeds, attention_mask=masks, position_ids=position_ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        final = out.last_hidden_state[0, -1].float()
        rows["final"] = out.last_hidden_state[0].float().clone()
        w = lm.embed_tokens.weight
        cap = cfg.final_logit_softcapping
        lab = torch.tanh((w[label_ids].float() @ final) / cap) * cap
        meta = {"name": name, "text": text, "system": system, "user": user, "ids": ids[0].tolist(),
                "labels": labels, "label_ids": label_ids, "label_logits": lab.tolist(),
                "images": [os.path.relpath(images[n], ROOT) for n in case["images"]],
                "soft_tokens": [int(f.shape[0]) for f in feats],
                "mode": "bf16" if a.bf16 else "fp32"}
        for t in (1.0, 2.0):
            z = lab.double()
            pz = torch.exp((z - z.max()) / t)
            meta[f"probs_T{t:g}"] = (pz / pz.sum()).tolist()
        d = os.path.join(out_root, name)
        os.makedirs(d, exist_ok=True)
        save_file({k: v.contiguous() for k, v in rows.items()}, os.path.join(d, "rows.safetensors"))
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1, ensure_ascii=False)
        print(f"{name}: {L} tokens ({image_mask.sum().item()} image) in {time.time() - t0:.0f} s; "
              f"p(T=1) {[round(x, 4) for x in meta['probs_T1']]}", flush=True)


if __name__ == "__main__":
    main()
