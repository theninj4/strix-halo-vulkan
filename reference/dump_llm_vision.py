"""Dump qwen3.8-flash-next's vision path for LLM-VISION.md V2 — the fp32
oracle the Go tower, the preprocessing and the prompt's positions are gated
against.

The language model is 180 B parameters and never loaded. Everything the
image path adds before the first decoder layer is small, though:

  * the processor: resize, normalize, patch. `pixel_values` and
    `image_grid_thw` for three deterministic cards: a square RGB one at
    exactly the processor's minimum, a 300x517 RGB one that has to be resized
    (neither side a multiple of 32), and an RGBA one that pins what the
    processor does with alpha. The cards' own uint8 pixels are dumped, so
    V3 can reproduce the processor from the same bytes;
  * the tower (`Qwen4ExpVisionModel`, 27 blocks, merger to 2560, no
    deepstack) built from the HF checkpoint's first shard, which holds all of
    `model.visual.*`. Its taps, and the square card's merged rows again in
    float64, the arbiter qimage's Q8 needed after 27 pre-norm blocks;
  * the text side of a two-image chat prompt: the rendered template, the
    expanded `input_ids`, `get_rope_index`'s 3-row positions and delta (run
    as the model's own method, on a stand-in `self` that carries only the
    config), and the PLE n-gram ids HF would hash for those ids. Those ids come from HF's
    `Qwen4ExpTextNGramEmbedding` built on the meta device with its lookup
    stubbed out, because its table is 320 M x 160.

    .venv/bin/python reference/dump_llm_vision.py   (~6 GB RSS, about a minute)
"""

import json
import os
import types

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor
from transformers.models.qwen4_exp import modeling_qwen4_exp as mq
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

OUT = "reference/out/llmvision"
MODEL = "models/Qwen3.8-Flash-Next-HF-vision"
PROMPT = "Compare these two pictures in one sentence."


def card_square(size):
    """dump_qi21_vision.py's RGBA card: colour ramps, hard edges, a gradient
    alpha. The RGB fixture is this with alpha dropped."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / (size - 1)
    r = np.where((x * 4).astype(int) % 2 == 0, x, 1 - y)
    g = np.sin(6.0 * np.pi * y) * 0.5 + 0.5
    b = np.where(((x * 8).astype(int) + (y * 8).astype(int)) % 2 == 0, 0.15, 0.85)
    a = np.clip(0.25 + 0.75 * x, 0, 1)
    return (np.stack([r, g, b, a], axis=-1) * 255).round().astype(np.uint8)


def card_odd(h, w):
    """Diagonal ramps, a coarse checker and fine stripes at an aspect ratio
    and size the resizer has to change, so the resampler's kernel is
    exercised at high frequency."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    y, x = y / (h - 1), x / (w - 1)
    r = np.clip((x + y) / 2 * 1.5, 0, 1)
    g = np.where(((x * 6).astype(int) + (y * 3).astype(int)) % 2 == 0, 0.9, 0.2)
    b = (np.sin(2 * np.pi * x * w / 7.0) * 0.5 + 0.5) * np.cos(2.0 * np.pi * y) ** 2
    return (np.stack([r, g, b], axis=-1) * 255).round().astype(np.uint8)


class IdsOut(torch.nn.Module):
    """Stands in for the n-gram table: returns the row ids it was asked for."""

    weight = types.SimpleNamespace(device=torch.device("meta"))

    def forward(self, x):
        return x.unsqueeze(-1)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = Qwen4ExpConfig.from_pretrained(MODEL)
    vcfg, tcfg = cfg.vision_config, cfg.text_config
    proc = AutoProcessor.from_pretrained(MODEL)
    manifest = {
        "prompt": PROMPT,
        "image_token_id": cfg.image_token_id,
        "vision_start_token_id": cfg.vision_start_token_id,
        "vision_end_token_id": cfg.vision_end_token_id,
        "processor": {
            "class": type(proc.image_processor).__name__,
            "size": dict(proc.image_processor.size),
            "resample": str(getattr(proc.image_processor, "resample", None)),
        },
        "tensors": {},
    }

    def dump(name, t):
        t = torch.as_tensor(t).detach().contiguous()
        if t.dtype != torch.float64:
            t = t.to(torch.float32)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten().double()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."),
            "count": int(flat.numel()), "sum": float(flat.sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:24s} {str(list(t.shape)):18s} sum={flat.sum():+.5f} absmax={flat.abs().max():.4g}")

    # --- the tower, from the HF shard -----------------------------------
    visual = mq.Qwen4ExpVisionModel._from_config(vcfg, attn_implementation="eager").float().eval()
    state = {}
    with safe_open(os.path.join(MODEL, "model-00001-of-00131.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("model.visual."):
                state[k.removeprefix("model.visual.")] = f.get_tensor(k).float()
    missing, unexpected = visual.load_state_dict(state, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]
    assert not missing and not unexpected, (missing, unexpected)

    # The checkpoint's own precision as a yardstick (LLM-VISION.md V4): the
    # same tower in bf16, which is how HF and vLLM serve it, against fp32 on
    # the same pixels. A device port in fp16 is held to *this*, not to taste.
    manifest["bf16_vs_fp32"] = {}
    v16 = None

    def bf16_gap(name, px, grid, ref):
        nonlocal v16
        if v16 is None:
            v16 = mq.Qwen4ExpVisionModel._from_config(vcfg, attn_implementation="sdpa").to(torch.bfloat16).eval()
            v16.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()}, strict=False)
        with torch.no_grad():
            got = v16(px.pixel_values.to(torch.bfloat16), grid, return_dict=True).pooler_output.double()
        ref = ref.double()
        rms = float((got - ref).pow(2).sum().sqrt() / ref.pow(2).sum().sqrt())
        cos = float(torch.nn.functional.cosine_similarity(got, ref, dim=1).min())
        manifest["bf16_vs_fp32"][name] = {"rms_rel": rms, "worst_cos": cos}
        print(f"  bf16 vs fp32 merged, {name}: rms-relative {rms:.3g}, worst row cosine {cos:.6f}")

    # --- the three cards through the processor --------------------------
    sq = card_square(256)
    odd = card_odd(300, 517)
    cards = {
        "sq": Image.fromarray(sq[..., :3], mode="RGB"),
        "odd": Image.fromarray(odd, mode="RGB"),
        "rgba": Image.fromarray(sq, mode="RGBA"),
        # Under min_pixels, so the resize is an *upscale* on both axes,
        # the branch where the filter is not stretched.
        "tiny": Image.fromarray(card_odd(100, 150), mode="RGB"),
    }
    manifest["cards"] = {}
    for name, img in cards.items():
        raw = np.array(img)
        dump(f"{name}_u8", torch.from_numpy(raw.astype(np.float32)))
        px = proc.image_processor(images=[img], return_tensors="pt")
        grid = px.image_grid_thw
        manifest["cards"][name] = {"hw": list(raw.shape[:2]), "mode": img.mode, "grid_thw": grid[0].tolist()}
        print(f"{name}: {img.mode} {raw.shape[1]}x{raw.shape[0]} -> grid {grid[0].tolist()}")
        dump(f"{name}_pixel_values", px.pixel_values)

        caps = {}
        hooks = [visual.patch_embed.register_forward_hook(lambda m, a, o: caps.__setitem__("patch_embed", o))]
        for i in (0, 1, 13, 26):
            hooks.append(visual.blocks[i].register_forward_hook(
                lambda m, a, o, i=i: caps.__setitem__(f"block{i}", o[0] if isinstance(o, tuple) else o)))
        with torch.no_grad():
            out = visual(px.pixel_values, grid, return_dict=True)
        for h in hooks:
            h.remove()
        dump(f"{name}_patch_embed", caps["patch_embed"])
        for i in (0, 1, 13):
            dump(f"{name}_block{i}_out", caps[f"block{i}"])
        dump(f"{name}_last_hidden", out.last_hidden_state)
        dump(f"{name}_merged", out.pooler_output)
        bf16_gap(name, px, grid, out.pooler_output)

        if name == "sq":
            # The better oracle: the same tower in float64. How far fp32
            # sits from it is the floor a Go port can be held to.
            v64 = mq.Qwen4ExpVisionModel._from_config(vcfg, attn_implementation="eager").double().eval()
            v64.load_state_dict({k: v.double() for k, v in state.items()}, strict=False)
            with torch.no_grad():
                o64 = v64(px.pixel_values.double(), grid, return_dict=True)
            del v64
            dump("sq_last_hidden64", o64.last_hidden_state)
            dump("sq_merged64", o64.pooler_output)
            d = (out.pooler_output.double() - o64.pooler_output).abs()
            rms = float(o64.pooler_output.pow(2).mean().sqrt())
            manifest["sq_fp32_vs_fp64"] = {"max_abs": float(d.max()), "rms": rms}
            print(f"  fp32 vs float64 merged: max abs {float(d.max()):.3g} (rms {rms:.3g})")

    # --- two large cards, tower output only ----------------------------
    #
    # The gates above run grids of a few hundred patches; the device's
    # attention and its arenas change shape at thousands (LLM-VISION.md V4).
    # 1024² is 4 096 patches and 2048² is 16 384. At those sizes eager
    # attention materialises [16, n, n], so these run through SDPA, which is
    # the same arithmetic.
    visual.config._attn_implementation = "sdpa"
    for name, side in (("big", 1024), ("huge", 2048)):
        img = Image.fromarray(card_odd(side, side), mode="RGB")
        px = proc.image_processor(images=[img], return_tensors="pt")
        grid = px.image_grid_thw
        manifest["cards"][name] = {"hw": [side, side], "mode": "RGB", "grid_thw": grid[0].tolist()}
        print(f"{name}: {side}x{side} -> grid {grid[0].tolist()}")
        dump(f"{name}_pixel_values", px.pixel_values)
        with torch.no_grad():
            out = visual(px.pixel_values, grid, return_dict=True)
        dump(f"{name}_merged", out.pooler_output)
        bf16_gap(name, px, grid, out.pooler_output)
    visual.config._attn_implementation = "eager"

    # --- the prompt: two images and a question ---------------------------
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"}, {"type": "text", "text": PROMPT}]}]
    rendered = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[rendered], images=[cards["sq"], cards["odd"]], return_tensors="pt")
    ids = inputs.input_ids
    manifest["rendered"] = rendered
    manifest["ids"] = ids[0].tolist()
    manifest["grid_thw"] = inputs.image_grid_thw.tolist()
    pads = [i for i, v in enumerate(ids[0].tolist()) if v == cfg.image_token_id]
    manifest["image_pad_positions"] = pads
    print(f"prompt: {ids.shape[1]} tokens, grids {inputs.image_grid_thw.tolist()}, {len(pads)} pads")

    # get_rope_index is a method of Qwen4ExpModel that reads only
    # self.config and self.get_vision_position_ids, so it runs on a stand-in
    # rather than on a model with a 180 B language half.
    me = types.SimpleNamespace(config=cfg)
    me.get_vision_position_ids = types.MethodType(mq.Qwen4ExpModel.get_vision_position_ids, me)
    pos, delta = mq.Qwen4ExpModel.get_rope_index(
        me, ids, inputs.mm_token_type_ids, image_grid_thw=inputs.image_grid_thw)
    dump("prompt_position_ids", pos[:, 0].to(torch.float64))
    manifest["mrope_delta"] = int(delta.item())
    print(f"  positions end at {int(pos.max())}, delta {int(delta.item())}")

    # The PLE's n-gram ids, from HF's own module with its table stubbed.
    with torch.device("meta"):
        ng = mq.Qwen4ExpTextNGramEmbedding(tcfg, tcfg.ple_embed_dim, layer_idx=tcfg.ple_layer_ids[0])
    ng.layer_multipliers = mq._build_layer_multipliers(ng.unigram_vocab_size, ng.ngram_size, ng.ple_layer_index, ng.seed)
    ng.ngram_heads_vocab_sizes = torch.tensor(ng.head_vocab_sizes, dtype=torch.long)
    ng.ngram_heads_offsets = torch.tensor(ng.head_offsets, dtype=torch.long)
    ng.ngram_embedding = IdsOut()
    with torch.no_grad():
        ngram_ids = ng(ids, None).reshape(ids.shape[1], -1)
    dump("prompt_ngram_ids", ngram_ids.to(torch.float64))

    # smart_resize over sizes chosen to hit its branches: exact multiples,
    # Python's half-to-even round() on a .5 (48 -> 32, 80 -> 64, 112 -> 128), the
    # min-pixels ceil and the max-pixels floor, and extreme aspect ratios.
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    ip = proc.image_processor
    factor = ip.patch_size * ip.merge_size
    cases = [(256, 256), (300, 517), (48, 48), (80, 112), (16, 16), (1, 199), (4000, 3000),
             (4096, 4096), (5000, 5000), (720, 1280), (1080, 1920), (33, 6000), (6000, 31),
             (144, 400), (511, 513), (8192, 2048), (240, 16384)]
    manifest["smart_resize"] = {
        "factor": factor, "min_pixels": ip.size["shortest_edge"], "max_pixels": ip.size["longest_edge"],
        "cases": [[h, w, *smart_resize(h, w, factor=factor, min_pixels=ip.size["shortest_edge"],
                                        max_pixels=ip.size["longest_edge"])] for h, w in cases],
    }

    # --- decoding: files in, PIL's convert("RGB") out --------------------
    #
    # The lossless formats have to match bit for bit. JPEG is a measurement:
    # Go's decoder is not libjpeg-turbo (LLM-VISION.md V3). The photos are
    # whatever of a short list exists on this machine, copied beside the dump.
    dec = os.path.join(OUT, "decode")
    os.makedirs(dec, exist_ok=True)
    files = []
    rgba_img = cards["rgba"]
    rgba_img.save(os.path.join(dec, "rgba.png"))
    files.append("rgba.png")
    pal = rgba_img.convert("RGB").quantize(64)
    tr = np.array(pal)
    pal.info["transparency"] = int(tr[0, 0])  # one palette entry fully transparent
    pal.save(os.path.join(dec, "palette.png"), transparency=int(tr[0, 0]))
    files.append("palette.png")
    rgba_img.convert("L").save(os.path.join(dec, "gray.png"))
    files.append("gray.png")
    pal.save(os.path.join(dec, "first.gif"))
    files.append("first.gif")
    cards["odd"].save(os.path.join(dec, "card.jpg"), quality=90)
    files.append("card.jpg")
    for src in ["/usr/share/sddm/themes/maldives/background.jpg",
                "/usr/share/doc/ImageMagick-7/www/image/convex-hull-barn.jpg",
                "/usr/share/sddm/themes/elarun/elarun.jpg"]:
        if os.path.exists(src):
            name = "photo_" + os.path.basename(src)
            with open(src, "rb") as fi, open(os.path.join(dec, name), "wb") as fo:
                fo.write(fi.read())
            files.append(name)
    manifest["decode"] = []
    for f in files:
        img = Image.open(os.path.join(dec, f))
        rgb = np.array(img.convert("RGB"))
        rgb.tofile(os.path.join(dec, f + ".rgb"))
        manifest["decode"].append({"file": f, "mode": img.mode, "hw": list(rgb.shape[:2])})
        print(f"  decode {f}: {img.mode} {rgb.shape[1]}x{rgb.shape[0]}")

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
