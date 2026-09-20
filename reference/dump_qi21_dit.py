"""Dump everything around Qwen-Image-2.1's DiT blocks, for Q2/Q3.

Same doctrine as dump_zimage.py: the blocks have their own oracle
(dump_qi21_dit_block.py), so the transformer here is instantiated with
**num_layers=0** — 9 tensors, ~600 MB fp32 instead of 28 GB — and what gets
dumped is the machinery a port gets subtly wrong while every shape checks out:

  * the joint sequence: txt_in over the VLM embeddings, img_in over packed
    latents, each `<|image_pad|>` slot expanded 4x (a 2x2 latent group) and
    the latents scattered into those positions. The VLM's own image-slot
    rows are *overwritten* by the latents — vision content reaches the DiT
    only through what the text tokens absorbed inside the text encoder;
  * the 3-axis RoPE: text advances all axes, image blocks freeze the frame
    axis and use zero-centred h/w grids with *negative* indices served by a
    flipped table, and the cursor advances by max(h, w) after a block;
  * the t=0 modulation row (`causal_condition`): timestep gets a zero
    appended, text/condition tokens read that row, target tokens their own;
  * `kv_cache_mode="cached"` slices rotary, modulation mask and the stream
    down to the target rows — with no blocks, its output must equal the
    prefill output's target rows exactly, and the manifest records the
    measured gap so the identity is checked rather than assumed.

Two cases: `t2i_` (text 7 tokens, target 4x4) and `edit_` (text, a 2x4
condition block, more text, target 4x4 — condition blocks come from VLM
slots so their latent sides must be even).

    .venv/bin/python reference/dump_qi21_dit.py
"""

import json
import os

import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
)
from safetensors import safe_open

OUT = "reference/out/qi21dit"
TRANSFORMER = "models/Qwen-Image-2.1/transformer"
SEED = 7
T = 0.317  # what the pipeline hands the transformer: timestep/1000

NONBLOCK = [
    "img_in.weight",
    "modulation.1.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "txt_in.in_layer.weight",
    "txt_in.out_layer.weight",
    "txt_in.text_norm.weight",
]


def load_nonblock(model):
    with open(os.path.join(TRANSFORMER, "diffusion_pytorch_model.safetensors.index.json")) as fh:
        index = json.load(fh)["weight_map"]
    state = {}
    for name in NONBLOCK:
        with safe_open(os.path.join(TRANSFORMER, index[name]), framework="pt") as fh:
            state[name] = fh.get_tensor(name).to(torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"checkpoint has tensors the model does not want: {unexpected}")
    if missing:
        raise SystemExit(f"model wants tensors the dump does not load: {missing}")


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(os.path.join(TRANSFORMER, "config.json")))
    model_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    model_cfg["num_layers"] = 0
    model = QwenImage21Transformer2DModel(**model_cfg).to(torch.float32).eval()
    load_nonblock(model)

    manifest = {
        "t": T, "seed": SEED,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "cases": {}, "tensors": {},
    }

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:28s} {str(list(t.shape)):18s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}")

    def run_case(label, text_lens, img_shapes):
        """text_lens[i] tokens precede image block i; img_shapes has the target last."""
        torch.manual_seed(SEED)
        dim = model.inner_dim
        target_tokens = img_shapes[-1][1] * img_shapes[-1][2]
        cond_tokens = sum(h * w for _, h, w in img_shapes[:-1])

        # img_mask at VLM granularity: text runs interleaved with slot runs
        # (each slot = 4 latent tokens), target slots appended at the end.
        mask_bits = []
        for i, (_, h, w) in enumerate(img_shapes[:-1]):
            mask_bits += [False] * text_lens[i] + [True] * (h * w // 4)
        mask_bits += [False] * text_lens[len(img_shapes) - 1]
        vlm_len = len(mask_bits)
        mask_bits += [True] * (target_tokens // 4)
        img_mask = torch.tensor(mask_bits)[None]

        encoder_hidden_states = torch.randn(1, vlm_len, cfg["context_in_dim"])
        latents = torch.randn(1, cond_tokens + target_tokens, cfg["in_channels"])
        timestep = torch.tensor([T])

        dump(label + "_encoder_hidden", encoder_hidden_states)
        dump(label + "_latents", latents)
        dump(label + "_img_mask", img_mask.to(torch.float32))

        with torch.no_grad():
            # --- the pieces, via the model's own modules -------------------
            dump(label + "_txt_in", model.txt_in(encoder_hidden_states))
            dump(label + "_img_in", model.img_in(latents))

            repeats = torch.where(img_mask, 4, 1)[0]
            image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
            rotary = model.pos_embed(img_shapes, image_pad_mask, device="cpu")
            dump(label + "_rope_real", torch.view_as_real(rotary)[..., 0])
            dump(label + "_rope_imag", torch.view_as_real(rotary)[..., 1])

            image_ids, target_mask = model.build_token_metadata(image_pad_mask, img_shapes)
            dump(label + "_image_ids", image_ids.to(torch.float32))
            dump(label + "_target_mask", target_mask.to(torch.float32))

            temb = model.time_text_embed(torch.cat([timestep, timestep.new_zeros(1)]), latents)
            dump(label + "_temb", temb)
            dump(label + "_modulation", model.modulation(temb))

            # --- the whole thing with no blocks: assembly + head + tail ----
            out_prefill = model(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=[img_shapes],
                img_mask=img_mask,
                return_dict=False,
            )[0]
            dump(label + "_out_prefill", out_prefill)

            cache = QwenImage21KVCache(0)
            # extract then cached, as the denoising loop does. With no blocks
            # the cache stays empty; what is under test is the slicing.
            model(
                hidden_states=latents, encoder_hidden_states=encoder_hidden_states,
                timestep=timestep, img_shapes=[img_shapes], img_mask=img_mask,
                kv_cache=cache, kv_cache_mode="extract", return_dict=False,
            )
            out_cached = model(
                hidden_states=latents, encoder_hidden_states=encoder_hidden_states,
                timestep=timestep, img_shapes=[img_shapes], img_mask=img_mask,
                kv_cache=cache, kv_cache_mode="cached", return_dict=False,
            )[0]
            dump(label + "_out_cached", out_cached)

        gap = (out_prefill[:, -target_tokens:] - out_cached[:, -target_tokens:]).abs().max()
        manifest["cases"][label] = {
            "text_lens": text_lens, "img_shapes": img_shapes, "vlm_len": vlm_len,
            "cond_tokens": cond_tokens, "target_tokens": target_tokens,
            "prefill_vs_cached_target_gap": float(gap),
        }
        print(f"{label}: prefill-vs-cached target gap {gap:.3g} (0 blocks: must be 0)")

    print("t2i case:")
    run_case("t2i", [7], [(1, 4, 4)])
    print("edit case:")
    run_case("edit", [3, 4], [(1, 2, 4), (1, 4, 4)])

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
