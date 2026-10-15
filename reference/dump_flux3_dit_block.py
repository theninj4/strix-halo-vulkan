"""Dump FLUX 3 Image Klein's DiT around single blocks, for X2 (research/flux3-vertical.md).

One fp32 forward of diffusers' Flux3Transformer2DModel on the CPU at a 512²
target with one 512² reference image, so all three modes run: the text is
the `short` prompt from dump_flux3_textenc.py (80 rows, pad rows included),
the reference is the checkpoint's own assets/1_figurine.png through the VAE
(posterior mode, patchify, BatchNorm — diffusers' Flux3VaeEncoderStep), and
the target is seeded noise at a mid-trajectory time.

Forward hooks record every input and the output of: the first text, image
and image_cond mode blocks, the first and last single blocks, and the output
norm; plus the model's own rotary tables and the final velocity. That is
enough for the Go port to gate one block of each kind against its exact
inputs, then the whole stack against the output.

A second, reference-free forward (t2i) dumps only its output, so the stack
gate covers the two-mode joint sequence as well.

    .venv-flux3/bin/python -I reference/dump_flux3_dit_block.py   (~26 GB RSS)
"""

import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel
from PIL import Image

OUT = "reference/out/flux3block"
MODEL = "models/flux-3-image-klein"
TEXT = "reference/out/flux3textenc"
SIDE = 512  # pixels, target and reference
T, T_NEXT = 0.75, 0.5  # current and jump-target flow times (a mid-trajectory step)
SCALE = 0.5  # image_rope_scale


def image_ids(h, w, l):
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ids = torch.stack([torch.zeros_like(rows), rows, cols, torch.full_like(rows, l)], dim=-1).reshape(h * w, 4).float()
    ids[:, 1:3] *= SCALE
    return ids


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"side": SIDE, "t": T, "t_next": T_NEXT, "image_rope_scale": SCALE, "tensors": {}}

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

    # --- text: the textenc oracle's short prompt, pad rows and all ------------
    tmani = json.load(open(os.path.join(TEXT, "manifest.json")))
    shape = tmani["tensors"]["short_prompt_embeds"]["shape"]
    txt = torch.from_numpy(np.fromfile(os.path.join(TEXT, "short_prompt_embeds.bin"), dtype=np.float32).reshape(shape))[None]
    L = txt.shape[1]
    txt_ids = torch.stack([torch.zeros(L), torch.zeros(L), torch.zeros(L), torch.arange(L).float()], dim=-1)

    # --- reference: VAE encode exactly as Flux3VaeEncoderStep -----------------
    vae = AutoencoderKLFlux2.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).eval()
    img = Image.open(os.path.join(MODEL, "assets/1_figurine.png")).convert("RGB").resize((SIDE, SIDE), Image.Resampling.LANCZOS)
    px = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()[None] / 127.5 - 1.0
    dump("ref_pixels", px[0])
    with torch.no_grad():
        lat = vae.encode(px).latent_dist.mode()
    b, c, h2, w2 = lat.shape
    lat = lat.view(b, c, h2 // 2, 2, w2 // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h2 // 2, w2 // 2)
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    lat = (lat - mean) / std
    gh, gw = lat.shape[2:]
    cond = lat[0].permute(1, 2, 0).reshape(1, gh * gw, c * 4)
    cond_ids = image_ids(gh, gw, -1)
    dump("cond_tokens", cond[0])
    del vae

    # --- target noise ---------------------------------------------------------
    g = torch.Generator().manual_seed(0)
    x = torch.randn((1, gh * gw, 128), generator=g).to(torch.bfloat16).float()
    img_ids = image_ids(gh, gw, 0)
    dump("noise", x[0])
    dump("img_ids", img_ids)
    dump("cond_ids", cond_ids)
    dump("txt_ids", txt_ids)

    model = Flux3Transformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.float32).eval()
    manifest["config"] = dict(model.config)

    hooks = []

    def tap(name, module):
        def hook(mod, args, kwargs, out):
            kind = next(k for k in ("mode", "single", "norm_out") if name.startswith(k))
            names = {
                "mode": ["hidden_states", "temb_mod", "image_rotary_emb"],
                "single": ["hidden_states", "encoder_hidden_states", "temb_mod", "temb_mod_context", "image_rotary_emb"],
                "norm_out": ["hidden_states", "temb"],
            }[kind]
            bound = dict(zip(names, args))
            bound.update(kwargs)
            for k, v in bound.items():
                if isinstance(v, tuple):  # rotary (cos, sin): [S, 128], shared by every block of a kind
                    if name != "single27":
                        dump(f"{name}_in_{k}_cos", v[0])
                        dump(f"{name}_in_{k}_sin", v[1])
                else:
                    dump(f"{name}_in_{k}", v[0])
            if isinstance(out, tuple):
                dump(f"{name}_out_encoder_hidden_states", out[0][0])
                dump(f"{name}_out_hidden_states", out[1][0])
            else:
                dump(f"{name}_out", out[0])
        hooks.append(module.register_forward_hook(hook, with_kwargs=True))

    tap("mode_txt0", model.txt_mode_blocks[0])
    tap("mode_image0", model.mode_blocks["image"][0])
    tap("mode_cond0", model.mode_blocks["image_cond"][0])
    tap("single0", model.single_transformer_blocks[0])
    tap("single27", model.single_transformer_blocks[-1])
    tap("norm_out", model.norm_out)

    print("edit forward (text + image + image_cond)")
    with torch.no_grad():
        v = model(
            hidden_states=x, encoder_hidden_states=txt, timestep=torch.tensor([T]),
            img_ids=img_ids, txt_ids=txt_ids,
            image_cond_hidden_states=cond, image_cond_ids=cond_ids,
            target_timestep=torch.tensor([T_NEXT]), return_dict=False,
        )[0]
    dump("edit_velocity", v[0])
    for hk in hooks:
        hk.remove()

    print("t2i forward (text + image)")
    with torch.no_grad():
        v = model(
            hidden_states=x, encoder_hidden_states=txt, timestep=torch.tensor([T]),
            img_ids=img_ids, txt_ids=txt_ids, target_timestep=torch.tensor([T_NEXT]), return_dict=False,
        )[0]
    dump("t2i_velocity", v[0])

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
