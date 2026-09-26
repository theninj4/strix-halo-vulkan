"""Dump MiniMax-H3's video VAE decode, for VIDEO.md M5.

The oracle the Go decode is gated against: diffusers' `AutoencoderKLMiniMaxH3`
in fp32 on the CPU, run on a *real* sample, M4's final latents (the 256x448 x
124-frame t2va after N = 8, `reference/out/h3dit/f6_latents`). The pipeline
decodes under fp16 autocast on CUDA; the CPU runs it in fp32, which is the
truth the fp16 port is measured against (M-o2).

Dumped:
  * `z`: the whole latent, unpatchified and denormalised (latent * std + mean),
    [24, 37, 16, 28], which is what `vae.decode` takes;
  * one tile-clip through the decoder alone: `tile_in` (post_quant_conv's
    output for latent frames 0..6, rows 0..15, columns 0..15), the residual
    into and out of blocks 0, 1, 18 and 35 ([1797, 2048]: 1792 patches, 4
    registers, the zero token), and `tile_out` [3, 28, 256, 256];
  * every block's absmax on that tile-clip: the residual, q/k/v, the attention
    output, the FFN's two halves and its down projection's input (M-o2);
  * `short_dec`: `vae.decode` of the first 12 latent frames (two temporal
    clips x two tiles, so both the temporal cross-fade and the tile blend
    run), [3, 39, 256, 448], ImageNet-normalised; decoded twice, and the
    manifest records whether the two runs are byte-identical;
  * `full_dec`: `vae.decode` of all 37 latent frames, [3, 124, 256, 448].

    .venv/bin/python reference/dump_h3_vae.py   (~12 GB RSS)
"""

import json
import os
import time

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

MODEL = "models/MiniMax-H3"
OUT = "reference/out/h3vae"
DIT = "reference/out/h3dit"
KEEP_BLOCKS = (0, 1, 18, 35)
SHORT = 12  # latent frames: two temporal clips


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    dmeta = json.load(open(f"{DIT}/manifest.json"))
    lf, lh, lw = dmeta["latent_frames"], dmeta["latent_height"], dmeta["latent_width"]
    rows = np.fromfile(f"{DIT}/f6_latents.bin", dtype=np.float32).reshape(-1, 96)

    vae = AutoencoderKLMiniMaxH3.from_pretrained(f"{MODEL}/vae", torch_dtype=torch.float32).eval()
    cfg = vae.config
    manifest = {"latent_frames": lf, "latent_height": lh, "latent_width": lw, "short_frames": SHORT,
                "keep_blocks": list(KEEP_BLOCKS), "tensors": {}, "block_stats": []}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        while t.dim() > 2 and t.shape[0] == 1:
            t = t[0]
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {"shape": list(t.shape), "count": int(flat.numel()),
                                     "sum": float(flat.double().sum()), "absmax": float(flat.abs().max())}

    def save():
        with open(os.path.join(OUT, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)

    # patchify_video_latents' inverse: rows are (f, h/2, w/2), columns (c, 1, 2, 2).
    z = torch.from_numpy(rows).reshape(1, lf, lh // 2, lw // 2, 24, 1, 2, 2)
    z = z.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 24, lf, lh, lw)
    mean = torch.tensor(cfg.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(cfg.latents_std).view(1, -1, 1, 1, 1)
    z = z * std + mean
    dump("z", z)

    # --- one tile-clip through the decoder, block by block -----------------
    dec = vae.decoder
    tile = vae.post_quant_conv(z[:, :, 0:7, 0:16, 0:16])
    dump("tile_in", tile)
    stats = [dict() for _ in dec.transformer_blocks]
    hooks = []

    def absmax(i, name):
        def put(t):
            stats[i][name] = max(stats[i].get(name, 0.0), float(t.abs().max()))
        return put

    for i, blk in enumerate(dec.transformer_blocks):
        def pre(m, args, i=i):
            absmax(i, "residual_in")(args[0])
            if i in KEEP_BLOCKS:
                dump(f"tile_block{i}_in", args[0])

        def post(m, args, out, i=i):
            absmax(i, "residual_out")(out)
            if i in KEEP_BLOCKS:
                dump(f"tile_block{i}_out", out)

        hooks.append(blk.register_forward_pre_hook(pre))
        hooks.append(blk.register_forward_hook(post))
        for name, mod in (("q", blk.attn.to_q), ("k", blk.attn.to_k), ("v", blk.attn.to_v),
                          ("attn_out", blk.attn.to_out[0]), ("ff_proj", blk.ff.net[0].proj),
                          ("ff_out", blk.ff.net[2])):
            hooks.append(mod.register_forward_hook(lambda m, a, o, f=absmax(i, name): f(o)))
        hooks.append(blk.attn.to_out[0].register_forward_pre_hook(lambda m, a, f=absmax(i, "ctx"): f(a[0])))
        hooks.append(blk.ff.net[2].register_forward_pre_hook(lambda m, a, f=absmax(i, "ff_down_in"): f(a[0])))
        hooks.append(blk.norm1.register_forward_hook(lambda m, a, o, f=absmax(i, "norm1"): f(o)))
    t0 = time.time()
    out = dec(tile)
    manifest["tile_seconds"] = time.time() - t0
    for h in hooks:
        h.remove()
    manifest["block_stats"] = stats
    dump("tile_out", out)
    save()
    print(f"tile-clip: {manifest['tile_seconds']:.1f} s", flush=True)

    # --- the short decode, twice --------------------------------------------
    t0 = time.time()
    a = vae.decode(z[:, :, :SHORT], return_dict=False)[0]
    manifest["short_seconds"] = time.time() - t0
    b = vae.decode(z[:, :, :SHORT], return_dict=False)[0]
    manifest["short_repeat_identical"] = bool(torch.equal(a, b))
    dump("short_dec", a)
    save()
    print(f"short: {manifest['short_seconds']:.1f} s, repeat identical {manifest['short_repeat_identical']}", flush=True)

    # --- the whole clip -------------------------------------------------------
    t0 = time.time()
    full = vae.decode(z, return_dict=False)[0]
    manifest["full_seconds"] = time.time() - t0
    dump("full_dec", full)
    save()
    print(f"full: {manifest['full_seconds']:.1f} s; wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
