"""Dump FLUX 3 Image Klein's autoencoder stagewise, for X4 (research/flux3-vertical.md).

The VAE is byte-identical to FLUX.2-dev's (AutoencoderKLFlux2, fp32). Two
oracles, each a forward with hooks on every top-level stage so the Go port
walks rather than bisects:

  decode — the latent the t2i run oracle actually decoded
           (reference/out/flux3run_t2i/vae_in.bin, already de-normalized and
           unpatchified), through post_quant_conv, conv_in, the mid block,
           each up block, the output norm and conv_out.
  encode — a non-square reference (assets/2_coffee.png, resized to 768x512)
           through conv_in, each down block, the mid block, the output norm,
           conv_out and quant_conv; then the posterior mode, the 2x2 patchify
           and the BatchNorm, i.e. the 128-channel tokens the DiT takes.

    .venv-flux3/bin/python -I reference/dump_flux3_vae.py   (~10 GB RSS)
"""

import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image

OUT = "reference/out/flux3vae"
MODEL = "models/flux-3-image-klein"
RUN = "reference/out/flux3run_t2i"
ENC_W, ENC_H = 768, 512


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"tensors": {}}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:28s} {str(list(t.shape)):22s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}", flush=True)

    vae = AutoencoderKLFlux2.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).eval()
    manifest["config"] = dict(vae.config)
    manifest["bn_eps"] = vae.config.batch_norm_eps
    dump("bn_mean", vae.bn.running_mean)
    dump("bn_var", vae.bn.running_var)

    def taps(prefix, module, names):
        hs = []
        for n in names:
            sub = module.get_submodule(n)
            hs.append(sub.register_forward_hook(lambda m, a, o, n=n: dump(f"{prefix}_{n.replace('.', '_')}", o[0])))
        return hs

    # --- decode ---------------------------------------------------------------
    rman = json.load(open(os.path.join(RUN, "manifest.json")))
    shape = rman["tensors"]["vae_in"]["shape"]
    z = torch.from_numpy(np.fromfile(os.path.join(RUN, "vae_in.bin"), dtype=np.float32).reshape(shape))[None]
    dump("dec_in", z[0])
    dec = vae.decoder
    names = ["conv_in", "mid_block"] + [f"up_blocks.{i}" for i in range(len(dec.up_blocks))] + ["conv_norm_out", "conv_out"]
    hs = taps("dec", dec, names)
    hs.append(vae.post_quant_conv.register_forward_hook(lambda m, a, o: dump("dec_post_quant_conv", o[0])))
    with torch.no_grad():
        img = vae.decode(z, return_dict=False)[0]
    for h in hs:
        h.remove()
    dump("dec_out", img[0])
    # the run's own PNG is this, clamped and quantized: check we decoded the same latent
    png = torch.from_numpy(np.asarray(Image.open(os.path.join(RUN, "image.png"))).astype(np.float32))
    mine = ((img[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0)
    manifest["dec_vs_run_png_max_levels"] = float((mine - png).abs().max())
    print(f"  decode vs the run's PNG: max {manifest['dec_vs_run_png_max_levels']:.3g} levels")

    # --- encode ---------------------------------------------------------------
    ref = Image.open(os.path.join(MODEL, "assets/2_coffee.png")).convert("RGB").resize((ENC_W, ENC_H), Image.Resampling.LANCZOS)
    px = torch.from_numpy(np.asarray(ref).astype(np.float32)).permute(2, 0, 1)[None] / 127.5 - 1.0
    dump("enc_in", px[0])
    enc = vae.encoder
    names = ["conv_in"] + [f"down_blocks.{i}" for i in range(len(enc.down_blocks))] + ["mid_block", "conv_norm_out", "conv_out"]
    hs = taps("enc", enc, names)
    hs.append(vae.quant_conv.register_forward_hook(lambda m, a, o: dump("enc_quant_conv", o[0])))
    with torch.no_grad():
        mode = vae.encode(px).latent_dist.mode()
    for h in hs:
        h.remove()
    dump("enc_mode", mode[0])
    b, c, h2, w2 = mode.shape
    lat = mode.view(b, c, h2 // 2, 2, w2 // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h2 // 2, w2 // 2)
    lat = (lat - vae.bn.running_mean.view(1, -1, 1, 1)) / torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    dump("enc_tokens", lat[0].permute(1, 2, 0).reshape(-1, c * 4))

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
