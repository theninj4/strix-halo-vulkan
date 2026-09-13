"""Dump a reference VAE decode, with intermediates, for the Go/Vulkan port to
check itself against.

The rule in PIPELINE.md is that every stage is validated against diffusers
rather than against a shape table. This writes the input latent, the final
image, and the output of every interesting submodule as raw little-endian
float32, plus a JSON manifest of shapes, so the Go side can compare piece by
piece instead of only at the end -- which is the difference between "the
image is wrong" and "up_blocks.1.resnets.0 is wrong".

    .venv/bin/python reference/dump_vae.py --latent-size 16

Everything runs in float32 on CPU: the VAE config sets force_upcast, and a
reference that is itself approximate is not a reference.
"""

import argparse
import json
import os
import struct

import torch
from diffusers import AutoencoderKL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="models/Z-Image-Turbo/vae")
    ap.add_argument("--out", default="reference/out/vae")
    ap.add_argument("--latent-size", type=int, default=16,
                    help="latent H=W; the image comes out 8x this")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32)
    vae.eval()
    dec = vae.decoder

    n = args.latent_size
    latent = torch.randn(1, vae.config.latent_channels, n, n, dtype=torch.float32)

    manifest = {
        "seed": args.seed,
        "latent_size": n,
        "config": {
            k: vae.config[k]
            for k in ("latent_channels", "block_out_channels", "layers_per_block",
                      "norm_num_groups", "scaling_factor", "shift_factor",
                      "act_fn", "mid_block_add_attention")
        },
        "tensors": {},
    }

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        path = os.path.join(args.out, name + ".bin")
        with open(path, "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape),
            "count": int(flat.numel()),
            # A checksum the Go side can compare before it bothers with a
            # full elementwise diff.
            "sum": float(flat.double().sum()),
            "absmax": float(flat.abs().max()),
        }
        print(f"  {name:46s} {str(list(t.shape)):24s} sum={flat.double().sum():+.6f}")

    # Capture the output of every submodule worth checking on its own.
    watch = {
        "conv_in": dec.conv_in,
        "mid.resnets.0": dec.mid_block.resnets[0],
        "mid.attn": dec.mid_block.attentions[0],
        "mid.resnets.1": dec.mid_block.resnets[1],
        "mid": dec.mid_block,
        "conv_norm_out": dec.conv_norm_out,
        "conv_out": dec.conv_out,
    }
    for i, up in enumerate(dec.up_blocks):
        for j, r in enumerate(up.resnets):
            watch[f"up.{i}.resnets.{j}"] = r
        if up.upsamplers:
            watch[f"up.{i}.upsample"] = up.upsamplers[0]
        watch[f"up.{i}"] = up

    captured = {}

    def hook(name):
        def fn(_mod, _inp, out):
            captured[name] = out[0] if isinstance(out, tuple) else out
        return fn

    handles = [m.register_forward_hook(hook(k)) for k, m in watch.items()]

    print("decoding...")
    with torch.no_grad():
        image = dec(latent)
    for h in handles:
        h.remove()

    print("tensors:")
    dump("latent", latent)
    # Emit in execution order so a failing stage is the first mismatch.
    order = ["conv_in", "mid.resnets.0", "mid.attn", "mid.resnets.1", "mid"]
    for i in range(len(dec.up_blocks)):
        for j in range(len(dec.up_blocks[i].resnets)):
            order.append(f"up.{i}.resnets.{j}")
        if dec.up_blocks[i].upsamplers:
            order.append(f"up.{i}.upsample")
        order.append(f"up.{i}")
    order += ["conv_norm_out", "conv_out"]
    for k in order:
        if k in captured:
            dump(k, captured[k])
    dump("image", image)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
