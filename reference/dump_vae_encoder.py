"""Dump a reference VAE *encode*, with intermediates, for IMAGE.md I7.

The mirror of dump_vae.py and the same contract: run a fixed input through
diffusers in float32 on the CPU, dump every submodule's output as raw
little-endian float32 plus a JSON manifest, so the Go port can be walked stage
by stage instead of only at the end.

The input is a **real image** rather than a random tensor, which is
PIPELINE.md's fifth rule. It matters more here than it did for the decoder:
the encoder's first convolution sees pixels, and a uniform-random image has
neither the flat regions nor the edges that the downsamplers' asymmetric
padding is visible on.

    .venv/bin/python reference/dump_vae_encoder.py --size 256

Also dumped, because the Go side needs them and they are one line each here:
the posterior's mode (the mean, which is what Encoder.Encode returns), the
same thing scaled into the transformer's latent space, and a decode of that
back to pixels so a round trip has a picture to be looked at.
"""

import argparse
import json
import os

import torch
from PIL import Image
from diffusers import AutoencoderKL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="models/Z-Image-Turbo/vae")
    ap.add_argument("--image", default="zimage.png.0.png")
    ap.add_argument("--out", default="reference/out/vaeenc")
    ap.add_argument("--size", type=int, default=256,
                    help="the image is resized to this square; the latent comes out size/8")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32)
    vae.eval()
    enc = vae.encoder

    img = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    img.save(os.path.join(args.out, "input.png"))
    x = (torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float() / 127.5 - 1.0)
    x = x.reshape(args.size, args.size, 3).permute(2, 0, 1).unsqueeze(0).contiguous()

    manifest = {
        "size": args.size,
        "image": args.image,
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
        with open(os.path.join(args.out, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape),
            "count": int(flat.numel()),
            "sum": float(flat.double().sum()),
            "absmax": float(flat.abs().max()),
        }
        print(f"  {name:34s} {str(list(t.shape)):24s} sum={flat.double().sum():+.6f} absmax={flat.abs().max():.5g}")

    watch = {"conv_in": enc.conv_in}
    for i, down in enumerate(enc.down_blocks):
        for j, r in enumerate(down.resnets):
            watch[f"down.{i}.resnets.{j}"] = r
        if down.downsamplers:
            watch[f"down.{i}.downsample"] = down.downsamplers[0]
        watch[f"down.{i}"] = down
    watch.update({
        "mid.resnets.0": enc.mid_block.resnets[0],
        "mid.attn": enc.mid_block.attentions[0],
        "mid.resnets.1": enc.mid_block.resnets[1],
        "mid": enc.mid_block,
        "conv_norm_out": enc.conv_norm_out,
        "conv_out": enc.conv_out,
    })

    captured = {}

    def hook(name):
        def fn(_mod, _inp, out):
            captured[name] = out[0] if isinstance(out, tuple) else out
        return fn

    handles = [m.register_forward_hook(hook(k)) for k, m in watch.items()]
    print("encoding...")
    moments = enc(x)
    for h in handles:
        h.remove()

    print("tensors:")
    dump("image_in", x)
    order = ["conv_in"]
    for i, down in enumerate(enc.down_blocks):
        for j in range(len(down.resnets)):
            order.append(f"down.{i}.resnets.{j}")
        if down.downsamplers:
            order.append(f"down.{i}.downsample")
        order.append(f"down.{i}")
    order += ["mid.resnets.0", "mid.attn", "mid.resnets.1", "mid", "conv_norm_out", "conv_out"]
    for k in order:
        if k in captured:
            dump(k, captured[k])
    dump("moments", moments)

    # The deterministic branch of the posterior: mode() is the mean, and is
    # what Encoder.Encode returns. Sampling would put a different latent
    # behind every edit of the same picture.
    mode = vae.encode(x).latent_dist.mode()
    dump("mode", mode)
    latent = (mode - vae.config.shift_factor) * vae.config.scaling_factor
    dump("latent", latent)

    # A round trip, so that "the encoder is right" has a picture as well as a
    # number behind it.
    back = vae.decode(mode, return_dict=False)[0]
    dump("roundtrip", back)
    px = ((back[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
    Image.fromarray(px).save(os.path.join(args.out, "roundtrip.png"))

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
