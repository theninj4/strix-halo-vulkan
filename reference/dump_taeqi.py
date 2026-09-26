"""Dump TAEQI2.1, the tiny decoder that replaces Q7's fitted preview matrix.

madebyollin shipped `taeqi2_1_{encoder,decoder}.pth` in taesd commit
401ce45e (2026-09-25, taesd issue #38): a 64-channel/16x RGBA tiny AE for
Qwen-Image-2.1's VAE, answering Q-o4. `reference/taesd/taesd.py` is that
commit's file verbatim (MIT, LICENSE beside it) and is the oracle.

This script does three things:

  * converts the decoder's .pth (a pickle Go cannot read) to
    `models/taeqi2_1/taeqi2_1_decoder.safetensors`, keys unchanged, dtype
    unchanged (the checkpoint is fp16);
  * decodes the *final* latent of reference/out/qi21run1024 (a real 40-step
    1024² trajectory, packed [1, 4096, 64] in the DiT's normalized space)
    and records it against that run's full-VAE image — the quality number a
    preview is priced at;
  * decodes a 24x40 crop of the same latent with every top-level layer of
    the Sequential hooked, for the Go port's stagewise gate; a rectangle
    because that is where a transposed H/W hides.

The decoder takes the **normalized** latent — the space the DiT works in,
the one the fitted matrix took too. Measured with the real VAE encoder on a
served picture: normalized lands rms 0.035 from the full VAE's decode
(the full VAE's own round trip is 0.030), raw lands 0.198.

    .venv/bin/python reference/dump_taeqi.py
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "taesd"))
from taesd import TAESD  # noqa: E402

MODEL = "models/taeqi2_1"
RUN = "reference/out/qi21run1024"
OUT = "reference/out/taeqi"


def main():
    torch.set_grad_enabled(False)
    os.makedirs(OUT, exist_ok=True)
    enc_pth = os.path.join(MODEL, "taeqi2_1_encoder.pth")
    dec_pth = os.path.join(MODEL, "taeqi2_1_decoder.pth")
    tae = TAESD(enc_pth, dec_pth).eval()
    assert (tae.latent_channels, tae.image_channels) == (64, 4)

    sd = torch.load(dec_pth, map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(MODEL, "taeqi2_1_decoder.safetensors"))
    # The oracle runs in fp32 on the fp16 weights, which is what the Go port
    # does too: widen once at load.
    dec = tae.decoder.float()

    manifest = {"source": "madebyollin/taesd@401ce45e", "run": RUN, "tensors": {}}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:22s} {str(list(t.shape)):22s} sum={flat.double().sum():+.4f} absmax={flat.abs().max():.4g}")

    run = json.load(open(os.path.join(RUN, "manifest.json")))
    last = max(int(k[4:-8]) for k in run["tensors"] if k.startswith("step") and k.endswith("_latents"))
    packed = torch.from_numpy(np.fromfile(os.path.join(RUN, f"step{last}_latents.bin"), dtype=np.float32))
    side = run["size"] // 16
    z = packed.view(side, side, 64).permute(2, 0, 1)[None].contiguous()  # [1, 64, h, w], normalized
    manifest["latent_step"] = last

    # Full size, against the run's own full-VAE image.
    dump("latent", z)
    img = dec(z).clamp(0, 1) * 2 - 1  # the taesd [0, 1] convention, moved to the pipeline's [-1, 1]
    dump("image", img)
    # The run dumps diffusers' "np" output: [H, W, 4] in [0, 1].
    full = torch.from_numpy(np.fromfile(os.path.join(RUN, "image.bin"), dtype=np.float32))
    full = full.view(img.shape[2], img.shape[3], 4).permute(2, 0, 1)[None] * 2 - 1
    d = img - full
    manifest["vs_full_vae"] = {
        "rms": float(d.pow(2).mean().sqrt()), "max": float(d.abs().max()),
        "rms_per_channel": [float(d[:, c].pow(2).mean().sqrt()) for c in range(4)],
    }
    print("  vs full VAE:", manifest["vs_full_vae"])

    # A rectangle, every layer tapped.
    crop = z[:, :, 8:32, 12:52].contiguous()
    dump("crop_latent", crop)
    x = crop
    for i, layer in enumerate(dec):
        x = layer(x)
        dump(f"crop_layer{i:02d}", x)
    dump("crop_image", x.clamp(0, 1) * 2 - 1)

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
