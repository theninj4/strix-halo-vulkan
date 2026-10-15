"""Dump taef2's decoder, the FLUX 3 preview decoder, for X6
(research/flux3-vertical.md).

taesd.py's TAESD(arch_variant="flux_2", latent_channels=32) in fp32 on the
CPU, loaded as the model card's DiffusersTAEF2Wrapper loads it (the
checkpoint's diffusers-style "decoder.layers.N" are taesd's N+1: its
Sequential opens with the Clamp). Input: the latent the t2i run oracle
decoded (reference/out/flux3run_t2i/vae_in.bin) **re-normalised** by the
VAE's BatchNorm and un-patchified again, [32, 128, 128]: taef2 takes the
DiT's normalised space, not the raw latent. The wrapper on the model card
says so by construction -- its BatchNorm has mean 0 and variance 1, so the
pipeline's de-normalisation is the identity in front of it -- and it
measures so: against the full VAE's PNG of this latent the normalised input
decodes at 32.8 dB with the picture's own mean and contrast (std 41.1 vs
40.6), the raw one at 17.9 dB with 1.7x the contrast (X6). Output: the
decoder's [0, 1] image as the wrapper's x*2-1 (unclamped), plus the tensor
after each top-level layer.

    .venv-flux3/bin/python -I reference/dump_taef2.py
"""

import json
import os
import sys

import numpy as np
import safetensors.torch as stt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "taesd"))
from taesd import TAESD  # noqa: E402

OUT = "reference/out/taef2"
RUN = "reference/out/flux3run_t2i"
CKPT = "models/taef2/taef2.safetensors"
MODEL = "models/flux-3-image-klein"


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"tensors": {}}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        manifest["tensors"][name] = {"shape": list(t.shape), "count": t.numel(),
                                     "absmax": float(t.abs().max())}

    m = TAESD(encoder_path=None, decoder_path=None, latent_channels=32, arch_variant="flux_2")
    sd = {}
    for k, v in stt.load_file(CKPT).items():
        encdec, _layers, index, *suffix = k.split(".")
        offset = 1 if encdec == "decoder" else 0
        sd[".".join([encdec, str(int(index) + offset), *suffix])] = v
    m.load_state_dict(sd)
    m = m.float().eval()

    with open(os.path.join(RUN, "manifest.json")) as fh:
        shape = json.load(fh)["tensors"]["vae_in"]["shape"]
    raw = torch.from_numpy(np.fromfile(os.path.join(RUN, "vae_in.bin"), dtype=np.float32).reshape(shape))[None]
    vae = stt.load_file(os.path.join(MODEL, "vae", "diffusion_pytorch_model.safetensors"))
    with open(os.path.join(MODEL, "vae", "config.json")) as fh:
        eps = json.load(fh)["batch_norm_eps"]
    c, h, w = shape
    # patchify (channel c*4 + 2*dy + dx), normalise, un-patchify
    p = raw.reshape(1, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(1, 4 * c, h // 2, w // 2)
    p = (p - vae["bn.running_mean"][None, :, None, None]) / torch.sqrt(vae["bn.running_var"][None, :, None, None] + eps)
    z = p.reshape(1, c, 2, 2, h // 2, w // 2).permute(0, 1, 4, 2, 5, 3).reshape(1, c, h, w)
    dump("latent", z[0])

    h = z
    with torch.no_grad():
        for i, layer in enumerate(m.decoder):
            h = layer(h)
            # taesd index i is the checkpoint's layers.(i-1); 0 is the clamp.
            dump("clamp" if i == 0 else f"layer{i - 1:02d}", h[0])
        dump("image", (h * 2 - 1)[0])
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    main()
