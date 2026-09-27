"""Dump ACE-Step 1.5's VAE encoder (Oobleck, 48 kHz stereo), for MUSIC.md A11a.

The inputs are the audio A5's oracle wrote (reference/out/acevae,
dump_ace_vae.py): upstream's final, -1 dBFS decodes of the fp32 DiT oracle's
two songs, so the encoder is gated on real music without a file on disk. It
records:

* stagewise, on the first 4 s (100 latents): conv1, each of the five
  encoder blocks, the final snake and conv2 (the posterior's 128 channels:
  mean then scale), so a port can be gated stage by stage;
* the whole encode of each song, untiled (diffusers' `vae.encode`), as the
  posterior's mean and scale;
* the same encode in bf16, upstream's CUDA dtype for the VAE
  (`_get_vae_dtype`): what an fp16 port is priced against (the M11a lesson);
* the mean through upstream's own tiling (`VaeEncodeChunksMixin`, chunk
  30 s, overlap 2 s, its choice above 8 GB) -- whether tiling moves the
  latents, as A-o6 asked of the decoder. Upstream *samples* the posterior
  per chunk, unseeded; the mean is what a tiling can be judged by, so the
  VAE here returns it.

Tensors are [channels, frames] float32, as diffusers holds them.

    HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_vae_enc.py
"""

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "models/ACE-Step-1.5-src")

from diffusers import AutoencoderOobleck  # noqa: E402

from acestep.core.generation.handler.vae_encode_chunks import VaeEncodeChunksMixin  # noqa: E402

VAE = "models/Ace-Step1.5/vae"
SRC = "reference/out/acevae"
OUT = "reference/out/acevaeenc"
CASES = ["full_metas", "defaults", "odd"]
# odd is the 120 s song cut to 100 s and 1,234 frames: not whole latents,
# so the frames past the last latent are read and the tiling has a tail.
ODD = 2500 * 1920 + 1234
STAGE_SAMPLES = 100 * 1920
CHUNK, OVERLAP = 48000 * 30, 48000 * 2


class MeanVAE:
    """The VAE as the tiler sees it, with the posterior's mean for a sample."""

    def __init__(self, vae):
        self.vae = vae
        self.dtype = torch.float32

    def encode(self, x):
        post = self.vae.encode(x).latent_dist
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: post.mean))


class Tiler(VaeEncodeChunksMixin):
    """The least of a handler that upstream's tiling mixin needs."""

    def __init__(self, vae):
        self.vae = MeanVAE(vae)
        self.device = "cpu"
        self.disable_tqdm = True


def main():
    torch.set_grad_enabled(False)
    os.makedirs(OUT, exist_ok=True)
    manifest = {"tensors": {}, "cases": {}}

    def dump(name, t):
        arr = t.detach().contiguous().cpu().float().numpy()
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(arr.tobytes())
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": "float32"}

    vae = AutoencoderOobleck.from_pretrained(VAE, torch_dtype=torch.float32).eval()
    enc = vae.encoder
    vae_bf16 = AutoencoderOobleck.from_pretrained(VAE, torch_dtype=torch.bfloat16).eval()

    def audio(label):
        cut = None
        if label == "odd":
            label, cut = "defaults", ODD
        wav = np.fromfile(os.path.join(SRC, label + "_final.bin"), dtype=np.float32).reshape(2, -1)[:, :cut]
        return torch.from_numpy(wav).clamp(-1, 1).unsqueeze(0).contiguous()  # [1, 2, N]

    # --- stagewise --------------------------------------------------------
    x = audio("full_metas")[:, :, :STAGE_SAMPLES]
    dump("stage_in", x[0])
    hooks = [enc.conv1.register_forward_hook(lambda _m, _i, o: dump("stage_conv1", o[0]))]
    for i, blk in enumerate(enc.block):
        hooks.append(blk.register_forward_hook(lambda _m, _i, o, i=i: dump(f"stage_block{i}", o[0])))
        hooks.append(blk.res_unit3.register_forward_hook(lambda _m, _i, o, i=i: dump(f"stage_block{i}_res", o[0])))
    hooks.append(enc.snake1.register_forward_hook(lambda _m, _i, o: dump("stage_snake", o[0])))
    h = enc(x)
    for hk in hooks:
        hk.remove()
    dump("stage_out", h[0])

    # --- whole encodes ----------------------------------------------------
    for label in CASES:
        a = audio(label)
        t0 = time.time()
        post = vae.encode(a).latent_dist
        untiled_s = time.time() - t0
        dump(label + "_mean", post.mean[0])
        dump(label + "_scale", post.scale[0])
        bf = vae_bf16.encode(a.bfloat16()).latent_dist
        dump(label + "_bf16_mean", bf.mean[0].float())
        dump(label + "_bf16_scale", bf.scale[0].float())
        samples = a.shape[-1]
        stride = CHUNK - 2 * OVERLAP
        steps = -(-samples // stride)
        t0 = time.time()
        if samples <= CHUNK:
            tiled = post.mean
        else:
            tiled = Tiler(vae)._tiled_encode_offload_cpu(a, 1, samples, stride, OVERLAP, steps, CHUNK)
        tiled_s = time.time() - t0
        dump(label + "_tiled_mean", tiled[0])
        d = (tiled - post.mean).abs()
        manifest["cases"][label] = {
            "samples": int(samples),
            "latents": int(post.mean.shape[-1]),
            "untiled_seconds": untiled_s,
            "tiled_seconds": tiled_s,
            "tiled_max_abs_diff": float(d.max()),
            "tiled_rms_diff": float(d.pow(2).mean().sqrt()),
            "mean_rms": float(post.mean.pow(2).mean().sqrt()),
            "std_mean": float((torch.nn.functional.softplus(post.scale) + 1e-4).mean()),
        }
        print(label, manifest["cases"][label])

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
