"""Dump ACE-Step 1.5's VAE decoder (Oobleck, 48 kHz stereo), for MUSIC.md A5.

The inputs are the fp32 DiT oracle's final latents (reference/out/acedit,
dump_ace_dit.py), so A5's gate and A6's end-to-end one share a reference.
It records:

* stagewise, on the first 100 latents (4 s): conv1, each of the five
  decoder blocks, the final snake and conv2, so a port can be gated stage by
  stage;
* the whole decode of each case, untiled (the defined answer, as
  diffusers' `vae.decode` does it);
* the same through upstream's own overlap-discard tiling
  (`VaeDecodeChunksMixin._tiled_decode_inner`) at its CUDA default, chunk
  512 and overlap 64: A-o6, whether tiling changes the audio;
* the audio upstream writes: the peak clamp after decode (divide by the
  peak if it passes 1) and then `normalize_audio` to -1 dBFS.

Tensors are [channels, frames] float32, as diffusers holds them.

    HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_vae.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "models/ACE-Step-1.5-src")

from diffusers import AutoencoderOobleck  # noqa: E402

from acestep.audio_utils import normalize_audio  # noqa: E402
from acestep.core.generation.handler.vae_decode_chunks import VaeDecodeChunksMixin  # noqa: E402

VAE = "models/Ace-Step1.5/vae"
DIT = "reference/out/acedit"
OUT = "reference/out/acevae"
CASES = ["full_metas", "defaults"]
STAGE_LATENTS = 100


class Tiler(VaeDecodeChunksMixin):
    """The least of a handler that upstream's tiling mixin needs."""

    def __init__(self, vae):
        self.vae = vae
        self.disable_tqdm = True

    def _empty_cache(self):
        pass


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
    dec = vae.decoder

    def latents(label):
        lat = np.fromfile(os.path.join(DIT, label + "_latents.bin"), dtype=np.float32).reshape(-1, 64)
        return torch.from_numpy(lat).T.unsqueeze(0).contiguous()  # [1, 64, T]

    # --- stagewise --------------------------------------------------------
    x = latents("full_metas")[:, :, :STAGE_LATENTS]
    dump("stage_in", x[0])
    hooks = [dec.conv1.register_forward_hook(lambda _m, _i, o: dump("stage_conv1", o[0]))]
    for i, blk in enumerate(dec.block):
        hooks.append(blk.register_forward_hook(lambda _m, _i, o, i=i: dump(f"stage_block{i}", o[0])))
        hooks.append(blk.conv_t1.register_forward_hook(lambda _m, _i, o, i=i: dump(f"stage_block{i}_up", o[0])))
    hooks.append(dec.snake1.register_forward_hook(lambda _m, _i, o: dump("stage_snake", o[0])))
    out = vae.decode(x).sample
    for h in hooks:
        h.remove()
    dump("stage_out", out[0])

    # --- whole decodes ----------------------------------------------------
    for label in CASES:
        z = latents(label)
        t0 = time.time()
        wav = vae.decode(z).sample
        untiled_s = time.time() - t0
        dump(label + "_untiled", wav[0])
        t0 = time.time()
        tiled = Tiler(vae)._tiled_decode_inner(z, 512, 64, False)
        tiled_s = time.time() - t0
        dump(label + "_tiled", tiled[0])
        # What upstream writes: the post-decode clamp, then -1 dBFS.
        peak = wav.abs().amax(dim=[1, 2], keepdim=True)
        clamped = wav / peak.clamp(min=1.0) if torch.any(peak > 1.0) else wav
        final = normalize_audio(clamped[0], -1.0)
        dump(label + "_final", final)
        d = (tiled - wav).abs()
        manifest["cases"][label] = {
            "latents": int(z.shape[-1]),
            "samples": int(wav.shape[-1]),
            "peak": float(peak.max()),
            "untiled_seconds": untiled_s,
            "tiled_seconds": tiled_s,
            "tiled_max_abs_diff": float(d.max()),
            "tiled_rms_diff": float(d.pow(2).mean().sqrt()),
            "rms": float(wav.pow(2).mean().sqrt()),
        }
        print(label, manifest["cases"][label])

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
