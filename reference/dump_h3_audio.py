"""Dump MiniMax-H3's audio VAE decode, for VIDEO.md M6.

diffusers' `AutoencoderKLMiniMaxH3Audio` in fp32 on the CPU, run on M4's
final audio latents (`reference/out/h3dit/f6_audio`, 2 x 207 rows: the
5.2 s soundtrack of the 8-step sample). The rows are unpacked as
`MiniMaxH3AfterDenoiseStep` does, (2, 207, 32) -> (2, 32, 207), and
denormalised as `MiniMaxH3AudioDecodeStep` does; the two stereo channels are
one batch of two through the mono codec.

Dumped, each [2, C, T]:
  * `z`: the denormalised latents;
  * `pre`: dec_in_proj then conv_pre, the BigVGAN trunk's input;
  * `stage{i}`: after upsampler i and its three averaged AMP blocks (i = 0..6);
  * `act0_in` / `act0_out`: stage 0's first alias-free SnakeBeta alone, so the
    resampling can be gated apart from the convolutions;
  * `wave`: the clamped waveform, [2, 1, samples].

    .venv/bin/python reference/dump_h3_audio.py   (~2 GB RSS, seconds)
"""

import json
import os
import time

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

MODEL = "models/MiniMax-H3"
OUT = "reference/out/h3audio"
DIT = "reference/out/h3dit"


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    dmeta = json.load(open(f"{DIT}/manifest.json"))
    n = dmeta["audio_latents"]
    rows = torch.from_numpy(np.fromfile(f"{DIT}/f6_audio.bin", dtype=np.float32).reshape(2 * n, 32))

    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(f"{MODEL}/audio_vae", torch_dtype=torch.float32).eval()
    cfg = vae.config
    manifest = {"audio_latents": n, "sampling_rate": cfg.sampling_rate, "tensors": {}}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {"shape": list(t.shape), "count": int(flat.numel()),
                                     "sum": float(flat.double().sum()), "absmax": float(flat.abs().max())}

    z = rows.reshape(2, n, 32).permute(0, 2, 1).contiguous()
    mean = torch.tensor(cfg.latents_mean).view(1, -1, 1)
    std = torch.tensor(cfg.latents_std).view(1, -1, 1)
    z = z * std + mean
    dump("z", z)

    dec = vae.decoder
    t0 = time.time()
    h = dec.conv_pre(vae.dec_in_proj(z))
    dump("pre", h)
    for i in range(dec.num_upsamples):
        h = dec.ups[i][0](h)
        if i == 0:
            act = dec.resblocks[0].activations[0]
            dump("act0_in", h)
            dump("act0_out", act(h))
        acc = None
        for j in range(dec.num_kernels):
            b = dec.resblocks[i * dec.num_kernels + j](h)
            acc = b if acc is None else acc + b
        h = acc / dec.num_kernels
        dump(f"stage{i}", h)
        print(f"stage {i}: {list(h.shape)} at {time.time() - t0:.1f} s", flush=True)
    h = dec.conv_post(dec.activation_post(h))
    h = torch.clamp(h, min=-1.0, max=1.0)
    manifest["seconds"] = time.time() - t0
    dump("wave", h)

    # The staged run is the model's forward, reassembled: check it is.
    ref = vae.decode(z, return_dict=False)[0]
    manifest["staged_equals_decode"] = bool(torch.equal(ref, h))
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"{manifest['seconds']:.1f} s; staged == decode: {manifest['staged_equals_decode']}")


if __name__ == "__main__":
    main()
