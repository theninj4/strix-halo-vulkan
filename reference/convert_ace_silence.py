"""Convert ACE-Step's silence_latent.pt to safetensors, for MUSIC.md A0.

The DiT's "no source audio" input and its default timbre reference are both
slices of this tensor. Upstream loads it as
`torch.load(path).transpose(1, 2)` -- [1, 15000, 64] fp32, 25 latents a
second for 10 minutes -- and slices time from the front. The Go side reads
no pickles, so it is written once, in that time-major layout, as
`silence_latent.safetensors` beside the original.

    .venv-acestep/bin/python reference/convert_ace_silence.py
"""

import torch
from safetensors.torch import save_file

SRC = "models/acestep-v15-xl-turbo/silence_latent.pt"
DST = "models/acestep-v15-xl-turbo/silence_latent.safetensors"


def main():
    s = torch.load(SRC, weights_only=True).transpose(1, 2).contiguous()
    assert s.shape == (1, 15000, 64) and s.dtype == torch.float32, (s.shape, s.dtype)
    save_file({"silence_latent": s[0]}, DST)
    print(f"{DST}: {tuple(s[0].shape)} absmax {s.abs().max():.4g} mean {s.mean():.4g} std {s.std():.4g}")


if __name__ == "__main__":
    main()
