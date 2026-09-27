"""Audit ACE-Step 1.5's weights against fp16, for MUSIC.md A0 / decision 4.

The DiT is stored fp32 and the LM and VAE bf16; the device path is fp16.
From bf16 the only losses are range; from fp32 there is mantissa rounding
too, which is harmless at 11 bits, so range is what this measures: a weight
above 65504 overflows, one below 2^-14 lands in fp16's subnormals and one
below 2^-24 flushes to zero. Streams every tensor once and records, per
tensor, its absmax and the subnormal and flush fractions of its nonzero
weights, then prints the worst offenders per component.

    .venv-acestep/bin/python reference/dump_ace_ranges.py [component ...]   (default: dit lm vae)
"""

import glob
import json
import os
import sys

import torch
from safetensors import safe_open

COMPONENTS = {
    "dit": "models/acestep-v15-xl-turbo",
    "lm": "models/acestep-5Hz-lm-4B",
    "vae": "models/Ace-Step1.5/vae",
}
OUT = "reference/out/aceranges"
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0**-14
FP16_MIN_SUB = 2.0**-24


def audit(component):
    rows = {}
    for shard in sorted(glob.glob(os.path.join(COMPONENTS[component], "*.safetensors"))):
        with safe_open(shard, "pt") as fh:
            for name in fh.keys():
                t = fh.get_tensor(name)
                if not t.is_floating_point():
                    continue
                a = t.float().abs()
                nz = a[a > 0]
                n = max(nz.numel(), 1)
                rows[name] = {
                    "dtype": str(t.dtype).removeprefix("torch."),
                    "shape": list(t.shape),
                    "absmax": float(a.max()),
                    "overflow": int((a > FP16_MAX).sum()),
                    "subnormal": float(((nz < FP16_MIN_NORMAL) & (nz >= FP16_MIN_SUB)).sum()) / n,
                    "flush": float((nz < FP16_MIN_SUB).sum()) / n,
                }
        print(f"  {component}/{os.path.basename(shard)}: {len(rows)} tensors so far", flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    for component in sys.argv[1:] or list(COMPONENTS):
        rows = audit(component)
        json.dump(rows, open(os.path.join(OUT, component + ".json"), "w"), indent=1)
        over = {k: v for k, v in rows.items() if v["overflow"]}
        print(f"{component}: {len(rows)} tensors, absmax {max(v['absmax'] for v in rows.values()):.4g}, "
              f"{len(over)} with fp16 overflow")
        for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["absmax"])[:8]:
            print(f"  absmax {v['absmax']:10.4g}  {k}")
        for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["flush"])[:8]:
            print(f"  flush {v['flush']:9.3%} subnormal {v['subnormal']:8.3%}  {k}")


if __name__ == "__main__":
    main()
