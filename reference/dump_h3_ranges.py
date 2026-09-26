"""Audit MiniMax-H3's bf16 weights against fp16, for VIDEO.md M0 / decision 4.

The checkpoint is bf16 and the device path is fp16. bf16 → fp16 keeps every
mantissa bit (8 of fp16's 10), so the only losses are range: a weight above
65504 overflows, and one below 2^-14 (6.1e-5) lands in fp16's subnormals,
losing precision until it underflows to zero below 2^-24 (6e-8). This streams
every tensor one at a time and records, per tensor, its absmax, the fraction
of nonzero weights in the subnormal band, and the fraction that would flush to
zero, and then prints the worst offenders per component.

    .venv/bin/python reference/dump_h3_ranges.py [component ...]   (default: transformer text_encoder)
"""

import json
import os
import sys

import torch
from safetensors import safe_open

MODEL = "models/MiniMax-H3"
OUT = "reference/out/h3ranges"
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0**-14
FP16_MIN_SUB = 2.0**-24


def audit(component):
    idx = json.load(open(next(os.path.join(MODEL, component, f) for f in os.listdir(os.path.join(MODEL, component))
                              if f.endswith(".index.json"))))
    shards = sorted(set(idx["weight_map"].values()))
    rows = {}
    for shard in shards:
        with safe_open(os.path.join(MODEL, component, shard), "pt") as fh:
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
        print(f"  {component}/{shard}: {len(rows)} tensors so far", flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    for component in sys.argv[1:] or ["transformer", "text_encoder"]:
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
