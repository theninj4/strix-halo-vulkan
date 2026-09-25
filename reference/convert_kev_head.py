"""Convert Kev's head.pt (a torch pickle) into files the Go loader reads.

CLASSIFICATION.md K1. head.pt holds the pointer head's four tensors and the
checkpoint's metadata (base, revision, LoRA rank, head_dim, temperature...).
Go has no pickle reader, and the head is tiny, so it is converted once:

  head.safetensors  q.weight [256, 2560], q.bias [256], k.weight, k.bias, fp32
  head.json         every non-tensor field of head.pt, as JSON

    .venv/bin/python reference/convert_kev_head.py models/kev-4b
"""
import json
import sys

import torch
from safetensors.torch import save_file

run = sys.argv[1] if len(sys.argv) > 1 else "models/kev-4b"
d = torch.load(f"{run}/head.pt", map_location="cpu")
head = d.pop("head")
save_file({k: v.float().contiguous() for k, v in head.items()}, f"{run}/head.safetensors")
with open(f"{run}/head.json", "w") as f:
    json.dump(d, f, indent=1, default=str)
print({k: tuple(v.shape) for k, v in head.items()}, "temperature", d["temperature"], "head_dim", d["head_dim"])
