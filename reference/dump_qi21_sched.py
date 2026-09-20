"""Dump Qwen-Image-2.1's noise schedule, for Q0/Q3.

Unlike Z-Image's fixed shift 3.0, 2.1 uses *dynamic* shifting: mu is linear in
the target token count between (base_image_seq_len 256, base_shift 0.5) and
(max_image_seq_len 8192, max_shift 0.9), the time shift is the exponential
form sigma' = e^mu / (e^mu + (1/sigma - 1)), and `shift_terminal: 0.02`
re-stretches the whole schedule so the last live sigma is exactly 0.02.
Every one of those is a place a port silently diverges, so this dumps the
sigma/timestep tables for the sizes and step counts the Go side will test
with, straight from diffusers' own FlowMatchEulerDiscreteScheduler under the
pipeline's own calculate_shift.

Also dumps one Euler step over deterministic vectors, same as the z-image
scheduler dump did, so the step arithmetic has an oracle of its own.

    .venv/bin/python reference/dump_qi21_sched.py
"""

import json
import os

import numpy as np
import torch
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

OUT = "reference/out/qi21sched"
SCHEDULER = "models/Qwen-Image-2.1/scheduler"

# (label, image side in pixels, steps). Token count = (side/16)^2.
CASES = [
    ("s256x4", 256, 4),
    ("s512x12", 512, 12),
    ("s1024x40", 1024, 40),
    ("s1024x16", 1024, 16),
    ("s2048x40", 2048, 40),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(SCHEDULER)
    cfg = dict(sched.config)
    manifest = {"config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "cases": {}, "tensors": {}}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:24s} {str(list(t.shape)):12s} sum={flat.double().sum():+.6f}")

    for label, side, steps in CASES:
        tokens = (side // 16) * (side // 16)
        mu = calculate_shift(
            tokens,
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
        sigmas_in = np.linspace(1.0, 1 / steps, steps)
        sched.set_timesteps(steps, sigmas=sigmas_in, mu=mu)
        manifest["cases"][label] = {"side": side, "tokens": tokens, "steps": steps, "mu": float(mu)}
        print(f"{label}: {tokens} tokens, mu={mu:.6f}")
        dump(label + "_sigmas", sched.sigmas)
        dump(label + "_timesteps", sched.timesteps)

    # One Euler step over vectors the Go side reproduces exactly. The pipeline
    # hands the transformer timestep/1000 and steps only the target latents.
    label, side, steps = CASES[0]
    tokens = (side // 16) * (side // 16)
    mu = calculate_shift(tokens, 256, sched.config.get("max_image_seq_len", 4096), 0.5, sched.config.get("max_shift", 1.15))
    sched.set_timesteps(steps, sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
    sched.set_begin_index(0)
    sample = torch.arange(32, dtype=torch.float32) / 8 - 2
    model_out = torch.cos(torch.arange(32, dtype=torch.float32))
    dump("step_sample", sample)
    dump("step_model_out", model_out)
    dump("step_result", sched.step(model_out, sched.timesteps[0], sample, return_dict=False)[0])
    dump("step_model_t", sched.timesteps.float() / 1000)

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
