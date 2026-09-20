"""Run Qwen-Image-2.1 end to end and dump every step, for Q3/Q6 — the oracle
the Go pipeline's image is compared against.

The whole pipeline in fp32 on CPU (~65 GB RSS), 256x256 in 4 steps — small on
purpose: this is the parity oracle, not a quality sample. `use_kv_cache=True`
because that is the shipped configuration (IMAGE.md decision 2); the cache-off
sample is a *different valid image* by design, so there is exactly one oracle.

The initial noise is generated here and passed in packed, so the Go side gets
the noise as data rather than having to reproduce torch's RNG. Dumped: the
prompt embeddings, the packed noise, the latents after every step, the final
latent, the RGBA image tensor, and a PNG for the eyeball.

Run it twice (the two-run convention) into different --out dirs and compare
manifests: CPU fp32 is deterministic, the sums must match exactly.

    .venv/bin/python reference/dump_qi21_run.py
    .venv/bin/python reference/dump_qi21_run.py --out reference/out/qi21run2

The one-time full-size oracle (~1 h CPU, ~65 GB RSS; holds ~half the machine,
so not while an LLM whole-model run is up):

    .venv/bin/python reference/dump_qi21_run.py --size 1024 --steps 40 --out reference/out/qi21run1024
"""

import argparse
import json
import os

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

MODEL = "models/Qwen-Image-2.1"
PROMPT = "a red bicycle leaning against a teal wall, morning light"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reference/out/qi21run")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    global SIZE, STEPS
    SIZE, STEPS = args.size, args.steps
    os.makedirs(args.out, exist_ok=True)

    pipe = QwenImage21Pipeline.from_pretrained(MODEL, torch_dtype=torch.float32)

    manifest = {
        "prompt": PROMPT, "seed": SEED, "size": SIZE, "steps": STEPS,
        "use_kv_cache": True, "tensors": {},
    }

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(args.out, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:18s} {str(list(t.shape)):20s} sum={flat.double().sum():+.6f} absmax={flat.abs().max():.5g}")

    latent_side = SIZE // 16
    generator = torch.Generator("cpu").manual_seed(SEED)
    noise = torch.randn((1, 1, 64, latent_side, latent_side), generator=generator, dtype=torch.float32)
    packed = pipe._pack_latents(noise, 1, 64, latent_side, latent_side)
    dump("noise", packed)

    def cb(p, i, t, kw):
        dump(f"step{i}_latents", kw["latents"])
        if i == 0:
            dump("prompt_embeds", kw["prompt_embeds"])
        return kw

    out = pipe(
        prompt=PROMPT,
        width=SIZE, height=SIZE,
        num_inference_steps=STEPS,
        latents=packed,
        output_type="np",
        use_kv_cache=True,
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
    )

    image = out.images[0]  # [H, W, C] in [0, 1]
    dump("image", torch.from_numpy(image))
    mode = "RGBA" if image.shape[-1] == 4 else "RGB"
    png = Image.fromarray((image * 255).round().astype(np.uint8), mode)
    png.save(os.path.join(args.out, "image.png"))
    manifest["image_mode"] = mode

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
