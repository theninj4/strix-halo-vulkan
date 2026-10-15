"""Dump a whole FLUX 3 Image Klein run, for X3/X4 and the end-to-end gates (research/flux3-vertical.md).

diffusers' own ModularPipeline, untouched, in fp32 on the CPU at its defaults
(16 steps, no guidance). Forward hooks record, for every transformer call,
the latents going in, the flow time, the jump-target time and the velocity
coming out — the per-step oracle and the schedule oracle at once — and the
latents the VAE decodes. The schedule is also checked against BFL's formula
(reference/flux3/bfl/sampling.py) and the gap written to the manifest.

    .venv-flux3/bin/python -I reference/dump_flux3_run.py t2i    (~45 GB RSS, ~15 min)
    .venv-flux3/bin/python -I reference/dump_flux3_run.py edit   (one reference: assets/1_figurine.png)

Nothing else heavy may run beside it.
"""

import json
import os
import sys

import numpy as np
import torch
from diffusers import ModularPipeline
from PIL import Image

MODEL = "models/flux-3-image-klein"
SEED = 0
STEPS = 16
PROMPTS = {
    "t2i": (
        "A wide-angle photograph of a lone American bison in a misty geothermal landscape, half hidden by white "
        "steam, a slate-blue stream and evergreens fading into the mist behind. Muted, cool palette, soft overcast "
        "light."
    ),
    "edit": (
        "The same figurine on the same surface, now lit by a single warm candle from the left, deep shadows on the "
        "right, a dark wooden background."
    ),
}


def bfl_schedule(steps, alpha=6.93):
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "t2i"
    out = f"reference/out/flux3run_{mode}"
    os.makedirs(out, exist_ok=True)
    manifest = {"mode": mode, "seed": SEED, "steps": STEPS, "prompt": PROMPTS[mode], "tensors": {}, "calls": []}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(out, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:24s} {str(list(t.shape)):18s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}", flush=True)

    pipe = ModularPipeline.from_pretrained(MODEL)
    pipe.load_components(dtype=torch.float32)
    pipe.to("cpu")

    step = {"i": 0}

    def on_transformer(mod, args, kwargs, result):
        i = step["i"]
        call = {"step": i, "timestep": float(kwargs["timestep"][0]), "target_timestep": float(kwargs["target_timestep"][0])}
        manifest["calls"].append(call)
        dump(f"step{i:02d}_latents", kwargs["hidden_states"][0])
        dump(f"step{i:02d}_velocity", result[0][0] if isinstance(result, tuple) else result.sample[0])
        if i == 0:
            dump("prompt_embeds", kwargs["encoder_hidden_states"][0])
            dump("img_ids", kwargs["img_ids"])
            dump("txt_ids", kwargs["txt_ids"])
            if kwargs.get("image_cond_hidden_states") is not None:
                dump("cond_tokens", kwargs["image_cond_hidden_states"][0])
                dump("cond_ids", kwargs["image_cond_ids"])
        print(f"step {i}: t={call['timestep']:.6f} -> {call['target_timestep']:.6f}", flush=True)
        step["i"] += 1

    pipe.transformer.register_forward_hook(on_transformer, with_kwargs=True)
    # the VAE's input after de-normalization and unpatchify, then the decoder's input after post_quant_conv
    pipe.vae.post_quant_conv.register_forward_pre_hook(lambda m, a: dump("vae_in", a[0][0]))
    pipe.vae.decoder.register_forward_pre_hook(lambda m, a: dump("decoder_in", a[0][0]))

    kwargs = dict(prompt=PROMPTS[mode], generator=torch.Generator().manual_seed(SEED), num_inference_steps=STEPS)
    if mode == "edit":
        ref = Image.open(os.path.join(MODEL, "assets/1_figurine.png")).convert("RGB")
        manifest["reference_size"] = list(ref.size)
        kwargs["image"] = ref
    with torch.no_grad():
        images = pipe(**kwargs, output="images")
    img = images[0]
    img.save(os.path.join(out, "image.png"))
    dump("image", torch.from_numpy(np.asarray(img).astype(np.float32)))
    manifest["image_size"] = list(img.size)

    ts = torch.tensor([c["timestep"] for c in manifest["calls"]] + [manifest["calls"][-1]["target_timestep"]], dtype=torch.float64)
    manifest["schedule"] = ts.tolist()
    manifest["schedule_vs_bfl_gap"] = float((ts - bfl_schedule(STEPS)).abs().max())
    print(f"schedule vs BFL formula gap {manifest['schedule_vs_bfl_gap']:.3g}")

    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"wrote {len(manifest['tensors'])} tensors to {out}")


if __name__ == "__main__":
    main()
