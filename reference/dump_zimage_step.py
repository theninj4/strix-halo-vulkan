"""Dump one transformer forward pass, phase by phase.

dump_zimage_run.py says whether the pipeline as a whole agrees with diffusers;
this says *where* it stops agreeing. It runs a single call of
ZImageTransformer2DModel.forward on the inputs the first denoising step sees --
the initial latent and the prompt embedding the run dumped -- with forward
hooks on the last module of each of the three phases, so the residual stream is
visible at every boundary the Go implementation also has one.

It needs dump_zimage_run.py to have run first, for latents_init and
prompt_embeds.

    .venv/bin/python reference/dump_zimage_step.py --step 0
"""

import argparse
import json
import os

import numpy as np
import torch
from diffusers import ZImageTransformer2DModel


def read(path, shape):
    raw = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(raw.reshape(shape).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Z-Image-Turbo")
    ap.add_argument("--run", default="reference/out/zimagerun")
    ap.add_argument("--out", default="reference/out/zimagestep")
    ap.add_argument("--step", type=int, default=0, help="which step's t and latent to use")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    run = json.load(open(os.path.join(args.run, "manifest.json")))
    steps = run["steps"]

    embeds = read(os.path.join(args.run, "prompt_embeds.bin"), run["tensors"]["prompt_embeds"]["shape"])
    name = "latents_init" if args.step == 0 else f"latents_{args.step - 1}"
    latents = read(os.path.join(args.run, name + ".bin"), run["tensors"][name]["shape"])

    # The pipeline's schedule, and the conditioning value it sends the model.
    sigmas = torch.linspace(1.0, 1 / steps, steps)
    shift = 3.0
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    t = torch.tensor([float(1 - sigmas[args.step])])
    print(f"step {args.step}: sigma {float(sigmas[args.step]):.5f}, model t {float(t[0]):.5f}")

    print("loading the transformer in fp32 (~25 GB)...")
    model = ZImageTransformer2DModel.from_pretrained(
        os.path.join(args.model, "transformer"), torch_dtype=torch.float32
    ).eval()

    manifest = {"step": args.step, "t": float(t[0]), "tensors": {}}

    def dump(tag, tensor):
        tensor = tensor.detach().contiguous().to(torch.float32)
        with open(os.path.join(args.out, tag + ".bin"), "wb") as fh:
            fh.write(tensor.numpy().tobytes())
        flat = tensor.flatten()
        manifest["tensors"][tag] = {
            "shape": list(tensor.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {tag:24s} {str(list(tensor.shape)):20s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.5g}")

    grabbed = {}

    def grab(tag):
        def hook(mod, inp, out):
            grabbed[tag] = out.detach().clone()
        return hook

    model.noise_refiner[0].register_forward_hook(grab("noise_refiner_0"))
    model.noise_refiner[-1].register_forward_hook(grab("after_noise_refiner"))
    model.context_refiner[-1].register_forward_hook(grab("after_context_refiner"))
    model.layers[0].register_forward_hook(grab("layers_0"))
    model.layers[-1].register_forward_hook(grab("after_layers"))
    model.all_final_layer["2-1"].register_forward_hook(grab("final"))
    model.cap_embedder.register_forward_hook(grab("cap_embedded"))
    model.t_embedder.register_forward_hook(grab("adaln_input"))

    # The pipeline's own call: the latent gains a frame axis and becomes a list.
    out = model([latents.unsqueeze(1)], t, [embeds], return_dict=False)[0]

    for tag in ["adaln_input", "cap_embedded", "after_context_refiner", "noise_refiner_0",
                "after_noise_refiner", "layers_0", "after_layers", "final"]:
        dump(tag, grabbed[tag][0] if grabbed[tag].ndim == 3 else grabbed[tag])
    dump("model_out", out[0])

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
