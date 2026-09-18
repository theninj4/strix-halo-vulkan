"""Run an SDEdit through the whole diffusers pipeline in fp32 and dump it.

This is IMAGE.md I7's composition oracle, and it exists for the same reason
dump_zimage_run.py does: the encoder, the scheduler and the loop each have
their own reference, but "an edit noises the encoded latent to the right sigma
and runs the right tail of the schedule" is a claim about the composition, and
nothing below it can check that.

What SDEdit is, in three lines:

    x0     = encode(image)                      the picture as a latent
    x_s    = (1 - sigma) * x0 + sigma * eps     noise it to an intermediate sigma
    image' = denoise(x_s, sigmas[start:])       run only the tail

`start` is diffusers' img2img arithmetic -- keep the last int(steps*strength)
steps -- and the sigma is the shifted schedule's at that index. The pipeline
takes `sigmas=` directly, and the shift it applies is elementwise, so handing
it the *tail of the raw linspace* produces exactly the tail of the full shifted
schedule the Go side walks.

`eps` is dumped because torch's generator and Go's do not agree and it is the
only random thing in here.

    .venv/bin/python reference/dump_zimage_edit.py --size 256 --strength 0.8

Then, against it:

    go test ./zimage/pipeline/ -run TestEditAgainstDiffusers
"""

import argparse
import json
import os

import torch
from PIL import Image
from diffusers import ZImagePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Z-Image-Turbo")
    ap.add_argument("--out", default="reference/out/zimageedit")
    ap.add_argument("--image", default="zimage.png.0.png")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", default="a red fox sitting in fresh snow, photograph")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    manifest = {
        "size": args.size, "steps": args.steps, "strength": args.strength,
        "seed": args.seed, "prompt": args.prompt, "image": args.image, "tensors": {},
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
        print(f"  {name:16s} {str(list(t.shape)):20s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.5g}")

    print("loading the pipeline in fp32 (~45 GB)...")
    pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=False)

    img = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    img.save(os.path.join(args.out, "input.png"))
    x = (torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float() / 127.5 - 1.0)
    x = x.reshape(args.size, args.size, 3).permute(2, 0, 1).unsqueeze(0).contiguous()
    dump("image_in", x[0])

    vae_cfg = pipe.vae.config
    mode = pipe.vae.encode(x).latent_dist.mode()
    x0 = (mode - vae_cfg.shift_factor) * vae_cfg.scaling_factor
    dump("latent_in", x0[0])

    # The schedule the pipeline would build for `steps`, shifted the way its
    # scheduler shifts it. Computed here rather than read off the scheduler so
    # that what `start` indexes is unambiguous.
    shift = pipe.scheduler.config.shift
    raw = torch.linspace(1.0, 1.0 / args.steps, args.steps)
    sigmas = shift * raw / (1 + (shift - 1) * raw)
    keep = min(max(int(args.steps * args.strength), 1), args.steps)
    start = args.steps - keep
    manifest["start"] = start
    manifest["sigma"] = float(sigmas[start])
    manifest["sigmas"] = [float(v) for v in sigmas]
    print(f"strength {args.strength} -> start at step {start} of {args.steps}, sigma {sigmas[start]:.4f}, "
          f"{keep} steps to run")

    gen = torch.Generator("cpu").manual_seed(args.seed)
    eps = torch.randn(x0.shape, generator=gen, dtype=torch.float32)
    dump("eps", eps[0])

    s = sigmas[start]
    noised = (1 - s) * x0 + s * eps
    dump("latents_start", noised[0])

    steps = []

    def on_step_end(p, i, t, kwargs):
        steps.append(kwargs["latents"][0].clone())
        return kwargs

    out = pipe(
        prompt=args.prompt,
        height=args.size, width=args.size,
        num_inference_steps=keep,
        sigmas=raw[start:].tolist(),
        guidance_scale=0.0,
        latents=noised.clone(),
        output_type="latent",
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images

    for i, st in enumerate(steps):
        dump(f"latents_{start + i}", st)
    dump("latents_final", out[0])

    image = pipe.vae.decode(
        (out.to(torch.float32) / vae_cfg.scaling_factor) + vae_cfg.shift_factor, return_dict=False,
    )[0]
    dump("image", image[0])
    pipe.image_processor.postprocess(image, output_type="pil")[0].save(
        os.path.join(args.out, "reference.png"))

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors and reference.png to {args.out}")


if __name__ == "__main__":
    main()
