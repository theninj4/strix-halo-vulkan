"""Run the whole Z-Image pipeline in fp32 on the CPU and dump every step.

This is stage 6's end-to-end oracle, and it is the only one that can exist:
the head, the blocks, the scheduler and the VAE each have their own reference,
but "the pipeline runs them in the right order with the right conventions" is a
claim about the composition and nothing below it can check it.

It is affordable only because the composition can be checked at a small size.
At a 256x256 image the unified sequence is 288 tokens rather than 4224, so a
denoising step is 3.5 TFLOP rather than 52, and diffusers walks all eight of
them on the CPU in a couple of minutes. What is under test is the wiring, and
the wiring does not know how big the image is.

The initial latent is dumped as well as the final one, because the Go side has
to start from the same noise: torch's generator and Go's do not agree, and
nothing else in the pipeline is random.

    .venv/bin/python reference/dump_zimage_run.py --size 256

Then, against it:

    go run ./cmd/zimage -width 256 -latents reference/out/zimagerun/latents_init.bin \\
        -dumplatent /tmp/latent.bin
"""

import argparse
import json
import os

import torch
from diffusers import ZImagePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Z-Image-Turbo")
    ap.add_argument("--out", default="reference/out/zimagerun")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--prompt", default="a red fox sitting in fresh snow, photograph")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    manifest = {
        "size": args.size, "steps": args.steps, "seed": args.seed,
        "prompt": args.prompt, "tensors": {},
    }

    def dump(name, tensor):
        tensor = tensor.detach().contiguous().to(torch.float32)
        with open(os.path.join(args.out, name + ".bin"), "wb") as fh:
            fh.write(tensor.numpy().tobytes())
        flat = tensor.flatten()
        manifest["tensors"][name] = {
            "shape": list(tensor.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:22s} {str(list(tensor.shape)):20s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.5g}")

    print("loading the pipeline in fp32 (~45 GB)...")
    pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=False)

    # The caption, dumped so the Go text encoder and cap_embedder can be
    # compared against it without re-running the whole thing.
    embeds, _ = pipe.encode_prompt(
        prompt=[args.prompt], do_classifier_free_guidance=False, device=torch.device("cpu"),
    )
    dump("prompt_embeds", embeds[0])
    manifest["prompt_tokens"] = int(embeds[0].shape[0])

    gen = torch.Generator("cpu").manual_seed(args.seed)
    latents = torch.randn(
        (1, pipe.transformer.in_channels, args.size // 8, args.size // 8),
        generator=gen, dtype=torch.float32,
    )
    dump("latents_init", latents[0])

    steps = []

    def on_step_end(p, i, t, kwargs):
        steps.append(kwargs["latents"][0].clone())
        return kwargs

    out = pipe(
        prompt=args.prompt,
        height=args.size, width=args.size,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        latents=latents.clone(),
        output_type="latent",
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images

    for i, s in enumerate(steps):
        dump(f"latents_{i}", s)
    dump("latents_final", out[0])

    image = pipe.vae.decode(
        (out.to(torch.float32) / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor,
        return_dict=False,
    )[0]
    dump("image", image[0])
    png = pipe.image_processor.postprocess(image, output_type="pil")[0]
    png.save(os.path.join(args.out, "reference.png"))

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors and reference.png to {args.out}")


if __name__ == "__main__":
    main()
