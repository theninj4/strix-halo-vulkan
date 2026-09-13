"""Encode a real image to a VAE latent, and decode it back with diffusers.

This gives the Vulkan decoder something better than noise to prove itself on:
a latent that decodes to a recognisable picture, plus the reference decode to
diff against. If the port has a structural bug the image is visibly wrong,
which is the check a relative-error number cannot make on its own.

    .venv/bin/python reference/encode_image.py --image <path> --size 512
"""

import argparse
import os
import struct

import torch
from PIL import Image
from diffusers import AutoencoderKL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="models/Z-Image-Turbo/vae")
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="reference/out/image")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32)
    vae.eval()

    img = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    img.save(os.path.join(args.out, "input.png"))
    x = torch.from_numpy(
        (torch.frombuffer(img.tobytes(), dtype=torch.uint8).float() / 127.5 - 1.0)
        .reshape(args.size, args.size, 3).permute(2, 0, 1).numpy()
    ).unsqueeze(0)

    with torch.no_grad():
        # The deterministic branch of the posterior: no sampling, so the Go
        # side reproduces this exactly.
        latent = vae.encode(x).latent_dist.mode()
        decoded = vae.decode(latent).sample

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(args.out, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        print(f"  {name:10s} {list(t.shape)}")

    dump("latent", latent)
    dump("decoded", decoded)

    px = ((decoded[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
    Image.fromarray(px).save(os.path.join(args.out, "reference.png"))

    with open(os.path.join(args.out, "shape.txt"), "w") as fh:
        fh.write(f"{latent.shape[1]} {latent.shape[2]} {latent.shape[3]}\n")
    print(f"wrote latent {tuple(latent.shape)} and reference.png to {args.out}")


if __name__ == "__main__":
    main()
