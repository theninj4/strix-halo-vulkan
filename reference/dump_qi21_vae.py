"""Dump Qwen-Image-2.1's VAE, for Q5.

A new port, not a variant of z-image's: Wan-lineage causal-3D residual conv
AE, RMS-norm (not group norm), avg-down/dup-up resamplers, encoder base 96 /
decoder base 144, z_dim 64, 16x spatial, 4-channel RGBA both ways. At T=1 the
temporal machinery must degenerate; this dump is what "degenerate" means
numerically.

What gets dumped:

  * decode: a seeded standard-normal latent, denormalized with the config's
    per-channel latents_mean/std exactly as the pipeline does, then
    `vae.decode` — plus every top-level child of the decoder (hooked
    generically by name) so the Go port compares stage by stage;
  * encode: a deterministic RGBA test card in [-1, 1] → posterior **mode**
    (the pipeline's argmax path; never sample), raw and normalized;
  * round trip: decode(encode(image)) for the eyeball and a tolerance record;
  * a non-square case, 192x320, because rectangular geometry is where
    resampler padding mistakes live.

    .venv/bin/python reference/dump_qi21_vae.py
"""

import json
import os

import torch
from diffusers import AutoencoderKLQwenImage21

OUT = "reference/out/qi21vae"
MODEL = "models/Qwen-Image-2.1"
SEED = 5


def test_card(h, w):
    """Deterministic RGBA image in [-1, 1]: gradients, a disc, banded alpha, seeded noise."""
    torch.manual_seed(SEED)
    y = torch.linspace(-1, 1, h)[:, None].expand(h, w)
    x = torch.linspace(-1, 1, w)[None, :].expand(h, w)
    disc = ((x**2 + y**2) < 0.25).float() * 2 - 1
    alpha = torch.where((y * 4).floor() % 2 == 0, torch.ones_like(y), -torch.ones_like(y))
    img = torch.stack([x, y, disc, alpha])
    img = (img + 0.05 * torch.randn(4, h, w)).clamp(-1, 1)
    return img[None, :, None]  # [1, 4, 1, H, W]


def main():
    os.makedirs(OUT, exist_ok=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).eval()
    cfg = vae.config
    mean = torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1)
    std = torch.tensor(cfg.latents_std).view(1, cfg.z_dim, 1, 1, 1)

    manifest = {
        "seed": SEED,
        "config": {k: v for k, v in dict(cfg).items() if not k.startswith("_")},
        "cases": {}, "tensors": {},
    }

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:26s} {str(list(t.shape)):24s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}")

    def hook_children(module, prefix, store):
        # Direct children, plus the entries of ModuleList children (up_blocks,
        # down_blocks are called in a loop, so the list itself never fires).
        targets = []
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ModuleList):
                targets += [(f"{name}.{i}", sub) for i, sub in enumerate(child)]
            else:
                targets.append((name, child))
        handles = []
        for name, child in targets:
            def make(name):
                def hook(mod, args, output):
                    out = output[0] if isinstance(output, tuple) else output
                    if torch.is_tensor(out):
                        store[f"{prefix}_{name}"] = out
                return hook
            handles.append(child.register_forward_hook(make(name)))
        return handles

    def run_case(label, h, w):
        latent_h, latent_w = h // 16, w // 16
        torch.manual_seed(SEED)
        z_norm = torch.randn(1, cfg.z_dim, 1, latent_h, latent_w)
        z = z_norm * std + mean  # what the pipeline feeds vae.decode
        dump(label + "_z_norm", z_norm)

        stages = {}
        handles = hook_children(vae.decoder, label + "_dec", stages)
        with torch.no_grad():
            image = vae.decode(z, return_dict=False)[0]
        for hd in handles:
            hd.remove()
        dump(label + "_decoded", image)
        for name, t in stages.items():
            dump(name, t)

        img = test_card(h, w)
        dump(label + "_card", img)
        stages = {}
        handles = hook_children(vae.encoder, label + "_enc", stages)
        with torch.no_grad():
            posterior = vae.encode(img).latent_dist
            z_mode = posterior.mode()
        for hd in handles:
            hd.remove()
        dump(label + "_encoded_mode", z_mode)
        dump(label + "_encoded_norm", (z_mode - mean) / std)
        for name, t in stages.items():
            dump(name, t)

        with torch.no_grad():
            rt = vae.decode(z_mode, return_dict=False)[0]
        dump(label + "_roundtrip", rt)
        err = (rt - img).abs()
        manifest["cases"][label] = {
            "h": h, "w": w, "latent": [latent_h, latent_w],
            "roundtrip_mae": float(err.mean()), "roundtrip_max": float(err.max()),
        }
        print(f"{label}: roundtrip mae {err.mean():.4f} max {err.max():.4f}")

    print("256x256:")
    run_case("s256", 256, 256)
    print("192x320:")
    run_case("r192x320", 192, 320)

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
