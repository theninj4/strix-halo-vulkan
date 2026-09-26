"""The official bf16 transformer's distance from the fp32 oracle, for the
int8 bank (research/qimage-vertical.md Q13; VIDEO.md M11a's method).

`QwenImage21Pipeline.from_pretrained(..., torch_dtype=torch.bfloat16)` is
what Qwen ships, and its error is the scale a quantised bank is priced on —
not our fp16 path's. This runs that transformer over one of the fp32
oracles (dump_qi21_run.py, or dump_qi21_edit.py for an edit), isolated to
the transformer: the oracle's own prompt embeddings, condition latents and
noise go in (narrowed to bf16, as the pipeline would hold them), the text
encoder and VAE are never loaded, and the KV cache is on as shipped.

The denoising loop is the pipeline's own (pipeline_qwenimage21.py, step 5),
copied rather than called so an edit can be fed the oracle's condition
latents and pad mask directly; on the t2i oracle it reproduces the
pipeline-driven run exactly.

Teacher-forced (default): after every step the bf16 latents are recorded and
the loop continues from the oracle's own step latents, so step i is the
released model's one-step error from the oracle's state — what qimage/dit's
teacher-forced gate measures of ours. --free lets the bf16 trajectory run
and, on a t2i oracle, decodes it with the bf16 VAE the way the pipeline's
tail does, into image.bin / image.png (qimage/pipeline's served gate).

    .venv/bin/python reference/dump_qi21_dit_bf16.py --ref reference/out/qi21run [--free]
    .venv/bin/python reference/dump_qi21_dit_bf16.py --ref reference/out/qi21edit [--free]
    .venv/bin/python reference/dump_qi21_dit_bf16.py --ref reference/out/qi21run1024   (~16 GB, ~15 min)
"""

import argparse
import inspect
import json
import os

import numpy as np
import torch
from diffusers import (AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler,
                       QwenImage21Pipeline, QwenImage21Transformer2DModel)
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

MODEL = "models/Qwen-Image-2.1"
BF16 = torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="reference/out/qi21run")
    ap.add_argument("--free", action="store_true")
    args = ap.parse_args()
    out_dir = args.ref.rstrip("/") + ("_bf16_free" if args.free else "_bf16")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(args.ref, "manifest.json")) as fh:
        man = json.load(fh)
    size, steps = man["size"], man["steps"]

    def load(name):
        meta = man["tensors"][name]
        a = np.fromfile(os.path.join(args.ref, name + ".bin"), dtype=np.float32)
        return torch.from_numpy(a.reshape(meta["shape"]))

    torch.set_num_threads(os.cpu_count())
    mod = inspect.getmodule(QwenImage21Pipeline)
    tr = QwenImage21Transformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=BF16).eval()
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder="scheduler")

    manifest = {"ref": args.ref, "dtype": "bfloat16", "free": args.free,
                "size": size, "steps": steps, "tensors": {}}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(out_dir, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        want = load(name).reshape(t.shape)
        print(f"  {name:18s} {str(list(t.shape)):20s} absmax={flat.abs().max():.5g}"
              f" rms diff {((t - want) ** 2).mean().sqrt():.4g}", flush=True)

    embeds = load("prompt_embeds").to(BF16)
    latents = load("noise").to(BF16)
    n = latents.shape[1]
    side = size // 16
    edit = "cond_latents" in man["tensors"]
    cond = load("cond_latents").to(BF16) if edit else None
    if edit:
        img_shapes = [[tuple(s) for s in man["img_shapes"]]]
        pad = load("image_pad_mask").reshape(1, -1) != 0
    else:
        img_shapes = [[(1, side, side)]]
        pad = torch.zeros(1, embeds.shape[1], dtype=torch.bool)
    pad = torch.cat([pad, pad.new_ones(1, n // 4)], dim=1)

    cfg = sched.config
    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu = mod.calculate_shift(n, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
    timesteps, _ = mod.retrieve_timesteps(sched, steps, "cpu", sigmas=sigmas, mu=mu)
    cache_on = tr.config.causal_condition
    cache = mod.QwenImage21KVCache(len(tr.transformer_blocks)) if cache_on else None
    sched.set_begin_index(0)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            kv_mode = ("extract" if i == 0 else "cached") if cache_on else None
            inp = latents if cond is None else torch.cat([cond, latents], dim=1)
            timestep = t.expand(1).to(latents.dtype)
            with tr.cache_context("cond"):
                pred = tr(hidden_states=inp, timestep=timestep / 1000, encoder_hidden_states=embeds,
                          encoder_hidden_states_mask=None, img_shapes=img_shapes, img_mask=pad,
                          kv_cache=cache, kv_cache_mode=kv_mode, return_dict=False)[0]
            pred = pred[:, -n:]
            latents = sched.step(pred, t, latents, return_dict=False)[0]
            dump(f"step{i}_latents", latents)
            if not args.free:
                latents = load(f"step{i}_latents").to(latents.dtype)

    if args.free and not edit:
        # The pipeline's tail (pipeline_qwenimage21.py, after step 5), bf16.
        vae = AutoencoderKLQwenImage21.from_pretrained(MODEL, subfolder="vae", torch_dtype=BF16).eval()
        scale = 16  # the pipeline's vae_scale_factor
        with torch.no_grad():
            lat = QwenImage21Pipeline._unpack_latents(latents, size, size, scale).to(vae.dtype)
            mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(lat.dtype)
            std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(lat.dtype)
            image = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
        image = VaeImageProcessor(vae_scale_factor=scale, vae_latent_channels=vae.config.z_dim).postprocess(image, output_type="np")[0]
        t = torch.from_numpy(image).contiguous().to(torch.float32)
        with open(os.path.join(out_dir, "image.bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        manifest["tensors"]["image"] = {"shape": list(t.shape), "count": int(t.numel()),
                                        "sum": float(t.double().sum()), "absmax": float(t.abs().max())}
        mode = "RGBA" if image.shape[-1] == 4 else "RGB"
        Image.fromarray((image * 255).round().astype(np.uint8), mode).save(os.path.join(out_dir, "image.png"))
        print(f"  image {list(t.shape)}")

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {out_dir}")


if __name__ == "__main__":
    main()
