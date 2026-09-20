"""Run Qwen-Image-2.1 end to end as an *edit* and dump every step, for Q8.3 —
the oracle the Go edit pipeline is compared against.

`dump_qi21_run.py` is this script's t2i twin and the two are deliberately
alike: same size, same step count, same seed, noise generated here and passed
in packed so the Go side gets it as data. What an edit adds is the prefix —
a condition image that is encoded twice, by the vision tower into text
context and by the VAE into latents the transformer attends over — and this
walks that half:

  * the condition image at both of its resolutions and in both of its copies
    (RGBA for the VAE, flattened over white for the tower);
  * its VAE latents, packed and normalised, which are the `cond` rows the
    denoiser prepends to the target at every step;
  * the prompt embeddings and the image-pad mask the transformer reads the
    layout from;
  * the per-step target latents and the final image.

`output_resolution` is set to the image's own size, which makes the
pipeline's condition resize the identity: that keeps this oracle about the
*pipeline* rather than about a resampler. The resize is a separate question
and is dumped separately at the bottom — `calculate_dimensions` over a
spread of aspect ratios, plus one image the pipeline actually resized, so a
Go resampler can be gated without running any of this.

The whole pipeline in fp32 on CPU is ~68 GB RSS and must run alone.

    .venv/bin/python reference/dump_qi21_edit.py
    .venv/bin/python reference/dump_qi21_edit.py --out reference/out/qi21edit2
"""

import argparse
import json
import os

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

from dump_qi21_vision import test_card

MODEL = "models/Qwen-Image-2.1"
PROMPT = "make the sky a deep orange sunset"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reference/out/qi21edit")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    size, steps = args.size, args.steps
    os.makedirs(args.out, exist_ok=True)

    pipe = QwenImage21Pipeline.from_pretrained(MODEL, torch_dtype=torch.float32)

    manifest = {
        "prompt": PROMPT, "seed": SEED, "size": size, "steps": steps,
        "output_resolution": size, "use_kv_cache": True, "tensors": {},
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
        print(f"  {name:22s} {str(list(t.shape)):22s} sum={flat.double().sum():+.6f} absmax={flat.abs().max():.5g}")

    # --- the condition image, the same card the tower dump uses ----------
    card = test_card(size)
    rgba = Image.fromarray(card, mode="RGBA")
    dump("cond_rgba", torch.from_numpy(card).permute(2, 0, 1).float()[None] / 127.5 - 1)

    # What the pipeline does to it before either encoder sees it. With
    # output_resolution == the card's own size this is the identity, and the
    # dumped tensor is what proves that rather than an assumption.
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (  # noqa: PLC0415
        calculate_dimensions,
    )
    in_w, in_h, _ = calculate_dimensions(size * size, card.shape[1] / card.shape[0])
    manifest["cond_input_size"] = [in_w, in_h]
    vae_image = pipe.image_processor.preprocess(rgba, width=in_w, height=in_h).unsqueeze(2)
    dump("cond_vae_image", vae_image)

    # The condition latents: posterior *mode*, per-channel normalised, packed
    # to one row per latent token. These are the rows the denoiser prepends to
    # the target at every step.
    cond_latents = pipe._encode_vae_image(vae_image.to(torch.float32), None)
    dump("cond_latents_grid", cond_latents)
    lat_h, lat_w = cond_latents.shape[3:]
    dump("cond_latents", pipe._pack_latents(cond_latents, 1, 64, lat_h, lat_w))
    manifest["cond_latent_shape"] = [1, int(lat_h), int(lat_w)]

    # --- the prompt side, so a failure can be localised before the loop --
    with torch.no_grad():
        embeds, _, pad_mask = pipe._get_qwen_prompt_embeds(PROMPT, [rgba], device="cpu")
    dump("prompt_embeds", embeds)
    dump("image_pad_mask", pad_mask.float())
    manifest["prompt_tokens"] = int(embeds.shape[1])
    manifest["pad_mask_count"] = int(pad_mask.sum())

    # --- the run ---------------------------------------------------------
    latent_side = size // 16
    generator = torch.Generator("cpu").manual_seed(SEED)
    noise = torch.randn((1, 1, 64, latent_side, latent_side), generator=generator, dtype=torch.float32)
    packed = pipe._pack_latents(noise, 1, 64, latent_side, latent_side)
    dump("noise", packed)

    manifest["img_shapes"] = [[1, int(lat_h), int(lat_w)], [1, latent_side, latent_side]]

    def cb(p, i, t, kw):
        dump(f"step{i}_latents", kw["latents"])
        return kw

    out = pipe(
        prompt=PROMPT,
        image=[rgba],
        width=size, height=size,
        output_resolution=size,
        num_inference_steps=steps,
        latents=packed,
        output_type="np",
        use_kv_cache=True,
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    image = out.images[0]
    dump("image", torch.from_numpy(image))
    mode = "RGBA" if image.shape[-1] == 4 else "RGB"
    Image.fromarray((image * 255).round().astype(np.uint8), mode).save(os.path.join(args.out, "image.png"))
    manifest["image_mode"] = mode

    # --- the resize, as its own question ---------------------------------
    #
    # Every condition image the server is handed goes through this and almost
    # none of them will be identity like the one above. The geometry is pure
    # arithmetic and is dumped as cases; the resampler is PIL's bicubic and is
    # dumped as one before/after pair, so a Go port can be gated on both
    # without staging a pipeline.
    cases = []
    for res in (256, 1024):
        for w, h in ((256, 256), (192, 384), (384, 192), (500, 333), (1000, 1000)):
            cw, ch, _ = calculate_dimensions(res * res, w / h)
            cases.append({"resolution": res, "src": [w, h], "out": [cw, ch]})
    manifest["calculate_dimensions"] = cases

    # Three pairs, because the resampler has three behaviours and a single
    # pair would only show one of them: Pillow stretches the filter when it
    # *downscales* and not when it upscales, and it skips the pass for an
    # axis whose size does not change. A photo at output_resolution 1024 is
    # usually an upscale, so that branch is not a corner case.
    def resize_pair(tag, src_img, out_w, out_h):
        dump(f"{tag}_src", torch.from_numpy(np.array(src_img)).permute(2, 0, 1).float()[None] / 127.5 - 1)
        dst = pipe.image_processor.resize(src_img, width=out_w, height=out_h)
        dump(f"{tag}_dst", torch.from_numpy(np.array(dst)).permute(2, 0, 1).float()[None] / 127.5 - 1)
        manifest[f"{tag}_out_size"] = [out_w, out_h]
        return dst

    wide = Image.fromarray(test_card(size), mode="RGBA").resize((500, 333), Image.BICUBIC)
    rw, rh, _ = calculate_dimensions(size * size, 500 / 333)
    down = resize_pair("resize", wide, rw, rh)          # 500x333 -> 320x224, both axes down
    small = Image.fromarray(test_card(size), mode="RGBA").resize((128, 96), Image.BICUBIC)
    uw, uh, _ = calculate_dimensions(size * size, 128 / 96)
    resize_pair("resize_up", small, uw, uh)             # 128x96 -> up on both axes
    resize_pair("resize_axis", down, rw, rh // 2)       # height only; width must not be refiltered

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
