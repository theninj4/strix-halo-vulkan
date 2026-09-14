"""Dump the pieces of the Z-Image transformer that are *not* a block, for stage 6.

dump_dit_block.py is the oracle for one block and dump_dit_stack.py for a
sequence of them. Everything around those blocks -- the head that turns a
latent and a caption into the two token streams, the tail that turns the
residual stream back into a latent, the positional ids that tie the two
streams together, and the scheduler that walks the eight steps -- has never
been dumped, because until stage 6 nothing ran it.

It is all index arithmetic and small linear layers, which is exactly the kind
of code that produces a plausible tensor when it is wrong:

  * the caption is padded to a multiple of 32 and the padded rows are replaced
    by a learned token *after* the embedder, not before;
  * the caption's own position ids start at 1, not 0, and the image's axis-0
    id is the padded caption length plus one, so a caption of 40 tokens and
    one of 64 put the image at *different* rotary positions;
  * `_pad_with_ids` builds the caption's ids from the already-padded length
    and then appends pad_len more, so the id list is longer than the stream
    and `_prepare_sequence` truncates it -- a detail no shape check catches;
  * patchify interleaves (ph, pw, c) into 64 features in one order and
    unpatchify undoes it in another;
  * the final layer's norm is a LayerNorm -- mean subtracted -- where every
    other norm in this model is an RMS norm.

So this runs diffusers' own methods for all of it and dumps what they produce.
The transformer is instantiated with **no blocks at all** (n_layers=0,
n_refiner_layers=0): the blocks are already covered, and without them the
model is 130 MB rather than 24 GB.

    .venv/bin/python reference/dump_zimage.py
"""

import argparse
import json
import os

import torch
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from safetensors import safe_open

# The tensors that are not inside a block. Everything else in the checkpoint
# belongs to layers/noise_refiner/context_refiner.
HEAD_TENSORS = [
    "t_embedder.mlp.0.weight",
    "t_embedder.mlp.0.bias",
    "t_embedder.mlp.2.weight",
    "t_embedder.mlp.2.bias",
    "all_x_embedder.2-1.weight",
    "all_x_embedder.2-1.bias",
    "cap_embedder.0.weight",
    "cap_embedder.1.weight",
    "cap_embedder.1.bias",
    "all_final_layer.2-1.adaLN_modulation.1.weight",
    "all_final_layer.2-1.adaLN_modulation.1.bias",
    "all_final_layer.2-1.linear.weight",
    "all_final_layer.2-1.linear.bias",
    "x_pad_token",
    "cap_pad_token",
]


def load_head(model, transformer, index):
    """Fill the model's non-block tensors from the checkpoint."""
    state = {}
    for name in HEAD_TENSORS:
        shard = index.get(name)
        if shard is None:
            raise SystemExit(f"{name} is not in the checkpoint")
        with safe_open(os.path.join(transformer, shard), framework="pt") as fh:
            state[name] = fh.get_tensor(name).to(torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"checkpoint has tensors the model does not want: {unexpected}")
    if missing:
        raise SystemExit(f"model wants tensors the dump does not load: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer", default="models/Z-Image-Turbo/transformer")
    ap.add_argument("--scheduler", default="models/Z-Image-Turbo/scheduler")
    ap.add_argument("--out", default="reference/out/zimage")
    ap.add_argument("--latent", type=int, default=32, help="latent side; the image is 8x this")
    ap.add_argument("--caption", type=int, default=40, help="caption tokens, deliberately not a multiple of 32")
    ap.add_argument("--steps", type=int, default=8, help="denoising steps for the schedule")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.transformer, "config.json")))
    with open(os.path.join(args.transformer, "diffusion_pytorch_model.safetensors.index.json")) as fh:
        index = json.load(fh)["weight_map"]

    # No blocks: this stage is everything around them.
    model_cfg = dict(cfg)
    model_cfg.pop("_class_name", None)
    model_cfg.pop("_diffusers_version", None)
    model_cfg["n_layers"] = 0
    model_cfg["n_refiner_layers"] = 0
    model = ZImageTransformer2DModel(**model_cfg).to(torch.float32).eval()
    load_head(model, args.transformer, index)

    dim = cfg["dim"]
    patch, f_patch = cfg["all_patch_size"][0], cfg["all_f_patch_size"][0]
    chan = cfg["in_channels"]
    L = args.latent
    img_tokens = (L // patch) * (L // patch)

    torch.manual_seed(args.seed)
    latent = torch.randn(chan, 1, L, L)
    cap_feats = torch.randn(args.caption, cfg["cap_feat_dim"])
    # t is what the pipeline hands the transformer: (1000 - timestep)/1000.
    t = torch.tensor([0.317])

    manifest = {
        "latent": L, "caption": args.caption, "dim": dim, "in_channels": chan,
        "patch_size": patch, "f_patch_size": f_patch, "img_tokens": img_tokens,
        "steps": args.steps, "seed": args.seed, "t": float(t[0]),
        "t_scale": cfg["t_scale"], "norm_eps": cfg["norm_eps"],
        "axes_dims": cfg["axes_dims"], "axes_lens": cfg["axes_lens"],
        "rope_theta": cfg["rope_theta"], "cap_feat_dim": cfg["cap_feat_dim"],
        "tensors": {},
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
        print(f"  {name:22s} {str(list(tensor.shape)):22s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}")

    print("inputs and head:")
    dump("latent", latent)
    dump("cap_feats", cap_feats)

    with torch.no_grad():
        # --- the timestep embedding -------------------------------------
        adaln = model.t_embedder(t * model.t_scale)
        dump("adaln_input", adaln)

        # --- patchify, and the ids both streams get ---------------------
        (
            img_patches, cap_padded, img_size, img_pos_ids, cap_pos_ids,
            img_pad_mask, cap_pad_mask,
        ) = model.patchify_and_embed([latent], [cap_feats], patch, f_patch)
        dump("img_patches", img_patches[0])
        dump("img_pos_ids", img_pos_ids[0].to(torch.float32))
        dump("cap_pos_ids", cap_pos_ids[0].to(torch.float32))

        # --- x: embed, pad token, rope ----------------------------------
        x_seqlens = [len(xi) for xi in img_patches]
        x = model.all_x_embedder[f"{patch}-{f_patch}"](torch.cat(img_patches, dim=0))
        x, x_freqs, _, _, _ = model._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), img_pos_ids, img_pad_mask, model.x_pad_token, None, None
        )
        dump("x_prepared", x[0])
        dump("x_freqs_cos", torch.view_as_real(x_freqs[0])[..., 0])
        dump("x_freqs_sin", torch.view_as_real(x_freqs[0])[..., 1])

        # --- cap: embed, pad token, rope --------------------------------
        cap_seqlens = [len(ci) for ci in cap_padded]
        cap = model.cap_embedder(torch.cat(cap_padded, dim=0))
        cap, cap_freqs, _, _, _ = model._prepare_sequence(
            list(cap.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, model.cap_pad_token, None, None
        )
        dump("cap_prepared", cap[0])
        dump("cap_freqs_cos", torch.view_as_real(cap_freqs[0])[..., 0])
        dump("cap_freqs_sin", torch.view_as_real(cap_freqs[0])[..., 1])

        # --- the unified sequence: [x, cap], and its rotary table --------
        unified, unified_freqs, _, _ = model._build_unified_sequence(
            x, x_freqs, x_seqlens, None, cap, cap_freqs, cap_seqlens, None,
            None, None, None, None, False, None,
        )
        dump("unified", unified[0])
        dump("unified_freqs_cos", torch.view_as_real(unified_freqs[0])[..., 0])
        dump("unified_freqs_sin", torch.view_as_real(unified_freqs[0])[..., 1])

        # --- the tail ----------------------------------------------------
        # The blocks are not here, so the tail is fed a tensor of its own
        # rather than their output: what is under test is the final layer and
        # the unpatchify, not the stack that has its own oracle.
        stream = torch.randn_like(unified)
        dump("stream", stream[0])
        final = model.all_final_layer[f"{patch}-{f_patch}"](stream, c=adaln)
        dump("final", final[0])
        out = model.unpatchify(list(final.unbind(dim=0)), img_size, patch, f_patch, None)
        dump("unpatchified", out[0])

    # --- the scheduler ---------------------------------------------------
    # FlowMatchEulerDiscrete under the pipeline's own call: the sigmas are
    # linspace(1, 1/N, N), shifted by the config's 3.0, with a terminal zero
    # appended. The pipeline then hands the transformer (1000 - timestep)/1000
    # and adds dt * (-model_out) to the latents.
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler)
    sigmas_in = torch.linspace(1.0, 1 / args.steps, args.steps).tolist()
    sched.set_timesteps(args.steps, sigmas=sigmas_in, mu=None)
    dump("sigmas", sched.sigmas)
    dump("timesteps", sched.timesteps)
    dump("model_t", (1000 - sched.timesteps.float()) / 1000)

    # One Euler step, over a tensor the Go side can reproduce exactly.
    sched.set_begin_index(0)
    sample = torch.arange(32, dtype=torch.float32) / 8 - 2
    model_out = torch.cos(torch.arange(32, dtype=torch.float32))
    dump("step_sample", sample)
    dump("step_model_out", model_out)
    dump("step_result", sched.step(model_out, sched.timesteps[0], sample, return_dict=False)[0])

    print("\nscheduler:", " ".join(f"{s:.5f}" for s in sched.sigmas.tolist()))
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
