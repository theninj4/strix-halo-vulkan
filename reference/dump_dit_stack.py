"""Dump a reference *stack* of Z-Image DiT blocks, for stage 4c.

dump_dit_block.py is the oracle for one block; this is the oracle for several
of them applied in sequence, each with its own weights. That is the question
stage 4c adds and nothing before it could ask: 34 blocks share one set of
pipelines and one pair of activation arenas, their weights are spread over
several storage buffers, and every block is addressed by an offset -- so the
way to get this wrong is for block i to read block j's weights, or the right
weights out of the wrong bank, and every such mistake still produces a
plausible tensor.

The chain here is a straight composition: x -> block0 -> block1 -> ... It is
not the transformer's topology (the refiners run on the image and caption
streams separately, before the layers run on the two concatenated), which is
stage 6's business. It is a stack of the real modules with the real weights,
which is what a stack of the real weights has to reproduce.

The default list deliberately mixes the two kinds of block: the context
refiners are the ones diffusers builds with modulation=False, so they have no
adaLN projection, no scale on either branch input and both residuals ungated.

    .venv/bin/python reference/dump_dit_stack.py --seq 320
"""

import argparse
import json
import os

import torch
from diffusers.models.transformers.transformer_z_image import (
    RopeEmbedder,
    ZImageTransformerBlock,
)
from safetensors import safe_open


def load_prefix(transformer, index, prefix, block):
    """Fill block's state dict from the checkpoint tensors under prefix."""
    want = list(block.state_dict())
    got = {}
    for key in want:
        src = f"{prefix}.{key}"
        shard = index.get(src)
        if shard is None:
            raise SystemExit(f"{src} is not in the checkpoint")
        with safe_open(os.path.join(transformer, shard), framework="pt") as fh:
            got[key] = fh.get_tensor(src).to(torch.float32)
    block.load_state_dict(got, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer", default="models/Z-Image-Turbo/transformer")
    ap.add_argument("--out", default="reference/out/ditstack")
    ap.add_argument("--seq", type=int, default=320)
    ap.add_argument("--caption", type=int, default=64, help="tokens carrying text ids")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument(
        "--blocks",
        default="noise_refiner.0,noise_refiner.1,context_refiner.0,context_refiner.1,layers.0,layers.1",
        help="checkpoint prefixes, applied in this order",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.transformer, "config.json")))
    dim, heads = cfg["dim"], cfg["n_heads"]
    head_dim = dim // heads
    prefixes = [p for p in args.blocks.split(",") if p]

    with open(os.path.join(args.transformer, "diffusion_pytorch_model.safetensors.index.json")) as fh:
        index = json.load(fh)["weight_map"]

    torch.manual_seed(args.seed)

    S = args.seq
    x = torch.randn(1, S, dim)
    adaln = torch.randn(1, 256)

    # The same positional ids dump_dit_block.py uses: caption tokens advance
    # along axis 0, image tokens tile a square grid on axes 1 and 2.
    side = int((S - args.caption) ** 0.5)
    if side * side != S - args.caption:
        raise SystemExit(f"{S - args.caption} image tokens is not a square grid")
    ids = torch.zeros(S, 3, dtype=torch.long)
    ids[: args.caption, 0] = torch.arange(args.caption)
    grid = torch.arange(side)
    ids[args.caption:, 1] = grid.repeat_interleave(side)
    ids[args.caption:, 2] = grid.repeat(side)

    rope = RopeEmbedder(theta=cfg["rope_theta"], axes_dims=cfg["axes_dims"], axes_lens=cfg["axes_lens"])
    freqs_cis = rope(ids)

    manifest = {
        "seq": S, "caption": args.caption, "dim": dim, "heads": heads,
        "head_dim": head_dim, "seed": args.seed, "blocks": prefixes,
        "axes_dims": cfg["axes_dims"], "axes_lens": cfg["axes_lens"],
        "rope_theta": cfg["rope_theta"], "norm_eps": cfg["norm_eps"],
        "tensors": {},
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
        print(f"  {name:24s} {str(list(t.shape)):22s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}")

    print("tensors:")
    dump("x", x)
    dump("adaln_input", adaln)
    dump("ids", ids.to(torch.float32))
    dump("freqs_cos", torch.view_as_real(freqs_cis)[..., 0])
    dump("freqs_sin", torch.view_as_real(freqs_cis)[..., 1])

    # One block is materialised at a time: six of them as fp32 is 4.3 GB, and
    # the only thing wanted out of each is the tensor it produces.
    h = x
    for i, prefix in enumerate(prefixes):
        modulated = not prefix.startswith("context_refiner")
        block = ZImageTransformerBlock(
            layer_id=i, dim=dim, n_heads=heads, n_kv_heads=cfg["n_kv_heads"],
            norm_eps=cfg["norm_eps"], qk_norm=cfg["qk_norm"], modulation=modulated,
        ).to(torch.float32).eval()
        load_prefix(args.transformer, index, prefix, block)
        with torch.no_grad():
            h = block(
                h, attn_mask=None, freqs_cis=freqs_cis.unsqueeze(0),
                adaln_input=adaln if modulated else None,
            )
        dump(f"block{i}", h)
        del block

    dump("out", h)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
