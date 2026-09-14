"""Dump a reference Z-Image DiT block, with intermediates, for the Go/Vulkan
port to check itself against.

Only layer 0 is instantiated -- the block is a plain nn.Module, so there is
no need to materialise the 24.6 GB transformer to get a reference for one of
its 34 identical blocks.

Besides the module outputs this also dumps the attention internals computed
by hand from the same weights (q/k/v, after qk-norm, after RoPE), because
those are the steps with no module boundary to hook and the ones a port is
most likely to get subtly wrong: the RoPE pairing convention and the
per-head RMS norm.

    .venv/bin/python reference/dump_dit_block.py --seq 512
"""

import argparse
import json
import os

import torch
from diffusers.models.transformers.transformer_z_image import (
    RopeEmbedder,
    ZImageTransformerBlock,
)
from safetensors.torch import load_file
import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer", default="models/Z-Image-Turbo/transformer")
    ap.add_argument("--out", default="reference/out/dit")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--caption", type=int, default=64, help="tokens carrying text ids")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.transformer, "config.json")))
    dim, heads = cfg["dim"], cfg["n_heads"]
    head_dim = dim // heads

    torch.manual_seed(args.seed)

    block = ZImageTransformerBlock(
        layer_id=0, dim=dim, n_heads=heads, n_kv_heads=cfg["n_kv_heads"],
        norm_eps=cfg["norm_eps"], qk_norm=cfg["qk_norm"], modulation=True,
    ).to(torch.float32).eval()

    # Pull just layer 0 out of whichever shard holds it.
    want = {k: None for k in block.state_dict()}
    got = {}
    for shard in sorted(glob.glob(os.path.join(args.transformer, "*.safetensors"))):
        sd = load_file(shard)
        for k in list(want):
            src = "layers.0." + k
            if src in sd:
                got[k] = sd[src].to(torch.float32)
        del sd
    missing = [k for k in want if k not in got]
    if missing:
        raise SystemExit(f"missing weights for layer 0: {missing}")
    block.load_state_dict(got, strict=True)

    S = args.seq
    x = torch.randn(1, S, dim)
    adaln = torch.randn(1, 256)

    # Positional ids: the caption tokens advance along axis 0, the image
    # tokens tile a square grid on axes 1 and 2, which is the shape
    # patchify_and_embed produces for a text prompt followed by latents.
    side = int((S - args.caption) ** 0.5)
    if side * side != S - args.caption:
        raise SystemExit(f"{S - args.caption} image tokens is not a square grid")
    ids = torch.zeros(S, 3, dtype=torch.long)
    ids[: args.caption, 0] = torch.arange(args.caption)
    grid = torch.arange(side)
    ids[args.caption:, 1] = grid.repeat_interleave(side)
    ids[args.caption:, 2] = grid.repeat(side)

    rope = RopeEmbedder(theta=cfg["rope_theta"], axes_dims=cfg["axes_dims"], axes_lens=cfg["axes_lens"])
    freqs_cis = rope(ids)                      # [S, head_dim/2] complex64

    manifest = {
        "seq": S, "caption": args.caption, "dim": dim, "heads": heads,
        "head_dim": head_dim, "seed": args.seed,
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

    captured = {}
    hooks = [
        block.attention_norm1.register_forward_hook(lambda m, i, o: captured.__setitem__("attention_norm1", o)),
        block.attention.register_forward_hook(lambda m, i, o: captured.__setitem__("attn_out", o)),
        block.attention_norm2.register_forward_hook(lambda m, i, o: captured.__setitem__("attention_norm2", o)),
        block.ffn_norm1.register_forward_hook(lambda m, i, o: captured.__setitem__("ffn_norm1", o)),
        block.feed_forward.register_forward_hook(lambda m, i, o: captured.__setitem__("feed_forward", o)),
        block.ffn_norm2.register_forward_hook(lambda m, i, o: captured.__setitem__("ffn_norm2", o)),
    ]

    with torch.no_grad():
        out = block(x, attn_mask=None, freqs_cis=freqs_cis.unsqueeze(0), adaln_input=adaln)
    for h in hooks:
        h.remove()

    # The attention internals, recomputed from the same weights.
    with torch.no_grad():
        mod = block.adaLN_modulation(adaln)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        h_in = block.attention_norm1(x) * scale_msa
        a = block.attention
        q = a.to_q(h_in).unflatten(-1, (heads, head_dim))
        k = a.to_k(h_in).unflatten(-1, (heads, head_dim))
        v = a.to_v(h_in).unflatten(-1, (heads, head_dim))
        qn, kn = a.norm_q(q), a.norm_k(k)

        def rope_apply(t):
            c = torch.view_as_complex(t.float().reshape(*t.shape[:-1], -1, 2))
            return torch.view_as_real(c * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(3)

        qr, kr = rope_apply(qn), rope_apply(kn)
        ctx = torch.nn.functional.scaled_dot_product_attention(
            qr.transpose(1, 2), kr.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2).flatten(2, 3)

    print("tensors:")
    dump("x", x)
    dump("adaln_input", adaln)
    dump("ids", ids.to(torch.float32))
    dump("freqs_cos", torch.view_as_real(freqs_cis)[..., 0])
    dump("freqs_sin", torch.view_as_real(freqs_cis)[..., 1])
    dump("mod", mod)
    dump("scale_msa", scale_msa.squeeze(1))
    dump("gate_msa", gate_msa.squeeze(1))
    dump("scale_mlp", scale_mlp.squeeze(1))
    dump("gate_mlp", gate_mlp.squeeze(1))
    dump("attention_norm1", captured["attention_norm1"])
    dump("attn_in", h_in)
    dump("q", q.flatten(2, 3))
    dump("q_normed", qn.flatten(2, 3))
    dump("q_roped", qr.flatten(2, 3))
    dump("k_roped", kr.flatten(2, 3))
    dump("v", v.flatten(2, 3))
    dump("attn_ctx", ctx)
    dump("attn_out", captured["attn_out"])
    dump("attention_norm2", captured["attention_norm2"])
    dump("ffn_norm1", captured["ffn_norm1"])
    dump("feed_forward", captured["feed_forward"])
    dump("ffn_norm2", captured["ffn_norm2"])
    dump("out", out)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
