"""Dump one Qwen-Image-2.1 DiT block, for Q2 — the oracle a Go block is built against.

The block is called directly (transformer_blocks[0], real weights, fp32) with
synthetic inputs, in the three shapes the denoising loop uses it:

  * **prefill** — the full joint sequence with the block-causal structure in
    segment form: one attention call per prefix segment (text segments get a
    causal triangle, image blocks attend [0, end) bidirectionally over
    themselves) plus one full call for the target rows;
  * **extract** — prefill plus the per-layer cache write: post-RoPE K and V of
    the prefix rows, dumped so the Go cache's contents can be compared, not
    just its effect;
  * **cached** — target rows only, K/V = [cached prefix ++ fresh target].

The cached output and the prefill output's target rows are the same
mathematical object computed along different tilings; the manifest records
the measured fp32 gap (~1e-6 expected — Q4's GPU gate reuses that bound).

Modulation is synthetic `[2, 16384]` (batch row + the t=0 row); the block
reads scale/gate slices and tanh-gates both residual branches. Everything the
block needs from the model — rotary tables, image ids, segments — comes from
the model's own modules with num_layers=1, so the metadata is diffusers' own.

    .venv/bin/python reference/dump_qi21_dit_block.py
"""

import json
import os

import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVLayerCache,
    QwenImage21Transformer2DModel,
    _qwenimage21_prefix_segments,
)
from safetensors import safe_open

OUT = "reference/out/qi21block"
TRANSFORMER = "models/Qwen-Image-2.1/transformer"
SEED = 11


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(os.path.join(TRANSFORMER, "config.json")))
    model_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    model_cfg["num_layers"] = 1
    model = QwenImage21Transformer2DModel(**model_cfg).to(torch.float32).eval()

    with open(os.path.join(TRANSFORMER, "diffusion_pytorch_model.safetensors.index.json")) as fh:
        index = json.load(fh)["weight_map"]
    state = {}
    for name, shard in index.items():
        if name.startswith("transformer_blocks.0."):
            with safe_open(os.path.join(TRANSFORMER, shard), framework="pt") as fh:
                state[name] = fh.get_tensor(name).to(torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected: {unexpected}")
    block_missing = [m for m in missing if m.startswith("transformer_blocks.")]
    if block_missing:
        raise SystemExit(f"block tensors not loaded: {block_missing}")
    block = model.transformer_blocks[0]

    manifest = {"seed": SEED, "cases": {}, "tensors": {}}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:26s} {str(list(t.shape)):18s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}")

    def run_case(label, text_lens, img_shapes):
        torch.manual_seed(SEED)
        target_tokens = img_shapes[-1][1] * img_shapes[-1][2]

        mask_bits = []
        for i, (_, h, w) in enumerate(img_shapes[:-1]):
            mask_bits += [False] * text_lens[i] + [True] * (h * w // 4)
        mask_bits += [False] * text_lens[len(img_shapes) - 1]
        mask_bits += [True] * (target_tokens // 4)
        img_mask = torch.tensor(mask_bits)[None]
        repeats = torch.where(img_mask, 4, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)

        rotary = model.pos_embed(img_shapes, image_pad_mask, device="cpu")
        image_ids, target_mask = model.build_token_metadata(image_pad_mask, img_shapes)
        prefix_len = int((~target_mask).sum())
        segments = _qwenimage21_prefix_segments(image_ids, prefix_len)

        S = image_pad_mask.shape[0]
        hidden = torch.randn(1, S, model.inner_dim)
        modulation = torch.randn(2, 4 * model.inner_dim)
        dump(label + "_hidden", hidden)
        dump(label + "_modulation", modulation)

        with torch.no_grad():
            cache = QwenImage21KVLayerCache()
            out_prefill = block(
                hidden_states=hidden, modulation=modulation, rotary_emb=rotary,
                target_token_mask=target_mask, layer_cache=cache,
                kv_cache_mode="extract", cache_write_slice=slice(0, prefix_len),
                segments=segments,
            )
            dump(label + "_out_prefill", out_prefill)
            dump(label + "_cache_k", cache.k)
            dump(label + "_cache_v", cache.v)

            out_cached = block(
                hidden_states=hidden[:, prefix_len:], modulation=modulation,
                rotary_emb=rotary[prefix_len:], target_token_mask=target_mask[prefix_len:],
                layer_cache=cache, kv_cache_mode="cached",
            )
            dump(label + "_out_cached", out_cached)

        gap = (out_prefill[:, prefix_len:] - out_cached).abs().max()
        manifest["cases"][label] = {
            "text_lens": text_lens, "img_shapes": img_shapes, "seq": S,
            "prefix_len": prefix_len, "target_tokens": target_tokens,
            "segments": segments, "prefill_vs_cached_target_gap": float(gap),
        }
        print(f"{label}: seq {S}, prefix {prefix_len}, segments {segments}, gap {gap:.3g}")

    print("t2i case:")
    run_case("t2i", [7], [(1, 4, 4)])
    print("edit case:")
    run_case("edit", [3, 4], [(1, 2, 4), (1, 4, 4)])

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
