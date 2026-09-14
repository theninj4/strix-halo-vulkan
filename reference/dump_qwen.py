"""Dump a reference run of Z-Image's Qwen3-4B text encoder, for stage 5.

The oracle is transformers' own Qwen3Model in fp32 on CPU, loaded from the
checkpoint's text_encoder/. What the pipeline actually wants out of it is
narrow -- `hidden_states[-2]`, the output of decoder layer 34 of 36, with no
final norm and no lm_head -- so that is what the Go side has to reproduce,
and everything else dumped here exists to find out *where* it stopped
reproducing it.

Three things this dump settles rather than assumes:

  * `hidden_states[-2]` really is layer 34's output. transformers v5 captures
    hidden states with a forward hook on the decoder layer instead of the
    v4 list, so the tuple's last entry is no longer the normed one; the
    manifest records the identity against a directly captured layer output.
  * Right padding to 512 does not change the real tokens' hidden states.
    The pipeline pads, runs, then masks the padding back out; the Go side
    skips the padding entirely. `pad_gap` in the manifest is the measured
    difference between the two.
  * The layer-0 internals are recomputed from the same weights, so a port
    can be walked stage by stage rather than bisected from the output.

    .venv/bin/python reference/dump_qwen.py
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, Qwen3Model

PROMPT = "a cat sitting on a windowsill at golden hour, 85mm lens"
PROMPT2 = "一只猫坐在窗台上，黄昏的光线"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="models/Z-Image-Turbo/text_encoder")
    ap.add_argument("--tokenizer", default="models/Z-Image-Turbo/tokenizer")
    ap.add_argument("--out", default="reference/out/qwen")
    ap.add_argument("--layer", type=int, default=0, help="layer whose internals are dumped")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = Qwen3Model.from_pretrained(
        args.encoder, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = model.config
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    rendered = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    ids = tok(rendered, return_tensors="pt").input_ids
    T = ids.shape[1]
    print(f"{T} tokens: {rendered!r}")

    manifest = {
        "prompt": PROMPT, "rendered": rendered, "ids": ids[0].tolist(), "seq": T,
        "hidden_size": cfg.hidden_size, "layers": cfg.num_hidden_layers,
        "heads": heads, "kv_heads": kv_heads, "head_dim": hd,
        "intermediate_size": cfg.intermediate_size,
        "rms_eps": cfg.rms_norm_eps, "rope_theta": cfg.rope_parameters["rope_theta"],
        "dump_layer": args.layer, "tensors": {},
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
        print(f"  {name:26s} {str(list(t.shape)):20s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}")

    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states
    print(f"hidden_states: {len(hs)} entries for {cfg.num_hidden_layers} layers")

    # What hidden_states[-2] is, checked rather than believed: run the first
    # 35 layers by hand and require the answer to be the same tensor.
    with torch.no_grad():
        pos = torch.arange(T).unsqueeze(0)
        h = model.embed_tokens(ids)
        cos, sin = model.rotary_emb(h, pos)
        mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
        for i in range(cfg.num_hidden_layers - 1):
            h = model.layers[i](h, attention_mask=mask, position_embeddings=(cos, sin))
    manifest["hidden_states_len"] = len(hs)
    manifest["minus2_is_layer"] = cfg.num_hidden_layers - 2  # 0-based index of the last layer run
    manifest["minus2_gap"] = float((h - hs[-2]).abs().max())
    print(f"hidden_states[-2] vs 35 layers by hand: {manifest['minus2_gap']:.3g}")

    # What the pipeline does: pad to 512, run, mask the padding back out.
    with torch.no_grad():
        padded = tok([rendered], padding="max_length", max_length=512,
                     truncation=True, return_tensors="pt")
        pm = padded.attention_mask.bool()
        pe = model(input_ids=padded.input_ids, attention_mask=pm,
                   output_hidden_states=True).hidden_states[-2]
        pipeline_out = pe[0][pm[0]]
    manifest["pad_gap"] = float((pipeline_out - hs[-2][0]).abs().max())
    manifest["pad_rms"] = float(hs[-2][0].pow(2).mean().sqrt())
    print(f"padded-to-512 vs unpadded: max abs {manifest['pad_gap']:.3g} "
          f"against rms {manifest['pad_rms']:.4g}")

    # Layer internals, recomputed from the same weights.
    layer = model.layers[args.layer]
    a = layer.self_attn
    with torch.no_grad():
        x = hs[args.layer]
        normed = layer.input_layernorm(x)
        q = a.q_proj(normed).view(1, T, heads, hd)
        k = a.k_proj(normed).view(1, T, kv_heads, hd)
        v = a.v_proj(normed).view(1, T, kv_heads, hd)
        qn, kn = a.q_norm(q), a.k_norm(k)

        def rope(t):  # NeoX style: the halves rotate against each other
            c, s = cos.unsqueeze(2), sin.unsqueeze(2)
            half = t.shape[-1] // 2
            rot = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
            return t * c + rot * s

        qr, kr = rope(qn), rope(kn)
        # GQA: each kv head serves heads/kv_heads query heads.
        rep = heads // kv_heads
        kx = kr.repeat_interleave(rep, dim=2)
        vx = v.repeat_interleave(rep, dim=2)
        scores = torch.einsum("bthd,bshd->bhts", qr, kx) * (hd ** -0.5)
        scores = scores + torch.full((T, T), float("-inf")).triu(1)[None, None]
        probs = scores.softmax(-1, dtype=torch.float32)
        ctx = torch.einsum("bhts,bshd->bthd", probs, vx).reshape(1, T, heads * hd)
        attn_out = a.o_proj(ctx)
        resid1 = x + attn_out
        normed2 = layer.post_attention_layernorm(resid1)
        gate = torch.nn.functional.silu(layer.mlp.gate_proj(normed2))
        mlp = layer.mlp.down_proj(gate * layer.mlp.up_proj(normed2))
        layer_out = resid1 + mlp

    print("tensors:")
    dump("input_ids", ids.to(torch.float32))
    dump("cos", cos[0])
    dump("sin", sin[0])
    dump("embeddings", hs[0][0])
    dump("input_layernorm", normed[0])
    dump("q", q.reshape(T, heads * hd))
    dump("k", k.reshape(T, kv_heads * hd))
    dump("v", v.reshape(T, kv_heads * hd))
    dump("q_normed", qn.reshape(T, heads * hd))
    dump("k_normed", kn.reshape(T, kv_heads * hd))
    dump("q_roped", qr.reshape(T, heads * hd))
    dump("k_roped", kr.reshape(T, kv_heads * hd))
    dump("attn_ctx", ctx[0])
    dump("attn_out", attn_out[0])
    dump("resid1", resid1[0])
    dump("post_attention_layernorm", normed2[0])
    dump("mlp", mlp[0])
    dump("layer_out", layer_out[0])
    dump("hidden_1", hs[1][0])
    dump("hidden_2", hs[2][0])
    dump("hidden_18", hs[18][0])
    dump("final", hs[-2][0])
    dump("pipeline_out", pipeline_out)

    # A second prompt, end to end only, in another script.
    rendered2 = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT2}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    ids2 = tok(rendered2, return_tensors="pt").input_ids
    with torch.no_grad():
        final2 = model(input_ids=ids2, output_hidden_states=True).hidden_states[-2][0]
    manifest["prompt2"] = PROMPT2
    manifest["rendered2"] = rendered2
    manifest["ids2"] = ids2[0].tolist()
    dump("final2", final2)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
