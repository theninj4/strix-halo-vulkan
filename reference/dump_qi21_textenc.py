"""Dump Qwen-Image-2.1's text encoding, for Q1.

The oracle is the pipeline's own `_get_qwen_prompt_embeds` — raw template
string (never apply_chat_template), the tokenized-system-message drop_idx,
and the forward hook that neutralizes the final RMSNorm so hidden_states[-1]
is the last decoder layer's *pre-norm* output on transformers 5.x — run
against the real Qwen3-VL-8B in fp32 on CPU. The pipeline is instantiated
with only the text encoder and processor; vae/transformer/scheduler are None
and never touched on this path.

Beyond the pipeline outputs, a direct forward dumps every decoder layer's
hidden state so the Go port walks rather than bisects, and the manifest
settles Q-o5 (IMAGE.md): the model's own rotary cos/sin for a text-only
prompt is compared against a plain NeoX RoPE table built from theta alone —
`mrope_gap` is the measured difference, expected 0, because interleaved
mrope with equal section positions *is* plain RoPE.

Three prompts: English, CJK, and empty (the pipeline turns "" into " " —
Qwen has no BOS, an empty string leaves the encoder nothing to read).

    .venv/bin/python reference/dump_qi21_textenc.py   (~34 GB RSS, minutes)
"""

import json
import os

import torch
from diffusers import QwenImage21Pipeline
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

OUT = "reference/out/qi21textenc"
MODEL = "models/Qwen-Image-2.1"
PROMPTS = {
    "en": "a cat sitting on a windowsill at golden hour, 85mm lens",
    "cjk": "一只猫坐在窗台上，黄昏的光线",
    "empty": "",
}
WALK_LAYER = 0  # layer whose output the manifest calls out for the port's first gate


def main():
    os.makedirs(OUT, exist_ok=True)
    proc = Qwen3VLProcessor.from_pretrained(MODEL, subfolder="processor")
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, subfolder="text_encoder", dtype=torch.float32, attn_implementation="eager"
    ).eval()
    pipe = QwenImage21Pipeline(
        scheduler=None, vae=None, text_encoder=te, processor=proc, transformer=None
    )

    tcfg = te.config.text_config
    manifest = {
        "template_t2i": pipe.prompt_template_t2i,
        "sys_prompt": pipe.sys_prompt,
        "drop_idx": pipe._drop_idx,
        "hidden_size": tcfg.hidden_size, "layers": tcfg.num_hidden_layers,
        "heads": tcfg.num_attention_heads, "kv_heads": tcfg.num_key_value_heads,
        "head_dim": tcfg.head_dim, "intermediate_size": tcfg.intermediate_size,
        "rms_eps": tcfg.rms_norm_eps,
        "rope_theta": tcfg.rope_parameters["rope_theta"],
        "mrope_section": tcfg.rope_parameters.get("mrope_section"),
        "mrope_interleaved": tcfg.rope_parameters.get("mrope_interleaved"),
        "walk_layer": WALK_LAYER,
        "prompts": {}, "tensors": {},
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
        print(f"  {name:24s} {str(list(t.shape)):18s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}")

    for label, prompt in PROMPTS.items():
        # --- the pipeline's own path: what the DiT is fed -------------------
        with torch.no_grad():
            embeds, mask, pad_mask = pipe._get_qwen_prompt_embeds(prompt, None, device="cpu")
        rendered = pipe.prompt_template_t2i.format(prompt if prompt else " ")
        ids = proc(text=[rendered], return_tensors="pt").input_ids
        manifest["prompts"][label] = {
            "prompt": prompt, "rendered": rendered, "ids": ids[0].tolist(),
            "seq": ids.shape[1], "embed_tokens": embeds.shape[1],
        }
        print(f"{label}: {ids.shape[1]} tokens, {embeds.shape[1]} after drop")
        dump(label + "_prompt_embeds", embeds)
        dump(label + "_ids", ids.to(torch.float32))

        # --- the layer walk: every decoder layer's output -------------------
        if label == "en":
            text_model = getattr(te.model, "language_model", te.model)
            captured = {}
            rope_handle = text_model.rotary_emb.register_forward_hook(
                lambda m, a, out: captured.update(cos=out[0], sin=out[1])
            )
            norm_handle = text_model.norm.register_forward_hook(lambda m, a, out: a[0])
            with torch.no_grad():
                outputs = te(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
            rope_handle.remove()
            norm_handle.remove()
            hs = outputs.hidden_states
            dump("en_embed_tokens", hs[0])
            for i in (WALK_LAYER, WALK_LAYER + 1):
                dump(f"en_layer{i}_out", hs[i + 1])
            dump("en_last_prenorm", hs[-1])

            # hidden_states[-1] with the hook == the pipeline's pre-drop rows
            gap = (hs[-1][:, pipe._drop_idx:] - embeds).abs().max()
            manifest["prompts"]["en"]["prenorm_vs_pipeline_gap"] = float(gap)

            # --- Q-o5: text-only mrope against a plain NeoX table ----------
            cos, sin = captured["cos"], captured["sin"]
            T, hd = ids.shape[1], tcfg.head_dim
            theta = float(tcfg.rope_parameters["rope_theta"])
            inv = 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
            ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None]
            plain_cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
            plain_sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
            mrope_gap = max(
                float((cos.reshape(T, hd) - plain_cos).abs().max()),
                float((sin.reshape(T, hd) - plain_sin).abs().max()),
            )
            manifest["mrope_gap"] = mrope_gap
            manifest["rope_shapes"] = {"cos": list(cos.shape), "sin": list(sin.shape)}
            dump("en_rope_cos", cos.reshape(T, hd) if cos.numel() == T * hd else cos)
            dump("en_rope_sin", sin.reshape(T, hd) if sin.numel() == T * hd else sin)
            print(f"  prenorm-vs-pipeline gap {gap:.3g}; mrope-vs-plain-rope gap {mrope_gap:.3g}")

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
