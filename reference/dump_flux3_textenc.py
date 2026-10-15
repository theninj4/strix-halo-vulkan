"""Dump FLUX 3 Image Klein's text encoding, for X1 (research/flux3-vertical.md).

The oracle is diffusers' own `Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds`
(reference/flux3/diffusers/modular/encoders.py) run against the checkpoint's
Qwen3-VL-4B in fp32 on the CPU: the user-only chat template, right padding to
a multiple of 80, pad position ids pinned to 1, and hidden_states[4, 8, ...,
32] stacked per token into 20480 columns. The pad rows are DiT tokens, so
they are dumped like any other row.

Beyond the embeddings, a direct forward with the same inputs dumps the
embedding table's output, two early layers and the last tapped layer, so the
Go port walks rather than bisects. The manifest records the rendered template,
the ids, the mask and the position ids exactly as the oracle built them.

Three prompts: a short one (pads 1 window), a long one (crosses into a second
window of 80), and a CJK one.

    .venv-flux3/bin/python -I reference/dump_flux3_textenc.py   (~20 GB RSS)
"""

import json
import math
import os

import torch
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

OUT = "reference/out/flux3textenc"
MODEL = "models/flux-3-image-klein"
LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
PAD = 80
PROMPTS = {
    "short": "a red fox in fresh snow, morning light",
    "long": (
        "A wide-angle photograph of a lone American bison in a misty geothermal landscape, half hidden by white "
        "steam, a slate-blue stream and evergreens fading into the mist behind. Muted, cool palette, soft overcast "
        "light. The ground is crusted with pale mineral deposits, and a thin boardwalk runs along the left edge. "
        "In the far distance a second, smaller herd grazes on a low ridge, rendered as soft grey shapes, and a single "
        "raven circles above the steam vents against an overcast, pearl-white sky."
    ),
    "cjk": "一只猫坐在窗台上，黄昏的光线",
}
WALK = (1, 2, 32)  # hidden_states indices dumped for the walk (0 is the embedding table's output)


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    proc = AutoProcessor.from_pretrained(MODEL, subfolder="text_encoder")
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, subfolder="text_encoder", dtype=torch.float32, attn_implementation="eager"
    ).eval()
    tcfg = te.config.text_config
    manifest = {
        "layers_tapped": list(LAYERS), "pad_multiple": PAD,
        "hidden_size": tcfg.hidden_size, "layers": tcfg.num_hidden_layers,
        "heads": tcfg.num_attention_heads, "kv_heads": tcfg.num_key_value_heads,
        "head_dim": tcfg.head_dim, "intermediate_size": tcfg.intermediate_size,
        "rms_eps": tcfg.rms_norm_eps, "rope_theta": tcfg.rope_parameters["rope_theta"],
        "pad_token_id": proc.tokenizer.pad_token_id,
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
        with torch.no_grad():
            embeds = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
                te, proc, prompt, device="cpu", pad_multiple=PAD, hidden_states_layers=LAYERS
            )
        # Rebuild the oracle's inputs the way it does, to record them and to run the walk.
        text = proc.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        real = proc.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)["input_ids"].shape[1]
        length = min(math.ceil(real / PAD) * PAD, 8192)
        inp = proc.tokenizer(text, return_tensors="pt", padding="max_length", padding_side="right",
                             truncation=True, max_length=length)
        ids, mask = inp["input_ids"], inp["attention_mask"]
        pos = torch.arange(length)[None].masked_fill(mask == 0, 1)
        manifest["prompts"][label] = {
            "prompt": prompt, "rendered": text, "real": real, "padded": length,
            "ids": ids[0].tolist(), "mask": mask[0].tolist(), "positions": pos[0].tolist(),
        }
        print(f"{label}: {real} real tokens, window {length}")
        dump(label + "_prompt_embeds", embeds[0])

        with torch.no_grad():
            out = te(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        # The walk's last tapped layer must be the embeddings' last 2560 columns, exactly.
        gap = float((hs[LAYERS[-1]][0] - embeds[0, :, -tcfg.hidden_size:]).abs().max())
        manifest["prompts"][label]["walk_vs_oracle_gap"] = gap
        if label == "short":
            for i in WALK:
                dump(f"{label}_hidden{i}", hs[i][0])
        print(f"  walk vs oracle gap {gap:.3g}")

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
