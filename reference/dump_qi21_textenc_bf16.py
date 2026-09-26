"""The official bf16 text encoder's prompt embeddings, for the int8 bank.

Q1 priced the fp16 GPU encoder against the released pipeline's own error
(rel 2.0, max abs 204 from fp32) with a torch probe that was never kept.
This is that probe as a dump: the same three prompts as
dump_qi21_textenc.py, through `_get_qwen_prompt_embeds`, with the text
encoder loaded at `torch_dtype=torch.bfloat16` and its default attention —
what `QwenImage21Pipeline.from_pretrained(..., torch_dtype=torch.bfloat16)`
runs. The Go int8 bank is held to *this* distance from the fp32 oracle in
reference/out/qi21textenc, as VIDEO.md M11a held the video encoder.

    .venv/bin/python reference/dump_qi21_textenc_bf16.py   (~18 GB RSS, a minute)
"""

import json
import os

import torch
from diffusers import QwenImage21Pipeline
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

OUT = "reference/out/qi21textenc_bf16"
MODEL = "models/Qwen-Image-2.1"
PROMPTS = {
    "en": "a cat sitting on a windowsill at golden hour, 85mm lens",
    "cjk": "一只猫坐在窗台上，黄昏的光线",
    "empty": "",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    proc = Qwen3VLProcessor.from_pretrained(MODEL, subfolder="processor")
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, subfolder="text_encoder", dtype=torch.bfloat16
    ).eval()
    pipe = QwenImage21Pipeline(
        scheduler=None, vae=None, text_encoder=te, processor=proc, transformer=None
    )
    manifest = {"dtype": "bfloat16", "attn": te.config._attn_implementation, "tensors": {}}
    for label, prompt in PROMPTS.items():
        with torch.no_grad():
            embeds, _, _ = pipe._get_qwen_prompt_embeds(prompt, None, device="cpu")
        t = embeds.detach().contiguous().to(torch.float32)
        name = label + "_prompt_embeds"
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:24s} {str(list(t.shape)):18s} absmax={flat.abs().max():.4g}")
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
