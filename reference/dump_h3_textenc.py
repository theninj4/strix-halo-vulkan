"""Dump MiniMax-H3's text conditioning, for VIDEO.md M2.

MiniMax-H3 conditions on `hidden_states[50]` of its Qwen3-VL-32B conditioner:
the *unnormalised* output of decoder layer 49 of 64. For `t2va` the prompt is
tokenised verbatim, with no chat template and no special tokens. The encoder
is Qwen-Image-2.1's Qwen3-VL at four times the size, so the Go side is
qimage/textenc's config adapter over zimage/qwen. This dump is what gates it.

There are two oracles, because torch here is CPU-only and 32 B params do not
fit in fp32:

  * **official**: diffusers' own `get_qwen3vl_prompt_embeds` over the full
    bf16 checkpoint. It is what the released pipeline feeds the transformer,
    and the thing the Go port is *characterised* against (as Q1 was).
  * **fp32**: the same weights with fp32 arithmetic. Each decoder layer is
    widened to fp32 in a pre-hook and narrowed back afterwards, which is
    lossless for bf16, so the model never holds more than one fp32 layer.
    The stack is truncated to the 50 layers that matter, and the output is
    read by a hook on layer 49, never from `hidden_states`, which is post-norm
    at the end of a truncated stack. This is the oracle the port is *gated*
    against.

Three prompts: a short English one, a CJK one, and the README's full
Context-IR t2va prompt (537 tokens), which is the shape real requests take.
The short prompt also dumps the embedding and layer 0/1 outputs, so a
failing port walks rather than bisects.

    .venv/bin/python reference/dump_h3_textenc.py   (~70 GB RSS, some minutes)
"""

import json
import os
import re

import torch
from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

OUT = "reference/out/h3textenc"
MODEL = "models/MiniMax-H3"
LAYER = 50


def readme_prompt():
    """The t2va request body's prompt in scripts/readme/reproducible-768p-t2va-request.sh."""
    body = open(f"{MODEL}/scripts/readme/reproducible-768p-t2va-request.sh").read()
    body = body.split("<<'JSON'\n", 1)[1].split("\nJSON\n", 1)[0]
    return json.loads(body)["prompt"]


PROMPTS = {
    "en": "A red fox trotting through a snowy pine forest, snow crunching underfoot",
    "cjk": "一只红狐狸在雪松林中小跑，脚下的雪嘎吱作响",
    "readme": None,  # filled in main()
}


def main():
    os.makedirs(OUT, exist_ok=True)
    PROMPTS["readme"] = readme_prompt()
    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    proc = Qwen3VLProcessor.from_pretrained(MODEL, subfolder="processor")
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, subfolder="text_encoder", dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    tcfg = te.config.text_config
    manifest = {
        "layer": LAYER,
        "hidden_size": tcfg.hidden_size, "layers": tcfg.num_hidden_layers,
        "heads": tcfg.num_attention_heads, "kv_heads": tcfg.num_key_value_heads,
        "head_dim": tcfg.head_dim, "intermediate_size": tcfg.intermediate_size,
        "rms_eps": tcfg.rms_norm_eps, "prompts": {}, "tensors": {},
        "added_tokens": {t.content: i for i, t in tok.added_tokens_decoder.items() if re.fullmatch(r"</?d>", t.content)},
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

    ids = {}
    for label, prompt in PROMPTS.items():
        ids[label] = tok(prompt, add_special_tokens=False)["input_ids"]
        manifest["prompts"][label] = {"prompt": prompt, "ids": ids[label], "tokens": len(ids[label])}
        print(f"{label}: {len(ids[label])} tokens")

    # --- official: the pipeline's function, full bf16 stack -----------------
    with torch.no_grad():
        for label in PROMPTS:
            emb = get_qwen3vl_prompt_embeds(te, proc, ids[label], {}, text_encoder_layer=LAYER, dtype=torch.float32)
            dump(label + "_bf16", emb[0])

    # --- fp32: widen a layer at a time, stop after layer 49 -----------------
    text = te.model.language_model
    text.layers = text.layers[:LAYER]
    captured = {}

    def widen(module, args, kwargs):
        module.float()

    def narrow(module, args, out):
        module.to(torch.bfloat16)

    for layer in text.layers:
        layer.register_forward_pre_hook(widen, with_kwargs=True)
        layer.register_forward_hook(narrow)
    text.embed_tokens.register_forward_hook(lambda m, a, out: out.float())
    text.layers[LAYER - 1].register_forward_hook(
        lambda m, a, out: captured.update(last=(out[0] if isinstance(out, tuple) else out).clone())
    )
    walk = {}
    for i in (0, 1):
        text.layers[i].register_forward_hook(
            lambda m, a, out, i=i: walk.update({i: (out[0] if isinstance(out, tuple) else out).clone()})
        )
    with torch.no_grad():
        for label in PROMPTS:
            x = torch.tensor([ids[label]])
            out = text(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, output_hidden_states=True)
            if out.hidden_states[0].dtype != torch.float32:
                raise SystemExit(f"embeddings are {out.hidden_states[0].dtype}, the fp32 walk did not take")
            dump(label + "_fp32", captured["last"][0])
            if label == "en":
                dump("en_embed", out.hidden_states[0][0])
                dump("en_layer0", walk[0][0])
                dump("en_layer1", walk[1][0])
            gap = (manifest["tensors"][label + "_bf16"]["absmax"], float(captured["last"].abs().max()))
            manifest["prompts"][label]["absmax_bf16_fp32"] = gap

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
