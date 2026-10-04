#!/usr/bin/env python3
"""Rune v3's reference for research/rune-vertical.md stage R3.

The oracle is transformers 5.17.0's Gemma 4 (`Gemma4ForConditionalGeneration`, text path only) over
the checkpoint in models/rune-26b-a4b, with the arithmetic in **fp32 and the weights stored bf16**:
a full fp32 copy is ~104 GB and does not fit beside the desktop, so each decoder layer is upcast
to fp32 just before it runs and put back after (~3 GB at a time, almost all of it the 128 experts).
The embedding is HF's own bf16 lookup, scale included (`embed_scale` is cast to bf16 first, so
the scale is 53.0, not sqrt(2816) = 53.07 -- what the model was trained with), then upcast.

The prompts are gemma4/testdata/prompts.json's, whose ids the Go tokenizer already reproduces
(R2), so the forward is compared on identical input. For each prompt this writes, under
reference/out/rune/<name>/:

  meta.json        the ids, the label ids (each label's one in-context token), the label logits
                   after the softcap, the probabilities at T = 1 and T = 2, and the full-vocab
                   top 10 at the answer position
  rows.safetensors embed [L, 2816], layer{i} [L, 2816] (decoder layer i's output) and final
                   (after the final norm); and for layers 0 (sliding) and 5 (the first full
                   layer) their internals: attn (self_attn's output), mlp (the dense MLP's),
                   router.weights / router.index (top-8), experts (the MoE's, before its norm),
                   in_norm

    .venv/bin/python reference/dump_rune.py [--only name,name] [--layers N]

--layers N stops after N layers (with no final norm or logits) for a quick check while the GPU
forward is being built.
"""

import argparse
import json
import os
import time

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ONLY = ["ticket/sentiment", "ticket/refund", "ticket/urgency", "romanian/tone",
                "many-options/product", "extra/wikitext-24k"]
INTERNAL_LAYERS = (0, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/rune-26b-a4b"))
    ap.add_argument("--prompts", default=os.path.join(ROOT, "gemma4/testdata/prompts.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reference/out/rune"))
    ap.add_argument("--only", default=",".join(DEFAULT_ONLY))
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--bf16", action="store_true",
                    help="compute in bf16 as served upstream (no per-layer upcast): the precision baseline Q8 is judged against (R7)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count())

    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True)
    model.eval()
    lm = model.model.language_model
    cfg = lm.config
    print(f"loaded in {time.time() - t0:.0f} s: {cfg.num_hidden_layers} layers, "
          f"{cfg.num_experts} experts, top {cfg.top_k_experts}", flush=True)

    # Upcast a layer for its forward and put it back after. The hidden state is fp32 throughout,
    # and every Gemma 4 op follows the hidden state's dtype (the norms' type_as, RoPE's .to).
    for layer in ([] if a.bf16 else lm.layers):
        layer.register_forward_pre_hook(lambda m, args, kwargs: (m.float(), None)[1], with_kwargs=True)
        layer.register_forward_hook(lambda m, args, out: (m.bfloat16(), None)[1])
    if not a.bf16:
        lm.norm.float()

    prompts = {p["name"]: p for p in json.load(open(a.prompts))["prompts"]}
    for name in [n for n in a.only.split(",") if n]:
        p = prompts[name]
        ids = torch.tensor([p["ids"]])
        label_ids = [tok.encode(p["text"] + lab, add_special_tokens=False)[-1] for lab in p["labels"]]
        t0 = time.time()
        rows, caps = run(lm, ids, a.layers)
        out = {}
        meta = {"name": name, "ids": p["ids"], "labels": p["labels"], "label_ids": label_ids,
                "layers_run": a.layers or cfg.num_hidden_layers}
        if not a.layers:
            final = rows["final"][-1]
            # The tied head, fp32, then Gemma's final softcap: logits/30 -> tanh -> *30.
            w = lm.embed_tokens.weight
            cap = cfg.final_logit_softcapping
            lab = (w[label_ids].float() @ final)
            lab = torch.tanh(lab / cap) * cap
            meta["label_logits"] = lab.tolist()
            for t in (1.0, 2.0):
                z = lab.double()
                pz = torch.exp((z - z.max()) / t)
                meta[f"probs_T{t:g}"] = (pz / pz.sum()).tolist()
            full = torch.cat([w[i:i + 32768].float() @ final for i in range(0, w.shape[0], 32768)])
            full = torch.tanh(full / cap) * cap
            top = torch.topk(full, 10)
            meta["top10"] = [[int(i), tok.decode([int(i)]), float(v)] for v, i in zip(top.values, top.indices)]
        out.update({k: v.contiguous() for k, v in rows.items()})
        out.update({k: v.contiguous() for k, v in caps.items()})
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        save_file(out, os.path.join(d, "rows.safetensors"))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        msg = f"{name}: {ids.shape[1]} tokens in {time.time() - t0:.0f} s"
        if "probs_T1" in meta:
            msg += f"; labels {p['labels'][:4]} p(T=1) {[round(x, 4) for x in meta['probs_T1'][:4]]}; top {meta['top10'][0][1]!r}"
        print(msg, flush=True)


def run(lm, ids, stop):
    """The text tower over one prompt, returning every layer's output and the internals of
    INTERNAL_LAYERS, all fp32 [L, hidden]."""
    caps = {}
    hooks = []
    for li in INTERNAL_LAYERS:
        layer = lm.layers[li]

        def grab(key):
            def hook(m, args, out):
                o = out[0] if isinstance(out, tuple) else out
                caps[key] = o.detach().float().reshape(-1, o.shape[-1]).clone()
            return hook

        def grab_router(key):
            def hook(m, args, out):
                _, weights, index = out
                caps[key + ".weights"] = weights.detach().float().clone()
                caps[key + ".index"] = index.detach().to(torch.int64).clone()
            return hook

        hooks.append(layer.input_layernorm.register_forward_hook(grab(f"layer{li}.in_norm")))
        hooks.append(layer.self_attn.register_forward_hook(grab(f"layer{li}.attn")))
        hooks.append(layer.mlp.register_forward_hook(grab(f"layer{li}.mlp")))
        hooks.append(layer.router.register_forward_hook(grab_router(f"layer{li}.router")))
        hooks.append(layer.experts.register_forward_hook(grab(f"layer{li}.experts")))

    rows = {}
    for i, layer in enumerate(lm.layers[: stop or len(lm.layers)]):
        def keep(m, args, out, i=i):
            o = out[0] if isinstance(out, tuple) else out
            rows[f"layer{i}"] = o.detach()[0].float().clone()
        hooks.append(layer.register_forward_hook(keep))

    embeds = lm.embed_tokens(ids)  # HF's bf16 lookup and bf16 scale
    embeds = embeds if lm.norm.weight.dtype == torch.bfloat16 else embeds.float()
    rows["embed"] = embeds[0].float().clone()
    layers = lm.layers
    if stop:
        lm.layers = layers[:stop]
    try:
        out = lm(inputs_embeds=embeds, use_cache=False)
    finally:
        lm.layers = layers
        for h in hooks:
            h.remove()
    if not stop:
        rows["final"] = out.last_hidden_state[0].float()
    return rows, caps


if __name__ == "__main__":
    main()
