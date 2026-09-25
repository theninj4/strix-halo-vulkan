"""Dump Kev-4B's fp32 reference for CLASSIFICATION.md stage K1.

The oracle is Kev's own code (reference/kev/, vendored unchanged at 9fdf054),
loaded the way `kev.serve` loads a checkpoint except for precision: fp32, the
path every number Kev publishes comes from. The base is the local
models/Qwen3.5-4B-Base snapshot at the revision head.pt pins, so the only edit
is where `Checkpoint` looks for it.

For every request in reference/kev_fixtures.json this writes, under
reference/out/kev/<name>/:

  record.json   the internal record (`api.to_record`), the packed encoding
                (`ids`, `seg`, `pos`, `decide_idx`, `opt_idx`, `opt`), the
                per-question row split (`model.rows_of`), Kev's
                probabilities, `to_answers`, `usage` and the logits each
                row produced before and after the temperature
  rows.safetensors
                per question k, the row it ran as (state + branch k):
                  row{k}.embed      token embeddings, [L, 2560]
                  row{k}.layer{i}   decoder layer i's output, every layer
                  row{k}.final      last_hidden_state (after the final norm)
                and the layer-0 (GDN) and layer-3 (attention) internals of
                row 0: in_norm, mixer, post_norm, mlp

Two paths are compared rather than assumed. Kev's `probs()` on a hybrid
backbone is its serving path (the state once, then each question on a copy
of its cache); the rows here are the plain `forward()` path (every row from
scratch). `prefix_vs_rows` is their largest difference.

    .venv/bin/python reference/dump_kev.py
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kev.api import SystemOneRequest, output_tokens, to_answers, to_record  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, rows_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="models/kev-4b")
    ap.add_argument("--base", default="models/Qwen3.5-4B-Base")
    ap.add_argument("--fixtures", default="reference/kev_fixtures.json")
    ap.add_argument("--out", default="reference/out/kev")
    ap.add_argument("--only", default="", help="comma-separated fixture names")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    ck = Checkpoint(a.run)
    want_rev = "1001bb4d826a52d1f399e183466143f4da7b741b"
    if ck.meta.base_revision != want_rev:
        raise SystemExit(f"head.pt pins {ck.meta.base_revision}, the local base is {want_rev}")
    ck.meta.base, ck.meta.base_revision = a.base, None
    tok, model = ck.load("cpu", LoadOptions())   # fp32, adapter merged, fitted temperature
    lm = model.lm
    layers = lm.layers if hasattr(lm, "layers") else lm.language_model.layers
    print(f"loaded: {type(lm).__name__}, {len(layers)} layers, T={model.head.temperature}, dtype={model.dtype}", flush=True)

    # Save the merged head alongside, for the Go side's conversion check.
    head = {"q.weight": model.head.q.weight, "q.bias": model.head.q.bias, "k.weight": model.head.k.weight, "k.bias": model.head.k.bias}

    fixtures = json.load(open(a.fixtures))
    only = set(filter(None, a.only.split(",")))
    for fx in fixtures:
        if only and fx["name"] not in only:
            continue
        out = os.path.join(a.out, fx["name"])
        os.makedirs(out, exist_ok=True)
        req = SystemOneRequest(**fx["request"])
        rec, meta = to_record(req)
        enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        probs = [p.tolist() for p in model.probs(enc)]
        answers = to_answers(probs, meta)

        S, Sp, rows = rows_of(enc)
        tensors, row_logits, row_probs = {}, [], []
        for k, r in enumerate(rows):
            ids = torch.tensor([S + r["ids"]])
            pos = torch.tensor([Sp + r["pos"]])
            cap, hooks = {}, []
            for i, layer in enumerate(layers):
                hooks.append(layer.register_forward_hook(lambda m, inp, o, i=i: cap.__setitem__(f"layer{i}", (o[0] if isinstance(o, tuple) else o)[0].clone())))
            if k == 0:
                for i in (0, 3):
                    L = layers[i]
                    mixer = L.linear_attn if hasattr(L, "linear_attn") else L.self_attn
                    hooks.append(L.input_layernorm.register_forward_hook(lambda m, inp, o, i=i: cap.__setitem__(f"l{i}.in_norm", o[0].clone())))
                    hooks.append(mixer.register_forward_hook(lambda m, inp, o, i=i: cap.__setitem__(f"l{i}.mixer", (o[0] if isinstance(o, tuple) else o)[0].clone())))
                    hooks.append(L.post_attention_layernorm.register_forward_hook(lambda m, inp, o, i=i: cap.__setitem__(f"l{i}.post_norm", o[0].clone())))
                    hooks.append(L.mlp.register_forward_hook(lambda m, inp, o, i=i: cap.__setitem__(f"l{i}.mlp", o[0].clone())))
            try:
                emb = lm.get_input_embeddings()(ids)[0]
                h = lm(input_ids=ids, position_ids=pos).last_hidden_state[0].float()
            finally:
                for hk in hooks:
                    hk.remove()
            d = len(S) + r["decide"]
            oi = [len(S) + o for o in r["opts"]]
            model.head.temperature, T = 1.0, model.head.temperature
            raw = model.head(h[d], h[torch.tensor(oi)])
            model.head.temperature = T
            row_logits.append(raw.tolist())
            row_probs.append(F.softmax(raw / T, -1).tolist())
            tensors[f"row{k}.embed"] = emb.float().contiguous()
            tensors[f"row{k}.final"] = h.contiguous()
            for name, t in cap.items():
                key = f"row{k}.{name}" if name.startswith("layer") else f"row0.{name}"
                tensors[key] = t.float().contiguous()

        gap = max(abs(x - y) for p, q in zip(probs, row_probs) for x, y in zip(p, q))
        save_file(tensors, os.path.join(out, "rows.safetensors"))
        body = {
            "name": fx["name"],
            "request": fx["request"],
            "record": rec,
            "meta": meta,
            "enc": {k: enc[k] for k in ("ids", "seg", "pos", "opt", "decide_idx", "opt_idx")},
            "state_truncated": enc["state_truncated"],
            "rows": [{"ids": S + r["ids"], "pos": Sp + r["pos"], "decide": len(S) + r["decide"], "opts": [len(S) + o for o in r["opts"]]} for r in rows],
            "state_len": len(S),
            "probs": probs,
            "row_logits_raw": row_logits,
            "row_probs": row_probs,
            "prefix_vs_rows": gap,
            "temperature": model.head.temperature,
            "answers": answers,
            "usage": {"input_tokens": len(enc["ids"]), "output_tokens": output_tokens(tok, answers)},
            "answers_json": json.dumps(answers),
        }
        with open(os.path.join(out, "record.json"), "w") as f:
            json.dump(body, f, indent=1, ensure_ascii=False)
        print(f"{fx['name']}: {len(enc['ids'])} packed tokens, state {len(S)}, {len(rows)} rows, prefix_vs_rows {gap:.2e}", flush=True)
        print("  " + json.dumps(answers), flush=True)

    save_file({k: v.detach().float().contiguous() for k, v in head.items()}, os.path.join(a.out, "head.safetensors"))


if __name__ == "__main__":
    main()
