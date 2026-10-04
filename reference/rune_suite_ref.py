#!/usr/bin/env python3
"""Rune R7's precision baseline: a suite's questions re-run through HF in bf16, as Rune is served
upstream, on the exact prompt ids our engine ran (cmd/rune -prompts), so a Q8 row and a bf16 row
differ only in the arithmetic.

For each prompt it runs the whole prompt (no shared prefix: the reference's own route), reads the
last position, takes the label rows of the tied head and Gemma's softcap, and writes a row in
cmd/rune's schema, its metadata (keys, label, variant...) copied from the Q8 row file, so
`cmd/rune -compare` reads the two side by side.

    .venv/bin/python reference/rune_suite_ref.py \\
        --prompts reference/out/rune/suites/prompts-transfer-v4-development.jsonl \\
        --rows reference/out/rune/suites/q8-transfer-v4-development.jsonl \\
        --out reference/out/rune/suites/bf16-transfer-v4-development.jsonl [--every 4]

Resumable: questions already in --out are skipped. ~55 GB of RAM; run nothing else big beside it.
"""
import argparse
import json
import math
import os
import time

import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/rune-26b-a4b"))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1, help="every k-th question, for a sample")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count())

    meta = {}
    for line in open(a.rows):
        r = json.loads(line)
        meta[(r["id"], r["question"])] = r
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["id"], r["question"]))

    model = Gemma4ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.eval()
    lm = model.model.language_model
    cap = lm.config.final_logit_softcapping
    w = lm.embed_tokens.weight

    out = open(a.out, "a")
    t0, n = time.time(), 0
    for i, line in enumerate(open(a.prompts)):
        if i % a.every:
            continue
        p = json.loads(line)
        key = (p["id"], p["question"])
        if key in done:
            continue
        h = lm(input_ids=torch.tensor([p["ids"]]), use_cache=False).last_hidden_state[0, -1]
        z = (w[p["labels"]].float() @ h.float())
        z = torch.tanh(z / cap) * cap
        zz = z.double()
        pr = torch.exp(zz - zz.max())
        pr = (pr / pr.sum()).tolist()
        r = dict(meta[key])
        r["p"], r["z"] = pr, z.tolist()
        out.write(json.dumps(r) + "\n")
        out.flush()
        n += 1
        if n % 25 == 0:
            print(f"{n} questions, {(time.time() - t0) / n:.2f} s each", flush=True)
    print(f"done: {n} questions in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
