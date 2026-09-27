"""Teacher-forced logits of ACE-Step's 5 Hz LM in fp32 and in bf16, for MUSIC.md A10.

A10 quantises the LM's projections to int8. What that costs is priced the
way M11a taught (VIDEO.md): against upstream's own serving precision, bf16,
and not only against fp32. This script gives both at every step, not just the
seven `dump_ace_lm.py` kept.

It reads `dump_ace_lm.py`'s manifest (prompts, masks, the sampled tokens) and,
for every case and phase that sampled, runs each row's prompt plus the first
`STEPS` sampled tokens as one causal forward -- teacher forcing in a single
pass -- once with the model in fp32 and once in bf16 (upstream's CUDA dtype).
It writes the fp32 logits at each step over the phase's range (phase 1: the
text tokens, [0, CODE_BASE); phase 2: the codes, both CFG rows) and records
per step, in the manifest, the bf16 run's KL(fp32 || bf16) at temperature 0.85
(per row, and after CFG 2.0 in phase 2) and whether its argmax agrees.

In phase 1 the KL is over what the FSM allows at that step, as the oracle
recorded it: the listed ids when it kept them (at most 64), the whole range
otherwise (a free caption or lyric), and null where it forced one token --
there the logits choose nothing, and bf16 is at its worst on exactly those
steps.

Left pads are dropped, as the Go gates drop them (RoPE makes them a no-op).
The fp32 logits at the steps `dump_ace_lm.py` kept are checked against it.

    .venv-acestep/bin/python reference/dump_ace_lm_tf.py
"""

import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

REF = "reference/out/acelm"
OUT = "reference/out/acelmtf"
LM = "models/acestep-5Hz-lm-4B"
CODE_BASE = 151669
NUM_CODES = 64000
STEPS = 100
TEMP = 0.85


def kl(want, got):
    """KL(want || got) of softmax(x / TEMP), fp64, over the last axis."""
    w = torch.log_softmax(want.double() / TEMP, -1)
    g = torch.log_softmax(got.double() / TEMP, -1)
    return (w.exp() * (w - g)).sum(-1)


def main():
    with open(os.path.join(REF, "manifest.json")) as fh:
        ref = json.load(fh)

    def load(name, dtype):
        info = ref["tensors"][name]
        return np.fromfile(os.path.join(REF, name + ".bin"), dtype=dtype).reshape(info["shape"])

    work = []
    for label, case in ref["cases"].items():
        for ph, p in enumerate(case["phases"]):
            if not p.get("steps"):
                continue
            ids = load(f"{label}_p{ph}_prompt_ids", np.int32)
            mask = load(f"{label}_p{ph}_prompt_mask", np.int32)
            rows = [ids[r][mask[r] != 0].tolist() for r in range(ids.shape[0])]
            toks = load(f"{label}_p{ph}_tokens", np.int32).tolist()
            work.append((label, ph, rows, toks[:STEPS], p["steps"]))

    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count())
    model = AutoModelForCausalLM.from_pretrained(LM, torch_dtype=torch.float32).eval()

    def run(rows, toks, lo, hi):
        """[steps][rows][hi-lo] logits: step i is after the prompt and toks[:i]."""
        out = []
        for r in rows:
            seq = torch.tensor([r + toks], dtype=torch.long)
            lg = model(input_ids=seq).logits[0, len(r) - 1:, lo:hi].float()
            out.append(lg)
        return torch.stack(out, 1)

    fp32 = {}
    for label, ph, rows, toks, _ in work:
        lo, hi = (CODE_BASE, CODE_BASE + NUM_CODES) if len(rows) == 2 else (0, CODE_BASE)
        lg = run(rows, toks, lo, hi)
        fp32[(label, ph)] = lg
        print(f"{label} p{ph}: fp32 {tuple(lg.shape)}", flush=True)
    model = model.to(torch.bfloat16)

    manifest = {"cases": {}, "tensors": {}}
    for label, ph, rows, toks, steps in work:
        codes = len(rows) == 2
        lo, hi = (CODE_BASE, CODE_BASE + NUM_CODES) if codes else (0, CODE_BASE)
        f = fp32[(label, ph)]
        b = run(rows, toks, lo, hi)
        name = f"{label}_p{ph}_logits"
        f.numpy().astype(np.float32).tofile(os.path.join(OUT, name + ".bin"))
        manifest["tensors"][name] = {"shape": list(f.shape), "dtype": "float32"}

        # the fp32 forward against dump_ace_lm.py's incremental one
        worst = 0.0
        for s in (0, 1, 2, 3, 10, 50, 100):
            n = f"{label}_p{ph}_logits{s}"
            if s >= f.shape[0] or n not in ref["tensors"]:
                continue
            want = torch.from_numpy(load(n, np.float32).reshape(len(rows), -1)[:, lo:hi].copy())
            worst = max(worst, float(((f[s] - want).norm() / want.norm())))
        rec = {"rows": len(rows), "lo": lo, "hi": hi, "steps": f.shape[0], "fp32_vs_incremental_rel": worst,
               "bf16_kl": kl(f, b).tolist(),
               "bf16_argmax": (f.argmax(-1) == b.argmax(-1)).tolist()}
        if not codes:
            fk, fa = [], []
            for i in range(f.shape[0]):
                st = steps[i]
                if st["fsm_allowed"] == 1:
                    fk.append(None)
                    fa.append(None)
                    continue
                ids = torch.tensor(st["fsm_ids"]) - lo if st.get("fsm_ids") else torch.arange(hi - lo)
                fi, bi = f[i, 0, ids], b[i, 0, ids]
                fk.append(float(kl(fi, bi)))
                fa.append(bool(fi.argmax() == bi.argmax()))
            rec["bf16_fsm_kl"], rec["bf16_fsm_argmax"] = fk, fa
        if codes:
            fc = f[:, 1] + 2.0 * (f[:, 0] - f[:, 1])
            bc = b[:, 1] + 2.0 * (b[:, 0] - b[:, 1])
            rec["bf16_cfg_kl"] = kl(fc, bc).tolist()
            rec["bf16_cfg_argmax"] = (fc.argmax(-1) == bc.argmax(-1)).tolist()
        manifest["cases"][f"{label}_p{ph}"] = rec
        kls = kl(f, b)
        if not codes:
            free = torch.tensor([v for v in rec["bf16_fsm_kl"] if v is not None])
            print(f"{label} p{ph}: {len(free)} unforced steps; bf16 KL over the FSM's set mean {free.mean():.2e} "
                  f"max {free.max():.2e}", flush=True)
        print(f"{label} p{ph}: fp32 vs incremental rel {worst:.2e}; bf16 KL mean {kls.mean():.2e} max {kls.max():.2e}; "
              f"argmax {int(rec['bf16_argmax'].count([True] * len(rows)))}/{f.shape[0]} steps", flush=True)

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
