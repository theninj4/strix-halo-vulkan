"""ACE-Step's phase-1 constrained-decoding FSM, driven through scripted CoTs,
for MUSIC.md A7c.

`MetadataConstrainedLogitsProcessor` decides what the 5 Hz LM may write in
its CoT: forced field names, prefix trees of valid bpm/duration/key/time
signature/language values, a free caption that ends when the model starts a
non-indented line, and the user's own metas injected token by token. Two of
its decisions read the logits (the argmax after a caption newline, and the
top-1 language); everything else is a function of the tokens so far.

So this runs upstream's own processor, configured as `_setup_constrained_
processor` configures it for phase 1 (single mode, stop at reasoning, genres
skipped), over synthetic logits: hash noise plus a large bonus on the token
that continues a scripted CoT. At every step it records the state, the
allowed set (its size, a checksum, and the ids when there are at most 64),
the raw argmax and the top-1 language candidate, and the token taken (the
masked argmax). A port replays the tokens and must produce the same sets.

    .venv-acestep/bin/python reference/dump_ace_fsm.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "models/ACE-Step-1.5-src")

from transformers import AutoTokenizer  # noqa: E402

from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor  # noqa: E402

OUT = "reference/out/acefsm"
LM = "models/acestep-5Hz-lm-4B"

CAP = ("A dreamy lo-fi hip-hop beat with dusty vinyl crackle, mellow Rhodes chords and a lazy\n"
       "  swing groove. Soft female humming floats over the top.")

SCENARIOS = [
    ("free", {}, f"<think>\nbpm: 128\ncaption: {CAP}\nduration: 95\nkeyscale: F# minor\nlanguage: ja\n"
                 "timesignature: 3\n</think>"),
    ("duration_given", {"duration": 45},
     "<think>\nbpm: 75\ncaption: Slow cinematic piano.\nduration: 45\nkeyscale: B♭ major\nlanguage: zh\n"
     "timesignature: 4\n</think>"),
    ("bpm_key_given", {"bpm": 100, "keyscale": "G major"},
     "<think>\nbpm: 100\ncaption: Cantopop ballad with strings.\nduration: 240\nkeyscale: G major\n"
     "language: yue\ntimesignature: 6\n</think>"),
    ("extremes", {}, "<think>\nbpm: 300\ncaption: Speedcore.\nduration: 600\nkeyscale: Ab minor\n"
                     "language: unknown\ntimesignature: 2\n</think>"),
    ("low", {"timesignature": "3"}, "<think>\nbpm: 30\ncaption: Drone.\nduration: 10\nkeyscale: E♭ major\n"
                                    "language: no\ntimesignature: 3\n</think>"),
    ("out_of_order", {}, "<think>\nbpm: 90\ncaption: Funk.\nkeyscale: C major\nlanguage: en\n"
                         "timesignature: 4\n</think>"),
    ("invalid_values", {}, "<think>\nbpm: 15\ncaption: Odd.\nduration: 5\nkeyscale: H major\nlanguage: xx\n"
                           "timesignature: 5\n</think>"),
]


def noise(step, scen, vocab):
    i = np.arange(vocab, dtype=np.uint64)
    h = (i * np.uint64(2654435761) + np.uint64(step * 40503 + scen * 9973 + 17)) % np.uint64(1 << 32)
    h = (h * np.uint64(2246822519)) % np.uint64(1 << 32)
    return (h.astype(np.float64) / float(1 << 32)).astype(np.float32)


def main():
    tok = AutoTokenizer.from_pretrained(LM)
    proc = MetadataConstrainedLogitsProcessor(tok, enabled=True, skip_genres=True)
    vocab = len(tok)
    eos = tok.eos_token_id
    lang_first = sorted(proc.language_prefix_tree[tuple()])
    out = {"vocab": vocab, "scenarios": []}
    for si, (name, user, target) in enumerate(SCENARIOS):
        proc.reset()
        proc.enabled = True
        proc.metadata_temperature = None
        proc.codes_temperature = None
        proc.set_target_duration(None)
        proc.set_user_metadata(user or None)
        proc.set_stop_at_reasoning(True)
        proc.set_skip_genres(True)
        proc.set_skip_caption(False)
        proc.set_skip_language(False)
        proc.set_generation_phase("cot")
        gen = []
        steps = []
        for step in range(700):
            scores = torch.from_numpy(noise(step, si, vocab)).unsqueeze(0)
            text = tok.decode(gen)
            if target.startswith(text) and len(text) < len(target):
                cand = tok.encode(target[len(text):], add_special_tokens=False)[0]
                scores[0, cand] += 50.0
            top = int(torch.argmax(scores[0]))
            lang_top = lang_first[int(torch.argmax(scores[0, lang_first]))]
            state = proc.state.name
            ids = torch.tensor([gen], dtype=torch.long) if gen else torch.zeros((1, 0), dtype=torch.long)
            masked = proc(ids, scores.clone())
            fin = torch.isfinite(masked[0])
            allowed = fin.nonzero().flatten()
            choice = int(torch.argmax(masked[0]))
            rec = {"state": state, "n": int(fin.sum()), "sum": int(allowed.sum()), "top": top, "lang": lang_top,
                   "tok": choice}
            if rec["n"] <= 64:
                rec["ids"] = allowed.tolist()
            steps.append(rec)
            proc.update_state(choice)
            gen.append(choice)
            if choice == eos:
                break
        text = tok.decode(gen)
        out["scenarios"].append({"name": name, "user": user, "target": target, "steps": steps, "text": text})
        print(f"{name}: {len(steps)} steps, {text!r}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fsm.json"), "w") as fh:
        json.dump(out, fh, ensure_ascii=False)


if __name__ == "__main__":
    main()
