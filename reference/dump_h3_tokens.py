"""Dump MiniMax-H3's prompt tokenisation, for VIDEO.md M2 (no weights).

A t2va prompt is tokenised verbatim with `add_special_tokens=False`, under
the H3 repository's own tokenizer, which adds `<d>`/`</d>` (dialogue) and a
few other tokens to Qwen3-VL's vocabulary. The prompts cover what Context-IR
output actually contains: section labels, shot markers, timecodes, `<d>`
dialogue with a language tag, CJK and Arabic, and the README's full prompt.

    .venv/bin/python reference/dump_h3_tokens.py
"""

import json

from transformers import AutoTokenizer

MODEL = "models/MiniMax-H3"
OUT = "reference/out/h3tokens/tokens.json"


def readme_prompt():
    body = open(f"{MODEL}/scripts/readme/reproducible-768p-t2va-request.sh").read()
    body = body.split("<<'JSON'\n", 1)[1].split("\nJSON\n", 1)[0]
    return json.loads(body)["prompt"]


PROMPTS = {
    "en": "A red fox trotting through a snowy pine forest, snow crunching underfoot",
    "cjk": "一只红狐狸在雪松林中小跑，脚下的雪嘎吱作响",
    "dialogue": "[Shot 1] At 00:02.500, <Subject 1> (S1) speaks softly, <d>[English] Follow the wind, live free.</d> "
    "Then <d>[Arabic] مرحبا بالعالم</d>\noverall_soundscape: rain.\n\nnon_diegetic_music: none.",
    "readme": readme_prompt(),
}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    out = {k: {"prompt": p, "ids": tok(p, add_special_tokens=False)["input_ids"]} for k, p in PROMPTS.items()}
    for k, v in out.items():
        print(f"{k}: {len(v['ids'])} tokens")
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
