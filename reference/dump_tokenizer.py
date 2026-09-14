"""Dump reference tokenizations for zimage/tokenizer, for stage 5.

The oracle is the same tokenizer the pipeline builds: HF's fast Qwen2
tokenizer, loaded from the checkpoint's own tokenizer/ directory. Each case
is dumped twice over -- the raw text and the chat-template rendering of it --
because the Go side has to reproduce both the BPE and the template, and a
mistake in either one produces a plausible list of ids.

The corpus is chosen for the places byte-level BPE goes wrong: the
pre-tokenizer's alternatives (contractions, digit-at-a-time, runs of
punctuation, a whitespace run before a word, a whitespace run before a
newline, a whitespace run at the end of the input), multi-byte scripts,
astral-plane emoji, and an added token written out in the middle of ordinary
text.

    .venv/bin/python reference/dump_tokenizer.py
"""

import argparse
import json
import os

from transformers import AutoTokenizer

CASES = [
    ("ascii", "a cat"),
    ("empty", ""),
    ("contractions", "it's Bob's, they're, I'll, we've, don't, I'd, IT'S"),
    ("digits", "2024-11-30 costs $1,234.56 or 007"),
    ("punctuation", "wow!!! ...what?? (really) [yes] {no} <tag/> ---"),
    ("spaces", "a  b   c    d"),
    ("space_before_newline", "a  \n  b\n\n\nc"),
    ("trailing_space", "a cat   "),
    ("tabs", "a\tb\t\tc"),
    ("newline_only", "\n\n"),
    ("chinese", "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。"),
    ("chinese_long", "画面巧妙地将文字概念视觉化：一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。"),
    ("japanese", "猫が窓辺で寝ている、柔らかい光。"),
    ("emoji", "a cat 🐈 on a 🛋️ at 5pm 👨‍👩‍👧‍👦"),
    ("mixed", "A 猫 named Bob, 2 years old — très bien! Ω≈ç√"),
    ("special_inline", "before <|im_end|> after <|endoftext|>tail"),
    ("cyrillic", "Кот сидит на окне"),
    ("arabic", "قطة على النافذة"),
    ("long_prompt", "A photorealistic portrait of an elderly fisherman mending his nets at dawn, "
                    "golden hour light, 85mm lens, shallow depth of field, salt-crusted hands, "
                    "weathered face, the harbour out of focus behind him."),
    ("newline_prompt", "line one\nline two\n\nline four"),
]

# The one case the Go tokenizer is expected to get wrong: "é" written as e +
# U+0301 rather than U+00E9. tokenizer.json asks for NFC and Go's standard
# library has no NFC, so this is dumped to measure the gap rather than to
# pass.
NFC_CASE = ("nfc_decomposed", "café au lait")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="models/Z-Image-Turbo/tokenizer")
    ap.add_argument("--out", default="reference/out/tokenizer")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def case(name, text):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        return {
            "name": name,
            "text": text,
            "ids": tok(text).input_ids,
            "rendered": rendered,
            "prompt_ids": tok(rendered).input_ids,
        }

    out = {
        "tokenizer": args.tokenizer,
        "vocab_size": len(tok),
        "cases": [case(n, t) for n, t in CASES],
        "nfc": case(*NFC_CASE),
    }
    path = os.path.join(args.out, "cases.json")
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    for c in out["cases"]:
        print(f"  {c['name']:22s} {len(c['ids']):4d} ids  {len(c['prompt_ids']):4d} with template")
    print(f"\nwrote {len(out['cases'])} cases to {path}")


if __name__ == "__main__":
    main()
