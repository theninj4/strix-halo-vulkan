"""Tokenizer fixtures for PaddleOCR-VL-1.6 (OCR.md O1).

HF's fast tokenizer over a corpus that hits what the Go port must get right:
the six task prompts and the whole chat template, digits (added tokens, one
per digit), <|LOC_N|> coordinates, specials embedded in text, runs of spaces
and newlines, CJK, emoji and characters outside the vocabulary (byte
fallback), OTSL tags, LaTeX. For decoding it adds every generation in
reference/out/ocr/*/record.json and random id sequences, including lone and
broken byte-fallback runs.

Writes reference/out/ocr/tokens.json:
  encode: [{text, ids}]
  decode: [{ids, skip, text}]   skip = skip_special_tokens

    .venv/bin/python reference/dump_ocr_tokens.py
"""

import glob
import json
import random

from transformers import AutoTokenizer

TEXTS = [
    "OCR:", "Table Recognition:", "Formula Recognition:", "Chart Recognition:",
    "Seal Recognition:", "Spotting:",
    "<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>OCR:\nAssistant:\n",
    "User: ", "\nAssistant:\n", "Assistant:\n", " ", "  ", "   leading spaces", "trailing  ", "\n\n\n", "\t tab\t",
    "2024年4月,由中企蜀道集团所属四川路桥承建的孔院教学楼项目在阿斯马拉开工建设",
    "0123456789 12.5% -3.87 1e-5",
    "<|LOC_0|><|LOC_12|><|LOC_999|> text <|LOC_1000|>",
    "<fcel>CRuncover<lcel><lcel><nl><ecel><xcel><ucel>",
    "\\[\\zeta_{0}(\\nu)=-\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}\\int_{\\mu}^{\\infty}d\\omega\\]",
    "emoji 😀🧪🇨🇳 and rare 𠀀𪚥 and combining é é",
    "Hello, World! It's a test — with “quotes” and ‘single’ ones…",
    "</s><s><unk> specials in text <|IMAGE_END|>x<|image_pad|>",
    "<|TEXT_START|>normalized added<|TEXT_END|>",
    "mixed English 中文 한국어 日本語 русский العربية हिन्दी",
    "a▁b ▁▁ already-marked",
    "| col | col2 |\n|---|---|\n| 1 | 2 |",
    "",
]


def main():
    tok = AutoTokenizer.from_pretrained("models/PaddleOCR-VL-1.6")
    rng = random.Random(0)
    enc = []
    for t in TEXTS:
        enc.append({"text": t, "ids": tok.encode(t, add_special_tokens=False)})
    # Random slices of real text, to exercise merge order on arbitrary joins.
    corpus = "".join(TEXTS)
    for _ in range(200):
        i = rng.randrange(len(corpus))
        s = corpus[i:i + rng.randrange(1, 40)]
        enc.append({"text": s, "ids": tok.encode(s, add_special_tokens=False)})

    seqs = [e["ids"] for e in enc]
    for path in sorted(glob.glob("reference/out/ocr/*/record.json")):
        rec = json.load(open(path))
        seqs.append(rec["generated_ids"])
    vocab = len(tok)
    byte0 = tok.convert_tokens_to_ids("<0x00>")
    for _ in range(200):
        n = rng.randrange(1, 12)
        s = []
        for _ in range(n):
            r = rng.random()
            if r < 0.3:
                s.append(byte0 + rng.randrange(256))       # byte fallback, often broken UTF-8
            elif r < 0.4:
                s.append(rng.choice([0, 1, 2, 100273, 100295, 101305, 101306, 101298]))
            else:
                s.append(rng.randrange(vocab))
        seqs.append(s)
    # Valid multi-byte UTF-8 split into byte tokens, and a truncated one.
    for ch in ["€", "😀", "中"]:
        b = ch.encode()
        seqs.append([byte0 + x for x in b])
        seqs.append([byte0 + x for x in b[:-1]] + [tok.convert_tokens_to_ids("a")])

    dec = []
    for s in seqs:
        for skip in (True, False):
            dec.append({"ids": s, "skip": skip, "text": tok.decode(s, skip_special_tokens=skip)})
    json.dump({"encode": enc, "decode": dec}, open("reference/out/ocr/tokens.json", "w"), ensure_ascii=False)
    print(f"{len(enc)} encodes, {len(dec)} decodes")


if __name__ == "__main__":
    main()
