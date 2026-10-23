"""MiniMax-Music3's prompt assembly, for research/music-minimax.md N7: the
reference's _clean_caption, _normalize_lyrics and token ids (diffusers
#14456 at dafe3733, encoders.py) over a spread of captions and lyrics -- the
A14e set, N0b/N0c's, the card's example, and edge cases for every rule
(markdown, special tags, text on a tag line, inline tags, " ^ ").

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv-mm3/bin/python -I \\
        reference/dump_mm3_prompt.py

Writes reference/out/mm3prompt/cases.json.
"""

import json
import os
import re
import sys

from diffusers.modular_pipelines.minimax_music3 import encoders as E
from transformers import Qwen2Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models", "MiniMax-Music3")
OUT = os.path.join(ROOT, "reference", "out", "mm3prompt")

EDGE = [
    ("## Title\n**Bold** caption with *italic* and - a list\n- item one\n+ item two\n* item three\n---\n• bullet    four spaces",
     "[Verse] text on the tag line is dropped\nline one\n[CHORUS][Bridge]\nsing ^ split here [inline] tag\n"),
    ("A caption with <|bpm 120|> and <|key Cmaj|> and <|solo|> tags\n\n\nand blank lines",
     "[intro]\n\n[verse 1]\nHello  world\n  [Outro]  trailing"),
    ("Café crème — naïve 😀 émigré", "Ünïcödé lyrics\n[Verse]\n日本語の歌詞\n"),
]


def main():
    tok = Qwen2Tokenizer.from_pretrained(os.path.join(MODELS, "tokenizer"))
    pairs = list(EDGE)
    sets = [os.path.join(ROOT, "cmd", "acecal", "set.json"), os.path.join(ROOT, "out", "mm3-n0b", "set.json"),
            os.path.join(ROOT, "out", "mm3-n0c", "set.json"), os.path.join(ROOT, "out", "mm3-n0b", "card-set.json")]
    for p in sets:
        if not os.path.exists(p):
            continue
        for s in json.load(open(p))["songs"]:
            pairs.append((s["caption"], s["lyrics"]))
    for p in [os.path.join(ROOT, "out", "mm3-n0c", "captions.json"), os.path.join(ROOT, "out", "mm3-n0b", "captions.json")]:
        if os.path.exists(p):
            for k, c in json.load(open(p)).items():
                if not k.startswith("_"):
                    pairs.append((c, "[verse]\nla la la"))
    cases = []
    for caption, lyrics in pairs:
        if not caption.strip() or not lyrics.strip():
            continue
        cc, nl = E._clean_caption(caption), E._normalize_lyrics(lyrics)
        text = (f"{E._IM_START}{E._CAPTION_START}{cc}{E._CAPTION_END}"
                f"{E._LYRICS_START}{nl}{E._LYRICS_END}{E._IM_END}{E._AUDIO_START}")
        ids = tok(text)["input_ids"]
        cases.append({"caption": caption, "lyrics": lyrics, "clean": cc, "normal": nl, "ids": ids})
    os.makedirs(OUT, exist_ok=True)
    json.dump(cases, open(os.path.join(OUT, "cases.json"), "w"), ensure_ascii=False)
    print(f"{len(cases)} cases")


if __name__ == "__main__":
    main()
