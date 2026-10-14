"""Breeze-TTS-2's prompt ids (research/tts-instruct-vertical.md B1).

For a corpus of requests over all four templates, this records what upstream's
`templates._prepare_one` gives each branch (the prompt, and the CFG negative
where the template has one): input_ids, which positions are text, and each
text segment's length. A clone's reference audio is stubbed to N frames of
zeros: only how many placeholders it lays down reaches the ids.

The corpus covers English, Chinese, the vocal events in both spellings, digits,
punctuation runs, emoji and byte fallback, a literal special token in the
text, and leading/trailing/inner whitespace (each segment is encoded,
decoded and encoded again upstream, which the Go side must agree with).

    .venv-breeze/bin/python -I reference/dump_breeze_tokens.py

Writes breeze/testdata/prompts.json.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "models/breeze-tts-src"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from breeze_infer import templates as T  # noqa: E402


class Cfg:
    num_codebooks = 16


TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Good morning everyone, and welcome back to the show! (laugh) We have got a great lineup for you today.",
    "It cost $1,234.56 on 2026-10-14 -- that's 3x more (sigh)...",
    "  leading and trailing spaces,   inner   runs  ",
    "Line one.\nLine two.\n\nTabs\there.",
    "大家好，欢迎收听今天的节目。[笑]",
    "他叹了口气[叹气]，然后说：“好吧，我们开始。”",
    "Mixed 中文 and English, with emoji 😀👍🏽 and ümlauts.",
    "A literal <ins_bos> and [S3] and <|AUDIO|> in the text.",
    "\U0001fae8 \x01 byte fallback",
    "Hi.",
]
INSTRUCTIONS = [
    "An enthusiastic radio DJ with a warm, energetic voice.",
    "一位温柔自信的年轻女性，声音清晰，语气亲切。",
    "Speak slowly with a restrained, serious tone.",
    "  a whispering, breathy voice  ",
]
REF_TEXT = "This is the reference speaker reading a short sentence."


def main():
    tok = AutoTokenizer.from_pretrained(os.path.join(REPO, "models/Breeze-TTS-2"), fix_mistral_regex=False)
    frames = {}
    T._resolve_segment_audio_codes = lambda _a, seg: torch.zeros(frames["n"], 16, dtype=torch.int16)

    def branch(segments):
        enc = T._prepare_one(tok, None, Cfg, segments)
        return {
            "ids": enc["input_ids"][0].tolist(),
            "text_mask": [int(x) for x in enc["text_ids_mask"][0].tolist()],
            "segments": enc["text_ids_len"].tolist(),
        }

    cases = []
    for i, text in enumerate(TEXTS):
        reqs = [{"text": text}]
        reqs.append({"text": text, "instruction": INSTRUCTIONS[i % len(INSTRUCTIONS)]})
        reqs.append({"text": text, "ref_text": REF_TEXT, "ref_audio_path": "stub", "frames": 3 + i})
        reqs.append({"text": text, "ref_text": REF_TEXT, "ref_audio_path": "stub", "frames": 5,
                     "instruction": INSTRUCTIONS[(i + 1) % len(INSTRUCTIONS)]})
        for r in reqs:
            r["speaker"] = "S0"
            frames["n"] = r.get("frames", 0)
            name = T.select_template_name(r)
            spec = T.get_template(name)
            c = {"template": name, "text": text, "instruction": r.get("instruction"),
                 "ref_text": r.get("ref_text"), "ref_frames": r.get("frames", 0),
                 "cond": branch(spec.build_segments(r))}
            if spec.build_negative_segments is not None:
                c["uncond"] = branch(spec.build_negative_segments(r))
            cases.append(c)
    out = os.path.join(REPO, "breeze/testdata/prompts.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"cases": cases}, fh, ensure_ascii=False, indent=1)
    print(f"{len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
