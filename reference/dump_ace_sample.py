"""ACE-Step's sample mode on the host side, for MUSIC.md A12.

Sample mode ("simple"/"inspiration" mode: `sample_query` in the API) has the
5 Hz LM write a whole song from a description. Around the LM pass (which
dump_ace_lm.py records) upstream does three things in Python that a port
must match exactly:

* ``parse_description_hints``: the query's language (first match of a
  word list, in dict order) and whether it asks for an instrumental;
* ``build_formatted_prompt_for_inspiration``: the chat prompt, through the
  tokenizer's own chat template;
* ``_extract_lyrics_from_output`` and create_sample's conversion of the
  parsed CoT (bpm to int, duration to float, "N/A" to empty).

This dumps all three over fuzzed inputs to reference/out/acesample.

    .venv-acestep/bin/python reference/dump_ace_sample.py
"""

import json
import os
import random
import sys

sys.path.insert(0, "models/ACE-Step-1.5-src")

from transformers import AutoTokenizer  # noqa: E402

from acestep.api.server_utils import parse_description_hints  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402

OUT = "reference/out/acesample"
LM = "models/acestep-5Hz-lm-4B"

QUERIES = [
    "", "a soft Bengali love song for a quiet evening", "Instrumental jazz", "piano solo", "solo",
    "a song in english", "EN pop", "en", "french café accordion", "français chanson",
    "Deutsch rap, hard", "a pure music meditation", "spanish, flamenco", "es: reggaeton", "k-pop idol song",
    "korean ballad (한국어)", "中文 流行", "mandarin rock", "日本語のアニメソング", "an italian opera aria",
    "portuguese bossa nova!", "русский рок", "hindi bollywood", "arabic oud", "thai pop", "vietnamese",
    "indonesian dangdut", "turkish", "dutch", "polish disco polo", "no language", "it's a party",
    "pt.", "ar;", "the end? th", "zh-cn", "Solo guitar solo", "  Instrumental  ", "a\tspanish\nsong",
    "piano SOLO", "pure instrument", "english and japanese", "japanese and english", "de", "hi there",
    "hi", "vi", "id card", "tr", "nl", "pl", "ja", "ko", "ru", "bn", "Bn!", "(fr)", "[de]",
]

OUTPUTS = [
    "", "<think>\nbpm: 90\n</think>", "<think>\nbpm: 90\n</think>\n\n# Lyric\n[Verse]\nla la\n<|im_end|>",
    "<think>\n</think>\n\n[Verse]\nhello<|im_end|>\n", "no think tag here\n[Verse]\nx",
    "<think>x</think>   \n  # Lyrics\n[Chorus]\nyeah<|im_end|>", "<think>x</think># lyric\nA\n",
    "<think>x</think>#Lyri\nB", "<think>x</think>#  LYRIC  \n\nC<|im_end|>", "<think>x</think>\n# Lyric[Verse]",
    "<think>x</think>\n# Lyric\n\n\n<|im_end|>", "<think>x</think></think>\nsecond<|im_end|>",
    "<think>x</think>\n# Lyri|\nD", "<think>x</think>\n# Lyris\nE", "<think>x</think>\n#　Lyric　\nF",
    "<think>x</think>\n    G <|im_end|>  \n", "<think>x</think>\n# Lyric\n[Instrumental]<|im_end|>",
    "<think>x</think><|im_end|>", "<think>x</think>\nH<|im_end|>tail", "<think>x</think>\nI<|im_end|><|im_end|>",
]

METADATA = [
    {}, {"bpm": 120, "duration": 180, "keyscale": "A minor", "timesignature": "4", "language": "en", "caption": "c"},
    {"bpm": "N/A", "duration": "N/A", "keyscale": "N/A", "timesignature": "N/A", "language": "N/A"},
    {"bpm": "", "duration": ""}, {"bpm": "12x", "duration": "3.5"}, {"bpm": 300, "duration": 600},
    {"language": "ja", "genres": "j-pop"}, {"keyscale": "F# major", "timesignature": "6"},
]


def convert(metadata, instrumental):
    """create_sample's field conversion (inference.py), verbatim."""
    caption = metadata.get('caption', '')
    lyrics = metadata.get('lyrics', '')
    keyscale = metadata.get('keyscale', '')
    language = metadata.get('language', metadata.get('vocal_language', ''))
    timesignature = metadata.get('timesignature', '')
    is_instrumental = metadata.get('instrumental', instrumental)
    bpm = None
    bpm_value = metadata.get('bpm')
    if bpm_value is not None and bpm_value != 'N/A' and bpm_value != '':
        try:
            bpm = int(bpm_value)
        except (ValueError, TypeError):
            pass
    duration = None
    duration_value = metadata.get('duration')
    if duration_value is not None and duration_value != 'N/A' and duration_value != '':
        try:
            duration = float(duration_value)
        except (ValueError, TypeError):
            pass
    if keyscale == 'N/A':
        keyscale = ''
    if language == 'N/A':
        language = ''
    if timesignature == 'N/A':
        timesignature = ''
    return {"caption": caption, "lyrics": lyrics, "bpm": bpm, "duration": duration, "keyscale": keyscale,
            "language": language, "timesignature": timesignature, "instrumental": is_instrumental}


def main():
    tok = AutoTokenizer.from_pretrained(LM)
    h = LLMHandler.__new__(LLMHandler)
    h.llm_tokenizer = tok
    rng = random.Random(7)
    words = ["english", "japanese", "solo", "instrumental", "pure music", "en", "de", "zh", "中文", "rock",
             "a", "song", ",", ".", "!", " ", "\n", "hindi", "arabic", "español", "português"]
    queries = list(QUERIES)
    for _ in range(300):
        queries.append(" ".join(rng.choice(words) for _ in range(rng.randint(1, 6))))
    out = {"hints": [], "prompts": [], "lyrics": [], "convert": []}
    for q in queries:
        lang, inst = parse_description_hints(q)
        out["hints"].append({"query": q, "language": lang, "instrumental": inst})
    for q in QUERIES[:12] + ["NO USER INPUT"]:
        for inst in (False, True):
            p = h.build_formatted_prompt_for_inspiration(q, instrumental=inst)
            out["prompts"].append({"query": q, "instrumental": inst, "prompt": p,
                                   "ids": tok.encode(p, add_special_tokens=False)})
    for o in OUTPUTS:
        out["lyrics"].append({"text": o, "lyrics": LLMHandler._extract_lyrics_from_output(h, o)})
    for m in METADATA:
        for inst in (False, True):
            out["convert"].append({"metadata": {k: str(v) for k, v in m.items()}, "instrumental": inst,
                                   "result": convert({k: str(v) for k, v in m.items()}, inst)})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sample.json"), "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"{len(out['hints'])} hints, {len(out['prompts'])} prompts, {len(out['lyrics'])} lyric extractions, "
          f"{len(out['convert'])} conversions")


if __name__ == "__main__":
    main()
