"""PyYAML's output for ACE-Step's phase-2 CoT, fuzzed, for MUSIC.md A7c.

`LLMHandler._format_metadata_as_cot` writes the metadata the codes phase
conditions on with `yaml.dump(allow_unicode=True, sort_keys=True)`, so where
the caption's line breaks fall (plain-scalar folding at width 80) and when a
value gets quoted are part of the prompt. This writes a few thousand
metadata dicts and upstream's own CoT text for each, so `ace/lm` can check
its emitter against the real one.

    .venv-acestep/bin/python reference/dump_ace_yaml.py
"""

import json
import os
import random
import sys

sys.path.insert(0, "models/ACE-Step-1.5-src")

from acestep.llm_inference import LLMHandler  # noqa: E402

OUT = "reference/out/aceyaml"

WORDS = ("a an the upbeat synth-pop track driven by punchy four-on-the-floor drum machine beat pulsing "
         "bassline clear melodic female vocal delivers main melody over bed of bright atmospheric pads "
         "chorus lifts energy with layered harmonies catchy hooks creating vibrant nocturnal danceable "
         "atmosphere perfect for city night. lo-fi hip-hop, jazz; piano (soft) 120bpm 4/4 "
         "supercalifragilisticexpialidociousandthensomemorelettersuntilitpassesthewidthofaline").split()
ODD = ["no", "yes", "on", "Off", "null", "~", "123", "1.5", "0x1f", "2024-01-02", "=", "<<", "",
       "- dash", "-dash", "? q", ":colon", "a: b", "a:b", "a #b", "a#b", "#tag", "'quoted'", "\"dq\"",
       "it's", "[x]", "{y}", "a, b", "&amp", "*star", "!bang", "|pipe", ">gt", "%pct", "@at", "`bt",
       " lead", "trail ", "two  spaces", "tab\there", "é à ü", "中文歌词", "emoji 🎵 notes", "---doc",
       "...dots", "back\\slash", "nbsp here", "zw​sp", "line\nbreak", "ctrl\x07bell"]


def caption(rng):
    n = rng.choice([1, 3, 8, 15, 25, 40, 60, 90])
    words = [rng.choice(WORDS) for _ in range(n)]
    for _ in range(rng.choice([0, 0, 0, 1, 2])):
        words.insert(rng.randrange(len(words) + 1), rng.choice(ODD))
    sep = rng.choice([" ", " ", " ", "  "])
    return sep.join(words)


def main():
    rng = random.Random(1234)
    h = LLMHandler.__new__(LLMHandler)
    cases = []
    for s in ODD:
        cases.append({"caption": s})
        cases.append({"language": s})
    for i in range(3000):
        m = {}
        if rng.random() < 0.8:
            m["bpm"] = rng.choice([str(rng.randrange(30, 301)), "086", "0"])
        if rng.random() < 0.9:
            m["caption"] = caption(rng)
        if rng.random() < 0.8:
            m["duration"] = str(rng.randrange(10, 601))
        if rng.random() < 0.8:
            m["keyscale"] = rng.choice(["C major", "F# minor", "Bb major", "C♯ minor", "E♭ major", "A minor"])
        if rng.random() < 0.8:
            m["language"] = rng.choice(["en", "zh", "ja", "no", "es", "unknown"])
        if rng.random() < 0.8:
            m["timesignature"] = rng.choice(["4", "3", "2", "6", "4/4", "3/4", "6/8"])
        cases.append(m)
    out = [{"meta": m, "cot": h._format_metadata_as_cot(dict(m))} for m in cases]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cots.json"), "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=0)
    folded = sum("\n  " in c["cot"] for c in out)
    quoted = sum(("'" in c["cot"].split("caption:")[-1][:3]) or ('"' in c["cot"]) for c in out)
    print(f"wrote {len(out)} CoTs ({folded} folded, {quoted} quoted) to {OUT}")


if __name__ == "__main__":
    main()
