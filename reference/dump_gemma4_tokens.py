#!/usr/bin/env python3
"""Rune R2's oracle for the tokenizer and the chat template: HF's ids for decisions prompts.

For every question of the decisions v1 golden file (decide/testdata/golden.json), plus a corpus of
extra states that exercise the tokenizer (Romanian, CJK, emoji, whitespace runs, byte fallback, a
literal special token inside the state, a long wikitext state), it renders the chat with
`apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` and records the text,
its ids, and whether every option label is one token *in context* (the prompt plus the label
tokenises to the prompt's ids plus one id). It also records the codebook, v1's labels past 26
options: A..Z then AA..ZZ, kept when the code is one token that decodes back to itself.

    .venv/bin/python reference/dump_gemma4_tokens.py [tokenizer dir]

The tokenizer dir defaults to models/gemma-4-tokenizer (google/gemma-4-26B-A4B-it's files); run
it again against models/rune-26b-a4b once the gated weights are down. Writes
gemma4/testdata/prompts.json.
"""
import json
import string
import sys
from itertools import product
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
tok_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models/gemma-4-tokenizer")
tok = AutoTokenizer.from_pretrained(tok_dir)

golden = json.loads((ROOT / "decide/testdata/golden.json").read_text())
SYSTEM = golden["cases"][0]["questions"][0]["system"]

wiki = (ROOT / "models/wikitext-2-raw/wiki.test.raw").read_text()[:24000]
extra_states = {
    "romanian-cjk-emoji": "Clientul a scris: «Comanda a întârziat» — 注文が遅れました 😀👍🏽 ﷽ é",
    "whitespace": "  leading\t\ttabs\n\n\n\nnewlines    and   runs    nbsp sep  ",
    "byte-fallback": "\U0001fae8 \U000e0041 ͸ \x01\x7f",
    "special-inside": "a literal <turn|> and <|channel> in the state <bos>",
    "wikitext-24k": wiki,
}


def ids_of(user, system=SYSTEM):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    # The template's own tokenisation must be encode() of its text: the Go side encodes the text.
    assert ids == tok.encode(text, add_special_tokens=False), "template ids are not encode(text)"
    return text, ids


def in_context(text, ids, labels):
    out = []
    for label in labels:
        more = tok.encode(text + label, add_special_tokens=False)
        out.append(len(more) == len(ids) + 1 and more[: len(ids)] == ids)
    return out


codebook = []
seen = set()
for code in list(string.ascii_uppercase) + ["".join(p) for p in product(string.ascii_uppercase, repeat=2)]:
    ids = tok.encode(code, add_special_tokens=False)
    if len(ids) == 1 and ids[0] not in seen and tok.decode(ids) == code:
        codebook.append(code)
        seen.add(ids[0])


def relabel(branch, labels):
    """An extended question's branch with the real codebook's codes in place of the golden file's
    synthetic ones (which keep every two-letter code)."""
    head, rest = branch.split("\nOPTIONS:\n", 1)
    lines = rest.split("\n")
    body = [lab + line[line.index(": "):] for lab, line in zip(labels, lines[:-1])]
    return head + "\nOPTIONS:\n" + "\n".join(body + lines[-1:])


prompts = []
for case in golden["cases"]:
    for q in case["questions"]:
        labels, branch_text = q["labels"], q["branch"]
        if len(labels) > 26:
            labels = codebook[: len(labels)]
            branch_text = relabel(branch_text, labels)
        user = case["state_text"] + branch_text
        text, ids = ids_of(user, q["system"])
        prompts.append({"name": f"{case['name']}/{q['name']}", "system": q["system"], "user": user,
                        "text": text, "ids": ids, "single": in_context(text, ids, labels),
                        "labels": labels})
branch = golden["cases"][0]["questions"][0]["branch"]
for name, state in extra_states.items():
    user = "SHARED STATE (JSON string):\n" + json.dumps(state, ensure_ascii=False) + "\n\n" + branch
    text, ids = ids_of(user)
    prompts.append({"name": "extra/" + name, "system": SYSTEM, "user": user, "text": text, "ids": ids,
                    "single": in_context(text, ids, ["A", "B", "C"]), "labels": ["A", "B", "C"]})
    # The raw state too, outside any template, to hit the tokenizer alone.
    prompts.append({"name": "raw/" + name, "text": state, "ids": tok.encode(state, add_special_tokens=False)})

out = ROOT / "gemma4/testdata/prompts.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"tokenizer": tok_dir, "codebook": codebook, "prompts": prompts}, ensure_ascii=False))
print(f"wrote {out}: {len(prompts)} prompts, {sum(len(p['ids']) for p in prompts)} ids, codebook {len(codebook)}; "
      f"labels not single in context: {[p['name'] for p in prompts if 'single' in p and not all(p['single'])]}")
