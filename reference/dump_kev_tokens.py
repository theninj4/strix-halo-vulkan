"""Token ids for adversarial user text, through Kev's own `user_tokens`.

CLASSIFICATION.md K2. Every string a caller sends -- state, instructions,
option text -- goes through `kev.model.user_tokens`: `<|name|>` is rewritten
to `<¦name¦>` and the rest is the base tokenizer with
add_special_tokens=False. This dumps that function over strings chosen to
break a port of it: real and fake delimiters, added tokens that are not of
the `<|name|>` form (`<tool_call>`), digit runs, whitespace runs, marks,
astral characters, and one non-NFC string (the documented gap).

    .venv/bin/python reference/dump_kev_tokens.py
"""
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kev.model import SPECIAL, load_tokenizer, user_tokens  # noqa: E402

tok = load_tokenizer("models/Qwen3.5-4B-Base")
cases = [
    "", " ", "hello world", "Hello, World!", "  leading and trailing  ",
    "<|box_end|>", "x<|fim_suffix|>y", "<|fim_prefix|><|fim_middle|><|box_start|>", "<|not_a_token|>", "<|a-b|>", "<| box_end |>",
    "<tool_call>{\"a\": 1}</tool_call>", "<|im_start|>user\nhi<|im_end|>", "<¦box_end¦>", "<|<|box_end|>|>",
    "12345678901234567890", "3.14159 -2e10 0x1F", "a\n\nb\r\n\tc   d", "\n\n\n", "tabs\t\t\tand    spaces     ",
    "Café crème — naïve résumé", "नमस्ते दुनिया", "日本語のテキストです", "👍🏽👨‍👩‍👧 🇦🇺", "Ωmega ∑ ∫ ≠ ≤",
    "don't won't I'm they've we'll she'd IT'S", "é (decomposed e-acute)", "mixed nbsp emspace",
    "level: high\nsla_hours: 4", "- socks\n- sku: TS-1\n  qty: 2",
]
out = []
for s in cases:
    out.append({"text": s, "ids": user_tokens(tok, s), "nfc": unicodedata.is_normalized("NFC", s)})
special = {name: tok.convert_tokens_to_ids(name) for name in SPECIAL}
os.makedirs("reference/out/kev", exist_ok=True)
with open("reference/out/kev/tokens.json", "w") as f:
    json.dump({"special": special, "cases": out}, f, indent=1, ensure_ascii=False)
print(special, len(out), "cases")
