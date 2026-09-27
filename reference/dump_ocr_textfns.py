"""PaddleX's text post-processing on edge cases, for OCR.md O8's Go port.

Runs in .venv-paddle. The five oracle pages exercise little of it (one
table, no repetition, one formula), so this feeds PaddleX's own functions
hand-written and generated inputs and records their outputs:
truncate_repetitive_content (min_count 50 and 5000), convert_otsl_to_html,
the formula-delimiter rewrite and the title formatter.

    .venv-paddle/bin/python reference/dump_ocr_textfns.py
"""

import json
import os
import random

from paddlex.inference.common.result.converter.markdown_format_funcs import format_title
from paddlex.inference.pipelines.paddleocr_vl.uilts import convert_otsl_to_html, truncate_repetitive_content


class B:
    def __init__(self, content):
        self.content = content


def delimiters(r, label):
    # _paddleocr_vl_assemble_parsing_results, verbatim.
    if ("\\(" in r and "\\)" in r) or ("\\[" in r and "\\]" in r):
        r = r.replace("$", "")
        r = (r.replace("\\(", " $ ").replace("\\)", " $").replace("\\[\\[", "\\[")
             .replace("\\]\\]", "\\]").replace("\\[", " $$ ").replace("\\]", " $$ "))
        if label == "formula_number":
            r = r.replace("$", "")
    return r


def otsl_case(s):
    # PaddleX's parser indexes past a row on some malformed tables and raises,
    # which fails the whole page there; recorded as such.
    try:
        return {"in": s, "out": convert_otsl_to_html(s)}
    except IndexError:
        return {"in": s, "error": True}


def main():
    rnd = random.Random(7)
    words = ["表格", "abc", "数据", "x", "12.5", "Total", "  ", "é", "—", "&<>\"'"]
    trunc = [
        "短文本", "a" * 60, "ab" * 40, "xyz" * 30 + "\n", "hello world " * 12,
        "前缀内容" * 3 + "重复的短语而已" * 20, "line one\n" * 12, ("line A\n" * 9) + "line B\n",
        ("same\n" * 11) + ("other\n" * 3), "  \n  ", "x" * 49, "一二三四五六七八" * 10,
        "The model keeps saying this. " * 8, "abcdefgh" * 5 + "ijklmnop" * 6,
    ]
    for _ in range(40):
        unit = "".join(rnd.choice(words) for _ in range(rnd.randint(1, 4)))
        head = "".join(rnd.choice(words) for _ in range(rnd.randint(0, 30)))
        trunc.append(head + unit * rnd.randint(1, 30))
    tags = ["<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>"]
    otsl = [
        "<fcel>a<fcel>b<nl><fcel>c<fcel>d<nl>", "<fcel>a<lcel><nl><fcel>b<fcel>c<nl>",
        "<fcel>h<fcel>i<nl><ucel><fcel>j<nl>", "<fcel>only", "", "<nl>", "text before<fcel>a<nl>",
        "<fcel>a<fcel>b<fcel>c<nl><fcel>d<nl><fcel>e<fcel>f<fcel>g<fcel>h<nl>",
        "<fcel>x<xcel><nl><ucel><xcel><nl>", "<ecel><ecel><nl><fcel>&amp;<fcel><b><nl>",
        " <fcel> spaced <fcel>\tcell\n<nl>\n",
    ]
    for _ in range(60):
        s = ""
        for _ in range(rnd.randint(1, 25)):
            t = rnd.choice(tags)
            s += t
            if t == "<fcel>":
                s += rnd.choice(words)
        otsl.append(s)
    formulas = [("\\(x\\)", "text"), ("$\\[a+b\\]$", "display_formula"), ("\\[\\[y\\]\\]", "display_formula"),
                ("(1)", "formula_number"), ("\\(1\\)", "formula_number"), ("no delimiters $x$", "text"),
                ("\\( only open", "text"), ("\\[ half \\) mix", "text")]
    titles = ["1. Introduction", "2.3 Methods.", "（一）概述", "三、结果", "IV. Discussion", "IV Discussion",
              "Plain title", "1.2.3 Deep", "(12) paren", "第一章", "  5、带顿号", "1.\nnext line", "X.Y.",
              "10 Ten", "0. zero", "一二三四", "A title with a dot. inside."]
    out = {
        "truncate": [{"in": s, "min_count": m, "out": truncate_repetitive_content(s, min_count=m)}
                     for s in trunc for m in (50, 5000)],
        "otsl": [otsl_case(s) for s in otsl],
        "formula": [{"in": s, "label": l, "out": delimiters(s, l)} for s, l in formulas],
        "title": [{"in": s, "out": format_title(B(s))} for s in titles],
    }
    os.makedirs("reference/out/ocr_page", exist_ok=True)
    with open("reference/out/ocr_page/textfns.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
