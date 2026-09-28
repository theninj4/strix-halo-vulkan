"""PaddleX's figure-in-table tokens, for OCR.md O10's Go port (ocr/page/figtoken.go).

Runs in .venv-paddle. Calls PaddleX's own tokenize_figure_of_table and
untokenize_figure_of_table (paddleocr_vl/uilts.py, cv2 4.10's putText
underneath) on generated tables: noise or flat crops of many sizes, one to
twelve figures inside and outside them, boxes from 25 px to most of the
table, so the token's font scale runs from 0.2 to its ceiling and its
thickness from 1 up. Records each painted crop as raw RGB bytes and the
token map, and untokenize on strings with known, unknown and repeated
tokens.

    .venv-paddle/bin/python reference/dump_ocr_figtoken.py

writes reference/out/ocr_page/figtoken/: cases.json, and each crop before
(case_<k>.in.rgb) and after (case_<k>.rgb).
"""

import json
import os
import random

import numpy as np

from paddlex.inference.pipelines.paddleocr_vl.uilts import tokenize_figure_of_table, untokenize_figure_of_table

OUT = os.path.join(os.path.dirname(__file__), "out", "ocr_page", "figtoken")


class B:
    def __init__(self, content):
        self.content = content


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(7)
    cases = []
    for k in range(120):
        w, h = rng.randint(40, 1400), rng.randint(40, 1100)
        if k % 3 == 0:
            img = np.full((h, w, 3), rng.randint(0, 255), np.uint8)
        else:
            img = np.random.default_rng(k).integers(0, 256, (h, w, 3), dtype=np.uint8)
        tx, ty = rng.randint(0, 300), rng.randint(0, 300)
        table = [tx, ty, tx + w, ty + h]
        figures = []
        for f in range(rng.randint(1, 12)):
            inside = rng.random() < 0.8
            fw = rng.choice([rng.randint(5, 40), rng.randint(20, max(21, w))])
            fh = rng.choice([rng.randint(5, 40), rng.randint(20, max(21, h))])
            fw, fh = min(fw, w), min(fh, h)
            x0 = tx + rng.randint(0, w - fw)
            y0 = ty + rng.randint(0, h - fh)
            if not inside:
                x0 += w
            label = rng.choice(["image", "image", "seal"])
            path = f"imgs/img_in_{label}_box_{x0}_{y0}_{x0 + fw}_{y0 + fh}.jpg"
            figures.append({"path": path, "label": label, "coordinate": (x0, y0, x0 + fw, y0 + fh)})
        img.tofile(os.path.join(OUT, f"case_{k}.in.rgb"))
        out, token_map, dropped = tokenize_figure_of_table(img, table, figures)
        out = np.ascontiguousarray(out)
        out.tofile(os.path.join(OUT, f"case_{k}.rgb"))
        # untokenize: every token, an unknown one, one twice; some figures
        # are page image blocks (with and without content), some are not.
        objs = {}
        for i, f in enumerate(figures):
            if i % 3 != 2:
                objs[f["path"]] = B("" if i % 2 == 0 else f"caption {i}")
        s = "<table><tr><td>" + "</td><td>".join(list(token_map) + ["[F777]", "[F]"] + list(token_map)[:1]) + "</td></tr></table>"
        un = untokenize_figure_of_table(s, token_map, objs)
        cases.append({
            "w": w, "h": h, "table": table, "figures": figures, "token_map": token_map,
            "dropped": dropped, "untok_in": s, "untok_objs": {p: o.content for p, o in objs.items()},
            "untok_out": un, "painted": bool(token_map),
        })
    json.dump(cases, open(os.path.join(OUT, "cases.json"), "w"), indent=1)
    print(len(cases), "cases,", sum(len(c["token_map"]) for c in cases), "tokens painted")


if __name__ == "__main__":
    main()
