"""eval_llm_vision: V10's structured eval, "does it see".

One fixed set of images with known answers, asked of any OpenAI-compatible
chat server, greedy and without thinking, then scored. The same client runs
against ours and against llama-server on the same GGUF and mmproj, so the
comparison is the two stacks and nothing else.

    .venv/bin/python reference/eval_llm_vision.py gen
    .venv/bin/python reference/eval_llm_vision.py run --url http://127.0.0.1:8080 --name ours
    .venv/bin/python reference/eval_llm_vision.py run --url http://127.0.0.1:8081 --name llamacpp
    .venv/bin/python reference/eval_llm_vision.py score ours llamacpp

The set is generated from a fixed seed, so `gen` rewrites identical files.
The synthetic cards are the ones a language prior cannot answer: random codes,
counts, positions on non-square canvases (a transposed grid reads them
backwards), and the difference between two pictures. The photos are the V3
fixtures.

The oracle server:

    ~/repos/llama.cpp/build/bin/llama-server --port 8081 -c 16384 -ngl 999 --jinja \
        -m models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
        --mmproj models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf --image-max-tokens 4096
"""

import argparse
import base64
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SET = ROOT / "out" / "llmvision_v10" / "set"
RESULTS = ROOT / "out" / "llmvision_v10"
PHOTOS = ROOT / "out" / "llmvision" / "decode"
SERIF = "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf"
SANS = "/usr/share/fonts/noto/NotoSans-Regular.ttf"

COLORS = {
    "red": (220, 30, 30),
    "blue": (30, 70, 220),
    "green": (30, 160, 60),
    "yellow": (235, 200, 20),
    "purple": (140, 40, 170),
    "orange": (245, 130, 20),
}

PARAGRAPH = (
    "The lighthouse keeper wrote in his log every evening at nine.\n"
    "On the fourth of March the wind turned east and the glass fell.\n"
    "Three ships passed before midnight, the last one without lights.\n"
    "He noted the time, 23:41, and the bearing, north by northwest.\n"
    "By morning the harbour was white with foam and nobody sailed."
)

SMALL = (
    "Invoice 20931 was issued to Harrow & Blythe Ltd on 7 June 2024.\n"
    "Items: 14 brass hinges at 3.20 each, 2 oak panels at 118.00 each,\n"
    "and one tin of varnish at 21.75. Delivery was charged at 9.50.\n"
    "Payment is due within 30 days to account 44-19-07 / 83152690.\n"
    "Late payment accrues interest at 1.5% per month, compounded."
)


def font(path, size):
    return ImageFont.truetype(path, size)


def text_card(text, path, size, width, pad=40, fg=(20, 20, 20), bg=(250, 248, 240)):
    f = font(path, size)
    lines = text.split("\n")
    lh = int(size * 1.45)
    img = Image.new("RGB", (width, pad * 2 + lh * len(lines)), bg)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((pad, pad + i * lh), line, font=f, fill=fg)
    return img


def shape(d, kind, cx, cy, r, color):
    if kind == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif kind == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif kind == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    else:
        raise ValueError(kind)


def scatter(rng, w, h, items, r, margin=10):
    """Places items (kind, color) at non-overlapping centres."""
    placed = []
    for kind, color in items:
        for _ in range(10000):
            cx = rng.randint(r + margin, w - r - margin)
            cy = rng.randint(r + margin, h - r - margin)
            if all((cx - x) ** 2 + (cy - y) ** 2 > (2 * r + margin) ** 2 for x, y, _, _ in placed):
                placed.append((cx, cy, kind, color))
                break
        else:
            raise RuntimeError("no room")
    return placed


def shapes_card(w, h, placed, r, bg=(255, 255, 255)):
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    for cx, cy, kind, color in placed:
        shape(d, kind, cx, cy, r, COLORS[color])
    return img


def bar_chart(values, labels_on_bars, w=900, h=600, title=None):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = font(SANS, 22)
    left, right, top, bottom = 80, w - 30, 60, h - 70
    vmax = 100
    if title:
        d.text((left, 15), title, font=font(SANS, 26), fill=(0, 0, 0))
    for v in range(0, vmax + 1, 20):
        y = bottom - (bottom - top) * v / vmax
        d.line([(left, y), (right, y)], fill=(220, 220, 220), width=1)
        d.text((left - 50, y - 13), f"{v:>3}", font=f, fill=(60, 60, 60))
    d.line([(left, top), (left, bottom), (right, bottom)], fill=(0, 0, 0), width=2)
    n = len(values)
    slot = (right - left) / n
    for i, (label, v) in enumerate(values):
        x0 = left + slot * i + slot * 0.2
        x1 = left + slot * (i + 1) - slot * 0.2
        y = bottom - (bottom - top) * v / vmax
        d.rectangle([x0, y, x1, bottom], fill=(70, 110, 190))
        tw = d.textlength(label, font=f)
        d.text(((x0 + x1) / 2 - tw / 2, bottom + 12), label, font=f, fill=(0, 0, 0))
        if labels_on_bars:
            s = str(v)
            tw = d.textlength(s, font=f)
            d.text(((x0 + x1) / 2 - tw / 2, y - 30), s, font=f, fill=(0, 0, 0))
    return img


def gen():
    rng = random.Random(10)
    SET.mkdir(parents=True, exist_ok=True)
    items = []

    def save(img, name):
        p = SET / name
        img.save(p)
        return name

    # Photos: description, and text a photo carries.
    items.append(dict(id="describe_barn", images=["photo:photo_convex-hull-barn.jpg"],
                      prompt="Describe this picture in one sentence.",
                      score="keywords", expect=[["barn"], ["mountain", "mountains", "peaks", "range", "teton"]]))
    items.append(dict(id="describe_pier", images=["photo:photo_background.jpg"],
                      prompt="Describe this picture in one sentence.",
                      score="keywords", expect=[["pier", "jetty", "boardwalk", "walkway", "dock"],
                                                ["water", "sea", "ocean", "lagoon", "bungalow", "bungalows", "villas"]]))
    items.append(dict(id="ocr_screen_date", images=["photo:photo_elarun.jpg"],
                      prompt="What date and time are shown in the login box? Answer with just the text shown.",
                      score="keywords", expect=[["15 february 2013"], ["14:52"]]))
    items.append(dict(id="ocr_screen_user", images=["photo:photo_elarun.jpg"],
                      prompt="What is typed in the username field, and what word is at the top left of the login box? "
                             "Answer as `username, word`.",
                      score="keywords", expect=[["user"], ["avci"]]))

    # OCR: prose, dense figures, and codes no prior can guess.
    items.append(dict(id="ocr_paragraph", images=[save(text_card(PARAGRAPH, SERIF, 30, 1000), "ocr_paragraph.png")],
                      prompt="Transcribe the text in this image exactly, keeping its line breaks. Output only the text.",
                      score="transcript", expect=PARAGRAPH))
    items.append(dict(id="ocr_small", images=[save(text_card(SMALL, SANS, 17, 700, pad=20), "ocr_small.png")],
                      prompt="Transcribe the text in this image exactly, keeping its line breaks. Output only the text.",
                      score="transcript", expect=SMALL))
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    codes = ["".join(rng.choice(alphabet) for _ in range(4)) + "-" + "".join(rng.choice(alphabet) for _ in range(4))
             for _ in range(6)]
    items.append(dict(id="ocr_codes", images=[save(text_card("\n".join(codes), SANS, 34, 420), "ocr_codes.png")],
                      prompt="These are six random codes. Transcribe them exactly, one per line. Output only the codes.",
                      score="transcript", expect="\n".join(codes)))

    # Charts.
    cats = ["Oslo", "Lima", "Kyiv", "Doha", "Baku", "Riga"]
    vals = [rng.randrange(10, 96) for _ in cats]
    chart = list(zip(cats, vals))
    items.append(dict(id="chart_values", images=[save(bar_chart(chart, True, title="Rainy days per year"), "chart_values.png")],
                      prompt="Read the bar chart. Reply with one line per bar, left to right, as `label: value`, and nothing else.",
                      score="pairs", expect={c: v for c, v in chart}))
    vals2 = rng.sample(range(15, 95), 6)
    chart2 = list(zip(["Ash", "Elm", "Fir", "Oak", "Yew", "Pine"], vals2))
    top = max(chart2, key=lambda p: p[1])[0]
    low = min(chart2, key=lambda p: p[1])[0]
    items.append(dict(id="chart_extremes", images=[save(bar_chart(chart2, False), "chart_extremes.png")],
                      prompt="Which bar is tallest and which is shortest? Answer as `tallest, shortest` using the labels.",
                      score="sequence", expect=[top.lower(), low.lower()]))

    # Counting, with distractors of the other colour and kind.
    for n, distract in ((4, 3), (7, 5), (12, 6)):
        placed = scatter(rng, 800, 600, [("circle", "red")] * n + [("square", "blue")] * distract
                         + [("circle", "blue")] * 2, 28)
        rng.shuffle(placed)
        items.append(dict(id=f"count_{n}", images=[save(shapes_card(800, 600, placed, 28), f"count_{n}.png")],
                          prompt="How many red circles are in this image? Answer with just the number.",
                          score="number", expect=n))

    # Positions on non-square canvases: a transposed grid reverses these.
    row = [("triangle", "green"), ("circle", "red"), ("square", "blue"), ("circle", "yellow")]
    wide = Image.new("RGB", (1400, 360), (255, 255, 255))
    d = ImageDraw.Draw(wide)
    for i, (k, c) in enumerate(row):
        shape(d, k, 175 + 350 * i, 180, 90, COLORS[c])
    items.append(dict(id="order_wide", images=[save(wide, "order_wide.png")],
                      prompt="List the shapes from left to right as `color shape`, separated by commas. Nothing else.",
                      score="sequence", expect=[f"{c} {k}" for k, c in row]))
    col = [("square", "purple"), ("triangle", "orange"), ("circle", "green")]
    tall = Image.new("RGB", (360, 1200), (255, 255, 255))
    d = ImageDraw.Draw(tall)
    for i, (k, c) in enumerate(col):
        shape(d, k, 180, 200 + 400 * i, 110, COLORS[c])
    items.append(dict(id="order_tall", images=[save(tall, "order_tall.png")],
                      prompt="List the shapes from top to bottom as `color shape`, separated by commas. Nothing else.",
                      score="sequence", expect=[f"{c} {k}" for k, c in col]))
    letters = rng.sample("ABCDEFGHJKLMNPRSTUVWXYZ", 15)
    grid = Image.new("RGB", (5 * 140, 3 * 140), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    f = font(SANS, 80)
    for r in range(3):
        for c in range(5):
            d.rectangle([c * 140, r * 140, c * 140 + 139, r * 140 + 139], outline=(0, 0, 0), width=3)
            L = letters[r * 5 + c]
            tw = d.textlength(L, font=f)
            d.text((c * 140 + 70 - tw / 2, r * 140 + 15), L, font=f, fill=(0, 0, 0))
    items.append(dict(id="grid_cell", images=[save(grid, "grid.png")],
                      prompt="The grid has 3 rows and 5 columns. What letter is in row 2, column 4 "
                             "(counting from the top left, starting at 1)? Answer with just the letter.",
                      score="exact", expect=letters[1 * 5 + 3]))
    items.append(dict(id="grid_rows", images=["grid.png"],
                      prompt="Read the grid row by row, top to bottom. Output each row's letters left to right "
                             "with no spaces, one row per line, and nothing else.",
                      score="transcript", expect="\n".join("".join(letters[r * 5:(r + 1) * 5]) for r in range(3))))

    # Two pictures: which changed, and which has more.
    base = scatter(rng, 640, 480, [("circle", "red"), ("square", "blue"), ("triangle", "green"),
                                   ("circle", "yellow"), ("square", "purple")], 45)
    changed = list(base)
    cx, cy, k, _ = changed[2]
    changed[2] = (cx, cy, k, "orange")
    items.append(dict(id="compare_change", images=[save(shapes_card(640, 480, base, 45), "compare_a.png"),
                                                   save(shapes_card(640, 480, changed, 45), "compare_b.png")],
                      prompt="These two pictures differ in exactly one shape. Which shape changed, and how? "
                             "Answer in one short sentence.",
                      score="keywords", expect=[["triangle"], ["green"], ["orange"]]))
    few = scatter(rng, 640, 480, [("circle", "blue")] * 3, 40)
    many = scatter(rng, 640, 480, [("circle", "blue")] * 6, 40)
    items.append(dict(id="compare_more", images=[save(shapes_card(640, 480, many, 40), "more_a.png"),
                                                 save(shapes_card(640, 480, few, 40), "more_b.png")],
                      prompt="How many circles are in the first picture, and how many in the second? "
                             "Answer as `first, second`.",
                      score="sequence", expect=["6", "3"]))

    (SET / "manifest.json").write_text(json.dumps(items, indent=1))
    print(f"{len(items)} items in {SET}")


def data_url(ref):
    p = PHOTOS / ref[len("photo:"):] if ref.startswith("photo:") else SET / ref
    mime = "image/jpeg" if p.suffix == ".jpg" else "image/png"
    return "data:%s;base64,%s" % (mime, base64.b64encode(p.read_bytes()).decode())


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run(url, name, model, only):
    items = json.loads((SET / "manifest.json").read_text())
    if model is None:
        with urllib.request.urlopen(url + "/v1/models") as r:
            ids = [m["id"] for m in json.load(r)["data"]]
        presets = {"chatting", "instruct", "thinking", "coding"}
        model = next((i for i in ids if i not in presets), ids[0])
    out = {}
    for it in items:
        if only and not re.search(only, it["id"]):
            continue
        content = [{"type": "image_url", "image_url": {"url": data_url(ref)}} for ref in it["images"]]
        content.append({"type": "text", "text": it["prompt"]})
        body = dict(model=model, messages=[{"role": "user", "content": content}],
                    temperature=0, top_k=1, max_tokens=400, seed=0,
                    chat_template_kwargs={"enable_thinking": False})
        t = time.time()
        r = post(url + "/v1/chat/completions", body)
        dt = time.time() - t
        text = r["choices"][0]["message"].get("content") or ""
        out[it["id"]] = dict(text=text, seconds=round(dt, 2), usage=r.get("usage"))
        print(f"{it['id']:18} {dt:6.1f}s  {text!r}"[:220], flush=True)
    path = RESULTS / f"{name}.json"
    prev = json.loads(path.read_text()) if only and path.exists() else {}
    prev.update(out)
    path.write_text(json.dumps(prev, indent=1))


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower().replace("`", "").replace("*", ""))


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_one(it, text):
    """Returns (pass, detail)."""
    t = norm(text)
    kind, exp = it["score"], it["expect"]
    if kind == "keywords":
        miss = [g for g in exp if not any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in g)]
        return not miss, "" if not miss else "missing " + "/".join(miss[0])
    if kind == "exact":
        return t.strip(" .") == norm(exp), ""
    if kind == "number":
        nums = re.findall(r"\d+", t)
        return nums[:1] == [str(exp)], ""
    if kind == "sequence":
        got = [norm(x).strip(" .") for x in re.split(r"[,\n]", text) if x.strip()]
        return got == exp, ""
    if kind == "pairs":
        got = {}
        for line in text.splitlines():
            m = re.match(r"\s*[-*]?\s*([A-Za-z]+)\s*:\s*(\d+)", line)
            if m:
                got[m.group(1)] = int(m.group(2))
        wrong = [k for k in exp if got.get(k) != exp[k]]
        return not wrong, f"{len(exp) - len(wrong)}/{len(exp)} bars"
    if kind == "transcript":
        a = [l.strip() for l in text.strip().replace("`", "").splitlines() if l.strip()]
        b = [l.strip() for l in exp.splitlines()]
        cer = edit_distance("\n".join(a), "\n".join(b)) / len(exp)
        return a == b, f"CER {cer:.3f}"
    raise ValueError(kind)


def score(names):
    items = json.loads((SET / "manifest.json").read_text())
    runs = {n: json.loads((RESULTS / f"{n}.json").read_text()) for n in names}
    print(f"{'item':18}" + "".join(f"{n:>22}" for n in names) + ("   same text" if len(names) == 2 else ""))
    totals = {n: 0 for n in names}
    same = 0
    for it in items:
        row = f"{it['id']:18}"
        for n in names:
            r = runs[n].get(it["id"])
            if r is None:
                row += f"{'-':>22}"
                continue
            ok, detail = score_one(it, r["text"])
            totals[n] += ok
            row += f"{('pass' if ok else 'FAIL') + (' ' + detail if detail else ''):>22}"
        if len(names) == 2 and all(it["id"] in runs[n] for n in names):
            s = runs[names[0]][it["id"]]["text"].strip() == runs[names[1]][it["id"]]["text"].strip()
            same += s
            row += "   " + ("yes" if s else "no")
        print(row)
    print(f"{'total':18}" + "".join(f"{f'{totals[n]}/{len(items)}':>22}" for n in names)
          + (f"   {same}/{len(items)}" if len(names) == 2 else ""))
    for n in names:
        toks = [runs[n][i["id"]]["usage"]["prompt_tokens"] for i in items
                if i["id"] in runs[n] and runs[n][i["id"]].get("usage")]
        print(f"{n}: prompt tokens {toks}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen")
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--name", required=True)
    r.add_argument("--model")
    r.add_argument("--only", help="regexp over item ids")
    s = sub.add_parser("score")
    s.add_argument("names", nargs="+")
    a = ap.parse_args()
    if a.cmd == "gen":
        gen()
    elif a.cmd == "run":
        run(a.url.rstrip("/"), a.name, a.model, a.only)
    else:
        score(a.names)


if __name__ == "__main__":
    sys.exit(main())
