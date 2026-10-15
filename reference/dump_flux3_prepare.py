"""Dump BFL's reference preparation, for X7's gate (research/flux3-vertical.md).

Runs reference/flux3/bfl/pipeline.py's own prepare_reference (one reference:
fit under MATCH_MAX_PIXELS, reflect-pad to the 16 grid, centred, and the box
to crop back to) and prepare_multi_reference (several: each under 1 MiP,
sides floored, centre-cropped to 16) on seeded pictures of awkward sizes,
plus the edit run's figurine. The single-reference cap is run at BFL's 4 MiP
and at the server's 1024² staging area (the decoder's ceiling caps it there),
by setting the module's constant.

    .venv-flux3/bin/python -I reference/dump_flux3_prepare.py      (~10 s)

Writes reference/out/flux3prepare/: <case>_in.png, <case>_out.png and
manifest.json (sizes, boxes, caps).
"""

import json
import os
import sys
import types

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg = types.ModuleType("flux3")
pkg.__path__ = [os.path.join(ROOT, "reference/flux3/bfl")]
sys.modules["flux3"] = pkg
import flux3.pipeline as bfl  # noqa: E402

OUT = os.path.join(ROOT, "reference/out/flux3prepare")


def picture(w, h, seed, mode="RGB"):
    """Smooth gradients plus noise: Lanczos has something to ring on."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 7) % 256], -1)
    arr = np.clip(base + rng.integers(-40, 41, (h, w, 3)), 0, 255).astype(np.uint8)
    if mode == "RGBA":
        alpha = rng.integers(0, 256, (h, w, 1)).astype(np.uint8)
        arr = np.concatenate([arr, alpha], -1)
    return Image.fromarray(arr, mode)


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = {
        "single_small": (picture(300, 200, 1), "single"),        # no resize, padded to 304x208
        "single_grid": (picture(640, 480, 2), "single"),         # on the grid: identity
        "single_big": (picture(3000, 2000, 3), "single"),        # 6 MP: fitted under the cap, Lanczos
        "single_odd": (picture(2999, 2001, 8), "single"),        # fitted off the grid: Lanczos, then padded
        "single_thin": (picture(10, 300, 4), "single"),          # side <= 16: symmetric pad, canvas 64 wide
        "single_narrow": (picture(20, 300, 9), "single"),        # reflect with a pad (22) past the side (20)
        "single_rgba": (picture(500, 333, 5, "RGBA"), "single"), # convert("RGB") drops alpha
        "multi_small": (picture(300, 200, 6), "multi"),          # cropped to 288x192
        "multi_big": (picture(2000, 1500, 7), "multi"),          # over 1 MiP: floored, Lanczos, cropped
        "figurine": (Image.open(os.path.join(ROOT, "models/flux-3-image-klein/assets/1_figurine.png")), "single"),
    }
    caps = {"bfl": bfl.MATCH_MAX_PIXELS, "served": 1024 * 1024}
    manifest = {"caps": caps, "cases": {}}
    for name, (img, kind) in cases.items():
        img.save(os.path.join(OUT, f"{name}_in.png"))
        runs = caps.items() if kind == "single" else [("multi", None)]
        for cap_name, cap in runs:
            key = name if kind == "multi" else f"{name}@{cap_name}"
            if kind == "multi":
                out, box = bfl.prepare_multi_reference(img), None
            else:
                bfl.MATCH_MAX_PIXELS = cap
                out, box = bfl.prepare_reference(img)
            out.save(os.path.join(OUT, f"{key}_out.png"))
            manifest["cases"][key] = {
                "in": f"{name}_in.png", "out": f"{key}_out.png", "kind": kind, "cap": cap,
                "in_size": list(img.size), "out_size": list(out.size), "box": list(box) if box else None,
            }
            print(f"{key:24s} {img.size} -> {out.size} box {box}")
        bfl.MATCH_MAX_PIXELS = caps["bfl"]
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
