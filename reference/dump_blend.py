"""Dump upstream's voice blends, for stage T8 of SPEECH.md.

A comma-joined voice is **not this repository's invention**. hexgrad's own
`KPipeline.load_voice` splits a voice on commas and returns

    torch.mean(torch.stack(packs), dim=0)

so "af_bella,af_sky" already means the equal mean of two style packs
everywhere else kokoro runs, and a server that made it mean anything else
would answer the same request with a different voice. That is what this dumps:
the oracle is `load_voice` itself, not a reimplementation of it, which is
possible offline because `load_single_voice` takes a path when the name ends
in `.pt` -- so no checkpoint is downloaded and no model is constructed
(`model=False`).

The weighted form, "af_bella:3,af_sky:1", is ours: upstream has no spelling
for it. It has no oracle either, so what is dumped for it is the definition --
`sum(w_i * p_i) / sum(w_i)` in torch -- and the point of dumping it is that
the Go side's float64 accumulation is checked against torch's float32 one
rather than against itself.

Each case writes the whole `[510, 256]` blended pack, not one row, because the
Go side blends *per row* on the grounds that a weighted mean is linear, and
the whole pack is what makes that claim checkable at every length rather than
at the one the corpus happens to use.

    .venv/bin/python reference/dump_blend.py
"""

import argparse
import json
import os

import torch

from kokoro.pipeline import KPipeline

# The three-way mix is the request that opened T8 -- a client asked
# /v1/audio/speech for "af_alloy,af_bella,af_heart" and got a 400 -- and is
# here so the case that motivated the stage is the case that is checked.
EQUAL = [
    "af_bella,af_sky",
    "af_alloy,af_bella,af_heart",
    "af_heart,am_michael,bf_emma,bm_george",
]
# Weighted mixes have no upstream. The single name is in this list rather than
# the one above because it is the property that matters most: a blend of one,
# however spelled, has to be the pack itself, bit for bit.
WEIGHTED = [
    ("af_bella:3,af_sky:1", [("af_bella", 3.0), ("af_sky", 1.0)]),
    ("af_bella:0.75,af_sky:0.25", [("af_bella", 0.75), ("af_sky", 0.25)]),
    ("af_heart:1", [("af_heart", 1.0)]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Kokoro-82M")
    ap.add_argument("--out", default="reference/out/blend")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    voice_dir = os.path.join(args.model, "voices")
    path = lambda name: os.path.join(voice_dir, name + ".pt")

    # model=False builds no KModel and downloads nothing; the G2P it does
    # build is unused here and is the same one T5 was dumped against.
    pipeline = KPipeline(lang_code="a", model=False, repo_id="hexgrad/Kokoro-82M")

    manifest = {
        "source": "kokoro.pipeline.KPipeline.load_voice",
        "delimiter": ",",
        "upstream_rule": "torch.mean(torch.stack(packs), dim=0)",
        "weighted_rule": "sum(w_i * p_i) / sum(w_i), this repository's extension",
        "cases": {},
    }

    def dump(spec, pack, kind, names, weights):
        pack = pack.detach().reshape(pack.shape[0], -1).contiguous().to(torch.float32)
        fn = "blend_%02d.bin" % len(manifest["cases"])
        with open(os.path.join(args.out, fn), "wb") as fh:
            fh.write(pack.numpy().tobytes())
        flat = pack.flatten()
        manifest["cases"][spec] = {
            "file": fn, "kind": kind, "names": names, "weights": weights,
            "shape": list(pack.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {spec:34s} {kind:8s} {str(list(pack.shape)):12s} "
              f"sum={flat.double().sum():+.5f} absmax={float(flat.abs().max()):.4g}")

    print("upstream, via KPipeline.load_voice:")
    for spec in EQUAL:
        names = spec.split(",")
        pack = pipeline.load_voice(",".join(path(n) for n in names))
        # The equal mean is what load_voice computes; recomputing it here is
        # the self-check that the paths above resolved to the packs meant.
        by_hand = torch.mean(torch.stack([torch.load(path(n), weights_only=True) for n in names]), dim=0)
        gap = float((pack - by_hand).abs().max())
        if gap != 0.0:
            raise SystemExit(f"load_voice disagrees with the mean it documents for {spec}: {gap}")
        dump(spec, pack, "upstream", names, [1.0 / len(names)] * len(names))

    print("weighted, this repository's definition:")
    for spec, parts in WEIGHTED:
        names = [n for n, _ in parts]
        raw = [w for _, w in parts]
        total = sum(raw)
        stack = torch.stack([torch.load(path(n), weights_only=True) for n in names])
        w = torch.tensor([x / total for x in raw], dtype=torch.float32).reshape(-1, *([1] * (stack.dim() - 1)))
        dump(spec, (stack * w).sum(dim=0), "weighted", names, [x / total for x in raw])

    # A blend of one has to be the pack, exactly. Checked here as well as in
    # Go, because the Go side takes a different route to it -- it never
    # allocates -- and a property that holds for two different reasons is the
    # one worth stating twice.
    single = torch.load(path("af_heart"), weights_only=True).reshape(510, -1)
    blended = torch.frombuffer(
        open(os.path.join(args.out, manifest["cases"]["af_heart:1"]["file"]), "rb").read(),
        dtype=torch.float32).reshape(510, -1)
    manifest["single_gap"] = float((single - blended).abs().max())
    print(f"\na blend of one against the pack itself: {manifest['single_gap']:.3g}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['cases'])} blends to {args.out}")


if __name__ == "__main__":
    main()
