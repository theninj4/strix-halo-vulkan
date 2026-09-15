"""Convert Kokoro-82M's PyTorch pickle into safetensors, for stage T1 of SPEECH.md.

`hexgrad/Kokoro-82M` ships one file, `kokoro-v1_0.pth`: a pickle holding five
`state_dict`s (`bert`, `bert_encoder`, `predictor`, `decoder`, `text_encoder`),
every key prefixed `module.` from the `DataParallel` it was trained under. The
Go side reads safetensors and nothing else, so nothing about this model can be
compared against anything until the pickle is a mapping. That is this script.

It is not a transcription. Three things happen on the way through, each of
which is a decision the Go loader would otherwise have to make at runtime, and
each of which is checked here rather than assumed:

  * **`weight_norm` is folded.** 116 of the convolutions are stored as a
    direction `weight_v` and a magnitude `weight_g`, and torch recomputes
    `w = g * v / ||v||` on every forward. The fold is done once, here, and
    checked against `torch.nn.utils.remove_weight_norm` -- the same shape of
    check as parakeet's BatchNorm fold, and for the same reason: a mistake in
    it would show up in Go looking like a convolution bug.

  * **The `InstanceNorm1d` affine inside every `AdaIN1d` is identity.** The
    module is built `affine=True` (an ONNX export workaround, per the upstream
    comment) but the checkpoint has no weights for it, so `KModel` loads
    `strict=False` and leaves it at init. That is load-bearing: AdaIN is
    `(1 + gamma) * instance_norm(x) + beta` with *no* learned scale under the
    gamma. Asserted here, then dropped from the output.

  * **What the checkpoint does not cover is enumerated.** `KModel` falls back
    to `strict=False` on a `load_state_dict` failure and logs at debug, so a
    genuinely missing tensor would load as noise in silence. The key sets are
    diffed both ways and the difference is written to the manifest.

The 54 voices are `[510, 1, 256]` pickles of their own -- one row per token
count, split 128/128 between the decoder's style vector and the predictor's --
and they go into a second file in the same directory, under `voice.<name>`, so
that one `safetensors.OpenSet` gives the Go side both the model and the voices.

    .venv/bin/python reference/convert_kokoro.py
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file

from kokoro.model import KModel


def weight_norm_fold(g, v, dim=0):
    """`torch._weight_norm(v, g, dim)`, written out.

    `g` is a per-output-channel magnitude shaped like `v` with every dimension
    but `dim` collapsed to 1, so the broadcast needs no reshaping: the norm is
    taken over every dimension *except* `dim`, which for a `Conv1d` means over
    (in_channels, kernel) and for a `ConvTranspose1d` -- whose weight is
    [in, out, k] -- means over (out_channels, kernel). That asymmetry is why
    this is written generically rather than as `v.flatten(1).norm(dim=1)`.
    """
    other = [d for d in range(v.dim()) if d != dim]
    norm = v.pow(2).sum(dim=other, keepdim=True).sqrt()
    return v * (g / norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Kokoro-82M")
    ap.add_argument("--out", default=None, help="directory for the .safetensors (default: --model)")
    args = ap.parse_args()
    out_dir = args.out or args.model
    os.makedirs(out_dir, exist_ok=True)

    pth = os.path.join(args.model, "kokoro-v1_0.pth")
    cfg_path = os.path.join(args.model, "config.json")
    raw = torch.load(pth, map_location="cpu", weights_only=True)
    with open(cfg_path, encoding="utf-8") as fh:
        config = json.load(fh)

    manifest = {
        "source": pth,
        "config": cfg_path,
        "modules": {k: {"tensors": len(v), "params": int(sum(t.numel() for t in v.values()))}
                    for k, v in raw.items()},
    }
    total_raw = sum(m["params"] for m in manifest["modules"].values())
    print(f"{pth}: {len(raw)} modules, {total_raw / 1e6:.3f} M parameters")
    for name, m in manifest["modules"].items():
        print(f"  {name:14s} {m['tensors']:4d} tensors {m['params'] / 1e6:8.3f} M")

    # ------------------------------------------------- what the pickle covers
    # `KModel.__init__` retries `load_state_dict` with strict=False when the
    # prefixed keys fail, which is every module here -- so a tensor the
    # checkpoint is missing stays at its random init and says nothing. The
    # diff below is the only thing standing between that and a silent bug.
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg_path, model=pth).eval()
    have = {f"{top}.{k[7:] if k.startswith('module.') else k}" for top, sd in raw.items() for k in sd}
    want = {n for n, _ in model.state_dict().items()}
    missing = sorted(want - have)
    extra = sorted(have - want)
    manifest["checkpoint_coverage"] = {
        "in_model_not_in_checkpoint": missing,
        "in_checkpoint_not_in_model": extra,
    }
    print(f"\ncheckpoint covers {len(have & want)} of the model's {len(want)} tensors; "
          f"{len(missing)} left at init, {len(extra)} unused")

    # Every one of those uncovered tensors must be an InstanceNorm1d affine
    # sitting at identity, or an InstanceNorm running count. Anything else
    # means the model is running on random numbers somewhere.
    sd = model.state_dict()
    bad = []
    for n in missing:
        t = sd[n]
        if n.endswith("num_batches_tracked"):
            continue
        if n.endswith(".norm.weight") and bool((t == 1).all()):
            continue
        if n.endswith(".norm.bias") and bool((t == 0).all()):
            continue
        bad.append(n)
    if bad:
        raise SystemExit(f"uncovered tensors that are not an identity affine: {bad}")
    manifest["adain_instancenorm_affine"] = {
        "count": len([n for n in missing if n.endswith((".norm.weight", ".norm.bias"))]) // 2,
        "identity": True,
        "note": "AdaIN1d builds InstanceNorm1d(affine=True) but the checkpoint has no "
                "weights for it, so the affine is 1/0: AdaIN is (1+gamma)*instance_norm(x)+beta",
    }
    print(f"AdaIN InstanceNorm affines: {manifest['adain_instancenorm_affine']['count']}, all identity")

    # ------------------------------------------------------- fold weight_norm
    # Folded by hand from the pickle, then checked against torch's own
    # recomputation via remove_weight_norm. The hand fold is what the manifest
    # describes; torch is the oracle for it.
    folded, gap, n_folded = {}, 0.0, 0
    for top, state in raw.items():
        for key, v in state.items():
            name = f"{top}.{key[7:] if key.startswith('module.') else key}"
            if name.endswith(".weight_v"):
                g = state[key[: -len("weight_v")] + "weight_g"]
                folded[name[: -len(".weight_v")] + ".weight"] = weight_norm_fold(g, v)
                n_folded += 1
            elif name.endswith(".weight_g"):
                continue
            else:
                folded[name] = v.contiguous()

    for mod in model.modules():
        if hasattr(mod, "weight_g"):
            torch.nn.utils.remove_weight_norm(mod)
    ref = model.state_dict()
    for name, t in folded.items():
        gap = max(gap, float((t - ref[name]).abs().max()))
    manifest["weight_norm"] = {"folded": n_folded, "max_abs_gap_vs_torch": gap}
    print(f"folded {n_folded} weight_norm convolutions: max abs gap vs torch {gap:.3g}")

    # ------------------------------------------------------------------ write
    # fp32, the checkpoint's own precision. Narrowing to fp16 is a decision for
    # the Vulkan port (T4), taken against the absmax survey below, not here.
    tensors = {n: t.contiguous().to(torch.float32) for n, t in folded.items()}
    model_path = os.path.join(out_dir, "model.safetensors")
    save_file(tensors, model_path, metadata={"format": "pt", "source": "kokoro-v1_0.pth"})
    params = sum(t.numel() for t in tensors.values())
    manifest["model_safetensors"] = {
        "path": model_path, "tensors": len(tensors), "params": int(params),
        "bytes": os.path.getsize(model_path),
        "params_dropped_by_fold": total_raw - int(params),
    }
    print(f"\nwrote {len(tensors)} tensors, {params / 1e6:.3f} M parameters "
          f"({os.path.getsize(model_path) / 1e6:.1f} MB) to {model_path}")
    print(f"  {total_raw - int(params)} parameters fewer than the pickle: the weight_g "
          f"magnitudes, now inside the folded weights")

    # ----------------------------------------------------------------- voices
    voice_dir = os.path.join(args.model, "voices")
    voices, rows, dim = {}, None, None
    for fn in sorted(os.listdir(voice_dir)):
        if not fn.endswith(".pt"):
            continue
        v = torch.load(os.path.join(voice_dir, fn), map_location="cpu", weights_only=True)
        v = v.squeeze(1).contiguous().to(torch.float32)  # [510, 1, 256] -> [510, 256]
        if rows is None:
            rows, dim = v.shape
        elif tuple(v.shape) != (rows, dim):
            raise SystemExit(f"{fn}: shape {tuple(v.shape)}, expected {(rows, dim)}")
        voices[f"voice.{fn[:-3]}"] = v
    voices_path = os.path.join(out_dir, "voices.safetensors")
    save_file(voices, voices_path, metadata={"format": "pt", "source": "voices/*.pt"})
    manifest["voices"] = {
        "path": voices_path, "count": len(voices), "rows": rows, "dim": dim,
        "names": sorted(n[len("voice."):] for n in voices),
        "note": "row i is the style for a token sequence of length i+1; the first 128 of "
                "the 256 go to the decoder, the last 128 to the predictor",
    }
    print(f"wrote {len(voices)} voices, [{rows}, {dim}] each, to {voices_path}")

    # ----------------------------------------------- how wide are the weights
    absmax = {n: float(t.abs().max()) for n, t in tensors.items()}
    manifest["fp16"] = {
        "max_weight": max(absmax.values()),
        "top": sorted(absmax.items(), key=lambda kv: -kv[1])[:10],
        "min_absmax": sorted(absmax.items(), key=lambda kv: kv[1])[:5],
    }
    print(f"\nwidest weight {manifest['fp16']['max_weight']:.4g} against fp16's 65504:")
    for n, a in manifest["fp16"]["top"][:5]:
        print(f"  {n:60s} {a:.4g}")

    man_path = os.path.join(out_dir, "convert_manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {man_path}")


if __name__ == "__main__":
    main()
