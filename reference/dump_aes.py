"""Dump Meta's Audiobox Aesthetics on songs, for MUSIC.md A14c.

Upstream's own predictor (audiobox_aesthetics 0.0.4, `infer.AesPredictor`),
fp32 on the CPU (its "bf16" precision never turns autocast on), with the
weights from models/audiobox-aesthetics (facebook/audiobox-aesthetics on HF,
loaded from its model.safetensors), records what a port must reproduce:

* the 48 kHz stereo input and its 16 kHz mono resample (torchaudio's
  sinc resampler, then the channel mean), `{case}_wav16`;
* the relative-position buckets for every offset a 10 s window has
  (-498..498), `buckets`;
* for the first and the last window of each case: the conv extractor's
  output, the projected features, the encoder's 13 hidden states (the input
  after the positional conv and layer norm, then each layer's output), and
  the frame padding mask;
* every window's four normalised predictions and its weight, and the
  song's four scores (PQ, PC, CE, CU on 1-10).

Cases: A11's 30 s synthwave source and 120 s lofi reference
(reference/out/acetasks/{src,ref}.wav: 3 and 12 full windows), the source
cut to 23.7 s (a last window of 3.7 s, padded and masked), and 12.5 s of
seeded white noise at -12 dBFS (the control).

    .venv-aes/bin/python -I reference/dump_aes.py
"""

import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

from audiobox_aesthetics.infer import AXES_NAME, AesPredictor, make_inference_batch
from audiobox_aesthetics.model.aes import AesMultiOutput, Normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reference/out/aes")
MODEL = os.path.join(ROOT, "models/audiobox-aesthetics")
TASKS = os.path.join(ROOT, "reference/out/acetasks")


class Predictor(AesPredictor):
    """AesPredictor with the weights from the local safetensors."""

    def setup_model(self):
        self.device = torch.device("cpu")
        model = AesMultiOutput.from_pretrained(MODEL)
        model.eval()
        self.model = model
        self.target_transform = {axis: Normalize(mean=model.target_transform[axis]["mean"],
                                                 std=model.target_transform[axis]["std"]) for axis in AXES_NAME}


def cases():
    src, sr = sf.read(os.path.join(TASKS, "src.wav"), dtype="float32", always_2d=True)
    ref, sr2 = sf.read(os.path.join(TASKS, "ref.wav"), dtype="float32", always_2d=True)
    assert sr == sr2 == 48000
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal((int(12.5 * 48000), 2)) * 10 ** (-12 / 20) / 3).astype(np.float32)
    return {
        "synthwave": src.T.copy(),
        "lofi": ref.T.copy(),
        "partial": src[: int(23.7 * 48000)].T.copy(),
        "noise": noise.T.copy(),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"cases": {}, "tensors": {}}

    def dump(name, t, dtype="float32"):
        arr = t.detach().contiguous().cpu().numpy().astype(dtype) if torch.is_tensor(t) else np.asarray(t, dtype)
        arr.tofile(os.path.join(OUT, name + ".bin"))
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": dtype}

    torch.set_grad_enabled(False)
    p = Predictor(checkpoint_pth=None, data_col="path")
    model = p.model
    wavlm = model.wavlm_model
    attn0 = wavlm.encoder.layers[0].self_attn
    rel = torch.arange(-498, 499)[None, :]
    dump("buckets", attn0._relative_positions_bucket(rel, bidirectional=True)[0], "int32")

    for label, wav in cases().items():
        x = torch.from_numpy(wav)
        dump(label + "_wav48", x)
        wavs16 = p.audio_resample_mono([{"path": x, "sample_rate": 48000}])
        dump(label + "_wav16", wavs16[0][0])
        wins, masks, weights, _ = make_inference_batch(wavs16, 10, 10, sample_rate=16000)
        keep = sorted({0, len(wins) - 1})
        rec = {}
        hooks = []

        def hook(name):
            def fn(_m, _i, out):
                rec[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            return fn
        hooks.append(wavlm.feature_extractor.register_forward_hook(hook("conv")))
        hooks.append(wavlm.post_extract_proj.register_forward_hook(hook("proj")))
        hooks.append(wavlm.encoder.layer_norm.register_forward_hook(hook("hidden0")))
        for i, layer in enumerate(wavlm.encoder.layers):
            hooks.append(layer.register_forward_hook(hook(f"hidden{i + 1}")))

        rows = p.forward([{"path": x, "sample_rate": 48000}])[0]
        # The same windows through the model once more, batched as the
        # predictor batched them, for the per-window predictions.
        preds = model({"wav": torch.stack(wins), "mask": torch.stack(masks)})
        for hk in hooks:
            hk.remove()
        fmask = wavlm.forward_padding_mask(torch.zeros(len(wins), 499, 1), ~torch.stack(masks).squeeze(1))
        for w in keep:
            pre = f"{label}_w{w}_"
            dump(pre + "conv", rec["conv"][w].T)
            dump(pre + "proj", rec["proj"][w])
            dump(pre + "hidden0", rec["hidden0"][w])
            for i in range(1, 13):
                dump(pre + f"hidden{i}", rec[f"hidden{i}"][:, w])  # T x B x C
            dump(pre + "pad", fmask[w], "int32")
        dump(label + "_preds", torch.stack([preds[a] for a in AXES_NAME], dim=1))
        manifest["cases"][label] = {"windows": len(wins), "weights": weights, "kept": keep, "scores": rows,
                                    "seconds": wav.shape[1] / 48000}
        print(label, len(wins), "windows", {k: round(v, 4) for k, v in rows.items()})
        with open(os.path.join(OUT, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
