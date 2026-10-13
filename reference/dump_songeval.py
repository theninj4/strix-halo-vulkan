"""Dump SongEval on songs, for MUSIC.md A14d.

SongEval (ASLP-lab, models/SongEval-src at 848fb2f) scores a whole song on
five aesthetic dimensions with a small head over MuQ-large-msd-iter's
layer-6 hidden states (OpenMuQ, models/MuQ-large-msd-iter). Its eval.py:
librosa.load(path, sr=24000) (mono, soxr_hq), MuQ over the whole song in
one pass, hidden_states[6], the head, and the five scores in the order
Coherence, Musicality, Memorability, Clarity, Naturalness. fp32 on the CPU.

The head (model.py's Generator) is restated here rather than imported from
the downloaded repo: FFN 1024→4096→1024 (ReLU), four 8-head self-attention
layers chained without residuals, mean of the last attention's output and
max of the FFN's over time, a linear to five, tanh·2 + 3. It loads the
repo's ckpt/model.safetensors with strict=False, as eval.py does.

Each song is scored twice: as eval.py runs it (librosa's soxr), and from
torchaudio's sinc resample of the same 48 kHz stereo (channels resampled,
then averaged) -- the resampler the Go side ports, so the port is gated on
that input and the gap between the two is the resampler's own effect. For
the torchaudio input the script records the mel features, the conv
subsampling's output, hidden_states 0-6, the head's FFN output, and the
scores; and it checks that transformers 5's hidden_states[6] is the sixth
layer's output, which is what it was under transformers 4.

    .venv-aes/bin/python -I reference/dump_songeval.py
"""

import json
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from muq import MuQ
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reference/out/songeval")
MUQ = os.path.join(ROOT, "models/MuQ-large-msd-iter")
CKPT = os.path.join(ROOT, "models/SongEval-src/ckpt/model.safetensors")
TASKS = os.path.join(ROOT, "reference/out/acetasks")
DIMS = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]


class Generator(nn.Module):
    """SongEval's model.Generator (in_features 1024, ffd 4096, 5 classes, 4 attention layers)."""

    def __init__(self, in_features=1024, ffd_hidden_size=4096, num_classes=5, attn_layer_num=4):
        super().__init__()
        self.attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=in_features, num_heads=8, dropout=0.2,
                                                         batch_first=True) for _ in range(attn_layer_num)])
        self.ffd = nn.Sequential(nn.Linear(in_features, ffd_hidden_size), nn.ReLU(),
                                 nn.Linear(ffd_hidden_size, in_features))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features * 2, num_classes)
        self.proj = nn.Tanh()

    def forward(self, ssl_feature, rec=None):
        ssl_feature = self.ffd(ssl_feature)
        if rec is not None:
            rec["ffd"] = ssl_feature
        tmp = ssl_feature
        for attn in self.attn:
            tmp, _ = attn(tmp, tmp, tmp)
        x = self.dropout(torch.concat([torch.mean(tmp, dim=1), torch.max(ssl_feature, dim=1)[0]], dim=1))
        return self.proj(self.fc(x)) * 2.0 + 3


def songs():
    src, sr = sf.read(os.path.join(TASKS, "src.wav"), dtype="float32", always_2d=True)
    ref, _ = sf.read(os.path.join(TASKS, "ref.wav"), dtype="float32", always_2d=True)
    assert sr == 48000
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal((int(12.5 * 48000), 2)) * 10 ** (-12 / 20) / 3).astype(np.float32)
    return {"synthwave": src.T.copy(), "lofi": ref.T.copy(), "noise": noise.T.copy()}


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"cases": {}, "tensors": {}}

    def dump(name, t, dtype="float32"):
        arr = t.detach().contiguous().cpu().numpy().astype(dtype) if torch.is_tensor(t) else np.asarray(t, dtype)
        arr.tofile(os.path.join(OUT, name + ".bin"))
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": dtype}

    torch.set_grad_enabled(False)
    muq = MuQ.from_pretrained(MUQ).eval()
    head = Generator().eval()
    missing, unexpected = head.load_state_dict(load_file(CKPT, device="cpu"), strict=False)
    assert not missing, missing

    for label, st in songs().items():
        rec = {"unexpected_keys": list(unexpected)}
        # eval.py's input: librosa.load(sr=24000) is to_mono, then resample (soxr_hq).
        lib = librosa.resample(librosa.to_mono(st), orig_sr=48000, target_sr=24000)
        ta = torchaudio.functional.resample(torch.from_numpy(st), 48000, 24000).mean(0).numpy()
        dump(label + "_wav48", st)
        dump(label + "_wav24", ta)
        for name, wav in (("librosa", lib), ("torchaudio", ta)):
            audio = torch.tensor(wav).unsqueeze(0)
            seen = {}
            hooks = []
            if name == "torchaudio":
                mm = muq.model

                def grab(key):
                    def fn(_m, _i, out):
                        seen[key] = (out[0] if isinstance(out, tuple) else out).detach().clone()
                    return fn
                hooks.append(mm.conv.register_forward_hook(grab("conv")))
                for i in range(6):
                    hooks.append(mm.conformer.layers[i].register_forward_hook(grab(f"layer{i}")))
            out = muq(audio, output_hidden_states=True)
            for hk in hooks:
                hk.remove()
            hs = out["hidden_states"]
            feats = hs[6]
            hrec = {}
            scores = head(feats, hrec if name == "torchaudio" else None).squeeze(0)
            rec[name] = {d: float(scores[i]) for i, d in enumerate(DIMS)}
            if name == "torchaudio":
                mel = muq.model.normalize(muq.model.preprocessing(audio, features=["melspec_2048"]))["melspec_2048"]
                dump(label + "_mel", mel[0])
                dump(label + "_conv", seen["conv"][0])
                for i in range(7):
                    dump(label + f"_hidden{i}", hs[i][0])
                rec["hidden6_is_layer5_out"] = bool(torch.equal(hs[6], seen["layer5"]))
                rec["hidden0_is_conv"] = bool(torch.equal(hs[0], seen["conv"]))
                rec["tokens"] = int(feats.shape[1])
                rec["hidden_states"] = len(hs)
                dump(label + "_ffd", hrec["ffd"][0])
        manifest["cases"][label] = rec
        print(label, json.dumps({k: v for k, v in rec.items() if k != "unexpected_keys"}))
        with open(os.path.join(OUT, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    main()
