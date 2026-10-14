"""The codec encoder's oracle (research/tts-instruct-vertical.md B8).

Voice cloning puts a reference clip's codes in the prompt; upstream gets them
from qwen_tts's Qwen3TTSTokenizer.encode, which is HF's MimiModel.encode
(SEANet, an 8-layer transformer, a stride-2 downsample, the split RVQ) cut to
16 codebooks and to ceil(samples / 1920) frames.

The clip here is testdata/jfk.wav (16 kHz, 11 s: past the transformer's
nominal 250-frame window at 25 Hz) resampled to 24 kHz the way qwen_tts does
it (librosa.resample's default, soxr_hq), and written out as the 24 kHz WAV
the Go side reads, so the resampler is not part of the gate. Each stage is
dumped: SEANet's output, the transformer's, the downsample's, the codes; and
a second clip, the design case's own audio (already 24 kHz), for a clone run.

    .venv-breeze/bin/python -I reference/dump_breeze_enc.py

Writes reference/out/breeze/enc/{jfk24.wav,tensors.safetensors,manifest.json}.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reference/out/breeze/enc")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from qwen_tts import Qwen3TTSTokenizer  # noqa: E402


def stages(model, wav):
    """MimiModel._encode_frame with its intermediates, then the qwen cut."""
    enc = model.encoder
    x = torch.from_numpy(wav)[None, None]
    st = {}
    h = enc.encoder(x)
    st["seanet"] = h[0].T  # [T25][512]
    t = enc.encoder_transformer(h.transpose(1, 2), return_dict=True)[0]
    st["transformer"] = t[0]
    d = enc.downsample(t.transpose(1, 2))
    st["downsample"] = d[0].T  # [T12][512]
    codes = enc.quantizer.encode(d, 32)  # [Q][1][T]
    n = -(-len(wav) // 1920)
    st["codes"] = codes[:16, 0, :n].T.float()  # [F][16]
    return st


def main():
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    tok = Qwen3TTSTokenizer.from_pretrained(os.path.join(REPO, "models/Breeze-TTS-2/audio_tokenizer"), device_map="cpu")
    model = tok.model.float()
    os.makedirs(OUT, exist_ok=True)
    out, man = {}, {}
    jfk, sr = sf.read(os.path.join(REPO, "testdata/jfk.wav"), dtype="float32")
    jfk24 = librosa.resample(y=jfk, orig_sr=sr, target_sr=24000).astype(np.float32)
    sf.write(os.path.join(OUT, "jfk24.wav"), jfk24, 24000, subtype="FLOAT")
    design, _ = sf.read(os.path.join(REPO, "reference/out/breeze/design/audio.wav"), dtype="float32")
    for name, wav in (("jfk", jfk24), ("design", design)):
        st = stages(model, wav)
        # The whole path is upstream's own: the codes must be what encode() says.
        ref = tok.encode(wav, sr=24000)["audio_codes"][0]
        assert torch.equal(ref.float(), st["codes"]), f"{name}: stages drifted from Qwen3TTSTokenizer.encode"
        for k, v in st.items():
            out[f"{name}_{k}"] = v.contiguous()
        out[f"{name}_wav"] = torch.from_numpy(wav).contiguous()
        man[name] = {"samples": len(wav), "frames": int(st["codes"].shape[0])}
        print(f"{name}: {len(wav)} samples -> {st['codes'].shape[0]} frames", flush=True)
    save_file(out, os.path.join(OUT, "tensors.safetensors"))
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
