"""Dump ACE-Step 1.5's audio-in tasks through upstream's own handler, for
MUSIC.md A11b-d: a reference audio's timbre, cover, cover-nofsq and repaint.

Every case runs `inference.generate_music` (thinking off; the LM is skipped
for these tasks anyway) on the CPU in fp32, seed 42, 8 steps, shift 3, with
the instruction upstream's API resolves for the task. The audio comes from
A5's oracle (reference/out/acevae): the 30 s synthwave song is the source
(src.wav) and the 120 s lofi one the reference (ref.wav), written as 48 kHz
float wavs so no resampling happens. Hooks record, per case:

* `src_wav`, `ref_wav`: the processed source and the 30 s the reference
  became, and `ref_offsets`: the three random.randint draws that chose its
  segments;
* `enc{i}_mean`, `enc{i}_sample`: every VAE encode, in call order (its
  posterior's mean, and the sample upstream went on with -- its noise is
  global torch RNG, so the gates feed the sample);
* the `generate_audio` inputs: `refer` (the packed timbre latents),
  `src_latents`, `chunk_masks`, `is_covers`, `repaint_mask`, `clean_src`,
  and the scalars;
* `cond{j}_enc`, `cond{j}_ctx`: every `prepare_condition` (the second is
  the non-cover one when audio_cover_strength < 1), its packed sequence and
  context latents;
* `pool`, `codes`: the audio tokenizer's pooled rows before the FSQ, and
  the code indices (cover);
* `noise`, `latents`: the DiT's noise and final latents;
* `audio`: the -1 dBFS audio upstream returns (after the repaint splice).

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv-acestep/bin/python reference/dump_ace_tasks.py [--bf16] [label ...]

--bf16 runs the same requests the way upstream serves them on CUDA: the
DiT, the text encoder and the VAE in bfloat16 (the handler's dtype there),
from the fp32 run's DiT noise, into reference/out/acetasks_bf16 -- the bar
a device path's drift from the fp32 oracle is priced against (decision 2).
Its VAE samples its own posterior noise, as upstream's does.
"""

import json
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "models/ACE-Step-1.5-src")
sys.path.insert(0, "reference")

from acestep.constants import TASK_INSTRUCTIONS  # noqa: E402
from acestep.core.generation.handler import io_audio  # noqa: E402
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from dump_ace_plan import LYRICS, PROJECT  # noqa: E402

BF16 = "--bf16" in sys.argv
FP32_OUT = "reference/out/acetasks"
OUT = "reference/out/acetasks_bf16" if BF16 else FP32_OUT
SRC_WAV = os.path.join(FP32_OUT, "src.wav")
REF_WAV = os.path.join(FP32_OUT, "ref.wav")

SYNTH = dict(caption="upbeat synthwave with female vocals", lyrics=LYRICS, bpm=120, keyscale="C major",
             timesignature="4", vocal_language="en")
FOLK = dict(caption="acoustic folk ballad with male vocals and fingerpicked guitar", lyrics=LYRICS,
            vocal_language="en")

# (label, GenerationParams overrides).
CASES = [
    ("ref_t2m", dict(SYNTH, duration=30.0, reference_audio=REF_WAV)),
    ("cover", dict(FOLK, task_type="cover", src_audio=SRC_WAV)),
    ("cover_mix", dict(FOLK, task_type="cover", src_audio=SRC_WAV, audio_cover_strength=0.5,
                       cover_noise_strength=0.3)),
    ("cover_nofsq", dict(FOLK, task_type="cover-nofsq", src_audio=SRC_WAV, reference_audio=REF_WAV)),
    ("repaint", dict(SYNTH, task_type="repaint", src_audio=SRC_WAV, repainting_start=10.0, repainting_end=20.0)),
    ("repaint_out", dict(SYNTH, task_type="repaint", src_audio=SRC_WAV, repainting_start=24.0,
                         repainting_end=34.0, repaint_mode="conservative", chunk_mask_mode="explicit")),
]


class RandLog:
    """io_audio's `random`, logging randint."""

    def __init__(self, log):
        self.log = log

    def randint(self, a, b):
        v = random.randint(a, b)
        self.log.append(v)
        return v


def main():
    labels = [a for a in sys.argv[1:] if not a.startswith("--")] or [c[0] for c in CASES]
    os.makedirs(OUT, exist_ok=True)
    manifest_path = os.path.join(OUT, "manifest.json")
    manifest = {"cases": {}, "tensors": {}}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)

    def dump(name, t, dtype=np.float32):
        if isinstance(t, torch.Tensor):
            t = t.detach().contiguous().cpu()
            t = t.float() if dtype == np.float32 else t
            arr = t.numpy().astype(dtype)
        else:
            arr = np.asarray(t, dtype)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(arr.tobytes())
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": np.dtype(dtype).name}

    for path, label in () if BF16 else ((SRC_WAV, "full_metas"), (REF_WAV, "defaults")):
        wav = np.fromfile(f"reference/out/acevae/{label}_final.bin", dtype=np.float32).reshape(2, -1)
        sf.write(path, wav.T, 48000, subtype="FLOAT")

    torch.set_grad_enabled(False)
    h = AceStepHandler()
    msg, ok = h.initialize_service(project_root=PROJECT, config_path="acestep-v15-xl-turbo", device="cpu",
                                   use_mlx_dit=False)
    assert ok, msg
    model = h.model
    if BF16:
        model.to(torch.bfloat16)
        h.dtype = torch.bfloat16
        h.text_encoder.to(torch.bfloat16)
        h.vae.to(torch.bfloat16)
        h._get_vae_dtype = lambda *_a, **_k: torch.bfloat16
        h.silence_latent = h.silence_latent.to(torch.bfloat16)
    cases = dict(CASES)

    for label in labels:
        p = label + "_"
        rec = {"enc": [], "cond": [], "ref_offsets": []}
        hooks = []

        # The VAE: every encode's posterior mean and the sample taken from it.
        real_encode = h.vae.encode

        def encode(x, *a, **k):
            out = real_encode(x, *a, **k)
            post = out.latent_dist
            real_sample = post.sample

            def sample(*sa, **sk):
                s = real_sample(*sa, **sk)
                rec["enc"].append((post.mean.clone(), s.clone()))
                return s
            post.sample = sample
            return out
        h.vae.encode = encode

        real_ref, real_src = h.process_reference_audio, h.process_src_audio

        def ref_audio(path):
            out = real_ref(path)
            rec["ref_wav"] = out
            return out

        def src_audio(path):
            out = real_src(path)
            rec["src_wav"] = out
            return out
        h.process_reference_audio, h.process_src_audio = ref_audio, src_audio
        real_random = io_audio.random
        io_audio.random = RandLog(rec["ref_offsets"])

        real_generate, real_prepare, real_noise = model.generate_audio, model.prepare_condition, model.prepare_noise

        def generate(**kw):
            rec["kw"] = kw
            out = real_generate(**kw)
            rec["latents"] = out["target_latents"]
            return out

        def prepare(**kw):
            out = real_prepare(**kw)
            rec["cond"].append(out)
            return out

        def noise(ctx, seed):
            if BF16:
                # The fp32 run's noise, so the two runs start from the same x.
                ref = np.fromfile(os.path.join(FP32_OUT, p + "noise.bin"), dtype=np.float32)
                out = torch.from_numpy(ref.reshape(1, -1, 64)).to(ctx.dtype)
            else:
                out = real_noise(ctx, seed)
            rec["noise"] = out
            return out
        model.generate_audio, model.prepare_condition, model.prepare_noise = generate, prepare, noise

        def tok_hook(_m, inp, out):
            rec.setdefault("codes", out[1])
        hooks.append(model.tokenizer.register_forward_hook(tok_hook))
        hooks.append(model.tokenizer.attention_pooler.register_forward_hook(
            lambda _m, _i, out: rec.setdefault("pool", out)))

        kw = dict(cases[label])
        task = kw.get("task_type", "text2music")
        params = GenerationParams(thinking=False, shift=3.0, seed=42, use_cot_metas=False, use_cot_caption=False,
                                  use_cot_language=False, instruction=TASK_INSTRUCTIONS[task], **kw)
        random.seed(0)
        torch.manual_seed(0)
        t0 = time.time()
        res = generate_music(h, None, params, GenerationConfig(batch_size=1, use_random_seed=False, seeds=[42]))
        elapsed = time.time() - t0
        for hk in hooks:
            hk.remove()
        h.vae.encode = real_encode
        h.process_reference_audio, h.process_src_audio = real_ref, real_src
        io_audio.random = real_random
        model.generate_audio, model.prepare_condition, model.prepare_noise = real_generate, real_prepare, real_noise
        assert res.success, res.error

        if "src_wav" in rec and rec["src_wav"] is not None:
            dump(p + "src_wav", rec["src_wav"])
        if "ref_wav" in rec and rec["ref_wav"] is not None:
            dump(p + "ref_wav", rec["ref_wav"])
        for i, (m, s) in enumerate(rec["enc"]):
            dump(f"{p}enc{i}_mean", m[0])
            dump(f"{p}enc{i}_sample", s[0])
        k = rec["kw"]
        dump(p + "refer", k["refer_audio_acoustic_hidden_states_packed"][0])
        dump(p + "src_latents", k["src_latents"][0])
        dump(p + "chunk_masks", k["chunk_masks"][0])
        dump(p + "text_hidden", k["text_hidden_states"][0])
        if k.get("non_cover_text_hidden_states") is not None:
            dump(p + "non_cover_text_hidden", k["non_cover_text_hidden_states"][0])
        if k.get("repaint_mask") is not None:
            dump(p + "repaint_mask", k["repaint_mask"][0].float())
            dump(p + "clean_src", k["clean_src_latents"][0])
        for j, (enc, _mask, ctx) in enumerate(rec["cond"]):
            dump(f"{p}cond{j}_enc", enc[0])
            dump(f"{p}cond{j}_ctx", ctx[0])
        if "pool" in rec:
            dump(p + "pool", rec["pool"][0])
            dump(p + "codes", rec["codes"][0].reshape(-1), np.int64)
        dump(p + "noise", rec["noise"][0])
        dump(p + "latents", rec["latents"][0])
        dump(p + "audio", res.audios[0]["tensor"])

        manifest["cases"][label] = {
            "params": {kk: v for kk, v in kw.items() if kk != "lyrics"},
            "instruction": TASK_INSTRUCTIONS[task],
            "seconds": elapsed,
            "ref_offsets": rec["ref_offsets"],
            "encodes": len(rec["enc"]),
            "conditions": len(rec["cond"]),
            "is_covers": k["is_covers"].tolist(),
            "audio_cover_strength": k.get("audio_cover_strength"),
            "cover_noise_strength": k.get("cover_noise_strength"),
            "repaint_crossfade_frames": k.get("repaint_crossfade_frames"),
            "repaint_injection_ratio": k.get("repaint_injection_ratio"),
            "latent_length": int(rec["latents"].shape[1]),
            "audio_samples": int(res.audios[0]["tensor"].shape[-1]),
        }
        print(label, json.dumps(manifest["cases"][label]))
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=1)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
