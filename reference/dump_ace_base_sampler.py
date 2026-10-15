"""Dump the XL base model's sampler, for research/music-vertical.md A15b.

The base model's `generate_audio` (acestep/models/xl_base, upstream ca1e85f)
is run with its decoder replaced by a stub whose velocity Go reproduces
exactly, as dump_ace_plan.py does for turbo. The stub tells the CFG pair's
rows apart: row 0 (conditional) is v = 0.5 x + t c and row 1 (the
null-condition row) is v = 0.4 x + t d, all fp32. So the whole loop is
recorded: the linspace + shift schedule, the CFG pair, APG's momentum, norm
clamp and projection, the CFG interval, Euler, and DCW when asked.

It also captures what `generate_music` hands the base model for a request
with Gradio's pure-base settings (32 steps, shift 3, guidance 7), so the
defaults are read from upstream and not assumed.

The handler runs on the CPU in fp32, from models/ace-checkpoints (see
dump_ace_plan.py for why not the HF clone).

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv-acestep/bin/python reference/dump_ace_base_sampler.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "models/ACE-Step-1.5-src")

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

OUT = "reference/out/acebasesampler"
PROJECT = "models/ace-checkpoints"

# (label, T, generate_audio overrides). Gradio's base defaults are steps 32,
# shift 3, guidance 7, APG, interval 0-1, DCW off.
CASES = [
    ("n32_s3", 40, dict(infer_steps=32, shift=3.0)),
    ("n32_s3_odd", 37, dict(infer_steps=32, shift=3.0)),
    ("n32_s3_long", 750, dict(infer_steps=32, shift=3.0)),
    ("n50_s1", 40, dict(infer_steps=50, shift=1.0)),
    ("n8_s3", 40, dict(infer_steps=8, shift=3.0)),
    ("n24_s2", 37, dict(infer_steps=24, shift=2.0)),
    ("n32_s3_g3", 40, dict(infer_steps=32, shift=3.0, diffusion_guidance_scale=3.0)),
    ("n32_s3_nocfg", 40, dict(infer_steps=32, shift=3.0, diffusion_guidance_scale=1.0)),
    ("n32_s3_interval", 40, dict(infer_steps=32, shift=3.0, cfg_interval_start=0.3, cfg_interval_end=0.8)),
    ("n32_s3_dcw", 37, dict(infer_steps=32, shift=3.0, dcw_enabled=True)),
    ("n17_s3_long", 3000, dict(infer_steps=17, shift=3.0)),
]


class Stop(Exception):
    pass


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"request": {}, "sampler": {}, "tensors": {}}

    def dump(name, t, dtype):
        arr = t.detach().contiguous().cpu().numpy().astype(dtype) if isinstance(t, torch.Tensor) else np.asarray(t, dtype)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(arr.tobytes())
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": np.dtype(dtype).name}

    h = AceStepHandler()
    msg, ok = h.initialize_service(project_root=PROJECT, config_path="acestep-v15-xl-base", device="cpu",
                                   use_mlx_dit=False)
    assert ok, msg
    assert h.dtype == torch.float32
    assert not h.is_turbo_model()
    silence = h.silence_latent[0]
    real_generate = h.model.generate_audio

    # --- what a Gradio-default base request hands the model ---------------
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        raise Stop()

    h.model.generate_audio = capture
    params = GenerationParams(thinking=False, inference_steps=32, shift=3.0, guidance_scale=7.0, seed=42,
                              caption="lofi hip hop beat", lyrics="", duration=30.0,
                              use_cot_metas=False, use_cot_caption=False, use_cot_language=False)
    generate_music(h, None, params, GenerationConfig(batch_size=1, use_random_seed=False, seeds=[42]))
    assert captured, "generate_audio was not reached"
    manifest["request"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                           for k, v in captured.items() if not isinstance(v, torch.Tensor)}
    print("request:", {k: manifest["request"].get(k) for k in (
        "infer_steps", "diffusion_guidance_scale", "use_adg", "cfg_interval_start", "cfg_interval_end",
        "shift", "dcw_enabled", "infer_method", "sampler_mode", "timesteps")}, flush=True)
    h.model.generate_audio = real_generate

    # --- the sampler over a stub decoder ------------------------------------
    real_decoder = h.model.decoder

    class StubDecoder(torch.nn.Module):
        def __init__(self, c, d):
            super().__init__()
            self.c, self.d = c, d
            self.calls = []

        def forward(self, hidden_states, timestep, **kwargs):
            self.calls.append((float(timestep[0]), hidden_states.shape[0]))
            t = timestep.view(-1, 1, 1)
            v = 0.5 * hidden_states[:1] + t[:1] * self.c
            if hidden_states.shape[0] == 2:
                v = torch.cat([v, 0.4 * hidden_states[1:] + t[1:] * self.d], dim=0)
            return v, kwargs.get("past_key_values")

    for label, T, over in CASES:
        c = torch.cos(torch.arange(T * 64, dtype=torch.float32) * 0.37).view(1, T, 64)
        d = torch.sin(torch.arange(T * 64, dtype=torch.float32) * 0.23).view(1, T, 64) * 1.5
        stub = StubDecoder(c, d)
        h.model.decoder = stub
        kw = dict(
            text_hidden_states=torch.zeros(1, 4, 1024), text_attention_mask=torch.ones(1, 4),
            lyric_hidden_states=torch.zeros(1, 4, 1024), lyric_attention_mask=torch.ones(1, 4),
            refer_audio_acoustic_hidden_states_packed=silence[:750].unsqueeze(0),
            refer_audio_order_mask=torch.zeros(1, dtype=torch.long),
            src_latents=silence[:T].unsqueeze(0), chunk_masks=torch.full((1, T, 64), 2.0),
            is_covers=torch.zeros(1, dtype=torch.bool), silence_latent=h.silence_latent,
            seed=[7], infer_method="ode", use_progress_bar=False,
            diffusion_guidance_scale=7.0, use_adg=False, cfg_interval_start=0.0, cfg_interval_end=1.0,
            dcw_enabled=False, dcw_mode="double", dcw_scaler=0.05, dcw_high_scaler=0.02, dcw_wavelet="haar",
        )
        kw.update(over)
        noise = h.model.prepare_noise(torch.zeros(1, T, 128), [7])
        out = h.model.generate_audio(**kw)["target_latents"][0]
        manifest["sampler"][label] = {
            "T": T, "steps": over["infer_steps"], "shift": over["shift"],
            "guidance": kw["diffusion_guidance_scale"], "interval": [kw["cfg_interval_start"], kw["cfg_interval_end"]],
            "dcw": kw["dcw_enabled"],
            "timesteps": [t for t, _ in stub.calls], "rows": [r for _, r in stub.calls],
        }
        dump(f"{label}_c", c[0], np.float32)
        dump(f"{label}_d", d[0], np.float32)
        dump(f"{label}_noise", noise[0], np.float32)
        dump(f"{label}_out", out, np.float32)
        print(f"{label}: {len(stub.calls)} calls, rows {sorted(set(r for _, r in stub.calls))}, "
              f"t[:3] {['%.6f' % t for t, _ in stub.calls[:3]]}", flush=True)
    h.model.decoder = real_decoder

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
