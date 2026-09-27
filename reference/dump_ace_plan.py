"""Dump ACE-Step 1.5's request plan, for MUSIC.md A1: everything a DiT-only
text2music request decides around the transformer.

Three oracles, all upstream's own code at models/ACE-Step-1.5-src (ca1e85f):

* **Requests.** `acestep.inference.generate_music` -- the public entry, with
  thinking off and no LM -- is driven for a handful of requests, and the
  arguments that reach `model.generate_audio` are captured and the run
  stopped there. That records what the pipeline builds around the DiT: the
  caption and lyric prompts and their token ids, the latent length a
  duration becomes, the source latents, chunk masks, cover flags, the
  timbre reference, the seed and every sampler argument. The text encoder's
  hidden states and the lyric embeddings come along for A2.
* **The sampler.** `generate_audio` is run again with the decoder replaced
  by a stub whose velocity Go reproduces exactly (v = 0.5 x + t c, fp32),
  so the whole upstream loop -- schedule choice, Euler, the final x0 jump,
  DCW -- is recorded step by step for each shift, with DCW on and off, at an
  odd and an even latent length.
* **FSQ.** `get_output_from_indices` on a spread of code indices, and the
  implicit codebook rows under them.

The handler is initialised on the CPU, which runs fp32. Its checkpoint
directory is models/ace-checkpoints/checkpoints: symlinks to the weights,
plus upstream's own model code (acestep/models/xl_turbo), which is what the
handler would sync into a checkpoint anyway -- pointing it at the HF clone
lets it overwrite the clone's files.

    HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_plan.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "models/ACE-Step-1.5-src")

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

OUT = "reference/out/aceplan"
PROJECT = "models/ace-checkpoints"

LONG_CAPTION = ", ".join(
    f"layer {i} of warm analog synth pads over a slow four-on-the-floor kick with shimmering hi-hats"
    for i in range(30)
)
LYRICS = """[Verse 1]
Neon rivers run beneath the city light
We were strangers dancing through the night

[Chorus]
Hold on, hold on, the morning's coming soon
Sing it loud beneath the paper moon
"""
ZH_LYRICS = """[verse]
月光洒在窗台上
我在等你回家的方向

[chorus]
风吹过 心跳着
"""

# (label, GenerationParams overrides). Every case runs thinking off, shift 3,
# seed 42, one item -- upstream's defaults otherwise.
CASES = [
    ("full_metas", dict(caption="upbeat synthwave with female vocals", lyrics=LYRICS, bpm=120,
                        keyscale="C major", timesignature="4", duration=30.0, vocal_language="en")),
    ("instrumental_60", dict(caption="calm solo piano, rain outside", lyrics="[Instrumental]",
                             duration=60.0, instrumental=True)),
    ("defaults", dict(caption="lofi hip hop beat", lyrics="")),
    ("zh_odd", dict(caption="温柔的中文流行歌曲，钢琴伴奏", lyrics=ZH_LYRICS, duration=12.34,
                    vocal_language="zh", keyscale="A minor")),
    ("short", dict(caption="a drum fill", lyrics="", duration=3.0, bpm=90)),
    ("long_caption", dict(caption=LONG_CAPTION, lyrics=LYRICS * 12, duration=200.0)),
]


class Stop(Exception):
    pass


class TokenizerProxy:
    """Forwards to the real tokenizer and records what it was asked."""

    def __init__(self, tok):
        self._tok = tok
        self.calls = []

    def __call__(self, text, *args, **kwargs):
        out = self._tok(text, *args, **kwargs)
        self.calls.append({"text": text, "max_length": kwargs.get("max_length"),
                           "ids": out["input_ids"][0].tolist()})
        return out

    def __getattr__(self, name):
        return getattr(self._tok, name)


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"cases": {}, "sampler": {}, "fsq": {}, "tensors": {}}

    def dump(name, t, dtype):
        arr = t.detach().contiguous().cpu().numpy().astype(dtype) if isinstance(t, torch.Tensor) else np.asarray(t, dtype)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(arr.tobytes())
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": np.dtype(dtype).name}

    h = AceStepHandler()
    msg, ok = h.initialize_service(project_root=PROJECT, config_path="acestep-v15-xl-turbo", device="cpu",
                                   use_mlx_dit=False)
    assert ok, msg
    assert h.dtype == torch.float32
    silence = h.silence_latent[0]  # [15000, 64]
    real_generate = h.model.generate_audio
    proxy = TokenizerProxy(h.text_tokenizer)
    h.text_tokenizer = proxy

    # --- requests --------------------------------------------------------
    for label, over in CASES:
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            raise Stop()

        h.model.generate_audio = capture
        proxy.calls.clear()
        params = GenerationParams(thinking=False, shift=3.0, seed=42, use_cot_metas=False, use_cot_caption=False,
                                  use_cot_language=False, **over)
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[42])
        generate_music(h, None, params, config)
        assert captured, f"{label}: generate_audio was not reached"

        text_call = next(c for c in proxy.calls if c["max_length"] == 256)
        lyric_call = next(c for c in proxy.calls if c["max_length"] == 2048)
        src = captured["src_latents"][0]
        T = src.shape[0]
        refer = captured["refer_audio_acoustic_hidden_states_packed"]
        scalars = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                   for k, v in captured.items() if not isinstance(v, torch.Tensor)}
        case = {
            "params": {k: v for k, v in over.items()},
            "text_prompt": text_call["text"],
            "lyrics_prompt": lyric_call["text"],
            "latent_length": T,
            "src_is_silence": bool(torch.equal(src, silence[:T])),
            # As the DiT receives them: prepare_condition casts to the model
            # dtype. "auto" writes 2.0 into this bool tensor, so it is 1.0 here.
            "chunk_mask_dtype": str(captured["chunk_masks"].dtype),
            "chunk_mask_values": sorted(set(captured["chunk_masks"].float().flatten().tolist())),
            "chunk_mask_shape": list(captured["chunk_masks"].shape),
            "is_covers": captured["is_covers"].tolist(),
            "attention_mask_all_ones": bool(captured["attention_mask"].eq(1).all()) if captured.get("attention_mask") is not None else None,
            "text_mask_all_ones": bool(captured["text_attention_mask"].bool().all()),
            "lyric_mask_all_ones": bool(captured["lyric_attention_mask"].bool().all()),
            "refer_shape": list(refer.shape),
            "refer_is_silence750": bool(torch.equal(refer[0], silence[:750])),
            "refer_order_mask": captured["refer_audio_order_mask"].tolist(),
            "scalars": scalars,
        }
        manifest["cases"][label] = case
        dump(label + "_text_ids", text_call["ids"], np.int32)
        dump(label + "_lyric_ids", lyric_call["ids"], np.int32)
        dump(label + "_text_hidden", captured["text_hidden_states"][0], np.float32)
        dump(label + "_lyric_embeds", captured["lyric_hidden_states"][0], np.float32)
        print(f"{label}: T={T} text {len(text_call['ids'])} lyric {len(lyric_call['ids'])} tokens, "
              f"chunk {case['chunk_mask_values']}, covers {case['is_covers']}, seed {scalars.get('seed')}, "
              f"shift {scalars.get('shift')}, dcw {scalars.get('dcw_enabled')}", flush=True)

    # The noise a CPU run draws for seed 42 at the first case's length.
    T0 = manifest["cases"]["full_metas"]["latent_length"]
    ctx = torch.zeros(1, T0, 128)
    dump("noise_seed42", h.model.prepare_noise(ctx, [42])[0], np.float32)

    # --- the sampler, over a stub decoder --------------------------------
    h.model.generate_audio = real_generate
    real_decoder = h.model.decoder

    class StubDecoder(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c
            self.calls = []

        def forward(self, hidden_states, timestep, **kwargs):
            self.calls.append(float(timestep[0]))
            v = 0.5 * hidden_states + timestep.view(-1, 1, 1) * self.c
            return v, kwargs.get("past_key_values")

    base = manifest["cases"]["full_metas"]
    for T in (37, 40):
        c = torch.cos(torch.arange(T * 64, dtype=torch.float32) * 0.37).view(1, T, 64)
        dump(f"sampler_c_T{T}", c[0], np.float32)
        for shift in (1.0, 2.0, 3.0):
            for dcw in (False, True):
                stub = StubDecoder(c)
                h.model.decoder = stub
                steps = []
                orig_dcw = None
                kw = dict(
                    text_hidden_states=torch.zeros(1, 4, 1024), text_attention_mask=torch.ones(1, 4),
                    lyric_hidden_states=torch.zeros(1, 4, 1024), lyric_attention_mask=torch.ones(1, 4),
                    refer_audio_acoustic_hidden_states_packed=silence[:750].unsqueeze(0),
                    refer_audio_order_mask=torch.zeros(1, dtype=torch.long),
                    src_latents=silence[:T].unsqueeze(0), chunk_masks=torch.full((1, T, 64), 2.0),
                    is_covers=torch.zeros(1, dtype=torch.bool), silence_latent=h.silence_latent,
                    seed=[7], fix_nfe=8, infer_method="ode", shift=shift,
                    dcw_enabled=dcw, dcw_mode="double", dcw_scaler=0.05, dcw_high_scaler=0.02, dcw_wavelet="haar",
                )
                noise = h.model.prepare_noise(torch.zeros(1, T, 128), [7])
                out = h.model.generate_audio(**kw)["target_latents"][0]
                label = f"sampler_T{T}_s{int(shift)}_{'dcw' if dcw else 'plain'}"
                manifest["sampler"][label] = {"T": T, "shift": shift, "dcw": dcw, "timesteps": stub.calls}
                dump(label + "_noise", noise[0], np.float32)
                dump(label + "_out", out, np.float32)
                print(f"{label}: timesteps {['%.4f' % t for t in stub.calls]}", flush=True)
    h.model.decoder = real_decoder

    # --- FSQ ---------------------------------------------------------------
    q = h.model.tokenizer.quantizer
    idx = torch.tensor([0, 1, 7, 8, 63, 64, 511, 512, 12345, 40000, 63999], dtype=torch.long)
    out = q.get_output_from_indices(idx.view(1, -1, 1))[0]
    codes = q.get_codes_from_indices(idx.view(1, -1, 1))  # [q, b, n, d]
    manifest["fsq"] = {"levels": list(h.model.config.fsq_input_levels), "indices": idx.tolist()}
    dump("fsq_indices", idx, np.int32)
    dump("fsq_codes", codes[0, 0], np.float32)
    dump("fsq_out", out, np.float32)

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
