"""MiniMax-Music3's flow-matching stage and vocoder, for
research/music-minimax.md N4-N6: what mm3's DiT, window sampler and vocoder
are gated against.

The frame hiddens are made the way the pipeline makes them, by the global
and local LMs (bf16, the reference's CUDA dtype, which is also 2x faster on
the CPU) for FRAMES frames of N0c's h3 from seed 11, and kept as
[frames, 8, 4096] fp32. Then, in fp32, the denoise blocks of
diffusers #14456 (dafe3733, denoise.py) are run by hand for every window of
those frames -- two windows of 200 frames at a hop of 100 -- with a seeded
generator, recording per window:

  - condition: the condition encoder's [L, 2048] (with the previous
    window's over the overlap, as ChunkConditionStep splices it);
  - noise: the initial latents [128, L];
  - lat<i>: the latents each step's transformer reads (after the overlap
    blend), v<i>: the guided velocity it gives (CFG 1.7 against a zero
    condition), and the final latents;
  - out_c0 / out_u0: step 0's conditional and unconditional predictions,
    the DiT's own gate (N4);
  - wave: the vocoder's [2, samples] of the final latents (N6);

and then the cropped, concatenated song, as decoders.py builds it. The
scheduler's timesteps and sigmas, and the largest |activation| into every
block's ff_out (the fp16 range check), go in the manifest.

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv-mm3/bin/python -I \\
        reference/dump_mm3_dit.py

Writes reference/out/mm3dit/<name>.bin (raw little-endian) and manifest.json.
"""

import gc
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)
from diffusers.modular_pipelines.minimax_music3 import decoders as DEC
from diffusers.modular_pipelines.minimax_music3 import denoise as DN
from diffusers.modular_pipelines.minimax_music3 import encoders as E
from diffusers.modular_pipelines.minimax_music3.before_denoise import _CHUNK_FRAMES, _CHUNK_HOP
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models", "MiniMax-Music3")
OUT = os.path.join(ROOT, "reference", "out", "mm3dit")
sys.path.insert(0, os.path.join(ROOT, "reference"))
from dump_mm3 import prompt_ids  # noqa: E402

FRAMES = 300
SEED = 11
STEPS = 30
GUIDANCE = 1.7

tensors = {}


def save(name, a):
    a = np.ascontiguousarray(a)
    a.tofile(os.path.join(OUT, name + ".bin"))
    tensors[name] = {"shape": list(a.shape), "dtype": str(a.dtype)}


def frames_bf16():
    """The pipeline's semantic stage in bf16: frame_hiddens [frames, 8*4096]."""
    tok = Qwen2Tokenizer.from_pretrained(os.path.join(MODELS, "tokenizer"))
    ids = prompt_ids(tok)
    lm = Qwen3ForCausalLM.from_pretrained(os.path.join(MODELS, "language_model"), torch_dtype=torch.bfloat16).eval()
    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(os.path.join(MODELS, "rvq_depth_decoder"),
                                                         torch_dtype=torch.bfloat16).eval()
    comp = SimpleNamespace(language_model=lm, rvq_depth_decoder=depth, num_codebooks=8, audio_vocab_size=1024)
    gen = torch.Generator("cpu").manual_seed(SEED)
    out = lm.model(inputs_embeds=lm.model.embed_tokens(ids), use_cache=True)
    pkv, last = out.past_key_values, out.last_hidden_state[:, -1]
    mask = torch.ones(lm.config.vocab_size, dtype=torch.bool)
    mask[E._AUDIO_CODE_OFFSET:E._AUDIO_CODE_OFFSET + E._SEMANTIC_VOCAB_SIZE] = False
    mask[E._AUDIO_END_TOKEN_ID] = False
    hiddens = []
    t0 = time.time()
    for f in range(FRAMES + 1):
        logits = lm.lm_head(last).float().masked_fill(mask, -float("inf"))
        c, u = logits[0:1], logits[1:2]
        g = u + (c - u) * E._AR_CFG_SCALE
        thr = torch.topk(c, E._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        g = g.masked_fill(c < thr, -float("inf")).masked_fill(mask.unsqueeze(0), -float("inf"))
        s = E._sample_top_k(g, gen)
        if int(s.item()) == E._AUDIO_END_TOKEN_ID:
            sys.exit(f"end of audio at frame {f}")
        fc, dh = E._generate_depth_codes(comp, last, (s - E._AUDIO_CODE_OFFSET).repeat(2), gen)
        if f > 0:
            hiddens.append(torch.cat((last[:1], dh), dim=-1)[0].float())
        out = lm.model(inputs_embeds=E._embed_audio_frame(comp, fc), past_key_values=pkv, use_cache=True)
        pkv, last = out.past_key_values, out.last_hidden_state[:, -1]
        if f % 50 == 0:
            print(f"frame {f} ({time.time() - t0:.0f} s)", flush=True)
    return torch.stack(hiddens)


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    fh = frames_bf16()
    save("frame_hiddens", fh.view(FRAMES, 8, -1).numpy())
    gc.collect()

    enc = MiniMaxMusic3ConditionEncoder.from_pretrained(os.path.join(MODELS, "condition_encoder"),
                                                        torch_dtype=torch.float32).eval()
    dit = MiniMaxMusic3Transformer1DModel.from_pretrained(os.path.join(MODELS, "transformer"),
                                                          torch_dtype=torch.float32).eval()
    voc = MiniMaxMusic3Vocoder.from_pretrained(os.path.join(MODELS, "vocoder"), torch_dtype=torch.float32).eval()
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(MODELS, "scheduler"))

    # The largest |activation| into each block's ff_out, over every forward.
    ffmax = [0.0] * len(dit.transformer_blocks)

    def hook(i):
        def f(m, inp):
            ffmax[i] = max(ffmax[i], float(inp[0].abs().max()))
        return f

    for i, b in enumerate(dit.transformer_blocks):
        b.ff_out.register_forward_pre_hook(hook(i))

    fh = fh.unsqueeze(0)
    starts = [0] if FRAMES <= _CHUNK_FRAMES else list(range(0, FRAMES - _CHUNK_HOP, _CHUNK_HOP))
    gen = torch.Generator("cpu").manual_seed(SEED)
    prev_lat, prev_cond, chunks = None, None, []
    t0 = time.time()
    for k, start in enumerate(starts):
        end = min(start + _CHUNK_FRAMES, FRAMES)
        cond = enc(fh[:, start:end])
        overlap = 0
        if prev_lat is not None:
            overlap = min(prev_lat.shape[-1], cond.shape[1])
            cond[:, :overlap] = prev_cond[:, :overlap]
        lat = randn_tensor((1, 128, cond.shape[1]), generator=gen, device=torch.device("cpu"), dtype=cond.dtype)
        save(f"w{k}_condition", cond[0].numpy())
        save(f"w{k}_noise", lat[0].numpy())
        noise_prompt = lat[..., :overlap].clone() if overlap > 0 else None
        sigmas = np.linspace(1.0, 1.0 / STEPS, STEPS)
        sched.set_timesteps(sigmas=sigmas, device="cpu")
        ts = sched.timesteps
        if k == 0:
            sched_info = {"timesteps": ts.tolist(), "sigmas": sched.sigmas.tolist()}
        zeros = torch.zeros_like(cond)
        for i, t in enumerate(ts):
            if overlap > 0:
                tv = t.to(lat.dtype)
                lat[..., :overlap] = (1.0 - (1.0 - 1e-6) * tv) * noise_prompt + tv * prev_lat[..., :overlap]
            save(f"w{k}_lat{i}", lat[0].numpy())
            tt = t.expand(1).to(lat.dtype)
            c = dit(hidden_states=lat, timestep=tt, encoder_hidden_states=cond, return_dict=False)[0]
            u = dit(hidden_states=lat, timestep=tt, encoder_hidden_states=zeros, return_dict=False)[0]
            if i == 0:
                save(f"w{k}_out_c0", c[0].numpy())
                save(f"w{k}_out_u0", u[0].numpy())
            v = u + GUIDANCE * (c - u)
            save(f"w{k}_v{i}", v[0].numpy())
            lat = sched.step(v, t, lat, return_dict=False)[0]
            print(f"window {k} step {i} ({time.time() - t0:.0f} s)", flush=True)
        if overlap > 0:
            lat[..., :overlap] = prev_lat[..., :overlap]
        save(f"w{k}_final", lat[0].numpy())
        os_ = max(0, lat.shape[-1] - 2 * DN._OVERLAP_LATENT_LENGTH)
        oe = max(os_, lat.shape[-1] - DN._OVERLAP_LATENT_LENGTH)
        prev_lat, prev_cond = lat[..., os_:oe], cond[:, os_:oe]
        chunks.append(lat)

    waves = []
    for k, lat in enumerate(chunks):
        w = voc(lat)
        save(f"w{k}_wave", w[0].float().numpy())
        left = 0 if k == 0 else DEC._CROP_LEFT_LATENT * 512
        right = 0 if k == len(chunks) - 1 else DEC._CROP_RIGHT_LATENT * 512
        waves.append(w[..., left:w.shape[-1] - right])
    song = torch.cat(waves, dim=-1).float().clamp(-1.0, 1.0)
    save("song", song[0].numpy())
    print(f"song: {song.shape[-1] / 44100:.2f} s; ff_out input max {max(ffmax):.1f}", flush=True)

    json.dump({"frames": FRAMES, "seed": SEED, "steps": STEPS, "guidance": GUIDANCE, "starts": starts,
               "overlap_latents": DN._OVERLAP_LATENT_LENGTH, "crop_left": DEC._CROP_LEFT_LATENT,
               "crop_right": DEC._CROP_RIGHT_LATENT, "scheduler": sched_info, "ff_out_max": ffmax,
               "tensors": tensors}, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
