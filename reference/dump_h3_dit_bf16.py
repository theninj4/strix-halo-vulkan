"""The official bf16 transformer's distance from the fp32 oracle, for VIDEO.md M11a.

What the int8 bank is priced against. `dump_h3_dit.py` is the fp32 oracle;
this runs the same 7 forwards of the same N = 8 t2va, **teacher-forced** (each
forward starts from that oracle's latents, as TestGPURun does), with the
transformer at the precision the released pipeline runs it: diffusers'
`torch_dtype=torch.bfloat16`, whose `_keep_in_fp32_modules` leaves the patch
projections, the timestep MLP and the output heads fp32 and puts everything
else, the AdaLN projections included, in bf16. The AdaLN projections are
applied once per block to every forward's timestep embedding and dropped, as
in the oracle, but here in bf16 on the fp32 embedding, which is what
`MiniMaxH3AdaLayerNormModulation` does in that pipeline.

Dumped (reference/out/h3dit_bf16): every forward's velocities and the latents
one scheduler step from the oracle's, and the same rel/rms the Go gates print.

`--free` runs the same 7 forwards free-running from the oracle's noise
instead (reference/out/h3dit_bf16_free): how far the released pipeline's own
run ends from the fp32 one, which is the scale TestE2E's free-running gate is
priced on for a bank that is not fp16.

    .venv/bin/python reference/dump_h3_dit_bf16.py [--free]   (~45 GB RSS)
"""

import json
import os
import sys
import time

import numpy as np
import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TokenRefiner,
    MiniMaxH3TransformerBlock,
)
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from safetensors import safe_open

MODEL = "models/MiniMax-H3"
ORACLE = "reference/out/h3dit"
FREE = "--free" in sys.argv
OUT = "reference/out/h3dit_bf16" + ("_free" if FREE else "")
TEXTENC = "reference/out/h3textenc"
BF16 = torch.bfloat16


class TableModulation(torch.nn.Module):
    def __init__(self, tables):
        super().__init__()
        self.tables = tables
        self.current = 0

    def forward(self, temb):
        return self.tables[self.current]


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    om = json.load(open(f"{ORACLE}/manifest.json"))
    cfg = {k: v for k, v in json.load(open(f"{MODEL}/transformer/config.json")).items() if not k.startswith("_")}
    wmap = json.load(open(f"{MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    handles = {}

    def tensor(key, dtype):
        shard = wmap[key]
        if shard not in handles:
            handles[shard] = safe_open(f"{MODEL}/transformer/{shard}", "pt")
        return handles[shard].get_tensor(key).to(dtype)

    def load(module, prefix, dtype, skip=()):
        state = {n: tensor(prefix + n, dtype) for n in module.state_dict() if not n.startswith(skip)}
        module.load_state_dict(state, strict=not skip)
        return module.to(dtype).eval()

    def oracle(name):
        meta = om["tensors"][name]
        return torch.from_numpy(np.fromfile(f"{ORACLE}/{name}.bin", dtype=np.float32).reshape(meta["shape"]))

    manifest = {"tensors": {}, "forwards": []}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        if t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        manifest["tensors"][name] = {"shape": list(t.shape), "count": int(t.numel())}

    def gap(got, want):
        # h3/dit's relGap: max |d| over max |want|, and rms(d) over rms(want).
        got, want = got.double().flatten(), want.double().flatten()
        d = got - want
        return float(d.abs().max() / want.abs().max()), float(d.pow(2).mean().sqrt() / want.pow(2).mean().sqrt())

    # --- the request, as dump_h3_dit.py builds it ---------------------------
    lf, lh, lw = om["latent_frames"], om["latent_height"], om["latent_width"]
    audio_latents = om["audio_latents"]
    tmeta = json.load(open(f"{TEXTENC}/manifest.json"))["tensors"]["readme_fp32"]
    text = torch.from_numpy(np.fromfile(f"{TEXTENC}/readme_fp32.bin", dtype=np.float32).reshape(tmeta["shape"]))[None]
    ntext = text.shape[1]
    pos, tags, vidx, aidx, tidx, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.full((ntext,), 1, dtype=torch.long), lf, lh, lw, audio_latents, (1, 2, 2), 2, 2, 0, ())
    vs = MiniMaxH3Scheduler.from_pretrained(f"{MODEL}/scheduler")
    aus = MiniMaxH3Scheduler.from_pretrained(f"{MODEL}/audio_scheduler")
    vs.set_timesteps(om["steps"])
    aus.set_timesteps(om["steps"])
    plans = [MiniMaxH3SetTimestepsStep.build_row_timesteps(vidx, aidx, 0, 0, ntext, float(t), float(ta), 0.999, 1.0)
             for t, ta in zip(vs.timesteps, aus.timesteps)]

    # --- the model: bf16 but for _keep_in_fp32_modules ----------------------
    H, E = cfg["hidden_size"], cfg["norm_eps"]
    patch = cfg["in_channels"] * 4
    f32 = torch.float32
    time_proj = Timesteps(num_channels=cfg["freq_dim"], flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedder = load(TimestepEmbedding(cfg["freq_dim"], cfg["time_embed_hidden_dim"], out_dim=cfg["time_embed_dim"]), "time_embedder.", f32)
    proj_in = load(torch.nn.Linear(patch, H), "proj_in.", f32)
    audio_proj_in = load(torch.nn.Linear(cfg["audio_in_channels"], H), "audio_proj_in.", f32)
    proj_out = load(torch.nn.Linear(H, patch), "proj_out.", f32)
    audio_proj_out = load(torch.nn.Linear(H, cfg["audio_in_channels"]), "audio_proj_out.", f32)
    context_embedder = load(torch.nn.Linear(cfg["text_dim"], H), "context_embedder.", BF16)
    refiner = load(MiniMaxH3TokenRefiner(H, cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["ffn_dim"],
                                         cfg["num_refiner_layers"], E, cfg["qk_norm_eps"], cfg["final_norm_eps"]), "token_refiner.", BF16)
    norm_out = load(MiniMaxH3AdaLayerNormOut(H, cfg["time_embed_dim"], cfg["final_norm_eps"]), "norm_out.", BF16)
    rope = MiniMaxH3RotaryPosEmbed(cfg["rope_freq_dim"], cfg["rope_theta"])
    tembs = [time_embedder(time_proj(u)) for u, _ in plans]

    blocks = []
    t0 = time.time()
    for i in range(cfg["num_layers"]):
        block = MiniMaxH3TransformerBlock(H, cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["ffn_dim"],
                                          cfg["time_embed_dim"], E, cfg["qk_norm_eps"])
        load(block, f"transformer_blocks.{i}.", BF16)
        tables = [tuple(c.clone() for c in block.adaln_proj(temb)) for temb in tembs]
        block.adaln_proj = TableModulation(tables)
        blocks.append(block)
        if i % 10 == 9:
            print(f"  loaded {i + 1} blocks in {time.time() - t0:.0f}s", flush=True)

    # --- 7 forwards, teacher-forced ------------------------------------------
    rotary_emb = rope(pos)
    text_embeds = refiner(context_embedder(text.to(BF16)))
    print("refiner rel %.3g rms %.3g" % gap(text_embeds, oracle("text_refined")), flush=True)
    latents, audio = oracle("noise_video"), oracle("noise_audio")
    for step, (u, rowt) in enumerate(plans):
        t0 = time.time()
        if not FREE:
            latents = oracle("noise_video" if step == 0 else f"f{step - 1}_latents")
            audio = oracle("noise_audio" if step == 0 else f"f{step - 1}_audio")
        video_embeds = proj_in(latents[None])
        audio_embeds = audio_proj_in(audio[None])
        hidden = text_embeds.new_zeros((1, pos.shape[0], H))
        hidden = hidden.index_copy(1, tidx, text_embeds)
        hidden = hidden.index_copy(1, vidx, video_embeds.to(BF16))
        hidden = hidden.index_copy(1, aidx, audio_embeds.to(BF16))
        adaln_indices = rowt * MINIMAX_H3_MODALITY_NUM + tags
        for block in blocks:
            block.adaln_proj.current = step
            hidden = block(hidden, tembs[step], adaln_indices, rotary_emb)
        normed = norm_out(hidden, tembs[step], rowt).to(f32)
        v = proj_out(normed).index_select(1, vidx)[0]
        a = audio_proj_out(normed).index_select(1, aidx)[0]
        dump(f"f{step}_v_video", v)
        dump(f"f{step}_v_audio", a)
        latents = vs.step(v.float(), vs.timesteps[step], latents, return_dict=False)[0]
        audio = aus.step(a.float(), aus.timesteps[step], audio, return_dict=False)[0]
        dump(f"f{step}_latents", latents)
        dump(f"f{step}_audio", audio)
        row = {"seconds": time.time() - t0}
        for key, got, name in (("v_video", v, f"f{step}_v_video"), ("v_audio", a, f"f{step}_v_audio"),
                               ("latents", latents, f"f{step}_latents"), ("audio", audio, f"f{step}_audio")):
            row[key] = gap(got, oracle(name))
        manifest["forwards"].append(row)
        print(f"forward {step} ({row['seconds']:.0f}s): velocity video rel %.3g rms %.3g, audio rel %.3g rms %.3g; "
              "latents video rel %.3g rms %.3g, audio rel %.3g rms %.3g" % (*row["v_video"], *row["v_audio"], *row["latents"], *row["audio"]),
              flush=True)
        with open(os.path.join(OUT, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    main()
