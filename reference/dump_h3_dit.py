"""Dump MiniMax-H3's whole transformer over a short t2va run, for VIDEO.md M4.

The whole-stack oracle the GPU port (M7) is gated against, in fp32. The
transformer in fp32 is 132 GB, so it is never built whole. Each block is
loaded on its own, and its AdaLN projection (a quarter of its bytes) is
applied at once to every timestep the run will use and then dropped. What
remains is ~80 GB fp32. The per-step tables replace the projection through a
module that returns the same six chunks `MiniMaxH3AdaLayerNormModulation`
would. The rest is diffusers' own modules, run in the order
`MiniMaxH3Transformer3DModel.forward` runs them (reproduced line for line, as
in dump_h3_dit_block.py), and stepped by diffusers' own scheduler.

The run is a t2va at 256x448 x 124 frames (37 latent frames, 207 audio
latents per channel) conditioned on M2's fp32 `readme` embedding (537
tokens): 5,095 rows. N = 8 (7 forwards). The noise is drawn here and saved.

Dumped:
  * every forward's video and audio velocity, and the latents after each step;
  * forward 0 block by block: the input to blocks 0, 1, 2, 25 and 49 and each
    one's output, so a GPU block can be teacher-forced anywhere in the stack;
  * forward 0, every block: absmax of the residual and of every GEMM input
    and output (the fp16 question, M-o1);
  * every block's AdaLN table for every forward (`adaln{i}_f{f}`, six [T*3, H]
    chunks; T is 1 at forward 0, where video and audio both start at t = 0),
    so the Go side can check the tables it builds against the ones used here.

    .venv/bin/python reference/dump_h3_dit.py   (~85 GB RSS; ~1-3 min a forward)
"""

import json
import os
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
    patchify_video_latents,
)
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from safetensors import safe_open

MODEL = "models/MiniMax-H3"
OUT = "reference/out/h3dit"
TEXTENC = "reference/out/h3textenc"
HEIGHT, WIDTH, FRAMES = 256, 448, 124
STEPS = 8
SEED = 42
KEEP_BLOCKS = (0, 1, 2, 25, 49)


class TableModulation(torch.nn.Module):
    """Stands in for adaln_proj: returns the precomputed chunks of the current forward."""

    def __init__(self, tables):
        super().__init__()
        self.tables = tables  # forward index -> six [T*3, H] tensors
        self.current = 0

    def forward(self, temb):
        return self.tables[self.current]


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    cfg = {k: v for k, v in json.load(open(f"{MODEL}/transformer/config.json")).items() if not k.startswith("_")}
    wmap = json.load(open(f"{MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    handles = {}

    def tensor(key):
        shard = wmap[key]
        if shard not in handles:
            handles[shard] = safe_open(f"{MODEL}/transformer/{shard}", "pt")
        return handles[shard].get_tensor(key).float()

    def load(module, prefix, skip=()):
        state = {n: tensor(prefix + n) for n in module.state_dict() if not n.startswith(skip)}
        module.load_state_dict(state, strict=not skip)
        return module.float().eval()

    manifest = {"height": HEIGHT, "width": WIDTH, "frames": FRAMES, "steps": STEPS, "seed": SEED,
                "keep_blocks": list(KEEP_BLOCKS), "tensors": {}, "block_stats": [], "forwards": []}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        if t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {"shape": list(t.shape), "count": int(flat.numel()),
                                     "sum": float(flat.double().sum()), "absmax": float(flat.abs().max())}

    # --- the request: layout, schedules, noise, conditioning ----------------
    lf = (FRAMES - 5) // 17 * 5 + 2
    lh, lw = HEIGHT // 16, WIDTH // 16
    audio_latents = int(round(FRAMES / 24 * 40))
    tmeta = json.load(open(f"{TEXTENC}/manifest.json"))["tensors"]["readme_fp32"]
    text = torch.from_numpy(np.fromfile(f"{TEXTENC}/readme_fp32.bin", dtype=np.float32).reshape(tmeta["shape"]))[None]
    ntext = text.shape[1]
    pos, tags, vidx, aidx, tidx, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.full((ntext,), 1, dtype=torch.long), lf, lh, lw, audio_latents, (1, 2, 2), 2, 2, 0, ())
    vs = MiniMaxH3Scheduler.from_pretrained(f"{MODEL}/scheduler")
    aus = MiniMaxH3Scheduler.from_pretrained(f"{MODEL}/audio_scheduler")
    vs.set_timesteps(STEPS)
    aus.set_timesteps(STEPS)
    plans = [MiniMaxH3SetTimestepsStep.build_row_timesteps(vidx, aidx, 0, 0, ntext, float(t), float(ta), 0.999, 1.0)
             for t, ta in zip(vs.timesteps, aus.timesteps)]
    gen = torch.Generator().manual_seed(SEED)
    noise = torch.randn((1, 24, lf, lh, lw), generator=gen)
    latents = patchify_video_latents(noise, (1, 2, 2))
    audio = torch.randn((audio_latents * 2, cfg["audio_in_channels"]), generator=gen)
    manifest.update(text_tokens=ntext, latent_frames=lf, latent_height=lh, latent_width=lw,
                    audio_latents=audio_latents, rows=int(pos.shape[0]))
    dump("noise_video", latents)
    dump("noise_audio", audio)
    print(f"layout: {pos.shape[0]} rows ({ntext} text, {aidx.numel()} audio, {vidx.numel()} video)", flush=True)

    # --- the model, block by block -----------------------------------------
    H, E = cfg["hidden_size"], cfg["norm_eps"]
    patch = cfg["in_channels"] * 4
    time_proj = Timesteps(num_channels=cfg["freq_dim"], flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedder = load(TimestepEmbedding(cfg["freq_dim"], cfg["time_embed_hidden_dim"], out_dim=cfg["time_embed_dim"]), "time_embedder.")
    proj_in = load(torch.nn.Linear(patch, H), "proj_in.")
    audio_proj_in = load(torch.nn.Linear(cfg["audio_in_channels"], H), "audio_proj_in.")
    context_embedder = load(torch.nn.Linear(cfg["text_dim"], H), "context_embedder.")
    refiner = load(MiniMaxH3TokenRefiner(H, cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["ffn_dim"],
                                         cfg["num_refiner_layers"], E, cfg["qk_norm_eps"], cfg["final_norm_eps"]), "token_refiner.")
    norm_out = load(MiniMaxH3AdaLayerNormOut(H, cfg["time_embed_dim"], cfg["final_norm_eps"]), "norm_out.")
    proj_out = load(torch.nn.Linear(H, patch), "proj_out.")
    audio_proj_out = load(torch.nn.Linear(H, cfg["audio_in_channels"]), "audio_proj_out.")
    rope = MiniMaxH3RotaryPosEmbed(cfg["rope_freq_dim"], cfg["rope_theta"])
    tembs = [time_embedder(time_proj(u)) for u, _ in plans]

    blocks = []
    t0 = time.time()
    for i in range(cfg["num_layers"]):
        block = MiniMaxH3TransformerBlock(H, cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["ffn_dim"],
                                          cfg["time_embed_dim"], E, cfg["qk_norm_eps"])
        # The block's own projection, in fp32, over every forward's temb; then gone.
        proj = block.adaln_proj
        proj.linear.weight = torch.nn.Parameter(tensor(f"transformer_blocks.{i}.adaln_proj.linear.weight"))
        proj.linear.bias = torch.nn.Parameter(tensor(f"transformer_blocks.{i}.adaln_proj.linear.bias"))
        tables = [tuple(c.clone() for c in proj.float()(temb)) for temb in tembs]
        for f, tab in enumerate(tables):
            dump(f"adaln{i}_f{f}", torch.stack(tab))
        block.adaln_proj = TableModulation(tables)
        del proj
        load(block, f"transformer_blocks.{i}.", skip=("adaln_proj",))
        blocks.append(block)
        if i % 10 == 9:
            print(f"  loaded {i + 1} blocks in {time.time() - t0:.0f}s", flush=True)

    # --- the run -------------------------------------------------------------
    rotary_emb = rope(pos)
    text_embeds = refiner(context_embedder(text))
    dump("text_refined", text_embeds)
    for step, (u, rowt) in enumerate(plans):
        t0 = time.time()
        stats = []
        hidden = text_embeds.new_zeros((1, pos.shape[0], H))
        hidden = hidden.index_copy(1, tidx, text_embeds)
        hidden = hidden.index_copy(1, vidx, proj_in(latents[None]))
        hidden = hidden.index_copy(1, aidx, audio_proj_in(audio[None]))
        adaln_indices = rowt * MINIMAX_H3_MODALITY_NUM + tags
        for i, block in enumerate(blocks):
            block.adaln_proj.current = step
            hooks = []
            if step == 0:
                row = {"block": i, "in": float(hidden.abs().max())}
                if i in KEEP_BLOCKS:
                    dump(f"f0_block{i}_in", hidden)

                def watch(name):
                    def hook(m, args, out):
                        row[name + "_in"] = max(row.get(name + "_in", 0.0), float(args[0].abs().max()))
                        row[name + "_out"] = max(row.get(name + "_out", 0.0), float(out.abs().max()))
                    return hook
                for name, mod in (("q", block.attn.to_q), ("k", block.attn.to_k), ("v", block.attn.to_v),
                                  ("o", block.attn.to_out[0]), ("up", block.ff.net[0].proj), ("down", block.ff.net[2])):
                    hooks.append(mod.register_forward_hook(watch(name)))
            hidden = block(hidden, tembs[step], adaln_indices, rotary_emb)
            for h in hooks:
                h.remove()
            if step == 0:
                row["out"] = float(hidden.abs().max())
                stats.append(row)
                if i in KEEP_BLOCKS:
                    dump(f"f0_block{i}_out", hidden)
        normed = norm_out(hidden, tembs[step], rowt)
        v = proj_out(normed).index_select(1, vidx)[0]
        a = audio_proj_out(normed).index_select(1, aidx)[0]
        dump(f"f{step}_v_video", v)
        dump(f"f{step}_v_audio", a)
        latents = vs.step(v.float(), vs.timesteps[step], latents, return_dict=False)[0]
        audio = aus.step(a.float(), aus.timesteps[step], audio, return_dict=False)[0]
        dump(f"f{step}_latents", latents)
        dump(f"f{step}_audio", audio)
        if step == 0:
            manifest["block_stats"] = stats
        manifest["forwards"].append({"timestep": float(vs.timesteps[step]), "audio_timestep": float(aus.timesteps[step]),
                                     "unique": u.tolist(), "seconds": time.time() - t0})
        print(f"forward {step}: {time.time() - t0:.0f}s, |v| {v.abs().max():.4g}", flush=True)
        with open(os.path.join(OUT, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
