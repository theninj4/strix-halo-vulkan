"""Dump MiniMax-H3's transformer front, first two blocks and tail, for VIDEO.md M3.

The whole transformer in fp32 is 132 GB and does not fit, so this never
builds it. It constructs the pieces a forward pass is made of — the timestep
MLP, the three input projections and the text refiner, two blocks with their
AdaLN projections, and the output norm and heads — from diffusers' own
modules, loads only their tensors (fp32) from the shards, and runs them in
the order MiniMaxH3Transformer3DModel.forward does. The forward's body is
reproduced line for line below, so a divergence between it and the library
would show up as a diff against transformer_minimax_h3.py, not as a number.

The layout is a real one from the layout step: 64 text rows, a 256x448
canvas at 7 latent frames and 40 audio latents per channel, 928 rows. The
video and audio rows are seeded noise at a real step (n20, forward 5); the
text rows stand in for the encoder's output with seeded noise at the scale
M2 measures, which is enough for a block gate — M4 runs the real encoder
output through the whole stack.

    .venv/bin/python reference/dump_h3_dit_block.py
"""

import json
import os

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
OUT = "reference/out/h3block"
TEXT, LF, LH, LW, AUDIO = 64, 7, 16, 28, 40
STEPS, FORWARD = 20, 5
BLOCKS = 2


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    cfg = {k: v for k, v in json.load(open(f"{MODEL}/transformer/config.json")).items() if not k.startswith("_")}
    wmap = json.load(open(f"{MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    handles = {}

    def load(module, prefix):
        state = {}
        for name in module.state_dict():
            key = prefix + name
            shard = wmap[key]
            if shard not in handles:
                handles[shard] = safe_open(f"{MODEL}/transformer/{shard}", "pt")
            state[name] = handles[shard].get_tensor(key).float()
        module.load_state_dict(state)
        return module.float().eval()

    H, E = cfg["hidden_size"], cfg["norm_eps"]
    time_proj = Timesteps(num_channels=cfg["freq_dim"], flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedder = load(TimestepEmbedding(cfg["freq_dim"], cfg["time_embed_hidden_dim"], out_dim=cfg["time_embed_dim"]), "time_embedder.")
    patch = cfg["in_channels"] * 4
    proj_in = load(torch.nn.Linear(patch, H), "proj_in.")
    audio_proj_in = load(torch.nn.Linear(cfg["audio_in_channels"], H), "audio_proj_in.")
    context_embedder = load(torch.nn.Linear(cfg["text_dim"], H), "context_embedder.")
    refiner = load(MiniMaxH3TokenRefiner(H, cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["ffn_dim"],
                                         cfg["num_refiner_layers"], E, cfg["qk_norm_eps"], cfg["final_norm_eps"]), "token_refiner.")
    blocks = [load(MiniMaxH3TransformerBlock(H, cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["ffn_dim"],
                                             cfg["time_embed_dim"], E, cfg["qk_norm_eps"]), f"transformer_blocks.{i}.")
              for i in range(BLOCKS)]
    norm_out = load(MiniMaxH3AdaLayerNormOut(H, cfg["time_embed_dim"], cfg["final_norm_eps"]), "norm_out.")
    proj_out = load(torch.nn.Linear(H, patch), "proj_out.")
    audio_proj_out = load(torch.nn.Linear(H, cfg["audio_in_channels"]), "audio_proj_out.")
    rope = MiniMaxH3RotaryPosEmbed(cfg["rope_freq_dim"], cfg["rope_theta"])

    text_tags = torch.full((TEXT,), 1, dtype=torch.long)
    pos, tags, vidx, aidx, tidx, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_tags, LF, LH, LW, AUDIO, (1, 2, 2), 2, 2, 0, ())
    vs = MiniMaxH3Scheduler.from_pretrained(f"{MODEL}/scheduler")
    aus = MiniMaxH3Scheduler.from_pretrained(f"{MODEL}/audio_scheduler")
    vs.set_timesteps(STEPS)
    aus.set_timesteps(STEPS)
    timestep, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        vidx, aidx, 0, 0, TEXT, float(vs.timesteps[FORWARD]), float(aus.timesteps[FORWARD]), 0.999, 1.0)

    video = torch.randn(1, vidx.numel(), patch)
    audio = torch.randn(1, aidx.numel(), cfg["audio_in_channels"])
    text = torch.randn(1, TEXT, cfg["text_dim"]) * 4.0

    manifest = {"text": TEXT, "latent_frames": LF, "latent_height": LH, "latent_width": LW, "audio_latents": AUDIO,
                "steps": STEPS, "forward": FORWARD, "blocks": BLOCKS, "rows": int(pos.shape[0]),
                "timestep": timestep.tolist(), "tensors": {}}

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        if t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {"shape": list(t.shape), "count": int(flat.numel()),
                                     "sum": float(flat.double().sum()), "absmax": float(flat.abs().max())}
        print(f"  {name:22s} {str(list(t.shape)):16s} absmax={flat.abs().max():.4g}")

    dump("in_video", video)
    dump("in_audio", audio)
    dump("in_text", text)
    dump("timestep_indices", timestep_indices.float())

    with torch.no_grad():
        # --- MiniMaxH3Transformer3DModel.forward, the parts before the stack --
        rotary_emb = rope(pos)
        video_embeds = proj_in(video)
        audio_embeds = audio_proj_in(audio)
        text_embeds = context_embedder(text)
        dump("text_projected", text_embeds)
        text_embeds = refiner(text_embeds)
        dump("text_refined", text_embeds)
        hidden = text_embeds.new_zeros((1, pos.shape[0], H))
        hidden = hidden.index_copy(1, tidx, text_embeds)
        hidden = hidden.index_copy(1, vidx, video_embeds)
        hidden = hidden.index_copy(1, aidx, audio_embeds)
        dump("packed", hidden)
        temb = time_embedder(time_proj(timestep))
        dump("temb", temb)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + tags
        for i, block in enumerate(blocks):
            # The AdaLN table this block reads: six [T*3, H] tensors.
            table = torch.stack(block.adaln_proj(temb))
            dump(f"block{i}_adaln", table)
            hidden = block(hidden, temb, adaln_indices, rotary_emb)
            dump(f"block{i}_out", hidden)
        # --- the tail, over the last block's output ---------------------------
        normed = norm_out(hidden, temb, timestep_indices)
        dump("tail_normed", normed)
        dump("out_video", proj_out(normed).index_select(1, vidx))
        dump("out_audio", audio_proj_out(normed).index_select(1, aidx))

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
