"""Dump MiniMax-H3's request plan, for VIDEO.md M1: everything a request
decides before a weight is touched.

That is the canvas a ratio resolves to, the 17n+5 frame snap and the latent
counts, the packed `[text | keyframes | audio L, R | video]` layout (float64
(t, h, w) rotary coordinates, modality tags, row indices), both sigma grids
(video shift 12, audio shift 3), the per-row timestep plan of every step, the
fp32 RoPE cos/sin the transformer computes from the layout, and one Euler
step. All of it comes straight from diffusers' own functions, so the Go port
in h3/plan is gated bit-exactly against the library rather than against a
re-derivation.

No weights are loaded: the transformer's RoPE module is constructed from its
config, and the schedulers from their scheduler_config.json.

    .venv/bin/python reference/dump_h3_plan.py
"""

import json
import os

import numpy as np
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

OUT = "reference/out/h3plan"
MODEL = "models/MiniMax-H3"

# The released checkpoint's constants, as MiniMaxH3ModularPipeline reads them.
CANVAS_MULTIPLE = 32
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
SPATIAL = 16
PATCH = (1, 2, 2)
AUDIO_CHANNELS = 2
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
KEYFRAME_NOISE_AUG = 0.999

# Aspect ratios the canvas resolver is checked on, including the extremes it
# accepts and a keyframe's raw pixel size.
CANVASES = [(16, 9), (9, 16), (1, 1), (4, 3), (3, 4), (21, 9), (4, 1), (1, 4), (1920, 1080), (1000, 777)]

# Frame requests, including ones the snap moves and the 5 s / 15 s edges.
FRAMES = [120, 124, 125, 141, 244, 360, 362]

# (label, height, width, frames, text tokens, keyframe anchors). The small
# canvases are the ones the CPU oracles run at; one is the served default.
LAYOUTS = [
    ("t2va_256x448", 256, 448, 124, 37, ()),
    ("t2va_480x864", 480, 864, 124, 211, ()),
    ("t2va_768x1344_10s", 768, 1344, 243, 5, ()),
    ("t2va_9x16", 448, 256, 141, 64, ()),
    ("first_256x448", 256, 448, 124, 40, ("first",)),
    ("last_256x448", 256, 448, 345, 40, ("last",)),
    ("firstlast_320x320", 320, 320, 158, 50, ("first", "last")),
]

# (label, steps). Steps count sigma grid points, the terminal zero included.
SCHEDULES = [("n8", 8), ("n20", 20), ("n50", 50), ("n2", 2)]


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"canvases": [], "frames": [], "layouts": {}, "schedules": {}, "tensors": {}}

    def dump(name, t, dtype):
        t = t.detach().contiguous()
        arr = t.numpy().astype(dtype)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(arr.tobytes())
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": np.dtype(dtype).name}

    for aw, ah in CANVASES:
        h, w = resolve_canvas_size(aw, ah, CANVAS_MULTIPLE, SHORT_EDGE, MAX_PIXELS)
        manifest["canvases"].append({"aspect": [aw, ah], "height": h, "width": w})

    for n in FRAMES:
        aligned = align_num_frames(n, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK)
        manifest["frames"].append(
            {
                "requested": n,
                "aligned": aligned,
                "latent_frames": video_latent_num_frames(aligned, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK),
                "audio_latents": audio_latent_num_frames(aligned),
            }
        )

    rope = MiniMaxH3RotaryPosEmbed(
        **{k: json.load(open(f"{MODEL}/transformer/config.json"))[k] for k in ("rope_freq_dim", "rope_theta")}
    )
    dump("inv_freq", rope.inv_freq, np.float32)

    for label, height, width, frames, text, anchors in LAYOUTS:
        lf = video_latent_num_frames(frames, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK)
        lh, lw = height // SPATIAL, width // SPATIAL
        audio = audio_latent_num_frames(frames)
        # The vision block of a keyframe is tagged video inside the text rows;
        # a plain alternating pattern exercises that the tags pass through.
        text_tags = torch.full((text,), TEXT_TAG, dtype=torch.long)
        if anchors:
            text_tags[1:5] = VIDEO_TAG
        pos, tags, vidx, aidx, tidx, ncond_v, ncond_a = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_tags, lf, lh, lw, audio, PATCH, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG, anchors
        )
        cos, sin = rope(pos)
        manifest["layouts"][label] = {
            "height": height, "width": width, "frames": frames, "text_tokens": text,
            "anchors": list(anchors), "latent_frames": lf, "latent_height": lh, "latent_width": lw,
            "audio_latents": audio, "sequence_length": int(pos.shape[0]),
            "condition_video_rows": ncond_v, "condition_audio_rows": ncond_a,
        }
        dump(label + "_pos", pos, np.float64)
        dump(label + "_tags", tags, np.int32)
        dump(label + "_text_tags", text_tags, np.int32)
        dump(label + "_video_idx", vidx, np.int32)
        dump(label + "_audio_idx", aidx, np.int32)
        dump(label + "_text_idx", tidx, np.int32)
        dump(label + "_cos", cos, np.float32)
        dump(label + "_sin", sin, np.float32)
        print(f"{label}: {pos.shape[0]} rows, {lf} latent frames, {audio} audio latents")

    video = MiniMaxH3Scheduler.from_pretrained(f"{MODEL}/scheduler")
    audio_s = MiniMaxH3Scheduler.from_pretrained(f"{MODEL}/audio_scheduler")
    manifest["shift"] = video.shift
    manifest["audio_shift"] = audio_s.shift
    lay = manifest["layouts"]["first_256x448"]
    _, _, vidx, aidx, _, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.full((lay["text_tokens"],), TEXT_TAG, dtype=torch.long), lay["latent_frames"],
        lay["latent_height"], lay["latent_width"], lay["audio_latents"], PATCH, AUDIO_CHANNELS,
        AUDIO_TAG, VIDEO_TAG, ("first",),
    )
    for label, steps in SCHEDULES:
        video.set_timesteps(steps)
        audio_s.set_timesteps(steps)
        manifest["schedules"][label] = {"steps": steps, "forwards": int(video.timesteps.numel()),
                                        "audio_forwards": int(audio_s.timesteps.numel())}
        dump(label + "_sigmas", video.sigmas, np.float32)
        dump(label + "_timesteps", video.timesteps, np.float32)
        dump(label + "_audio_sigmas", audio_s.sigmas, np.float32)
        dump(label + "_audio_timesteps", audio_s.timesteps, np.float32)
        # The row plan of every step on the first-keyframe layout, whose
        # conditioning rows take the max(t, 0.999) branch.
        uniq, inv = [], []
        for t, ta in zip(video.timesteps, audio_s.timesteps):
            u, i = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                vidx, aidx, lay["condition_video_rows"], 0, lay["text_tokens"],
                float(t), float(ta), max(float(t), KEYFRAME_NOISE_AUG), 1.0,
            )
            uniq.append(torch.nn.functional.pad(u, (0, 4 - u.numel()), value=-1.0))
            inv.append(i)
        dump(label + "_plan_unique", torch.stack(uniq), np.float32)
        dump(label + "_plan_index", torch.stack(inv), np.int32)
        print(f"{label}: {video.timesteps.numel()} video / {audio_s.timesteps.numel()} audio forwards")

    # One step of each scheduler, over vectors Go reproduces exactly, at every
    # step index of the 8-point grid (both sides of sigma = 0.5).
    sample = torch.arange(96, dtype=torch.float32) / 16 - 3
    model_out = torch.cos(torch.arange(96, dtype=torch.float32) * 0.37)
    dump("step_sample", sample, np.float32)
    dump("step_model_out", model_out, np.float32)
    for name, sched in (("video", video), ("audio", audio_s)):
        sched.set_timesteps(8)
        outs = []
        x = sample.clone()
        for t in sched.timesteps:
            x = sched.step(model_out, t, x, return_dict=False)[0]
            outs.append(x.clone())
        dump(f"step_{name}_chain", torch.stack(outs), np.float32)

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
