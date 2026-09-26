"""Dump MiniMax-H3's fl2va conditioning path, for VIDEO.md M10.

A keyframe reaches the transformer twice, and this dumps both routes and what
joins them:

  * **the presentation**: `"<Picture i>: "` + a Qwen3-VL vision block per
    keyframe, then the prompt verbatim; the vision rows are tagged *video*.
    The conditioner runs the vision tower and injects its merged rows and
    deepstack features, as Qwen-Image's edits do (qimage/textenc).
  * **the anchor rows**: the keyframe through the video VAE's *encoder* (a
    causal CNN, which at one frame is a 2-D CNN on the last temporal tap),
    posterior-sampled under its own generator seeded 42, rounded to fp16,
    normalised, then noised to t = 0.999 with the request's generator and
    packed ahead of the generated video rows. They are never stepped.

The input is the keyframe of the README's reproducible fl2va request
(`models/MiniMax-H3/assets/fl2va_keyframe.png`, 1920x1080), at M4's oracle
shape: 256x448 x 124 frames, height and width given, so the keyframe is
stretched onto the canvas. A second keyframe, a 900x1080 portrait crop of the
first, is the `last` follower and exercises the cover-crop.

Phases, each its own run (argv), all writing reference/out/h3fl2va:

  prep  resize (stretch and cover-crop), the processor's pixel_values and
        grids, the presentations' ids and tags for [first] and [first, last].
        No weights.
  vae   the encoder stage by stage on the first 256x256 tile, the moments of
        both keyframes at 256x448 (two tiles), the posterior noise (seed 42),
        the sample and the normalised latents `encode_vae_condition` returns;
        and the first keyframe at 480x864 (15 tiles, widened overlaps).
  text  the vision tower in fp32 (merged + 3 deepstack) for both keyframes,
        and hidden_states[50] over the README prompt's presentation, official
        bf16 and the fp32 walk, for [first] and [first, last].
  dit   two forwards of an N = 8 fl2va run on the fp32 [first] conditioning:
        noise (condition, video, audio, in the generator's order), the
        condition rows, both forwards' velocities, latents after each step,
        and blocks 0 and 49's input/output at forward 0.

    .venv/bin/python reference/dump_h3_fl2va.py prep vae    (~12 GB)
    .venv/bin/python reference/dump_h3_fl2va.py text        (~75 GB, minutes)
    .venv/bin/python reference/dump_h3_fl2va.py dit         (~85 GB, ~10 min)
"""

import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

MODEL = "models/MiniMax-H3"
OUT = "reference/out/h3fl2va"
KEYFRAME = f"{MODEL}/assets/fl2va_keyframe.png"
HEIGHT, WIDTH, FRAMES = 256, 448, 124
BIG_H, BIG_W = 480, 864
STEPS, FORWARDS = 8, 2
SEED = 42
LAYER = 50
KEEP_BLOCKS = (0, 49)

manifest = {}


def load_manifest():
    global manifest
    path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(path)) if os.path.exists(path) else {"tensors": {}}


def save_manifest():
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


def dump(name, t, squeeze=True):
    t = torch.as_tensor(t).detach().contiguous().to(torch.float32)
    while squeeze and t.dim() > 2 and t.shape[0] == 1:
        t = t[0]
    with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
        fh.write(t.numpy().tobytes())
    flat = t.flatten()
    manifest["tensors"][name] = {"shape": list(t.shape), "count": int(flat.numel()),
                                 "sum": float(flat.double().sum()), "absmax": float(flat.abs().max())}
    print(f"  {name:28s} {str(list(t.shape)):22s} absmax={flat.abs().max():.4g}", flush=True)


def readme_prompt():
    body = open(f"{MODEL}/scripts/readme/reproducible-768p-fl2va-request.sh").read()
    body = body.split("<<'JSON'\n", 1)[1].split("\nJSON\n", 1)[0]
    return json.loads(body)["prompt"]


def keyframes():
    """The two keyframes put on the canvas as MiniMaxH3ResizeStep does."""
    from diffusers.image_processor import VaeImageProcessor

    proc = VaeImageProcessor(vae_scale_factor=16)
    first_src = Image.open(KEYFRAME).convert("RGB")
    last_src = first_src.crop((600, 0, 1500, 1080))

    def put(img, index, height, width):
        if img.size == (width, height):
            return img
        if index == 0:
            return proc.resize(img, height=height, width=width)
        scale = max(width / img.size[0], height / img.size[1])
        size = (max(width, round(img.size[0] * scale)), max(height, round(img.size[1] * scale)))
        left, top = max(0, (size[0] - width) // 2), max(0, (size[1] - height) // 2)
        return img.resize(size, Image.Resampling.LANCZOS).crop((left, top, left + width, top + height))

    return first_src, last_src, put


def pixels_of(img):
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)[None, :, None]


def presentation(tok, proc, prompt, frames):
    vision = proc.image_processor(images=frames, return_tensors="pt")
    grid = vision["image_grid_thw"]
    merge = proc.image_processor.merge_size ** 2
    ids, tags = [], []
    for i in range(len(frames)):
        label = tok(f"<Picture {i + 1}>: ", add_special_tokens=False)["input_ids"]
        n = int(grid[i].prod()) // merge
        block = ([tok.convert_tokens_to_ids("<|vision_start|>")] + [tok.convert_tokens_to_ids("<|image_pad|>")] * n
                 + [tok.convert_tokens_to_ids("<|vision_end|>")])
        ids += label + block
        tags += [1] * len(label) + [0] * len(block)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    return ids + p, tags + [1] * len(p), vision


def phase_prep():
    from transformers import AutoTokenizer, Qwen3VLProcessor

    first_src, last_src, put = keyframes()
    first = put(first_src, 0, HEIGHT, WIDTH)
    last = put(last_src, 1, HEIGHT, WIDTH)
    big = put(first_src, 0, BIG_H, BIG_W)
    for name, img in (("key_first", first), ("key_last", last), ("key_big", big)):
        arr = np.array(img)
        dump(name, torch.from_numpy(arr.astype(np.float32)), squeeze=False)
    manifest["keyframe_src"] = list(first_src.size)
    manifest["last_src"] = list(last_src.size)
    manifest.update(height=HEIGHT, width=WIDTH, frames=FRAMES, big=[BIG_H, BIG_W])

    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    proc = Qwen3VLProcessor.from_pretrained(MODEL, subfolder="processor")
    prompt = readme_prompt()
    manifest["prompt"] = prompt
    manifest["presentations"] = {}
    for label, frames in (("f", [first]), ("fl", [first, last])):
        ids, tags, vision = presentation(tok, proc, prompt, frames)
        manifest["presentations"][label] = {"ids": ids, "tags": tags, "grids": vision["image_grid_thw"].tolist()}
        print(f"{label}: {len(ids)} tokens, grids {vision['image_grid_thw'].tolist()}")
        if label == "fl":
            dump("pixel_values_fl", vision["pixel_values"])
    # The same resolve the ResizeStep does without height/width, for a few short edges.
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size

    manifest["auto_canvas"] = {str(se): list(resolve_canvas_size(*first_src.size, 32, se, 768 * 1344))
                               for se in (256, 480, 768)}
    manifest["auto_canvas_last"] = {str(se): list(resolve_canvas_size(*last_src.size, 32, se, 768 * 1344))
                                    for se in (256, 480, 768)}


def phase_vae():
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

    vae = AutoencoderKLMiniMaxH3.from_pretrained(f"{MODEL}/vae", torch_dtype=torch.float32).eval()
    mean_px, std_px = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    first_src, last_src, put = keyframes()
    frames = {"first": put(first_src, 0, HEIGHT, WIDTH), "last": put(last_src, 1, HEIGHT, WIDTH),
              "big": put(first_src, 0, BIG_H, BIG_W)}

    def normalised(img):
        px = pixels_of(img).float().div(255.0)
        return (px - torch.tensor(mean_px).view(1, -1, 1, 1, 1)) / torch.tensor(std_px).view(1, -1, 1, 1, 1)

    # --- one tile, stage by stage ------------------------------------------
    x = normalised(frames["first"])
    dump("first_pixels", x)
    tile = x[..., :256, :256]
    enc = vae.encoder
    caps = {}
    hooks = [enc.conv_in.register_forward_hook(lambda m, a, o: caps.__setitem__("conv_in", o))]
    for i, blk in enumerate(enc.down_blocks):
        hooks.append(blk.resnets[0].register_forward_hook(lambda m, a, o, i=i: caps.__setitem__(f"down{i}_r0", o)))
        hooks.append(blk.register_forward_hook(lambda m, a, o, i=i: caps.__setitem__(f"down{i}", o)))
    hooks.append(enc.norm_out.register_forward_hook(lambda m, a, o: caps.__setitem__("norm_out", o)))
    t0 = time.time()
    moments = vae.quant_conv(enc(tile))
    for h in hooks:
        h.remove()
    print(f"tile: {time.time() - t0:.1f}s")
    for k in ["conv_in"] + [f"down{i}_r0" for i in range(len(enc.down_blocks))] + \
             [f"down{i}" for i in range(len(enc.down_blocks))] + ["norm_out"]:
        dump(f"tile_{k}", caps[k][:, :, 0])
    dump("tile_moments", moments[:, :, 0])

    # --- whole keyframes ------------------------------------------------------
    gen_seed = 42
    for name, img in frames.items():
        x = normalised(img)
        if name != "first":
            dump(f"{name}_pixels", x)
        t0 = time.time()
        m = vae._encode(x)
        print(f"{name}: encode {time.time() - t0:.1f}s")
        dump(f"{name}_moments", m[:, :, 0])
        post = DiagonalGaussianDistribution(m)
        noise = torch.randn(post.mean.shape, generator=torch.Generator().manual_seed(gen_seed))
        dump(f"{name}_post_noise", noise[:, :, 0])
        dump(f"{name}_post_std", post.std[:, :, 0])
        sample = post.sample(generator=torch.Generator().manual_seed(gen_seed))
        dump(f"{name}_sample", sample[:, :, 0])
        official = encode_vae_condition(vae, pixels_of(img), mean_px, std_px, gen_seed)
        dump(f"{name}_latents", official[:, :, 0])
        # The same with the posterior mean in place of the sample, for scale:
        # how much the seed-42 draw moves the conditioning.
        lm = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        ls = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        mode = (post.mean.to(torch.float16).float() - lm) / ls
        gap = float((official - mode).abs().max())
        manifest.setdefault("sample_vs_mode", {})[name] = gap
        print(f"  {name}: sample vs mode absmax {gap:.4g}, std max {float(post.std.max()):.4g}")


def load_te():
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    proc = Qwen3VLProcessor.from_pretrained(MODEL, subfolder="processor")
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, subfolder="text_encoder", dtype=torch.bfloat16, attn_implementation="eager").eval()
    return tok, proc, te


def phase_text():
    from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds

    torch.set_grad_enabled(False)
    tok, proc, te = load_te()
    first_src, last_src, put = keyframes()
    first, last = put(first_src, 0, HEIGHT, WIDTH), put(last_src, 1, HEIGHT, WIDTH)
    prompt = readme_prompt()
    cases = {"f": [first], "fl": [first, last]}
    built = {}
    for label, frames in cases.items():
        ids, tags, vision = presentation(tok, proc, prompt, frames)
        if ids != manifest["presentations"][label]["ids"]:
            raise SystemExit(f"{label}: the presentation differs from prep's")
        built[label] = (ids, {"pixel_values": vision["pixel_values"], "image_grid_thw": vision["image_grid_thw"]})

    # --- official: the pipeline's own function, bf16 -----------------------
    for label, (ids, vis) in built.items():
        if f"{label}_bf16" in manifest["tensors"]:
            continue
        t0 = time.time()
        emb = get_qwen3vl_prompt_embeds(te, proc, ids, vis, text_encoder_layer=LAYER, dtype=torch.float32)
        dump(f"{label}_bf16", emb[0])
        print(f"{label}: official in {time.time() - t0:.0f}s")

    # --- the vision tower in fp32 -------------------------------------------
    visual = te.model.visual.float()
    ids, vis = built["fl"]
    vout = visual(vis["pixel_values"].float(), vis["image_grid_thw"], return_dict=True)
    dump("vis_merged_fl", vout.pooler_output)
    for i, f in enumerate(vout.deepstack_features):
        dump(f"vis_deepstack{i}_fl", f)

    # --- fp32: widen a layer at a time, stop after layer 49 -----------------
    text = te.model.language_model
    text.layers = text.layers[:LAYER]
    captured = {}
    def widen(module, args, kwargs):
        module.float()

    def narrow(module, args, out):
        module.to(torch.bfloat16)

    for layer in text.layers:
        layer.register_forward_pre_hook(widen, with_kwargs=True)
        layer.register_forward_hook(narrow)
    text.embed_tokens.register_forward_hook(lambda m, a, out: out.float())
    text.layers[LAYER - 1].register_forward_hook(
        lambda m, a, out: captured.update(last=(out[0] if isinstance(out, tuple) else out).clone()))
    for label, (ids, vis) in built.items():
        t0 = time.time()
        x = torch.tensor([ids])
        mm = torch.tensor(proc.create_mm_token_type_ids([ids]), dtype=torch.long)
        te.model(input_ids=x, attention_mask=torch.ones_like(x), mm_token_type_ids=mm, use_cache=False,
                 pixel_values=vis["pixel_values"].float(), image_grid_thw=vis["image_grid_thw"])
        if captured["last"].dtype != torch.float32:
            raise SystemExit("the fp32 walk did not take")
        dump(f"{label}_fp32", captured["last"][0])
        print(f"{label}: fp32 in {time.time() - t0:.0f}s")


class TableModulation(torch.nn.Module):
    def __init__(self, tables):
        super().__init__()
        self.tables = tables
        self.current = 0

    def forward(self, temb):
        return self.tables[self.current]


def phase_dit():
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

    # --- the request ----------------------------------------------------------
    lf = (FRAMES - 5) // 17 * 5 + 2
    lh, lw = HEIGHT // 16, WIDTH // 16
    audio_latents = int(round(FRAMES / 24 * 40))
    meta = manifest["tensors"]["f_fp32"]
    text = torch.from_numpy(np.fromfile(f"{OUT}/f_fp32.bin", dtype=np.float32).reshape(meta["shape"]))[None]
    tags_text = torch.tensor(manifest["presentations"]["f"]["tags"], dtype=torch.long)
    ntext = text.shape[1]
    pos, tags, vidx, aidx, tidx, ncond, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        tags_text, lf, lh, lw, audio_latents, (1, 2, 2), 2, 2, 0, ("first",))
    vs = MiniMaxH3Scheduler.from_pretrained(f"{MODEL}/scheduler")
    aus = MiniMaxH3Scheduler.from_pretrained(f"{MODEL}/audio_scheduler")
    vs.set_timesteps(STEPS)
    aus.set_timesteps(STEPS)
    plans = [MiniMaxH3SetTimestepsStep.build_row_timesteps(vidx, aidx, ncond, 0, ntext, float(t), float(ta),
                                                           max(float(t), 0.999), 1.0)
             for t, ta in zip(vs.timesteps, aus.timesteps)][:FORWARDS]

    lmeta = manifest["tensors"]["first_latents"]
    cond = torch.from_numpy(np.fromfile(f"{OUT}/first_latents.bin", dtype=np.float32).reshape(lmeta["shape"]))
    cond = cond[None, :, None]  # (1, 24, 1, lh, lw)
    gen = torch.Generator().manual_seed(SEED)
    cnoise = torch.randn(cond.shape, generator=gen)
    noised = vs.scale_noise(cond, 0.999, cnoise)
    cond_rows = patchify_video_latents(noised, (1, 2, 2))
    noise = torch.randn((1, 24, lf, lh, lw), generator=gen)
    latents = torch.cat([cond_rows, patchify_video_latents(noise, (1, 2, 2))])
    audio = torch.randn((audio_latents * 2, cfg["audio_in_channels"]), generator=gen)
    dump("cond_noise", cnoise[:, :, 0])
    dump("cond_rows", cond_rows)
    dump("noise_video", latents[ncond:])
    dump("noise_audio", audio)
    manifest["dit"] = {"text_tokens": ntext, "latent_frames": lf, "latent_height": lh, "latent_width": lw,
                       "audio_latents": audio_latents, "rows": int(pos.shape[0]), "cond_rows": ncond,
                       "steps": STEPS, "forwards": [], "seed": SEED}
    print(f"layout: {pos.shape[0]} rows ({ntext} text, {ncond} cond, {aidx.numel()} audio, "
          f"{vidx.numel() - ncond} video)", flush=True)

    H, E = cfg["hidden_size"], cfg["norm_eps"]
    patch = cfg["in_channels"] * 4
    time_proj = Timesteps(num_channels=cfg["freq_dim"], flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedder = load(TimestepEmbedding(cfg["freq_dim"], cfg["time_embed_hidden_dim"],
                                           out_dim=cfg["time_embed_dim"]), "time_embedder.")
    proj_in = load(torch.nn.Linear(patch, H), "proj_in.")
    audio_proj_in = load(torch.nn.Linear(cfg["audio_in_channels"], H), "audio_proj_in.")
    context_embedder = load(torch.nn.Linear(cfg["text_dim"], H), "context_embedder.")
    refiner = load(MiniMaxH3TokenRefiner(H, cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["ffn_dim"],
                                         cfg["num_refiner_layers"], E, cfg["qk_norm_eps"], cfg["final_norm_eps"]),
                   "token_refiner.")
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
        proj = block.adaln_proj
        proj.linear.weight = torch.nn.Parameter(tensor(f"transformer_blocks.{i}.adaln_proj.linear.weight"))
        proj.linear.bias = torch.nn.Parameter(tensor(f"transformer_blocks.{i}.adaln_proj.linear.bias"))
        block.adaln_proj = TableModulation([tuple(c.clone() for c in proj.float()(temb)) for temb in tembs])
        del proj
        load(block, f"transformer_blocks.{i}.", skip=("adaln_proj",))
        blocks.append(block)
        if i % 10 == 9:
            print(f"  loaded {i + 1} blocks in {time.time() - t0:.0f}s", flush=True)

    rotary_emb = rope(pos)
    text_embeds = refiner(context_embedder(text))
    dump("text_refined", text_embeds)
    for step, (u, rowt) in enumerate(plans):
        t0 = time.time()
        hidden = text_embeds.new_zeros((1, pos.shape[0], H))
        hidden = hidden.index_copy(1, tidx, text_embeds)
        hidden = hidden.index_copy(1, vidx, proj_in(latents[None]))
        hidden = hidden.index_copy(1, aidx, audio_proj_in(audio[None]))
        adaln_indices = rowt * MINIMAX_H3_MODALITY_NUM + tags
        for i, block in enumerate(blocks):
            block.adaln_proj.current = step
            if step == 0 and i in KEEP_BLOCKS:
                dump(f"f0_block{i}_in", hidden)
            hidden = block(hidden, tembs[step], adaln_indices, rotary_emb)
            if step == 0 and i in KEEP_BLOCKS:
                dump(f"f0_block{i}_out", hidden)
        normed = norm_out(hidden, tembs[step], rowt)
        v = proj_out(normed).index_select(1, vidx)[0]
        a = audio_proj_out(normed).index_select(1, aidx)[0]
        dump(f"f{step}_v_video", v)
        dump(f"f{step}_v_audio", a)
        latents[ncond:] = vs.step(v[ncond:].float(), vs.timesteps[step], latents[ncond:], return_dict=False)[0]
        audio = aus.step(a.float(), aus.timesteps[step], audio, return_dict=False)[0]
        dump(f"f{step}_latents", latents)
        dump(f"f{step}_audio", audio)
        manifest["dit"]["forwards"].append({"timestep": float(vs.timesteps[step]), "unique": u.tolist(),
                                            "seconds": time.time() - t0})
        print(f"forward {step}: {time.time() - t0:.0f}s, |v| {v.abs().max():.4g}", flush=True)
        save_manifest()


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    load_manifest()
    phases = {"prep": phase_prep, "vae": phase_vae, "text": phase_text, "dit": phase_dit}
    for name in sys.argv[1:]:
        t0 = time.time()
        print(f"--- {name}", flush=True)
        phases[name]()
        manifest.setdefault("seconds", {})[name] = time.time() - t0
        save_manifest()
    print(f"{len(manifest['tensors'])} tensors in {OUT}")


if __name__ == "__main__":
    main()
