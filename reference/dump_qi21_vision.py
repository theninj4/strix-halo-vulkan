"""Dump Qwen-Image-2.1's vision tower and its image-conditioned text
encoding, for Q8 — the oracle the Go port of an *edit's* prefix is gated
against.

Everything an edit adds over a generation is on this path and nowhere else.
Q2 already validated the DiT's edit case (per-reference RoPE blocks, the
segments, the prefix cache) against `out/qi21dit` and `out/qi21block`, and
Q5 the VAE encoder that supplies the reference latents. What has never been
run in Go is the other half of a condition image: the 27-layer ViT that
turns its pixels into vision context, the three deepstack taps that are
injected into the text model's first layers, and the 3-D mrope the text
model uses once real image positions exist.

So this walks all of it, stage by stage:

  * the processor's patching — `pixel_values` is [patches, 1536] where 1536
    is 3 channels x 2 temporal x 16 x 16, and the temporal axis is the same
    frame twice. The Go side has to reproduce that layout exactly, so it is
    dumped rather than described;
  * the tower's *interpolated* position embedding. The checkpoint holds 2304
    = 48x48 learned positions and any image grid is a bilinear resample of
    them, which is four (index, weight) pairs per patch — dumped as the
    indices and weights the model itself computed, because reproducing the
    resample is the single most mistakable piece here;
  * the vision rotary table, every block's output, the three deepstack
    features (layers 8/16/24) and the 2x2 patch merger;
  * the text side: the 3-D mrope position ids for a sequence that contains
    image tokens, the input embeddings *after* the merger's output has been
    scattered into the `<|image_pad|>` slots, and the pipeline's own
    `prompt_embeds` and `image_pad_mask`.

A fact this dump exists to pin, and the one Q8 rests on: the DiT overwrites
the image-slot rows with VAE latents, so vision content reaches it **only
through what the text tokens absorbed inside the encoder's attention**. The
tower conditions the text; the VAE conditions the pixels. `edit_merged_rows`
and `edit_prompt_embeds` are the two sides of that.

The condition image is generated here — a deterministic 256x256 RGBA card —
so the dump needs no asset and two runs are byte-identical.

    .venv/bin/python reference/dump_qi21_vision.py   (~36 GB RSS, minutes)

The end-to-end edit oracle is a separate, heavier script: the full pipeline
in fp32 is ~68 GB and must run alone.
"""

import copy
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

OUT = "reference/out/qi21vision"
MODEL = "models/Qwen-Image-2.1"
PROMPT = "make the sky a deep orange sunset"
# 256x256 is the processor's own minimum (shortest_edge is 65536 pixels), so
# the card passes through unresized: the grid is 16x16 patches, 64 image-pad
# slots, and every stage below stays small enough to read.
SIZE = 256


def test_card(size):
    """A deterministic RGBA card: colour ramps, hard edges and a gradient
    alpha, so the tower sees structure at every frequency it could care
    about. Same idea as dump_qi21_vae.py's card, at the tower's size."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / (size - 1)
    r = np.where((x * 4).astype(int) % 2 == 0, x, 1 - y)
    g = np.sin(6.0 * np.pi * y) * 0.5 + 0.5
    b = np.where(((x * 8).astype(int) + (y * 8).astype(int)) % 2 == 0, 0.15, 0.85)
    a = np.clip(0.25 + 0.75 * x, 0, 1)
    card = np.stack([r, g, b, a], axis=-1)
    return (card * 255).round().astype(np.uint8)


def test_card_wide(h, w):
    """A second deterministic RGBA card, at a different aspect ratio and with
    a different pattern from test_card's — diagonal ramps and a coarser
    checker — so that two images in one sequence cannot be confused for each
    other by symmetry."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    y, x = y / (h - 1), x / (w - 1)
    r = np.clip((x + y) / 2 * 1.5, 0, 1)
    g = np.where(((x * 6).astype(int) + (y * 3).astype(int)) % 2 == 0, 0.9, 0.2)
    b = np.cos(4.0 * np.pi * x) * 0.5 + 0.5
    a = np.clip(1.0 - 0.75 * y, 0, 1)
    card = np.stack([r, g, b, a], axis=-1)
    return (card * 255).round().astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    proc = Qwen3VLProcessor.from_pretrained(MODEL, subfolder="processor")
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, subfolder="text_encoder", dtype=torch.float32, attn_implementation="eager"
    ).eval()
    pipe = QwenImage21Pipeline(
        scheduler=None, vae=None, text_encoder=te, processor=proc, transformer=None
    )

    vcfg = te.config.vision_config
    tcfg = te.config.text_config
    manifest = {
        "template_ti2i": pipe.prompt_template_ti2i,
        "drop_idx": pipe._drop_idx,
        "image_token_id": pipe._img_token_id,
        "prompt": PROMPT,
        "size": SIZE,
        "vision": {
            "depth": vcfg.depth, "hidden_size": vcfg.hidden_size,
            "num_heads": vcfg.num_heads, "head_dim": vcfg.hidden_size // vcfg.num_heads,
            "intermediate_size": vcfg.intermediate_size, "hidden_act": vcfg.hidden_act,
            "patch_size": vcfg.patch_size, "spatial_merge_size": vcfg.spatial_merge_size,
            "temporal_patch_size": vcfg.temporal_patch_size,
            "in_channels": vcfg.in_channels,
            "num_position_embeddings": vcfg.num_position_embeddings,
            "out_hidden_size": vcfg.out_hidden_size,
            "deepstack_visual_indexes": list(vcfg.deepstack_visual_indexes),
        },
        "text": {
            "mrope_section": tcfg.rope_parameters.get("mrope_section"),
            "mrope_interleaved": tcfg.rope_parameters.get("mrope_interleaved"),
            "rope_theta": tcfg.rope_parameters["rope_theta"],
            "head_dim": tcfg.head_dim,
        },
        "tensors": {},
    }

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": float(flat.abs().max()),
        }
        print(f"  {name:26s} {str(list(t.shape)):20s} sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}")

    # --- the condition image, both copies -------------------------------
    #
    # The pipeline flattens alpha over white for the *vision* copy only and
    # feeds all four channels to the VAE. Both are dumped, because a port
    # that flattens once and uses it twice is a bug this pins.
    card = test_card(SIZE)
    rgba = Image.fromarray(card, mode="RGBA")
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.getchannel("A"))
    dump("card_rgba", torch.from_numpy(card).permute(2, 0, 1).float()[None] / 127.5 - 1)
    dump("card_vision_rgb", torch.from_numpy(np.array(white)).permute(2, 0, 1).float()[None] / 127.5 - 1)

    # --- what the processor produces ------------------------------------
    rendered = pipe.prompt_template_ti2i.format(PROMPT)
    inputs = proc(text=[rendered], images=[white], padding=True, padding_side="left", return_tensors="pt")
    ids = inputs.input_ids
    grid = inputs.image_grid_thw
    manifest["rendered"] = rendered
    manifest["ids"] = ids[0].tolist()
    manifest["grid_thw"] = grid.tolist()
    manifest["seq"] = int(ids.shape[1])
    manifest["image_pad_positions"] = [i for i, v in enumerate(ids[0].tolist()) if v == pipe._img_token_id]
    print(f"{ids.shape[1]} tokens, grid {grid.tolist()}, "
          f"{len(manifest['image_pad_positions'])} image-pad slots, drop_idx {pipe._drop_idx}")
    dump("pixel_values", inputs.pixel_values)
    dump("ids", ids.float())
    dump("mm_token_type_ids", inputs.mm_token_type_ids.float())

    # --- the vision tower, stage by stage -------------------------------
    visual = te.model.visual
    caps = {}
    handles = [
        visual.patch_embed.register_forward_hook(lambda m, a, o: caps.__setitem__("patch_embed", o)),
        visual.pos_embed.register_forward_hook(lambda m, a, o: caps.__setitem__("pos_embed_rows", o)),
        visual.rotary_pos_emb.register_forward_hook(lambda m, a, o: caps.__setitem__("rope", o)),
        visual.merger.register_forward_hook(lambda m, a, o: caps.__setitem__("merger_in", a[0])),
    ]
    for i, blk in enumerate(visual.blocks):
        handles.append(blk.register_forward_hook(
            lambda m, a, o, i=i: caps.__setitem__(f"block{i}", o[0] if isinstance(o, tuple) else o)))

    with torch.no_grad():
        vout = visual(inputs.pixel_values, grid, return_dict=True)
    for h in handles:
        h.remove()

    dump("vis_patch_embed", caps["patch_embed"])
    # The tower's input is patch_embed + the interpolated position embedding;
    # block0's input is that, so it is recoverable and dumped as the sum the
    # model actually formed.
    dump("vis_block0_out", caps["block0"])
    dump("vis_block1_out", caps["block1"])
    for i in manifest["vision"]["deepstack_visual_indexes"]:
        dump(f"vis_block{i}_out", caps[f"block{i}"])
    dump("vis_last_hidden", vout.last_hidden_state)
    dump("vis_merger_in", caps["merger_in"])
    dump("vis_merged", vout.pooler_output)
    for i, f in enumerate(vout.deepstack_features):
        dump(f"vis_deepstack{i}", f)
    cos, sin = caps["rope"] if isinstance(caps["rope"], tuple) else (caps["rope"], None)
    dump("vis_rope_cos", cos)
    if sin is not None:
        dump("vis_rope_sin", sin)

    # The interpolated position grid, recomputed with the model's own helper
    # so the dump carries the indices and weights rather than a description
    # of how to get them.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )
    idx, wts = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.config.spatial_merge_size, kwargs={},
    )
    dump("vis_interp_indices", idx.float())
    dump("vis_interp_weights", wts)
    dump("vis_position_ids", get_vision_position_ids(grid, visual.spatial_merge_size, kwargs={}).float())
    # patch_embed + pos: the actual first-block input.
    pos = (visual.pos_embed(idx) * wts[:, :, None]).sum(1)
    dump("vis_pos_embeds", pos)
    dump("vis_block_input", caps["patch_embed"] + pos)

    # --- the text side --------------------------------------------------
    text_model = getattr(te.model, "language_model", te.model)
    tcaps = {}
    th = [
        text_model.rotary_emb.register_forward_hook(
            lambda m, a, o: tcaps.update(cos=o[0], sin=o[1])),
        text_model.norm.register_forward_hook(lambda m, a, o: a[0]),
        text_model.layers[0].register_forward_hook(
            lambda m, a, o: tcaps.__setitem__("layer0", o[0] if isinstance(o, tuple) else o)),
        text_model.embed_tokens.register_forward_hook(
            lambda m, a, o: tcaps.__setitem__("embed_tokens", o)),
    ]
    fwd = {
        "input_ids": ids, "attention_mask": inputs.attention_mask,
        "pixel_values": inputs.pixel_values, "image_grid_thw": grid,
        "output_hidden_states": True,
    }
    fwd["mm_token_type_ids"] = inputs.mm_token_type_ids
    with torch.no_grad():
        outputs = te(**fwd)
    for h in th:
        h.remove()

    dump("edit_embed_tokens", tcaps["embed_tokens"])
    dump("edit_layer0_out", tcaps["layer0"])
    dump("edit_last_prenorm", outputs.hidden_states[-1])
    dump("edit_rope_cos", tcaps["cos"])
    dump("edit_rope_sin", tcaps["sin"])

    # The 3-D mrope position ids for this sequence: the number every text
    # token shares three of, and every image token gets a real (t, h, w) for.
    pos_ids = te.model.compute_3d_position_ids(
        input_ids=ids, image_grid_thw=grid, video_grid_thw=None,
        inputs_embeds=None, attention_mask=inputs.attention_mask,
        mm_token_type_ids=inputs.mm_token_type_ids,
    )
    if isinstance(pos_ids, tuple):
        pos_ids = pos_ids[0]
    dump("edit_position_ids", pos_ids.float())

    # --- the pipeline's own output, which is what the DiT is fed --------
    with torch.no_grad():
        embeds, mask, pad_mask = pipe._get_qwen_prompt_embeds(PROMPT, [rgba], device="cpu")
    dump("edit_prompt_embeds", embeds)
    dump("edit_image_pad_mask", pad_mask.float())
    manifest["embed_tokens"] = int(embeds.shape[1])
    manifest["pad_mask_count"] = int(pad_mask.sum())
    # The merged rows the tower produced, as they appear inside the encoder's
    # input: the DiT will overwrite these with VAE latents, so this is the
    # tensor whose *influence* on the surrounding text rows is all that
    # survives. Dumped so a port can check the scatter landed.
    scattered = tcaps["embed_tokens"].clone()
    pad_positions = torch.tensor(manifest["image_pad_positions"])
    scattered[0, pad_positions] = vout.pooler_output.to(scattered.dtype)
    dump("edit_merged_rows", scattered[0, pad_positions][None])
    gap = float((outputs.hidden_states[0] - scattered).abs().max())
    manifest["scatter_gap"] = gap
    print(f"  merger-scatter vs the model's own inputs_embeds: {gap:.3g}")

    # --- two condition images, which is a different sequence ------------
    #
    # Everything above is one image, and one image hides three things an
    # edit with several references has to get right: the second run's
    # " <image2>" carries a leading space that tokenizes; the position
    # counter advances between images by the grid's *span* (max(h, w) over
    # the merge) rather than by its token count, so two images of different
    # aspect ratios shift everything after them differently; and the
    # deepstack features of all the images are one stacked tensor in pad
    # order rather than a loop. The second card is 192x384 — a different
    # aspect ratio, 72 tokens against the square's 64 and a span of 12
    # against 8 — and a different pattern, so an ordering mistake cannot
    # produce the right answer by symmetry.
    card2 = test_card_wide(192, 384)
    rgba2 = Image.fromarray(card2, mode="RGBA")
    white2 = Image.new("RGB", rgba2.size, (255, 255, 255))
    white2.paste(rgba2, mask=rgba2.getchannel("A"))
    dump("card2_vision_rgb", torch.from_numpy(np.array(white2)).permute(2, 0, 1).float()[None] / 127.5 - 1)

    replace = ("<image1><|vision_start|><|image_pad|><|vision_end|>"
               " <image2><|vision_start|><|image_pad|><|vision_end|>")
    multi_template = pipe.prompt_template_ti2i.replace(
        "<image1><|vision_start|><|image_pad|><|vision_end|>", replace)
    multi_rendered = multi_template.format(PROMPT)
    multi_inputs = proc(text=[multi_rendered], images=[white, white2], padding=True,
                        padding_side="left", return_tensors="pt")
    manifest["multi_rendered"] = multi_rendered
    manifest["multi_ids"] = multi_inputs.input_ids[0].tolist()
    manifest["multi_grid_thw"] = multi_inputs.image_grid_thw.tolist()
    manifest["multi_seq"] = int(multi_inputs.input_ids.shape[1])
    manifest["multi_sizes"] = [[SIZE, SIZE], [192, 384]]
    manifest["multi_pad_positions"] = [
        i for i, v in enumerate(multi_inputs.input_ids[0].tolist()) if v == pipe._img_token_id
    ]
    print(f"two images: {multi_inputs.input_ids.shape[1]} tokens, grids "
          f"{multi_inputs.image_grid_thw.tolist()}, {len(manifest['multi_pad_positions'])} slots")

    # The tower over both images at once, sliced to the *second* — the one
    # with a non-square grid. Attention is block-diagonal over cu_seqlens, so
    # a port that runs one image at a time is equivalent and these rows are
    # what it has to reproduce.
    n1 = SIZE // 16 * SIZE // 16  # patches of the first image
    m1 = n1 // 4  # its merged rows
    vcaps = {}
    vh = [
        visual.patch_embed.register_forward_hook(lambda m, a, o: vcaps.__setitem__("patch_embed", o)),
        visual.blocks[0].register_forward_hook(
            lambda m, a, o: vcaps.__setitem__("block0", o[0] if isinstance(o, tuple) else o)),
    ]
    with torch.no_grad():
        vout2 = visual(multi_inputs.pixel_values, multi_inputs.image_grid_thw, return_dict=True)
    for h in vh:
        h.remove()
    dump("multi_pixel_values2", multi_inputs.pixel_values[n1:])
    dump("vis2_patch_embed", vcaps["patch_embed"][n1:])
    dump("vis2_block0_out", vcaps["block0"][n1:])
    dump("vis2_last_hidden", vout2.last_hidden_state[n1:])
    dump("vis2_merged", vout2.pooler_output[m1:])
    for i, f in enumerate(vout2.deepstack_features):
        dump(f"vis2_deepstack{i}", f[m1:])

    # How far the reference disagrees with *itself* on these rows. A port
    # runs one condition image at a time; the reference ran both in one
    # packed forward with block-diagonal attention. The two are the same
    # arithmetic in a different summation order, so the gap between them is
    # this stage's fp32 noise floor — and the tower earns one, because its
    # residual stream reaches absmax 1.4e4 by the last hidden state and 27
    # pre-norm blocks of cancellation sit on top of it. Measured rather than
    # assumed, with the same metric the Go gate uses.
    def reldev(got, want):
        got, want = got.detach().to(torch.float64), want.detach().to(torch.float64)
        rms = float(want.pow(2).mean().sqrt())
        d = (got - want).abs()
        return {
            "rel": float((d / torch.clamp(want.abs(), min=rms)).max()),
            "max_abs": float(d.max()), "rms": rms,
        }

    with torch.no_grad():
        alone2 = visual(multi_inputs.pixel_values[n1:], multi_inputs.image_grid_thw[1:], return_dict=True)
        alone1 = visual(multi_inputs.pixel_values[:n1], multi_inputs.image_grid_thw[:1], return_dict=True)
    manifest["batched_vs_alone"] = {
        "image2_last_hidden": reldev(alone2.last_hidden_state, vout2.last_hidden_state[n1:]),
        "image2_merged": reldev(alone2.pooler_output, vout2.pooler_output[m1:]),
        "image2_deepstack2": reldev(alone2.deepstack_features[2], vout2.deepstack_features[2][m1:]),
        "image1_last_hidden": reldev(alone1.last_hidden_state, vout2.last_hidden_state[:n1]),
        "image1_merged": reldev(alone1.pooler_output, vout2.pooler_output[:m1]),
    }
    for k, v in manifest["batched_vs_alone"].items():
        print(f"  reference against itself, {k:22s} rel {v['rel']:.3g}  max abs {v['max_abs']:.3g}")

    # That gap is 0 — packing perturbs nothing — so it does not bound this
    # stage. What does is the arbiter Q5 used on the VAE's tail: the same
    # tower in float64. It says how far the *dumped fp32 reference* is from
    # the answer, which is the only way to read a port's disagreement with it
    # after 27 pre-norm blocks over a residual stream that reaches 1.4e4. The
    # float64 rows are dumped too, so a port can be compared against the
    # better oracle rather than against fp32's noise.
    visual64 = copy.deepcopy(visual).to(torch.float64)
    with torch.no_grad():
        f64_2 = visual64(multi_inputs.pixel_values[n1:].to(torch.float64),
                         multi_inputs.image_grid_thw[1:], return_dict=True)
        f64_1 = visual64(multi_inputs.pixel_values[:n1].to(torch.float64),
                         multi_inputs.image_grid_thw[:1], return_dict=True)
    del visual64
    dump("vis2_last_hidden64", f64_2.last_hidden_state)
    dump("vis2_merged64", f64_2.pooler_output)
    for i, f in enumerate(f64_2.deepstack_features):
        dump(f"vis2_deepstack{i}64", f)
    dump("vis_last_hidden64", f64_1.last_hidden_state)
    dump("vis_merged64", f64_1.pooler_output)
    manifest["fp32_vs_fp64"] = {
        "image2_last_hidden": reldev(vout2.last_hidden_state[n1:], f64_2.last_hidden_state),
        "image2_merged": reldev(vout2.pooler_output[m1:], f64_2.pooler_output),
        "image2_deepstack2": reldev(vout2.deepstack_features[2][m1:], f64_2.deepstack_features[2]),
        "image1_last_hidden": reldev(vout2.last_hidden_state[:n1], f64_1.last_hidden_state),
        "image1_merged": reldev(vout2.pooler_output[:m1], f64_1.pooler_output),
    }
    for k, v in manifest["fp32_vs_fp64"].items():
        print(f"  the fp32 reference vs float64, {k:22s} rel {v['rel']:.3g}  max abs {v['max_abs']:.3g}")

    multi_pos = te.model.compute_3d_position_ids(
        input_ids=multi_inputs.input_ids, image_grid_thw=multi_inputs.image_grid_thw,
        video_grid_thw=None, inputs_embeds=None, attention_mask=multi_inputs.attention_mask,
        mm_token_type_ids=multi_inputs.mm_token_type_ids,
    )
    if isinstance(multi_pos, tuple):
        multi_pos = multi_pos[0]
    dump("multi_position_ids", multi_pos.float())

    # The text side of the two-image sequence, stage by stage, so a port that
    # disagrees at the end can be told *where*: the scattered inputs_embeds
    # separate the tower and the scatter from the 36 layers entirely.
    tcaps2 = {}
    th2 = [
        # The same final-norm neutralization the single-image section and the
        # pipeline both install: without it transformers 5.x ties
        # hidden_states[-1] to the normalized last_hidden_state, and the
        # dumped "prenorm" is a third of the signal the DiT reads.
        text_model.norm.register_forward_hook(lambda m, a, o: a[0]),
        text_model.layers[0].register_forward_hook(
            lambda m, a, o: tcaps2.__setitem__("layer0", o[0] if isinstance(o, tuple) else o)),
    ]
    with torch.no_grad():
        outputs2 = te(
            input_ids=multi_inputs.input_ids, attention_mask=multi_inputs.attention_mask,
            pixel_values=multi_inputs.pixel_values, image_grid_thw=multi_inputs.image_grid_thw,
            mm_token_type_ids=multi_inputs.mm_token_type_ids, output_hidden_states=True,
        )
    for h in th2:
        h.remove()
    dump("multi_inputs_embeds", outputs2.hidden_states[0])
    dump("multi_layer0_out", tcaps2["layer0"])
    dump("multi_last_prenorm", outputs2.hidden_states[-1])

    with torch.no_grad():
        m_embeds, _, m_pad_mask = pipe._get_qwen_prompt_embeds(PROMPT, [rgba, rgba2], device="cpu")
    dump("multi_prompt_embeds", m_embeds)
    dump("multi_image_pad_mask", m_pad_mask.float())
    manifest["multi_embed_tokens"] = int(m_embeds.shape[1])
    manifest["multi_pad_mask_count"] = int(m_pad_mask.sum())

    # --- the VAE's view of the same card --------------------------------
    try:
        vae = AutoencoderKLQwenImage21.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).eval()
        px = torch.from_numpy(card).permute(2, 0, 1).float()[None] / 127.5 - 1
        with torch.no_grad():
            posterior = vae.encode(px.unsqueeze(2)).latent_dist
        mode = posterior.mode()
        dump("card_latent_raw", mode)
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        dump("card_latent_norm", (mode - mean) / std)
    except Exception as exc:  # noqa: BLE001 - the tower is the point; the VAE is a convenience
        print(f"  (skipping the VAE latents: {exc})")

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
