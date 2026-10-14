"""Breeze-TTS-2's oracle (research/tts-instruct-vertical.md B0).

Upstream's runtime (`models/breeze-tts-src`, commit 6eba901) refuses to stream
without CUDA (`FastBreezeStreamingRuntime.__init__`), so this script drives
upstream's own modules on the CPU with the loop `iter_audio_chunks` runs:

  1. `templates.prepare_inputs` builds the prompt (and the CFG negative).
  2. `_merge_input_ids_with_input_values` encodes each text segment with the
     T5Gemma2 encoder and lays the projections into a zero sequence.
  3. The backbone prefills, then decodes a frame a step: codebook 0 from
     `lm_head` (CFG over the two rows, `sample_logits` with the 1.1 penalty
     over codebook 0's history), codebooks 1-15 from the depth decoder fed
     the backbone's normed last hidden state (CFG per codebook, the token
     shared by both rows, as `DepthDecoderGraph._cfg_sample`).
  4. The codec decodes the frames in one causal pass (`decoder.forward`,
     not `chunked_decode`: the streaming runtime keeps exact causal caches,
     so the whole-clip forward is the defined answer).

It runs in fp32 over the bf16 weights -- the defined answer the Go side is
gated against -- and then teacher-forces the frames it sampled through the
backbone (one causal pass) and through the depth decoder (one 16-position
pass a frame), so the gates do not depend on reproducing torch's RNG.

Each case writes reference/out/breeze/<case>/{tensors.safetensors,
manifest.json,audio.wav}. Run twice; the bytes must not move.

    .venv-breeze/bin/python -I reference/dump_breeze.py [case ...]
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "models/breeze-tts-src")
CKPT = os.path.join(REPO, "models/Breeze-TTS-2")
OUT = os.path.join(REPO, "reference/out/breeze")
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze  # noqa: E402
from breeze_infer.templates import get_template, prepare_inputs, select_template_name  # noqa: E402
from models.cudagraph.sampling import sample_logits  # noqa: E402
from models.logits_process import mask_invalid_codec_token_logits  # noqa: E402

CASES = {
    # Voice design: the vertical's reason to exist. CFG 4 is the README's.
    "design": dict(
        text="Good morning everyone, and welcome back to the show! (laugh) We have got a great lineup for you today.",
        instruction="An enthusiastic radio DJ with a warm, energetic voice.",
        cfg=4.0,
        seed=7,
    ),
    # Plain TTS: one row, no CFG, a random speaker.
    "plain": dict(
        text="The quick brown fox jumps over the lazy dog.",
        cfg=1.0,
        seed=11,
    ),
    # Chinese voice design, for the tokenizer and the encoder off the Latin path.
    "zh": dict(
        text="大家好，欢迎收听今天的节目。[笑]",
        instruction="一位温柔自信的年轻女性，声音清晰，语气亲切。",
        cfg=4.0,
        seed=3,
    ),
    # Past the codec transformer's 72-frame window and its 8-layer reach.
    "long": dict(
        text="The lighthouse keeper climbed the spiral stairs every evening at dusk. He trimmed the wick, "
        "polished the great lens, and watched the ships slip past the rocks. In forty years he had never "
        "once missed a night, and the village below slept soundly because of it.",
        instruction="A calm, deep-voiced storyteller speaking slowly.",
        cfg=4.0,
        seed=5,
    ),
    # Voice clone (B8): jfk at 24 kHz (dump_breeze_enc.py writes it) and its transcript.
    "clone": dict(
        text="We choose to build this machine not because it is easy, but because it is hard.",
        ref_audio="reference/out/breeze/enc/jfk24.wav",
        ref_text="And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
        cfg=1.0,
        seed=9,
    ),
    # Voice direction: the clone, steered.
    "direct": dict(
        text="We choose to build this machine not because it is easy, but because it is hard.",
        ref_audio="reference/out/breeze/enc/jfk24.wav",
        ref_text="And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
        instruction="Speak slowly, in a quiet, intimate whisper.",
        cfg=4.0,
        seed=9,
    ),
}

MAX_FRAMES = 400  # 32 s; every case stops on EOS long before
REPETITION_PENALTY = 1.1
BACKBONE = dict(temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
DEPTH = dict(temperature=0.9, top_k=50, top_p=1.0, do_sample=True)


def left_pad(t, n, value):
    if t.shape[1] == n:
        return t
    pad = torch.full((t.shape[0], n - t.shape[1], *t.shape[2:]), value, dtype=t.dtype)
    return torch.cat([pad, t], dim=1)


def branches(model, inputs):
    """[(name, input_ids, text_ids_mask, text_ids_len, inputs_embeds)], cond first."""
    out = []
    keys = [("cond", "input_ids", "attention_mask", "text_ids_mask", "text_ids_len")]
    if "cfg_negative_prompt_ids" in inputs:
        keys.append(("uncond", "cfg_negative_prompt_ids", "cfg_negative_prompt_attention_mask",
                     "cfg_negative_text_ids_mask", "cfg_negative_text_ids_len"))
    for name, ids, mask, tmask, tlen in keys:
        assert bool(inputs[mask].all()), "a single request is never padded"
        merged = model._merge_input_ids_with_input_values(
            input_ids=inputs[ids], attention_mask=inputs[mask],
            text_ids_mask=inputs[tmask], text_ids_len=inputs[tlen],
            input_values=inputs.get("input_values" if name == "cond" else "cfg_negative_input_values"),
        )
        out.append((name, inputs[ids][0], inputs[tmask][0], inputs[tlen], merged["inputs_embeds"][0]))
    return out


def encoder_segments(model, ids, tmask, tlen):
    """Each text segment's encoder output (before the projection), as the Go side computes it."""
    text = ids[tmask]
    segs, states, layers = [], [], []
    for seg in text.split(tlen.tolist()):
        o = model.text_encoder(input_ids=seg[None], attention_mask=torch.ones_like(seg)[None],
                               position_ids=torch.arange(seg.shape[0])[None], output_hidden_states=True)
        segs.append(seg)
        states.append(o.last_hidden_state[0])
        layers.append(torch.stack([h[0] for h in o.hidden_states]))
    return segs, states, layers


def generate(model, embeds, cfg, seed):
    """The sampled loop of iter_audio_chunks; returns frames [F,16] (int64)."""
    from transformers import DynamicCache

    rows = len(embeds)
    n = max(e.shape[0] for e in embeds)
    x = torch.stack([left_pad(e[None], n, 0.0)[0] for e in embeds])
    mask = torch.stack([left_pad(torch.ones(1, e.shape[0], dtype=torch.long), n, 0)[0] for e in embeds])
    pos = mask.cumsum(-1) - 1
    pos.masked_fill_(mask == 0, 1)
    cache = DynamicCache(config=model.backbone_model.config)
    torch.manual_seed(seed)
    o = model.backbone_model(inputs_embeds=x, attention_mask=mask, position_ids=pos,
                             past_key_values=cache, use_cache=True)
    hidden = o.last_hidden_state[:, -1]
    reserved = tuple(range(2048, model.config.vocab_size))

    def head(h):
        lg = model.lm_head(h.float()).float()
        return lg[:1] + 0 if rows == 1 else lg[1:] + cfg * (lg[:1] - lg[1:])

    token = sample_logits(head(hidden), suppress_tokens=reserved, **BACKBONE).view(1)
    history, frames = [], []
    E = model.depth_decoder.model.embed_tokens.weight
    V = model.config.vocab_size
    for step in range(MAX_FRAMES):
        if int(token) == V:  # backbone EOS
            break
        codes = [int(token)]
        dcache = DynamicCache(config=model.depth_decoder.config)
        for k in range(15):
            if k == 0:
                inp = torch.stack([hidden, E[codes[0]].expand(rows, -1)], dim=1)
            else:
                inp = E[k * V + codes[k]].expand(rows, 1, -1)
            d = model.depth_decoder.model(inputs_embeds=inp, past_key_values=dcache, use_cache=True)
            h = d.last_hidden_state[:, -1:].float()
            lg = torch.nn.functional.linear(h[:, 0], model.depth_decoder.codebooks_head.weight[k].float().T)
            lg = lg[:1] if rows == 1 else lg[1:] + cfg * (lg[:1] - lg[1:])
            mask_invalid_codec_token_logits(lg, codebook_size=2048, token_vocab_size=V)
            codes.append(int(sample_logits(lg, **DEPTH)))
        frames.append(codes)
        frame = torch.tensor(codes).view(1, 1, 16).expand(rows, 1, 16)
        mask = torch.cat([mask, torch.ones(rows, 1, dtype=torch.long)], dim=1)
        p = pos[:, -1:] + 1
        pos = torch.cat([pos, p], dim=1)
        o = model.backbone_model(inputs_embeds=model.backbone_model.embed_tokens(frame), attention_mask=mask,
                                 position_ids=p, past_key_values=cache, use_cache=True)
        hidden = o.last_hidden_state[:, -1]
        history.append(int(token))
        token = sample_logits(head(hidden), token_history=torch.tensor(history),
                              repetition_penalty=REPETITION_PENALTY, suppress_tokens=reserved,
                              **BACKBONE).view(1)
    return torch.tensor(frames, dtype=torch.long).view(-1, 16)


def teacher_forced(model, embed, frames):
    """One branch: backbone hidden/logits after the prompt and after each frame, and the
    depth decoder's 15 logits for each frame, all from single causal passes."""
    V = model.config.vocab_size
    fe = model.backbone_model.embed_tokens(frames[None])[0] if len(frames) else embed[:0]
    x = torch.cat([embed, fe])[None]
    o = model.backbone_model(inputs_embeds=x, use_cache=False)
    hid = o.last_hidden_state[0, embed.shape[0] - 1:]  # [F+1, 2048]
    logits = model.lm_head(hid.float()).float()
    E = model.depth_decoder.model.embed_tokens.weight
    W = model.depth_decoder.codebooks_head.weight.float()
    depth = []
    for f in range(len(frames)):
        codes = frames[f]
        inp = torch.cat([hid[f][None], E[codes[:15] + torch.arange(15) * V]])[None]  # [1,16,2048]
        d = model.depth_decoder.model(inputs_embeds=inp, use_cache=False).last_hidden_state[0].float()
        depth.append(torch.stack([d[k + 1] @ W[k] for k in range(15)]))
    depth = torch.stack(depth) if depth else torch.zeros(0, 15, 2051)
    return hid, logits, depth


def codec_stages(codec, frames):
    """The decoder's forward with its intermediates (Qwen3TTSTokenizerV2Decoder.forward)."""
    dec = codec.model.decoder
    st = {}
    codes = frames.T[None]  # [1,16,F]
    h = dec.quantizer.decode(codes)
    st["codec_quant"] = h[0]
    h = dec.pre_conv(h).transpose(1, 2)
    st["codec_preconv"] = h[0]
    h = dec.pre_transformer(inputs_embeds=h).last_hidden_state.permute(0, 2, 1)
    st["codec_pretf"] = h[0]
    for i, blocks in enumerate(dec.upsample):
        for b in blocks:
            h = b(h)
        st[f"codec_up{i}"] = h[0]
    wav = h
    for i, b in enumerate(dec.decoder):
        wav = b(wav)
        st[f"codec_dec{i}"] = wav[0]
    wav = wav.clamp(min=-1, max=1)
    ref = dec(codes)
    assert torch.equal(ref, wav), "codec_stages drifted from decoder.forward"
    st["wav"] = wav[0, 0]
    return st


def bf16_pass(model, names):
    """--bf16: upstream's serving precision over the same teacher forcing.

    For each dumped case, the fp32 prompt embeddings (cast) and the sampled
    frames run through the backbone and the depth decoder with the model in
    bf16, as upstream serves it, and the logits go to tensors_bf16.
    safetensors: what an int8 bank is priced against (research/tts-instruct-
    vertical.md B7, the way MUSIC.md A10 priced ace/lm's).
    """
    from safetensors.torch import load_file

    model.to(torch.bfloat16)
    # Upstream's graphs run both heads in fp32 (BackboneGraph, DepthDecoderGraph).
    model.lm_head.float()
    for name in names:
        d = os.path.join(OUT, name)
        ref = load_file(os.path.join(d, "tensors.safetensors"))
        frames = ref["frames"].long()
        out = {}
        for br in ("cond", "uncond"):
            if f"{br}_embeds" not in ref:
                continue
            emb = ref[f"{br}_embeds"].to(torch.bfloat16)
            hid, logits, depth = teacher_forced(model, emb, frames)
            out[f"{br}_logits"] = logits.float()
            out[f"{br}_depth_logits"] = depth.float()
            print(f"{name}/{br}: bf16 teacher-forced", flush=True)
        save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(d, "tensors_bf16.safetensors"))


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(CASES)
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count())
    torch.use_deterministic_algorithms(True)
    tokenizer, model, codec = load_runtime(__import__("pathlib").Path(CKPT), device="cpu",
                                           attn_implementation="eager")
    update_generation_config_for_breeze(model)
    # Never called at inference (research/tts-instruct-vertical.md); drop before fp32.
    model.embed_text_tokens = None
    model.codec_model = None
    if "--bf16" in sys.argv:
        bf16_pass(model, names)
        return
    model.float()
    codec.model.float()

    for name in names:
        case = CASES[name]
        req = {"id": name, "text": case["text"], "speaker": "S0"}
        if case.get("instruction"):
            req["instruction"] = case["instruction"]
        if case.get("ref_audio"):
            req["ref_audio_path"] = os.path.join(REPO, case["ref_audio"])
            req["ref_text"] = case["ref_text"]
        tmpl = select_template_name(req)
        inputs = prepare_inputs(tokenizer, codec, model, [req], get_template(tmpl), guidance_scale=case["cfg"],
                                guidance_scale_ref=None, guidance_scale_ins=None)
        br = branches(model, inputs)
        frames = generate(model, [b[4] for b in br], case["cfg"], case["seed"])
        print(f"{name}: {tmpl}, {[b[1].shape[0] for b in br]} prompt tokens, {len(frames)} frames "
              f"= {len(frames) / 12.5:.2f} s", flush=True)

        t = {"frames": frames.to(torch.int32)}
        if inputs.get("input_values") is not None:
            t["ref_codes"] = inputs["input_values"][0].to(torch.int32)
        man = {"case": case, "template": tmpl, "frames": len(frames), "branches": []}
        for bname, ids, tmask, tlen, embed in br:
            segs, states, layers = encoder_segments(model, ids, tmask, tlen)
            hid, logits, depth = teacher_forced(model, embed, frames)
            t[f"{bname}_ids"] = ids.to(torch.int32)
            t[f"{bname}_text_mask"] = tmask.to(torch.int32)
            for i, (s, h, ly) in enumerate(zip(segs, states, layers)):
                t[f"{bname}_seg{i}_ids"] = s.to(torch.int32)
                t[f"{bname}_seg{i}_enc"] = h
                t[f"{bname}_seg{i}_enc_layers"] = ly
            t[f"{bname}_embeds"] = embed
            t[f"{bname}_hidden"] = hid
            t[f"{bname}_logits"] = logits
            t[f"{bname}_depth_logits"] = depth
            man["branches"].append({"name": bname, "tokens": ids.shape[0], "segments": tlen.tolist()})
        st = codec_stages(codec, frames)
        t.update(st)
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        # Ids go out as fp32 (exact below 2^24): the Go reader takes float dtypes only.
        save_file({k: (v.float() if not v.is_floating_point() else v).contiguous() for k, v in t.items()},
                  os.path.join(d, "tensors.safetensors"))
        man["tensors"] = {k: list(v.shape) for k, v in t.items()}
        with open(os.path.join(d, "manifest.json"), "w") as fh:
            json.dump(man, fh, indent=1, ensure_ascii=False)
        sf.write(os.path.join(d, "audio.wav"), st["wav"].numpy(), 24000, subtype="PCM_16")


if __name__ == "__main__":
    main()
