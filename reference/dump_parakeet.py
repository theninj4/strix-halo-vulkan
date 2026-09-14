"""Dump a reference run of parakeet-tdt-0.6b-v3, for stages S2-S5 of SPEECH.md.

The oracle is transformers' own `ParakeetForTDT` in fp32 on CPU, run on
`testdata/jfk.wav` -- 11.000 s of 16 kHz speech whose transcript is known, so
the end of the chain is checkable as an exact string rather than a tolerance.

The dump walks the whole model rather than only its ends, because every stage
of the Go port wants a different point in it:

  * the mel front end, which is the one part that is signal processing rather
    than linear algebra (S3);
  * the subsampling stack layer by layer -- the stride-2 conv2ds are the first
    operator the engine does not already have (S4);
  * one encoder layer's internals, including both halves of the relative
    position attention and the `_rel_shift` that joins them, which is the only
    open kernel question in STT (S4);
  * every layer's output, so a divergence is bisected rather than hunted (S4);
  * the prediction network's LSTM state and the joint, step by step, plus the
    whole TDT greedy trace of (frame, token, duration) triples (S5).

Four things it settles rather than assumes, each recorded in the manifest:

  * `attention` -- whether the encoder attends over the whole clip or over a
    limited context window. NeMo's FastConformer usually windows;
    `config.json` names no window, and what the library actually builds is
    what matters at 3000 frames.
  * `stack_gap` -- that running the layers by hand reproduces
    `encoder(...)`, so the by-hand internals dumped here describe the real
    forward pass.
  * `manual_matches_generate` -- that the hand-written TDT loop below emits
    the same tokens and durations as `model.generate`. The Go decoder is a
    port of that loop, not of `generate`.
  * `fp16` -- the absmax of every activation on the path, since the repo's
    recurring lesson is that fp16 headroom is surveyed before it is assumed.

    .venv/bin/python reference/dump_parakeet.py
"""

import argparse
import json
import os
import wave

import numpy as np
import torch
from transformers import AutoProcessor, ParakeetForTDT

# The transcript of testdata/jfk.wav, as this checkpoint punctuates it. The
# words are the clip's; the comma after "so" and the final full stop are the
# model's, since it is trained with a <|pnc|> option -- so this string is the
# check that the *reference* is transcribing rather than hallucinating, and
# the Go side's bar is to reproduce it exactly.
EXPECTED = (
    "And so, my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)


def read_wav(path):
    """Read a 16-bit PCM WAV exactly the way audio.DecodeWAV does in Go."""
    with wave.open(path) as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: {8 * w.getsampwidth()}-bit, expected 16-bit PCM")
        rate, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(-1)
    return x, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/parakeet-tdt-0.6b-v3")
    ap.add_argument("--audio", default="testdata/jfk.wav")
    ap.add_argument("--out", default="reference/out/parakeet")
    ap.add_argument("--layer", type=int, default=0, help="encoder layer whose internals are dumped")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    audio, rate = read_wav(args.audio)
    print(f"{args.audio}: {len(audio)} samples at {rate} Hz = {len(audio) / rate:.3f} s")

    processor = AutoProcessor.from_pretrained(args.model)
    model = ParakeetForTDT.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = model.config
    enc_cfg = cfg.encoder_config
    heads, hd = enc_cfg.num_attention_heads, enc_cfg.hidden_size // enc_cfg.num_attention_heads

    manifest = {
        "audio": args.audio, "samples": len(audio), "sampling_rate": rate,
        "expected": EXPECTED,
        "front_end": {
            "n_fft": processor.feature_extractor.n_fft,
            "hop_length": processor.feature_extractor.hop_length,
            "win_length": processor.feature_extractor.win_length,
            "preemphasis": processor.feature_extractor.preemphasis,
            "n_mels": processor.feature_extractor.feature_size,
            "log_zero_guard": 2.0**-24, "norm_eps": 1e-5,
            "window": "hann, periodic=False, zero-padded to n_fft",
            "center": True, "pad_mode": "constant",
        },
        "encoder": {
            "layers": enc_cfg.num_hidden_layers, "hidden_size": enc_cfg.hidden_size,
            "heads": heads, "head_dim": hd, "intermediate_size": enc_cfg.intermediate_size,
            "conv_kernel_size": enc_cfg.conv_kernel_size,
            "subsampling_factor": enc_cfg.subsampling_factor,
            "subsampling_channels": enc_cfg.subsampling_conv_channels,
            "activation": enc_cfg.hidden_act,
        },
        "tdt": {
            "vocab_size": cfg.vocab_size, "blank_token_id": cfg.blank_token_id,
            "durations": cfg.durations, "decoder_hidden_size": cfg.decoder_hidden_size,
            "num_decoder_layers": cfg.num_decoder_layers,
            "max_symbols_per_step": cfg.max_symbols_per_step,
            "joint_activation": cfg.hidden_act,
        },
        "dump_layer": args.layer, "tensors": {}, "fp16": {},
    }

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(args.out, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        absmax = float(flat.abs().max()) if flat.numel() else 0.0
        manifest["tensors"][name] = {
            "shape": list(t.shape), "count": int(flat.numel()),
            "sum": float(flat.double().sum()), "absmax": absmax,
        }
        print(f"  {name:28s} {str(list(t.shape)):22s} sum={flat.double().sum():+.5f} absmax={absmax:.4g}")

    # ---------------------------------------------------------------- front end
    fe = processor.feature_extractor
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt")

    # The front end's own intermediates, so that S3 is walked rather than
    # bisected: the preemphasised signal, the window torch.stft is handed, the
    # librosa filterbank, and the power spectrogram the bank is applied to.
    with torch.no_grad():
        wave_t = torch.from_numpy(audio.copy())[None, :]
        pre = torch.cat([wave_t[:, :1], wave_t[:, 1:] - fe.preemphasis * wave_t[:, :-1]], dim=1)
        window = torch.hann_window(fe.win_length, periodic=False)
        stft = torch.stft(pre, fe.n_fft, hop_length=fe.hop_length, win_length=fe.win_length,
                          window=window, return_complex=True, pad_mode="constant")
        power = torch.view_as_real(stft).pow(2).sum(-1)
        log_mel = torch.log(fe.mel_filters @ power + 2.0**-24).permute(0, 2, 1)
    mel, mask = inputs.input_features, inputs.attention_mask.bool()
    frames, valid = mel.shape[1], int(mask.sum())
    print(f"mel {tuple(mel.shape)}, {valid} of {frames} frames valid")
    manifest["front_end"]["frames"] = frames
    manifest["front_end"]["valid_frames"] = valid

    # ------------------------------------------------------------------ encoder
    encoder = model.encoder
    sub = encoder.subsampling
    with torch.no_grad():
        h = mel.unsqueeze(1)
        lengths = mask.sum(-1)
        sub_outs = []
        for i, layer in enumerate(sub.layers):
            h = layer(h)
            if isinstance(layer, torch.nn.Conv2d):
                lengths = sub._get_output_length(lengths, layer)
                keep = torch.arange(h.shape[2]) < lengths[:, None]
                h = h * keep[:, None, :, None]
                sub_outs.append((f"sub_{i}", h.clone()))
        flat = h.transpose(1, 2).reshape(h.shape[0], h.shape[2], -1)
        subsampled = sub.linear(flat)
    T = subsampled.shape[1]
    enc_valid = int(encoder._get_subsampling_output_length(torch.tensor([valid])).item())
    print(f"subsampled {tuple(subsampled.shape)}: {T} frames, {enc_valid} valid "
          f"({enc_cfg.subsampling_factor}x from {frames})")
    manifest["encoder"]["frames"] = T
    manifest["encoder"]["valid_frames"] = enc_valid

    with torch.no_grad():
        pos_embed = encoder.encode_positions(subsampled)
        enc_mask = encoder._get_output_attention_mask(mask, target_length=T)
        attn_mask = enc_mask.unsqueeze(1).expand(-1, T, -1)
        attn_mask = (attn_mask & attn_mask.transpose(1, 2)).unsqueeze(1)

    # Is the attention windowed, or full over the clip? The mask the library
    # builds is the answer: a window would be a band, padding alone is a
    # rectangle. Measured, not read off config.json.
    valid_block = attn_mask[0, 0, :enc_valid, :enc_valid]
    manifest["encoder"]["attention"] = {
        "mask_is_all_true_over_valid": bool(valid_block.all()),
        "masked_positions": int((~attn_mask[0, 0]).sum()),
        "att_context_size": getattr(enc_cfg, "att_context_size", None),
        "note": "full (non-causal) attention over the valid frames; the mask is padding only",
    }
    print(f"attention: {'full' if valid_block.all() else 'WINDOWED'} over {enc_valid} valid frames, "
          f"{int((~attn_mask[0, 0]).sum())} masked pairs of {T * T}")

    # One layer's internals, recomputed from the same weights so that a port
    # can be walked forwards instead of bisected.
    layer = encoder.layers[args.layer]
    a = layer.self_attn
    with torch.no_grad():
        x0 = subsampled
        n1 = layer.norm_feed_forward1(x0)
        ff1 = layer.feed_forward1(n1)
        r1 = x0 + 0.5 * ff1

        n2 = layer.norm_self_att(r1)
        q = a.q_proj(n2).view(1, T, heads, hd).transpose(1, 2)
        k = a.k_proj(n2).view(1, T, heads, hd).transpose(1, 2)
        v = a.v_proj(n2).view(1, T, heads, hd).transpose(1, 2)
        qu = q + a.bias_u.view(1, heads, 1, hd)
        qv = q + a.bias_v.view(1, heads, 1, hd)
        rel_k = a.relative_k_proj(pos_embed).view(1, -1, heads, hd)
        bd_raw = qv @ rel_k.permute(0, 2, 3, 1)
        bd = a._rel_shift(bd_raw)[..., :T] * a.scaling
        ac = (qu @ k.transpose(2, 3)) * a.scaling
        scores = ac + bd.masked_fill(~attn_mask, float("-inf"))
        probs = scores.softmax(-1, dtype=torch.float32)
        ctx = (probs @ v).transpose(1, 2).reshape(1, T, heads * hd)
        attn_out = a.o_proj(ctx)
        r2 = r1 + attn_out

        n3 = layer.norm_conv(r2)
        c = layer.conv
        ct = n3.transpose(1, 2)
        pw1 = c.pointwise_conv1(ct)
        glu = torch.nn.functional.glu(pw1, dim=1)
        glu_masked = glu.masked_fill(torch.all(~attn_mask, dim=2), 0.0)
        dw = c.depthwise_conv(glu_masked)
        bn = c.norm(dw)
        act = c.activation(bn)
        conv_out = c.pointwise_conv2(act).transpose(1, 2)
        r3 = r2 + conv_out

        n4 = layer.norm_feed_forward2(r3)
        ff2 = layer.feed_forward2(n4)
        r4 = r3 + 0.5 * ff2
        layer_out = layer.norm_out(r4)

    # BatchNorm at inference is an affine, so it folds into the pointwise
    # convolution at load time. The fold is checked here rather than in Go,
    # where a mistake would look like a conv bug.
    with torch.no_grad():
        scale = c.norm.weight / torch.sqrt(c.norm.running_var + c.norm.eps)
        shift = c.norm.bias - c.norm.running_mean * scale
        folded = dw * scale[None, :, None] + shift[None, :, None]
    manifest["encoder"]["batchnorm_fold_gap"] = float((folded - bn).abs().max())
    manifest["encoder"]["batchnorm_eps"] = float(c.norm.eps)
    print(f"batchnorm folded to an affine: max abs gap {manifest['encoder']['batchnorm_fold_gap']:.3g}")

    # The whole stack by hand, which is what makes the internals above
    # describe the real forward pass.
    with torch.no_grad():
        h = subsampled
        hiddens = []
        for lyr in encoder.layers:
            h = lyr(h, attention_mask=attn_mask, position_embeddings=pos_embed)
            hiddens.append(h)
        enc_out = model.encoder(input_features=mel, attention_mask=mask)
        projected = model.encoder_projector(enc_out.last_hidden_state)
    manifest["encoder"]["stack_gap"] = float((h - enc_out.last_hidden_state).abs().max())
    manifest["encoder"]["layer0_gap"] = float((layer_out - hiddens[args.layer]).abs().max())
    print(f"by-hand stack vs encoder(): {manifest['encoder']['stack_gap']:.3g}; "
          f"by-hand layer {args.layer}: {manifest['encoder']['layer0_gap']:.3g}")

    # ------------------------------------------------- prediction net and joint
    dec = model.decoder
    blank = cfg.blank_token_id
    with torch.no_grad():
        start = torch.tensor([[blank]])
        emb0 = dec.embedding(start)
        lstm0, (h0, c0) = dec.lstm(emb0)
        dec0 = dec.decoder_projector(lstm0)
        # A second step, from the first step's state, so the port has a case
        # where the recurrence actually carries something.
        tok1 = torch.tensor([[1]])
        emb1 = dec.embedding(tok1)
        lstm1, (h1, c1) = dec.lstm(emb1, (h0, c0))
        dec1 = dec.decoder_projector(lstm1)

        joint_sum = projected[:, :1] + dec0
        joint_act = model.joint.activation(joint_sum)
        joint_logits = model.joint.head(joint_act)

    # ----------------------------------------------------- the TDT greedy loop
    # A hand-written port of ParakeetTDTGenerationMixin: the decoder state
    # advances only on a non-blank emission, the encoder cursor advances by
    # the predicted duration, and a blank that predicts duration 0 is forced
    # to 1 so the loop cannot stall.
    trace = []
    with torch.no_grad():
        t, state, prev = 0, None, torch.tensor([[blank]])
        dec_out = None
        while t < enc_valid and len(trace) < cfg.max_symbols_per_step * enc_valid:
            if dec_out is None or int(prev) != blank:
                emb = dec.embedding(prev)
                out, state = dec.lstm(emb, state)
                dec_out = dec.decoder_projector(out)
            logits = model.joint.head(model.joint.activation(projected[:, t : t + 1] + dec_out))[0, 0]
            token = int(logits[: cfg.vocab_size].argmax())
            duration = int(cfg.durations[int(logits[cfg.vocab_size :].argmax())])
            if token == blank and duration == 0:
                duration = 1
            trace.append({"t": t, "token": token, "duration": duration})
            prev = torch.tensor([[token]])
            t += duration

    tokens = [s["token"] for s in trace]
    emitted = [tok for tok in tokens if tok != blank]
    manual_text = processor.tokenizer.decode(emitted, skip_special_tokens=True, group_tokens=False)

    with torch.no_grad():
        gen = model.generate(input_features=mel, attention_mask=inputs.attention_mask)
    gen_tokens = gen.sequences[0].tolist()[1:]  # drop the prepended start token
    gen_durations = gen.durations[0].tolist()[1:]
    gen_text = processor.batch_decode(gen.sequences, skip_special_tokens=True)[0]

    manifest["decode"] = {
        "steps": len(trace), "trace": trace,
        "tokens": tokens, "emitted": emitted,
        "text": manual_text, "generate_text": gen_text,
        "generate_tokens": gen_tokens, "generate_durations": gen_durations,
        "manual_matches_generate": tokens == gen_tokens
        and [s["duration"] for s in trace] == gen_durations,
        "matches_expected": manual_text.strip() == EXPECTED,
    }
    print(f"\ntranscript ({len(trace)} steps, {len(emitted)} tokens): {manual_text!r}")
    print(f"  generate(): {gen_text!r}")
    print(f"  manual == generate: {manifest['decode']['manual_matches_generate']}")
    print(f"  == expected:        {manifest['decode']['matches_expected']}")

    # ------------------------------------------------------------------- dumps
    print("\ntensors:")
    dump("audio", torch.from_numpy(audio.copy()))
    dump("preemphasised", pre[0])
    dump("window", window)
    dump("mel_filters", fe.mel_filters)
    dump("stft_power", power[0].transpose(0, 1))
    dump("log_mel", log_mel[0])
    dump("mel", mel[0])
    dump("mel_mask", mask[0].float())
    for name, t in sub_outs:
        dump(name, t[0])
    dump("subsampled", subsampled[0])
    dump("pos_embed", pos_embed[0])
    dump("norm_ff1", n1[0])
    dump("ff1", ff1[0])
    dump("resid1", r1[0])
    dump("norm_self_att", n2[0])
    dump("q", q.transpose(1, 2).reshape(T, heads * hd))
    dump("k", k.transpose(1, 2).reshape(T, heads * hd))
    dump("v", v.transpose(1, 2).reshape(T, heads * hd))
    dump("rel_k", rel_k.reshape(-1, heads * hd))
    dump("matrix_bd_raw", bd_raw[0])
    dump("matrix_bd", bd[0])
    dump("matrix_ac", ac[0])
    dump("attn_probs", probs[0])
    dump("attn_ctx", ctx[0])
    dump("attn_out", attn_out[0])
    dump("resid2", r2[0])
    dump("norm_conv", n3[0])
    dump("conv_pw1", pw1[0])
    dump("conv_glu", glu[0])
    dump("conv_dw", dw[0])
    dump("conv_bn", bn[0])
    dump("conv_out", conv_out[0])
    dump("resid3", r3[0])
    dump("norm_ff2", n4[0])
    dump("ff2", ff2[0])
    dump("layer_out", layer_out[0])
    for i in (1, 2, 12, enc_cfg.num_hidden_layers - 1):
        dump(f"hidden_{i}", hiddens[i][0])
    dump("encoder_out", enc_out.last_hidden_state[0])
    dump("encoder_projected", projected[0])
    dump("dec_embed", emb0[0])
    dump("dec_lstm", lstm0[0])
    dump("dec_h", h0[:, 0])
    dump("dec_c", c0[:, 0])
    dump("dec_out", dec0[0])
    dump("dec_embed1", emb1[0])
    dump("dec_lstm1", lstm1[0])
    dump("dec_h1", h1[:, 0])
    dump("dec_c1", c1[:, 0])
    dump("dec_out1", dec1[0])
    dump("joint_sum", joint_sum[0])
    dump("joint_act", joint_act[0])
    dump("joint_logits", joint_logits[0])

    # --------------------------------------------------- does fp16 hold here?
    # Every activation on the path, biggest first. fp16's largest finite is
    # 65504 and its subnormals start at 6e-5; the number that matters is how
    # much headroom the widest tensor has.
    survey = {n: v["absmax"] for n, v in manifest["tensors"].items()}
    manifest["fp16"] = {
        "max_activation": max(survey.values()),
        "top": sorted(survey.items(), key=lambda kv: -kv[1])[:10],
        "weight_absmax": {
            n: float(p.detach().abs().max())
            for n, p in sorted(model.named_parameters(), key=lambda kv: -float(kv[1].detach().abs().max()))[:10]
        },
    }
    print(f"\nfp16 headroom: largest activation {manifest['fp16']['max_activation']:.4g} "
          f"against 65504; largest weight {max(manifest['fp16']['weight_absmax'].values()):.4g}")

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
