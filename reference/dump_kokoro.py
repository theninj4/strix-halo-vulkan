"""Dump a reference run of Kokoro-82M, for stages T2-T4 of SPEECH.md.

The oracle is hexgrad's own `kokoro.KModel` in fp32 on CPU, driven from a fixed
phoneme string so that grapheme-to-phoneme stays outside the model, where
SPEECH.md puts it: the first vertical takes phonemes, not text. The phonemes
here came from `misaki.en.G2P` and are recorded verbatim in the manifest, so
this dump reproduces without misaki, spacy or espeak installed.

Like `dump_parakeet.py` this walks the model rather than its ends, because
every stage of the Go port wants a different point in it. Unlike parakeet, the
walk has to settle four things first, and each one is a finding rather than a
tensor:

  * **The vocoder is stochastic.** `SineGen` draws a random initial phase for
    each of the eight harmonics and adds Gaussian noise to the excitation, so
    two runs of `KModel` on the same input give different samples. There is no
    seed to match in Go and no tolerance that survives it. So the primary dump
    runs with every random draw replaced by zeros -- deterministic, and what
    the Go port is checked against -- and a second, seeded run is written as a
    WAV next to it with the difference between the two recorded. The noise is
    then an isolated, explicit input rather than a mystery in the residuals.

  * **`weight_norm` and the AdaIN affine.** Both are settled by
    `convert_kokoro.py`; this script reads the folded safetensors' own
    `model.safetensors` *and* the live `KModel`, and checks the two agree, so
    the dump describes the file the Go side actually opens.

  * **The length regulator.** `pred_dur` makes the output length depend on the
    input's *content*, which nothing in z-image or parakeet did. The durations,
    the alignment matrix and both sides of the expansion are dumped, since a
    single off-by-one there desynchronises everything downstream.

  * **The iSTFT.** The vocoder ends in `torch.istft` at `n_fft=20`, `hop=5`,
    a periodic Hann window and `center=True`. `audio/` already has an inverse
    FFT; what it does not have is this overlap-add, so the spectrogram going
    in and the waveform coming out are both dumped to check it in isolation.

Self-checks, each recorded in the manifest: the by-hand ALBERT layer against
the library's, the by-hand duration encoder, AdaIN block, decoder and generator
against their modules, and the whole by-hand chain against
`KModel.forward_with_tokens`.

    .venv/bin/python reference/dump_kokoro.py
"""

import argparse
import contextlib
import json
import math
import os
import wave

import numpy as np
import torch
import torch.nn.functional as F

from kokoro.model import KModel

# A sentence whose every word is in misaki's dictionary, so the phonemes below
# need no espeak fallback and no ❓. Recorded rather than recomputed: see the
# module docstring.
TEXT = "The quick brown fox jumps over the lazy dog."
PHONEMES = "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ."


@contextlib.contextmanager
def no_randomness():
    """Replace every random draw with zeros, for the length of a forward pass.

    `SineGen._f02sine` calls `torch.rand` for the harmonics' initial phases and
    `SourceModuleHnNSF.forward` calls `torch.randn_like` twice. Those are the
    only three, and both modules reach them through the module-level `torch`,
    so patching the two functions is enough.
    """
    rand, randn_like = torch.rand, torch.randn_like
    torch.rand = lambda *a, **k: torch.zeros(*a, **{kk: vv for kk, vv in k.items() if kk != "generator"})
    torch.randn_like = lambda t, *a, **k: torch.zeros_like(t)
    try:
        yield
    finally:
        torch.rand, torch.randn_like = rand, randn_like


def write_wav(path, x, rate=24000):
    pcm = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767.0).round().astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def adain(norm_module, fc, x, s):
    """`AdaIN1d.forward`, written out: instance norm over time, then a style affine.

    The `InstanceNorm1d` is per (batch, channel) over the time axis, unbiased
    variance off, and its own affine is identity -- see `convert_kokoro.py`.
    Written out here because it is the operator the whole decoder is built from
    and because "normalise over T, not over C" is the one thing a port of it
    gets wrong.
    """
    h = fc(s).view(s.shape[0], -1, 1)
    gamma, beta = torch.chunk(h, 2, dim=1)
    mean = x.mean(dim=2, keepdim=True)
    var = x.var(dim=2, unbiased=False, keepdim=True)
    xn = (x - mean) / torch.sqrt(var + norm_module.eps)
    return (1 + gamma) * xn + beta


def snake(x, alpha):
    """Snake1D: `x + (1/a) * sin(a*x)^2`, the generator's activation."""
    return x + (1.0 / alpha) * torch.sin(alpha * x).pow(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Kokoro-82M")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--out", default="reference/out/kokoro")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    cfg_path = os.path.join(args.model, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        config = json.load(fh)
    model = KModel(
        repo_id="hexgrad/Kokoro-82M", config=cfg_path,
        model=os.path.join(args.model, "kokoro-v1_0.pth"),
    ).eval()

    # ---------------------------------------------------------------- the input
    vocab = model.vocab
    unknown = sorted({p for p in PHONEMES if p not in vocab})
    if unknown:
        raise SystemExit(f"phonemes outside the vocabulary: {unknown}")
    ids = [vocab[p] for p in PHONEMES]
    input_ids = torch.LongTensor([[0, *ids, 0]])
    N = input_ids.shape[1]
    voice = torch.load(
        os.path.join(args.model, "voices", args.voice + ".pt"),
        map_location="cpu", weights_only=True,
    )
    ref_s = voice[len(PHONEMES) - 1]  # KPipeline's index: one row per phoneme count
    s_pred, s_dec = ref_s[:, 128:], ref_s[:, :128]
    print(f"{TEXT!r}\n  -> {PHONEMES!r}\n  -> {N} tokens (0 + {len(ids)} + 0), "
          f"voice {args.voice} row {len(PHONEMES) - 1} of {voice.shape[0]}")

    manifest = {
        "text": TEXT, "phonemes": PHONEMES, "input_ids": input_ids[0].tolist(),
        "voice": args.voice, "voice_row": len(PHONEMES) - 1,
        "voice_shape": list(voice.shape), "speed": args.speed,
        "sampling_rate": 24000,
        "config": {
            "n_token": config["n_token"], "hidden_dim": config["hidden_dim"],
            "style_dim": config["style_dim"], "max_dur": config["max_dur"],
            "n_layer": config["n_layer"],
            "text_encoder_kernel_size": config["text_encoder_kernel_size"],
            "istftnet": config["istftnet"],
            "plbert": dict(config["plbert"], **{
                "hidden_act": model.bert.config.hidden_act,
                "embedding_size": model.bert.config.embedding_size,
                "layer_norm_eps": model.bert.config.layer_norm_eps,
                "shared_layers": True,
            }),
        },
        "tensors": {}, "fp16": {},
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
        print(f"  {name:26s} {str(list(t.shape)):20s} sum={flat.double().sum():+.5f} absmax={absmax:.4g}")

    # ------------------------------------------------------------------- ALBERT
    # No padding at batch 1, so the mask is all ones and attention is full; it
    # is passed anyway, the way KModel passes it.
    bert = model.bert
    text_mask = torch.zeros(1, N, dtype=torch.bool)
    with torch.no_grad():
        bert_dur = bert(input_ids, attention_mask=(~text_mask).int())

        emb = bert.embeddings(input_ids)
        h0 = bert.encoder.embedding_hidden_mapping_in(emb)
        layer = bert.encoder.albert_layer_groups[0].albert_layers[0]
        a, heads = layer.attention, bert.config.num_attention_heads
        hd = bert.config.hidden_size // heads

        # One layer by hand: post-LayerNorm on both the attention and the FFN,
        # which is what makes ALBERT not a conformer layer.
        q = a.query(h0).view(1, N, heads, hd).transpose(1, 2)
        k = a.key(h0).view(1, N, heads, hd).transpose(1, 2)
        v = a.value(h0).view(1, N, heads, hd).transpose(1, 2)
        scores = (q @ k.transpose(2, 3)) * a.scaling
        probs = scores.softmax(-1)
        ctx = (probs @ v).transpose(1, 2).reshape(1, N, heads * hd)
        attn_out = a.LayerNorm(h0 + a.dense(ctx))
        ffn = layer.ffn_output(layer.activation(layer.ffn(attn_out)))
        layer_out = layer.full_layer_layer_norm(ffn + attn_out)

        hiddens, h = [], h0
        for _ in range(bert.config.num_hidden_layers):
            h = layer(h)
            hiddens.append(h)

    manifest["bert"] = {
        "layers": bert.config.num_hidden_layers, "heads": heads, "head_dim": hd,
        "layer0_gap": float((layer_out - hiddens[0]).abs().max()),
        "stack_gap": float((h - bert_dur).abs().max()),
        "note": "12 layers sharing one weight group; post-LayerNorm; "
                f"{bert.config.hidden_act} in the FFN",
    }
    print(f"albert: by-hand layer 0 {manifest['bert']['layer0_gap']:.3g}, "
          f"stack vs bert() {manifest['bert']['stack_gap']:.3g}")

    # ------------------------------------------------- duration encoder + LSTM
    pred = model.predictor
    with torch.no_grad():
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)          # [1, 512, N]

        # DurationEncoder by hand: three (bidirectional LSTM, AdaLayerNorm)
        # pairs, with the 128-wide style vector re-concatenated after each
        # normalisation, so every LSTM sees 512 + 128 = 640 inputs.
        s_rep = s_pred.unsqueeze(0).expand(N, 1, -1)                    # [N, 1, 128]
        x = torch.cat([d_en.permute(2, 0, 1), s_rep], dim=-1)           # [N, 1, 640]
        x = x.transpose(0, 1).transpose(-1, -2)                         # [1, 640, N]
        de_steps = []
        for i, block in enumerate(pred.text_encoder.lstms):
            if isinstance(block, torch.nn.LSTM):
                y, _ = block(x.transpose(-1, -2))
                x = y.transpose(-1, -2)
                de_steps.append((f"dur_lstm_{i // 2}", x.clone()))
            else:
                x = block(x.transpose(-1, -2), s_pred).transpose(-1, -2)
                x = torch.cat([x, s_rep.permute(1, 2, 0)], dim=1)
                de_steps.append((f"dur_norm_{i // 2}", x.clone()))
        d_hand = x.transpose(-1, -2)                                    # [1, N, 640]
        d = pred.text_encoder(d_en, s_pred, torch.LongTensor([N]), text_mask)

        lstm_out, _ = pred.lstm(d)                                      # [1, N, 512]
        dur_logits = pred.duration_proj(lstm_out)                       # [1, N, 50]
        duration = torch.sigmoid(dur_logits).sum(axis=-1) / args.speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

    manifest["predictor"] = {"duration_encoder_gap": float((d_hand - d).abs().max())}
    print(f"duration encoder: by-hand vs module {manifest['predictor']['duration_encoder_gap']:.3g}")

    # --------------------------------------------------------- length regulator
    L = int(pred_dur.sum())
    indices = torch.repeat_interleave(torch.arange(N), pred_dur)
    aln = torch.zeros(N, L)
    aln[indices, torch.arange(L)] = 1
    aln = aln.unsqueeze(0)
    manifest["length_regulator"] = {
        "tokens": N, "frames": L, "durations": pred_dur.tolist(),
        "samples": L * 600, "seconds": L * 600 / 24000,
        "note": "one alignment frame is 600 samples at 24 kHz = 25 ms; the expansion is "
                "a gather (frame f reads token indices[f]), not the [N, L] matmul torch writes",
    }
    print(f"length regulator: {N} tokens -> {L} frames -> {L * 600} samples "
          f"({L * 600 / 24000:.3f} s), durations {pred_dur.min()}..{pred_dur.max()}")

    # ------------------------------------------------------------- F0 and noise
    with torch.no_grad():
        en = d.transpose(-1, -2) @ aln                                  # [1, 640, L]
        shared_out, _ = pred.shared(en.transpose(-1, -2))               # [1, L, 512]

        # One AdainResBlk1d by hand, against the module: the predictor's first
        # F0 block. Every other block in the predictor and the decoder is the
        # same shape of thing, so this is the check that covers them.
        blk = pred.F0[0]
        xb = shared_out.transpose(-1, -2)
        r = adain(blk.norm1.norm, blk.norm1.fc, xb, s_pred)
        r = blk.actv(r)
        r = blk.conv1(r)
        r = adain(blk.norm2.norm, blk.norm2.fc, r, s_pred)
        r = blk.actv(r)
        r = blk.conv2(r)
        blk_hand = (r + xb) * torch.rsqrt(torch.tensor(2.0))
        blk_ref = blk(xb, s_pred)

        f0_steps, F0 = [], shared_out.transpose(-1, -2)
        for i, block in enumerate(pred.F0):
            F0 = block(F0, s_pred)
            f0_steps.append((f"f0_blk_{i}", F0.clone()))
        F0_pred = pred.F0_proj(F0).squeeze(1)                           # [1, 2L]
        n_steps, Nx = [], shared_out.transpose(-1, -2)
        for i, block in enumerate(pred.N):
            Nx = block(Nx, s_pred)
            n_steps.append((f"n_blk_{i}", Nx.clone()))
        N_pred = pred.N_proj(Nx).squeeze(1)                             # [1, 2L]

        F0_ref, N_ref = pred.F0Ntrain(en, s_pred)
    manifest["predictor"]["adain_resblk_gap"] = float((blk_hand - blk_ref).abs().max())
    manifest["predictor"]["f0n_gap"] = float(max((F0_pred - F0_ref).abs().max(),
                                                 (N_pred - N_ref).abs().max()))
    print(f"AdainResBlk1d by hand: {manifest['predictor']['adain_resblk_gap']:.3g}; "
          f"F0Ntrain by hand: {manifest['predictor']['f0n_gap']:.3g}")

    # ------------------------------------------------------------ text encoder
    with torch.no_grad():
        te = model.text_encoder
        m = text_mask.unsqueeze(1)
        xt = te.embedding(input_ids).transpose(1, 2)
        te_steps = [("te_embed", xt.clone())]
        for i, c in enumerate(te.cnn):
            xt = c(xt)
            te_steps.append((f"te_cnn_{i}", xt.clone()))
        xt, _ = te.lstm(xt.transpose(1, 2))
        t_en_hand = xt.transpose(-1, -2)
        t_en = te(input_ids, torch.LongTensor([N]), text_mask)
        asr = t_en @ aln                                                # [1, 512, L]
    manifest["text_encoder"] = {"gap": float((t_en_hand - t_en).abs().max())}
    print(f"text encoder: by-hand vs module {manifest['text_encoder']['gap']:.3g}")

    # ---------------------------------------------------------------- iSTFTNet
    dec, gen = model.decoder, model.decoder.generator
    with torch.no_grad(), no_randomness():
        F0_c = dec.F0_conv(F0_pred.unsqueeze(1))                        # [1, 1, L]
        N_c = dec.N_conv(N_pred.unsqueeze(1))
        x = torch.cat([asr, F0_c, N_c], axis=1)                         # [1, 514, L]
        x = dec.encode(x, s_dec)                                        # [1, 1024, L]
        asr_res = dec.asr_res(asr)                                      # [1, 64, L]
        dec_steps = [("dec_encode", x.clone())]
        res = True
        for i, block in enumerate(dec.decode):
            if res:
                x = torch.cat([x, asr_res, F0_c, N_c], axis=1)
            x = block(x, s_dec)
            dec_steps.append((f"dec_decode_{i}", x.clone()))
            if block.upsample_type != "none":
                res = False

        # The source module, by hand, with the randomness zeroed: an F0 curve
        # upsampled 300x, wrapped to a phase, decimated, integrated and
        # re-interpolated, then nine harmonics of it.
        up = math.prod(config["istftnet"]["upsample_rates"]) * config["istftnet"]["gen_istft_hop_size"]
        f0_up = gen.f0_upsamp(F0_pred[:, None]).transpose(1, 2)         # [1, 600L, 1]
        fn = f0_up * torch.arange(1, gen.m_source.l_sin_gen.harmonic_num + 2, dtype=torch.float32)
        rad = (fn / 24000.0) % 1
        rad_d = F.interpolate(rad.transpose(1, 2), scale_factor=1 / up, mode="linear").transpose(1, 2)
        phase = torch.cumsum(rad_d, dim=1) * 2 * math.pi
        phase = F.interpolate(phase.transpose(1, 2) * up, scale_factor=up, mode="linear").transpose(1, 2)
        sines = torch.sin(phase) * gen.m_source.l_sin_gen.sine_amp
        uv = (f0_up > gen.m_source.l_sin_gen.voiced_threshold).float()
        sine_waves = sines * uv                                         # noise zeroed
        har_source_hand = gen.m_source.l_tanh(gen.m_source.l_linear(sine_waves))
        har_ref, _, uv_ref = gen.m_source(f0_up)

        har_source = har_source_hand.transpose(1, 2).squeeze(1)         # [1, 600L]
        har_spec, har_phase = gen.stft.transform(har_source)            # [1, 11, 120L+1]
        har = torch.cat([har_spec, har_phase], dim=1)                   # [1, 22, 120L+1]

        gx, gen_steps = x, []
        for i in range(gen.num_upsamples):
            gx = F.leaky_relu(gx, negative_slope=0.1)
            x_source = gen.noise_res[i](gen.noise_convs[i](har), s_dec)
            gx = gen.ups[i](gx)
            if i == gen.num_upsamples - 1:
                gx = gen.reflection_pad(gx)
            gx = gx + x_source
            xs = None
            for j in range(gen.num_kernels):
                b = gen.resblocks[i * gen.num_kernels + j](gx, s_dec)
                xs = b if xs is None else xs + b
            gx = xs / gen.num_kernels
            gen_steps.append((f"gen_up_{i}", gx.clone()))
        gx = F.leaky_relu(gx)
        post = gen.conv_post(gx)                                        # [1, 22, 120L+1]
        spec = torch.exp(post[:, : gen.post_n_fft // 2 + 1, :])
        ph = torch.sin(post[:, gen.post_n_fft // 2 + 1:, :])
        audio_hand = gen.stft.inverse(spec, ph).squeeze()

        gen_ref = gen(x, s_dec, F0_pred).squeeze()
        dec_ref = dec(asr, F0_pred, N_pred, s_dec).squeeze()
        audio_ref, dur_ref = model.forward_with_tokens(input_ids, ref_s, args.speed)

    manifest["decoder"] = {
        "source_gap": float((har_source_hand - har_ref).abs().max()),
        "generator_gap": float((audio_hand - gen_ref).abs().max()),
        "decoder_gap": float((audio_hand - dec_ref).abs().max()),
        "end_to_end_gap": float((audio_hand - audio_ref).abs().max()),
        "durations_match": pred_dur.tolist() == dur_ref.tolist(),
        "samples": int(audio_hand.numel()),
        "istft": {
            "n_fft": gen.stft.filter_length, "hop": gen.stft.hop_length,
            "win_length": gen.stft.win_length, "window": "hann, periodic=True",
            "center": True, "frames": int(spec.shape[-1]),
            "note": "torch.istft's default center=True and normalized=False; the "
                    "window-sum normalisation is over the same frames",
        },
    }
    print(f"source {manifest['decoder']['source_gap']:.3g}, "
          f"generator {manifest['decoder']['generator_gap']:.3g}, "
          f"decoder {manifest['decoder']['decoder_gap']:.3g}, "
          f"end to end {manifest['decoder']['end_to_end_gap']:.3g}, "
          f"durations match {manifest['decoder']['durations_match']}")
    print(f"audio: {audio_hand.numel()} samples = {audio_hand.numel() / 24000:.3f} s")

    # ------------------------------------------ what the noise is actually worth
    torch.manual_seed(0)
    with torch.no_grad():
        audio_rand, _ = model.forward_with_tokens(input_ids, ref_s, args.speed)
    diff = (audio_rand - audio_ref)
    manifest["stochastic"] = {
        "max_abs_diff": float(diff.abs().max()),
        "rms_diff": float(diff.pow(2).mean().sqrt()),
        "rms_signal": float(audio_ref.pow(2).mean().sqrt()),
        "note": "SineGen draws 8 harmonic phases from torch.rand and adds Gaussian noise; "
                "the primary dump zeroes both, so the Go port is deterministic and the "
                "noise is an explicit input rather than a residual",
    }
    snr = 20 * math.log10(manifest["stochastic"]["rms_signal"] / manifest["stochastic"]["rms_diff"])
    manifest["stochastic"]["snr_db"] = snr
    print(f"randomness is worth {manifest['stochastic']['rms_diff']:.4g} rms against "
          f"{manifest['stochastic']['rms_signal']:.4g} of signal ({snr:.1f} dB)")

    write_wav(os.path.join(args.out, "audio.wav"), audio_ref.numpy())
    write_wav(os.path.join(args.out, "audio_stochastic.wav"), audio_rand.numpy())

    # ------------------------------------------------------------------- dumps
    print("\ntensors:")
    dump("input_ids", input_ids[0].float())
    dump("ref_s", ref_s[0])
    dump("bert_embed", emb[0])
    dump("bert_hidden_in", h0[0])
    dump("bert_q", q.transpose(1, 2).reshape(N, heads * hd))
    dump("bert_k", k.transpose(1, 2).reshape(N, heads * hd))
    dump("bert_v", v.transpose(1, 2).reshape(N, heads * hd))
    dump("bert_probs", probs[0])
    dump("bert_attn_out", attn_out[0])
    dump("bert_layer_out", layer_out[0])
    for i in (5, bert.config.num_hidden_layers - 1):
        dump(f"bert_hidden_{i}", hiddens[i][0])
    dump("bert_out", bert_dur[0])
    dump("d_en", d_en[0])
    for name, t in de_steps:
        dump(name, t[0])
    dump("dur_encoded", d[0])
    dump("dur_lstm_out", lstm_out[0])
    dump("dur_logits", dur_logits[0])
    dump("durations", duration[0])
    dump("pred_dur", pred_dur.float())
    dump("alignment", aln[0])
    dump("en", en[0])
    dump("shared_out", shared_out[0])
    for name, t in f0_steps + n_steps:
        dump(name, t[0])
    dump("f0_pred", F0_pred[0])
    dump("n_pred", N_pred[0])
    for name, t in te_steps:
        dump(name, t[0])
    dump("t_en", t_en[0])
    dump("asr", asr[0])
    dump("f0_conv", F0_c[0])
    dump("n_conv", N_c[0])
    dump("asr_res", asr_res[0])
    for name, t in dec_steps:
        dump(name, t[0])
    dump("f0_upsampled", f0_up[0])
    dump("sine_waves", sine_waves[0])
    dump("uv", uv_ref[0])
    dump("har_source", har_source[0])
    dump("har_spec", har_spec[0])
    dump("har_phase", har_phase[0])
    dump("noise_conv_0", gen.noise_convs[0](har)[0])
    for name, t in gen_steps:
        dump(name, t[0])
    dump("gen_post", post[0])
    dump("gen_spec", spec[0])
    dump("gen_phase", ph[0])
    dump("audio", audio_hand)

    # --------------------------------------------------- does fp16 hold here?
    survey = {n: v["absmax"] for n, v in manifest["tensors"].items()}
    manifest["fp16"] = {
        "max_activation": max(survey.values()),
        "top": sorted(survey.items(), key=lambda kv: -kv[1])[:10],
    }
    print(f"\nfp16 headroom: largest activation {manifest['fp16']['max_activation']:.4g} "
          f"against 65504")
    for n, a in manifest["fp16"]["top"][:5]:
        print(f"  {n:26s} {a:.4g}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
