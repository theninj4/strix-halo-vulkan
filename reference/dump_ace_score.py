"""Dump ACE-Step 1.5's own song scorers, for MUSIC.md A14b.

Upstream's Gradio "Auto Score" (ui/gradio/events/results/scoring.py) runs
two scorers on a finished song; its API exposes neither. Both are driven
here through upstream's own functions, fp32 on the CPU, with hooks that
record what a port must reproduce:

* **PMI** (`core/scoring/lm_score.calculate_pmi_score_per_condition`): the
  5 Hz LM reads the song's codes in the understand prompt and is
  teacher-forced on each condition. Every `_get_logits_and_target_for_scoring`
  call is recorded in order: the prompt and target text, the full ids, and
  per target position the log-softmax of the target token and the top-10
  ids. Then the per-condition scores and the reward. Cases:

  - ``pmi_match``: the codes of A1's full_metas song (A13's ``understand``
    case, `reference/out/acelm/understand_codes`), scored against the
    request that made it (caption, lyrics, bpm 120, C major, 4, 30 s, en);
  - ``pmi_mismatch``: the lofi reference's codes (``understand_lofi``)
    against the same request -- the control the score has to rank lower.

* **Lyric alignment** (`core/generation/handler/lyric_score.get_lyric_score`
  and `core/scoring/dit_score.MusicLyricScorer`): one decoder forward on two
  rows (seed-42 noise at t = 1; the final latent noised to t = 1/8), the
  cross-attention of the checkpoint's `lyric_alignment_layers_config` heads,
  DTW, coverage² · monotonicity² · confidence. A1's requests are generated
  first (thinking off, seed 42, 8 steps), as Gradio scores what it just
  made. Per case: the decoder inputs, the noise, the heads' attention
  probabilities [2, heads, tokens, encoder rows], the lyric slice, and for
  each row the energy matrix, the DTW path, the three metrics and the score.
  Cases: ``align_full_metas`` (30 s, English), ``align_zh_odd`` (12.34 s,
  Chinese), ``align_instrumental_60`` (60 s, "[Instrumental]": no sung
  rows).

--bf16 runs both the way upstream serves them on CUDA, into
reference/out/acescore_bf16: the LM and the DiT in bfloat16, the alignment
cases from the fp32 run's inputs. That is the bar our drift is priced
against (decision 2).

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv-acestep/bin/python reference/dump_ace_score.py [--bf16] [label ...]
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "models/ACE-Step-1.5-src")
sys.path.insert(0, "reference")

import acestep.core.scoring.lm_score as lm_score  # noqa: E402
from acestep.core.scoring.dit_score import MusicLyricScorer  # noqa: E402
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from dump_ace_plan import CASES, PROJECT  # noqa: E402

BF16 = "--bf16" in sys.argv
OUT = "reference/out/acescore_bf16" if BF16 else "reference/out/acescore"
FP32_OUT = "reference/out/acescore"
CKPT = os.path.join(PROJECT, "checkpoints")
LABELS = ["pmi_match", "pmi_mismatch", "align_full_metas", "align_zh_odd", "align_instrumental_60"]
STEPS = 8
TOPK = 10


class Stop(Exception):
    pass


class TokenizerProxy:
    """Forwards to the real tokenizer and records what it was asked."""

    def __init__(self, tok):
        self._tok = tok
        self.calls = []

    def __call__(self, text, *args, **kwargs):
        out = self._tok(text, *args, **kwargs)
        self.calls.append({"text": text, "max_length": kwargs.get("max_length"),
                           "ids": out["input_ids"][0].tolist()})
        return out

    def __getattr__(self, name):
        return getattr(self._tok, name)


def main():
    labels = [a for a in sys.argv[1:] if not a.startswith("--")] or LABELS
    os.makedirs(OUT, exist_ok=True)
    manifest = {"cases": {}, "tensors": {}}
    path = os.path.join(OUT, "manifest.json")
    if os.path.exists(path):  # other labels' cases are kept
        with open(path) as fh:
            manifest = json.load(fh)

    def dump(name, t, dtype="float32"):
        arr = t.detach().contiguous().cpu().float().numpy().astype(dtype) if torch.is_tensor(t) \
            else np.asarray(t, dtype=dtype)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(arr.tobytes())
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": dtype}

    def load(name, dtype=np.float32, out=FP32_OUT):
        with open(os.path.join(out, "manifest.json")) as fh:
            shape = json.load(fh)["tensors"][name]["shape"]
        return np.fromfile(os.path.join(out, name + ".bin"), dtype=dtype).reshape(shape)

    torch.set_grad_enabled(False)
    h = AceStepHandler()
    msg, ok = h.initialize_service(project_root=PROJECT, config_path="acestep-v15-xl-turbo", device="cpu",
                                   use_mlx_dit=False)
    assert ok, msg
    if BF16:
        h.model.to(torch.bfloat16)
        h.dtype = torch.bfloat16
        if getattr(h, "text_encoder", None) is not None:
            h.text_encoder.to(torch.bfloat16)
    lm = None

    full = dict(CASES)["full_metas"]
    for label in labels:
        t0 = time.time()
        rec = {}
        if label.startswith("pmi_"):
            if lm is None:
                lm = LLMHandler()
                msg, ok = lm.initialize(checkpoint_dir=CKPT, lm_model_path="acestep-5Hz-lm-4B", backend="pt",
                                        device="cpu", dtype=torch.float32)
                assert ok, msg
                lm.disable_tqdm = True
                if BF16:
                    lm.get_hf_model_for_scoring().to(torch.bfloat16)
            src = {"pmi_match": "understand", "pmi_mismatch": "understand_lofi"}[label]
            codes = load(f"{src}_codes", np.int32, "reference/out/acelm").tolist()
            code_str = "".join(f"<|audio_code_{c}|>" for c in codes)
            metadata = {"bpm": int(full["bpm"]), "keyscale": full["keyscale"], "timesignature": full["timesignature"],
                        "duration": int(full["duration"]), "language": full["vocal_language"],
                        "caption": full["caption"]}
            rec.update({"codes_from": src, "codes": len(codes), "caption": full["caption"],
                        "lyrics": full["lyrics"], "metadata": metadata, "calls": []})

            real = lm_score._get_logits_and_target_for_scoring

            def spy(handler, formatted_prompt, target_text, rec=rec, label=label):
                logits, target = real(handler, formatted_prompt, target_text)
                i = len(rec["calls"])
                lp = F.log_softmax(logits, dim=-1)
                gathered = lp[torch.arange(target.shape[0]), target]
                top = torch.topk(logits, k=TOPK, dim=-1).indices
                full_ids = handler.llm_tokenizer(formatted_prompt + target_text, add_special_tokens=True)["input_ids"]
                dump(f"{label}_call{i}_ids", full_ids, "int32")
                dump(f"{label}_call{i}_logp", gathered.float())
                dump(f"{label}_call{i}_top", top, "int32")
                rec["calls"].append({"prompt_tail": formatted_prompt[-120:], "unconditional": "NO USER INPUT" in
                                     formatted_prompt, "target": target_text, "prompt_len":
                                     len(full_ids) - int(target.shape[0]), "target_len": int(target.shape[0]),
                                     "mean_logp": float(gathered.float().mean())})
                return logits, target
            lm_score._get_logits_and_target_for_scoring = spy
            scores, reward, status = lm_score.calculate_pmi_score_per_condition(
                llm_handler=lm, audio_codes=code_str, caption=full["caption"], lyrics=full["lyrics"],
                metadata=dict(metadata), temperature=1.0, topk=TOPK, score_scale=0.5)
            lm_score._get_logits_and_target_for_scoring = real
            assert scores, status
            rec.update({"scores": scores, "reward": reward, "score_scale": 0.5})
            print(f"{label}: reward {reward:.4f} {scores}")

        elif label.startswith("align_"):
            case = label[len("align_"):]
            p = label + "_"
            if BF16:
                enc = torch.from_numpy(load(p + "encoder_states"))[None]
                ctx = torch.from_numpy(load(p + "context_latents"))[None]
                lat = torch.from_numpy(load(p + "latents"))[None]
                ids = load(p + "lyric_ids", np.int32).tolist()
                with open(os.path.join(FP32_OUT, "manifest.json")) as fh:
                    lang = json.load(fh)["cases"][label]["vocal_language"]
            else:
                # Generate the song as Gradio would have, then score it.
                captured = {}

                def capture(**kwargs):
                    captured.update(kwargs)
                    raise Stop()
                real_generate = h.model.generate_audio
                h.model.generate_audio = capture
                proxy = TokenizerProxy(h.text_tokenizer)
                h.text_tokenizer = proxy
                params = GenerationParams(thinking=False, shift=3.0, seed=42, use_cot_metas=False,
                                          use_cot_caption=False, use_cot_language=False, **dict(CASES)[case])
                generate_music(h, None, params, GenerationConfig(batch_size=1, use_random_seed=False, seeds=[42]))
                h.model.generate_audio = real_generate
                h.text_tokenizer = proxy._tok
                assert captured, label
                lyric_call = next(c for c in proxy.calls if c["max_length"] == 2048)
                ids = lyric_call["ids"]
                prompt = lyric_call["text"]
                lang = prompt.split("# Languages\n", 1)[1].split("\n\n# Lyric", 1)[0]
                seen = {}

                def enc_out(_m, _inp, out):
                    seen.setdefault("enc", out[0].detach().clone())

                def dec_pre(_m, _args, kwargs):
                    seen.setdefault("ctx", kwargs["context_latents"].detach().clone())
                hooks = [h.model.encoder.register_forward_hook(enc_out),
                         h.model.decoder.register_forward_pre_hook(dec_pre, with_kwargs=True)]
                lat = h.model.generate_audio(**captured)["target_latents"]
                for hk in hooks:
                    hk.remove()
                enc, ctx = seen["enc"], seen["ctx"]
                dump(p + "encoder_states", enc[0])
                dump(p + "context_latents", ctx[0])
                dump(p + "latents", lat[0])
                dump(p + "lyric_ids", ids, "int32")
                rec["lyrics_prompt"] = prompt

            dt = h.dtype
            cfg = h.custom_layers_config
            heads = [(l, hd) for l in sorted(cfg) for hd in cfg[l]]
            rec.update({"vocal_language": lang, "layers_config": {str(k): v for k, v in cfg.items()},
                        "heads": heads, "steps": STEPS, "latent_length": int(lat.shape[1]),
                        "encoder_len": int(enc.shape[1])})

            # The attention, as get_lyric_score's decoder call returns it.
            dec = h.model.decoder
            real_fwd = dec.forward
            got = {}

            def fwd(*args, **kwargs):
                out = real_fwd(*args, **kwargs)
                got["attn"] = out[2]
                got["xt"] = kwargs["hidden_states"].detach().clone()
                return out
            dec.forward = fwd
            res = h.get_lyric_score(pred_latent=lat.to(dt), encoder_hidden_states=enc.to(dt),
                                    encoder_attention_mask=torch.ones(enc.shape[:2], dtype=dt),
                                    context_latents=ctx.to(dt), lyric_token_ids=torch.tensor([ids]),
                                    vocal_language=lang, inference_steps=STEPS, seed=42)
            dec.forward = real_fwd
            assert res["success"], res
            noise = h._sample_noise_like(lat.to(dt), 42)
            dump(p + "noise", noise[0])
            dump(p + "xt", got["xt"])
            # [2, heads, tokens, encoder rows]: each listed head's softmax.
            probs = torch.stack([got["attn"][l][:, hd] for l, hd in heads], dim=1)
            dump(p + "probs", probs)

            _, pure, start, end = h._extract_lyric_segment(ids, lang)
            rec.update({"lyric_start": start, "lyric_end": end, "pure_lyric_ids": pure})
            scorer = MusicLyricScorer(h.text_tokenizer)
            rec["type_mask"] = scorer._generate_token_type_mask(pure).tolist()
            rows = {}
            for r, name in enumerate(["lm", "dit"]):
                stacked = torch.stack([a.transpose(-1, -2) for a in got["attn"]])[:, r]
                m = stacked[..., start:end, :]
                info = scorer.lyrics_alignment_info(attention_matrix=m, token_ids=pure, custom_config=cfg,
                                                    return_matrices=False, medfilt_width=1)
                energy = info["energy_matrix"]
                cov, mono, conf = scorer._compute_alignment_metrics(
                    energy_matrix=torch.tensor(energy, dtype=torch.float32), path_coords=torch.tensor(
                        info["path_coords"]), type_mask=torch.tensor(info["type_mask"]))
                score = scorer.calculate_score(energy_matrix=energy, type_mask=info["type_mask"],
                                               path_coords=info["path_coords"])["lyrics_score"]
                dump(p + f"{name}_energy", torch.from_numpy(np.asarray(energy, dtype=np.float32)))
                dump(p + f"{name}_path", np.asarray(info["path_coords"], dtype=np.int32), "int32")
                rows[name] = {"coverage": cov, "monotonicity": mono, "confidence": conf, "score": score}
            rec.update({"rows": rows, "lm_score": res["lm_score"], "dit_score": res["dit_score"]})
            print(f"{label}: T={rec['latent_length']} enc {rec['encoder_len']} lyrics [{start}:{end}] "
                  f"lm {res['lm_score']:.4f} dit {res['dit_score']:.4f} {rows}")
        else:
            raise SystemExit(f"unknown label {label}")

        rec["seconds"] = time.time() - t0
        manifest["cases"][label] = rec
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=1)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
