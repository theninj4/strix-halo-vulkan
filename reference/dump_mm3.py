"""The frame loop of MiniMax-Music3's global and local LMs, for
research/music-minimax.md N1-N3: what the Go LM is gated against.

One case (N0c's h3: a 500-word caption in the skill's style, full lyrics) runs
through the reference's own functions (diffusers #14456 at dafe3733,
encoders.py) in fp32 on the CPU for FRAMES frames, free-running from a seeded
generator, and records, per decode step f (step 0 is the one that only
advances past <|audio_start|>):

  - head_cond / head_uncond: the global LM's logits over [HEAD_LO, HEAD_HI),
    end-of-audio (151670) through the last semantic code; 151671-151674 are
    masked upstream and are in the span only to keep it contiguous;
  - hidden: last_hidden (after the final norm), both rows;
  - codes: the frame's 8 codes (semantic, then c1..c7);
  - depth_hidden: the local LM's 7 conditional hiddens (the condition's
    input), and depth_cond / depth_uncond its 7 heads' logits;
  - feedback: the next step's input embedding (both rows are the same).

Then the global LM is loaded again in bf16 (upstream's CUDA dtype) and run
teacher-forced over the same prompt and the fp32 run's codes, with the
feedback recomputed by its own bf16 tables, as upstream bf16 would: its head
logits are kept, and the manifest records per step the KL(fp32 || bf16) of
each row and after CFG (before the top-50 cut), how many of the top 50
they share, and whether the guided argmax agrees. N1's int8 bank is priced against that, as A10 priced ACE's.

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv-mm3/bin/python -I \\
        reference/dump_mm3.py

Writes reference/out/mm3lm/<name>.bin (raw little-endian) and manifest.json.
"""

import gc
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from diffusers import MiniMaxMusic3RVQDepthDecoder
from diffusers.modular_pipelines.minimax_music3 import encoders as E
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models", "MiniMax-Music3")
OUT = os.path.join(ROOT, "reference", "out", "mm3lm")
CASE = "h3"
FRAMES = 48
SEED = 11
HEAD_LO = E._AUDIO_END_TOKEN_ID
HEAD_HI = E._AUDIO_CODE_OFFSET + E._SEMANTIC_VOCAB_SIZE

tensors = {}


def save(name, a):
    a = np.ascontiguousarray(a)
    a.tofile(os.path.join(OUT, name + ".bin"))
    tensors[name] = {"shape": list(a.shape), "dtype": str(a.dtype)}


def prompt_ids(tok):
    spec = json.load(open(os.path.join(ROOT, "out", "mm3-n0c", "set.json")))
    song = next(s for s in spec["songs"] if s["id"] == CASE)
    caption = json.load(open(os.path.join(ROOT, "out", "mm3-n0c", "captions.json")))[CASE]
    text = (
        f"{E._IM_START}{E._CAPTION_START}{E._clean_caption(caption)}{E._CAPTION_END}"
        f"{E._LYRICS_START}{E._normalize_lyrics(song['lyrics'])}{E._LYRICS_END}{E._IM_END}{E._AUDIO_START}"
    )
    ids = tok(text, return_tensors="pt")["input_ids"]
    unc = ids.clone()
    unc[:, 1:-2] = E._AUDIO_CFG_TOKEN_ID
    return torch.cat((ids, unc), dim=0)


def guided(cond, uncond):
    """Upstream's guided logits over the head span: CFG 1.5, cond's top 50, the mask."""
    g = uncond + (cond - uncond) * E._AR_CFG_SCALE
    thr = torch.topk(cond, E._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
    g = g.masked_fill(cond < thr, -float("inf"))
    g[..., 1:E._AUDIO_CODE_OFFSET - HEAD_LO] = -float("inf")
    return g


def cfg(cond, uncond):
    """CFG 1.5 over the span, before the top-50 cut (which would give each
    side its own support): the KL the gates compare."""
    g = uncond + (cond - uncond) * E._AR_CFG_SCALE
    g[..., 1:E._AUDIO_CODE_OFFSET - HEAD_LO] = -float("inf")
    return g


def kl(want, got):
    """KL(want || got) of softmax, fp64; -inf entries of want carry no mass."""
    w = torch.log_softmax(want.double(), -1)
    g = torch.log_softmax(got.double(), -1)
    p = w.exp()
    return float(torch.where(p > 0, p * (w - g), torch.zeros_like(p)).sum())


def masked(logits):
    """The full-vocabulary logits' head span, with upstream's mask applied."""
    s = logits[:, HEAD_LO:HEAD_HI].float().clone()
    s[:, 1:E._AUDIO_CODE_OFFSET - HEAD_LO] = -float("inf")
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    tok = Qwen2Tokenizer.from_pretrained(os.path.join(MODELS, "tokenizer"))
    ids = prompt_ids(tok)
    save("prompt_ids", ids.numpy().astype(np.int32))
    print(f"prompt: {ids.shape[1]} tokens a row", flush=True)

    t0 = time.time()
    lm = Qwen3ForCausalLM.from_pretrained(os.path.join(MODELS, "language_model"), torch_dtype=torch.float32).eval()
    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(os.path.join(MODELS, "rvq_depth_decoder"),
                                                         torch_dtype=torch.float32).eval()
    comp = SimpleNamespace(language_model=lm, rvq_depth_decoder=depth, num_codebooks=8, audio_vocab_size=1024)
    print(f"fp32 loaded in {time.time() - t0:.1f} s", flush=True)

    # The depth heads' logits, as each runs (both rows).
    head_out = []
    hooks = [h.register_forward_hook(lambda m, i, o: head_out.append(o.float().clone()))
             for h in depth.audio_heads]

    gen = torch.Generator("cpu").manual_seed(SEED)
    t0 = time.time()
    out = lm.model(inputs_embeds=lm.model.embed_tokens(ids), use_cache=True)
    pkv, last = out.past_key_values, out.last_hidden_state[:, -1]
    print(f"fp32 prefill {time.time() - t0:.1f} s", flush=True)

    hc, hu, hid, codes, dh, dc, du, fb = [], [], [], [], [], [], [], []
    for f in range(FRAMES + 1):
        logits = lm.lm_head(last).float()
        span = masked(logits)
        hc.append(span[0].numpy())
        hu.append(span[1].numpy())
        hid.append(last.float().numpy())
        g = guided(span[0:1], span[1:2])
        s = int(E._sample_top_k(g, gen).item()) + HEAD_LO
        if s == E._AUDIO_END_TOKEN_ID:
            sys.exit(f"end of audio at step {f}; pick another case or seed")
        sem = torch.tensor([s - E._AUDIO_CODE_OFFSET]).repeat(2)
        head_out.clear()
        fc, dhid = E._generate_depth_codes(comp, last, sem, gen)
        codes.append(fc[0].numpy().astype(np.int32))
        dh.append(dhid[0].view(7, -1).float().numpy())
        dl = torch.stack(head_out)  # [7, 2, 1024]
        dc.append(dl[:, 0].numpy())
        du.append(dl[:, 1].numpy())
        feed = E._embed_audio_frame(comp, fc)
        fb.append(feed[0, 0].float().numpy())
        out = lm.model(inputs_embeds=feed, past_key_values=pkv, use_cache=True)
        pkv, last = out.past_key_values, out.last_hidden_state[:, -1]
        if f % 8 == 0:
            print(f"step {f}: codes {fc[0].tolist()} ({time.time() - t0:.0f} s)", flush=True)
    for h in hooks:
        h.remove()
    save("head_cond", np.stack(hc))
    save("head_uncond", np.stack(hu))
    save("hidden", np.stack(hid))
    save("codes", np.stack(codes))
    save("depth_hidden", np.stack(dh))
    save("depth_cond", np.stack(dc))
    save("depth_uncond", np.stack(du))
    save("feedback", np.stack(fb))
    all_codes = torch.from_numpy(np.stack(codes)).long()
    del lm, depth, comp, pkv, out
    gc.collect()

    # bf16, teacher-forced on the fp32 run's codes.
    t0 = time.time()
    lm = Qwen3ForCausalLM.from_pretrained(os.path.join(MODELS, "language_model"), torch_dtype=torch.bfloat16).eval()
    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(os.path.join(MODELS, "rvq_depth_decoder"),
                                                         torch_dtype=torch.bfloat16).eval()
    comp = SimpleNamespace(language_model=lm, rvq_depth_decoder=depth, num_codebooks=8, audio_vocab_size=1024)
    print(f"bf16 loaded in {time.time() - t0:.1f} s", flush=True)
    out = lm.model(inputs_embeds=lm.model.embed_tokens(ids), use_cache=True)
    pkv, last = out.past_key_values, out.last_hidden_state[:, -1]
    bc, bu, steps = [], [], []
    for f in range(FRAMES + 1):
        span = masked(lm.lm_head(last).float())
        bc.append(span[0].numpy())
        bu.append(span[1].numpy())
        wc, wu = torch.from_numpy(hc[f]), torch.from_numpy(hu[f])
        gw, gg = guided(wc[None], wu[None])[0], guided(span[0:1], span[1:2])[0]
        top = lambda c: set(torch.topk(c, E._AR_CFG_TOP_K).indices.tolist())
        steps.append({"kl_cond": kl(wc, span[0]), "kl_uncond": kl(wu, span[1]),
                      "kl_cfg": kl(cfg(wc, wu), cfg(span[0], span[1])),
                      "top50_shared": len(top(wc) & top(span[0])),
                      "argmax": bool(gw.argmax() == gg.argmax())})
        feed = E._embed_audio_frame(comp, all_codes[f].repeat(2, 1))
        out = lm.model(inputs_embeds=feed, past_key_values=pkv, use_cache=True)
        pkv, last = out.past_key_values, out.last_hidden_state[:, -1]
    save("bf16_head_cond", np.stack(bc))
    save("bf16_head_uncond", np.stack(bu))
    kls = np.array([s["kl_cfg"] for s in steps])
    print(f"bf16 vs fp32, CFG KL: mean {kls.mean():.3g}, max {kls.max():.3g}; "
          f"argmax agrees on {sum(s['argmax'] for s in steps)}/{len(steps)}", flush=True)

    json.dump({"case": CASE, "seed": SEED, "frames": FRAMES, "head_lo": HEAD_LO, "head_hi": HEAD_HI,
               "code_offset": E._AUDIO_CODE_OFFSET, "cfg": E._AR_CFG_SCALE, "steps": steps, "tensors": tensors},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
