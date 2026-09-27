"""Dump ACE-Step 1.5's 5 Hz LM on the thinking path, for MUSIC.md A7 (and A8).

It runs upstream's own `generate_music` with thinking on (LLMHandler, the
PyTorch backend, fp32 on the CPU) and records what a port must reproduce:

* every prompt the LM reads (phase 1's CoT prompt; phase 2's conditional and
  unconditional prompts) as text and token ids, and the attention mask the
  CFG pair is left-padded with;
* at every sampling step, per phase: the FSM state before the step, how many
  tokens the FSM alone allows (and which, when at most 64), how many are
  left after top-k/top-p too (and which), and the sampled token. That is
  A-o5: what the default path masks;
* the raw logits (the model's own, before CFG and the FSM) of both rows at a
  few steps of each phase, for teacher-forced logit gates;
* the LM's outputs (metadata, the code string), and what reaches the DiT
  (`generate_audio`'s inputs: the rewritten caption's prompt, the LM hints),
  for A8.

Cases (label: what the request gives; the LM plans the rest):

* ``given_duration``: caption + lyrics + duration 30 s.
* ``all_metas``: A1's full_metas request (bpm, key, time signature,
  duration 30 s, language).

    HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_lm.py [label ...]
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "models/ACE-Step-1.5-src")
sys.path.insert(0, "reference")

from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor  # noqa: E402
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from dump_ace_plan import CASES, PROJECT  # noqa: E402

OUT = "reference/out/acelm"
CKPT = os.path.join(PROJECT, "checkpoints")
LOGIT_STEPS = (0, 1, 2, 3, 10, 50, 100)


class Stop(Exception):
    pass


def main():
    labels = sys.argv[1:] or ["given_duration", "all_metas"]
    os.makedirs(OUT, exist_ok=True)
    manifest = {"cases": {}, "tensors": {}}

    def dump(name, t, dtype="float32"):
        arr = t.detach().contiguous().cpu().numpy().astype(dtype) if torch.is_tensor(t) else np.asarray(t, dtype=dtype)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(arr.tobytes())
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": dtype}

    torch.set_grad_enabled(False)
    h = AceStepHandler()
    msg, ok = h.initialize_service(project_root=PROJECT, config_path="acestep-v15-xl-turbo", device="cpu",
                                   use_mlx_dit=False)
    assert ok, msg
    lm = LLMHandler()
    msg, ok = lm.initialize(checkpoint_dir=CKPT, lm_model_path="acestep-5Hz-lm-4B", backend="pt", device="cpu",
                            dtype=torch.float32)
    assert ok, msg
    lm.disable_tqdm = True
    tok = lm.llm_tokenizer

    base = dict(CASES)["full_metas"]
    requests = {
        "given_duration": {"caption": base["caption"], "lyrics": base["lyrics"], "duration": 30.0},
        "all_metas": base,
    }

    for label in labels:
        rec = {"phases": [], "prompts": []}
        state = {"phase": -1, "step": 0}

        real_fwd = lm._forward_pass

        def fwd(model, generated_ids, model_kwargs, past_key_values, use_cache):
            if past_key_values is None:
                # A new generation: a new phase, whose prompt(s) this is.
                state["phase"] += 1
                state["step"] = 0
                ph = state["phase"]
                rows = generated_ids.shape[0]
                rec["phases"].append({"rows": rows, "steps": []})
                ids = generated_ids.cpu()
                dump(f"{label}_p{ph}_prompt_ids", ids, "int32")
                am = model_kwargs.get("attention_mask")
                if am is not None:
                    dump(f"{label}_p{ph}_prompt_mask", am.cpu(), "int32")
                rec["prompts"].append([tok.decode(r) for r in ids])
            out = real_fwd(model, generated_ids, model_kwargs, past_key_values, use_cache)
            if state["step"] in LOGIT_STEPS:
                dump(f"{label}_p{state['phase']}_logits{state['step']}", out.logits[:, -1, :].float())
            return out
        lm._forward_pass = fwd

        real_call = MetadataConstrainedLogitsProcessor.__call__

        def fsm_call(self, input_ids, scores):
            state["fsm"] = self.state.name
            out = real_call(self, input_ids, scores)
            fin = torch.isfinite(out[0])
            state["fsm_allowed"] = int(fin.sum())
            if state["fsm_allowed"] <= 64:
                state["fsm_ids"] = fin.nonzero().flatten().tolist()
            return out
        MetadataConstrainedLogitsProcessor.__call__ = fsm_call

        real_sample = lm._sample_tokens

        def sample(logits, temperature):
            toks = real_sample(logits, temperature)
            fin = torch.isfinite(logits[0])
            n = int(fin.sum())
            step = {"fsm": state.pop("fsm", None), "fsm_allowed": state.pop("fsm_allowed", None),
                    "allowed": n, "token": int(toks[0])}
            if "fsm_ids" in state:
                step["fsm_ids"] = state.pop("fsm_ids")
            if n <= 64:
                step["allowed_ids"] = fin.nonzero().flatten().tolist()
            rec["phases"][state["phase"]]["steps"].append(step)
            state["step"] += 1
            return toks
        lm._sample_tokens = sample

        # Stop at the DiT's door: A8's inputs, not another DiT run.
        captured = {}
        real_generate = h.model.generate_audio

        def capture(**kwargs):
            captured.update(kwargs)
            raise Stop()
        h.model.generate_audio = capture

        params = GenerationParams(thinking=True, seed=42, shift=3.0, **requests[label])
        t0 = time.time()
        try:
            res = generate_music(h, lm, params, GenerationConfig(batch_size=1, use_random_seed=False, seeds=[42]))
            err = getattr(res, "status_message", "")
        except Stop:
            err = None
        elapsed = time.time() - t0
        h.model.generate_audio = real_generate
        lm._forward_pass, lm._sample_tokens = real_fwd, real_sample
        MetadataConstrainedLogitsProcessor.__call__ = real_call

        for ph, p in enumerate(rec["phases"]):
            toks = [s["token"] for s in p["steps"]]
            dump(f"{label}_p{ph}_tokens", toks, "int32")
            p["text"] = tok.decode(toks)
            p["n_steps"] = len(toks)
        if captured:
            for k in ("src_latents", "precomputed_lm_hints_25Hz", "text_hidden_states", "lyric_hidden_states"):
                v = captured.get(k)
                if torch.is_tensor(v):
                    dump(f"{label}_dit_{k}", v[0].float())
            rec["dit_is_covers"] = [bool(x) for x in captured["is_covers"].flatten()] if torch.is_tensor(
                captured.get("is_covers")) else None
            rec["dit_chunk_mask_values"] = sorted({float(x) for x in captured["chunk_masks"].flatten().unique()}) \
                if torch.is_tensor(captured.get("chunk_masks")) else None
        rec["seconds"] = elapsed
        rec["error"] = err
        rec["request"] = requests[label]
        manifest["cases"][label] = rec
        summary = {ph: {"rows": p["rows"], "steps": p["n_steps"],
                        "fsm_states": sorted({s["fsm"] for s in p["steps"] if s["fsm"]})}
                   for ph, p in enumerate(rec["phases"])}
        print(f"{label}: {elapsed:.0f} s, {summary}")
        for ph, p in enumerate(rec["phases"]):
            print(f"  phase {ph} text: {p['text'][:400]!r}")
        with open(os.path.join(OUT, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
