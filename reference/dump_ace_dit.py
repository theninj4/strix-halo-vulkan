"""Dump ACE-Step 1.5's condition encoder and DiT, for MUSIC.md A2 and A3.

Starting from what upstream's pipeline hands `generate_audio` (captured the
way dump_ace_plan.py does, through `inference.generate_music` with thinking
off), this runs upstream's own `generate_audio` -- 8 steps, shift 3, DCW on,
fp32 on the CPU -- with hooks that record:

* A2, the conditioning: the text projector's output, every lyric-encoder
  layer and its final norm, the timbre encoder's layers and CLS row, the
  packed cross-attention sequence, and the context latents.
* A3, the DiT: at the first forward, the timestep tables, the patch
  embedding, the condition embedder's output, the output of layers 0, 1, 2,
  15 and 31, and the velocity; at every forward, the input latents and the
  velocity; and the latents it ends on.
* A-o1, whether fp16 holds: the absmax of every GEMM operand in every layer
  at every forward (residual stream, the normed inputs of the attention and
  FFN projections, the attention contexts, the SwiGLU product that feeds the
  down projection), and of every layer's output.

Cases are A1's requests by label (default: full_metas, 30 s, 375 tokens;
defaults, 120 s, 1,500 tokens).

    HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_dit.py [--bf16] [label ...]

--bf16 runs the same request the way upstream serves it on CUDA (the
handler's dtype there is bfloat16; on ROCm and the CPU it is fp32), into
reference/out/acedit_bf16: the bar a device path's own drift from the fp32
oracle is priced against (decision 2).
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "models/ACE-Step-1.5-src")
sys.path.insert(0, "reference")

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from dump_ace_plan import CASES, PROJECT  # noqa: E402

BF16 = "--bf16" in sys.argv
OUT = "reference/out/acedit_bf16" if BF16 else "reference/out/acedit"
KEEP_LAYERS = (0, 1, 2, 15, 31)


class Stop(Exception):
    pass


def main():
    labels = [a for a in sys.argv[1:] if not a.startswith("--")] or ["full_metas", "defaults"]
    os.makedirs(OUT, exist_ok=True)
    manifest = {"cases": {}, "tensors": {}}
    manifest_path = os.path.join(OUT, "manifest.json")

    def dump(name, t):
        arr = t.detach().contiguous().cpu().float().numpy()
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(arr.tobytes())
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": "float32"}

    torch.set_grad_enabled(False)
    h = AceStepHandler()
    msg, ok = h.initialize_service(project_root=PROJECT, config_path="acestep-v15-xl-turbo", device="cpu",
                                   use_mlx_dit=False)
    assert ok, msg
    model = h.model
    if BF16:
        model.to(torch.bfloat16)
        h.dtype = torch.bfloat16
        if getattr(h, "text_encoder", None) is not None:
            h.text_encoder.to(torch.bfloat16)
    real_generate = model.generate_audio
    cases = dict(CASES)

    for label in labels:
        # --- capture the inputs, as dump_ace_plan.py does --------------------
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            raise Stop()

        model.generate_audio = capture
        params = GenerationParams(thinking=False, shift=3.0, seed=42, use_cot_metas=False, use_cot_caption=False,
                                  use_cot_language=False, **cases[label])
        generate_music(h, None, params, GenerationConfig(batch_size=1, use_random_seed=False, seeds=[42]))
        model.generate_audio = real_generate
        assert captured, label
        p = label + "_"

        # --- hooks -----------------------------------------------------------
        state = {"forward": -1}
        stats = []  # per forward: per layer dict
        hooks = []

        def amax(t):
            return float(t.detach().abs().max())

        enc = model.encoder

        def keep(name):
            def hook(_m, _inp, out):
                dump(p + name, (out[0] if isinstance(out, tuple) else out)[0])
            return hook

        hooks.append(enc.text_projector.register_forward_hook(keep("text_proj")))
        for i, layer in enumerate(enc.lyric_encoder.layers):
            hooks.append(layer.register_forward_hook(keep(f"lyric_layer{i}")))
        hooks.append(enc.lyric_encoder.norm.register_forward_hook(keep("lyric_out")))
        for i, layer in enumerate(enc.timbre_encoder.layers):
            hooks.append(layer.register_forward_hook(keep(f"timbre_layer{i}")))
        hooks.append(enc.timbre_encoder.norm.register_forward_hook(keep("timbre_norm")))

        def enc_out(_m, _inp, out):
            dump(p + "encoder_states", out[0][0])
            manifest["cases"].setdefault(label, {})["encoder_mask_all_ones"] = bool(out[1].bool().all())
            manifest["cases"][label]["encoder_len"] = int(out[0].shape[1])
        hooks.append(enc.register_forward_hook(enc_out))

        dec = model.decoder

        def dec_pre(_m, args, kwargs):
            state["forward"] += 1
            f = state["forward"]
            stats.append([{} for _ in dec.layers])
            dump(p + f"x_in{f}", kwargs["hidden_states"][0])
            if f == 0:
                dump(p + "context_latents", kwargs["context_latents"][0])
                manifest["cases"][label]["timesteps"] = []
            manifest["cases"][label]["timesteps"].append(float(kwargs["timestep"][0]))

        def dec_out(_m, _args, _kwargs, out):
            dump(p + f"v{state['forward']}", out[0][0])
        hooks.append(dec.register_forward_pre_hook(dec_pre, with_kwargs=True))
        hooks.append(dec.register_forward_hook(dec_out, with_kwargs=True))

        def first(name):
            def hook(_m, _inp, out):
                if state["forward"] == 0:
                    dump(p + name, (out[0] if isinstance(out, tuple) else out)[0])
            return hook
        hooks.append(dec.proj_in.register_forward_hook(first("proj_in")))
        hooks.append(dec.condition_embedder.register_forward_hook(first("cond_emb")))
        def temb(suffix):
            # A forward hook's return value replaces the output: return None.
            def hook(_m, _i, out):
                if state["forward"] == 0:
                    dump(p + "temb_" + suffix, out[0][0])
                    dump(p + "tproj_" + suffix, out[1][0])
            return hook
        hooks.append(dec.time_embed.register_forward_hook(temb("t")))
        hooks.append(dec.time_embed_r.register_forward_hook(temb("r")))

        for li, layer in enumerate(dec.layers):
            def rec(key, li=li):
                def hook(_m, inp, _out=None):
                    stats[state["forward"]][li][key] = amax(inp[0])
                return hook

            def layer_in(_m, args, kwargs, li=li):
                stats[state["forward"]][li]["residual_in"] = amax(args[0] if args else kwargs["hidden_states"])

            def layer_out(_m, _args, _kwargs, out, li=li):
                stats[state["forward"]][li]["residual_out"] = amax(out[0])
                if state["forward"] == 0 and li in KEEP_LAYERS:
                    dump(p + f"layer{li}", out[0][0])
            hooks.append(layer.register_forward_pre_hook(layer_in, with_kwargs=True))
            hooks.append(layer.register_forward_hook(layer_out, with_kwargs=True))
            hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(rec("sa_in")))
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(rec("sa_ctx")))
            hooks.append(layer.self_attn.o_proj.register_forward_hook(
                lambda _m, _i, out, li=li: stats[state["forward"]][li].__setitem__("sa_out", amax(out))))
            hooks.append(layer.cross_attn.q_proj.register_forward_pre_hook(rec("ca_in")))
            hooks.append(layer.cross_attn.o_proj.register_forward_pre_hook(rec("ca_ctx")))
            hooks.append(layer.cross_attn.o_proj.register_forward_hook(
                lambda _m, _i, out, li=li: stats[state["forward"]][li].__setitem__("ca_out", amax(out))))
            hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(rec("ff_in")))
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(rec("ff_down_in")))
            hooks.append(layer.mlp.down_proj.register_forward_hook(
                lambda _m, _i, out, li=li: stats[state["forward"]][li].__setitem__("ff_out", amax(out))))

        # --- the run -----------------------------------------------------------
        if BF16:
            # The fp32 run's noise, so the two runs start from the same x:
            # a bf16 prepare_noise draws a different tensor.
            ref = np.fromfile(f"reference/out/acedit/{label}_noise.bin", dtype=np.float32)
            ref = torch.from_numpy(ref.reshape(1, -1, 64))

            def fixed_noise(context_latents, seed, ref=ref):
                return ref.to(context_latents.dtype)
            model.prepare_noise = fixed_noise
        t0 = time.time()
        out = model.generate_audio(**captured)["target_latents"][0]
        elapsed = time.time() - t0
        for hk in hooks:
            hk.remove()
        dump(p + "latents", out)
        dump(p + "noise", model.prepare_noise(captured["src_latents"].new_zeros(1, out.shape[0], 128), captured["seed"])[0])

        c = manifest["cases"][label]
        c.update({"latent_length": int(out.shape[0]), "tokens": int((out.shape[0] + 1) // 2), "seconds": elapsed,
                  "stats": stats})
        worst = {}
        for f in stats:
            for li, d in enumerate(f):
                for k, v in d.items():
                    if v > worst.get(k, (0, 0))[0]:
                        worst[k] = (v, li)
        c["worst"] = {k: {"absmax": v, "layer": li} for k, (v, li) in worst.items()}
        print(f"{label}: T={out.shape[0]} enc {c['encoder_len']} in {elapsed:.1f} s; timesteps {c['timesteps']}")
        for k, (v, li) in sorted(worst.items()):
            print(f"  {k:13s} absmax {v:10.4g} (layer {li})")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=1)

    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
