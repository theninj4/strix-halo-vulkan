"""Dump ACE-Step 1.5's audio detokenizer, for MUSIC.md A4.

The LM path turns 5 Hz audio codes into 25 Hz "LM hints" the DiT reads as
its source latents: `tokenizer.quantizer.get_output_from_indices` (the FSQ
codebook and project_out), then `detokenizer` (embed_tokens, x5 plus 5
learned tokens, 2 encoder layers over each code's 5 rows on their own,
norm, proj_out). This is upstream's `_decode_audio_codes_to_latents`, fp32
on the CPU, with hooks.

The codes are real: the fp32 DiT oracle's final latents (dump_ace_dit.py)
through the model's own audio tokenizer, the cover path's `tokenize`.

    HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_detok.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "models/ACE-Step-1.5-src")
sys.path.insert(0, "reference")

from acestep.handler import AceStepHandler  # noqa: E402
from dump_ace_plan import PROJECT  # noqa: E402

DIT = "reference/out/acedit"
OUT = "reference/out/acedetok"
CASES = ["full_metas", "defaults"]


def main():
    torch.set_grad_enabled(False)
    os.makedirs(OUT, exist_ok=True)
    manifest = {"tensors": {}, "cases": {}}

    def dump(name, t, dtype="float32"):
        arr = t.detach().contiguous().cpu().numpy().astype(dtype)
        with open(os.path.join(OUT, name + ".bin"), "wb") as fh:
            fh.write(arr.tobytes())
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": dtype}

    h = AceStepHandler()
    msg, ok = h.initialize_service(project_root=PROJECT, config_path="acestep-v15-xl-turbo", device="cpu",
                                   use_mlx_dit=False)
    assert ok, msg
    model = h.model
    det = model.detokenizer
    for label in CASES:
        p = label + "_"
        lat = np.fromfile(os.path.join(DIT, label + "_latents.bin"), dtype=np.float32).reshape(1, -1, 64)
        x = torch.from_numpy(lat)
        mask = torch.ones(1, x.shape[1], dtype=torch.bool)
        _, indices, _ = model.tokenize(x, h.silence_latent, mask)
        codes = indices.flatten()
        dump(p + "codes", codes, "int32")
        q = model.tokenizer.quantizer.get_output_from_indices(indices)
        dump(p + "quantized", q[0])
        hooks = [det.embed_tokens.register_forward_hook(lambda _m, _i, o, p=p: dump(p + "embed", o[0]))]
        for i, layer in enumerate(det.layers):
            # [codes, 5, 2048]: every code's own sequence
            hooks.append(layer.register_forward_hook(lambda _m, _i, o, i=i, p=p: dump(p + f"layer{i}", o[0])))
        hooks.append(det.norm.register_forward_hook(lambda _m, _i, o, p=p: dump(p + "norm", o)))
        hints = det(q)
        for hk in hooks:
            hk.remove()
        dump(p + "hints", hints[0])
        # And the handler's own entry point, from the code string the LM writes.
        s = "".join(f"<|audio_code_{int(c)}|>" for c in codes)
        via = h._decode_audio_codes_to_latents(s)
        manifest["cases"][label] = {
            "codes": int(codes.numel()),
            "latents": int(x.shape[1]),
            "handler_matches": bool(torch.equal(via, hints)),
        }
        print(label, manifest["cases"][label])
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {len(manifest['tensors'])} tensors to {OUT}")


if __name__ == "__main__":
    main()
