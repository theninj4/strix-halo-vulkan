"""Render A14e set songs through MiniMax-Music3's reference pipeline, for
research/music-minimax.md N0: the go/no-go against ACE-Step's renders of the
same requests (out/a14e, scored and rated).

The pipeline is the diffusers modular one from huggingface/diffusers#14456 at
dafe3733, on the CPU. Each song is the set's caption and lyrics as given (no
structured-caption rewrite, and the set has no metas to write in, N-d3), its
duration, and the set's seed. The model's native output (the vocoder's rate,
N-o1) is written as a float wav, with a json of the request and the wall time:

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv-mm3/bin/python -I \\
        reference/render_mm3.py [--bf16] [--seconds S] [--out DIR] id ...

--bf16 runs every component in bfloat16, the reference's own CUDA dtype; the
default is fp32. --seconds overrides the set's duration (a timing probe).
"""

import argparse
import json
import os
import sys
import time

import soundfile as sf
import torch
from diffusers import ModularPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--captions", default="", help="a json of id -> caption that replaces the set's (tagged -sc)")
    ap.add_argument("--seed", type=int, default=0, help="0 takes the set's first seed")
    ap.add_argument("--set", default=os.path.join(ROOT, "cmd", "acecal", "set.json"), help="the set of songs (N0b: out/mm3-n0b/set.json)")
    ap.add_argument("--models", default=os.path.join(ROOT, "models", "MiniMax-Music3"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "mm3-n0"))
    args = ap.parse_args()

    with open(args.set) as f:
        spec = json.load(f)
    songs = {s["id"]: s for s in spec["songs"]}
    for i in args.ids:
        if i not in songs:
            sys.exit(f"render_mm3: no song {i!r} in {args.set}")
    if args.captions:
        with open(args.captions) as f:
            for i, c in json.load(f).items():
                if i in songs:
                    songs[i] = {**songs[i], "caption": c}
    seed = args.seed or spec["seeds"][0]
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    pipe = ModularPipeline.from_pretrained(args.models)
    # modular_model_index.json names the hub repo for every component; read
    # them from the local copy instead (offline, and the files we checked).
    for spec in pipe._component_specs.values():
        if spec.pretrained_model_name_or_path:
            spec.pretrained_model_name_or_path = args.models
    pipe.load_components(dtype=dtype)
    missing = [n for n in pipe._component_specs if getattr(pipe, n, None) is None and n != "guider"]
    if missing:
        sys.exit(f"render_mm3: components not loaded: {missing}")
    pipe.to("cpu")
    load = time.time() - t0
    print(f"loaded in {load:.1f} s ({dtype}, {torch.get_num_threads()} threads)", flush=True)

    for i in args.ids:
        s = songs[i]
        seconds = args.seconds or float(s["duration"])
        tag = (f"{i}-{seed}" + ("-bf16" if args.bf16 else "") + ("-sc" if args.captions else "")
               + (f"-{seconds:g}s" if args.seconds else ""))
        t0 = time.time()
        audio = pipe(
            prompt=s["caption"],
            lyrics=s["lyrics"],
            audio_duration=seconds,
            generator=torch.Generator("cpu").manual_seed(seed),
            output="audios",
        )[0]
        wall = time.time() - t0
        wav = audio.T.float().cpu().numpy() if torch.is_tensor(audio) else audio.T
        sf.write(os.path.join(args.out, tag + ".wav"), wav, pipe.sampling_rate, subtype="FLOAT")
        rec = {
            "id": i, "seed": seed, "dtype": str(dtype), "caption": s["caption"], "lyrics": s["lyrics"],
            "requested_seconds": seconds, "seconds": len(wav) / pipe.sampling_rate,
            "sampling_rate": pipe.sampling_rate, "wall_seconds": wall, "load_seconds": load,
        }
        with open(os.path.join(args.out, tag + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"{tag}: {rec['seconds']:.1f} s of audio at {pipe.sampling_rate} Hz in {wall:.0f} s", flush=True)


if __name__ == "__main__":
    main()
