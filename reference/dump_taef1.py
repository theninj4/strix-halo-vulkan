"""Dump a reference taef1 decode for the Go/Vulkan preview decoder (IMAGE.md I2).

Two things are needed before a preview decoder can be written, and this script
produces both.

**The stagewise oracle.** `--latent-size N` decodes a seeded random latent and
writes the output of every one of the decoder's 19 layers, so a Go port that is
wrong is wrong at a named layer rather than "the picture looks odd". Same
pattern and same manifest shape as `dump_vae.py`, which is what the Go tests
already know how to read.

**Two conventions**, both of which `--from-run` settles by decoding the *real*
z-image latents `dump_zimage_run.py` already saved.

*The latent's space.* taef1's model card says it "uses the same latent API as
FLUX.1's VAE", and `IMAGE.md` is explicit that the dump decides this rather
than the card: the pipeline holds latents in the *diffusion* space and hands
`AutoencoderKL` `(z / 0.3611) + 0.1159`, so a preview decoder either wants the
same mapping or wants the raw latent. Note that the answer is not obvious by
eye -- the decoder opens with `tanh(x/3)*3`, so a latent that is 2.8x too large
is *squashed* rather than blown out and both conventions produce a recognisable
picture. What separates them is the distance from the full VAE's image for the
same latent, and the wrong one is several times further.

*What to decode at step k.* The obvious answer is the state the loop holds,
and it is wrong: this is a flow-matching schedule, so `latents` at step k is
`x_t`, an interpolation that is still 75% noise four steps into an eight-step
run. A preview has to decode the *denoised estimate* `x0 = x_t - sigma*v`,
which the loop can form for free because it has the velocity in hand. The
script decodes both and prints how each sequence approaches the final image.

    .venv/bin/python reference/dump_taef1.py --latent-size 16
    .venv/bin/python reference/dump_taef1.py --from-run reference/out/zimagerun

Everything is float32 on the CPU: a reference that is itself approximate is not
a reference. taef1 is 2.5 M parameters, so this costs seconds rather than the
minutes `dump_zimage_run.py` does.
"""

import argparse
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKL, AutoencoderTiny

# The z-image checkpoint's AutoencoderKL config. Spelled out rather than read
# so that a mismatch is visible here; --from-run checks it against the file.
FLUX_SCALE, FLUX_SHIFT = 0.3611, 0.1159


def load_tiny(path):
    tiny = AutoencoderTiny.from_pretrained(path, torch_dtype=torch.float32)
    tiny.eval()
    return tiny


def dumper(out, manifest):
    os.makedirs(out, exist_ok=True)

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(out, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape),
            "count": int(flat.numel()),
            "sum": float(flat.double().sum()),
            "absmax": float(flat.abs().max()),
        }
        print(f"  {name:28s} {str(list(t.shape)):22s} sum={flat.double().sum():+.6f} absmax={flat.abs().max():.5g}")

    return dump


def stagewise(args):
    """Decode a random latent, dumping every layer's output."""
    tiny = load_tiny(args.taef1)
    dec = tiny.decoder
    torch.manual_seed(args.seed)

    n = args.latent_size
    latent = torch.randn(1, tiny.config.latent_channels, n, n, dtype=torch.float32)

    manifest = {
        "seed": args.seed,
        "latent_size": n,
        "config": {k: tiny.config[k] for k in (
            "latent_channels", "decoder_block_out_channels", "num_decoder_blocks",
            "act_fn", "upsampling_scaling_factor", "scaling_factor", "shift_factor",
            "latent_magnitude", "latent_shift")},
        "tensors": {},
    }
    dump = dumper(args.out, manifest)

    # One hook per layer of the Sequential. The names are the indices the
    # checkpoint uses -- "layers.12" is a block, "layers.11" a bare conv -- so
    # a failing tensor names the weights it came from with no lookup table in
    # between.
    captured = {}

    def hook(name):
        def fn(_m, _i, o):
            captured[name] = o[0] if isinstance(o, tuple) else o
        return fn

    handles = [m.register_forward_hook(hook(f"layers.{i}")) for i, m in enumerate(dec.layers)]

    print("decoding...")
    with torch.no_grad():
        image = dec(latent)
    for h in handles:
        h.remove()

    print("tensors:")
    dump("latent", latent)
    # The clamp in front of the layers is part of DecoderTiny.forward rather
    # than a module, so it is recomputed here rather than hooked. It is the
    # first thing a Go port gets wrong, so it is dumped.
    dump("clamped", torch.tanh(latent / 3) * 3)
    for i in range(len(dec.layers)):
        key = f"layers.{i}"
        if key in captured:
            dump(key, captured[key])
    # And the tail: x.mul(2).sub(1), which is what puts the image in the same
    # [-1, 1] the full VAE produces and the PNG writer already expects.
    dump("image", image)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors to {args.out}")


def sigmas(steps, shift):
    """The pipeline's own schedule: linspace(1, 1/N, N) bent by the shift.

    Duplicated from zimage/pipeline/scheduler.go rather than read from the
    checkpoint, because what the x0 estimate needs is the sigma the *pipeline*
    integrated, and a reference that asks the code under test for it cannot
    catch a wrong one.
    """
    out = []
    for i in range(steps):
        raw = 1.0 if steps == 1 else 1 + (1 / steps - 1) * i / (steps - 1)
        out.append(shift * raw / (1 + (shift - 1) * raw))
    return out + [0.0]


def from_run(args):
    """Decode real z-image latents every way and say which conventions hold."""
    run = args.from_run
    with open(os.path.join(run, "manifest.json")) as fh:
        ref = json.load(fh)
    steps = ref["steps"]

    def load(name):
        shape = ref["tensors"][name]["shape"]
        a = np.fromfile(os.path.join(run, name + ".bin"), dtype=np.float32)
        return torch.from_numpy(a.reshape(shape)).unsqueeze(0)

    tiny = load_tiny(args.taef1)
    full = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32)
    full.eval()
    assert abs(full.config.scaling_factor - FLUX_SCALE) < 1e-6, full.config.scaling_factor
    assert abs(full.config.shift_factor - FLUX_SHIFT) < 1e-6, full.config.shift_factor

    manifest = {
        "from_run": run, "size": ref["size"], "steps": steps, "prompt": ref["prompt"],
        "shift": args.shift, "sigmas": sigmas(steps, args.shift),
        "latent_space": {}, "preview_state": {}, "tensors": {},
    }
    dump = dumper(args.out, manifest)

    z = load("latents_final")
    print(f"latent {list(z.shape)} from {run}\n")

    truth = full.decode(z / FLUX_SCALE + FLUX_SHIFT, return_dict=False)[0]
    rng = float(truth.abs().max())

    # --- which space the latent is in ---------------------------------
    print("the latent's space, on the final latent:")
    spaces = {
        # The model card's claim: the same latent the transformer produced.
        "raw": z,
        # The other reading: the VAE-space latent the full decoder is fed.
        "unscaled": z / FLUX_SCALE + FLUX_SHIFT,
    }
    for name, x in spaces.items():
        img = tiny.decode(x, return_dict=False)[0]
        err = (img - truth).abs().mean().item()
        manifest["latent_space"][name] = {
            "mean_abs_err": err, "rel": err / rng, "absmax": float(img.abs().max())}
        print(f"  {name:10s} mean|taef1 - vae| = {err:.4f}  ({100 * err / rng:.1f}% of the VAE's range), "
              f"absmax {img.abs().max():.4f}")
        save_png(img, os.path.join(args.out, f"final_{name}.png"))

    space = min(manifest["latent_space"], key=lambda k: manifest["latent_space"][k]["mean_abs_err"])
    manifest["latent_space_convention"] = space
    ratio = (manifest["latent_space"]["unscaled"]["mean_abs_err"]
             / manifest["latent_space"]["raw"]["mean_abs_err"])
    print(f"  -> the {space!r} latent, by {ratio:.1f}x\n")

    def to_tiny(x):
        return x if space == "raw" else x / FLUX_SCALE + FLUX_SHIFT

    # --- which state to preview ---------------------------------------
    # The loop holds x_t; the estimate of where it is heading is
    # x0 = x_t - sigma*v, and v is recoverable here from two consecutive
    # states because the Euler step is x_{k+1} = x_k + (s_{k+1} - s_k) v.
    sig = sigmas(steps, args.shift)
    traj = [load("latents_init")] + [load(f"latents_{i}") for i in range(steps)]
    xt, x0 = [], []
    for k in range(steps):
        v = (traj[k + 1] - traj[k]) / (sig[k + 1] - sig[k])
        xt.append(traj[k + 1])
        x0.append(traj[k + 1] - sig[k + 1] * v)

    print("what to decode at step k, as mean|preview - vae final| over the run:")
    print(f"  {'k':>2s}  {'sigma':>7s}  {'x_t':>8s}  {'x0':>8s}")
    for name, seq in (("x_t", xt), ("x0", x0)):
        errs = []
        for k in range(steps):
            img = tiny.decode(to_tiny(seq[k]), return_dict=False)[0]
            errs.append((img - truth).abs().mean().item())
            save_png(img, os.path.join(args.out, f"{name}_{k}.png"))
            if name == "x0":
                # Both halves: the latent the Go side feeds its decoder and
                # the image it has to produce from it. Dumping only the image
                # would make the test depend on Go reconstructing x0 the same
                # way, which is the other half of what is under test.
                dump(f"x0_{k}", seq[k])
                dump(f"preview_{k}", img)
        manifest["preview_state"][name] = {"errs": errs, "rel": [e / rng for e in errs]}
    for k in range(steps):
        print(f"  {k:2d}  {sig[k + 1]:7.4f}  "
              f"{manifest['preview_state']['x_t']['errs'][k]:8.4f}  "
              f"{manifest['preview_state']['x0']['errs'][k]:8.4f}")
    # The discriminator is the *first* step, where the two differ most: at the
    # last one they coincide, since the terminal sigma is zero and x0 is x_t.
    manifest["preview_state_convention"] = "x0"
    print("  -> x0, and the gap at k=0 is the whole argument: x_t there is the "
          "initial noise barely moved\n")

    # Dumped with the batch axis, so zimage/vae's loadRef reads them with the
    # same four-dimensional shape check as every other dump in this package.
    dump("latent_final", z)
    dump("preview_final", tiny.decode(to_tiny(z), return_dict=False)[0])
    dump("vae_final", truth)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['tensors'])} tensors and {2 * steps + 2} PNGs to {args.out}")


def save_png(img, path):
    from PIL import Image
    a = ((img[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
    Image.fromarray(a).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taef1", default="models/taef1")
    ap.add_argument("--vae", default="models/Z-Image-Turbo/vae")
    ap.add_argument("--out", default="")
    ap.add_argument("--latent-size", type=int, default=16,
                    help="latent H=W for the stagewise dump; the image comes out 8x this")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shift", type=float, default=3.0,
                    help="the scheduler shift the run used; z-image ships 3.0")
    ap.add_argument("--from-run", default="",
                    help="a dump_zimage_run.py directory; decodes its real latents instead")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    if args.from_run:
        args.out = args.out or "reference/out/taef1run"
        from_run(args)
    else:
        args.out = args.out or "reference/out/taef1"
        stagewise(args)


if __name__ == "__main__":
    main()
