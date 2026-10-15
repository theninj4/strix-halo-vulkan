# IMAGE — generation and editing (FLUX 3 Image Klein)

> **Live tracking and session handoff doc, opened 2026-10-15.** This
> replaces Qwen-Image-2.1 ([`qimage-vertical.md`](qimage-vertical.md), Q0–Q14,
> now frozen) as `GOALS.md` item 4, the same day
> `black-forest-labs/flux-3-image-klein` was read. Stage letters are **X**
> ("fluX"; F collides with `F16`/`F32` and kokoro's `F0`, and I, Q, K, B are
> taken). It is updated here in place: tick a stage when its gate passes, put
> its measured numbers under it, and keep **§ Handoff** at the bottom current.
> Long stage write-ups go in `x*.md` files beside it.
>
> **Qwen-Image-2.1 keeps serving until X6 and X7 replace both endpoints.**
> Unlike the z-image → Qwen-Image pivot, breaking the served vertical
> mid-port was *not* agreed, so `qimage/` stays whole until X8 deletes it.

## What we are building

`GOALS.md` item 4: prompt (+ up to 10 reference images) → PNG, end to end in
Go on Vulkan, served at `POST /v1/images/generations` and
`POST /v1/images/edits` with in-progress previews — the same two endpoints
`qimage/` answers today, from a model that is **smaller (5.6 B DiT vs 7.1 B),
distilled to 16 steps with no CFG (vs 40), and the checkpoint behind BFL's own
API**.

### Sources, and who is ground truth

Read 2026-10-15:

- **Checkpoint**: [`black-forest-labs/flux-3-image-klein`](https://huggingface.co/black-forest-labs/flux-3-image-klein)
  at `8454017f` (created 2026-10-13, gated "auto" — the `theninj4` account
  has access). Downloaded in its **diffusers layout** to
  `models/flux-3-image-klein/` (`transformer/`, `vae/`, `text_encoder/` and
  the root configs; ~20 GB). The root `model.safetensors`/`ae.safetensors`
  (BFL's own format, the same weights again) and `variants/{fp8r,nvfp4}` were
  skipped — fetch them only if X0 needs BFL's CLI to run.
- **BFL reference**: [`black-forest-labs/flux-3-image-klein`](https://github.com/black-forest-labs/flux-3-image-klein)
  at `45e57c9f` (2026-10-15) — 1,379 lines of Python, `src/flux3/{dit,
  sampling,pipeline,text_encoder,autoencoder}.py` are the spec. It is the
  **readable** source and the one that settles semantics (reference
  preparation, cropping, seeds).
- **diffusers**: `main` at `7ecb1e1b` (2026-10-15) carries
  `models/transformers/transformer_flux3.py` (656 lines) and the modular
  `modular_pipelines/flux3/` (Flux3DistilledAutoBlocks). It is the **oracle**
  every dump comes from, as diffusers was for Qwen-Image — it loads the
  diffusers-layout weights we downloaded and runs in fp32 on the CPU. Where
  it and BFL's code disagree, BFL's wins and the difference is written down.
- **License**: FLUX Kommunity License v1.0 — non-commercial use, plus
  commercial *outputs* for "Qualifying Users" under US$5M revenue; Section 4
  lists prohibited uses. Stricter than Qwen-Image's Qwen Research License.
  The text encoder is Apache-2.0. Fine for this project; worth remembering if
  the server ever grows other users.

## The model, from the configs and the reference code

Three components:

| | | |
|---|---|---|
| DiT | `Flux3Transformer2DModel` — hidden 3072 (24 heads x 128), **5 mode blocks per mode + 28 shared single-stream blocks**, every block the same parallel attention + SwiGLU (9216) with **no biases**, q/k RMSNorm, 4-axis RoPE; 5,548,681,984 params | 11.1 GB bf16 |
| Text encoder | **Qwen3-VL-4B-Instruct**, text model only: 36 layers (only **32 run** — the taps stop there), hidden 2560, 32 q / 8 kv heads of 128, theta 5e6, vocab 151936. The vision tower ships but **is never called** | ~7.2 GB fp16 for the 32 layers |
| VAE | `AutoencoderKLFlux2` — **byte-identical to FLUX.2-dev's** (sha256 `d64f3a68…` on both repos): the FLUX.1 autoencoder's structure (ch 128, mult 1/2/4/4, 2 res blocks, mid attention, GroupNorm 32) with 32 latent channels, quant/post-quant 1x1 convs, and a non-affine BatchNorm over the 2x2-patchified 128-channel latent | 0.34 GB fp32 |

Geometry: VAE /8 then 2x2 patchify → **one token = one 16x16 pixel tile, 128
channels**. Sides must be multiples of 16. 1024² = 4096 image tokens, the
same sequence length as Qwen-Image at 1024². The reference allows sides 64 to
4096 and a single reference canvas up to 4 MiP.

### How a step runs (the five mechanisms)

**1. Three modes, separate weights early, shared late.** Tokens come in three
modes — `image` (the target), `image_cond` (all reference images), `text`.
Each mode has its **own** 5 early blocks and attends **only within its own
mode** (the references attend to each other; the target sees no text yet).
Then the three are concatenated per sample (image, image_cond, text order)
and run through 28 shared single-stream blocks with full joint attention.
No masks anywhere inside a sample.

**2. One block shape.** `x + gate * (attn_out(attn) + mlp_out(silu(g) * u))`
with `x_mod = (1 + scale) * LayerNorm(x) + shift` (non-affine, eps 1e-6),
q/k/v separate 3072→3072, `mlp_in` 3072→18432 as **gate|up concatenated**
(exactly the `C_SWIGLU` epilogue's operand order from kernels G3, §2.14),
`mlp_out` 9216→3072. One gate for the sum, not two. The q/k RMSNorm and the
RoPE are done in **fp32** in the reference (`rmsnorm_rope`), even under bf16
autocast.

**3. Modulation is per mode, not per block.** Two `ModeModulation`s (early,
single), each one Linear(3072→9216) per mode on `silu(vec)` → (shift, scale,
gate), shared by every block of that stage. `vec[mode] =
time_in(emb(t_mode)) + ref_time_in(emb(t_next))`, where `t_mode` is the
current time for `image` and **0** for text and references, and `t_next` (the
flow map's jump target) is added to **every** mode. Timestep embedding is
sinusoidal over `1000 t`, cos first. Final layer: image tokens only,
AdaLN(shift, scale) from `vec[image]`, 3072→128.

**Consequence: there is no prefix cache.** Qwen-Image's step-independent
prefix (decision 2 there) does not exist here: `t_next` changes every step and
enters the text and reference tokens' modulation, so every step recomputes
every token. Only the text encoder's output is computed once.

**4. RoPE: 4 axes x 32 dims (t, h, w, l), theta 10000, adjacent pairs.**
Image ids are `(0, row/2, col/2, 0)` — **fractional**, `image_rope_scale
0.5`; reference *k* gets `l = -(1+k)` with its own (scaled) row/col grid;
text gets `(0, 0, 0, position)`. The angle table is computed in float64 in the
reference.

**5. Flow-map sampling, 16 steps, no guidance.** Schedule `t_i = α s / (1 +
(α−1) s)` over `s = linspace(1, 0, steps+1)`, **α = 6.93**, no resolution
shift. Each step `x += (t_{i+1} − t_i) · DiT(x, t_i, t_{i+1})`. Noise is drawn
in bf16 values and the state kept in fp32 across steps.

### Text encoding, exactly

- `apply_chat_template([{"role": "user", "content": prompt}],
  add_generation_prompt=True)` — no system message, **no token drop**
  (Qwen-Image dropped the system prefix; this keeps every token).
- **Right-padded to a multiple of 80** (max 4096) with an attention mask;
  **the pad rows are DiT tokens** — their hidden states go into the sequence,
  so a pad query must be computed exactly as the reference computes it
  (attending to the real keys only). Pad positions are 1, real ones
  `cumsum − 1`.
- Output: `hidden_states[k]` for k in **(4, 8, …, 32)**, stacked per token
  as 8 x 2560 = **20480** wide, then `txt_in` 20480→3072. Layers 33–36 never
  need to run.
- No vision tower, even for edits: references reach the DiT only as VAE
  latents. The text encoder never sees the pictures.

### Edits

- References: one → fitted under a 4 MiP canvas, **reflect-padded** (centred)
  to a multiple of 16, and the output **cropped back** to the fitted box;
  several → each fitted under 1 MiP and centre-cropped to the 16 grid.
  Default output size is the first prepared reference's.
- VAE encode uses the posterior **mean**, then patchify + BatchNorm.
- Cost grows with the references' tokens in both the early `image_cond`
  blocks and every single-stream block; there is no cache to amortise it.

### Prompt upsampling, safety classifier, boxes

The card is blunt that **short prompts give generic output**; BFL's CLI runs
an 18.8 GB prompt upsampler (`flux-3-image-klein-pup`, a Qwen3.5 VLM with two
system prompts) before every generation, plus an always-on NSFW classifier.
Neither is part of the image model. Bounding boxes are prompt text (a JSON
list appended to the caption) and need nothing from us. See decisions 5–6.

## What carries over

| piece | verdict |
|---|---|
| `zimage/qwen` + `qimage/textenc` (Qwen3-VL text model, CPU + GPU, int8 bank) | **reused** — Qwen3-VL-4B is a config change of the 8B (`LoadConfig` reads `text_config`). New: the user-only template, pad-to-80 with a key mask, an 8-layer tap. Q13's lesson travels: find the massive-activation layers before int8 |
| `qimage/vision` | **not needed** by this model; stays for `h3`, `ocr`, `llm` vision |
| FLUX.1 VAE port (`zimage/vae` at `77f6627^`: decoder, encoder, GPU paths, fp16 WMMA conv, taef1) | **resurrected as the starting point** — same structure; add 32-ch latent, quant convs, BatchNorm, 2x2 patchify. z-image *shipped* fp16 WMMA convs on this VAE family; Qwen-Image's refusal (Q9b) was its own tail norm, so the fp16 ladder (`TestConvFP16Ladder`) is re-run here, not assumed either way |
| taef1 port (same commit) → **taef2** | previews; `madebyollin/taef2` decodes FLUX.2 latents, which these are |
| shaders: `dit_gemm` (LDS w32 + w64 short), `h3_attn_t_kvlds`, `C_SWIGLU`/`C_PACK` epilogues, q8 dequant | **reused** — 3072/9216/18432 are new shapes for the screens, nothing new in kind. RMSNorm+RoPE needs a 4-axis fractional-position variant |
| `qimage/pipeline` sampler, geometry, cancellation, previews hook | **rewritten** in a new `flux3/` tree (new sampler, new packing); the cancellation and progress plumbing pattern copies over |
| `backend/image.go`, `api` image endpoints, `-image`/`-edits N` | **re-wired at X6/X7**, shape unchanged. `background: "transparent"` loses its mechanism (decision 4) |
| `qimage/pipeline.Resize`, `qimage/vae.Tensor` | still imported by `h3/pipeline/keyframe.go` and `backend/resize.go` — **hoisted before X8 deletes their packages** |

## What it should cost on this machine (estimate, to be replaced by X5)

Per step at 1024² with a 320-token caption: 33 block-passes x 4416 tokens x
245 MFLOP + attention ≈ **43 TFLOP**. Qwen-Image's cached 1024² step does
~67 TFLOP in ~1.95 s on the same kernels, so ≈ **1.3 s a step, ~21 s of DiT
for 16 steps**, plus a FLUX.1-class decode (z-image's was 0.9 s in fp16) — a
**~25 s image against Qwen-Image's 1m28.8s**. A one-reference 1024² edit is
~8.5k tokens through the single blocks, ≈ 96 TFLOP a step, **~45 s against
1m54s**.

Resident: DiT 11.1 GB fp16 / ~5.6 GB int8, encoder ~7.2 / ~3.6 GB, VAE 0.34
GB → **~19 GB fp16, ~10 GB int8** against Qwen-Image's 20.4 GB int8. It
shares the swap slot with music and video either way.

## Decisions (so future sessions don't relitigate)

1. **New `flux3/` tree** (`flux3/dit`, `flux3/vae`, `flux3/pipeline`,
   `cmd/flux3`); `qimage/textenc` and `qimage/vision` are imported, not
   copied. A rename of those two out of `qimage/` is mechanical and waits for
   X8.
2. **diffusers fp32 CPU is the oracle; BFL's code is the spec.** The dump env
   is a separate `.venv-flux3` (diffusers `7ecb1e1b`, transformers ≥ 5.17,
   torch CPU) so the shared `.venv` keeps diffusers `80c7ed26`, which the
   Qwen-Image, H3 and ACE dumps were taken with.
3. **Qwen-Image serves until FLUX 3 replaces each endpoint**; X8 deletes it
   afterwards in one stage with every surviving gate re-run.
4. **No RGBA.** This VAE is RGB. `background: "transparent"` becomes a 400
   saying so (it was native on Qwen-Image-2.1). Opaque output only.
5. **Prompt upsampling is the client's** at first — the precedent is video's
   M12. The upsampler's two system prompts are public
   (`src/flux3/pup_prompts.json`), so routing through the served LLM is a
   later, separate decision (X-o1), not a port of BFL's 18.8 GB model.
6. **No safety classifier** — the Qwen-Image vertical had none and the
   diffusers pipeline has none.
7. **Seeded noise is ours** (Go's `rand` over the packed buffer, as qimage):
   reproducing torch's CPU generator is not a goal. Gates feed the dump's
   noise.
8. **16 steps is the served default**, the checkpoint's own; `steps` stays a
   request field. A sweep comes only if a prompt kind breaks.

## The stages

Same doctrine as every vertical: **CPU reference first, validated against a
dump, then the GPU port debugged against the CPU one.** One stage, one gate,
two-run numbers; every tolerance measured with its instrument named.

| # | Stage | Gate | State |
|---|---|---|---|
| X0 | Weights, `.venv-flux3`, reference code copied to `reference/flux3/`, oracles (`reference/dump_flux3_*.py`): textenc, one block per kind, the stack per step, sampler schedule, VAE enc/dec, a full 1024² t2i run and a one-reference edit | two runs byte-identical; diffusers vs BFL's code on one step and the reference preparation | **in progress** |
| X1 | Text encoder in Go (`flux3/textenc` over `zimage/qwen` at the 4B config; template, pad-80 window as one pad row, 8-layer tap) | CPU vs dump at its own noise floor; GPU fp16 vs bf16's error | **CPU done 2026-10-15** — template, window, tokenizer exact; the 20480-wide embedding **rel 4.2e-5** at depth 32 (gate 2e-4); **GPU done 2026-10-15 (X5)**: every tap ≤ 1.1e-3 rms, 54–62 ms a prompt, the pad row via a `PAD_LAST` attention build |
| X2 | One DiT block on CPU (`flux3/dit`): mode block, single block, modulation, 4-axis RoPE with fractional ids, both time embeddings | block ≤ 4e-5, negative controls | **done 2026-10-15** — every block kind ≤ 2.8e-5 on exact inputs; the 33-block stack **rel 2.5e-5 (edit) / 1.7e-5 (t2i)**; RoPE bit-identical; three controls 1.9e4–1.6e7x the bound |
| X3 | Stack + flow-map sampler on CPU (`flux3/pipeline`) | per-step latents vs the dump | **done 2026-10-15** — schedule exact, Euler steps bit-identical, BN/patchify bridge ≤ 2 ulps both ways, teacher-forced 1024² velocity ≤ 7.7e-5; free run moved to X5 |
| X4 | VAE on CPU (`flux3/vae`, from `77f6627^`'s FLUX.1 port) | stagewise vs the dump; the run's latent → its PNG | **done 2026-10-15** — every stage ≤ 2.8e-5, free encode 1.4e-5, free 1024² decode 3.0e-5, **the run's PNG to within 1 level on 74 of 3.1M values**; end-to-end *prompt* → image moves to X5 with the free-running DiT |
| X5 | GPU: DiT (`dit_gemm`, `h3_attn_t_kvlds`, `C_SWIGLU`), text encoder, VAE decoder; fp16 ladder for the VAE | teacher-forced per step; s/step at 1024² | **done 2026-10-15** — prompt → PNG at 1024² **19.40 / 19.49 s** (two runs: encode 52 ms, 16 steps 18.4 s at 1.15 s, decode 0.81 s) against Qwen-Image's 1m28.8s; teacher-forced rms ≤ 1.0e-3 (t2i) / 1.7e-3 (edit); prompt text → picture on the device **63.1 dB** against the fp32 run's PNG |
| X6 | Serve t2i (`backend/image.go` → `flux3/pipeline`), taef2 previews, the area ceiling re-measured, `transparent` refused | served 1024² wall clock, two runs; oracle picture | **done 2026-10-15, deployed** — served 1024² **19.54 / 19.46 s** on ai.service (33.4 s cold, 13.7 s of it staging); seed for seed bit-identical to `cmd/flux3`; taef2 previews 99 ms, 81 dB from upstream; ceiling **1,517,824 px (1232²)**; edits routed to Qwen-Image until X7 |
| X7 | Edits: VAE encoder on the device, reference preparation (reflect pad + crop; multi-ref ≤1 MiP), `image_cond` mode, up to `-edits N` ≤ 10 | served edit vs the oracle edit | **done 2026-10-15** — preparation **bit-exact** to BFL's on 18 cases; device encoder fp16 2.6e-3 (fp32 arm 1.7e-6); the oracle edit **58.8 dB** from reference pixels through 16 free steps; served one-reference ~1 MP edit **40.7 / 40.6 s** (Qwen-Image 124 s), two references 69 s; `ImageRouter` gone, Qwen-Image no longer staged |
| X8 | Delete Qwen-Image-2.1: `qimage/{dit,pipeline,vae}`, `cmd/qimage`, its shaders and tests; hoist `Resize`/`Tensor`; README/PERFORMANCE/api docs | every surviving gate re-run, no digit moved | |
| X9 | Percents: int8 banks (Q13's method, priced against bf16), step profile, fusions, kernel screens at 3072/9216/18432 | bit-identical or priced against bf16 | |

### X0 — weights, environment, oracles

- [x] weights: diffusers layout to `models/flux-3-image-klein/` (2026-10-15)
- [x] `.venv-flux3`: torch 2.14.0+cpu, transformers 5.17.0, diffusers
      `7ecb1e1b` (0.41.0.dev0) — the shared `.venv`'s versions otherwise
- [x] `reference/flux3/`: BFL's `src/flux3/*.py` at `45e57c9f` (Apache-2.0,
      with its LICENSE) and diffusers' `transformer_flux3.py` +
      `modular_pipelines/flux3/` at `7ecb1e1b`, verbatim; `README.md` there
- [x] **textenc oracle** (`reference/dump_flux3_textenc.py` →
      `reference/out/flux3textenc`, 9 s): `short` and `cjk` (17 real tokens →
      window 80), `long` (118 → 160, two windows); every prompt's 20480-wide
      embeddings, ids, mask and position ids, plus hidden states 1, 2 and 32
      of `short`. The walk's layer 32 equals the oracle's last 2560 columns
      **exactly**; two runs byte-identical
- [x] **block oracle** (`reference/dump_flux3_dit_block.py` →
      `reference/out/flux3block`, 26 s): one 512² edit forward (80 text rows,
      1024 target, 1024 reference from `assets/1_figurine.png`, t 0.75 →
      0.5) with the exact inputs and outputs of the first text/image/cond
      mode blocks, single blocks 0 and 27, `norm_out`, the rotary tables, the
      VAE-encoded reference, and both velocities (edit and t2i); two runs
      byte-identical
- [x] **t2i run oracle** (`reference/dump_flux3_run.py t2i` →
      `reference/out/flux3run_t2i`, **8m00s** fp32 on the CPU, ~40 GB RSS):
      diffusers' own modular pipeline at 1024²/16, seed 0, the bison prompt,
      hooked for every step's latents, both times and the velocity, the
      VAE's input and the PNG (a correct picture). The schedule matches BFL's
      `α s / (1 + (α−1) s)` to **1.2e-7** — the same 17 points
      (1.0, 0.990, …, 0.497, 0.316, 0)
- [x] **edit run oracle** (`dump_flux3_run.py edit` →
      `reference/out/flux3run_edit`, ~15 min): `assets/1_figurine.png`
      (896x1120) as the one reference → 3,920 `image_cond` tokens, 80 text
      rows, a convincing 912x1152 edit. Schedule gap to BFL 1.2e-7 again
- [x] **VAE oracle** (`reference/dump_flux3_vae.py` →
      `reference/out/flux3vae`, 10 s): the t2i run's own latent decoded
      stagewise (post-quant conv, conv_in, mid, four up blocks, norm, conv_out)
      — it reproduces the run's PNG within **0.5 of a level**, i.e. the PNG's
      own quantization — and a non-square 768x512 encode stagewise down to
      the 1536 x 128 BatchNormed tokens. **Decoder activations peak at 457**
      (up block 3), encoder at 217: three orders below the 2.6e5 that refused
      fp16 on Qwen-Image's VAE (its decision 7), so the fp16 ladder at X5 is
      worth running rather than assumed lost
- [ ] cross-check: diffusers vs BFL's `DiT.forward` on one step (needs the
      root BFL-format `model.safetensors`, 11 GB — not fetched yet)
- [x] reference preparation, diffusers vs BFL (X-o2): **they differ** — see
      finding 6

**What X0 found (2026-10-15)** — each one changes a port:

1. **All pad rows of the text encoding are bit-identical** (max difference
   0.0 in all three prompts). Right padding under a causal mask means the
   real rows never see a pad, and every pad row has the same token (151643),
   the same position (**1**, pinned by diffusers because transformers 5.x
   would give 0) and the same visible keys. So X1 is the *unpadded* causal
   run plus **one extra row**: a query at position 1 over all the real keys,
   *not* its own, broadcast to the pad count. In the DiT the pad rows are
   distinct again — `txt_ids` gives them sequential positions.
2. **The residual stream does not fit fp16.** The text rows enter the first
   single block at absmax 6.2e5 and leave the last at **1.1e7**; `txt_in`'s
   output is already 1.15e5; the image rows reach 5.8e4. The text encoder's
   layer-32 output carries 1.12e4 in its first token on every prompt (an
   attention-sink channel). X5 keeps the residual in fp32 and narrows only
   LayerNorm outputs; any GEMM whose output is added to the residual stores
   fp32. The modulated activations are small (`norm_out` 3.7).
3. diffusers orders the joint sequence `[text, image, image_cond]`, BFL
   `[image, image_cond, text]`; with no mask the two are the same function,
   and the Go port may pick the order that suits the kernels.
4. Modulation is shared across every block of a stage (single0's and
   single27's `temb_mod` are the same tensor), so it is 6 GEMVs a step, not
   per block.
5. transformers warns that the tokenizer's regex is "incorrect" (a Mistral
   pattern check). The ids are Qwen's ordinary BPE and both oracles load it
   the same way; X1's tokenizer gate is against the dumped ids, not the
   warning.
6. **Edit sizes differ between the two references.** BFL fits one reference
   under 4 MiP and reflect-pads it to the 16 grid (several: each under
   1 MiP, centre-cropped), and returns the reference's own size, cropped
   back. diffusers keeps each reference at its own resolution (downscaled
   only past 4096², centre-cropped to 16) and returns **1 MP at the first
   reference's aspect**: the figurine (896x1120, already on the grid) came
   back 912x1152 where BFL's code would return 896x1120. The DiT does not
   care — X7's gates feed the dump's own tokens — but the server must pick
   one: **BFL's, by decision 2** (it is the checkpoint behind BFL's API),
   with the reflect pad and crop ported at X7.
7. **Noise: diffusers draws fp32, BFL rounds to bf16 values.** Ours is our
   own generator anyway (decision 7); it follows BFL's rounding (decision 2).
   Gates feed the dump's own noise, so it moves no gate.

### X1 — the text encoder

- [x] `flux3/textenc` (2026-10-15): `Template`, `Window`, `Encoder.IDs`,
      `Hidden`, `Encode`. The pad rows are **one** extra row: the real ids
      plus `PadID` at rotary position 1, run through `PadAttention` (causal
      over the real rows, the last row over the real keys only), and that
      row copied out to the window. The seam is `qwen.Layer.ForwardAttn`, a
      pure refactor of `Layer.Forward` (which now calls it with the causal
      `Attention`); no other caller changed.
- [x] `TestTemplate`, `TestTokenizer`: rendered template, window rule, pad
      id/position and the real ids exact on all three prompts (CJK included —
      transformers' "incorrect regex" warning does not bite them)
- [x] `TestLayerWalk`: hidden states 1 and 2 of `short`, all 80 rows, **max
      abs 1.4e-5 / 7.3e-4, rel 2.6e-5 / 2.0e-5** against the 2e-4 per-layer bound
- [x] `TestEncoder`: the full 20480-wide embedding of all three prompts,
      pad rows included, **rel 3.0e-5 / 4.2e-5 / 3.0e-5** (short / long /
      cjk), max abs 0.025 against rms 17–24, in 11.7 s on the CPU (~15 GB).
      The per-layer 2e-4 bound holds at depth 32, so it is the gate —
      qimage's 1e-3 (needed on the 8B at depth 36) is not borrowed
- [ ] GPU: `qwen.GPUEncoder` with the pad row — its attention is causal in
      the kernel, so the pad query needs a build or a second dispatch

### X2 — the DiT on the CPU (done 2026-10-15)

`flux3/dit`: `Load` (by `Parts`) / `LoadBlocks`, `NewRope`, `Block.Forward`
over row `Segment`s, `Model.Modulations`, `Forward`, `Final`/`FinalNorm`. One
`Block` type serves both kinds — a mode block is a block whose sequence is
one segment; a single block gets `[text, image, cond]` as three. Every gate
runs on the X0 block dump's **exact** inputs (`go test ./flux3/dit`):

| gate | measured (rel, worst element) | bound |
|---|---|---|
| `TestRope` — txt, image, cond tables from the dumped ids | **0** (bit-identical) | 1e-6 |
| `TestModulation` — image temb, all six (shift, scale, gate) sets | ≤ 3.4e-6 (temb), ≤ 5.9e-7 (mods) | 4e-5 |
| `TestModeBlocks` — txt / image / cond block 0 | 9.5e-6 / 2.8e-5 / 2.3e-5 | 4e-5 |
| `TestSingleBlocks` — block 0, block 27 (text rows; content rows) | 8.6e-6, 3.6e-6; 3.7e-6, 2.5e-5 | 4e-5 |
| `TestFinal` — norm_out; proj_out | 1.1e-5; 6.0e-6 | 4e-5 |
| `TestForward` — the whole model, 512² + one 512² reference / t2i | **2.5e-5 / 1.7e-5** (5 min CPU, 22 GB) | 1e-4 |
| `TestControls` — t_next = t; shift/scale swapped; ids unscaled | 0.75; 624; 6.7 | must exceed 4e-3 |

**The one bug the gates caught:** `norm_out` first landed at rel 1.0e-4.
Measured before blamed: torch's fp32 is 1.2e-5 from a float64 LayerNorm,
so the error was ours, and it was not the LayerNorm (float64 here) but the
**modulation GEMV**: `qwen.Linear` sums 3072 terms sequentially in float32,
and where `1+scale` nearly cancels against a large normalised channel
(inputs to 764) a 1e-6 error in scale becomes 1e-4 of the output. The
conditioning path is single rows, so `gemv` accumulates in float64 for free:
modulations 4.5e-6 → 5e-7, norm_out → 1.1e-5 (the reference's own distance
from float64). **X5 carries this**: the GPU's modulation GEMVs want fp32
accumulation with an fp32 (not fp16) scale, or a re-measure.

Error does not grow with depth: 33 blocks land inside the one-block bound,
even with text rows at absmax 1.1e7 (relative error is what fp32 keeps).

### X3 — the sampler on the CPU (2026-10-15)

`flux3/pipeline`: `NewSchedule`, `Schedule.T` / `Step`, `Noise`, `Denoise`,
and the latent bridge `BN.ToVAE` / `FromVAE` (BatchNorm + 2x2 patchify, both
directions). Gated against `reference/out/flux3run_t2i` (1024², 16 steps):

| gate | measured | bound |
|---|---|---|
| `TestSchedule` — the 17 times the model was handed, and each jump target | **exact** (all 34 values) | exact |
| `TestEulerSteps` — each step's latents rebuilt from the previous step's and its dumped velocity | **bit-identical**, all 15 transitions | exact |
| `TestToVAE` — the last step finished, de-normalised, unpatchified, against the VAE's input | ≤ 2 ulps (of `|v|·std + |mean|`) | 4 ulps |
| `TestFromVAE` — the encoder's posterior mode → normalised tokens | ≤ 2 ulps | 4 ulps |
| `TestNoise` — reproducible, every value bf16, mean/std 0/1 | ✓ | |
| `TestTeacherForced` — the dumped latents at steps 0, 8, 15 through the whole model | **3.1e-5 / 7.1e-5 / 7.7e-5** (25 min CPU) | 1e-4 |

**The teacher-forced margin is thin, and grows with the sequence.** The
whole stack was 2.5e-5 at 512² (2,128 tokens, X2) and is 7.7e-5 at 1024²
(4,176 tokens), against a 1e-4 bound set before measuring: 1.3x of margin.
It is deterministic, so it does not flake, and it is the shape of float32
sequential sums over more keys (the naive `attend`, `qwen.Linear`). Not
loosened; if a larger size needs it, measure the reference against float64
first, as `norm_out` was.

**The Go CPU path is 8.5 min a 1024² step** (torch fp32: ~30 s), so a free
16-step run is >2 h: the free-running prompt → image gate moves to X5, on the
GPU, against `reference/out/flux3run_t2i/image.png` and its per-step latents.

**What made the times exact.** diffusers builds the schedule in numpy
float32 after a float64 linspace — `shift * σ / (1 + (shift−1) * σ)` with the
Python scalars cast to float32 — and hands the model `(σ·1000)/1000`, which is
not σ. `NewSchedule` and `T` do the same arithmetic, with every float32
product an explicit conversion so the Go compiler cannot fuse it into an FMA.

**What did not need to be exact, and why.** torch 2.14's AVX-512 float32
`sqrt` is **not correctly rounded**: it disagrees with IEEE sqrt in 29 of the
128 BatchNorm stds (its add is exact). `LoadBN` takes the IEEE one, so the
bridge is within 2 ulps of the dump rather than equal — the reference is the
less accurate side, the third time in two verticals.

### X4 — the VAE on the CPU (done 2026-10-15)

`flux3/vae`: zimage/vae's FLUX.1 port restored from `77f6627^`
(`decoder.go`, `encoder.go`, `tensor.go`, `math.go`) with what
AutoencoderKLFlux2 adds — a 32-channel latent, `post_quant_conv` in front of
the decoder and `quant_conv` after the encoder — `LoadConfig` reading
`config.json` instead of a hardcoded FLUX.1 shape, FLUX.1's scaling/shift
pair gone (the BatchNorm in `pipeline.BN` replaces it), and `RGB8`, which
is diffusers' postprocess (`x·0.5+0.5`, clamp, `·255`, round half to even).
The checkpoint's tensor names are AutoencoderKL's, unchanged. Every stage is
fed the dump's input to that stage (`go test ./flux3/vae`, ~7 min):

| gate | measured (rel) | bound |
|---|---|---|
| decoder: post_quant_conv, conv_in, mid, up 0–3, norm, conv_out | 7.1e-7, 3.7e-6, **1.2e-5**, 1.4e-5 / 1.4e-5 / 2.8e-5 / 3.0e-6, 3.5e-7, 3.0e-6 | 1e-4 |
| encoder: conv_in, down 0–3, mid, norm, conv_out, quant_conv, mode | 1.2e-6, 8.7e-6 / 7.6e-6 / 9.0e-6 / 6.2e-6, 7.7e-6, 2.8e-7, 5.9e-6, 8.3e-7, exact | 1e-4 |
| encode free-running, 768x512 | 1.4e-5 | 1e-4 |
| decode free-running, 1024², the t2i run's own latent | 3.0e-5 | 1e-4 |
| that decode through `RGB8` against the run's PNG | **74 of 3,145,728 levels differ, each by 1** | ≤ 1 level |

**The one bug:** the decoder's mid block first landed at rel 1.08e-4. Its
single-head attention is over 16,384 pixels at 1024² (the encoder's, at
6,144, passed at 4.8e-5), and torch's fp32 measured **1.8e-6** from a
float64 run of the same block — so the error was ours: float32 sequential
sums in the scores, the softmax denominator and P·V. The CPU oracle now
accumulates those in float64 (mid block 1.2e-5; free decode 6.5e-5 → 3.0e-5).
That is the second time a long float32 sum was the gap (X2's modulation
GEMVs were the first); the GPU attention accumulates in fp32 over tiles,
which is a different sum, so X5 measures it rather than inheriting this.

The CPU decode is ~3 min at 1024² and the encode ~45 s at 768x512 on 32
cores — an oracle's speed. **What X5 inherits:** this decoder's activations
peak at 457 (X0), so the fp16 ladder (`FP16ConvOperands`, kept in
`tensor.go` with `convTap` for exactly that) is the first thing to run
before choosing the device path; Qwen-Image's refusal (its decision 7) was
its own tail norm at 2.6e5, not a property of VAEs.

### X5 — the GPU (done 2026-10-15)

`flux3/dit.GPU` (`gpu.go`): `NewGPU(dev, dir, maxRows, maxText, chunk)`,
`Begin(txt, txtIDs, imgIDs, cond, condIDs)`, `Forward(img, t, tNext)`,
`Profile`. It is h3/dit's graph — row chunks through one scratch, fp32
residual, per-run modulation, the LDS GEMM with the SwiGLU epilogue, the
transposed K/V-through-LDS attention — fitted to this block:

- **Parallel block, two adds.** A chunk's first pass runs the norm, q/k/v
  into the planes *and the whole FFN*, adding gate·ff at once (its rows'
  q/k/v are already packed, so the residual is not read again); the second
  pass adds gate·attn once every key is in. x + g·f + g·a, not x + g·(a+f):
  two fp32 adds, no scratch kept across the passes.
- **Modes are row ranges.** The residual is the joint [text, image, cond];
  a mode block is the same pass over its mode's rows with planes from row 0
  and attention over those rows only. The embedders write ascending so each
  128-row GEMM overrun is covered by the next mode's write.
- **One new shader build**, `flux3_norm_mod.spv`: `h3_norm_mod.comp` with
  `LAYER=1` (mean subtracted, then the variance about it; the default build
  is byte-identical). Everything else is an existing `.spv`.
- **The host keeps what needs precision and is not per-token**: every
  modulation (the CPU oracle's float64 GEMVs, X2's bug) and `txt_in`, once a
  request in fp32 — the embedding reaches 1.1e4 and its projection 1.1e5, and
  it does not depend on the step.
- **The key tail is zeroed in the graph** before every attention (the planes
  are rewritten at four lengths a forward). Measured not load-bearing on the
  LDS build, which masks the tail from the row max itself; kept for the
  plain fallback build, where h3 found the 0/0.

Gates (`go test -p 1 -run GPU ./flux3/...`, ~2 min; the DiT's share one
~7 s staging):

| gate | measured (rms rel; max/absmax) | bound |
|---|---|---|
| `TestGPUBlocks` — mode block 0 ×3, single 0 and 27, exact inputs | 3.9e-5 – 2.3e-4; ≤ 3.5e-4 | 1e-3 |
| `TestGPUForward` — 512² t2i / edit | 5.4e-4 / 5.8e-4; ≤ 1.4e-3 | 4e-3 |
| `TestGPUTeacherForced` — 1024² t2i, all 16 steps | 3.5e-4 – 9.7e-4; ≤ 7.4e-3 | 4e-3 |
| ″ — 912x1152 edit, one 896x1120 reference, all 16 | 4.0e-4 – 1.7e-3; ≤ 2.9e-2 | 4e-3 |
| `TestGPUFreeRun` — the t2i run's noise, 16 free steps | final latent 1.3e-3 (×1.4 a step from 4.4e-6) | 1e-2 |
| ″ — decoded on the CPU, against the run's PNG | **65.5 dB**, mean 0.02 levels | 45 dB |
| ″ — since: prompt text, device encoder and fp16 decoder | 1.95e-3; **63.1 dB**, mean 0.03 levels | 1e-2; 45 dB |
| `TestGPUControls` — RMS pre-norm (mean kept) | 6.7e-2 | must exceed 4e-3 |

qimage/dit's 2e-2 step bound was inherited first and was too loose here: the
RMS-norm control read only 3x of it. 4e-3 is 4x the worst t2i step.

**Where a 1024² step goes** (`TestGPUProfile`, a fence a dispatch, 1127 ms
over 1012 dispatches): GEMMs 75% at 38–43 TFLOP/s (gate|up 33%, down 18%,
q/k/v 18%, o 6%), attention 16% at 39 TFLOP/s, the gated adds 4%, the packs
4%, the norms 1%. Nothing anomalous; X9's list starts with the int8 bank
and the v-pack epilogue (h3's `C_PACK`), not with a kernel.

**The decoder: z-image's fp16 route, restored** (`flux3/vae/gpu*.go`, the
nine `.comp` files and their `shaders.go` block, all from `77f6627^`
verbatim). The FLUX.1 autoencoder's GPU path — implicit-GEMM 3x3 convs on
the matrix cores (`vae_pack_conv` + `vae_conv_wmma`), the mid block's
attention and projections in fp16 — with `post_quant_conv` (1x1, 32→32) in
front. The ladder was run first, on the CPU (`TestConvFP16Ladder`,
`FLUX3_LADDER=1`, the t2i run's latent):

| arm | decoded vs fp32 dump | 8-bit vs the run's PNG |
|---|---|---|
| fp32 (X4) | 3.0e-5 rel | 74 levels differ, all by 1 |
| **3x3** narrowed | max 3.6e-3, rms 2.3e-4 | 19,876 differ, all by 1; 70.1 dB |
| **all** narrowed (1x1s too) | max 3.0e-3, rms 3.2e-4 | 28,851 differ, all by 1; 68.5 dB |

So fp16 is safe here, 1x1s included: Qwen-Image's refusal (Q9b) was its tail
norm's 2.6e5, not a property of the family, and this decoder peaks at 457.
On the device (`TestGPUDecode`, 1024², two runs): **fp16 path 0.81 s**,
rms 3.3e-4, 30,261 levels off by 1 (68.3 dB) — the "all" arm's prediction
almost exactly; the fp32 scalar path 5.9 s, 92.8 dB.

**A watchdog finding on the restored code.** The scalar path submitted four
dispatches a batch (z-image's cap), and at 1024² the scalar mid block's
batch — three ~0.1 s projections and the ~1.5 s attention — sits at amdgpu's
2 s ring timeout: it passed once and was reset twice (`journalctl -k`: ring
gfx_0.0.0 timeout, reset succeeded; ai.service unaffected). That path now
submits a dispatch at a time. The first reset came from running the four
packages' GPU tests concurrently (`go test ./flux3/...` without `-p 1`), so
**every GPU gate now skips under `-short`**, and they run with `-p 1`.

**The text encoder** (`flux3/textenc.GPU`): `qwen.GPUEncoder` over 32 layers
with X1's pad row — the real tokens plus one `<|endoftext|>` at position 1 —
and the taps read back between layers (`RunHooked`). The pad row must see the
real keys and *not itself*, which a causal kernel cannot say, so
`dit_attention_wmma.comp` gained `PAD_LAST` (the last query row also skips
its diagonal key, in the row max and in P; every existing build is
byte-identical, checked over all 40 `.spv`s) and `GPUEncoder` gained
`UsePadLastAttention`. `TestGPUEncode`, three prompts (17 / 118 / 17 real
tokens, CJK included): every tap **≤ 7.2e-4 rms on the real rows, ≤ 1.1e-3 on
the pad rows** (bound 5e-3), **54–62 ms** a prompt. The control —
the plain causal build, the pad row seeing itself — puts tap 32's pad rows at
1.03 (205x the bound) and leaves the real rows at 3.8e-4.

**End to end** (`flux3/pipeline.GPU`, `cmd/flux3`; `pipeline.ImageIDs` /
`TextIDs` match every dumped id table exactly): `TestGPUFreeRun` takes the
t2i run's prompt *text* and noise through the device encoder, 16 free steps
and the fp16 decoder — final latent rms 1.95e-3, picture **63.1 dB** against
the fp32 run's PNG (mean 0.03 levels). `cmd/flux3` at 1024², the default
fox prompt, seeds 1 and 2: **19.40 s and 19.49 s** (encode 52/54 ms,
denoise 18.45/18.53 s = 1.15 s a step, decode 0.81 s), staged in 11 s at
10.6 GB of DiT banks + 0.5 GB activations, ~7 GB encoder, 0.3 GB VAE.

### X6 — serving text-to-image (done 2026-10-15, deployed)

**What serves.** `backend.Flux3` (an `api.ImageBackend` and a swap-slot
member over `flux3/pipeline.GPU`) answers `/v1/images/generations`;
`backend.ImageRouter` puts it in front of Qwen-Image, which answers
`/v1/images/edits` until X7. Qwen-Image is now `Lazy`: staged on the first
edit, not at startup, taking its own turn in the slot. `cmd/serve`: `-image`
is FLUX 3 (`-image-model models/flux-3-image-klein`, `-image-steps 16`,
`-image-size`), `-edits N` keeps Qwen-Image's edits (`-qimage-model`,
`-qimage-steps 40`, `-image-fp16` now names its banks). ai.service's line
did not change.

**taef2, and the one mistake worth recording.** `flux3/vae/tiny*.go` is
z-image's taef1 port (`77f6627^`) with taef2's addition — TAESD's `flux_2`
variant puts a pooled branch, `x + conv1x1(relu(GroupNorm(4)(conv1x1(x))))`
at 64→256→64 channels, in front of the first three blocks — built from
kernels X5 already restored. Oracle: `reference/dump_taef2.py` over
upstream's `taesd.py` (vendored, MIT, `reference/taesd/`). The port matched
it from the first run (every layer ≤ 1.8e-5, free decode 3.3e-5; device fp16
81 dB, **99 ms at 1024²**). **The input did not**: it was first fed the raw
latent (what the full decoder takes), and the previews came out
oversaturated. taef2 takes the **normalised** latent, un-patchified — the
model card's wrapper says so by construction (its BatchNorm is mean 0,
variance 1, so the pipeline's de-normalisation is the identity) — and it
measures so against the full VAE's PNG of the t2i run's latent: **32.8 dB
normalised, 17.8 dB raw** (contrast 1.7x). `TestTinyConvention` now gates
that; Qwen-Image's TAEQI2.1 had the same convention (Q14).

**Previews** decode each chosen step's denoised estimate x − t·v (the step's
input less its time times its velocity; under the flow map an estimate, not
the end). Three partials cost ~0.45 s of a 19.5 s image.

**The ceiling (X-o3).** `TestArenaCeiling` plans both decoders' graphs with
no device: the full decoder's fp32 activation arena is 2816 MB at 1024² and
reaches the 4 GiB storage-buffer range at **1232² = 1,517,824 pixels**, the
same at 2464x616 and 616x2464 — it scales with the area alone. taef2's is
832 MB at 1024². `pipeline.MaxPixels` and `NewFlux3` refuse past it.

**Served gate** (side server on :11500, then ai.service):

| request | result |
|---|---|
| 1024², seeds 1 and 2, side server | 19.50 s, 19.57 s; seed 1 **bit-identical** to `cmd/flux3`'s |
| 1024², ai.service after deploy | 33.4 s cold (music evicted, FLUX 3 staged 13.7 s), then **19.54 s, 19.46 s** |
| 1344x768 (same area) | 19.3 s |
| `stream: true`, `partial_images: 3` | steps 0, 5, 10 + the final, 19.96 s |
| `background: "transparent"` | **400**, naming the RGB autoencoder |
| 1000x1000 | 400 (not a multiple of 16) |
| an edit (one reference) | routed to Qwen-Image: 124 s including its 25.6 s staging; the next generation swaps back in 11.4 s |

Staging is 23.3 GB at 1024² (6.5 encoder, 10.6 transformer, 0.31 decoders,
6.0 activations) against Qwen-Image's 20.4 int8 — X9's int8 banks are what
bring it down.

### X7 — edits (done 2026-10-15)

**What serves.** `backend.Flux3` answers both endpoints; `-edits N` (≤ 10)
stages the VAE encoder and sizes the DiT's rows for N references
(`pipeline.CondRows`: one reference's canvas is capped at the staging's area,
several at min(1 MiP, area) each — 12,288 cond rows at `-edits 3`, 1024²).
`backend.ImageRouter` and the lazy Qwen-Image staging are gone from
`cmd/serve`, with the `-qimage-model`, `-qimage-steps`,
`-image-condition-size` and `-image-fp16` flags; `qimage/` itself waits for
X8. ai.service's line (`-image -edits 3`) is unchanged. Staging at 1024² is
**25.8 GB** (6.5 encoder + 10.6 transformer + 0.51 VAEs + 8.3 activations;
the encoder's fp32 arena is 1.5 GB of that — it could share the idle
decoder's at X9).

**The device encoder** (`flux3/vae/gpu_encoder.go`) is z-image's I7 graph
from `77f6627^` with the 1x1 `quant_conv` after `conv_out`, the scalar path
submitting a dispatch at a time (the decoder's watchdog finding), and the
arena check by area (any shape of no more pixels; `TestArenaCeiling` plans it
at 1.5 GB fp32 at 1024², 2.2 GB at the 1232² ceiling, the same in 4:1 and
1:4). `TestGPUEncode`, the dump's 768x512 picture, every stage from the
pixels (RunTo):

| arm | worst stage (rms rel) | mode | time |
|---|---|---|---|
| **fp16 cores** (served) | 2.9e-3 (down block 1) | 2.6e-3 | 144 ms |
| fp32 scalar | 1.6e-6 | 1.7e-6 | 1.04 s |

**Reference preparation** (`flux3/pipeline/prepare.go`) is BFL's
`prepare_reference` / `prepare_multi_reference` / `reference_canvas` with the
cap a parameter: Python's half-to-even `round`, the 0.995 shrink loop,
`convert("RGB")` dropping alpha without compositing, qimage's Pillow-exact
Lanczos (Q8.3), numpy's `reflect` (and `symmetric` at a side ≤ 16) as the
periodic mirror, which is what numpy's repeated reflection comes to for pads
longer than the side. Oracle `reference/dump_flux3_prepare.py` runs BFL's own
functions (the package aliased, nothing copied) on nine pictures at BFL's
4 MiP and at a 1024² cap; `TestPrepare`: **all 18 cases exact** — sizes,
boxes and every 8-bit value — on the first run. The served cap is the
staging's area, not 4 MiP: the canvas is the output's default size, and the
decoder's ceiling is 1232². **Not ported: EXIF orientation** (the API's
decoders do not read it; a rotated phone photo is edited as stored).

**The pipeline** (`GPU.Generate` with `Request.References`): prepare, encode
each reference (`ToTensor` is BFL's x/127.5 − 1), `BN.FromVAE` to normalised
tokens, ids `ImageIDs(h, w, −(1+k))`, concatenated in reference order into
`DiT.Begin`'s cond operand. No size → the first reference's canvas, and with
one reference the decoded picture **and every preview** are cropped back to
the fitted box. BFL rounds the reference tokens to bf16 (its DiT's dtype);
we do not — the fp32 oracle does not either.

**The gate** (`TestGPUEdit`): the edit run's prompt text and the figurine
(896x1120, on the grid, so BFL's and diffusers' preparations agree), the run's
noise, 16 free steps at the run's 912x1152:

| encoder | cond tokens vs dump | final latent | picture vs the run's PNG | references |
|---|---|---|---|---|
| **fp16** (served) | 1.8e-3 | 4.6e-3 | **58.8 dB**, mean 0.07 levels | 0.42 s |
| fp32 scalar (`FLUX3_EDIT_SCALAR=1`) | 2.9e-6 | 1.6e-3 | 62.9 dB, mean 0.03 levels | 3.9 s |

Both clear the 45 dB bar by 14 dB; the fp32 encoder buys 4 dB for 3.5 s of a
41 s edit, so fp16 ships. Denoise is 40.3 s (2.5 s a step at 8.1k rows), as
X5's teacher-forced edit predicted.

**Served** (side server `-image -edits 3` on :11500, `-token=`):

| request | result |
|---|---|
| figurine, no size, seeds 1 and 2 | **896x1120** (BFL's default), **40.76 s, 40.64 s**; the candle-lit edit asked for |
| 2999x2001 picture, no size | fitted to 1247x832, padded to 1248, **cropped back: 1247x832**, 42.2 s |
| 300x200 picture | padded to 304x208, cropped back to 300x200, 3.4 s |
| two references (figurine + 2000x1500) | 896x1120 (3920 + 4015 cond tokens), 69.2 s, both pictures used |
| figurine, `size: 1024x1024` | 1024x1024, 41.9 s |
| four references | **400** naming the 3 |
| `background: "transparent"` | 400 |
| `stream: true`, `partial_images: 2` on the 300x200 | two partials and the final, **all 300x200** |
| a 1024² generation on the same staging | 19.4 s |

**One API fix found on the way:** an edit's stream was named
`image_generation.*`; OpenAI names it `image_edit.partial_image` /
`image_edit.completed` on `/v1/images/edits`, and it now does.

## Open questions

- **X-o1**: should the server upsample prompts (the served LLM with BFL's
  system prompts), or leave it to clients? Quality depends on it more than
  on anything we port.
- ~~**X-o2**: does diffusers reproduce BFL's reference preparation?~~
  **Settled 2026-10-15 (X0): no** — finding 6; the server follows BFL.
- ~~**X-o3**: the VAE arena ceiling~~ **Settled 2026-10-15 (X6):
  1,517,824 pixels (1232²), any shape**, set by the full decoder's fp32
  arena; the reference's 4 MiP would need a tiled decode.
- **X-o4**: does the text encoder carry a massive-activation channel in
  particular layers, as Qwen-Image's 8B did in 6 and 16? (X9's int8 bank.)
- **X-o5**: is a lower step count safe for any prompt kind? Not planned; the
  checkpoint is already distilled to 16.

## Handoff

**2026-10-15, session 6: X6 done and deployed.** ai.service answers
`/v1/images/generations` with FLUX 3 (19.5 s at 1024², taef2 previews,
`transparent` a 400) and `/v1/images/edits` with Qwen-Image, staged on the
first edit (`backend.ImageRouter`). Gates: `go test -short ./flux3/...`
(CPU) and `go test -p 1 -run 'GPU|Tiny' ./flux3/...` (device, one package at
a time); `TestArenaCeiling` needs no device.

**2026-10-15, session 7: X7 done.** FLUX 3 answers both endpoints; edits
follow BFL's preparation exactly, 40.7 s for a one-reference ~1 MP edit.
Gates: `go test -short ./flux3/...`, `go test -p 1 -run 'GPU|Tiny'
./flux3/...` (TestGPUEdit ~1 min, TestGPUEncode 10 s).

**Next: X8, delete Qwen-Image-2.1** (hoist `qimage/pipeline.Resize` — now
also flux3's — and `qimage/vae.Tensor` first). Then X9. Small open items:
EXIF orientation; the encoder's arena could share the decoder's.

**Superseded handoff (session 6): next was X7, edits on FLUX 3.** The DiT side is done and gated (the edit run
teacher-forced at 1.7e-3, 2.5 s a step at 8.1k rows). Missing: the VAE
encoder on the device (`77f6627^`'s `gpu_encoder.go` beside the restored
decoder; the CPU encoder is X4's), the reference preparation (BFL's: one
reference fitted under 4 MiP — capped by our 1.52 MP — reflect-padded to the
16 grid and the output cropped back; several each under 1 MiP, centre-cropped;
decided at X0 that the server follows BFL, not diffusers), the `image_cond`
ids (`ImageIDs(h, w, -(1+k))`), `-edits N` sizing the DiT's rows, and the
gate against `reference/out/flux3run_edit` (its PNG). Then `ImageRouter`
goes and X8 deletes Qwen-Image.
