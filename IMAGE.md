# IMAGE — the Qwen-Image-2.1 vertical

> **Rewritten through 2026-09-20 — Q0–Q8 are done, the vertical's last
> capability is served, and Q9's first pass has been taken**: `POST
> /v1/images/generations` answers at **1m32s for a 1024²/40-step image** and
> `POST /v1/images/edits` at **1m59s for a 1024² edit on one reference**,
> 31.5 and 39.4 GB resident, reproducing the reference edit at **max abs
> 0.0014** — 24x tighter than the t2i path's own served number, because an
> edit's trajectory is pinned by its prefix. Q9 so far: a 6–7% step, bit for
> bit the same picture, from one badly-shaped launch the profile caught.
>
> **2026-09-21 — Q10: the ceiling is an area, not a box.** A served 16:9
> request was coming back 1024x576 against arenas that hold 1.05 Mpx, because
> the geometry limited each *side*. The VAE's arena turns out to be **exactly
> 3060 bytes a pixel for every aspect ratio** (`TestArenaShape`), so the shape
> was never a constraint: the same server now answers **1344x768**, 1.75x the
> pixels, at the same wall clock, and `-image-size 1184x1184` reaches
> **1536x864**. Nothing about residency or the numerics moved.
>
> **2026-09-21 — Q11: a hung-up client now stops the run.** The context
> reaches the sampler and the VAE's submit loop, so an abandoned request costs
> **one step (21% of a short run) or 295 ms of a decode** instead of the whole
> picture — and stops holding the process-wide device lock against every other
> vertical. The run after a cancellation is bit-identical to one before it.
>
> **Q0–Q7, one day in** and
> the model was *served*: `POST /v1/images/generations` answered from
> Qwen-Image-2.1 at **1m38s for a 1024²/40-step image** (Q6's own
> measurement; Q9 has since taken it to 1m32), 31.7 GB resident,
> with native RGBA and in-progress previews. On the CPU the port reproduces
> diffusers' image to under a quarter of an 8-bit quantization step; on the
> device the served fp16 path reproduces the fp32 oracle's own picture at
> **mean 3.4e-4**. What remains for the vertical is percents (Q9) — plus
> finishing the Z-Image deletion, which Q6 owes and half did.
> This file is live again: the old root
> `IMAGE.md` was frozen into
> [`research/zimage-vertical.md`](research/zimage-vertical.md) at the 2026-09-20
> consolidation, and this one is the **replacement of Z-Image-Turbo by
> `Qwen/Qwen-Image-2.1`** for both generation and edits. Same rules as every
> vertical file: rewritten each session, not appended to; closed stages go to
> `research/`. Stage letters here are **Q0, Q1, …** (I-stages stay z-image's,
> in the archive).
>
> Breaking the current vertical while we do this was accepted up front
> (2026-09-20 session). Z-Image code gets deleted, not kept in parallel —
> except `zimage/qwen`, which `embed`, `llm` and `parakeet` also import
> (see "what survives").

**Target**: `Qwen/Qwen-Image-2.1` — prompt → PNG (optionally RGBA) at up to
2048x2048, and native reference-image editing (up to 10 refs), end to end in
Go on Vulkan, served at `POST /v1/images/generations` and
`POST /v1/images/edits`, with in-progress previews.

## Where the work stands

| # | Stage | State |
|---|---|---|
| Q0 | Reference env + dumps (six oracles, `reference/dump_qi21_*.py`) | **done 2026-09-20** — incl. the saved 1024²/40 fp32 run; two runs byte-identical |
| Q1 | Text encoding in Go (`qimage/textenc` over an unchanged `zimage/qwen`) | **done** — CPU ≤7.1e-4 (its own noise floor measured), GPU fp16 characterized 10-85x tighter than the official bf16 |
| Q2 | One DiT block on CPU (`qimage/dit`) | **done** — block, cache contents, both regimes ≤4e-5; negative control 1e5x |
| Q3 | Stack + scheduler + sampler on CPU (`qimage/pipeline`) | **done** — per-step latents 2.1e-5→2.5e-4 vs the oracle |
| Q5 | VAE on CPU (`qimage/vae`) + the end-to-end image gate | **done** — decoder/encoder stagewise; **prompt→image matches diffusers at max abs 9.1e-4** (24x inside the 2.2e-2 precedent) |
| Q4 | DiT on the GPU (`qimage/dit/gpu.go`) | **done for t2i** — teacher-forced flat ≤8e-3 over 40 steps; **2.27 s/step at 1024²** |
| Q5g | VAE decoder on the GPU (`qimage/vae/gpu.go`) | **done 2026-09-20** — stagewise on the dump's own bounds; **1024² in 7.4 s**, 117 dispatches. Encoder stays CPU until Q8 needs it |
| Q6 | Serve t2i (`/v1/images/generations`, RGBA, ceiling geometry) | **done 2026-09-20** — `qimage/pipeline` resident at **31.7 GB**, served 1024²/40 in **1m38.2s / 1m41.3s**, `background: "transparent"` answered; Z-Image deletion still owed |
| Q7 | Previews (fitted linear 64→RGB; no tiny AE exists for this VAE) | **done 2026-09-20** — R² 0.97, **159 µs a frame**, three partials cost 0.3% of a request; previews are unconditional, the flags are gone |
| Q8 | Edits (vision tower, multi-ref, VAE encoder serving) | **done 2026-09-20** — `/v1/images/edits` answers from a **2m8s served edit at 1024²**, 39.4 GB resident, reproducing the reference edit at **max abs 0.0014, mean 1.7e-4** (24x tighter than t2i's own served number); seventeen controls firing |
| Q9 | Percents (fusion ports, tile re-screens, the full-seq re-pack) | **first pass done 2026-09-20** — the DiT attributed (`TestGPUStepProfile`), the GEMM swizzle re-screen closed with a measurement (SWZ=8 wins here too), the fragment pack taken **44 → 131 GB/s**: image 1m38→**1m32**, edit 2m8→**1m59**, output bit-identical. The VAE's two priced ports are next |
| Q10 | The ceiling is an area (`api`, `qimage/pipeline`) | **done 2026-09-21** — the arena measured at **3060 B/px for every aspect ratio** (`TestArenaShape`), so a side box was costing 16:9 **44% of its pixels**: the same server now answers `aspect_ratio: "16:9"` with **1344x768 instead of 1024x576**, at the same wall clock |
| Q11 | Cancellation (`qimage/*`, `backend`) | **done 2026-09-21** — the context reaches the sampler and the VAE's submit batches: a hung-up client stops in **21% of a run** (one step) or **295 ms of a decode** instead of paying for the whole image, and the next run is bit-identical (`TestCancellation`) |

Every gate is dump-driven and every tolerance in this file is measured, with
the instrument named beside it. The day's method finding, three times over:
when a port exceeds a borrowed tolerance, measure the *reference against
itself* (swap its attention kernel, its thread count, its dtype) before
hunting a bug — four of the bounds below are set that way, and the third
instrument, a float64 run, twice found the **reference** to be the less
accurate side (Q5's decoder tail, Q8.2's wide condition image).

**References, in the order to reach for them:**

- `models/Qwen-Image-2.1/` — the checkpoint is cloned; configs quoted below
  are read from it, not from docs.
- `reference/qwenimage21/{pipeline,transformer,autoencoder_kl}_qwenimage21.py`
  — the **merged** diffusers implementation (PR
  [huggingface/diffusers#14804](https://github.com/huggingface/diffusers/pull/14804)),
  copied verbatim 2026-09-20. This is the oracle every dump comes from.
- [Model card](https://huggingface.co/Qwen/Qwen-Image-2.1) ·
  [blog](https://qwen.ai/blog?id=qwen-image-2.1) (JS-only page, fetch fails —
  the model card and PR carry the same facts) ·
  [examples repo](https://github.com/QwenLM/Qwen-Image-2.1) ·
  [sglang port](https://github.com/sgl-project/sglang/pull/39983) (a second
  implementation to cross-check against; its follow-ups
  [#40447](https://github.com/sgl-project/sglang/pull/40447) document the
  per-layer prefix-cache layout).
- License: **Qwen Research License** (`models/Qwen-Image-2.1/LICENSE`) — not
  Apache like Z-Image was. Fine for this project; worth remembering if the
  server ever grows other users.

---

## The model, as the checkpoint describes it

Three components, ~32.3 GB on disk (bf16), against Z-Image's ~20 GB resident:

| | | |
|---|---|---|
| DiT | `QwenImage21Transformer2DModel` — **32 single-stream blocks**, dim 4096 (32 heads x 128), SwiGLU **12288** (mlp_ratio 3), **no biases anywhere**, per-head-dim q/k RMSNorm (eps 1e-6), 7.1 B params | 14 GB |
| Text encoder | **Qwen3-VL-8B** (`Qwen3VLForConditionalGeneration`): text model 36 layers, hidden 4096, 32 q / 8 kv heads of 128, theta **5e6**, vocab 151936; plus a 27-layer vision tower (hidden 1152, patch 16, spatial-merge 2, deepstack taps at layers 8/16/24) | 17 GB |
| VAE | `AutoencoderKLQwenImage21` — Wan-lineage causal-3D residual conv AE, encoder base 96 / decoder base **144**, dim_mult [1,2,4,8,8], **z_dim 64, 16x spatial**, **4-channel RGBA in and out**, per-channel `latents_mean/std` (64 values each) | 1.3 GB |

Geometry: **one latent token = one 16x16 pixel tile** (patch_size 1, VAE /16).
Pixel sides must be multiples of **32** (the VLM groups latent tokens 2x2).
1024² = 4096 image tokens — the same DiT sequence length as Z-Image at 1024².
2048² (the model's native size) = 16384. Defaults: **40 steps, no CFG**
(`true_cfg_scale=1.0`; true-CFG exists but is off by design), pipeline default
resolution **1024²** (the README's examples use 2048² explicitly).

### The four mechanisms that make it different from Z-Image

**1. One joint sequence, block-causal.** Text embeddings and image latents
share a single stream. The mask is `(q_idx >= kv_idx) OR same_image_block`:
the sequence is causal, text tokens strictly so, but every image block (each
condition image, and the target) is internally bidirectional. In diffusers the
non-flex processor decomposes this exactly into one attention call per prefix
segment (text segments get a causal triangle) plus one full-attention call for
the target — that decomposition, not a masked kernel, is the shape we port.

**2. `causal_condition` + prefix KV cache.** Text and condition-image tokens
are modulated from **t = 0** (the modulation tensor carries batch+1 rows; the
target's tokens read their sample's row, everything else reads the t=0 row).
That makes the prefix's activations step-independent, so **step 0 runs the
full block-causal prefill and stores per-layer post-RoPE K/V for the prefix;
steps 1..N-1 recompute only the target's tokens**, attending over
[cached prefix K/V ++ fresh target K/V] with no mask at all (batch 1, no
padding). For t2i the prefix is just the prompt (tens of tokens — small win);
for edits it is thousands of tokens per reference image (large win). Caching
on/off changes rounding, not correctness — diffusers documents that the two
settings give *visibly different but equally valid* samples. **We ship
cache-on and validate against diffusers `use_kv_cache=True`.**

**3. Sequence assembly and RoPE.** The VLM sequence's `<|image_pad|>` slots
each stand for a **2x2 group of latent tokens**: the transformer expands those
positions four-fold and drops VAE latents into them; the target's slots are
appended after the text. Text embeddings pass through `txt_in` (zero-centered
RMSNorm → 4096→4096 → GELU-tanh → 4096→4096); latents through `img_in`
(64→4096, no bias). RoPE is 3-axis (frame/h/w, dims 16/56/56, theta 10000),
complex adjacent-pair rotation — **the same pairing as Z-Image's DiT**, not
the NeoX halves of the text encoder. Text advances all three axes together;
each image block freezes the frame axis and lays h/w out on a **zero-centered
grid with negative indices** (a flipped negative-frequency table serves
indices −1024..−1); after an image block the position cursor advances by
`max(h, w)`.

**4. Shared modulation.** One `SiLU → Linear(4096→16384)` for the whole
model — blocks own no adaLN parameters. Each block chunks it into
(scale1, gate1, scale2, gate2); gates pass through **tanh**; norms are plain
non-affine LayerNorm. Final head: scale-only AdaLN + `proj_out` 4096→64.
Timestep embedding: sinusoidal 256 (cos first half — Z-Image convention
differs, check the dump), then a bias-free MLP to 4096.

### Text encoding, exactly

Raw template strings, **not** `apply_chat_template` (the two tokenize
differently and the checkpoint expects the raw form):

```
t2i:  <|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
edit: same, with "<image1><|vision_start|><|image_pad|><|vision_end|>" (then " <image2>…" etc.) prefixed to the user text
```

- The embedding is the **last decoder layer's output *before* the final
  RMSNorm** (all 36 layers run — different from Z-Image's `hidden_states[-2]`
  where the last layer never ran). transformers 5.x silently normalizes
  `hidden_states[-1]`; diffusers neutralizes that with a forward hook on the
  final norm, and Q0 **verified the hook on transformers 5.17.0**: the hooked
  `hidden_states[-1]` equals the pipeline's output exactly (gap 0), so 5.17
  is the dump env — the model card requires >= 5.17 anyway.
- Drop the first `drop_idx` tokens, where drop_idx = length of the tokenized
  system message (computed, not hardcoded). Empty prompt becomes `" "`.
- For edits, condition images additionally go through the vision tower (RGBA
  flattened over white **for the vision copy only**); mrope with real 3D
  positions at image slots. Text-only prompts make mrope collapse to standard
  NeoX RoPE — **verified in Q0**: the model's own cos/sin tables for a text
  prompt match a plain theta-5e6 table to 0.0 (`mrope_gap` in the textenc
  manifest), so nothing mrope-shaped is needed before Q8.
- A fact Q8 rests on, confirmed while building the head dump: **the VLM's
  image-slot embedding rows are overwritten by the VAE latents** in the joint
  stream. Vision content reaches the DiT only through what the *text* tokens
  absorbed from the image inside the text encoder's own attention. The vision
  tower conditions the text; the VAE conditions the pixels.

### Scheduler

FlowMatchEulerDiscrete with **dynamic shifting**: sigmas `linspace(1, 1/N, N)`,
`mu` linear in the **target** token count between (256, 0.5) and (8192, 0.9)
— at 1024², mu = 0.6935, and **unclamped past 8192**: 2048² extrapolates to
mu = 1.3129 (measured, Q0) — `time_shift_type: exponential`,
`shift_terminal: 0.02`, `stochastic_sampling: false`. Port from diffusers'
scheduler, not from Z-Image's fixed shift 3.0. The scheduler steps **only the
target latents** (the model's output rows for the prefix are discarded).
VAE decode applies `z * std + mean` per channel; encode the inverse, and uses
the **posterior mode** (`argmax`), which keeps our seed-reproducibility rule
for free.

---

## What this changes about the vertical (semantics, not code)

- **Edits stop being SDEdit.** Today an edit is a truncated generation
  (strength ⇒ steps skipped ⇒ cheaper). In 2.1 an edit is a *conditional
  generation*: reference latents + vision context as prefix, target from pure
  noise, all 40 steps. An edit becomes **slightly more expensive** than a
  generation (bigger prefix), the `strength` knob disappears, and up to **10
  reference images** arrive (OpenAI's `image[]` array already allows several).
  `/v1/images/edits` keeps its shape; its documentation changes meaning.
  **Measured, Q8.6**: 2m8s against a generation's 1m38s at 1024² with one
  reference, `strength` is a 400 explaining the change, and how many
  references a given server takes is `-edits N` and `max_reference_images`.
- **RGBA is native.** The VAE emits 4 channels always; transparency is asked
  for in the prompt (the model card's recommended phrasing). Map OpenAI's
  `background: "transparent"` to that prompt prefix + PNG-with-alpha out;
  opaque requests composite over white.
- **Masked / annotated local edits** (circles, painted regions, separate
  masks) are a *capability the model has* — the current 501 for masks may
  become closable. How a separate mask is fed (extra condition image?) is not
  in the diffusers PR; needs the examples repo before promising it (open
  question Q-o3).
- **The cost profile inverts — now measured, no longer estimated.**
  Z-Image-Turbo was an 8-step distillation at 14.3 s an image; 2.1 wants 40
  steps, and Q4's GPU DiT measures **2.27 s a cached step, 2.29 s the
  prefill: 90.7 s of transformer per 1024² image** (the morning's estimate
  said ~70 s; the gap is unported fusions and the full-seq re-pack, priced
  under Q9). **Q6 measured the whole thing served: 1m38.2s / 1m41.3s at
  1024²/40** — 92% the denoising steps, 7.6% the VAE, 0.1% the text encoder.
  Q9's two priced VAE ports would take the decode from 7.5 s to ~1.3 s, i.e.
  the image to ~92 s; everything past that is the DiT's.
  2048² is not reachable at all today — the VAE's arena ceiling is 1184²
  (Q6) — so the DiT's x4/x16 estimate for it is moot until that is lifted.
  The two levers, in order: **the step count** (sweep 40/24/16/12 for
  quality — the knob exists, and at 2.27 s/step it is the difference between
  ~90 s and ~27 s) and **a distilled/turbo release or LoRA** (the ecosystem
  — lightx2v etc. — is where Z-Image-Turbo itself came from; watch for it).
  **Q6's sweep decides the served default**; 1024²/40 is the starting point,
  not a commitment.

## The preview problem (taef1 has no successor here)

`madebyollin/taehv`'s **taew2_1 does *not* fit this model**: it decodes the
Wan-2.1 16-channel/8x latent that the *original* Qwen-Image borrowed; 2.1's
VAE is a new 64-channel/16x RGBA design, and as of today taehv lists nothing
for it. So, in order:

1. **Q7 shipped a fitted linear preview** (done 2026-09-20): a 64→4 matrix,
   least-squares over (latent, 1/16-scale image) pairs — the latent2rgb
   approach. One GEMV per latent pixel, measured at 159 µs a frame; a 1024²
   generation previews at 64x64, upscaled client-side. The fitting data is
   *generated* rather than taken from Q0's dumps (`cmd/previewfit`, 24
   prompts spanning colour and content), because a pair the pipeline
   produced itself cannot disagree with it about normalisation or layout.
2. **Watch taehv** for a 2.1 variant and port it like `zimage/vae/tiny.go`
   (at T=1 its temporal machinery degenerates; it would be a small 2D conv
   stack again). Do not build our own distilled decoder — that is a training
   project this repo does not want.

## What survives, what is new, what dies

| piece | verdict |
|---|---|
| `zimage/qwen` (Qwen3 text transformer, CPU + GPU) | **survives and grows** — it is shared infra (`embed`, `llm`, `parakeet` import it). Q1 adds: theta as config (5e6), 36-layer/pre-final-norm output mode beside the existing `EncoderLayers()` convention, and (Q8 only) mrope + deepstack + the vision tower. The Qwen3-Embedding caller must keep passing its tests untouched. |
| `zimage/tokenizer` | **survives** — same BPE family; new `added_tokens.json` (`<image1>`…, vision markers) and the raw template above. |
| shaders: `dit_gemm_*`, `dit_attention_*` (WMMA), gpu arenas | **survived, confirmed by Q4** — the whole graph runs on z-image's kernels plus two new small shaders (`dit_attn_causal`, `dit_copy`). The new shapes run on z-image's measured winners *uncontested*: the tile re-screen at M=16384/K=12288 is a Q9 percent (hypothesis 5 in `TODO.md`). |
| `zimage/vae/gpu_conv*` | **survives as infrastructure** — confirmed by Q5g: six of the decoder's ten kernels are z-image's unchanged (conv2d, add, 2x upsample, the row shuffles, the linear, the transposed-K attention), and `gpu_conv`'s matrix-core implicit GEMM is what Q9's 69% is waiting on. Only `qvae_chnorm`, `qvae_dupup` and a `-DDIM=1152` attention build are new. |
| `zimage/dit`, `zimage/pipeline` | **deleted 2026-09-20 (Q6)** — replaced by `qimage/dit` and `qimage/pipeline`. New names because almost no line survived: different block, different mask. |
| `zimage/vae` (model code, `tiny*`, both GPU paths) | **stayed past its replacement, deliberately** — `gpu_conv.go`'s matrix-core convolution is what Q9 wants and `TestGPUConvMatchesScalar` is the proof it works. It goes when Q9 has taken the packing helpers across, not before. |
| `cmd/zimage`, `cmd/ditstack`, `cmd/ditbench`, `cmd/ditblock` | **deleted 2026-09-20 (Q6)** — `cmd/qimage` is the replacement driver. `cmd/vaebench`/`vaedecode`/`vaeprof` stay while `zimage/vae` does. |
| `models/Z-Image-Turbo`, `models/taef1` (62 GB) | unreferenced since Q6 and still on disk. One `rm -rf` whenever the space is wanted; not deleted unasked. |
| serving: `/v1/images/*`, streaming, geometry-under-a-ceiling | **survived, confirmed by Q6** — `api.ImageBackend`, the geometry split, the SSE envelope and `backend.partialSteps` all carried over untouched; the adapter re-wired to `qimage/pipeline` and the two endpoints that lost their mechanism (streaming, edits) refused with the stage that owed them until Q7 and Q8.6 closed both. Two fields were added and one removed: `background` and `max_reference_images` in, `default_strength` out, because 2.1 has no strength. |

Two-machine deployment holds: image goes ~26 → **31.7 GB resident for
generation alone and 39.4 GB with one 1024² reference image's editing**
(Q8.6's measurement, inside the ~36–40 GB this file estimated before any of
it was built), sharing a 128 GB box with speech + TTS + embeddings (~3 GB).
Still no footprint quantisation.

## The stages

Same doctrine as every vertical: **CPU reference first, validated against a
dump, then the GPU port debugged against the CPU one.** One stage, one gate,
two-run numbers.

- **Q0 — reference environment and dumps. Done 2026-09-20**, full-size
  oracle included: `out/qi21run1024` is the saved 1024²/40 fp32 CPU run
  (~52 s/step, ~35 min, ~68 GB — nothing else heavy may run beside it; a
  34 GB Go test alongside got both OOM-killed), and its PNG is a
  photorealistic render of the prompt. The env: the shared `.venv` with torch
  2.14.0+cpu, transformers 5.17.0, torchvision 0.29.0+cpu, accelerate, and
  **diffusers 0.41.0.dev0 pinned to git commit `80c7ed26`** (stable 0.40.0
  predates the merge); `reference/qwenimage21/` is byte-identical to the
  installed package. Five scripts, all fp32 CPU, all in `reference/`:
  - `dump_qi21_sched.py` → `out/qi21sched` — sigma/timestep tables for five
    size/step cases plus one Euler step over deterministic vectors.
  - `dump_qi21_dit.py` → `out/qi21dit` — everything around the blocks
    (num_layers=0, 9 tensors): txt_in/img_in, joint assembly, RoPE tables,
    image_ids/target mask, the t=0 modulation row, prefill and cached
    forwards, for a t2i case and a synthetic edit case. The
    prefill-vs-cached identity at 0 blocks measured exactly 0.
  - `dump_qi21_dit_block.py` → `out/qi21block` — block 0 with real weights
    in prefill/extract/cached modes, cache K/V contents included; the fp32
    cached-vs-prefill gap on target rows measured **≤ 3.8e-6** (t2i), 0
    (edit) — Q4's numeric bound.
  - `dump_qi21_vae.py` → `out/qi21vae` — decode from seeded noise and encode
    of a deterministic RGBA test card (posterior **mode**), per-child and
    per-up/down-block intermediates hooked, 256² and 192x320. Decode clamps
    to [-1,1]; encoder conv_out is 128 = 64 mean + 64 logvar.
  - `dump_qi21_run.py` → `out/qi21run` + `out/qi21run2` — the end-to-end
    oracle: 256²/4 steps, seed 42, kv-cache on, noise passed in packed (the
    Go side gets it as data, not torch RNG), per-step latents, final RGBA
    image + PNG. **Two runs byte-identical**, and the PNG is recognizably
    the prompt. `--size 1024 --steps 40 --out out/qi21run1024` is the saved
    full-size run.
  - `dump_qi21_textenc.py` → `out/qi21textenc` — runs the pipeline's own
    `_get_qwen_prompt_embeds` (en / cjk / empty prompts), plus a layer walk
    (embed, layers 0-1, last-prenorm) and the rope cos/sin capture that
    settled Q-o5. drop_idx = 14 for the shipped system prompt.
- **Q1 — text encoding in Go. Done 2026-09-20.** `zimage/qwen` needed **no
  change at all** — `qimage/textenc` (the tree's first package) supplies the
  nested-`text_config` adapter (Prefix `model.language_model.`), the raw t2i
  template, the tokenized-system-message drop, and `Drop`; `Load(dir,
  cfg.NumLayers)` already expresses "all 36 layers, no final norm". Gates:
  tokenizer exact on en/cjk/empty; layer walk ≤ 2e-5; **CPU encoder en
  1.6e-4 / cjk 8.6e-5 / empty 7.1e-4 against a `deepTol` of 1e-3** — measured,
  not chosen: the reference disagrees with *itself* by 3.2e-4 on the empty
  prompt when only the attention kernel changes (eager vs sdpa), so
  z-image's 2e-4 sits below this model's own fp32 noise floor at depth 36.
  **GPU encoder (fp16, unchanged kernels) en/cjk 0.024-0.027, empty 0.21**,
  and that number has a mechanism (`TestGPULadder`, kept as the permanent
  instrument): the residual stream grows a ~13k massive-activation channel
  at layer 6, carries it flat, and **cancels it in layers 34-35 — the two
  layers Z-Image never ran** — so the fp16 bulk-rounding variance the
  channel accumulated surfaces at the end (protecting just the spike
  elements measurably does not help). Priced against deployment: the
  **official bf16 pipeline deviates from fp32 by rel 2.0 (abs 204)** through
  the same mechanism — our fp16 path is 10-85x closer to fp32 than what the
  ecosystem ships. Not taken (priced): an fp32-activation GEMM arm for the
  encoder — at prompt-sized M (~40 rows) the encoder is weight-bandwidth-
  bound so it would be cheap; build it only if Q3/Q6 show prompt adherence
  suffering.
- **Q2 — one DiT block on CPU. Done 2026-09-20** (`qimage/dit`): layout
  (image ids/target mask/segments) **exact** vs the dump, RoPE tables at
  ~5e-7 (negative-index grids included), head pieces at ~1e-5, the no-block
  prefill/cached forwards at ~4e-6, and the block gate — out_prefill,
  cache K/V contents, out_cached, both cases — at **≤ 4e-5** against
  `out/qi21block`. The negative control (NeoX-halves rotary pairing) is
  caught at 1e5x the bound. `Load(dir, blocks)` takes 0/1/32 blocks, so Q3's
  stack is a loop away. One porting trap found and worth remembering:
  `chunk(2, -1)` splits *rows* — rewrapping the modulation's backing array
  at half width hands row 1 the second half of row 0, and every downstream
  tensor fails at rel ~10 while every shape checks out.
- **Q3 — the stack + scheduler + t2i sampler on CPU. Done 2026-09-20 at the
  latent gate** (`qimage/pipeline`): the dynamic-shift scheduler matches
  diffusers at ~1e-6 across all five dumped size/step cases (mu exact, the
  exponential shift, the terminal stretch, the Euler step), and
  `Denoise` — 32 blocks, ModeExtract at step 0 then cached decodes, exactly
  as served — reproduces the 256²/4-step oracle's **per-step latents at
  rel 2.1e-5 → 2.5e-4** (bound 1e-3, 4x measured; 2.3 min a sample on CPU).
  **Still owed to Q5**: the *image* gate — prompt → PNG with our own text
  encoder and VAE decode against the oracle's `image` tensor (Z-Image's
  2.2e-2 precedent). The three green gates (textenc, denoise, decoder) do
  not compose into that claim; it gets measured, once the decoder exists.
- **Q4 — DiT on the GPU. Done 2026-09-20 for t2i** (`qimage/dit/gpu.go`,
  ~750 lines on z-image's shader set): **1024²/40 steps in 90.7 s — prefill
  2.29 s, cached steps 2.27 s** — against CPU Go's ~2.5 h and torch fp32's
  ~35 min. The mapping that made it small: the block's LayerNorm+modulate+
  narrow is `dit_final_norm.comp` verbatim; the shared modulation is host
  arithmetic (ten uploaded vectors) with the t=0 row-select expressed as two
  dispatch ranges; SwiGLU's fp16 range problem is solved *exactly* by
  scaling 1/16 into fp16 and folding 16 into the uploaded tanh(gate) vector
  (2.1's down-proj feeds a gated residual, so z-image's norm-invariance
  trick was unavailable). Two new shaders: `dit_attn_causal.comp` (the
  prefill runs the WMMA kernel bidirectionally over the whole sequence and
  this recomputes the text rows' context causally — tens of rows, once per
  image) and `dit_copy.comp` (offset copies for the KV cache). The cache is
  each block's post-RoPE prefix k/v as fp32 arena rows, restored under the
  fresh target rows each step before a full re-pack (fine at t2i's tiny
  prefix; Q8's reference images want it fp16 + tile-masked, and the
  full-seq re-pack is a Q9 percent). **Gates, per
  `two-row-regime-needs-a-graph-gate`, both regimes against the oracle**:
  teacher-forced (each step fed the oracle's own latents — bounds the
  model) flat at 1.5e-3–8e-3 over all 40 steps at 1024², bound 2e-2;
  free-running drifts linearly at ~1.3e-3/step to 0.050 at 40 steps — a
  perturbed ODE diverging, the phenomenon diffusers documents for
  cache-on/off — bounded at 0.12 with the image judgment deferred to Q6's
  eyeball. Not done: tile re-screen at M=16384/K=12288 (running on
  z-image's measured winners uncontested — a Q9 percent), the VAE's GPU
  stage (with Q5's fp16-range constraint), and the negative-control
  breakages the z-image graph carries.
- **Q5 — the VAE in Go. CPU done 2026-09-20** (`qimage/vae`, on
  `zimage/vae`'s Tensor/Conv2D — whose Pad/PadEnd/Stride already carried the
  identical downsampler shape): decoder and encoder, stagewise against
  `out/qi21vae`, RGBA round trip included. The T=1 semantics are explicit in
  the package doc: causal 3D convs fold to 2D, time_convs never run,
  AvgDown3D *means in a zero frame* on temporal blocks, DupUp3D keeps only
  the last temporal copy. Two measured bounds: conv stages at 2e-4 (they run
  at ≤5e-5), and `normTol` 8e-3 for the stages where the model amplifies
  fp32 noise — the decoder past its per-pixel L2 tail norm (the dumped fp32
  reference is itself rel 1.3e-3 from a float64 decode there; our decoded
  image sits max abs 8.7e-4 from the dump, under a quarter of one 8-bit
  step) and the encoder's deep middle (the reference's own thread-order
  noise jumps 47x at down_blocks.3; we track ~5x that and land at 4.7e-5 on
  the posterior mode).
  **The end-to-end image gate Q3 owed is closed**: `TestEndToEndImage` runs
  prompt → image through our tokenizer, text encoder, DiT, scheduler and
  decoder — only the noise is the oracle's — and matches the 256²/4-step
  reference image at **max abs 9.1e-4, mean 6e-6** in [0, 1] space, 24x
  inside the 2.2e-2 precedent. (~65 GB fp32 passes through, sequenced to a
  ~34 GB peak; it must run alone.)
- **Q5g — the decoder on the GPU. Done 2026-09-20** (`qimage/vae/gpu.go`,
  ~700 lines): **1024² decodes in 7.4 s** — two runs, 7.39 / 7.49 s; an
  earlier pair spanned 7.36–7.77 — over 117 dispatches, and 256² in 0.38 s
  against the CPU port's ~20 s. Gated stagewise against
  `out/qi21vae` on **vae_test.go's own bounds, unchanged** — conv_in through
  every up block at rel 1e-5–4e-5, the image at **max abs 7.3e-4** (s256)
  and 7.8e-4 (192x320), both under the CPU port's own 8.7e-4 — plus a direct
  GPU-vs-CPU decode at **max abs 4.9e-4** and a negative control (every
  DupUp temporal factor set to 1) caught at **18443x the bound**.
  The port is small because z-image's decoder graph and six of its kernels
  carry over unchanged; three things in this VAE are not in that one, and
  each is one shader: `qvae_chnorm` (the per-pixel channel L2 norm, with the
  SiLU that always follows it fused in), `qvae_dupup` (the up blocks'
  parameter-free shortcut, whose channel mapping still depends on a temporal
  factor with no axis left), and a `-DDIM=1152` build of the scalar
  attention, whose query row and context live in shared memory. The mid
  block's projections are `to_qkv`'s three row blocks read at offsets — a
  1x1 convolution is a linear over the pixel rows — so there is no qkv
  convolution in the dispatch list.
  **Q0's fp16 constraint turned out to bind one convolution, not the
  graph** (`TestConvInputRanges`, the permanent instrument): the channel
  norm in front of every resnet convolution bounds what it reads, so **43 of
  44 convolutions peak below 1.1e4** and only the `288→144` 1x1 shortcut in
  up_blocks.4 — reading the raw residual at **absmax 2.6e5** — is outside
  fp16. That is what prices Q9's matrix-core port, below.
  **The encoder stayed on the CPU here**: nothing t2i serves needs it, and
  its stride-2 asymmetric-pad downsampler looked like a kernel the decoder's
  graph does not have. It went to the GPU in Q8.3c, which is the first thing
  that wanted it — and the downsampler turned out to need no kernel at all,
  because z-image's subsampling identity covers it.
- **Q6 — serve t2i. Done 2026-09-20**, except the Z-Image deletion it owes
  (below). `qimage/pipeline`'s `Pipeline` is the resident three-model graph —
  Q1's text encoder, Q4's DiT, Q5g's decoder — and everything between them is
  host arithmetic per request, so **a size is a parameter and only the
  ceiling is residency**. `backend.Image` re-wires `api.ImageBackend` onto it,
  `cmd/serve -image` loads it, and `cmd/qimage` is the CLI (cmd/zimage's
  replacement).
  **The first honest served wall clock, 1024²/40, two runs**: **1m38.2s /
  1m41.3s** at **31.7 GB resident** (13.2 encoder + 13.3 transformer + 1.0
  VAE + 5.0 activations), staged in 28 s. By stage: text encoder 107/105 ms,
  DiT prefill 2.275/2.285 s, 39 cached steps mean 2.261/2.344 s, VAE decode
  7.59/7.47 s. **An image is the transformer and almost nothing else**: 92%
  the steps, 7.6% the VAE, 0.1% the text encoder.
  **Gate** (`TestServedOracle256`), both regimes as the rule asks: fed the
  oracle's own final latents, our decoder lands the image at **max abs
  2.0e-5**; run free from the oracle's noise — the served path exactly, fp16
  through 36 encoder layers and 32 DiT blocks — the image is **max abs
  0.0333, mean 3.4e-4** from the fp32 reference, which is 1.5x Z-Image's
  *fp32* end-to-end precedent on a path that narrows everything. Latents
  track at rel 0.0088 over the four steps. **The free-run eyeball Q4
  deferred is passed**: `out/qi21served/served1024_40steps.png` is a
  photorealistic fox.
  **The step sweep settles the default at 40, and finds the failure mode is
  the prompt, not the count** (`TestStepSweep`, same seed, three prompt
  kinds, PNGs in `out/qi21served/`):

  | steps | wall | photographic | painterly | structured/technical |
  |---|---|---|---|---|
  | 40 | 1m38–1m42 | good | good | good |
  | 24 | 1m4 | good | good | good |
  | 16 | 44–45 s | good | good | **washed out, structure fragmenting** |
  | 12 | 35–36 s | good | good | **broken — the mechanism comes apart** |

  A fox photograph and an impasto harbour are convincing at **12 steps**, at
  36 % of the cost. A bicycle drivetrain diagram is coherent at 24 and has
  disintegrated by 12 — floating parts, ghosted tubes, contrast collapsing
  toward white. So **the served default stays 40**, the checkpoint's own,
  because it is the only count safe across prompt kinds; **24 is the honest
  fast setting** (−35 %, no visible loss on any of the three) and belongs in
  the docs as such; 16 and below are a per-prompt choice the `steps` field
  already exposes. The pixel distances the sweep also reports (mean
  0.025–0.068 against the 40-step image) are **not** the judgement and say
  so in the test: fewer steps integrate a different trajectory rather than a
  worse one, so the number moves even where the eye sees no loss.
  One thing no step count fixed: the diagram prompt asked for "labelled
  parts" and got none at 40 either. Prompt adherence on rendered text is
  untested territory — Q1 priced an fp32-activation GEMM arm for the encoder
  against exactly this symptom and did not build it; if text turns out to
  matter, that is the known lever.

  **RGBA is served.** `background: "transparent"` is OpenAI's field and it is
  answerable because the VAE is natively 4-channel — but it is not a mode:
  the server prepends and appends the model card's own recommended phrasing
  and keeps the alpha plane, while `opaque`/`auto` composite over white.
  Transparent + JPEG is a 400 naming both fields.
  **Two capabilities regressed with the model.** `stream: true` was closed
  the same day by Q7 (below) and previews are now unconditional. Editing
  (Q8) is still refused, naming the stage: `-edits` is a startup *error*
  rather than a flag that quietly does nothing.
  **The ceiling is a number, and it is not 2048²**: the VAE's activation
  arena is one storage buffer, this device caps one at
  `maxStorageBufferRange` = 4 GiB − 4, and the decode needs **3060 MB at
  1024², 4090 MB at 1184² and 4315 MB at 1216²** (`TestArenaCeiling`). So
  **1184² is the largest square this decoder decodes**, against a model whose
  own README examples are 2048². `pipeline.New` refuses a larger ceiling with
  the arithmetic *before* staging 27 GB (`TestGeometryCeiling`). Closing it is
  a capability, not a percent, and there are two routes: tiled decode (what
  diffusers' `enable_tiling` does, and the up blocks are local enough for it)
  or splitting the arena across several buffers bound as a descriptor array —
  `vk.PipelineSpec.Counts` already exists for exactly this, and the LLM's
  77 GB bank is the precedent.
  **Amended 2026-09-21 (Q10): that ceiling is an *area*, and shipping it as a
  pair of side limits was costing every non-square request 44% of its
  pixels** — see the stage below.
  **The Z-Image deletion is half done, on purpose.** Gone: `zimage/dit`,
  `zimage/pipeline`, `cmd/zimage`, `cmd/ditstack`, `cmd/ditbench`,
  `cmd/ditblock` — nothing else imported them and `qimage/` replaces all of
  it. **Kept for now: `zimage/vae`** (decoder, encoder, both GPU paths,
  `tiny*`) and `cmd/vaebench`/`vaedecode`/`vaeprof` with it, because
  `gpu_conv.go` is the *validated test bed* for the one kernel Q9 wants —
  `TestGPUConvMatchesScalar` proves the matrix-core convolution against a
  scalar oracle, and deleting it before Q9 has taken the packing helpers
  across would throw away the proof and keep the problem. It goes when Q9
  lands. `zimage/qwen` and `zimage/tokenizer` stay permanently (`embed`,
  `llm`, `parakeet` and `qimage` all import them).
  **The 62 GB of weights on disk are not deleted** — `models/Z-Image-Turbo`
  and `models/taef1`. Nothing in the tree reads them now, so it is one
  command whenever you want the space back, but it is not a change worth
  making on a machine's behalf.
- **Q7 — previews. Done 2026-09-20** (`qimage/pipeline/preview.go`,
  `preview_matrix.go`, `cmd/previewfit`). `stream: true` and
  `partial_images` answer again, and the regression Q6 recorded is closed.
  **The decoder is one 64x4 matrix and a bias** — 260 float32s compiled in,
  least-squares fitted. A preview decode measures **159 µs against a 587 ms
  step (0.03%)**, and three partials over HTTP cost **0.3%** of the request
  (10.27 s against 10.24 at 512²/16) — against Z-Image's 4.4% through taef1.
  That is why **previews are not behind a flag any more**: `-preview` and
  `-previews` are deleted, not repurposed. Under Z-Image they bought 1.0 GB
  of activation arena; here there is nothing to load.
  **The fit** (`cmd/previewfit`) generates its own pairs rather than reading
  Q0's two: it runs the served pipeline over 24 prompts chosen to span
  colour and content — primaries, skin, foliage, night, snow, flat graphic
  colour, neutral greys — at 512²/8 steps, box-filters each 16x16 image
  block onto the latent pixel that produced it, and solves one 65x65 normal
  system with four right-hand sides. 24576 samples for 260 parameters.
  The shipped matrix is fit on all 24 prompts and scores **R² 0.969–0.974
  on RGB** (rms 0.098–0.114 in [-1, 1], 12–15 of 255 8-bit levels).
  **`-report` is the honest number**: holding the last six prompts out
  entirely gives **R² 0.83–0.90 on unseen content** (rms 0.176–0.198). The
  gap is real and is what a 260-parameter linear map costs — six held-out
  prompts are six palettes it never saw — and it is the right order for a
  latent2rgb preview, whose job is composition and colour rather than
  fidelity. Alpha is near-constant on opaque prompts, so its R² (0.63 train,
  0.26 held out) over an rms of 0.0014 means nothing; it is reported rather
  than hidden.
  **The tensor decoded is the denoised estimate x0, not the sample** — the
  schedule is flow matching, so ten steps into forty the sample is still
  three quarters noise. `Run` forms it after the Euler move, where the
  identity is x0 = x_{t+1} − σ_next·v and needs no copy of the pre-step
  latents. `TestPreviewFrames` carries that as a **negative control**:
  decoding the sample instead is 2.5x further from the finished image at
  step 0 (0.394 against 0.157), and the two are asserted *identical* at the
  last step, where the terminal sigma makes them the same quantity.
  What a preview cannot do is resolve what the latent does not carry
  per-pixel: it is 1/16 scale with no texture, upscaled by whatever displays
  it. What it carries is composition, colour and layout — at 512²/16 the
  first frame lands at **2.27 s against a 10.25 s image**.
- **Q8 — edits. Done 2026-09-20**, in six sub-stages: the reference and the
  vision tower, the edit text encoding, the CPU pipeline with its resampler
  and GPU VAE encoder and GPU tower, the GPU edit text encoding, the GPU DiT
  edit path, and the served pipeline with its endpoint. The
  stage is (a) VAE encoder serving reference latents, (b) the vision tower,
  (c) the multi-image template, mrope with real 3-D positions and deepstack
  injection, (d) the pipeline and the endpoint. `/v1/images/edits` drops
  `strength` and gains multi-ref.
  **Less of it was new than the plan assumed.** The DiT side — per-reference
  RoPE blocks, the segments, the prefix cache across refs — was already
  ported *and gated* by Q2 against the synthetic edit case in
  `out/qi21dit`/`out/qi21block` (≤4e-5, both regimes), and `dit.NewLayout`
  already takes a list of image shapes. Q5 built the VAE encoder. So (a) and
  (c)'s DiT half were done before this stage opened.
  **Q8.0 — the reference. Done** (`reference/dump_qi21_vision.py`,
  `out/qi21vision`): 50 tensors over two deterministic RGBA cards, a 256²
  one and a 192x384 one, two runs byte-identical. It walks the processor's patching, the interpolated
  position grid (indices *and* weights, as the model computed them), the
  vision rope table, every block the deepstack taps, the three deepstack
  features, the merger, the 3-D mrope position ids, and the pipeline's own
  `prompt_embeds` and `image_pad_mask`. It also pins the fact Q8 rests on,
  at gap **exactly 0**: the merger's rows are what the encoder scatters into
  the `<|image_pad|>` slots, and the DiT then overwrites those rows with VAE
  latents — so vision reaches the transformer *only* through what the text
  tokens absorbed from them. The tower conditions the text; the VAE
  conditions the pixels.
  **Q8.1 — the vision tower on CPU. Done** (`qimage/vision`, ~600 lines):
  all 27 blocks against the dump at **rel ≤ 1.1e-4**, the three deepstack
  features at ≤9.5e-5, the merged rows at 8.9e-5, and patchify / position
  grid / rope table exact to fp32 rounding (1e-7–5e-6). Four things here are
  not what a plain ViT does, and each is a **negative control that fires**:
  token order is 2x2-block-major rather than raster (raster is caught at
  **40717x** the bound); the block MLP's GELU is the tanh approximation
  while both mergers use the exact erf one (swapping them, **221x**); the
  rotation is NeoX halves and not the adjacent-pair convention
  `qimage/dit` uses one import away (**3796x**); and the output merger
  normalizes each 1152-wide row *before* concatenating four while the three
  deepstack mergers normalize the 4608-wide concatenation after — which the
  checkpoint's own norm widths refuse at load rather than answering wrongly.
  **Q8.2 — the edit text encoding on CPU. Done 2026-09-20**
  (`qimage/textenc/edit.go`, ~340 lines, plus one seam in `zimage/qwen`:
  `ForwardEmbeds` runs the layers over caller-supplied embeddings, a
  caller-supplied rotary table and a per-layer hook, and `Forward` is now
  that function with a token lookup and a plain 0..T-1 table in front of it —
  `embed`, `llm` and `parakeet` are untouched). Three mechanisms, each
  gated:
  - **the template**, with each image's `<|image_pad|>` expanded to one token
    per merged 2x2 group before tokenizing. `<image1>` is *not* a special
    token — it is four ordinary BPE ids — and the leading space before
    `<image2>` tokenizes, so both are rendered exactly.
  - **3-D positions**, transformers' `get_rope_index`: text runs advance all
    three axes off a shared counter, an image block lays its merged grid out
    in (t, h, w) raster order at the counter, and the counter then advances
    by the grid's **span**, `max(H, W)/merge` — 8 positions for a 64-token
    square image, 12 for a 72-token 192x384 one. Then the **interleaved**
    mrope table: frequency j reads h when j%3 == 1 and w when j%3 == 2 below
    3*section, t everywhere else.
  - **deepstack injection**: the merger's rows scatter into the pad slots
    before layer 0, and the three features are added at those slots after
    layers 0, 1 and 2. Several images are one stacked tensor in pad order,
    not a loop.

  **Gates.** Without weights, exact: the ids, the pad slots and all three
  position axes for both the one- and two-image sequences, and the rotary
  tables at 2.7e-6 / 4.2e-6 against the ones the model built for itself.
  With weights, against the pipeline's own `prompt_embeds`: **one image —
  layer 0 at 1.4e-4, the pre-norm state 3.3e-4, `prompt_embeds` 4.3e-4**;
  **two images — the scattered `inputs_embeds` exactly (rel 0, max abs 0),
  layer 0 at 1.1e-5, the pre-norm state 4.1e-4, `prompt_embeds` 4.8e-4**,
  all against Q1's measured `deepTol` of 1e-3. Controls: plain RoPE instead
  of mrope **13315x** the bound, no deepstack injection **6057x**, and the
  conditions in the wrong order refused at the row count rather than
  answered.

  **The dump grew a two-image case** — the 256² card plus a 192x384 one with
  a different pattern, 177 tokens and 136 slots — because one image hides
  the leading space, the span arithmetic and the cross-image stacking, and
  the endpoint promises ten. Two dump bugs came out of it, both in the
  reference rather than the port: the new section was missing the
  final-norm neutralization hook the single-image one installs (so its
  "prenorm" came back *normalized*, absmax 142 against `prompt_embeds`'
  661.9), and the fp32 tensors turned out not to be the right oracle for the
  wide card at all — see below.

  **The method finding, a third time: measure the reference against itself.**
  The two-image gate first failed at rel 1.3e-2, with the worst element
  inside the second image's rows. Everything structural was already
  excluded — the pixel rows, the patch embedding and block 0 match to fp32
  rounding, and every geometric piece feeds them — and the error *grew with
  depth* (3.3e-5 at layer 8, 1.1e-4 at 16, 3.5e-4 at 24, 1.0e-3 at the last
  hidden state). Packing was not the cause: the reference is **bit-identical**
  running the two images batched or alone. What settled it was Q5's arbiter,
  the same tower in **float64**: the dumped **fp32 reference is itself rel
  8.9e-4 (last hidden) and 1.4e-3 (merged) from float64 on the wide card,
  against 1.0e-4 and 1.2e-4 on the square one** — the same code on a picture
  that amplifies rounding ten times harder. Against the float64 rows this
  port lands at **1.3e-4, 2.1e-4 and 4.8e-5, six to seven times closer than
  the fp32 reference**, because `qimage/vision` accumulates dot products in
  float64 while holding activations in float32. So the wide card's deep
  tower stages are gated against **dumped float64 rows at 5e-4**
  (`towerTol`), and the edit text encoding is gated with the *dump's own*
  condition rows, which is what makes it a gate on the text encoding rather
  than on the pair. The composed run — our tower feeding our encoder, which
  is what a served edit does — sits at **rel 1.3e-2 from a 1.6e-3 difference
  in the tower's merged rows**, reported with that attribution and guarded
  at an order of magnitude, not presented as a precision claim.

  **Q8.3 — the edit pipeline on CPU. Done 2026-09-20**
  (`qimage/pipeline/edit.go`, `dit.LayoutFromPadMask`, and the oracle
  `reference/dump_qi21_edit.py` → `out/qi21edit`). The oracle is the t2i
  dump's deliberate twin — same size, same steps, same seed, noise passed in
  packed — with `output_resolution` set to the card's own size so the
  pipeline's condition resize is the *identity* and the gate is about the
  pipeline rather than a resampler. Two runs byte-identical over all 14
  tensors.
  Three pieces were new and each is small, because an edit reuses the t2i
  graph: **`LayoutFromPadMask`** turns the text encoder's own image-pad mask
  into the joint sequence's geometry (runs of true are the condition images,
  what is between them is text) and *generalises* `NewLayout` rather than
  sitting beside it — a t2i prompt has no slots and the two agree, which the
  test asserts so the edit path cannot become a second layout implementation;
  **`EncodeCondition`** runs the VAE encoder, takes the posterior mode,
  normalises per channel and packs; and **`CalcDimensions`**, diffusers'
  `calculate_dimensions`, whose rounding is Python's break-to-even and not
  away-from-zero.
  **The gate** (`TestEndToEndEdit`, the edit twin of `TestEndToEndImage`):
  condition image + prompt through our tokenizer, vision tower, text
  encoder, VAE encoder, DiT, scheduler and decoder, only the noise from the
  oracle — **prompt_embeds 4.7e-4, condition latents 6.5e-5, per-step
  latents 1.4e-4 → 4.4e-5, and the edited image at max abs 1.0e-5, mean
  2e-6** against the 2.2e-2 precedent. That is 90x tighter than the t2i
  gate's own image number, and the reason is the prefix: an edit's
  trajectory is pinned by 277 rows of conditioning where a generation's is
  free.
  **The finding, and it is a sharp one: the condition image must be
  quantized exactly like the reference's.** The vision copy is composited
  over white by PIL, which works on **uint8**, so every composited pixel
  lands on a level. Doing that arithmetic in float instead leaves 97% of the
  pixels half a level away — and the tower turns that into **rel 3.49 on the
  merged rows and rel 11.1 on the prompt embedding**, which is how this port
  first failed. It is now the stage's negative control, firing at 11115x the
  bound. Read together with the float64 measurement above, the tower
  amplifies an input perturbation by roughly **10³**, and the consequence is
  a requirement on the resampler, which is why it is ported exactly rather
  than approximately (below). None of this means an edit is fragile — a
  differently-quantized condition gives a different *valid* edit, the same
  way cache-on and cache-off do — but it does mean parity is only measurable
  against the same pixels.

  **Q8.3b — the resampler. Done 2026-09-20** (`qimage/pipeline/resize.go`,
  ~200 lines). It is **Lanczos, not bicubic**: `VaeImageProcessor`'s default
  `resample` is `lanczos` and the pipeline constructs it without overriding
  that, so every condition image is resized by
  `PIL.Image.resize(resample=LANCZOS)`. The port reproduces Pillow's
  arithmetic rather than its intent — 22-bit fixed-point coefficients with
  its away-from-zero rounding, an int32 accumulator with its rounding term,
  its clip-and-shift, two passes with a **uint8 intermediate**, and its skip
  of a pass whose axis does not change — and the gate is **exact 8-bit
  equality**, not a tolerance, because at 10³ amplification "close" is not a
  claim worth making.
  Two things had to be got right that reading the filter would not tell you.
  The filter is stretched by the **downscale** factor only, so downscaling
  antialiases and upscaling does not; and **an RGBA image is not resampled
  as four independent channels** — `Image.resize` converts it to `RGBa`
  (premultiplied), resamples, and converts back, both conversions 8-bit and
  lossy. Missing the premultiply left 28% of samples up to four levels out,
  which is how the port first failed. It now matches on all three
  behaviours: **500x333 → 320x224 (down), 128x96 → 288x224 (up), and
  320x224 → 320x112 (one axis)**, every sample of all three exact.

  **Q8.3c — the VAE encoder on the GPU. Done 2026-09-20**
  (`qimage/vae/gpu_encoder.go`, ~300 lines; `qvae_avgdown.comp` and a
  `-DDIM=768` attention build are the only new shader work). **A 1024²
  condition image encodes in 1.65–1.72 s** over 94 dispatches and 1648 MB of
  activations, against a CPU encoder that costs minutes at that size — the
  same argument that moved the decoder in Q5g, now paid per reference image
  per request.
  It is a small file because the encoder is the decoder's graph read
  backwards over the same kernels, and the one thing that looked like new
  work was not: **the stride-2 downsampler needs no kernel**. z-image's
  encoder established that a 3x3 stride-2 convolution over a (0, 1, 0, 1)
  pad computes exactly what the stride-1 pad-1 form computes at the *odd*
  pixels; Qwen's downsampler has the identical shape, so `vae_downsample2x`
  — which keeps (2h+1, 2w+1) — is reused unmodified and the ordinary
  convolution does the work, at four times the arithmetic and no second
  shader. Genuinely new: `qvae_avgdown`, the mirror of `qvae_dupup`, whose
  temporal factor at T = 1 still averages in a **zero frame** and still
  divides by the full group; and a second build of the attention kernel,
  because the encoder's mid block is **768** wide against the decoder's 1152
  (base_dim 96 against decoder_base_dim 144) and DIM is a shared-array
  extent rather than a push constant.
  The decoder and encoder now share an `engine` — buffers, pipelines,
  weights, arena — the same split `zimage/vae` made between its own two
  directions, so the second graph added no Vulkan lifecycle at all.
  **Gates**, on `vae_test.go`'s own bounds unchanged: every stage of both
  dump cases (conv_in, all five down blocks, the mid block, the fused tail
  norm, conv_out) and the posterior mode raw and normalised — **conv_in at
  1.1e-6 through to the mode at 2.6e-5** — plus a direct GPU-vs-CPU encode
  at **4.6e-5**. Two controls fire: the AvgDown temporal factor at
  **12173x** the bound (measured on the first block that *has* one — the
  checkpoint's `temperal_downsample` is [false, true, true, true], so
  breaking it and looking at block 0 shows nothing, which is how the control
  first passed for the wrong reason), and a symmetric-pad stride-2
  downsampler refused at construction rather than answered wrongly.

  **Q8.3d — the vision tower on the GPU. Done 2026-09-20**
  (`qimage/vision/gpu.go`, ~600 lines; `qvit_bias_act.comp` and
  `qvit_rope.comp` are the only new shaders). **A 1024² condition image runs
  the 27 blocks in 6.27 s** — two runs 6.283/6.265 — against roughly twenty
  minutes on the CPU port, and about 6% of an edit's total.
  **The precision was measured before it was chosen**, which is the part
  worth keeping. This tower sits between the tree's two conventions: the VAE
  is fp32 because its activations reach 1e5 against half's 65504, the DiT
  and text encoder are fp16 on the matrix cores because that is where their
  speed is, and this one's residual stream reaches absmax 1.3e4 — inside
  half's range but with three digits left — while Q8.2 measured the tower
  amplifying perturbations by ~1600x. So `TestFP16Ladder` runs the *CPU
  oracle* with every linear taking fp16 operands and accumulating in fp32,
  which is what a matrix core does, and reports **rel 0.135 at the last
  hidden state, 0.14 on the merged rows, no overflow** — sub-linear growth
  from 3.6e-3 at block 0. That is what said the matrix cores are usable, and
  it is the bound the port is gated at. The device then landed **within a
  few percent of the prediction at every stage** (merged 0.127 against 0.14,
  deepstack 0.016/0.031/0.050 against 0.015/0.035/0.053), which is better
  evidence of a correct port than the bound itself: a structural mistake in
  this tower moves these tensors by rel 0.76 to 8.1, per the four controls
  `qimage/vision` already carries.
  Almost every kernel is the DiT's, and three of the reuses are the reason
  the file is small. **The affine LayerNorm is `dit_final_norm` plus
  algebra**: that kernel computes LN(x)·scale, the ViT wants LN(x)·γ + β
  feeding W(·) + b, and W(LN(x)·γ + β) + b = W(LN(x)·γ) + (W·β + b) — so β
  is folded into the next projection's bias at load and the kernel is used
  verbatim. **The residual add is `dit_gate_add` with its gate dropped**,
  the path the DiT's two unmodulated blocks already take. **The patch
  merger's concatenation is free**: four consecutive rows of 1152 in a
  row-major buffer *are* one row of 4608, so the output merger normalises
  with a tight row stride and the GEMM reads the same bytes four rows at a
  time, while the three deepstack mergers normalise the 4608 directly.
  Attention is the **scalar** kernel, which takes its head width from a push
  constant and runs this tower's 72 unchanged; the matrix-core one is built
  per head dim and 72 does not divide its 16-wide tile. Padding to 80 would
  work and is a **percent, not a capability** — the 6.27 s says whether it is
  worth it, and at 6% of an edit it is not yet. The MLP's 4304 is padded to
  4352 for the same tile reason, with zeros that GELU fixes at zero and the
  second layer multiplies by zero rows.

  **Q8.4 — the edit text encoding on the GPU. Done 2026-09-20**
  (`qimage/textenc/edit.go`'s `EncodeGPU`, ~110 lines, plus four seams in
  `zimage/qwen`'s `GPUEncoder`). Q8.2's CPU path is minutes at a condition
  image's token count; this is **~0.12 s** at the dump's 99 and 177 rows, and
  it is what makes a served edit possible at all.
  **The transformer did not change.** The three mechanisms each landed on one
  seam, and the seams are the device counterparts of the three `ForwardEmbeds`
  grew in Q8.2: `Embeddings`+`UploadEmbeds` (the scatter is a host edit of
  the lookup, because the ids do not determine the rows), `SetRoPE` (the
  mrope table is uploaded rather than built as 0..T-1), and
  `RunHooked`+`AddRows` (the deepstack features are added at the pad slots
  between layers — one dispatch per image per layer, on `dit_gate_add` with
  its gate switched off, staged through the attention branch's own dead
  output tensor). `upload` is now `Embeddings` followed by `UploadEmbeds` and
  the plain table, so the t2i path runs the same code: `TestGPUEncoder`
  re-measures Q1's en/cjk/empty at **0.024 / 0.027 / 0.21, unchanged**.
  **Gates**, against `out/qi21vision` with the dump's *own* tower rows as the
  conditions (which makes it a gate on the text encoding rather than on the
  pair, exactly as Q8.2's two-image case argued): **one image 0.059, two
  images 0.064**, the pad mask exact, and layer 0 at **1.1e-3** against a
  4x bound — the shallow reading is what keeps a mistake in the three new
  mechanisms from hiding under the fp16 one. Controls: plain RoPE instead of
  mrope **133x** the bound, no deepstack injection **61x**, the conditions
  swapped refused at the row count. Two runs identical.
  **The bound is measured and it is 2.4x the t2i path's, and that is not a
  bug**: `TestGPUEditLadder` (the permanent instrument, `QI21_LADDER=1`,
  ~50 GB) walks the same run against the CPU port layer by layer and finds
  **Q1's mechanism unchanged and in the same place** — flat at ~1.1e-3 through
  layers 0-5, the massive-activation channel appearing at layer 6 (rms
  0.98 → 21.1) carrying an absolute error of 4.46 *flat*, a second jump at
  layer 16 whose 8.83 stays 8.83 **in the same column for 18 layers**, and
  then the cancellation in layers 34-35 surfacing it: 0.0049 → 0.0146 →
  0.0532. The last two layers are the whole of the deviation. And the
  *absolute* error does not move between the two paths at all — the worst
  element is **2.38 for the t2i prompt and for both edit prompts**, measured
  in the same hour; what the 2.4x buys is a different worst relative element
  under a lower rms floor. The official bf16 pipeline is rel 2.0 on this
  checkpoint, 30x further out.

  **Q8.5 — the DiT's edit path on the GPU. Done 2026-09-20**
  (`qimage/dit/gpu.go`, ~120 lines changed; `dit_kvcache.comp` and a rewrite
  of `dit_attn_causal.comp` are the shader work). **An edit's transformer at
  the served shape — a 1024² target on one 1024² reference — is prefill
  5.47/5.52 s and cached steps 2.70/2.71 s, so 40 steps is 110.7/111.3 s**
  against a generation's 90.7. That is the "slightly more expensive than a
  generation" this file predicted, now measured: +22%, all of it the prefix
  in every step's attention.
  Four things had to change, and three of them are about the prefix being
  thousands of rows instead of forty:
  - **The condition latents go through `img_in`.** They are latents in the
    same sequence, so they take the target's own projection — one GEMM per
    condition block off the same narrowed `aLat`, issued in ascending row
    order because a GEMM writes whole 128-row tiles and overruns a short
    block. The projected *text* rows then land by copy dispatch rather than
    by host write, since a host write cannot be ordered after a dispatch
    that has not been submitted.
  - **The block-causal mask is repaired segment by segment, last segment
    first.** Every row attends to a contiguous key prefix — its own index
    for a text token, its image block's last index for an image token — so
    the matrix-core pass still runs bidirectionally over the whole sequence
    (right for the target) and each prefix segment is then recomputed with
    its own key count. Backwards, because the WMMA kernel's query block
    starts at row 0 and an image segment's pass necessarily rewrites every
    row before it; running in reverse makes each row's last writer its own
    segment. It costs about a quarter again of the prefill's attention and
    needs no new matrix-core kernel.
  - **`dit_attn_causal.comp` became an online softmax** over chunks of keys,
    with a query-row offset and an optional fixed key count. That lifts the
    512-key cap it carried (a text run in the middle of an edit's prefix
    attends over the condition image before it) and costs nothing at t2i:
    for a prefix inside one chunk the arithmetic order is unchanged, and
    `TestGPUOracle256` comes back **bit-identical** (0.001503 / 0.00148
    before and after).
  - **The prefix KV cache moved into fp16 banks of its own**
    (`dit_kvcache.comp`). 32 blocks x 2 x 4136 rows x 4096 is **4.34 GB of
    fp32**, past this device's 4 GiB single-buffer cap; as halves it is
    **2.07 GB**, measured. The narrowing is *free*, not a trade: the pack
    narrows those rows to halves before attention reads them either way, so
    storing `fp32(fp16(x))` and packing it gives the same half.

  **One real hazard came out of this, and it was worth the 4.9x**
  (`TAIL_MAX` in `dit_attention_wmma.comp`). The kernel masks keys past
  `pc.tokens` out of P but not out of the row *max*, which is sound in every
  existing caller because the packed plane is zeroed past the sequence — a
  key that does not exist scores 0 and can only set the scale on a row whose
  every real score is negative. The edit prefill's image-segment repair
  breaks that premise: it runs with the key count cut to a block's end, in
  the middle of a plane whose later rows are real and large. The softmax is
  shift-invariant so the answer stays right, but **P is stored in fp16**, and
  a max set by a key outside the mask shifts every real weight down before it
  is narrowed. A build with the mask reaching the max took the prefill step
  from **rel 0.0154 to 0.00314**. The mask has to reach the max to keep the
  precision, not to keep the answer.
  **Gate** (`TestGPUEditOracle`, against `out/qi21edit`, both regimes as the
  rule asks, the prompt embedding / condition latents / noise all the
  oracle's own so it is a gate on the transformer alone): teacher-forced
  **3.1e-3 at the prefill and 2.3e-4–1.4e-3 after it**, free-running
  **2.3e-3 over four steps**, against Q4's own bounds unchanged — an edit
  needed no bound of its own, because the arithmetic per row is identical and
  only the row count changed. Two runs identical. Three controls fire:
  condition latents zeroed **35x** the bound, the pad mask ignored so the
  condition never enters the sequence **54x**, and the block-causal repair
  switched off — the mask being wrong while every shape checks out —
  **29x**. And t2i is untouched, re-measured the same hour: 1024²/40 at
  **prefill 2.29 s, cached mean 2.27 s, 1m30.7s total**.

  **Q8.6 — the edit pipeline and the endpoint. Done 2026-09-20**
  (`qimage/pipeline/edit.go`'s `Pipeline.Edit`, `backend.Image`, and
  `/v1/images/edits`). **A served edit at 1024² with one reference image is
  2m7.6s / 2m7.9s at 39.4 GB resident** — against a generation's 1m38s, which
  is the "slightly more expensive than a generation" this file predicted
  before any of it was built. By stage: the reference's own encoding 7.8 s
  (the vision tower 6.3, the VAE encoder 1.7), text encoding 0.73 s, prefill
  5.49 s, 40 cached steps at 2.70 s, decode 7.5 s.
  **The one thing that needed new infrastructure was the tower's sizing.**
  `vision.GPU` fixed its patch grid at construction, and a served edit hands
  it a different grid every request — `calculate_dimensions` fixes a
  reference's *area* and lets its sides follow its aspect ratio. It is now
  staged for a **patch budget** and takes the grid at `Forward`, which is the
  same split the VAE's graphs already made; `TestGPUTower` runs two
  differently-shaped cards through one staging and re-runs the first at
  **rel 0** to catch state left behind. Everything else composed: the VAE's
  GPU encoder already re-planned per image, and `Pipeline.Edit` is the eight
  stages the CPU gate composes with the device's versions substituted.
  **A second bound came out of the tower's sizing, and it is a finding.**
  `fp16Tol` was 0.25, measured by `TestFP16Ladder` on the square card. Run on
  the **wide** card the same ladder gives **0.297 / 0.438** — three times as
  much from the same weights and the same code — and the device lands at
  0.562 against that 0.438 prediction, the same agreement the square card
  shows (0.127 against 0.14). That is the fp16 face of what Q8.2 measured in
  fp32 (the dumped reference is itself 8.9e-4 from float64 on this card
  against 1.0e-4 on the square one): **some pictures amplify rounding an
  order of magnitude harder than others**, and a tower gated on one of them
  would be wrong about the other. Each card is now gated at its own
  measurement.
  **Gate** (`TestServedEditOracle256`, the edit twin of Q6's
  `TestServedOracle256`, both regimes): the whole request — the reference
  resized by our Lanczos port, composited over white on the 8-bit levels,
  through the GPU tower and the GPU VAE encoder, positioned by our mrope,
  injected into the GPU text encoder, laid out from the pad mask, denoised
  over a real prefix and decoded, only the noise the oracle's — reproduces
  the reference edit at **max abs 0.0014, mean 1.7e-4**. That is **24x
  tighter than the served t2i path's 0.0333**, for the reason Q8.3 found on
  the CPU: an edit's trajectory is pinned by its prefix where a generation's
  is free. Teacher-forced, the oracle's own latents decode at max abs
  0.00000. Controls: a blank white reference lands **366x** further out
  (measured against the run, not the bound), and two references on a
  one-reference server and an edit with no reference are both refusals.
  **The geometry needed one deviation from the reference, and it is named**
  (`targetFor`, `TestTargetFor`): `calculate_dimensions` fixes the *area* and
  lets the sides run, so at the shipped defaults — condition area and ceiling
  both 1024² — a 4:3 reference asks for 1184x896, whose long side is outside
  a 1024x1024 ceiling even though its area is not. Refusing would make every
  non-square edit a 400 out of the box; enlarging the ceiling would be
  residency nobody asked for. So the aspect ratio is kept and the area shrunk
  until both sides fit, and a ceiling above the condition area leaves
  diffusers' own answer untouched.
  **Mostly repaired by Q10** (2026-09-21): against an area ceiling only the
  snap is left to deal with — 1184x896 is 1% past 1024²'s area, not a side
  outside a box — so a 4:3 edit is **1152x896 rather than 1024x768**, a
  panorama 2048x512 rather than 1024x256, and the deviation is now one 32-px
  step instead of a whole shrink.
  **The endpoint's shape changed with the mechanism**, which is what Q6 said
  it would: `-edits` is now `-edits N` (residency — the tower and the VAE
  encoder are 1.4 GB, and each reference is ~2.1 GB of prefix KV cache at
  1024²), `image[]` is a real list, `strength` is a **400 that explains what
  changed** rather than a silently ignored field, `max_reference_images`
  joins the geometry in `GET /v1/models`, and an edit with no `size` follows
  the last reference's aspect ratio at the condition area instead of the
  server's default. Streaming needed no code: `partialSteps(0, …)` was
  already parameterised for an edit that starts at 0, which under 2.1 every
  edit does. Driven over HTTP end to end: a red mug becomes a blue mug on the
  same table with the same grain and the same light, and the streamed form
  sends two 32x32 partials and the finished 512².
- **Q9 — percents. First pass done 2026-09-20: the served image is 6% faster
  and the served edit 7%, bit for bit the same picture.** 1024²/40 goes
  **1m38.2s/1m41.3s → 1m32.3s/1m33.5s**, an edit **2m7.6s/2m7.9s →
  1m58.6s/1m59.2s**, and the DiT's cached step **2.27 s → 2.12–2.14 s**.
  Nothing about the numerics moved: `TestGPUOracle256` and the two served
  oracles return the same digits they did before (0.001503, max abs 0.0333,
  mean 3.4e-4).
  **The attribution came first and is the permanent instrument**
  (`TestGPUStepProfile`, both regimes at the served shape, one dispatch at a
  time). A cached step, before: **GEMMs 69% at 35–40 TFLOP/s**, attention
  10.6% at 37, and 21% in elementwise passes — pack 6.6%, swiglu 3.8%,
  qkpack 3.1%, gate 3.0%, the rest 4.2%. Q4 had run this graph on z-image's
  measured winners *uncontested*, and that is what the profile was for.
  **Two screens, and they disagreed about where the percent was.**
  - *The GEMM swizzle arm re-screen* — IMAGE.md's own hypothesis, since
    z-image chose SWZ=8 at M=16384/K=12288 and this model runs M=4096 —
    **found nothing, and that is the result** (`TestGPUGEMMScreen`, four arms
    on one staging): SWZ=8 wins here too, by 1.4–1.9% over SWZ=4 and 3.9–6.9%
    over the others, twice over. The GEMMs are at this kernel's ceiling and
    the inheritance was right.
  - *The fragment pack* was not a hypothesis anyone had, and it was the
    percent. The profile caught it moving **102 MB in 2.3 ms — 44 GB/s —
    where the SwiGLU dispatch beside it in the same block reaches 194** on
    the same bus. The layout it writes is why the matrix-core attention is
    worth using and was never in question; the *shape of the launch* was.
    One token tile per workgroup is 2048 elements over 256 threads — eight
    each, with two barriers around them — which is a latency-bound shape and
    not a bandwidth-bound one. `TPW` tiles per workgroup took the pack to
    **131 GB/s (3.0x)** and the fused q pack from 69 ms to 20, and the step
    fell **5.9%** (`TestGPUPackScreen`, 1/2/4/8 arms). The screen asserts the
    four arms' outputs **bit-identical** rather than close, because a screen
    that only timed them could pick one that packs wrongly.
  **What is left is at the bus, which is the next finding.** Re-profiled
  after the fix, every remaining elementwise pass measures 190–194 GB/s:
  swiglu 4.0%, gate 3.2%, pack 2.4%, narrow/rope/rmsnorm/ffn/attn 5.5%. None
  of them is slow; there are simply round trips. So the next percents in the
  DiT are *fusions* and are priced from the traffic they remove: the
  attention writing its context straight into the o-projection's A operand
  (z-image's stage-10 `OUT_F16`, the build already exists) is ~1%, and a
  gated-accumulate GEMM epilogue that absorbs both residual adds is ~2% and
  a new kernel. The GEMMs are 73% of what remains and at their ceiling.
  **The bigger fish is elsewhere**: the VAE decode is 7.5 s of a 92 s image
  and 68.8% of it is conv3x3 at 3.2 TFLOP/s, against the 38.3 the
  matrix-core implicit GEMM measured in z-image with kernels already in this
  tree. That is ~6 s, larger than anything left in the transformer, and it
  is priced below.
- **Q9's remaining ledger.** The I3/I5/I8/I9 fusion lesson
  (elementwise passes absorbed into consumers) transfers to a new but
  same-shaped budget; attribute before optimising, same-hour controls, plans
  measured never composed. Two of them are already attributed and priced, by
  `TestGPUDecodeProfile` on a 1024² decode (7.46 s one dispatch at a time):
  - **conv3x3 is 68.8% — 5.13 s at 3.2 TFLOP/s**, which is z-image's stage-8
    starting point to three digits (3.18 s, 3.0–3.2 TFLOP/s). That stage's
    matrix-core implicit GEMM took it to 258 ms and 38.3 TFLOP/s, its
    kernels and packing pass are already in the tree, and the range
    measurement above says **every conv3x3 in this graph can feed it** —
    the one convolution that cannot is a 1x1 shortcut worth a fraction of
    the 3.8% that all four 1x1s cost together, and it simply stays scalar.
    Expected: ~7.4 s → ~2.8 s.
  - **the mid block's four projections are 21.0% — 1.57 s at 35 GFLOP/s** on
    `vae_linear`, the same naive kernel and nearly the same number (62
    GFLOP/s) that z-image's stage 7 replaced with `dit_gemm` unmodified.
    Expected: another ~1.5 s.
  Everything else in the decode is under 4%: attention 250 ms (the latent is
  4096 rows here, not z-image's 16384 — this is *not* the 26% it was there),
  chnorm 1.7% with its SiLU already fused, add 1.0%, the two shuffles 0.4%.

  **What that port has to price first, and it is not in z-image's notes.**
  Every percent taken on 2026-09-20 was **bit-identical** — a launch shape
  and a kernel choice, no arithmetic moved. The conv port is not: the
  matrix-core convolution narrows its *operands* to fp16, and this decoder's
  convolutions read activations at **absmax 1.1e4** where z-image's peaked
  at **497**. Both are inside half's 65504, so neither overflows and the
  measurement that matters is not range but precision: an fp16 operand
  carries ~5e-4 of relative error, and `vae_test.go`'s conv stages are gated
  at **2e-4** with the decoded image at max abs 7.3e-4 against the dump.
  So the port moves the numbers this vertical's tightest gates are written
  against, and the first thing it owes is the same instrument every other
  bound here came from — run the CPU decoder with fp16 conv operands and
  fp32 accumulators, measure what it costs stagewise and on the image, and
  gate the device at *that* rather than at the fp32 bounds it will not hold.
  z-image made the same trade and kept its gates, but 20x less input range
  is 20x less headroom, and assuming it transfers is exactly the composition
  this file's method refuses. The infrastructure is otherwise ready:
  `zimage/vae`'s `hbuf`/`w16buf` engine split, `vae_pack_conv.comp`,
  nine `vae_conv_wmma` builds and `TestGPUConvMatchesScalar` are all in the
  tree and are why `zimage/vae` was kept past its replacement.
- **Q10 — the ceiling is an area, not a box. Done 2026-09-21**, and it came
  out of a served complaint rather than a plan: a 16:9 request against the
  shipped server was answering **1024x576**, 0.59 Mpx, where the same arenas
  hold 1.05. Nothing was broken — `fitRatio` was fitting the ratio inside
  `MaxWidth`/`MaxHeight` exactly as written — the *rule* was wrong, and it
  was wrong because it was inherited.
  **The measurement first** (`qimage/vae`'s `TestArenaShape`, the permanent
  instrument): plan the decode for a dozen shapes and the activation arena is
  **exactly 3060 bytes a pixel for every one of them** — 1024x1024, 1344x768,
  2048x512, 256x4096, 4096x256, to the byte, with no per-axis term at all.
  The DiT's rows are the pixel count over 256 by construction and the text
  encoder does not see the geometry, so **residency in this pipeline is a
  function of area alone**. The side box came from `api.ImageGeometry`'s
  z-image-era doc, which argued the opposite and was *right about z-image*:
  that VAE holds blocked fp16 copies of each convolution's input, padded per
  axis, so 128x512 genuinely cost it more than 256x256. Q5g's decoder keeps
  its activations flat. The constraint was carried across with the comment
  that justified it and never re-measured — the same composition-without-
  measurement this file's method refuses everywhere else.
  **What it cost, and it is the largest single number this vertical has
  moved**: at the shipped 1024² ceiling, `aspect_ratio: "16:9"` goes
  **1024x576 → 1344x768 (1.75x the pixels)**, 3:2 → 1248x832 (1.51x), 21:9 →
  1536x672 (2.42x), and an edit that names no size goes 1024x768 → 1152x896
  on a 4:3 reference. At the hardware's own ceiling (`-image-size 1184x1184`,
  1,403,584 px) 16:9 is **1536x864**, which is 2.25x what the box gave.
  **And it is free**, which is the point: a 1344x768 image is 4032 latent
  tokens against a square's 4096, so it costs *less* than the picture it
  replaces. Measured on the device at 8 steps, one square staging serving all
  four shapes: 1024x1024 27.0 s, **1344x768 26.5 s**, 768x1344 26.4 s,
  2048x512 27.0 s, steps flat at 2.36 s (`TestGeometryShapes`, the graph gate
  the new regime needed — arithmetic could not have settled whether a
  re-planned decode and a non-square RoPE grid actually run, and a transposed
  grid would produce a plausible tensor, so the PNGs are written and looked
  at). The full article is
  `out/qi21served/served1344x768_40steps.png`, a photorealistic 16:9 fox at
  **1m44.5s** (encode 113 ms, prefill 2.41 s, 39 steps at 2.42 s, decode
  7.38 s) — read against Q9's 1m32 at 1024² with the step at 2.12 s, the 14%
  is device contention: the `ai` service was resident and serving while this
  ran, and the per-step number is the tell. Nothing here is per-pixel more
  expensive than the square.
  **The fit had to grow a rule**, because the grid is coarse: both sides are
  multiples of 32 px, so an exact 16:9 exists only at 512m x 288m and there is
  nothing between 1024x576 and 1536x864. `fitAspect` now maximises the area
  **of the shape that was asked for** — a candidate's area discounted by what
  survives a centre-crop back to the requested ratio — with near-ties (within
  0.5%) broken toward the closer ratio. That is what picks 1344x768 at 1.05
  Mpx and the *exact* 1536x864 at 1.40, where a plain "largest that fits"
  would have taken 1568x864 and been 2% wider than asked.
  Two things deliberately did not change: the hard limit is still the VAE's
  single storage buffer (now stated as **1,403,584 pixels** rather than
  1184x1184, which is the same number said correctly), and `-image-size` is
  still one flag naming both the default size and the ceiling — its *area* is
  now what the ceiling means.
- **Q11 — a hung-up client stops the run. Done 2026-09-21.** The context now
  reaches the sampler, the VAE's submit loop, the vision tower and the VAE
  encoder, instead of being checked once on the way into `backend.Image`.
  **The granularity argument was half right and that is what made it worth
  fixing.** A denoising step *is* a submit-and-fence with nothing to abandon
  inside it — that much was true and is why no kernel changed here — but
  between two steps nothing is in flight, and the VAE already batches its
  dispatches four at a time for the driver's watchdog, which is 30 more
  boundaries in a 1024² decode. So the finest available granularity was never
  "the whole image"; it was one step and one batch, and nobody had gone and
  taken it.
  **Measured** (`TestCancellation`, the gate, at 512²/8 where a full run is
  6.27 s): cancelled after step 1, the run returns in **1.30 s — 21% of the
  full run** — with `pipeline: cancelled at step 2 of 8: context canceled`;
  cancelled an eighth of the way into a 1.71 s decode, it returns in
  **295 ms, after 12 of 117 dispatches**. At the served 1024²/40 shape that
  is ~2.2 s of a 92 s image instead of 92.
  **The third assertion is the one that would have hurt**: a cancelled run
  leaves a recorded graph half-submitted and the arenas holding an abandoned
  image, so the gate runs a *fourth* generation after the two cancellations
  and requires it **bit-identical** to the one before them. It is. Nothing
  about the device state survives a cancel, because nothing is left in
  flight to survive it — which is the same property that made the step
  boundary safe in the first place.
  **What this is really worth is the lock.** `backend.Device` is one mutex
  for every vertical in the process (one queue, nothing in `vk` externally
  synchronised), and it is held for the whole run — so an abandoned image
  was holding every queued speech, transcription and embedding request
  behind it for its full duration, not merely wasting its own GPU time.
  Text completions have always cancelled between tokens; **speech and
  transcription still check only on the way in**, and are the same shape of
  fix if an utterance ever gets long enough to care.
  One thing came free with it: `cmd/qimage` takes its context from
  `signal.NotifyContext`, so Ctrl-C now unwinds a run through the same path a
  hung-up client takes rather than killing the process mid-submit.

## Decisions taken now (so future sessions don't relitigate)

1. **New `qimage/` tree; `zimage/qwen` and `zimage/tokenizer` stay put** until
   z-image is fully gone, then a mechanical rename can hoist them.
2. **KV cache always on** — validate against diffusers `use_kv_cache=True`;
   cache-off exists only as a debug path if Q4 needs it.
3. **Posterior mode (argmax) for every VAE encode** — matches the reference
   and our reproducibility rule.
4. **1024² default, 2048² behind the ceiling flag**, pending Q6's numbers —
   amended 2026-09-20: 2048² is not a flag away. The decoder's arena stops
   at **1184²** on one storage buffer (Q5g), so serving above that needs
   tiling or a multi-buffer arena first. 1024² is the shipped default and
   `-image-size` refuses anything larger than 1184x1184 at startup.
   Amended again 2026-09-21 (Q10): **the ceiling is that rectangle's area and
   not its sides** — 1,403,584 pixels, whatever shape they are in — because
   the arena is 3060 bytes a pixel for every aspect ratio and a side limit
   was charging landscape requests for a constraint that does not exist.
5. **No self-trained tiny decoder**; linear preview + watch taehv.
6. **No CFG path in the Go port** (true_cfg_scale stays a refusal if asked
   for over HTTP) until something demands it — it would double every step.

## Open questions

- ~~**Q-o1**: how far below 40 steps does quality hold?~~ **Settled
  2026-09-20 (Q6 sweep)**: it depends on the prompt, which is why the
  default stays 40. Photographic and painterly prompts hold to **12 steps**
  (36 s); a structured technical drawing is fine at 24 and broken at 12. The
  headline is therefore **~1m38s at the safe default**, with 24 steps at
  1m4s available to any client that sends `steps`.
- **Q-o2**: does a turbo/distilled 2.1 checkpoint or step-distillation LoRA
  appear? (Watch the examples repo and lightx2v.)
- **Q-o3**: how are separate masks / painted annotations fed for local edits?
  (Examples repo, before `/v1/images/edits` promises masks.)
- **Q-o4**: does taehv grow a 2.1 variant? (Q7 fallback becomes an upgrade.)
- ~~**Q-o5**: text-only mrope — confirmed equivalent to plain RoPE by dump?~~
  **Settled 2026-09-20 (Q0)**: gap 0.0 against a plain theta-5e6 NeoX table;
  `zimage/qwen` needs nothing mrope-shaped before Q8.
