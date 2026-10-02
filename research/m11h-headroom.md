# M11h — where the headroom is (2026-09-29)

*The M11h stage, broken out of `VIDEO.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`video-vertical.md`](video-vertical.md).*


An assessment against the goal, which is the most out of this hardware,
not the fastest video. A 480p forward re-profiled today (`TestGPUShapes`,
`H3_SHAPES=480 H3_PROFILE=1`, int8, two passes): **28.98 / 29.33 s**.

| kind | share | rate (TFLOP/s) |
|---|---|---|
| attention | 33% | 37.4–37.5 |
| gemm qkv / gate / up / o | 17 / 11 / 11 / 6% | 37.2–37.9 / 35–36.5 |
| gemm down (two K passes) | 13–14% | **29.0–31.3** |
| swiglu, packs, gates, norms, dequant | 6.5% | (elementwise) |

**The exact ceiling.** A 480p forward is 978 TFLOP (614 in the GEMMs, 364
in the attention), so today's forward runs at **33.4–33.8 TFLOP/s, 60% of
the 55.5 WMMA peak and 80% of the 42 TFLOP/s that the best kernel on this
machine has ever reached** (research/ideas.md, the ceiling table). If every
MMA ran at 42 the forward would be 23.3 s: **1.25x is all that exact
kernel work can give**, and most of the pieces are already at 37–38. So
the dense pipeline is within ~1.15x of what this hardware does with the
arithmetic the model asks for; every larger multiple is fewer forwards or
cheaper attention, and both change the output.

**The exact levers, priced:**

- **The row chunk: 8192 → 4096 is 1.04x, measured, not shipped.** The
  same run at `H3_CHUNK=4096` is **27.94 / 28.11 s**, and 2048 is 29.98 s.
  The whole gain is the down projection, 29–31 → **34–37 TFLOP/s**: at
  chunk 8192 its A chunk is 235 MB (M11d), past the MALL, and at 2048 its
  grid is 16 × 21 workgroups. The other GEMMs and the attention move by
  under 2% either way. This closes M11d's "29 against `o`'s 35" gap
  without touching the kernel, and it should be bit-identical (the chunk
  is a row partition, and every GEMM and query tile makes the same MMAs in
  the same order). One line in `pipeline.Options.Chunk`'s default, a
  `TestGPUForward` byte comparison, and the 768p shape timed.
- **Gate + up + SwiGLU as one GEMM** (Kev's K7.6, `kev_gemm_q8_glu.comp`,
  for the big-tile fp16 GEMM): removes the swiglu pass (0.57 s, 2%) and
  one stream of the A chunk. ≤ 2.5% of a forward.
- **The attention**, 37.5 TFLOP/s at 68% of peak, spilling 112 VGPRs.
  Before any kernel work, price it the M11c way: gut the transposed loop to
  its MMAs alone. If that arm is ~40, the softmax and spills cost nothing
  worth chasing; if it is 45+, a spill-free QT2 is worth up to 15% of the
  attention, 5% of a forward.
- **The elementwise 6.5%**: the two gates could ride in the GEMM epilogue
  and the v pack in the qkv GEMM's. ≤ 3%, several kernels.
- **The video VAE decode**, 37 s at 480p (~6% of a request; ~40% of the
  80 s served small-canvas request). GEMM-bound at ~26 TFLOP/s, K = 2048:
  the same chunk/grid screen as above, at `H3_VAE_SEQS`. ≤ 1.5% of a
  480p request.
- **The stagings** (M11g): 4–7 s a request, ~1% at 480p.
- Refused: the transformer resident between video requests (12 s a
  request, 2%, for 28 GB standing in the swap slot against the image
  model); int8 GEMMs (compute-bound, M11a); a device audio decode (M11f).

Together the exact levers are ~1.10–1.15x a request, the chunk being a
third of it for one line.

**The lossy levers, which are where the multiples are.** All of these are
the community's, measured on other hardware, and none is in this tree:

- **MiniMax's own sparse attention** is still "a future update" on the
  card as of today (M-o4 stays open).
- **SGLang's H200 post** (2026-08-27): fused kernels are 1.95x lossless
  over diffusers (we have those fusions: modulated norm, gated residual,
  packed RoPE). **Cache-DiT** (a forward is skipped when the residual's
  change is under a threshold) and **SubBlock sparse attention** (64-token
  query and key blocks, 20–25% of key blocks kept) compose to **6.24x at
  SSIM 0.76–0.91** against their lossless run. At 480p the attention is
  33% of a forward, so sparsity alone is worth ≤ 1.3x here; step caching
  is the bigger of the two.
- **Lightx2v's Turbo-SLA** (2026-08-20): a **4-step LoRA** over the base
  transformer, shifts 6 / 3, with 85% sparse attention through a SageAttention
  operator. **fl2va only, 768p**, 2.5x on a 5090 against the 30-step base.
  The step count is the lever: N = 4 is 3 forwards against 19. Whether
  the LoRA holds up dense (without its sparse operator) is unmeasured.

These change the output, so if any is built it is a request parameter
with the dense schedule as the default, as `steps` is, and its gate is
perceptual (M8's PSNR/SSIM against the dense run), never a tolerance.
Whether to chase them is a product decision: the honest number for a
lossy mode is "N× faster at SSIM s against the dense run".

**Not yet measured:** the served 480p × 20 request after M11c–M11e
(estimated ~640 s from M11b's 741; one request through `ai.service` gives
the real number).
