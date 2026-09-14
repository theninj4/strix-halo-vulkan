<!-- Pipeline stage findings, not an IDEAS section: these come from building
     the z-image slice rather than from a numbered experiment. Referenced
     from PIPELINE.md, which stays short by pointing here. -->

[← PIPELINE.md](../PIPELINE.md) · [research index](README.md) · pipeline stage 2

# Stage 2 — the VAE decoder

Built as a CPU reference first (`zimage/vae`, validated stagewise against
diffusers) and then as ten Vulkan shaders. 1024x1024 in **5.6 s**, matching
diffusers to 2.4e-5.

## fp16 cannot hold this model's intermediates

- **Activations are fp16-safe; the attention scores are not.** Activation
  absmax through the whole decoder is **497**, comfortably inside fp16 — but
  the mid block's score matrix reaches **1.16e7**, 177x past fp16's 65504.
  Scores must accumulate in fp32 (the WMMA fp16->fp32 path does this
  natively) and the row max must be subtracted before any exponential. This
  is what the VAE config's `force_upcast` is for.
- Stage 3 then found the same thing again in the DiT, whose feed-forward
  output reaches 5.99e5. Two of two stages, so an fp16 port must keep fp32
  accumulators rather than assume it can fold them.

## That attention is fully saturated, and it defeats an obvious test

Scores reach 1.16e7 and the softmax is a **hard one-hot, entropy 0**, with
the argmax set by `||k_j||` rather than by q's direction. Transposing `to_q`
therefore changes the output by **1.5e-5** — i.e. not at all — so no
tolerance can catch a q-projection bug at this input. Validate `to_v`
instead, and do not read a passing q test as coverage.

## The four fixes, three of them found by profiling

The Vulkan decoder was correct on its first run, because the CPU stage had
already settled the architecture: every bug left was a shader bug. Everything
after that was performance or a resource limit.

| Fix | Effect |
|---|---|
| Register-block conv over 8 output channels | conv3x3 **551 → 3342 GFLOP/s** |
| Transpose K before the attention score loop | attention **1.03 s → 51 ms** |
| Best-fit arena with coalescing | activations **5.28 → 3.12 GB** |
| Read back only the output, not the whole arena | latent-32 wall **4.9 s → 217 ms** |

- **Staging weights in shared memory was the wrong fix, and measuring said
  so.** conv3x3 was 90.3% of the decode at ~750 GFLOP/s; caching the filter
  in LDS moved it only to ~850. That near-miss is what proved the weights
  were never the bottleneck — the kernel re-reads each *activation* once per
  output channel. Register-blocking the accumulators, which is [§2.1](2.1-register-blocking.md)'s
  result applied to a different operator, was worth **6.1x**.
- **[§5.1b](5.1b-mall-cliff-and-stride.md)'s coverage law predicted the
  attention bug exactly.** The score loop had thread `t` read key row
  `base+t` at a fixed component, so consecutive threads landed `dim*4` =
  2048 B apart: `min(1, C/gcd(stride, 4096))` = 4/2048, **0.2% of the bus**,
  measured at 50 GFLOP/s and 67 GB/s. Transposing K so consecutive keys are
  consecutive addresses, with the row stride padded 64 elements to land the
  gcd inside §5.1b's [128, 256] window, was worth **20x**.
- **The profile kept being surprising.** Attention looked like the worry and
  was 0.4% of the decode at 256²; conv looked solved after the LDS change
  and was not. At 1024² the split is conv3x3 54.7%, attention 26.5%, linear
  9.4%.

## Two device limits that shape the engine

- **A single Vulkan storage buffer addresses 4.29 GB here**
  (`maxStorageBufferRange` and `maxMemoryAllocationSize` both), which a bump
  allocator that never reuses exceeds at 1024x1024. A best-fit free list with
  coalescing brings the arena to 3.12 GB against a ~2.7 GB theoretical peak.
- **A command buffer holding the whole 119-dispatch graph is 5.5 s of GPU
  work**, which trips the driver's reset watchdog and returns
  `VK_ERROR_DEVICE_LOST`. The identical dispatches submitted in batches of
  eight all complete. Batch submissions; do not assume one submit per graph.
