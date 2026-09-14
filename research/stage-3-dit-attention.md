<!-- Pipeline stage findings, not an IDEAS section. Referenced from
     PIPELINE.md, which stays short by pointing here. -->

[← PIPELINE.md](../PIPELINE.md) · [research index](README.md) · pipeline stage 3

# Stage 3 — the DiT block and its attention

`zimage/dit` implements Z-Image's transformer block on CPU and its attention
stack on Vulkan. Both matched diffusers on their first run. Attention is
**correct but not yet fast**: 707 GFLOP/s, 2.1% of fp32 peak.

## Attention is two GEMMs and has to be built as such

Three successive structural fixes bought **1.53x in total**, and the third
is the one that says why no fourth will help:

| Change | Effect |
|---|---|
| Baseline: one query row per workgroup | 836 ms @ 4096 tokens, 463 GFLOP/s |
| QB=8 queries share an LDS key tile | **1.09x** — and it *lost* occupancy |
| Remove the LDS staging again | 1.11x |
| One key per thread, all QB queries in registers | 1.26x → 546 ms, 707 GFLOP/s |

- The LDS tile's 16 KB took the CU from six resident workgroups to two, and
  the latency hiding lost paid for the reuse gained. Occupancy is the budget
  that decides this kernel, not shared-memory capacity.
- After the register-blocking fix the kernel runs at **2.67 FLOP/byte where
  ~28 is needed** to saturate 22.9 TFLOP/s against the MALL. Reaching that by
  register blocking alone would need a query block near 32-64, which LDS
  cannot hold.
- So [IDEAS §3.3](../IDEAS.md)'s original instruction — build attention's two
  matmuls out of the register-blocked WMMA kernel — is now *measured* advice
  rather than a guess. The scalar-FMA path tops out an order of magnitude
  short.

**Cost today**: 546 ms for one block's attention at 4096 tokens, i.e. ~148 s
per image over 34 blocks and 8 steps. The linears are 79% of a block's FLOPs
but would run ~15x faster on the WMMA path, so attention is ~95% of the DiT's
time until this is fixed.

## Two hazards a port hits here

Both are covered by negative controls that catch them at **12,000x to
100,000x** the tolerance:

- **RoPE pairs adjacent components** — (2j, 2j+1) as one complex number. The
  other convention in common use pairs j with j+headDim/2; it is equally
  plausible, differs only in index arithmetic, and is wrong everywhere
  (measured at 12.6 against a 2e-4 bound).
- **The q/k RMS norms are per head**, over 128 components, not over the full
  3840. Normalising over the full width measures 2.43.

Two more the controls cover: adaLN's four chunks are concatenated
(scale_msa, gate_msa, scale_mlp, gate_mlp) rather than interleaved, and the
gated residuals are easy to drop.

## The checkpoint disagrees with the config

The DiT has **34 attention blocks, not 30** — 30 `layers` plus two
`context_refiner` and two `noise_refiner`, all the same module. And each
carries an `adaLN_modulation` of `[15360, 256]` that `bench/modelshapes.go`
does not model at all. Both were found by reading the weights rather than
`config.json`.
