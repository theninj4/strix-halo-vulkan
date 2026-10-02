# M11e — the VAE's attention, transposed (2026-09-29)

*The M11e stage, broken out of `VIDEO.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`video-vertical.md`](video-vertical.md).*


The decoder's attention (36 layers, 32 heads of 64, a tile-clip's 1,797
tokens against its own keys, 8 tile-clips a batch) ran the plain
`dit_attention_wmma.comp` at head 64. It now runs `h3_attn_t.comp` at
`-DHEAD_DIM=64`, chosen at the first run by the same element-order probe
the transformer uses (`dit.TransposedAttentionOK`, exported for this; the
plain build stays the fallback, `vae.AttnVariant`, `GPU.SetAttention`).

One change to the kernel. The tile-clips sit side by side at a 16-row
stride (1,808), so at QT = 2 the last workgroup's second query tile is
the *next* sequence's first 16 rows, and storing it would race that
sequence's dispatch. `-DSTORE_TAIL=0` skips a query tile that starts
past the sequence. The transformer's build leaves it at 1 and its
SPIR-V is byte-identical to before. `maxKeyBlock` went from 4 to 8 tiles,
for the KTIL 8 builds (the plane's pad rows).

**Screen** (`TestGPUAttentionScreen`, block 0's planes of 8 random
tile-clips, one batch-layer, median of 7, two runs):

| build | ms | TFLOP/s | rms vs plain |
|---|---|---|---|
| plain QT1 KTIL4 (was shipped) | 8.91 / 9.14 | 23–24 | — |
| T QT1 KTIL4 | 6.49 / 6.54 | 32.5 | 8.7e-5 |
| T QT1 KTIL8 | 6.76 / 6.56 | 31–32 | 1.1e-4 |
| **T QT2 KTIL4** | **6.19 / 6.21** | **34.1** | 8.7e-5 |
| T QT2 KTIL8 | 6.85 / 6.87 | 31 | 1.1e-4 |
| T QT4 KTIL4 | 10.6 / 10.8 | 20 | 8.7e-5 |

The transformer's choice (QT2 KTIL4) wins here too, 1.44x. QT4 spills.

**Decode** (`TestGPUShapes`, `H3_VAE_ATTN_PLAIN=1` the control, two
passes of each arm, interleaved): 480p **37.0 / 37.2 s** against 38.4 /
38.4, 768p **69.4 / 69.2** against 71.5 / 71.5 — 1.035x, which is the
kernel's saving (14 batches × 36 layers × 2.9 ms ≈ 1.5 s) and nothing
else. The attention was ~12% of the decode, so this is its ceiling.
**Gate** (`TestGPUDecoder`): blocks teacher-forced at rel ≤ 1.4e-4, the
tile-clip and the 12-frame decode at **PSNR 81.7 dB** against fp32, as
in M5.
