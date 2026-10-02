# M11f — the audio decode on the device, priced (2026-09-29)

*The M11f stage, broken out of `VIDEO.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`video-vertical.md`](video-vertical.md).*


The handoff's lever was "7.2 s on the CPU against torch's 1.1". But
`Generate` has run the audio decode in a goroutine beside the video
decode since M8. So the question is not the audio's time but what it
adds to the request. `TestDecodeOverlap` (`h3/pipeline`, `H3_OVERLAP=1`)
answers that at the served 480p × 124 frames (105 tile-clips, 5.17 s of
stereo), on random latents, two passes:

| audio workers | audio alone | together | video's device time | past the video alone |
|---|---|---|---|---|
| — (video alone) | | 37.6–37.7 s | 37.13–37.17 s | |
| 32 (shipped) | 7.5–7.7 s | 40.1–40.2 s | 39.4–39.5 s | **+2.4–2.6 s** |
| 8 | 15.4–15.5 s | 39.2–39.3 s | 38.7–38.8 s | +1.6 s |
| 4 | 27.6 s | 38.9 s | 38.4 s | +1.2–1.3 s |

The audio always finishes inside the video decode. The cost is that the
*device* runs slower while the CPU decodes. ~~That is contention for the
shared memory, not host scheduling~~ **Corrected 2026-09-29 (KERNELS.md
G0, research §0.6): it is the package power budget.** The GPU runs a
forward at 2600–2650 MHz and ~149 W; 32 busy CPU threads take it to
1300–1400 MHz and a forward from 30 to 50 s. The audio decode's 32
workers draw from the same ~150 W, and the device slows in proportion.
The host-side part of the video decode is unchanged either way. Fewer workers spread the same traffic over a longer
time, so the cost falls but never goes away. At four workers the audio
also takes 28 s, which would put it on the critical path of any canvas
smaller than 480p. `audiovae.SetWorkers` is the knob, and
`H3_OVERLAP_WORKERS=a,b,...` sets the arms.

**Price.** A device BigVGAN would remove at most the ~2.4 s of contention,
minus its own device time (~480 GFLOP of fp32 1-D convolutions). That is
≤ 0.3% of a served 480p request (~740 s). The port is the whole codec:
dilated convolutions, transposed-convolution upsampling, and SnakeBeta
with its replicate-padded ×2 kaiser resampling, all in fp32 (M6: bf16 is
20 dB quieter). **Not built.** A worker count that adapts to the tile-clip
count (the audio finishing just before the video decode) would get about
half the saving without a port. It was not shipped either: its model of
the audio's time is per-machine, for ~1 s a request.
