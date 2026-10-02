# M11g — the last two stagings, timed (2026-09-29)

*The M11g stage, broken out of `VIDEO.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`video-vertical.md`](video-vertical.md).*


M11b left two stagings untimed. `TestStagings` (`h3/pipeline`,
`H3_STAGINGS=1`) times each on its own. For a cold read, the files are
first evicted with `posix_fadvise(DONTNEED)`, as in M11b. Two warm runs
and two cold:

| staging | warm | cold | where in a request |
|---|---|---|---|
| the text encoder's embedding table (151,936 × 5,120 bf16 → 3.11 GB fp32) | 0.73–0.74 s | 1.01–1.02 s | inside the encoder's staging |
| the video VAE decoder (4.88 GB fp16 banks + 1.86 GB activations at 8 × 1,808 rows) | 4.58–4.65 s | 5.44–5.50 s | first act of the decode |
| the same, beside the 5.17 s audio decode | **8.44–8.53 s** | | *as a request runs it* |

The embedding table is noise: under a second, 0.1% of a 480p request.
Leave it.

The VAE staging is the one that counts. It is a serial loop over 36
blocks. Each block reads fp32 from the checkpoint (the decoder is 9.8 GB
of fp32), adds the bias column, packs to fp16 and writes into the mapped
bank: ~2.1 GB/s, and cold is only 0.8 s slower than warm, so it is CPU,
not disk. And `Generate` starts the audio decode just *before*
`decodeVideo`, whose first act is this staging. The two contend for the
CPU, and the staging stretches from 4.6 to **8.5 s**.

**This corrects M11f's picture.** M11f timed the audio beside an
already-staged video decode (+2.4 s of device time). In a request, the
7.5 s audio decode instead spends most of its time beside the VAE
staging, and costs that staging ~3.9 s of host time. The conclusion holds
(a device port still buys only seconds), but the cheap levers are here:

- stage the VAE's blocks in parallel (the loop has no dependency between
  blocks; bytes identical);
- or start the audio decode after the staging, so it overlaps the video
  decode's device time instead of the staging's CPU time;
- or cache the packed fp16 banks as M11b does (4.9 GB of disk, ~1 s
  cold at M11b's 4.7 GB/s).

Together these are worth ~4–7 s of a request (~1% at 480p). Not built.
