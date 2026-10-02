# M11a — int8 banks for the encoder and the transformer (2026-09-26)

*The M11a stage, broken out of `VIDEO.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`video-vertical.md`](video-vertical.md).*


The deployed service leaves ~35 GB free, and a request staged 50 GB (the
encoder) then 42 GB (the transformer). Both now hold their projections as
**int8, one fp16 scale per 32 k of a row**. That is Kev K7.1's
quantisation (q rounded against the fp16 scale) in the LLM's L8 fragment
layout, `qwen.PackQ8`.

**Not an int8 GEMM.** A layer's (or block's) first seven dispatches expand
its projections into one fp16 scratch (`shaders/dit_dequant_q8.comp`, 64
threads a 16×16 tile), and the validated fp16 GEMMs read the scratch as
before. The expansion is **bit-identical** to the host's
(`qwen.TestGPUDequantQ8`). At these row counts the GEMMs are compute-bound,
so an int8 kernel would buy nothing (TODO.md: int8 WMMA runs at fp16 rate).
Writing 0.77 GB of fp16 a block is invisible next to a 7 s forward:

| | fp16 | int8 |
|---|---|---|
| encoder staged | 50 GB, 51 s | **26.9 GB**, 60 s |
| encoder, 537 tokens | 1.0 s | 1.4 s |
| transformer staged (weights) | ~40 GB, 53 s | **22.1 GB** (21.3 int8 + 0.77 scratch + heads), 50 s |
| a forward, 5,095 rows, same day | 8.26 s | **7.29 s** |

At 5,095 rows int8 is **12% faster** a forward. (M7's 7.4 s for fp16 did
not reproduce on the day.) Every block's GEMMs read the same hot 0.77 GB
scratch rather than a stack spread over 40 GB, which is the likely reason,
but that is untested. At the served 480p, where attention dominates, it
is even: 35.7 s against M7's 35.6.

`pipeline.Options.Bank`, `cmd/h3 -bank q8|fp16` (default q8), and `serve
-video-fp16` for the control (default int8). Tests take `H3_BANK=q8`.

**Gates.** The encoder is held to its fp32 oracle as M2 was. The transformer
is held to the **released pipeline**: `reference/dump_h3_dit_bf16.py` runs
M4's seven forwards teacher-forced at diffusers' `torch_dtype=bfloat16` (its
`_keep_in_fp32_modules` split). That is the error MiniMax ships, and fp16
was never the right yardstick for a quantised bank.

| what | fp16 | **int8** | official bf16 |
|---|---|---|---|
| encoder, README prompt, rel of absmax (en / cjk) | 1.1e-4 | **3.0e-4** (2.1e-4 / 1.7e-4) | 1.2e-2 |
| refiner | 5.4e-4 | 1.7e-3 | 1.3e-2 |
| blocks 0/1/2/25/49, teacher-forced | ≤ 1.9e-4 | ≤ 4.2e-4 | |
| forward 0 video velocity, rms | 1.8e-3 | 1.8e-2 | 2.0e-2 |
| step latents rms (video), steps 0…6 | 7.9e-5 … 1.7e-3 | 7.9e-4, 1.0e-3, 2.1e-3, 2.1e-3, 2.2e-3, 3.2e-3, **6.9e-3** | 8.6e-4, 1.7e-3, 3.0e-3, 3.3e-3, 3.5e-3, 6.6e-3, **1.6e-2** |
| step latents rms (audio), steps 0…6 | ≤ 4.0e-4 | 6.9e-4 … 4.8e-3 | 1.1e-3 … 8.5e-3 |

**Every int8 step is at or inside bf16's, on both modalities** (0.46×–0.99×;
step 1's audio is the 0.99). `TestGPURun` now gates int8 at 1.25× bf16's
step, a step at a time, and fp16 at its old 3e-3.

**Where the error comes from** (`TestGPUQ8Ablation`, `H3_Q8_ABLATION=1`:
chosen projections of an fp16 bank held at their int8 values; the "all" arm
reproduces the real bank's numbers exactly). Putting any one of the seven
back to fp16 moves the worst step by at most 11% (6.1e-3 … 7.0e-3 against
6.9e-3). The error is spread evenly, so a mixed bank buys nothing worth its
memory.

**End to end** (`TestE2E`, free-running from the oracle's noise, 448×256 ×
N = 8): 3 m 4 s against 3 m 28 s, most of it staging. Frames PSNR **19.8 dB**
against the oracle's decode (fp16: 25.9), soundtrack SNR 16.5 dB (21.5), and
final latents rms 0.263. That is one draw of a chaotic divergence.
`TestGPUSensitivity` now takes `H3_SENSITIVITY_DRAWS`, and 8 draws of a
1e-4 move of the starting noise alone end 0.056–0.212 rms apart (median
0.10). fp16 lands at 0.121 and bf16 (`dump_h3_dit_bf16.py --free`) at 0.146.
Int8's draw is 1.24× past the chaos spread's top, while its every step is
inside bf16's. Frames 0/60/120 against the oracle's show the same shots,
character and lighting; framing and ship placement drift, as free runs do.
`TestE2E`'s floor for int8 is 18 dB (`q8PSNRFloor`): it catches a broken
run, and `TestGPURun` holds the precision.

**Memory at a request's peak**, measured, not summed: MemAvailable
sampled every 0.5 s against idle, over `TestE2E`. The int8 banks alone
took the peak only from 57.1 to 39.3 GB, because the host was holding the
rest. Two host-side fixes brought it to 31:

| | peak | encoder phase | transformer phase | wall (448×256, N = 8) |
|---|---|---|---|---|
| fp16 | 57.1 GB | | | 3 m 20 s |
| int8 | 39.3 GB | 33–39 | 29 | 3 m 3 s |
| + `dit.LoadHost`, a GC a staged layer | 34.0 GB | 30–34 | 21 | 3 m 19 s |
| + tables beside the transformer's staging | **31.2 GB** | 30 | 21–24 | 3 m 22 s |

- **`dit.LoadHost`.** `Tables` and `NewGPU` each called `Load(dir, 0)`,
  which materialises the two refiner blocks in fp32 (3.1 GB). Neither
  needs them: `Tables` wants the time MLP, and the device stages the
  refiner from the checkpoint.
- **`runtime.GC()` once per staged layer or block** (`qwen.stageWeights`,
  the DiT's staging, `Tables`). A 32 B encoder layer is ~2 GB of fp32 on
  the host, dead once quantised, and at the default GOGC the heap's
  doubling headroom kept several of them. `GOGC=25` showed the size of it
  (39.3 → 35.9 GB). The collection is pointer-free and cheap, and it
  changes no process-wide setting in a server that runs every vertical.
- **The AdaLN tables moved from beside the encoder's staging to beside the
  transformer's**, which now has the headroom. With `Resident` the
  transformer is already staged, so they stay beside the encoder.

That costs ~20 s of wall at this shape (the tables contend with the
transformer's staging for disk, plus the collections), ~2% of a served
480p request. The frames are the same bits throughout.

**The served shape** (`cmd/h3`, README prompt, 864×480 × 124 frames,
N = 20, 15,936 rows): **14 m 25 s**, peak **32.6 GB** (in the encoder's
staging), the transformer phase 19–24 GB, 35.7 s a forward, decode 52 s.
That looked like it fit inside the ~35 GB the deployed service left, with
~2 GB to spare. **It did not.**

**Served, it was OOM-killed** (2026-09-26 19:34:56, 7 s after submit;
systemd restarted the unit and the in-memory job was lost). The kernel's
report had 79.7 GB of GPU pages and 40.6 GB of anon, which is all of RAM.
The standalone measurements had missed the host side of a *server*.
Rerun by hand with `GODEBUG=gctrace=1`, `H3_MEMLOG=1` (the pipeline logs
the Go heap and RssAnon every 250 ms) and a guard that kills the server
below 3 GB free, the Go heap went **14.6 → 28.6 GB in 2 s**, before the
encoder's first layer, with no collection run. Two causes:

- **`safetensors.Tensor.F32(nil)` grew its output by `append`.** A large
  slice grows ~1.25× at a time, so the 3.1 GB embedding table churned
  ~15 GB of copies, and every fp32 load of every model paid ~5× its size.
  Standalone, the heap was small and the GC collected constantly, which
  hid it. In the server, with a 14 GB live heap, the next target was
  29 GB. `F32` and `F16` now reserve their capacity (`slices.Grow`).
- **The server kept ~11 GB of collected garbage resident** from staging
  its other models: 24 GB RssAnon against a 13 GB live heap. `serve` now
  calls `debug.FreeOSMemory()` once every backend is up, and the pipeline
  calls it at the start of a request and after each staging is freed.

| the deployed configuration (image, kev, speech, video) | available at rest | a 480p request's low point |
|---|---|---|
| before | 36.9 GB | OOM at +7 s |
| after | **47.5 GB** | **15.4 GB free** (+17 s, encoder staging); 18.7 GB in the transformer phase |

The server now gives every vertical ~10 GB more room at rest, not only
video. **Served end to end** (hand-run server with the unit's flags, the
README prompt, 864×480 × 20 steps): completed in **826 s**, 124 frames of
h264 and stereo AAC, and the frames follow the prompt.

**Through `ai.service` itself** (the unit's line restored to `-image
-edits 3 … -video`, the same request on :11434): completed in **834 s**,
low point **15.4 GB free**. The deployed machine now serves video beside
the image model. Don't measure headroom with anything else running: a
background `go test -short` sweep stages models too, and it tripped the
guard on the first attempt.
