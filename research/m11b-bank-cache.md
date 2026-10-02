# M11b — the stagings from a bank cache (2026-09-29)

*The M11b stage, broken out of `VIDEO.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`video-vertical.md`](video-vertical.md).*


M11a left every request re-making its int8 banks from the checkpoint: 48 GB
of bf16 for the encoder and 42 GB for the transformer read, widened to fp32
and `PackQ8`'d, plus 17 GB more for the AdaLN tables beside it. At 3.7 GB/s
off this disk that is ~30 s of reads. **The rest was CPU**, and all of it
produces the same bytes every time.

**The cache** (`qwen.Q8Cache`, `zimage/qwen/q8cache.go`) keeps a staging's
int8 banks on disk byte for byte as the device holds them, one file a bank,
in `bank-cache/` beside each component (`text_encoder/`, `transformer/`).
A miss stages as before and writes each packed matrix at its bank offset
into a temporary file, renamed into place once the staging is complete. A
hit reads the files straight into the mapped banks, 64 MB chunks on 8
readers, and loads only the norms from the checkpoint. The key hashes the
layout (every matrix's shape, bank and offset, so a different layer count
or fp16 keep list is a different file set), `q8CacheVersion`, and each
shard's name, size and mtime. It does not hash the shards' bytes (the LLM's
bank cache makes the same trade, P4c). The AdaLN tables are cached the same
way (`dit.TablesCached`), keyed by the request's timestep bits: one file a
step count and t2va/fl2va, 19 MB a distinct timestep (0.25 GB at N = 8, 0.72 GB at N = 20).

On by default: `pipeline.Options.BankCache`, `serve -video-bank-cache`,
`cmd/h3 -bank-cache`. The tests leave it off. It takes **47 GB of disk**
(25.9 GB encoder, 21.3 GB transformer), written by the first request, which
pays ~14 s for the writes.

| 448×256 × N = 2, `cmd/h3` | encode (staging + forward) | transformer staging | tables | request |
|---|---|---|---|---|
| M11a (no cache) | 40.5 s | 34.3 s | 19.2 s (beside it) | 97 s |
| first run (writes the cache) | 48.2 s | 40.4 s | 19.3 s | 111 s |
| cached, files evicted from the page cache | **7.5 s** | **4.5 s** | **8 ms** | **33 s** |

The cold row is honest: the files were evicted with
`posix_fadvise(DONTNEED)` first (no root needed), and buff/cache fell from
71 to 13 GB. That is 26 GB in ~6 s and 21 GB in 4.5 s, ~4.7 GB/s. Every
run's mp4 is **byte-identical** to the uncached one (same seed).

Served (hand-run `serve -video`, README-style request at `short_edge 256`,
`7:4`, N = 8): **74 s** end to end, against M9's 3 m 19 s. The default shape
(864×480 × 124 frames, N = 20): **741 s** against M11a's 826 s. That run
also computed the N = 20 tables for the first time (0.7 GB, 19 s, which
hide behind the transformer's staging), so a repeat should come in ~10 s
under. The forwards are now ~95% of a served request.

What is left of a request's staging is the video VAE's (inside the decode's
14 s at the small canvas) and the embedding table (3.1 GB of fp32 widened
from bf16 on every encoder staging). Neither has been timed on its own yet.
