# G4b — the attention's key blocks through LDS (2026-09-30)

*The G4b stage, broken out of `KERNELS.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`kernels-vertical.md`](kernels-vertical.md).*


Filed as **§3.9** in `research/3.9-attention-kv-lds.md`, with the ladder.

**The build.** `h3_attn_t.comp -DKVLDS=1 -DWAVES=8 -DQT=1 -DKTIL=4
-DLAZY=1 -DTAIL_SPLIT=1`: a workgroup of eight wave32 waves, a query tile
each, over one head. A key block's 16 KB of K and 16 KB of V^T are each
fetched once by the workgroup's 256 lanes (four `uvec4` a lane) and
written to an LDS plane — K as key rows at 132 halves, V^T as component
rows at 68, both conflict-free for `ds_load_b64` — and every K and V
fragment is read from there. One plane each (two of each is 68.6 KB), so
two barriers a block: K(b+1) is fetched at the top of block b and stored
behind the barrier that ends the S^T reads, V(b+1) fetched there and
stored behind the barrier that ends the O^T reads. 34.8 KB of LDS, four
waves a SIMD, 256 VGPRs, no spill. The same MMAs in the same order per
accumulator as `h3_attn_t_qt2_kt4`: **bit-identical**.

| 480p forward, int8, two runs each, interleaved | attention | forward | sclk |
|---|---:|---:|---:|
| the one-wave build (`H3_ATTN_LDS=0`) | 9.79 / 9.82 s, 37.1–37.2 TFLOP/s | 25.23 / 25.21 s | 2653–2660 |
| **`h3_attn_t_kvlds_w8_kt4`** | **8.91 / 8.90 s, 40.8–40.9 (1.10x)** | **24.27 / 24.35 s (1.037x)** | 2717–2718 |
| 768p, one run each | 60.5 → **54.3 s** (34.6 → 38.6, 1.115x) | 95.9 → **92.5 s** | 2658 → 2704 |

| the screen, 480p, ms (the one-wave build 193.8–197.5 across the runs) | |
|---|---:|
| first cut (fetches a block ahead, staging registers across the back-edge) | 196–200 |
| fetched and stored inside one trip | 193.9–194.9 |
| … `LAZY` / `MAX_TREE LAZY TAIL_SPLIT` | 187.4 / 177.7–178.2 |
| **… `LAZY TAIL_SPLIT` (shipped)** | **175.1–175.6**; 768p 1.195 → 1.046 s; 3,100 keys 7.58 → 7.05 ms |
| its controls: no barriers / staging all L0 / softmax gutted | 170.3 / 164.6 / 165.6 |
| `KV_DEEP` / `KV_VTOP` / `KV_BSOFT` / sixteen waves / `QREG=0` / `KTIL2` | 194 / 183 / 184 / 186–202 / 220 / 200 |

**The three findings.** (1) **A register carried across a loop's
back-edge is copied there behind a wait for what it holds**: the first
cut's staging registers crossed the back-edge, the copy into the phi's
registers sat behind `s_waitcnt vmcnt(0)`, and the "block ahead" fetch was
waited out every block (its L0 control read 8–9%); §2.9's rule about
fragments holds for plain `uvec4`. The same back-edge rotated all 64
accumulator registers. Fetch and store inside one trip. (2) **§3.8's exact
softmax knobs were dead only at the register limit**: on a spill-free
build at 222 of 256 VGPRs `LAZY` is 3–4% and `TAIL_SPLIT` another 6%
(`MAX_TREE` still loses). (3) **The fetch is a quarter of the bytes** (32
KB a block for 128 queries, not for 32), and the forward's clock rises
60 MHz at the same package power. §3.8's estimate (≤ 2% of a forward) had
the LDS side right — two barriers are 3% — and missed (2) and (3).

**Carry-in.** `h3/dit` runs it by default wherever the transposed kernel
runs (`AttnVariant{QT: 8, KTIL: 4, T: true}`: `QT` is query tiles a
*workgroup*); `TestGPUForward` and `TestGPURun` pass. `h3/vae`: the
head-64 build reads 6.10 ms against the shipped 6.22 on the VAE's screen
(8 × 1,808 keys, 14.5 VALU an MMA), inside its spread — not embedded.
`qimage/dit`: the transposed kernel is opt-in behind the edit gate (§3.8)
and this is the same bits; not carried.
