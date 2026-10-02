# G6 — the streaming kernels (2026-09-30)

*The G6 stage, broken out of `KERNELS.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`kernels-vertical.md`](kernels-vertical.md).*


Filed as **§5.4** in `research/5.4-grid-order-and-channel-phase.md`, with
the arms and the full tables.

**The finding.** A pass that reads a slice of each row a workgroup keeps
the next few dozen workgroups, in grid order, in flight beside it. If one
step along the grid's X axis moves the address by a whole number of the
4 KB DRAM channel rotation (§5.1b), all of them load the same channels and
the rest idle. That is the LLM's `[T][4][2560]` residual walked with the
token on X (a token is ten rotations; a stream's 10,240 B row asks one half
of the rotation for three bytes and the other for two: 5/6 of the bus), and
the DiT family's q/k pack walked with the token tile on X (one head of
consecutive rows is 512 B of a row that is 7, 4, 2 or 1 whole rotations:
an eighth or a sixteenth of the channels). The one-line test: the same
grid walked in the other order. Hoisting every load of a row and
sixteen-byte lanes moved nothing at DRAM sizes, which is why P11, P12 and
P15 each spent a hypothesis on `hc.cn` and found nothing — and a third of
its "missing" rate was never missing: 134 GB/s counted the residual once,
and an in-place update is a read and a write (184).

**The table the gate asks for** — every streaming pass over 1 ms a block,
before → after, GB/s of read + written against a copy's 236:

| host (shape) | pass | before | after |
|---|---|---:|---:|
| LLM, 8,192 rows | `hc.cn` (94 a pass) | 184 (4.98 ms) | **200** (4.62–4.66 ms) |
| | `hc.norm` | 183 | **217** |
| | `hc.combine` | 191 real (the block output read 4x) | **210**, 1.48x |
| `h3/dit`, 480p | `qkpack q`, `k` | 117 | **209** |
| | `attn in`, `ffn in` | 193 | 207 |
| | `gate attn`, `gate ffn` | 211 | 210 |
| `h3/vae`, 8 × 1,808 rows | `qkpack q`, `k` | (no profile before; a sixteenth of the channels) | 198–199 |
| | norms / gates | | 205 / 204–213 |
| `ace/dit`, 7,500 rows | `qkpack q`, `cross q` / `k` | 120 / 93 | **208 / 187** |
| | norms / gates | 181 / 193–208 | 181+ / 193–208 (hoisted: +0.6–1.8% a forward) |
| `qimage/dit`, 1024² | `pack k`, `v` / `qkpack q` | 126 / 151 | **173 / 189** |
| | `gate` / `dequant` | 175 / 240 | not touched |

**What it is worth**, every row bit-identical to the binary before it:
LLM prefill **1520 → 1538 tok/s at 2048, 1552 → 1570 at 4096, 1493 → 1498
at 8192**, decode **36.87 → 37.35 tok/s** (`hc.cn` 7.0 → 2.4 µs a
dispatch: the hoisted loads, which is where they matter); a 480p video
forward **25.58 / 25.44 → 25.25 / 25.21 s**; a 480p VAE decode **29.95 →
27.5 s**, 768p 55.7 → 51.6; a music forward **2113 → 2050 ms at 10 min,
745 → 723 at 4 min, 111.4 → 109.8 at 30 s**; an image step 1.895 → 1.886 s.

**What changed.** `llm_hc_{norm,combine,cn}.comp`: a stream across and a
token down, `hc.cn` a workgroup a (token, stream) at every width (P11's
walk and `LLM_HC_CN_SPLIT_ROWS` are gone), a row's loads hoisted.
`h3_qk_pack`, `dit_qk_pack`, `dit_pack_f16`: `-DHEAD_X=1` builds, used by
`h3/dit`, `h3/vae`, `ace/dit` and (at the served eight tiles) `qimage/dit`.
`h3_norm_mod`: hoisted loads. `dit_gate_add_hoist`: a hoisted build for the
three hosts that re-ran their gates. `h3/dit`, `h3/vae`, `ace/dit`: a
`Bytes` field a stage and a GB/s column in the profile; `h3/vae` has a
profile now (`Profile`, `TestGPUProfile`).

**Left.** The gate fused with the norm that reads what it wrote (one read
of the residual, ~a fifth of those two passes: ~1.5% of a music forward).
The hoisted gate on qimage, kev, ocr, zimage and `ace/lm`. The hosts with
no bytes column yet: kev, ocr, `zimage/qwen` (its pack walks one head of
many rows of 2.5 rotations: predicted a quarter of the channels), the
speech encoders (cache-resident, where the law does not bind), and the
LLM's `move`, `moe.combine`, `dn.*`, `ple.*`.
