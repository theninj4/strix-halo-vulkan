<!-- LLM.md P1c. The decode step recorded once and replayed, the position
     moved out of the push constants to make that legal, and the two host
     rows it was priced to remove. Cited from llm/graph.go, llm/record.go,
     vk/engine.go, vk/shim.c and shaders/llm_common.glsl. -->

[← LLM.md](../LLM.md) · [← LLM2.md](../LLM2.md) · [research index](README.md) · [P1](p1-decode-attribution.md) · [P1a](p1a-hyper-connection-shape.md) · [P1b](p1b-moe-decode-rescreen.md) · phase 2

# P1c — the decode step, recorded once: 1.90 ms of the priced 1.96, and the machine moved more than the fix

**Result: the decode step is one pre-recorded command buffer, replayed with a
fence per token, and it is worth its estimate.** P1 priced idea 2 at 1.127 ms
of host recording plus 0.828 ms of hand-over — 1.96 ms, 6.4% of the step —
and on matched machine state the fix returns **1.90 ms a token**: 39.39 →
37.50 ms a step, **25.4 → 26.7 tok/s**, the `record` row 1.127 → 0.16-0.17 ms
and the hand-over 0.82 → 0.25. Two runs each arm, 0.03% apart. The gate is
bit-level: 48 greedy tokens *and all 48 full logit rows* identical to the
re-recording arm on the four-layer prefix, the whole model's greedy text
identical to EOG, and prefill at ubatch 2048 unregressed (1070.3 against the
baseline binary's 1061.5 the same hour).

**And the reason every number above is quoted against a same-day control:
the machine moved 8.7 ms a step overnight with nothing changed.** Yesterday's
binary, byte for byte, measured 30.51 ms a step for P1b and measures 39.20
today. See the last section.

## The premise, measured first

The whole idea rests on the decode graph being shape-stable, so the first
thing built was the instrument (`TestDecodeDispatchDiff`): capture the
recorded sequence of consecutive one-token passes and diff them dispatch for
dispatch — pipeline, grid, and every push-constant dword, mapped back to its
field name by reflection over the `push` block. Over 40 steps on the
four-layer prefix:

- **Every pipeline, every grid, every label, in the same order, every step.**
- **111 of 131 dispatches carried identical push constants already.**
- The other 20 differed in exactly **one dword: `lowRank`, carrying the
  sequence position** — twelve kinds (`attn.pack/idx/score/attn`,
  `dn.conv/hist/norm/scan`, `ple.conv/gate/hist/kv`), of which only seven
  shaders actually read it (`SEQ_PAST`); the rest carried it because their
  block's base push block did.

So the entire varying state of a decode step was one uint. The token id never
touches a push constant or a grid — it enters through the arenas the host
writes (the embedding rows, the n-gram gather).

## Finding 1 — the position moves into the arena, and the step becomes bytes

`SEQ_PAST` is now **dword 0 of each block's own fp32 activation arena**, read
through the `actu` view every pipeline already binds (`actu[0]`), written by
each block's `SetPast` as a 4-byte mapped write. It is the first allocation
of the three sequence-carrying blocks (attention, DeltaNet, PLE), asserted at
offset zero because the shaders hardcode it; the `lowRank` field stays zero
in those blocks' base push blocks and keeps its other meanings elsewhere (the
hyper-connection's actual low rank, the GEMV rungs' int8 split). One macro
changed in `llm_common.glsl`; nine `.spv` recompiled; no pipeline layout, no
new binding, no spec constant.

After the change the diff instrument reports **131 of 131 dispatches
byte-identical over 40 steps**, and every SEQ_PAST-consuming gate passes
unchanged: the cache-is-a-chunk-split tests, the pooled-block tests, the
selection tests at 4k, the graph chunk-split, `TestGraphPrefix`.

Why not a uniform buffer? The push block is 64 uints — this device's whole
range — and full; a new binding would have re-cut every descriptor layout for
one uint. Dword 0 of a buffer every kernel already binds costs one scalar
load that ACO hoists, and nothing else.

## Finding 2 — record once, replay with a fence

`vk.Prerecorded` (shim: `shim_prerecord_multi` / `shim_submit_prerecorded`)
records the same sequence `DispatchMultiMarked` would — same barriers, same
per-dispatch timestamps, the query-pool reset *inside* the command buffer so
the marks re-arm on every submit — into its own command pool, once. Submit is
a fence reset, one `vkQueueSubmit`, a fence wait and the timestamp read.
Because the marks survive, **`-gen -attrib` works identically on the replay
path**, which is how the table below exists.

The graph captures at the first one-token `Extend` with `past > 0` — the
capture *is* the sequence that token submitted, taken at flush before the
recorder resets — and replays from the second token on
(`extendPrerecorded`): the two gathers, the uploads, the `SetPast`s and the
host-side `Resize(1)`s (field sets and pad clears, kept because a pass after
a fresh prefill would otherwise run with the prompt's row count in host
state), then one submit of all **1407 dispatches in one buffer** where the
live path chunks into two (27 ms of GPU against the watchdog's 2 s; the
recording sizes its own query pool). `past > 0` because a fresh sequence is a
different shape: at position zero the attention block rebuilds its whole
pooled table and `attn.idx`'s grid is NBlocks, not one. `PinSchedule` drops
the capture; `LLM_NO_PRERECORD=1` is the control arm.

## The step, before and after (same machine state, same hour)

64 greedy tokens, whole model, shipped bank. Control is `LLM_NO_PRERECORD=1`;
`baseline` is yesterday's binary rebuilt from HEAD in a worktree.

| arm | ms/step | tok/s | record | hand-over | dispatches (GPU) |
|---|---:|---:|---:|---:|---:|
| baseline binary (P1b's code) | 39.20 | 25.51 | 1.111 | 0.824 | 35.88 |
| control (this code, re-recording) | 39.39 | 25.38 | 1.127 | 0.822 | 35.93 |
| **prerecorded** | **37.50 / 37.49** | **26.68 / 26.70** | **0.171 / 0.160** | **0.259 / 0.249** | 35.59 / 35.57 |

Three things worth saying:

- **The control equals the baseline.** Phase A (the arena slot, the shifted
  arena offsets, the buffer-read `SEQ_PAST` in seven kernels) costs nothing
  measurable — decode or prefill (1070.3 vs 1061.5 tok/s at ubatch 2048,
  new code faster within noise).
- **The win is 1.90 of the priced 1.96 ms.** What remains in the two rows is
  real work the fix never claimed: ~0.16 ms of per-token uploads (the
  embedding rows, the n-gram slab, the pads) that used to hide inside
  `record`, and ~0.25 ms of one submit + fence, which is the floor for
  handing anything to the GPU. The `record` row's *recording* — building
  push constants for 1407 dispatches and the cgo command encoding — is gone.
- **On P1b's machine state this is ~28.6 ms a step, ~34.9 tok/s** — the
  +2.2 tok/s P1 priced — but that state was not available to measure (below),
  so the claim stays arithmetic and the measured claim is +1.3 tok/s on a
  day when the whole step was 29% slower.

`results/p1c_decode_attrib.csv` is the prerecorded arm's full table.

## Finding 3 — the environment moved 8.7 ms overnight, and it names its kernels

P1b ended with a measured **1.33 ms environment gap** and the note that a
dispatch inside a step is 8-22% slower than alone on a cold bank. Today the
environment demonstrated a much larger degree of freedom: **yesterday's
binary, unchanged, measures 39.20 ms a step where it measured 30.51** — and
the inflation is not uniform:

| label | P1b, us | today, us | ratio |
|---|---:|---:|---:|
| dn.qkv | 114.9 | 206.2 | 1.79 |
| head.head | 1964 | 3531 | 1.80 |
| attn.qkv | 95.6 | 181.9 | 1.90 |
| dn.out | 45.9 | 72.8 | 1.59 |
| hc.up | 16.8 | 26.2 | 1.56 |
| moe.up | 106.9 | 108.9 | **1.02** |
| moe.down | 72.2 | 69.8 | **0.97** |
| moe.shexp.up | 26.4 | 25.0 | **0.95** |

**Every dense-bank kernel is 1.6-1.9x slower; every MoE expert-bank rung is
unchanged.** The split is exactly the staging path: the dense banks are built
through host floats into buffers allocated early, the expert banks are staged
byte-for-byte into per-layer buffers allocated last. The machine at
measurement time: 32 days up, swap 100% consumed, 78 GB free. The suspicion
is allocation-time physical fragmentation of the dense banks' GTT pages —
huge-page-assembly failing on a long-lived machine — but with no root access
it stays a suspicion. Two consequences worth recording:

- **A cross-day tok/s comparison on this machine is not a measurement.**
  D16 said a rung above the bus is an L3 measurement; this is the same rule
  one level up: a whole-model number is only comparable against a control
  staged the same hour. Every number in this file obeys that.
- The 53 tok/s honest ceiling is a *good-day* ceiling; on a day like today
  the same arithmetic gives ~41. Whatever assembles the dense banks'
  physical pages is worth a look if the machine cannot be rebooted between
  serving sessions.

## Finding 4 — the first run of a staged graph is not the second

Found while gating: on one staged graph, **run 1 diverges from every later
run** of the same tokens at ~1.6e-5 relative in the logits from about the
fifth decode step — prerecording or not (control vs control reproduces it);
runs 2 onward are bit-repeatable, and the replay arm is bit-identical to
run 2+. Greedy tokens don't move. Pre-existing, arena-content-dependent
(fresh zeros against a previous run's leftovers in masked cells or pooled
blocks), harmless to every gate this vertical runs — but it is why
`TestPrerecordedDecodeIsTheRecordedDecode` warms the graph once before
comparing, and it is written down here so the next person who diffs run 1
against run 2 doesn't chase it as a regression.

## Files

- `shaders/llm_common.glsl` — `SEQ_PAST` is `actu[0]`; the doc block carries the contract
- `llm/gpu_attn.go`, `llm/gpu_deltanet.go`, `llm/gpu_ple.go` — the slot as first allocation, `writeSeq` in `SetPast`/`Reset`
- `vk/shim.c`, `vk/shim.h` — `shim_prerecord_multi`, `shim_submit_prerecorded`
- `vk/engine.go` — `vk.Prerecorded`, `Buffer.WriteUint32At`
- `llm/graph.go` — the capture in `flush`, `extendPrerecorded`, `Prerecord()`/`LLM_NO_PRERECORD`
- `llm/prerecord_test.go` — the diff instrument and the bit-level gate
- `results/p1c_decode_attrib.csv` — the prerecorded arm's attribution
