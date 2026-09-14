# Stage 4c — the whole DiT resident, and a weight arena that does not fit one buffer

**Result: all 34 DiT blocks live on the device at once — 12.54 GB of weights
across three storage buffers — and one denoising step is 1.60 s, which is
12.8 s per image** at 1024x1024 and 8 steps. The per-block time inside the
stack is **49.27 ms at 4096 tokens**, the same 49.3 ms stage 4b measured for a
single block on its own: splitting the arena costs nothing per dispatch, and
a run with the weights in six banks is **bit-identical** to the same run with
them in one.

The stage's whole problem was addressing. A block's arithmetic did not change
— it is the same 18 dispatches 4b left — but 34 of them share one set of
pipelines and one pair of activation arenas, their projection weights do not
fit in a single Vulkan storage buffer (4.29 GB here, against 12.0 GB of
weights), and every block is reached by an offset plus a bank. Each of those
is a way to produce a completely plausible wrong tensor, which is what the
controls below are for.

## The layout

| | | |
|---|---|---|
| shared, one copy | fp32 activations | 0.87 GB at 4224 tokens |
| | fp16 activations | 0.42 GB |
| | fp32 weights: the rotary table, and every block's five norms and adaLN projection | 0.51 GB |
| | every non-GEMM pipeline | 11 elementwise builds, plus attention |
| per **bank** | one fp16 storage buffer, and the GEMM pipelines that read it | 3 banks: 4.25, 4.25, 3.54 GB |
| per **block** | eight fp32 offsets, seven fp16 offsets, a bank index, a modulation flag | 34 rows |

**Why a bank is a pipeline.** The buffer a GEMM reads its weight out of lives
in the pipeline's *descriptor set*, not in its push constants, so a second
bank is a second pipeline — the one thing about the split that is not free.
It costs almost nothing in practice: only the two GEMM builds read binding 3,
so the real stack compiles six pipelines instead of two, and nothing changes
in the dispatch path, because `vk.DispatchMultiTimed` already binds each
dispatch's own descriptor set (that is what IDEAS §1.12 built it for).

**Why the bank split is greedy and uneven.** A block is 353.9 MB, so twelve
fit a 4.29 GB buffer and the 34 come out 12, 12, 10. Nothing tries to balance
them: what has to hold is that a block is in exactly one bank, since a
projection's address is a single uint32 into one bound buffer.

**Sizing a weight for a layout it will not be read in cost 0.35 GB.** The
single-block code sized each weight for the largest of the three B layouts, so
that the `wrongBLayout` negative control could stage a weight in a layout its
kernel does not expect without running off the end. Across 34 blocks that pad
is 2.9% of the arena and 0.35 GB of memory for a debug switch, so it is now
applied only when the control is on.

## The unmodulated block

Two of the 34 — the context refiners — are built by diffusers with
`modulation=False`, and stage 4c is where that had to be implemented: no adaLN
projection in the weight arena, no scale on either branch input, both
residuals ungated. It is the block with four of its inputs missing rather than
a different module, so the graph drops the `adaln` dispatch (17 instead of 18)
and tells four sites not to read a modulation vector, through the same
`pc.aux2` flag `dit_norm_scale_f16.comp` already had. `dit_gate_add.comp` and
`dit_norm_gate_add.comp` gained it; nothing else changed, and every other
`.spv` in `shaders/` is byte-identical after `go generate`.

## Three phases at three lengths

The transformer does not run its 34 blocks over one sequence, and stage 4b's
"34 x 8 x 49.3 ms" budget quietly assumed it did. From
`ZImageTransformer2DModel.forward`: the **noise refiners** run over the image
tokens, the **context refiners** over the caption, and the **30 layers** over
the two concatenated. So 30 of the 34 blocks run at `4096 + caption`, not at
4096, and two of them run at the caption length alone.

A stack therefore has to take a run length shorter than the one its arenas
were built for. That is safe here for a reason worth recording: the rows past
the sequence hold whatever the last longer run left there and nothing reads
them, because a GEMM's rows are independent and the attention kernel masks its
key tail against `pc.tokens` rather than trusting the pad to be zero (stage
3c's `NO_TAIL_MASK` control). The packed q/k/v plane stride stays the arena's,
so a short run writes a prefix of each plane.

## The measurement

`go run ./cmd/ditstack -image 4096 -caption 128`, best of three, GPU
timestamps per dispatch, run twice (and a third time with `-v`). The runs
agree to **0.997-1.008** on every cell.

| phase | blocks | tokens | per block | total | GEMMs | attention | TFLOP/s |
|---|---|---|---|---|---|---|---|
| `noise_refiner` | 2 | 4096 | 49.27 ms | 98.6 ms | 71.6 | 13.6 | 34.6 |
| `context_refiner` | 2 | 128 | 2.60 ms | 5.2 ms | 4.9 | 0.03 | 17.5 |
| `layers` | 30 | 4224 | 49.81 ms | 1.49 s | 1.08 s | 204.8 ms | 35.5 |
| **step** | **34** | | | **1.60 s** | | | |
| **image**, 8 steps | | | | **12.8 s** | | | |

One layer's 18 dispatches at 4224 tokens, from `-v`:

| | ms | | ms |
|---|---|---|---|
| `adaln` | 0.17 | `gemm o` | 3.53 |
| `attn in` | 0.69 | `gate msa` | 0.90 |
| `gemm q` / `k` / `v` | 3.31 / 3.16 / 3.31 | `ffn in` | 0.53 |
| `qkpack q` / `k` | 0.71 / 0.74 | `gemm w1` / `w3` | 8.37 / 8.36 |
| `pack v` | 0.62 | `swiglu` | 1.40 |
| `attention` | 7.42 | `gemm w2` | 8.65 |
| `narrow ctx` | 0.42 | `gate mlp` | 0.85 |

**Wall clock is +0.3-0.4% over the sum of the dispatch timings.** A step is
612 dispatches submitted in batches of eight — 77 fence waits — and against
1.6 s of GPU work that is 5 ms. The batching is not an optimisation to
revisit; it is there because a command buffer holding the whole graph outlives
the driver's reset watchdog (stage 2).

**Residency.** 12.54 GB against the 83.79 GiB device-local *and* host-visible
heap this APU reports. The text encoder and VAE are another 8.5 GB as fp16, so
the whole pipeline fits with room to spare — which was never in doubt for the
memory, only for the buffer.

**Load.** 15.1 s to read 24.6 GB of fp32 off disk, narrow 6.0e9 values to fp16
and pack them into the layout the matrix cores read. Packing is parallel over
output rows, which is worth **1.56x** (23.5 s at `GOMAXPROCS=1`); what is left
is the checkpoint read and `LoadBlock`'s own widening of each tensor to fp32,
neither of which is parallel. Peak host memory is one block, 724 MB: the stack
loads, packs and drops them one at a time, because all 34 at once is 24.6 GB.

## Correctness

The reference is `reference/dump_dit_stack.py`: six real blocks with their
real weights, applied in sequence, four modulated and two not. Straight
composition is not the transformer's topology — that is stage 6 — but it is
exactly the question this stage adds.

| check | result |
|---|---|
| each of the six blocks, on the *reference's* own input | 5.1e-2, 1.5e-2, 7.6e-2, 2.6e-2, 1.5e-3, 6.2e-3 — all inside the single-block bound (8e-2) |
| the six chained on the device | **1.8e-1**, bound 4e-1 |
| the same six chained on the **CPU** in fp32 | **4.2e-4**, bound 2e-3 |
| six banks against one bank | **every one of 1,228,800 elements identical** |
| the 34-block bank plan, without a device | every bank inside 4.29 GB, every block in one bank, no two projections overlapping |

The three negative controls are all addressing, since that is what the stage
added. Each leaves every kernel, shape and weight exactly as it was:

| control | rel | x the bound |
|---|---|---|
| the blocks run in reverse order | 198 | **496x** |
| every block reads block 0's weights | 105 | **263x** |
| one block per bank, every block reading the *next* bank | 140 | **351x** |

## What it says

**1. Splitting the weight arena is free, and that is a measurement, not an
argument.** 49.27 ms per block inside a three-bank 34-block stack against
49.3 ms for one block in its own buffer, and six banks bit-identical to one.
The cost of the split is entirely at build time: one extra GEMM pipeline per
bank.

**2. The fp32 CPU chain is what makes the fp16 chain's error readable.** Six
chained blocks drift to 1.8e-1 of the tensor's RMS, over twice the bound a
single block gets, and there is no way to tell compounding rounding from a
wiring mistake by staring at it. The same chain in fp32 lands at 4.2e-4 — the
worst element is 1.6e-5 of itself — so the block order, the unmodulated
variant and the weight addressing are right independently of the narrowing,
and what is left is arithmetic. Running each block on the *reference's* input
rather than the stack's is the other half: in isolation every one of the six
is inside the single-block bound.

**3. The per-image budget was 4% pessimistic, for two reasons that nearly
cancel.** 34 x 8 x 49.3 ms = 13.4 s assumed every block sees 4096 tokens. Two
of them see 128 instead (2.60 ms a block, 19x cheaper), which is worth -0.7 s;
but 30 of them see 4224 rather than 4096, which gives +0.1 s back. The
measured figure is 12.8 s.

**4. A block is sub-linear in tokens between 4096 and 4224.** 49.27 ms to
49.81, +1.1%, for +3.1% of GEMM rows and +6.4% of attention. Not explained.
The candidate is §5.1b: an activation tensor's row stride is the same at both
lengths, but the *number* of rows sets how a grid's workgroups tile over the
channels, and 4096 is the one value in this model that is a power of two.
Worth an hour before anything is concluded from a single 2.6% cell.

## What this leaves open

- **The stack is 72% GEMM, 14% attention, 14% elementwise, unchanged from one
  block.** At
  12.8 s of an 18.4 s image the next-largest item is still the **VAE's 5.6 s**,
  which has had no optimisation pass at all.
- **`attention` at 4224 tokens is 7.42 ms, 15% of a block**, against the
  GEMMs it used to lead. Stage 4's open list already had it; the stack makes
  it 1.7 s of the image.
- **One rotary table, three phases.** `GPUStack.SetRoPE` rewrites it between
  phases (4 MB against 12 GB), which is the right shape for stage 6, but
  nothing has yet run the three phases with their own position ids — the
  timing run uses a prefix of the unified table, which is the same work.
- **Load is 15 s and none of it is the device.** If that matters it is
  `LoadBlock`'s fp32 widening, not the packing.
