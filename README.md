# strix-halo-vulkan

A Go + Vulkan compute harness for exploring shaders on the Strix Halo iGPU
(gfx1151 / AMD Radeon 8060S, RADV driver), plus a benchmark suite
(`cmd/bench`) that measures the operations neural-network inference is
built from — GEMM, GEMV, elementwise/activation, softmax/RMSNorm — across
the implementation "flavours" that matter on this hardware: naive vs.
shared-memory-tiled vs. subgroup-reduced vs. cooperative-matrix
(`VK_KHR_cooperative_matrix`, RDNA3.5's matrix-multiply accelerator), and
fp32 vs. fp16 vs. quantized (int8/int4, GGML-style block scales) weights vs.
W8A8 and W4A8 (int8 or 4-bit weights against int8 activations, reduced via
`VK_KHR_shader_integer_dot_product`'s packed-dot instructions).

Two kernels here are the current answers for the two shapes inference comes
in. For **decode** (GEMV, memory-bound), **W4A8**
(`shaders/gemv_w4a8.comp`): 4-bit weights fed to the packed-int8 dot
instruction through its mixed-signedness overload, at **239 GB/s against
DRAM-resident weights — 101% of this machine's 236 GB/s measured memory
bandwidth**, 2.4x the int8-weight kernel it replaces. The last 13% of that is
one compile-time constant. Size the load width so that a single lane-step
covers a whole weight row — `VEC = N/(8·WAVE)`, i.e. `uvec4` at N=2048, two
at N=4096, four at N=8192 for a wave64 pipeline — and the kernel reads
**99-103% of the bus at every one of those N**: 243, 239 and 235 GB/s. Get it
wrong and it does not: the same kernel at one `uvec4` reads 164 GB/s at
N=8192, 70% of the bus. This is the `stride` family's coverage law (below)
applied from inside a kernel, with the weight row as the stride, and it is
what an earlier reading of the same data mistook for a **wave32** effect —
holding the bytes-per-lane-step fixed and varying the wave size instead
measures the same bandwidth to 0.05%, so the wave32 pin that used to be
recommended here is withdrawn (IDEAS §1.7). For **prefill** (GEMM,
compute-bound), the register-blocked cooperative-matrix GEMM
(`shaders/gemm_wmma.comp`): **39.0 TFLOP/s, 70% of this chip's measured
55.5 TFLOP/s of matrix-core throughput** (30.6 TFLOP/s at N=4096), 7.6x the
straightforward coopmat kernel it replaces at that shape. Two thirds of that
came from neither tiling nor instruction selection but from how the operands
meet the memory system. First *stride*: every fragment load in a WMMA GEMM is
K-strided, a power-of-two leading dimension aims all the addresses of one
load at the same memory channel, and padding each stride 256 B off a 4 KB
multiple — a host-side allocation change — is worth up to 1.35x on the
kernels that were already winning and 2.6x on the [N,K] weight layout. Then
*concurrency*: issuing a whole K-slab's fragment loads before the slab's
first MMA, rather than one K-tile's worth at a time, is worth a further
**2.1x** for identical bytes, identical instruction counts and an identical
tile — only the scheduling moves, from 16 loads in flight to 45. The same
slab depth *without* that change leaves 16 in flight and is worth nothing,
which is what says the variable is concurrency rather than depth. Together
they make the [N,K] layout a real `Linear` weight already has the fastest
one, where it used to be 2.8x the slowest.

The `stride` family then measured that memory system directly
(`shaders/strided_read.comp`) and found a second, larger effect with no
kernel-side cause at all: channel interleave on this part is 256 B across
sixteen channels, so a row shorter than `gcd(stride, 4096)` bytes **never
addresses some of them**, and the achievable fraction of DRAM bandwidth is
exactly `min(1, rowBytes/gcd(stride, 4096))`. A K=1024 fp16 weight matrix
whose rows were aligned to 4 KB reads at **half** this chip's bandwidth; 1 KB
rows at an 8 KB stride, a **quarter** — and unlike the aliasing above this
hits a plainly contiguous read as hard as a gather.

Swapping the probe's traversal so consecutive *waves* start on consecutive
rows — the same bytes, the same instructions, the same checksum — then
generalised that into one law about concurrency: the achievable
fraction of peak is `min(1, C/gcd(stride, 4096))` where **C is the
contiguous run of bytes the requests in flight at one moment hold in a
row**, which is the whole row only when consecutive waves walk along it. A
*densely* packed tensor is not safe after all: 1 KB rows packed at a 1 KB
stride, read with 256 B of each row in flight, run at **61 of 243 GB/s**,
with no padding anywhere to blame. One wave streaming a *whole* row
— the GEMV kernels' shape — is clean at every stride, and 4 KB of a row in
flight per wave restores full bandwidth whatever the stride is. What is
exposed is tiling: a 1 KB-wide panel of a K=4096 fp16 matrix gets a quarter
of the bus. Pad every row stride to 256 B past a multiple of 4 KB and all
three effects go away, for every panel width and every traversal — and where
the stride is not the engine's to choose, depth in the inner loop buys the
same thing, which is the 2.1x the GEMM above collects. `IDEAS.md`
explains how each kernel gets where it is and what is still on the table.

All of the above was measured on square N x N x N, which no transformer layer
is. The `shapes` family runs the same kernels over the actual (M,N,K) triples
of the models in `GOALS.md` — read out of their own configs — and two of the
answers change. **Decode gets easier**: every reduction length those models
use is a non-power-of-two, so its 4-bit weight row has a *smaller* `gcd` with
the 4 KB channel rotation than the square sweep's rows did, and the load width
stops mattering. At K=2560, the width every Qwen projection reduces over, the
kernel reads 239-242 GB/s at *all four* built widths including the narrowest
— where at the square sweep's N=8192 only the widest got there. Only K=1024
still needs a `uvec4` (211 → 239 GB/s), and nothing in any of the five models
needs a load wider than the kernel already has. **Prefill gets a second
kernel**: the suite's headline GEMM wins every shape with 1024 or more tokens
in the batch and *loses every shape below that*, by up to 2.8x, to the
small-tile wave32 variant the square sweep only ever crowned at N=1024. Real
batches are below it constantly — 128 caption tokens, 384 audio frames, and
40 tokens per expert when a 2048-token prompt is routed across 512 of them.
The crossover is in M alone, and a sweep with M tied to N and K cannot see
it.

The one shape none of that covers is the **MoE** FFN: qwen3.8-flash-next
routes each token to 10 of 512 experts, so its feed-forward block is 512
GEMMs over 40-row groups rather than one GEMM, and at prefill it is 1440 of
the 1861 matmuls a token costs. The `moe` family builds it — the same WMMA
kernel with its tile origin read out of a table instead of derived from
`gl_WorkGroupID`, so one dispatch covers every expert — and measures it
against the expert-at-a-time loop it replaces, plus the gather and combine
passes it makes necessary. Three answers, none of which was the expected one.
**Grouping is worth 1.1-4.1x, and it is an occupancy effect, not a launch
one**: sort the cells by how many workgroups one expert's dispatch launches
and the speedup falls monotonically from 4.1x at 10 workgroups to ~1.1x at
80, which is exactly the two-waves-per-CU this part needs to issue WMMA at
rate — the 300 ns launch itself is nothing against a 20 µs dispatch. **Tile
padding is real and costs nothing**: a 16-row tile takes an expert's group
from 62% useful to 84%, and is *slower*, because at these shapes the binding
constraint is weight bytes and a smaller tile reads more of them. The grouped
kernel moves 190 GB/s of a 236 GB/s bus at the real prefill shape, so there is
no room for the FLOPs to matter. And **the "pad every stride by 256 B" rule
this suite has been carrying is the wrong statement of itself**: sweeping
`gcd(row stride, 4096)` over every power of two at two non-power-of-two
reduction lengths gives a plateau at 128-256 B and a cliff on both sides, so
what matters is landing the gcd at 128-256 B — which the 640-wide down
projection already does unpadded, and which padding it by 256 B *breaks*, for
1.19x.

That left MoE prefill with exactly one lever — it was at 93% of the ceiling
its *format* implies, and that ceiling is 5.03 GB of fp16 expert weights per
block. Holding the same kernel and the same schedule and putting the bank at
**4 bits** (`shaders/gemm_wmma_q4.comp`) takes one MoE block from 28.13 ms to
**13.37 ms** and a 48-layer prompt chunk from 1.35 s to **0.64 s**. Removing
4x of the bytes buys 2.1x of the time, and the gap is the finding: the phase
stops being a memory problem partway through the change. Weight traffic goes
from 190 GB/s of a 236 GB/s bus to 113, while the padded tiles' own rate goes
from 22% of the matrix cores to **51%** — half of each ceiling, pinned
against neither, with the dequant loop body left as what binds.

Three things that cost more than the arithmetic did. **The operand has to
round-trip through LDS**, because no cooperative-matrix extension exposes a
fragment's lane layout — `m[i]` is defined, which (row, col) it means is not
— so 4-bit weights cannot be unpacked into registers and declared a fragment.
**That LDS tile's rows must be padded off the 32-bank rotation**: a slab row
is 128 B, exactly one rotation, so a 16-row fragment load hits one bank
sixteen times, and eight halves of pad are worth **1.68-1.83x** for 28 bytes
of code. And **the quantization block size is worth 1.24x between two
binaries that are instruction-for-instruction identical** — same VGPRs, same
code size, same VALU and VMEM counts, 4.4% of nominal bytes standing behind a
24% difference — which is still unattributed. Two predictions did come true:
the 16-row tile that *lost* at fp16 because padding cost FLOPs and FLOPs were
free now wins, and grouping is worth *more* than it was (5-8x at ten
workgroups per expert dispatch, against fp16's 3-4x), because the idle time
an under-filled dispatch leaves is fixed and a faster kernel makes it a bigger
share. One rule got amended instead: **the stride window above stops paying
once a kernel is not bandwidth-bound**, since padding into it still costs
bytes — on the down projection, climbing from its natural gcd of 64 into the
128-256 B plateau costs 1.20-2.40x the traffic and measures slower.

All of that is prefill. **Decode** on the same block was, until now, measured
with that same grouped GEMM run at one row per expert — 15 of every 16 rows of
each cooperative-matrix tile being padding, and the 8 MB an expert slice
occupies sitting inside the 32 MiB MALL, so the timing loop re-read it out of
cache. The honest kernel is the W4A8 GEMV with the same table
(`shaders/gemv_w4a8.comp -DGROUPED=1`): one subgroup per output row, one
workgroup per (row, routed pair), and **no gather pass at all**, since the
gather exists only to make an expert's rows consecutive for a fragment load
and a subgroup reads whichever row the table names. It is **1.3-3.3x the GEMM
at M=1** — same weight bytes, but 2-6 GB/s of activation traffic against 17-41
and no 7-11%-useful tiles — reads **235 GB/s of distinct weight bytes, 100% of
the bus**, and takes one token's MoE FFN from 9.56 ms to **5.47 ms across 48
layers (183 tok/s), 91% of the 5.00 ms its 1.18 GB of 4-bit weights cost at the
bus**. Three things fell out of it. **Grouping is worth 1.29-1.81x even though
every dispatch it merges already fills the machine**, because what a barrier
costs a memory-bound dispatch is the *drain* — 1163 ns median, against the
300 ns an empty shader's launch measures. **The crossover back to the GEMM is
at about 1.2 rows per expert**, not at the 16 a tile holds, because the
question is how many rows of *one* expert routing supplies. And, reversing the
rule above once more, **an expert bank should not be padded at all**: with
every stride 64 B-aligned so the pad is never read, the unpadded stride wins at
both projections — including the down projection's gcd-64 row, which the
plateau says should be the slow end — and padding into the window costs
1.12-1.15x. The plateau is a property of a *tiled* read, where a fragment holds
32 B of a row; a kernel that walks whole matrices just wants them contiguous.

That 1.2-rows-per-expert crossover is the one thing a serving engine cannot
live with, so the same kernel took an **M block** (`-DMROWS=n`): a workgroup
loads an expert's weight row once and dots it against *n* routed tokens, each
with its own accumulator and output row, with short groups padded out to keep
the inner loop branchless. At 256 sequences in flight — 5.05 rows per expert —
it is worth **1.39-1.66x**, and it takes the FFN to **694 tok/s against the
GEMM's 546**.
At *one* row per expert it is worth **0.39x**, which is the other half of the
result: the block has to be picked from the routing, not compiled in once, and
the rule is arithmetic — a fitted cost model puts one group (an expert's
845 KB through all its output rows) at **6.5-10.5x one slot**, so blocking
pays as soon as it removes one group per 6-10 pads it creates, i.e. `MROWS` ≈
the mean rows per expert. What the block actually takes off is **not the bus
but the cache**: unblocked at five rows per expert the kernel asks for
**746 GB/s**, 3.2x the bus, while DRAM underneath it runs at 148 — four fifths
of the requests are re-reads the MALL is serving, and that is what sets the
time. Blocked, issue falls 4.7x and DRAM reaches **206 GB/s, 87% of the bus**.
The GEMM crossover moves out 4x and splits by projection: the `down` tie at
t=64 becomes a 1.16x win, and at t=256 `gate_up` stays 1.70x ahead of the GEMM
while `down` goes to it, because a 16-row tile covers 5.05 rows in one pass
where a compile-time block of 4 needs 1.62. Single-stream decode is unchanged
at 5.43 ms and 184 tok/s — this is a throughput lever with no latency effect.

The *other* axis of that block is what closed the one cell this suite had left
open on the decode path. The `down` projection read **198 GB/s** per distinct
expert where `gate_up` read **235**, for the same 845 KB of weights either
way; what differs is the aspect ratio, 2560 rows of 640 against 640 rows of
2560, so down launches 4x the workgroups and issues 4x the output writes. Two
arms that divide the workgroup count *identically* separate the causes:
packing several subgroups into a workgroup (`-DROWS=n`, same waves, same
loads, same stores) is worth **1.02x**, while giving one subgroup several of
an expert's weight rows against one activation row (`-DNROWS=n`, so the wave
count falls too) is worth **1.17x** and takes down to **98% of gate_up's
rate**. Fitting `T = A/n + B` prices the difference exactly: **19.8% of down's
time is a fixed cost per output row** — one `subgroupAdd`, one store, one
activation row, paid four times as often against a quarter of the loads — and
**-0.4% of gate_up's is**. It is not the lanes, twice over: halving the wave
is a wash, and the one build that actually *fills* down's 64 lanes (VEC=1, 80
loads per row) is **1.05x slower** than the VEC=4 build whose 44 idle lanes
the N block leaves exactly as idle as it found them. The sizing rule is the
load-width rule read along the other axis of the matrix — **one lane-step
should cover a weight row (`VEC`), and one wave-step should cover the lanes
(`NROWS >= 8*VEC*WAVE/K`)** — which is 4 at K=640 and 1 at K=2560, as
measured. Unlike the M block this one takes nothing from routing, never pads,
and is a **latency** lever: a single token's MoE FFN goes to **5.15 ms over 48
layers, 194 tok/s, 97% of its 5.00 ms bus floor**, and `down` beats the Q4
grouped GEMM by 1.38x at 64 sequences where the M block managed 1.16x.

Those two blocks are the two axes of one kernel, and **putting both in one
wave is where a serving batch actually sits**. A workgroup takes `MROWS` of an
expert's routed tokens and each of its subgroups takes `NROWS` of that
expert's weight rows, so every weight load is dotted `MROWS` times and every
activation load `NROWS` times. They **compose**: at 256 sequences in flight it
is **1.12x each of its own axes on `gate_up` and 1.19x on `down`**, **1.10x
the best single-axis build of any width in the sweep** at both shapes, and
1.54x/2.11x the unblocked kernel. At 16-64 tokens it ties the best
single-axis build within 1% — those cells are already at the bus — and below
that, where the M block is a real loss, the corner loses with it. What that buys is the last of the decode path's
bandwidth: `gate_up` reads **228 GB/s of distinct weight bytes, 96% of the
bus**, where the M block alone reached 87% and the unblocked kernel 63%, and
the FFN runs at **818 tok/s against the Q4 grouped GEMM's 545** — so `down`,
which the M block had handed to the GEMM at five rows per expert (0.77x),
comes back to **0.98x** there and `gate_up` to 1.88x. Three things fell out of
it. An N block makes the M block's **pad slots 4x cheaper on `gate_up` and
1.2x on `down`** — a pad costs a per-wave part, which `NROWS` divides, plus a
per-output-row part, which it does not, and down's slot is 2560 output rows
against gate_up's 640 — so the fitted "block when it removes one group per
6-10 pads" becomes one per **45** pads on gate_up and one per 6.5 on down, and
the M block may be sized *above* the mean rows per expert where its rows are
few. The row block **partly substitutes for the load width**: `NROWS=4` is
worth 1.67x at VEC=1 against 1.37x at VEC=4, and `VEC=1` + `NROWS=4` is the
fastest single-axis build at gate_up's 256-sequence cell — a width §1.7's rule
calls the worst available. And the corner **never spills**: 83 VGPRs at
(4, 4) against the plain kernel's 47, with no scratch access anywhere in the
family, so the register competition the arm was built to look for does not
appear. Single-stream decode is unchanged at 195 tok/s, since the M axis is a
loss at one row per expert.

The last cell under the bus — `down` at 256 sequences, 66% of it — closed by
taking the M block out of the binary. `MROWS` is a compile-time constant and
routing is not: at 256 sequences an expert gets 5.05 pairs on average with a
tail out to 12, so a compiled block of four spends **23% of its slots on
padding**. A two-term cost model (`T = groups*w + slots*s`, fitted from the
rows the sweep already produces) reads the width off the routing histogram
instead, and it **picks the width that actually won in 28 of 40 cells and is
within 1% in 32**, against a 1.11-1.66x spread across the four built widths —
and **calibrating it once, at the cheapest batch there is, is as good as
re-fitting it per batch**. Then the width stops having to be one number at all:
a group's slots must all name one expert, but the width is fixed per
*dispatch*, so bucketing the groups by width and issuing one dispatch per
bucket covers the histogram with several widths at once, out of the binaries
that already exist. On `down` at a serving batch that is worth **1.09-1.13x**
over the best fixed width — 2.761 ms to **2.530 ms**, 157 to **169 GB/s** of
distinct weight bytes, 66% to **72%** of the bus — and takes a token's MoE FFN
at 256 sequences to **848 tok/s** against the Q4 grouped GEMM's 545, with
`down` itself back from the GEMM it had gone to (0.99x → **1.07x**), so both
projections are the GEMV's at every batch measured.

The interesting half is the plan that *loses*. Two ways to cover an expert
routed five pairs: one group of eight (three pads, one weight read) or a four
and a one (no pads, two weight reads). The pad-minimal plan gets the slot count
down to a 5% pad where the fixed build ran 23% — and it is **0.72-0.96x**,
losing by up to 1.4x, while the plan that gives each expert a single width
repeated wins. What separates them is the one quantity a cost model written in
groups and slots cannot see: whether an expert's groups land in the same
dispatch. Pricing the residual against a count of the ones that do not gives
**3.3-5.9 µs per crossing, against 3.58 µs for a cold DRAM read of that
expert's 845 KB**, where a group *inside* the part is priced at 0.45-1.67 µs.
So a group is not a fixed cost: one whose expert the previous dispatch just
read is a cache hit, and one whose expert was last read a dispatch ago pays the
bus again — the same mechanism the M block was buying all along, seen from the
side where it can be lost. Barriers between the parts cost +0.2-1.3%, so it is
locality and not scheduling. **An expert belongs to exactly one dispatch**, and
the fitted "a group is worth 45 pad slots on gate_up, 6.5 on down" turns out to
be a *cache-resident* number — at 256 sequences it is 1.0-2.7 on both.
What is left of down's deficit is not the table: it is the per-output-row cost
inside the wave, paid over four times as many rows, which is the one kernel arm
this file has never built.

No third-party Go modules — `go.mod` has no dependencies. Vulkan access is a
hand-written cgo binding straight against the system Vulkan loader
(`vulkan/vulkan.h` + `libvulkan.so`), with the struct-heavy parts of the
Vulkan API built in a small C shim (`vk/shim.c`/`vk/shim.h`) rather than Go,
since cgo forbids passing a Go struct across the boundary when one of its
fields is itself a pointer into other Go memory — and Vulkan's
`*CreateInfo` structs are built almost entirely out of such chains.

## Layout

- `vk/` — the engine: `shim.c`/`shim.h` do the actual Vulkan calls
  (instance/device setup with optional fp16/int8/cooperative-matrix/
  subgroup-size-control device features, buffer allocation preferring
  device-local+host-visible memory, N-buffer pipeline creation with push
  constants, specialization constants and an optional required subgroup size,
  GPU-timestamp-timed dispatch); `engine.go` is the idiomatic Go wrapper
  (`Instance`, `Device`, `Buffer`, `ShaderModule`, `ComputePipeline`).
- `main.go` — the original demo: picks the Strix Halo iGPU, uploads a float
  array, runs `shaders/double.comp`, reads it back, verifies it.
- `shaders/*.comp` — the compute shaders (GLSL); `shaders/shaders.go`
  embeds their compiled SPIR-V via `go:embed`. Precision/tile-size variants
  of the same source are generated via `glslc -D` flags, not duplicated GLSL.
- `cmd/probe/main.go` — dispatches a single `.spv` once, so
  `RADV_DEBUG=asm` / `RADV_DEBUG=shaderstats` can be pointed at any shader to
  read its disassembly or its VGPR/LDS/spill counts. An optional second
  argument pins the wave size the shader is compiled for, since the register
  file is per lane and the VGPR count is a different number at each. Note that
  RADV caches compiled pipelines, so a repeat probe of the same shader prints
  nothing — set `MESA_SHADER_CACHE_DISABLE=1`. Compilation-time questions
  only; it binds dummy buffers and computes nothing meaningful.
- `bench/` — the benchmark harness: GPU-timestamp-based timing
  (`bench.go`), fp16/int8/int4 quantization helpers matching GGML-style
  block scales (`quant.go`), clock/power instrumentation (`sysmon.go`), and
  one file per op family (`ops_*.go`) wiring shader variants + buffer
  layouts + a CPU-reference correctness check to a size sweep.
  `modelshapes.go` is the odd one out: not code but data — the weight-matrix
  dimensions of the models in `GOALS.md`, with the config each was read from.
  `ops_moe.go` is the only family that needs more than one dispatch shape per
  measurement: its per-expert baseline is a command buffer of 512 dispatches
  with differing push constants, which is `vk.ComputePipeline`'s
  `DispatchSequenceTimed`. `ops_moe_q4.go` is its 4-bit arm — the same
  routing, tile table and schedules against a Q4 expert bank, so the two
  formats' rows subtract. `ops_moe_gemv.go` is the decode kernel and its two
  compile-time blocks, and `ops_moe_select.go` is what replaces them with a
  choice: a cost model fitted from those rows, a plan that covers the routing
  histogram with several block widths at once, and `vk.DispatchMultiTimed` —
  the one timed sequence in the harness whose dispatches differ in their
  *pipeline* rather than just their push constants.
- `cmd/bench/main.go` — the benchmark CLI; `bench/families.go` is the
  registry of targetable op families it dispatches to.
- `results/` — one CSV per op family, the committed measurements.

## Requirements

- Go, gcc/clang, `pkg-config`
- Vulkan loader + headers (`vulkan-icd-loader`, `vulkan-headers` on Arch)
- `glslc` (from `shaderc`) to compile shaders — only needed when a `.comp`
  file changes, not to build/run the Go programs themselves.

## Usage

```sh
go generate ./...   # only needed after editing a .comp file
go build ./...

./strix-halo-vulkan   # original demo

go run ./cmd/bench -list          # the op families a run can target
go run ./cmd/bench gemv gemv_cold # run two of them
go run ./cmd/bench all            # the whole suite (~60 min)
                                  # -h for the sweep flags (sizes, blocks,
                                  # warmup/iters, per-family sweeps)
```

A run has to name what it is measuring: the full suite takes well over half
an hour and produces ~2600 rows, so sweeping all of it to answer one question is
mostly waste. `all` is still there for the occasions that want it.

`cmd/bench` prints a results table per op family and writes one CSV per
family into `-resultsdir` (default `results/`) — `results/gemv.csv`,
`results/stride.csv`, and so on. Each file is written as soon as its family
finishes, so an interrupted run keeps what completed, and re-running one
family refreshes only its own file and leaves the rest of the committed
numbers untouched. Every file has the same columns:
`op,variant,weight_format,block_size,size,ns_per_iter,gflops,gbps` followed
by `sclk_mhz,sclk_mhz_min,sclk_mhz_max,power_w,detail`, so
`tail -q -n +2 results/*.csv` reconstructs the single flat table the suite
used to write if an analysis wants it.

The families are the twelve listed by `-list`: `peak`, `overhead`,
`bandwidth`, `stride`, `elementwise`, `gemv`, `gemv_cold`, `gemm`,
`gemm_wmma`, `shapes`, `moe`, `reduce`. `gemm_wmma` — the hand-tuned WMMA
ladder — is separate from `gemm` because it is the larger half of the GEMM
rows and is iterated on by itself. `shapes` and `moe` are the two that sweep
nothing: `shapes` runs the kernels the other families picked over the real
model dimensions in `bench/modelshapes.go` (~5 min), and `moe` runs the
grouped/MoE GEMM over qwen3.8-flash-next's 512-expert bank at fp16 and at
4 bits, plus the grouped W4A8 GEMV the decode half of that block wants
(~18 min). Both ignore `-sizes`/`-blocks`, because their shapes are the
models' rather than the flags'. `moe` also allocates the largest buffers in
the suite — a 512-expert fp16 weight bank is 1.8-4.0 GB depending on the row
stride under test — so it wants headroom rather than a loaded machine; its Q4
GEMM arm runs in a second pass once those are freed and its GEMV arm in a
third, so no two banks are ever live at the same time.

## Reading the numbers honestly

Four op families exist to keep the rest of the suite interpretable. They
are cheap to run and worth running first.

- **`peak`** measures the hardware's instruction-issue ceilings with kernels
  whose operands are register-resident before the loop, so nothing in the
  timed loop touches memory (`shaders/alu_peak.comp`): scalar fp32 FMA,
  packed-fp16 FMA, `dotPacked4x8` (int8 packed dot), and cooperative-matrix
  fp16 and int8. Every other kernel's GFLOP/s should be read as a fraction
  of the relevant one of these — not of a spec-sheet figure extrapolated
  from a different chip. `-cus` (default 40) expresses each ceiling as
  ops/clock/CU, the form comparable against published RDNA3 figures.
- **`overhead`** measures a shader that does nothing, isolating the cost of
  a dispatch plus the pipeline barrier between iterations from any real
  work. It bounds how much a long chain of small kernels (decode) can lose
  to launch cost, and explains bandwidth measurements at sizes small enough
  to be dominated by a fixed cost.
- **`stride`** measures what the memory system delivers for a *fixed* set of
  bytes as the row stride and the request shape vary — the qualifier on
  `bandwidth`, which only ever measures a contiguous sweep and therefore
  reports best-case ceilings a strided kernel does not automatically get. One
  kernel (`shaders/strided_read.comp`) reads every touched 16-byte chunk
  exactly once in every variant, so the byte set and the load count are
  identical and only the stride and the lane→address mapping change;
  `LANES_PER_ROW` sets how many consecutive chunks of a row go to
  consecutive lanes, so one request spans 1 row (contiguous) up to 64 rows (a
  gather); `CROSS_WAVE` swaps the traversal so consecutive *requests* advance
  down the rows rather than along one, which is the only way to put
  concurrent waves a stride apart; and `LOADS_PER_WAVE` gives each request
  more of its row in flight, up to a whole row, which is what a real kernel's
  inner loop does. `-stridepads`, `-striderowbytes` and `-stridefootprints`
  set the three sweeps; the printed grid carries a `model` row giving
  `min(1, C/gcd(stride, 4096))`, and the cross-wave grid under it carries the
  same model per shape, which the measurements track to within 2% wherever C
  is a genuine contiguous run and only indicatively for the gather shapes. Each case is checked against a host checksum
  over its (row, chunk) set, so a mis-derived index fails instead of
  reporting a plausible bandwidth — and the shapes where both traversals
  compile to the same mapping are marked, since those cells measure
  repeatability rather than traversal.
- **`gemv_cold`** measures the decode-shape GEMV against weights several
  times larger than the last-level cache. The square `gemv` sweep's largest
  case holds 33.5MB of fp16 weights (16.8MB at int8, 8.4MB at int4), which
  fits this chip's ~32MB MALL, and then re-reads it hundreds of times — so
  it reports bandwidth above what the DRAM bus can deliver and flatters the
  formats that read *more* bytes, since cached bytes are nearly free. Real
  decode streams a whole model from DRAM per token. `-coldfootprints` sets
  the swept weight footprints in MB (fp16-equivalent, so every format in a
  row holds the same number of weights), `-coldn` the reduction length.
- **`shapes`** is the check on everything above: the same kernels, at the
  dimensions the models in `GOALS.md` are actually made of, because a square
  N x N x N sweep ties three independent variables together and every real
  layer separates them. Its decode arm gives each reduction length a
  DRAM-resident twin — the model's own row geometry, with enough rows that
  the weights cannot fit the MALL — because a single real weight matrix is
  usually small enough to be cache-resident on its own while the model it
  belongs to is not. Its prefill arm pads M and N up to the kernel's tile and
  charges the padding to the result: the reported GFLOP/s is the model's own
  FLOPs over the padded dispatch's time, so a 40-row expert in a 64-row tile
  reads as the 62%-efficient thing it is, with the rate the kernel itself hit
  in `exec_gflops`. K is never padded — that would compute a different
  product — so a variant whose K-slab does not divide the model's K is
  skipped instead. The arm runs a 200-iteration floor rather than the suite's
  20: these dispatches are tens of microseconds, and at 20 iterations a few
  cells per run came back 2x low at a pinned clock.

Three measurement hazards the harness handles rather than leaves to the
reader:

- **Clock.** This GPU idles at 637 MHz out of 2900 (see
  `/sys/class/drm/cardN/device/pp_dpm_sclk`), and host-side work between
  cases — generating and quantizing hundreds of MB of weights — lets it
  drop back. Before each timed measurement the harness drives the GPU with
  ALU-heavy work until the clock reaches 97% of its advertised maximum
  (`-warmclock` bounds how long it will try; it stops as soon as the target
  is reached, so a hot GPU costs one 20ms burst), and records the clock and
  power actually observed during the measurement in every result row. An
  earlier sweep without this reported the same kernel at 190 GFLOP/s and
  522 GFLOP/s depending only on how long the preceding case had kept the
  CPU busy.
- **GPU-hang watchdog.** `bench.TimeDispatch` probes each case with one
  dispatch before batching more into a single timed submission, capping the
  batch so it can't run long enough to trip the kernel driver's GPU-hang
  timeout (amdgpu's default TDR is commonly ~10s) — a slow kernel (e.g.
  naive GEMM at large sizes) is simply measured with fewer iterations
  rather than risking `VK_ERROR_DEVICE_LOST`, which would otherwise poison
  the device for every case still queued.
- **Operand stride.** A sweep over powers of two sweeps, by construction,
  only the strides that alias worst on this memory system. Two distinct
  things go wrong there (IDEAS §2.3, §5.1b): a load whose addresses are all
  one stride apart aims them at a single 256 B-interleaved channel when that
  stride is a multiple of the 4 KB interleave rotation, costing up to 1.34x
  of bandwidth and up to 1.6x on a WMMA GEMM; and — the larger one —
  whenever the bytes the concurrent waves hold in flight cover less of the
  4 KB rotation than `gcd(stride, 4096)`, the remaining channels are never
  addressed at all, costing 2x, 4x or 8x even for a contiguous read and even
  with no padding present. The `gemm` family's WMMA cases therefore take
  both leading dimensions as push constants and carry `strideA`/`strideB` in
  their `detail` column, and the `_pada128`/`_pad*` rows are the same SPIR-V
  at a padded stride — so a row's stride is visible rather than implied by
  its size. Nothing outside that family has been measured against a padded
  stride yet; assume its numbers are the aliased ones.
- **Wave size.** Every kernel whose workgroup *is* one subgroup takes `WAVE`
  as a `glslc -D`, and the pipeline pins the matching size with
  `VK_EXT_subgroup_size_control`'s `requiredSubgroupSize`
  (`PipelineSpec.RequiredSubgroupSize`) — the two must agree or a 32-thread
  binary would run as a half-idle wave64 and `subgroupAdd` would reduce half a
  row. The `*_w32` rows are those pairs; they are dropped whole, with a note on
  stderr, on a device that will not let a pipeline name a size. It is not a
  free knob in either direction: up to 2.1x on a fragment-heavy WMMA tile, and
  0.58-0.9x almost everywhere else (IDEAS §6.2). The 1.07x it used to be worth
  on the DRAM-resident W4A8 decode GEMV was not a wave-size effect at all —
  at equal bytes-of-row-per-lane-step the two sizes measure the same to 0.05%
  (IDEAS §1.7). Rows without `_w32` are at the driver's default of 64.
- **W4A8 load width (`VEC`) and rows per workgroup (`ROWS`).**
  `gemv_w4a8.comp` takes both as `glslc -D`s: `VEC` ∈ {1,4,8,16} is `uvec4`s
  of weights per lane per step (`subgroup`, `_vec4`, `_vec8`, `_vec16` rows),
  `ROWS` is subgroups per workgroup (`_r2`). `VEC` is the one that matters and
  its right value depends on the reduction length — `N/(8·WAVE)`, so a row is
  covered in one step — which is why the rows are swept rather than one
  variant kept. `ROWS` changes nothing anywhere except the single cell where
  `VEC` is one step short of a row, and is kept as the control that showed
  that (IDEAS §1.7).
- **The grouped arm's two blocks (`MROWS`, `NROWS`).** `gemv_w4a8.comp
  -DGROUPED=1 -DMROWS=m -DNROWS=n` gives one workgroup *m* routed tokens of
  the same expert and each of its subgroups *n* of that expert's weight rows
  (`_m2`, `_n4`, `_m4_n4` rows). They combine, and they are sized by different
  things. `NROWS` is a model constant like `VEC` — `8*VEC*WAVE/K`, so 4 at
  K=640 and 1 at K=2560 — and never hurts at the shapes it is sized for.
  `MROWS` has to match how many rows routing gives an expert and is a real
  loss when it does not (0.39x at one row per expert, 1.66x at five), so an
  engine keeps several builds and picks per dispatch. Both require `GROUPED`;
  `NROWS` is capped at 4 when combined with an M block, and `ROWS` (subgroups
  per workgroup) works under `GROUPED` too, as the control that showed the
  cost is inside the wave rather than in the grid (IDEAS §1.9, §1.10, §1.11).
- **Cache-resident measurements are contended.** A MALL-resident working
  set is shared with everything else touching memory — the display this iGPU
  also drives, or a second benchmark process — and losing part of it drops a
  case towards DRAM speed. With another benchmark running alongside, one in
  ten MALL-resident `stride` cases lands at 0.3-0.5x, on a different cell
  each run, with sclk and fclk both pinned; with sole use of the GPU the
  large dropouts vanish and ~10% one-sided scatter remains. Contention can
  only make a case slower, so that family runs each case three times and
  keeps the fastest. DRAM-resident cases reproduce to within 2% and are
  unaffected. Run one benchmark at a time, and treat cache-resident numbers
  elsewhere in the suite as carrying the same one-sided noise.
- **The first cache-resident case after a host fill is not a measurement.**
  Filling a 128 MiB buffer from the CPU cost the next case 13-17%,
  reproducibly by *position* rather than by access pattern: it came out at
  exactly 782 GB/s in all three row-length groups of one run, while a
  shape-identical twin measured immediately afterwards read 942 and the
  clock differed by 1%. `RunStride` now runs and discards one case per
  group, which puts those cells back at 884-950. Writeback from the fill
  competing for DRAM is the likeliest cause; it was never visible in the
  DRAM-resident groups.

## Adding a new shader

1. Write `shaders/foo.comp`.
2. Add a `//go:generate glslc ... -o foo.spv foo.comp` line and a
   `//go:embed foo.spv` var in `shaders/shaders.go`.
3. Run `go generate ./...`.
4. Build a pipeline for it via `Device.NewPipeline(shader, vk.PipelineSpec{...})`
   in `vk/engine.go` — it takes any number of storage-buffer bindings, an
   optional push-constant size, and optional specialization constants.
5. To benchmark it, add a `bench/ops_*.go` case following the existing
   ones: build buffers/pipeline, run one untimed dispatch and compare
   against a CPU reference before trusting any timing, then call
   `bench.TimeDispatch`. Verify once at a small fixed size, never inside
   the size sweep — an O(N³) CPU reference GEMM at N=4096 is ~137 billion
   scalar operations in pure Go.
