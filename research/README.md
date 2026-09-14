# research/ — the findings archive

One file per **completed** experiment. The open backlog, the measured
roofline, and the order of attack stay in [`../IDEAS.md`](../IDEAS.md);
this directory is where a finished item's full write-up lives once it is
closed, so the backlog stays readable and an agent can load one finding
without loading all of them.

## The section number is the address — do not renumber

`§N.M` is a stable identifier, cited **972 times** across this repo: in
`IDEAS.md` and `TODO.md`, and — the part that matters — in **293 comments
across 29 files** in `shaders/`, `bench/`, `cmd/` and `vk/`, where it is the
only link between a kernel and the experiment that justified its shape. For
example `shaders/gemv_w4a8.comp` cites §1.1, §1.2 and §1.3 to explain three
separate decisions in one file.

So: a finding may be **moved, renamed or rewritten**, but its number is
permanent. A new experiment takes the next free number in its section
rather than reusing or reflowing an old one. Filenames carry the number so
that `grep -rl '§2\.3' .` and `ls research/2.3*` find the same thing.

## Index

| § | Finding | Headline result |
|---|---|---|
| [§0](0-measurement-validity.md)<br>(§0.1-§0.5) | Measurement validity | Ceilings measured on-device; three predictions falsified, incl. **WMMA int8 is not 2x fp16 here** |
| [§1.1](1.1-w4a8-gemv.md) | W4A8 GEMV | **2.4x**, 819 GFLOP/s, 89% of the DRAM bus |
| [§1.7](1.7-load-width.md) | Match load width to the weight row | **99-103% of the bus at every N**; `VEC = N/(8*WAVE)` |
| [§1.8](1.8-grouped-gemv.md) | Grouped GEMV for MoE decode | **1.8-2.0x** the GEMM at M=1; don't pad an expert bank |
| [§1.9](1.9-m-block.md) | An M block for it | **1.66x** at 256 sequences, **0.39x** at one — read it off the routing |
| [§1.10](1.10-n-block.md) | An N block for it | `down`'s deficit is a fixed cost per output row; **195 tok/s, 97% of the bus floor** |
| [§1.11](1.11-mn-corner.md) | Both blocks in one wave | They compose: **1.10-1.19x**, FFN at **818 tok/s** |
| [§1.12](1.12-m-block-from-histogram.md) | M block from the routing histogram | **1.09-1.13x**; an expert belongs to exactly one dispatch |
| [§2.1](2.1-register-blocking.md) | Register-block the coopmat GEMM | **6.0x**, 25.2 TFLOP/s; LDS staging turned out unnecessary |
| [§2.2](2.2-q4-coopmat.md) | Q4 into the coopmat path | **2.10x a MoE block**; LDS round-trip is unavoidable |
| [§2.3](2.3-channel-aliasing.md) | The [N,K] contradiction | It was **DRAM channel aliasing**; pad strides off a 4 KB multiple |
| [§2.7](2.7-kslab-hoist.md) | Hoist the K-slab's fragment loads | **2.1x → 38990 GFLOP/s, 70% of the ceiling**; concurrency, not depth |
| [§3.4](3.4-model-shapes.md) | The models' real shapes | Prefill has **two winners split by M alone**; the per-model budget table |
| [§3.5](3.5-grouped-moe-gemm.md) | Grouped / MoE GEMM | **1.1-4.1x**, and the mechanism is **occupancy** |
| [§5.1b](5.1b-mall-cliff-and-stride.md) | The MALL cliff and the stride probe | The coverage law `min(1, C/gcd(stride, 4096))`; rotation is **4 KB** |
| [§6.2](6.2-wave32-vs-wave64.md) | wave32 vs wave64 | Splits three ways; decode **96% of the bus**, the best GEMM **spills** |

`§0.1`-`§0.5` are subsections *inside* `0-measurement-validity.md` rather
than files of their own — they are 10-38 lines each. `§0.1` (the measured
MMA ceiling) is the most-cited of them, at 20 references.

### Pipeline stage findings

These come from building the z-image slice (PIPELINE.md) rather than from a
numbered experiment, so they carry names instead of section numbers:

| Stage | Findings |
|---|---|
| [Stage 2](stage-2-vae-decoder.md) | VAE decoder — fp16 overflow, the conv intensity lesson, two device limits |
| [Stage 3](stage-3-dit-attention.md) | DiT attention — why it needs WMMA, and the two hazards a port hits |

### Closed, but small enough to have stayed in the backlog

`§4.1` (per-dispatch cost: ~300 ns, worry falsified) and `§3.7` (the
reduction kernels: the 3.3x gap is lane count, closed as a side effect of
§6.2) are answered in one paragraph each and live in `../IDEAS.md`.

## The three layers, and which file to put a thing in

| Layer | Lives in | Grows |
|---|---|---|
| What the chip can do | `../IDEAS.md` — the roofline table | Rarely; only when a ceiling is re-measured |
| What to try next | `../IDEAS.md` — the open items and the order of attack | Shrinks as items close |
| What we learned | **here**, one file per § | Forever — which is why it is not in `IDEAS.md` |
| What happened in a session | `../TODO.md` | Forever; append-only handoffs |

When an item closes: move its body here as `N.M-slug.md`, leave the heading
in `IDEAS.md` with a one-line result and a link, and add a row above.
