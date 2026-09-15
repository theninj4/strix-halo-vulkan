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

§2.4 (the workgroup swizzle), §2.6 (fp16 output and epilogue fusion) and §2.8
(a weight stored as fragment tiles) were all measured as part of stage 4 and
are written up in [`stage-4-dit-graph.md`](stage-4-dit-graph.md) rather than
in files of their own.

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
| [Stage 4](stage-4-dit-graph.md) | The DiT block as a graph — **49.3 ms/block, 13.4 s/image**; a tiled weight (§2.8), a swizzled grid (§2.4) and a fused tail (§2.6) compound to **1.83x**, and all three are about order rather than arithmetic |
| [Stage 4c](stage-4c-dit-stack.md) | The whole DiT resident — **12.54 GB in three storage buffers, 12.8 s/image**; splitting the weight arena costs one pipeline per bank and nothing per dispatch, and six banks are bit-identical to one |
| [Stage 5](stage-5-text-encoder.md) | The tokenizer and the text encoder — **35 of 36 layers, 2.08 s on the CPU to 57 ms on the GPU**; the model is **memory-bound at T flop/byte against a 235 crossover**, so unlike the DiT **the winning kernel moves with the prompt length** (1.24x for getting it wrong); Qwen3's massive activations break an RMS-normalised error bound |
| [Stage 6](stage-6-pipeline.md) | The scheduler and the driver — **prompt to PNG, 1024x1024 in 19.8 s**, matching diffusers' fp32 CPU pipeline to 2.5e-2 of the image; every bug this stage had was in the *composition* and invisible to each component's own oracle — SwiGLU **overflows fp16 on a real prompt and never on a random one**, a shared shader's new push constant silently zeroed the text encoder, and the VAE's watchdog cap is a proxy for time on a graph whose dispatches differ 100x |
| [Stage 7](stage-7-vae-mid-block.md) | The VAE mid block on the matrix cores — attention **1.43 s to 15.1 ms** and the four projections **503 ms to 1.12 ms**, an image **19.8 s to 17.8 s**; the head is 512 wide so the *workgroup* owns it rather than the wave, and the block's saturated softmax means **a kernel that drops three quarters of every dot product decodes the same image** — no end-to-end tolerance can validate this attention |
| [Stage 8](stage-8-vae-conv.md) | conv2d on the matrix cores — conv3x3 **3.02 s to 244 ms**, 3.1 to 41-42 TFLOP/s, the decode **3.62 s to 876 ms** and an image **17.8 s to 15.0 s**; the layout is the finding, because a patch fragment is only contiguous if the *channel* axis is the tiled one (tiling the pixel axis makes the `dw = ±1` window straddle two tiles), and a one-pixel zero border puts the convolution's padding **in the data** so the nine taps are branchless. Afterwards the decoder has no arithmetic left in it: the largest item is a **group norm** and 63% of the decode is four bandwidth-bound elementwise passes |
| [Stage 9](stage-9-head-and-tail.md) | The transformer's head and tail on the device — the host's share of a step **59 ms to 5-10 ms**, an image **14.89 s to 14.49 s**. A bias and a *pad token* are extra columns of K, so `y = Wx + b` and `y = x_pad_token` come out of one biasless GEMM with no branch and no host write. And the correction: the engine's "a host read-back runs at 0.2 GB/s" is true only below about **8 GB of allocation**, where `vk.NewBuffer` still gets the device-local heap — the 20.5 GB pipeline reads the same arena **83x faster**, so the 0.8 s an image this stage was partly priced on was never there |
| [Stage 10](stage-10-layout-epilogues.md) | The block's two layout passes as epilogues — 18 dispatches to **16**, a block **53.17 ms to 51.80**, an image **14.49 s to 14.26 s**. `pack v` and `narrow ctx` moved no information and are now the store instruction of the kernel above them, which is possible because in both cases the data was already in the right registers in the right 16x16 shape. Two other things came out of it: the item had been on the backlog for four sessions priced at **1.0 s an image when the measurement behind it said 1.0 ms a block** — a units slip, 3.4x the real figure — and the old path was doing a **double rounding** that the fused one does not, which is why 84 halves in 1.2 M differ and why every one of them is an exact fp16 tie |

### Speech vertical findings

From building the two audio verticals (SPEECH.md), so they carry stage names
rather than section numbers.

| Stage | Findings |
|---|---|
| [S6](s6-parakeet-encoder.md) | The parakeet encoder on Vulkan — the 24 conformer layers **2.45 s to 13.8 ms**, 12.9 TFLOP/s, transcript and decode trace unchanged. Transformer-XL's position term is an **additive bias on the score tiles**, one coopmat load and one add. The bug: the flash kernel takes its row max *before* the tail mask, which encoded an assumption that a real score is near a pad key's 0 — with a position bias it is not, and a key that does not exist set the scale until the softmax underflowed fp16; one `break`, 100x. Padding the clip to one alignment instead of two cost 1.28x, **wave32 beats wave64 on every rung at every clip length**, and two of the four negative controls change the encoder output by 33% and 3.9% while leaving the transcript alone |
| [S7](s7-parakeet-subsampling.md) | The parakeet subsampling stack on Vulkan — five convolutions and a linear, **92 ms on the host to 0.97 ms on the device**, 2.9 TFLOP/s, whole pipeline 331 → 247 ms. The port *is* a layout: held channel-last as `[T, F, C]` the depthwise convolution's lanes walk contiguous memory, the 1x1 convolutions become plain `[8832, 256] x [256, 256]` GEMMs with **no packing pass**, and the flatten is a row copy once the linear's weight columns are permuted at upload. Stage 8's implicit-GEMM conv answers the same layout question the *opposite* way because its input has 512 channels and this one has **one** — nine wave-uniform scalars shared by 256 lanes, no fragment at all. **wave32 wins a third set of shapes**, and the wide four-wave tile loses the 8832-row shape it was built for because N=256 is only four tiles wide. The flatten-order control is the model's cleanest plausible-tensor failure: 1.22 relative, empty transcript |
| [T4](t4-kokoro-vocoder.md) | Kokoro's vocoder on Vulkan — **3626 ms to 38 ms**, an utterance **0.90x real time to 12.90x**, waveform 59 dB from the CPU path and *exactly as far from torch as the CPU path is*. Three shape changes turned out to be addressing rules rather than passes: a dilated convolution's im2col is `+j*(d*lda)` on the A operand (`A_CONV`), a transposed convolution at `k = 2s` is **one** GEMM whose N is `s*C_out` read back at a different stride, and a `torch.cat` per block is a **row stride**. `A_CONV=2` extends the first to channel counts that are not powers of two — 514 and 1090 — by *carrying* the tap index across the K loop instead of dividing, so the inner loop costs the same; padding to a multiple of BK rather than to a power of two is 5.7% of arithmetic instead of 88%. The iSTFT is a **gather**: sample i reads the four frames covering it, one dispatch, no atomics. And the readback was most of what remained twice over — 38 ms for one `[15601, 128]`, 14.4 for a `[2600, 256]` — removed by a **shared fp32 arena** in which each object's output *is* the next one's input, with every `1/N` between them folded into a weight |
| [T5](t5-kokoro-g2p.md) | Grapheme to phoneme in Go — `cmd/tts` takes **text**, and the phonemes are character for character misaki's on a corpus in which every branch fires. The stage opened with a *measurement instead of code*, because misaki runs spacy: **91% of running English is a dictionary lookup** and a part-of-speech tagger decides only **1.74%** of tokens, half of what looks tag-conditioned being decided by whether a vowel follows. Four components are exact against dumped tables (204 word-path tokens, 10115 integers in three spellings, 1760 `get_number` cases, 67 subtoken splits, 39 espeak words) — and the tables earned it: one caught a backtracking bug that split `3.50` into `3.5` and `0`. The homograph rules scored **worse than no tagger** on the first attempt, all of the damage in one word, and the fix was measured rather than reasoned: the syntactically obvious refinement costs 8.5 points because "that is" is a determiner in running prose. espeak turned out to be a library rather than a binary, so the fallback is **cgo plus dlopen** and its absence degrades instead of breaking |
| [S8](s8-parakeet-decode.md) | The parakeet transducer tail on Vulkan — projector, LSTM prediction network, joint and TDT greedy loop, **208 ms to 4.8 ms**, whole clip 255 → 43 ms at **257x real time**. The first *latency*-bound stage in the engine: 1.1 GFLOP that took 208 ms because each of 46 emissions was a host round trip. **No new GEMM** — M = 1 runs on the existing rungs with M padded to the tile, and `BM = 32 ties BM = 16 exactly`, doing double the arithmetic for the same time, which is the proof the tail is bandwidth. The finding: **24 MB of weights against a 32 MiB MALL**, so the loop re-reads the whole model from cache 46 times and the GEMMs measure 449-702 GB/s — above the DRAM bus. §5.1b's budget met rather than fallen off. 38% of an emission is still the submit and the fence |

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
