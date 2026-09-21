<!-- P12. Prompt processing, the round after P11: two bit-exact kernel changes
     that hold, four measured refusals that close P11's own open list, and
     where the remaining prefill headroom actually is.
     Cited from TODO.md, shaders/llm_attn_wmma.comp, shaders/llm_moe_gemm.comp,
     shaders/llm_hc_cn.comp and llm/gpu_moe.go. -->

[← TODO.md](../TODO.md) · [research index](README.md) · [P11](p11-prefill.md) · [P8](p8-decode-attention-split.md) · [L5b](l5b-moe-gpu.md)

# P12 — prefill, round two: the lane split and the block header, and four things that are not there

**Result. Prefill is 1.012-1.025x on both axes, bit-exact, and P11's open
list is now mostly closed by refusal rather than by work.** Two changes hold —
the attention kernel's row max spread across the whole wave, and a K-quant
block's header read as one sixteen-byte load — and four ideas that looked
right, including *both* of the ones P11 named for `hc.cn`, cost nothing or
buy nothing and are not in the tree.

Two passes of each arm, interleaved, same hour. `results/p12_pp_*.csv` and
`results/p12_depth_*.csv`.

**Short context**, 48 layers, `-graph`, shipped banks:

| ubatch | before | after | | against llama.cpp |
|---:|---:|---:|---:|---:|
| 512 | 683.8 | 683.3 | 1.000x | 1.75x |
| 1024 | 924.4 | **935.6** | 1.012x | 2.39x |
| 2048 | 1191.2 | **1207.7** | 1.014x | 3.08x |
| 4096 | 1337.1 | **1363.5** | 1.020x | 3.48x |
| 8192 | 1370.5 | **1403.9** | **1.025x** | **3.59x** |

**Against context depth**, `-pp 2048`:

| depth | before | after | |
|---:|---:|---:|---:|
| 0 | 1140.7 | **1154.6** | 1.012x |
| 8 000 | 946.2 | **968.1** | 1.023x |
| 16 000 | 878.5 | **894.3** | 1.018x |
| 32 000 | 787.5 | **805.7** | 1.023x |
| 64 000 | 651.3 | **667.7** | **1.025x** |

The arms are tight enough that 1.012x is a real number: the two base passes
at 8192 rows came out **1370.5 and 1370.5**, and at 64 000 cells **651.3 and
651.2**.

## P12-1: the attention row max, on sixty-four lanes instead of sixteen

P11's first open item, and it is worth what it looked like. `llm_attn_wmma.comp`
took a key block's row max with `for (i = lane; i < BM; i += WAVE)` — BM is 16
at the shipped `qt1_kt2` rung and WAVE is 64, so **sixteen lanes worked and
forty-eight sat out**, each of the sixteen then walking TILE cells serially
with a `selected()` bit test on every one. It ran KTIL times a block, with a
cooperative-matrix store and two barriers around each pass.

It is now one pass. The whole key block's scores are staged at once — `stage`
holds QT*KTIL tiles rather than QT, which is 2 KB of LDS at the shipped rung —
and RCL = WAVE/BM lanes share a row, each scanning RCPL = BN/RCL of its
columns, folded by a clustered subgroup max. Every shipped rung has BM ≤ 32,
so RCL is 4 or 2. The `-inf` initialiser and its barrier go with it: every row
is now written exactly once, by its cluster's first lane.

**It is bit-identical, and not merely close.** A max is associative,
commutative and idempotent and has no rounding in it, so it is exact under any
reassociation; and the set of cells each row maxes over is unchanged, because
the causal test and the selection bit are the same predicates read in the same
order. The wave is converged at that point — the `q0`/`head` return and the
block skip are both workgroup-uniform — which is what makes the clustered op
legal. Gated by `TestAttnGPUCacheSizeDoesNotChangeTheAnswer` and the other
three exact-equality attention tests, plus `TestGraphIsAChunkSplit`.

On the kernel, `cmd/llm -attn`, µs a layer:

| tokens | before | after | |
|---:|---:|---:|---:|
| 512 | 268.7 | **231.1** | **1.16x** |
| 2048 | 3408.4 | 3264.2 | 1.04x |
| 8192 | 50 408 | 47 711 | 1.06x |

On the graph at depth it is **+1.6-1.8%** of a prefill token from 8 000 cells
on (`results/p12_depth_rowmax_only.csv`), which is the whole of the depth
column above bar the MoE's share.

## P12-2: a K-quant block's header is one load, not four

Q4_K and Q5_K open every block with `d`, `dmin` and twelve packed 6-bit
scale/min bytes — **sixteen bytes**, and the unpack was reading them as four
separate dwords. At BK 32 a lane unpacks two rows a K-step, so that was eight
scalar loads a K-step against the four `uvec4`s the nibbles take. One `uvec4`
does it, and this is L5b-7's finding for the third time: *the kernel is bound
by how many load instructions its unpack issues, not by the bytes they fetch.*

It is legal because every block of these two formats is sixteen-byte aligned,
and that is three facts rather than an assumption — each bank region starts at
`roundUpInt(nbytes, 16)`, a Q4_K row is `k/256 * 144` bytes and a Q5_K row
`k/256 * 176`, and the blocks inside a row are 144 and 176 apart. `MoEGPU.build`
now checks the first of those rather than trusting it, as P11 checked the
gather's.

Beside it, the nibble comes out in two ops instead of four: `(w >> (8*b + 4*sub)) & 0xF`
where it was a byte extract and then a select on `sub`.

One layer, real 4k routing (`cmd/llm -moe`), µs a layer:

| tokens | `moe.up` before | after | | `moe.down` |
|---:|---:|---:|---:|---:|
| 512 | 6762.0 | 6645.3 | 1.018x | unchanged |
| 2048 | 12 118.2 | **11 755.4** | **1.031x** | unchanged |
| 4096 | 17 826.4 | 17 467.0 | 1.021x | unchanged |

**`moe.down` is the control and it is flat**: it is IQ4_NL, which has no
K-quant header, and its body was not touched.

## P12-3: what the MoE unpack would be worth if it were free, and why it is not

Before writing P12-2 the term was priced by gutting it — `float16_t(nib)` in
place of `float16_t(ds * float(nib) - dm)`, which also kills the header and
scale loads behind it. That arm is **1.37x / 1.21x / 1.17x** at 512 / 2048 /
4096 tokens, so the whole scale path is 17-37% of `moe.up`, the single largest
kernel in a prefill.

P12-2 collects the part of that which is loads and integer ops. The rest is
the **per-element affine itself** — an int→float convert, an FMA and an f16
convert, three instructions on every one of the 1.6 billion weights a
2048-token chunk unpacks — and there is no bit-exact way to make it cheaper.
The two obvious routes both change the answer: doing it in packed fp16 costs
about thirteen bits of mantissa on every dequantised weight, and
`v_cvt_pkrtz_f16_f32` rounds toward zero where `float16_t()` rounds to
nearest. Either would need a perplexity run to justify and neither is
attempted here. **This is the floor of the current format choice, not a
kernel defect**: at 21.1 TFLOP/s and 61 GB/s of bank, `moe.up` is neither
arithmetic-bound on its matrix cores nor bandwidth-bound — it is bound by
unpacking a 4.5-bit bank into fp16 three instructions at a time.

## P12-4: `hc.cn` — both of P11's leads cost nothing

P11 left `hc.cn` at 134 GB/s where a copy gets 236, with two named suspects.
**Both are wrong, and the probes are one line each.**

- **The reduction tree is free.** P11 proposed the norm's tail — "an
  eight-barrier 256-way tree where a subgroup reduction is one barrier" — and
  noted it would not be bit-exact. Deleting the tree outright (wrong answer,
  right cost) measures **5331.5 µs against 5325.8** at 8192 tokens. The eight
  barriers and the 256-way tree, four times a token, cost **nothing at all**.
  So the non-bit-exact rewrite that was queued behind it would have bought
  zero, and `llm_hc_norm.comp` does not have to move.
- **The gamma read is free.** The other candidate was the kernel's load count:
  each of the four streams reads a 10 KB `gamma` row per token, a third of all
  its memory operations. Replacing it with a constant measures **5332.9 µs**.
  It is the same 40 KB for every token and the MALL serves it.

So `hc.cn`'s 57% of copy bandwidth is neither its barriers nor its redundant
loads, and it is still unexplained. What is now ruled out is everything that
was on the list.

## P12-5: the block skip hoisted out of the loop is worth zero at prefill

The one structural idea tried and reverted. At 64 000 cells a query tile walks
~2000 key blocks and P11-7 measured only 25.9% of them live, so ~1480 blocks
pay a **dependent** global read — the loop cannot decide to skip until the
block's mask words come back — and then do nothing with it. That is P8/P9/P10's
shape one level up, and it predicted well: a cost model in which a dead block
costs ~20% of a live one reproduces `attn.attn`'s measured 9.8 TFLOP/s at 64k
against 16.0 at depth zero almost exactly.

It was built. One streaming pass over the key range before the loop, lane `b`
taking block `b` so a wave reads 64 consecutive mask words at a time, and a
subgroup ballot turning the wave's 64 predicates straight into the two bitmap
words they are — no atomics, no per-block barrier, 512 bytes of LDS, the same
words read, bit-identical by construction.

**It measures 1.00x at every depth**, against the row max arm in the same
interleaved sweep (`results/p12_depth_prepass_refused.csv`):

| depth | row max only | + prepass |
|---:|---:|---:|
| 0 | 1143.2 | 1143.9 |
| 8 000 | 960.2 | 960.3 |
| 16 000 | 888.7 | 888.4 |
| 32 000 | 800.7 | 800.5 |
| 64 000 | 663.0 | 662.5 |

The reason is the grid, and it is the **exact converse of P8's finding**. A
decode step dispatches 24 single-wave workgroups and nothing hides a serial
walk; a 2048-token prefill dispatches (2048/16) × 24 = **3072** of them, so
every one of those dependent loads is already covered by another wave's work.
*Latency that matters at decode does not matter at prefill, because prefill
has occupancy.* Reverted; the per-block vote stands.

> The corollary is worth keeping: this idea is still live **for decode**,
> where the grid is 24 workgroups (384 with P8's splits) and the walk is
> exactly what P8 shortened. It was deliberately built under `#ifndef SPLITK`
> and so was never measured there.

## P12-6: the selection mask loop is 3.5-8%, not the rest of the gap

P11's other candidate for `attn.attn`'s depth gap was "the mask loop that
every live block now passes through" — BM×BN elements with a `selected()` bit
test on each. Removing it entirely (wrong answer, right cost) is **1.08x /
1.05x / 1.035x** at 512 / 2048 / 8192 tokens. It is real but small, and with
the row max now widened there is no large under-parallelised loop left in the
block body.

## P12-7: `-llm-batch` is 4096, and the second rate moves the other way

P11 left the ubatch as the largest number on the board and a memory budget to
be decided. It was decided by measuring it **through the server**, which is
also how the thing P11 did not see got found.

Four distinct prompts of 4104-4477 tokens (wikitext, disjoint slices so prefix
reuse cannot touch them), `-llm-ctx 8192`, two passes of each arm, the arms
interleaved and each server started fresh. The passes agree to 0.5%:

| `-llm-batch` | prefill | | TTFT, 4249 tokens | decode | |
|---:|---:|---:|---:|---:|---:|
| 2048 | 1020.9 tok/s | | 4.33 s | 28.04 tok/s | |
| **4096** | **1146.3** | **1.12x** | **3.87 s** | **25.81** | **0.92x** |

**The wider arenas cost 8% of decode.** It is not noise: sixteen samples, the
two arms do not overlap at all (the slowest 2048 decode is 30.63 tok/s in the
short-generation run and the fastest 4096 is 30.15), and it reproduces at both
8 and 256 generated tokens. A decode step streams 4.1 GB of weights a token
and the extra 1.5 GB of resident arenas sits in the same memory; P1c already
saw this shape once, where a whole-model rate moved 1.6-1.9x on the dense
banks with an identical binary and GTT fragmentation was the suspect.

So the two rates move in **opposite directions** and the trade has a
break-even. On the 4249-token prompt, 4096 saves 0.46 s of TTFT and costs
3.1 ms a generated token, so it pays for itself for the first ~150 tokens:

| generated | 2048 | 4096 | |
|---:|---:|---:|---|
| 8 | 4.60 s | **4.17 s** | 4096 by 1.10x |
| 150 | ~9.7 s | ~9.7 s | break-even |
| 256 | **13.48 s** | 13.81 s | 2048 by 1.02x |
| 1024 | **40.9 s** | 43.5 s | 2048 by 1.07x |

**4096 is the right default for an interactive server and the wrong one for a
batch summariser**, and `-llm-max-tokens` defaults to 1024 — so the shipped
default is now 4096 and API.md says in as many words that a deployment which
really generates that much should say `-llm-batch 2048` and take the TTFT
back. The lesson is the one P11-1 already stated and this is the second
instance of: **a ubatch measured on the prefill ladder alone is a hypothesis
about prefill**, and the graph ladder cannot see the other half because it
never decodes.

## What is open, in order

- **The decode cost of wide arenas is unexplained and is now the interesting
  one.** 8% of decode for 1.5 GB of arenas that a decode step never reads is
  not obviously anybody's fault, and if it is layout rather than volume it may
  be recoverable — which would make 4096 free and 8192 worth having. The probe
  is to stage the wide arenas and prefill in narrow chunks anyway, which
  separates *allocated* from *used*.
- **The ubatch ladder itself has not flattened**: 2048 → 8192 is 1.16x on the
  graph (1207.7 → 1403.9). Whether any of that survives the decode trade above
  is unmeasured.
- **`hc.cn` at 134 GB/s** is now the cleanest unexplained number in the
  prefill graph *and* has no remaining hypothesis — P12-4 spent both. The
  next probe should be the access pattern itself: it runs four read-modify-write
  streams plus an fp16 write per token, where a copy runs two.
- **`attn.select` at 64 000 cells** is untouched and is 7.5% of a prefill
  token there. It streams the row out of DRAM five times — four radix passes
  and an emit, 1.28 MB a token a layer — at ~133 GB/s. The two routes are
  P11's (select over `nKV/ratio` block scores and expand once, ~2.5x the
  traffic back, with the causal boundary's tie fill as the risk) and a
  candidate pass: collect the keys matching the winning bucket during pass 2
  so passes 3 and 4 run out of LDS, which is 5 reads → 3 and needs no new
  tie semantics.
- **The MoE's per-element affine** is the floor described in P12-3 and only
  moves if the format or the rounding moves.
- **P12-5's prepass for decode**, where the grid cannot hide the latency.

## Reproducing

	go build -o /tmp/pp ./cmd/llm        # once, never `go run` per arm
	export LLM_BANK_CACHE=models/Qwen3.8-Flash-Next-GGUF/bank-cache
	export LLM_DENSE_BANK=deltanet=q4_k/32,hyper_conn=q5_k/32,lm_head=q5_k/32,full_attn=q5_k/32,qsa_indexer=q5_k/32
	export LLM_MOE_BANK=gate_shexp=q4_k,up_shexp=q4_k,down_shexp=q5_1,down_exps=iq4_nl
	M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

	/tmp/pp -model $M -graph -tokens 512,1024,2048,4096,8192 -ctx 8192
	/tmp/pp -model $M -depth -depths 0,8000,16000,32000,64000 -pp 2048 -tg 4 -ctx 70000
	/tmp/pp -model $M -moe  -tokens 512,2048,4096 -iters 3 -layers 1    # 228 ms to stage
	/tmp/pp -model $M -attn -tokens 512,2048,8192 -ctx 16384 -layers 2 -iters 3

**`-attn` is the loop this round was worked in and it was not before**: two
layers, 1.9 seconds end to end, real routing and a real selection, per-label
µs a layer. It tops out at ~16 384 tokens — a 32 768-token batch holds the
graphics ring past P0's watchdog and is reset mid-submit — and it runs at
`past = 0`, so it reaches the selection's regime but not depth's. Every
probe in P12-3, P12-4 and P12-6 is one line in a shader and one run of
`-attn` or `-moe`, which is why four of them were affordable.

The gates after touching any of this:

	go test ./llm/ -run TestAttn -timeout 30m
	go test ./llm/ -run TestMoEGPU
	go test ./llm/ -run TestGraphIsAChunkSplit

Note the **full** `./llm/` suite needs `-timeout 60m`: it runs past Go's
ten-minute default in `TestQSASelectionSurvivesOurScores`, which reads as a
failure and is not one.
