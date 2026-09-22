<!-- P13. Prompt processing at 128k: the submit budget that was blind to depth,
     the key-block width that was chosen for the wrong regime, and the
     compaction that is measured inert. Cited from llm/record.go,
     llm/gpu_attn.go, shaders/llm_attn_wmma.comp and shaders/llm_attn_blocks.comp. -->

[← TODO.md](../TODO.md) · [research index](README.md) · [P0](p0-ring-watchdog.md) · [P7](p7-context-depth.md) · [P11](p11-prefill.md) · [P12](p12-prefill-round-two.md)

# P13 — 128k completes, and the key block was twice as wide as it should have been

> **Two results and one refusal.** A 128 000-cell prefill **runs**, for the
> first time in this vertical, because the submit budget had no depth term in
> it — and it runs at **471.7 tok/s**. The attention kernel's key block then
> goes from 32 cells to 16, which is **1.61x on `attn.attn` at 128 000 cells**
> and takes the prefill there to **563.6 tok/s (1.195x)**: the 1.40x P11-7
> priced and failed to collect as a branch, collected as a rung. The refusal
> is the one that looked most promising going in — compacting the live key
> blocks into a per-query-tile list is **1.00x at every depth**, which is
> P11-4's finding a third time.

## P13-1: why 128k did not complete, and it was not the marks

TODO.md carried this as "the fill dies at ~115k cells in the P0 timestamp
pathology, reached by depth instead of row count", with "make
`DispatchMultiMarked`'s per-dispatch marks optional" as the untried route. The
marks were never the problem.

P0 established that a command buffer holding the gfx ring for more than ~2 s
is killed by amdgpu's watchdog, and fitted a cost model to chunk under it:

	322 us a dispatch  +  582 ns a dispatch-row

Every measurement behind that fit was taken **from cell zero**. P11-5 then
measured what a prefill costs at depth — 0.872 ms a token at depth zero and
1.529 at 64 000 cells — and nothing connected the two. So `batchFor(rows)`
went on handing back the same 660 dispatches at 2048 rows whether the cache
was empty or held 128 000 cells, and at 64 000 those 660 already ran for
~1.53 s of the 2 s cliff while the budget believed it was spending 0.86. At
~115 000 the same two chunks cross it.

The fix is the term that was missing. From P12's shipped ladder at 2048 rows,
1207.7 tok/s at depth zero is a 1.696 s pass and 667.7 at 64 000 cells is a
3.067 s one, so the depth term is 1.371 s over `rows * cells`, which is
**10.46 ns each** — `nsPerRowCell = 11` with a little margin, because a submit
that runs long is a ring reset and a dead process where one that runs short is
~40 us of fence wait.

Two things about its shape:

- **Only the attention dispatches carry it.** Every other block of the model
  is flat in depth (P11-5), so charging the term uniformly would make the MoE
  pay for the indexer. The recorder knows each dispatch's owner — that is what
  the attribution is for — so the chunker walks the recorded sequence and cuts
  where the *accumulated* prediction reaches the budget, charging each
  dispatch what its own label costs. At `past = 0` that is exactly P0's answer.
- **It checks itself, and only upwards.** The constants are a fit and a fit is
  a hypothesis, so every submit compares its measured GPU time against what
  was predicted and the ratio scales the next prediction. The recorder already
  got the GPU time back for the attribution; P0's own "what this leaves open"
  suggested the hook. The floor of 1.0 is the part that matters: a decode step
  is 1443 dispatches over *one* row, so the affine model charges it 322 us
  each and predicts ~330 ms where the step is ~36 ms — the per-dispatch
  constant was fitted at prefill row counts and is an order out at one row,
  harmlessly, because `maxBatch` bounds a decode step anyway. A scale that
  believed those readings would converge towards 0.25 over a conversation's
  decode steps and then hand the *next prefill* four times the budget it asked
  for, which is a ring reset 40 seconds into a 128k prompt.
  `TestObserveNeverLoosensTheBudget` is that line.

Measured, the four chunks of a 2048-row pass at 64 000 cells come in at 1.12,
0.86, 0.97 and 0.57 s against a 2 s cliff, and at 128 000 the pass is five
chunks. `TestChunkEndIsUnderTheWatchdogAtDepth` walks a synthetic 48-layer
pass at every (rows, depth) a 128k context reaches and asserts no chunk is
modelled over budget — and that a one-row pass still chunks only at `maxBatch`,
so **nothing about decode moved**.

### The curve that could not be measured before

48 layers, shipped banks, `-pp 2048 -tg 8 -ctx 139000`, `wiki.test.raw`, one
sequence walked forward (`results/p13_depth_base.csv`, the 16-cell arm `results/p13_depth_kt1.csv`, the compaction arm `results/p13_depth_blocklist.csv`, each with a `-labels` table beside it):

| depth | pp tok/s | vs d0 | tg tok/s | ms a token |
|---:|---:|---:|---:|---:|
| 0 | 1131.1 | 1.00x | 29.29 | 0.870 |
| 32 000 | 796.7 | 0.70x | 25.84 | 1.205 |
| 64 000 | 661.0 | 0.58x | 24.68 | 1.467 |
| 96 000 | 511.1 | 0.45x | 23.24 | 1.907 |
| **128 000** | **471.7** | **0.42x** | **22.94** | **2.070** |

And where a prefill token goes at 128 000 cells, by label (ms a token):

	attn.attn     0.9107    44.0%      moe.up        0.2770
	attn.select   0.2341    11.3%      moe.down      0.1310
	attn.score    0.0788                dn.qkv        0.0753
	attn.expand   0.0516                hc.cn         0.0581

**Everything that is not the attention block is flat in depth to within 1%**
across the whole sweep — `moe.up` is 0.2828 at depth zero and 0.2770 at
128 000 — which is P11-5's finding carried to twice the depth. The floor is
**0.803 ms a token**, so a prefill at 128 000 cells could not exceed 1245
tok/s at this ubatch even with the attention deleted.

## P13-2: the key block was chosen for the dense regime

P11-7 measured what the QSA selection actually lets the kernel skip, and the
table said what to do:

> the whole remaining headroom is the 1.40x between a 32-cell block and a
> 16-cell one

It then tried to collect it *inside* the kernel — a per-k-tile `kLive[kk]`
flag and a `continue` on it in the QK loop, the row max, the exponential and
the PV loop — measured **1.18x slower**, reverted it, and drew the rule:
**change the loop's granularity, not what happens inside it.** What nobody
did was change the granularity, which is a build and not a branch: the rung
`qt1_kt1` did not exist.

The live-pair table at 128 000 cells (`cmd/llm -depth` prints it per depth):

| tile | 16 | 32 | 64 | one row |
|---:|---:|---:|---:|---:|
| 16 | **11.7%** | 17.4% | 25.5% | 7.8% |
| 32 | 16.3 | 23.9 | 34.5 | 7.8 |
| 64 | 22.7 | 32.7 | 45.8 | 7.8 |

A query's cost is `live% x nKV`, so a 16-cell block is 14 976 cells a query
against a 32-cell block's 22 272 — **1.49x**, predicted before anything ran.

Measured, 4 layers (one full-attention layer), `-pp 2048 -ctx 136000`, the
attention block in ms, two passes of the control agreeing to 0.2%:

| rung | depth 0 | 64 000 | 128 000 | |
|---|---:|---:|---:|---:|
| `qt1_kt4` (64-cell block) | 16.9 | 280.1 | 597.4 | 0.63x |
| `qt1_kt2` (32-cell, shipped) | 16.0 | 186.3 | 379.3 | 1.00x |
| **`qt1_kt1` (16-cell)** | 16.3 | **127.2** | **250.1** | **1.46x / 1.52x** |

And at 48 layers, which is the number that counts — `attn.attn` in ms a
token, with every other label as the control:

| | 0 | 32 000 | 64 000 | 96 000 | 128 000 |
|---|---:|---:|---:|---:|---:|
| `attn.attn` 32-cell | 0.0157 | 0.2999 | 0.4817 | 0.8339 | 0.9107 |
| `attn.attn` 16-cell | 0.0167 | **0.2525** | **0.3547** | **0.5198** | **0.5659** |
| | 0.94x | 1.19x | 1.36x | 1.60x | **1.61x** |
| `attn.select` | 0.0026 | 0.0570 | 0.1129 | 0.1722 | 0.2341 |
| the same, 16-cell | 0.0026 | 0.0570 | 0.1129 | 0.1730 | 0.2327 |

The three labels the change cannot touch — `attn.select`, `attn.score`,
`attn.expand` — come back identical to three decimal places at every depth,
which is what says this measured one thing. The whole pass:

| depth | before | after | |
|---:|---:|---:|---:|
| 0 | 1131.1 | 1129.3 | 1.00x |
| 32 000 | 796.7 | 825.8 | 1.04x |
| 64 000 | 661.0 | 719.6 | 1.09x |
| 96 000 | 511.1 | 607.9 | 1.19x |
| **128 000** | **471.7** | **563.6** | **1.195x** |

The falloff from depth zero goes **0.42x to 0.50x**, and the gain grows with
depth exactly as the live-pair table says it should — 1.19x at 32 000, where
a 32-cell block is already half live, and 1.61x at 128 000, where it is not.

**The ladder is monotone in the block width at depth and flat at depth zero.**
That is the shape the live-pair table predicts and not the shape L2f's reuse
argument predicts — L2f chose `qt1_kt2` on 512-token prefills from cell zero,
where a 32-cell block is two cooperative-matrix tiles of reuse against one and
the key loop is dense. At depth this kernel is not short of reuse; it is short
of cells it is allowed to skip. So the rung now follows `sparse`:
`AttnKernelFor(sparse)` is `qt1_kt1` where the selection is live and
`qt1_kt2` where it is not, which keeps the dense control arm on the ladder
L2f measured.

`TestAttnGPURungsAgree` is the gate the ladder never had, and the 16-cell rung
is why it is needed: a key block narrower than a mask word is the first shape
in this file where `base` is not a multiple of 32, so the staged bitmask slice
is *half* a word and every `selected` bit moves by `base & 31`. Get that wrong
and the kernel is still a perfectly well-formed attention over the wrong
cells, which every tolerance against llama.cpp passes by being roughly a dense
causal attention. Every rung agrees with the shipped one at **4.6e-5 rms** on
the 4k fixture, where the dense bank the layer reads is itself 5.4e-3 from the
f32 model.

## P13-3: the live-block list, and it is 1.00x

The hypothesis, and it was a good one on paper. `llm_attn_wmma.comp` skips a
key block the selection left empty (P7) but still *walks the axis* to find
out: at 128 000 cells it visits 4000 blocks to do real work in 696 of them,
and each visit stages `BM` words of the bitmask — sixteen separate 64-byte
lines, because a row's mask is `nKV/32` dwords from the next row's — **once
per head**, because the mask is per token and the grid is (query block, head).
Twenty-four heads times 128 query tiles times every block, at 8x line
amplification, is ~6 GB a layer.

So `shaders/llm_attn_blocks.comp` does that OR once for all twenty-four heads,
coalesced, and compacts the live block indices into an ascending list per
query tile with a ballot and a prefix over waves; the attention kernel then
iterates the list instead of the axis. It is **bit-identical** — the list holds
exactly the blocks the skip would not have skipped, in the order the loop
would have met them — and
`TestAttnGPUBlockListDoesNotChangeTheAnswer` asserts that as an equality over
three chunk schedules.

It buys nothing. 48 layers, the same sweep, arm against arm (ms a token):

| | 32 000 | 64 000 | 128 000 |
|---|---:|---:|---:|
| `attn.attn` without | 0.2999 | 0.4817 | 0.9107 |
| `attn.attn` with | 0.2979 | 0.4725 | 0.9281 |
| `attn.blocks` | 0.0003 | 0.0005 | 0.0010 |

The compaction itself costs what it was predicted to cost — a thousandth of a
millisecond a token, three orders below what it was meant to save — and the
kernel it feeds does not move. **The mask read was never the cost**: the same
sixteen lines are read by all twenty-four heads of a query tile, the tile's
whole mask is 128 KB at 128 000 cells, and the MALL serves it.

This is **P11-4's finding for the third time**, and at this point it is a
property of the engine rather than an anecdote:

> *Latency and redundant reads that matter at decode do not matter at prefill,
> because prefill has occupancy.* A prefill dispatches 3072 single-wave
> workgroups where a decode step dispatches 24. P11-4 hoisted the block skip
> out of the loop and got 1.00x; P12-4 deleted `hc.cn`'s reduction tree and
> its gamma read and got 1.00x twice; this deletes 83% of the key-block visits
> and gets 1.00x.

And it does not become worth anything at the narrow rung either, where there
are twice as many dead blocks to skip: at 4 layers and 128 000 cells the
attention block is **238.6 ms without the list and 251.2 with it**, because
the compaction's own scan doubles with the block count while what it buys
stays zero.

The machinery is kept and defaults **off** (`LLM_ATTN_BLOCK_LIST=1` turns it
on) for two reasons: it is the exact converse regime at decode, where the
split kernel walks the same axis with 24 workgroups and no occupancy to hide
behind, and it is the compaction a per-cell gather would be built on.

## What this leaves, and the arithmetic of 900 tok/s at 128k

The target is 900 tok/s at 128 000 cells. After P13-1 and P13-2 the budget at
that depth, at ubatch 2048, is:

	floor (everything flat in depth)   0.803 ms a token
	attn.attn      0.911 -> 0.566      the 16-cell rung, measured
	attn.select    0.233
	attn.score     0.078
	attn.expand    0.051
	                                  = 1.731 ms a token, 563.6 on the clock

900 tok/s is **1.111 ms a token**. Two of the four terms above are worth
having and neither is enough on its own:

- **`attn.select` and `attn.expand` together are 0.286**, and they are one
  change. The expansion writes `nKV` floats a token so that the selection can
  radix-select over cells — but all `ratio` cells of a pooled block carry one
  block score, so a select over `nKV/ratio` entries with a per-block *weight*
  is the same answer over a quarter of the traffic, and the expansion is then
  not needed at all. It also frees **1.14 GB of arena** at ubatch 2048, which
  is what a wider ubatch wants. P11 priced the traffic at ~2.5x and named the
  risk: the causal boundary, where a block is partly masked and the
  reference's tie fill is a block split. `TestAttnGPUSelectionIsTheCPUs` is
  an exact gate and would catch any drift.
- **The ubatch.** P12 measured 1.13x from 2048 to 4096 rows on the flat floor,
  which is ~0.09 ms a token here. `-llm-batch` already defaults to 4096;
  `-depth` does not.

Both together put 128 000 cells at about **1.42 ms a token, ~700 tok/s**. The
rest has to come from the one thing that has been named since L4b and priced
twice since:

> **The gather.** At 128 000 cells a query tile of sixteen rows selects a
> union of ~4 574 cells and the 16-cell rung reads 14 976 of them — **3.3x**
> more than are wanted, and that is the floor of what a cooperative-matrix
> tile can skip, because the tile is sixteen cells and the selection's runs
> are four. Compacting the union's K and V into a contiguous scratch and
> running a dense attention over it is flat in depth by construction and cuts
> both terms at once: P13's own probe says the kernel at 128 000 cells is
> roughly **half arithmetic and half K/V traffic** (fitting the `qt2_kt2`
> control, which trades 1.46x less K/V a query for 1.37x more work and comes
> out 1.02x *worse*), so 3.3x fewer cells is 3.3x off both halves.
>
> Priced: ~0.20 ms a token at 128 000 including the gather's own traffic,
> against 0.566 — **2.8x**, where before the rung change it would have had to
> be 4.9x. With the selection change and the ubatch that is **~1.06 ms a
> token, ~940 tok/s** — the first arrangement of these numbers that reaches
> the target, and the only one.
>
> The open question is the arena. A per-(query tile, kv head) scratch at 8192
> gathered cells is 2.1 GB, which only fits once the expansion's 1.14 GB is
> freed and the cap is honest; the alternative is chunking the gathered axis
> and folding the partials through P8's combine, which already exists.

Smaller things this sweep priced in passing:

- **`attn.score` is 0.079 ms a token at 128 000** and linear in depth. The
  indexer must score every pooled block, so it is the one term here that is
  the model rather than the implementation.
- **`maxStorageBufferRange` caps the cache at ~148k cells**, because the KV
  planes share a buffer with the arenas. 128k fits; 256k needs L6a's
  array-of-buffers, one a layer.
- **Decode at 128 000 cells is 22.94 tok/s**, 0.78x of its depth-zero rate —
  the falloff P7-P10 closed holds to twice the depth they measured it at.
- **The narrow rung is prefill's and not decode's**, and the sweep that tried
  it on both is why: with `qt1_kt1` everywhere, decode at 128 000 went
  **22.94 -> 22.27 tok/s** while prefill went 471.7 -> 563.6. That is P8's
  finding in the other direction — a decode step is 24 single-wave workgroups
  that all fit at once, so it costs one wave's *serial walk*, and halving the
  key block doubles the walk. `AttnKernelFor` takes the rung from the row
  count at the same `attnSplitMaxRows` boundary the split already turns on.

## Reproducing

	go build -o /tmp/pp ./cmd/llm        # once, never `go run` per arm
	export LLM_BANK_CACHE=models/Qwen3.8-Flash-Next-GGUF/bank-cache
	export LLM_DENSE_BANK=deltanet=q4_k/32,hyper_conn=q5_k/32,lm_head=q5_k/32,full_attn=q5_k/32,qsa_indexer=q5_k/32
	export LLM_MOE_BANK=gate_shexp=q4_k,up_shexp=q4_k,down_shexp=q5_1,down_exps=iq4_nl
	M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

	# the headline, ~6 minutes; two passes agree to 0.05-0.2% at every depth
	#   pass 1  1129.3 / 825.8 / 719.6 / 607.9 / 563.6
	#   pass 2  1127.2 / 825.3 / 721.1 / 607.9 / 563.3
	/tmp/pp -model $M -depth -depths 0,32000,64000,96000,128000 -pp 2048 -tg 8 -ctx 139000 -csv results/p13_depth.csv

	# **the fast loop for attention at depth, and it is 40 seconds an arm**
	LLM_ATTN_KERNEL=qt1_kt1 /tmp/pp -model $M -depth -depths 0,64000,128000 -pp 2048 -tg 4 -ctx 136000 -layers 4

The four-layer prefix is the instrument P13 was built on and it is worth
knowing about: it stages one full-attention layer in ~1 s, fills to 128 000
cells in 15 s, and reports the same `attn.attn` ratios the 48-layer sweep does
at a twelfth of the wall clock. `cmd/llm -attn` cannot replace it — that bench
runs at `past = 0` and so never reaches depth's regime at all.

The gates:

	go test ./llm/ -run TestChunkEndIsUnderTheWatchdogAtDepth        # P13-1
	go test ./llm/ -run TestObserveNeverLoosensTheBudget             # P13-1's hazard
	go test ./llm/ -run TestAttnGPURungsAgree                        # P13-2, every rung
	go test ./llm/ -run TestAttnGPUBlockListDoesNotChangeTheAnswer   # P13-3, exact
	go test ./llm/ -run TestAttnGPU                                  # the block
	go test ./llm/ -run TestGraphIsAChunkSplit                       # the graph
