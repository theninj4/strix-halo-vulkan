# P15 — token generation at depth: the host term, the stripe, and the split gather

**2026-09-22.** Three changes and one measured refusal. A decode step at
128 000 cells goes from **22.58 to 27.20 tok/s (1.20x)** and the falloff from
depth zero from **0.76x to 0.91x**, with **prefill and depth-zero decode flat
as the controls** — 826.5 → 829.4 tok/s at 128k and 29.70 → 29.73 at depth
zero, both inside the run-to-run spread.

P7–P10 closed the *device's* depth term down to what the attention kernels
cost. P14 closed prefill's. What was left was decode's, and it was four terms,
one of which was not on the device at all.

| depth | pp base | pp final | tg base | tg final |
|------:|--------:|---------:|--------:|---------:|
| 0 | 1154.4 | 1150.7 | 29.70 | 29.73 |
| 8 000 | 972.8 | 969.6 | 28.07 | **28.78** |
| 32 000 | 916.2 | 915.3 | 25.87 | **28.72** |
| 64 000 | 891.0 | 892.8 | 24.27 | **28.14** |
| 128 000 | 826.5 | 829.4 | 22.58 | **27.20** |
| falloff | 0.72x | 0.72x | 0.76x | **0.91x** |

48 layers, shipped banks, `-pp 2048 -tg 16 -ctx 139000`, `wiki.test.raw`.
**Two passes of the final arm, same hour**, agreeing to 0.2-1.2% on tg
(29.73/29.78, 28.78/29.14, 28.72/28.80, 28.14/28.26, 27.20/27.41) and 0.3% on
pp (1150.7/1153.7 at depth zero, 829.4/831.8 at 128 000).

## Where a 44.3 ms token went

The whole of it, and the reason this session went where it did. Per token at
128 000 cells, base against final:

| term | base | final | Δ |
|---|---:|---:|---:|
| `attn.attn.split` | 4.109 | 1.571 | **−2.538** |
| `attn.score` | 2.533 | 0.995 | **−1.538** |
| host `gather` (PLE rows) | 5.175 | 1.263 | **−3.912** |
| `attn.select` | 0.862 | 0.862 | — (the control) |
| `attn.attn.gather` + `gathmask` | — | 0.249 | +0.249 |
| everything else | flat | flat | ~0 |
| **a token** | **44.3** | **36.8** | **−7.5** |

Every other kernel in the model is flat in depth and was already flat before
this: `moe.up` 5.44 → 5.44, `dn.qkv` 4.39 → 4.39, `hc.cn` 2.33 → 2.33. **The
depth question in this model is four kernels and one host function**, and it
has been since P7.

## 1. `PLERows` hashed the whole sequence, every token

The largest term in a deep decode step was **on the host**, and it was not
attention.

`Graph.hidden` gathers the n-gram embedding rows for the tokens it is about to
push. It did that by calling `PLERows(cfg, g.ids)` — which allocates
`len(ids)*NHeads` int32s and hashes **every position in the sequence** — and
then slicing off all but the tail:

```go
rows := PLERows(g.pleCfg, g.ids)[g.past*g.pleCfg.NHeads:]
```

At 128 000 cells that is 8.2 MB allocated and 2.05 million rows hashed, per
decode step, to use sixteen of them. Measured: **5.18 ms of a 44.3 ms token,
and 3.9 of the 10.6 ms that token gained over one at depth zero.** It was
39% of the whole falloff and it never touched the device.

`PLERowsFrom(cfg, ids, from)` is the same loop over the tail. The rows are
**identical and not merely close, by construction**: a position's row depends
on `ids[i-NGram+1 .. i]` and nothing else — the `cut` flag is reset at the top
of each position, so no state carries between them — so a suffix that starts
far enough back reproduces every row exactly.
`TestPLERowsFromIsTheTail` asserts that at *every* cut point of a sequence with
an EOS in the middle, which is the one piece of per-position state there is.

It is an equality test and not a tolerance one, for `TestPLEGather`'s reason:
these are row indices into a 28.80 GB table, so a wrong index is a plausible
embedding from somewhere else and nothing downstream would flinch.

**The lesson is the profile's shape.** `-depth` had been printing a host
`gather` line that went 3.1% → 11.7% between depth zero and 128k for as long as
the bench has existed, and every session read past it because the
question was always "which kernel". A term that grows 5x with depth is worth
looking at whichever side of the bus it is on.

## 2. `attn.score` ran on sixteen workgroups of a forty-CU device

P9 unpinned the indexer's score from the one workgroup a token it ran on and
striped it `attnCellSplits` ways. It set that to 16 and stopped, and nothing
since asked whether 16 was the number.

It is a **grid and not a reduction** — the kernel's own comment says nothing
sums across the stripe, so the tensor it writes is the same bits however the
grid is cut — so the only thing that bounds the count is having blocks left to
give a workgroup. Measured at decode and 128 000 cells, with
`attn.attn.split` flat to 0.4% as the control:

| splits | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| `attn.score` µs | 211.6 | 118.5 | **80.9** | 78.5 | 86.5 |

64 is the knee; 128 is inside its noise and 256 is past it, where a stripe has
too few blocks to pay for its workgroup. It is **better at every depth and not
only at the deep end** — 0.126 → 0.106 ms a token at depth zero, 2.533 → 0.995
at 128 000 — so there is no crossover to pick a default around, which is why
`attnCellSplits` is a constant and not a function of the cache.

**1.5 ms a token for a one-line change**, and the reason it was there to find
is that P9 was measuring whether the stripe worked at all, not how wide it
wanted to be.

## 3. The gather and the split compose, and neither had the other's factor

This is the change with the argument in it.

P14 built the per-cell gather for prefill and wrote down why decode should not
take it: *a decode step is 24 single-wave workgroups that all fit at once, so
what it costs is one wave's serial walk — which a gather in front of it
lengthens rather than cuts.* That is **right about the gather alone** and it is
measured here. What it misses is that the two cut different things.

Per dispatch at 128 000 cells and one row, four arms, `LLM_ATTN_SPLITS` and
`LLM_ATTN_GATHER` the knobs:

| arm | `attn.attn` µs |
|---|---:|
| unsplit, block walk | 2948.7 |
| unsplit + P13's block list | 2401.1 |
| unsplit + gather | 986.7 (+22.4 of gather) |
| **split 16, block walk** (shipped) | **419.8** |
| **split 16 + gather** (P15) | **129.9** (+21.5 of gather) |

The split is **7.12x on the walk**: it cuts how many key blocks one wave has to
step through. The gather is **2.99x on the work**: one query row selects 2051
cells however deep the cache is, and the block kernel reads every cell of every
32-cell block that holds one of them — at 128k that is roughly eight cells
visited per cell selected. Neither factor is the other's, and **together they
are 3.23x on the shipped kernel** (419.8 → 129.9, two passes agreeing to
129.9/129.9 and 420.7/418.9), 2.77x with the gather's own two dispatches
charged against it.

The gather's cost at decode is the number that decided this, and it was priced
*before* anything was built: **22.4 µs against the 419.8 the kernel it feeds
was taking**, 5%. A decode tile is one real row and fifteen pad rows, so the
"union of sixteen rows" that costs prefill 4.9x over a single row's reach costs
decode nothing — the union *is* the row.

### What it took

`llm_attn_wmma.comp` gains a `GATHER`+`SPLITK` build. Three things had to move:

- **`GATHER` is tested before `SPLITK`**, not after. A build can now be both,
  and when it is, the axis it walks is the gathered list.
- **The list comes in through `cellOff`, not `resOff`.** `resOff` is the
  partial arena to every split build and to `llm_attn_combine.comp` behind
  them. `cellOff` is the indexer's expanded per-cell score, which P14-1 deleted
  and which no attention build has ever read. The push block is full at 64
  uints (`TestAttnGPUPushBlockFits`), so a spare field is the whole budget
  there is.
- The epilogue is `SPLITK`'s, unchanged, and the combine behind it is the one
  P8 already wrote.

Striding the gathered list is sound where striding a *block* list would not
have been. P13 refused that one because a slice has to be a set of absolute key
blocks that does not move under rechunking (L7a's gate). A gathered pass is
already outside that gate by construction — see `SetGather` — and is already
pinned with the other reassociating kernels. What is *not* given up is the
shape of a decode step: it is one row, so there is no second row for a chunk
boundary to fall between.

`TestAttnGPUGatherSplitIsTheGather` is the gate, and it is held to the
**reference** rather than to the unsplit gather, because a split folds the
online softmax's maxima in a different order. rms 1.07e-4 to 1.25e-4 against a
1e-3 bar, and **flat in the slice count** across 2, 3, 5, 8, 16 and 32 — which
is the signature of reordering and not of a dropped slice, because a dropped
slice does not care how many there are.

**The first draft of that test passed instantly with 0.000e+00 on every arm**,
because it uploaded the whole fixture at once and `attnSplits` refuses any
batch wider than `attnSplitMaxRows`. A split that never engaged looks exactly
like a perfect one. The test now runs 64 single-token steps over a 4032-cell
cache and **fails on a zero**.

### The bound, and the first decode plan that moves with the depth

The gathered arm **loses at shallow depth**, and that is not a surprise once
stated: below the selection's 2051-cell width every live cell is selected, the
gathered list *is* the key axis in the same order, and the compaction is two
dispatches and a scattered read paid for nothing. Measured at depth zero,
`attn.attn` is **1.09 → 1.56 ms** a token the wrong way, and the whole model
2.3% down.

So the arm turns on at `2 * selWidth` live cells — the shallowest cache in
which the selection is skipping half of what the block kernel would read —
and the bound is on **the cells that exist, not the cells the cache was
allocated for**, which is P7's rule. `LLM_ATTN_GATHER_MIN` is the ladder.

That makes it **the first knob in a decode step that is a function of the
depth**, and P1c's prerecorded command buffer is built on the opposite
assumption: a one-token pass is the same dispatches every time, so the first
one is captured and every later one replays it. The graph now compares
`decodeEpoch` before each replay and re-records the single step that crosses.
It crosses at most once in a sequence, because the depth only grows.

**The obvious test for that cannot fail**, and the first one written did not:
diffing the logits across the bound with the replay on and off is bit-identical
at any depth a short fixture can reach, precisely because below 2051 cells the
two kernels fold the same partials the same way. So
`TestPrerecordedDecodeCrossesTheGatherBound` asserts the *mechanism* — the
buffer being replayed must be the buffer the current depth asks for, checked
after every step — and with the epoch comparison deleted it fails at the
crossing step by name.

## The refusal: the arena's memory type costs the kernels nothing

`llm/arena.go` has asked for this control since L6b and nothing had ever run
it. The activation arenas come from a HOST_CACHED memory type, because the host
reads them at 24.76 GB/s there against 0.18 on the write-combined type the
device-local heap offers. The file says outright that this is "a claim about
GPU time, and a claim like that needs a run with the knob the other way".

The claim was worth re-testing because **the allocation it was made about grew
by two orders**: since P13/P14 the KV planes share that buffer, which is
3.85 GB at 139 000 cells, and `hc.cn` — a kernel that is pure arena traffic and
does no arithmetic — has sat at 134 GB/s where a copy gets 236, with both of
P12-4's hypotheses spent.

`LLM_ARENA_UNCACHED=1`, whole model, depth 0 and 128 000:

| | cached | uncached |
|---|---:|---:|
| `hc.cn` (prefill, µs) | 1272.2 | 1260.4 |
| `moe.up` (prefill, µs) | 12170.1 | 12131.0 |
| `dn.qkv` (prefill, µs) | 4265.2 | 4259.4 |
| GPU total, prefill (ms) | 1748.5 | 1744.2 |
| GPU total, decode (ms) | 516.9 | 523.9 |
| **host glue, decode (ms)** | **2.7** | **78.7** |
| decode tok/s | 29.74 | 25.73 |

**Every kernel is within 1%.** The memory type is not `hc.cn`'s 134 GB/s, it is
not the wide-arena decode cost, and there is no win available from splitting
the KV planes onto their own type. What moves is host glue, 29x, exactly as
`arena.go` predicted — which is also the evidence that the knob was really
thrown.

That is L6b's claim confirmed at the new scale and a third standing hypothesis
for `hc.cn` spent. What is left there is the access pattern itself.

## Reproducing

	go build -o /tmp/pp ./cmd/llm        # once, never `go run` per arm
	export LLM_BANK_CACHE=models/Qwen3.8-Flash-Next-GGUF/bank-cache
	export LLM_DENSE_BANK=deltanet=q4_k/32,hyper_conn=q5_k/32,lm_head=q5_k/32,full_attn=q5_k/32,qsa_indexer=q5_k/32
	export LLM_MOE_BANK=gate_shexp=q4_k,up_shexp=q4_k,down_shexp=q5_1,down_exps=iq4_nl
	M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

	# the whole table, ~11 minutes
	/tmp/pp -model $M -depth -depths 0,8000,32000,64000,128000 -pp 2048 -tg 16 -ctx 139000

	# the fast loop, ~60 s an arm, and it reproduces the attn.attn ratios
	/tmp/pp -model $M -depth -depths 128000 -pp 2048 -tg 8 -ctx 139000 -layers 4

	# the control arms
	LLM_ATTN_GATHER=0            # P8's block split, which is what decode had
	LLM_ATTN_GATHER_MIN=<cells>  # the depth bound, 2*selWidth by default
	LLM_ATTN_CELL_SPLITS=16      # P9's stripe count
	LLM_ARENA_UNCACHED=1         # the memory type
	LLM_NO_PRERECORD=1           # re-record every decode step

The gates:

	go test ./llm/ -run TestPLERowsFromIsTheTail                  # P15-1, exact
	go test ./llm/ -run TestAttnGPUGatherSplitIsTheGather         # P15-3, the tolerance
	go test ./llm/ -run TestPrerecordedDecodeCrossesTheGatherBound # P15-3, the epoch
	go test ./llm/ -run TestAttnGPU                               # the block
	go test ./llm/ -run TestGraphIsAChunkSplit                    # the graph
	go test ./llm/ -run TestPrerecordedDecodeIsTheRecordedDecode  # P1c

## What is left of decode at depth

The falloff is 0.91x and the remaining ~2 ms between depth zero and 128 000 is,
in order:

- **`attn.select`, 0.862 ms a token and untouched** — it is the one kernel here
  that cannot be striped, because four radix passes over a row are a reduction
  (P10). What it has not had is a *two-level* arrangement: per-stripe histograms
  and a merge. That is a real change, not a constant.
- **`attn.attn.split` is still 1.571 against 1.092 at depth zero.** The gathered
  list is 2051 cells at every depth, so what remains is not the cell count — it
  is that the same 2051 cells are scattered over a deeper cache and each 128-byte
  line costs more to reach.
- **The host `gather` is 1.263 ms and no longer grows**, but `Embeddings` still
  walks a 675 MB mmap per token.

## The ubatch, re-priced at this cache depth

Prefill did not move here at all, on purpose: it is P14's, and its next step is
the ubatch. That trade was measured again, at depth zero and `-ctx 139000`, and
**it is worse than P12-7 found at a shallower cache**:

| ubatch | pp tok/s | tg tok/s |
|---|---:|---:|
| 2048 | 1153.8 | 30.20 |
| 4096 | 1286.7 (**1.12x**) | 26.75 (**0.886x**) |
| 8192 | 1327.5 (**1.15x**) | 23.43 (**0.776x**) |

P12-7 priced 4096 at 1.12x prefill for 0.92x decode and shipped it as the
interactive default. At a 139 000-cell cache the prefill side is unchanged and
**the decode side is 0.886x, not 0.92x** — and 8192 costs 22.4% of decode for
2.6% more prefill than 4096. So 8192 is not a candidate at this depth, and the
`API.md` note that a batch summariser should set 2048 now has a second reason.

The cost is still unexplained, but P15 narrowed it twice.

It is **not the arenas' memory type** (the refusal above). And it is **not the
allocation**, which is the half of "allocated versus used" that needs no new
code: `-ctx` moves only what is reserved — the KV planes are 3.85 GB at 139 000
cells and 0.1 at 4096 — and at depth zero none of it is read. Decode at depth
zero, ubatch fixed at 2048:

| `-ctx` | 4 096 | 32 768 | 139 000 |
|---|---:|---:|---:|
| tg tok/s | 30.23 | 30.28 | 30.17 |
| pp tok/s | 1178.0 | 1167.0 | 1157.1 |

**3.85 GB of extra allocated arena costs decode 0.2%.** So reserving memory a
decode step never reads is free, and the ubatch's 11% is not volume — it is
something about the pass that *wrote* those arenas immediately before. The
decode dispatches themselves are identical across the ubatch ladder (`aSplit`
is sized for `attnSplitMaxRows` and `aPart` for `GEMVMaxRows`, neither of which
moves), so what is left is the residency the wide prefill leaves behind it.

That makes the remaining probe a narrower one than TODO has been carrying:
not "allocated versus used" — that half is answered — but stage the wide
arenas, prefill in **narrow** chunks anyway, and see whether the 11% follows
the arena's width or the prefill's footprint.
