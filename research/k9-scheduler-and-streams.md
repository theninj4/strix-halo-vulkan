# K9 — scheduling, the host head, and chunked states (2026-09-28)

*The K9 stage, broken out of `CLASSIFICATION.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`classification-vertical.md`](classification-vertical.md).*


Kev alone on the machine, measured served (`cmd/serve -kev`) with a new
open-loop client, `cmd/kevload`: short requests are the README ticket
(~100 packed tokens) and long ones the 2,300-token report of
`TestGPUCacheLatency`, every text distinct so the cache never answers.
Production's log over the week before: 936 requests, p50 52 ms, p90 95,
p99 197, max 888.

**The baseline's three problems.**
1. **All or nothing.** The worker drained up to eight requests and ran
   them as one pass, or, if they did not all fit one, one at a time.
2. **Head-of-line blocking.** A short request queued with a long text
   rode in the long one's pass: eight shorts sent with one long came back
   in 1068 ms, not ~250.
3. **A pass is not preemptible.** A long text that finds the device idle
   starts its 760 ms pass, and every short request that arrives meanwhile
   waits for all of it: at 10 shorts a second plus a long every 2 s, short
   p90 was 847 ms.

**What changed.**
- **The scheduler** (`backend/kev.go`, `pickKev`). The worker keeps a
  waiting list, prices each request in pass rows (`GPU.PassCost`: its
  branches, plus its state unless cached) and packs a pass
  cheapest-first up to `-kev-batch` requests and **`-kev-batch-tokens`
  (1024) rows**. Past ~500 rows a pass is compute-bound (K7.1), so a
  bigger one buys nothing but waiting. A request over the budget runs
  alone, after cheaper ones, and one that has waited a second goes first.
  A request that does not fit is skipped, not the end of the pass
  (`GPU.FitsStep`).
- **The pointer head on every core** (`Head.LogitsMany`). The head was
  4.2 ms of a 50 ms request: a dozen 256×2560 projections in float64 on
  one core, and eight times that for a batch of eight, with the GPU idle.
  Now every row's projection of the whole pass is split over the cores
  in 32-output blocks, and each output is still one sum in index order,
  so the bits are the serial loop's (`TestHeadLogitsManyIsSerial`). A
  short request's host time, measured stage by stage: parse 0.03, encode
  0.15, upload 0.2, submits 0.8 over the kernels, readback 0.07, head
  **4.16 → 0.62**, cache save 0.7 ms.
- **Streams: a long state in chunks** (`kev.GPU.BeginStream`/`Step`,
  `-kev-chunk`, 512). A request over the budget whose state is longer
  than a chunk becomes the stream. Its state runs 512 tokens a pass, and
  other requests' passes go in between: the stream's next chunk goes when
  nothing cheaper waits, or once it has waited a second, with cheaper
  requests riding along inside the budget. The stream holds GDN slot 0
  and attention cells `[0, Ls)` until it finishes, and the planner
  (`planStep`) gives everything else the slots and cells after it. One
  stream runs at a time, and a second long request waits for it. The
  kernels needed three things:
  1. A **later chunk continues the scan from its slot's S**, flagged in
     the segment table's spare fourth word (`kev_gdn_scan.comp`).
  2. It **attends to the earlier chunks' cells** as its request's state
     range `[sLo, sHi)`. That is the mask that already existed.
  3. It **reads its conv history from the previous chunk's tail**. A
     chunk also writes its own tail, and in one dispatch a workgroup
     reading the tail and another writing it would race. So chunks
     alternate between the tail and a second one (`tail2`, per layer and
     slot, 19 MB), chosen by bit 1 of the row-kind word, and the last
     chunk always writes the first tail, which branches and the cache
     read.

  Chunks are multiples of the 64-key block, so every key block is the
  same as in the whole state. **The answers are the same bits**
  (`TestGPUStreamMatchesWhole`): alone, with other passes between chunks,
  and sharing passes with them, at chunks of 128 and 192, on both banks
  and both attentions. So are the requests around the stream, and the
  stream's cached state on a hit. Breaking the tail alternation moves
  answers by 3.8e-2, so the test can see it. Over HTTP the long request's
  response is identical with `-kev-chunk 512` and `-1`.

**Gate passed.** transfer-v4 dev rows are **byte-identical** to the K7.7
build, at batches of 1 and of 8 (0.8140). Wall time is 36.7 → 35.4 s and
22.4 → 21.5 s, all of it host time. `go test ./kev/` passes, and so do
the batch, chunk, stale-cache, isolation, cache and stream tests in all
four `KEV_BANK`/`KEV_ATTN` combinations.

**Served, before → after** (ms, `cmd/kevload`, `-kev-chunk 512`):

| | baseline | K9 |
|---|---|---|
| 1 short | 51.7 | **48.2** |
| 16 shorts at once: p50 / max, throughput | 317 / 540, 29.5/s | **296 / 503, 31.6/s** |
| 8 shorts + 1 long at once: shorts p50 / long | 1068 / 1070 | **247 / 934** |
| 10 shorts/s: p50 / p90 / p99 | 53 / 119 / 190 | **49 / 104 / 167** |
| 10 shorts/s + a long every 2 s: shorts p50 / p90 / p99 / max | 167 / 847 / 1060 / 1073 | **87 / 187 / 263 / 290** |
| the same: long p50 | **771** | 1031 |
| 25 shorts/s: p50 / p90 / p99 | 158 / 325 / 507 | **126 / 238 / 373** |
| 1 long alone | 753 | **679** |
| 4 longs at once: mean / last | 2702 / 3349 | **1709 / 2729** |

**The chunk** (the same binary, `-kev-chunk`):

| chunk | long alone | mixed: shorts p90 / p99 | mixed: long p50 | 4 longs: mean |
|---|---|---|---|---|
| off | 753 | 812 / 976 | 762 | 1883 |
| 256 | 771 | **133 / 168** | 1259 | 1923 |
| **512** | **679** | 187 / 263 | 1031 | **1709** |
| 1024 | 726 | 364 / 430 | 1029 | 1800 |

512 is the default. It is the fastest for a long text even alone, which
was not the aim. The likely reason is that a 512-row pass gets the
row-block-fast GEMM rungs and eight layers a submit (a pass over 2048
rows submits one layer at a time), but that is not measured. What the
mixed load pays for its shorts is the long request's latency, 762 → 1031
ms, while it shares the device.

**Not done.** Two concurrent requests with the same new text each
compute it. The scheduler could hold the second until the first has
cached it, which pays only when clients send parallel questions about
one text. The short-request floor is still the int8 GEMMs' ~125 GB/s
read (K7.1). Past 32 req/s the passes are compute-bound.
