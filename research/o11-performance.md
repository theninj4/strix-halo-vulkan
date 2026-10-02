# O11 — performance

*The O11 stage, broken out of `OCR.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`ocr-vertical.md`](ocr-vertical.md).*


**O11a: a page's regions decode together** (decision 10). Measured first
(`TestLMRowLadder`, a decode step of N rows, one a slot, depth ~200, idle
device):

| rows | 1 | 2 | 3 | 4 | 8 | 12 | 16 | 24 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| O4's kernels (GEMV to 3 rows, GEMM past) | 4.46 | 4.53 | 4.69 | 8.43 | 8.55 | 8.76 | 8.98 | 9.46 | 10.38 | 15.22 |
| **16-row GEMV** (GEMM past 16) | 4.45 | 4.59 | 4.80 | **4.88** | **5.44** | **5.98** | **6.36** | 9.45 | 10.62 | 15.29 |

ms a step, wall, logits for every row included. Sixteen regions decode for
1.4 steps of one. What was built:

- **A paged KV cache** (`ocr/lm.go`, the prep and attention shaders'
  `PAGED` build): every layer's K and V are a pool of 64-position pages
  (the attention's key block, so a block of keys is one page), each slot a
  table of them in the fp32 arena (`pc.heads`, `pc.span` its length).
  `LM.Reserve`/`Release`/`FreePages`; `Pass` reserves what its rows need.
  ACE's `ace_lm_*` binaries are byte-identical (the switch defaults off).
  32 slots x 10,240 positions unpaged would be 6 GB; the pool is 64k
  positions, 1.2 GB.
- **Logits for every row a pass finishes** (one a slot, up to 96), the head
  as a GEMM past 16 rows; the logits are the LM's own reused buffers.
- **16-row GEMVs**: `llm_gemv.comp`'s `MAXROWS` is overridable and OCR builds
  `ocr_gemv_k{1,16}.spv` at 16 (the LLM's binaries unchanged). `gemvRows`
  is the crossover; `OCR_GEMV_ROWS` moves it in the ladder.
- **The scheduler** (`ocr/engine.go`): one goroutine owns the tower and the
  LM. A pass is a row for every decoding request, then the prompts of new
  ones, oldest first, up to `Rows` (2,048), with every row that ends a
  request's input at the tail for its logits (rows may come in any order:
  prep writes every row's K/V before any row attends). Prefill, decode and
  recompute are one code path: a request needs positions [filled, prompt +
  tokens). Admission takes a free slot and the pages for the prompt plus a
  page for each running request; when the pool runs dry anyway, the
  youngest request gives its cache back and prefills prompt + tokens again
  later (vLLM's recompute preemption). Tokens reach `OnToken` through a
  channel on the caller's goroutine, so a slow streaming client stalls no
  one; cancellation is reaped every pass. `RecognizeAll` queues a page's
  regions in one go (a page alone is reproducible; see below), and
  `page.Parser.Recognize` now takes the whole page's requests.
- `Engine.Stats()` counts towers, passes, rows and preemptions;
  `cmd/ocr -page` prints them.

**The one thing batching changes: a request's numbers depend a little on
its company.** A pass of up to 16 rows sums its projections in the GEMV's
order and a larger one in the GEMM's, so a near-tie can flip between a
region run alone and in a crowd. On the O10 subset this flipped **2 of 331
pages** (a space in "Part 1 演示", `align*` for `aligned`); the score did not
move.

Gates:

- `TestEngine` (every case alone) and **`TestEngineBatched`**: all six
  cases at once, token-identical to fp32 HF; the "tight" arm (16 pages,
  admission without the page check) forces **137 preemptions**, still
  token-identical.
- `TestParse`: every oracle page's recognitions identical to the serial
  ones, markdown = PaddleX's.
- `TestOCR*` (backend): chat, streaming, the document door.
- **PaddleOCR's `doc_parser`, unchanged, against `serve -ocr`**: the demo
  page's 27 concurrent requests batch; exit 0, markdown byte-identical
  across three runs, 7.6 s end to end (Paddle's CPU layout included).
  Three streams dropped mid-generation are reaped and the server keeps
  answering.
- **The O10 subset again** (`pred/go_rect_o11`): 18 min 12 s for 331 pages
  (O10: 66 min of recognition alone); recognition 992 s against 3,897,
  **3.0 s a page mean against 11.8**; **96.13, every column identical to
  O10's go_rect**; 329 pages byte-identical. That run had the GEMM past 3
  rows; **the rerun with the 16-row GEMVs** (`pred/go_rect_o11b`): 17 min
  8 s, recognition 929 s (2.8 s a page mean), **96.13 again, every column
  identical**, 328 pages byte-identical to O10.

Where a page's time goes now (`cmd/ocr -page`, idle device except where
said):

| page | regions | serial (O10) | O11a | towers | LM passes | note |
|---|---|---|---|---|---|---|
| demo | 27 | 8.3 s | **2.7 s** | 0.87 s | 1.58 s, 244 | the longest region is 240 tokens |
| newspaper `0b1bb8d0…_1` | 194 | 143 s | **18.2 s** | 8.3 s | 9.3 s, 755 (78 rows a pass) | towers are half |
| Washington Post p. 42 | 103 | 120 s | **28.8 s** | 4.5 s | 23.6 s, 4,097 | one region loops to the 4,096 cap and decodes alone |

(The newspaper rows ran beside the OmniDocBench scorer on the CPU.)

**Refused: several crops in one tower pass.** O3 read the small crop's
33 ms as ~75 synchronous submits. It is not: the whole tower in one submit
is 29 ms against 32 (line, 608 patches), 123 against 128 (seal), 346 against
355 (page), and a `ForwardMany` that ran 27 crops in 4 passes (shared GEMMs,
attention a crop at a time) took 907 ms against 919 one at a time. 608
patches are ~0.9 TFLOP, so 29 ms is the ~30 TFLOP/s our GEMMs reach: the
tower is compute. Built, measured, reverted; what stayed is the submit size
scaled by patch count (`vision.GPU.PerSubmit`, 8 x 5120/patches a submit),
worth the 3 ms.

**Left for O11b**, by where the time is: the towers (half of a dense
page: int8 or faster GEMM rungs, since it is compute); the long tail (a
single region decoding alone at depth, the scalar attention O5 saw, and
repetition loops PaddleX truncates afterwards but generates to the cap, as
vLLM does); a device argmax (sampling is ~1 ms a pass at 64 rows); a GEMV
past 16 rows.
