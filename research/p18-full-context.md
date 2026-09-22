# P18 — the full trained context: 262 144 cells

**2026-09-22.** The model is trained to 262 144 tokens
(`qwen4exp.context_length`), and the server could hold about 148k of them.
The KV cache shared one fp16 arena with the prefill activations, and
`maxStorageBufferRange` is 4 GiB - 4. It now holds all 262 144, and the answer
is checked there, not just the rate:

- **Perplexity over positions 131 072-262 143 of one 262 144-token window:
  4.0344 ± 0.024** (`-ppl -ctx 262144 -chunks 1 -ubatch 4096 -head-rows 2048`,
  5m27s, `results/p18_ppl_262144.txt`). Every scored token reads the cache past
  the old cap. A cache that read back zeros there — the failure P14 found is
  silent — would not score 4.03.
- **Recall through the server:** a 237 286-token prompt (the first 1.03 MB of
  `wiki.test.raw`) asking who the *first* article is about. It answers
  **"Robert Boulter"**, which is the first line of the file. Prefill averaged
  **1022.3 tok/s** over the whole prompt, and decode ran at **29.86 tok/s at
  237k deep** (`results/p18_serve_262144.txt`).

## The change

The key plane, the value plane and the indexer's raw and pooled keys each get
a buffer of their own (`AttnGPU.kbuf/vbuf/ibuf`, bindings 8, 9 and 10 in
`llm_common.glsl`). Over twelve layers that is 12 KB, 12 KB and 3.8 KB a cell,
so the trained context is 3.22 + 3.22 + 0.81 GB, and K and V reach the range
at ~349k cells. The five kernels that touch the cache read and write it
through the new bindings: `pack` (writes K, V and the raw index key), `idx`
(pools the raw keys; the pooled key goes to the cache and the query to the
arena), both `score` builds, and `wmma` (K and V, through `coopMatLoad` and the
gather's `stagePlane`). Offsets are unchanged in meaning, just relative to
the new buffers. Every attention pipeline binds 0-10, with the bank padding
slots 5-7. Binding 5 was already the bank for the quantised GEMMs and the
GEMV, so `gemmBufs` is now the same list.

Two supporting changes:

- **`vk/shim.c`'s `SHIM_MAX_BINDINGS` 8 → 16.** Pipeline creation failed with
  `VkResult(-3)` at 11 bindings. It is a stack array bound, nothing more.
- **`cmd/llm -ppl -ubatch N`**: a window wider than any arena is prefilled in
  `N`-token batches (with P17-2's prefetch), and the scored half goes
  through `ExtendRows` a head-arena at a time. On the 4-layer prefix at a
  4096 window it agrees with the one-pass path to 2e-5. `-chunks N` now
  needs N windows of corpus, not llama.cpp's two, so one 262 144-token window
  fits `wiki.test.raw`'s 297 193 tokens.

The cap check (P14's `checkBufferRange`) now runs per buffer, and its message
names the per-plane limit.

## Gates

	go test ./llm/ -run 'TestAttnGPU|TestGraphIsAChunkSplit|TestPrerecordedDecode|TestInPortPadsTheRunNotTheArena|TestSpec|TestRewind|TestSnapshot'

All pass unchanged, including the exact ones (cache size, chunk split, the
rollback snapshot of the pooled keys, which now reads `ibuf`). Moving a plane
between buffers changes no arithmetic.

## The served configuration

`ai.service`'s LLM line: `-llm -llm-ctx 262144 -llm-batch 8192`. It stages in
**1m05.7s at ~83 GB**, leaving ~33 GB available on the 117 GB machine.
Batch 8192 because P16 showed a wider batch costs decode nothing and it is the
faster prefill for exactly the prompts this context exists for. At 262 144
cells it fits both the fp32 arena (the indexer's scores are 8192 x 65 536
floats, 2.1 GB) and the range. 4-layer probe at 248 000 cells, ubatch 4096:
prefill 0.86x of depth zero, decode 0.98x.

**Not changed:** `-llm-max-tokens` stays 1024. It is what a request that names
no `max_tokens` gets, a product choice rather than a rate, and this model
reasons before it answers.
