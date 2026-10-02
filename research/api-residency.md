# Residency, and the one place it is not resident yet

*The residency section, broken out of `API.md` when it moved to `research/` on 2026-10-02; the server's overview, flags and envelopes are in [`api-server.md`](api-server.md).*


**The language model is resident, and it is most of the machine.** `-llm`
stages all 48 layers, the dense banks and the 512-expert bank at startup —
**33.5 s**, and the process then holds ~84 GB — so a request is a prefill and
one forward pass per token with no weight ever moving again. Two numbers are
decided at startup rather than per request: `-llm-ctx`, the attention cache's
cell count and so the longest conversation the process can hold, and
`-llm-batch`, how many tokens the prefill arenas hold. A conversation past the
cache is refused with both numbers rather than truncated at one end; a prompt
past the batch is prefilled in chunks of it, which is the same computation by
L7b's gate.

**`-llm-batch` is 4096, and a wider batch costs memory and nothing else**
(P16). Measured through this server on four ~3.9-4.3k-token prompts, 256
generated tokens, two passes agreeing to 0.1%: 2048 prefills at **1052
tok/s**, 4096 at **1199** (1.14x, ~0.5 s off the time to first token) and
8192 at **1233**, and all three **decode at 34.2 tok/s**. (P12 measured 4096
costing 8% of decode; that was a bug, fixed in P16.) 8192 is only faster on
prompts longer than 4096 tokens, holds ~3 GB more arena, and lowers the
longest `-llm-ctx` the attention cache can reach, so it is the setting for a
deployment whose prompts are long.

**The device lock is taken per forward pass and not per request.** A
completion is seconds long and the speech models are milliseconds, so holding
the queue for a whole generation would make a transcription wait behind it for
nothing: between two decode steps there is no work in flight. What is held for
the length of a request is a *slot* of the graph, whose cache, PLE ring and
DeltaNet state are one sequence's.

**`-llm-slots N` holds N conversations at once** (CONCURRENCY.md, C1/C2). Each
slot is its own KV cache (27.8 KB a cell, in buffers of its own since C6, so
every slot can hold the full 262 144 cells), DeltaNet state and rings (120
MB). The served line is `-llm-slots 3 -llm-ctx 262144`, ~98 GB resident. A
request holds one slot for its life, and a scheduler interleaves every live
request's work a **unit** at a time — one prefill chunk or one decode step.
The decode steps of concurrent conversations of one class **share a pass**,
a row each (C5, `-llm-batch-decode`). Rows cost less than passes: three
streams get ~18 tok/s each rather than a third of ~34 each, 56 tok/s in
total at 48 layers. The order is:

1. **Interactive before background, strictly.** A request is interactive if
   it sends `X-Priority: interactive` or `service_tier: "priority"`, and
   background with `X-Priority: background` or `service_tier: "flex"`;
   otherwise `-llm-class` decides (background by default). While an
   interactive request holds a slot no background unit starts, so a voice
   command decodes at the full single-stream rate. Measured on the 4-layer
   prefix: a 16-token interactive request beside two decoding background ones
   finished in 166 ms, against 419 ms sharing with them as an equal.
2. **Within a class, the least device time served goes first**, so a long
   prompt cannot take the device from a conversation that is decoding.
3. **A background prefill runs in `-llm-preempt-chunk` pieces** (2048), because
   a unit cannot be interrupted once it is on the device and an interactive
   request can arrive at any moment.

`-llm-reserve` (1) slots are only for interactive requests, so background
agents can never occupy every slot. A request that finds no slot it may take
waits for one, and the log line says `(Ns queued)`. With more than one slot the
line also names the slot, the class and how long the request's units
`stalled` behind other conversations'.

**Parakeet is resident.** `-stt` stages the encoder and the transducer tail
once, sized for `-max-audio` seconds (60 by default), and every shorter clip
runs in the same arenas — `UploadMel` settles the geometry per run. A request
costs a front end on the host and two submits. A longer clip is refused with a
message saying what the server was sized for, rather than being truncated.

**The embedding model is resident**, and it is the cheapest thing here:
`-embed` stages 0.88 GB of fp16 banks in **1.3 s**, and an input then costs
**11.5 ms** whatever it says — a 27-token query and an 8-token document take
the same time, because at these lengths the run is the weights read once
(research/embedding-vertical.md E6). `-embed-tokens` sizes the arenas and is therefore where an
input is truncated: the front of the text survives and the end-of-text token
is re-appended, because last-token pooling takes *that* row and an input
truncated through it would be pooled from an ordinary word.

**Image, music and video take turns in one slot** (`backend.Swap`,
2026-09-27). Each is 26-46 GB and asked for rarely, so a request makes its
vertical resident, unloading whichever other one is, and runs with the slot
held; after `-swap-idle` (10 min) with no request the slot is emptied. Each
is staged once at startup so a bad checkpoint fails then, and the last one
staged (music, when it is on) is resident when serving begins. A load takes
no device lock -- staging makes no queue submit (`TestStagingSkipsTheQueue`,
`STAGE_GPU=1`) -- so speech is answered through it. Served, with the unit's
flags (`-embed -tts -tts-gpu -stt -kev -image -edits 3 -video -music`), one
image → music → video → image cycle, 20 ms sampling:

| state | process over the machine's idle | switch |
|---|---|---|
| nothing resident (idle unload) | 9.4 GB | image unloaded in 0.5 s |
| music resident | 35 GB | loaded in 18.6 s; a 30 s song in 20.5 s end to end |
| image resident (`-edits 3`) | 56 GB (38.1 GB device, ~8.6 GB of host heap) | loaded in 24.7-30.6 s; unloaded in 0.3-0.5 s |
| video, at its peak | 40 GB | nothing to load; 448x256 x 8 steps in 2m23s |
| **worst moment of the cycle** | **60 GB** (an image staging) | |

All three resident was ~112 GB at a video's peak, which is why the unit ran
without image and video. Speech through the cycle: median 19 ms, p95 105 ms,
≤ 0.86 s through every load; the two waits past a second (5.4 and 6.5 s)
were image generations, which hold the device for their whole run.

**The image pipeline is resident while it holds the slot, and its
resolution is not part of what is resident.** `-image` stages 13.2 GB of text encoder, 13.3 GB of transformer
and 1.0 GB of VAE in **28 s**, sizes 5.0 GB of activation arenas for
`-image-size`, and then a request moves no weight. That last clause is what
this endpoint needed and did not have: the pipeline used to fix its width
and height at construction, so a size was a residency question.

It is not one, and none of the three graphs had to change to stop it being
one. The transformer states its run length per image, the VAE re-records its
graph from the latent it is handed and checks the arena rather than assuming
it, and the rotary table is rebuilt per run anyway. So `-image-size` is a
*ceiling* and a default, the way `-max-audio` is for parakeet, and every
smaller image runs in the same arenas. Measured on one 1024x1024 process,
two runs, at the default 40 steps:

| stage | 1024x1024 |
|---|---|
| text encoder | 107 ms / 105 |
| DiT prefill (step 0, which also fills the prefix KV cache) | 2.136 s / 2.163 |
| 39 cached steps, mean | 2.117 s / 2.143 |
| VAE decode | 3.96 s / 3.96 |
| **request** | **1m28.8 s** / 1m29.8 |

**An image is the transformer and almost nothing else**: 95% of that wall
clock is the 40 denoising steps, 4.5% is the VAE and 0.1% is the text
encoder. The two levers are therefore the step count — genuinely per request
here, unlike under a turbo distillation — and the unported percents
research/qimage-vertical.md's Q9 prices. (The VAE was 8.1% until Q9b halved its decode, in
fp32: the matrix cores are refused there on precision, so what changed was
a register block and a GEMM tile, and the output is bit for bit the same.)

An *edit* at the same size, with one reference image, is **1m54.2 s /
1m56.6**: the reference's own encoding 7.1 s (a 27-layer vision tower and a
VAE encode), text encoding 0.73 s, the prefill 5.19 s against a generation's
2.14, 40 steps at 2.47 s against 2.12, and the same 4.0 s decode. The extra
is the prefix — thousands of rows of reference latents that every step
attends over.

Over HTTP, on a `-image -image-size 512x512 -image-steps 24` process: a
512x512 image at 24 steps comes back in **16.8 s then 14.5 s**, base64 body
included. **The step count is worth sending.** The sweep in
research/qimage-vertical.md (same seed, three prompt kinds, 1024x1024) found
the answer is prompt-dependent rather than a single number:

| steps | wall | photographic | painterly | structured/technical |
|---|---|---|---|---|
| 40 (default) | 1m38 | good | good | good |
| 24 | 1m4 | good | good | good |
| 16 | 45 s | good | good | washed out, structure fragmenting |
| 12 | 36 s | good | good | broken |

(The wall clocks are the sweep's own, taken before Q9 and Q9b's 10% — the
ratios are what it was measuring, and 40 steps is 1m28.8 today.)

40 is the default because it is the only count safe across all three; **24
is the honest fast setting** at −35% with no visible loss on any of them,
and a client that knows it is asking for a photograph can go to 12 for a
third of the cost.

**The ceiling is 1184x1184, and it is not a budget.** The VAE decoder's
activation arena is a single storage buffer, this device caps one at
`maxStorageBufferRange` = 4 GiB − 4, and a decode needs 3060 MB at 1024²,
4090 MB at 1184² and 4315 MB at 1216². So `-image-size 2048x2048` — the size
the model card's own examples use — is refused at startup with the
arithmetic, rather than failing after 27 GB of staging. Lifting it means
tiling the decode or splitting that arena across buffers, and is Q6's own
follow-up.

**Kokoro is not.** Its arenas are sized for one utterance's frame count, and
the durations decide that, so nothing can be staged until the phoneme side has
run (research/speech-vertical.md T2, T4c). `-tts-gpu` therefore stages *per request* — the
sequence `cmd/tts` uses, unchanged, because it is the one T4 and T6 measured —
and it is **off by default** until that cost is measured against the host
path, which is what `-tts` alone runs.

Two things would fix it, and both are measurements rather than arguments:

- **Bucketed residency.** Size the arenas above the utterance and zero-pad the
  alignment up to them, trimming the tail by `Prosody.Samples`. The phoneme
  side already accepts any frame count up to the one it was built for
  (`GPUPhonemes.sharedGraph`); whether the generator's stages tolerate the
  padding is the open question.
- **Skipping the second prosody pass.** The vocoder still reads `asr`, `f0`
  and `energy` from the host (research/speech-vertical.md T7), so the host prosody that settles
  the length could feed the device vocoder directly and the phoneme side would
  not need staging at all.
