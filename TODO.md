# TODO — the state of play

> **Rewritten 2026-09-21.** The per-vertical progress files were
> consolidated into [`research/`](research/README.md) on 2026-09-20 (`LLM.md`,
> `LLM2.md`, `SPEECH.md`, `TTS.md`, `IMAGE.md`, `PIPELINE.md`,
> `EMBEDDING.md`, `IDEAS.md` and the old session-log `TODO.md`) — see the
> file map at the bottom. The root `IMAGE.md` then came back live for one day
> to carry the Qwen-Image-2.1 replacement, and was frozen in turn on
> **2026-09-21** as
> [`research/qimage-vertical.md`](research/qimage-vertical.md). **No live
> progress file remains at the root**: this one is it — what each vertical
> is, what it measures today, and what is open. It is **rewritten**, not
> appended to; a closed item's write-up goes to `research/` and history
> lives in git.
>
> Addresses: `§N.M` (cited from code) resolves in
> [`research/ideas.md`](research/ideas.md); stage letters (L7d, T9, S10,
> Q9b, E7, P6…) resolve in the vertical archives listed at the bottom;
> decisions **D1–D21** are in
> [`research/llm-vertical.md`](research/llm-vertical.md). A code comment
> citing `IMAGE.md` resolves by what follows it: **Q-stages, numbered
> decisions and Q-o numbers** are `research/qimage-vertical.md`, **I-stages**
> are `research/zimage-vertical.md`. `GOALS.md` is the target; `API.md`
> documents the server as it answers today.

## The five verticals, at a glance

All five of `GOALS.md`'s models run end to end in Go on Vulkan, validated
against a reference, and are served by `cmd/serve`. The competition from
here is our own ceiling, not a reference implementation.

| vertical | model | headline, measured | open |
|---|---|---|---|
| text generation | qwen3.8-flash-next (180 B, 6 B active) | decode **36.19 tok/s, 1.44x** llama.cpp at **+1.74%** perplexity; prefill **1213.5 tok/s at 8192 rows, 3.10x**, still climbing where llama.cpp plateaus | batching (P6); context depth; speculation parked at 0.95x |
| speech → text | parakeet-tdt-0.6b-v3 | an 11 s clip in **43 ms — 257x real time**, whole model resident | S10 front end (48% of the pipeline); S9 long clips |
| text → speech | Kokoro-82M | **31 ms for 3.25 s (105x)**, **162 ms for 19.5 s (120x)** — flat per second of audio; the endpoint answers in 59 ms | the vocoder's 20 ms of arithmetic; three small boundaries |
| image generation + editing | Qwen-Image-2.1 | 1024², 40 steps in **1m28.8s**, 31.5 GB resident, native RGBA; streaming previews cost **0.3%**; the fp32 oracle's picture to mean **3.4e-4**. **Edits answer too**: **1m54.2s** on one reference at 1024², 39.4 GB, the oracle's edit to max abs **0.0014** | **parked 2026-09-21** — Q0–Q12 all closed; the 1184²-area ceiling is the one capability left unbuilt |
| embeddings | Qwen3-Embedding-0.6B | a text in **11.5 ms**, the card's similarity matrix to 1.3e-4 over HTTP | E7 batching, worth up to 10x on short texts |

**The server** (`API.md`): one process, one flag per vertical, OpenAI-shaped
(`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/audio/*`,
`/v1/images/*`, `/v1/embeddings`), plus a Wyoming door for Home Assistant
(`-wyoming`, byte-identical audio to the HTTP door). Everything unimplemented
is refused with a reason, never faked.

**Deployment is two machines**: the language model alone on one (~84 GB
resident), the other four verticals on the other (image ~32 GB, the rest
~3 GB together). `-llm` and `-image` do not fit in one 128 GB process, on
purpose — so no footprint quantisation is planned for the small verticals.

**Where the work goes next.** With the image vertical parked, nothing in the
repo is mid-stage: every open item below is a fresh start, and each vertical's
list is in its own rough order of value. Across all five, the largest
**measured regression** is the language model's context depth — decode falls
**24.27 → 3.98 tok/s between depth 0 and 64k (6.1x) and all of it is
attention**, with 128k not completing at all. It is also the only open item
that is a *regression* rather than an unbuilt capability or an unclaimed
percent, and it still owes a second run before anything is attributed. After
that the two capability gaps are P6 batching (blocked on a product question:
will the API serve more than one stream?) and E7's batched embeddings, worth
up to 10x on short texts; the largest single-vertical percent is S10, the
speech front end at 48% of its pipeline.

---

## Text generation (archive: [`research/llm-vertical.md`](research/llm-vertical.md), review: [`research/llm-review.md`](research/llm-review.md))

**Where it stands.** Phases 1 and 2 are done on both axes and every P-stage
through P5 is closed. The shipped configuration is **D19 + D20 + D21**
(4.5-bit dense bank with a fifth bit on three families, the MoE rows
transcoded, `ffn_down_exps` at IQ4_NL): 4.132 GB a token against a 58.6
tok/s ceiling, perplexity 4.0992 (+1.74% of our own 4.0289, which is itself
−0.13% against llama.cpp's at identical weights). Decode 36.19 tok/s,
prefill 1089 tok/s at ubatch 2048. Served with prefix reuse; a second turn
extends the graph's state rather than re-prefilling.

**Speculation (P5) is built, lossless, and parked at 0.95x.** The rollback
costs nothing when off. What would take it past 1.0, in order: the draft
head's **acceptance on a real workload** — 65.6% on prose against a
break-even of ~0.72; measure chat/code with the observer
(`cmd/llm -mtp`, ~3 min, no machinery) *before* building anything, and only
above a ≈ 0.8 does the rest get interesting — then the recovery round
(~0.95 → 1.05x) and pre-recording the verification pass (~2%). Hard
ceiling: speculation refuses past 2048 cells (`blk.48` has no compress
ratio). Narrowing the trunk makes speculation worse, twice — the draft
stages at checkpoint widths and acceptance falls as the trunk moves away
from what the draft predicts.

**Open, in rough order of value:**

- **P6 — batching.** Blocked on the product question: will the API serve
  more than one stream? Each sequence owns 113 MB of DeltaNet state plus
  rings and KV. P5b already built the first stage (R-row decode GEMVs,
  shipped at R = 2); raising `GEMVMaxRows` must land in the same commit as
  its rung in `TestGraphIsAChunkSplit` — R rows were correct in a block test
  and wrong in the model three separate ways (P5c).
- **Context depth is now the largest measured regression.** One-run figures
  from `cmd/llm -depth` (2026-09-19, shipped banks, no reproducibility pair
  yet): decode **24.27 → 3.98 tok/s between depth 0 and 64k — 6.1x — and it
  is all attention** (tg attention share 36.4% → 86.1%; MoE flat). Prefill
  halves. The QSA sparse selection does not stop it. And **128k does not
  complete**: the fill dies at ~115k cells in the P0 timestamp pathology
  ("N of 1025 timestamp slots never became ready"), reached by depth instead
  of row count — the untried route is making `DispatchMultiMarked`'s
  per-dispatch marks optional, since they are pure instrumentation. Needs a
  second run, then attribution of the attention growth (the 2051-cell read
  cap says the *selection* is not the term that grows).
- **Long-context gates** (idea 7, enabled by P0): perplexity at ctx 4096 is
  done (3.9392, −2.23% against ctx 2048 — the selection helps); a needle
  test through the API is not run.
- **The downstream task eval** — the last unpriced thing about the widths.
  At +1.74% of wikitext perplexity the instrument may no longer separate
  plans; a few hundred multiple-choice items through the HTTP API is an
  evening.
- **`Reset` is owed a test.** A fresh sequence is not independent of the one
  before it (P5c finding 6): three plain greedy runs over one prompt give
  three different texts, deterministically, when the previous run left cells
  past the new prompt's end. Any future token-for-token claim needs the
  plain-against-plain row printed beside it.
- **Priced and not taken** (in the archives, with numbers): the router's
  padding (+0.09 tok/s), re-screening the GEMV rungs at the q5 width, fp16
  DeltaNet state (+~1.3 ceiling), the unpack prefetch (~1.1x prefill), the
  asymmetric epilogue's GEMM arm, W4A8, the hot-expert fast path, the
  float-atomic combine, the vision tower. Also unexplained and written
  down: a dispatch is 8–22% slower inside a step than alone on a cold bank
  (P1b's 1.33 ms environment gap), and dispatch time is not a function of
  its bytes in either direction (P3a/P4b, opposite signs).

**Instrument rules that must survive** (each earned the hard way): a
whole-model number needs a **same-hour control** (P1c — the machine moved
8.7 ms a step overnight); a ladder rate above 242 GB/s is an L3 hit, and
its *byte count* can lie too (D16, P4a); a plan is **measured, never
composed** (additivity leaks both ways through the n-gram block); a control
has to be able to fail (P5b shipped green tests on stale SPIR-V — `.comp`
edits need `go generate` before measuring); benchmarks need
`LLM_BANK_CACHE` set or a shipped-bank run re-fits for 8 minutes
(`cmd/serve` sets it, `cmd/llm` does not).

## Speech → text (archive: [`research/speech-vertical.md`](research/speech-vertical.md))

**Where it stands.** S1–S8 done: front end 21 ms + encoder 17 + decode 5 =
43 ms for jfk.wav, transcript exact, word/segment timings from the model's
own TDT durations, served at `/v1/audio/transcriptions` and over Wyoming
(with resampling at that door only).

**Open:**

- **S10 — the front end is 48% of the pipeline**: a few thousand 512-point
  float64 FFTs on the host. Either a float32 radix-4 on the host or the
  STFT + mel filterbank as two dispatches (the filterbank is a
  `[T, 257] x [257, 128]` GEMM; `kokoro_istft.comp` is the worked inverse).
- **The submit+fence is 38% of an emission** (40 µs of 105 per token). A
  persistent kernel — which is also what streaming transcripts would want —
  and/or speculating on blanks (consecutive-frame joints are independent
  during a blank run: one GEMM at M = 16 for the price of M = 1).
- **S9 — long clips.** Full attention means a chunk boundary changes every
  frame; chunking belongs in the design. The quadratic term arrives around
  1024 frames (~82 s); `-max-audio` refuses past the sizing today.
- Small: cache `UploadMel`'s sinusoidal position rows (1 ms, depends only
  on T); the eight per-head position-score GEMMs are §3.5's grouped shape.

## Text → speech (archive: [`research/speech-vertical.md`](research/speech-vertical.md), recap: [`research/tts-recap.md`](research/tts-recap.md))

**Where it stands.** T1–T10 and W1 done. Every stage on the device, staged
once for the life of the server (T9: the endpoint went 550 → 59 ms with
byte-identical audio); voices blend in upstream's own spelling (T8); T10
put PL-BERT's attention on the matrix cores so synthesis is a straight
8.0 ms per second of audio at every length. The round trip
(`cmd/roundtrip`) closes at 70.1x real time, six prose cases exact.

**Open, in order of what a round trip buys:**

- **The generator's 12 ms and the tail's 8** — the only arithmetic-bound
  parts of the model, 65% of a short utterance, measured since T4.
- **The host embedding stack** (R1's #2): `bert` is 12 ms of a paragraph
  and only 2.5 on the device — the "not worth a dispatch" comment is stale
  the same way the attention kernel's was.
- **Three boundaries, one change each**: the excitation's 0.2 ms is 88%
  submit+readback (move the two noise convolutions onto the device and
  nothing of that stage touches the bus); the phoneme side's remaining 6 ms
  is readbacks and submits (move the vocoder's input boundary); a 16 kHz
  path out of the vocoder would delete the client resample (9.7% of the
  loop — can the iSTFT head be asked for the rate directly?).
- **G2P on unseen text**: designed corpus 24/24; on 400 unseen sentences
  68.8% of sentences, 92.4% of phoneme words agree with misaki. Closing the
  gap is more *measured* rules — the syntactically obvious ones scored
  worse than no tagger.
- **Known, unsettled**: the style row is indexed by the phoneme *character*
  count (upstream's behaviour), so a front end emitting different characters
  picks a different row. And kokoro spells numbers out where parakeet writes
  digits back — the round trip measures those three cases and does not
  count them as failures.

## Image generation — **parked** (archive: [`research/qimage-vertical.md`](research/qimage-vertical.md))

**Where it stands.** Done and **parked 2026-09-21** — parked because the plan
ran out, not because it stalled. (The superseded z-image-turbo vertical is
[`research/zimage-vertical.md`](research/zimage-vertical.md) plus
[`research/zimage-pipeline.md`](research/zimage-pipeline.md); its stages are
**I0–I7** and none of its code survives except `zimage/qwen` and
`zimage/tokenizer`, which `embed`, `llm` and `parakeet` import.) The vertical was replaced 2026-09-20
(Z-Image-Turbo out, `Qwen/Qwen-Image-2.1` in, for native RGBA and
reference-image editing — `GOALS.md` #4), and **stages Q0–Q12 all closed
inside two days**. Both endpoints are served: `POST /v1/images/generations`
answers at **1m28.8s / 1m29.8s for a 1024²/40-step image** (31.5 GB resident,
matching the fp32 oracle's own picture at mean 3.4e-4) and `POST
/v1/images/edits` at **1m54.2s / 1m56.6s for a 1024² edit on one reference**
(39.4 GB, matching the oracle's edit at max abs 0.0014), both with native
RGBA and unconditional in-progress previews (a fitted 64x4 matrix, 159 µs a
frame, three partials for 0.3% of a request — no flag, because there is
nothing to load). Q10 turned the ceiling from a side box into an area, so
16:9 comes back **1344x768** instead of 1024x576; Q11 put the client's
hang-up through to the sampler and the VAE's submit batches; Q12 finished the
Z-Image deletion (**6,839 lines of Go and 895 of GLSL** out, every gate
re-run with no digit changed). The full write-up, every tolerance with its
instrument named, is in the archive.

**What to read before touching this code again** — three precision facts,
each of which has already caught a port:

- **the VAE cannot take fp16 operands anywhere** (Q9b). Its tail norm divides
  a per-pixel L2 out of a residual stream at absmax 2.6e5, so a 5e-4 relative
  perturbation becomes an absolute one: **one** narrowed convolution costs the
  decoded image max abs 0.0885 and the whole 3x3 set costs 0.178, against an
  fp32 port sitting at 7.3e-4. The rule is not "watch the range", it is "do
  not narrow". `TestConvFP16Ladder` is the instrument; re-run it before
  pointing any narrowing kernel at `qimage/vae`. Q12 deleted
  `vae_conv_wmma`/`vae_attention_wmma` outright, so the shortcut is not in
  the tree — resurrect from git history only if that ladder says otherwise.
- **the vision tower amplifies an input perturbation by ~10³**, so a
  condition image must be quantized exactly as the reference's is —
  compositing alpha over white in float rather than on 8-bit levels moves the
  prompt embedding by rel 11 (a firing control). That is why the Lanczos
  resampler is gated on exact 8-bit equality and not a tolerance.
- **on a non-square condition image the fp32 dump is the less accurate
  side** — rel 1.4e-3 from a float64 run where the Go tower sits 2.1e-4 — so
  that stage is gated against dumped float64 rows.

**If it is ever unparked**, in the archive's order:

- **The 1184²-area ceiling** — the only remaining *capability*, not a
  percent. The VAE decoder's activation arena is one storage buffer against a
  4 GiB − 4 device limit at a measured **3060 bytes a pixel for every aspect
  ratio** (`TestArenaShape`), which caps a request at 1,403,584 pixels
  whatever shape they are in, so the model's own 2048² examples do not
  decode. Two routes: tiled decode, or a multi-buffer arena
  (`vk.PipelineSpec.Counts`, with the LLM's 77 GB bank as precedent). Tiling
  is the less attractive of the two against a tail norm that is a per-pixel
  L2 over the whole feature map.
- **conv3x3's last ceiling** — after Q9b's register block it is **85.9% of
  the decode, 3.36 s at 5.0 TFLOP/s**, and its remaining limit is one shared
  read per multiply-add. A pixel block is priced at ~1.5 s and is the only
  port here that would **not** be bit-identical. Low value now: the VAE is
  4.5% of an image, the DiT is 95%.
- **The DiT's remaining percents are fusions** — that is where the image's
  time actually is, post-Q9.
- **Masked edits** (Q-o3) — still a 501, but the reason moved: 2.1 *can* do
  masked and annotated local edits; how a mask is fed is not in the diffusers
  implementation this port follows. External research, not a port.
- **Two watch items**: Q-o2, whether a turbo/distilled 2.1 checkpoint or
  step-distillation LoRA appears (the examples repo and lightx2v); Q-o4,
  whether [taehv](https://github.com/madebyollin/taehv) grows a 2.1 variant,
  which would turn the fitted linear preview from a fallback into an upgrade.

**Settled, so it does not get relitigated**: the step count — 40 stays the
default because it is the only count safe across prompt kinds. A fox
photograph and an impasto harbour are convincing at **12 steps**, at 36% of
the cost; a bicycle drivetrain diagram is coherent at 24 and has
disintegrated by 12 (floating parts, ghosted tubes, contrast collapsing
toward white). **24 is the honest fast setting** — −35%, no visible loss on
any of the three prompt kinds. `steps` is a request field, so a client that
knows its prompt takes the discount itself. Note the sweep's wall clocks
(40: 1m38–1m42, 24: 1m4, 16: 44–45 s, 12: 35–36 s) were measured **before
Q9 and Q9b**, so the seconds are stale by the 6–7% those two took off every
step while the *ratios* stand; re-run with `QI21_SWEEP=1 go test
./qimage/pipeline -run TestStepSweep` (~10 minutes of device time) before
quoting a number from it.

**Not planned**: quantisation (compute-bound at every servable size, §3.4;
int8 WMMA runs at fp16 rate, §0; and the two-machine deployment removes the
footprint argument). Sampling the encoder's posterior (breaks seed
reproducibility). A self-trained tiny decoder for previews — a training
project this repo does not want. No CFG path (`true_cfg_scale` stays a
refusal) — it would double every step.

## Embeddings (archive: [`research/embedding-vertical.md`](research/embedding-vertical.md))

**Where it stands.** E0–E8 done except E7. Same `zimage/qwen` transformer,
third caller; 500 lines of package. A ≤32-token text costs what the weights
cost (11.5 ms, 32% of the bus); cosine 0.999999+ against fp32. Served with
MRL `dimensions` and the non-OpenAI `instruct` field.

**Open: E7, batching — up to 10x on short texts.** The projections already
take M; the blocker is attention (two texts in one causal stream need a
block-diagonal mask). Measure the cheap shape first: **attention per
segment, projections batched** — attention is 1.5% of the graph and the
push block already carries per-text offsets, so no shader changes. The
segmented-mask kernel is the fallback. Quantisation stays unplanned until
a batch makes the weights the limit again.

## The server, cross-vertical (docs: [`API.md`](API.md))

- **Constrained decoding** — the one refusal that is a missing capability:
  would close `response_format`, `text.format` and named `tool_choice`.
- **More than one conversation** — the graph is one sequence's; two clients
  take turns evicting each other's prefix. A cache per conversation is a
  measurement (residency) plus P6's scheduler question.
- **The image adapter holds the device lock for the whole run** (the LLM's
  is per forward pass). Same fix shape; what a waiting speech request
  actually gains is a measurement.
- `n > 1` images are serial (capped at 4); a forced alignment for
  transcript timings if the 80 ms grid ever isn't enough.

---

## Where the old files went (2026-09-20 consolidation, and IMAGE.md again on 2026-09-21)

| was | now |
|---|---|
| `LLM.md` | [`research/llm-vertical.md`](research/llm-vertical.md) — stages, decisions D1–D21, open questions, how-to-run |
| `LLM2.md` | [`research/llm-review.md`](research/llm-review.md) — hypotheses checked, decode budget, P0–P6 as closed |
| `SPEECH.md` | [`research/speech-vertical.md`](research/speech-vertical.md) — S1–S8, T1–T10, R1, W1 write-ups |
| `TTS.md` | [`research/tts-recap.md`](research/tts-recap.md) |
| `IMAGE.md` (z-image, until 2026-09-20) | [`research/zimage-vertical.md`](research/zimage-vertical.md) |
| `IMAGE.md` (Qwen-Image-2.1, 2026-09-20–21) | [`research/qimage-vertical.md`](research/qimage-vertical.md) — frozen 2026-09-21 with the vertical; **Q-stages, decisions 1–7 and Q-o numbers resolve there** |
| `PIPELINE.md` | [`research/zimage-pipeline.md`](research/zimage-pipeline.md) — inventory, validation rules, budget |
| `EMBEDDING.md` | [`research/embedding-vertical.md`](research/embedding-vertical.md) |
| `IDEAS.md` | [`research/ideas.md`](research/ideas.md) — the `§N.M` backlog and the measured roofline |
| `TODO.md` (session log) | distilled into `research/` as each stage closed; the phase-1 tail is [`research/phase1-backlog.md`](research/phase1-backlog.md); the full log is in git history |

Per-experiment findings remain one file each in [`research/`](research/README.md),
indexed there. `§N.M` numbers are permanent addresses — never renumber.
