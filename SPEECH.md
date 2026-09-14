# SPEECH — the two audio verticals

> **Current work.** `PIPELINE.md` (the z-image slice) is **parked** at 14.26 s
> an image; this is what happens next. Same rules as that file: it is
> rewritten rather than appended to, history goes to `TODO.md` and closed
> findings to `research/`.

**Targets**: `wav → text` for `parakeet-tdt-0.6b-v3` and `text → wav` for
`Kokoro-82M`, both end to end in Go on Vulkan, both validated against a Python
reference dump before anything is optimised. Two of `GOALS.md`'s five models,
and the two smallest.

**Status**: nothing built. Both checkpoints are on disk and both were
surveyed on 2026-09-14 — the inventories below are read off the files, not off
a paper.

## Why these two, and why now

The z-image slice is done in every sense that matters to a *vertical*: it
turns a prompt into a PNG in 14.26 s and everything left in it is the last
quarter of the WMMA ceiling on seven GEMMs. That is a deep problem and it will
still be there. These two are shallow ones — 0.63 B and 0.08 B against
6.2 B — and each closes a whole capability rather than a percent.

They are also a different *regime*, which is the interesting part. Every
kernel this repo has tuned was chosen at M=4096 with 12.5 GB resident. Parakeet
at fp16 is 1.25 GB and a 30 s clip is **375 encoder frames**; Kokoro is 164 MB
and a sentence is a few hundred phonemes. Neither is a residency problem and
neither sits where the DiT sits on the roofline. Stage 5c already found what
that does — the winning GEMM tile *moves with the sequence length* and
re-planning per run is free — and these two will test that finding rather than
inherit it.

## What we already have

The engine is most of both models. Taking it seriously is the point of doing
these next:

| They need | We have | Fits? |
|---|---|---|
| GEMM, N/K of 1024, 4096, 640 | `dit_gemm.comp`, 11 rungs + stage 5c's narrow-M rungs | yes, and the narrow-M rungs are aimed exactly at this T |
| Bidirectional MHA, 8 heads x **128** | `dit_attention_wmma.comp`; `wmmaHeadDim` is 128 | geometry yes, **positional scheme no** — see below |
| LayerNorm (mean subtracted) | `dit_final_norm.comp` | yes |
| conv2d 3x3 as implicit GEMM | `vae_conv_wmma.comp` (stage 8) | **stride 1 only**; parakeet subsamples at stride 2 |
| Checkpoint loading | `safetensors/`, `cmd/inspect` | parakeet yes but for one dtype; kokoro **no**, it is a pickle |
| A validated-against-the-library method | `reference/dump_*.py`, `RunTo`, negative controls | yes, and it is the reason to keep the order below |
| An HTTP surface | `api/speech.go`, `api/transcription.go` | request/response types only, no server |

## Parakeet-TDT 0.6B v3 — as the checkpoint describes it

`models/parakeet-tdt-0.6b-v3/`, 723 tensors, **0.627 B params, F32**. It ships
as **HF `transformers`**, not as NeMo: `config.json` says `ParakeetForTDT`,
there is a `model.safetensors`, and `transformers 5.17.0` in `.venv` has
`ParakeetForTDT`, `ParakeetProcessor` and `ParakeetFeatureExtractor`. **So the
oracle exists today with no conversion work** — the same position stage 5 was
in with Qwen3, and the `.nemo` and `.gguf` files beside it are not needed.

**Front end** (`processor_config.json`): 16 kHz mono, preemphasis 0.97,
`n_fft` 512, `win_length` 400, `hop_length` 160, **128 mel bins**. 30 s of
audio is 3000 frames.

**Encoder**, FastConformer, 24 layers, d=1024, 8 heads x 128, FFN 4096, SiLU:

| | |
|---|---|
| Subsampling | 3 x conv2d 3x3 **stride 2**, 1 -> 256 -> 256 -> 256 channels, then `linear [1024, 4096]`. 4096 is 256 channels x 16 surviving mel bins; the factor is **8** in both time and frequency |
| Per layer | macaron FF1 (x0.5) -> rel-pos MHA -> conv module -> FF2 -> `norm_out`. Five LayerNorms a layer, all with bias |
| Attention | `q/k/v/o_proj [1024,1024]`, **no bias**, plus `relative_k_proj [1024,1024]` and `bias_u`/`bias_v` `[8,128]` |
| Conv module | `pointwise_conv1 [2048,1024,1]` (GLU) -> `depthwise_conv [1024,1,9]` -> **BatchNorm** -> SiLU -> `pointwise_conv2 [1024,1024,1]` |
| Projection | `encoder_projector [640, 1024]` |

**Prediction network**: `embedding [8193, 640]`, a **2-layer LSTM** (640
hidden, `weight_ih/hh [2560, 640]`), `decoder_projector [640, 640]`.

**Joint**: `joint.head [8198, 640]`. 8198 is 8193 token logits (blank at
**8192**) followed by **5 duration logits** for durations `[0,1,2,3,4]` —
which `generation_config.json` confirms by suppressing 8193-8197 from the
token argmax. That is the whole of TDT: emit a token, then jump the encoder
cursor by the predicted duration instead of by one.

**Tokenizer**: `tokenizer.json`, **BPE with a Metaspace pre-tokenizer**
(SentencePiece-style), 8192 pieces plus control tokens —
`<|startoftranscript|>`, `<|pnc|>`/`<|nopnc|>`, `<|itn|>`/`<|noitn|>`,
`<|timestamp|>`/`<|notimestamp|>`. Note this is **not** byte-level BPE, so
`zimage/tokenizer` does not transfer — but ASR only ever **decodes**, and
decoding is a vocab lookup and a `▁ -> space` rule. Encoding is not needed at
all.

## Kokoro-82M — as the checkpoint describes it

`models/Kokoro-82M/`, **82 M params**, a StyleTTS2 derivative. Five modules in
one `kokoro-v1_0.pth` **PyTorch pickle** (327 MB F32), so unlike parakeet it
needs a conversion pass before `safetensors/` can see it:

| module | tensors | params | what it is |
|---|---|---|---|
| `bert` | 25 | 6.29 M | ALBERT, 12 layers x 768, 12 heads, FFN 2048 — 25 tensors because the layers **share weights** |
| `bert_encoder` | 2 | 0.39 M | one linear, 768 -> 512 |
| `text_encoder` | 24 | 5.61 M | phoneme embedding, a conv stack (kernel 5), an LSTM |
| `predictor` | 122 | 16.19 M | duration (`max_dur` 50) + F0/N prosody, AdaIN-conditioned |
| `decoder` | 375 | **53.28 M** | the iSTFTNet vocoder — two thirds of the model |

From `config.json`: `n_token` 178 with the **IPA vocabulary inline** (114
symbols), `style_dim` 128, `hidden_dim` 512, `n_mels` 80. The vocoder is
`upsample_rates [10, 6]`, `upsample_kernel_sizes [20, 12]`, resblocks
`(3,7,11) x dilations (1,3,5)`, `upsample_initial_channel` 512, and a final
**iSTFT** at `n_fft` 20 / `hop` 5. Output is **24 kHz**.

**Voices**: `voices/*.pt`, 54 of them, each a `[510, 1, 256]` fp32 tensor —
one style vector per phoneme-sequence length, 256 wide, which the model splits
128/128 between the predictor and the decoder. Confirm the indexing rule
(`len(phonemes)`) against the reference rather than assuming it.

**G2P is the real dependency and it is not a kernel.** Kokoro takes IPA
phonemes; `misaki` (plus `espeak-ng` for some languages) is what turns text
into them, and porting that to Go is a language problem, not an inference one.
**So the first vertical should take phonemes, not text** — the whole network
is then validatable against the reference on day one, and G2P becomes a
separate, separable stage that can start as a shell-out to `espeak-ng`.

## What is genuinely new

Shared by both, and **all of it belongs on the host first**. A 30 s clip is
3000 512-point FFTs; a sentence of speech is a few hundred phonemes. Stage 4's
rule applies — build the thing that runs, let the profiler choose what moves
to the GPU:

- **WAV in and out**, 16-bit PCM. Small, and needed by every test.
- **FFT both ways**: forward STFT for parakeet's mel front end, **inverse**
  STFT for kokoro's vocoder. One radix-2 implementation covers both.
- **LSTM.** Both models have one and both are small (640 and 512 hidden).
  Sequential by nature, so it is a latency problem rather than a throughput
  one, and parakeet's runs once per *emitted token* rather than per frame.

Parakeet only:

- **Relative-position attention.** `bias_u`, `bias_v` and `relative_k_proj`
  are Transformer-XL style rel-pos as Conformer uses it —
  `(q+u)·k^T + (q+v)·rel_k^T` with a shift — and that is **not** what
  `dit_attention_wmma.comp` computes. This is the one real kernel question in
  STT, and it is worth answering on the CPU reference first so that the shape
  of the change is known before a shader is written.
- **stride-2 conv2d**, over a 1-channel input. The stage-8 implicit GEMM is
  stride 1; whether stride 2 is a push constant or a different patch gather is
  the first thing to find out.
- **Depthwise 1-D conv (k=9), GLU, BatchNorm.** BatchNorm at inference is an
  affine, so **fold it into the pointwise convolution at load** and it costs
  nothing at run time. `running_mean`/`running_var` are in the checkpoint.
- **The TDT greedy loop** on the host: `joint(enc[t], dec[u])`, argmax over
  8193, emit, advance `u`, advance `t` by the duration argmax, cap at
  `max_symbols_per_step` 10.

Kokoro only:

- **A pickle-to-safetensors conversion** (`reference/convert_kokoro.py`), five
  submodules, once.
- **1-D transposed convolution** and the AdaIN-conditioned resblock stack —
  a new operator family, and two thirds of the model's parameters.
- **The length regulator**: expand phoneme states by predicted integer
  durations. Trivial arithmetic, but it makes the output length data-dependent,
  which no stage of z-image was.

## Order, and the recommendation

**Do parakeet first.** Not because it is smaller — it is 8x larger — but
because everything it needs to be *validated* already exists:

1. Its weights are safetensors and `transformers 5.17.0` can run it today, so
   the oracle is a `reference/dump_parakeet.py` away. Kokoro needs a
   conversion script before a single number can be compared.
2. Its encoder is 90% shapes the GEMM ladder already wins, at the head
   dimension the attention kernel is compiled for.
3. Its correctness bound is **an exact string**. Every stage of z-image needed
   an argument about what tolerance means; a transcript is right or it is not.
   That is the cheapest oracle this project has had.
4. It has exactly **one** open kernel question (rel-pos attention). Kokoro has
   three independent new things — conversion, G2P, a vocoder operator family —
   and its output is a waveform, which needs a perceptual argument the repo
   has no precedent for.

Then kokoro, phonemes-first, with G2P last.

## Stages

Same shape as `PIPELINE.md`'s: a CPU implementation in Go first, validated
against the library dump, then the Vulkan port debugged against it.

| # | Stage | State |
|---|---|---|
| S1 | `safetensors` reads the parakeet checkpoint; WAV reader | |
| S2 | `reference/dump_parakeet.py` — mel, 24 layer outputs, joint, decode trace | |
| S3 | Mel front end in Go, against S2's mel | |
| S4 | Encoder, CPU reference, layer by layer | |
| S5 | Prediction net + joint + TDT greedy decode; **first transcript** | |
| S6 | Encoder on Vulkan (GEMMs, LayerNorm, attention, conv) | |
| S7 | `cmd/asr`, profile, then optimise | |
| T1 | `reference/convert_kokoro.py` + `reference/dump_kokoro.py` | |
| T2 | Phoneme encoder + ALBERT + predictor, CPU, against T1 | |
| T3 | iSTFTNet decoder, CPU; **first waveform** | |
| T4 | Vulkan port; `cmd/tts` | |
| T5 | G2P — `espeak-ng` shell-out, then a port if it is worth one | |

## First concrete steps

1. **`safetensors` cannot open the parakeet checkpoint.** `go run ./cmd/inspect
   models/parakeet-tdt-0.6b-v3` fails with

       tensor "encoder.layers.16.conv.norm.num_batches_tracked" has unsupported dtype "I64"

   24 of the 723 tensors are BatchNorm's `num_batches_tracked`, scalar I64, and
   **inference never reads them**. The decision to make is whether `DType` gains
   an `I64` case (and the float accessors reject it) or whether `Open` skips
   non-float tensors. The first is smaller and keeps `cmd/inspect` honest about
   what is in the file.
2. **The reference env needs audio.** `uv pip install --python .venv/bin/python
   soundfile librosa` — there is no `pip` in `.venv`. `torch 2.14.0+cpu` and
   `transformers 5.17.0` are already there.
3. **Pick the fixture clip** and commit it: a few seconds of 16 kHz speech with
   a known transcript, small enough to live in the repo, used by every test
   from S2 onward. The samples in `models/Kokoro-82M/samples/*.wav` are 24 kHz
   Kokoro output and would make a pleasing but circular fixture — use real
   speech.

## Open questions, to settle by measurement

- **Does the encoder use limited-context attention?** NeMo's FastConformer
  usually does (`att_context_size`), and `config.json` shows only
  `max_position_embeddings: 5000` with no window. What `transformers` actually
  runs is what matters, and it changes both the kernel and the cost at 3000
  frames.
- **Does fp16 hold?** The repo's recurring lesson (stages 2, 6): survey absmax
  through the encoder on a real clip before committing. A conformer's conv
  module and a 4096-wide FFN are the places to look.
- **Where is the roofline crossover for this T?** At 375 frames the encoder's
  GEMMs sit near stage 5c's 235 flop/byte crossover, and a 5 s clip (63
  frames) is well below it. If the winning tile moves with clip length the way
  the text encoder's moved with prompt length, `qwen.PlanFor` is the precedent.
- **Is kokoro's voice pack indexed by `len(phonemes)`?** 510 rows, and the
  reference will say.
- **What does streaming mean here?** `api/transcription.go` already defines
  `transcript.text.delta`. Nothing below needs to be streaming to be correct,
  but the TDT loop is naturally incremental and the choice of chunking belongs
  in the design rather than after it.
