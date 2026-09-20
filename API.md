# API — the models behind one HTTP server

`cmd/serve` is the long-lived process: an OpenAI-shaped API over the models in
this repository, with one flag per vertical deciding what is resident.

    go run ./cmd/serve -llm -embed -tts -stt

`GOALS.md`'s last line — "we will ultimately serve up a HTTP API serving these
features, in Go" — is what this is, and all five verticals are now behind it.

The same process has a **second door** for the two speech verticals, because
the thing on the other side of a house's microphones does not speak OpenAI:

    go run ./cmd/serve -tts -stt -wyoming 0.0.0.0:10300

See [Home Assistant speaks Wyoming](#home-assistant-speaks-wyoming-not-openai).

## What answers today

| Endpoint | State |
|---|---|
| `GET /v1/models` | **done** — lists what the process actually loaded, with the speech model's voices |
| `POST /v1/chat/completions` | **done** — qwen3.8-flash-next, `-llm`, buffered and SSE |
| `POST /v1/responses` | **done** — the same generation, OpenAI's newer envelope |
| `POST /v1/messages` | **done** — the same generation, Anthropic's envelope |
| `POST /v1/audio/speech` | **done** — kokoro, `-tts` |
| `POST /v1/audio/transcriptions` | **done** — parakeet, `-stt`, with word and segment timings |
| `POST /v1/embeddings` | **done** — Qwen3-Embedding-0.6B, `-embed`, float or base64, MRL widths |
| `POST /v1/images/generations` | **done** — qwen-image-2.1, `-image`, any size the arenas hold, RGBA with `background: "transparent"`, streaming previews with no flag |
| `POST /v1/images/edits` | **done** — qwen-image-2.1, `-edits N`, up to N reference images, conditional generation rather than SDEdit (no `strength`) |

Every endpoint is *routed*, including the ones that are not implemented: a
client gets a 501 that says what is missing and, where a flag would have fixed
it, which flag. A 404 would mean a misspelled path and nothing else.

## The three packages

    cmd/serve   flags, load order, graceful shutdown
    backend/    the adapters: weights, device residency, one lock each
    api/        routing, request shapes, encoding; imports no model

The direction of the dependency is `cmd/serve -> backend -> api`, never the
other way. `api` *declares* the backend interfaces (`api.SpeechBackend`,
`api.TranscriptionBackend`, `api.CompletionBackend`, `api.EmbeddingBackend`,
`api.ImageBackend`) and implements none of them, so it pulls in no checkpoint
loader and no Vulkan, and its tests run against fakes in milliseconds —
`go test ./api/` needs no `models/` directory and no GPU, including for the
three chat envelopes and their streams and for every size the image endpoint
resolves.

What crosses the boundary is deliberately model-shaped and not
container-shaped. A speech backend returns `*audio.Clip` — samples and a rate
— and the HTTP layer encodes the WAV or the raw PCM, because which one was
asked for is the request's business and not the model's. A transcription
backend is handed an already-decoded clip for the same reason in reverse.

**The completion backend's signature is a streaming one**, and the buffered
response is written in terms of it rather than the other way round. A token
on this part costs ~44 ms of the wall clock at 22.7 tok/s, so a 500-token
answer is 22 seconds: an interface that returned the whole thing would make
the streamed endpoint impossible to build on top of it, where the buffered one
is a string builder over the streaming one. That is also what keeps the two
answers to the same prompt identical — a server whose streaming path is a
separate implementation is a server whose two answers eventually differ, over
some whitespace one of them strips.

## The log does not keep the conversations

`-log-bodies` is off by default, and that default is the point: this server's
request and response bodies are its users' prompts and answers, and a process
that stays up would otherwise write every one of them to disk as a side effect
of having an access log. With it off the log still carries the method, the
path, the status, the duration and the sizes — which is what an operator reads
anyway. An event stream is never kept whatever the flag says: the first
kilobyte of one is a fragment of a frame.

## Requests are serialised at the queue

A `vk.Device` holds a single queue and nothing in `vk` is externally
synchronised, so two requests recording and submitting at once is undefined
behaviour rather than contention. `backend.Device` is the device plus the lock
that fixes that: every run and every staging pass goes through `Do`, and the
server is concurrent at the HTTP layer and serial at the GPU.

That is also the only arrangement that makes sense on this part. These models
are each sized to saturate the iGPU; two of them interleaved would not go
faster, they would share a 236 GB/s bus.

## Residency, and the one place it is not resident yet

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

**The device lock is taken per forward pass and not per request.** A
completion is seconds long and the speech models are milliseconds, so holding
the queue for a whole generation would make a transcription wait behind it for
nothing: between two decode steps there is no work in flight. What is held for
the length of a request is the *graph*, whose cache, PLE ring and DeltaNet
state are one sequence's — two conversations interleaved in them would not be
slower, they would be one conversation.

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

**The image pipeline is resident, and its resolution is not part of what is
resident.** `-image` stages 13.2 GB of text encoder, 13.3 GB of transformer
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
| DiT prefill (step 0, which also fills the prefix KV cache) | 2.151 s / 2.163 |
| 39 cached steps, mean | 2.117 s / 2.140 |
| VAE decode | 7.48 s / 7.75 |
| **request** | **1m32.3 s** / 1m33.5 |

**An image is the transformer and almost nothing else**: 91% of that wall
clock is the 40 denoising steps, 8.1% is the VAE and 0.1% is the text
encoder. The two levers are therefore the step count — genuinely per request
here, unlike under a turbo distillation — and the unported percents
`IMAGE.md`'s Q9 prices.

An *edit* at the same size, with one reference image, is **1m58.6 s /
1m59.2**: the reference's own encoding 7.8 s (a 27-layer vision tower and a
VAE encode), text encoding 0.71 s, the prefill 5.19 s against a generation's
2.15, 40 steps at 2.47 s against 2.12, and the same 7.5 s decode. The extra
is the prefix — thousands of rows of reference latents that every step
attends over.

Over HTTP, on a `-image -image-size 512x512 -image-steps 24` process: a
512x512 image at 24 steps comes back in **16.8 s then 14.5 s**, base64 body
included. **The step count is worth sending.** IMAGE.md's sweep (same seed,
three prompt kinds, 1024x1024) found the answer is prompt-dependent rather
than a single number:

| steps | wall | photographic | painterly | structured/technical |
|---|---|---|---|---|
| 40 (default) | 1m38 | good | good | good |
| 24 | 1m4 | good | good | good |
| 16 | 45 s | good | good | washed out, structure fragmenting |
| 12 | 36 s | good | good | broken |

(The wall clocks are the sweep's own, taken before Q9's 6% — the ratios are
what it was measuring, and 40 steps is 1m32 today.)

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

## Flags

    -addr        127.0.0.1:8080   loopback by default
    -token       …                bearer token; empty serves without authorization
    -gpu         true             use the device where an adapter has a resident path
    -max-upload  128              largest body carrying a file, in MB
    -log-bodies  false            print request and response bodies in the access log

    -llm             false        load qwen3.8-flash-next
    -llm-model       models/Qwen3.8-Flash-Next-GGUF/…-00001-of-00004.gguf
    -llm-ctx         4096         cache cells: the longest conversation, prompt plus completion
    -llm-batch       512          tokens the prefill arenas hold
    -llm-max-tokens  1024         what a request that names no max_tokens gets
    -llm-layers      0            stage the first N layers only; a fast start, not an answer

    -tts         false            load Kokoro-82M
    -tts-model   models/Kokoro-82M
    -tts-gpu     false            stage per utterance; see above
    -lexicon     models/misaki    empty takes phonemes only
    -espeak      true             espeak-ng for words outside the lexicon
    -british     false            en-GB lexicon and fallback
    -voice       af_heart         what a request that names none gets
    -noise       0                excitation noise seed; 0 leaves it off

    -stt         false            load parakeet-tdt-0.6b-v3
    -stt-model   models/parakeet-tdt-0.6b-v3
    -max-audio   60               longest clip the arenas are sized for, in seconds

    -wyoming             ""       also serve the Wyoming protocol on this address; empty is off
    -wyoming-name        strix-halo   program name `info` advertises, and what Home Assistant
                                      names its entities
    -wyoming-all-voices  false    advertise all 54 packs, not just the 28 -lexicon can pronounce
    -wyoming-rate        0        resample synthesised speech before sending it; 0 is the model's
    -wyoming-max-conns   32       concurrent connections

    -embed         false          load Qwen3-Embedding-0.6B
    -embed-model   models/Qwen3-Embedding-0.6B
    -embed-tokens  512            longest input the arenas hold; longer inputs are truncated

    -image             false               load Qwen-Image-2.1
    -image-model       models/Qwen-Image-2.1
    -image-size        1024x1024           the largest image the arenas hold, and the default; <= 1184x1184
    -image-steps       40                  what a request that names no steps gets
    -image-max-prompt  512                 longest prompt the image text encoder is built for
    -edits                   0              reference images an edit may carry; 0 refuses /v1/images/edits
    -image-condition-size    1024           the area every reference image is resized to, and an edit's
                                            default output size (diffusers' output_resolution)

There is no `-preview`/`-previews` flag: this model's preview decoder is a
fitted 64x4 matrix compiled in, so `stream: true` always answers.

**-llm and -image do not fit together.** ~84 GB and ~32 GB against 128 GB of
unified memory the rest of the machine is also in: they are separate
processes on this part, or separate runs.

## A second turn is a continuation

A chat client re-sends the whole conversation every turn, so the naive server
prefills ten turns of history to answer the tenth. This one does not: the
adapter keeps the token sequence the graph is holding, and when a request
continues it — the same tokens, plus more — the new tokens are `Extend`ed onto
the state that is already there. That is the same call a generated token
makes, and L7b's gate is that a sequence in chunks is the sequence whole.

Measured on the second turn of a two-turn conversation: **90 prompt tokens, 71
of them reused, 216 ms against 421**. The log says which, per request.

The reuse is all-or-nothing, and the reason is the gated DeltaNet. A
divergence part way through would mean rewinding to the fork — and the
attention cache is masked by position and both convolution rings are addressed
by it, so those would survive, but a DeltaNet layer's recurrent state is a
running product with no inverse. So the prefix has to be the whole of what is
held, or the sequence starts again. A miss costs one comparison; a hit costs
nothing and saves the history.

It also means **the graph is one conversation's**. Two clients are served
correctly — the second waits, and re-prefills — but they take turns evicting
each other's prefix, which is the first thing a cache per conversation would
fix.

## One prompt, three envelopes, and a template that is not ours

The three chat endpoints are one generation. `/v1/chat/completions` is
OpenAI's, `/v1/responses` is OpenAI's newer one, `/v1/messages` is
Anthropic's, and each is a translation into the same `api.CompletionRequest`
and back out of the same token stream. Nothing but the envelope differs, which
is why the tests for two of them are about the translation and not about the
model.

They do disagree about **reasoning**, and usefully. A chat completion carries
it in a `reasoning_content` field beside the content — not OpenAI's field, but
the one every client that talks to a reasoning model already reads. A response
carries it as an *item of its own*, before the message it led to. A message
carries it as a `thinking` block. All three come from the same place: this
checkpoint's generation prompt opens a `<think>` block, so the model's first
token is inside it, and `llm.ChatDecoder` splits the stream at `</think>` —
holding back any tail that could still be the start of a marker, so a `<` at
the end of a token is never forwarded as an answer and then taken back.

**The prompt itself is a transcription and not a design.** A chat request is a
list of messages and the model reads one string, and what turns one into the
other is `tokenizer.chat_template` in the GGUF's own metadata: 180 lines of
Jinja that merge the leading system turns, add a reasoning-effort sentence,
write the tool block, keep each assistant turn's `<think>`, and open the
generation prompt. A server that improvises that produces a prompt the model
has never been shown — which does not fail, it answers slightly wrong,
everywhere, with nothing to point at. So `llm.RenderChat` reproduces it, and
the acceptance criterion is Jinja's own output: `reference/dump_chat_template.py`
renders the template out of the checkpoint with the environment transformers
builds, and `TestRenderChat` diffs 23 cases character for character. The cases
that made it worth doing are the small ones — Python's `json.dumps` writes a
space after every separator where Go's encoder writes none, and escapes
neither `<` nor `&` where Go escapes both, so the tool block was four
differences away from being someone else's prompt.

**A stop sequence is another marker.** The same holdback that keeps
`</think>` from being forwarded half-written keeps the first character of a
stop sequence out of the stream, so a streamed answer and a buffered one are
the same string — the sequence itself appears in neither. They are watched in
the *answer* and not in the reasoning, because a `stop` of `"\n\n"` would
otherwise end every generation inside its own working. OpenAI's envelopes fold
this into `finish_reason: "stop"`; Anthropic's reports `stop_sequence` and
says which one matched, and so does this.

**Tool calls are read back through the schema.** The template writes a string
argument raw between its tags and everything else as JSON, so rebuilding
`arguments` needs the tool's own parameter types: `123` is the number for an
integer parameter and the text `"123"` for a string one. They are also the one
thing that is not streamed — a call is not a call until it has closed, and a
client sent half of one would have to be told to take it back.

## Transcripts have times in them

Every emission the transducer makes records the encoder frame it was made at,
and a TDT model also predicts how many frames to skip afterwards — so a word
has an end as well as a beginning without a forced alignment. `verbose_json`
reports both granularities OpenAI defines, `srt` and `vtt` are the same
segments in their two containers, and `json` stays the text alone.

Two things about them are worth stating rather than discovering. **The timings
are the model's own**: the duration head is approximate in a way a Viterbi
alignment against the audio is not, and 80 ms is the finest it can be. And
**Whisper's fields are absent rather than invented** — `avg_logprob`,
`no_speech_prob`, `compression_ratio`, `seek` and `temperature` are properties
of Whisper's decoder, and a number in them here would be a number a client
could filter on. `language` is likewise the request's or absent: this
checkpoint is multilingual and does not report which language it heard.

Words are grouped by the tokenizer's own rule (SentencePiece's U+2581 marks a
word start) and segments are sentences, which this model can do because it
punctuates — a sentence end is a token rather than a guess about silence.

## Extensions, and what is refused rather than faked

`POST /v1/embeddings` takes an **`"instruct"`** field that is not OpenAI's.
Qwen3-Embedding is instruction aware: a *query* is meant to be prefixed with a
one-sentence description of the retrieval task and a *document* is not, and
the card measures 1-5% on downstream tasks for it. There is nowhere in
OpenAI's envelope to say so, so the field carries the task description —
`"instruct": "default"` asks for the card's own — and leaving it off embeds
the text as given, which is the document behaviour and the safe half. The
prefix is applied to the *text*, so it is inside the `usage` the response
reports. `"dimensions"` is honoured the way the card asks (MRL): the vector is
truncated and renormalised, so a 32-wide vector is still a unit vector.

`POST /v1/audio/speech` takes a **`"phonemes"`** field that is not OpenAI's: IPA
to speak directly, winning over `input`. The checkpoint's vocabulary is 178 IPA
symbols and the front end that reaches them is a separate model's worth of
rules (research/speech-vertical.md T5), so a caller that has already done that says so instead of
having it guessed. It is also how the endpoint is driven without a lexicon on
disk. `GET /v1/models` carries a **`voices`** array on the speech model, so a
client does not need a second call to populate a menu.

Its **`"voice"`** also takes a *mixture*: `"af_bella,af_sky"` is the equal mean
of two style packs, which is not this server's invention — hexgrad's own
`KPipeline.load_voice` gives that spelling exactly that meaning, so the same
request is the same voice here as anywhere else kokoro runs (research/speech-vertical.md T8).
`"af_bella:3,af_sky:1"` weights the mix, which is ours, and the weights are
normalised by their sum so they need not add up. Either every component
carries a weight or none does: `"af_bella:0.7,af_sky"` is a 400, because it
reads as either 0.3 for the rest or one share before normalising and a mix
that guessed would still sound like a voice. An unknown name is a 400 naming
*which component* was unknown, since that is the one thing a three-part string
does not tell you. The `voices` array stays the list of single names: a blend
is every comma-joined subset of them and not a list anything could enumerate.

`POST /v1/images/generations` takes three fields that are not OpenAI's, and
each of them exists because the parameter is real here and there is nowhere
else to put it. **`"seed"`** names the initial latent's seed, and a response
reports the seed whether or not the request named one — a drawn seed that the
client is not told is an image that cannot be asked for twice. **`"steps"`**
is the denoising schedule's length, which is a host-side array and therefore
genuinely per request. And **`"aspect_ratio"`** — `"16:9"` and the like — is
fitted to the largest image *this* server holds, so a client wanting a
landscape picture does not have to know the ceiling to ask for one; `size`
wins when both are set, because that is the field every other server reads.

**The ceiling is an area and the shape is free**, which is worth knowing
before reading `max_size`. What a server stages is a number of pixels — the
image pipeline's arenas are sized by the pixel count alone, the VAE's at a
measured 3060 bytes a pixel whatever the aspect ratio — so a request is
accepted when `width x height` fits `max_pixels`, and both 1344x768 and
2048x512 are served by a box started at `-image-size 1024x1024`. That is why
`aspect_ratio: "16:9"` answers 1344x768 there rather than 1024x576: a
landscape picture gets 16:9's share of the arenas instead of what is left of
a square after it has been flattened. A size past the budget is a 400 naming
the megapixels.

`GET /v1/models` carries an **`image`** object on the image model, alongside
the speech model's voices and for the same reason: `default_size`, `max_size`,
`max_pixels`, `size_multiple`, `default_steps`, `previews`,
`max_partial_images`, `edits` and `max_reference_images`, which are decided by
what this process staged. `max_pixels` is the rule and `max_size` is the
largest *square* obeying it, both reported because a client asking for a
square needs only the second and one asking for 16:9 cannot derive it from
the second alone. The last four are how a client finds out that `stream: true`
and an edit will be answered, and with how many pictures, rather than
discovering it from a 501.

**`background: "transparent"` is OpenAI's field and it is answerable here**,
because Qwen-Image-2.1's VAE is natively RGBA — four channels out of the
decoder on every image. It is not a mode, though, and that is the part worth
knowing: what actually makes a background transparent is *asking for it in
the prompt*, so the server prepends and appends the checkpoint's own
recommended phrasing ("This is an RGBA image with transparency. … The image
has alpha channel and the background is transparent.") and keeps the alpha
plane. `"opaque"` and `"auto"` leave the prompt alone and composite the
result over white, which is what an ordinary picture wants — the alpha plane
exists either way and on an ordinary prompt it is whatever the model painted
there. `background: "transparent"` with `output_format: "jpeg"` is a 400
naming both fields rather than a silently flattened image.

**Streaming an image** is OpenAI's `stream` and `partial_images`. The frames
are `image_generation.partial_image` — `b64_json`, `size`, `output_format`
and `partial_image_index`, plus a `step`/`steps` pair this server adds so a
client watching one arrive knows how much is left — followed by one
`image_generation.completed` carrying the finished image and the `seed` and
`steps` that produced it. There is **no `[DONE]` sentinel**: these events are
named, so the terminal one is already unambiguous, and the sentinel exists on
the chat endpoint only because its frames are not.

**It needs no flag, and that is the change from Z-Image-Turbo rather than an
omission.** There, previewing meant loading `madebyollin/taef1` beside the
full VAE — 1.0 GB of activation arena, so streaming was a residency decision
and `-preview` was how you made it. Qwen-Image-2.1's 64-channel VAE has no
distilled decoder in existence, so the preview here is a **fitted linear
64→RGBA matrix**: 260 float32s compiled into the binary, least-squares
fitted against the real decoder's own output (IMAGE.md Q7, R² 0.97 on the
fit and 0.83–0.90 on held-out prompts).
There is nothing to load and nothing to turn on.

Measured on one `-image -image-size 512x512 -image-steps 16` process, two
runs each:

| request | wall clock | first picture |
|---|---|---|
| no stream | 10.25 s / 10.23 | 10.3 s |
| `stream: true` | 10.26 s / 10.28 | 10.3 s |
| `stream: true, partial_images: 3` | **10.27 s** / 10.28 | **2.27 s** |

So the framing is free and three in-progress frames cost **0.3%** — against
4.4% through taef1 — and what they buy is a picture at 2.3 seconds instead
of at ten. The frames land at 2.27, 4.34 and 6.43 s, from steps 3, 7 and 11
of the sixteen.

A preview decode itself is **159 µs** against a 587 ms step; everything else
in that 0.3% is encoding the PNG and writing it. Partial frames go out at
`png.BestSpeed` — 15% more bytes for a fifth of the latency — while the
finished image keeps the careful encoder. A transient frame and a deliverable
are not the same object. **A frame is 1/16 scale** (32x32 for a 512x512
image, 64x64 for a 1024x1024 one) and is upscaled by whatever displays it: a
latent2rgb map carries composition, colour and layout, and cannot carry
texture the latent does not hold per-pixel.

**What the frame decodes is the denoised estimate, not the current latent**,
and the distinction is load-bearing. The schedule is flow matching, so the
sample at step k is an interpolation x_t = (1−σ)·x0 + σ·ε — ten steps into
forty it is still three quarters noise, and mapping *that* through any
decoder gives a picture of noise. The estimate x0 = x_t − σ·v is what looks
like the image, and the loop forms it for nothing because it has the
velocity in hand.

**Which steps a partial comes from is the backend's decision**, not the
handler's: the step count may not have been named in the request, and how
finished a picture looks at step k is the scheduler's business. `backend`
spreads them evenly over the steps that will actually run and never takes the
last, whose denoised estimate *is* the final latent — a frame there would be
the finished image sent twice, once through each decoder.

### Editing a picture

`POST /v1/images/edits` answers when the server is started with **`-edits N`**,
where N is how many reference images an edit may carry. The mechanism changed
with the model, and so did what a client sends. Under Z-Image-Turbo this
endpoint did SDEdit: the picture was encoded to a latent, the latent mixed
with noise at an intermediate point on the schedule, and only the tail run —
`strength` being how far back up the schedule that point was.

**Qwen-Image-2.1 does not edit that way.** An edit is a *conditional
generation*: every reference image is resized to the condition area and
encoded twice — by a 27-layer vision tower into context the prompt's tokens
absorb, and by the VAE into latents that become rows of the transformer's own
sequence — and the target is then denoised from pure noise through the whole
schedule, attending over that prefix at every step. Four consequences a
client sees:

- **`strength` is gone.** Nothing is partially renoised, so there is no knob
  to have. A request that still sends one is a 400 saying so, rather than
  having it quietly ignored.
- **An edit costs slightly *more* than a generation**, not less: the prefix
  is thousands of tokens rather than tens. Measured at 1024² with one
  reference, an edit is **1m59s** against a generation's 1m32s — the
  reference's own encoding is 7.8 s of it (a 27-layer vision tower and a VAE
  encode), and the rest is the prefix riding along in every denoising step.
- **`image[]` is a real list.** Up to `max_reference_images` pictures are
  accepted, in the order they were sent, and the model's own limit is ten.
  How many *this* server takes is residency — every reference adds its latent
  rows to the prefix and its share of the prefix KV cache — so it is fixed by
  `-edits N` at startup and reported in `GET /v1/models`.
- **An edit with no `size` follows the last reference image's aspect ratio**
  at the condition area, which is what diffusers does. It is not the server's
  default size, and it is not the picture's own pixel dimensions either.

`-edits N` is residency and not a feature flag: it stages the vision tower
and the VAE encoder (~1.4 GB together) and sizes the transformer's prefix KV
cache, which at one 1024² reference is ~2.1 GB. Both encodings of the request
are unchanged — `multipart/form-data` for OpenAI's clients, and a JSON body
whose `image` is base64 for curl.

What is refused rather than faked:

- **`stream: true` on speech.** Kokoro decides the whole utterance's durations
  before a single sample exists, so there is nothing to send early. A whole
  body dressed as a stream would be a lie about latency.
- **`stream: true` on transcription.** The decode loop emits token by token
  and could stream, but the 40 µs submit-and-fence per emission is 38% of it
  (research/speech-vertical.md), so what that endpoint wants is the persistent kernel, not a
  goroutine.
- **Containers other than WAV, and audio formats other than wav/pcm.**
  `audio/` is 16-bit PCM in and out on purpose. An mp3 request gets a 400
  naming what the server does encode. *Transcripts* are a different question
  and are no longer refused: `json`, `verbose_json`, `text`, `srt` and `vtt`
  all answer, because the timings they need exist.
- **`/v1/images/edits` without `-edits N`.** A 501 naming the flag, because
  editing is residency: the vision tower and the VAE encoder are staged or
  they are not.
- **`strength` on an edit.** A 400 explaining that the model conditions on
  the whole reference rather than seeding a truncated schedule, so there is
  no fraction of the schedule to name. Ignoring it would be a picture that
  did something other than what was asked.
- **A `mask` on an edit.** Still refused, but the reason has moved: 2.1 can
  do masked and annotated local edits, and how a separate mask is fed is not
  in the diffusers implementation this port follows (IMAGE.md's open question
  Q-o3). It is a 501 that says what it would be rather than a picture that
  ignored it.
- **`background: "transparent"` with a JPEG container.** A 400 naming both
  fields. JPEG has no alpha channel, and flattening it silently would hand
  back an opaque picture that the prompt rewrite had also made worse.

And on the chat endpoints, the same principle with a longer list: **`n > 1`**
(n completions are n runs of a model sized to saturate the device), **`min_p`
and `repeat_penalty`** (the sampler is temperature, top_k and top_p),
**`response_format` and `text.format`** other than text, a **`tool_choice`**
that names a function, **`thinking.budget_tokens`**, and any **image content
block**. Every one of them is a 400 that says what the server
does instead. The reason is the same each time: a knob accepted and ignored is
worse than a refused one, because the client never learns that turning it did
nothing.

## Home Assistant speaks Wyoming, not OpenAI

`-wyoming` opens a second listener on the same process: the two speech
backends over [Wyoming](https://github.com/rhasspy/wyoming), which is the
protocol Home Assistant's voice pipeline talks to a microphone and a speaker
with. It is a door and not a copy — one staging of the weights, one GPU
queue, one `models/` — so a request that arrives on 10300 and the same
request on 8080 are the same run of the same model. They are checked against
each other: the same text and voice through both produces byte-identical
samples.

    In Home Assistant: Settings -> Devices & services -> Add integration
    -> Wyoming Protocol -> the host this runs on, port 10300

One port carries both services. Wyoming's convention is 10300 for
speech-to-text and 10200 for text-to-speech, but a client discovers what is
behind a port by asking rather than by which port it is, so this answers
`describe` with both and Home Assistant creates an STT entity and a TTS
entity from the one entry.

**There is no authentication, and there cannot be.** The protocol has nowhere
to put a credential: it is a framed JSON conversation over a bare TCP socket,
and every implementation of it assumes a trusted LAN. `-token` does not reach
this listener, and the startup line says so. That is why the flag takes an
address rather than a boolean — which interface it binds is the only control
there is, and it should be a decision somebody made.

    13:03:47 wyoming: 1 asr, 1 tts (28 voices) as "strix-halo" on 0.0.0.0:10300;
             the protocol carries no credential, so -token does not apply to this port

**28 voices, not 54.** Kokoro's packs cover nine languages and misaki's
English grapheme-to-phoneme is the one that is ported (research/speech-vertical.md T5), so
`jf_alpha` over this protocol would be a Japanese voice reading English
phonemes. Over HTTP that is the caller's business, because
`/v1/audio/speech` also takes IPA directly and a caller with its own lexicon
is entitled to every pack; Wyoming has no phoneme field, so the default is to
advertise only what can be pronounced. `-wyoming-all-voices` offers the rest.
Each is described the way a menu wants it — `af_heart` is "Heart (American
English, female)" — and the name Home Assistant sends back is the pack's own,
so a blend still reaches `backend.TTS` by being spelled into it.

**What it does not do.** Neither streaming direction is implemented, and both
are measurements rather than gaps: `supports_synthesize_streaming` would buy
the time between the first sentence of an answer and the last, and this part
synthesises nineteen seconds of speech in 162 ms (research/speech-vertical.md T10), so the
whole utterance is ready before a satellite could have played the first
sentence of a streamed one — and kokoro decides an utterance's durations all
at once, so the pieces would have to agree about prosody they cannot see.
`supports_transcript_streaming` is the same shape at 257x real time. Wake
word detection, intent handling and satellite control are other people's
programs; `info` reports empty lists for them rather than claiming them.

**The audio is converted where the protocol says it arrives, not where the
model wants it.** A `transcribe` stream declares its own rate, width and
channels, and this side reads 8-, 16- and 32-bit PCM at any rate and any
channel count and hands parakeet the 16 kHz mono it reads — through
`audio.Resample`, which is a filter and not an interpolation, because an ASR
front end with 80 mel filters up to 8 kHz treats aliased energy as a feature.
The HTTP endpoint still refuses a clip at the wrong rate, and the difference
is who is calling: there the caller chose a file and can convert it, here the
caller is a microphone. What is *not* converted is a stream that changes
format halfway through — concatenating two rates gives a clip whose timeline
is wrong in a way the transcript would not show.

    wyoming[14]: resampled 3.25 s from 24000 Hz to 16000
    wyoming[14]: transcribe: 3.25 s of audio at 16000 Hz -> 44 characters,
                 in 23ms (140.4x real time)

**A failure is an `error` event, not a hang-up.** Home Assistant reports a
closed socket as "connection lost" and reports an error event with its text,
so the difference is whether a misspelt voice shows up in the log as
`no voice "bm_geroge"; this checkpoint has 54` or as nothing at all. The
connection survives it: a client that asked for the wrong thing can ask
again.

## What is left

- **Constrained decoding**, which is the one refusal above that is a missing
  capability rather than a missing shape. It would close `response_format`,
  `text.format` and a `tool_choice` that names a function in one go.
- **More than one conversation at a time.** The graph is one sequence's, so a
  second request waits — and evicts the first's reusable prefix. What would
  change that is a cache per conversation and the residency to hold them,
  which is a measurement rather than an argument.
- **MTP speculative decoding is built, lossless, and parked at 0.95x**
  (research/p5c-speculative-loop.md) — it costs nothing while off. What
  would make it pay is the draft head's acceptance on a real workload,
  which is a measurement (`cmd/llm -mtp`), not engineering; see `TODO.md`.
- **Image editing costs more than it did**, and that is the model rather
  than the port: an edit in 2.1 is a conditional generation over a 27-layer
  vision tower (IMAGE.md Q8), not an SDEdit, so there is no truncated
  schedule to make it cheap. 1m59s at 1024² against a generation's 1m32s, and
  no `strength` to trade quality for time with. The endpoint's shape is
  unchanged underneath; what changed is that `image[]` is a real list.
- **The image adapter holds the device lock for the whole run**, where the
  language model's takes it per forward pass. The same fix applies — between
  two denoising steps there is no work in flight — but it would mean the
  pipeline calling back out around every submit, and a step at 1024x1024 is
  1.7 s, so what a speech request waiting behind it would actually gain is a
  measurement rather than an argument.
- **`n > 1` images are serial**, and capped at four, because the device lock
  is. Four 1024x1024 images is a minute in one request.
- **A forced alignment** for the transcript timings, if the 80 ms grid and the
  duration head's approximation ever turn out not to be enough.
