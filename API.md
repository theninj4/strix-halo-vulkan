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
| `POST /v1/images/generations` | **done** — z-image-turbo, `-image`, any size the arenas hold, streaming previews with `-preview` |
| `POST /v1/images/edits` | **done** — SDEdit over the VAE's encoder, `-image -edits`, multipart or JSON |

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
resident.** `-image` stages 7.2 GB of text encoder, 12.5 GB of transformer and
0.3 GB of VAE in **21.2 s**, sizes 4.9 GB of activation arenas for
`-image-size`, and then a request moves no weight. That last clause is what
this endpoint needed and did not have: `zimage/pipeline` used to fix its width
and height at construction, so a size was a residency question.

It is not one, and none of the three graphs had to change to stop it being
one. The transformer states its run length per upload (`GPUStack.Upload`,
which the caption phase has always used to run 32 rows in a 4128-row arena),
the VAE re-records its graph from the latent it is handed and checks the arena
rather than assuming it, and the rotary table is rebuilt per run anyway. So
`-image-size` became a *ceiling* and a default, the way `-max-audio` is for
parakeet, and every smaller image runs in the same arenas. Measured on one
1024x1024 process, two runs each:

| size | image tokens | request |
|---|---|---|
| 1024x1024 | 4096 | **14.55 s** / 14.62 |
| 1024x576 | 2304 | 7.95 s |
| 512x512 | 1024 | **3.65 s** / 3.64 |
| 256x256 | 256 | 1.33 s |

`-previews` adds 1.0 GB of arena to that and no measurable staging time —
24.9 GB becomes **25.9** — and the two processes render the same non-streamed
1024x1024 image in 14.64/14.72 s and 14.72/14.80. A preview decoder that is
not asked for costs residency and nothing else. `-edits` is the same bargain
one size up: 0.21 GB of encoder weights and 1.8 GB of arena at a 1024x1024
ceiling, and the encoder runs only on a request that carries a picture.

**What bounds a request is each side, and not the area** — which is worth
stating because the plausible guess is the other one. A diffusion model's cost
is its token count, so the same area in another shape should be the same work,
and for the transformer it is: 512x2048 is the same 4096 rows as the square.
The decoder disagrees. Stage 8 gave it an fp16 arena of *blocked* copies of
each convolution's input, and a block is padded on each axis separately, so
redistributing the same area over a taller grid needs more arena than it has —
measured at 33 MB against 32. A run that got that far would fail in the VAE
after the whole denoising loop, so `pipeline.geomFor` refuses it at the door
and the endpoint refuses it before that. Both sides inside the ceiling makes
every tensor in all three graphs smaller elementwise, which is the condition
that actually holds; `zimage/pipeline`'s `TestSameAreaTallerShapeIsRefused` is
the measurement.

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

    -image             false               load Z-Image-Turbo
    -image-model       models/Z-Image-Turbo
    -image-size        1024x1024           the largest image the arenas hold, and the default
    -image-steps       8                   what a request that names no steps gets
    -image-max-prompt  512                 longest prompt the image text encoder is built for
    -preview           ""                  a madebyollin/taef1 checkpoint; loads the preview decoder
    -previews          false               shorthand for -preview models/taef1
    -edits             false               load the VAE's encoder, which is what lets /v1/images/edits answer

**-llm and -image do not fit together.** ~84 GB and ~25 GB against 128 GB of
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
`GET /v1/models` carries an **`image`** object on the image model, alongside
the speech model's voices and for the same reason: `default_size`, `max_size`,
`size_multiple`, `default_steps`, `previews`, `max_partial_images`, `edits` and
`default_strength`, which are decided by what this process staged. The last
four are how a client finds out that `stream: true` and an edit will be
answered rather than discovering it from a 501.

**Streaming an image** is OpenAI's `stream` and `partial_images`, and it needs
`-preview`. The frames are `image_generation.partial_image` — `b64_json`,
`size`, `output_format` and `partial_image_index`, plus a `step`/`steps` pair
this server adds so a client watching one arrive knows how much is left —
followed by one `image_generation.completed` carrying the finished image and
the `seed` and `steps` that produced it. There is **no `[DONE]` sentinel**:
these events are named, so the terminal one is already unambiguous, and the
sentinel exists on the chat endpoint only because its frames are not.

Measured on one `-image -previews` process at 1024x1024, two runs each:

| request | wall clock | first picture |
|---|---|---|
| no stream | 14.72 s / 14.80 | 14.7 s |
| `stream: true` | 14.76 s / 14.75 | 14.8 s |
| `stream: true, partial_images: 3` | **15.40 s** / 15.41 | **3.6 s** |

So the framing is free and three in-progress frames cost **4.4%** — and what
they buy is a picture at 3.6 seconds instead of at fifteen. The frames land at
3.6, 7.1 and 10.7 s, from steps 1, 3 and 5 of the eight.

Two numbers inside that 4.4% are worth separating, because only one of them is
the model. A preview decode is **87 ms** (`madebyollin/taef1`, research/zimage-vertical.md I2);
encoding the frame is the other **60**, and it would have been **344** at Go's
default PNG compression — four times the decode. Partial frames therefore go
out at `png.BestSpeed`, 15% more bytes for a fifth of the latency, while the
finished image keeps the careful encoder. A transient frame and a deliverable
are not the same object.

**Which steps a partial comes from is the backend's decision**, not the
handler's: the step count may not have been named in the request, and how
finished a picture looks at step k is the scheduler's business. `backend`
spreads them evenly over *the steps that will actually run* and never takes
the last, whose denoised estimate *is* the final latent — a frame there would
be the finished image sent twice, once through each decoder. The "actually
run" is not pedantry: an edit at strength 0.5 runs the last four steps of
eight, and frames spread over all eight would every one of them fall before it
began.

### Editing a picture

`POST /v1/images/edits` needs `-image -edits`, and it takes **both encodings**:
`multipart/form-data`, which is what OpenAI's clients send and the only thing
their endpoint accepts, and a JSON body whose `image` is base64 (a `data:` URL
is fine), which is what curl and a test can write by hand. Both fill the same
struct.

**What it does is SDEdit** (research/zimage-vertical.md I7): the picture is encoded to a latent,
the latent is mixed with noise at an intermediate point on the schedule, and
only the tail of the schedule runs. **`strength`** is how far back up the
schedule that point is and is the whole behaviour of the endpoint — near zero
returns almost the picture that was sent, 1 discards it and is an ordinary
generation, and the default is 0.8. It is an extension because OpenAI has no
such field: their edit endpoint is an inpainting model driven by a mask, which
is a different mechanism.

Everything else is the generation endpoint's, including streaming, because
underneath an edit *is* a generation with a different starting latent. At
1024x1024 an edit is **cheaper** than a generation, and by exactly the steps it
skips:

| strength | steps run | wall |
|---|---|---|
| 1.0 | 8 of 8 | 14.66 s — and bit-identical to a generation from the same seed |
| 0.8 (default) | 6 | **11.29 s** |
| 0.5 | 4 | 7.94 s |
| 0.3 | 2 | 4.63 s |

The encode is a flat 430 ms of that, 3.6% at the default strength. A 512x512
edit over HTTP including the base64 body is **2.51 s**.

**The size an edit with no `size` gets is the input picture's own shape**,
fitted inside the ceiling and rounded to a multiple of 16 — not the server's
default, which would silently reframe what was sent — and it is never scaled
*up*, because four times the tokens cannot paint detail the input never had. A
picture whose shape does not match the size being rendered is **cover-cropped**
rather than stretched, since an edit whose point is to keep the composition
should not begin by distorting it.

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
- **`stream: true` without `-preview`.** It is a 501 naming the flag rather
  than a 400, because the request is well formed and the server was started
  without the decoder that answers it. With the flag it streams; see above.
  `/v1/images/edits` without `-edits` is the same answer about the encoder.
- **A `mask` on an edit.** A 501 that says what it would be rather than a
  picture that ignored it: a mask is blended into the latent at *every*
  denoising step, which is a mechanism and not a parameter, where `strength` is
  one number over the whole picture. More than one input image is a 400 for
  the same reason — OpenAI's field is a list because their model composites
  several, and this one does not.

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
- **Masked edits.** The encoder is ported and `/v1/images/edits` answers, so
  the mask is the only piece of that endpoint still missing: a blend into the
  latent at every denoising step, which is a mechanism rather than a
  parameter. Nothing in `GOALS.md` asks for it — the endpoint exists because
  OpenAI's surface does — and `strength` already covers editing the whole
  picture.
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
