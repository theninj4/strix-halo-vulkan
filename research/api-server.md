# API — the models behind one HTTP server

> **Moved to `research/` on 2026-10-02** — this was `API.md` at the repo
> root, and code comments citing `API.md` resolve here. It documents the server as it answers today and is kept
> current, not frozen. The longer topics are broken out into `api-*.md` files
> beside this one, linked where each section was.

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
| `POST /v1/chat/completions` | **done** — qwen3.8-flash-next, `-llm`, buffered and SSE; images with `-llm-mmproj` |
| `POST /v1/responses` | **done** — the same generation, OpenAI's newer envelope, `input_image` included |
| `POST /v1/messages` | **done** — the same generation, Anthropic's envelope, `image` blocks included |
| `POST /v1/audio/speech` | **done** — kokoro, `-tts` |
| `POST /v1/audio/transcriptions` | **done** — parakeet, `-stt`, with word and segment timings |
| `POST /v1/embeddings` | **done** — Qwen3-Embedding-0.6B, `-embed`, float or base64, MRL widths |
| `POST /v1/images/generations` | **done** — qwen-image-2.1, `-image`, any size the arenas hold, RGBA with `background: "transparent"`, streaming previews with no flag |
| `POST /v1/images/edits` | **done** — qwen-image-2.1, `-edits N`, up to N reference images, conditional generation rather than SDEdit (no `strength`), RGBA with `background: "transparent"` |
| `POST /v1/videos` (+ `GET /v1/videos`, `GET`/`DELETE /v1/videos/{id}`, `GET /v1/videos/{id}/content`) | **done** — MiniMax-H3, `-video`: OpenAI's asynchronous video jobs, text (and optionally a first and/or last keyframe, `fl2va`) to a 24 fps mp4 with a 32 kHz stereo soundtrack; SGLang's H3 request shape too. See [Videos are jobs](#videos-are-jobs) |
| `POST /v1/music` (+ `GET /v1/music`, `GET`/`DELETE /v1/music/{id}`, `GET /v1/music/{id}/content`) | **done** — ACE-Step 1.5 XL turbo + its 5 Hz LM, `-music`: caption and lyrics (or just a description, sample mode) to a 48 kHz stereo song of 10 s to 10 min, as jobs in `/v1/videos`' shape; ACE-Step's own `release_task` fields and aliases are read. See [Music is a job](#music-is-a-job) |
| `POST /v1/systemone` | **done** — Kev-4B, `-kev`: TypeSafe's System One (typed `noul` / `choice` / `score` questions about a state, calibrated probabilities, no generation); the TypeSafe Python SDK works unchanged. See [`classification-vertical.md`](classification-vertical.md) |

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

Broken out to [`api-residency.md`](api-residency.md).

## Flags

    -addr        127.0.0.1:8080   loopback by default
    -token       …                bearer token; empty serves without authorization
    -gpu         true             use the device where an adapter has a resident path
    -max-upload  128              largest body carrying a file, in MB
    -log-bodies  false            print request and response bodies in the access log

    -swap-idle       10m          how long image, music or video stays staged with no request; 0 keeps it until another is asked for

    -video           false        load MiniMax-H3 and serve /v1/videos; ~0.3 GB at rest, ~31 GB at a request's peak
    -video-model     models/MiniMax-H3             the diffusers-layout checkpoint root
    -video-dir       ""           where finished mp4s wait to be fetched; empty is a temporary directory
    -video-ttl       24h          how long a finished job and its file are kept
    -video-queue     16           jobs that may wait behind the running one; past it is a 429
    -video-max-prompt 0           longest prompt in tokens; 0 is 4096

    -music           false        load ACE-Step 1.5 (XL turbo, the 5 Hz LM, VAE, text encoder) and serve /v1/music; ~26 GB while it holds the swap slot
    -music-models    models       the root holding acestep-v15-xl-turbo/, acestep-5Hz-lm-4B/, Ace-Step1.5/vae, Qwen3-Embedding-0.6B/
    -music-lm        true         load the 5 Hz LM; false serves only thinking=false, ~10 GB lighter
    -music-dir       ""           where finished songs wait to be fetched; empty is a temporary directory
    -music-ttl       24h          how long a finished job and its file are kept
    -music-queue     32           jobs that may wait behind the running one; past it is a 429

    -kev             false        load Kev-4B (Qwen3.5-4B-Base + LoRA + pointer head) and serve /v1/systemone
    -kev-model       models/kev-4b                 adapter, converted head (reference/convert_kev_head.py), tokenizer
    -kev-base        models/Qwen3.5-4B-Base        the base, at the revision head.json names (1001bb4d)
    -kev-tokens      8192         the longest request one pass holds: the state once plus every question
    -kev-cache       4            states kept, so a repeated text pays for its questions only; 0 is off
    -kev-cache-tokens 4096        the longest state kept (32 KB of KV a token, plus 52.7 MB a state)
    -kev-batch       8            the most requests one pass answers; a lone request waits for nothing
    -kev-fp16        false        stage the weights as fp16 instead of int8 (K7.1's control: 1.27x slower, 3.3 GB more)

    -llm             false        load qwen3.8-flash-next
    -llm-model       models/Qwen3.8-Flash-Next-GGUF/…-00001-of-00004.gguf
    -llm-ctx         4096         cache cells: the longest conversation, prompt plus completion
    -llm-batch       4096         tokens the prefill arenas hold (P16: wider costs memory only)
    -llm-max-tokens  1024         what a request that names no max_tokens gets
    -llm-layers      0            stage the first N layers only; a fast start, not an answer
    -llm-slots       1            conversations held at once, each of -llm-ctx cells
    -llm-preempt-chunk 2048       longest background prefill chunk, with more than one slot
    -llm-reserve     1            slots only interactive requests may take, with more than one
    -llm-class       background   priority of a request that names none
    -llm-checkpoints true         keep each slot's state before the last user turn (C4)
    -llm-batch-decode true        decode concurrent conversations a row each in one pass (C5)
    -llm-presets     models.ini   llama-server preset file: virtual models over the same weights
    -llm-mmproj      ""           the vision tower (…/mmproj-BF16.gguf); empty serves text and refuses images
    -llm-vision-tokens 4096       the most tokens one image becomes (32x32 px a token); capped at -llm-batch
    -llm-vision-images 8          the most images one request may carry

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

There is no `-preview`/`-previews` flag: this model's preview decoder,
TAEQI2.1, is always staged (29 MB of weights, 224 MB of arena at 1024²) from
`taeqi2_1/` beside `-image-model`, so `stream: true` always answers.

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

With `-llm-slots 1` **the graph is one conversation's**: two clients are
served correctly — the second waits, and re-prefills — but they take turns
evicting each other's prefix. With more slots each holds its own, and a
request is given the free slot already holding the longest prefix of its
prompt, else the least recently used one.

**Each slot also keeps a checkpoint** ([C4](c4-checkpoints.md),
`-llm-checkpoints`, on by default). It is the slot's state at the end of
everything before the last user turn: the system prompt and the history. A
request that shares that prefix but diverges after it restores the
checkpoint and prefills only the rest. That is the voice case, the same
system prompt with a new utterance: at 48 layers a 1.94k-token Home-Assistant
prompt goes from a **1.83 s** time to first token to **0.19 s**. It also
covers a conversation whose re-rendered history no longer matches the tokens
that were generated. A checkpoint is ~120 MB of host memory a slot, and the
first request on a new prefix pays ~0.2 s once to take it. The log line says
`(N from a checkpoint)` where a continued prefix says `(N cached)`. A
checkpoint is restored only in the slot that took it, so between two slots
the reuse is still all-or-nothing.

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
builds, and `TestRenderChat` diffs 28 cases character for character. The cases
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

### Images in a conversation

With `-llm-mmproj` the three chat endpoints read images, which this
checkpoint was trained on (LLM-VISION.md is the vertical's record). The
shapes, each landing as the same `image_url` block of `api.CompletionRequest`
in its place between the texts:

- chat: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}`
- responses: `{"type": "input_image", "image_url": "data:…"}`
- messages: `{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "…"}}`

**Image URLs are fetched where OpenAI's API fetches them** (and Anthropic's,
for Messages' `url` source): an `image_url` is "a fully qualified URL or a
base64-encoded data URL", so an `http(s)` URL is downloaded before the vision
tower is taken, and a `data:` URL is read in place. The fetch is capped at
50 MB and 60 s, follows up to 5 redirects, and needs a 2xx; any failure is a
400 naming the URL (without its query string). Other schemes are refused,
and so is a Responses `file_id`, because there is no Files API. **The server
fetches whatever the client names, the LAN included**, as an OpenAI-shaped
server does; a deployment that must not reach its own network on a client's
say-so needs that boundary in front of it. PNG, JPEG and GIF (the first
frame) decode; **WebP is refused**, because the module takes no dependencies
and the standard library has no decoder. Images are read in **user turns**:
the template itself refuses one in a system message, and one inside a tool
result is not wired. Audio, video and files are refused as before.

**What the model sees is what HF's processor gives it.** The picture is
resized with smart-resize to 32-pixel multiples (alpha dropped, not
composited) using torch's own antialiased bicubic, bit for bit. It goes
through the 27-layer vision tower on the matrix cores and is placed at the
image's `<|image_pad|>` with 3-D rotary positions. An image of *W x H* pixels
is about *W·H/1024* tokens, counted in `prompt_tokens`, with at least 64.
`-llm-vision-tokens` caps it (4096 is a 2048² picture), and an image always
runs in one prefill pass, so the cap is also held under `-llm-batch`.
Measured through this server: a 500² photo is 256 tokens and **53 ms** of
tower, a 1920×1080 one 2 040 tokens and **0.77 s**. The tower runs before the
request takes a slot, so it is never time a slot sits idle. JPEG decodes a
level or two apart from libjpeg on ~2% of samples (Go's IDCT), which this
tower turns into what invisible noise would.

**A second turn continues the first, picture and all**, and a checkpoint
reaches past an image. Reuse is keyed on the picture and not on the ids:
every image cell holds the same pad token, so two different pictures of one
size would otherwise be one prefix. The per-request `llm:` line says how many
images a prompt held and the tower's time on them.

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

Broken out to [`api-extensions.md`](api-extensions.md).

## Home Assistant speaks Wyoming, not OpenAI

Broken out to [`api-wyoming-and-systemone.md`](api-wyoming-and-systemone.md).

## System One is not an OpenAI envelope

Broken out to [`api-wyoming-and-systemone.md`](api-wyoming-and-systemone.md).

## Videos are jobs

Broken out to [`api-jobs.md`](api-jobs.md).

## Music is a job

Broken out to [`api-jobs.md`](api-jobs.md).

## What is left

- **Constrained decoding**, which is the one refusal above that is a missing
  capability rather than a missing shape. It would close `response_format`,
  `text.format` and a `tool_choice` that names a function in one go.
- **Speculative decoding is served** (`-llm-draft`, [`llm-plan.md`](llm-plan.md)
  P20f–P20h): a conversation decoding alone runs the MTP draft head, 56.8 tok/s
  greedy and 56.6 sampled against 41.1 plain. Several conversations decode as
  rows of one pass ([C5](c5-batched-decode.md)). What is open for both is in
  [`llm-plan.md`](llm-plan.md) and [`concurrency.md`](concurrency.md) § Next.
- **Image editing costs more than it did**, and that is the model rather
  than the port: an edit in 2.1 is a conditional generation over a 27-layer
  vision tower (research/qimage-vertical.md Q8), not an SDEdit, so there is no truncated
  schedule to make it cheap. 1m54s at 1024² against a generation's 1m28.8s, and
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
