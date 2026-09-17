# API — the models behind one HTTP server

`cmd/serve` is the long-lived process: an OpenAI-shaped API over the models in
this repository, with one flag per vertical deciding what is resident.

    go run ./cmd/serve -llm -tts -stt

`GOALS.md`'s last line — "we will ultimately serve up a HTTP API serving these
features, in Go" — is what this is. It is stage one of that: the routing
layer, three of the five verticals wired to it, and the shape the other two
plug into.

## What answers today

| Endpoint | State |
|---|---|
| `GET /v1/models` | **done** — lists what the process actually loaded, with the speech model's voices |
| `POST /v1/chat/completions` | **done** — qwen3.8-flash-next, `-llm`, buffered and SSE |
| `POST /v1/responses` | **done** — the same generation, OpenAI's newer envelope |
| `POST /v1/messages` | **done** — the same generation, Anthropic's envelope |
| `POST /v1/audio/speech` | **done** — kokoro, `-tts` |
| `POST /v1/audio/transcriptions` | **done** — parakeet, `-stt`, with word and segment timings |
| `POST /v1/embeddings` | 501 — `models/Qwen3-Embedding-0.6B` has not been started |
| `POST /v1/images/generations`, `/edits` | 501 — `zimage/pipeline` runs, it is not wired here |

Every endpoint is *routed*, including the ones that are not implemented: a
client gets a 501 that says what is missing and, where a flag would have fixed
it, which flag. A 404 would mean a misspelled path and nothing else.

## The three packages

    cmd/serve   flags, load order, graceful shutdown
    backend/    the adapters: weights, device residency, one lock each
    api/        routing, request shapes, encoding; imports no model

The direction of the dependency is `cmd/serve -> backend -> api`, never the
other way. `api` *declares* the backend interfaces (`api.SpeechBackend`,
`api.TranscriptionBackend`, `api.CompletionBackend`) and implements none of
them, so it pulls in no checkpoint loader and no Vulkan, and its tests run
against fakes in milliseconds — `go test ./api/` needs no `models/` directory
and no GPU, including for the three chat envelopes and their streams.

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

**Kokoro is not.** Its arenas are sized for one utterance's frame count, and
the durations decide that, so nothing can be staged until the phoneme side has
run (SPEECH.md T2, T4c). `-tts-gpu` therefore stages *per request* — the
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
  and `energy` from the host (SPEECH.md T7), so the host prosody that settles
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

## Two extensions, and three refusals

`POST /v1/audio/speech` takes a **`"phonemes"`** field that is not OpenAI's: IPA
to speak directly, winning over `input`. The checkpoint's vocabulary is 178 IPA
symbols and the front end that reaches them is a separate model's worth of
rules (SPEECH.md T5), so a caller that has already done that says so instead of
having it guessed. It is also how the endpoint is driven without a lexicon on
disk. `GET /v1/models` carries a **`voices`** array on the speech model, so a
client does not need a second call to populate a menu.

What is refused rather than faked:

- **`stream: true` on speech.** Kokoro decides the whole utterance's durations
  before a single sample exists, so there is nothing to send early. A whole
  body dressed as a stream would be a lie about latency.
- **`stream: true` on transcription.** The decode loop emits token by token
  and could stream, but the 40 µs submit-and-fence per emission is 38% of it
  (SPEECH.md), so what that endpoint wants is the persistent kernel, not a
  goroutine.
- **Containers other than WAV, and audio formats other than wav/pcm.**
  `audio/` is 16-bit PCM in and out on purpose. An mp3 request gets a 400
  naming what the server does encode. *Transcripts* are a different question
  and are no longer refused: `json`, `verbose_json`, `text`, `srt` and `vtt`
  all answer, because the timings they need exist.

And on the chat endpoints, the same principle with a longer list: **`n > 1`**
(n completions are n runs of a model sized to saturate the device), **`min_p`
and `repeat_penalty`** (the sampler is temperature, top_k and top_p),
**`response_format` and `text.format`** other than text, a **`tool_choice`**
that names a function, **`thinking.budget_tokens`**, and any **image content
block**. Every one of them is a 400 that says what the server
does instead. The reason is the same each time: a knob accepted and ignored is
worse than a refused one, because the client never learns that turning it did
nothing.

## What is left

- Embeddings, which need a model at all.
- **Constrained decoding**, which is the one refusal above that is a missing
  capability rather than a missing shape. It would close `response_format`,
  `text.format` and a `tool_choice` that names a function in one go.
- **More than one conversation at a time.** The graph is one sequence's, so a
  second request waits — and evicts the first's reusable prefix. What would
  change that is a cache per conversation and the residency to hold them,
  which is a measurement rather than an argument.
- **MTP speculative decoding** (LLM.md L9), which is the next thing worth
  1.5-1.8x on the endpoint that has just been wired.
- Images: `zimage/pipeline` is already the right shape for a server — resident
  weights, `Generate(prompt, seed, progress)` — but its width and height are
  fixed at construction, so `aspect_ratio` in the request is a residency
  question and not a parameter.
- **A forced alignment** for the transcript timings, if the 80 ms grid and the
  duration head's approximation ever turn out not to be enough.
