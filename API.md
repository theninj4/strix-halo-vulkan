# API — the models behind one HTTP server

`cmd/serve` is the long-lived process: an OpenAI-shaped API over the models in
this repository, with one flag per vertical deciding what is resident.

    go run ./cmd/serve -tts -stt

`GOALS.md`'s last line — "we will ultimately serve up a HTTP API serving these
features, in Go" — is what this is. It is stage one of that: the routing
layer, the two speech verticals wired to it, and the shape the other three
plug into.

## What answers today

| Endpoint | State |
|---|---|
| `GET /v1/models` | **done** — lists what the process actually loaded, with the speech model's voices |
| `POST /v1/audio/speech` | **done** — kokoro, `-tts` |
| `POST /v1/audio/transcriptions` | **done** — parakeet, `-stt` |
| `POST /v1/chat/completions` | 501 — waits on the generation loop (LLM.md L7) |
| `POST /v1/responses`, `POST /v1/messages` | 501 — the same model, two more envelopes |
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
`api.TranscriptionBackend`) and implements none of them, so it pulls in no
checkpoint loader and no Vulkan, and its tests run against fakes in
milliseconds — `go test ./api/` needs no `models/` directory and no GPU.

What crosses the boundary is deliberately model-shaped and not
container-shaped. A speech backend returns `*audio.Clip` — samples and a rate
— and the HTTP layer encodes the WAV or the raw PCM, because which one was
asked for is the request's business and not the model's. A transcription
backend is handed an already-decoded clip for the same reason in reverse.

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
- **Containers other than WAV, and formats other than wav/pcm.** `audio/` is
  16-bit PCM in and out on purpose. An mp3 request gets a 400 naming what the
  server does encode.

## What is left

- The language model: `/v1/chat/completions` first, then `/v1/responses` and
  `/v1/messages` over the same backend, with SSE. The interface is not
  declared yet on purpose — a chat backend's real signature is a streaming
  one, and what that looks like is decided by the generation loop in `llm`
  (LLM.md L7), not in advance here.
- Embeddings, which need a model at all.
- Images: `zimage/pipeline` is already the right shape for a server — resident
  weights, `Generate(prompt, seed, progress)` — but its width and height are
  fixed at construction, so `aspect_ratio` in the request is a residency
  question and not a parameter.
- `verbose_json` transcripts. The transducer records the frame every emission
  was made at, in 80 ms units, so the timings exist; only the shape is missing.
