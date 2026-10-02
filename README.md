# strix-halo-vulkan

A self-hosted AI server for the AMD Strix Halo APU (Ryzen AI Max, Radeon
8060S iGPU, gfx1151) written in Go, with every model running on hand-written
Vulkan compute shaders. There is no PyTorch, ROCm or llama.cpp at runtime.
One process, `cmd/serve`, puts text generation, vision, speech, embeddings,
classification, OCR, images, video and music behind an API shaped like
OpenAI's.

This file covers what the server answers and how to build, deploy and call it.
For the internals, see [`research/`](research/README.md) (one file per vertical
and per experiment), [`PERFORMANCE.md`](PERFORMANCE.md) (headline numbers) and
[`GOALS.md`](GOALS.md).

## Models

Each model is turned on by its own flag. A model whose flag was not given is
never loaded, and its endpoints answer `501` with the name of the flag that
would load it.

| Flag | Model | Endpoints | Resident |
|---|---|---|---|
| `-llm` | Qwen3.8-Flash-Next (GGUF, MoE) | `/v1/chat/completions`, `/v1/responses`, `/v1/messages` | ~84 GB (more with long context) |
| `-llm-mmproj` | Its vision tower | images in the three chat endpoints | small |
| `-llm-draft` | Its MTP draft head | speculative decoding, ~1.4x on greedy decode | small |
| `-embed` | Qwen3-Embedding-0.6B | `/v1/embeddings` | 0.9 GB |
| `-tts` | Kokoro-82M (+ misaki lexicon, espeak-ng fallback) | `/v1/audio/speech` | small |
| `-stt` | parakeet-tdt-0.6b-v3 | `/v1/audio/transcriptions` | small |
| `-kev` | Kev-4B (TypeSafe System One) | `/v1/systemone` | ~9 GB |
| `-ocr` | PaddleOCR-VL-1.6 + PP-DocLayoutV3 | `/v1/ocr`, and `/v1/chat/completions` as model `PaddleOCR-VL-1.6-0.9B` | small |
| `-image` / `-edits N` | Qwen-Image-2.1 | `/v1/images/generations`, `/v1/images/edits` | ~20 GB while resident |
| `-video` | MiniMax-H3 (video with a soundtrack) | `/v1/videos` | ~0.3 GB at rest, ~31 GB per request |
| `-music` | ACE-Step 1.5 XL turbo + 5 Hz LM | `/v1/music` | ~18–26 GB while resident |

`-image`, `-video` and `-music` share **one residency slot**. A request loads
its own model into the slot, unloading whichever of the other two is there, and
the slot empties after `-swap-idle` (default 10m) with no requests. The three
together therefore need only as much memory as the largest of them. Switching
models costs a load, about 15–30 s.

`-llm` and the image model do not fit in 128 GB together. In practice the
language model runs on one machine and everything else on another (see
[Deploying](#deploying)).

## Building

Requirements:

- Linux with Mesa RADV (developed on Mesa 26.x, Arch).
- Go (see `go.mod`), a C compiler and `pkg-config`, because `vk/` uses cgo.
- The Vulkan loader and headers (`vulkan-icd-loader`, `vulkan-headers`).
- `glslc` (from `shaderc`). The SPIR-V files are not committed, so the shaders
  must be compiled once after cloning and again after any `.comp` file changes.
- Runtime tools, needed only by the models that use them: `espeak-ng` (`-tts`, for
  words that are not in the lexicon; loaded as a shared library), `ffmpeg`
  (`-video` muxing and `-music` output formats) and poppler's
  `pdfinfo`/`pdftoppm` (`-ocr`, for PDFs).

```sh
go generate ./shaders      # compile shaders/*.comp to SPIR-V
go build -o ai ./cmd/serve
```

### Model checkpoints

Checkpoints live under `models/`, which is git-ignored. Every flag has a
`-<name>-model` partner whose default is the directory the server expects:

```
models/Qwen3.8-Flash-Next-GGUF/   -llm (the UD-Q4_K_XL shards), mmproj-BF16.gguf, mtp-*.gguf
models/Qwen3-Embedding-0.6B/      -embed (also used by -music)
models/Kokoro-82M/  models/misaki/ -tts
models/parakeet-tdt-0.6b-v3/      -stt
models/kev-4b/  models/Qwen3.5-4B-Base/   -kev
models/PaddleOCR-VL-1.6/  models/PP-DocLayoutV3/   -ocr
models/Qwen-Image-2.1/  models/taeqi2_1/   -image
models/MiniMax-H3/                -video
models/acestep-v15-xl-turbo/  models/acestep-5Hz-lm-4B/  models/Ace-Step1.5/vae   -music
```

Most of these are Hugging Face downloads used as-is. A few (Kokoro, misaki,
the Kev head) are first converted by a script in `reference/`. Each
vertical's file in `research/` records exactly which revision it was built
against and any conversion step.

Some models stage int8 weight banks the first time they load. The video model
writes its banks to `bank-cache/` beside the checkpoint (~50 GB of disk), which
saves about 70 s on every request after the first.

## Running

```sh
./ai -addr 127.0.0.1:8080 -token=secret -tts -stt -embed
./ai -llm -llm-ctx 262144 -llm-slots 3 -llm-mmproj models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf
./ai -image -edits 3 -video -music -swap-idle 10m
./ai -tts -stt -wyoming 0.0.0.0:10300          # also answer Home Assistant's Wyoming protocol
./ai -llm -llm-layers 4                         # a 7 GB server for working on the HTTP side (not real answers)
```

`./ai -h` lists every flag. The ones that most affect deployment:

| Flag | Default | |
|---|---|---|
| `-addr` | `127.0.0.1:8080` | listen address (loopback by default) |
| `-token` | built-in | bearer token. **Leaving it out does not disable auth**: the built-in default token is used. Pass `-token=` (empty) to serve with no auth |
| `-max-upload` | `128` | largest body that carries a file, in MB |
| `-log-bodies` | `false` | log prompts and answers (off by default, because they are conversations) |
| `-swap-idle` | `10m` | how long image/video/music stay loaded with no requests |
| `-llm-ctx` | `4096` | KV cells per conversation: prompt plus completion, up to 262144 |
| `-llm-batch` | `4096` | prefill chunk width. A wider chunk costs memory, not speed |
| `-llm-slots` | `1` | conversations held at once. Prefill and decode are interleaved between them, and decode is batched |
| `-llm-reserve` | `1` | slots that only interactive requests may take |
| `-llm-class` | `background` | priority of a request that sets none |
| `-llm-presets` | `models.ini` | llama-server preset file (see [Presets](#presets)) |
| `-llm-max-tokens` | `1024` | `max_tokens` for a request that sets none |
| `-embed-tokens` | `512` | longest embedding input; longer inputs are truncated |
| `-max-audio` | `60` | longest transcription clip, in seconds |
| `-image-size` | `1024x1024` | largest image *area*. Any aspect ratio within that many pixels is served |
| `-video-dir`, `-music-dir` | temp dir | where finished jobs' files are kept until `-*-ttl` (24h) |

Requests share one GPU queue. HTTP handling is concurrent, but GPU work runs one
request at a time, apart from the batching each model does internally (the
LLM's slots, Kev's and the embedder's request batching, OCR's paged decode).

## Deploying

The server runs as a **user** systemd unit called `ai`. `deploy.sh` builds `./ai`,
writes `ai.service` into `~/.config/systemd/user/` with this checkout's path
filled in, restarts the unit and follows the log:

```sh
./deploy.sh
```

One-time setup on a new machine:

```sh
systemctl --user daemon-reload
systemctl --user enable --now ai
loginctl enable-linger $USER    # keep it running without a login session
```

`ai.service` holds two `ExecStart` lines. One runs the language model by
itself, with full 262k context, three slots, vision and the draft head (~98 GB).
The other runs everything else with the image, video and music swap slot. Keep
the line you want on each machine and comment out the other.
Startup takes about 25 s. The logs are at `journalctl --user -u ai -f`.

## API

Every route is under `/v1` and needs `Authorization: Bearer <token>` unless the
server was started with `-token=`. CORS preflight is answered. Errors use
OpenAI's `{"error": {...}}` envelope. The server follows OpenAI's documented
behaviour as well as its request shapes: for example, an image URL is fetched
wherever OpenAI would fetch it. Parameters the server cannot honour are refused
by name, not silently ignored.

### Discovery

| | |
|---|---|
| `GET /v1/models` | The models this process actually loaded, including TTS voices and the LLM presets. A client can discover what an instance can do from this list. |

### Text generation

| | |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions: buffered or SSE (`stream`, `stream_options.include_usage`), tools / `tool_choice`, `response_format` with JSON schema, `reasoning_effort`, `image_url` content parts (with `-llm-mmproj`), plus llama.cpp/vLLM extras (`top_k`, `min_p`, `repetition_penalty`, `chat_template_kwargs`, `seed`). |
| `POST /v1/responses` | OpenAI Responses: the same generation in OpenAI's newer envelope, including `input_image`. |
| `POST /v1/messages` | Anthropic Messages: the same generation, including `image` blocks. |

The chat endpoint routes by `model`: `PaddleOCR-VL-1.6-0.9B` goes to the OCR
model, and every other name goes to the LLM.

With more than one slot, a follow-up turn of a conversation resumes from that
conversation's cached state instead of prefilling the whole history again.

**Priority.** Requests are `interactive` or `background`. Set the class with
the `X-Priority` header or with `service_tier` in the body (`priority` counts as
interactive; `flex`, `default` and `auto` count as background). Interactive
requests always run first, and a background prefill yields to them every
`-llm-preempt-chunk` tokens. The header exists for clients, such as a voice
assistant, that cannot add fields to the request body.

#### Presets

`models.ini` is llama-server's `--models-preset` format. Each section is an
extra model name that uses the same weights with fixed sampling and template
settings. The shipped file defines `chatting`, `instruct`, `thinking` and
`coding`. Settings in the file override the request's, field by field; a
setting the file leaves out still comes from the request. An unknown key is an
error at startup.

```sh
curl localhost:8080/v1/chat/completions -H "Authorization: Bearer $TOKEN" \
  -d '{"model": "thinking", "messages": [{"role": "user", "content": "Why is the sky blue?"}], "stream": true}'
```

### Embeddings

`POST /v1/embeddings`. Request fields: `input` (a string or a list),
`encoding_format` (`float` or `base64`), `dimensions` (Matryoshka truncation),
and `instruct`, an extension that sets the query instruction.

### Speech

| | |
|---|---|
| `POST /v1/audio/speech` | `input`, `voice` (a Kokoro voice, or a comma-joined mix of voices), `speed` 0.5–2.0, `response_format` `wav` (the default) or `pcm`, `stream`. There is no audio encoder, so `mp3`, `opus`, `aac` and `flac` are refused with a `400` (OpenAI's default is `mp3`, so set `wav` explicitly). Extension: `phonemes` bypasses the text-to-phoneme step. |
| `POST /v1/audio/transcriptions` | multipart `file`. `response_format` `json`, `text`, `verbose_json`, `srt` or `vtt`; `timestamp_granularities` `word` and/or `segment`; `stream` (SSE `transcript.text.delta` / `.done`). |

**Wyoming.** `-wyoming host:port` also serves the same two speech models over
Home Assistant's Wyoming protocol, from the same process. Wyoming has no
authentication, so anyone who can reach that port can use both models.

### Classification: System One

`POST /v1/systemone` implements TypeSafe's System One contract as Kev serves it:
one text and typed questions in, calibrated probabilities out, with nothing
generated. The TypeSafe Python SDK works against it unchanged. Model names:
`kev-latest`, `jev-latest`.

```json
{"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
 "model": "kev-latest",
 "questions": {
   "department":  {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong items",
                                "shipping": "Delivery status, delays",
                                "billing": "Charges, invoices, payments"}},
   "escalate":    {"type": "noul",  "instructions": "Does this need urgent human attention?"},
   "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}}}
```

The response has `answers` keyed by question id, plus `usage` and
`latency_ms`. The `X-Typesafe-Request-Id` header is echoed back, or created
when the request has none. A malformed question gets a `422`.

### OCR and document parsing

| | |
|---|---|
| `POST /v1/ocr` | Mistral's OCR API. `document` is a `document_url` (PDF) or an `image_url`. Also `pages`, `include_image_base64`, `table_format`, `extract_header` / `extract_footer`. Returns `pages[].markdown` with image references, `images[]` with bounding boxes, `dimensions` and `usage_info`. Up to `-ocr-max-pages` pages. |
| `POST /v1/chat/completions` with `model: "PaddleOCR-VL-1.6-0.9B"` | Element-level OCR in PaddleOCR-VL's own prompt format. This is what PaddleOCR's pipeline expects when it calls a VLM server. |

### Images

| | |
|---|---|
| `POST /v1/images/generations` | `prompt`, `size` or `aspect_ratio`, `n`, `seed`, `steps`, `output_format` / `output_compression`, `background: "transparent"` (RGBA output), `stream` with `partial_images` (preview frames from TAEQI2.1). |
| `POST /v1/images/edits` | Requires `-edits N`. Takes up to N reference images (`image` or `images`, as multipart or JSON). The edit is conditional generation, not SDEdit, so `strength` and `mask` are refused. |

`size` is limited by the area set with `-image-size`, not by its shape: at
`1024x1024`, `aspect_ratio: "16:9"` gives 1344x768.

### Video (asynchronous jobs)

OpenAI's video job API. Each video is an mp4 at 24 fps with a 32 kHz stereo
soundtrack.

```sh
curl localhost:8080/v1/videos -d '{"prompt": "…", "size": "864x480", "seconds": "5"}'
curl localhost:8080/v1/videos -F prompt="…" -F input_reference=@start.png   # from a first frame
curl localhost:8080/v1/videos/video_…            # status, progress, stage, estimated_seconds
curl localhost:8080/v1/videos/video_…/content -o out.mp4
```

| | |
|---|---|
| `POST /v1/videos` | Create a job. Takes OpenAI's shape, or SGLang's MiniMax-H3 shape (`task`, `target`, `conditions` with first/last keyframes, `seed`, `steps`). |
| `GET /v1/videos` | List jobs. |
| `GET /v1/videos/{id}` | Job status. |
| `GET /v1/videos/{id}/content` | The finished mp4. |
| `DELETE /v1/videos/{id}` | Delete a job and its file. |

Up to `-video-queue` jobs wait behind the one that is running; beyond that the
server returns `429`. Finished jobs expire after `-video-ttl`.

### Music (asynchronous jobs)

Jobs work the same way as video, under `/v1/music`, with the same five routes.
The request body uses ACE-Step's own `release_task` field names and aliases, as
JSON or a form. A client written for ACE-Step's server only needs to change the
URL and read a job back instead of a `task_id`.

```sh
curl localhost:8080/v1/music -d '{"prompt": "warm acoustic folk, fingerpicked guitar, male vocals",
  "lyrics": "[Verse]\nMorning light on the river\n…", "audio_duration": 30, "audio_format": "flac"}'
curl localhost:8080/v1/music -F task_type=cover -F src_audio=@song.mp3 -F prompt="acoustic folk ballad"
```

Songs are 48 kHz stereo and 10 s to 10 min long. `thinking` (the default)
uses the 5 Hz LM to plan the song's metadata. Sample mode generates a song
from a description alone. `task_type` is `text2music` (the default), `cover`,
`repaint` or a reference-audio task.

## Other commands

`cmd/serve` is the product. The rest of `cmd/` are drivers and measurement
tools, one per vertical, used during development; each command's `-h`
describes it:

- `llm`, `qimage`, `h3`, `ace`, `ocr`, `kev`, `tts`, `asr`, `embed`: run one model from the command line.
- `loadgen`, `kevload`, `roundtrip`: drive a running server.
- `bench`, `probe`, `bus`, `quanterr`, `gguf`, `inspect`: the kernel benchmark suite and other inspection tools.

## Repository layout

```
cmd/serve/   flags and wiring
api/         HTTP routes and request/response shapes; no models, tests run without a GPU
backend/     adapters that own weights, residency and scheduling
wyoming/     the Wyoming protocol door
vk/          the Vulkan engine (cgo + a small C shim)
shaders/     GLSL compute shaders, compiled with go generate
llm/ embed/ kokoro/ parakeet/ kev/ ocr/ qimage/ h3/ ace/   one package per model
reference/   Python oracles and checkpoint converters
research/    design notes and measurements, one file per vertical or experiment
```

## Testing

```sh
go test ./api/        # fast: no GPU and no models/ needed
go test ./backend/    # needs the GPU and checkpoints
```

The model packages' tests compare against reference dumps from `reference/` and
need both the checkpoints and the GPU. The full `llm` suite takes the whole
machine. See [`research/llm-plan.md`](research/llm-plan.md) before running it.
