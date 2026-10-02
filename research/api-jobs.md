# API — videos and music are jobs

*The video and music job sections, broken out of `API.md` when it moved to `research/` on 2026-10-02; the server's overview, flags and envelopes are in [`api-server.md`](api-server.md).*

## Videos are jobs

A video is minutes at the served 480p and hours at the trained 768p
(VIDEO.md), so `/v1/videos` is OpenAI's asynchronous shape and not a request
that waits: `POST` answers at once with a job, the client polls `GET
/v1/videos/{id}` until `status` is `completed` (or `failed`, with an `error`),
and fetches `GET /v1/videos/{id}/content`, an H.264/AAC mp4. `DELETE` cancels
a queued or running job — within one ≤ 800 ms submission — and forgets it. A
finished job and its file are kept for `-video-ttl` and then 404, as does
everything after a restart: jobs live in memory.

```sh
curl -s localhost:8080/v1/videos -H 'Content-Type: application/json' \
  -d '{"prompt": "…", "size": "864x480", "seconds": "5"}'          # OpenAI's shape
curl -s localhost:8080/v1/videos -H 'Content-Type: application/json' \
  -d '{"task": "t2va", "prompt": "…", "seed": 0,
       "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10}}'  # SGLang's
curl -s localhost:8080/v1/videos -F prompt="…" -F input_reference=@start.png          # OpenAI's, from a first frame
curl -s localhost:8080/v1/videos -H 'Content-Type: application/json' \
  -d '{"task": "fl2va", "prompt": "…", "target": {"short_edge": 480, "aspect_ratio": "auto"},
       "conditions": [{"type": "image", "uri": "data:image/png;base64,…", "role": "keyframe", "frame_index": 0},
                      {"type": "image", "uri": "data:image/jpeg;base64,…", "role": "keyframe", "frame_index": -1}]}'
```

Both envelopes are read, and a multipart form as OpenAI's SDK sends one.
**`size` is read as an aspect ratio and a short edge**, not a canvas: the
model's canvases are on a 32-pixel grid, so `1280x720` runs as 1280x704 and
the job's `size` says so. `seconds` below the model's 5 s floor is raised
to it (OpenAI's clients default to "4"); past 15 s is a 400. So is a request
the transformer's arenas cannot span — 87,296 rows, which is the trained
768p canvas to ~11.8 s — and that is found at submit time, before 50 GB of
text encoder is staged for it. Extensions: `steps` (or
`num_inference_steps`; default 20, the release's 50), `seed` (reported
either way), and on the job `seed`, `steps`, `frames`, `stage` and
`estimated_seconds`, the backend's estimate of the run from measured rates.
`variant` other than `video` is a 400.

**Keyframes (`fl2va`, VIDEO.md M10)** pin the video's first frame, its last,
or both. OpenAI's `input_reference` is the first frame: a multipart file, or in JSON
OpenAI's `{"image_url": …}` (an http(s) URL is fetched, as OpenAI fetches it;
a data: URL is read) — `{"file_id": …}` is a 400, there being no Files API —
or, leniently, a bare URL or base64 string. SGLang's `conditions` are up to
two images with a `frame_index`: 0 is the first frame, and -1 — or any index
at or past the last requested frame — is the last; without one, the first
condition is the first frame and a second the last. A condition's `uri` is
fetched the same way (SGLang fetches it too), so the README's requests run
unchanged. URLs are fetched at submit, so an unreadable picture is a 400 and
not a failed job; the fetch is the chat images' (50 MB, 60 s). png and jpeg
are read, up to 64 megapixels. With no `size` and `aspect_ratio`
`auto` (or none), the canvas takes the first keyframe's aspect ratio; the
first keyframe is stretched onto the canvas and a second one cover-cropped,
as the reference pipeline does. `task: t2va` with a keyframe, or `ref2va`
(references, VIDEO.md M13), is a 400.

**The queue is in `api`, and one job runs at a time.** The backend is
synchronous — one mp4 to a path — and a request's stagings peak at ~50 GB,
so two at once would not fit. **The device is not held for the job.** The
stagings (~2 of a request's minutes, reading bf16 from disk) allocate and
write mapped memory and never touch the queue, so they take no lock; what
does — the encoder's forward, each transformer forward, the VAE decode —
yields the device between submissions, so a speech request behind a 35 s
forward waits for one submission, not the forward.

## Music is a job

OpenAI has no music endpoint, so `/v1/music` takes the shape of the one it
has for a generator that outlasts a connection: `/v1/videos`, route for
route (MUSIC.md A9). `POST` answers at once with a job, the client polls
`GET /v1/music/{id}` until `status` is `completed` (or `failed`, with an
`error`), and fetches `GET /v1/music/{id}/content`. `DELETE` cancels a queued
or running job and forgets it. A finished job and its file are kept for
`-music-ttl`; jobs live in memory, so a restart 404s them.

```sh
curl -s localhost:8080/v1/music -H 'Content-Type: application/json' \
  -d '{"prompt": "warm acoustic folk, fingerpicked guitar, male vocals",
       "lyrics": "[Verse]\nMorning light on the river\n…", "audio_duration": 30, "audio_format": "flac"}'
curl -s localhost:8080/v1/music/music_…            # status, progress, stage, and the plan once there is one
curl -s localhost:8080/v1/music/music_…/content -o song.flac
```

**The request is ACE-Step's own** (`POST /release_task` in its API docs):
its field names and their aliases, flat or inside a `metas`, `metadata` or
`user_metadata` object, as JSON or a form. So a client of ACE-Step's server
changes the URL and reads a job instead of a `task_id`.

- `prompt`/`caption`, `lyrics` (section tags: `[Verse]`, `[Chorus]`,
  `[Instrumental]`…). One of caption or lyrics is required, unless the
  request is in sample mode.
- `sample_query` (or `description`/`desc`): **sample mode**, upstream's
  "simple mode" (MUSIC.md A12). From a description alone ("a melancholy
  synthwave song about driving through a neon city at night") the LM first
  writes the caption, bpm, key, time signature, duration, genres and the
  lyrics, and the song is then made from those. A language named in the
  description ("… japanese city pop …") holds the lyrics to it, as does a
  `vocal_language` (upstream ignores an `en` there, its default; here it
  holds the lyrics to English); "instrumental", "pure music" or a
  trailing "solo" asks for no lyrics. `sample_mode: true` with no query
  lets the LM choose the whole song. A caption or lyrics sent with it is a
  400: upstream would discard them. Unlike upstream, the request's own
  `audio_duration`, `bpm`, `key_scale` and `time_signature` are kept: the
  LM writes the song around them.
- `thinking` (default **true**, upstream's pipeline default; its API server
  defaults to false): the 5 Hz LM writes the metadata the request leaves out
  and the audio codes the DiT renders. `thinking: false` is the DiT alone.
- `audio_duration`/`duration` (10–600 s). Without it the LM chooses the
  length when thinking, and it is upstream's 120 s when not.
- `bpm` (30–300), `key_scale`/`keyscale`, `time_signature` (2, 3, 4, 6 or
  2/4, 3/4, 4/4, 6/8), `vocal_language` (the LM's language codes).
- `audio_format`/`response_format`: `mp3` (default), `wav`, `flac`, `opus`.
- `seed` (the DiT's noise) and `lm_seed` (the LM's sampling): given, they
  are used (upstream also wants `use_random_seed: false`; here `true`
  still draws); drawn, they are 32-bit and reported, so a job can be rerun.
- `shift` and `timesteps` (the DiT's schedule), `lm_temperature` (0 is
  greedy), `lm_cfg_scale`, `lm_top_p`.
- `inference_steps` and `guidance_scale` are accepted and change nothing,
  as upstream's turbo model ignores them too (MUSIC.md A1).

**Audio in** (MUSIC.md A11) is upstream's multipart upload, in any format
ffmpeg reads:

```sh
curl -s localhost:8080/v1/music -F task_type=cover -F src_audio=@song.mp3 \
  -F prompt="acoustic folk ballad, male vocals" -F lyrics="$(cat song.txt)"
curl -s localhost:8080/v1/music -F task_type=repaint -F src_audio=@song.mp3 \
  -F repainting_start=10 -F repainting_end=20 -F prompt="…, a saxophone solo"
curl -s localhost:8080/v1/music -F reference_audio=@voice.flac -F prompt="…" -F audio_duration=60
```

- `task_type`: `text2music` (default), or the turbo model's three audio
  tasks. They need `src_audio` (or `ctx_audio`), skip the LM as upstream's
  do whatever `thinking` says, and last as long as the source, so an
  `audio_duration` is a 400.
  - `cover`: the source through the audio tokenizer is what the DiT
    renders, so its structure and melody are kept and the caption and
    lyrics restyle it. `audio_cover_strength` (default 1) is the share of
    steps conditioned on the source; the rest are plain text2music.
    `cover_noise_strength` (default 0) starts that close to the source
    instead of from noise.
  - `cover-nofsq`: the same, from the source's latents as they are, with
    no tokenizer bottleneck.
  - `repaint`: `repainting_start`/`repainting_end` (seconds; an end of 0
    or −1 is the source's end) is regenerated and the rest kept. A span past
    either end extends the song (outpainting). `repaint_mode` is
    `conservative`, `balanced` (default, with `repaint_strength` 0.5) or
    `aggressive`, and sets how long the kept part is pinned during
    sampling and the crossfades at the span's edges. `chunk_mask_mode:
    "explicit"` also tells the DiT where the span is.
- `reference_audio` (or `ref_audio`), with any task: 30 s from three
  random places in it give the song its timbre.
- **Departures:** upstream's `src_audio_path` and `reference_audio_path`
  name files on its server, and this server reads none of its own, so they
  are a 400 that says to upload. A source at another rate is resampled by
  ffmpeg, not by torchaudio, and a file of more than two channels is
  downmixed rather than cut to its first two. The base model's tasks
  (`lego`, `extract`, `complete`) are a 400. So is a field its task would
  not use, which upstream drops silently: a cover's strengths on
  text2music, a repaint's span on anything else, a custom `instruction`,
  and the crossfade fields that `repaint_mode` overrides.

**What ACE-Step's API offers and this server does not run is a 400 naming
the field**, never quietly ignored: the base model's tasks, audio named by
a server path, `batch_size` above 1 (submit one job per song),
format mode, `audio_code_string`, the `sde` sampler, top-k, a
repetition penalty or negative prompt, and turning off the LM's CoT caption,
language, metas or constrained decoding. Their upstream defaults are
accepted. So is a request that would not fit: a duration or bpm out of
range, or lyrics so long the LM's cache could not hold them with the codes
behind — found at submit, not twenty seconds in.

**The job tells a client what the LM planned before the audio exists.**
`seconds` is null until the length is known (at once when the request gave
it; after the CoT when the LM chose it), `metadata` holds what the DiT was
asked for (the rewritten caption, bpm, key, time signature, duration,
language, and in sample mode the `lyrics` and `genres` the LM wrote, there
as soon as it has written them), `stage` is `sample`, `think`, `codes`,
`dit`, `vae` or `write`, and
`estimated_seconds` is redone from measured rates once the length is known.
Three jobs submitted together on this machine (MUSIC.md A9, before A10's
int8 LM halved its step):

| request | LM chose | estimated | ran |
|---|---|---|---|
| folk, lyrics, 30 s, flac | the metas; the length was given | 17 s | 16.2 s |
| techno, instrumental, no duration | 233 s, 136 bpm, E♭ minor | 43 → 76 s | 78.7 s |
| piano, `thinking: false`, 20 s | — | 1 s | 1.4 s |

Since A10 (the LM's layers in int8, `-music-lm-fp16` for the control), a
30 s folk song with lyrics runs in 9.6 s and a 3-minute techno track in
34.9 s.

**Everything is resident while it holds the swap slot, and one job runs at
a time.** The pipeline holds ~18 GB, and staging it per request (~16-19 s)
would be most of a request, so it stays staged between songs until image or
video is asked for or `-swap-idle` passes; a job that has to wait for the
load reports `stage: "load"`. **The
device is not held for the song**: the job yields it after every LM step
(~25 ms) and every DiT forward, so speech behind a song waits for one of
those. Measured with `-tts` beside it, a short speech request took 32 ms
idle, 32–71 ms while the LM planned a 3-minute song, and 0.14–0.31 s during
its DiT forwards.
