# Extensions, and what is refused rather than faked

*The extensions section (image edits and presets with it), broken out of `API.md` when it moved to `research/` on 2026-10-02; the server's overview, flags and envelopes are in [`api-server.md`](api-server.md).*


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

**It needs no flag.** Under Z-Image-Turbo previewing meant loading
`madebyollin/taef1` and 1.0 GB of activation arena, so `-preview` was a
residency decision. Here the preview decoder is **TAEQI2.1**
(`madebyollin/taesd`, 2026-09-25), a tiny decoder distilled against this
very VAE: 7.6 M parameters, 29 MB of weights and 224 MB of arena at a 1024²
ceiling, always staged. It replaced Q7's fitted 64→RGBA matrix, which
stood in while no such decoder existed. On the final latent of a real 1024²
run it lands rms 0.087 from the full VAE's image against the matrix's 0.20
(research/qimage-vertical.md Q14).

Measured under the fitted matrix, on one `-image -image-size 512x512
-image-steps 16` process, two runs each (not yet re-measured with TAEQI2.1,
which adds ~20 ms of device time a frame at 512²):

| request | wall clock | first picture |
|---|---|---|
| no stream | 10.25 s / 10.23 | 10.3 s |
| `stream: true` | 10.26 s / 10.28 | 10.3 s |
| `stream: true, partial_images: 3` | **10.27 s** / 10.28 | **2.27 s** |

So the framing is free and three in-progress frames cost **0.3%** — against
4.4% through taef1 — and what they buy is a picture at 2.3 seconds instead
of at ten. The frames land at 2.27, 4.34 and 6.43 s, from steps 3, 7 and 11
of the sixteen.

A preview decode is **20 ms at 512² and 63 ms at 1024²** on the device,
against ~0.6 s and ~2.2 s steps. Partial frames go out at `png.BestSpeed` —
15% more bytes for a fifth of the latency — while the finished image keeps
the careful encoder. A transient frame and a deliverable are not the same
object. **A frame is the finished image's size**, with texture, and it
follows the request's `background`: a transparent request's frames are
transparent too.

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

**A client that hangs up stops the run**, streamed or not. A denoising step
is a submit-and-fence and cannot be abandoned from outside, but between two
steps nothing is in flight, and that is where the request's context is
checked — likewise between the VAE's submit batches, and between reference
images and inside the vision tower on an edit. So an abandoned 1024²/40
request costs the step it was in (~2.2 s of 92) or a fraction of a second of
the decode, rather than the whole picture. That is worth more than the
arithmetic it saves: one lock serialises every model in the process, so an
abandoned image used to hold the queued speech, transcription and embedding
requests behind it for its full duration. Text completions have always
cancelled this way, between tokens; images, speech and transcription checked
only on the way in, and images now check throughout.

## Editing a picture

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
  reference, an edit is **1m54s** against a generation's 1m28.8s — the
  reference's own encoding is 7.1 s of it (a 27-layer vision tower and a VAE
  encode), and the rest is the prefix riding along in every denoising step.
- **References arrive as OpenAI sends them**: multipart `image` / `image[]`
  parts, JSON base64 `image`, or OpenAI's JSON `images: [{"image_url": …}]`,
  where an http(s) URL is fetched as OpenAI fetches it (the same capped
  fetch as a chat image) and a data: URL is read; `{"file_id": …}` is a 400,
  there being no Files API. A `mask`, in any of its forms, is the 501 below.
- **`image[]` is a real list.** Up to `max_reference_images` pictures are
  accepted, in the order they were sent, and the model's own limit is ten.
  How many *this* server takes is residency — every reference adds its latent
  rows to the prefix and its share of the prefix KV cache — so it is fixed by
  `-edits N` at startup and reported in `GET /v1/models`.
- **An edit with no `size` follows the last reference image's aspect ratio**
  at the condition area, which is what diffusers does. It is not the server's
  default size, and it is not the picture's own pixel dimensions either.
- **`background` means what it means on a generation.** `"transparent"`
  rewrites the prompt and keeps the result's alpha plane; anything else
  composites it over white. A reference image's own alpha reaches the model
  either way, because the VAE encodes all four channels, so a transparent
  PNG sent in is seen as transparent whatever `background` says.

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
  in the diffusers implementation this port follows
  (research/qimage-vertical.md's open question Q-o3). It is a 501 that says
  what it would be rather than a picture that ignored it.
- **`background: "transparent"` with a JPEG container.** A 400 naming both
  fields. JPEG has no alpha channel, and flattening it silently would hand
  back an opaque picture that the prompt rewrite had also made worse.

And on the chat endpoints, the same principle with a longer list: **`n > 1`**
(n completions are n runs of a model sized to saturate the device),
**`response_format` and `text.format`** other than text, a **`tool_choice`**
that names a function, **`thinking.budget_tokens`**, and any **image content
block**. Every one of them is a 400 that says what the server
does instead. The reason is the same each time: a knob accepted and ignored is
worse than a refused one, because the client never learns that turning it did
nothing.

## Presets: one set of weights, several model names

`models.ini` is llama-server's preset file, read at startup (`-llm-presets`;
the default is skipped if it is missing, and a key the server does not apply
fails the start). Each section is a model name a chat request can put in
`model` -- `chatting`, `instruct`, `thinking`, `coding` -- and `GET
/v1/models` lists them beside the checkpoint. A preset is **an override**:
whatever it sets of `temperature`, `top_p`, `top_k`, `min_p`,
`repeat_penalty`, `presence_penalty` and `chat_template_kwargs` replaces what
the request sent, and what it leaves out is still the request's. The
response's `model` is the preset's name. A
model name that is not a preset gets the checkpoint's own defaults, as before.

The sampler takes llama.cpp's **`min_p`, `repeat_penalty` and
`presence_penalty`**, the penalties over the last 64 tokens (prompt included,
as llama-server does). **`chat_template_kwargs`** is llama-server's request
field; the template reads `enable_thinking`, `preserve_thinking` and
`reasoning_effort`, and any other is a 400. A top-level `reasoning_effort`
(and the Responses API's `reasoning.effort`, which is the same field) decides
thinking outright -- except under a preset that sets `enable_thinking` or
`reasoning_effort`, where it is dropped. All four shipped presets set both, so
`chatting` and `instruct` never think and `thinking` and `coding` always do,
whatever the client sends.
