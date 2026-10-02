# API — the two doors that are not OpenAI's

*The Wyoming and System One sections, broken out of `API.md` when it moved to `research/` on 2026-10-02; the server's overview, flags and envelopes are in [`api-server.md`](api-server.md).*

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

## System One is not an OpenAI envelope

`POST /v1/systemone` is TypeSafe's contract, as
[Kev](https://github.com/jaredpalmer/kev) serves it, and it is answered the
way Kev answers it rather than translated into anything of ours:

```json
{"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
 "model": "kev-latest",
 "questions": {
   "department":  {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                "shipping": "Delivery status, delays, lost packages",
                                "billing": "Charges, invoices, payment problems"}},
   "escalate":    {"type": "noul",  "instructions": "Does this need urgent human attention?"},
   "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}}}
```

gets `answers` keyed by question id, with a `noul` answer as p(yes), a `choice`
as its argmax, `probabilities` and `confidence`, and a `score` as the expected
level index with `legend`, `probabilities` and `confidence`. The response also
carries `usage` and `latency_ms`, the model time. `usage.output_tokens` counts
the serialised answers, not generated tokens (none are generated). The
`x-typesafe-request-id` header is echoed, or minted when the caller sends
none.

- **The body is not decoded by `api`.** Key order in an object state, and
  int-against-float in a number, both change the text the model reads, and
  criteria order decides the option slots. So the bytes go to the backend,
  which parses them by Kev's rules (`kev/value.go`).
- **Invalid is 422**, as TypeSafe and Kev answer it: an unknown `type`,
  missing criteria, 0 or more than 255 options, a question whose row (state
  plus branch) is over 8,192 tokens, or a state or single question longer
  than `-kev-tokens`. A longer *request* runs as several passes, which
  answer bit-identically to one. A state over 8,191 tokens is truncated
  silently, as Kev does.
- **A repeated text is cached.** The last four states' KV and recurrent
  state (up to 4096 tokens each, `-kev-cache`/`-kev-cache-tokens`) are kept
  by token ids. A new question about a text already sent pays for its
  question only, and gets the bits it would have got without the cache.
  `latency_ms` shows it: a 2,269-token text is ~0.76 s new and ~0.09 s again.
- **Concurrent requests share passes.** One worker takes whatever is
  queued when the device frees up, up to `-kev-batch`, and runs it as one
  pass, each request in its own state slot. Sixteen concurrent short
  requests run at ~29/s against ~19.5/s one after another. `latency_ms` is
  the pass's model time, so it grows with the batch. Answers can move by up
  to ~6e-4 with the batch's composition (the matrix-core attention); no
  argmax changed over 764 suite questions.
- **`kev-latest` and `jev-latest`** both name this model. The second is the
  SDK's default, so an unconfigured client works. `GET /v1/models` lists
  both. TypeSafe's own `{"models": [...]}` card is not served, since the SDK
  does not read it.
- The questions are **isolated**. Rewriting or removing one leaves every
  other answer bit-identical (`TestGPUQuestionsAreIsolated`). Options inside
  one question are *not* independent, and reordering them can move the
  answer. That is Kev's and Jev's behaviour, not a defect of this server.
