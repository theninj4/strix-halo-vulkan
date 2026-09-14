# testdata

Fixtures shared by the speech verticals (`SPEECH.md`).

## `jfk.wav`

11.000 s of 16 kHz mono 16-bit PCM — 176000 samples, 352 KB — from John F.
Kennedy's 1961 inaugural address. A work of the United States federal
government, so public domain.

Fetched from whisper.cpp's sample set, which is where it is a de facto ASR
fixture:

    curl -L -o testdata/jfk.wav \
      https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav

Its transcript, as parakeet-tdt-0.6b-v3 punctuates it:

    And so, my fellow Americans, ask not what your country can do for you,
    ask what you can do for your country.

That string is the correctness bound for the whole STT vertical — it is
`transcript` in `parakeet/reference_test.go` and `EXPECTED` in
`reference/dump_parakeet.py`, and both the Go model and the transformers
reference have to produce it exactly.

The decode of the file itself is pinned separately, by a sha256 of the
float32 samples in `audio/wav_test.go` against an independent read by
CPython's `wave` module. Every downstream comparison starts from those
samples, so a change to the WAV decoder has to fail there rather than show up
as a model regression.
