package wyoming

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"log"
	"math"
	"net"
	"os"
	"strings"
	"testing"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/audio"
)

// TestMain silences the server's own logging. Every test here drives a whole
// conversation and the package logs a line per connection and per request, so
// leaving it on would bury the failures in transcripts of the passes.
func TestMain(m *testing.M) {
	log.SetOutput(io.Discard)
	code := m.Run()
	log.SetOutput(os.Stderr)
	os.Exit(code)
}

// The fakes. They are the same shape as the ones in api/: an interface this
// package consumes, implemented by something that records what it was asked
// and answers from a field, so that the tests are about the protocol and not
// about kokoro.

type fakeSpeech struct {
	voices []string
	rate   int
	clip   []float32
	err    error

	last *api.SpeechRequest
}

func (f *fakeSpeech) Models() []api.Model {
	return []api.Model{{ID: "kokoro-82m", Object: "model", OwnedBy: "local"}}
}

func (f *fakeSpeech) Voices() []string { return append([]string(nil), f.voices...) }

func (f *fakeSpeech) Speak(_ context.Context, req *api.SpeechRequest) (*audio.Clip, error) {
	f.last = req
	if f.err != nil {
		return nil, f.err
	}
	return &audio.Clip{Samples: f.clip, Rate: f.rate}, nil
}

type fakeTranscription struct {
	text string
	rate int
	err  error

	last    *audio.Clip
	lastReq *api.TranscriptionRequest
	calls   int
}

func (f *fakeTranscription) Models() []api.Model {
	return []api.Model{{ID: "parakeet-tdt-0.6b-v3", Object: "model", OwnedBy: "local"}}
}

func (f *fakeTranscription) SampleRate() int { return f.rate }

func (f *fakeTranscription) Transcribe(_ context.Context, clip *audio.Clip,
	req *api.TranscriptionRequest,
) (*api.TranscriptionResponse, error) {
	f.calls++
	f.last, f.lastReq = clip, req
	if f.err != nil {
		return nil, f.err
	}
	return &api.TranscriptionResponse{Text: f.text}, nil
}

// harness is one client talking to one server goroutine over an in-memory
// connection. net.Pipe rather than a loopback socket because it is
// synchronous: a write that the server never reads is a deadlock the test
// fails on rather than a flake it passes through.
type harness struct {
	t    *testing.T
	srv  *Server
	conn net.Conn
	r    *bufio.Reader
	w    *bufio.Writer
	done chan struct{}
}

func newHarness(t *testing.T, opt Options) *harness {
	t.Helper()
	srv, err := New(opt)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	client, server := net.Pipe()
	h := &harness{
		t: t, srv: srv, conn: client,
		r: bufio.NewReader(client), w: bufio.NewWriter(client),
		done: make(chan struct{}),
	}
	go func() {
		defer close(h.done)
		srv.handle(context.Background(), server)
	}()
	t.Cleanup(func() {
		_ = client.Close()
		select {
		case <-h.done:
		case <-time.After(5 * time.Second):
			t.Error("the connection goroutine did not finish")
		}
	})
	return h
}

func (h *harness) send(typ string, data any) {
	h.t.Helper()
	ev, err := event(typ, data)
	if err != nil {
		h.t.Fatalf("event %s: %v", typ, err)
	}
	h.sendEvent(ev)
}

func (h *harness) sendEvent(ev *Event) {
	h.t.Helper()
	_ = h.conn.SetWriteDeadline(time.Now().Add(5 * time.Second))
	if err := WriteEvent(h.w, ev); err != nil {
		h.t.Fatalf("writing %s: %v", ev.Type, err)
	}
}

func (h *harness) recv() *Event {
	h.t.Helper()
	_ = h.conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	ev, err := ReadEvent(h.r)
	if err != nil {
		h.t.Fatalf("reading: %v", err)
	}
	return ev
}

// expect reads one event and fails unless it is of this type, naming what
// arrived instead -- including the text of an `error`, which is the failure
// this saves the most time on.
func (h *harness) expect(typ string) *Event {
	h.t.Helper()
	ev := h.recv()
	if ev.Type == typ {
		return ev
	}
	if ev.Type == typeError {
		var e errorData
		_ = ev.Unmarshal(&e)
		h.t.Fatalf("wanted %s, got an error: %s", typ, e.Text)
	}
	h.t.Fatalf("wanted %s, got %s", typ, ev.Type)
	return nil
}

func (h *harness) expectError() string {
	h.t.Helper()
	ev := h.expect(typeError)
	var e errorData
	if err := ev.Unmarshal(&e); err != nil {
		h.t.Fatalf("decoding the error: %v", err)
	}
	return e.Text
}

func speechBackend() *fakeSpeech {
	return &fakeSpeech{
		voices: kokoroVoices,
		rate:   24000,
		clip:   ramp(3000),
	}
}

// ramp is a signal whose every sample is distinct, so that a chunking or
// ordering mistake in the audio stream shows up as a wrong value rather than
// as silence that happens to match.
func ramp(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i) * 0.01))
	}
	return out
}

func TestNewNeedsABackend(t *testing.T) {
	if _, err := New(Options{}); err == nil || !strings.Contains(err.Error(), "-tts") {
		t.Errorf("err = %v, want one naming the flags", err)
	}
}

func TestNewRefusesAnUnpronounceableVoiceList(t *testing.T) {
	// Every pack Japanese, no English front end to read it with: advertising
	// them would be advertising a bug, and an empty voice list is a service
	// Home Assistant silently will not use, so this is a startup failure.
	_, err := New(Options{Speech: &fakeSpeech{voices: []string{"jf_alpha"}, rate: 24000}})
	if err == nil || !strings.Contains(err.Error(), "-wyoming-all-voices") {
		t.Errorf("err = %v, want one naming the override", err)
	}
	if _, err := New(Options{
		Speech:    &fakeSpeech{voices: []string{"jf_alpha"}, rate: 24000},
		AllVoices: true,
	}); err != nil {
		t.Errorf("with -wyoming-all-voices: %v", err)
	}
}

// TestDescribe checks the `info` message against what the reference client
// requires of it, which is stricter than what this package's own types say: a
// missing attribution or a null where a list belongs is a stack trace on the
// client, and the client is Home Assistant's config flow.
func TestDescribe(t *testing.T) {
	h := newHarness(t, Options{
		Speech: speechBackend(), Transcription: &fakeTranscription{rate: 16000},
		Name: "strix-halo",
	})
	h.send(typeDescribe, nil)
	ev := h.expect(typeInfo)

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(ev.Data, &raw); err != nil {
		t.Fatalf("info is not an object: %v", err)
	}
	for _, key := range []string{"asr", "tts", "handle", "intent", "wake", "mic", "snd"} {
		val, ok := raw[key]
		if !ok {
			t.Errorf("%q is missing", key)
			continue
		}
		if string(val) == "null" {
			t.Errorf("%q is null; the reference client iterates it", key)
		}
	}

	var info Info
	if err := ev.Unmarshal(&info); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(info.ASR) != 1 || len(info.TTS) != 1 {
		t.Fatalf("asr = %d, tts = %d, want one each", len(info.ASR), len(info.TTS))
	}
	asr := info.ASR[0]
	if asr.Name != "strix-halo" || !asr.Installed {
		t.Errorf("asr program = %q installed=%v", asr.Name, asr.Installed)
	}
	if !asr.RequiresExternalVAD {
		t.Error("requires_external_vad is false; nothing here decides where a command ended")
	}
	if asr.SupportsTranscriptStreaming {
		t.Error("supports_transcript_streaming is true and no transcript-chunk is ever sent")
	}
	if len(asr.Models) != 1 || asr.Models[0].Name != "parakeet-tdt-0.6b-v3" {
		t.Errorf("asr models = %+v", asr.Models)
	}
	if len(asr.Models[0].Languages) == 0 {
		t.Error("the asr model lists no languages; Home Assistant offers it for none")
	}
	tts := info.TTS[0]
	if tts.SupportsSynthesizeStreaming {
		t.Error("supports_synthesize_streaming is true and no synthesize-chunk is handled")
	}
	if len(tts.Voices) != 28 {
		t.Errorf("%d voices, want the 28 English packs", len(tts.Voices))
	}
	for _, art := range []Artifact{asr.Artifact, tts.Artifact, asr.Models[0].Artifact} {
		if art.Attribution.Name == "" || art.Attribution.URL == "" {
			t.Errorf("%q has no attribution, which the reference client will not decode", art.Name)
		}
	}

	// Describe is answered the same way every time: Home Assistant polls it.
	h.send(typeDescribe, nil)
	again := h.expect(typeInfo)
	if string(again.Data) != string(ev.Data) {
		t.Error("two describes, two answers")
	}
}

func TestDescribeWithOneBackend(t *testing.T) {
	// A process started with -tts and not -stt advertises a text-to-speech
	// service and no speech-to-text one, rather than an empty one.
	h := newHarness(t, Options{Speech: speechBackend()})
	h.send(typeDescribe, nil)
	var info Info
	if err := h.expect(typeInfo).Unmarshal(&info); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(info.ASR) != 0 || len(info.TTS) != 1 {
		t.Errorf("asr = %d, tts = %d, want 0 and 1", len(info.ASR), len(info.TTS))
	}
	// And it says so when asked to do the thing it cannot.
	h.send(typeTranscribe, transcribeData{Language: "en"})
	if text := h.expectError(); !strings.Contains(text, "-stt") {
		t.Errorf("error = %q, want one naming the flag", text)
	}
}

func TestPing(t *testing.T) {
	h := newHarness(t, Options{Speech: speechBackend()})
	text := "are you there"
	h.send(typePing, pingData{Text: &text})
	var p pingData
	if err := h.expect(typePong).Unmarshal(&p); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if p.Text == nil || *p.Text != text {
		t.Errorf("pong text = %v, want %q copied back", p.Text, text)
	}
}

func TestUnknownEventsAreDropped(t *testing.T) {
	// The protocol's compatibility rule: a server ignores what it does not
	// know, so a newer client keeps working against this one. The proof is
	// that the connection survives and still answers.
	h := newHarness(t, Options{Speech: speechBackend()})
	for _, typ := range []string{"detect", "run-pipeline", "synthesize-start", "synthesize-stop"} {
		h.send(typ, map[string]any{"whatever": 1})
	}
	h.send(typePing, pingData{})
	h.expect(typePong)
}

func TestSynthesize(t *testing.T) {
	speech := speechBackend()
	h := newHarness(t, Options{Speech: speech, Voice: "af_heart"})
	h.send(typeSynthesize, synthesizeData{
		Text:  "The quick brown fox.",
		Voice: &synthesizeVoice{Name: "bm_george"},
	})

	var start audioFormat
	if err := h.expect(typeAudioStart).Unmarshal(&start); err != nil {
		t.Fatalf("audio-start: %v", err)
	}
	if start.Rate != 24000 || start.Width != 2 || start.Channels != 1 {
		t.Errorf("audio-start = %+v, want 24000/2/1", start)
	}

	var got []float32
	var chunks int
	for {
		ev := h.recv()
		if ev.Type == typeAudioStop {
			break
		}
		if ev.Type != typeAudioChunk {
			t.Fatalf("wanted audio-chunk or audio-stop, got %s", ev.Type)
		}
		chunks++
		var f audioFormat
		if err := ev.Unmarshal(&f); err != nil {
			t.Fatalf("audio-chunk: %v", err)
		}
		if f.Rate != start.Rate || f.Width != start.Width || f.Channels != start.Channels {
			t.Fatalf("chunk %d format %+v, want %+v", chunks, f, start)
		}
		// Every chunk repeats the format, and the timestamp is where in
		// the stream it belongs -- a satellite plays from these.
		if f.Timestamp == nil {
			t.Fatalf("chunk %d has no timestamp", chunks)
		}
		if want := int(math.Round(float64(len(got)) * 1000 / float64(start.Rate))); *f.Timestamp != want {
			t.Errorf("chunk %d timestamp = %d ms, want %d", chunks, *f.Timestamp, want)
		}
		got = append(got, decodePCM16(ev.Payload)...)
	}
	if chunks == 0 {
		t.Fatal("no audio")
	}

	if speech.last.Voice != "bm_george" {
		t.Errorf("voice = %q, want the request's", speech.last.Voice)
	}
	if speech.last.Input != "The quick brown fox." {
		t.Errorf("input = %q", speech.last.Input)
	}
	if speech.last.Speed != 1 {
		t.Errorf("speed = %v, want 1: the protocol has no way to ask for another", speech.last.Speed)
	}
	if len(got) != len(speech.clip) {
		t.Fatalf("%d samples out, %d in", len(got), len(speech.clip))
	}
	// 16-bit PCM round trip: within half a step of full scale.
	for i := range got {
		if math.Abs(float64(got[i]-speech.clip[i])) > 1.0/32767 {
			t.Fatalf("sample %d = %v, want %v", i, got[i], speech.clip[i])
		}
	}
}

func TestSynthesizeDefaults(t *testing.T) {
	for _, tt := range []struct {
		name  string
		voice *synthesizeVoice
		want  string
	}{
		{"no voice at all", nil, "af_heart"},
		{"an empty voice object", &synthesizeVoice{}, "af_heart"},
		{"a language, exactly", &synthesizeVoice{Language: "en-GB"}, "bf_alice"},
		{"a language, primary only", &synthesizeVoice{Language: "en"}, "af_alloy"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			speech := speechBackend()
			h := newHarness(t, Options{Speech: speech, Voice: "af_heart"})
			h.send(typeSynthesize, synthesizeData{Text: "hello", Voice: tt.voice})
			h.expect(typeAudioStart)
			drainAudio(t, h)
			if speech.last.Voice != tt.want {
				t.Errorf("voice = %q, want %q", speech.last.Voice, tt.want)
			}
		})
	}
}

func TestSynthesizeRefusals(t *testing.T) {
	for _, tt := range []struct {
		name string
		req  synthesizeData
		want string
	}{{
		"no text", synthesizeData{Text: "   "}, "no text",
	}, {
		"ssml", synthesizeData{Text: "hi", TextFormat: "ssml"}, "plain text",
	}, {
		// A kokoro pack is one speaker; a blend is spelled in the name.
		"a speaker", synthesizeData{Text: "hi", Voice: &synthesizeVoice{Name: "af_heart", Speaker: "2"}},
		"one speaker",
	}, {
		"a language nobody speaks", synthesizeData{Text: "hi", Voice: &synthesizeVoice{Language: "cy"}},
		"no advertised voice",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			h := newHarness(t, Options{Speech: speechBackend(), Voice: "af_heart"})
			h.send(typeSynthesize, tt.req)
			if text := h.expectError(); !strings.Contains(text, tt.want) {
				t.Errorf("error = %q, want one mentioning %q", text, tt.want)
			}
			// The connection survives a refusal: a client that asked for
			// the wrong voice should be able to ask again.
			h.send(typePing, pingData{})
			h.expect(typePong)
		})
	}
}

// TestSynthesizeBackendErrorReachesTheClient is the whole reason failures go
// back as `error` events instead of a hang-up: Home Assistant reports a
// dropped connection as "connection lost" and reports this with its text.
func TestSynthesizeBackendError(t *testing.T) {
	speech := speechBackend()
	speech.err = errors.New("no voice \"bm_geroge\"; this checkpoint has 54")
	h := newHarness(t, Options{Speech: speech})
	h.send(typeSynthesize, synthesizeData{Text: "hi", Voice: &synthesizeVoice{Name: "bm_geroge"}})
	if text := h.expectError(); !strings.Contains(text, "bm_geroge") {
		t.Errorf("error = %q, want the backend's own text", text)
	}
}

func TestSynthesizeOutputRate(t *testing.T) {
	speech := speechBackend()
	h := newHarness(t, Options{Speech: speech, OutputRate: 16000})
	h.send(typeSynthesize, synthesizeData{Text: "hi"})
	var start audioFormat
	if err := h.expect(typeAudioStart).Unmarshal(&start); err != nil {
		t.Fatalf("audio-start: %v", err)
	}
	if start.Rate != 16000 {
		t.Errorf("rate = %d, want the resampled 16000", start.Rate)
	}
	n := drainAudio(t, h)
	// 3000 samples at 24 kHz is 2000 at 16 kHz.
	if n != 2000 {
		t.Errorf("%d samples, want 2000", n)
	}
}

// TestTranscribe is Home Assistant's exact speech-to-text conversation:
// transcribe, audio-start, chunks at 16 kHz mono 16-bit, audio-stop.
func TestTranscribe(t *testing.T) {
	stt := &fakeTranscription{text: "and so my fellow Americans", rate: 16000}
	h := newHarness(t, Options{Transcription: stt, MaxSeconds: 60})

	samples := ramp(8000) // half a second
	h.send(typeTranscribe, transcribeData{Language: "en"})
	h.send(typeAudioStart, audioFormat{Rate: 16000, Width: 2, Channels: 1, Timestamp: intptr(0)})
	for off := 0; off < len(samples); off += 1024 {
		end := min(off+1024, len(samples))
		ev := mustEvent(t, typeAudioChunk, audioFormat{Rate: 16000, Width: 2, Channels: 1})
		ev.Payload = encodePCM16(samples[off:end])
		h.sendEvent(ev)
	}
	h.send(typeAudioStop, audioStopData{Timestamp: intptr(500)})

	reply := h.expect(typeTranscript)
	var out transcriptData
	if err := reply.Unmarshal(&out); err != nil {
		t.Fatalf("transcript: %v", err)
	}
	if out.Text != stt.text {
		t.Errorf("text = %q, want %q", out.Text, stt.text)
	}
	if stt.last.Rate != 16000 {
		t.Errorf("the backend got %d Hz", stt.last.Rate)
	}
	if len(stt.last.Samples) != len(samples) {
		t.Fatalf("%d samples reached the backend, %d were sent", len(stt.last.Samples), len(samples))
	}
	// Within a step of 16-bit, plus the scale asymmetry the repository uses
	// on purpose: everything here encodes at 32767 and clamps, so a full
	// scale +1.0 stays at the top of the range, and decodes at 32768, which
	// is what soundfile and librosa do and what the reference dumps were
	// taken with. The two are not exact inverses, by a part in 32768.
	for i := range samples {
		if math.Abs(float64(stt.last.Samples[i]-samples[i])) > 1e-4 {
			t.Fatalf("sample %d = %v, want %v", i, stt.last.Samples[i], samples[i])
		}
	}
	// The language hint is carried but not turned into a result: a TDT
	// transducer reports no language, so the transcript has no field for it.
	if stt.lastReq.Language != "en" {
		t.Errorf("language = %q, want the request's", stt.lastReq.Language)
	}
	if strings.Contains(string(reply.Data), "language") {
		t.Error("the transcript claims a language the model did not detect")
	}
}

func TestTranscribeResamples(t *testing.T) {
	// A satellite whose microphone is 48 kHz. The HTTP endpoint refuses a
	// clip at the wrong rate because its caller chose the file; here the
	// caller is a microphone, so this side filters.
	stt := &fakeTranscription{text: "hello", rate: 16000}
	h := newHarness(t, Options{Transcription: stt})
	samples := ramp(48000)
	h.send(typeAudioStart, audioFormat{Rate: 48000, Width: 2, Channels: 1})
	ev := mustEvent(t, typeAudioChunk, audioFormat{Rate: 48000, Width: 2, Channels: 1})
	ev.Payload = encodePCM16(samples)
	h.sendEvent(ev)
	h.send(typeAudioStop, nil)
	h.expect(typeTranscript)

	if stt.last.Rate != 16000 {
		t.Errorf("the backend got %d Hz, want 16000", stt.last.Rate)
	}
	if got, want := len(stt.last.Samples), 16000; got != want {
		t.Errorf("%d samples, want %d -- one second either way", got, want)
	}
}

func TestTranscribeWidthsAndChannels(t *testing.T) {
	// The three sample widths the protocol carries, and a stereo stream,
	// each decoded to the same mono float32 the WAV reader produces.
	for _, tt := range []struct {
		name            string
		width, channels int
		payload         []byte
		want            []float32
	}{
		{"8-bit", 1, 1, []byte{128, 255, 0, 192}, []float32{0, 127.0 / 128, -1, 0.5}},
		{"16-bit", 2, 1, pcm16(0, 16384, -32768), []float32{0, 0.5, -1}},
		{"32-bit", 4, 1, pcm32(0, 1<<30, -1<<31), []float32{0, 0.5, -1}},
		{"stereo", 2, 2, pcm16(16384, -16384, 32767, 32767), []float32{0, 32767.0 / 32768}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			stt := &fakeTranscription{text: "x", rate: 16000}
			h := newHarness(t, Options{Transcription: stt})
			ev := mustEvent(t, typeAudioChunk, audioFormat{
				Rate: 16000, Width: tt.width, Channels: tt.channels,
			})
			ev.Payload = tt.payload
			h.sendEvent(ev)
			h.send(typeAudioStop, nil)
			h.expect(typeTranscript)
			got := stt.last.Samples
			if len(got) != len(tt.want) {
				t.Fatalf("%d samples, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if math.Abs(float64(got[i]-tt.want[i])) > 1e-6 {
					t.Errorf("sample %d = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestTranscribeRefusesAFormatChange(t *testing.T) {
	// Concatenating two rates would give a clip whose timeline is wrong in a
	// way the transcript would not show, and no client does it by accident.
	stt := &fakeTranscription{text: "x", rate: 16000}
	h := newHarness(t, Options{Transcription: stt})
	h.send(typeAudioStart, audioFormat{Rate: 16000, Width: 2, Channels: 1})
	first := mustEvent(t, typeAudioChunk, audioFormat{Rate: 16000, Width: 2, Channels: 1})
	first.Payload = pcm16(1, 2, 3, 4)
	h.sendEvent(first)
	second := mustEvent(t, typeAudioChunk, audioFormat{Rate: 48000, Width: 2, Channels: 1})
	second.Payload = pcm16(1, 2, 3, 4)
	h.sendEvent(second)

	if text := h.expectError(); !strings.Contains(text, "48000") || !strings.Contains(text, "16000") {
		t.Errorf("error = %q, want one naming both rates", text)
	}
	// The half-received utterance was dropped rather than transcribed.
	h.send(typeAudioStop, nil)
	h.send(typePing, pingData{})
	h.expect(typePong)
	if stt.calls != 0 {
		t.Errorf("the backend ran %d times on a stream that was refused", stt.calls)
	}
}

func TestTranscribeOverflow(t *testing.T) {
	// Past the ceiling the samples are dropped rather than held: this side
	// buffers the whole utterance in memory, and the encoder's arenas would
	// refuse it anyway.
	stt := &fakeTranscription{text: "x", rate: 16000}
	h := newHarness(t, Options{Transcription: stt, MaxSeconds: 0.25})
	h.send(typeAudioStart, audioFormat{Rate: 16000, Width: 2, Channels: 1})
	for range 4 {
		ev := mustEvent(t, typeAudioChunk, audioFormat{Rate: 16000, Width: 2, Channels: 1})
		ev.Payload = encodePCM16(ramp(2000)) // 125 ms each
		h.sendEvent(ev)
	}
	h.send(typeAudioStop, nil)
	if text := h.expectError(); !strings.Contains(text, "-max-audio") {
		t.Errorf("error = %q, want one naming the flag", text)
	}
	if stt.calls != 0 {
		t.Errorf("the backend ran %d times on an utterance that was refused", stt.calls)
	}
	// And the next utterance on the same connection is fine.
	ev := mustEvent(t, typeAudioChunk, audioFormat{Rate: 16000, Width: 2, Channels: 1})
	ev.Payload = encodePCM16(ramp(1000))
	h.sendEvent(ev)
	h.send(typeAudioStop, nil)
	h.expect(typeTranscript)
}

func TestTranscribeSilence(t *testing.T) {
	// A pipeline that heard nothing gets an empty transcript, not an error:
	// Home Assistant treats an error as a broken engine and an empty string
	// as "nothing was said".
	stt := &fakeTranscription{text: "x", rate: 16000}
	h := newHarness(t, Options{Transcription: stt})
	h.send(typeAudioStart, audioFormat{Rate: 16000, Width: 2, Channels: 1})
	h.send(typeAudioStop, nil)
	var out transcriptData
	if err := h.expect(typeTranscript).Unmarshal(&out); err != nil {
		t.Fatalf("transcript: %v", err)
	}
	if out.Text != "" {
		t.Errorf("text = %q, want empty", out.Text)
	}
	if stt.calls != 0 {
		t.Errorf("the backend ran %d times on silence", stt.calls)
	}
}

func TestTranscribeBadFormat(t *testing.T) {
	h := newHarness(t, Options{Transcription: &fakeTranscription{rate: 16000}})
	h.send(typeAudioStart, audioFormat{Rate: 16000, Width: 3, Channels: 1})
	if text := h.expectError(); !strings.Contains(text, "3-byte") {
		t.Errorf("error = %q, want one naming the width", text)
	}
	// A chunk that is not a whole number of frames.
	ev := mustEvent(t, typeAudioChunk, audioFormat{Rate: 16000, Width: 2, Channels: 1})
	ev.Payload = []byte{1, 2, 3}
	h.sendEvent(ev)
	if text := h.expectError(); !strings.Contains(text, "frames") {
		t.Errorf("error = %q, want one about the frame size", text)
	}
}

func TestSelectProgram(t *testing.T) {
	h := newHarness(t, Options{Speech: speechBackend(), Name: "strix-halo"})
	h.send(typeSelectProgram, selectProgramData{Name: "strix-halo"})
	h.send(typePing, pingData{})
	h.expect(typePong)
	h.send(typeSelectProgram, selectProgramData{Name: "piper"})
	if text := h.expectError(); !strings.Contains(text, "piper") {
		t.Errorf("error = %q, want one naming the program", text)
	}
}

// TestServeAndDrain is the whole listener: a real socket, a real client, and
// a cancellation that has to close it without cutting an answer in half.
func TestServeAndDrain(t *testing.T) {
	srv, err := New(Options{Speech: speechBackend(), Transcription: &fakeTranscription{rate: 16000}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- srv.Serve(ctx, ln) }()

	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	r, w := bufio.NewReader(conn), bufio.NewWriter(conn)
	ev := mustEvent(t, typeDescribe, nil)
	if err := WriteEvent(w, ev); err != nil {
		t.Fatalf("describe: %v", err)
	}
	got, err := ReadEvent(r)
	if err != nil {
		t.Fatalf("info: %v", err)
	}
	if got.Type != typeInfo {
		t.Fatalf("got %s, want info", got.Type)
	}

	// An idle connection unwinds on cancellation rather than holding the
	// drain open, and Serve returns only once it has.
	cancel()
	select {
	case err := <-served:
		if err != nil {
			t.Errorf("Serve: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Serve did not return")
	}
	_ = conn.Close()
	if _, err := net.DialTimeout("tcp", ln.Addr().String(), time.Second); err == nil {
		t.Error("the listener is still accepting after Serve returned")
	}
}

// drainAudio reads chunks until audio-stop and reports how many samples came
// back.
func drainAudio(t *testing.T, h *harness) int {
	t.Helper()
	n := 0
	for {
		ev := h.recv()
		switch ev.Type {
		case typeAudioStop:
			return n
		case typeAudioChunk:
			n += len(ev.Payload) / 2
		default:
			t.Fatalf("wanted audio-chunk or audio-stop, got %s", ev.Type)
		}
	}
}

func decodePCM16(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = float32(int16(binary.LittleEndian.Uint16(b[2*i:]))) / 32767
	}
	return out
}

func pcm16(vals ...int16) []byte {
	out := make([]byte, 0, 2*len(vals))
	for _, v := range vals {
		out = binary.LittleEndian.AppendUint16(out, uint16(v))
	}
	return out
}

func pcm32(vals ...int32) []byte {
	out := make([]byte, 0, 4*len(vals))
	for _, v := range vals {
		out = binary.LittleEndian.AppendUint32(out, uint32(v))
	}
	return out
}
