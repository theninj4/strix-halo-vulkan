package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"strix-halo-vulkan/audio"
)

// The point of the backend interfaces is that the routing layer can be tested
// without a checkpoint, a device or a second of staging. These two fakes are
// what that buys: every test below runs in microseconds.

type fakeSpeech struct {
	req  *SpeechRequest
	err  error
	clip *audio.Clip
}

func (f *fakeSpeech) Models() []Model {
	return []Model{{ID: "kokoro-82m", Object: "model", OwnedBy: "local"}}
}
func (f *fakeSpeech) Voices() []string { return []string{"bm_george", "af_heart"} }
func (f *fakeSpeech) Speak(_ context.Context, req *SpeechRequest) (*audio.Clip, error) {
	f.req = req
	if f.err != nil {
		return nil, f.err
	}
	if f.clip != nil {
		return f.clip, nil
	}
	return &audio.Clip{Rate: 24000, Samples: []float32{0, 0.5, -0.5, 1}}, nil
}

type fakeSTT struct {
	clip *audio.Clip
	req  *TranscriptionRequest
	resp *TranscriptionResponse
	err  error
}

func (f *fakeSTT) Models() []Model {
	return []Model{{ID: "parakeet-tdt-0.6b-v3", Object: "model", OwnedBy: "local"}}
}
func (f *fakeSTT) Transcribe(_ context.Context, clip *audio.Clip,
	req *TranscriptionRequest,
) (*TranscriptionResponse, error) {
	f.clip, f.req = clip, req
	if f.err != nil {
		return nil, f.err
	}
	if f.resp != nil {
		return f.resp, nil
	}
	return &TranscriptionResponse{Text: "the quick brown fox"}, nil
}

// do runs one request against a server's whole middleware stack, which is
// what makes these tests about the server and not about a handler.
func do(t *testing.T, s *Server, req *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	if s.Token != "" {
		req.Header.Set("Authorization", "Bearer "+s.Token)
	}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	return rec
}

func jsonRequest(method, path string, body any) *http.Request {
	b, err := json.Marshal(body)
	if err != nil {
		panic(err)
	}
	req := httptest.NewRequest(method, path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func TestModelsListsWhatIsLoaded(t *testing.T) {
	s := &Server{Speech: &fakeSpeech{}, Transcription: &fakeSTT{}}
	rec := do(t, s, httptest.NewRequest("GET", "/v1/models", http.NoBody))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var got ModelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != 2 {
		t.Fatalf("%d models, want 2: %+v", len(got.Data), got.Data)
	}
	// The voices belong to the speech model and to nothing else.
	for _, m := range got.Data {
		switch m.ID {
		case "kokoro-82m":
			if len(m.Voices) != 2 || m.Voices[0] != "af_heart" {
				t.Errorf("voices %v, want them sorted", m.Voices)
			}
		case "parakeet-tdt-0.6b-v3":
			if len(m.Voices) != 0 {
				t.Errorf("the transcription model has voices: %v", m.Voices)
			}
		default:
			t.Errorf("unexpected model %q", m.ID)
		}
	}
}

func TestModelsEmptyWithoutBackends(t *testing.T) {
	rec := do(t, &Server{}, httptest.NewRequest("GET", "/v1/models", http.NoBody))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	// An empty list, not null: a client iterating the array should not have
	// to special-case a server with nothing loaded.
	if got := strings.TrimSpace(rec.Body.String()); !strings.Contains(got, `"data":[]`) {
		t.Errorf("body %s, want an empty data array", got)
	}
}

func TestSpeechWAV(t *testing.T) {
	fake := &fakeSpeech{}
	s := &Server{Token: "t", Speech: fake}
	rec := do(t, s, jsonRequest("POST", "/v1/audio/speech", SpeechRequest{
		Input: "hello", Voice: "bm_george", Speed: 1.5,
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "audio/wav" {
		t.Errorf("content type %q", ct)
	}
	if got := rec.Body.Bytes(); len(got) < 44 || string(got[:4]) != "RIFF" {
		t.Errorf("body is not a WAV: %d bytes", len(got))
	}
	if rec.Header().Get("X-Sample-Rate") != "24000" {
		t.Errorf("sample rate header %q", rec.Header().Get("X-Sample-Rate"))
	}
	if fake.req.Voice != "bm_george" || fake.req.Speed != 1.5 {
		t.Errorf("the backend saw %+v", fake.req)
	}
}

// The PCM body is the WAV body without its 44-byte header, sample for
// sample: the two encoders have to agree, because a caller switching format
// is not asking for different audio.
func TestSpeechPCMMatchesWAVPayload(t *testing.T) {
	s := &Server{Speech: &fakeSpeech{}}
	wav := do(t, s, jsonRequest("POST", "/v1/audio/speech", SpeechRequest{Input: "hi"}))
	pcm := do(t, s, jsonRequest("POST", "/v1/audio/speech", SpeechRequest{Input: "hi", ResponseFormat: "pcm"}))
	if pcm.Code != http.StatusOK {
		t.Fatalf("status %d: %s", pcm.Code, pcm.Body)
	}
	if ct := pcm.Header().Get("Content-Type"); ct != "audio/pcm" {
		t.Errorf("content type %q", ct)
	}
	if got, want := pcm.Body.Bytes(), wav.Body.Bytes()[44:]; !bytes.Equal(got, want) {
		t.Errorf("pcm % x, wav payload % x", got, want)
	}
}

// Full scale must stay at the top of the range rather than wrapping to the
// most negative sample, which is audio.EncodeWAV's rule and has to hold for
// the raw path too.
func TestSpeechPCMClampsFullScale(t *testing.T) {
	s := &Server{Speech: &fakeSpeech{clip: &audio.Clip{
		Rate: 24000, Samples: []float32{1, -1, 2, float32(math.Inf(-1))},
	}}}
	rec := do(t, s, jsonRequest("POST", "/v1/audio/speech", SpeechRequest{Input: "hi", ResponseFormat: "pcm"}))
	got := rec.Body.Bytes()
	want := []byte{0xff, 0x7f, 0x01, 0x80, 0xff, 0x7f, 0x00, 0x80}
	if !bytes.Equal(got, want) {
		t.Errorf("% x, want % x", got, want)
	}
}

func TestSpeechRejects(t *testing.T) {
	cases := []struct {
		name   string
		req    SpeechRequest
		status int
		body   string
	}{
		{"no input", SpeechRequest{}, http.StatusBadRequest, "input is empty"},
		{"streaming", SpeechRequest{Input: "x", Stream: true}, http.StatusBadRequest, "streaming"},
		{"speed", SpeechRequest{Input: "x", Speed: 9}, http.StatusBadRequest, "speed"},
		{"format", SpeechRequest{Input: "x", ResponseFormat: "mp3"}, http.StatusBadRequest, "mp3"},
	}
	s := &Server{Speech: &fakeSpeech{}}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, s, jsonRequest("POST", "/v1/audio/speech", tc.req))
			if rec.Code != tc.status {
				t.Fatalf("status %d, want %d: %s", rec.Code, tc.status, rec.Body)
			}
			if !strings.Contains(rec.Body.String(), tc.body) {
				t.Errorf("body %s, want it to mention %q", rec.Body, tc.body)
			}
		})
	}
}

// Phonemes win over input, so a caller with its own front end can reach the
// model's vocabulary directly (SPEECH.md T5).
func TestSpeechPassesPhonemes(t *testing.T) {
	fake := &fakeSpeech{}
	s := &Server{Speech: fake}
	do(t, s, jsonRequest("POST", "/v1/audio/speech", SpeechRequest{
		Input: "hello there", Phonemes: "hɛlˈO wˈɜɹld.",
	}))
	if fake.req.Phonemes != "hɛlˈO wˈɜɹld." {
		t.Errorf("the backend saw phonemes %q", fake.req.Phonemes)
	}
}

// An unsupported voice is the client's mistake, not the server's: the
// backend says so with ErrUnsupported and it comes back as a 400.
func TestSpeechUnsupportedIsBadRequest(t *testing.T) {
	s := &Server{Speech: &fakeSpeech{err: fmt.Errorf("no voice %q: %w", "nope", ErrUnsupported)}}
	rec := do(t, s, jsonRequest("POST", "/v1/audio/speech", SpeechRequest{Input: "x"}))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}

func TestSpeechNotLoaded(t *testing.T) {
	rec := do(t, &Server{}, jsonRequest("POST", "/v1/audio/speech", SpeechRequest{Input: "x"}))
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "-tts") {
		t.Errorf("body %s, want it to name the flag", rec.Body)
	}
}

// testWAV is a short clip in the only container the server decodes.
func testWAV(t *testing.T, rate int, n int) []byte {
	t.Helper()
	samples := make([]float32, n)
	for i := range samples {
		samples[i] = float32(math.Sin(float64(i) * 0.1))
	}
	b, err := (&audio.Clip{Rate: rate, Samples: samples}).EncodeWAV()
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestTranscriptionMultipart(t *testing.T) {
	wav := testWAV(t, 16000, 800)
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	part, err := mw.CreateFormFile("file", "clip.wav")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(wav); err != nil {
		t.Fatal(err)
	}
	for k, v := range map[string]string{
		"model": "parakeet-tdt-0.6b-v3", "language": "en", "temperature": "0.25",
	} {
		if err := mw.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}

	fake := &fakeSTT{}
	s := &Server{Transcription: fake}
	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := do(t, s, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var got TranscriptionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Text != "the quick brown fox" {
		t.Errorf("text %q", got.Text)
	}
	// The backend is handed a decoded clip, never a container.
	if fake.clip.Rate != 16000 || len(fake.clip.Samples) != 800 {
		t.Errorf("clip %d Hz, %d samples", fake.clip.Rate, len(fake.clip.Samples))
	}
	if fake.req.Language != "en" || fake.req.Temperature != 0.25 {
		t.Errorf("the backend saw %+v", fake.req)
	}
}

// The JSON encoding of the same request: "file" as base64, which is what a
// hand-written curl can send.
func TestTranscriptionJSONBody(t *testing.T) {
	fake := &fakeSTT{}
	s := &Server{Transcription: fake}
	rec := do(t, s, jsonRequest("POST", "/v1/audio/transcriptions", TranscriptionRequest{
		Data: testWAV(t, 16000, 800), ResponseFormat: "text",
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("content type %q", ct)
	}
	if rec.Body.String() != "the quick brown fox" {
		t.Errorf("body %q", rec.Body)
	}
}

func TestTranscriptionRejects(t *testing.T) {
	s := &Server{Transcription: &fakeSTT{}}
	cases := []struct {
		name string
		req  TranscriptionRequest
		body string
	}{
		{"no audio", TranscriptionRequest{}, "no audio"},
		{"not a wav", TranscriptionRequest{Data: []byte("not a riff header at all")}, "16-bit PCM WAV"},
		{"streaming", TranscriptionRequest{Data: testWAV(t, 16000, 800), Stream: true}, "streaming"},
		{
			"format",
			TranscriptionRequest{Data: testWAV(t, 16000, 800), ResponseFormat: "xml"},
			"verbose_json",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, s, jsonRequest("POST", "/v1/audio/transcriptions", tc.req))
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status %d: %s", rec.Code, rec.Body)
			}
			if !strings.Contains(rec.Body.String(), tc.body) {
				t.Errorf("body %s, want it to mention %q", rec.Body, tc.body)
			}
		})
	}
}

func TestTranscriptionNotLoaded(t *testing.T) {
	rec := do(t, &Server{}, jsonRequest("POST", "/v1/audio/transcriptions", TranscriptionRequest{}))
	if rec.Code != http.StatusNotImplemented || !strings.Contains(rec.Body.String(), "-stt") {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}

// The endpoints without a model answer 501 rather than 404, and the message
// distinguishes "not started with it" from "not written yet".
func TestUnimplementedEndpoints(t *testing.T) {
	s := &Server{}
	for _, path := range []string{
		"/v1/chat/completions", "/v1/responses", "/v1/messages",
		"/v1/embeddings", "/v1/images/generations", "/v1/images/edits",
	} {
		rec := do(t, s, jsonRequest("POST", path, map[string]string{}))
		if rec.Code != http.StatusNotImplemented {
			t.Errorf("%s: status %d", path, rec.Code)
		}
	}
}

func TestCORSPreflightNeedsNoToken(t *testing.T) {
	s := &Server{Token: "secret"}
	req := httptest.NewRequest("OPTIONS", "/v1/audio/speech", http.NoBody)
	req.Header.Set("Origin", "http://localhost:5173")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status %d", rec.Code)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "http://localhost:5173" {
		t.Errorf("allow-origin %q", got)
	}
}

func TestUnauthorized(t *testing.T) {
	s := &Server{Token: "secret", Speech: &fakeSpeech{}}
	req := jsonRequest("POST", "/v1/audio/speech", SpeechRequest{Input: "x"})
	rec := httptest.NewRecorder() // no Authorization header
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status %d", rec.Code)
	}
}

// An empty token is an open server, which is the loopback development case.
func TestEmptyTokenIsOpen(t *testing.T) {
	s := &Server{Speech: &fakeSpeech{}}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, jsonRequest("POST", "/v1/audio/speech", SpeechRequest{Input: "x"}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}

func TestMessageContentAcceptsBothShapes(t *testing.T) {
	var msg Message
	if err := json.Unmarshal([]byte(`{"role":"user","content":"hello"}`), &msg); err != nil {
		t.Fatal(err)
	}
	if len(msg.Content) != 1 || msg.Content[0].Type != "text" || msg.Content.Text() != "hello" {
		t.Fatalf("string content decoded as %+v", msg.Content)
	}
	if err := json.Unmarshal([]byte(
		`{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url"},{"type":"text","text":"b"}]}`,
	), &msg); err != nil {
		t.Fatal(err)
	}
	if len(msg.Content) != 3 || msg.Content.Text() != "ab" {
		t.Fatalf("array content decoded as %+v, text %q", msg.Content, msg.Content.Text())
	}
	// An assistant turn that is only tool calls has a null content.
	if err := json.Unmarshal([]byte(`{"role":"assistant","content":null}`), &msg); err != nil {
		t.Fatal(err)
	}
	if len(msg.Content) != 0 {
		t.Fatalf("null content decoded as %+v", msg.Content)
	}
}

// A body over the cap is a 413 and not a confused 400, and the handler
// never sees it.
func TestUploadLimit(t *testing.T) {
	fake := &fakeSTT{}
	s := &Server{Transcription: fake, MaxUploadBytes: 1 << 10}
	rec := do(t, s, jsonRequest("POST", "/v1/audio/transcriptions", TranscriptionRequest{
		Data: testWAV(t, 16000, 4000), // 8 KB of samples, base64 on the wire
	}))
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if fake.clip != nil {
		t.Errorf("the backend was handed a clip anyway")
	}
}

// TestTranscriptionFormats covers the shapes the timings made possible.
// The fake returns a transcript with two sentences of words in it, which is
// what a backend hands over: the grouping is the model's business and the
// container is this package's.
func transcriptFake() *fakeSTT {
	return &fakeSTT{resp: &TranscriptionResponse{
		Text:     "Hello there. General Kenobi.",
		Duration: 2.4,
		Words: []TranscriptionWord{
			{Word: "Hello", Start: 0, End: 0.4},
			{Word: "there.", Start: 0.4, End: 0.8},
			{Word: "General", Start: 1.2, End: 1.76},
			{Word: "Kenobi.", Start: 1.76, End: 2.4},
		},
		Segments: []TranscriptionSegment{
			{ID: 0, Start: 0, End: 0.8, Text: "Hello there."},
			{ID: 1, Start: 1.2, End: 2.4, Text: "General Kenobi."},
		},
	}}
}

func transcriptionRequest(t *testing.T, body map[string]any) *http.Request {
	t.Helper()
	body["file"] = testWAV(t, 16000, 800)
	return jsonRequest("POST", "/v1/audio/transcriptions", body)
}

func TestTranscriptionJSONIsTheTextAlone(t *testing.T) {
	s := &Server{Transcription: transcriptFake()}
	rec := do(t, s, transcriptionRequest(t, map[string]any{}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	// OpenAI's `json` is the text and nothing else: a client that asked for
	// the short shape should not have to ignore four more fields.
	var got map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got["text"] != "Hello there. General Kenobi." {
		t.Errorf("body %v", got)
	}
}

func TestTranscriptionVerboseJSON(t *testing.T) {
	s := &Server{Transcription: transcriptFake()}
	rec := do(t, s, transcriptionRequest(t, map[string]any{
		"response_format":         "verbose_json",
		"language":                "en",
		"timestamp_granularities": []string{"word", "segment"},
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var got TranscriptionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Task != "transcribe" || got.Language != "en" || got.Duration != 2.4 {
		t.Errorf("%+v", got)
	}
	if len(got.Words) != 4 || got.Words[0].Word != "Hello" {
		t.Errorf("words %+v", got.Words)
	}
	if len(got.Segments) != 2 || got.Segments[1].Text != "General Kenobi." {
		t.Errorf("segments %+v", got.Segments)
	}
	// Whisper's decoder fields are absent rather than invented: a number in
	// `no_speech_prob` would be one a client could filter on.
	for _, absent := range []string{"avg_logprob", "no_speech_prob", "compression_ratio", "seek"} {
		if strings.Contains(rec.Body.String(), absent) {
			t.Errorf("the body carries %q, which this model has no value for", absent)
		}
	}
}

// TestTranscriptionGranularities: segments are the default, and asking for
// words alone leaves the segments out rather than sending both.
func TestTranscriptionGranularities(t *testing.T) {
	cases := []struct {
		name                string
		granularities       any
		wantWords, wantSegs bool
	}{
		{"unset", nil, false, true},
		{"word", []string{"word"}, true, false},
		{"segment", []string{"segment"}, false, true},
		{"both", []string{"word", "segment"}, true, true},
	}
	for _, c := range cases {
		body := map[string]any{"response_format": "verbose_json"}
		if c.granularities != nil {
			body["timestamp_granularities"] = c.granularities
		}
		s := &Server{Transcription: transcriptFake()}
		rec := do(t, s, transcriptionRequest(t, body))
		var got TranscriptionResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if (len(got.Words) > 0) != c.wantWords {
			t.Errorf("%s: %d words", c.name, len(got.Words))
		}
		if (len(got.Segments) > 0) != c.wantSegs {
			t.Errorf("%s: %d segments", c.name, len(got.Segments))
		}
	}
}

func TestTranscriptionSubtitles(t *testing.T) {
	s := &Server{Transcription: transcriptFake()}
	rec := do(t, s, transcriptionRequest(t, map[string]any{"response_format": "srt"}))
	wantSRT := "1\n00:00:00,000 --> 00:00:00,800\nHello there.\n\n" +
		"2\n00:00:01,200 --> 00:00:02,400\nGeneral Kenobi.\n\n"
	if rec.Body.String() != wantSRT {
		t.Errorf("srt:\n%q\nwant\n%q", rec.Body.String(), wantSRT)
	}

	s = &Server{Transcription: transcriptFake()}
	rec = do(t, s, transcriptionRequest(t, map[string]any{"response_format": "vtt"}))
	wantVTT := "WEBVTT\n\n00:00:00.000 --> 00:00:00.800\nHello there.\n\n" +
		"00:00:01.200 --> 00:00:02.400\nGeneral Kenobi.\n\n"
	if rec.Body.String() != wantVTT {
		t.Errorf("vtt:\n%q\nwant\n%q", rec.Body.String(), wantVTT)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/vtt") {
		t.Errorf("vtt content-type %q", ct)
	}
}

func TestTimecode(t *testing.T) {
	cases := []struct {
		in   float64
		want string
	}{
		{0, "00:00:00,000"},
		{-1, "00:00:00,000"},
		{0.8, "00:00:00,800"},
		{61.5, "00:01:01,500"},
		{3723.456, "01:02:03,456"},
	}
	for _, c := range cases {
		if got := timecode(c.in, ","); got != c.want {
			t.Errorf("timecode(%v) = %s, want %s", c.in, got, c.want)
		}
	}
}
