package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// errBoom is a run that failed for a reason the client cannot fix, i.e. the
// case that has to become a frame rather than a status.
var errBoom = errors.New("boom: the device went away")

// previewGeometry is a backend that has a preview decoder. Everything in this
// file is about what the handler does with `stream` and `partial_images`; that
// the frames are pictures of anything is the VAE's business.
func previewGeometry() ImageGeometry {
	return ImageGeometry{Width: 1024, Height: 1024, MaxPixels: 1024 * 1024,
		Multiple: 16, Steps: 8, Previews: true, MaxPartials: 3}
}

// sseFrame is one parsed event of a stream.
type sseFrame struct {
	event string
	data  ImageStreamEvent
}

// parseSSE splits a recorded body into frames. It is deliberately strict about
// the framing -- a blank line after every `data:` -- because a stream that
// parses only under a lenient reader is a stream that will not work.
func parseSSE(t *testing.T, body string) []sseFrame {
	t.Helper()
	var out []sseFrame
	for _, block := range strings.Split(strings.TrimSuffix(body, "\n\n"), "\n\n") {
		if strings.TrimSpace(block) == "" {
			continue
		}
		var f sseFrame
		for _, line := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(line, "event: "):
				f.event = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &f.data); err != nil {
					t.Fatalf("frame %q: %v", block, err)
				}
			default:
				t.Fatalf("unexpected line %q in %q", line, block)
			}
		}
		out = append(out, f)
	}
	return out
}

func stream(t *testing.T, s *Server, body any) *httptest.ResponseRecorder {
	t.Helper()
	return do(t, s, jsonRequest("POST", "/v1/images/generations", body))
}

// TestImageStreamSendsPartialsThenTheImage is the shape of the endpoint: three
// partial frames numbered from zero, then one completed frame carrying the seed
// and the step count, and nothing after it.
func TestImageStreamSendsPartialsThenTheImage(t *testing.T) {
	fake := &fakeImage{geo: previewGeometry()}
	s := &Server{Image: fake}
	rec := stream(t, s, ImageGenerationRequest{
		Prompt: "a fox", Size: "64x64", Stream: true, PartialImages: 3, Seed: ptr(int64(7)),
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type %q", ct)
	}
	frames := parseSSE(t, rec.Body.String())
	if len(frames) != 4 {
		t.Fatalf("%d frames, want 3 partials and a completed", len(frames))
	}
	for i := 0; i < 3; i++ {
		f := frames[i]
		if f.event != eventPartialImage || f.data.Type != eventPartialImage {
			t.Errorf("frame %d: event %q / type %q", i, f.event, f.data.Type)
		}
		if f.data.PartialImageIndex == nil || *f.data.PartialImageIndex != i {
			t.Errorf("frame %d: partial_image_index %v", i, f.data.PartialImageIndex)
		}
		if f.data.Step == nil || f.data.Steps != 8 {
			t.Errorf("frame %d: step %v of %d", i, f.data.Step, f.data.Steps)
		}
		if f.data.Size != "64x64" || f.data.OutputFormat != "png" {
			t.Errorf("frame %d: size %q format %q", i, f.data.Size, f.data.OutputFormat)
		}
		decodePNG(t, f.data.B64JSON, 64, 64)
	}

	last := frames[3]
	if last.event != eventImageDone || last.data.Type != eventImageDone {
		t.Errorf("last frame: event %q / type %q", last.event, last.data.Type)
	}
	if last.data.PartialImageIndex != nil {
		t.Errorf("the completed frame carries a partial_image_index: %d", *last.data.PartialImageIndex)
	}
	// The seed and the step count are what make a streamed image reproducible,
	// and the completed frame is the only place they can go.
	if last.data.Seed != 7 || last.data.Steps != 8 {
		t.Errorf("completed frame: seed %d, steps %d", last.data.Seed, last.data.Steps)
	}
	decodePNG(t, last.data.B64JSON, 64, 64)

	// And the backend was asked for exactly what the request said.
	if len(fake.reqs) != 1 {
		t.Fatalf("%d backend calls", len(fake.reqs))
	}
	if got := fake.reqs[0].PartialImages; got != 3 {
		t.Errorf("the backend was asked for %d partials, not 3", got)
	}
	if fake.reqs[0].Partial == nil {
		t.Error("the backend was given no callback to send partials to")
	}
}

// TestImageStreamWithoutPartials is OpenAI's default: `stream: true` alone
// asks for the framing and not for any in-progress work, so the only frame is
// the finished image.
func TestImageStreamWithoutPartials(t *testing.T) {
	fake := &fakeImage{geo: previewGeometry()}
	s := &Server{Image: fake}
	rec := stream(t, s, ImageGenerationRequest{Prompt: "a fox", Size: "64x64", Stream: true})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	frames := parseSSE(t, rec.Body.String())
	if len(frames) != 1 || frames[0].event != eventImageDone {
		t.Fatalf("%d frames, first %q; want one completed", len(frames), frames[0].event)
	}
	if fake.reqs[0].PartialImages != 0 {
		t.Errorf("the backend was asked for %d partials", fake.reqs[0].PartialImages)
	}
}

// TestImageStreamNeedsThePreviewDecoder is the refusal a backend without a
// preview decoder gets.
//
// It is a 501 and not a 400 because the request is well formed and the
// loaded backend cannot answer it. The shipped image backend always can --
// its preview decoder is a matrix compiled in, not a checkpoint to load --
// so this covers the interface's contract rather than a configuration the
// server has: `previews: false` in GET /v1/models has to mean `stream: true`
// is refused, and the refusal has to say which field said so.
func TestImageStreamNeedsThePreviewDecoder(t *testing.T) {
	geo := previewGeometry()
	geo.Previews, geo.MaxPartials = false, 0
	fake := &fakeImage{geo: geo}
	s := &Server{Image: fake}
	rec := stream(t, s, ImageGenerationRequest{Prompt: "a fox", Stream: true})
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	// It points at the field a client could have read first rather than at a
	// flag: the shipped image backend always has a preview decoder, so a
	// backend that reports none is a property of the model, not of how the
	// server was started.
	body := rec.Body.String()
	if !strings.Contains(body, "image.previews") {
		t.Errorf("body %s, want it to name the capability field", rec.Body)
	}
	if len(fake.reqs) != 0 {
		t.Error("the backend ran anyway")
	}
}

// TestImageStreamRefusals covers the rest, and the common thread is that every
// one of them is answerable before the stream opens -- which is what lets them
// be a status code rather than an `error` event.
func TestImageStreamRefusals(t *testing.T) {
	for _, c := range []struct {
		name string
		req  ImageGenerationRequest
		want string
	}{
		{"too many partials", ImageGenerationRequest{Prompt: "p", Stream: true, PartialImages: 4}, "0 to 3"},
		{"negative partials", ImageGenerationRequest{Prompt: "p", Stream: true, PartialImages: -1}, "0 to 3"},
		{"partials without stream", ImageGenerationRequest{Prompt: "p", PartialImages: 2}, "needs stream: true"},
		{"n with stream", ImageGenerationRequest{Prompt: "p", Stream: true, N: 2}, "the event stream carries one image"},
		{"bad size", ImageGenerationRequest{Prompt: "p", Stream: true, Size: "1000x1000"}, "multiple of 16"},
	} {
		t.Run(c.name, func(t *testing.T) {
			fake := &fakeImage{geo: previewGeometry()}
			s := &Server{Image: fake}
			rec := stream(t, s, c.req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status %d: %s", rec.Code, rec.Body)
			}
			if !strings.Contains(rec.Body.String(), c.want) {
				t.Errorf("body %s, want it to mention %q", rec.Body, c.want)
			}
			if len(fake.reqs) != 0 {
				t.Error("the backend ran anyway")
			}
		})
	}
}

// TestImageStreamErrorAfterAFrame is the case the status code cannot cover: a
// run that fails once the stream is open. The status was committed with the
// first byte, so the failure has to be a frame.
func TestImageStreamErrorAfterAFrame(t *testing.T) {
	fake := &fakeImage{geo: previewGeometry(), err: errBoom}
	s := &Server{Image: fake}
	rec := stream(t, s, ImageGenerationRequest{Prompt: "p", Size: "64x64", Stream: true})
	// 200, because the stream opened. This is not a bug being papered over: a
	// client reading an event stream reads frames, and the error frame is
	// where a failure belongs once a frame has been promised.
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	frames := parseSSE(t, rec.Body.String())
	if len(frames) != 1 || frames[0].event != "error" {
		t.Fatalf("frames %+v, want one error", frames)
	}
	if !strings.Contains(rec.Body.String(), "boom") {
		t.Errorf("the error frame does not carry the message: %s", rec.Body)
	}
}

// TestImageStreamJPEG pins that a partial is in the same container as the
// finished image. A client decoding the stream should not have to switch
// decoders halfway through it.
func TestImageStreamJPEG(t *testing.T) {
	s := &Server{Image: &fakeImage{geo: previewGeometry()}}
	rec := stream(t, s, ImageGenerationRequest{
		Prompt: "p", Size: "64x64", Stream: true, PartialImages: 2, OutputFormat: "jpeg",
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	for i, f := range parseSSE(t, rec.Body.String()) {
		if f.data.OutputFormat != "jpeg" {
			t.Errorf("frame %d: output_format %q", i, f.data.OutputFormat)
		}
		raw, err := base64.StdEncoding.DecodeString(f.data.B64JSON)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if len(raw) < 2 || raw[0] != 0xff || raw[1] != 0xd8 {
			t.Errorf("frame %d is not a JPEG", i)
		}
	}
}

// TestModelsReportsPreviews is the discovery half: a client should be able to
// find out whether `stream: true` will work without sending one and reading
// the refusal.
func TestModelsReportsPreviews(t *testing.T) {
	for _, previews := range []bool{true, false} {
		geo := previewGeometry()
		geo.Previews = previews
		if !previews {
			geo.MaxPartials = 0
		}
		s := &Server{Image: &fakeImage{geo: geo}}
		rec := do(t, s, httptest.NewRequest("GET", "/v1/models", http.NoBody))
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body)
		}
		var got struct {
			Data []struct {
				Image *struct {
					Previews    bool `json:"previews"`
					MaxPartials int  `json:"max_partial_images"`
				} `json:"image"`
			} `json:"data"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if len(got.Data) != 1 || got.Data[0].Image == nil {
			t.Fatalf("no image geometry in %s", rec.Body)
		}
		if got.Data[0].Image.Previews != previews {
			t.Errorf("previews=%v reported as %v", previews, got.Data[0].Image.Previews)
		}
		if want := 0; !previews && got.Data[0].Image.MaxPartials != want {
			t.Errorf("max_partial_images %d with no preview decoder", got.Data[0].Image.MaxPartials)
		}
	}
}

func decodePNG(t *testing.T, b64 string, w, h int) {
	t.Helper()
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if b := img.Bounds(); b.Dx() != w || b.Dy() != h {
		t.Errorf("frame is %dx%d, want %dx%d", b.Dx(), b.Dy(), w, h)
	}
}

func ptr[T any](v T) *T { return &v }
