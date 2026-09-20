package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeImage is an image backend that renders a flat colour, records what it
// was asked for, and answers a geometry. Everything in this file is about the
// translation between a request and that struct, which is the whole of what
// `api` owns: the model is a device away.
type fakeImage struct {
	reqs           []ImageRequest
	geo            ImageGeometry
	err            error
	returnsNothing bool // (nil, nil), which a backend should never do
}

func (f *fakeImage) Models() []Model {
	return []Model{{ID: "z-image-turbo", Object: "model", OwnedBy: "local"}}
}

func (f *fakeImage) Geometry() ImageGeometry {
	if f.geo.Width == 0 {
		return ImageGeometry{Width: 1024, Height: 1024, MaxWidth: 1024, MaxHeight: 1024,
			Multiple: 16, Steps: 8}
	}
	return f.geo
}

func (f *fakeImage) Generate(_ context.Context, req *ImageRequest) (*ImageResult, error) {
	f.reqs = append(f.reqs, *req)
	if f.err != nil {
		return nil, f.err
	}
	// The partials, which a real backend picks the steps for. This one sends
	// exactly as many as were asked for, so the handler's framing is what is
	// under test rather than the schedule's spacing.
	if req.Partial != nil && req.PartialImages > 0 {
		steps := req.Steps
		if steps == 0 {
			steps = f.Geometry().Steps
		}
		for i := 0; i < req.PartialImages; i++ {
			w, h := req.Width, req.Height
			if w == 0 || h == 0 {
				w, h = f.Geometry().Width, f.Geometry().Height
			}
			frame := image.NewRGBA(image.Rect(0, 0, w, h))
			if err := req.Partial(ImagePartial{
				Index: i, Step: i, Steps: steps, Image: frame, Width: w, Height: h,
			}); err != nil {
				return nil, err
			}
		}
	}
	if f.returnsNothing {
		return nil, nil
	}
	w, h := req.Width, req.Height
	if w == 0 || h == 0 {
		w, h = f.Geometry().Width, f.Geometry().Height
	}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{1, 2, 3, 255})
		}
	}
	seed := int64(99)
	if req.Seed != nil {
		seed = *req.Seed
	}
	steps := req.Steps
	if steps == 0 {
		steps = f.Geometry().Steps
	}
	return &ImageResult{Image: img, Width: w, Height: h, Steps: steps, Seed: seed}, nil
}

func generate(t *testing.T, s *Server, body any) (*httptest.ResponseRecorder, ImageGenerationResponse) {
	t.Helper()
	rec := do(t, s, jsonRequest("POST", "/v1/images/generations", body))
	var got ImageGenerationResponse
	if rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatalf("%v: %s", err, rec.Body)
		}
	}
	return rec, got
}

func TestImageGenerationDefaultSize(t *testing.T) {
	fake := &fakeImage{}
	s := &Server{Token: "t", Image: fake}
	rec, got := generate(t, s, ImageGenerationRequest{Prompt: "a red fox in the snow"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if len(fake.reqs) != 1 {
		t.Fatalf("%d backend calls", len(fake.reqs))
	}
	// A request that named no size is handed the backend's own, resolved
	// here: the backend should never have to ask what its default was.
	if fake.reqs[0].Width != 1024 || fake.reqs[0].Height != 1024 {
		t.Errorf("backend got %dx%d, want the default 1024x1024",
			fake.reqs[0].Width, fake.reqs[0].Height)
	}
	if fake.reqs[0].Seed != nil {
		t.Errorf("seed %v, want nil so the backend draws one", *fake.reqs[0].Seed)
	}
	if got.Size != "1024x1024" || got.OutputFormat != "png" {
		t.Errorf("echoed %q / %q", got.Size, got.OutputFormat)
	}
	if len(got.Data) != 1 {
		t.Fatalf("%d images", len(got.Data))
	}
	// The seed the server drew comes back, because that is the only way a
	// client can ask for the same image twice.
	if got.Data[0].Seed != 99 || got.Data[0].Steps != 8 {
		t.Errorf("seed %d, steps %d", got.Data[0].Seed, got.Data[0].Steps)
	}
	raw, err := base64.StdEncoding.DecodeString(got.Data[0].B64JSON)
	if err != nil {
		t.Fatal(err)
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if format != "png" || cfg.Width != 1024 || cfg.Height != 1024 {
		t.Errorf("%s %dx%d", format, cfg.Width, cfg.Height)
	}
}

func TestImageGenerationSizeReachesTheBackend(t *testing.T) {
	for _, c := range []struct {
		name          string
		req           ImageGenerationRequest
		width, height int
	}{
		{"explicit", ImageGenerationRequest{Prompt: "p", Size: "512x768"}, 512, 768},
		{"auto", ImageGenerationRequest{Prompt: "p", Size: "auto"}, 1024, 1024},
		// A non-square out of a server built square, which is the whole of
		// what the pipeline change bought: half the tokens, the same arenas.
		{"landscape", ImageGenerationRequest{Prompt: "p", Size: "1024x512"}, 1024, 512},
		{"aspect ratio", ImageGenerationRequest{Prompt: "p", AspectRatio: "16:9"}, 1024, 576},
		// Size wins over aspect_ratio, because one of them is OpenAI's field
		// and a client that sent both meant the one every other server reads.
		{"size beats ratio", ImageGenerationRequest{Prompt: "p", Size: "256x256", AspectRatio: "16:9"}, 256, 256},
	} {
		t.Run(c.name, func(t *testing.T) {
			fake := &fakeImage{}
			s := &Server{Image: fake}
			rec, got := generate(t, s, c.req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status %d: %s", rec.Code, rec.Body)
			}
			if fake.reqs[0].Width != c.width || fake.reqs[0].Height != c.height {
				t.Errorf("backend got %dx%d, want %dx%d",
					fake.reqs[0].Width, fake.reqs[0].Height, c.width, c.height)
			}
			if want := formatSize(c.width, c.height); got.Size != want {
				t.Errorf("echoed size %q, want %q", got.Size, want)
			}
		})
	}
}

// The sizes a resizable pipeline still has to refuse, and the reason each one
// is refused rather than rounded: a client that asked for 1000x1000 and got
// 992x992 has no way to find out.
func TestImageGenerationRejectsSizes(t *testing.T) {
	for _, c := range []struct{ name, size, want string }{
		{"not a size", "big", "WIDTHxHEIGHT"},
		{"no height", "1024x", "not a height"},
		{"negative", "-16x16", "not a width"},
		{"off the grid", "1000x1000", "multiple of 16"},
		{"past the ceiling", "2048x2048", "neither side may be past it"},
		{"past it on one side only", "1024x1088", "neither side may be past it"},
		// The same area in a taller shape, which the transformer would take
		// and the VAE will not -- see pipeline.geomFor, where it is measured.
		{"same area, wrong shape", "512x2048", "neither side may be past it"},
	} {
		t.Run(c.name, func(t *testing.T) {
			fake := &fakeImage{}
			s := &Server{Image: fake}
			rec, _ := generate(t, s, ImageGenerationRequest{Prompt: "p", Size: c.size})
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status %d: %s", rec.Code, rec.Body)
			}
			if !strings.Contains(rec.Body.String(), c.want) {
				t.Errorf("body %s, want it to mention %q", rec.Body, c.want)
			}
			if len(fake.reqs) != 0 {
				t.Errorf("the backend ran anyway")
			}
		})
	}
}

// A ratio is fitted to the *server's* ceiling, so the same request answers on
// a server started at a different size. It is the one place this endpoint
// computes a size rather than checking one, and rounding down on both sides is
// what keeps the answer inside the budget.
func TestAspectRatioFitsTheCeiling(t *testing.T) {
	for _, c := range []struct {
		ratio         string
		geo           ImageGeometry
		width, height int
	}{
		{"1:1", ImageGeometry{MaxWidth: 1024, MaxHeight: 1024, Multiple: 16}, 1024, 1024},
		{"16:9", ImageGeometry{MaxWidth: 1024, MaxHeight: 1024, Multiple: 16}, 1024, 576},
		{"9:16", ImageGeometry{MaxWidth: 1024, MaxHeight: 1024, Multiple: 16}, 576, 1024},
		{"2:1", ImageGeometry{MaxWidth: 512, MaxHeight: 512, Multiple: 16}, 512, 256},
		// The width binds and the height comes out well under its own
		// ceiling, which is the ratio being honoured rather than the area
		// being spent.
		{"3:2", ImageGeometry{MaxWidth: 512, MaxHeight: 512, Multiple: 16}, 512, 336},
		// A non-square ceiling: the height binds for the first of these and
		// the width for the second, which is the only way to tell the two
		// halves of the fit apart.
		{"1:1", ImageGeometry{MaxWidth: 1024, MaxHeight: 512, Multiple: 16}, 512, 512},
		{"2:1", ImageGeometry{MaxWidth: 1024, MaxHeight: 512, Multiple: 16}, 1024, 512},
	} {
		t.Run(c.ratio+" into "+formatSize(c.geo.MaxWidth, c.geo.MaxHeight), func(t *testing.T) {
			w, h, err := fitRatio(c.ratio, c.geo)
			if err != nil {
				t.Fatal(err)
			}
			if w != c.width || h != c.height {
				t.Fatalf("%s into %dx%d gave %dx%d, want %dx%d", c.ratio,
					c.geo.MaxWidth, c.geo.MaxHeight, w, h, c.width, c.height)
			}
			if w > c.geo.MaxWidth || h > c.geo.MaxHeight {
				t.Errorf("%dx%d is outside the ceiling it was fitted to", w, h)
			}
		})
	}
}

func TestImageGenerationN(t *testing.T) {
	fake := &fakeImage{}
	s := &Server{Image: fake}
	seed := int64(7)
	rec, got := generate(t, s, ImageGenerationRequest{Prompt: "p", N: 3, Seed: &seed})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if len(got.Data) != 3 || len(fake.reqs) != 3 {
		t.Fatalf("%d images from %d runs", len(got.Data), len(fake.reqs))
	}
	// Three images from one named seed are three *different* images, or the
	// parameter did something a client did not ask for.
	for i, r := range fake.reqs {
		if r.Seed == nil || *r.Seed != seed+int64(i) {
			t.Errorf("run %d got seed %v, want %d", i, r.Seed, seed+int64(i))
		}
		if got.Data[i].Seed != seed+int64(i) {
			t.Errorf("run %d reported seed %d", i, got.Data[i].Seed)
		}
	}
}

func TestImageGenerationRefusals(t *testing.T) {
	for _, c := range []struct {
		name string
		req  ImageGenerationRequest
		want string
	}{
		{"empty prompt", ImageGenerationRequest{Prompt: "  "}, "prompt is empty"},
		{"n too large", ImageGenerationRequest{Prompt: "p", N: 9}, "1 to 4"},
		{"url", ImageGenerationRequest{Prompt: "p", ResponseFormat: "url"}, "nowhere to host"},
		{"webp", ImageGenerationRequest{Prompt: "p", OutputFormat: "webp"}, "png and jpeg"},
		{"quality", ImageGenerationRequest{Prompt: "p", OutputCompression: 101}, "outside [0, 100]"},
	} {
		t.Run(c.name, func(t *testing.T) {
			fake := &fakeImage{}
			s := &Server{Image: fake}
			rec, _ := generate(t, s, c.req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status %d: %s", rec.Code, rec.Body)
			}
			if !strings.Contains(rec.Body.String(), c.want) {
				t.Errorf("body %s, want it to mention %q", rec.Body, c.want)
			}
			if len(fake.reqs) != 0 {
				t.Errorf("the backend ran anyway")
			}
		})
	}
}

func TestImageGenerationJPEG(t *testing.T) {
	s := &Server{Image: &fakeImage{}}
	rec, got := generate(t, s, ImageGenerationRequest{
		Prompt: "p", Size: "64x64", OutputFormat: "jpeg", OutputCompression: 50,
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if got.OutputFormat != "jpeg" {
		t.Errorf("output_format %q", got.OutputFormat)
	}
	raw, err := base64.StdEncoding.DecodeString(got.Data[0].B64JSON)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := jpeg.Decode(bytes.NewReader(raw)); err != nil {
		t.Fatalf("not a JPEG: %v", err)
	}
}

// An unsupported request from the backend is the client's fault; anything else
// is ours. The distinction is the whole reason api.ErrUnsupported exists, and
// it is a wrapped sentinel rather than a message so that rewording the message
// cannot silently turn a 400 into a 500.
func TestImageGenerationBackendErrors(t *testing.T) {
	for _, c := range []struct {
		name string
		fake *fakeImage
		want int
	}{
		{"unsupported", &fakeImage{err: fmt.Errorf("the prompt is 700 tokens: %w", ErrUnsupported)}, 400},
		{"broken", &fakeImage{err: errors.New("device lost")}, 500},
		{"nothing at all", &fakeImage{returnsNothing: true}, 500},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := &Server{Image: c.fake}
			rec, _ := generate(t, s, ImageGenerationRequest{Prompt: "p"})
			if rec.Code != c.want {
				t.Fatalf("status %d, want %d: %s", rec.Code, c.want, rec.Body)
			}
		})
	}
}

// /v1/images/edits on a server that was not started for editing. It is a 501
// naming the flag, because under Qwen-Image-2.1 editing is residency: the
// 27-layer vision tower and the VAE's encoder are staged or they are not, and
// each reference image costs prefix KV cache besides. A client that gets this
// knows exactly what to change.
func TestImageEditNeedsTheFlag(t *testing.T) {
	s := &Server{Image: &fakeImage{}}
	rec := do(t, s, jsonRequest("POST", "/v1/images/edits", ImageEditRequest{Prompt: "p"}))
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "-edits") {
		t.Errorf("body %s, want it to name the flag", rec.Body)
	}
}

func TestImageGenerationNeedsTheFlag(t *testing.T) {
	rec, _ := generate(t, &Server{}, ImageGenerationRequest{Prompt: "p"})
	if rec.Code != http.StatusNotImplemented || !strings.Contains(rec.Body.String(), "-image") {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}

// The geometry rides on the model object for the same reason the voices do: a
// client that has to guess this server's ceiling will guess wrong on a server
// started at anything but the default.
func TestModelsCarriesTheImageGeometry(t *testing.T) {
	s := &Server{
		Speech: &fakeSpeech{},
		Image: &fakeImage{geo: ImageGeometry{
			Width: 768, Height: 768, MaxWidth: 1024, MaxHeight: 1024, Multiple: 16, Steps: 8,
		}},
	}
	rec := do(t, s, httptest.NewRequest("GET", "/v1/models", http.NoBody))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var got struct {
		Data []struct {
			ID    string `json:"id"`
			Image *struct {
				DefaultSize  string `json:"default_size"`
				MaxSize      string `json:"max_size"`
				SizeMultiple int    `json:"size_multiple"`
				DefaultSteps int    `json:"default_steps"`
			} `json:"image"`
			Voices []string `json:"voices"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	for _, m := range got.Data {
		switch m.ID {
		case "z-image-turbo":
			if m.Image == nil {
				t.Fatal("the image model has no geometry")
			}
			if m.Image.DefaultSize != "768x768" || m.Image.MaxSize != "1024x1024" {
				t.Errorf("sizes %q / %q", m.Image.DefaultSize, m.Image.MaxSize)
			}
			if m.Image.SizeMultiple != 16 || m.Image.DefaultSteps != 8 {
				t.Errorf("geometry %+v", *m.Image)
			}
			if len(m.Voices) != 0 {
				t.Errorf("the image model has voices: %v", m.Voices)
			}
		case "kokoro-82m":
			if m.Image != nil {
				t.Errorf("the speech model has an image geometry: %+v", *m.Image)
			}
		default:
			t.Errorf("unexpected model %q", m.ID)
		}
	}
}

// --- /v1/images/edits -------------------------------------------------

// editGeo is a server that can edit: the default geometry with the encoder
// resident and previews on, so the streaming cases have something to answer.
func editGeo() ImageGeometry {
	return ImageGeometry{
		Width: 1024, Height: 1024, MaxWidth: 1024, MaxHeight: 1024,
		Multiple: 16, Steps: 8, Previews: true, MaxPartials: 3,
		Edits: true, MaxRefs: 3,
	}
}

// pngOf is a base64 PNG of the given size, which is what an `image` field
// carries.
func pngOf(t *testing.T, w, h int) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x), uint8(y), 128, 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func edit(t *testing.T, s *Server, body any) (*httptest.ResponseRecorder, ImageGenerationResponse) {
	t.Helper()
	rec := do(t, s, jsonRequest("POST", "/v1/images/edits", body))
	var got ImageGenerationResponse
	if rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatalf("%v: %s", err, rec.Body)
		}
	}
	return rec, got
}

// TestImageEditPassesThePicture is the whole of what this endpoint owns: the
// base64 becomes an image.Image at the size it was sent, several of them
// arrive in order, and a request that named no size leaves the geometry to
// the backend rather than inventing one.
func TestImageEditPassesThePicture(t *testing.T) {
	fake := &fakeImage{geo: editGeo()}
	s := &Server{Image: fake}
	rec, got := edit(t, s, ImageEditRequest{
		Prompt: "at night", Image: StringList{pngOf(t, 640, 480), pngOf(t, 128, 256)},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if len(fake.reqs) != 1 {
		t.Fatalf("%d backend calls", len(fake.reqs))
	}
	req := fake.reqs[0]
	if len(req.Init) != 2 {
		t.Fatalf("the backend was given %d reference images, want 2", len(req.Init))
	}
	if b := req.Init[0].Bounds(); b.Dx() != 640 || b.Dy() != 480 {
		t.Errorf("the backend got a %dx%d picture, want the 640x480 that was sent -- "+
			"resizing is the backend's, so the handler must not have done it", b.Dx(), b.Dy())
	}
	if b := req.Init[1].Bounds(); b.Dx() != 128 || b.Dy() != 256 {
		t.Errorf("the second reference arrived %dx%d, want 128x256 -- the order is the client's",
			b.Dx(), b.Dy())
	}
	// An edit that named no size leaves it at zero for the backend, which
	// resolves it from the *last* reference image's aspect ratio at the
	// model's condition area — arithmetic the handler cannot do, because the
	// condition area is the backend's.
	if req.Width != 0 || req.Height != 0 {
		t.Errorf("the handler resolved %dx%d; an edit that named no size leaves it to the backend",
			req.Width, req.Height)
	}
	// And the echoed size is the one that came *back*, not the one that went
	// in — which is the only way a client learns what it got.
	if got.Size != "1024x1024" {
		t.Errorf("echoed size %q, want the backend's resolved 1024x1024", got.Size)
	}
}

// TestImageEditSizeRules covers the three ways an edit's geometry is arrived
// at. Two of them are the generation endpoint's rules unchanged; the third is
// not a rule here at all, and that is the point — under SDEdit the handler
// fitted the input's own bounds into the ceiling, and under conditional
// generation the size follows the *condition area*, which only the backend
// knows.
func TestImageEditSizeRules(t *testing.T) {
	for _, c := range []struct {
		name          string
		req           ImageEditRequest
		width, height int
	}{
		{"no size is left to the backend", ImageEditRequest{}, 0, 0},
		{"an explicit size wins", ImageEditRequest{Size: "512x512"}, 512, 512},
		{"an aspect ratio wins", ImageEditRequest{AspectRatio: "1:1"}, 1024, 1024},
	} {
		t.Run(c.name, func(t *testing.T) {
			fake := &fakeImage{geo: editGeo()}
			s := &Server{Image: fake}
			c.req.Prompt = "p"
			c.req.Image = StringList{pngOf(t, 640, 480)}
			rec, _ := edit(t, s, c.req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status %d: %s", rec.Code, rec.Body)
			}
			if fake.reqs[0].Width != c.width || fake.reqs[0].Height != c.height {
				t.Errorf("rendered %dx%d, want %dx%d", fake.reqs[0].Width, fake.reqs[0].Height, c.width, c.height)
			}
		})
	}

	// A named size past the ceiling is still the generation endpoint's
	// refusal, because that one *is* the handler's: it is checked against
	// Geometry() and not against any picture.
	fake := &fakeImage{geo: editGeo()}
	s := &Server{Image: fake}
	rec, _ := edit(t, s, ImageEditRequest{Prompt: "p", Image: StringList{pngOf(t, 64, 64)}, Size: "4096x4096"})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d for a size past the ceiling, want 400: %s", rec.Code, rec.Body)
	}
}

// TestImageEditRefusals is the negative control for the door. Every one of
// these would otherwise come back as a picture that quietly did something
// other than what was asked.
func TestImageEditRefusals(t *testing.T) {
	good := pngOf(t, 64, 64)
	for _, c := range []struct {
		name string
		req  ImageEditRequest
		want int
		says string
	}{
		{"no image", ImageEditRequest{Prompt: "p"}, 400, "image"},
		{"no prompt", ImageEditRequest{Image: StringList{good}}, 400, "prompt"},
		{"more references than the server takes",
			ImageEditRequest{Prompt: "p", Image: StringList{good, good, good, good}}, 400, "reference images"},
		{"a mask", ImageEditRequest{Prompt: "p", Image: StringList{good}, Mask: good}, 501, "mask"},
		{"not base64", ImageEditRequest{Prompt: "p", Image: StringList{"not base64!!"}}, 400, "base64"},
		{"not an image", ImageEditRequest{Prompt: "p", Image: StringList{
			base64.StdEncoding.EncodeToString([]byte("hello"))}}, 400, "png and jpeg"},
		{"a strength", ImageEditRequest{Prompt: "p", Image: StringList{good}, Strength: 0.5}, 400, "strength"},
		{"a url response", ImageEditRequest{Prompt: "p", Image: StringList{good}, ResponseFormat: "url"}, 400, "b64_json"},
		{"n past the cap", ImageEditRequest{Prompt: "p", Image: StringList{good}, N: 9}, 400, "n is 9"},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := &Server{Image: &fakeImage{geo: editGeo()}}
			rec, _ := edit(t, s, c.req)
			if rec.Code != c.want {
				t.Fatalf("status %d, want %d: %s", rec.Code, c.want, rec.Body)
			}
			if !strings.Contains(rec.Body.String(), c.says) {
				t.Errorf("body %s, want it to mention %q", rec.Body, c.says)
			}
		})
	}
}

// TestImageEditMultipart is the encoding OpenAI's own clients send, and the
// only one their endpoint accepts. It has to arrive at the same struct the
// JSON body does, which is what this asserts by checking the fields that
// travel as form values rather than as JSON types.
func TestImageEditMultipart(t *testing.T) {
	raw, err := base64.StdEncoding.DecodeString(pngOf(t, 320, 320))
	if err != nil {
		t.Fatal(err)
	}
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	part, err := mw.CreateFormFile("image", "in.png")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(raw); err != nil {
		t.Fatal(err)
	}
	for k, v := range map[string]string{
		"prompt": "at night", "steps": "6", "seed": "1234",
		"output_format": "jpeg", "size": "256x256",
	} {
		if err := mw.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}

	fake := &fakeImage{geo: editGeo()}
	s := &Server{Image: fake}
	req := httptest.NewRequest("POST", "/v1/images/edits", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := do(t, s, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	got := fake.reqs[0]
	if len(got.Init) != 1 {
		t.Fatalf("%d reference images came out of the multipart body, want 1", len(got.Init))
	}
	if b := got.Init[0].Bounds(); b.Dx() != 320 || b.Dy() != 320 {
		t.Errorf("the picture arrived %dx%d, want 320x320", b.Dx(), b.Dy())
	}
	if got.Steps != 6 || got.Seed == nil || *got.Seed != 1234 {
		t.Errorf("steps %d, seed %v", got.Steps, got.Seed)
	}
	if got.Width != 256 || got.Height != 256 {
		t.Errorf("rendered %dx%d, want the 256x256 the form asked for", got.Width, got.Height)
	}
	var resp ImageGenerationResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.OutputFormat != "jpeg" {
		t.Errorf("output_format %q, want the jpeg the form asked for", resp.OutputFormat)
	}
}

// TestImageEditStreams is the claim that made an edit a field on ImageRequest
// rather than a second method: the streaming path is the generation's, so an
// edit gets partial frames with no code of its own.
func TestImageEditStreams(t *testing.T) {
	fake := &fakeImage{geo: editGeo()}
	s := &Server{Image: fake}
	rec := do(t, s, jsonRequest("POST", "/v1/images/edits", ImageEditRequest{
		Prompt: "p", Image: StringList{pngOf(t, 256, 256)}, Stream: true, PartialImages: 2,
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	body := rec.Body.String()
	if n := strings.Count(body, "event: "+eventPartialImage); n != 2 {
		t.Errorf("%d partial frames in the stream, want 2:\n%s", n, body)
	}
	if !strings.Contains(body, "event: "+eventImageDone) {
		t.Error("no completion event")
	}
	if len(fake.reqs[0].Init) == 0 {
		t.Error("the streamed run was not given the picture")
	}
}
