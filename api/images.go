package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// OpenAI Image Generation
// https://platform.openai.com/docs/api-reference/images/create

// ImageGenerationRequest is the OpenAI image request, plus the three
// extensions this model has and OpenAI's does not.
//
// `size` is the field that matters here, and it is the reason this endpoint
// took a pipeline change to wire up rather than a handler: every arena used
// to be built for one width and one height, so a size was a residency
// question. It is now a ceiling, and a size inside it is a parameter -- see
// ImageGeometry.
type ImageGenerationRequest struct {
	host   string
	Model  string `json:"model,omitempty"`
	Prompt string `json:"prompt"`
	// Size is "WIDTHxHEIGHT", or "auto" for the server's default. Both sides
	// must be a multiple of ImageGeometry.Multiple.
	Size string `json:"size,omitempty"`
	// AspectRatio is an extension: "16:9" and the like, fitted to the largest
	// image the server holds. It is here because a client that wants a
	// landscape picture should not have to know this server's ceiling to ask
	// for one. Size wins when both are set.
	AspectRatio string `json:"aspect_ratio,omitempty"`
	N           int    `json:"n,omitempty"`
	// ResponseFormat is OpenAI's "b64_json" or "url". This server has no
	// object store, so only the first answers.
	ResponseFormat string `json:"response_format,omitempty"`
	// OutputFormat is the container: "png" or "jpeg".
	OutputFormat string `json:"output_format,omitempty"`
	// OutputCompression is JPEG quality, 0-100. It is OpenAI's field and is
	// ignored for PNG, which is lossless.
	OutputCompression int `json:"output_compression,omitempty"`
	// Seed is an extension: the initial latent's seed. Absent draws one, and
	// the response says which, so an image a caller liked can be asked for
	// again.
	Seed *int64 `json:"seed,omitempty"`
	// Steps is an extension: the denoising schedule's length. Qwen-Image-2.1
	// is not a distillation -- its default schedule is 40 -- so unlike under
	// Z-Image-Turbo this is a knob with real range in it: fewer steps is a
	// proportionally faster image, and how far down quality holds is the
	// question IMAGE.md's Q-o1 tracks.
	Steps int `json:"steps,omitempty"`
	// Background is OpenAI's "transparent" / "opaque" / "auto". It is
	// answerable here because this model's VAE is natively RGBA, and it is
	// not a mode: transparency is asked for *in the prompt* (the checkpoint's
	// own recommended phrasing), so "transparent" rewrites the prompt and
	// keeps the alpha plane, and everything else composites over white.
	Background string `json:"background,omitempty"`
	// Stream sends the image over Server-Sent Events, with in-progress
	// frames as it goes. It needs a preview decoder loaded; see
	// handleImageGeneration.
	Stream bool `json:"stream,omitempty"`
	// PartialImages is how many in-progress frames to send before the
	// finished one, 0 to ImageGeometry.MaxPartials. It is OpenAI's field and
	// OpenAI's default of zero, which means a stream that carries only the
	// finished image -- so `stream: true` alone is a framing choice and
	// `partial_images` is what costs anything.
	PartialImages int    `json:"partial_images,omitempty"`
	User          string `json:"user,omitempty"`
}

// ImageEditRequest is OpenAI's edit request: a picture, a prompt, and what to
// do to the one with the other.
//
// **Both encodings are accepted**, the same as /v1/audio/transcriptions:
// multipart/form-data, which is what OpenAI's clients send and the only thing
// their own endpoint takes, and a JSON body whose `image` is base64, which is
// what curl and a test can write by hand. The multipart parse fills the same
// struct, so nothing below the door knows which arrived.
type ImageEditRequest struct {
	host        string
	AspectRatio string `json:"aspect_ratio,omitempty"`
	Model       string `json:"model,omitempty"`
	// Image is the reference images an edit is conditioned on, base64, in
	// the order they should appear in the prompt. OpenAI's field is a list
	// because their model composites several, and this model takes a list
	// for its own reason: up to ten references become rows of the
	// transformer's prefix. How many *this* server takes is residency and is
	// reported as `max_reference_images`.
	Image StringList `json:"image"`
	// Mask is OpenAI's transparency mask. It is parsed so that a request
	// carrying one gets told this server does not blend rather than getting a
	// picture that quietly ignored it; see handleImageEdit.
	Mask   string `json:"mask,omitempty"`
	Prompt string `json:"prompt"`
	Size   string `json:"size,omitempty"`
	N      int    `json:"n,omitempty"`
	// Strength was SDEdit's knob — how much of the denoising schedule to run
	// over the encoded picture — and Qwen-Image-2.1 has no such quantity: a
	// reference conditions the whole trajectory instead of seeding a
	// truncated one, and every step runs.
	//
	// It is still parsed, and that is the point. A field that was removed
	// from the struct would be ignored by the JSON decoder and the request
	// would quietly do something else; parsed, a non-zero value is a 400
	// that explains what changed.
	Strength float64 `json:"strength,omitempty"`

	ResponseFormat    string `json:"response_format,omitempty"`
	OutputFormat      string `json:"output_format,omitempty"`
	OutputCompression int    `json:"output_compression,omitempty"`
	Seed              *int64 `json:"seed,omitempty"`
	Steps             int    `json:"steps,omitempty"`
	Stream            bool   `json:"stream,omitempty"`
	PartialImages     int    `json:"partial_images,omitempty"`
	User              string `json:"user,omitempty"`
}

// ImageGenerationResponse is OpenAI's image response.
type ImageGenerationResponse struct {
	Created int64       `json:"created"`
	Data    []ImageData `json:"data"`
	// Size and OutputFormat echo what was actually produced, which is not
	// always what was asked for: "auto" and an aspect ratio both resolve here.
	Size         string `json:"size,omitempty"`
	OutputFormat string `json:"output_format,omitempty"`
}

// ImageData is one generated image.
type ImageData struct {
	B64JSON string `json:"b64_json,omitempty"`
	// RevisedPrompt is OpenAI's, and is always absent here: nothing in this
	// server rewrites a prompt, and an echo of the original dressed as a
	// revision would be a lie a client could act on.
	RevisedPrompt string `json:"revised_prompt,omitempty"`
	// Seed and Steps are extensions, and they are what makes an image
	// reproducible: a request that named neither still gets told what it got.
	Seed  int64 `json:"seed,omitempty"`
	Steps int   `json:"steps,omitempty"`
}

// maxImages bounds `n`. Each image is a full run of a model sized to saturate
// the device -- 14 s at 1024x1024 -- and they are serial, because the device
// lock is. So `n` is a multiplier on a wall clock that is already seconds, and
// four of them is a minute: past that a client should send four requests and
// see three of them come back.
const maxImages = 4

// handleImageGeneration renders a prompt, over one JSON body or over
// Server-Sent Events.
//
// **Streaming needs the preview decoder**, and that is the whole of the
// condition. Decoding an in-progress latent through the full VAE is 0.87 s
// against a 1.67 s step, so a frame would cost half a step and three of them
// a fifth of the image; through madebyollin/taef1 it is 87 ms, 5% of a step.
// So a server started without `-preview` refuses `stream: true` and says
// which flag it wants, rather than answering with one frame at the end.
func (s *Server) handleImageGeneration(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Image == nil {
		notLoaded(ctx, w, "image generation", "-image")
		return
	}
	var req ImageGenerationRequest
	if !decodeJSON(ctx, w, r, &req) {
		return
	}
	if strings.TrimSpace(req.Prompt) == "" {
		badRequest(ctx, w, "prompt is empty")
		return
	}
	geo := s.Image.Geometry()
	if !s.checkStreaming(ctx, w, geo, req.Stream, req.PartialImages) {
		return
	}
	n, ok := s.imageCount(ctx, w, req.N, req.Stream)
	if !ok {
		return
	}
	out, ok := s.imageOutput(ctx, w, req.ResponseFormat, req.OutputFormat, req.OutputCompression)
	if !ok {
		return
	}

	width, height, err := resolveSize(req.Size, req.AspectRatio, geo)
	if err != nil {
		badRequest(ctx, w, err.Error())
		return
	}
	transparent, ok := s.imageBackground(ctx, w, req.Background, out.format)
	if !ok {
		return
	}

	s.render(w, r, &imageRun{
		prompt: req.Prompt, width: width, height: height, steps: req.Steps, seed: req.Seed,
		n: n, format: out.format, compression: out.compression,
		stream: req.Stream, partials: req.PartialImages, transparent: transparent,
	})
}

// imageBackground validates `background` and says whether the alpha plane
// survives.
//
// JPEG has no alpha, so "transparent" with `output_format: "jpeg"` is a
// request that cannot be answered as asked -- and answering it by quietly
// flattening would hand back an opaque picture that the prompt rewrite had
// also made worse. It is a 400 naming both fields.
func (s *Server) imageBackground(ctx context.Context, w http.ResponseWriter, background, format string) (transparent, ok bool) {
	switch background {
	case "", "auto", "opaque":
		return false, true
	case "transparent":
		if format != "png" {
			badRequest(ctx, w, "background \"transparent\" needs output_format \"png\"; "+
				strconv.Quote(format)+" has no alpha channel")
			return false, false
		}
		return true, true
	default:
		badRequest(ctx, w, "background "+strconv.Quote(background)+
			" is not supported; use \"transparent\", \"opaque\" or \"auto\"")
		return false, false
	}
}

// imageOutputs is the container a request resolved to.
type imageOutputs struct {
	format      string
	compression int
}

// imageOutput validates response_format, output_format and
// output_compression, which are the same three fields on both endpoints.
func (s *Server) imageOutput(ctx context.Context, w http.ResponseWriter, responseFormat, outputFormat string, compression int) (imageOutputs, bool) {
	switch responseFormat {
	case "", "b64_json":
	case "url":
		badRequest(ctx, w, "response_format \"url\" is not supported; this server has nowhere to host an image, "+
			"so it returns b64_json")
		return imageOutputs{}, false
	default:
		badRequest(ctx, w, "response_format "+strconv.Quote(responseFormat)+
			" is not supported; this server returns b64_json")
		return imageOutputs{}, false
	}
	format := outputFormat
	if format == "" {
		format = "png"
	}
	switch format {
	case "png", "jpeg", "jpg":
	default:
		badRequest(ctx, w, "output_format "+strconv.Quote(format)+
			" is not supported; this server encodes png and jpeg")
		return imageOutputs{}, false
	}
	if compression < 0 || compression > 100 {
		badRequest(ctx, w, "output_compression is "+strconv.Itoa(compression)+", outside [0, 100]")
		return imageOutputs{}, false
	}
	return imageOutputs{format: format, compression: compression}, true
}

// imageCount resolves and bounds `n`.
func (s *Server) imageCount(ctx context.Context, w http.ResponseWriter, n int, stream bool) (int, bool) {
	if n == 0 {
		n = 1
	}
	if n < 1 || n > maxImages {
		badRequest(ctx, w, "n is "+strconv.Itoa(n)+"; this server renders 1 to "+strconv.Itoa(maxImages)+
			" images per request, serially, because each one is a full run of the model")
		return 0, false
	}
	if stream && n != 1 {
		badRequest(ctx, w, "n is "+strconv.Itoa(n)+" with stream: true; the event stream carries one image, "+
			"and its frames have no field that would say which")
		return 0, false
	}
	return n, true
}

// checkStreaming is the preview decoder's condition, which both endpoints
// share because a partial frame of an edit is a partial frame.
func (s *Server) checkStreaming(ctx context.Context, w http.ResponseWriter, geo ImageGeometry, stream bool, partials int) bool {
	if stream && !geo.Previews {
		// A 501 rather than a 400: the request is well formed and the loaded
		// backend cannot answer it.
		//
		// **The image backend this server ships always can**, because its
		// preview decoder is a fitted 64x4 matrix compiled in rather than a
		// checkpoint to load (IMAGE.md Q7). So this branch is reached only
		// by a backend that reports no previews, and the message stays
		// generic rather than naming a flag that no longer exists: what
		// would be wrong is a model without one, not a server started
		// without one.
		writeError(w, http.StatusNotImplemented, "not_implemented",
			"streaming an image needs a preview decoder, and the loaded image backend reports none; "+
				"GET /v1/models says so in image.previews. Send stream: false")
		return false
	}
	if partials < 0 || partials > geo.MaxPartials {
		badRequest(ctx, w, "partial_images is "+strconv.Itoa(partials)+"; this server sends 0 to "+
			strconv.Itoa(geo.MaxPartials)+" in-progress frames, because each one is a decode")
		return false
	}
	if partials > 0 && !stream {
		badRequest(ctx, w, "partial_images needs stream: true; there is nowhere to put an in-progress "+
			"frame in a single JSON response")
		return false
	}
	return true
}

// imageRun is one resolved render, and it is what /generations and /edits have
// in common once their own fields have been read.
//
// **An edit is a generation with reference images in front of it**, which is
// why there is one of these rather than two handlers: the references become
// the transformer's prefix and the target still starts from noise, so `n`,
// the seed, the container, the streaming and the partial frames all mean
// exactly what they already meant. Everything that differs between the two
// endpoints is in the one line at the bottom.
type imageRun struct {
	prompt        string
	width, height int
	steps         int
	seed          *int64
	n             int
	format        string
	compression   int
	stream        bool
	partials      int
	transparent   bool

	// init holds an edit's reference images and is empty on a generation.
	init []image.Image
}

// render answers one resolved request, buffered or over SSE.
func (s *Server) render(w http.ResponseWriter, r *http.Request, run *imageRun) {
	ctx := r.Context()
	if run.stream {
		s.streamImage(w, r, run)
		return
	}
	resp := ImageGenerationResponse{
		Created:      time.Now().Unix(),
		Size:         formatSize(run.width, run.height),
		OutputFormat: run.format,
		Data:         make([]ImageData, 0, run.n),
	}
	for i := 0; i < run.n; i++ {
		// A request that named a seed and asks for four images means four
		// different images, so the seed walks. Naming none walks nothing: the
		// backend draws one per call and reports it.
		seed := run.seed
		if seed != nil && i > 0 {
			next := *seed + int64(i)
			seed = &next
		}
		out, err := s.Image.Generate(r.Context(), &ImageRequest{
			Prompt: run.prompt, Width: run.width, Height: run.height,
			Steps: run.steps, Seed: seed, Transparent: run.transparent,
			Init: run.init,
		})
		if err != nil {
			backendError(ctx, w, "images", err)
			return
		}
		if out == nil || out.Image == nil {
			serverError(ctx, w, "images", errNoImage)
			return
		}
		body, err := encodeImage(out.Image, run.format, run.compression)
		if err != nil {
			serverError(ctx, w, "images", err)
			return
		}
		// The size is read back from the result rather than echoed from the
		// request: an edit that named none had it resolved by the backend,
		// from the last reference image's aspect ratio.
		resp.Size = formatSize(out.Width, out.Height)
		resp.Data = append(resp.Data, ImageData{
			B64JSON: base64.StdEncoding.EncodeToString(body),
			Seed:    out.Seed,
			Steps:   out.Steps,
		})
	}
	writeJSON(w, http.StatusOK, resp)
}

// ImageStreamEvent is one frame of a streamed image generation.
//
// It is OpenAI's envelope -- `type` names the event, `partial_image_index`
// numbers the in-progress frames -- plus the three fields this server has that
// theirs does not: `seed` and `steps`, without which a streamed image could not
// be asked for again, and `step`/`steps` on a partial, so a client watching one
// arrive knows how much is left. OpenAI's client reads the fields it knows and
// ignores these.
type ImageStreamEvent struct {
	Type      string `json:"type"`
	B64JSON   string `json:"b64_json,omitempty"`
	CreatedAt int64  `json:"created_at"`
	Size      string `json:"size,omitempty"`
	// OutputFormat is the container the frame is in, and it is the same for a
	// partial as for the finished image: a client decoding the stream should
	// not have to switch decoders halfway through it.
	OutputFormat string `json:"output_format,omitempty"`
	// PartialImageIndex is present on partials only, and counts from zero.
	PartialImageIndex *int `json:"partial_image_index,omitempty"`
	// Step and Steps are the denoising step a partial came from, out of how
	// many. Extensions.
	Step  *int  `json:"step,omitempty"`
	Steps int   `json:"steps,omitempty"`
	Seed  int64 `json:"seed,omitempty"`
}

// The event names, which are OpenAI's.
const (
	eventPartialImage = "image_generation.partial_image"
	eventImageDone    = "image_generation.completed"
)

// streamImage renders one image over Server-Sent Events.
//
// **There is no `[DONE]` sentinel**, which is the one place this differs from
// the chat stream next door. OpenAI's image events are named, and a named
// terminal event -- `image_generation.completed` -- is already unambiguous;
// the sentinel exists on the chat endpoint because its frames are unnamed and
// nothing else marks the last one.
//
// An error *after* the first frame cannot be a status code, because the status
// was committed when the stream opened. It goes out as an `error` event, which
// is what the chat stream does and what a client can act on; an error before
// any frame is still a 400 or a 500, which is why every check the handler can
// make happens before newSSE is called.
func (s *Server) streamImage(w http.ResponseWriter, r *http.Request, run *imageRun) {
	ctx := r.Context()
	size := formatSize(run.width, run.height)
	format := run.format
	str := newSSE(w)

	// The partial frames go out from inside the backend's run, on its
	// goroutine, before Generate returns. Encoding and writing them here is
	// what makes the stream a stream: there is no queue, and a client that
	// reads slowly slows the denoiser down rather than filling memory.
	sendPartial := func(p ImagePartial) error {
		if err := r.Context().Err(); err != nil {
			return err
		}
		body, err := encode(p.Image, format, run.compression, true)
		if err != nil {
			return err
		}
		idx, step := p.Index, p.Step
		return str.send(eventPartialImage, ImageStreamEvent{
			Type:              eventPartialImage,
			B64JSON:           base64.StdEncoding.EncodeToString(body),
			CreatedAt:         time.Now().Unix(),
			Size:              formatSize(p.Width, p.Height),
			OutputFormat:      format,
			PartialImageIndex: &idx,
			Step:              &step,
			Steps:             p.Steps,
		})
	}

	out, err := s.Image.Generate(r.Context(), &ImageRequest{
		Prompt: run.prompt, Width: run.width, Height: run.height,
		Steps: run.steps, Seed: run.seed, Transparent: run.transparent,
		Init:          run.init,
		PartialImages: run.partials, Partial: sendPartial,
	})
	if err != nil {
		streamError(ctx, str, "images", err)
		return
	}
	if out == nil || out.Image == nil {
		streamError(ctx, str, "images", errNoImage)
		return
	}
	body, err := encodeImage(out.Image, format, run.compression)
	if err != nil {
		streamError(ctx, str, "images", err)
		return
	}
	if out.Width > 0 && out.Height > 0 {
		size = formatSize(out.Width, out.Height)
	}
	_ = str.send(eventImageDone, ImageStreamEvent{
		Type:         eventImageDone,
		B64JSON:      base64.StdEncoding.EncodeToString(body),
		CreatedAt:    time.Now().Unix(),
		Size:         size,
		OutputFormat: format,
		Seed:         out.Seed,
		Steps:        out.Steps,
	})
}

// streamError reports a failure that happened after the status was committed.
// A client that hung up gets nothing, because there is nobody to tell.
func streamError(ctx context.Context, str *sse, where string, err error) {
	if errors.Is(err, context.Canceled) {
		logf(ctx, "%s: client cancelled", where)
		return
	}
	logf(ctx, "%s: %v", where, err)
	_ = str.send("error", errorResponse{Error: errorBody{Message: err.Error(), Type: "server_error"}})
}

// handleImageEdit edits a picture: conditional generation over
// Qwen-Image-2.1's vision tower (IMAGE.md Q8).
//
// **What an edit is, in four lines.** Every reference image is resized to the
// model's condition area and encoded twice -- by a 27-layer vision tower into
// context the prompt's tokens absorb, and by the VAE into latents that become
// rows of the transformer's own sequence. The target then starts from pure
// noise like any generation and runs the whole schedule, attending over that
// prefix at every step. Everything else here is the generation endpoint's,
// including the streaming.
//
// **`strength` is gone, and it is gone because the mechanism is.** Under
// SDEdit an edit was a truncated generation and `strength` was how much of
// the schedule to run; here the reference conditions the whole trajectory
// rather than seeding it, every step runs, and an edit costs slightly *more*
// than a generation rather than less. A request that still sends one is
// refused saying so, rather than having it quietly ignored.
//
// One thing OpenAI's endpoint has that this one still refuses: a `mask`.
// This model can do local edits -- that is what its reference images and
// annotations are for -- but how a separate mask is fed is not in the
// diffusers implementation (IMAGE.md Q-o3), and answering with a whole-image
// edit would be a picture that silently did something else.
//
// The size is the one place this differs from /generations in kind. A
// generation with no `size` takes the server's default; an edit with no
// `size` takes **the last reference image's aspect ratio** at the model's
// condition area, which is diffusers' own rule and the only answer that does
// not silently reframe what was sent.
func (s *Server) handleImageEdit(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Image == nil {
		notLoaded(ctx, w, "image generation", "-image")
		return
	}
	geo := s.Image.Geometry()
	if !geo.Edits {
		// A 501 naming the flag, because editing is residency: the 27-layer
		// vision tower and the VAE's encoder are staged or they are not, and
		// each reference image then costs prefix KV cache besides.
		notLoaded(ctx, w, "image editing", "-edits N")
		return
	}

	var req ImageEditRequest
	if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
		if !parseImageEditForm(w, r, &req) {
			return
		}
	} else if !decodeJSON(ctx, w, r, &req) {
		return
	}

	if strings.TrimSpace(req.Prompt) == "" {
		badRequest(ctx, w, "prompt is empty; an edit is a prompt applied to a picture, so there is nothing to apply")
		return
	}
	if req.Mask != "" {
		writeError(w, http.StatusNotImplemented, "not_implemented",
			"mask is not implemented: Qwen-Image-2.1 conditions on whole reference images, and how a "+
				"separate mask is fed to it is not in the reference implementation (IMAGE.md Q-o3). "+
				"Without one the whole picture is edited")
		return
	}
	if req.Strength != 0 {
		badRequest(ctx, w, "strength is not a knob this model has: Qwen-Image-2.1 edits by conditional "+
			"generation rather than by SDEdit, so the reference conditions the whole trajectory instead "+
			"of seeding it, and every denoising step runs. Send the request without it")
		return
	}
	if len(req.Image) == 0 {
		badRequest(ctx, w, "no image: send multipart/form-data with an 'image' part, or JSON with a base64 'image'")
		return
	}
	if len(req.Image) > geo.MaxRefs {
		badRequest(ctx, w, strconv.Itoa(len(req.Image))+" reference images; this server is started for "+
			strconv.Itoa(geo.MaxRefs)+". Every reference becomes rows of the transformer's prefix and its "+
			"own share of the prefix KV cache, so the number is fixed when the server starts")
		return
	}
	refs := make([]image.Image, 0, len(req.Image))
	var kind string
	for i, field := range req.Image {
		raw, err := decodeImageField(field)
		if err != nil {
			badRequest(ctx, w, "image "+strconv.Itoa(i)+": "+err.Error())
			return
		}
		img, k, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			badRequest(ctx, w, "image "+strconv.Itoa(i)+": this server decodes png and jpeg: "+err.Error())
			return
		}
		refs = append(refs, img)
		kind = k
	}
	init := refs[len(refs)-1]

	out, ok := s.imageOutput(ctx, w, req.ResponseFormat, req.OutputFormat, req.OutputCompression)
	if !ok {
		return
	}
	n, ok := s.imageCount(ctx, w, req.N, req.Stream)
	if !ok {
		return
	}
	if !s.checkStreaming(ctx, w, geo, req.Stream, req.PartialImages) {
		return
	}

	// The geometry. With a `size` this is the generation endpoint's rule;
	// with none it is left to the backend, which takes the last reference's
	// aspect ratio at the model's condition area -- diffusers' own rule, and
	// not something the HTTP layer can compute, since the condition area is
	// the backend's.
	b := init.Bounds()
	var width, height int
	if (req.Size != "" && req.Size != "auto") || req.AspectRatio != "" {
		var err error
		width, height, err = resolveSize(req.Size, req.AspectRatio, geo)
		if err != nil {
			badRequest(ctx, w, err.Error())
			return
		}
	}
	logf(ctx, "images/edits: %d %s reference(s), last %dx%d, out %s",
		len(refs), kind, b.Dx(), b.Dy(), formatSize(width, height))

	s.render(w, r, &imageRun{
		prompt: req.Prompt, width: width, height: height, steps: req.Steps, seed: req.Seed,
		n: n, format: out.format, compression: out.compression,
		stream: req.Stream, partials: req.PartialImages,
		init: refs,
	})
}

// decodeImageField reads the base64 an `image` field carries, accepting a
// `data:` URL as well as the bare payload. A browser that built the field from
// a FileReader sends the prefix, and stripping it here is cheaper than every
// client remembering not to.
func decodeImageField(v string) ([]byte, error) {
	v = strings.TrimSpace(v)
	if strings.HasPrefix(v, "data:") {
		if _, after, ok := strings.Cut(v, ","); ok {
			v = after
		}
	}
	raw, err := base64.StdEncoding.DecodeString(v)
	if err != nil {
		return nil, fmt.Errorf("the field is not base64: %w", err)
	}
	if len(raw) == 0 {
		return nil, errors.New("the field is empty")
	}
	return raw, nil
}

// parseImageEditForm reads the multipart encoding OpenAI's clients send into
// the same struct the JSON one fills.
func parseImageEditForm(w http.ResponseWriter, r *http.Request, req *ImageEditRequest) bool {
	ctx := r.Context()
	// ParseMultipartForm's argument is how much it keeps in memory; the rest
	// spills to a temporary file, and Server.limitBody is what bounds the
	// upload.
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		if tooLarge(ctx, w, err) {
			return false
		}
		badRequest(ctx, w, "malformed multipart body: "+err.Error())
		return false
	}
	defer func() { _ = r.MultipartForm.RemoveAll() }()

	// OpenAI's clients send the part as `image` and, when there are several,
	// as `image[]`. Both are read so that a count of two is a 400 saying so
	// rather than a silently dropped second picture.
	var parts []string
	for _, field := range []string{"image", "image[]"} {
		for _, fh := range r.MultipartForm.File[field] {
			f, err := fh.Open()
			if err != nil {
				badRequest(ctx, w, "reading the '"+field+"' part: "+err.Error())
				return false
			}
			raw, err := io.ReadAll(f)
			_ = f.Close()
			if err != nil {
				if tooLarge(ctx, w, err) {
					return false
				}
				badRequest(ctx, w, "reading the '"+field+"' part: "+err.Error())
				return false
			}
			// Back to base64 so that the two encodings converge on one struct
			// rather than on two code paths. An image is a few hundred KB and
			// this is one allocation on a request that is about to spend
			// seconds on a GPU.
			parts = append(parts, base64.StdEncoding.EncodeToString(raw))
		}
	}
	req.Image = parts
	if fh := r.MultipartForm.File["mask"]; len(fh) > 0 {
		// Only its presence matters: the handler refuses it either way, and
		// reading a mask it will not use would be a megabyte for nothing.
		req.Mask = "(a mask part)"
	}

	req.Model = r.FormValue("model")
	req.Prompt = r.FormValue("prompt")
	req.Size = r.FormValue("size")
	req.AspectRatio = r.FormValue("aspect_ratio")
	req.ResponseFormat = r.FormValue("response_format")
	req.OutputFormat = r.FormValue("output_format")
	req.User = r.FormValue("user")
	for _, f := range []struct {
		name string
		dst  *int
	}{{"n", &req.N}, {"steps", &req.Steps}, {"output_compression", &req.OutputCompression},
		{"partial_images", &req.PartialImages}} {
		if v := r.FormValue(f.name); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil {
				badRequest(ctx, w, f.name+" is not a number: "+v)
				return false
			}
			*f.dst = n
		}
	}
	if v := r.FormValue("strength"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			badRequest(ctx, w, "strength is not a number: "+v)
			return false
		}
		req.Strength = f
	}
	if v := r.FormValue("seed"); v != "" {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			badRequest(ctx, w, "seed is not a number: "+v)
			return false
		}
		req.Seed = &n
	}
	if v := r.FormValue("stream"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			badRequest(ctx, w, "stream is not a boolean: "+v)
			return false
		}
		req.Stream = b
	}
	return true
}

// resolveSize turns the request's size or aspect ratio into pixels, and is the
// whole of this endpoint's geometry policy.
//
// The checks are ordered by how much they tell the client. A side that is not
// a multiple of 16 is a typo; an area past the ceiling is a server that was
// started smaller, and the message says which so the operator can be asked for
// a bigger one.
func resolveSize(size, ratio string, geo ImageGeometry) (width, height int, err error) {
	switch {
	case size != "" && size != "auto":
		if width, height, err = parseSize(size); err != nil {
			return 0, 0, err
		}
	case ratio != "":
		if width, height, err = fitRatio(ratio, geo); err != nil {
			return 0, 0, err
		}
	default:
		return geo.Width, geo.Height, nil
	}

	m := geo.Multiple
	if m > 1 && (width%m != 0 || height%m != 0) {
		return 0, 0, fmt.Errorf("size %s: both sides must be a multiple of %d", formatSize(width, height), m)
	}
	if width > geo.MaxWidth || height > geo.MaxHeight {
		return 0, 0, fmt.Errorf(
			"size %s: this server's arenas were built for %s, and neither side may be past it "+
				"-- a ceiling is what was staged, not a budget per request",
			formatSize(width, height), formatSize(geo.MaxWidth, geo.MaxHeight))
	}
	return width, height, nil
}

// parseSize reads OpenAI's "1024x1024".
func parseSize(s string) (int, int, error) {
	w, h, ok := strings.Cut(strings.ToLower(strings.TrimSpace(s)), "x")
	if !ok {
		return 0, 0, fmt.Errorf("size %s is not WIDTHxHEIGHT or \"auto\"", strconv.Quote(s))
	}
	width, err := strconv.Atoi(strings.TrimSpace(w))
	if err != nil || width <= 0 {
		return 0, 0, fmt.Errorf("size %s: %s is not a width", strconv.Quote(s), strconv.Quote(w))
	}
	height, err := strconv.Atoi(strings.TrimSpace(h))
	if err != nil || height <= 0 {
		return 0, 0, fmt.Errorf("size %s: %s is not a height", strconv.Quote(s), strconv.Quote(h))
	}
	return width, height, nil
}

// fitRatio is the aspect-ratio extension: the largest image of that shape
// that fits inside both side ceilings, rounded down to the size multiple.
//
// Rounding *down* is what keeps the answer inside the ceiling after the
// rounding. The ratio is honoured and the size gives, which is the right way
// round: a client that asked for 16:9 wants 16:9, and how many pixels this
// particular server has is not something they said anything about.
func fitRatio(ratio string, geo ImageGeometry) (int, int, error) {
	a, b, ok := strings.Cut(strings.TrimSpace(ratio), ":")
	if !ok {
		return 0, 0, fmt.Errorf("aspect_ratio %s is not W:H", strconv.Quote(ratio))
	}
	rw, errW := strconv.ParseFloat(strings.TrimSpace(a), 64)
	rh, errH := strconv.ParseFloat(strings.TrimSpace(b), 64)
	if errW != nil || errH != nil || rw <= 0 || rh <= 0 {
		return 0, 0, fmt.Errorf("aspect_ratio %s is not W:H with both sides positive", strconv.Quote(ratio))
	}
	width, height, err := fitAspect(rw, rh, geo)
	if err != nil {
		return 0, 0, fmt.Errorf("aspect_ratio %s: %w", strconv.Quote(ratio), err)
	}
	return width, height, nil
}

// fitAspect is the arithmetic behind fitRatio: the largest image of that shape
// that fits inside both ceilings, rounded down to the size multiple.
func fitAspect(rw, rh float64, geo ImageGeometry) (int, int, error) {
	return scaleAspect(rw, rh, geo, math.Min(float64(geo.MaxWidth)/rw, float64(geo.MaxHeight)/rh))
}

// fitBounds is fitAspect for a picture rather than for a ratio: the same shape
// at the same size, shrunk only as far as the ceiling requires.
//
// **It never scales up**, which is the difference and the reason it is its own
// function. A ratio a client typed carries no size, so "16:9" has to mean the
// largest 16:9 this server does; a 512x512 picture already has one, and
// upsampling it to a 1024x1024 ceiling would cost four times the tokens to
// paint detail the input never had.
func fitBounds(w, h int, geo ImageGeometry) (int, int, error) {
	rw, rh := float64(w), float64(h)
	scale := math.Min(float64(geo.MaxWidth)/rw, float64(geo.MaxHeight)/rh)
	return scaleAspect(rw, rh, geo, math.Min(scale, 1))
}

func scaleAspect(rw, rh float64, geo ImageGeometry, scale float64) (int, int, error) {
	m := geo.Multiple
	if m < 1 {
		m = 1
	}
	width := int(rw*scale) / m * m
	height := int(rh*scale) / m * m
	if width < m || height < m {
		return 0, 0, fmt.Errorf("%g:%g does not fit a %s image in steps of %d",
			rw, rh, formatSize(geo.MaxWidth, geo.MaxHeight), m)
	}
	return width, height, nil
}

// formatSize is OpenAI's spelling of a size, which is the one a client can
// send back.
func formatSize(width, height int) string {
	if width <= 0 || height <= 0 {
		return ""
	}
	return strconv.Itoa(width) + "x" + strconv.Itoa(height)
}

// encodeImage puts the picture in a container. JPEG's quality is
// output_compression where the request gave one and 90 where it did not,
// which is high enough that a client comparing formats is comparing the
// containers and not this default.
//
// **fast trades bytes for latency, and it is only ever set on a partial
// frame.** Measured on a 1024x1024 image on this machine: Go's default PNG
// encoder is 344 ms, `png.BestSpeed` is 60 ms for a file 15% larger, and JPEG
// at quality 90 is 23 ms. Those are not rounding errors next to what a preview
// costs to produce -- the taef1 decode behind one is 87 ms -- so the default
// encoder would make the *container* three quarters of a preview's price and
// put a frame 430 ms behind the step it came from. A finished image keeps the
// slower encoder, because it is the deliverable and it is written once.
func encodeImage(img image.Image, format string, compression int) ([]byte, error) {
	return encode(img, format, compression, false)
}

func encode(img image.Image, format string, compression int, fast bool) ([]byte, error) {
	var buf bytes.Buffer
	switch format {
	case "jpeg", "jpg":
		q := compression
		if q == 0 {
			q = defaultJPEGQuality
		}
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: q}); err != nil {
			return nil, err
		}
	default:
		enc := png.Encoder{}
		if fast {
			enc.CompressionLevel = png.BestSpeed
		}
		if err := enc.Encode(&buf, img); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

const defaultJPEGQuality = 90

// errNoImage is its own value so that a backend that returned nothing without
// saying why is distinguishable in the log from one that returned an error.
var errNoImage = errors.New("the image backend returned no image")
