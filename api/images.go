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
	"log"
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
// took a pipeline change to wire up rather than a handler: every arena in
// `zimage/pipeline` used to be built for one width and one height, so a size
// was a residency question. It is now a ceiling, and a size inside it is a
// parameter -- see ImageGeometry.
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
	// Steps is an extension: the denoising schedule's length. The checkpoint
	// is a turbo distillation whose NFE is 8, so this is a knob for looking at
	// the trade rather than one a client should normally turn.
	Steps int `json:"steps,omitempty"`
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

// ImageEditRequest is OpenAI's edit request. It is parsed and refused; see
// handleImageEdit.
type ImageEditRequest struct {
	host        string
	AspectRatio string   `json:"aspect_ratio,omitempty"`
	Model       string   `json:"model,omitempty"`
	Image       []string `json:"image"`
	Prompt      string   `json:"prompt"`
	Size        string   `json:"size,omitempty"`
	Stream      bool     `json:"stream,omitempty"`
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
	if s.Image == nil {
		notLoaded(w, "image generation", "-image")
		return
	}
	var req ImageGenerationRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.Prompt) == "" {
		badRequest(w, "prompt is empty")
		return
	}
	geo := s.Image.Geometry()
	if req.Stream && !geo.Previews {
		// A 501 rather than a 400, and the same kind of answer -image itself
		// gives: the request is well formed and the server was started
		// without the thing that would answer it. writeError rather than
		// notLoaded because what is missing is a decoder inside a model that
		// *is* loaded, which its sentence does not fit.
		writeError(w, http.StatusNotImplemented, "not_implemented",
			"streaming image generation needs the preview decoder, which this server was started without; "+
				"pass -preview with a madebyollin/taef1 checkpoint (it is what makes an in-progress frame "+
				"cost 87 ms instead of the full VAE's 876)")
		return
	}
	if req.PartialImages < 0 || req.PartialImages > geo.MaxPartials {
		badRequest(w, "partial_images is "+strconv.Itoa(req.PartialImages)+"; this server sends 0 to "+
			strconv.Itoa(geo.MaxPartials)+" in-progress frames, because each one is a decode")
		return
	}
	if req.PartialImages > 0 && !req.Stream {
		badRequest(w, "partial_images needs stream: true; there is nowhere to put an in-progress "+
			"frame in a single JSON response")
		return
	}
	n := req.N
	if n == 0 {
		n = 1
	}
	if n < 1 || n > maxImages {
		badRequest(w, "n is "+strconv.Itoa(n)+"; this server renders 1 to "+strconv.Itoa(maxImages)+
			" images per request, serially, because each one is a full run of the model")
		return
	}
	switch req.ResponseFormat {
	case "", "b64_json":
	case "url":
		badRequest(w, "response_format \"url\" is not supported; this server has nowhere to host an image, "+
			"so it returns b64_json")
		return
	default:
		badRequest(w, "response_format "+strconv.Quote(req.ResponseFormat)+
			" is not supported; this server returns b64_json")
		return
	}
	format := req.OutputFormat
	if format == "" {
		format = "png"
	}
	switch format {
	case "png", "jpeg", "jpg":
	default:
		badRequest(w, "output_format "+strconv.Quote(format)+
			" is not supported; this server encodes png and jpeg")
		return
	}
	if req.OutputCompression < 0 || req.OutputCompression > 100 {
		badRequest(w, "output_compression is "+strconv.Itoa(req.OutputCompression)+", outside [0, 100]")
		return
	}

	if req.Stream && n != 1 {
		badRequest(w, "n is "+strconv.Itoa(n)+" with stream: true; the event stream carries one image, "+
			"and its frames have no field that would say which")
		return
	}

	width, height, err := resolveSize(req.Size, req.AspectRatio, geo)
	if err != nil {
		badRequest(w, err.Error())
		return
	}

	if req.Stream {
		s.streamImage(w, r, &req, width, height, format)
		return
	}

	resp := ImageGenerationResponse{
		Created:      time.Now().Unix(),
		Size:         formatSize(width, height),
		OutputFormat: format,
		Data:         make([]ImageData, 0, n),
	}
	for i := 0; i < n; i++ {
		// A request that named a seed and asks for four images means four
		// different images, so the seed walks. Naming none walks nothing: the
		// backend draws one per call and reports it.
		seed := req.Seed
		if seed != nil && i > 0 {
			next := *seed + int64(i)
			seed = &next
		}
		out, err := s.Image.Generate(r.Context(), &ImageRequest{
			Prompt: req.Prompt, Width: width, Height: height, Steps: req.Steps, Seed: seed,
		})
		if err != nil {
			backendError(w, "images", err)
			return
		}
		if out == nil || out.Image == nil {
			serverError(w, "images", errNoImage)
			return
		}
		body, err := encodeImage(out.Image, format, req.OutputCompression)
		if err != nil {
			serverError(w, "images", err)
			return
		}
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
func (s *Server) streamImage(w http.ResponseWriter, r *http.Request, req *ImageGenerationRequest,
	width, height int, format string) {
	size := formatSize(width, height)
	str := newSSE(w)

	// The partial frames go out from inside the backend's run, on its
	// goroutine, before Generate returns. Encoding and writing them here is
	// what makes the stream a stream: there is no queue, and a client that
	// reads slowly slows the denoiser down rather than filling memory.
	sendPartial := func(p ImagePartial) error {
		if err := r.Context().Err(); err != nil {
			return err
		}
		body, err := encode(p.Image, format, req.OutputCompression, true)
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
		Prompt: req.Prompt, Width: width, Height: height, Steps: req.Steps, Seed: req.Seed,
		PartialImages: req.PartialImages, Partial: sendPartial,
	})
	if err != nil {
		streamError(str, "images", err)
		return
	}
	if out == nil || out.Image == nil {
		streamError(str, "images", errNoImage)
		return
	}
	body, err := encodeImage(out.Image, format, req.OutputCompression)
	if err != nil {
		streamError(str, "images", err)
		return
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
func streamError(str *sse, where string, err error) {
	if errors.Is(err, context.Canceled) {
		log.Printf("api: %s: client cancelled", where)
		return
	}
	log.Printf("api: %s: %v", where, err)
	_ = str.send("error", errorResponse{Error: errorBody{Message: err.Error(), Type: "server_error"}})
}

// handleImageEdit is the one endpoint on this server that no flag fixes.
//
// An edit starts from a picture, and bringing a picture into the transformer's
// latent space is the VAE's *encoder*. **Its weights are in the checkpoint** --
// z-image ships a stock Flux `AutoencoderKL`, and the 34 M encoder parameters
// are in the same file `-image` already opens for the decoder's 50 M -- so
// this is a missing *port* and not a missing download. `zimage/vae`
// implements the decoder only, because generating never needed to go that
// way. Saying which of the two it is is more use to a caller than a 501 that
// names a flag they cannot pass.
func (s *Server) handleImageEdit(w http.ResponseWriter, _ *http.Request) {
	writeError(w, http.StatusNotImplemented, "not_implemented",
		"/v1/images/edits is not implemented: an edit needs the VAE's encoder to bring the input image "+
			"into the latent space, and this server implements the decoder only. The encoder's weights "+
			"are in the checkpoint; the port is not written, so no flag turns this on")
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
				"-- the same area in a taller shape does not fit",
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

	m := geo.Multiple
	if m < 1 {
		m = 1
	}
	// The largest scale at which neither side is past its ceiling: the
	// smaller of the two each side would allow on its own.
	scale := math.Min(float64(geo.MaxWidth)/rw, float64(geo.MaxHeight)/rh)
	width := int(rw*scale) / m * m
	height := int(rh*scale) / m * m
	if width < m || height < m {
		return 0, 0, fmt.Errorf("aspect_ratio %s does not fit a %s image in steps of %d",
			strconv.Quote(ratio), formatSize(geo.MaxWidth, geo.MaxHeight), m)
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
