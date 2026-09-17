package api

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
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
	Steps  int    `json:"steps,omitempty"`
	Stream bool   `json:"stream,omitempty"`
	User   string `json:"user,omitempty"`
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

// handleImageGeneration renders a prompt.
//
// It does not stream. OpenAI's streaming image endpoint sends partial images
// as the denoiser passes them, and the pipeline does produce the latent after
// every step -- but decoding one is the full VAE, 0.8 s against a 1.7 s step,
// which would make a preview cost half the image. What that wants is the
// small decoder GOALS.md names for it (madebyollin/taef1), and until that is
// loaded a `stream: true` is refused rather than answered with one frame at
// the end.
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
	if req.Stream {
		badRequest(w, "streaming image generation is not implemented; "+
			"decoding a preview costs the whole VAE, and the small decoder that would make it cheap is not loaded")
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

	geo := s.Image.Geometry()
	width, height, err := resolveSize(req.Size, req.AspectRatio, geo)
	if err != nil {
		badRequest(w, err.Error())
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
func encodeImage(img image.Image, format string, compression int) ([]byte, error) {
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
		if err := png.Encode(&buf, img); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

const defaultJPEGQuality = 90

// errNoImage is its own value so that a backend that returned nothing without
// saying why is distinguishable in the log from one that returned an error.
var errNoImage = errors.New("the image backend returned no image")
