package util

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Image URLs, read the way OpenAI's API reads them.
//
// **This server fetches an image URL wherever OpenAI's API does**, and only
// there: a chat `image_url`, a Responses `input_image`, an images/edits
// `images[].image_url`, a video's `input_reference.image_url` (and
// SGLang's `conditions[].uri`, which SGLang fetches too), and Anthropic's
// `url` image source. Each of those takes "a fully qualified URL or a
// base64-encoded data URL", so both forms land here and come back as bytes.
// A `file_id` is a different matter — there is no Files API here — and each
// door refuses it itself.
//
// The fetch is a plain GET with caps rather than a policy: http and https
// only, MaxImageBytes of body, FetchTimeout end to end, and a 2xx status.
// It fetches whatever the client names, the local network included, as a
// server speaking OpenAI's API does; a deployment that must not reach its
// own LAN on a client's say-so has to put that boundary in front of it.

// MaxImageBytes caps a fetched image: OpenAI's own per-image limit is
// 50 MB, and nothing this server decodes needs more.
const MaxImageBytes = 50 << 20

// FetchTimeout bounds one fetch, connection to last byte.
const FetchTimeout = 60 * time.Second

// ErrImageURL wraps every failure to read an image URL. It is the
// client's: a bad URL, a host that does not answer, a body too large.
var ErrImageURL = errors.New("image url")

var fetchClient = &http.Client{
	Timeout: FetchTimeout,
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 5 {
			return errors.New("more than 5 redirects")
		}
		if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
			return fmt.Errorf("a redirect to a %s URL", req.URL.Scheme)
		}
		return nil
	},
}

// FetchImage returns the bytes an image URL names: a data: URL's base64
// payload, or an http(s) URL's body.
func FetchImage(ctx context.Context, url string) ([]byte, error) {
	url = strings.TrimSpace(url)
	if strings.HasPrefix(url, "data:") {
		meta, payload, ok := strings.Cut(url[len("data:"):], ",")
		if !ok || !strings.HasSuffix(meta, ";base64") {
			return nil, fmt.Errorf("%w: a data: URL that is not base64", ErrImageURL)
		}
		b, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			if b, err = base64.RawStdEncoding.DecodeString(payload); err != nil {
				return nil, fmt.Errorf("%w: a data: URL whose payload is not base64: %v", ErrImageURL, err)
			}
		}
		if len(b) == 0 {
			return nil, fmt.Errorf("%w: an empty data: URL", ErrImageURL)
		}
		return b, nil
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		kind := "a schemeless"
		if i := strings.Index(url, ":"); i > 0 && i < 12 {
			kind = "a " + url[:i]
		}
		return nil, fmt.Errorf("%w: %s URL; an image URL is http(s) or a base64 data: URL", ErrImageURL, kind)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrImageURL, err)
	}
	req.Header.Set("Accept", "image/*")
	resp, err := fetchClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: fetching %s: %v", ErrImageURL, redact(url), err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("%w: fetching %s: %s", ErrImageURL, redact(url), resp.Status)
	}
	if resp.ContentLength > MaxImageBytes {
		return nil, fmt.Errorf("%w: %s is %d bytes; an image is at most %d MB", ErrImageURL, redact(url), resp.ContentLength, MaxImageBytes>>20)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, MaxImageBytes+1))
	if err != nil {
		return nil, fmt.Errorf("%w: reading %s: %v", ErrImageURL, redact(url), err)
	}
	if len(b) > MaxImageBytes {
		return nil, fmt.Errorf("%w: %s is past %d MB; an image is at most that", ErrImageURL, redact(url), MaxImageBytes>>20)
	}
	if len(b) == 0 {
		return nil, fmt.Errorf("%w: %s is empty", ErrImageURL, redact(url))
	}
	return b, nil
}

// redact is a URL for an error message: without its query, which is where
// signed URLs carry their credentials.
func redact(url string) string {
	if i := strings.IndexAny(url, "?#"); i >= 0 {
		return url[:i] + "?…"
	}
	return url
}
