package gemma4

// A request's images between the HTTP layer and the pass (R10).

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"sync"

	"strix-halo-vulkan/llm/pixels"
)

// The image placeholders as the processor expands `<|image|>`: one per soft
// token, between the begin and end markers.
const (
	imageBegin = "<|image>"
	imageToken = "<|image|>"
	imageEnd   = "<image|>"
)

// ImageText is the user turn's head for images of these soft-token counts,
// in order: what Gemma4Processor puts where the template wrote <|image|>.
func ImageText(soft []int) string {
	var b strings.Builder
	for _, n := range soft {
		b.WriteString(imageBegin)
		b.WriteString(strings.Repeat(imageToken, n))
		b.WriteString(imageEnd)
	}
	return b.String()
}

// Picture is one request image: its patches, then its features once
// EncodePictures has run the tower.
type Picture struct {
	Key     [32]byte // sha256 of the encoded file: the feature cache's key
	Patches *Patches
	Feats   []float32 // [Patches.Soft][hidden]
}

// NewPicture decodes and patchifies an encoded image (PNG, JPEG, GIF), on
// the host.
func (v *VisionConfig) NewPicture(data []byte) (*Picture, error) {
	img, _, err := pixels.Decode(data)
	if err != nil {
		return nil, err
	}
	p, err := v.Patchify(img)
	if err != nil {
		return nil, err
	}
	return &Picture{Key: sha256.Sum256(data), Patches: p}, nil
}

// featureCache keeps the last images' features (~3.2 MB each): a client
// that asks several requests about one image pays the tower once.
type featureCache struct {
	mu    sync.Mutex
	order [][32]byte
	m     map[[32]byte][]float32
}

const featureCacheSize = 32

func (c *featureCache) get(k [32]byte) ([]float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	f, ok := c.m[k]
	return f, ok
}

func (c *featureCache) put(k [32]byte, f []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.m == nil {
		c.m = map[[32]byte][]float32{}
	}
	if _, ok := c.m[k]; ok {
		return
	}
	if len(c.order) == featureCacheSize {
		delete(c.m, c.order[0])
		c.order = c.order[1:]
	}
	c.order = append(c.order, k)
	c.m[k] = f
}

// Cached fills a picture's features from the cache; false if it must run
// the tower. It takes no device.
func (g *GPU) Cached(p *Picture) bool {
	if f, ok := g.feats.get(p.Key); ok {
		p.Feats = f
		return true
	}
	return false
}

// EncodePictures runs the tower over every picture whose features are not
// cached. The caller holds the device.
func (g *GPU) EncodePictures(pics []*Picture) error {
	if len(pics) > 0 && g.Vis == nil {
		return fmt.Errorf("gemma4: no vision tower staged")
	}
	for _, p := range pics {
		if p.Feats != nil || g.Cached(p) {
			continue
		}
		f, err := g.Vis.Encode(p.Patches, nil)
		if err != nil {
			return err
		}
		p.Feats = f
		g.feats.put(p.Key, f)
	}
	return nil
}
