package backend

// The language model's eyes (LLM-VISION.md V8, V9): a request's images,
// from the bytes a client sent to the rows the graph reads.
//
// An image is decoded and preprocessed exactly as HF's processor would
// (llm/pixels), encoded by the vision tower on the device (qimage/vision,
// the mmproj's weights), and handed to the prompt as one PromptImage, which
// llm.ExpandImages places at its pad. The tower runs under the device lock,
// between the scheduler's units, before the request takes a slot.
//
// **Scheduling sees an image as keys, not ids.** Every cell of every image
// holds the same pad id, so two different pictures would be the same
// prefix to a scheduler comparing ids, and a slot would "continue" a
// conversation about a cat with a picture of a dog. The scheduler compares
// keys instead: a text cell's key is its id, and an image cell's is a
// negative number derived from the image's hash and the cell's index in it.
// The graph itself refuses a checkpoint whose images differ (Restore compares
// spans), so the keys are what makes reuse *happen* for the same picture, and
// the graph is what makes it *safe*.

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"
	"sync"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/qimage/vision"
	"strix-halo-vulkan/vk"
)

// imagePad is `<|image_pad|>`, the id every image cell holds.
const imagePad int32 = 248056

const (
	// defaultVisionTokens is an image's token cap when the server names
	// none: 2048² of pixels, 3.1 s of tower at 4096 tokens (V4's curve).
	// HF allows 16 384, which is 33 s of tower and more rows than a
	// -llm-batch of 4096 can take in one pass.
	defaultVisionTokens = 4096
	defaultVisionImages = 8

	// towerSlice is how long the tower holds the device before it lets a
	// waiting unit run (LLM-VISION.md V12). An image at the 4096-token cap
	// is ~2.1 s of tower, and held whole it delayed a voice command's first
	// token by all of it (0.22 s -> 1.97 s). In 50 ms slices the voice
	// command's TTFT is 0.28 s and it decodes at ~10 tok/s through the tower,
	// which pays +0.7 s for it. 100 ms was +0.5 s, but ~6 tok/s for the voice.
	// LLM_TOWER_SLICE overrides it, and 0 holds the device for the whole
	// image, which is the control.
	towerSlice = 50 * time.Millisecond
)

// llmVision is the tower, staged once for the largest image a request may
// send.
type llmVision struct {
	gpu       *vision.GPU
	cfg       vision.Config
	proc      pixels.Processor
	maxImages int
	// slice is towerSlice or LLM_TOWER_SLICE. tower is held through a whole
	// Forward. The device lock no longer is: a yield lets another request's
	// Do in, and that must not be a second image into the same arenas.
	slice time.Duration
	tower sync.Mutex

	// cache holds encoded images by the hash of their bytes. A chat client
	// resends the whole conversation every turn, picture included, and the
	// slot already reuses the prefill; without this the tower would still
	// run again on every follow-up (0.77 s for a 1920x1080 photo). Bounded
	// by the rows' memory, least recently used out first.
	mu       sync.Mutex
	cache    map[uint64]*cachedImage
	cached   int // bytes of rows held
	maxCache int
	tick     uint64
}

type cachedImage struct {
	img  llm.PromptImage
	used uint64
}

// visionCacheBytes is how much encoded image the cache keeps: ~12 photos at
// 2048 tokens, or 25 at 1024.
const visionCacheBytes = 256 << 20

func (v *llmVision) lookup(hash uint64) (llm.PromptImage, bool) {
	v.mu.Lock()
	defer v.mu.Unlock()
	c, ok := v.cache[hash]
	if !ok {
		return llm.PromptImage{}, false
	}
	v.tick++
	c.used = v.tick
	return c.img, true
}

func (v *llmVision) store(img llm.PromptImage) {
	size := 4 * len(img.Embd)
	v.mu.Lock()
	defer v.mu.Unlock()
	if size > v.maxCache {
		return
	}
	if v.cache == nil {
		v.cache = map[uint64]*cachedImage{}
	}
	for v.cached+size > v.maxCache {
		var oldest uint64
		first := true
		for h, c := range v.cache {
			if first || c.used < v.cache[oldest].used {
				oldest, first = h, false
			}
		}
		v.cached -= 4 * len(v.cache[oldest].img.Embd)
		delete(v.cache, oldest)
	}
	v.tick++
	v.cache[img.Hash] = &cachedImage{img: img, used: v.tick}
	v.cached += size
}

// newLLMVision stages the tower from an mmproj for images of up to
// maxTokens merged tokens.
func newLLMVision(dev *Device, mmproj string, maxTokens, maxImages int) (*llmVision, error) {
	_, tower, err := vision.LoadGGUF(mmproj, 0)
	if err != nil {
		return nil, fmt.Errorf("backend: vision tower %s: %w", mmproj, err)
	}
	merge := tower.Cfg.SpatialMergeSize * tower.Cfg.SpatialMergeSize
	v := &llmVision{cfg: tower.Cfg, maxImages: maxImages, proc: pixels.Default, maxCache: visionCacheBytes,
		slice: towerSlice}
	if s, ok := os.LookupEnv("LLM_TOWER_SLICE"); ok {
		if v.slice, err = time.ParseDuration(s); err != nil {
			return nil, fmt.Errorf("backend: LLM_TOWER_SLICE: %w", err)
		}
	}
	// The processor's own ceiling, lowered to the server's: a token is
	// 32 x 32 pixels, so maxTokens of them is maxTokens*1024 pixels.
	side := v.cfg.PatchSize * v.cfg.SpatialMergeSize
	v.proc.MaxPixels = min(v.proc.MaxPixels, maxTokens*side*side)
	err = dev.Do(func(d *vk.Device) error {
		v.gpu, err = vision.NewGPU(d, tower, maxTokens*merge)
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("backend: staging the vision tower: %w", err)
	}
	return v, nil
}

func (v *llmVision) close() {
	if v != nil && v.gpu != nil {
		v.gpu.Destroy()
	}
}

// requestImage is one image as the request sent it, in conversation order:
// its URL, and once fetched (util.FetchImage, before the tower is taken)
// its bytes.
type requestImage struct {
	url  string
	data []byte
}

// encode turns one image into the prompt's rows. It returns the tower time
// for the request's log line: wall time, including the units that ran in its
// yields.
//
// The tower yields the device every v.slice to whoever is waiting for it.
// It is still one pass as far as its arenas go (v.tower), and an interactive
// request's tower counts as an interactive job to the scheduler (towerBegin),
// so what runs in its gaps is another interactive unit and never a
// background prefill chunk.
func (v *llmVision) encode(ctx context.Context, dev *Device, img requestImage) (llm.PromptImage, time.Duration, error) {
	data := img.data
	sum := sha256.Sum256(data)
	hash := binary.LittleEndian.Uint64(sum[:8])
	if p, ok := v.lookup(hash); ok {
		return p, 0, nil
	}
	rgb, _, err := pixels.Decode(data)
	if err != nil {
		return llm.PromptImage{}, 0, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
	}
	planes, h, w, err := v.proc.Planes(rgb)
	if err != nil {
		return llm.PromptImage{}, 0, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
	}
	rows, gh, gw, err := v.cfg.Patchify(planes, h, w)
	if err != nil {
		return llm.PromptImage{}, 0, err
	}
	var out *vision.Output
	v.tower.Lock()
	defer v.tower.Unlock()
	start := time.Now()
	held := start
	v.gpu.Between = nil
	if v.slice > 0 {
		v.gpu.Between = func() {
			if time.Since(held) >= v.slice {
				dev.yield()
				held = time.Now()
			}
		}
	}
	err = dev.Do(func(*vk.Device) error {
		var err error
		out, err = v.gpu.Forward(ctx, rows, gh, gw)
		return err
	})
	if err != nil {
		return llm.PromptImage{}, 0, fmt.Errorf("vision tower: %w", err)
	}
	p := llm.PromptImage{
		GridH: gh / v.cfg.SpatialMergeSize, GridW: gw / v.cfg.SpatialMergeSize,
		Hash: hash, Embd: out.Merged.Data,
	}
	v.store(p)
	return p, time.Since(start), nil
}

// imageKey is an image cell's scheduling key: negative, so it can never be
// a token id, and a function of the image's hash and the cell's place in it.
func imageKey(hash uint64, k int) int32 {
	h := hash ^ uint64(k+1)*0x9e3779b97f4a7c15
	h ^= h >> 29
	return -int32(h&0x3fffffff) - 1
}

// promptKeys is the sequence the scheduler matches slots against.
func promptKeys(in llm.Input) []int32 {
	keys := append([]int32(nil), in.IDs...)
	for _, im := range in.Images {
		for k := range im.GridH * im.GridW {
			keys[im.At+k] = imageKey(im.Hash, k)
		}
	}
	return keys
}

// keyIDs is the ids a run of keys stands for: an image cell is the pad.
func keyIDs(keys []int32) []int32 {
	ids := make([]int32, len(keys))
	for i, k := range keys {
		if k < 0 {
			ids[i] = imagePad
		} else {
			ids[i] = k
		}
	}
	return ids
}
