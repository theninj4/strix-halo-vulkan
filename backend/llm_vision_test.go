package backend

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	_ "image/jpeg"
	"image/png"
	"os"
	"runtime"
	"sync"
	"testing"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/vk"
)

const visionTestMMProj = "../models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf"

// mirrorURL is an image flipped left to right, as a PNG data: URL.
func mirrorURL(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("no test image %s", path)
	}
	defer f.Close()
	src, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	b := src.Bounds()
	dst := image.NewNRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(b.Max.X-1-(x-b.Min.X), y, src.At(x, y))
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, dst); err != nil {
		t.Fatal(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

func imageURL(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no test image %s", path)
	}
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(b)
}

// TestPromptKeys: an image cell's key is its picture's, so two images under
// the same ids part at their first cell, and the same image does not.
func TestPromptKeys(t *testing.T) {
	in := func(hash uint64) llm.Input {
		return llm.Input{IDs: []int32{1, 2, imagePad, imagePad, imagePad, imagePad, 3},
			Images: []llm.InputImage{{At: 2, GridH: 2, GridW: 2, Hash: hash}}}
	}
	a, b, a2 := promptKeys(in(7)), promptKeys(in(8)), promptKeys(in(7))
	if commonPrefix(a, b) != 2 || commonPrefix(a, a2) != len(a) {
		t.Fatalf("keys: different images share %d, the same image %d of %d", commonPrefix(a, b), commonPrefix(a, a2), len(a))
	}
	for i, id := range keyIDs(a) {
		if id != in(7).IDs[i] {
			t.Fatalf("keyIDs gives %d at %d, the ids hold %d", id, i, in(7).IDs[i])
		}
	}
	// A mark after one image moves past its other cells.
	short := []int32{1, imagePad, 2, 3, imagePad, 4}
	imgs := []llm.PromptImage{{GridH: 2, GridW: 3}, {GridH: 1, GridW: 2}}
	if got := expandedMark(3, short, imgs); got != 3+5 {
		t.Fatalf("a mark at 3 behind a 6-token image is %d, want 8", got)
	}
}

// TestLLMImageCheckpoint is V9's gate on the scheduler over the four-layer
// prefix: a conversation whose earlier turn holds an image is checkpointed
// at its last user turn, after the image (expandedMark), and a later question
// restores it (the slot's checkpoint keys, the graph's spans) and streams
// exactly what a fresh prefill streams. The same conversation around a
// *different* picture must not restore: its keys part at the image.
func TestLLMImageCheckpoint(t *testing.T) {
	if _, err := os.Stat(visionTestMMProj); err != nil {
		t.Skipf("no mmproj at %s", visionTestMMProj)
	}
	l := newTestLLM(t, LLMOptions{Slots: 1, MMProj: visionTestMMProj})
	barn := imageURL(t, "../reference/out/llmvision/barn.jpg")
	// The other picture is the barn mirrored: the same size, so the same
	// grid and the *same ids*, and only the keys can tell the two apart.
	mirrored := mirrorURL(t, "../reference/out/llmvision/barn.jpg")
	conv := func(img, q string) *api.CompletionRequest {
		r := greedy("", 12)
		r.Messages = []api.Message{
			{Role: "user", Content: api.MessageContent{
				{Type: "image_url", ImageURL: &struct {
					URL string `json:"url"`
				}{URL: img}},
				{Type: "text", Text: "Describe this picture."}}},
			{Role: "assistant", Content: api.MessageContent{{Type: "text", Text: "A barn under mountains."}}},
			{Role: "user", Content: api.MessageContent{{Type: "text", Text: q}}},
		}
		return r
	}
	ctx := context.Background()
	run := func(req *api.CompletionRequest) string {
		t.Helper()
		out, err := complete(ctx, l, req)
		if err != nil {
			t.Fatal(err)
		}
		return joined(out)
	}

	run(greedy("Something else entirely, to fill the slot.", 4))
	want := run(conv(barn, "Is it winter?"))
	if n := l.sched.restores; n != 0 {
		t.Fatalf("%d restores before any checkpoint", n)
	}
	run(conv(barn, "How many windows does it have?"))
	got := run(conv(barn, "Is it winter?"))
	if n := l.sched.restores; n != 2 {
		t.Fatalf("%d restores; the two later questions should each restore the image turn", n)
	}
	if got != want {
		t.Fatalf("restored past an image:\n%q\nprefilled fresh:\n%q", got, want)
	}
	other := run(conv(mirrored, "Is it winter?"))
	if n := l.sched.restores; n != 2 {
		t.Fatalf("a different picture under the same ids restored the checkpoint (%d restores)", n)
	}
	if other == want {
		t.Logf("note: the mirrored picture gives the same %d bytes at four layers", len(other))
	}
	t.Logf("restored past a 256-token image, twice, and streamed the fresh prefill's %q exactly; "+
		"the mirrored picture, under identical ids, was not restored", got)
}

// TestVisionCacheEvictsOldest: the encoded-image cache holds at most its
// budget of rows, dropping the least recently used first, and a lookup
// counts as a use.
func TestVisionCacheEvictsOldest(t *testing.T) {
	v := &llmVision{maxCache: 3 * 400}
	img := func(h uint64) llm.PromptImage { return llm.PromptImage{Hash: h, Embd: make([]float32, 100)} }
	v.store(img(1))
	v.store(img(2))
	v.store(img(3))
	if _, ok := v.lookup(1); !ok {
		t.Fatal("image 1 is not cached")
	}
	v.store(img(4)) // evicts 2, the oldest since 1 was just used
	for h, want := range map[uint64]bool{1: true, 2: false, 3: true, 4: true} {
		if _, ok := v.lookup(h); ok != want {
			t.Errorf("image %d cached %v, want %v", h, ok, want)
		}
	}
	if v.cached != 3*400 {
		t.Errorf("%d bytes held, want %d", v.cached, 3*400)
	}
	v.store(llm.PromptImage{Hash: 9, Embd: make([]float32, 1000)}) // larger than the whole budget
	if _, ok := v.lookup(9); ok {
		t.Error("an image over the whole budget was cached")
	}
}

// TestDeviceYieldHandsOff: a yield inside Do lets exactly the one waiting Do
// run before the yielder continues, every time, even though that waiter
// queues its next Do at once, and is free when nobody waits. On a sync.Mutex
// this fails both ways: the yielder barges back in, or the waiter does.
func TestDeviceYieldHandsOff(t *testing.T) {
	d := &Device{}
	if err := d.Do(func(*vk.Device) error {
		if d.yield() {
			t.Error("yield handed off with nobody waiting")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	const rounds = 20
	var order []string // appended only by whoever holds the device
	inside := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		d.Do(func(*vk.Device) error {
			close(inside)
			for i := 0; i < rounds; i++ {
				order = append(order, "tower")
				for d.mu.waiters() == 0 {
					runtime.Gosched()
				}
				if !d.yield() {
					t.Error("yield with a waiter handed nothing off")
				}
			}
			return nil
		})
	}()
	<-inside
	for i := 0; i < rounds; i++ {
		d.Do(func(*vk.Device) error {
			order = append(order, "unit")
			return nil
		})
	}
	<-done
	for i, who := range order {
		if want := []string{"tower", "unit"}[i%2]; who != want {
			t.Fatalf("turn %d went to %s, want %s: %v", i, who, want, order)
		}
	}
	if len(order) != 2*rounds {
		t.Fatalf("%d turns, want %d", len(order), 2*rounds)
	}
}

// TestTowerHoldsBackground: while an interactive request's images are on the
// tower, which yields the device between slices, rule 1 keeps background
// units off it and lets interactive ones run in the gaps. A background
// request's tower holds nothing back.
func TestTowerHoldsBackground(t *testing.T) {
	s := &llmSched{slots: make([]llmSlot, 2)}
	s.wake = sync.NewCond(&s.mu)
	bg := &llmUnit{job: &llmJob{class: classBackground}}
	s.pending = []*llmUnit{bg}

	done := s.towerBegin(classBackground)
	if got := s.pick(); got != bg {
		t.Fatal("a background tower held back a background unit")
	}
	done()

	done = s.towerBegin(classInteractive)
	if got := s.pick(); got != nil {
		t.Fatal("a background unit was picked during an interactive tower")
	}
	fg := &llmUnit{job: &llmJob{class: classInteractive}}
	s.pending = append(s.pending, fg)
	if got := s.pick(); got != fg {
		t.Fatal("an interactive unit was not picked during an interactive tower")
	}
	s.pending = s.pending[:1]
	done()
	if got := s.pick(); got != bg {
		t.Fatal("the background unit is still held after the tower ended")
	}
}
