package pipeline

import (
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/h3/plan"
	"strix-halo-vulkan/h3/vae"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// The fl2va reference is reference/dump_h3_fl2va.py (VIDEO.md M10): the
// README fl2va request's keyframe, placed on a 256x448 canvas as the first
// keyframe, a 900x1080 portrait crop of it placed as the last, and the
// keyframe again on 480x864.
const fl2vaRef = "../../reference/out/h3fl2va"

type fl2vaManifest struct {
	Prompt        string           `json:"prompt"`
	AutoCanvas    map[string][]int `json:"auto_canvas"`
	AutoCanvasEnd map[string][]int `json:"auto_canvas_last"`
	Presentations map[string]struct {
		IDs []int32 `json:"ids"`
	} `json:"presentations"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func loadFL2VA(t *testing.T) *fl2vaManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(fl2vaRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump (%v); run reference/dump_h3_fl2va.py prep", err)
	}
	var m fl2vaManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// keyframeSources is the README keyframe and the portrait crop of it the
// dump uses as a last keyframe.
func keyframeSources(t *testing.T) (first, last image.Image) {
	t.Helper()
	f, err := os.Open(filepath.Join(modelDir, "assets/fl2va_keyframe.png"))
	if err != nil {
		t.Skipf("no keyframe (%v)", err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	type sub interface {
		SubImage(image.Rectangle) image.Image
	}
	return img, img.(sub).SubImage(image.Rect(600, 0, 1500, 1080))
}

func (m *fl2vaManifest) bytes(t *testing.T, name string) []byte {
	t.Helper()
	v := readF32(t, filepath.Join(fl2vaRef, name+".bin"))
	out := make([]byte, len(v))
	for i, x := range v {
		out[i] = byte(x)
	}
	return out
}

// TestPlaceKeyframe holds the stretch and the cover-crop to PIL's bytes.
func TestPlaceKeyframe(t *testing.T) {
	m := loadFL2VA(t)
	first, last := keyframeSources(t)
	for _, c := range []struct {
		name  string
		img   image.Image
		index int
		h, w  int
	}{
		{"key_first", first, 0, 256, 448},
		{"key_last", last, 1, 256, 448},
		{"key_big", first, 0, 480, 864},
	} {
		got, err := PlaceKeyframe(c.img, c.index, c.h, c.w)
		if err != nil {
			t.Fatal(err)
		}
		want := m.bytes(t, c.name)
		diff, worst := 0, 0
		for i := range want {
			if d := int(got[i]) - int(want[i]); d != 0 {
				diff++
				worst = max(worst, d, -d)
			}
		}
		t.Logf("%s: %d of %d bytes differ, worst %d", c.name, diff, len(want), worst)
		if diff != 0 {
			t.Errorf("%s: %d bytes differ from PIL's", c.name, diff)
		}
	}
}

// TestResolveFL2VA resolves the dump's request — its presentation and
// layout — and the canvas a keyframe picks when the request names none.
func TestResolveFL2VA(t *testing.T) {
	m := loadFL2VA(t)
	tok, err := tokenizer.Load(filepath.Join(modelDir, "tokenizer"))
	if err != nil {
		t.Skipf("no tokenizer: %v", err)
	}
	first, last := keyframeSources(t)
	r, err := Resolve(tok, &Request{Prompt: m.Prompt, First: first, Height: 256, Width: 448, Frames: 124})
	if err != nil {
		t.Fatal(err)
	}
	want := m.Presentations["f"].IDs
	if len(r.Tokens) != len(want) {
		t.Fatalf("%d tokens, want %d", len(r.Tokens), len(want))
	}
	for i := range want {
		if r.Tokens[i] != want[i] {
			t.Fatalf("token %d: %d, want %d", i, r.Tokens[i], want[i])
		}
	}
	if len(r.Layout.Pos) != 5709 || r.Layout.CondVideoRows != 112 || len(r.Keyframes) != 1 {
		t.Errorf("%d rows, %d keyframe rows, %d keyframes; want 5709, 112, 1", len(r.Layout.Pos), r.Layout.CondVideoRows, len(r.Keyframes))
	}
	r2, err := Resolve(tok, &Request{Prompt: "x", First: first, Last: last, Height: 256, Width: 448})
	if err != nil {
		t.Fatal(err)
	}
	if r2.Layout.CondVideoRows != 224 || len(r2.Keyframes) != 2 {
		t.Errorf("first+last: %d keyframe rows, %d keyframes", r2.Layout.CondVideoRows, len(r2.Keyframes))
	}
	for se, hw := range m.AutoCanvas {
		n := atoi(t, se)
		for _, c := range []struct {
			req  *Request
			want []int
		}{
			{&Request{Prompt: "x", First: first, ShortEdge: n}, hw},
			{&Request{Prompt: "x", Last: last, ShortEdge: n}, m.AutoCanvasEnd[se]},
		} {
			r, err := Resolve(tok, c.req)
			if err != nil {
				t.Fatal(err)
			}
			// resolve_canvas_size returns (height, width).
			if r.Height != c.want[0] || r.Width != c.want[1] {
				t.Errorf("short edge %s: %dx%d, want %dx%d", se, r.Width, r.Height, c.want[1], c.want[0])
			}
		}
	}
}

func atoi(t *testing.T, s string) int {
	n := 0
	for _, c := range s {
		n = n*10 + int(c-'0')
	}
	return n
}

// TestKeyframeNoise builds the keyframe rows from the dump's anchor latent
// and its augmentation noise, and holds them to the oracle's rows bit for
// bit: `scale_noise` then the patchify.
func TestKeyframeNoise(t *testing.T) {
	m := loadFL2VA(t)
	if _, ok := m.Tensors["cond_rows"]; !ok {
		t.Skip("the dump has no dit phase")
	}
	frame := func(name string) *vae.Tensor {
		s := m.Tensors[name].Shape
		return &vae.Tensor{C: s[0], T: 1, H: s[1], W: s[2], Data: readF32(t, filepath.Join(fl2vaRef, name+".bin"))}
	}
	anchor, err := vae.Patchify(frame("first_latents"))
	if err != nil {
		t.Fatal(err)
	}
	noise, err := vae.Patchify(frame("cond_noise"))
	if err != nil {
		t.Fatal(err)
	}
	lay, err := plan.NewLayout([]int32{1, 1}, 37, 16, 28, 207, []plan.Anchor{plan.First})
	if err != nil {
		t.Fatal(err)
	}
	video, _, err := (&Pipeline{}).noise(&Request{CondNoise: noise}, lay, &qwen.Mat{Rows: len(anchor) / 96, Cols: 96, Data: anchor})
	if err != nil {
		t.Fatal(err)
	}
	want := readF32(t, filepath.Join(fl2vaRef, "cond_rows.bin"))
	for i := range want {
		if video.Data[i] != want[i] {
			t.Fatalf("keyframe row value %d: %v, want %v", i, video.Data[i], want[i])
		}
	}
}
