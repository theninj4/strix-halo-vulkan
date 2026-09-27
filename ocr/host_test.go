package ocr

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/safetensors"
)

const (
	modelDir = "../models/PaddleOCR-VL-1.6"
	refDir   = "../reference/out/ocr"
)

type record struct {
	Name        string     `json:"name"`
	Image       string     `json:"image"`
	Task        string     `json:"task"`
	Prompt      string     `json:"prompt"`
	GridTHW     []int      `json:"grid_thw"`
	ResizedHW   []int      `json:"resized_hw"`
	InputIDs    []int32    `json:"input_ids"`
	PositionIDs [3][]int32 `json:"position_ids"`
	RopeDelta   int32      `json:"rope_delta"`
	Generated   []int32    `json:"generated_ids"`
	Text        string     `json:"text"`
	TextRaw     string     `json:"text_raw"`
}

func loadTok(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := LoadTokenizer(modelDir)
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	return tok
}

func records(t *testing.T) []record {
	t.Helper()
	paths, _ := filepath.Glob(filepath.Join(refDir, "*", "record.json"))
	if len(paths) == 0 {
		t.Skip("no reference; run reference/dump_ocr.py")
	}
	var out []record
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var r record
		if err := json.Unmarshal(raw, &r); err != nil {
			t.Fatal(err)
		}
		out = append(out, r)
	}
	return out
}

func refTensor(t *testing.T, name, tensor string) ([]float32, []int) {
	t.Helper()
	f, err := safetensors.Open(filepath.Join(refDir, name, "tensors.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	ten, err := f.Get(tensor)
	if err != nil {
		t.Fatal(err)
	}
	v, err := ten.F32(nil)
	if err != nil {
		t.Fatal(err)
	}
	return v, append([]int(nil), ten.Shape...)
}

// TestTokenizer holds Encode and Decode to HF's fast tokenizer over
// reference/dump_ocr_tokens.py's corpus.
func TestTokenizer(t *testing.T) {
	tok := loadTok(t)
	raw, err := os.ReadFile(filepath.Join(refDir, "tokens.json"))
	if err != nil {
		t.Skip("no reference; run reference/dump_ocr_tokens.py")
	}
	var fx struct {
		Encode []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		} `json:"encode"`
		Decode []struct {
			IDs  []int32 `json:"ids"`
			Skip bool    `json:"skip"`
			Text string  `json:"text"`
		} `json:"decode"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for _, c := range fx.Encode {
		got := tok.Encode(c.Text)
		if !equalIDs(got, c.IDs) {
			if bad++; bad <= 5 {
				t.Errorf("encode %q:\n got  %v\n want %v", c.Text, got, c.IDs)
			}
		}
	}
	for _, c := range fx.Decode {
		if got := tok.Decode(c.IDs, c.Skip); got != c.Text {
			if bad++; bad <= 10 {
				t.Errorf("decode %v skip=%v:\n got  %q\n want %q", c.IDs, c.Skip, got, c.Text)
			}
		}
	}
	t.Logf("%d encodes, %d decodes, %d wrong", len(fx.Encode), len(fx.Decode), bad)
}

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestPromptAndDecode builds every dumped case's prompt from its grid and
// task, and decodes its generation: ids, positions, the rope delta and both
// texts must be HF's exactly.
func TestPromptAndDecode(t *testing.T) {
	tok := loadTok(t)
	for _, r := range records(t) {
		p, err := tok.BuildPrompt(Tasks[r.Task], r.GridTHW[1]/MergeSize, r.GridTHW[2]/MergeSize)
		if err != nil {
			t.Fatal(err)
		}
		if !equalIDs(p.IDs, r.InputIDs) {
			t.Errorf("%s: ids differ (%d vs %d tokens)", r.Name, len(p.IDs), len(r.InputIDs))
			continue
		}
		for i, pos := range p.Pos {
			for a := 0; a < 3; a++ {
				if pos[a] != r.PositionIDs[a][i] {
					t.Fatalf("%s: token %d axis %d at %d, want %d", r.Name, i, a, pos[a], r.PositionIDs[a][i])
				}
			}
		}
		if d := p.Next - int32(len(p.IDs)); d != r.RopeDelta {
			t.Errorf("%s: rope delta %d, want %d", r.Name, d, r.RopeDelta)
		}
		if got := tok.Decode(r.Generated, true); got != r.Text {
			t.Errorf("%s: decoded %q, want %q", r.Name, got, r.Text)
		}
		if got := tok.Decode(r.Generated, false); got != r.TextRaw {
			t.Errorf("%s: decoded raw %q, want %q", r.Name, got, r.TextRaw)
		}
	}
}

// TestProcess runs each case's image through the processor. PNG must be
// bit-exact; a JPEG decodes through Go's IDCT, not libjpeg's, so its gap is
// reported and bounded rather than required to be zero (LLM-VISION.md V3).
func TestProcess(t *testing.T) {
	for _, r := range records(t) {
		data, err := os.ReadFile(filepath.Join("..", r.Image))
		if err != nil {
			t.Fatal(err)
		}
		rgb, format, err := pixels.Decode(data)
		if err != nil {
			t.Fatal(err)
		}
		im, err := Process(rgb, DefaultMinPixels, DefaultMaxPixels)
		if err != nil {
			t.Fatal(err)
		}
		if im.GridH != r.GridTHW[1] || im.GridW != r.GridTHW[2] {
			t.Errorf("%s: grid %dx%d, want %dx%d", r.Name, im.GridH, im.GridW, r.GridTHW[1], r.GridTHW[2])
			continue
		}
		want, _ := refTensor(t, r.Name, "pixels")
		if len(want) != len(im.Patches) {
			t.Fatalf("%s: %d values, want %d", r.Name, len(im.Patches), len(want))
		}
		var worst float64
		off := 0
		for i := range want {
			d := math.Abs(float64(im.Patches[i] - want[i]))
			worst = max(worst, d)
			if d != 0 {
				off++
			}
		}
		t.Logf("%s (%s %dx%d): %d of %d values off, max %.4f (%.1f levels)",
			r.Name, format, rgb.W, rgb.H, off, len(want), worst, worst*127.5)
		switch {
		case strings.EqualFold(format, "jpeg"):
			if worst*127.5 > 12 || float64(off) > 0.05*float64(len(want)) {
				t.Errorf("%s: jpeg gap beyond V3's measurement", r.Name)
			}
		case off != 0:
			t.Errorf("%s: %s must be bit-exact", r.Name, format)
		}
	}
}

// TestSmartResize covers the branch Qwen2-VL's rule does not have: a side
// under 28 is raised to 28 first, and the aspect check runs after that.
// Expectations are HF's smart_resize, run by hand.
func TestSmartResize(t *testing.T) {
	for _, c := range []struct{ h, w, wh, ww int }{
		{70, 668, 112, 1064},    // ocr_demo.jpg
		{1368, 1524, 924, 1036}, // the demo page, max_pixels
		{10, 300, 84, 1848},
		{300, 14, 1568, 84},
		{27, 27, 336, 336},
		{29, 29, 336, 336},
		{42, 1000, 84, 1652},
		{13, 4000, 0, 0}, // 28 x 8615 after the raise: aspect 307, refused
		{5000, 20, 0, 0},
	} {
		h, w, err := SmartResize(c.h, c.w, DefaultMinPixels, DefaultMaxPixels)
		if c.wh == 0 {
			if err == nil {
				t.Errorf("%dx%d: want the aspect refusal, got %dx%d", c.w, c.h, w, h)
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		if h != c.wh || w != c.ww {
			t.Errorf("%dx%d -> %dx%d, want %dx%d", c.w, c.h, w, h, c.ww, c.wh)
		}
	}
}
