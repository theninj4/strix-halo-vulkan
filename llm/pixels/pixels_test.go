package pixels

import (
	"encoding/binary"
	"encoding/json"
	"image"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/qimage/vision"
)

// reference/dump_llm_vision.py: HF's processor over four cards, and
// smart_resize over a table of sizes (LLM-VISION.md V3).
const refDir = "../../reference/out/llmvision"

type manifest struct {
	Cards map[string]struct {
		HW      []int  `json:"hw"`
		Mode    string `json:"mode"`
		GridTHW []int  `json:"grid_thw"`
	} `json:"cards"`
	SmartResize struct {
		Factor    int     `json:"factor"`
		MinPixels int     `json:"min_pixels"`
		MaxPixels int     `json:"max_pixels"`
		Cases     [][]int `json:"cases"`
	} `json:"smart_resize"`
	Decode  []decodeCase `json:"decode"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func load(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference (%v); run reference/dump_llm_vision.py", err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func f32s(t *testing.T, m *manifest, name string) []float32 {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(refDir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, meta.Count)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// TestSmartResize holds the size rule to transformers' over its branches,
// including the half-to-even ties Python's round() takes.
func TestSmartResize(t *testing.T) {
	m := load(t)
	sr := m.SmartResize
	p := Processor{Factor: sr.Factor, MinPixels: sr.MinPixels, MaxPixels: sr.MaxPixels}
	if p != Default {
		t.Errorf("preprocessor_config says %+v, Default is %+v", p, Default)
	}
	for _, c := range sr.Cases {
		h, w, err := p.SmartResize(c[0], c[1])
		if err != nil || h != c[2] || w != c[3] {
			t.Errorf("%dx%d: got %dx%d (%v), HF %dx%d", c[0], c[1], h, w, err, c[2], c[3])
		}
	}
}

// TestPixelValues is V3's gate: from the cards' own bytes to the tower's
// input rows, **bit for bit** against the processor. The four cards cover no
// resize, a downscale on both axes, an upscale on both, and alpha.
func TestPixelValues(t *testing.T) {
	m := load(t)
	cfg := &vision.Config{PatchSize: 16, SpatialMergeSize: 2, TemporalPatchSize: 2, InChannels: 3}
	for name, card := range m.Cards {
		t.Run(name, func(t *testing.T) {
			if _, ok := m.Tensors[name+"_u8"]; !ok {
				t.Skip("a tower-only card: the dump keeps no source pixels for it")
			}
			ch := 3
			if card.Mode == "RGBA" {
				ch = 4
			}
			u8 := f32s(t, m, name+"_u8")
			// Every card enters as a Go image and goes through FromImage, so
			// the RGBA one proves the alpha is dropped and not composited.
			src := image.NewNRGBA(image.Rect(0, 0, card.HW[1], card.HW[0]))
			for i := 0; i < card.HW[0]*card.HW[1]; i++ {
				src.Pix[i*4+3] = 255
				for c := 0; c < ch; c++ {
					src.Pix[i*4+c] = uint8(u8[i*ch+c])
				}
			}
			img := FromImage(src)
			planes, h, w, err := Default.Planes(img)
			if err != nil {
				t.Fatal(err)
			}
			rows, gh, gw, err := cfg.Patchify(planes, h, w)
			if err != nil {
				t.Fatal(err)
			}
			if gh != card.GridTHW[1] || gw != card.GridTHW[2] {
				t.Fatalf("grid %dx%d, HF %v", gh, gw, card.GridTHW)
			}
			want := f32s(t, m, name+"_pixel_values")
			if len(rows.Data) != len(want) {
				t.Fatalf("%d values, HF %d", len(rows.Data), len(want))
			}
			bad := 0
			for i := range want {
				if math.Float32bits(rows.Data[i]) != math.Float32bits(want[i]) {
					if bad < 3 {
						t.Errorf("value %d: %g, HF %g", i, rows.Data[i], want[i])
					}
					bad++
				}
			}
			if bad == 0 {
				t.Logf("%dx%d -> %dx%d, %d values bit-identical", img.W, img.H, w, h, len(want))
			} else {
				t.Errorf("%d of %d values differ", bad, len(want))
			}
		})
	}
}
