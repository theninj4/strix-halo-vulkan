package pipeline

import (
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

const prepareDir = "../../reference/out/flux3prepare"

func readPNG(t *testing.T, path string) image.Image {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// TestPrepare runs every case BFL's own prepare_reference and
// prepare_multi_reference were dumped on (reference/dump_flux3_prepare.py):
// sizes, crop boxes and every 8-bit value exact.
func TestPrepare(t *testing.T) {
	buf, err := os.ReadFile(filepath.Join(prepareDir, "manifest.json"))
	if err != nil {
		t.Skipf("no dump: run reference/dump_flux3_prepare.py (%v)", err)
	}
	var m struct {
		Cases map[string]struct {
			In, Out, Kind string
			Cap           int
			OutSize       []int `json:"out_size"`
			Box           []int
		}
	}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	for name, c := range m.Cases {
		src := readPNG(t, filepath.Join(prepareDir, c.In))
		var ref Reference
		if c.Kind == "multi" {
			ref, err = PrepareMultiReference(src, MultiReferenceMaxPixels)
		} else {
			ref, err = PrepareReference(src, c.Cap)
		}
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want := opaque(readPNG(t, filepath.Join(prepareDir, c.Out)))
		got := ref.Image
		if got.Rect.Dx() != c.OutSize[0] || got.Rect.Dy() != c.OutSize[1] {
			t.Errorf("%-22s %v, want %v", name, got.Rect.Size(), c.OutSize)
			continue
		}
		wantBox := image.Rectangle{}
		if c.Box != nil {
			wantBox = image.Rect(c.Box[0], c.Box[1], c.Box[2], c.Box[3])
		}
		if ref.Box != wantBox {
			t.Errorf("%-22s box %v, want %v", name, ref.Box, wantBox)
		}
		diff, worst := 0, 0
		for i := range want.Pix {
			if d := abs(int(got.Pix[i]) - int(want.Pix[i])); d > 0 {
				diff++
				worst = max(worst, d)
			}
		}
		if diff > 0 {
			t.Errorf("%-22s %d values differ (worst %d levels)", name, diff, worst)
			continue
		}
		t.Logf("%-22s %v -> %v box %v: exact", name, src.Bounds().Size(), got.Rect.Size(), ref.Box)
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
