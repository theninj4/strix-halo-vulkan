package pixels

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type decodeCase struct {
	File string `json:"file"`
	Mode string `json:"mode"`
	HW   []int  `json:"hw"`
}

// TestDecode holds Decode to PIL's `Image.open(f).convert("RGB")`. PNG and
// GIF are lossless and must match bit for bit, including a palette entry
// that is fully transparent (PIL keeps its RGB) and an RGBA file (alpha
// dropped). JPEG is not gated, only measured: Go's IDCT and colour conversion
// are not libjpeg-turbo's (LLM-VISION.md V3).
func TestDecode(t *testing.T) {
	m := load(t)
	if len(m.Decode) == 0 {
		t.Skip("the dump has no decode cases; rerun reference/dump_llm_vision.py")
	}
	for _, c := range m.Decode {
		t.Run(c.File, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join(refDir, "decode", c.File))
			if err != nil {
				t.Fatal(err)
			}
			want, err := os.ReadFile(filepath.Join(refDir, "decode", c.File+".rgb"))
			if err != nil {
				t.Fatal(err)
			}
			got, format, err := Decode(data)
			if err != nil {
				t.Fatal(err)
			}
			if got.H != c.HW[0] || got.W != c.HW[1] {
				t.Fatalf("%dx%d, PIL %dx%d", got.W, got.H, c.HW[1], c.HW[0])
			}
			var diff, maxd int
			var sum float64
			for i := range want {
				d := int(got.Pix[i]) - int(want[i])
				if d != 0 {
					diff++
				}
				if d < 0 {
					d = -d
				}
				maxd = max(maxd, d)
				sum += float64(d)
			}
			frac := float64(diff) / float64(len(want))
			if strings.HasSuffix(c.File, ".jpg") {
				t.Logf("%s (%s, %dx%d): %.1f%% of samples differ from libjpeg, max %d, mean %.3f levels",
					c.File, format, got.W, got.H, 100*frac, maxd, sum/float64(len(want)))
				return
			}
			if diff != 0 {
				t.Errorf("%s (%s): %d of %d samples differ from PIL, max %d", c.File, c.Mode, diff, len(want), maxd)
				return
			}
			t.Logf("%s (%s, %s): %d samples identical", c.File, c.Mode, format, len(want))
		})
	}
	if _, _, err := Decode([]byte("RIFF\x00\x00\x00\x00WEBPVP8 ")); !errors.Is(err, ErrUnsupported) {
		t.Errorf("WebP was not refused as unsupported: %v", err)
	}
}
