package ocr

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"strix-halo-vulkan/llm/pixels"
)

// TestEngine runs every dumped case end to end, from its image file through
// the Go processor, the device tower and the device ERNIE, and holds the
// greedy generation to the fp32 reference's token for token (OCR.md O5).
func TestEngine(t *testing.T) {
	recs := records(t)
	dev, done := newTestDevice(t)
	defer done()
	start := time.Now()
	e, err := Load(dev, modelDir, DefaultOptions())
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	defer e.Destroy()
	t.Logf("loaded in %v, %.2f GB on the device", time.Since(start).Round(time.Millisecond), float64(e.DeviceBytes())/1e9)

	sort.Slice(recs, func(i, j int) bool { return recs[i].Name < recs[j].Name })
	for _, rec := range recs {
		t.Run(rec.Name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("..", rec.Image))
			if err != nil {
				t.Fatal(err)
			}
			img, _, err := pixels.Decode(data)
			if err != nil {
				t.Fatal(err)
			}
			wall := time.Now()
			res, err := e.Recognize(context.Background(), Request{Image: img, Prompt: rec.Prompt, MaxTokens: len(rec.Generated)})
			if err != nil {
				t.Fatal(err)
			}
			total := time.Since(wall)
			first := -1
			for i := range res.IDs {
				if i >= len(rec.Generated) || res.IDs[i] != rec.Generated[i] {
					first = i
					break
				}
			}
			if first < 0 && len(res.IDs) != len(rec.Generated) {
				first = len(res.IDs)
			}
			t.Logf("%d prompt + %d new (%s): process %v, tower %v, prefill %v, decode %v (%.2f ms a token); total %v; first divergence %d",
				res.PromptTokens, len(res.IDs), res.Finish, res.Process.Round(time.Millisecond), res.Tower.Round(time.Millisecond),
				res.Prefill.Round(time.Millisecond), res.Decode.Round(time.Millisecond),
				float64(res.Decode.Microseconds())/1e3/float64(len(res.IDs)), total.Round(time.Millisecond), first)
			if first >= 0 {
				t.Errorf("diverges from the reference at token %d of %d", first, len(rec.Generated))
			}
			if res.Text != rec.Text {
				t.Errorf("text differs:\n got %q\nwant %q", res.Text, rec.Text)
			}
		})
	}
}
