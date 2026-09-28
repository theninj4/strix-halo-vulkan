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

// TestEngineBatched runs every case at once, as a page's regions run (OCR.md
// O11), and holds each to the fp32 reference as TestEngine does. "tight"
// leaves the page out, gives the pool 16 pages and admits every request it
// has a slot for, so they outgrow it and the youngest are preempted and
// recomputed.
func TestEngineBatched(t *testing.T) {
	recs := records(t)
	sort.Slice(recs, func(i, j int) bool { return recs[i].Name < recs[j].Name })
	var reqs []Request
	for _, rec := range recs {
		data, err := os.ReadFile(filepath.Join("..", rec.Image))
		if err != nil {
			t.Fatal(err)
		}
		img, _, err := pixels.Decode(data)
		if err != nil {
			t.Fatal(err)
		}
		reqs = append(reqs, Request{Image: img, Prompt: rec.Prompt, MaxTokens: len(rec.Generated)})
	}
	dev, done := newTestDevice(t)
	defer done()
	for _, arm := range []struct {
		name string
		lm   LMOptions
	}{
		{"roomy", DefaultOptions().LM},
		{"tight", LMOptions{MaxLen: 640, Slots: 6, Rows: 2048, CachePositions: 16 * PageSize}},
	} {
		t.Run(arm.name, func(t *testing.T) {
			recs, reqs := recs, reqs
			if arm.name == "tight" {
				for i := range recs {
					if recs[i].Name == "page" {
						recs = append(recs[:i:i], recs[i+1:]...)
						reqs = append(reqs[:i:i], reqs[i+1:]...)
						break
					}
				}
			}
			e, err := Load(dev, modelDir, Options{LM: arm.lm})
			if err != nil {
				t.Skipf("no checkpoint (%v)", err)
			}
			defer e.Destroy()
			e.greedyAdmit = arm.name == "tight"
			wall := time.Now()
			res, err := e.RecognizeAll(context.Background(), reqs)
			if err != nil {
				t.Fatal(err)
			}
			total, tokens, preempted := time.Since(wall), 0, 0
			for i, rec := range recs {
				r := res[i]
				tokens += len(r.IDs)
				preempted += r.Preempted
				first := -1
				for k := range r.IDs {
					if k >= len(rec.Generated) || r.IDs[k] != rec.Generated[k] {
						first = k
						break
					}
				}
				if first < 0 && len(r.IDs) != len(rec.Generated) {
					first = len(r.IDs)
				}
				t.Logf("%s: %d prompt + %d new: wait %v, tower %v, prefill %v, decode %v, preempted %d; first divergence %d",
					rec.Name, r.PromptTokens, len(r.IDs), r.Wait.Round(time.Millisecond), r.Tower.Round(time.Millisecond),
					r.Prefill.Round(time.Millisecond), r.Decode.Round(time.Millisecond), r.Preempted, first)
				if first >= 0 {
					t.Errorf("%s diverges from the reference at token %d of %d", rec.Name, first, len(rec.Generated))
				}
				if r.Text != rec.Text {
					t.Errorf("%s: text differs:\n got %q\nwant %q", rec.Name, r.Text, rec.Text)
				}
			}
			t.Logf("%d requests, %d tokens in %v; %d preemptions", len(reqs), tokens, total.Round(time.Millisecond), preempted)
			if arm.name == "tight" && preempted == 0 {
				t.Errorf("the tight pool preempted nothing")
			}
		})
	}
}
