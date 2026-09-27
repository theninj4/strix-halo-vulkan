package backend

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"strix-halo-vulkan/api"
)

const ocrTestModel = "../models/PaddleOCR-VL-1.6"

// TestOCRRequestRefusals: every knob that would change the answer and is not
// implemented is a 400, not a silent default.
func TestOCRRequestRefusals(t *testing.T) {
	img := []api.Content{{Type: "image_url", ImageURL: &struct {
		URL string `json:"url"`
	}{URL: "data:,"}}, {Type: "text", Text: "OCR:"}}
	f := func(v float64) *float64 { return &v }
	ok := func() *api.CompletionRequest {
		return &api.CompletionRequest{Messages: []api.Message{{Role: "user", Content: img}}, Temperature: f(0)}
	}
	if _, p, err := ocrRequest(ok(), 1003520); err != nil || p != "OCR:" {
		t.Fatalf("the plain request: %q, %v", p, err)
	}
	for name, mod := range map[string]func(*api.CompletionRequest){
		"temperature":  func(r *api.CompletionRequest) { r.Temperature = f(0.7) },
		"stop":         func(r *api.CompletionRequest) { r.Stop = api.StringList{"x"} },
		"repeat":       func(r *api.CompletionRequest) { r.RepeatPenalty = f(1.1) },
		"two messages": func(r *api.CompletionRequest) { r.Messages = append(r.Messages, r.Messages[0]) },
		"system":       func(r *api.CompletionRequest) { r.Messages[0].Role = "system" },
		"no image":     func(r *api.CompletionRequest) { r.Messages[0].Content = img[1:] },
		"two images":   func(r *api.CompletionRequest) { r.Messages[0].Content = append([]api.Content{img[0]}, img...) },
		"max_pixels":   func(r *api.CompletionRequest) { r.MMProcessorKwargs = &api.MMProcessorArgs{MaxPixels: 2000000} },
		"min over max": func(r *api.CompletionRequest) { r.MMProcessorKwargs = &api.MMProcessorArgs{MinPixels: 9, MaxPixels: 8} },
		"penalty <= 0": func(r *api.CompletionRequest) { r.RepetitionPenalty = f(0) },
	} {
		r := ok()
		mod(r)
		if _, _, err := ocrRequest(r, 1003520); !errors.Is(err, api.ErrUnsupported) {
			t.Errorf("%s: %v, want ErrUnsupported", name, err)
		}
	}
}

// TestOCRChat runs every dumped case through the chat backend, buffered and
// streamed, from a data: URL of its image file: both must be the fp32
// reference's text, and the stream's deltas must concatenate to it.
func TestOCRChat(t *testing.T) {
	paths, _ := filepath.Glob("../reference/out/ocr/*/record.json")
	if len(paths) == 0 {
		t.Skip("no reference; run reference/dump_ocr.py")
	}
	if _, err := os.Stat(ocrTestModel); err != nil {
		t.Skip("no checkpoint")
	}
	dev, err := OpenDevice("ocr-chat-test")
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	defer dev.Close()
	b, err := NewOCR(OCROptions{Model: ocrTestModel, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var rec struct {
			Name, Image, Prompt, Text string
			Generated                 []int32 `json:"generated_ids"`
		}
		if err := json.Unmarshal(raw, &rec); err != nil {
			t.Fatal(err)
		}
		t.Run(rec.Name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("..", rec.Image))
			if err != nil {
				t.Fatal(err)
			}
			url := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)
			req := &api.CompletionRequest{
				Model: OCRModelID, MaxCompletionTokens: len(rec.Generated),
				Messages: []api.Message{{Role: "user", Content: []api.Content{
					{Type: "image_url", ImageURL: &struct {
						URL string `json:"url"`
					}{URL: url}},
					{Type: "text", Text: rec.Prompt},
				}}},
			}
			var deltas []string
			res, err := b.Complete(context.Background(), req, func(d api.Delta) error {
				deltas = append(deltas, d.Content)
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			got := strings.Join(deltas, "")
			if got != rec.Text {
				t.Errorf("text differs:\n got %q\nwant %q", got, rec.Text)
			}
			for _, d := range deltas {
				if strings.Contains(d, "�") {
					t.Errorf("a delta carries a partial character: %q", d)
				}
			}
			t.Logf("%d deltas, %d prompt + %d tokens, %s", len(deltas), res.Usage.PromptTokens, res.Usage.CompletionTokens, res.FinishReason)
		})
	}
}

// TestOCRDocument parses every page oracle through the backend's document
// path (reference/dump_ocr_page.py: PaddleX's glue on HF's layout) and a
// two-page PDF: a PNG page's markdown and prunedResult are PaddleX's, and
// the PDF renders at PaddleX's 144 dpi.
func TestOCRDocument(t *testing.T) {
	paths, _ := filepath.Glob("../reference/out/ocr_page/*/record.json")
	if len(paths) == 0 {
		t.Skip("no reference; run reference/dump_ocr_page.py")
	}
	dev, err := OpenDevice("ocr-document-test")
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	defer dev.Close()
	b, err := NewOCR(OCROptions{Model: ocrTestModel, Device: dev, LayoutModel: "../models/PP-DocLayoutV3"})
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	defer b.Close()
	for _, p := range paths {
		raw, _ := os.ReadFile(p)
		var rec struct {
			Name, Image, Markdown string
			Pruned                map[string]any `json:"pruned"`
		}
		if err := json.Unmarshal(raw, &rec); err != nil {
			t.Fatal(err)
		}
		if strings.HasSuffix(rec.Image, ".jpg") {
			continue // Go's JPEG decoder is not libjpeg (O5)
		}
		data, _ := os.ReadFile(filepath.Join("..", rec.Image))
		res, err := b.ParseDocument(context.Background(), &api.DocumentRequest{Data: data})
		if err != nil {
			t.Fatal(err)
		}
		pg := res.Pages[0]
		if pg.Markdown != rec.Markdown {
			t.Errorf("%s: markdown differs from PaddleX's", rec.Name)
		}
		gj, _ := json.Marshal(pg.Pruned)
		var got any
		_ = json.Unmarshal(gj, &got)
		wj, _ := json.Marshal(rec.Pruned)
		var want any
		_ = json.Unmarshal(wj, &want)
		// The device layout's scores are fp16's (within 3e-3 of HF's, O7),
		// and a box edge at a rounding boundary can land a pixel over (the
		// demo page's block 8: HF's 656.5 against 656.52); the rest is exact.
		if !sameWithin(got, want, map[string]float64{"score": 3e-3, "coordinate": 1, "block_bbox": 1}) {
			t.Errorf("%s: prunedResult differs:\n got %s\nwant %s", rec.Name, gj, wj)
		}
	}
	data, err := os.ReadFile("../testdata/ocr/two_pages.pdf")
	if err != nil {
		t.Skip("no testdata/ocr/two_pages.pdf")
	}
	start := time.Now()
	res, err := b.ParseDocument(context.Background(), &api.DocumentRequest{Data: data, PDF: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.PDFPages != 2 || len(res.Pages) != 2 || res.Pages[0].Width != 1524 || res.Pages[0].Height != 1368 ||
		res.Pages[1].Width != 551 || res.Pages[1].Height != 132 || res.Pages[0].DPI != 144 {
		t.Fatalf("PDF pages: %d of %d, sizes %dx%d and %dx%d", len(res.Pages), res.PDFPages,
			res.Pages[0].Width, res.Pages[0].Height, res.Pages[1].Width, res.Pages[1].Height)
	}
	if pc := res.Pages[1].Pruned["page_count"]; pc == nil || *pc.(*int) != 2 {
		t.Errorf("page_count %v", pc)
	}
	t.Logf("two-page PDF in %v; page 2: %.60q", time.Since(start).Round(time.Millisecond), res.Pages[1].Markdown)
	sel, err := b.ParseDocument(context.Background(), &api.DocumentRequest{Data: data, PDF: true, Pages: []int{1}})
	if err != nil || len(sel.Pages) != 1 || sel.Pages[0].Index != 1 {
		t.Errorf("page selection: %v", err)
	}
}

// sameWithin is reflect.DeepEqual over decoded JSON, but the numbers under
// the keys of loose may differ by their tolerance.
func sameWithin(a, b any, loose map[string]float64) bool {
	return within(a, b, loose, -1)
}

func within(a, b any, loose map[string]float64, tol float64) bool {
	if tol >= 0 {
		if x, ok := a.(float64); ok {
			y, ok := b.(float64)
			return ok && x-y <= tol && y-x <= tol
		}
	}
	switch av := a.(type) {
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for k, v := range av {
			w, ok := bv[k]
			if !ok {
				return false
			}
			t := tol
			if lt, ok := loose[k]; ok {
				t = lt
			}
			if !within(v, w, loose, t) {
				return false
			}
		}
		return true
	case []any:
		bv, ok := b.([]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if !within(av[i], bv[i], loose, tol) {
				return false
			}
		}
		return true
	}
	return reflect.DeepEqual(a, b)
}
