package backend

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
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
			Name, Image   string
			MarkdownPlain string `json:"markdown_plain"`
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
		// Mistral's markdown is PaddleX's plain markdown with each picture
		// referenced as Mistral does.
		want := rec.MarkdownPlain
		for i, im := range res.Pages[0].MistralImages {
			want = strings.Replace(want, "![](imgs/", "![PENDING"+strconv.Itoa(i)+"](imgs/", 1)
			j := strings.Index(want, "![PENDING"+strconv.Itoa(i)+"](")
			k := j + strings.Index(want[j:], ")")
			want = want[:j] + "![" + im.ID + "](" + im.ID + ")" + want[k+1:]
		}
		if got := res.Pages[0].Mistral; got != want {
			t.Errorf("%s: markdown differs from PaddleX's:\n got %q\nwant %q", rec.Name, got, want)
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
	t.Logf("two-page PDF in %v; page 2: %.60q", time.Since(start).Round(time.Millisecond), res.Pages[1].Mistral)
	sel, err := b.ParseDocument(context.Background(), &api.DocumentRequest{Data: data, PDF: true, Pages: []int{1}})
	if err != nil || len(sel.Pages) != 1 || sel.Pages[0].Index != 1 {
		t.Errorf("page selection: %v", err)
	}
}
