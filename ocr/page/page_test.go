package page

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/ocr/layout"
)

const (
	pageRef   = "../../reference/out/ocr_page"
	layoutRef = "../../reference/out/doclayout"
)

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no reference (%v); run reference/dump_ocr_page.py", err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
}

// TestTextFunctions holds the text post-processing to PaddleX's own
// functions over reference/dump_ocr_textfns.py's cases.
func TestTextFunctions(t *testing.T) {
	var fx struct {
		Truncate []struct {
			In       string `json:"in"`
			MinCount int    `json:"min_count"`
			Out      string `json:"out"`
		} `json:"truncate"`
		OTSL []struct {
			In    string `json:"in"`
			Out   string `json:"out"`
			Error bool   `json:"error"`
		} `json:"otsl"`
		Formula []struct {
			In    string `json:"in"`
			Label string `json:"label"`
			Out   string `json:"out"`
		} `json:"formula"`
		Title []struct {
			In  string `json:"in"`
			Out string `json:"out"`
		} `json:"title"`
	}
	readJSON(t, filepath.Join(pageRef, "textfns.json"), &fx)
	bad := 0
	for _, c := range fx.Truncate {
		if got := truncateRepetitive(c.In, c.MinCount); got != c.Out {
			bad++
			t.Errorf("truncate(%q, %d) = %q, want %q", c.In, c.MinCount, got, c.Out)
		}
	}
	skipped := 0
	for _, c := range fx.OTSL {
		if c.Error {
			skipped++
			continue
		}
		if got := otslToHTML(c.In); got != c.Out {
			bad++
			t.Errorf("otsl(%q)\n got %q\nwant %q", c.In, got, c.Out)
		}
	}
	for _, c := range fx.Formula {
		if got := formulaDelimiters(c.In, c.Label); got != c.Out {
			bad++
			t.Errorf("delimiters(%q, %s) = %q, want %q", c.In, c.Label, got, c.Out)
		}
	}
	for _, c := range fx.Title {
		if got := formatTitle(c.In); got != c.Out {
			bad++
			t.Errorf("title(%q) = %q, want %q", c.In, got, c.Out)
		}
	}
	t.Logf("%d truncations, %d OTSL tables (%d that PaddleX raises on, skipped), %d formulas, %d titles; %d wrong",
		len(fx.Truncate), len(fx.OTSL), skipped, len(fx.Formula), len(fx.Title), bad)
}

type oracle struct {
	Name      string `json:"name"`
	Image     string `json:"image"`
	Size      [2]int `json:"size"`
	LayoutDet []struct {
		ClsID      int     `json:"cls_id"`
		Label      string  `json:"label"`
		Score      float64 `json:"score"`
		Coordinate [4]int  `json:"coordinate"`
		Order      *int    `json:"order"`
	} `json:"layout_det_res"`
	Blocks []struct {
		Label       string   `json:"label"`
		Box         [4]int   `json:"box"`
		GroupID     *int     `json:"group_id"`
		MergeAligns []string `json:"merge_aligns"`
		Img         *struct {
			Size   [2]int `json:"size"`
			SHA256 string `json:"sha256"`
		} `json:"img"`
	} `json:"blocks"`
	VLM []struct {
		Block     int    `json:"block"`
		Prompt    string `json:"prompt"`
		MinPixels int    `json:"min_pixels"`
		MaxPixels int    `json:"max_pixels"`
		Size      [2]int `json:"size"`
		SHA256    string `json:"sha256"`
		Raw       string `json:"raw"`
	} `json:"vlm"`
	Parsing       []JSONBlock `json:"parsing_res_list"`
	Markdown      string      `json:"markdown"`
	MarkdownPlain string      `json:"markdown_plain"`
}

func oracles(t *testing.T) []oracle {
	t.Helper()
	paths, _ := filepath.Glob(filepath.Join(pageRef, "*", "record.json"))
	if len(paths) == 0 {
		t.Skip("no reference; run reference/dump_ocr_page.py")
	}
	var out []oracle
	for _, p := range paths {
		var o oracle
		readJSON(t, p, &o)
		out = append(out, o)
	}
	return out
}

// hfDetections reads the layout dump's detections at 0.3 as Regions.
func hfDetections(t *testing.T, name string) []layout.Region {
	var rec struct {
		D []struct {
			Query   int        `json:"query"`
			LabelID int        `json:"label_id"`
			Score   float64    `json:"score"`
			Box     [4]float64 `json:"box"`
			Order   int        `json:"order"`
		} `json:"detections_03"`
	}
	readJSON(t, filepath.Join(layoutRef, name, "record.json"), &rec)
	var out []layout.Region
	for _, d := range rec.D {
		out = append(out, layout.Region{Label: layout.Labels[d.LabelID], LabelID: d.LabelID, Score: d.Score, Box: d.Box, Order: d.Order, Query: d.Query})
	}
	return out
}

func rgbSHA(img *pixels.RGB) string {
	s := sha256.Sum256(img.Pix)
	return hex.EncodeToString(s[:])
}

func decode(t *testing.T, path string) (*pixels.RGB, string) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("../..", path))
	if err != nil {
		t.Fatal(err)
	}
	img, format, err := pixels.Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	return img, format
}

// TestGlue runs PaddleX's glue in Go on HF's detections and holds every
// step to the oracle: the layout boxes, the blocks (crops by hash), the
// recognition entries (images by hash), and, from the oracle's own raw
// recognitions, the block list and the markdown.
func TestGlue(t *testing.T) {
	for _, o := range oracles(t) {
		t.Run(o.Name, func(t *testing.T) {
			img, format := decode(t, o.Image)
			exactPixels := format != "jpeg" // Go's JPEG decoder is not libjpeg
			boxes := LayoutPost(hfDetections(t, o.Name), img.W, img.H)
			if len(boxes) != len(o.LayoutDet) {
				t.Fatalf("%d layout boxes, PaddleX has %d", len(boxes), len(o.LayoutDet))
			}
			for i, w := range o.LayoutDet {
				g := boxes[i]
				order := 0
				if w.Order != nil {
					order = *w.Order
				}
				if g.ClsID != w.ClsID || g.Coord != w.Coordinate || g.Order != order || float64(g.Score) != float64(float32(w.Score)) {
					t.Errorf("box %d: %+v, want %+v (order %d)", i, g, w, order)
				}
			}
			figures := gatherFigures(boxes)
			var blocks []Block
			for _, b := range filterOverlap(boxes) {
				blocks = append(blocks, Block{Label: b.Label, Box: b.Coord, Img: crop(img, b.Coord), GroupID: -1})
			}
			nonMerge := map[string]bool{"table": true}
			for l := range imageLabels {
				nonMerge[l] = true
			}
			blocks = mergeBlocks(blocks, nonMerge)
			if len(blocks) != len(o.Blocks) {
				t.Fatalf("%d blocks, PaddleX has %d", len(blocks), len(o.Blocks))
			}
			for i, w := range o.Blocks {
				g := blocks[i]
				gid := -1
				if w.GroupID != nil {
					gid = *w.GroupID
				}
				if g.Label != w.Label || g.Box != w.Box || g.GroupID != gid || !reflect.DeepEqual(nonNil(g.MergeAligns, w.MergeAligns), w.MergeAligns) {
					t.Errorf("block %d: %s %v group %d aligns %v, want %s %v group %d aligns %v",
						i, g.Label, g.Box, g.GroupID, g.MergeAligns, w.Label, w.Box, gid, w.MergeAligns)
				}
				if (g.Img == nil) != (w.Img == nil) {
					t.Errorf("block %d: image %v, PaddleX's %v", i, g.Img != nil, w.Img != nil)
				} else if g.Img != nil && ([2]int{g.Img.W, g.Img.H} != w.Img.Size || (exactPixels && rgbSHA(g.Img) != w.Img.SHA256)) {
					t.Errorf("block %d: crop %dx%d differs from PaddleX's %v", i, g.Img.W, g.Img.H, w.Img.Size)
				}
			}
			ents, drop := entries(blocks, figures)
			if len(ents) != len(o.VLM) {
				t.Fatalf("%d recognitions, PaddleX has %d", len(ents), len(o.VLM))
			}
			texts := make([]string, len(ents))
			for i, w := range o.VLM {
				g := ents[i]
				if g.Block != w.Block || g.Prompt != w.Prompt || g.MinPixels != w.MinPixels || g.MaxPixels != w.MaxPixels ||
					[2]int{g.Img.W, g.Img.H} != w.Size || (exactPixels && rgbSHA(g.Img) != w.SHA256) {
					t.Errorf("entry %d: block %d %q [%d, %d] %dx%d, want block %d %q [%d, %d] %v",
						i, g.Block, g.Prompt, g.MinPixels, g.MaxPixels, g.Img.W, g.Img.H, w.Block, w.Prompt, w.MinPixels, w.MaxPixels, w.Size)
				}
				texts[i] = w.Raw
			}
			results := assemble(blocks, ents, texts, drop)
			if got := JSON(results); !reflect.DeepEqual(got, o.Parsing) {
				gj, _ := json.Marshal(got)
				wj, _ := json.Marshal(o.Parsing)
				t.Errorf("parsing_res_list differs:\n got %s\nwant %s", gj, wj)
			}
			if md := Markdown(results, img.W); md != o.Markdown {
				t.Errorf("markdown differs:\n got %q\nwant %q", md, o.Markdown)
			}
			if md := MarkdownPlain(results); md != o.MarkdownPlain {
				t.Errorf("plain markdown differs:\n got %q\nwant %q", md, o.MarkdownPlain)
			}
			t.Logf("%d boxes, %d blocks, %d recognitions, %d markdown bytes: PaddleX's (pixels exact: %v)",
				len(boxes), len(blocks), len(ents), len(o.Markdown), exactPixels)
		})
	}
}

// nonNil lets a nil and an empty list compare equal where PaddleX's JSON
// has [] for a group of one.
func nonNil(got, want []string) []string {
	if got == nil && want != nil && len(want) == 0 {
		return []string{}
	}
	return got
}
