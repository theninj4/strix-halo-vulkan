package llm

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// The HF oracle for the image path: reference/dump_llm_vision.py
// (LLM-VISION.md V2). The language model never ran; these are the pieces
// of a two-image chat prompt that precede its first layer.
const visionRefDir = "../reference/out/llmvision"

type visionManifest struct {
	IDs        []int32 `json:"ids"`
	ImageToken int32   `json:"image_token_id"`
	Pads       []int   `json:"image_pad_positions"`
	GridTHW    [][]int `json:"grid_thw"`
	Delta      int     `json:"mrope_delta"`
	Tensors    map[string]struct {
		Shape []int  `json:"shape"`
		DType string `json:"dtype"`
		Count int    `json:"count"`
	} `json:"tensors"`
}

func loadVisionRef(t *testing.T) *visionManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(visionRefDir, "manifest.json"))
	if err != nil {
		t.Skipf("no vision reference (%v); run reference/dump_llm_vision.py", err)
	}
	var m visionManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// visionRefF64 reads a tensor the dump wrote as float64, which it does for
// integers too large for a float32's mantissa (the n-gram rows reach 3.2e8).
func visionRefF64(t *testing.T, m *visionManifest, name string) []float64 {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok || meta.DType != "float64" {
		t.Fatalf("reference has no float64 tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(visionRefDir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float64, meta.Count)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return out
}

// TestPLEImagePrompt holds PLERows to HF's own n-gram hash over a prompt
// with two images in it. HF hashes `input_ids` as they are, the image-pad id at
// every image slot, with ordinary history. That is what
// LLM-VISION.md decided to follow over llama.cpp, which cuts the window at
// every image cell. So the graph's image input has to keep 248056 in `ids`
// and nothing else changes. This proves the claim on the rows it concerns:
// inside an image, and the two tokens after each one.
func TestPLEImagePrompt(t *testing.T) {
	ref := loadVisionRef(t)
	m, err := Open(modelDir)
	if err != nil {
		t.Skipf("checkpoint absent: %v", err)
	}
	defer m.Close()
	c, ok, err := m.PLEConfig()
	if err != nil || !ok {
		t.Fatalf("PLE config: ok=%v err=%v", ok, err)
	}
	if c.Image != ref.ImageToken {
		t.Errorf("ple.image_token_id %d, HF's image_token_id %d", c.Image, ref.ImageToken)
	}
	want := visionRefF64(t, ref, "prompt_ngram_ids")
	got := PLERows(c, ref.IDs)
	if len(got) != len(want) {
		t.Fatalf("%d rows, HF has %d", len(got), len(want))
	}
	bad := 0
	for i := range got {
		if float64(got[i]) != want[i] {
			if bad < 5 {
				t.Errorf("token %d head %d: row %d, HF %d", i/c.NHeads, i%c.NHeads, got[i], int64(want[i]))
			}
			bad++
		}
	}
	if bad == 0 {
		t.Logf("%d tokens x %d heads identical, %d of them image pads", len(ref.IDs), c.NHeads, len(ref.Pads))
	}
}

// promptSpans is the dump's two images as spans: each run of consecutive
// pads, with the grid the processor reported for it (in patches, so halved
// for the 2x2 merge).
func promptSpans(t *testing.T, ref *visionManifest) []ImageSpan {
	t.Helper()
	var spans []ImageSpan
	for i := 0; i < len(ref.Pads); {
		j := i
		for j+1 < len(ref.Pads) && ref.Pads[j+1] == ref.Pads[j]+1 {
			j++
		}
		g := ref.GridTHW[len(spans)]
		spans = append(spans, ImageSpan{Start: ref.Pads[i], End: ref.Pads[j] + 1, GridH: g[1] / 2, GridW: g[2] / 2})
		if err := spans[len(spans)-1].valid(); err != nil {
			t.Fatal(err)
		}
		i = j + 1
	}
	return spans
}

// TestImagePositions holds cellPositions to HF's get_rope_index over the
// dump's two-image prompt: every cell's (t, h, w), and the delta the text
// after the last image continues at.
func TestImagePositions(t *testing.T) {
	ref := loadVisionRef(t)
	want := visionRefF64(t, ref, "prompt_position_ids") // [3][n]
	n := len(ref.IDs)
	spans := promptSpans(t, ref)
	got := cellPositions(spans, 0, n)
	bad := 0
	for i := 0; i < n; i++ {
		for a := 0; a < 3; a++ {
			if float64(got[i][a]) != want[a*n+i] {
				if bad < 5 {
					t.Errorf("cell %d axis %d: %d, HF %v", i, a, got[i][a], want[a*n+i])
				}
				bad++
			}
		}
	}
	if d := cellDelta(spans, n); d != ref.Delta {
		t.Errorf("delta after the prompt %d, HF %d", d, ref.Delta)
	}
	if bad == 0 {
		t.Logf("%d cells x 3 axes identical to HF; %d images, delta %d", n, len(spans), ref.Delta)
	}
	// Cutting a sequence inside an image is refused; at its edges it is not.
	if _, err := truncateSpans(spans, spans[0].Start+1); err == nil {
		t.Error("a cut inside an image was accepted")
	}
	if kept, err := truncateSpans(spans, spans[1].Start); err != nil || len(kept) != 1 {
		t.Errorf("a cut at the second image's start kept %d spans (%v)", len(kept), err)
	}
}

// TestExpandImagesIsTheProcessor is V7's gate on the text half of an image
// prompt. The dump's two-image message, rendered by RenderChat, has to be
// HF's apply_chat_template output character for character. Tokenized by the
// checkpoint's tokenizer and widened by ExpandImages with the processor's
// grids, it has to be the processor's input_ids, id for id.
func TestExpandImagesIsTheProcessor(t *testing.T) {
	ref := loadVisionRef(t)
	var extra struct {
		Rendered string `json:"rendered"`
		Prompt   string `json:"prompt"`
	}
	buf, err := os.ReadFile(filepath.Join(visionRefDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(buf, &extra); err != nil {
		t.Fatal(err)
	}
	msgs := []ChatMessage{{Role: "user", Parts: []ChatPart{{Image: true}, {Image: true}, {Text: extra.Prompt}}}}
	rendered, err := RenderChat(msgs, ChatOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if rendered != extra.Rendered {
		t.Fatalf("rendering: %s", firstDiff(rendered, extra.Rendered))
	}
	m, err := Open(modelDir)
	if err != nil {
		t.Skipf("checkpoint absent: %v", err)
	}
	defer m.Close()
	tok, err := LoadTokenizer(m.Set)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode(rendered)
	if err != nil {
		t.Fatal(err)
	}
	var imgs []PromptImage
	for _, g := range ref.GridTHW {
		imgs = append(imgs, PromptImage{GridH: g[1] / 2, GridW: g[2] / 2})
	}
	in, err := ExpandImages(ids, ref.ImageToken, imgs)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(in.IDs, ref.IDs) {
		t.Fatalf("ids differ from the processor's: %d against %d", len(in.IDs), len(ref.IDs))
	}
	spans := promptSpans(t, ref)
	for i, im := range in.Images {
		if im.At != spans[i].Start || im.GridH != spans[i].GridH || im.GridW != spans[i].GridW {
			t.Errorf("image %d at %d %dx%d, the processor's at %d %dx%d", i, im.At, im.GridH, im.GridW,
				spans[i].Start, spans[i].GridH, spans[i].GridW)
		}
	}
	t.Logf("%d chars rendered as HF does; %d ids, two images, identical to the processor's", len(rendered), len(in.IDs))

	// A pad the images do not account for is refused.
	if _, err := ExpandImages(ids, ref.ImageToken, imgs[:1]); err == nil {
		t.Error("two pads for one image were accepted")
	}
}
