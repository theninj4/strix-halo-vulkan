package page

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/llm/pixels"
)

// TestFigureTokens checks tokenize_figure_of_table and its inverse against
// PaddleX's own (reference/dump_ocr_figtoken.py): every painted crop byte
// for byte (cv2's Hershey putText), the tokens, the dropped paths and the
// untokenized strings.
func TestFigureTokens(t *testing.T) {
	dir := filepath.Join(pageRef, "figtoken")
	var cases []struct {
		W, H    int
		Table   [4]int
		Figures []struct {
			Path       string
			Label      string
			Coordinate [4]int
		}
		TokenMap  map[string]string `json:"token_map"`
		Dropped   []string
		UntokIn   string            `json:"untok_in"`
		UntokObjs map[string]string `json:"untok_objs"`
		UntokOut  string            `json:"untok_out"`
	}
	readJSON(t, filepath.Join(dir, "cases.json"), &cases)
	badPix, badCases := 0, 0
	for k, c := range cases {
		want, err := os.ReadFile(filepath.Join(dir, fmt.Sprintf("case_%d.rgb", k)))
		if err != nil {
			t.Fatal(err)
		}
		in, err := os.ReadFile(filepath.Join(dir, fmt.Sprintf("case_%d.in.rgb", k)))
		if err != nil {
			t.Fatal(err)
		}
		img := &pixels.RGB{W: c.W, H: c.H, Pix: in}
		var figs []Box
		for _, f := range c.Figures {
			figs = append(figs, Box{Label: f.Label, Coord: f.Coordinate})
		}
		got, toks, dropped := tokenizeFigures(img, c.Table, figs)
		n := 0
		for i := range want {
			if got.Pix[i] != want[i] {
				n++
			}
		}
		if n > 0 {
			badCases++
			badPix += n
			if badCases <= 5 {
				t.Errorf("case %d (%dx%d): %d bytes differ", k, c.W, c.H, n)
			}
		}
		if len(toks) != len(c.TokenMap) {
			t.Errorf("case %d: %d tokens, want %d", k, len(toks), len(c.TokenMap))
		}
		for _, tk := range toks {
			if c.TokenMap[tk.token] != tk.path {
				t.Errorf("case %d: token %s -> %s, want %s", k, tk.token, tk.path, c.TokenMap[tk.token])
			}
		}
		if fmt.Sprint(dropped) != fmt.Sprint(c.Dropped) {
			t.Errorf("case %d: dropped %v, want %v", k, dropped, c.Dropped)
		}
		if u := untokenizeFigures(c.UntokIn, toks, c.UntokObjs); u != c.UntokOut {
			t.Errorf("case %d: untokenize\n got %q\nwant %q", k, u, c.UntokOut)
		}
	}
	t.Logf("%d cases, %d with differing bytes (%d bytes)", len(cases), badCases, badPix)
}
