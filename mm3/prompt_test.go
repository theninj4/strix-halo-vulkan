package mm3

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/zimage/tokenizer"
)

// TestPrompt holds CleanCaption, NormalizeLyrics and the token ids to the
// reference's on reference/dump_mm3_prompt.py's cases.
func TestPrompt(t *testing.T) {
	buf, err := os.ReadFile("../reference/out/mm3prompt/cases.json")
	if err != nil {
		t.Skipf("no reference (run reference/dump_mm3_prompt.py): %v", err)
	}
	var cases []struct {
		Caption, Lyrics, Clean, Normal string
		IDs                            []int32
	}
	if err := json.Unmarshal(buf, &cases); err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(filepath.Join(modelDir, "tokenizer"))
	if err != nil {
		t.Skipf("no tokenizer: %v", err)
	}
	bad := 0
	for i, c := range cases {
		if got := CleanCaption(c.Caption); got != c.Clean {
			t.Errorf("case %d caption:\n got %q\nwant %q", i, got, c.Clean)
			bad++
			continue
		}
		if got := NormalizeLyrics(c.Lyrics); got != c.Normal {
			t.Errorf("case %d lyrics:\n got %q\nwant %q", i, got, c.Normal)
			bad++
			continue
		}
		ids, unc, err := Prompt(tok, c.Caption, c.Lyrics)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != len(c.IDs) {
			t.Errorf("case %d: %d ids, want %d", i, len(ids), len(c.IDs))
			bad++
			continue
		}
		for j := range ids {
			if ids[j] != c.IDs[j] {
				t.Errorf("case %d: id %d is %d, want %d", i, j, ids[j], c.IDs[j])
				bad++
				break
			}
		}
		if unc[0] != ids[0] || unc[len(unc)-1] != ids[len(ids)-1] || unc[1] != AudioCFGToken {
			t.Errorf("case %d: unconditional row %v...", i, unc[:3])
		}
	}
	t.Logf("%d cases, %d wrong", len(cases), bad)
}
