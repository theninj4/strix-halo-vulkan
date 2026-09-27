package lm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

const sampleRef = "../../reference/out/acesample"

// TestSampleHost checks sample mode's host side against
// reference/dump_ace_sample.py: the query hints, the prompt (text and ids),
// the lyrics cut from the output, and create_sample's conversion.
func TestSampleHost(t *testing.T) {
	buf, err := os.ReadFile(filepath.Join(sampleRef, "sample.json"))
	if err != nil {
		t.Skipf("no sample reference (%v); run reference/dump_ace_sample.py", err)
	}
	var ref struct {
		Hints []struct {
			Query        string  `json:"query"`
			Language     *string `json:"language"`
			Instrumental bool    `json:"instrumental"`
		} `json:"hints"`
		Prompts []struct {
			Query        string  `json:"query"`
			Instrumental bool    `json:"instrumental"`
			Prompt       string  `json:"prompt"`
			IDs          []int32 `json:"ids"`
		} `json:"prompts"`
		Lyrics []struct {
			Text, Lyrics string
		} `json:"lyrics"`
		Convert []struct {
			Metadata     map[string]string `json:"metadata"`
			Instrumental bool              `json:"instrumental"`
			Result       struct {
				Caption, Lyrics, Keyscale, Language, Timesignature string
				BPM                                                *int     `json:"bpm"`
				Duration                                           *float64 `json:"duration"`
				Instrumental                                       bool     `json:"instrumental"`
			} `json:"result"`
		} `json:"convert"`
	}
	if err := json.Unmarshal(buf, &ref); err != nil {
		t.Fatal(err)
	}
	for _, h := range ref.Hints {
		want := ""
		if h.Language != nil {
			want = *h.Language
		}
		lang, inst := DescriptionHints(h.Query)
		if lang != want || inst != h.Instrumental {
			t.Errorf("hints %q: (%q, %v), want (%q, %v)", h.Query, lang, inst, want, h.Instrumental)
		}
	}
	tk := loadTokenizer(t)
	for _, p := range ref.Prompts {
		if p.Query == "" {
			continue // create_sample_from_query never builds it: an empty query is NO USER INPUT
		}
		got := SamplePrompt(p.Query, p.Instrumental)
		if p.Query == NoUserInput && SamplePrompt("", p.Instrumental) != got {
			t.Errorf("an empty query is not NO USER INPUT's prompt")
		}
		if got != p.Prompt {
			t.Errorf("prompt %q/%v:\n got %q\nwant %q", p.Query, p.Instrumental, got, p.Prompt)
			continue
		}
		ids, err := tk.Encode(got)
		if err != nil {
			t.Fatal(err)
		}
		if equalIDs(ids, p.IDs) >= 0 {
			t.Errorf("prompt %q/%v: ids differ at %d", p.Query, p.Instrumental, equalIDs(ids, p.IDs))
		}
	}
	for _, l := range ref.Lyrics {
		if got := ExtractLyrics(l.Text); got != l.Lyrics {
			t.Errorf("lyrics of %q: %q, want %q", l.Text, got, l.Lyrics)
		}
	}
	for _, c := range ref.Convert {
		m := Meta{}
		for k, v := range c.Metadata {
			m[k] = v
		}
		s := SongOf(m, m["lyrics"], c.Instrumental)
		r := c.Result
		bpm, dur := 0, 0.0
		if r.BPM != nil {
			bpm = *r.BPM
		}
		if r.Duration != nil {
			dur = *r.Duration
		}
		if s.Caption != r.Caption || s.KeyScale != r.Keyscale || s.Language != r.Language ||
			s.TimeSignature != r.Timesignature || s.BPM != bpm || s.Duration != dur || s.Instrumental != r.Instrumental {
			t.Errorf("convert %v: %+v, want %+v", c.Metadata, s, r)
		}
	}
	t.Logf("%d hints, %d prompts, %d lyric cuts, %d conversions", len(ref.Hints), len(ref.Prompts), len(ref.Lyrics), len(ref.Convert))
}
