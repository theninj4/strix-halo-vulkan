package lm

import (
	"math/rand/v2"
	"strconv"
	"testing"
)

// understandLabels are reference/dump_ace_lm.py's understand cases
// (MUSIC.md A13): A11's 30 s synthwave source and 120 s lofi reference,
// captioned by upstream's full_analysis_only.
var understandLabels = []string{"understand", "understand_lofi"}

// TestUnderstandPrompt: the understand prompt from the oracle's codes, id
// for id, both spliced (UnderstandPrompt) and encoded from upstream's text
// (the claim that the two agree); and the song parsed from what upstream's
// LM wrote, field for field against understand_audio_from_codes' own.
func TestUnderstandPrompt(t *testing.T) {
	m := loadLMManifest(t)
	tk := loadTokenizer(t)
	p, err := NewPlanner(nil, tk)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, label := range understandLabels {
		c, ok := m.Cases[label]
		if !ok {
			continue
		}
		n++
		_, codes32 := refInt32(t, m, label+"_codes")
		codes := make([]int, len(codes32))
		for i, c := range codes32 {
			codes[i] = int(c)
		}
		want := promptRows(t, m, label, 0)[0]
		got, err := p.UnderstandPrompt(codes)
		if err != nil {
			t.Fatal(err)
		}
		if at := equalIDs(got, want); at >= 0 {
			t.Errorf("%s: spliced prompt differs at %d of %d ids", label, at, len(want))
		}
		enc, err := tk.Encode(c.Prompts[0][0])
		if err != nil {
			t.Fatal(err)
		}
		if at := equalIDs(enc, want); at >= 0 {
			t.Errorf("%s: encoded prompt differs at %d of %d ids", label, at, len(want))
		}
		t.Logf("%s: %d codes, %d prompt ids exact", label, len(codes), len(want))

		text := c.Phases[0].Text
		song := SongOf(ParseCoT(text), ExtractLyrics(text), false)
		w := c.Understand
		for k, g := range map[string]string{"caption": song.Caption, "genres": song.Genres, "keyscale": song.KeyScale,
			"timesignature": song.TimeSignature, "language": song.Language, "lyrics": song.Lyrics,
			"bpm": strconv.Itoa(song.BPM), "duration": strconv.FormatFloat(song.Duration, 'f', -1, 64)} {
			wv, ok := w[k]
			if !ok {
				wv = map[string]string{"bpm": "0", "duration": "0"}[k]
			}
			if k == "duration" { // upstream keeps the int the CoT wrote
				if f, err := strconv.ParseFloat(wv, 64); err == nil {
					wv = strconv.FormatFloat(f, 'f', -1, 64)
				}
			}
			if g != wv {
				t.Errorf("%s: %s %q, want %q", label, k, g, wv)
			}
		}
		t.Logf("%s: %d-token answer: %q, %s, %d bpm, %s, %s, %v s, %s", label, len(c.Phases[0].Steps),
			song.Genres, song.KeyScale, song.BPM, song.TimeSignature, song.Language, song.Duration, song.Caption)
	}
	if n == 0 {
		t.Skip("no understand cases; run reference/dump_ace_lm.py understand understand_lofi")
	}
}

// TestUnderstandPlanner runs understand mode live on the oracle's codes:
// a song with every field the FSM holds valid and genres from upstream's
// vocabulary. The draws are our RNG's; this gates the loop, not the words.
func TestUnderstandPlanner(t *testing.T) {
	m := loadLMManifest(t)
	g := loadGPU(t)
	p, err := NewPlanner(g, loadTokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	if err := p.LoadGenres(genresPath); err != nil {
		t.Skip(err)
	}
	p.S = UnderstandSampling
	n := 0
	for _, label := range understandLabels {
		if _, ok := m.Cases[label]; !ok {
			continue
		}
		n++
		_, codes32 := refInt32(t, m, label+"_codes")
		codes := make([]int, len(codes32))
		for i, c := range codes32 {
			codes[i] = int(c)
		}
		song, text, st, err := p.Understand(codes, rand.New(rand.NewPCG(42, 1)))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: %v\n%s", label, st, text)
		if song.Caption == "" || song.BPM < 30 || song.BPM > 300 || song.Duration < 10 || song.Duration > 600 ||
			song.KeyScale == "" || song.TimeSignature == "" || song.Genres == "" || song.Language == "" {
			t.Errorf("%s: song %+v", label, song)
		}
		if _, whole := p.v.genres.children(pyLower(song.Genres)); !whole {
			t.Errorf("%s: genres %q are not in the vocabulary", label, song.Genres)
		}
	}
	if n == 0 {
		t.Skip("no understand cases; run reference/dump_ace_lm.py understand understand_lofi")
	}
}
