package lm

// Sample mode (MUSIC.md A12), upstream's "simple" or "inspiration" mode:
// from a description alone the LM writes the song -- a caption, the metas,
// genres and the lyrics -- in one unguided pass (create_sample_from_query),
// and the thinking path then runs on what it wrote.
//
// This file is the host side, each piece a port checked against
// reference/dump_ace_sample.py: the query's language and instrumental hints
// (parse_description_hints), the chat prompt, the lyrics cut from the text
// after `</think>` (_extract_lyrics_from_output), and create_sample's
// conversion of the parsed fields.

import (
	"fmt"
	"math/rand/v2"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// InspiredInstruction is DEFAULT_LM_INSPIRED_INSTRUCTION.
const InspiredInstruction = "Expand the user's input into a more detailed and specific musical description:"

// SamplePrompt is build_formatted_prompt_for_inspiration. An empty query
// is upstream's NO USER INPUT: a song of the LM's own choosing.
func SamplePrompt(query string, instrumental bool) string {
	if pyStrip(query) == "" {
		query = NoUserInput
	}
	return chatPrompt(InspiredInstruction, query+"\n\ninstrumental: "+strconv.FormatBool(instrumental))
}

// hintLanguages is parse_description_hints' table, in its (dict) order:
// the first name found in the query wins.
var hintLanguages = [][2]string{
	{"english", "en"}, {"en", "en"}, {"chinese", "zh"}, {"中文", "zh"}, {"zh", "zh"}, {"mandarin", "zh"},
	{"japanese", "ja"}, {"日本語", "ja"}, {"ja", "ja"}, {"korean", "ko"}, {"한국어", "ko"}, {"ko", "ko"},
	{"spanish", "es"}, {"español", "es"}, {"es", "es"}, {"french", "fr"}, {"français", "fr"}, {"fr", "fr"},
	{"german", "de"}, {"deutsch", "de"}, {"de", "de"}, {"italian", "it"}, {"italiano", "it"}, {"it", "it"},
	{"portuguese", "pt"}, {"português", "pt"}, {"pt", "pt"}, {"russian", "ru"}, {"русский", "ru"}, {"ru", "ru"},
	{"bengali", "bn"}, {"bn", "bn"}, {"hindi", "hi"}, {"hi", "hi"}, {"arabic", "ar"}, {"ar", "ar"},
	{"thai", "th"}, {"th", "th"}, {"vietnamese", "vi"}, {"vi", "vi"}, {"indonesian", "id"}, {"id", "id"},
	{"turkish", "tr"}, {"tr", "tr"}, {"dutch", "nl"}, {"nl", "nl"}, {"polish", "pl"}, {"pl", "pl"},
}

// DescriptionHints is parse_description_hints: the language a query names
// ("" for none) and whether it asks for an instrumental.
func DescriptionHints(query string) (language string, instrumental bool) {
	q := pyStrip(pyLower(query))
	if q == "" {
		return "", false
	}
	for _, l := range hintLanguages {
		if findWord(q, l[0], utf8.RuneCountInString(l[0]) <= 2) {
			language = l[1]
			break
		}
	}
	instrumental = strings.Contains(q, "instrumental") || strings.Contains(q, "pure music") ||
		strings.Contains(q, "pure instrument") || strings.HasSuffix(q, " solo") || q == "solo"
	return language, instrumental
}

// findWord is the two patterns upstream searches with: a name of at most
// two characters must sit between the ends, whitespace or .,;:!? ; a longer
// one between `\b`s (Python's, which count any letter or digit as a word
// character).
func findWord(s, name string, short bool) bool {
	edge := func(r rune, at bool) bool {
		if !at {
			return true // ^ or $
		}
		if short {
			return isPySpace(r) || strings.ContainsRune(".,;:!?", r)
		}
		return !(unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_')
	}
	for i := 0; i <= len(s)-len(name); {
		j := strings.Index(s[i:], name)
		if j < 0 {
			return false
		}
		j += i
		end := j + len(name)
		before, _ := utf8.DecodeLastRuneInString(s[:j])
		after, _ := utf8.DecodeRuneInString(s[end:])
		if edge(before, j > 0) && edge(after, end < len(s)) {
			return true
		}
		_, w := utf8.DecodeRuneInString(s[j:])
		i = j + w
	}
	return false
}

// ExtractLyrics is _extract_lyrics_from_output: what follows `</think>`,
// stripped, less a leading `# Lyric` header and a trailing `<|im_end|>`.
// Its header pattern is `^#\s*Lyri[c|cs]?\s*\n`, so "# Lyrics" stays.
func ExtractLyrics(text string) string {
	i := strings.Index(text, "</think>")
	if i < 0 {
		return ""
	}
	s := pyStrip(text[i+len("</think>"):])
	if s == "" {
		return ""
	}
	s = cutLyricHeader(s)
	if j := strings.LastIndex(s, "<|im_end|>"); j >= 0 && strings.TrimFunc(s[j+len("<|im_end|>"):], isPySpace) == "" {
		s = s[:j]
	}
	return pyStrip(s)
}

func cutLyricHeader(s string) string {
	if !strings.HasPrefix(s, "#") {
		return s
	}
	rest := strings.TrimLeftFunc(s[1:], isPySpace)
	if len(rest) < 4 || !strings.EqualFold(rest[:4], "lyri") {
		return s
	}
	rest = rest[4:]
	// The whitespace run after, ending at its last newline.
	tail := func(t string) (string, bool) {
		n := len(t) - len(strings.TrimLeftFunc(t, isPySpace))
		if k := strings.LastIndexByte(t[:n], '\n'); k >= 0 {
			return t[k+1:], true
		}
		return "", false
	}
	if rest != "" && strings.ContainsRune("cCsS|", rune(rest[0])) {
		if t, ok := tail(rest[1:]); ok {
			return t
		}
	}
	if t, ok := tail(rest); ok {
		return t
	}
	return s
}

// Song is what sample mode wrote, as create_sample returns it: bpm and
// duration zero when missing, the other fields empty for N/A.
type Song struct {
	Caption, Lyrics                   string
	BPM                               int
	Duration                          float64
	KeyScale, TimeSignature, Language string
	Genres                            string
	Instrumental                      bool
}

// SongOf is create_sample's conversion of the parsed CoT and lyrics.
func SongOf(m Meta, lyrics string, instrumental bool) Song {
	s := Song{Caption: m["caption"], Lyrics: lyrics, KeyScale: m["keyscale"], Language: m["language"],
		TimeSignature: m["timesignature"], Genres: m["genres"], Instrumental: instrumental}
	if v, ok := m["bpm"]; ok && v != "N/A" && v != "" {
		if n, err := strconv.Atoi(strings.TrimPrefix(pyStrip(v), "+")); err == nil {
			s.BPM = n
		}
	}
	if v, ok := m["duration"]; ok && v != "N/A" && v != "" {
		if f, err := strconv.ParseFloat(pyStrip(v), 64); err == nil {
			s.Duration = f
		}
	}
	for _, p := range []*string{&s.KeyScale, &s.Language, &s.TimeSignature} {
		if *p == "N/A" {
			*p = ""
		}
	}
	return s
}

// LoadGenres reads upstream's genres vocabulary, which holds sample mode's
// `genres:` value (and the thinking path's, should its CoT write one).
func (p *Planner) LoadGenres(path string) error {
	g, err := loadGenres(p.v, path)
	if err != nil {
		return fmt.Errorf("lm: genres vocabulary: %w", err)
	}
	p.v.genres = g
	return nil
}

// HasGenres reports that LoadGenres succeeded: sample mode needs it.
func (p *Planner) HasGenres() bool { return p.v.genres != nil }

// Sample is create_sample_from_query: one row, no CFG, the FSM in its
// understand phase with user's fields injected (upstream injects only the
// language; a request's own bpm, duration, key and time signature go in the
// same way), until EOS. The text is what the LM wrote.
func (p *Planner) Sample(query string, instrumental bool, user Meta, rng *rand.Rand) (Song, string, Stats, error) {
	var st Stats
	if !p.HasGenres() {
		return Song{}, "", st, fmt.Errorf("lm: sample mode needs the genres vocabulary (LoadGenres)")
	}
	ids, err := p.Tk.Encode(SamplePrompt(query, instrumental))
	if err != nil {
		return Song{}, "", st, err
	}
	text, st, err := p.decode(ids, newFSM(p.v, user, true), 0, rng)
	if err != nil {
		return Song{}, "", st, err
	}
	lyrics := ExtractLyrics(text)
	if lyrics == "" && instrumental {
		lyrics = "[Instrumental]"
	}
	return SongOf(ParseCoT(text), lyrics, instrumental), text, st, nil
}

// decode runs one unguided row from a prompt under an FSM until EOS or
// maxCoT tokens (_generate_with_constrained_decoding), and returns the text.
func (p *Planner) decode(ids []int32, f *FSM, phase int, rng *rand.Rand) (string, Stats, error) {
	st := Stats{Prompt: len(ids)}
	var err error
	if st.Prefill, err = p.prefill([][]int32{ids}); err != nil {
		return "", st, err
	}
	step := []Tok{{ID: ids[len(ids)-1], Slot: 0, Pos: len(ids) - 1}}
	var gen []int32
	t0 := time.Now()
	for len(gen) < maxCoT {
		if step[0].Pos >= p.G.o.MaxLen {
			return "", st, fmt.Errorf("lm: the generation outgrew the cache")
		}
		out, _, err := p.G.Pass(step, Range{Lo: 0, Hi: p.G.vocab})
		if err != nil {
			return "", st, err
		}
		scores := out[0]
		f.Apply(scores)
		topP(scores, p.S.TopP)
		tok := int32(draw(scores, p.S.Temperature, rng))
		f.Update(tok)
		gen = append(gen, tok)
		if tok == ImEndID || tok == EndOfTextID {
			break
		}
		if err := p.after(phase, len(gen), 0); err != nil {
			return "", st, err
		}
		step[0].ID = tok
		step[0].Pos++
	}
	st.Steps, st.Tokens = time.Since(t0), len(gen)
	text, err := p.Tk.Decode(gen)
	return text, st, err
}
