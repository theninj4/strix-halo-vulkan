package mm3

// The prompt (N7): upstream's encoders.py at dafe3733, which the card says is
// part of the checkpoint's contract ("even whitespace-level changes to the
// assembled prompt change the generated audio"). _clean_caption and
// _normalize_lyrics are transcribed rule for rule; the places Go's regexp
// differs from Python's re are done by hand:
//
//   - Python's \s on a str is Unicode whitespace (str.isspace), Go's is
//     ASCII: the patterns spell out pyspace.
//   - str.splitlines splits on more than \n (\r, \r\n, \v, \f, \x1c-\x1e,
//     \x85,  ,  ) and drops a trailing empty line: splitLines.
//   - `(?<!\*)\*([^*\n]+)\*(?!\*)` needs look-around, which RE2 lacks: a
//     scan (italics).
//
// reference/dump_mm3_prompt.py pins all of it against the reference's own
// functions and token ids (TestPrompt).

import (
	"fmt"
	"regexp"
	"strings"

	"strix-halo-vulkan/zimage/tokenizer"
)

const (
	// AudioCFGToken replaces every prompt token but the first and the last
	// two in the unconditional row.
	AudioCFGToken = 151654
	// MaxPromptTokens is upstream's bound.
	MaxPromptTokens = 5000
)

// pyspace is Python's \s over str: what str.isspace() accepts.
const pyspace = `[\t\n\v\f\r \x{1c}-\x{1f}\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}]`

var (
	specialTagRE = regexp.MustCompile(`<\|([^|]*)\|>`)
	headingRE    = regexp.MustCompile(`^` + pyspace + `{0,3}#{1,6}` + pyspace + `+`)
	bulletRE     = regexp.MustCompile(`^` + pyspace + `*[*+-]` + pyspace + `+`)
	starBulletRE = regexp.MustCompile(`^` + pyspace + `*\*` + pyspace + `+`)
	boldRE       = regexp.MustCompile(`\*\*([^*]+)\*\*`)
	ruleRE       = regexp.MustCompile(`(?m)^` + pyspace + `*[-*_]{3,}` + pyspace + `*$`)
	blankRunRE   = regexp.MustCompile(`\n{2,}`)
	leadingTags  = regexp.MustCompile(`^[ \t]*((?:\[[^\]]+\][ \t]*)+)`)
	tagRE        = regexp.MustCompile(`\[([^\]]+)\]`)
	pyspaceRE    = regexp.MustCompile(pyspace + `+`)
)

// CleanCaption is upstream's _clean_caption: special tags rewritten
// (<|bpm 120|> is "bpm is 120"), the markdown forms the model's input
// contract accepts stripped, blank lines collapsed.
func CleanCaption(caption string) string {
	text := specialTagRE.ReplaceAllStringFunc(caption, func(m string) string {
		inner := pyStrip(m[2 : len(m)-2])
		// str.split(None, 1): the first whitespace run splits, the rest stays.
		if loc := pyspaceRE.FindStringIndex(inner); loc != nil {
			return inner[:loc[0]] + " is " + inner[loc[1]:]
		}
		return inner
	})
	lines := splitLines(text)
	for i, line := range lines {
		line = headingRE.ReplaceAllString(line, "")
		line = bulletRE.ReplaceAllString(line, "")
		line = starBulletRE.ReplaceAllString(line, "")
		for strings.Contains(line, "**") {
			updated := boldRE.ReplaceAllString(line, "$1")
			if updated == line {
				break
			}
			line = updated
		}
		lines[i] = pyRStrip(italics(line))
	}
	text = strings.Join(lines, "\n")
	text = ruleRE.ReplaceAllString(text, "")
	text = strings.ReplaceAll(text, "• ", "")
	text = strings.ReplaceAll(text, "    ", "")
	return blankRunRE.ReplaceAllString(text, "\n")
}

// italics is re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", s): a '*' not
// after a '*', a run of neither '*' nor newline, a '*' not before a '*'.
// The run cannot end anywhere but at the next '*' or newline, so the match
// is decided there; the look-behind reads the original string.
func italics(s string) string {
	var b strings.Builder
	i, last := 0, 0
	for i < len(s) {
		if s[i] != '*' || (i > 0 && s[i-1] == '*') {
			i++
			continue
		}
		j := i + 1
		for j < len(s) && s[j] != '*' && s[j] != '\n' {
			j++
		}
		if j > i+1 && j < len(s) && s[j] == '*' && (j+1 >= len(s) || s[j+1] != '*') {
			b.WriteString(s[last:i])
			b.WriteString(s[i+1 : j])
			i, last = j+1, j+1
			continue
		}
		i++
	}
	b.WriteString(s[last:])
	return b.String()
}

// NormalizeLyrics is upstream's _normalize_lyrics: a line's leading
// structure tags kept and the rest of the line dropped, tags split onto
// their own lines, " ^ " a line break, tags lowercased, "[start]" first.
func NormalizeLyrics(lyrics string) string {
	lines := strings.Split(lyrics, "\n")
	for i, line := range lines {
		if m := leadingTags.FindStringSubmatch(line); m != nil {
			lines[i] = pyStrip(m[1])
		}
	}
	text := strings.Join(lines, "\n")
	text = strings.ReplaceAll(text, "] ", "]\n")
	text = strings.ReplaceAll(text, " [", "\n[")
	text = strings.ReplaceAll(text, " ^ ", "\n")
	text = tagRE.ReplaceAllStringFunc(text, func(m string) string {
		return "[" + strings.ToLower(m[1:len(m)-1]) + "]"
	})
	return "[start]\n" + text
}

// PromptText is the assembled prompt string.
func PromptText(caption, lyrics string) string {
	return "<|im_start|><|caption_start|>" + CleanCaption(caption) + "<|caption_end|>" +
		"<|lyrics_start|>" + NormalizeLyrics(lyrics) + "<|lyrics_end|><|im_end|><|audio_start|>"
}

// Prompt is the CFG pair's token ids: the prompt, and its unconditional
// counterpart (every token but the first and the last two replaced).
func Prompt(tok *tokenizer.Tokenizer, caption, lyrics string) (cond, uncond []int32, err error) {
	if strings.TrimSpace(caption) == "" || strings.TrimSpace(lyrics) == "" {
		return nil, nil, fmt.Errorf("mm3: the caption and the lyrics must both be non-empty")
	}
	cond, err = tok.Encode(PromptText(caption, lyrics))
	if err != nil {
		return nil, nil, err
	}
	if len(cond) > MaxPromptTokens {
		return nil, nil, fmt.Errorf("mm3: the prompt is %d tokens; the most is %d", len(cond), MaxPromptTokens)
	}
	uncond = append([]int32(nil), cond...)
	for i := 1; i < len(uncond)-2; i++ {
		uncond[i] = AudioCFGToken
	}
	return cond, uncond, nil
}

func isPySpace(r rune) bool {
	switch {
	case r == '\t', r == '\n', r == '\v', r == '\f', r == '\r', r == ' ', r >= 0x1c && r <= 0x1f, r == 0x85, r == 0xa0,
		r == 0x1680, r >= 0x2000 && r <= 0x200a, r == 0x2028, r == 0x2029, r == 0x202f, r == 0x205f, r == 0x3000:
		return true
	}
	return false
}

func pyStrip(s string) string  { return strings.TrimFunc(s, isPySpace) }
func pyRStrip(s string) string { return strings.TrimRightFunc(s, isPySpace) }

// splitLines is str.splitlines(): every line boundary Python knows, \r\n as
// one, no trailing empty line.
func splitLines(s string) []string {
	var out []string
	rs := []rune(s)
	var b strings.Builder
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch r {
		case '\n', '\v', '\f', 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029:
			out = append(out, b.String())
			b.Reset()
		case '\r':
			out = append(out, b.String())
			b.Reset()
			if i+1 < len(rs) && rs[i+1] == '\n' {
				i++
			}
		default:
			b.WriteRune(r)
		}
	}
	if b.Len() > 0 {
		out = append(out, b.String())
	}
	return out
}
