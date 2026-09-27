// Package lm is ACE-Step 1.5's 5 Hz LM (acestep-5Hz-lm-4B), the planner of
// the thinking path (MUSIC.md A7): a Qwen3-4B fine-tune that writes a YAML
// CoT of metadata (phase 1) and then 5 audio codes a second (phase 2, CFG
// against an unconditional prompt).
//
// This file is the host side that decides what the model reads: the chat
// prompts of both phases, upstream's parse of a phase-1 CoT, and the phase-2
// CoT it re-serialises through PyYAML (`_format_metadata_as_cot`), whose
// plain-scalar folding at width 80 decides where the caption's line breaks
// fall -- and so which tokens the model conditions on.
package lm

import (
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"strix-halo-vulkan/ace/plan"
)

// Instruction is DEFAULT_LM_INSTRUCTION.
const Instruction = "Generate audio semantic tokens based on the given conditions:"

// NoUserInput is the unconditional row's user turn with no negative prompt.
const NoUserInput = "NO USER INPUT"

// Token ids in the LM's vocabulary.
const (
	ImStartID   = 151644
	ImEndID     = 151645 // the EOS of both phases
	EndOfTextID = 151643 // the pad the unconditional row is left-padded with
	ThinkID     = 151667 // <think>
	ThinkEndID  = 151668 // </think>
	// CodeBase is <|audio_code_0|>; codes 0..NumCodes-1 are the valid ones
	// (FSQ levels 8·8·8·5·5·5), and the vocabulary holds 65,535.
	CodeBase = 151669
	NumCodes = 64000
	// CodesPerSecond is the LM's rate: a code is 200 ms, five 25 Hz latents.
	CodesPerSecond = 5
)

// ChatPrompt is the Qwen3 chat template over the system instruction and one
// user turn, with the generation prompt: what apply_chat_template renders
// for upstream's two-message conversations.
func ChatPrompt(user string) string {
	return "<|im_start|>system\n# Instruction\n" + Instruction + "\n\n<|im_end|>\n" +
		"<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
}

// userTurn is the conditional user turn of both phases.
func userTurn(caption, lyrics string) string {
	return "# Caption\n" + caption + "\n\n# Lyric\n" + lyrics + "\n"
}

// Phase1Prompt is build_formatted_prompt: the CoT phase's only row (no CFG).
// The FSM then forces `<think>` as the first generated token.
func Phase1Prompt(caption, lyrics string) string {
	return ChatPrompt(userTurn(caption, lyrics))
}

// Phase2Prompt is build_formatted_prompt_with_cot: the assistant turn left
// open after the CoT and its `\n\n`, where the codes go.
func Phase2Prompt(caption, lyrics, cot string) string {
	return ChatPrompt(userTurn(caption, lyrics)) + cot + "\n\n"
}

// Phase2Uncond is the codes phase's unconditional row with no negative
// prompt: the raw `NO USER INPUT` and an empty reasoning block.
func Phase2Uncond() string {
	return ChatPrompt(NoUserInput) + "<think>\n\n</think>\n\n"
}

// metaKeys is _format_metadata_as_cot's key list, which is also the sorted
// order yaml.dump writes them in.
var metaKeys = []string{"bpm", "caption", "duration", "keyscale", "language", "timesignature"}

// Meta is a CoT's metadata as upstream holds it between the phases: each
// key present or not, its value the text parse_lm_output (or the request)
// gave it. Digit-only values are ints to yaml.dump, which writes them the
// same way, so strings are enough.
type Meta map[string]string

// CoT is _format_metadata_as_cot: the present keys through
// `yaml.dump(allow_unicode=True, sort_keys=True)`, stripped, inside
// `<think>\n…\n</think>`. A time signature `N/4` is written as `N`.
func (m Meta) CoT() string {
	var b strings.Builder
	for _, k := range metaKeys {
		v, ok := m[k]
		if !ok {
			continue
		}
		if k == "timesignature" && strings.HasSuffix(v, "/4") {
			v = strings.SplitN(v, "/", 2)[0]
		}
		b.WriteString(k)
		b.WriteString(":")
		if isDigits(v) {
			// Python's int(v) then yaml's repr: leading zeros go.
			v = strings.TrimLeft(v, "0")
			if v == "" {
				v = "0"
			}
			b.WriteString(" " + v + "\n")
			continue
		}
		col := len(k) + 1
		b.WriteString(yamlScalar(v, col))
		b.WriteString("\n")
	}
	return "<think>\n" + strings.TrimSpace(b.String()) + "\n</think>"
}

// isDigits is Python's str.isdigit on ASCII: non-empty, digits only.
func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// ParseCoT is parse_lm_output's metadata half: the text between <think>
// and </think> (or before the first code), one `key: value` a line, with
// indented continuation lines joined into the value. A caption's YAML
// folding is undone (postprocess_caption); every other value is stripped.
func ParseCoT(text string) Meta {
	reasoning := ""
	if i := strings.Index(text, "<think>"); i >= 0 {
		if j := strings.Index(text[i+len("<think>"):], "</think>"); j >= 0 {
			reasoning = strings.TrimSpace(text[i+len("<think>") : i+len("<think>")+j])
		}
	}
	if reasoning == "" {
		if i := strings.Index(text, "<|audio_code_"); i >= 0 {
			text = text[:i]
		}
		reasoning = strings.TrimSpace(text)
	}
	m := Meta{}
	key := ""
	var lines []string
	save := func() {
		if key != "" && len(lines) > 0 {
			v := strings.Join(lines, "\n")
			switch key {
			case "caption":
				m[key] = postprocessCaption(v)
			case "bpm", "duration", "genres", "keyscale", "language", "timesignature":
				// bpm and duration are int() when they parse, which prints
				// the same digits as the stripped text for every value a
				// digit-only CoT yields.
				m[key] = pyStrip(v)
			}
		}
		key, lines = "", nil
	}
	for _, line := range strings.Split(reasoning, "\n") {
		if strings.HasPrefix(pyStrip(line), "<") {
			continue
		}
		if line != "" && !isPySpace(firstRune(line)) && strings.Contains(line, ":") {
			save()
			parts := strings.SplitN(line, ":", 2)
			key = strings.ToLower(pyStrip(parts[0]))
			if pyStrip(parts[1]) != "" {
				lines = append(lines, parts[1])
			}
		} else if strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t") {
			if key != "" {
				lines = append(lines, line)
			}
		}
	}
	save()
	return m
}

// postprocessCaption is MetadataConstrainedLogitsProcessor.postprocess_caption:
// every non-empty stripped line, joined by one space.
func postprocessCaption(s string) string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if l = pyStrip(l); l != "" {
			out = append(out, l)
		}
	}
	return strings.Join(out, " ")
}

func firstRune(s string) rune {
	r, _ := utf8.DecodeRuneInString(s)
	return r
}

// isPySpace is Python's str.isspace for one character.
func isPySpace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\r', '\v', '\f', 0x1c, 0x1d, 0x1e, 0x1f, 0x85, 0xa0,
		0x1680, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000:
		return true
	}
	return r >= 0x2000 && r <= 0x200a
}

// pyStrip is Python's str.strip() with no argument.
func pyStrip(s string) string { return strings.TrimFunc(s, isPySpace) }

// ---- PyYAML's emitter, for one string value of a top-level block mapping.
//
// The value is written after `key:` at column col; continuation lines are
// indented 2 (expect_scalar's increase_indent over the mapping's 0), and a
// line breaks at a single space once the column passes best_width, 80.

const yamlWidth = 80

// yamlResolvers are the implicit resolvers a str's plain form must not
// match, or yaml.dump quotes it: bool, float, int, merge, null, timestamp,
// value and yaml, keyed as PyYAML keys them by first character.
var yamlResolvers = []struct {
	first string
	re    *regexp.Regexp
}{
	{"yYnNtTfFoO", regexp.MustCompile(`^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$`)},
	{"-+0123456789.", regexp.MustCompile(`^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$`)},
	{"-+0123456789", regexp.MustCompile(`^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$`)},
	{"<", regexp.MustCompile(`^(?:<<)$`)},
	{"~nN", regexp.MustCompile(`^(?:~|null|Null|NULL|)$`)},
	{"0123456789", regexp.MustCompile(`^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$`)},
	{"=", regexp.MustCompile(`^(?:=)$`)},
	{"!&*", regexp.MustCompile(`^(?:!|&|\*)$`)},
}

// yamlImplicitStr is the serializer's implicit[0] for a str: its plain form
// resolves back to a str.
func yamlImplicitStr(s string) bool {
	if s == "" {
		return false // the null resolver's '' entry
	}
	f := string(firstRune(s))
	for _, r := range yamlResolvers {
		if strings.Contains(r.first, f) && r.re.MatchString(s) {
			return false
		}
	}
	return true
}

func isYAMLBreak(r rune) bool { return r == '\n' || r == 0x85 || r == 0x2028 || r == 0x2029 }

func isYAMLSpaceOrEnd(r rune) bool {
	return r == 0 || r == ' ' || r == '\t' || r == '\r' || isYAMLBreak(r)
}

// yamlAnalysis is analyze_scalar's verdicts for the three styles a
// top-level block value can take.
type yamlAnalysis struct {
	empty, multiline                 bool
	blockPlain, singleQuoted, simple bool
}

func analyzeScalar(rs []rune) yamlAnalysis {
	if len(rs) == 0 {
		return yamlAnalysis{empty: true, blockPlain: true, singleQuoted: true}
	}
	s := string(rs)
	blockInd := strings.HasPrefix(s, "---") || strings.HasPrefix(s, "...")
	var lineBreaks, special, leadSpace, leadBreak, trailSpace, trailBreak, breakSpace, spaceBreak bool
	preceded := true
	followed := len(rs) == 1 || isYAMLSpaceOrEnd(rs[1])
	prevSpace, prevBreak := false, false
	for i, ch := range rs {
		if i == 0 {
			if strings.ContainsRune("#,[]{}&*!|>'\"%@`", ch) {
				blockInd = true
			}
			if (ch == '?' || ch == ':') && followed {
				blockInd = true
			}
			if ch == '-' && followed {
				blockInd = true
			}
		} else {
			if ch == ':' && followed {
				blockInd = true
			}
			if ch == '#' && preceded {
				blockInd = true
			}
		}
		if isYAMLBreak(ch) {
			lineBreaks = true
		}
		if !(ch == '\n' || (ch >= 0x20 && ch <= 0x7e)) {
			// allow_unicode: printable non-ASCII is not special.
			if !((ch == 0x85 || (ch >= 0xa0 && ch <= 0xd7ff) || (ch >= 0xe000 && ch <= 0xfffd) ||
				(ch >= 0x10000 && ch < 0x10ffff)) && ch != 0xfeff) {
				special = true
			}
		}
		switch {
		case ch == ' ':
			if i == 0 {
				leadSpace = true
			}
			if i == len(rs)-1 {
				trailSpace = true
			}
			if prevBreak {
				breakSpace = true
			}
			prevSpace, prevBreak = true, false
		case isYAMLBreak(ch):
			if i == 0 {
				leadBreak = true
			}
			if i == len(rs)-1 {
				trailBreak = true
			}
			if prevSpace {
				spaceBreak = true
			}
			prevSpace, prevBreak = false, true
		default:
			prevSpace, prevBreak = false, false
		}
		preceded = isYAMLSpaceOrEnd(ch)
		followed = i+2 >= len(rs) || isYAMLSpaceOrEnd(rs[i+2])
	}
	a := yamlAnalysis{multiline: lineBreaks, blockPlain: true, singleQuoted: true}
	if leadSpace || leadBreak || trailSpace || trailBreak {
		a.blockPlain = false
	}
	if breakSpace {
		a.blockPlain, a.singleQuoted = false, false
	}
	if spaceBreak || special {
		a.blockPlain, a.singleQuoted = false, false
	}
	if lineBreaks || blockInd {
		a.blockPlain = false
	}
	return a
}

// yamlScalar writes a str value the way yaml.dump does after `key:` at
// column col, including the separating space.
func yamlScalar(v string, col int) string {
	rs := []rune(v)
	a := analyzeScalar(rs)
	w := &yamlWriter{col: col}
	switch {
	case yamlImplicitStr(v) && a.blockPlain:
		w.plain(rs)
	case a.singleQuoted:
		w.singleQuoted(rs)
	default:
		w.doubleQuoted(rs)
	}
	return w.b.String()
}

// yamlWriter tracks what the emitter tracks while it writes one value.
type yamlWriter struct {
	b          strings.Builder
	col        int
	whitespace bool // the last thing written was whitespace
	indention  bool
}

const yamlIndent = 2

func (w *yamlWriter) write(s string) {
	w.b.WriteString(s)
	w.col += utf8.RuneCountInString(s)
}

func (w *yamlWriter) lineBreak(s string) {
	w.whitespace, w.indention = true, true
	w.b.WriteString(s)
	w.col = 0
}

func (w *yamlWriter) indent() {
	if !w.indention || w.col > yamlIndent || (w.col == yamlIndent && !w.whitespace) {
		w.lineBreak("\n")
	}
	if w.col < yamlIndent {
		w.whitespace = true
		w.b.WriteString(strings.Repeat(" ", yamlIndent-w.col))
		w.col = yamlIndent
	}
}

// indicator is write_indicator for a quote: preceded by a space unless the
// last thing written was whitespace.
func (w *yamlWriter) indicator(s string, needWhitespace bool) {
	if !(w.whitespace || !needWhitespace) {
		s = " " + s
	}
	w.whitespace = false
	w.indention = false
	w.write(s)
}

// lineBreaks writes a run of breaks as write_plain and write_single_quoted
// do: a leading '\n' doubles.
func (w *yamlWriter) breaks(run []rune) {
	if run[0] == '\n' {
		w.lineBreak("\n")
	}
	for _, br := range run {
		w.lineBreak(string(br))
	}
	w.indent()
}

func (w *yamlWriter) plain(rs []rune) {
	if len(rs) == 0 {
		return
	}
	if !w.whitespace {
		w.write(" ")
	}
	w.whitespace, w.indention = false, false
	spaces, brk := false, false
	start := 0
	for end := 0; end <= len(rs); end++ {
		var ch rune = -1
		if end < len(rs) {
			ch = rs[end]
		}
		switch {
		case spaces:
			if ch != ' ' {
				if start+1 == end && w.col > yamlWidth {
					w.indent()
					w.whitespace, w.indention = false, false
				} else {
					w.write(string(rs[start:end]))
				}
				start = end
			}
		case brk:
			if ch < 0 || !isYAMLBreak(ch) {
				w.breaks(rs[start:end])
				w.whitespace, w.indention = false, false
				start = end
			}
		default:
			if ch < 0 || ch == ' ' || isYAMLBreak(ch) {
				w.write(string(rs[start:end]))
				start = end
			}
		}
		if ch >= 0 {
			spaces, brk = ch == ' ', isYAMLBreak(ch)
		}
	}
}

func (w *yamlWriter) singleQuoted(rs []rune) {
	w.indicator("'", true)
	spaces, brk := false, false
	start := 0
	for end := 0; end <= len(rs); end++ {
		var ch rune = -1
		if end < len(rs) {
			ch = rs[end]
		}
		switch {
		case spaces:
			if ch != ' ' {
				if start+1 == end && w.col > yamlWidth && start != 0 && end != len(rs) {
					w.indent()
				} else {
					w.write(string(rs[start:end]))
				}
				start = end
			}
		case brk:
			if ch < 0 || !isYAMLBreak(ch) {
				w.breaks(rs[start:end])
				start = end
			}
		default:
			if ch < 0 || ch == ' ' || isYAMLBreak(ch) || ch == '\'' {
				if start < end {
					w.write(string(rs[start:end]))
					start = end
				}
			}
		}
		if ch == '\'' {
			w.write("''")
			start = end + 1
		}
		if ch >= 0 {
			spaces, brk = ch == ' ', isYAMLBreak(ch)
		}
	}
	w.indicator("'", false)
}

var yamlEscapes = map[rune]string{
	0: "0", 0x07: "a", 0x08: "b", 0x09: "t", 0x0a: "n", 0x0b: "v", 0x0c: "f", 0x0d: "r",
	0x1b: "e", '"': "\"", '\\': "\\", 0x85: "N", 0xa0: "_", 0x2028: "L", 0x2029: "P",
}

func (w *yamlWriter) doubleQuoted(rs []rune) {
	w.indicator("\"", true)
	start := 0
	for end := 0; end <= len(rs); end++ {
		var ch rune = -1
		if end < len(rs) {
			ch = rs[end]
		}
		if ch < 0 || strings.ContainsRune("\"\\\u0085\u2028\u2029\ufeff", ch) ||
			!((ch >= 0x20 && ch <= 0x7e) || (ch >= 0xa0 && ch <= 0xd7ff) || (ch >= 0xe000 && ch <= 0xfffd)) {
			if start < end {
				w.write(string(rs[start:end]))
				start = end
			}
			if ch >= 0 {
				var data string
				if e, ok := yamlEscapes[ch]; ok {
					data = "\\" + e
				} else if ch <= 0xff {
					data = "\\x" + hexUpper(uint32(ch), 2)
				} else if ch <= 0xffff {
					data = "\\u" + hexUpper(uint32(ch), 4)
				} else {
					data = "\\U" + hexUpper(uint32(ch), 8)
				}
				w.write(data)
				start = end + 1
			}
		}
		if end > 0 && end < len(rs)-1 && (ch == ' ' || start >= end) && w.col+(end-start) > yamlWidth {
			data := "\\" // Python's text[start:end] is empty once start passes end
			if start < end {
				data = string(rs[start:end]) + data
			}
			if start < end {
				start = end
			}
			w.write(data)
			w.indent()
			w.whitespace, w.indention = false, false
			if rs[start] == ' ' {
				w.write("\\")
			}
		}
	}
	w.indicator("\"", false)
}

func hexUpper(v uint32, n int) string {
	const digits = "0123456789ABCDEF"
	out := make([]byte, n)
	for i := n - 1; i >= 0; i-- {
		out[i] = digits[v&15]
		v >>= 4
	}
	return string(out)
}

// UserMeta is generate_music's user_metadata: the metas a request gives,
// which the FSM injects in phase 1 and which skip it when all four of bpm,
// key, time signature and duration are there (HasAll). Language is not one
// of them.
func UserMeta(r *plan.Request) Meta {
	m := Meta{}
	if r.BPM > 0 {
		m["bpm"] = strconv.Itoa(r.BPM)
	}
	if ks := pyStrip(r.KeyScale); ks != "" && strings.ToLower(ks) != "n/a" {
		m["keyscale"] = ks
	}
	if ts := pyStrip(r.TimeSignature); ts != "" && strings.ToLower(ts) != "n/a" {
		m["timesignature"] = ts
	}
	if r.Duration > 0 {
		m["duration"] = strconv.Itoa(int(r.Duration))
	}
	return m
}

// HasAll is has_all_metas: with these four given, phase 1 does not run and
// the CoT is the user's metas alone.
func (m Meta) HasAll() bool {
	for _, k := range []string{"bpm", "keyscale", "timesignature", "duration"} {
		if _, ok := m[k]; !ok {
			return false
		}
	}
	return true
}
