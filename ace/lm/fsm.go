package lm

// The phase-1 constrained-decoding FSM: upstream's
// MetadataConstrainedLogitsProcessor as the default thinking path configures
// it (MUSIC.md A-o5) -- single mode, phase "cot", stop at reasoning, genres
// skipped, caption and language generated, the request's metas injected.
//
// It is a port of the Python state for state, including its quirks: a
// caption ends when the raw argmax after a newline is not indentation (and
// the model then writes the next field name unconstrained, which may skip
// fields); the language is the top-1 candidate, forced; a user's value is
// injected as the tokens of " value\n". reference/dump_ace_fsm.py drives the
// upstream processor through scripted CoTs and TestFSM replays them.
//
// Sample mode (MUSIC.md A12) is the same processor as create_sample_from_
// query configures it: phase "understand", genres generated, no stop at
// reasoning. It writes `</think>` and goes on to free-form lyrics, where
// only the audio codes are masked. A `genres:` value is held to upstream's
// vocabulary (genres.go) whenever it was loaded; without it the value is
// free until its newline.

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"strix-halo-vulkan/zimage/tokenizer"
)

type fsmState int

const (
	stThinkTag fsmState = iota
	stNewlineAfterThink
	stBPMName
	stBPMValue
	stCaptionName
	stCaptionValue
	stDurationName
	stDurationValue
	stGenresName
	stGenresValue
	stKeyscaleName
	stKeyscaleValue
	stLanguageName
	stLanguageValue
	stTimesigName
	stTimesigValue
	stThinkEndTag
	stCodes
	stCompleted
)

var fsmStateNames = map[fsmState]string{
	stThinkTag: "THINK_TAG", stNewlineAfterThink: "NEWLINE_AFTER_THINK", stBPMName: "BPM_NAME",
	stBPMValue: "BPM_VALUE", stCaptionName: "CAPTION_NAME", stCaptionValue: "CAPTION_VALUE",
	stDurationName: "DURATION_NAME", stDurationValue: "DURATION_VALUE", stGenresName: "GENRES_NAME",
	stGenresValue: "GENRES_VALUE", stKeyscaleName: "KEYSCALE_NAME", stKeyscaleValue: "KEYSCALE_VALUE",
	stLanguageName: "LANGUAGE_NAME", stLanguageValue: "LANGUAGE_VALUE", stTimesigName: "TIMESIG_NAME",
	stTimesigValue: "TIMESIG_VALUE", stThinkEndTag: "THINK_END_TAG", stCodes: "CODES_GENERATION",
	stCompleted: "COMPLETED",
}

func (s fsmState) String() string { return fsmStateNames[s] }

var fixedStrings = map[fsmState]string{
	stThinkTag: "<think>", stNewlineAfterThink: "\n", stBPMName: "bpm:", stCaptionName: "caption:",
	stDurationName: "duration:", stGenresName: "genres:", stKeyscaleName: "keyscale:",
	stLanguageName: "language:", stTimesigName: "timesignature:", stThinkEndTag: "</think>",
}

// The value sets, from acestep/constants.py.
var (
	// ValidLanguages are the codes the FSM lets the LM write.
	ValidLanguages = []string{"ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
		"es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
		"is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
		"pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
		"ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
		"unknown"}
	validTimesigs = []string{"2", "3", "4", "6"}
)

const (
	bpmMin, bpmMax           = 30, 300
	durationMin, durationMax = 10, 600
	captionMaxTokens         = 512
)

// fieldOrder is the CoT's, genres included.
var fieldOrder = []string{"bpm", "caption", "duration", "genres", "keyscale", "language", "timesignature"}

func validKeyscales() []string {
	var out []string
	for _, note := range []string{"A", "B", "C", "D", "E", "F", "G"} {
		for _, acc := range []string{"", "#", "b", "♯", "♭"} {
			for _, mode := range []string{"major", "minor"} {
				out = append(out, note+acc+" "+mode)
			}
		}
	}
	return out
}

// prefixTree maps a token-id prefix (joined) to the set of ids that may
// follow it; a complete value allows the newline.
type prefixTree map[string][]int32

func treeKey(ids []int32) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(strconv.Itoa(int(id)))
		b.WriteByte(',')
	}
	return b.String()
}

func (t prefixTree) add(key string, id int32) {
	for _, x := range t[key] {
		if x == id {
			return
		}
	}
	t[key] = append(t[key], id)
}

func (t prefixTree) has(key string, id int32) bool {
	for _, x := range t[key] {
		if x == id {
			return true
		}
	}
	return false
}

// fsmVocab is what the FSM precomputes from the tokenizer, once.
type fsmVocab struct {
	tk                                     *tokenizer.Tokenizer
	vocab                                  int
	newline, backtick                      int32
	bpm, duration, timesig, keyscale, lang prefixTree
	langFirst                              []int32
	decoded                                map[int32]string
	fixed                                  map[string]int32 // remaining fixed text -> its longest single-token prefix
	fixedLen                               map[string]int
	genres                                 *genreVocab // nil until loaded
}

func newFSMVocab(tk *tokenizer.Tokenizer) (*fsmVocab, error) {
	v := &fsmVocab{tk: tk, vocab: tk.Size(), decoded: map[int32]string{}, fixed: map[string]int32{}, fixedLen: map[string]int{}}
	last := func(s string) (int32, error) {
		ids, err := tk.Encode(s)
		if err != nil || len(ids) == 0 {
			return 0, fmt.Errorf("lm: encoding %q: %v", s, err)
		}
		return ids[len(ids)-1], nil
	}
	var err error
	if v.newline, err = last("\n"); err != nil {
		return nil, err
	}
	if v.backtick, err = last("`"); err != nil {
		return nil, err
	}
	build := func(values []string, match, tokenize string, firstIsNote bool) (prefixTree, error) {
		ctx, err := tk.Encode(match)
		if err != nil {
			return nil, err
		}
		t := prefixTree{}
		for _, val := range values {
			full, err := tk.Encode(tokenize + val)
			if err != nil {
				return nil, err
			}
			if len(full) < len(ctx) || equalIDs(full[:len(ctx)], ctx) >= 0 {
				continue
			}
			ids := full[len(ctx):]
			if len(ids) == 0 {
				continue
			}
			if firstIsNote {
				s := strings.TrimLeftFunc(v.decode(ids[0]), isPySpace)
				if s == "" || !strings.ContainsRune("ABCDEFG", unicode.ToUpper(firstRune(s))) {
					continue
				}
			}
			for i := 0; i <= len(ids); i++ {
				key := treeKey(ids[:i])
				if _, ok := t[key]; !ok {
					t[key] = nil
				}
				if i < len(ids) {
					t.add(key, ids[i])
				} else {
					t.add(key, v.newline)
				}
			}
		}
		return t, nil
	}
	nums := func(lo, hi int) []string {
		var out []string
		for i := lo; i <= hi; i++ {
			out = append(out, strconv.Itoa(i))
		}
		return out
	}
	if v.bpm, err = build(nums(bpmMin, bpmMax), "bpm:", "bpm: ", false); err != nil {
		return nil, err
	}
	if v.duration, err = build(nums(durationMin, durationMax), "duration:", "duration: ", false); err != nil {
		return nil, err
	}
	if v.timesig, err = build(validTimesigs, "timesignature:", "timesignature: ", false); err != nil {
		return nil, err
	}
	if v.keyscale, err = build(validKeyscales(), "keyscale:", "keyscale: ", true); err != nil {
		return nil, err
	}
	if v.lang, err = build(ValidLanguages, "language:", "language: ", false); err != nil {
		return nil, err
	}
	v.langFirst = append([]int32(nil), v.lang[""]...)
	sort.Slice(v.langFirst, func(i, j int) bool { return v.langFirst[i] < v.langFirst[j] })
	return v, nil
}

// decode is tokenizer.decode([id]) as the FSM uses it.
func (v *fsmVocab) decode(id int32) string {
	if s, ok := v.decoded[id]; ok {
		return s
	}
	s, err := v.tk.Decode([]int32{id})
	if err != nil {
		s = ""
	}
	v.decoded[id] = s
	return s
}

// fixedToken is _get_allowed_tokens_for_fixed_string's first pass: the
// longest prefix of the remaining text that encodes to one token.
func (v *fsmVocab) fixedToken(remaining string) (int32, bool) {
	if id, ok := v.fixed[remaining]; ok {
		return id, true
	}
	rs := []rune(remaining)
	for end := len(rs); end > 0; end-- {
		ids, err := v.tk.Encode(string(rs[:end]))
		if err == nil && len(ids) == 1 {
			v.fixed[remaining] = ids[0]
			return ids[0], true
		}
	}
	return 0, false
}

// FSM is one phase-1 (or sample-mode) generation's state.
type FSM struct {
	v    *fsmVocab
	user Meta
	eos  int32
	// sample is the understand phase: genres generated, `</think>` written
	// rather than stopped at, then lyrics.
	sample bool

	state               fsmState
	pos                 int // characters of the fixed string written
	accIDs              []int32
	accValue            string
	captionAfterNewline bool
	captionEnding       bool
	captionCount        int
	pending             string
	queue               []int32
	curField            string
	next                map[fsmState]fsmState
}

// newFSM starts phase 1 with the request's metas (UserMeta), or with
// sample set, sample mode's pass with its (the language, and any metas the
// request gave).
func newFSM(v *fsmVocab, user Meta, sample bool) *FSM {
	f := &FSM{v: v, user: user, eos: ImEndID, state: stThinkTag, sample: sample}
	f.next = map[fsmState]fsmState{
		stThinkTag: stNewlineAfterThink, stNewlineAfterThink: stBPMName, stThinkEndTag: stCodes, stCodes: stCompleted,
		stBPMName: stBPMValue, stBPMValue: f.nextField("bpm"),
		stCaptionName: stCaptionValue, stCaptionValue: f.nextField("caption"),
		stDurationName: stDurationValue, stDurationValue: f.nextField("duration"),
		stKeyscaleName: stKeyscaleValue, stKeyscaleValue: f.nextField("keyscale"),
		stLanguageName: stLanguageValue, stLanguageValue: f.nextField("language"),
		stTimesigName: stTimesigValue, stTimesigValue: stThinkEndTag,
	}
	if sample {
		f.next[stGenresName] = stGenresValue
		f.next[stGenresValue] = f.nextField("genres")
	}
	return f
}

// nextField is _get_next_field_state: genres are skipped but in sample mode.
func (f *FSM) nextField(field string) fsmState {
	states := map[string]fsmState{"bpm": stBPMName, "caption": stCaptionName, "duration": stDurationName,
		"genres": stGenresName, "keyscale": stKeyscaleName, "language": stLanguageName, "timesignature": stTimesigName}
	for i, name := range fieldOrder {
		if name != field {
			continue
		}
		for _, n := range fieldOrder[i+1:] {
			if n == "genres" && !f.sample {
				continue
			}
			return states[n]
		}
	}
	return stThinkEndTag
}

var negInf = float32(math.Inf(-1))

// whitelist keeps only the allowed ids' scores.
func whitelist(scores []float32, allowed []int32) {
	saved := make([]float32, len(allowed))
	for i, id := range allowed {
		saved[i] = scores[id]
	}
	for i := range scores {
		scores[i] = negInf
	}
	for i, id := range allowed {
		scores[id] = saved[i]
	}
}

func argmax(scores []float32) int32 {
	best := 0
	for i, s := range scores {
		if s > scores[best] {
			best = i
		}
	}
	return int32(best)
}

// Apply masks one step's full-vocabulary scores in place.
func (f *FSM) Apply(scores []float32) {
	if f.state == stCompleted {
		if f.sample {
			for i := CodeBase; i < CodeBase+NumCodes; i++ {
				scores[i] = negInf // the lyrics are text
			}
		}
		return
	}
	f.process(scores)
}

func (f *FSM) inject(scores []float32, field string) bool {
	val, ok := f.user[field]
	if !ok || len(f.queue) > 0 {
		return false
	}
	ids, err := f.v.tk.Encode(" " + val + "\n")
	if err != nil || len(ids) == 0 {
		return false
	}
	f.queue, f.curField = ids, field
	whitelist(scores, ids[:1])
	return true
}

func (f *FSM) process(scores []float32) {
	if len(f.queue) > 0 {
		whitelist(scores, f.queue[:1])
		return
	}
	if fixed, ok := fixedStrings[f.state]; ok {
		rs := []rune(fixed)
		if f.pos < len(rs) {
			remaining := string(rs[f.pos:])
			if f.state == stThinkEndTag && !f.sample && len(rs)-f.pos <= 10 {
				whitelist(scores, []int32{f.eos}) // stop at reasoning
				return
			}
			id, ok := f.v.fixedToken(remaining)
			if !ok {
				panic(fmt.Sprintf("lm: no single token starts %q", remaining))
			}
			whitelist(scores, []int32{id})
			return
		}
		if f.state == stThinkEndTag && !f.sample {
			whitelist(scores, []int32{f.eos})
			return
		}
		f.transition()
		if _, ok := fixedStrings[f.state]; ok {
			return
		}
		for i := range scores {
			scores[i] = 0 // upstream's scores.zero_() before it recurses
		}
		f.process(scores)
		return
	}
	key := treeKey(f.accIDs)
	switch f.state {
	case stBPMValue, stDurationValue:
		field, tree := "bpm", f.v.bpm
		if f.state == stDurationValue {
			field, tree = "duration", f.v.duration
		}
		if len(f.accIDs) == 0 && f.inject(scores, field) {
			return
		}
		whitelist(scores, tree[key]) // a complete value's set holds the newline
	case stCaptionValue:
		if _, ok := f.user["caption"]; ok && f.accValue == "" && f.inject(scores, "caption") {
			return
		}
		if f.captionAfterNewline {
			top := f.v.decode(argmax(scores))
			if top != "" && top[0] != ' ' && top[0] != '\t' {
				f.captionAfterNewline = false
				f.captionEnding = true
				f.pending = ""
				return
			}
			f.captionAfterNewline = false
		}
		if f.captionEnding {
			return
		}
		scores[f.v.backtick] = negInf
		for i := CodeBase; i < CodeBase+NumCodes; i++ {
			scores[i] = negInf
		}
		if f.captionCount >= captionMaxTokens {
			whitelist(scores, []int32{f.v.newline})
		}
	case stGenresValue:
		if f.v.genres != nil {
			if allowed := f.v.genres.allowed(f.accValue, f.v.newline); len(allowed) > 0 {
				whitelist(scores, allowed)
			} else {
				whitelist(scores, []int32{f.v.newline})
			}
		}
	case stKeyscaleValue, stLanguageValue:
		field, tree := "keyscale", f.v.keyscale
		if f.state == stLanguageValue {
			field, tree = "language", f.v.lang
		}
		if len(f.accIDs) == 0 && f.inject(scores, field) {
			return
		}
		if f.state == stLanguageValue && len(f.accIDs) == 0 {
			best := f.v.langFirst[0]
			for _, id := range f.v.langFirst {
				if scores[id] > scores[best] {
					best = id
				}
			}
			whitelist(scores, []int32{best})
			return
		}
		if tree.has(key, f.v.newline) {
			whitelist(scores, []int32{f.v.newline})
			return
		}
		if allowed := tree[key]; len(allowed) > 0 {
			whitelist(scores, allowed)
			return
		}
		whitelist(scores, []int32{f.v.newline})
	case stTimesigValue:
		if len(f.accIDs) == 0 && f.inject(scores, "timesignature") {
			return
		}
		if f.v.timesig.has(key, f.v.newline) {
			whitelist(scores, []int32{f.v.newline})
			return
		}
		whitelist(scores, f.v.timesig[key])
	}
}

func (f *FSM) transition() {
	n, ok := f.next[f.state]
	if !ok {
		return
	}
	f.state = n
	f.pos = 0
	f.accValue = ""
	f.accIDs = nil
	f.captionAfterNewline = false
	f.captionCount = 0
	f.captionEnding = false
	f.pending = ""
}

// Update advances the state past a sampled token.
func (f *FSM) Update(tok int32) {
	if f.state == stCompleted || f.state == stCodes {
		return
	}
	if len(f.queue) > 0 {
		f.queue = f.queue[1:]
		if len(f.queue) == 0 {
			f.state = f.nextField(f.curField)
			f.curField = ""
			f.pos = 0
			f.accValue = ""
			f.accIDs = nil
		}
		return
	}
	s := f.v.decode(tok)
	if fixed, ok := fixedStrings[f.state]; ok {
		f.pos += utf8.RuneCountInString(s)
		if f.pos >= utf8.RuneCountInString(fixed) {
			if f.state == stThinkEndTag {
				f.state = stCompleted
				f.pos = 0
				f.accValue = ""
				f.accIDs = nil
			} else {
				f.transition()
			}
		}
		return
	}
	switch f.state {
	case stBPMValue, stDurationValue, stTimesigValue:
		if tok == f.v.newline {
			f.transition()
			return
		}
		f.accIDs = append(f.accIDs, tok)
		if t := pyStrip(s); isDigits(t) {
			f.accValue += t
		}
	case stGenresValue:
		if tok == f.v.newline {
			f.transition()
			return
		}
		f.accValue += s
	case stCaptionValue:
		f.captionCount++
		f.accValue += s
		f.captionAfterNewline = strings.Contains(s, "\n")
		if f.captionEnding {
			f.pending += s
			if strings.Contains(s, ":") {
				name := strings.ToLower(pyStrip(strings.TrimRight(pyStrip(f.pending), ":")))
				to, ok := map[string]fsmState{"duration": stDurationValue, "genres": stGenresValue,
					"keyscale": stKeyscaleValue, "language": stLanguageValue, "timesignature": stTimesigValue}[name]
				if ok {
					f.state = to
					f.pos = 0
					f.accValue = ""
					f.accIDs = nil
					f.captionEnding = false
					f.pending = ""
				} else {
					f.captionEnding = false
					f.pending = ""
					f.transition()
				}
			}
		}
	case stKeyscaleValue, stLanguageValue:
		if tok == f.v.newline {
			f.transition()
			return
		}
		f.accIDs = append(f.accIDs, tok)
		f.accValue += s
	}
}

// Done reports that phase 1 has written its EOS.
func (f *FSM) Done() bool { return f.state == stCompleted }

// equalIDs is the index of the first difference between two id lists (the
// shorter length when one is a prefix of the other), or -1.
func equalIDs(a, b []int32) int {
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			return i
		}
	}
	return -1
}
