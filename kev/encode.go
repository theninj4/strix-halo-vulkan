package kev

// The packed encoding: Kev's `model.encode` (kev/model.py at 9fdf054).
//
//	[<state>] state...  then per question k:  [<q>] instr... ([<opt>] option... [</opt>])xK [<decide>]
//
// seg is 0 on the state and k on question k's branch; pos runs 0.. over the
// state and every branch *restarts* at the state's length, so each branch is
// positioned as if it were the only one. The readouts are `<decide>` (the
// last token of each branch) and each option's `</opt>`.
//
// The delimiters are existing Qwen tokens, reused so that no embedding row
// had to be trained, and their roles are not the ones their names suggest:
// the state marker is fim_prefix, a question fim_middle and the decision
// fim_suffix. Their ids are read from the tokenizer, never written down.

import (
	"fmt"
	"path/filepath"
	"regexp"

	"strix-halo-vulkan/zimage/tokenizer"
)

// Delimiters, as Kev's SPECIAL lists them: state, q, opt, /opt, decide.
var delimiterTokens = [5]string{"<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"}

// Serving limits (kev/model.py SERVE_MAX_STATE, SERVE_MAX_BRANCH): the state
// is truncated to MaxState tokens including its marker, and a row (state
// plus one branch) longer than MaxRow is refused. Training only saw states of
// up to 384 tokens.
const (
	MaxState = 8192
	MaxRow   = 8192
)

// Option markers in Encoding.Opt: OptNone for state and instruction tokens,
// OptDecide for <decide>, 0..K-1 for the tokens of option k's span.
const (
	OptNone   = -1
	OptDecide = -2
)

// Encoder tokenises records. It holds the base tokenizer and the delimiter ids.
type Encoder struct {
	Tok *tokenizer.Tokenizer
	// State, Q, Open, Close and Decide are the delimiter ids.
	State, Q, Open, Close, Decide int32
}

// LoadEncoder reads tokenizer.json from dir. Kev's checkpoint directory
// carries a copy of the base's tokenizer (identical vocabulary and merges,
// with the merges stored as pairs, which is the form zimage/tokenizer reads;
// the base snapshot stores them as "a b" strings).
func LoadEncoder(dir string) (*Encoder, error) {
	tok, err := tokenizer.Load(filepath.Clean(dir))
	if err != nil {
		return nil, err
	}
	e := &Encoder{Tok: tok}
	dst := []*int32{&e.State, &e.Q, &e.Open, &e.Close, &e.Decide}
	for i, name := range delimiterTokens {
		id, ok := tok.ID(name)
		if !ok {
			return nil, fmt.Errorf("kev: tokenizer has no %s", name)
		}
		*dst[i] = id
	}
	return e, nil
}

var delimiterLike = regexp.MustCompile(`<\|([A-Za-z0-9_]+)\|>`)

// UserTokens tokenises caller-supplied text so that it can never produce a
// delimiter or control token: every `<|name|>` becomes `<¦name¦>` first
// (U+00A6), then the text is tokenised without special tokens. Added tokens
// of another form, like `<tool_call>`, are still matched, as the reference
// matches them.
//
// The reference also NFC-normalises; zimage/tokenizer does not (its
// TestNFCIsTheKnownGap). Text that is already NFC, which is nearly all text,
// tokenises identically.
func (e *Encoder) UserTokens(text string) ([]int32, error) {
	return e.Tok.Encode(delimiterLike.ReplaceAllString(text, "<¦$1¦>"))
}

// Encoding is one packed record.
type Encoding struct {
	IDs, Seg, Pos, Opt []int32
	// Decide[k] is the index of question k's <decide>, and Opts[k][j] the
	// index of its option j's </opt>, both into IDs.
	Decide []int
	Opts   [][]int
	// StateLen is the state's token count including its marker.
	StateLen       int
	StateTruncated bool
}

// OverflowError is a branch that does not fit its row: a 422.
type OverflowError struct{ Msg string }

func (e *OverflowError) Error() string { return e.Msg }

// Encode packs a record. The state is truncated to maxState tokens
// (including its marker); a branch longer than maxRow minus the state is
// an OverflowError.
func (e *Encoder) Encode(rec Record, maxState, maxRow int) (*Encoding, error) {
	st, err := e.UserTokens(rec.State)
	if err != nil {
		return nil, err
	}
	enc := &Encoding{StateTruncated: len(st)+1 > maxState}
	if enc.StateTruncated {
		st = st[:maxState-1]
	}
	S := append([]int32{e.State}, st...)
	enc.StateLen = len(S)
	for i, id := range S {
		enc.push(id, 0, int32(i), OptNone)
	}
	for k, q := range rec.Questions {
		instr, err := e.UserTokens(q.Instr)
		if err != nil {
			return nil, err
		}
		br := append([]int32{e.Q}, instr...)
		opt := make([]int32, len(br))
		for i := range opt {
			opt[i] = OptNone
		}
		var ends []int
		for j, o := range q.Options {
			ot, err := e.UserTokens(o)
			if err != nil {
				return nil, err
			}
			br = append(br, e.Open)
			br = append(br, ot...)
			br = append(br, e.Close)
			for range len(ot) + 2 {
				opt = append(opt, int32(j))
			}
			ends = append(ends, len(br)-1)
		}
		br = append(br, e.Decide)
		opt = append(opt, OptDecide)
		if len(br) > maxRow-len(S) {
			return nil, &OverflowError{Msg: fmt.Sprintf("branch too long: %d tokens with a %d-token state (row limit %d)", len(br), len(S), maxRow)}
		}
		base := len(enc.IDs)
		for i, id := range br {
			enc.push(id, int32(k+1), int32(len(S)+i), opt[i])
		}
		enc.Decide = append(enc.Decide, base+len(br)-1)
		oi := make([]int, len(ends))
		for j, end := range ends {
			oi[j] = base + end
		}
		enc.Opts = append(enc.Opts, oi)
	}
	return enc, nil
}

func (enc *Encoding) push(id, seg, pos, opt int32) {
	enc.IDs = append(enc.IDs, id)
	enc.Seg = append(enc.Seg, seg)
	enc.Pos = append(enc.Pos, pos)
	enc.Opt = append(enc.Opt, opt)
}

// Questions is the number of branches.
func (enc *Encoding) Questions() int { return len(enc.Decide) }

// Branch returns question k's [start, end) in IDs.
func (enc *Encoding) Branch(k int) (int, int) {
	start := enc.StateLen
	if k > 0 {
		start = enc.Decide[k-1] + 1
	}
	return start, enc.Decide[k] + 1
}
