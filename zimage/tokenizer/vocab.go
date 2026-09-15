package tokenizer

import (
	"fmt"
	"sort"
	"strings"
)

// A GGUF carries its tokenizer in the metadata rather than beside the
// checkpoint: `tokenizer.ggml.tokens` is the vocabulary indexed by id,
// `tokenizer.ggml.merges` the merge table as space-separated pairs,
// `tokenizer.ggml.token_type` one ggml token type per id, and
// `tokenizer.ggml.pre` the name of a pre-tokenizer regex that lives in
// llama.cpp's source rather than in the file.
//
// So there is no tokenizer.json to read, and Load's path does not apply.
// FromVocab is the same machinery reached from those four arrays instead,
// which is what qwen3.8-flash-next needs: a 248 320-entry vocabulary whose
// specials include the multimodal ones, and the `qwen35` pre-tokenizer.

// TokenType is a ggml token type — the `tokenizer.ggml.token_type` array.
// The numbering is llama.cpp's llama_token_attr.
type TokenType int

const (
	TokenUndefined   TokenType = 0
	TokenNormal      TokenType = 1
	TokenUnknown     TokenType = 2
	TokenControl     TokenType = 3
	TokenUserDefined TokenType = 4
	TokenUnused      TokenType = 5
	TokenByte        TokenType = 6
)

// Vocab is a tokenizer as a GGUF states it.
type Vocab struct {
	// Tokens is the vocabulary indexed by id, over the byte-level alphabet.
	Tokens []string
	// Merges are "left right" pairs in priority order, lowest index first.
	Merges []string
	// Types is one TokenType per id. It may be nil, in which case no token
	// is special and only Specials applies.
	Types []TokenType
	// Specials names tokens to match as whole tokens in the input, for a
	// checkpoint that marks none of them. Normally empty: Types says it.
	Specials []string
	// Pre is the pre-tokenizer name from `tokenizer.ggml.pre`. Only the two
	// Qwen ones are implemented, because they are the only two this project
	// has a model for.
	Pre string
}

// FromVocab builds a tokenizer from a GGUF's metadata arrays.
//
// Control and user-defined tokens become the added tokens Encode matches
// whole — which for this checkpoint is the chat markup and the multimodal
// specials (`<|vision_start|>`, `<|image_pad|>` and the rest). Every other
// token, byte-level ones included, is ordinary BPE input.
func FromVocab(v Vocab) (*Tokenizer, error) {
	if len(v.Tokens) == 0 {
		return nil, fmt.Errorf("tokenizer: the vocabulary is empty")
	}
	if v.Types != nil && len(v.Types) != len(v.Tokens) {
		return nil, fmt.Errorf("tokenizer: %d token types for %d tokens", len(v.Types), len(v.Tokens))
	}
	// The pre-tokenizer is named, not written, in the file. An unknown name
	// has to be an error rather than a default: the two differ by one
	// character class, so the wrong one is silently wrong on exactly the
	// inputs nobody tests with.
	var marks bool
	switch v.Pre {
	case "qwen2", "":
		marks = false
	case "qwen35":
		marks = true
	default:
		return nil, fmt.Errorf("tokenizer: pre-tokenizer %q is not implemented (qwen2, qwen35)", v.Pre)
	}

	t := &Tokenizer{
		vocab:      make(map[string]int32, len(v.Tokens)),
		tokens:     make([]string, len(v.Tokens)),
		ranks:      make(map[string]int, len(v.Merges)),
		specialID:  make(map[string]int32),
		runeToByte: make(map[rune]byte, 256),
		marks:      marks,
	}
	copy(t.tokens, v.Tokens)
	for id, tok := range v.Tokens {
		// A duplicate would be a defect in the checkpoint; the first id wins,
		// which is what llama.cpp's token_to_id does.
		if _, dup := t.vocab[tok]; !dup {
			t.vocab[tok] = int32(id)
		}
	}
	for i, m := range v.Merges {
		// "Ġ Ġ" — one space, and neither half can hold one, because the
		// byte-level alphabet maps 0x20 to Ġ before BPE ever sees it.
		sp := strings.IndexByte(m, ' ')
		if sp <= 0 || sp == len(m)-1 || strings.IndexByte(m[sp+1:], ' ') >= 0 {
			return nil, fmt.Errorf("tokenizer: merge %d is %q, want two space-separated halves", i, m)
		}
		t.ranks[m[:sp]+"\x00"+m[sp+1:]] = i
	}

	special := map[string]bool{}
	for _, s := range v.Specials {
		special[s] = true
	}
	for id, tok := range v.Tokens {
		if v.Types != nil {
			switch v.Types[id] {
			case TokenControl, TokenUserDefined:
				special[tok] = true
			}
		}
		if special[tok] {
			t.specialID[tok] = int32(id)
		}
	}
	for s := range t.specialID {
		t.specials = append(t.specials, s)
	}
	// Longest first, so "<|im_start|>" is not shadowed by a shorter added
	// token that happens to be a prefix of it. Ties break by name, so the
	// order does not depend on Go's map iteration.
	sort.Slice(t.specials, func(i, j int) bool {
		if len(t.specials[i]) != len(t.specials[j]) {
			return len(t.specials[i]) > len(t.specials[j])
		}
		return t.specials[i] < t.specials[j]
	})

	t.buildByteAlphabet()
	return t, nil
}
