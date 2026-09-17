package embed

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"strix-halo-vulkan/zimage/tokenizer"
)

// Tokenizer is the checkpoint's Qwen2 byte-level BPE with its post-processor.
//
// The BPE itself is `zimage/tokenizer` unchanged -- same vocabulary format,
// same pre-tokenizer regex, `ignore_merges` and `byte_fallback` both false --
// and what is here is the one thing that package does not do, because
// Z-Image's tokenizer.json does not ask for it: a `TemplateProcessing`
// post-processor that appends `<|endoftext|>` to every sequence.
//
// That append is not cosmetic. Pooling takes the *last* row, so the vector
// this model returns is the hidden state of the appended token; drop it and
// every embedding silently becomes the last real token's instead, which is a
// plausible-looking vector that scores wrong. The id is read out of
// tokenizer.json rather than written down here, and it is **not**
// `eos_token_id`: the config's eos is 151645 (`<|im_end|>`) and what the
// post-processor appends is 151643 (`<|endoftext|>`).
type Tokenizer struct {
	*tokenizer.Tokenizer
	// Append is the special token the post-processor adds, by id.
	Append int32
	// AppendToken is its text, kept for error messages and tests.
	AppendToken string
}

// postProcessor is the part of tokenizer.json this reads: the sequence of
// processors, and the single-sequence template of whichever one is a
// TemplateProcessing.
type postProcessor struct {
	PostProcessor struct {
		Type       string `json:"type"`
		Processors []struct {
			Type   string `json:"type"`
			Single []struct {
				SpecialToken *struct {
					ID string `json:"id"`
				} `json:"SpecialToken"`
				Sequence *struct {
					ID string `json:"id"`
				} `json:"Sequence"`
			} `json:"single"`
		} `json:"processors"`
	} `json:"post_processor"`
}

// LoadTokenizer reads a checkpoint directory's tokenizer.json.
func LoadTokenizer(dir string) (*Tokenizer, error) {
	base, err := tokenizer.Load(dir)
	if err != nil {
		return nil, err
	}
	buf, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	var raw postProcessor
	if err := json.Unmarshal(buf, &raw); err != nil {
		return nil, fmt.Errorf("embed: parsing tokenizer.json: %w", err)
	}
	t := &Tokenizer{Tokenizer: base}
	for _, p := range raw.PostProcessor.Processors {
		if p.Type != "TemplateProcessing" {
			continue
		}
		// The template is [Sequence A, SpecialToken X]: one sequence
		// followed by one special token. Anything else -- a prefix token, a
		// second suffix -- would change what a sequence is, so it is
		// rejected rather than partly honoured.
		if len(p.Single) != 2 || p.Single[0].Sequence == nil || p.Single[1].SpecialToken == nil {
			return nil, fmt.Errorf("embed: tokenizer.json has a post-processing template this does not implement")
		}
		t.AppendToken = p.Single[1].SpecialToken.ID
		id, ok := base.ID(t.AppendToken)
		if !ok {
			return nil, fmt.Errorf("embed: the post-processor appends %q, which is not in the vocabulary", t.AppendToken)
		}
		t.Append = id
	}
	if t.AppendToken == "" {
		return nil, fmt.Errorf("embed: tokenizer.json has no TemplateProcessing; this checkpoint should append a token")
	}
	return t, nil
}

// Encode turns text into the ids the model is run over, post-processor
// included.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	ids, err := t.Tokenizer.Encode(text)
	if err != nil {
		return nil, err
	}
	return append(ids, t.Append), nil
}

// EncodeLimit is Encode truncated to at most limit ids, the appended token
// included. limit <= 0 does not truncate.
//
// The front of the text is what survives, which is HF's `truncation=True`,
// and the appended token is re-added afterwards rather than being cut off
// with the tail. That second half is not cosmetic: pooling takes the last
// row, so an input truncated *through* its end-of-text token would have its
// vector taken from an ordinary word while every other input's came from the
// appended one, and the two are not comparable.
func (t *Tokenizer) EncodeLimit(text string, limit int) ([]int32, error) {
	ids, err := t.Tokenizer.Encode(text)
	if err != nil {
		return nil, err
	}
	if limit > 0 && len(ids) > limit-1 {
		ids = ids[:limit-1]
	}
	return append(ids, t.Append), nil
}

// EncodeBare is Encode without the appended token -- HF's
// `add_special_tokens=False`. It exists for the test that pins which of the
// two the model is run on, and for nothing else.
func (t *Tokenizer) EncodeBare(text string) ([]int32, error) {
	return t.Tokenizer.Encode(text)
}
