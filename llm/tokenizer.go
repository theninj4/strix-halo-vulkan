package llm

// The tokenizer and the stop set, which are the checkpoint's and not a
// command's.
//
// Both of these used to live in cmd/llm, which was right while the only
// caller was a benchmark. They moved here when `backend/llm.go` became a
// second caller: what token ends a generation is a fact about this
// checkpoint, and two copies of that fact is exactly the kind of thing that
// is fixed in one of them.

import (
	"fmt"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/zimage/tokenizer"
)

// LoadTokenizer builds the tokenizer out of a checkpoint's metadata. Only
// shard 1 carries it, and it is 12 MB of strings, so this is the one part of
// a 111 GB checkpoint that is copied onto the Go heap.
func LoadTokenizer(set *gguf.Set) (*tokenizer.Tokenizer, error) {
	tokens, ok := set.Strings("tokenizer.ggml.tokens")
	if !ok {
		return nil, fmt.Errorf("llm: the checkpoint has no tokenizer.ggml.tokens")
	}
	merges, ok := set.Strings("tokenizer.ggml.merges")
	if !ok {
		return nil, fmt.Errorf("llm: the checkpoint has no tokenizer.ggml.merges")
	}
	v := tokenizer.Vocab{Tokens: tokens, Merges: merges}
	v.Pre, _ = set.Str("tokenizer.ggml.pre")
	if raw, ok := set.Ints("tokenizer.ggml.token_type"); ok {
		v.Types = make([]tokenizer.TokenType, len(raw))
		for i, t := range raw {
			v.Types[i] = tokenizer.TokenType(t)
		}
	}
	return tokenizer.FromVocab(v)
}

// EndOfGeneration is the set of tokens that stop the loop.
//
// It is **not** `tokenizer.ggml.eos_token_id` alone, and the difference is
// visible on the first prompt anyone tries: this checkpoint's EOS is 248046
// and what the model actually emits to finish a completion is 248044,
// `<|endoftext|>`. llama.cpp marks a control token end-of-generation by
// *name* as well as by the metadata key — `llama_vocab` flags `<|endoftext|>`,
// `<|im_end|>` and `<|eot_id|>` wherever the vocabulary has them — so a loop
// that trusts the key alone runs past the end of the text and then repeats
// one token until -n is exhausted.
func EndOfGeneration(set *gguf.Set, tok *tokenizer.Tokenizer) map[int32]bool {
	eog := map[int32]bool{}
	if v, ok := set.Uint("tokenizer.ggml.eos_token_id"); ok {
		eog[int32(v)] = true
	}
	for _, k := range []string{"eot_token_id", "eom_token_id"} {
		if v, ok := set.Uint("tokenizer.ggml." + k); ok {
			eog[int32(v)] = true
		}
	}
	for _, name := range []string{"<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end|>"} {
		if id, ok := tok.ID(name); ok {
			eog[id] = true
		}
	}
	return eog
}
