package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"strix-halo-vulkan/gguf"
)

// checkpoint is the local UD-Q4_K_XL checkout. The test is skipped when it is
// absent, so the package still tests on a machine without 111 GB of weights.
const checkpoint = "../../models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"

// TestTokenizerMatchesLlamaCpp checks the GGUF-backed tokenizer against
// llama.cpp's, which is the acceptance criterion L1 asks for:
//
//	llama-tokenize -m …-00001-of-00004.gguf -f testdata/tokenizer_corpus.txt --ids
//
// The corpus is chosen for the places a byte-level BPE can differ rather than
// for length: contractions, digits one at a time, a tab and a blank line,
// **decomposed combining marks** (the one class where the `qwen35`
// pre-tokenizer differs from `qwen2` — see split()), CJK with no spaces, a
// ZWJ emoji sequence, and the chat and multimodal special tokens, which have
// to be matched whole rather than merged.
func TestTokenizerMatchesLlamaCpp(t *testing.T) {
	if _, err := os.Stat(checkpoint); err != nil {
		t.Skipf("no checkpoint at %s", checkpoint)
	}
	text, err := os.ReadFile(filepath.Join("testdata", "tokenizer_corpus.txt"))
	if err != nil {
		t.Fatal(err)
	}
	refText, err := os.ReadFile(filepath.Join("testdata", "tokenizer_ids.txt"))
	if err != nil {
		t.Fatal(err)
	}
	var want []int32
	for _, f := range strings.Fields(string(refText)) {
		id, err := strconv.Atoi(f)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, int32(id))
	}

	set, err := gguf.OpenSet(checkpoint)
	if err != nil {
		t.Skipf("opening %s: %v", checkpoint, err)
	}
	defer set.Close()

	tok, err := LoadTokenizer(set)
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Size(); got != 248320 {
		t.Errorf("vocabulary is %d tokens, the checkpoint says 248320", got)
	}
	got, err := tok.Encode(string(text))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d tokens, llama.cpp gets %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			// Decoding the neighbourhood says *what* diverged, which is
			// always more useful than the ids.
			lo := max(0, i-3)
			gs, _ := tok.Decode(got[lo : i+1])
			ws, _ := tok.Decode(want[lo : i+1])
			t.Fatalf("token %d: got %d (%q), llama.cpp gets %d (%q)", i, got[i], gs, want[i], ws)
		}
	}

	// Round trip: the corpus decodes back to itself byte for byte.
	back, err := tok.Decode(got)
	if err != nil {
		t.Fatal(err)
	}
	if back != string(text) {
		t.Errorf("decode round trip differs from the corpus")
	}
}
