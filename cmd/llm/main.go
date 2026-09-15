// Command llm is the qwen3.8-flash-next vertical's driver.
//
// At L1 it does what can be done with the container and the metadata alone:
// it loads the GGUF, builds the tokenizer out of its metadata arrays, and
// tokenizes. Generation arrives at L7; the flags it will need are not
// invented here in advance.
//
//	go run ./cmd/llm -tokenize 'hello world'
//	go run ./cmd/llm -model … -tokenize-file prompt.txt -ids-only
//
// The acceptance criterion for the tokenizer is llama.cpp's own, so the
// output format is llama-tokenize's:
//
//	$L/build/bin/llama-tokenize -m …-00001-of-00004.gguf -p 'hello world'
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/zimage/tokenizer"
)

const defaultModel = "models/Qwen3.8-Flash-Next-GGUF"

func main() {
	log.SetFlags(0)
	model := flag.String("model", defaultModel, "GGUF checkpoint, shard or directory")
	text := flag.String("tokenize", "", "tokenize this text and print the ids")
	file := flag.String("tokenize-file", "", "tokenize the contents of this file")
	idsOnly := flag.Bool("ids-only", false, "print ids alone, one line, space separated")
	chat := flag.Bool("chat", false, "wrap the text in the chat template first")
	flag.Parse()

	if *text == "" && *file == "" {
		fmt.Fprintf(os.Stderr, "usage: %s [-model dir] -tokenize <text> | -tokenize-file <path>\n", os.Args[0])
		os.Exit(2)
	}
	in := *text
	if *file != "" {
		buf, err := os.ReadFile(*file)
		if err != nil {
			log.Fatal(err)
		}
		in = string(buf)
	}
	if *chat {
		in = tokenizer.ChatPrompt(in)
	}
	if err := run(*model, in, *idsOnly); err != nil {
		log.Fatal(err)
	}
}

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

func run(model, text string, idsOnly bool) error {
	set, err := gguf.OpenSet(model)
	if err != nil {
		return err
	}
	defer set.Close()

	tok, err := LoadTokenizer(set)
	if err != nil {
		return err
	}
	ids, err := tok.Encode(text)
	if err != nil {
		return err
	}

	if idsOnly {
		var b strings.Builder
		for i, id := range ids {
			if i > 0 {
				b.WriteByte(' ')
			}
			fmt.Fprintf(&b, "%d", id)
		}
		fmt.Println(b.String())
		return nil
	}

	fmt.Printf("%s: %s, vocab %d, pre %q\n", model, set.Arch(), tok.Size(), mustStr(set, "tokenizer.ggml.pre"))
	fmt.Printf("%d tokens\n", len(ids))
	// llama-tokenize's layout, so the two can be diffed line for line.
	for _, id := range ids {
		s, err := tok.Decode([]int32{id})
		if err != nil {
			return err
		}
		fmt.Printf("%6d -> '%s'\n", id, s)
	}
	return nil
}

func mustStr(set *gguf.Set, key string) string { v, _ := set.Str(key); return v }
