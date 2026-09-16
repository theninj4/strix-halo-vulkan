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
// At L2 it also drives the first kernel of the vertical, the fused
// hyper-connection block, against the per-op attribution of llama.cpp's own
// prefill graph that L2a produced (hc.go):
//
//	go run ./cmd/llm -hc
//	go run ./cmd/llm -hc -tokens 512 -ladder -csv results/l2c_hc.csv
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

// graphDefaultPrompt is what -graph prefills when the caller names nothing.
// It is English prose rather than a repeated token because the MoE routing
// and the QSA selection are both properties of the text (L5a-4, L4b-6).
const graphDefaultPrompt = "The capital of France is Paris. " +
	"Hyper-connections replace the residual stream with four of them, and every " +
	"block reads a learned mix and writes back with learned per-stream weights. " +
	"Three quarters of the layers are a gated DeltaNet; the other quarter is " +
	"sparse attention with a query-aware indexer. Ninety-seven per cent of the " +
	"parameters are in five hundred and twelve experts, ten of which run per token."

func main() {
	log.SetFlags(0)
	model := flag.String("model", defaultModel, "GGUF checkpoint, shard or directory")
	text := flag.String("tokenize", "", "tokenize this text and print the ids")
	file := flag.String("tokenize-file", "", "tokenize the contents of this file")
	idsOnly := flag.Bool("ids-only", false, "print ids alone, one line, space separated")
	chat := flag.Bool("chat", false, "wrap the text in the chat template first")
	hc := flag.Bool("hc", false, "benchmark the fused hyper-connection block")
	ple := flag.Bool("ple", false, "benchmark the PLE n-gram block")
	attn := flag.Bool("attn", false, "benchmark the full-attention layer and the QSA indexer")
	dn := flag.Bool("dn", false, "benchmark the gated DeltaNet layer")
	moe := flag.Bool("moe", false, "benchmark the MoE block")
	graph := flag.Bool("graph", false, "run the whole model end to end, tokens to logits, and report tok/s")
	prompt := flag.String("prompt", graphDefaultPrompt, "the text -graph prefills; repeated to reach -tokens")
	resident := flag.Bool("resident", false, "stage every layer of every block on the device and report the plan")
	dense := flag.Bool("dense", false, "for -resident: leave the 77 GB expert bank out and stage the dense half alone")
	bank := flag.String("bank", "", "for -resident: stage these many expert banks instead of all 48 — a list, one stage each, to price residency against itself")
	ctx := flag.Int("ctx", 2048, "cache cells for -attn; llama.cpp's measured graph had 2048")
	attnLayers := flag.Int("layers", 2, "how many layers to stage for -attn and -dn; for -moe the default is one, because a layer's expert bank is 1.57 GB")
	sel := flag.String("sel", "auto", "the QSA selection for -attn: auto (only where it bites), on (price it where it is the identity), off (the dense control)")
	tokens := flag.String("tokens", "64,128,256,512,1024,2048", "token counts for -hc")
	mixers := flag.Int("mixers", 8, "how many real mixers to stage for -hc; the sweep needs more than the 32 MiB MALL")
	iters := flag.Int("iters", 20, "repetitions per timed dispatch for -hc")
	ladder := flag.Bool("ladder", false, "run every kernel rung for -hc")
	gemmLadder := flag.Bool("gemm-ladder", false, "also cross both GEMM row blocks for -dn")
	csvPath := flag.String("csv", "", "write the -hc table here")
	flag.Parse()

	if *ple {
		toks, err := parseInts(*tokens)
		if err != nil {
			log.Fatal(err)
		}
		if err := pleBench(*model, toks, *iters, *csvPath); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *graph {
		toks, err := parseInts(*tokens)
		if err != nil {
			log.Fatal(err)
		}
		layers := 0
		if flagSet("layers") {
			layers = *attnLayers
		}
		if err := graphBench(*model, *prompt, toks, layers, *ctx, *csvPath); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *resident {
		toks, err := parseInts(*tokens)
		if err != nil {
			log.Fatal(err)
		}
		// One arena size, not a sweep: residency is about the weights, so
		// the arenas are sized for llama.cpp's own best ubatch unless the
		// caller named a length of their own.
		maxTok := 512
		if flagSet("tokens") {
			for _, t := range toks {
				maxTok = max(maxTok, t)
			}
		}
		var banks []int
		if *bank != "" {
			if banks, err = parseInts(*bank); err != nil {
				log.Fatal(err)
			}
		}
		if err := residency(*model, maxTok, max(*ctx, maxTok), banks, *dense, *csvPath); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *moe {
		toks, err := parseInts(*tokens)
		if err != nil {
			log.Fatal(err)
		}
		if err := moeBench(*model, toks, *attnLayers, *iters, *ladder, *csvPath); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *dn {
		toks, err := parseInts(*tokens)
		if err != nil {
			log.Fatal(err)
		}
		if err := dnBench(*model, toks, *attnLayers, *iters, *ladder, *gemmLadder, *csvPath); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *attn {
		toks, err := parseInts(*tokens)
		if err != nil {
			log.Fatal(err)
		}
		if err := attnBench(*model, toks, *ctx, *attnLayers, *iters, *ladder, *sel, *csvPath); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *hc {
		toks, err := parseInts(*tokens)
		if err != nil {
			log.Fatal(err)
		}
		if err := hcBench(*model, toks, *mixers, *iters, *ladder, *csvPath); err != nil {
			log.Fatal(err)
		}
		return
	}

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

// flagSet reports whether the caller named a flag on the command line, as
// against taking its default. -resident needs it: every other mode sweeps
// -tokens and it defaults to a list, where residency wants one length.
func flagSet(name string) bool {
	found := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			found = true
		}
	})
	return found
}
