// Command llm is the qwen3.8-flash-next vertical's driver.
//
// At L1 it does what can be done with the container and the metadata alone:
// it loads the GGUF, builds the tokenizer out of its metadata arrays, and
// tokenizes. Generation arrives at L7; the flags it will need are not
// invented here in advance.
//
//	go run ./cmd/llm -tokenize 'hello world'
//	go run ./cmd/llm -model … -tokenize-file prompt.txt -ids-only
//	go run ./cmd/llm -model … -chat-template     # the checkpoint's own Jinja
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
	"strix-halo-vulkan/llm"
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
	chatTemplate := flag.Bool("chat-template", false, "print the checkpoint's own tokenizer.chat_template and exit")
	hc := flag.Bool("hc", false, "benchmark the fused hyper-connection block")
	ple := flag.Bool("ple", false, "benchmark the PLE n-gram block")
	attn := flag.Bool("attn", false, "benchmark the full-attention layer and the QSA indexer")
	dn := flag.Bool("dn", false, "benchmark the gated DeltaNet layer")
	moe := flag.Bool("moe", false, "benchmark the MoE block")
	head := flag.Bool("head", false, "benchmark the lm head on both dense banks, L8's int8 and the halves it replaces")
	graph := flag.Bool("graph", false, "run the whole model end to end, tokens to logits, and report tok/s")
	gen := flag.Bool("gen", false, "generate: prefill the prompt, then decode one token at a time")
	nPredict := flag.Int("n", 64, "for -gen: how many tokens to generate")
	temp := flag.Float64("temp", 0, "for -gen: sampling temperature; 0 is greedy, which is what a comparison needs")
	topK := flag.Int("top-k", 0, "for -gen: keep the k highest logits, 0 for all")
	topP := flag.Float64("top-p", 0, "for -gen: nucleus mass, 0 or 1 for all")
	seed := flag.Int64("seed", 0, "for -gen: the sampler's seed")
	showTop := flag.Int("top", 0, "for -gen: print the n highest logits beside each token, on stderr")
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
	ppl := flag.Bool("ppl", false, "measure perplexity over a corpus, in llama.cpp's own protocol")
	pplFile := flag.String("ppl-file", "models/wikitext-2-raw/wiki.test.raw", "for -ppl: the corpus")
	chunks := flag.Int("chunks", 0, "for -ppl: stop after this many chunks, 0 for the whole corpus")
	headRows := flag.Int("head-rows", 256, "for -ppl: rows of logits the head's arena holds; a row is 0.99 MB")
	gemmLadder := flag.Bool("gemm-ladder", false, "also cross both GEMM row blocks for -dn and -attn, and at one token the decode GEMV's split per projection")
	attrib := flag.Bool("attrib", false, "for -gen: attribute a decode step per dispatch label and per host phase (P1)")
	mtp := flag.Bool("mtp", false, "P5a: run the MTP draft head as a passive observer beside an ordinary decode, and report its acceptance rate")
	mtpDraft := flag.String("mtp-draft", "models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf", "for -mtp: the draft head's checkpoint")
	mtpN := flag.Int("mtp-n", 256, "for -mtp: how many draft rounds to observe")
	mtpDepth := flag.Int("mtp-depth", 4, "for -mtp: how many tokens deep to chain each draft")
	mtpFree := flag.Bool("mtp-free", false, "for -mtp: advance on the trunk's own argmax instead of the corpus's tokens")
	mtpPrime := flag.Int("mtp-prime", 64, "for -mtp-free: how many corpus tokens to prime with before generating, so the greedy run does not degenerate from one token")
	mtpWire := flag.String("mtp-wire", "res,mixed,zero", "for -mtp: which readings of the nextn block to screen — res, res-flip, mixed, mixed-flip, zero (the negative control)")
	spec := flag.Bool("spec", false, "P5c: generate through the speculative loop and against a same-hour plain one, and report the multiplier")
	specPasses := flag.Int("spec-passes", 2, "for -spec: how many times to run each arm, interleaved")
	specWire := flag.String("spec-wire", "res", "for -spec: the nextn wiring — P5a's measured winner is res")
	specPrefix := flag.Int("spec-prefix", 0, "for -spec: prompt with this many tokens of -ppl-file instead of -prompt, which is the long-context arm")
	csvPath := flag.String("csv", "", "write the -hc table here")
	flag.Parse()

	if *chatTemplate {
		if err := printChatTemplate(*model); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *head {
		toks := []int{1, 8, 64, 512}
		if flagSet("tokens") {
			var err error
			if toks, err = parseInts(*tokens); err != nil {
				log.Fatal(err)
			}
		}
		if err := headBench(*model, toks, *iters, *csvPath); err != nil {
			log.Fatal(err)
		}
		return
	}

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

	if *mtp {
		layers := 0
		if flagSet("layers") {
			layers = *attnLayers
		}
		if err := mtpObserve(mtpOpts{
			model: *model, draft: *mtpDraft, file: *pplFile, n: *mtpN, ctx: *ctx,
			layers: layers, depth: *mtpDepth, free: *mtpFree, prime: *mtpPrime, wires: *mtpWire, csv: *csvPath,
		}); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *spec {
		layers := 0
		if flagSet("layers") {
			layers = *attnLayers
		}
		p := specPromptDefault
		if flagSet("prompt") {
			p = *prompt
		}
		if err := speculate(specOpts{
			model: *model, draft: *mtpDraft, prompt: p, file: *pplFile, prefix: *specPrefix,
			n: *nPredict, ctx: *ctx,
			layers: layers, passes: *specPasses, wire: specWireList(*specWire), chat: *chat, csv: *csvPath,
		}); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *ppl {
		layers := 0
		if flagSet("layers") {
			layers = *attnLayers
		}
		if err := perplexity(pplOpts{
			model: *model, file: *pplFile, ctx: *ctx, chunks: *chunks,
			headRows: *headRows, layers: layers, csv: *csvPath,
		}); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *gen {
		layers := 0
		if flagSet("layers") {
			layers = *attnLayers
		}
		if err := generate(genOpts{
			model: *model, prompt: *prompt, n: *nPredict, ctx: *ctx, layers: layers,
			temp: *temp, topK: *topK, topP: *topP, seed: *seed, chat: *chat, top: *showTop,
			csv: *csvPath, attrib: *attrib,
		}); err != nil {
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
		if err := attnBench(*model, toks, *ctx, *attnLayers, *iters, *ladder, *gemmLadder, *sel, *csvPath); err != nil {
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

// printChatTemplate writes the checkpoint's own chat template to stdout.
//
// It is here because `llm.RenderChat` is a transcription of that template and
// the oracle it is checked against is Jinja rendering the original
// (reference/dump_chat_template.py) -- which needs the original, and the only
// copy of it is 180 lines of metadata inside a 107 GB GGUF:
//
//	go run ./cmd/llm -model … -chat-template > reference/out/chat/template.jinja
func printChatTemplate(model string) error {
	set, err := gguf.OpenSet(model)
	if err != nil {
		return err
	}
	defer set.Close()
	tmpl, ok := set.Str("tokenizer.chat_template")
	if !ok {
		return fmt.Errorf("%s carries no tokenizer.chat_template", model)
	}
	_, err = os.Stdout.WriteString(tmpl)
	return err
}

func run(model, text string, idsOnly bool) error {
	set, err := gguf.OpenSet(model)
	if err != nil {
		return err
	}
	defer set.Close()

	tok, err := llm.LoadTokenizer(set)
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
