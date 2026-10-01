// Command vocabfreq ranks the checkpoint's vocabulary by how often each token
// occurs in a corpus, for the trimmed draft head (research/p20-llamacpp-mtp.md
// §9): the draft proposes only from the first K ids of the ranking, so the
// ranking decides what the draft can ever say.
//
//	go run ./cmd/vocabfreq -train a.txt,b.txt -test c.txt -out rank.bin
//
// It prints, for each test file, the share of its tokens the top K covers
// under two rankings — counted on the training files, and the vocabulary's
// own id order (BPE merge order, a frequency ranking on the tokenizer's
// corpus) and the two merged — and writes the merged ranking as
// little-endian int32 ids, the whole vocabulary, first id first.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"strix-halo-vulkan/llm"
)

func main() {
	model := flag.String("model", "models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", "the trunk's first shard")
	train := flag.String("train", "models/wikitext-2-raw/wiki.train.raw", "comma-separated files to count")
	test := flag.String("test", "models/wikitext-2-raw/wiki.test.raw", "comma-separated files to report coverage on")
	out := flag.String("out", "", "write the counted ranking here")
	flag.Parse()

	m, err := llm.Open(*model)
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	tok, err := llm.LoadTokenizer(m.Set)
	if err != nil {
		log.Fatal(err)
	}
	const vocab = 248320 // output.weight's rows; Encode never names an id past it
	enc := func(path string) []int32 {
		buf, err := os.ReadFile(path)
		if err != nil {
			log.Fatal(err)
		}
		ids, err := tok.Encode(string(buf))
		if err != nil {
			log.Fatal(err)
		}
		return ids
	}

	count := make([]int, vocab)
	n := 0
	for _, f := range strings.Split(*train, ",") {
		ids := enc(f)
		for _, id := range ids {
			count[id]++
		}
		n += len(ids)
		fmt.Printf("counted %s: %d tokens\n", f, len(ids))
	}
	rank := make([]int32, vocab)
	for i := range rank {
		rank[i] = int32(i)
	}
	sort.SliceStable(rank, func(a, b int) bool { return count[rank[a]] > count[rank[b]] })
	seen := 0
	for _, c := range count {
		if c > 0 {
			seen++
		}
	}
	fmt.Printf("%d tokens counted, %d distinct of %d\n\n", n, seen, vocab)

	// The merge: the counted ranking and id order alternately, each id at
	// its first appearance, so the top K holds the top ~K/2 of each — the
	// corpus's words and the tokenizer's own broad frequency both.
	merged := make([]int32, 0, vocab)
	in := make([]bool, vocab)
	for i := 0; len(merged) < vocab; i++ {
		for _, id := range []int32{rank[i], int32(i)} {
			if !in[id] {
				in[id] = true
				merged = append(merged, id)
			}
		}
	}

	ks := []int{4096, 8192, 16384, 32768, 65536}
	for _, f := range strings.Split(*test, ",") {
		ids := enc(f)
		pos, mpos := make([]int, vocab), make([]int, vocab)
		for i, id := range rank {
			pos[id] = i
		}
		for i, id := range merged {
			mpos[id] = i
		}
		fmt.Printf("%s, %d tokens: share inside the top K\n", f, len(ids))
		fmt.Printf("%8s %10s %10s %10s\n", "K", "counted", "id order", "merged")
		for _, k := range ks {
			c, o, g := 0, 0, 0
			for _, id := range ids {
				if pos[id] < k {
					c++
				}
				if int(id) < k {
					o++
				}
				if mpos[id] < k {
					g++
				}
			}
			pc := func(x int) float64 { return 100 * float64(x) / float64(len(ids)) }
			fmt.Printf("%8d %9.2f%% %9.2f%% %9.2f%%\n", k, pc(c), pc(o), pc(g))
		}
		fmt.Println()
	}
	if *out != "" {
		f, err := os.Create(*out)
		if err != nil {
			log.Fatal(err)
		}
		if err := binary.Write(f, binary.LittleEndian, merged); err != nil {
			log.Fatal(err)
		}
		if err := f.Close(); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("wrote %s\n", *out)
	}
}
