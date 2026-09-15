// Command gguf reports what is inside a GGUF checkpoint and what it would
// cost to run on this machine.
//
// It is the Go counterpart of reference/gguf_inventory.py, which produced
// LLM.md's inventory of `unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL` from
// 35 MB of HTTP range requests before anything was downloaded. The Python
// stays as the range-request tool — it tolerates a truncated tensor table,
// which is the whole trick — and this one is the check on the Go reader:
// same grouping, same block arithmetic, same decode budget, so the two
// disagreeing means one of them is wrong about the file.
//
//	go run ./cmd/gguf models/Qwen3.8-Flash-Next-GGUF
//	go run ./cmd/gguf -tensors -kv …-00001-of-00004.gguf
//	go run ./cmd/gguf -check tensors.json …          # against the Python's dump
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"text/tabwriter"

	"strix-halo-vulkan/gguf"
)

// What this machine can do, measured: IDEAS §1.7 for the bus. The same two
// constants the Python carries, so the budget lines can be compared.
const busGBs = 242.0

func main() {
	log.SetFlags(0)
	tensors := flag.Bool("tensors", false, "list every tensor instead of collapsing repeated layers")
	kv := flag.Bool("kv", false, "print the metadata table")
	check := flag.String("check", "", "compare the tensor table against reference/gguf_inventory.py's tensors.json")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "usage: %s [-tensors] [-kv] [-check tensors.json] <checkpoint|shard|dir>\n", os.Args[0])
		os.Exit(2)
	}
	if err := run(flag.Arg(0), *tensors, *kv, *check); err != nil {
		log.Fatal(err)
	}
}

func run(path string, listTensors, listKV bool, check string) error {
	set, err := gguf.OpenSet(path)
	if err != nil {
		return err
	}
	defer set.Close()

	fmt.Printf("%s\n", path)
	for i, f := range set.Files {
		fmt.Printf("  shard %d   %-64s %d tensors, %d kv\n", i+1, shorten(f.Path), len(f.Tensors), len(f.KV))
	}
	fmt.Printf("  arch       %s\n", set.Arch())
	fmt.Printf("  tensors    %d\n", set.Len())
	fmt.Printf("  parameters %.3f B\n", float64(set.Params())/1e9)
	fmt.Printf("  on disk    %.2f GB, %.2f bits/weight\n\n",
		float64(set.Bytes())/1e9, float64(set.Bytes())*8/float64(set.Params()))

	if listKV {
		printKV(set)
	}

	printGroups(set)

	if listTensors {
		printTensors(set)
	}
	if check != "" {
		return compare(set, check)
	}
	return nil
}

// group buckets a qwen4exp tensor by the role it plays in the decode budget.
//
// Order matters, and it is the Python's order for the same reason: `hc_attn_*`
// contains "attn", and the DeltaNet layers' input projections are called
// `attn_qkv`/`attn_gate`, so the specific tests have to come before the
// generic "attn" one.
func group(name string) string {
	switch {
	case strings.HasPrefix(name, "per_layer_token_embd"):
		return "ngram_ple_table"
	case strings.HasPrefix(name, "token_embd"):
		return "embed(lookup)"
	case strings.HasPrefix(name, "output."):
		return "lm_head"
	case strings.Contains(name, "exps"):
		return "moe_experts"
	case strings.Contains(name, "shexp"):
		return "moe_shared"
	case strings.Contains(name, "ffn_gate_inp"):
		return "moe_router"
	case strings.Contains(name, "ssm"), strings.Contains(name, "attn_qkv"), strings.Contains(name, "attn_gate"):
		return "deltanet"
	case strings.Contains(name, "indexer"):
		return "qsa_indexer"
	case strings.Contains(name, "hc_"):
		return "hyper_conn"
	case strings.Contains(name, "attn"):
		return "full_attn"
	case strings.Contains(name, "ple"):
		return "ple_proj"
	}
	return "norms/other"
}

// gathered are the groups read a row at a time rather than streamed whole, so
// they cost capacity but essentially no bandwidth (LLM.md D2).
var gathered = map[string]bool{"ngram_ple_table": true, "embed(lookup)": true}

type groupStat struct {
	params int64
	bytes  int64
	mix    map[gguf.Type]int64
}

func printGroups(set *gguf.Set) {
	g := map[string]*groupStat{}
	for _, t := range set.Tensors {
		name := group(t.Name)
		s := g[name]
		if s == nil {
			s = &groupStat{mix: map[gguf.Type]int64{}}
			g[name] = s
		}
		s.params += t.Elems()
		s.bytes += int64(len(t.Data))
		s.mix[t.Type] += int64(len(t.Data))
	}

	nExpert, _ := set.Uint(set.Arch() + ".expert_count")
	nUsed, _ := set.Uint(set.Arch() + ".expert_used_count")

	names := make([]string, 0, len(g))
	for n := range g {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool { return g[names[i]].bytes > g[names[j]].bytes })

	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "GROUP\tPARAMS B\tGB\tBITS/W\tREAD\tQUANT MIX (GB)")
	for _, n := range names {
		s := g[n]
		how := "every tok"
		switch {
		case gathered[n]:
			how = "gather"
		case n == "moe_experts" && nExpert > 0:
			how = fmt.Sprintf("%d/%d", nUsed, nExpert)
		}
		fmt.Fprintf(tw, "%s\t%.3f\t%.2f\t%.2f\t%s\t%s\n", n,
			float64(s.params)/1e9, float64(s.bytes)/1e9,
			float64(s.bytes)*8/float64(s.params), how, mixString(s.mix))
	}
	tw.Flush()

	// The decode budget. Everything but the experts and the gathered tables
	// is read once per token; the experts are read `expert_used/expert_count`
	// of the way through.
	var table, dense, expertBytes int64
	for n, s := range g {
		switch {
		case n == "ngram_ple_table":
			table += s.bytes
		case gathered[n]:
		case n == "moe_experts":
			expertBytes = s.bytes
		default:
			dense += s.bytes
		}
	}
	experts := float64(expertBytes)
	if nExpert > 0 {
		experts = float64(expertBytes) * float64(nUsed) / float64(nExpert)
	}
	perToken := float64(dense) + experts
	core := set.Bytes() - table

	fmt.Printf("\nresident core (all but the n-gram table): %.2f GB\n", float64(core)/1e9)
	fmt.Printf("n-gram table, off-heap and mmap'd:        %.2f GB\n", float64(table)/1e9)
	fmt.Printf("\ndecode: dense %.3f + experts %.3f = %.3f GB/token\n",
		float64(dense)/1e9, experts/1e9, perToken/1e9)
	fmt.Printf("  at %.0f GB/s (IDEAS §1.7) that is a %.1f tok/s ceiling\n", busGBs, busGBs/(perToken/1e9))
	fmt.Printf("  dense is %.0f%% of it\n", 100*float64(dense)/perToken)
}

func mixString(mix map[gguf.Type]int64) string {
	type kv struct {
		t gguf.Type
		b int64
	}
	var xs []kv
	for t, b := range mix {
		xs = append(xs, kv{t, b})
	}
	sort.Slice(xs, func(i, j int) bool { return xs[i].b > xs[j].b })
	var parts []string
	for _, x := range xs {
		parts = append(parts, fmt.Sprintf("%s:%.1f", x.t, float64(x.b)/1e9))
	}
	return strings.Join(parts, " ")
}

func printKV(set *gguf.Set) {
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "KEY\tVALUE")
	for _, k := range set.Keys() {
		fmt.Fprintf(tw, "%s\t%s\n", k, abbrev(set.KVFile().KV[k]))
	}
	tw.Flush()
	fmt.Println()
}

// abbrev renders a metadata value in one line. The tokenizer's arrays are
// 248 320 entries, so arrays print their length and first few elements.
func abbrev(v any) string {
	const head = 4
	switch a := v.(type) {
	case []string:
		return fmt.Sprintf("[%d strings] %s", len(a), strings.Join(quoteN(a, head), " "))
	case []uint64:
		return fmt.Sprintf("[%d uints] %v", len(a), a[:min(head, len(a))])
	case []int64:
		return fmt.Sprintf("[%d ints] %v", len(a), a[:min(head, len(a))])
	case []float64:
		return fmt.Sprintf("[%d floats] %v", len(a), a[:min(head, len(a))])
	case []bool:
		return fmt.Sprintf("[%d bools] %v", len(a), a[:min(head, len(a))])
	case string:
		if len(a) > 120 {
			return fmt.Sprintf("%q… (%d bytes)", a[:120], len(a))
		}
		return fmt.Sprintf("%q", a)
	}
	return fmt.Sprintf("%v", v)
}

func quoteN(a []string, n int) []string {
	out := make([]string, 0, n)
	for _, s := range a[:min(n, len(a))] {
		out = append(out, fmt.Sprintf("%q", s))
	}
	return out
}

func printTensors(set *gguf.Set) {
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "\nTENSOR\tTYPE\tDIMS\tPARAMS\tMB\tSHARD")
	for _, t := range set.Tensors {
		fmt.Fprintf(tw, "%s\t%s\t%v\t%d\t%.1f\t%d\n",
			t.Name, t.Type, t.Dims, t.Elems(), float64(len(t.Data))/1e6, t.Shard+1)
	}
	tw.Flush()
}

// compare checks the Go tensor table against the Python's tensors.json dump,
// which is a [[name, dims, type-name], …]. Name for name, dim for dim, type
// for type — the acceptance criterion for the reader.
func compare(set *gguf.Set, path string) error {
	buf, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var ref [][]json.RawMessage
	if err := json.Unmarshal(buf, &ref); err != nil {
		return fmt.Errorf("parsing %s: %w", path, err)
	}
	seen := map[string]bool{}
	bad := 0
	for _, row := range ref {
		var name, typ string
		var dims []int64
		if err := json.Unmarshal(row[0], &name); err != nil {
			return err
		}
		if err := json.Unmarshal(row[1], &dims); err != nil {
			return err
		}
		if err := json.Unmarshal(row[2], &typ); err != nil {
			return err
		}
		seen[name] = true
		t, err := set.Get(name)
		if err != nil {
			fmt.Printf("  MISSING %s\n", name)
			bad++
			continue
		}
		if t.Type.String() != typ {
			fmt.Printf("  TYPE    %s: go %s, python %s\n", name, t.Type, typ)
			bad++
		}
		if len(t.Dims) != len(dims) {
			fmt.Printf("  RANK    %s: go %v, python %v\n", name, t.Dims, dims)
			bad++
			continue
		}
		for i := range dims {
			if t.Dims[i] != dims[i] {
				fmt.Printf("  DIMS    %s: go %v, python %v\n", name, t.Dims, dims)
				bad++
				break
			}
		}
	}
	for _, t := range set.Tensors {
		if !seen[t.Name] {
			fmt.Printf("  EXTRA   %s\n", t.Name)
			bad++
		}
	}
	fmt.Printf("\ncheck against %s: %d reference tensors, %d in the checkpoint, %d disagreements\n",
		path, len(ref), set.Len(), bad)
	if bad != 0 {
		return fmt.Errorf("%d disagreements", bad)
	}
	return nil
}

func shorten(p string) string {
	if i := strings.LastIndexByte(p, '/'); i >= 0 {
		return p[i+1:]
	}
	return p
}
