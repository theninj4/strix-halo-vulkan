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
	width := flag.Float64("width", 0, "L8c: re-price the decode budget with the streamed dense families at this many bits a weight")
	expertWidth := flag.Float64("expert-width", 0, "L8c: and the 512 expert banks at this many")
	routerWidth := flag.Float64("router-width", 0, "L8c: and the F32 router at this many (16 is D3's fp16)")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "usage: %s [-tensors] [-kv] [-check tensors.json] <checkpoint|shard|dir>\n", os.Args[0])
		os.Exit(2)
	}
	w := widths{dense: *width, experts: *expertWidth, router: *routerWidth}
	if err := run(flag.Arg(0), *tensors, *kv, *check, w); err != nil {
		log.Fatal(err)
	}
}

// widths is L8c's what-if: a candidate bits-per-weight for each of the three
// parts of the budget that could move, zero meaning "as the checkpoint ships
// it". It prices a format; it says nothing about whether the model survives
// it, which is what `cmd/llm -ppl` under `LLM_DENSE_SIM` is for.
type widths struct{ dense, experts, router float64 }

func (w widths) any() bool { return w.dense > 0 || w.experts > 0 || w.router > 0 }

// streamedDense are the groups a dense re-quantisation would reach: every
// matmul read once per token that is not the router and not an expert. It is
// the same list as llm.SimFamilies(), so an accuracy row and a byte row name
// the same thing.
var streamedDense = map[string]bool{
	"deltanet": true, "hyper_conn": true, "lm_head": true,
	"full_attn": true, "qsa_indexer": true, "ple_proj": true,
}

func run(path string, listTensors, listKV bool, check string, w widths) error {
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

	printGroups(set, w)

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
	// rows is every distinct row length in the group — ggml's Dims[0], the
	// axis a block runs along — mapped to the bytes at that length.
	//
	// It is here because P4 needed it and no table had it. A K-quant's
	// super-block is QK_K = 256 elements, so a tensor whose row is not a
	// multiple of 256 **cannot be stored as Q4_K at all**, and re-pricing it
	// at 4.5 bits prices a format that does not exist. `ffn_down_exps` is
	// [640, 2560, 512] — 640 is 2.5 super-blocks — which is why unsloth
	// shipped the one expert tensor in the model at Q5_1 and Q8_0, both of
	// them block-32 formats, while gate and up (row 2560) are Q4_K.
	rows map[int64]int64
}

// qkK is ggml's K-quant super-block, in elements.
const qkK = 256

// kquantable reports whether every row in the group divides the super-block,
// which is what a Q4_K/Q5_K re-pricing silently assumes.
func (s *groupStat) kquantable() bool {
	for n := range s.rows {
		if n%qkK != 0 {
			return false
		}
	}
	return true
}

// badRows is the row lengths that do not, smallest first, for the line that
// says why a re-pricing was refused.
func (s *groupStat) badRows() []int64 {
	var xs []int64
	for n := range s.rows {
		if n%qkK != 0 {
			xs = append(xs, n)
		}
	}
	sort.Slice(xs, func(i, j int) bool { return xs[i] < xs[j] })
	return xs
}

func printGroups(set *gguf.Set, w widths) {
	g := map[string]*groupStat{}
	for _, t := range set.Tensors {
		name := group(t.Name)
		s := g[name]
		if s == nil {
			s = &groupStat{mix: map[gguf.Type]int64{}, rows: map[int64]int64{}}
			g[name] = s
		}
		s.params += t.Elems()
		s.bytes += int64(len(t.Data))
		s.mix[t.Type] += int64(len(t.Data))
		if len(t.Dims) > 0 {
			s.rows[t.Dims[0]] += int64(len(t.Data))
		}
	}

	nExpert, _ := set.Uint(set.Arch() + ".expert_count")
	nUsed, _ := set.Uint(set.Arch() + ".expert_used_count")

	names := make([]string, 0, len(g))
	for n := range g {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool { return g[names[i]].bytes > g[names[j]].bytes })

	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "GROUP\tPARAMS B\tGB\tBITS/W\tREAD\tROW\tQUANT MIX (GB)")
	for _, n := range names {
		s := g[n]
		how := "every tok"
		switch {
		case gathered[n]:
			how = "gather"
		case n == "moe_experts" && nExpert > 0:
			how = fmt.Sprintf("%d/%d", nUsed, nExpert)
		}
		row := "k-quant"
		if !s.kquantable() {
			row = fmt.Sprintf("block-32 (%s)", joinInts(s.badRows()))
		}
		fmt.Fprintf(tw, "%s\t%.3f\t%.2f\t%.2f\t%s\t%s\t%s\n", n,
			float64(s.params)/1e9, float64(s.bytes)/1e9,
			float64(s.bytes)*8/float64(s.params), how, row, mixString(s.mix))
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

	if r := g["moe_router"]; r != nil && nExpert > 0 {
		staged := stagedRouterBytes(r, nExpert)
		was := float64(r.bytes)
		fmt.Printf("\n  ...but the router is **not** read at the width it ships in: llm/gpu_moe.go\n")
		fmt.Printf("  stages it as halves, %d columns padded to %d, so its row is %.3f GB a token\n",
			nExpert+1, roundUp64(int64(nExpert)+1, 64), staged/1e9)
		fmt.Printf("  and not %.3f. As this repo stages it: **%.3f GB/token, a %.1f tok/s ceiling** (P4a).\n",
			was/1e9, (perToken-was+staged)/1e9, busGBs/((perToken-was+staged)/1e9))
	}

	if w.any() {
		printWidths(g, names, w, nExpert, nUsed, perToken)
	}
}

// printWidths re-prices the budget at a candidate set of widths.
//
// Bytes are the half of L8c that is arithmetic: a group's parameter count is
// a fact about the checkpoint, so what it costs at 4.25 bits needs no
// measurement. The half that does is whether the model survives, and that is
// `LLM_DENSE_SIM` plus `cmd/llm -ppl`.
//
// **P4a adds the half that is neither**: whether the format exists for that
// group's rows. A width between 4 and 5.5 bits means a K-quant here, and a
// K-quant's super-block is 256 elements, so pricing `moe_experts` at 4.5 bits
// prices a Q4_K that `ffn_down_exps` cannot be stored as — its row is 640.
// The table says so per group rather than leaving it to be re-derived.
func printWidths(g map[string]*groupStat, names []string, w widths, nExpert, nUsed uint64, was float64) {
	bytesAt := func(name string, s *groupStat) float64 {
		bits := 0.0
		switch {
		case streamedDense[name]:
			bits = w.dense
		case name == "moe_experts":
			bits = w.experts
		case name == "moe_router":
			bits = w.router
		}
		if bits <= 0 {
			return float64(s.bytes)
		}
		return float64(s.params) * bits / 8
	}
	fmt.Printf("\nre-priced: dense %s, experts %s, router %s\n",
		bitsLabel(w.dense), bitsLabel(w.experts), bitsLabel(w.router))
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "GROUP\tBITS/W\tGB\tGB/TOKEN\tWAS\t")
	var dense, expertBytes float64
	for _, n := range names {
		s := g[n]
		if gathered[n] {
			continue
		}
		now := bytesAt(n, s)
		share := 1.0
		if n == "moe_experts" && nExpert > 0 {
			share = float64(nUsed) / float64(nExpert)
			expertBytes = now
		} else if n != "moe_experts" {
			dense += now
		}
		note := ""
		if bits := now * 8 / float64(s.params); now != float64(s.bytes) && checkpointBytes[n] && kquantWidth(bits) && !s.kquantable() {
			if b32 := block32Format(bits); b32 != "" {
				// P4c: the row takes no K-quant, but this particular width
				// is one a block-32 format meets, and the MoE kernels have
				// an arm for it. So the price is real — it is just not the
				// format the width's name suggests.
				note = fmt.Sprintf("  <- rows %s take no K-quant; at this width that is %s (P4c)",
					joinInts(s.badRows()), b32)
			} else {
				note = fmt.Sprintf("  <- no such format: rows %s are not multiples of %d", joinInts(s.badRows()), qkK)
			}
		}
		fmt.Fprintf(tw, "%s\t%.2f\t%.2f\t%.3f\t%.3f\t%s\n", n, now*8/float64(s.params),
			now/1e9, now*share/1e9, float64(s.bytes)*share/1e9, note)
	}
	tw.Flush()
	experts := expertBytes
	if nExpert > 0 {
		experts = expertBytes * float64(nUsed) / float64(nExpert)
	}
	now := dense + experts
	fmt.Printf("decode: dense %.3f + experts %.3f = %.3f GB/token  (was %.3f, %.2fx)\n",
		dense/1e9, experts/1e9, now/1e9, was/1e9, was/now)
	fmt.Printf("  at %.0f GB/s that is a %.1f tok/s ceiling (was %.1f)\n",
		busGBs, busGBs/(now/1e9), busGBs/(was/1e9))
}

// stagedRouterBytes is the fused router as `MoEGPU.stage` writes it: halves,
// the experts' 512 columns plus the shared expert's gate, padded up to the
// plain GEMM's 64-wide block. P4a.
func stagedRouterBytes(r *groupStat, nExpert uint64) float64 {
	perLayer := r.params / int64(nExpert) // nEmbd
	return float64(perLayer * roundUp64(int64(nExpert)+1, 64) * 2)
}

func roundUp64(n, to int64) int64 { return (n + to - 1) / to * to }

func joinInts(xs []int64) string {
	var parts []string
	for _, x := range xs {
		parts = append(parts, fmt.Sprintf("%d", x))
	}
	return strings.Join(parts, ",")
}

// checkpointBytes are the groups whose device bank is the checkpoint's own
// bytes, read by a shader that implements a ggml format (L5b: `MoEGPU.stage`
// memcpy's them). For those a width is only buyable if ggml has a format for
// it *at that row length*, which is what the note on the table checks.
//
// Every other streamed group goes through `bank_q4.go`, which is **not**
// ggml's layout — its own fragment tiling, and a super-block that is the
// whole row for the 320-wide hyper-connection family. That bank takes any row
// length the tiling divides, so the same note would be wrong about it: the
// dense families are 4.5 and 5.5 bits today at rows of 320.
var checkpointBytes = map[string]bool{"moe_experts": true, "moe_shared": true}

// kquantWidth is whether a bits-per-weight figure is in the range where the
// question "is this row a multiple of 256" has to be asked at all: between
// them the K-quants and the block-32 formats cover it, and which of the two a
// width lands on decides whether a 640-wide row can have it.
func kquantWidth(bits float64) bool { return bits > 4.0 && bits < 8.0 }

// block32Format names the ggml format at `bits` whose block is 32 elements,
// so it fits any row a multiple of 32 — which is what the down projection's
// 640 is. It is the other half of the note on the table: a K-quant width on a
// 640-wide row is unbuildable, but several *widths* are met by both kinds of
// format, and at those the re-pricing is honest.
//
// P4c built the two that matter. IQ4_NL wins 4.5 over Q4_0 at identical bytes
// (D4, L8c-1), so it is the one named.
func block32Format(bits float64) string {
	switch {
	case near(bits, 4.5):
		return "IQ4_NL"
	case near(bits, 5.0):
		return "Q4_1"
	case near(bits, 6.0):
		return "Q5_1"
	case near(bits, 8.5):
		return "Q8_0"
	}
	return ""
}

func near(a, b float64) bool { return a-b < 0.01 && b-a < 0.01 }

func bitsLabel(b float64) string {
	if b <= 0 {
		return "as shipped"
	}
	return fmt.Sprintf("%.2f bits", b)
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
