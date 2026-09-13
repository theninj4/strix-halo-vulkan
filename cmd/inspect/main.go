// Command inspect reports what is actually inside a safetensors checkpoint:
// the tensor inventory, the dtypes, the parameter count, and the shapes
// collapsed over repeated layers.
//
// It exists because bench/modelshapes.go's dimensions were transcribed from
// config.json files by hand, and that transcription had gaps — the real
// Z-Image transformer has 34 blocks rather than 30 (30 `layers` plus two
// `context_refiner` and two `noise_refiner`) and a per-layer
// adaLN_modulation of [15360, 256] that the shape table does not model at
// all. Reading the checkpoint is the only way to not repeat that.
//
//	go run ./cmd/inspect models/Z-Image-Turbo/transformer
//	go run ./cmd/inspect -tensors models/Z-Image-Turbo/vae
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"
	"text/tabwriter"

	"strix-halo-vulkan/safetensors"
)

func main() {
	log.SetFlags(0)
	full := flag.Bool("tensors", false, "list every tensor instead of collapsing repeated layers")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "usage: %s [-tensors] <checkpoint-dir>\n", os.Args[0])
		os.Exit(2)
	}
	if err := run(flag.Arg(0), *full); err != nil {
		log.Fatal(err)
	}
}

// layerIndex matches the numeric component of a tensor path so that
// `layers.0.attention.to_q.weight` and `layers.29.attention.to_q.weight`
// collapse onto one row.
var layerIndex = regexp.MustCompile(`\.\d+\.`)

func run(dir string, full bool) error {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return err
	}
	defer set.Close()

	fmt.Printf("%s\n", dir)
	fmt.Printf("  tensors    %d\n", set.Len())
	fmt.Printf("  parameters %.3f B\n", float64(set.Params())/1e9)
	fmt.Printf("  on disk    %.2f GB\n", float64(set.Bytes())/1e9)

	dtypes := map[safetensors.DType]int{}
	for _, n := range set.Names() {
		t, err := set.Get(n)
		if err != nil {
			return err
		}
		dtypes[t.DType]++
	}
	var ds []string
	for d, c := range dtypes {
		ds = append(ds, fmt.Sprintf("%s x%d", d, c))
	}
	sort.Strings(ds)
	fmt.Printf("  dtypes     %s\n", strings.Join(ds, ", "))
	// Every model here is destined for the fp16 matrix path, so the number
	// that decides whether a stage fits in a resident buffer is the fp16
	// size, not the on-disk one.
	fmt.Printf("  as fp16    %.2f GB\n\n", float64(set.Params()*2)/1e9)

	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	if full {
		fmt.Fprintln(tw, "TENSOR\tDTYPE\tSHAPE\tPARAMS")
		names := append([]string(nil), set.Names()...)
		sort.Strings(names)
		for _, n := range names {
			t, err := set.Get(n)
			if err != nil {
				return err
			}
			fmt.Fprintf(tw, "%s\t%s\t%v\t%d\n", n, t.DType, t.Shape, t.Elems())
		}
		return tw.Flush()
	}

	type group struct {
		pattern string
		dtype   safetensors.DType
		shape   []int
		count   int
		params  int64
	}
	groups := map[string]*group{}
	var order []string
	for _, n := range set.Names() {
		t, err := set.Get(n)
		if err != nil {
			return err
		}
		key := layerIndex.ReplaceAllString(n, ".N.")
		g, ok := groups[key]
		if !ok {
			g = &group{pattern: key, dtype: t.DType, shape: t.Shape}
			groups[key] = g
			order = append(order, key)
		}
		g.count++
		g.params += int64(t.Elems())
	}
	sort.Strings(order)

	fmt.Fprintln(tw, "COUNT\tPATTERN\tDTYPE\tSHAPE\tPARAMS")
	for _, k := range order {
		g := groups[k]
		fmt.Fprintf(tw, "x%d\t%s\t%s\t%v\t%.1f M\n", g.count, g.pattern, g.dtype, g.shape, float64(g.params)/1e6)
	}
	return tw.Flush()
}
