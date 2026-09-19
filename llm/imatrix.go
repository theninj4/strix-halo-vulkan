package llm

// The importance matrix: LLM.md L8c-2.
//
// L8c-1 measured every width with round-to-nearest and no importance
// weighting at all, and concluded that D3's ~4.25 bits costs 18.5% of
// perplexity. That is a measurement of *uncalibrated* quantisation, against a
// checkpoint whose own experts are imatrix-quantised — so the number it puts
// on the format is a lower bound on what the format can do, and the gap is
// whatever calibration is worth. Unsloth publish theirs, which makes the
// question free to ask:
//
//	models/Qwen3.8-Flash-Next-GGUF/imatrix_unsloth.gguf   580 MB, 1852 tensors
//	general.type = "imatrix", 45 chunks of 18432 tokens
//
// **It is a GGUF, and our own reader opens it.** Two tensors per weight:
// `<name>.in_sum2`, the sum over calibration tokens of the squared activation
// on each *input column*, and `<name>.counts`, how many were summed. The
// importance of column j is `in_sum2[j] / counts` — llama.cpp's own
// normalisation (`tools/quantize/quantize.cpp`), and the length is the weight
// matrix's k, which is exactly the axis a scale group runs along.
//
// Expert tensors carry one row per expert (`[k, 512]` with `[1, 512]` counts)
// and are not read here: L5b stages the expert banks byte for byte out of the
// checkpoint, so the simulation never sees them (sim.go).

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"strix-halo-vulkan/gguf"
)

// defaultImatrix is where reference/fetch_llm_checkpoint.sh puts it, tried
// from the repo root and from one directory down — the same two places the
// tests' `modelDir` covers, because `go test ./llm/` runs in `llm/` and every
// command runs in the root.
var defaultImatrix = []string{
	"models/Qwen3.8-Flash-Next-GGUF/imatrix_unsloth.gguf",
	"../models/Qwen3.8-Flash-Next-GGUF/imatrix_unsloth.gguf",
}

// Imatrix is the published importance matrix, held open.
type Imatrix struct {
	set   *gguf.Set
	mu    sync.Mutex
	cache map[string][]float32
	// Datasets and Chunks are the provenance, for a header line: a
	// calibration set is part of a quantisation's identity.
	Datasets string
	Chunks   uint64
}

// OpenImatrix opens one.
func OpenImatrix(path string) (*Imatrix, error) {
	set, err := gguf.OpenSet(path)
	if err != nil {
		return nil, fmt.Errorf("llm: imatrix %s: %w", path, err)
	}
	if t, ok := set.Str("general.type"); !ok || t != "imatrix" {
		set.Close()
		return nil, fmt.Errorf("llm: %s is general.type %q, want \"imatrix\"", path, t)
	}
	im := &Imatrix{set: set, cache: map[string][]float32{}}
	if ds, ok := set.Strings("imatrix.datasets"); ok {
		im.Datasets = strings.Join(ds, " ")
	}
	im.Chunks, _ = set.Uint("imatrix.chunk_count")
	return im, nil
}

// Close releases the mapping.
func (im *Imatrix) Close() error { return im.set.Close() }

// Columns returns the per-input-column importance for a weight tensor, or nil
// if the imatrix has no entry for it — which is not an error: the calibration
// run only covers what its graph executed.
//
// The values are `in_sum2 / counts`, llama.cpp's own normalisation. Only the
// *relative* weights within a scale group matter to make_qx_quants — a common
// factor cancels out of both `sumlx/suml2` and the `sumlx^2/suml2` comparison
// — so the division is for legibility rather than for correctness.
func (im *Imatrix) Columns(weight string) ([]float32, error) {
	im.mu.Lock()
	defer im.mu.Unlock()
	if v, ok := im.cache[weight]; ok {
		return v, nil
	}
	sums, err := im.set.Get(weight + ".in_sum2")
	if err != nil {
		im.cache[weight] = nil
		return nil, nil
	}
	counts, err := im.set.Get(weight + ".counts")
	if err != nil {
		return nil, fmt.Errorf("llm: imatrix %s has in_sum2 but no counts", weight)
	}
	v, err := sums.Dequantize(nil)
	if err != nil {
		return nil, fmt.Errorf("llm: imatrix %s.in_sum2: %w", weight, err)
	}
	c, err := counts.Dequantize(nil)
	if err != nil {
		return nil, fmt.Errorf("llm: imatrix %s.counts: %w", weight, err)
	}
	if len(c) != 1 {
		// A per-expert entry: [k, nExpert] values against [1, nExpert]
		// counts. Nothing here stages an expert bank through the host, so
		// rather than pick a row this refuses.
		return nil, fmt.Errorf("llm: imatrix %s has %d counts, this reader handles one", weight, len(c))
	}
	if c[0] > 0 {
		for i := range v {
			v[i] /= c[0]
		}
	} else {
		for i := range v {
			v[i] = 1
		}
	}
	im.cache[weight] = v
	return v, nil
}

// ExpertColumns is `Columns` for a per-expert entry: the published matrix
// stores an expert tensor's importance as `[k, nExpert]` against `[1,
// nExpert]` counts, one row per expert, and this returns expert `e`'s.
//
// The comment at the top of this file used to say expert tensors "are not
// read here: L5b stages the expert banks byte for byte out of the checkpoint,
// so the simulation never sees them". P4 is where that stops being true —
// the shared expert's two 2560-wide matrices and the five Q8_0 layers of
// `ffn_down_exps` are transcoded, and a transcode without the calibration is
// the thing L8c-3 measured at three times the cost.
//
// The normalisation is the same `in_sum2 / counts`, per expert: an expert
// that the calibration run never routed to has a zero count, and rather than
// divide by it this falls back to uniform importance, which is exactly what
// an absent entry does one level up.
func (im *Imatrix) ExpertColumns(weight string, e int) ([]float32, error) {
	im.mu.Lock()
	defer im.mu.Unlock()
	key := fmt.Sprintf("%s#%d", weight, e)
	if v, ok := im.cache[key]; ok {
		return v, nil
	}
	sums, err := im.set.Get(weight + ".in_sum2")
	if err != nil {
		im.cache[key] = nil
		return nil, nil
	}
	counts, err := im.set.Get(weight + ".counts")
	if err != nil {
		return nil, fmt.Errorf("llm: imatrix %s has in_sum2 but no counts", weight)
	}
	c, err := counts.Dequantize(nil)
	if err != nil {
		return nil, fmt.Errorf("llm: imatrix %s.counts: %w", weight, err)
	}
	if e < 0 || e >= len(c) {
		return nil, fmt.Errorf("llm: imatrix %s has %d experts, asked for %d", weight, len(c), e)
	}
	if int64(len(c)) != sums.Rows() {
		return nil, fmt.Errorf("llm: imatrix %s has %d counts against %d rows of in_sum2",
			weight, len(c), sums.Rows())
	}
	v, err := sums.DequantizeRow(int64(e), nil)
	if err != nil {
		return nil, fmt.Errorf("llm: imatrix %s.in_sum2 row %d: %w", weight, e, err)
	}
	if c[e] > 0 {
		for i := range v {
			v[i] /= c[e]
		}
	} else {
		for i := range v {
			v[i] = 1
		}
	}
	im.cache[key] = v
	return v, nil
}

var (
	imatrixOnce sync.Once
	imatrixOpen *Imatrix
	imatrixErr  error
)

// DefaultImatrix opens the published matrix once, from LLM_IMATRIX or the
// path the fetch script writes.
func DefaultImatrix() (*Imatrix, error) {
	imatrixOnce.Do(func() {
		if path := os.Getenv("LLM_IMATRIX"); path != "" {
			imatrixOpen, imatrixErr = OpenImatrix(path)
			return
		}
		for _, path := range defaultImatrix {
			if _, err := os.Stat(path); err != nil {
				continue
			}
			imatrixOpen, imatrixErr = OpenImatrix(path)
			return
		}
		imatrixErr = fmt.Errorf("llm: no imatrix at %v; reference/fetch_llm_checkpoint.sh fetches it",
			defaultImatrix)
	})
	return imatrixOpen, imatrixErr
}
