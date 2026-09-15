package bench

import (
	"fmt"
	"io"
	"math"
	"runtime"
	"sort"
	"sync"
	"text/tabwriter"
)

// Quantization *accuracy*, which is the one thing this suite has never
// measured and the one thing IDEAS §7 says the format decision should turn on.
// LLM.md **L0d**.
//
// Everything else here ranks formats by GFLOP/s. That ranking is now very
// well established — W4A8 decode reads 99-103% of the bus (§1.7), a 4-bit MoE
// prefill is 2.10x its fp16 twin (§2.2), and L0c took the fine scale block
// from a 14.2% tax to 1.0%. What none of it says is whether the *outputs* are
// any good, and two live decisions depend on that and nothing else:
//
//  1. **W4A8 or W4A16.** Every decode number in LLM.md assumes int8
//     activations, because `dotPacked4x8EXT` is what reaches the bus. If int8
//     activations are unacceptable the fallback is unpacking 4-bit weights to
//     fp16, which §1.1 measured at **74 GB/s — 31% of the bus**. That is a 3x
//     cliff standing behind a question nobody has asked.
//  2. **Symmetric or asymmetric, and at what block size.** These trade against
//     each other at *equal bits*: symmetric costs 4 + 16/block bits a weight
//     and asymmetric 4 + 32/block, so asymmetric at block 64 and symmetric at
//     block 32 are both 4.5 bits and only a measurement separates them.
//
// The measurement is a real (weight, activation) pair from a real forward
// pass, not a synthetic matrix. That matters more here than anywhere else in
// this file: activation outliers are the whole risk in int8, they are a
// property of a trained model, and a Gaussian test matrix has none. Stage 6
// already ran into this from the other side — SwiGLU overflows fp16 "on a real
// prompt and never on a random one".
//
// Accumulation is float64 everywhere, reference and candidate alike, so the
// only thing that differs between rows is the quantization. The real W4A8
// kernel accumulates int32 within a block and fp32 across blocks, which is
// very slightly worse than this; the gap is not what these numbers are about.

// QuantSite is one projection drawn from a real forward pass: the weight, and
// the activations that were actually fed to it.
type QuantSite struct {
	Name   string    // "L0.q", "L0.down", ...
	Rows   int       // out features
	Cols   int       // in features — the reduction axis, and the block axis
	W      []float32 // [Rows][Cols]
	X      []float32 // [Tokens][Cols]
	Tokens int
}

// QuantErrorRow is one (weight format, activation format) pair's error at one
// site.
type QuantErrorRow struct {
	Site       string
	Weight     string
	Act        string
	BitsPerW   float64 // including the scale (and, asymmetric, the min)
	RelRMS     float64 // ||Yhat - Y|| / ||Y||
	MaxRelErr  float64 // max|Yhat - Y| / max|Y|
	SNRdB      float64
	CosineLoss float64 // 1 - cos(Yhat, Y), the part a normalised model feels
	// WRelRMS is the weight *reconstruction* error, ||What - W|| / ||W||,
	// with no activation involved. Carried alongside the output error because
	// the two can disagree — the output error is the format's error seen
	// through one projection's conditioning, and separating them is the only
	// way to tell a bad format from a badly-conditioned matrix.
	WRelRMS float64
}

// weightFormat is a quantize/dequantize round trip plus what it costs.
type weightFormat struct {
	name  string
	bits  func(block int) float64
	block int
	apply func(w []float32, rows, cols, block int) []float32
}

// QuantWeightFormats returns the weight formats to sweep at the given block
// sizes. fp16 is included as the floor: it is what the *unquantized* path
// costs, so a 4-bit row that lands near it has nothing left to win.
func QuantWeightFormats(blocks []int) []weightFormat {
	out := []weightFormat{{
		name: "fp16", block: 0,
		bits:  func(int) float64 { return 16 },
		apply: func(w []float32, _, _, _ int) []float32 { return float16RoundTrip(w) },
	}}
	for _, b := range blocks {
		b := b
		out = append(out,
			weightFormat{
				// The same scheme with the scale kept in fp32 instead of
				// fp16. It exists because L0d found `quantizeQ8`'s fp16
				// scale going *subnormal*: the scale is maxAbs/127, so a
				// block whose largest weight is under 7.75e-3 lands below
				// fp16's smallest normal and keeps only a few mantissa bits,
				// at which point the scale is the error rather than the
				// int8. Q4 never sees it — its scale is maxAbs/7, eighteen
				// times larger. This arm is the control that proves the
				// diagnosis; the fix in the real quantizer is to floor the
				// scale at fp16's smallest normal, which costs a little
				// resolution and keeps the whole mantissa.
				name: "q8sym-f32s", block: b,
				bits: func(b int) float64 { return 8 + 32/float64(b) },
				apply: func(w []float32, rows, cols, block int) []float32 {
					return q8F32ScaleRoundTrip(w, rows, cols, block)
				},
			},
			weightFormat{
				name: "q8sym", block: b,
				bits: func(b int) float64 { return 8 + 16/float64(b) },
				apply: func(w []float32, rows, cols, block int) []float32 {
					q, s := quantizeQ8(w, rows, cols, block)
					return dequantizeQ8(q, s, rows, cols, block)
				},
			},
			weightFormat{
				name: "q4sym", block: b,
				bits: func(b int) float64 { return 4 + 16/float64(b) },
				apply: func(w []float32, rows, cols, block int) []float32 {
					p, s := quantizeQ4(w, rows, cols, block)
					return dequantizeQ4(p, s, rows, cols, block)
				},
			},
			weightFormat{
				name: "q4asym", block: b,
				bits: func(b int) float64 { return 4 + 32/float64(b) },
				apply: func(w []float32, rows, cols, block int) []float32 {
					p, s, m := quantizeQ4Asym(w, rows, cols, block)
					return dequantizeQ4Asym(p, s, m, rows, cols, block)
				},
			})
	}
	return out
}

// q8F32ScaleRoundTrip is quantizeQ8/dequantizeQ8 with the block scale left in
// fp32. Only the scale's storage differs from the shader-facing pair.
func q8F32ScaleRoundTrip(data []float32, rows, cols, block int) []float32 {
	out := make([]float32, len(data))
	blocksPerRow := cols / block
	for r := 0; r < rows; r++ {
		for bi := 0; bi < blocksPerRow; bi++ {
			start := r*cols + bi*block
			var maxAbs float32
			for i := 0; i < block; i++ {
				if a := absF32(data[start+i]); a > maxAbs {
					maxAbs = a
				}
			}
			scale := maxAbs / 127.0
			if scale == 0 {
				scale = 1.0
			}
			for i := 0; i < block; i++ {
				q := clampInt(int(math.Round(float64(data[start+i]/scale))), -127, 127)
				out[start+i] = float32(q) * scale
			}
		}
	}
	return out
}

// actFormat is the activation side. "per token" is one scale per row, which is
// what a real engine computes in the RMSNorm epilogue (§3.1) and is free;
// "per tensor" is the cheaper thing a naive implementation does and the one
// the outlier literature warns about.
type actFormat struct {
	name  string
	apply func(x []float32, tokens, cols int) []float32
}

// QuantActFormats returns the activation round trips to sweep.
func QuantActFormats() []actFormat {
	return []actFormat{
		{name: "exact", apply: func(x []float32, _, _ int) []float32 { return x }},
		{name: "fp16", apply: func(x []float32, _, _ int) []float32 { return float16RoundTrip(x) }},
		{name: "int8/token", apply: func(x []float32, tokens, cols int) []float32 {
			return int8RoundTrip(x, tokens, cols)
		}},
		{name: "int8/tensor", apply: func(x []float32, tokens, cols int) []float32 {
			return int8RoundTrip(x, 1, tokens*cols)
		}},
	}
}

// int8RoundTrip quantizes to symmetric int8 with one scale per row and back.
// Rows of one give the per-tensor variant.
func int8RoundTrip(x []float32, rows, cols int) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		row := x[r*cols : (r+1)*cols]
		var maxAbs float32
		for _, v := range row {
			if a := absF32(v); a > maxAbs {
				maxAbs = a
			}
		}
		scale := maxAbs / 127.0
		if scale == 0 {
			scale = 1
		}
		for i, v := range row {
			q := clampInt(int(math.Round(float64(v/scale))), -127, 127)
			out[r*cols+i] = float32(q) * scale
		}
	}
	return out
}

// AnalyseQuantSite measures every (weight format, activation format) pair at
// one site against a float64 reference.
func AnalyseQuantSite(s QuantSite, blocks []int) []QuantErrorRow {
	ref := matmul64(s.X, s.W, s.Tokens, s.Rows, s.Cols)
	refNorm, refMax := norm64(ref), max64(ref)

	var rows []QuantErrorRow
	for _, wf := range QuantWeightFormats(blocks) {
		if wf.block > 0 && s.Cols%wf.block != 0 {
			continue
		}
		wq := wf.apply(s.W, s.Rows, s.Cols, wf.block)
		wErr := relRMS32(wq, s.W)
		for _, af := range QuantActFormats() {
			// The cross product is mostly uninteresting. What is wanted is
			// each axis alone, plus the two combinations a real engine would
			// ship: W4A16 and W4A8.
			if af.name != "exact" && wf.name != "fp16" && wf.name != "q4sym" {
				continue
			}
			xq := af.apply(s.X, s.Tokens, s.Cols)
			got := matmul64(xq, wq, s.Tokens, s.Rows, s.Cols)
			name := wf.name
			if wf.block > 0 {
				name = fmt.Sprintf("%s/%d", wf.name, wf.block)
			}
			rows = append(rows, QuantErrorRow{
				Site: s.Name, Weight: name, Act: af.name, BitsPerW: wf.bits(wf.block),
				RelRMS:     diffNorm64(got, ref) / refNorm,
				MaxRelErr:  maxDiff64(got, ref) / refMax,
				SNRdB:      20 * math.Log10(refNorm/diffNorm64(got, ref)),
				CosineLoss: 1 - cosine64(got, ref),
				WRelRMS:    wErr,
			})
		}
	}
	return rows
}

// QuantOutliers describes how far from Gaussian one site's activations are,
// which is what decides whether int8 is safe. maxOverRMS is the classic
// number — a channel 20x the RMS spends most of int8's 127 codes on itself —
// and channelSpread is the ratio of the widest input channel to the median
// one, which is what a *per-tensor* scale has to cover and a per-token one
// does not.
type QuantOutliers struct {
	Site          string
	MaxOverRMS    float64
	ChannelSpread float64
	Kurtosis      float64
}

// AnalyseQuantOutliers computes those over one site's activations.
func AnalyseQuantOutliers(s QuantSite) QuantOutliers {
	var sum2, sum4 float64
	for _, v := range s.X {
		d := float64(v)
		sum2 += d * d
		sum4 += d * d * d * d
	}
	n := float64(len(s.X))
	rms := math.Sqrt(sum2 / n)
	chanMax := make([]float64, s.Cols)
	for t := 0; t < s.Tokens; t++ {
		for c := 0; c < s.Cols; c++ {
			if a := math.Abs(float64(s.X[t*s.Cols+c])); a > chanMax[c] {
				chanMax[c] = a
			}
		}
	}
	sorted := append([]float64(nil), chanMax...)
	sort.Float64s(sorted)
	median := sorted[len(sorted)/2]
	spread := 0.0
	if median > 0 {
		spread = sorted[len(sorted)-1] / median
	}
	return QuantOutliers{
		Site:          s.Name,
		MaxOverRMS:    max64(s.X) / rms,
		ChannelSpread: spread,
		Kurtosis:      (sum4 / n) / ((sum2 / n) * (sum2 / n)),
	}
}

// ---- float64 reference arithmetic ------------------------------------------

// matmul64 is Y[t, o] = sum_c X[t, c] * W[o, c], accumulated in float64 so the
// arithmetic is the same for the reference and for every candidate and the
// only thing that differs is what was quantized.
func matmul64(x, w []float32, tokens, rows, cols int) []float64 {
	out := make([]float64, tokens*rows)
	parallelChunks(rows, func(o int) {
		wr := w[o*cols : (o+1)*cols]
		for t := 0; t < tokens; t++ {
			xr := x[t*cols : (t+1)*cols]
			var sum float64
			for c, wv := range wr {
				sum += float64(wv) * float64(xr[c])
			}
			out[t*rows+o] = sum
		}
	})
	return out
}

func parallelChunks(n int, fn func(i int)) {
	workers := runtime.NumCPU()
	if workers > n {
		workers = n
	}
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			for i := start; i < n; i += workers {
				fn(i)
			}
		}(w)
	}
	wg.Wait()
}

// relRMS32 is ||a - b|| / ||b|| over float32 data, accumulated in float64.
func relRMS32(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		num += d * d
		den += float64(b[i]) * float64(b[i])
	}
	if den == 0 {
		return 0
	}
	return math.Sqrt(num / den)
}

func norm64(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

func diffNorm64(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s)
}

func maxDiff64(a, b []float64) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

func cosine64(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / math.Sqrt(na*nb)
}

func max64[T float32 | float64](v []T) float64 {
	var m float64
	for _, x := range v {
		if a := math.Abs(float64(x)); a > m {
			m = a
		}
	}
	return m
}

// ---- reporting --------------------------------------------------------------

// PrintQuantError prints the per-site error table plus the two cross-site
// digests the format decision actually reads: the bits-vs-error front, and
// what the activation format costs.
func PrintQuantError(w io.Writer, rows []QuantErrorRow, outliers []QuantOutliers) {
	fmt.Fprintf(w, "\nL0d — quantization error on real weights and real activations\n")
	fmt.Fprintf(w, "relative RMS error of one projection's output against a float64 reference\n\n")

	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "SITE\tWEIGHT\tACT\tBITS/W\tW REL RMS\tOUT REL RMS\tMAX REL\tSNR dB\t1-COS")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%.3f\t%.3e\t%.3e\t%.3e\t%.1f\t%.2e\n",
			r.Site, r.Weight, r.Act, r.BitsPerW, r.WRelRMS, r.RelRMS, r.MaxRelErr, r.SNRdB, r.CosineLoss)
	}
	tw.Flush()

	printQuantPareto(w, rows)
	printQuantActCost(w, rows)
	printQuantOutliers(w, outliers)
}

// printQuantPareto is the front IDEAS §7 asked for in place of a speed
// ranking: bits per weight against error, averaged over sites, with the
// activations exact so it is the weight format alone.
func printQuantPareto(w io.Writer, rows []QuantErrorRow) {
	type agg struct {
		bits, sum float64
		n         int
		worst     float64
		wsum      float64
	}
	by := map[string]*agg{}
	var order []string
	for _, r := range rows {
		if r.Act != "exact" {
			continue
		}
		a, ok := by[r.Weight]
		if !ok {
			a = &agg{bits: r.BitsPerW}
			by[r.Weight] = a
			order = append(order, r.Weight)
		}
		a.sum += r.RelRMS
		a.wsum += r.WRelRMS
		a.n++
		if r.RelRMS > a.worst {
			a.worst = r.RelRMS
		}
	}
	sort.Slice(order, func(i, j int) bool { return by[order[i]].bits < by[order[j]].bits })
	fmt.Fprintf(w, "\n  weight format alone (activations exact), mean over sites, cheapest first\n")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "  BITS/W\tFORMAT\tMEAN OUT REL RMS\tWORST SITE\tMEAN WEIGHT REL RMS")
	for _, name := range order {
		a := by[name]
		fmt.Fprintf(tw, "  %.3f\t%s\t%.3e\t%.3e\t%.3e\n",
			a.bits, name, a.sum/float64(a.n), a.worst, a.wsum/float64(a.n))
	}
	tw.Flush()
	fmt.Fprint(w, "  equal-bits pairs are the comparison to read: q4sym/32 against q4asym/64 (both 4.5),\n"+
		"  and q4sym/64 against q4asym/128 (both 4.25)\n")
}

// printQuantActCost is the W4A8-vs-W4A16 question, which is the 3x one.
func printQuantActCost(w io.Writer, rows []QuantErrorRow) {
	sites := map[string]map[string]float64{}
	var order []string
	for _, r := range rows {
		if r.Weight != "q4sym/32" {
			continue
		}
		if _, ok := sites[r.Site]; !ok {
			sites[r.Site] = map[string]float64{}
			order = append(order, r.Site)
		}
		sites[r.Site][r.Act] = r.RelRMS
	}
	if len(order) == 0 {
		return
	}
	fmt.Fprintf(w, "\n  what the activation format costs, at q4sym/32 weights (rel RMS)\n")
	acts := []string{"exact", "fp16", "int8/token", "int8/tensor"}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprint(tw, "  SITE")
	for _, a := range acts {
		fmt.Fprintf(tw, "\t%s", a)
	}
	fmt.Fprint(tw, "\tint8tok/fp16\n")
	for _, s := range order {
		fmt.Fprintf(tw, "  %s", s)
		for _, a := range acts {
			fmt.Fprintf(tw, "\t%.3e", sites[s][a])
		}
		ratio := 0.0
		if sites[s]["fp16"] > 0 {
			ratio = sites[s]["int8/token"] / sites[s]["fp16"]
		}
		fmt.Fprintf(tw, "\t%.1fx\n", ratio)
	}
	tw.Flush()
	fmt.Fprint(w, "  int8/token is W4A8, the path that reaches the bus (§1.7); fp16 is W4A16, which\n"+
		"  §1.1 measured at 31% of the bus. The last column is what the 3x buys.\n")
}

func printQuantOutliers(w io.Writer, o []QuantOutliers) {
	if len(o) == 0 {
		return
	}
	fmt.Fprintf(w, "\n  activation outliers, which are why int8 might not be safe\n")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "  SITE\tMAX/RMS\tCHANNEL SPREAD\tKURTOSIS")
	for _, r := range o {
		fmt.Fprintf(tw, "  %s\t%.1f\t%.1f\t%.1f\n", r.Site, r.MaxOverRMS, r.ChannelSpread, r.Kurtosis)
	}
	tw.Flush()
	fmt.Fprint(w, "  CHANNEL SPREAD is the widest input channel over the median one: what a per-tensor\n"+
		"  scale must cover and a per-token one need not. Gaussian kurtosis is 3.\n")
}
