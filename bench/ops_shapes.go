package bench

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"strix-halo-vulkan/vk"
)

// This file is IDEAS §3.4: run the kernels the rest of the suite settled on
// against the shapes in modelshapes.go — the (M,N,K) triples the four models
// in GOALS.md are actually made of — instead of the square power-of-two
// sweep every other family uses.
//
// It is deliberately not another ablation. The tuning questions are
// answered elsewhere; what this family asks is which of those answers
// survive contact with real dimensions, and it asks it in three parts:
//
//  1. decode (M=1): the W4A8 GEMV at each load width, at every reduction
//     length the models decode at. §1.7's rule says the width that reaches
//     the DRAM bus is the one where a lane-step covers a whole weight row —
//     but that rule was only ever checked at powers of two, and no model has
//     a power-of-two reduction length in its FFN.
//  2. prefill/encode (M>1): the best WMMA GEMM variants over the real
//     rectangles, with M and N padded up to the tile and the padding charged
//     against the result, which is what an engine actually pays.
//  3. the accounting the first two feed: how many weight bytes and FLOPs one
//     forward pass of each model costs, and therefore what the measured
//     numbers imply for tokens (or frames, or images) per second.

const (
	// shapesBlock is the quantization block the decode arm runs at. It is
	// fixed rather than swept because this family is not a block-size study
	// and the block-size effect does not survive warming (see the README):
	// 128 is the smallest value the widest load (VEC=16, 128 weights per
	// lane-step) is allowed to use, so one block keeps all four widths
	// comparable.
	shapesBlock = 128

	// shapesTwinMB is the footprint of the DRAM-resident twin each decode
	// reduction length also gets. A real decode streams the whole model
	// through the cache once per token, so no weight matrix is ever
	// MALL-resident — but most single matrices are small enough that a
	// benchmark re-reading one measures the MALL. The twin keeps the row
	// geometry (which is what §1.7's rule is about) and raises only the row
	// count, until the weights are several times the 32 MB last-level cache.
	shapesTwinMB = 192

	// shapesMaxOperandFloats caps one prefill case's host-side operands.
	// Nothing in modelshapes.go comes close; it exists so that adding a
	// shape that does fails loudly rather than by thrashing.
	shapesMaxOperandFloats = 1 << 28
)

// shapesDecodeVariants names the W4A8 load widths the decode arm runs, in
// increasing width. Wave32 arms are deliberately absent: §1.7 established
// that the wave size is not itself a variable — hold the bytes of a row a
// lane-step covers fixed and wave32 and wave64 measure the same to 0.05% —
// so running both sizes here would double the arm to re-answer a settled
// question.
var shapesDecodeVariants = []string{
	"subgroup", "subgroup_vec4", "subgroup_vec8", "subgroup_vec16",
}

// shapesGEMMVariants names the prefill kernels: the three tile geometries
// that win somewhere in results/gemm_wmma.csv, the wave32 arm of the AI-16
// grid (which takes the N=1024 crown, and small-M shapes are what this
// family is full of), and the AI-85 tile for the large ones. All have
// BK <= 64, so K is never padded — only M and N are, which is the padding an
// engine actually does. A kernel whose K-slab did not divide the model's K
// would be computing a different product, not a padded one.
var shapesGEMMVariants = []string{
	"wmma_reg64_bt_hka4_padab128",
	"wmma_reg64_bt_hkab2_padab128",
	"wmma_reg32_bt_hkab4_padab128",
	"wmma_reg32_bt_hkab4_w32_padab128",
	"wmma_wg128x256_pada128",
}

// shapesMinIters is the floor on the timed batch this family uses, against
// the suite's default of 20.
//
// It is here because the shapes are small. The square sweep's cases run for
// hundreds of microseconds each; a 128-token GEMM runs for forty, so twenty
// of them is under a millisecond of timed work, and at that length a handful
// of cells per run came back one-sided low — 2x low, at a pinned 2814 MHz, so
// not a clock effect. Two runs at 20 iterations disagree by up to 105% on
// their worst cell; two at 200 disagree by at most 15%, with the median
// unchanged at 0.2%. TimeDispatch still caps the batch to its 500 ms budget,
// so this costs nothing on the cases that were already long enough.
const shapesMinIters = 200

// RunShapes measures the suite's chosen kernels at the models' own shapes.
func RunShapes(dev *vk.Device, phys *vk.PhysicalDevice, warmup, iters uint32) ([]Result, error) {
	if iters < shapesMinIters {
		iters = shapesMinIters
	}
	decode, err := runShapesDecode(dev, phys, warmup, iters)
	if err != nil {
		return nil, err
	}
	prefill, err := runShapesPrefill(dev, phys, warmup, iters)
	if err != nil {
		return nil, err
	}
	return append(decode, prefill...), nil
}

// decodeShape is one reduction length that some model decodes at, with the
// output widths (and the layers) that share it. The kernel's cost is set by
// the row geometry, so the reduction length is the unit of the sweep and the
// row count follows.
type decodeShape struct {
	K      int
	rows   []int    // distinct output widths at this reduction length
	labels []string // "model/layer" for each, for the Detail column
}

func decodeShapes() []decodeShape {
	byK := map[int]*decodeShape{}
	for _, s := range modelShapes {
		if s.M != 1 {
			continue
		}
		d, ok := byK[s.K]
		if !ok {
			d = &decodeShape{K: s.K}
			byK[s.K] = d
		}
		seen := false
		for _, r := range d.rows {
			if r == s.N {
				seen = true
			}
		}
		if !seen {
			d.rows = append(d.rows, s.N)
			d.labels = append(d.labels, s.model+"/"+s.layer)
		}
	}
	out := make([]decodeShape, 0, len(byK))
	for _, d := range byK {
		sort.Ints(d.rows)
		out = append(out, *d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].K < out[j].K })
	return out
}

func runShapesDecode(dev *vk.Device, phys *vk.PhysicalDevice, warmup, iters uint32) ([]Result, error) {
	feat, err := phys.SupportedFeatures()
	if err != nil {
		return nil, err
	}
	if !feat.IntegerDotProduct {
		fmt.Fprintln(os.Stderr, "shapes decode: shaderIntegerDotProduct not supported, skipping")
		return nil, nil
	}

	var results []Result
	for _, name := range shapesDecodeVariants {
		v, ok := lookupW4A8Variant(name)
		if !ok {
			return nil, fmt.Errorf("shapes: no W4A8 variant %q", name)
		}
		if !w4a8BlockOK(shapesBlock, v.weightsPerLoad) {
			return nil, fmt.Errorf("shapes: variant %q needs a block wider than %d", name, shapesBlock)
		}
		mod, err := dev.NewShaderModule(v.spirv)
		if err != nil {
			return nil, err
		}
		defer mod.Destroy()

		for _, d := range decodeShapes() {
			if d.K%shapesBlock != 0 || d.K%v.weightsPerLoad != 0 {
				continue
			}
			// The model's own matrices first: one case per distinct output
			// width at this reduction length.
			for i, rows := range d.rows {
				r, err := timeShapesDecode(dev, mod, v, rows, d.K, d.labels[i], warmup, iters)
				if err != nil {
					return nil, fmt.Errorf("shapes decode %s %s: %w", name, d.labels[i], err)
				}
				results = append(results, r)
			}
			// Then the DRAM-resident twin: the same row, enough of them that
			// the weights cannot sit in the MALL.
			twinRows := shapesTwinMB * 1024 * 1024 * 2 / d.K
			if twinRows*d.K%2 != 0 {
				twinRows++
			}
			r, err := timeShapesDecode(dev, mod, v, twinRows, d.K, "*/dram-twin", warmup, iters)
			if err != nil {
				return nil, fmt.Errorf("shapes decode %s twin K=%d: %w", name, d.K, err)
			}
			results = append(results, r)
		}
	}
	return results, nil
}

// timeShapesDecode runs one W4A8 GEMV over a [rows, K] weight matrix. Note
// the coldCase field names are the kernel's, not this file's: its M is the
// output row count and its N is the reduction length.
func timeShapesDecode(dev *vk.Device, mod *vk.ShaderModule, v w4a8Variant, rows, K int, label string, warmup, iters uint32) (Result, error) {
	m, err := timeColdGEMV(dev, mod, coldCase{
		format: "w4a8", variant: v.name, M: rows, N: K, block: shapesBlock,
		weightBytes: rows * K / 2,
		fillWeights: fillNibblePattern,
		kind:        coldW4A8,
		waveSize:    v.waveSize,
		rowsPerWG:   v.rowsPerWG,
	}, warmup, iters)
	if err != nil {
		return Result{}, err
	}

	// C is the contiguous run of one weight row a lane-step holds, the
	// quantity §1.7 found predicts the bandwidth; the row stride is the
	// weight row itself, four bits per weight. Recording both next to the
	// measurement is what makes the coverage law checkable per row rather
	// than only in the summary.
	wave := 64
	if v.waveSize != 0 {
		wave = int(v.waveSize)
	}
	lanes := K / (8 * (v.weightsPerLoad / 8))
	if lanes > wave {
		lanes = wave
	}
	inFlight := lanes * v.weightsPerLoad / 2
	rowBytes := K / 2

	return Result{
		Op: "decode", Variant: v.name, WeightFormat: "w4a8", BlockSize: shapesBlock, Size: K,
		Detail: fmt.Sprintf("shape=%s;rows=%d;K=%d;weightMB=%.1f;C=%dB;rowB=%dB;cover=%.2f",
			label, rows, K, m.bytesRead/(1024*1024), inFlight, rowBytes,
			strideChannelCoverage(inFlight, rowBytes)),
		NsPerIter: m.ns,
		GFLOPS:    float64(2*rows*K) / (m.ns / 1e9) / 1e9,
		GBPS:      m.bytesRead / (m.ns / 1e9) / 1e9,
		Clocks:    m.clocks,
	}, nil
}

func runShapesPrefill(dev *vk.Device, phys *vk.PhysicalDevice, warmup, iters uint32) ([]Result, error) {
	shape, ok, err := findCoopMatShape(phys, vk.ComponentFloat16, vk.ComponentFloat32)
	if err != nil {
		return nil, err
	}
	if !ok || shape.M != 16 || shape.N != 16 || shape.K != 16 {
		fmt.Fprintln(os.Stderr, "shapes prefill: no 16x16x16 cooperative-matrix shape, skipping")
		return nil, nil
	}

	var variants []wmmaVariant
	for _, name := range shapesGEMMVariants {
		v, ok := lookupWMMAVariant(name)
		if !ok {
			return nil, fmt.Errorf("shapes: no WMMA variant %q", name)
		}
		variants = append(variants, v)
	}
	variants = filterWaveVariants(variants, mustSubgroupSizeControl(phys),
		func(v wmmaVariant) (string, uint32) { return "shapes " + v.name, v.waveSize })

	mods := make([]*vk.ShaderModule, len(variants))
	for i, v := range variants {
		if mods[i], err = dev.NewShaderModule(v.spirv); err != nil {
			return nil, err
		}
		defer mods[i].Destroy()
		// One correctness check per variant, at the small clean shape
		// verifyGEMMWMMA picks. The shapes below are padded and unverified by
		// design (a host reference over 4096x10240x3840 is minutes of CPU),
		// so this is what stands between a mistyped tile and a plausible
		// number.
		if err := verifyGEMMWMMA(dev, mods[i], v); err != nil {
			return nil, fmt.Errorf("shapes %s correctness check: %w", v.name, err)
		}
	}

	var results []Result
	for _, s := range prefillShapes() {
		// A and B are generated once per (padded extent) and shared by every
		// variant that pads to it, because generating and converting tens of
		// millions of floats costs more than the dispatch does.
		type operands struct{ a, b []float32 }
		cache := map[string]operands{}

		for i, v := range variants {
			if s.K%v.bk != 0 {
				// Padding K would change the product, not just its shape.
				continue
			}
			pm, pn := roundUpTo(s.M, v.bm), roundUpTo(s.N, v.bn)
			if float64(pm)*float64(s.K) > shapesMaxOperandFloats || float64(s.K)*float64(pn) > shapesMaxOperandFloats {
				fmt.Fprintf(os.Stderr, "shapes %s %s: operands too large, skipping\n", v.name, s.label())
				continue
			}
			key := fmt.Sprintf("%dx%d", pm, pn)
			ops, ok := cache[key]
			if !ok {
				ops = operands{a: randomFloats(pm * s.K), b: randomFloats(s.K * pn)}
				cache[key] = ops
			}
			ns, clocks, err := runWMMACase(dev, mods[i], v, pm, pn, s.K, ops.a, ops.b, warmup, iters)
			if err != nil {
				return nil, fmt.Errorf("shapes %s %s: %w", v.name, s.label(), err)
			}

			useful := float64(2 * s.M * s.N * s.K)
			executed := float64(2 * pm * pn * s.K)
			results = append(results, Result{
				Op: "prefill", Variant: v.name, WeightFormat: "fp16", Size: s.M,
				Detail: fmt.Sprintf("shape=%s;M=%d;N=%d;K=%d;pad=%dx%d;useful=%.0f%%;exec_gflops=%.0f;wave=%d",
					s.label(), s.M, s.N, s.K, pm, pn, 100*useful/executed,
					executed/(ns/1e9)/1e9, waveOrDefault(v.waveSize)),
				NsPerIter: ns,
				// The reported rate is the *useful* one: the FLOPs the model
				// needs, over the time the padded dispatch took. That is what
				// an engine gets, and it is the only way a shape whose M is
				// 40 rows in a 64-row tile reads as the problem it is. The
				// rate the kernel itself achieved is in exec_gflops.
				GFLOPS: useful / (ns / 1e9) / 1e9,
				Clocks: clocks,
			})
		}
	}
	return results, nil
}

// prefillShape is one deduplicated (M,N,K) triple with the model layers that
// share it. Several models use the same rectangle — every 2560-wide Qwen
// block does — and measuring it once per model would be the same dispatch
// three times.
type prefillShape struct {
	M, N, K int
	labels  []string
}

func (s prefillShape) label() string { return strings.Join(s.labels, "+") }

func prefillShapes() []prefillShape {
	var order []string
	byKey := map[string]*prefillShape{}
	for _, s := range modelShapes {
		if s.M == 1 {
			continue
		}
		key := fmt.Sprintf("%d/%d/%d", s.M, s.N, s.K)
		p, ok := byKey[key]
		if !ok {
			p = &prefillShape{M: s.M, N: s.N, K: s.K}
			byKey[key] = p
			order = append(order, key)
		}
		label := s.model + "/" + s.layer
		dup := false
		for _, l := range p.labels {
			if l == label {
				dup = true
			}
		}
		if !dup {
			p.labels = append(p.labels, label)
		}
	}
	out := make([]prefillShape, 0, len(order))
	for _, k := range order {
		out = append(out, *byKey[k])
	}
	return out
}

func lookupW4A8Variant(name string) (w4a8Variant, bool) {
	for _, v := range w4a8Variants {
		if v.name == name {
			return v, true
		}
	}
	return w4a8Variant{}, false
}

func lookupWMMAVariant(name string) (wmmaVariant, bool) {
	for _, v := range wmmaVariants {
		if v.name == name {
			return v, true
		}
	}
	return wmmaVariant{}, false
}

func roundUpTo(v, m int) int { return (v + m - 1) / m * m }

// PrintShapesSummary is where this family earns its place: the flat table is
// one row per (kernel, shape), and the reading that matters is per model.
func PrintShapesSummary(w io.Writer, results []Result, p Params) {
	printDecodeWidthGrid(w, results)
	printPrefillWinners(w, results)
	printModelBudget(w, results)
}

// printDecodeWidthGrid checks §1.7's load-width rule at the models' own
// reduction lengths. The rule was derived and confirmed at powers of two;
// every FFN reduction length in GOALS.md's models is not one, and a
// non-power-of-two row stride has a *smaller* gcd with the 4 KB channel
// rotation — so the prediction is that the real shapes need narrower loads
// than the square sweep did, not wider.
func printDecodeWidthGrid(w io.Writer, results []Result) {
	rows := map[int]map[string]float64{} // K -> variant -> GB/s, DRAM twin only
	var ks []int
	for _, r := range results {
		if r.Op != "decode" || detailField(r.Detail, "shape") != "*/dram-twin" {
			continue
		}
		if _, ok := rows[r.Size]; !ok {
			rows[r.Size] = map[string]float64{}
			ks = append(ks, r.Size)
		}
		rows[r.Size][r.Variant] = r.GBPS
	}
	if len(ks) == 0 {
		return
	}
	sort.Ints(ks)

	fmt.Fprintf(w, "\ndecode load width vs reduction length (DRAM-resident W4A8, GB/s of %.0f):\n", dramPeakGBPS)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprint(tw, "K\trowB\tgcd4K\tneed")
	for _, v := range shapesDecodeVariants {
		fmt.Fprintf(tw, "\t%s", widthLabel(v))
	}
	fmt.Fprintln(tw, "\tmodels")
	for _, k := range ks {
		rowBytes := k / 2
		fmt.Fprintf(tw, "%d\t%d\t%d\t%s", k, rowBytes, gcdInt(rowBytes, 4096), neededWidth(k))
		for _, name := range shapesDecodeVariants {
			if gbps, ok := rows[k][name]; ok {
				fmt.Fprintf(tw, "\t%.0f", gbps)
			} else {
				fmt.Fprint(tw, "\t-")
			}
		}
		fmt.Fprintf(tw, "\t%s\n", decodeUsers(k))
	}
	tw.Flush()
	fmt.Fprintln(w, "  need: the narrowest built load width whose lane-step covers gcd(rowB, 4096) bytes (IDEAS §1.7)")
}

// widthLabel names a W4A8 load-width variant by its VEC rather than by its
// kernel name, since the width is the only thing that varies along that axis.
func widthLabel(variant string) string {
	if variant == "subgroup" {
		return "vec1"
	}
	return strings.TrimPrefix(variant, "subgroup_")
}

func waveOrDefault(size uint32) int {
	if size == 0 {
		return 64
	}
	return int(size)
}

// neededWidth is §1.7's rule evaluated for one reduction length: the
// narrowest of the built load widths whose lane-step covers a full
// gcd(rowBytes, 4096) run, which is the coverage law's threshold.
func neededWidth(K int) string {
	rowBytes := K / 2
	g := gcdInt(rowBytes, 4096)
	for _, name := range shapesDecodeVariants {
		v, ok := lookupW4A8Variant(name)
		if !ok || K%v.weightsPerLoad != 0 {
			continue
		}
		lanes := K / v.weightsPerLoad
		if lanes > 64 {
			lanes = 64
		}
		if lanes*v.weightsPerLoad/2 >= g {
			return widthLabel(name)
		}
	}
	return ">vec16"
}

func decodeUsers(K int) string {
	var out []string
	for _, s := range modelShapes {
		if s.M != 1 || s.K != K {
			continue
		}
		l := s.model + "/" + s.layer
		dup := false
		for _, o := range out {
			if o == l {
				dup = true
			}
		}
		if !dup {
			out = append(out, l)
		}
	}
	if len(out) > 3 {
		out = append(out[:3], fmt.Sprintf("+%d", len(out)-3))
	}
	return strings.Join(out, " ")
}

// printPrefillWinners gives one line per real rectangle: which kernel wins it
// and what fraction of the dispatch was padding. The square sweep cannot show
// either, because every shape in it tiles exactly and every shape in it is
// large.
func printPrefillWinners(w io.Writer, results []Result) {
	type best struct {
		variant string
		gflops  float64
		exec    float64
		useful  string
		detail  string
	}
	var order []string
	bests := map[string]best{}
	for _, r := range results {
		if r.Op != "prefill" {
			continue
		}
		key := detailField(r.Detail, "M") + "/" + detailField(r.Detail, "N") + "/" + detailField(r.Detail, "K")
		b, ok := bests[key]
		if !ok {
			order = append(order, key)
		}
		if !ok || r.GFLOPS > b.gflops {
			exec, _ := strconv.ParseFloat(detailField(r.Detail, "exec_gflops"), 64)
			bests[key] = best{
				variant: r.Variant, gflops: r.GFLOPS, exec: exec,
				useful: detailField(r.Detail, "useful"), detail: r.Detail,
			}
		}
	}
	if len(order) == 0 {
		return
	}
	fmt.Fprintln(w, "\nprefill: best kernel per real rectangle")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "M\tN\tK\tBEST\tUSEFUL GFLOP/S\tKERNEL GFLOP/S\tUSEFUL\tSHAPE")
	for _, k := range order {
		b := bests[k]
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%.0f\t%.0f\t%s\t%s\n",
			detailField(b.detail, "M"), detailField(b.detail, "N"), detailField(b.detail, "K"),
			b.variant, b.gflops, b.exec, b.useful, detailField(b.detail, "shape"))
	}
	tw.Flush()
	fmt.Fprintln(w, "  useful < 100% is tile padding: the kernel ran a bigger GEMM than the model needed")
}

// printModelBudget adds the shapes up. This is the number GOALS.md is
// ultimately asking for — what one token, one second of audio, one image
// costs — and it is only computable once the shapes are written down.
func printModelBudget(w io.Writer, results []Result) {
	// Best measured decode bandwidth over the DRAM-resident twins, which is
	// the rate a streamed model reads its weights at.
	var decodeGBPS float64
	for _, r := range results {
		if r.Op == "decode" && detailField(r.Detail, "shape") == "*/dram-twin" && r.GBPS > decodeGBPS {
			decodeGBPS = r.GBPS
		}
	}

	type agg struct {
		q4Bytes, flops float64
		memBytes       float64 // weight bytes in matrices whose own intensity is below the crossover
		dispatches     int
	}
	var order []string
	byModel := map[string]*agg{}
	for _, s := range modelShapes {
		phase := "prefill"
		if s.M == 1 {
			phase = "decode"
		}
		key := s.model + " " + phase
		a, ok := byModel[key]
		if !ok {
			a = &agg{}
			byModel[key] = a
			order = append(order, key)
		}
		bytes := float64(s.count) * float64(s.N) * float64(s.K) / 2
		a.q4Bytes += bytes
		a.flops += float64(s.count) * 2 * float64(s.M) * float64(s.N) * float64(s.K)
		a.dispatches += s.count
		// One matrix's own arithmetic intensity against 4-bit weights is
		// 2*M*N*K flops over N*K/2 bytes, i.e. 4*M — it depends on the batch
		// and on nothing else. Aggregating the ratio hides that: a prefill
		// pass can be compute-bound overall while every expert in it is
		// memory-bound, which is exactly what the MoE rows do.
		if 4*float64(s.M) < crossoverFlopPerByte {
			a.memBytes += bytes
		}
	}

	fmt.Fprintln(w, "\nwhat one forward pass costs, added up over every matrix above")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "MODEL/PHASE\tWEIGHTS Q4\tGFLOP\tFLOP/BYTE\tMEM-BOUND\tDISPATCHES")
	for _, k := range order {
		a := byModel[k]
		fmt.Fprintf(tw, "%s\t%.2f GB\t%.1f\t%.1f\t%.0f%%\t%d\n",
			k, a.q4Bytes/1e9, a.flops/1e9, a.flops/a.q4Bytes, 100*a.memBytes/a.q4Bytes, a.dispatches)
	}
	tw.Flush()
	fmt.Fprintf(w, "  flop/byte is against 4-bit weights, where one matrix's own intensity is just 4*M;\n"+
		"  mem-bound is the share of the weight bytes below the %.0f flop/byte crossover of this\n"+
		"  part's 55.5 TFLOP/s matrix rate and 236 GB/s of DRAM (IDEAS §2.1)\n", crossoverFlopPerByte)
	if decodeGBPS > 0 {
		if a, ok := byModel["qwen3.8-flash-next decode"]; ok {
			ms := a.q4Bytes / (decodeGBPS * 1e9) * 1e3
			// IDEAS §4.1 measured a dispatch plus its barrier at ~300 ns.
			// Decode is the one phase with enough of them for that to be
			// worth putting next to the bandwidth figure.
			overheadMs := float64(a.dispatches) * dispatchNs / 1e6
			fmt.Fprintf(w, "  at the best measured DRAM-resident decode rate (%.0f GB/s), qwen3.8-flash-next\n"+
				"  reads %.2f GB per token = %.1f ms = %.0f tok/s before attention;\n"+
				"  its %d matmul dispatches add %.2f ms of launch cost on top (%.0f%%, IDEAS §4.1)\n",
				decodeGBPS, a.q4Bytes/1e9, ms, 1000/ms, a.dispatches, overheadMs, 100*overheadMs/ms)
		}
	}
}

// crossoverFlopPerByte is where this part's measured 55.5 TFLOP/s matrix
// rate and its 236 GB/s of DRAM meet: below it a matmul is memory-bound
// however well it is tiled.
const crossoverFlopPerByte = 235.0

// dispatchNs is §4.1's measured cost of one dispatch plus the barrier after
// it, which decode pays once per matrix per layer.
const dispatchNs = 300.0

// dramPeakGBPS is the measured DRAM read ceiling this chip's decode path is
// scored against (results/bandwidth.csv, IDEAS §0.3).
const dramPeakGBPS = 236.0

// detailField reads one "key=value" out of a Result's Detail, which is where
// this family's extra dimensions live because Result has one Size and these
// shapes have three.
func detailField(detail, key string) string {
	for _, part := range strings.Split(detail, ";") {
		if v, ok := strings.CutPrefix(part, key+"="); ok {
			return v
		}
	}
	return ""
}
