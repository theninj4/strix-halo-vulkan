package llm

// Simulating a bank width before there is a kernel for it: LLM.md L8c.
//
// L8c has two halves that are usually done together and do not have to be.
// One is **which widths** — 4 bits or 5, a scale per 32 elements or per 128,
// which families stay where they are — and the other is **the kernel and the
// layout** that make a narrow bank fast. The second is weeks of work whose
// only purpose is throughput; the first is a question about the model, and
// L8c-0 just built the instrument that answers it.
//
// So a width is graded first, by staging the weights a candidate format would
// produce and running them through the kernels that already exist. A dense
// weight is dequantised on its way to the device anyway (`Model.F32`), so the
// simulation is one round trip inserted there:
//
//	w -> (d, q) at the candidate width -> fp16(float(q) * float(d))
//
// **The result is not an approximation of the candidate bank; it is the
// candidate bank's own numbers.** The last step is the point. L8b-2 settled
// that a kernel reading a quantised tile multiplies in fp16 — `float16_t(q) *
// d` is one f16 instruction — so the half a real Q4 kernel would form is the
// correctly-rounded fp16 product of a small integer and an fp16 scale. An f32
// product of a 4-bit level and an 11-bit significand is exact, and rounding
// that to fp16 is the same correctly-rounded half. So the simulation stages
// exactly the values the kernel would multiply, and a perplexity measured on
// it is the format's perplexity.
//
// The one thing it cannot say is what the format *costs*, because it stages
// halves: residency and tok/s under a simulation are the fp16 arm's, not the
// candidate's. Bytes are arithmetic (`cmd/gguf`'s decode budget); accuracy is
// not, which is why this is the half that gets an instrument.
//
// The one place that reasoning is spelled differently is L8c-3's asymmetric
// arm. A K-quant group is affine — `d*sc*l - dmin*m` — and the subtraction is
// done in f32 by the one kernel here that already reads the format
// (`llm_moe_gemv.comp`, over the checkpoint's own experts), so that arm
// stages the f32 the kernel forms and lets the bank's own staging round it.
// The half is the same either way; what changes is which step does it.
//
//	LLM_DENSE_SIM=q4sym/32 go run ./cmd/llm -ppl
//	LLM_DENSE_SIM=q4_0/32  go run ./cmd/llm -gen -n 64
//	LLM_DENSE_SIM=q4_k/32  go run ./cmd/llm -ppl   # L8c-3, the same 4.5 bits
//	LLM_DENSE_SIM=off      # the default: the checkpoint's own widths

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	"strix-halo-vulkan/safetensors"
)

// QuantSim is a candidate weight format: a symmetric level count and the run
// of contiguous elements along a row that shares one fp16 scale.
//
// Zero value is off, and off is an identity — not "fp16 round trip", not
// "q8", nothing. A run with no `LLM_DENSE_SIM` stages what L8a and L8b stage.
type QuantSim struct {
	// Name is the spec as written, for a header line that makes a number
	// attributable.
	Name string
	// Bits is the level count's width: 4 means 15 or 16 levels depending on
	// Ggml below. Zero is off.
	Bits int
	// Group is the elements per scale, along a row — which is along k, since
	// Model.F32 returns a weight as out rows of in values.
	Group int
	// Mode is how the levels and the scale are chosen, and it is the axis
	// L8c-2 is about:
	//
	//	rtn      round to nearest against d = -maxval/2^(b-1), which is
	//	         ggml's own uncalibrated path (quantize_row_q4_0_ref) and
	//	         what every L8c-1 rung was measured with;
	//	search   ggml's make_qx_quants with rmse_type 1 — the same levels,
	//	         then a least-squares scale and a 19-rung search over the
	//	         initial one, weighted by sqrt(sigma2 + x^2);
	//	imatrix  the same search with the published importance matrix
	//	         folded into that weight, which is exactly
	//	         quantize_row_q4_0_impl.
	//
	// Keeping `search` separate from `imatrix` is the whole point: they are
	// two different things and ggml does them in one function, so without the
	// middle arm an improvement cannot be attributed to calibration.
	Mode string
	// Ggml picks the scale convention. False is L0d's `q4sym`, d = amax/7
	// with q clamped to [-8, 7], which is what every number in
	// research/l0d-quant-error.md was measured with — 15 reachable levels,
	// because amax/7 * -8 is below -amax. True is ggml's Q4_0, d = -maxval/8
	// where maxval is the signed element of largest magnitude, which reaches
	// all 16. They differ by one level out of fifteen and L8c-1 prices it.
	Ggml bool
	// Asym is ggml's K-quant form rather than a symmetric one, and it is the
	// axis L8c-3 is about — D7, reopened.
	//
	// A symmetric group has one number: a scale, and the levels straddle
	// zero. An asymmetric one has two — a scale and a min, with the levels
	// unsigned in [0, 2^b - 1] — and ggml does not spend an fp16 on each of
	// them per group of 32. It nests: a **super-block** of Sub groups carries
	// one fp16 `d` and one fp16 `dmin`, and each group carries a 6-bit scale
	// and a 6-bit min quantised against them. That is 12 bits a group plus 32
	// a super-block, which at eight groups of 32 is 0.5 bits a weight —
	// **exactly what a symmetric fp16 scale per 32 costs**, so `q4_k/32` and
	// `q4_0/32` are 4.5 bits apiece and the comparison is free of a width
	// argument. It is the form this checkpoint's own experts are in, and the
	// form unsloth's imatrix was collected against.
	Asym bool
}

// Off reports whether this simulation does nothing.
func (q QuantSim) Off() bool { return q.Bits == 0 }

// String is the spec, or "off".
func (q QuantSim) String() string {
	if q.Off() {
		return "off"
	}
	return q.Name
}

// BitsPerWeight is what the format would cost on disk and on the bus: the
// levels plus the fp16 scale spread over the group. It is the number that
// makes an accuracy result comparable to D3's ~4.25 target, and it is
// arithmetic rather than a measurement.
func (q QuantSim) BitsPerWeight() float64 {
	if q.Off() {
		return 0
	}
	if q.Asym {
		return q.bitsPerWeightSuper(asymSuperBlocks)
	}
	return float64(q.Bits) + 16/float64(q.Group)
}

// bitsPerWeightSuper is the asymmetric cost at a given super-block: the
// levels, plus 12 bits a group for the 6-bit scale and min, plus the 32 bits
// of the fp16 pair spread over the whole super-block.
//
// It takes the super-block because `Apply` picks it from the row (see
// superBlocks): eight groups is ggml's QK_K and what every number below is
// quoted at, but a 320-wide row has ten and costs 4.475 rather than 4.500.
func (q QuantSim) bitsPerWeightSuper(sub int) float64 {
	return float64(q.Bits) + (12*float64(sub)+16*2)/float64(sub*q.Group)
}

// ParseQuantSim reads a spec: "off", or "<format>/<group>" where format is
// q2sym…q8sym, q4_0, or q4_k / q5_k for L8c-3's asymmetric arm.
func ParseQuantSim(s string) (QuantSim, error) {
	s = strings.TrimSpace(s)
	if s == "" || s == "off" {
		return QuantSim{}, nil
	}
	name, group, ok := strings.Cut(s, "/")
	if !ok {
		return QuantSim{}, fmt.Errorf("llm: quant sim %q is not <format>/<group>", s)
	}
	g, err := strconv.Atoi(group)
	if err != nil || g < 8 || g%8 != 0 {
		return QuantSim{}, fmt.Errorf("llm: quant sim group %q is not a multiple of 8 at least 8", group)
	}
	q := QuantSim{Name: s, Group: g}
	switch {
	case name == "q4_0":
		q.Bits, q.Ggml = 4, true
	case name == "q4_k" || name == "q5_k":
		// Only the two ggml has. Q6_K is not this format — it is symmetric
		// with an 8-bit scale per sixteen — and Q2_K/Q3_K are below D4's
		// floor, so a `q6_k` here would be a name for something that does
		// not exist rather than a rung.
		q.Bits, q.Asym = int(name[1]-'0'), true
		if g != 32 {
			// ggml's sub-block is 32 and the two constants that steer the
			// search (`rmin`, `rdelta`) were fitted at that length. A
			// different group would be a different format wearing the name
			// of one whose numbers are published, which is the thing this
			// whole file exists not to do.
			return QuantSim{}, fmt.Errorf("llm: %s is defined on groups of 32, not %d", name, g)
		}
	case strings.HasPrefix(name, "q") && strings.HasSuffix(name, "sym"):
		b, err := strconv.Atoi(name[1 : len(name)-3])
		if err != nil || b < 2 || b > 8 {
			return QuantSim{}, fmt.Errorf("llm: quant sim %q: bits must be 2-8", name)
		}
		q.Bits = b
	default:
		return QuantSim{}, fmt.Errorf("llm: unknown quant sim format %q", name)
	}
	return q, nil
}

var (
	denseSimOnce sync.Once
	densePlan    QuantPlan
)

// QuantPlan is a width per streamed dense family — L8c's actual deliverable,
// because the families do not want the same width.
//
// The per-family screen is why it is a plan and not a number: at 4.5 bits the
// hyper-connection block costs 4x the perplexity the gated DeltaNet does, and
// the DeltaNet is 46% of a dense token where the block is 14%. A single width
// over all six is the one arrangement guaranteed to be wrong at both ends.
type QuantPlan struct {
	// Name is the spec as written, for a header that makes a number
	// attributable.
	Name string
	// Src restricts the plan by the width the *checkpoint* ships: "q8" is
	// what L8a stages as int8, "tail" the four families it could not, "" is
	// both. It is L8a-2's split, and the axis a question about the fp16 tail
	// has to be asked along.
	Src string
	// Mode is how every format in the plan chooses its levels and scale:
	// rtn, search or imatrix (QuantSim.Mode). It is plan-wide because it is
	// the axis L8c-2 sweeps, not a property of a family.
	Mode string

	byFamily map[string]QuantSim
}

// Off reports whether this plan does nothing at all.
func (p QuantPlan) Off() bool { return len(p.byFamily) == 0 }

// String is the spec, or "off".
func (p QuantPlan) String() string {
	if p.Off() {
		return "off"
	}
	return p.Name
}

// For returns the format to apply to a tensor, if any.
//
// q8Source is whether the checkpoint ships it as Q8_0 — which is what makes
// L8a's int8 bank an identity for it (D13) — and is what `Src` selects on.
func (p QuantPlan) For(name string, q8Source bool) (QuantSim, bool) {
	if p.Off() {
		return QuantSim{}, false
	}
	fam := simFamily(name)
	if fam == "" {
		return QuantSim{}, false
	}
	q, ok := p.byFamily[fam]
	if !ok {
		return QuantSim{}, false
	}
	switch p.Src {
	case "q8":
		if !q8Source {
			return QuantSim{}, false
		}
	case "tail":
		if q8Source {
			return QuantSim{}, false
		}
	}
	return q, true
}

// Widths lists the plan, one family per row, sorted — for a header line and
// for a write-up's table.
func (p QuantPlan) Widths() []string {
	out := make([]string, 0, len(p.byFamily))
	for _, f := range SimFamilies() {
		if q, ok := p.byFamily[f]; ok {
			out = append(out, fmt.Sprintf("%s=%s (%.3f bits)", f, q, q.BitsPerWeight()))
		}
	}
	return out
}

// ParseQuantPlan reads LLM_DENSE_SIM. Two grammars, because the two questions
// are different shapes:
//
//	q4_0/32                             one width, every streamed dense family
//	deltanet=q4_0/32,hyper_conn=q6sym/32   a width per family; the rest stay
//
// The second is the one a real bank is described by. Families are the names
// SimFamilies returns, which are `cmd/gguf`'s group names, so an accuracy row
// and a byte row name the same thing.
func ParseQuantPlan(spec, families, src, mode string) (QuantPlan, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || spec == "off" {
		return QuantPlan{}, nil
	}
	p := QuantPlan{Name: spec, byFamily: map[string]QuantSim{}}
	if strings.Contains(spec, "=") {
		if strings.TrimSpace(families) != "" {
			return QuantPlan{}, fmt.Errorf("llm: a per-family spec and LLM_DENSE_SIM_FAMILIES are two ways to say the same thing; use one")
		}
		for _, part := range strings.Split(spec, ",") {
			fam, f, ok := strings.Cut(strings.TrimSpace(part), "=")
			if !ok {
				return QuantPlan{}, fmt.Errorf("llm: %q is not family=format/group", part)
			}
			if !simFamilies[fam] {
				return QuantPlan{}, fmt.Errorf("llm: unknown dense family %q (have %v)", fam, SimFamilies())
			}
			q, err := ParseQuantSim(f)
			if err != nil {
				return QuantPlan{}, err
			}
			if q.Off() {
				continue
			}
			p.byFamily[fam] = q
		}
	} else {
		q, err := ParseQuantSim(spec)
		if err != nil {
			return QuantPlan{}, err
		}
		want := simFamilies
		if fams := strings.TrimSpace(families); fams != "" {
			want = map[string]bool{}
			for _, f := range strings.Split(fams, ",") {
				f = strings.TrimSpace(f)
				if !simFamilies[f] {
					return QuantPlan{}, fmt.Errorf("llm: unknown dense family %q (have %v)", f, SimFamilies())
				}
				want[f] = true
			}
			p.Name += ":" + fams
		}
		for f := range want {
			p.byFamily[f] = q
		}
	}
	switch mode = strings.TrimSpace(mode); mode {
	case "", "rtn":
		mode = "rtn"
	case "search", "imatrix", "search+gain", "imatrix+gain":
		p.Name += "+" + mode
	default:
		return QuantPlan{}, fmt.Errorf("llm: LLM_DENSE_SIM_QUANT is %q, want rtn, search, imatrix or either +gain", mode)
	}
	p.Mode = mode
	for f, q := range p.byFamily {
		q.Mode = mode
		p.byFamily[f] = q
	}
	if src = strings.TrimSpace(src); src != "" && src != "all" {
		if src != "q8" && src != "tail" {
			return QuantPlan{}, fmt.Errorf("llm: LLM_DENSE_SIM_SRC is %q, want q8, tail or all", src)
		}
		p.Src = src
		p.Name += ":" + src
	}
	return p, nil
}

// DensePlan is the simulation this process runs, from LLM_DENSE_SIM plus the
// two scope variables. It is read once, and a bad spec is fatal rather than
// ignored: a run that silently measured the wrong bank would be worse than
// one that did not start.
//
//	LLM_DENSE_SIM=q4_0/32                        every streamed dense family
//	LLM_DENSE_SIM=deltanet=q4_0/32,lm_head=q5sym/32   a width per family
//	LLM_DENSE_SIM_FAMILIES=deltanet,lm_head      restrict a single-width spec
//	LLM_DENSE_SIM_SRC=tail                       only what the checkpoint
//	                                             does not ship as Q8_0
func DensePlan() QuantPlan {
	denseSimOnce.Do(func() {
		p, err := ParseQuantPlan(os.Getenv("LLM_DENSE_SIM"),
			os.Getenv("LLM_DENSE_SIM_FAMILIES"), os.Getenv("LLM_DENSE_SIM_SRC"),
			os.Getenv("LLM_DENSE_SIM_QUANT"))
		if err != nil {
			panic(err)
		}
		densePlan = p
	})
	return densePlan
}

var (
	denseBankOnce sync.Once
	denseBankPlan QuantPlan
)

// DenseBankPlan is which dense families are staged on **the real 4.5-bit
// bank** (bank_q4.go), from `LLM_DENSE_BANK` — the same grammar `LLM_DENSE_SIM`
// takes, and `LLM_DENSE_BANK_QUANT` beside it, because a plan is a plan
// whether it is simulated or built.
//
//	LLM_DENSE_BANK=lm_head=q4_k/32              the family L8c-4 has a kernel for
//	LLM_DENSE_BANK_QUANT=imatrix                the default; rtn is the control
//
// It defaults to `imatrix` where `LLM_DENSE_SIM` defaults to `rtn`, and that
// is deliberate: a simulation's default arm is the one every L8c-1 rung was
// measured with, where a *bank*'s is the one L8c-3 recommends building
// (+4.24% against the calibrated symmetric form's +15.63%). A family the
// matrix does not cover falls back to round-to-nearest the way ggml does.
//
// The two are separate variables and a run that sets both is measuring
// something incoherent — a weight quantised twice — so `-ppl` and the graph
// refuse it rather than average it.
func DenseBankPlan() QuantPlan {
	denseBankOnce.Do(func() {
		mode := os.Getenv("LLM_DENSE_BANK_QUANT")
		if strings.TrimSpace(mode) == "" {
			mode = "imatrix"
		}
		p, err := ParseQuantPlan(os.Getenv("LLM_DENSE_BANK"), "", "", mode)
		if err != nil {
			panic(err)
		}
		if !p.Off() && !DensePlan().Off() {
			panic(fmt.Errorf("llm: LLM_DENSE_BANK=%s and LLM_DENSE_SIM=%s would quantise the same weight twice",
				p, DensePlan()))
		}
		denseBankPlan = p
	})
	return denseBankPlan
}

// ApplyTo round-trips one staged weight through the plan, if the plan covers
// it, and records what it touched.
//
// It is the single entry point both staging paths use — `Model.F32` for
// everything and `gpu_head.go` for the lm head, which is too big to reach the
// device as floats — because the two behaving differently is exactly the bug
// L8c-1 found. x is [n][k] row-major, which is ggml's own order for a weight
// stated [k, n].
func (p QuantPlan) ApplyTo(name string, q8Source bool, x []float32, k int) error {
	q, ok := p.For(name, q8Source)
	if !ok {
		return nil
	}
	var qw []float32
	if q.Mode == "imatrix" {
		im, err := DefaultImatrix()
		if err != nil {
			return err
		}
		if qw, err = im.Columns(name); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		if qw == nil {
			// **The published matrix has no entry for this tensor**, which
			// is not a failure and is not rare: unsloth's covers `blk.N.*`
			// only, so `output.weight` and the head mixer's two projections
			// — 0.636 B weights, 14% of a dense token — have no calibration
			// data at all.
			//
			// ggml does exactly this: `quantize_row_q4_0_impl` returns
			// `quantize_row_q4_0_ref` when `quant_weights` is null, so an
			// uncovered tensor is round-to-nearest. Following it keeps the
			// simulation's arms comparable to a real `llama-quantize`, and
			// the tally below is what stops that being invisible — a plan
			// that is half calibrated has to say so.
			q.Mode = "rtn"
			simCountUncal(name, len(x))
		}
	}
	if err := q.ApplyWeighted(x, k, qw); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	simCount(name, len(x))
	return nil
}

// imatrixCols is the published importance row for a tensor, or nil — which is
// both "this mode does not calibrate" and "the matrix has no entry for this
// tensor", because ggml treats the second as the first
// (`quantize_row_q4_K_impl` falls back to `_ref` on a null `quant_weights`)
// and so does everything here.
//
// It is `ApplyTo`'s lookup as a function, because L8c-4's *bank* needs the
// same row the *simulation* used and the two reading different matrices would
// be the bug this file was written to avoid.
func imatrixCols(name, mode string) ([]float32, error) {
	if mode != "imatrix" && mode != "imatrix+gain" {
		return nil, nil
	}
	im, err := DefaultImatrix()
	if err != nil {
		return nil, err
	}
	qw, err := im.Columns(name)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return qw, nil
}

// The simulation's own tally, by family.
//
// It exists because of the bug it would have caught. `lm_head` screened as
// **exactly** unchanged at 4 bits — 0.636 B parameters re-quantised and not
// one logit moved — because the head is the one dense weight that does not
// reach the device through Model.F32: it is 2.54 GB as floats, so gpu_head.go
// dequantises it a slab at a time and the hook was not there. A family that
// is silently skipped reads as "this family tolerates 4 bits", which is the
// most expensive wrong answer L8c could produce. So every run reports what it
// touched, and a zero is visible rather than inferred.
var (
	simTallyMu sync.Mutex
	simTally   = map[string]int64{}
	simUncal   = map[string]int64{}
)

// simCount records that n weights of a tensor were round-tripped.
func simCount(name string, n int) {
	fam := simFamily(name)
	if fam == "" {
		return
	}
	simTallyMu.Lock()
	simTally[fam] += int64(n)
	simTallyMu.Unlock()
}

// simCountUncal records that n of them had no imatrix entry and fell back to
// round-to-nearest, the way ggml does.
func simCountUncal(name string, n int) {
	fam := simFamily(name)
	if fam == "" {
		return
	}
	simTallyMu.Lock()
	simUncal[fam] += int64(n)
	simTallyMu.Unlock()
}

// SimUncalibratedLine names the weights an imatrix run could not calibrate,
// or "" when there are none.
func SimUncalibratedLine() string {
	simTallyMu.Lock()
	defer simTallyMu.Unlock()
	var total int64
	parts := make([]string, 0, len(simUncal))
	for _, f := range SimFamilies() {
		if n, ok := simUncal[f]; ok && n > 0 {
			parts = append(parts, fmt.Sprintf("%s %.3fB", f, float64(n)/1e9))
			total += n
		}
	}
	if total == 0 {
		return ""
	}
	return fmt.Sprintf("%.3f B of them have no imatrix entry and fell back to rtn — %s",
		float64(total)/1e9, strings.Join(parts, ", "))
}

// SimTally is what the simulation has touched so far, weights per family.
func SimTally() map[string]int64 {
	simTallyMu.Lock()
	defer simTallyMu.Unlock()
	out := make(map[string]int64, len(simTally))
	for k, v := range simTally {
		out[k] = v
	}
	return out
}

// SimTallyLine is that tally as one line, for a header that makes a number
// attributable — and for making a family the hook never reached visible.
func SimTallyLine() string {
	t := SimTally()
	if len(t) == 0 {
		return "nothing"
	}
	var total int64
	parts := make([]string, 0, len(t))
	for _, f := range SimFamilies() {
		if n, ok := t[f]; ok {
			parts = append(parts, fmt.Sprintf("%s %.3fB", f, float64(n)/1e9))
			total += n
		}
	}
	return fmt.Sprintf("%.3f B weights — %s", float64(total)/1e9, strings.Join(parts, ", "))
}

// simFamilies is the whitelist's range, for validating a filter and for
// naming the rows of a per-family screen. They are `cmd/gguf`'s group names,
// so an accuracy row and a byte row line up.
var simFamilies = map[string]bool{
	"deltanet": true, "full_attn": true, "hyper_conn": true,
	"qsa_indexer": true, "lm_head": true, "ple_proj": true,
}

// SimFamilies lists them, sorted.
func SimFamilies() []string {
	out := make([]string, 0, len(simFamilies))
	for f := range simFamilies {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

// Apply round-trips a weight in place: rows of k values, groups of q.Group
// contiguous elements along a row.
//
// k must be a whole number of groups. Every dense matrix in this model has
// k in {2560, 6144, 10240, 640, 336, 1024}, so the largest group that divides
// all of them is 16 — and the ones that matter are the first three, which are
// whole multiples of 128. A group that does not divide a row is an error
// rather than a short last group, because a short group is a different format
// and would quietly make the bits-per-weight column wrong.
func (q QuantSim) Apply(x []float32, k int) error { return q.ApplyWeighted(x, k, nil) }

// ApplyWeighted is Apply with the importance of each input column — the
// imatrix row for this tensor, length k, or nil.
//
// It is a separate entry point rather than a field because the weights belong
// to the *tensor* and the format belongs to the plan, and the caller is the
// only one holding both.
func (q QuantSim) ApplyWeighted(x []float32, k int, qw []float32) error {
	if q.Off() || len(x) == 0 {
		return nil
	}
	if k <= 0 || len(x)%k != 0 {
		return fmt.Errorf("llm: quant sim over %d values in rows of %d", len(x), k)
	}
	if k%q.Group != 0 {
		return fmt.Errorf("llm: quant sim group %d does not divide a row of %d", q.Group, k)
	}
	if qw != nil && len(qw) != k {
		return fmt.Errorf("llm: %d importance columns for a row of %d", len(qw), k)
	}
	if q.Mode == "imatrix" && qw == nil {
		return fmt.Errorf("llm: mode imatrix with no importance columns")
	}
	if q.Asym {
		return q.applyAsym(x, k, qw)
	}
	half := 1 << (q.Bits - 1)
	lo, hi := float32(-half), float32(half-1)
	rows := len(x) / k
	if strings.HasPrefix(q.Mode, "search") || strings.HasPrefix(q.Mode, "imatrix") {
		parallelFor(rows, func(r int) {
			row := x[r*k : (r+1)*k]
			// sigma2 is the row's mean square, which is what ggml's
			// quantize_row_q*_0_impl computes once per row and folds into
			// every group's weight. Its accumulator is `float` there, and a
			// wider one here would shift every weight in the row and with it
			// the rung the search picks — the same reason makeQxQuants keeps
			// float32 sums.
			var sum2 float32
			for _, v := range row {
				sum2 += v * v
			}
			sigma2 := sum2 / float32(k)
			w := make([]float32, q.Group)
			lv := make([]int8, q.Group)
			for g := 0; g < k; g += q.Group {
				blk := row[g : g+q.Group]
				for j := range blk {
					w[j] = sqrt32(sigma2 + blk[j]*blk[j])
					if qw != nil {
						w[j] *= qw[g+j]
					}
				}
				d := makeQxQuants(blk, half, w, lv)
				if q.gainFix() {
					d = unbiasedScale(blk, lv, d)
				}
				ds := safetensors.F16ToF32(safetensors.F32ToF16(d))
				for j := range blk {
					blk[j] = safetensors.F16ToF32(safetensors.F32ToF16(float32(lv[j]) * ds))
				}
			}
		})
		return nil
	}
	parallelFor(rows, func(r int) {
		row := x[r*k : (r+1)*k]
		for g := 0; g < k; g += q.Group {
			blk := row[g : g+q.Group]
			// The scale. Both conventions are decided by the element of
			// largest magnitude; they disagree about what to divide by and
			// about the sign.
			var amax, maxval float32
			for _, v := range blk {
				a := v
				if a < 0 {
					a = -a
				}
				if a > amax {
					amax, maxval = a, v
				}
			}
			var d float32
			if q.Ggml {
				d = maxval / lo // lo is negative: ggml's d = -maxval/8
			} else {
				d = amax / hi
			}
			if d == 0 {
				// An all-zero group. Leave it alone rather than inventing a
				// scale; every level is zero either way.
				continue
			}
			// The scale is stored as an fp16 and the kernel reads it back,
			// so the quantisation has to be against the *stored* value or
			// the levels are not the ones a kernel would pick.
			ds := safetensors.F16ToF32(safetensors.F32ToF16(d))
			if ds == 0 {
				continue
			}
			id := 1 / ds
			for i, v := range blk {
				l := roundHalfAway(v * id)
				if l < lo {
					l = lo
				}
				if l > hi {
					l = hi
				}
				// The half the kernel forms: float16_t(q) * d, correctly
				// rounded. The f32 product is exact, so rounding it once is
				// the same value (L8b-2, D10's neighbour).
				blk[i] = safetensors.F16ToF32(safetensors.F32ToF16(l * ds))
			}
		}
	})
	return nil
}

// makeQxQuants is ggml's `make_qx_quants` with rmse_type 1, which is the
// function every imatrix-aware quantiser in ggml-quants.c goes through.
//
// It does two things, and L8c-2 needs them apart. First it picks levels
// against `iscale = -nmax/maxval` — the same initial scale ggml's
// *uncalibrated* path uses, which is why the two arms share a starting point
// — and then it replaces the scale with the weighted least-squares one,
// `sumlx/suml2`. Second it sweeps that initial scale over nineteen rungs
// (`nmax + 0.1*is`, is in [-9,9] and not zero) and keeps whichever maximises
// `sumlx^2/suml2`, which is the objective's value at its own optimum.
//
// The weights are the caller's: `sqrt(sigma2 + x^2)` alone is ggml's
// rmse_type 1, and that times the imatrix column is
// `quantize_row_q4_0_impl`. Levels come back in lv as signed values in
// [-nmax, nmax-1] rather than ggml's biased [0, 2*nmax), because nothing here
// packs nibbles — the simulation stages the halves the levels stand for.
func makeQxQuants(x []float32, nmax int, w []float32, lv []int8) float32 {
	var amax, maxval float32
	for _, v := range x {
		a := v
		if a < 0 {
			a = -a
		}
		if a > amax {
			amax, maxval = a, v
		}
	}
	// GROUP_MAX_EPS, ggml-quants.c: an all-but-zero group has no scale worth
	// searching for.
	if amax < 1e-15 {
		for i := range lv {
			lv[i] = 0
		}
		return 0
	}
	lon, hin := float32(-nmax), float32(nmax-1)
	clampLevel := func(f float32) float32 {
		l := nearestInt(f)
		if l < lon {
			l = lon
		}
		if l > hin {
			l = hin
		}
		return l
	}
	// **float32 accumulators, not float64.** ggml's are `float`, and the
	// nineteen rungs below are chosen by `sumlx*sumlx > best*suml2` — a
	// comparison between near-equal quantities, where a wider accumulator
	// picks a different rung on about one group in a thousand. Widening it
	// is "more accurate" and produces a bank that is not the reference's, so
	// the port keeps the reference's width.
	iscale := -float32(nmax) / maxval
	var sumlx, suml2 float32
	for i, v := range x {
		l := clampLevel(iscale * v)
		lv[i] = int8(l)
		sumlx += w[i] * v * l
		suml2 += w[i] * l * l
	}
	scale := float32(0)
	if suml2 != 0 {
		scale = sumlx / suml2
	}
	best := scale * sumlx
	for is := -9; is <= 9; is++ {
		if is == 0 {
			continue
		}
		try := -(float32(nmax) + 0.1*float32(is)) / maxval
		var sx, s2 float32
		for i, v := range x {
			l := clampLevel(try * v)
			sx += w[i] * v * l
			s2 += w[i] * l * l
		}
		if s2 > 0 && sx*sx > best*s2 {
			for i, v := range x {
				lv[i] = int8(clampLevel(try * v))
			}
			scale = sx / s2
			best = scale * sx
		}
	}
	return scale
}

// gainFix reports whether this format replaces make_qx_quants' scale with the
// unbiased one (L8c-2's `+gain`).
func (q QuantSim) gainFix() bool { return strings.HasSuffix(q.Mode, "+gain") }

// unbiasedScale replaces a group's scale with the one that makes the
// reconstruction unbiased, keeping the levels make_qx_quants chose.
//
// **This is the correction L8c-2's mechanism implies.** make_qx_quants picks
// `d = sum(w*x*l) / sum(w*l*l)`, the weighted least-squares scale, which
// minimises squared error by trading a little *bias* for a lot of variance —
// the right trade for one group alone and the wrong one for a weight read 97
// times a pass, because bias compounds through depth where noise averages
// out. The unbiased scale instead solves `sum (l*d) * x = sum x*x`, so the
// reconstruction has no systematic component along the original:
//
//	d = sum(x*x) / sum(l*x)
//
// It is deliberately *unweighted*: a gain error costs the model in proportion
// to the whole row, not to the calibration's view of it. Falls back to the
// search's own scale where the denominator is degenerate.
func unbiasedScale(x []float32, lv []int8, d float32) float32 {
	var num, den float64
	for i, v := range x {
		num += float64(v) * float64(v)
		den += float64(lv[i]) * float64(v)
	}
	if den == 0 || num == 0 {
		return d
	}
	g := float32(num / den)
	// A sign flip or a wild magnitude means the levels carry no usable
	// information about this group; keep what the search returned.
	if g == 0 || (g < 0) != (d < 0) || g/d > 2 || g/d < 0.5 {
		return d
	}
	return g
}

// asymSuperBlocks is ggml's QK_K/32: eight groups of 32 share one fp16 pair.
// It is the shape every published K-quant number is quoted at.
const asymSuperBlocks = 8

// superBlocks is how many groups of `q.Group` share one fp16 (d, dmin) pair
// in a row of k, and it is the one place this arm departs from ggml.
//
// ggml's K-quants require `k % 256 == 0` and `llama-quantize` falls back to a
// different type where a row is not a whole number of super-blocks. **Every
// dense row in this model is, except one family and it is the family the
// question is about**: `hc_{attn,ffn}_up` and `output_hc_up` read the
// low-rank space, so their k is 320 — and `hc_attn_up` is L8c-2's outlier,
// the tensor whose importance is concentrated in 4.4 effective columns of 32
// and which carries the whole imatrix regression. Falling back to Q4_0 there
// would answer a different question than the one asked.
//
// So the super-block is ggml's eight where eight fits, and **the whole row**
// where it does not and the row is short enough for a 6-bit scale to still be
// amortised. That is a real format — nothing in `make_qkx3_quants` or
// `make_qp_quants` knows the count — and it costs 4.475 bits a weight at 320
// rather than 4.500, which the caller states rather than rounds off.
func (q QuantSim) superBlocks(k int) (int, error) {
	if k%q.Group != 0 {
		return 0, fmt.Errorf("llm: group %d does not divide a row of %d", q.Group, k)
	}
	if k%(asymSuperBlocks*q.Group) == 0 {
		return asymSuperBlocks, nil
	}
	if n := k / q.Group; n >= 2 && n <= 2*asymSuperBlocks {
		return n, nil
	}
	return 0, fmt.Errorf("llm: a row of %d is neither a multiple of %d nor short enough to be one super-block",
		k, asymSuperBlocks*q.Group)
}

// applyAsym round-trips a weight through ggml's K-quant form: LLM.md L8c-3.
//
// Two arms, and they are the same two the symmetric side has. `rtn` is
// `quantize_row_q4_K_ref` — `make_qkx2_quants` per group against
// `av_x + |x|`, then the group scales and mins quantised to six bits against
// their own maxima. `search`/`imatrix` is `quantize_row_q4_K_impl` —
// `make_qkx3_quants` per group, then `make_qp_quants` over the group scales
// weighted by how much calibration mass each group carries. Only the weight
// vector differs between `search` and `imatrix`, which is what keeps the
// attribution honest (L8c-2's rule).
//
// The value staged is the one a kernel forms: `d*sc*l - dmin*m` in f32, which
// is exactly `llm_moe_gemv.comp`'s Q4_K unpack — the kernel that already
// reads this format here. Unlike the symmetric arm it is not rounded to fp16
// on the way out, because the affine form has a subtraction in it and the
// shader does that subtraction in f32; the bank's own staging rounds the
// result once, which is the same half either way.
func (q QuantSim) applyAsym(x []float32, k int, qw []float32) error {
	sub, err := q.superBlocks(k)
	if err != nil {
		return err
	}
	n := sub * q.Group
	rows := len(x) / k
	parallelFor(rows, func(r int) {
		row := x[r*k : (r+1)*k]
		e := newAsymEnc(q, sub)
		for s := 0; s < k; s += n {
			blk := row[s : s+n]
			var cols []float32
			if qw != nil {
				cols = qw[s : s+n]
			}
			e.Encode(blk, cols)
			for i := range blk {
				blk[i] = e.Dequant(i)
			}
		}
	})
	return nil
}

// makeQkxQuants is ggml's `make_qkx2_quants` and `make_qkx3_quants`, which
// are the same function.
//
// They differ in two places and neither is reachable: qkx3 accepts a null
// weight vector where qkx2 requires one (every call site here passes one),
// and qkx3's degenerate test is `max <= min` where qkx2's is `max == min`,
// after a line that has already forced `min <= 0 <= max`. So the port is one
// function and the arms are the caller's three constants.
//
// It fits an affine group: levels `l` in [0, nmax] and a pair (scale, min)
// with `x ~ scale*l + min`, min <= 0. The first pass takes the obvious
// `iscale = nmax/(max-min)`; then `nstep+1` rungs slide that scale, and for
// each one the *weighted least squares* (scale, min) is solved in closed form
// over the levels it produces and kept if it lowers the weighted error. It
// returns the scale and `-min`, which is ggml's sign convention: the stored
// `dmin` is positive and the dequant subtracts.
func makeQkxQuants(x []float32, nmax int, w []float32, lv, laux []uint8,
	rmin, rdelta float32, nstep int) (scale, theMin float32) {

	min, max := x[0], x[0]
	sumW, sumX := w[0], w[0]*x[0]
	for i := 1; i < len(x); i++ {
		if x[i] < min {
			min = x[i]
		}
		if x[i] > max {
			max = x[i]
		}
		sumW += w[i]
		sumX += w[i] * x[i]
	}
	if min > 0 {
		min = 0
	}
	if max <= min {
		for i := range lv {
			lv[i] = 0
		}
		return 0, -min
	}
	iscale := float32(nmax) / (max - min)
	scale = 1 / iscale
	clampLevel := func(f float32) uint8 {
		l := nearestInt(f)
		if l < 0 {
			l = 0
		}
		if l > float32(nmax) {
			l = float32(nmax)
		}
		return uint8(l)
	}
	var bestErr float32
	for i, v := range x {
		lv[i] = clampLevel(iscale * (v - min))
		diff := scale*float32(lv[i]) + min - v
		bestErr += w[i] * diff * diff
	}
	if nstep < 1 {
		return scale, -min
	}
	for is := 0; is <= nstep; is++ {
		iscale = (rmin + rdelta*float32(is) + float32(nmax)) / (max - min)
		var sumL, sumL2, sumXL float32
		for i, v := range x {
			l := clampLevel(iscale * (v - min))
			laux[i] = l
			fl := float32(l)
			sumL += w[i] * fl
			sumL2 += w[i] * fl * fl
			sumXL += w[i] * fl * v
		}
		det := sumW*sumL2 - sumL*sumL
		if det <= 0 {
			continue
		}
		thisScale := (sumW*sumXL - sumX*sumL) / det
		thisMin := (sumL2*sumX - sumL*sumXL) / det
		if thisMin > 0 {
			thisMin = 0
			thisScale = sumXL / sumL2
		}
		var err float32
		for i, v := range x {
			diff := thisScale*float32(laux[i]) + thisMin - v
			err += w[i] * diff * diff
		}
		if err < bestErr {
			copy(lv, laux)
			bestErr = err
			scale, min = thisScale, thisMin
		}
	}
	return scale, -min
}

// makeQpQuants is ggml's `make_qp_quants`: the non-negative quantiser the
// calibrated K-quant path uses for the super-block's own scales and mins.
//
// It is `make_qx_quants`'s sibling for values that cannot be negative — nine
// rungs of an initial scale, then **five sweeps of coordinate descent**, each
// asking of every level whether moving it raises `sumlx^2/suml2`. The weight
// is `sw`, the calibration mass of each group, so a group the imatrix cares
// about gets a 6-bit scale nearer its own optimum.
func makeQpQuants(x []float32, nmax int, w []float32, lv []int) float32 {
	var max float32
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	if max < 1e-15 { // GROUP_MAX_EPS
		for i := range lv {
			lv[i] = 0
		}
		return 0
	}
	iscale := float32(nmax) / max
	for i, v := range x {
		lv[i] = int(nearestInt(iscale * v))
	}
	scale := 1 / iscale
	var bestMSE float32
	for i, v := range x {
		diff := v - scale*float32(lv[i])
		bestMSE += w[i] * diff * diff
	}
	for is := -4; is <= 4; is++ {
		if is == 0 {
			continue
		}
		iscaleIs := (0.1*float32(is) + float32(nmax)) / max
		scaleIs := 1 / iscaleIs
		var mse float32
		for i, v := range x {
			l := clamp(nearestInt(iscaleIs*v), nmax)
			diff := v - scaleIs*l
			mse += w[i] * diff * diff
		}
		if mse < bestMSE {
			bestMSE = mse
			iscale = iscaleIs
		}
	}
	var sumlx, suml2 float32
	for i, v := range x {
		l := clamp(nearestInt(iscale*v), nmax)
		lv[i] = int(l)
		sumlx += w[i] * v * l
		suml2 += w[i] * l * l
	}
	for try := 0; try < 5; try++ {
		changed := 0
		for i, v := range x {
			slx := sumlx - w[i]*v*float32(lv[i])
			sl2 := suml2 - w[i]*float32(lv[i])*float32(lv[i])
			if slx <= 0 || sl2 <= 0 {
				continue
			}
			newL := clamp(nearestInt(v*sl2/slx), nmax)
			if int(newL) == lv[i] {
				continue
			}
			slx += w[i] * v * newL
			sl2 += w[i] * newL * newL
			if slx*slx*suml2 > sumlx*sumlx*sl2 {
				lv[i], sumlx, suml2 = int(newL), slx, sl2
				changed++
			}
		}
		if changed == 0 {
			break
		}
	}
	if suml2 > 0 {
		return sumlx / suml2
	}
	return 0
}

// clamp is make_qp_quants' MIN(nmax, l), which has no lower bound because
// every value it is given is non-negative.
func clamp(v float32, nmax int) float32 {
	if v > float32(nmax) {
		return float32(nmax)
	}
	return v
}

func clamp63(v float32) int {
	if v < 0 {
		return 0
	}
	if v > 63 {
		return 63
	}
	return int(v)
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// nearestInt is ggml's, and it is **not** roundHalfAway.
//
// `nearest_int` adds 1.5*2^23 to force the significand's low bits out under
// the FPU's current rounding mode and reads the integer back out of the
// bits — which makes it round-half-to-**even**, where `quantize_row_q4_0_ref`
// and every quantiser in this repo round half away from zero. The two differ
// only on exact ties, which is why `rtn` matched ggml bit for bit without it
// and the nineteen-rung search did not.
func nearestInt(v float32) float32 { return float32(math.RoundToEven(float64(v))) }

func sqrt32(v float32) float32 {
	if v <= 0 {
		return 0
	}
	return float32(math.Sqrt(float64(v)))
}

// roundHalfAway is ggml's roundf: ties away from zero, which is what every
// quantiser in this repo and in ggml uses, and not Go's math.RoundToEven.
func roundHalfAway(v float32) float32 {
	if v >= 0 {
		return float32(int32(v + 0.5))
	}
	return float32(int32(v - 0.5))
}

// simFamily names the streamed dense family a tensor belongs to, or "" if the
// simulation must not touch it.
//
// The list is the answer to "what does D3 mean by everything *streamed*", and
// it is a whitelist rather than a filter because the cost of being wrong is
// asymmetric: re-quantising a norm weight or the router's ties would produce
// a perplexity that is not the format's, and nothing downstream would say so.
//
// Out of scope by construction, and each for its own reason:
//
//   - **the 512 expert banks** never pass through Model.F32 at all — L5b
//     stages them byte for byte out of the mmap'd checkpoint, so they keep
//     the checkpoint's Q4_K/Q5_K/Q5_1/Q8_0 whatever this is set to, and D4
//     says not to go below them anyway;
//   - **`token_embd` and the n-gram table** are gathers, not matmuls (D2);
//   - **every norm, `ssm_a`, `ssm_dt.bias` and `ssm_conv1d`** are elementwise
//     parameters, 1-D or nearly, and are not read a row at a time;
//   - **`ffn_gate_inp`, the router** is F32 in the checkpoint and D3 keeps it
//     at fp16: it is 0.25 GB a token and its ties decide which ten of 512
//     experts run (L5a).
func simFamily(name string) string {
	base := name
	if i := strings.Index(name, "."); i >= 0 && strings.HasPrefix(name, "blk.") {
		if j := strings.Index(name[i+1:], "."); j >= 0 {
			base = name[i+1+j+1:]
		}
	}
	switch base {
	case "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
		"hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight":
		return "hyper_conn"
	case "attn_q.weight", "attn_k.weight", "attn_v.weight":
		return "full_attn"
	case "attn_output.weight":
		return "full_attn"
	case "indexer.q_proj.weight", "indexer.k_proj.weight":
		return "qsa_indexer"
	case "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight":
		return "deltanet"
	case "ssm_alpha.weight", "ssm_beta.weight":
		return "deltanet"
	}
	switch base {
	case "ple_key.weight", "ple_value.weight":
		// L2d's fused key/value projection. It runs once, at layer 1, for
		// 0.6% of a decode step — small enough that LLM.md wrote it down
		// rather than narrowing it at L8b, and cheap enough to price here
		// rather than leave as the one streamed matmul nobody measured.
		return "ple_proj"
	}
	switch name {
	case "output.weight":
		return "lm_head"
	case "output_hc_down.weight", "output_hc_up.weight":
		return "hyper_conn"
	}
	return ""
}
