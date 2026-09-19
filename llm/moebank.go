package llm

// The MoE bank's widths: LLM.md P4.
//
// # What P4 asked for, and what the checkpoint allows
//
// The review's P4 reads: "the expert banks toward ~4.25 bits: transcode
// `ffn_down_exps`'s Q5_1 (27.1 GB) and the five Q8_0 layers to Q4_K with the
// imatrix — the existing `llm_moe_gemm/gemv` arms already read Q4_K, so this
// is a transcode and a corpus run, not a kernel."
//
// **It is not buildable as written, and the reason is one number.** A ggml
// K-quant's super-block is QK_K = 256 elements along the reduction axis, so a
// tensor can only *be* a Q4_K if its row divides 256. `ffn_down_exps` is
// `[640, 2560, 512]` — its row is **640**, two and a half super-blocks — and
// so are `ffn_down_shexp` and, off the MoE path, the n-gram table at 160.
// That is why unsloth shipped the one expert tensor in this model at Q5_1 and
// Q8_0 while gate and up (row 2560) are Q4_K: not because the imatrix said
// the down projection is sensitive, which is how `LLM.md`'s inventory read
// it, but because **llama-quantize had no K-quant to offer it**. The two
// readings predict opposite things about what a re-quantisation costs, and
// the format one is the true one.
//
// So the down projection below six bits is a *kernel* stage — a block-32
// format the down mode has no arm for. **That stage is P4c, and it is now
// built**: `q4_1` (5.0 bits) and `iq4_nl` (4.5, the calibrated non-linear
// form llama-quantize itself falls back to) are targets here and arms of
// `llm_moe_gemm.comp` and `llm_moe_gemv.comp`, so `down_exps` — 0.615 GB a
// token, 41% of the expert traffic — can be narrowed by the same grammar
// that narrowed the shared expert.
//
// P4c also found where the "the bank is the checkpoint's own bytes" rule
// stops: it binds the tensors staged **verbatim**, and a transcoded one is
// written here. IQ4_NL's eighteen-byte record is not a multiple of four, and
// leaving it in ggml's layout cost the decode kernel **28% of its read rate**
// — so `pairIQ4NLRow` rearranges each row into 36-byte pairs. Same bits, same
// levels, four-aligned, and a block's scale still beside its own nibbles,
// which is what the prefill kernel needs and a plane of scales destroys.
//
// # What is a transcode, and is what this file does
//
// Three rows are at **8.5 bits** for no reason but that unsloth left them
// there, and every one of them has a ggml format the MoE kernels already
// read:
//
//	ffn_gate_shexp   row 2560   Q8_0 -> Q4_K   0.0393 GB a token saved
//	ffn_up_shexp     row 2560   Q8_0 -> Q4_K   0.0393
//	ffn_down_shexp   row  640   Q8_0 -> Q5_1   0.0246
//	ffn_down_exps    row  640   Q8_0 -> Q5_1   0.0256   (the 5 layers of 48)
//
// 0.129 GB a token of 4.171, and the first three come off the **shared
// expert**, which P1 measured at 136.5 GB/s — the second-slowest family on
// the board — so the time they buy is worth more than their share of the
// bytes.
//
// # The grammar
//
//	LLM_MOE_BANK=gate_shexp=q4_k,up_shexp=q4_k,down_shexp=q5_1,down_exps=q5_1
//
// A family names a **ceiling**: a tensor already at or below that width is
// left alone, which is what makes `down_exps=q5_1` mean "the five Q8_0
// layers" without naming them. Widening is refused rather than performed —
// a bank wider than the checkpoint is not a thing anybody wants by accident.
//
// `LLM_MOE_BANK_QUANT` is `imatrix` (the default) or `rtn`, the same two arms
// and the same default as `LLM_DENSE_BANK`, and for the same reason: a bank
// is built with the calibration L8c-3 recommends, and the uncalibrated arm
// exists so the difference can be attributed.

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/safetensors"
)

// MoEBankPlan is which MoE families are narrowed and to what.
//
// The zero value is off, which is what every `results/*.csv` before P4 was
// measured on and what `cmd/llm` still defaults to.
type MoEBankPlan struct {
	to   map[string]gguf.Type
	mode string
	spec string
}

// moeFamilies maps a plan's family name onto the tensor suffix it selects.
// They are the six the MoE block stages, and the names are the checkpoint's
// own with `ffn_` dropped.
var moeFamilies = map[string]string{
	"gate_exps":  "ffn_gate_exps",
	"up_exps":    "ffn_up_exps",
	"down_exps":  "ffn_down_exps",
	"gate_shexp": "ffn_gate_shexp",
	"up_shexp":   "ffn_up_shexp",
	"down_shexp": "ffn_down_shexp",
}

// moeBankFormats are the ggml types a transcode can target, which is exactly
// the set `moeSPIRV` has arms for on the mode that reads them. Q4_K and Q5_K
// are the up mode's; Q5_1, and since P4c Q4_1 and IQ4_NL, are the down
// mode's. Asking for a format the kernel cannot read would stage a bank that
// reads as noise.
//
// **Q4_0 is deliberately absent.** It is IQ4_NL's bytes exactly — 18 for 32
// weights — with evenly spaced levels instead of the calibrated codebook, and
// this vertical has measured the symmetric linear form against the asymmetric
// one twice at the same width (D4, and L8c-1's +18.5%). A row that will take
// IQ4_NL has no reason to be Q4_0, so building the arm would only add a
// pipeline nothing should choose.
var moeBankFormats = map[string]gguf.Type{
	"q4_k":   gguf.Q4_K,
	"q5_k":   gguf.Q5_K,
	"q5_1":   gguf.Q5_1,
	"q4_1":   gguf.Q4_1,
	"iq4_nl": gguf.IQ4_NL,
}

// moeBitsPerWeight is a format's width, for the ceiling comparison. It is
// `BlockBytes*8/BlockElems`, written out so the ordering below is legible.
func moeBitsPerWeight(t gguf.Type) float64 {
	return float64(t.BlockBytes()) * 8 / float64(t.BlockElems())
}

// Off reports whether the plan narrows nothing.
func (p MoEBankPlan) Off() bool { return len(p.to) == 0 }

// String is the spec as written, so a CSV header can name what it measured.
func (p MoEBankPlan) String() string {
	if p.Off() {
		return "off"
	}
	return p.spec + " (" + p.mode + ")"
}

// Mode is `imatrix` or `rtn`.
func (p MoEBankPlan) Mode() string { return p.mode }

// For reports the format tensor `name` should be staged in, and whether that
// differs from the type it ships in.
//
// The ceiling rule lives here: a tensor already at or below the family's
// width stays exactly as it is, bit for bit, which is what keeps
// `down_exps=q5_1` from touching the 43 layers that are already Q5_1.
func (p MoEBankPlan) For(name string, have gguf.Type) (gguf.Type, bool) {
	for fam, suffix := range moeFamilies {
		if !strings.Contains(name, suffix) {
			continue
		}
		to, ok := p.to[fam]
		if !ok {
			return have, false
		}
		if moeBitsPerWeight(have) <= moeBitsPerWeight(to) {
			return have, false
		}
		return to, true
	}
	return have, false
}

// ParseMoEBankPlan reads the grammar above.
func ParseMoEBankPlan(spec, mode string) (MoEBankPlan, error) {
	spec = strings.TrimSpace(spec)
	if mode = strings.TrimSpace(mode); mode == "" {
		mode = "imatrix"
	}
	if mode != "imatrix" && mode != "rtn" {
		return MoEBankPlan{}, fmt.Errorf("llm: LLM_MOE_BANK_QUANT=%q, want imatrix or rtn", mode)
	}
	p := MoEBankPlan{mode: mode, spec: spec}
	if spec == "" || spec == "off" {
		return MoEBankPlan{mode: mode, spec: "off"}, nil
	}
	p.to = map[string]gguf.Type{}
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		fam, want, ok := strings.Cut(part, "=")
		if !ok {
			return MoEBankPlan{}, fmt.Errorf("llm: LLM_MOE_BANK entry %q is not family=format", part)
		}
		fam, want = strings.TrimSpace(fam), strings.ToLower(strings.TrimSpace(want))
		if _, ok := moeFamilies[fam]; !ok {
			return MoEBankPlan{}, fmt.Errorf("llm: no MoE family %q (have %s)", fam, moeFamilyList())
		}
		t, ok := moeBankFormats[want]
		if !ok {
			return MoEBankPlan{}, fmt.Errorf("llm: MoE family %s cannot be staged as %q; the kernels read %s", fam, want, moeFormatList())
		}
		p.to[fam] = t
	}
	return p, nil
}

func moeFormatList() string {
	var xs []string
	for f := range moeBankFormats {
		xs = append(xs, f)
	}
	sort.Strings(xs)
	return strings.Join(xs, ", ")
}

func moeFamilyList() string {
	var xs []string
	for fam := range moeFamilies {
		xs = append(xs, fam)
	}
	sort.Strings(xs)
	return strings.Join(xs, ", ")
}

// ShippedMoEBank is P4b's decision, D20: the MoE plan a product run stages
// when nothing on the command line says otherwise.
//
// The four rows are every tensor in this block that ships at 8.5 bits and
// has a ggml format one of the MoE kernels already builds. Measured as one
// complete 145-chunk plan on D19's dense bank:
//
//	D19            4.0948   +1.63%   4.408 GB a token
//	D19 + P4b      4.0970   +1.69%   4.279
//	D19 + this     4.0992   +1.74%   4.132   (P4c's `down_exps=iq4_nl`)
//
// **0.0022 points of perplexity for 0.1288 GB a token — 0.017 pp/GB**, where
// D19 refused to *buy* a family's fifth bit at 1.6 and took three above 5.9.
// Paired over the 145 chunks the delta is +0.00054 nll a chunk against a
// standard error of 0.00059 (t = 0.91, worse in 85 and better in 60), so the
// accuracy cost is **not resolvable by the instrument** while the bytes are
// arithmetic. It also takes 1.41 GB off residency.
//
// Like `ShippedDenseBank` it is deliberately not `MoEBankPlanFromEnv`'s
// default: `cmd/serve` opts in, `cmd/llm` does not, so no CSV in `results/`
// is ambiguous about which bank it measured.
//
// **P4c adds the fifth row, and it is the largest: D21.** `down_exps` at
// `iq4_nl` is 0.148 GB a token more — the row was 41% of the expert traffic —
// for **+0.05 points** (4.0992 against 4.0970, t = 0.79 paired over the 145
// chunks, which is not resolvable) and a measured **+0.92 tok/s**. The 5.0-bit
// `q4_1` arm was built beside it and lost on both axes: fewer bytes freed and
// +0.0122 points, resolvable at t = 4.11.
const ShippedMoEBank = "gate_shexp=q4_k,up_shexp=q4_k,down_shexp=q5_1,down_exps=iq4_nl"

var (
	moeBankOnce sync.Once
	moeBankPlan MoEBankPlan
)

// MoEBankPlanFromEnv reads `LLM_MOE_BANK` and `LLM_MOE_BANK_QUANT` once.
//
// Like `DenseBankPlan` it defaults to **off** rather than to the shipped
// plan: a measurement tool that narrowed a bank nobody named would make
// every CSV in `results/` ambiguous about what it measured. `cmd/serve` opts
// in; `cmd/llm` does not.
func MoEBankPlanFromEnv() MoEBankPlan {
	moeBankOnce.Do(func() {
		p, err := ParseMoEBankPlan(os.Getenv("LLM_MOE_BANK"), os.Getenv("LLM_MOE_BANK_QUANT"))
		if err != nil {
			panic(err)
		}
		moeBankPlan = p
	})
	return moeBankPlan
}

// TranscodeMoE re-quantises one staged MoE tensor into `to`, returning the
// bytes the bank should hold.
//
// `t` is the checkpoint's tensor, two- or three-dimensional: `[k, n]` for a
// shared-expert matrix and `[k, n, nExpert]` for a routed one. The row is
// `Dims[0]` on both, which is the axis a scale group and an importance vector
// both run along, so the only thing the third dimension changes is **which
// importance row** a row is fitted against — the published matrix carries one
// per expert (`[k, nExpert]` against `[1, nExpert]` counts).
//
// It dequantises through `gguf.Dequantize` and re-fits, which is a loss on
// top of a loss and is the honest thing to measure: the original floats are
// not in the checkpoint and nothing here can recover them.
func TranscodeMoE(t *gguf.Tensor, to gguf.Type, mode string) ([]byte, error) {
	if len(t.Dims) < 2 {
		return nil, fmt.Errorf("llm: %s is %dD, a MoE weight is 2 or 3", t.Name, len(t.Dims))
	}
	k := int(t.Dims[0])
	// P4c's IQ4_NL layout pairs blocks, so a row has to hold an even number
	// of them. Both 640-wide tensors in this model hold twenty; anything
	// else is refused rather than written in a layout the kernels cannot
	// address.
	if to == gguf.IQ4_NL && (k/to.BlockElems())%2 != 0 {
		return nil, fmt.Errorf("llm: %s has rows of %d, which is %d IQ4_NL blocks — "+
			"the paired layout needs an even number (P4c)", t.Name, k, k/to.BlockElems())
	}
	if k%to.BlockElems() != 0 {
		return nil, fmt.Errorf("llm: %s has rows of %d and %s blocks are %d elements — "+
			"there is no such format for this tensor (P4)", t.Name, k, to, to.BlockElems())
	}
	rows := int(t.Rows())
	nExpert := 1
	if len(t.Dims) > 2 {
		nExpert = int(t.Dims[2])
	}
	rowsPerExpert := rows / nExpert

	n, err := to.SizeOf(t.Elems())
	if err != nil {
		return nil, err
	}
	// The cache, if one is configured. A transcode is minutes of CPU for a
	// bank that is a pure function of (tensor, format, arm) — see
	// `moeCachePath`.
	cache := moeCachePath(t, to, mode)
	if b, ok := readMoECache(cache, n); ok {
		return b, nil
	}
	out := make([]byte, n)
	rowBytes := k / to.BlockElems() * to.BlockBytes()

	// The importance rows, one per expert, fetched once rather than per row.
	qws := make([][]float32, nExpert)
	if mode == "imatrix" {
		im, err := DefaultImatrix()
		if err != nil {
			return nil, err
		}
		for e := range qws {
			var v []float32
			if nExpert > 1 {
				v, err = im.ExpertColumns(t.Name, e)
			} else {
				v, err = im.Columns(t.Name)
			}
			if err != nil {
				return nil, err
			}
			if v != nil && len(v) != k {
				return nil, fmt.Errorf("llm: imatrix %s row %d is %d wide, the tensor's row is %d",
					t.Name, e, len(v), k)
			}
			qws[e] = v
		}
	}

	var perr error
	var perrOnce sync.Once
	parallelFor((rows+packChunk-1)/packChunk, func(ch int) {
		var enc *asymEnc
		if to == gguf.Q4_K || to == gguf.Q5_K {
			sim := QuantSim{Bits: 4, Group: 32, Mode: mode, Asym: true}
			if to == gguf.Q5_K {
				sim.Bits = 5
			}
			enc = newAsymEnc(sim, asymSuperBlocks)
		}
		row := make([]float32, 0, k)
		for r := ch * packChunk; r < minInt((ch+1)*packChunk, rows); r++ {
			row = row[:0]
			var err error
			if row, err = t.DequantizeRow(int64(r), row); err != nil {
				perrOnce.Do(func() { perr = err })
				return
			}
			qw := qws[r/rowsPerExpert]
			dst := out[r*rowBytes : (r+1)*rowBytes]
			switch to {
			case gguf.Q4_K, gguf.Q5_K:
				if err := packKRow(dst, row, qw, enc, to); err != nil {
					perrOnce.Do(func() { perr = err })
					return
				}
			case gguf.Q5_1:
				packQ51Row(dst, row, qw)
			case gguf.Q4_1:
				packQ41Row(dst, row, qw)
			case gguf.IQ4_NL:
				packIQ4NLRow(dst, row, qw)
				pairIQ4NLRow(dst)
			}
		}
	})
	if perr != nil {
		return nil, perr
	}
	writeMoECache(cache, out)
	return out, nil
}

// moeCacheVersion is bumped whenever a packer here changes what it writes for
// the same inputs. It is part of every cache file's name, so an old file is
// never read by new code — it is simply not looked for.
//
// v1: P4c, `iq4_nl` in ggml's records and `q4_1` at ggml's normalisation.
// v2: `iq4_nl` in 36-byte pairs instead — a layout change, so the same fit
// and different bytes, which is exactly the case this constant exists for.
const moeCacheVersion = 2

// moeCachePath is where a transcoded tensor is kept, or "" for no cache.
//
// **A transcode is expensive and perfectly reproducible**: fitting all 48
// layers of `ffn_down_exps` to IQ4_NL is 40 billion weights through a
// sixteen-scale search, about **eight minutes** of a 32-core machine, and it
// produces the same bytes every time. A measurement tool can pay that; a
// server that stages in 34 seconds cannot, so `cmd/serve` points
// `LLM_BANK_CACHE` at the checkpoint's own directory and the second start
// reads 22.6 GB off disk instead.
//
// The key is everything the bytes depend on that is *not* the file's content:
// the tensor's name and shape, the type it ships in, the type it is being
// written as, the calibration arm, and `moeCacheVersion`. The one thing it
// does not hash is the source bytes themselves — hashing 30 GB to save eight
// minutes of arithmetic would give most of the saving back — so the cache is
// scoped to a checkpoint by living **inside its directory**, and a checkpoint
// edited in place under a stable name is the case it cannot see. That is the
// same assumption `mmap` already makes about the file.
func moeCachePath(t *gguf.Tensor, to gguf.Type, mode string) string {
	dir := strings.TrimSpace(os.Getenv("LLM_BANK_CACHE"))
	if dir == "" {
		return ""
	}
	dims := make([]string, len(t.Dims))
	for i, d := range t.Dims {
		dims[i] = strconv.FormatInt(d, 10)
	}
	return filepath.Join(dir, fmt.Sprintf("%s.%s.%s-%s.%s.v%d.bin",
		t.Name, strings.Join(dims, "x"), t.Type, to, mode, moeCacheVersion))
}

// readMoECache returns the cached bytes if they are there and the right
// length. A short or unreadable file is a miss, not an error: the transcode
// that follows is the truth and it will overwrite it.
func readMoECache(path string, want int64) ([]byte, bool) {
	if path == "" {
		return nil, false
	}
	b, err := os.ReadFile(path)
	if err != nil || int64(len(b)) != want {
		return nil, false
	}
	return b, true
}

// writeMoECache stores a transcode, through a temporary file in the same
// directory so that a reader never sees a partial one. Every failure is
// silent by design — a cache that cannot be written costs time and nothing
// else, and a staging run should not die because a disk is full.
func writeMoECache(path string, b []byte) {
	if path == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return
	}
	tmp := f.Name()
	if _, err := f.Write(b); err != nil {
		f.Close()
		os.Remove(tmp)
		return
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
	}
}

// packKRow writes one row as ggml `block_q4_K`/`block_q5_K` records.
//
// The fit is `asymEnc`, which is the same encoder `bank_q4.go` packs into
// *our* dense layout — so the levels a transcoded expert carries and the
// levels L8c-4's bank carries come from one function, and a difference
// between the two banks can never be a difference of quantiser.
//
// The packing is ggml's own, inverted from `gguf.dequantQ4_K`: within each 64
// elements the low nibbles of all 32 bytes come first and then the high
// nibbles, so sub-blocks `2h` and `2h+1` share 32 bytes.
func packKRow(dst []byte, row, qw []float32, e *asymEnc, to gguf.Type) error {
	const sb = 256
	blockBytes := to.BlockBytes()
	for b := 0; b*sb < len(row); b++ {
		blk := row[b*sb : (b+1)*sb]
		var w []float32
		if qw != nil {
			w = qw[b*sb : (b+1)*sb]
		}
		e.Encode(blk, w)
		rec := dst[b*blockBytes : (b+1)*blockBytes]
		for i := range rec {
			rec[i] = 0
		}
		// The first sixteen bytes of a ggml K-quant block are d, dmin and
		// `get_scale_min_k4`'s twelve — which is `bank_q4.go`'s record
		// exactly, so the two banks share the writer as they share the fit.
		if err := packQ4KRecord(rec, e); err != nil {
			return err
		}
		ql, qh := rec[16:], []byte(nil)
		if to == gguf.Q5_K {
			qh, ql = rec[16:48], rec[48:]
		}
		for h := 0; h < 4; h++ {
			for l := 0; l < 32; l++ {
				lo, hi := e.Q[h*64+l], e.Q[h*64+32+l]
				ql[h*32+l] = lo&0xF | hi&0xF<<4
				if qh != nil {
					// The fifth bits sit two per byte per 64 elements: bit
					// 2h for the low half and 2h+1 for the high, which is
					// `u1`/`u2` shifting left by two in the dequantiser.
					if lo&0x10 != 0 {
						qh[l] |= 1 << uint(2*h)
					}
					if hi&0x10 != 0 {
						qh[l] |= 1 << uint(2*h+1)
					}
				}
			}
		}
	}
	return nil
}

// packQ51Row writes one row as ggml `block_q5_1`: a 32-element block with an
// fp16 scale, an fp16 min, a 32-bit plane of fifth bits and 16 bytes of
// nibbles. 24 bytes, 6 bits a weight.
//
// The fit is ours, and **P4b's reason for that was wrong**. That stage wrote
// "Q5_1 is fitted with the imatrix, which ggml does not do", from reading
// `quantize_row_q5_1_ref`; the entry point `llama-quantize` actually reaches
// is `ggml_quantize_chunk` -> `quantize_q5_1` -> `quantize_row_q5_1_impl`,
// which **is** wired to the matrix and fits it with
// `make_qkx3_quants(32, 31, ..., -0.9, 0.05, 36)` — our three constants
// exactly. So the arms are not "ours against none" but ours against ggml's,
// and they differ in two places: `sigma2` is the block's `2*sum/32` here and
// the row's `sum/n` there, and the levels are chosen against the stored
// halves here (D10) and against the f32 pair there.
//
// It is left as it was measured rather than corrected, because D20 is a
// shipped bank whose perplexity was measured on these bytes; P4c's two
// packers below take ggml's normalisation, and what the difference is worth
// is an open question rather than a claim. The `rtn` arm is
// `quantize_row_q5_1_ref` and is ggml's on both readings.
//
// One asymmetry between the arms is worth stating rather than discovering:
// `makeQkxQuants` clamps the group's min to zero, which is ggml's convention
// in every calibrated path, where `quantize_row_q5_1_ref` keeps the block's
// true min. On a
// block whose values are all one sign that costs the calibrated arm a level
// of range — rare in a weight matrix, and the search recovers more than it
// gives up, but it is why the two arms are not the same fit with a weight
// applied.
func packQ51Row(dst []byte, row, qw []float32) {
	const g = 32
	var lv, laux [g]uint8
	var w [g]float32
	for b := 0; b*g < len(row); b++ {
		blk := row[b*g : (b+1)*g]
		var d, min float32
		if qw != nil {
			var sum2 float32
			for _, v := range blk {
				sum2 += v * v
			}
			sigma2 := 2 * sum2 / float32(g)
			for i, v := range blk {
				w[i] = qw[b*g+i] * sqrt32(sigma2+v*v)
			}
			// 31 levels, ggml's Q5_K constants: the same search, one block
			// wide. `makeQkxQuants` returns (scale, -min) in the K-quants'
			// convention, where a value is `scale*l - min`.
			d, min = makeQkxQuants(blk, 31, w[:], lv[:], laux[:], -0.9, 0.05, 36)
		} else {
			lo, hi := blk[0], blk[0]
			for _, v := range blk {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			// `quantize_row_q5_1_ref` takes the block's own min and max
			// with no clamp to zero — that is Q5_0's symmetric path, not
			// this one.
			d, min = (hi-lo)/31, -lo
		}
		// Stored as halves and read back, so the levels are chosen against
		// the stored pair (D10's neighbour) exactly as `asymEnc` does.
		dh, mh := safetensors.F32ToF16(d), safetensors.F32ToF16(-min)
		df, mf := safetensors.F16ToF32(dh), safetensors.F16ToF32(mh)
		rec := dst[b*24 : (b+1)*24]
		for i := range rec {
			rec[i] = 0
		}
		binary.LittleEndian.PutUint16(rec[0:], dh)
		binary.LittleEndian.PutUint16(rec[2:], mh)
		var qh uint32
		for i, v := range blk {
			var l int
			if df != 0 {
				l = int(nearestInt((v - mf) / df))
			}
			if l < 0 {
				l = 0
			}
			if l > 31 {
				l = 31
			}
			if l&0x10 != 0 {
				qh |= 1 << uint(i)
			}
			if i < 16 {
				rec[8+i] |= byte(l & 0xF)
			} else {
				rec[8+i-16] |= byte(l&0xF) << 4
			}
		}
		rec[4] = byte(qh)
		rec[5] = byte(qh >> 8)
		rec[6] = byte(qh >> 16)
		rec[7] = byte(qh >> 24)
	}
}

// moeRowSigma2 is the normalisation ggml's calibrated legacy paths use:
// `sum(x*x)/n` over the **whole row**, which is not the `2*sum/32` a K-quant
// computes over a super-block. It is a fact about the entry point rather than
// about the format — `quantize_row_q4_1_impl` and `quantize_row_q5_1_impl`
// both take it — and it is a separate function here because getting it wrong
// is invisible: the weights it scales are a ranking, so a factor that is
// uniform across a block changes the fit only through `sqrt(sigma2 + x*x)`'s
// curvature, and the result stays plausible.
func moeRowSigma2(row []float32) float32 {
	var sum float32
	for _, v := range row {
		sum += v * v
	}
	return sum / float32(len(row))
}

// packQ41Row writes one row as ggml `block_q4_1`: a 32-element block with an
// fp16 scale, an fp16 min and 16 nibble pairs. 20 bytes, **5.0 bits a
// weight**, and it is `packQ51Row` with the plane of fifth bits deleted —
// the same asymmetric `scale*l + min` reconstruction, the same 0..15 / 16..31
// split of the pairs, fifteen levels instead of thirty-one.
//
// **The fit is ggml's own calibrated path**, `quantize_row_q4_1_impl`:
// `make_qkx3_quants(32, 15, ..., -0.9, 0.05, 36)` against
// `qw[j]*sqrt(sigma2 + x*x)`, with `sigma2` taken over the row. `makeQkxQuants`
// is that function (`sim.go` says why qkx2 and qkx3 are one port), and the
// three constants are the ones every calibrated fit in this vertical already
// uses, so the transcription is the weights and the normalisation.
//
// There is **one departure, and it is D10's**: the levels are chosen against
// the (d, m) pair *as stored* — halves — where ggml chooses them against the
// f32 pair it then rounds. A kernel reads the halves, so the level that is
// nearest under f32 is not always the level that reconstructs nearest to what
// the shader will compute. `TestPackQ41IsGgmlsQuantiser` is the measurement
// of that choice rather than an assertion about it: it packs the same rows
// both ways and reports which reconstructs the row better under the
// importance weight.
//
// `rtn` is `quantize_row_q4_1_ref`, min/max with no clamp of the min to zero,
// which is what `quantize_q4_1` falls back to when there is no matrix.
func packQ41Row(dst []byte, row, qw []float32) {
	const g = 32
	var lv, laux [g]uint8
	var w [g]float32
	sigma2 := moeRowSigma2(row)
	for b := 0; b*g < len(row); b++ {
		blk := row[b*g : (b+1)*g]
		var d, theMin float32
		if qw != nil {
			for i, v := range blk {
				w[i] = qw[b*g+i] * sqrt32(sigma2+v*v)
			}
			d, theMin = makeQkxQuants(blk, 15, w[:], lv[:], laux[:], -0.9, 0.05, 36)
		} else {
			lo, hi := blk[0], blk[0]
			for _, v := range blk {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			d, theMin = (hi-lo)/15, -lo
		}
		dh, mh := safetensors.F32ToF16(d), safetensors.F32ToF16(-theMin)
		df, mf := safetensors.F16ToF32(dh), safetensors.F16ToF32(mh)
		rec := dst[b*20 : (b+1)*20]
		for i := range rec {
			rec[i] = 0
		}
		binary.LittleEndian.PutUint16(rec[0:], dh)
		binary.LittleEndian.PutUint16(rec[2:], mh)
		packAffineNibbles(rec[4:], blk, df, mf, 15)
	}
}

// packAffineNibbles is the level assignment and the pair split both affine
// block-32 formats here share: element j in the low nibble of byte j and
// element j+16 in its high one, each the nearest level of [0, nmax] under the
// **stored** (d, m).
func packAffineNibbles(dst []byte, blk []float32, df, mf float32, nmax int) {
	for i, v := range blk {
		var l int
		if df != 0 {
			l = int(nearestInt((v - mf) / df))
		}
		if l < 0 {
			l = 0
		}
		if l > nmax {
			l = nmax
		}
		if i < 16 {
			dst[i] |= byte(l & 0xF)
		} else {
			dst[i-16] |= byte(l&0xF) << 4
		}
	}
}

// packIQ4NLRow writes one row as ggml `block_iq4_nl`: an fp16 scale and 16
// nibble pairs indexing the sixteen-level **non-linear codebook**, 18 bytes
// and 4.5 bits a weight — the same bytes as Q4_0 for a lower error, which is
// why the symmetric linear form is not built here at all (D4, and L8c-1's
// +18.5% for the symmetric arm at these bits).
//
// **The fit is ggml's, and for a codebook it has to be.** Every affine packer
// in this file shares its quantiser with `bank_q4.go`, so a difference between
// the two banks can never be a difference of fit; that argument does not reach
// here, because with unevenly spaced levels there is no (scale, min) to fit —
// only a scale, and an assignment that a search has to find. So this is
// `quantize_row_iq4_nl_impl` transcribed: a first assignment from
// `d = -max/values[0]`, the least-squares scale it implies, then fifteen more
// scales swept around it, each scored by the same weighted sum.
//
// Both arms go through it, which is the *other* thing to know about this
// format: `quantize_iq4_nl` passes `ntry = 7` whether or not it has a matrix
// and only the weights change (`qw*sqrt(sigma2 + x*x)` against `x*x`), so the
// uncalibrated arm here is that call with a null matrix and **not**
// `quantize_row_iq4_nl_ref`, which is the deterministic-file path
// `llama-quantize` never reaches.
//
// The block is the whole super-block (QK4_NL = 32), so the scale plane IQ4_XS
// carries does not exist and `sigma2` is the block's own. D10's departure is
// the same one `packQ41Row` makes: the final assignment is against the stored
// half of d.
//
// **What this writes is ggml's record**, so that the oracle can compare it
// with `quantize_iq4_nl` byte for byte; `pairIQ4NLRow` then rearranges the
// row into the layout the device reads, which is a permutation and is tested
// as one.
func packIQ4NLRow(dst []byte, row, qw []float32) {
	const g = 32
	const ntry = 7
	values := gguf.IQ4NLValues()
	var w [g]float32
	var lv [g]uint8
	for b := 0; b*g < len(row); b++ {
		blk := row[b*g : (b+1)*g]
		var sum2 float32
		for _, v := range blk {
			sum2 += v * v
		}
		sigma2 := 2 * sum2 / float32(g)
		for i, v := range blk {
			if qw != nil {
				w[i] = qw[b*g+i] * sqrt32(sigma2+v*v)
			} else {
				w[i] = v * v
			}
		}
		var amax, max float32
		for _, v := range blk {
			if ax := abs32(v); ax > amax {
				amax, max = ax, v
			}
		}
		rec := dst[b*18 : (b+1)*18]
		for i := range rec {
			rec[i] = 0
		}
		// ggml's GROUP_MAX_EPS. A block with no magnitude stores a zero
		// scale, and every level then reconstructs to zero whichever one it
		// is; ggml leaves `L` at whatever the previous block wrote, so the
		// nibbles are the one thing here that is deliberately not ggml's.
		if amax < 1e-15 {
			for i := range lv {
				lv[i] = uint8(bestIQ4NLIndex(values, 0))
			}
			packIQ4NLNibbles(rec[2:], lv[:])
			continue
		}
		// The sign convention is ggml's and it is not a typo: `values[0]` is
		// -127 against a top level of +113, so the codebook is not symmetric
		// and `-max/values[0]` is a different starting assignment from
		// `max/values[0]` rather than the same one mirrored.
		d := -max / float32(values[0])
		id := 1 / d
		var sumqx, sumq2 float32
		for i, v := range blk {
			q := float32(values[bestIQ4NLIndex(values, id*v)])
			sumqx += w[i] * q * v
			sumq2 += w[i] * q * q
		}
		if sumq2 > 0 {
			d = sumqx / sumq2
		} else {
			d = 0
		}
		best := d * sumqx
		for itry := -ntry; itry <= ntry; itry++ {
			id := (float32(itry) + float32(values[0])) / max
			sumqx, sumq2 = 0, 0
			for i, v := range blk {
				q := float32(values[bestIQ4NLIndex(values, id*v)])
				sumqx += w[i] * q * v
				sumq2 += w[i] * q * q
			}
			if sumq2 > 0 && sumqx*sumqx > best*sumq2 {
				d = sumqx / sumq2
				best = d * sumqx
			}
		}
		dh := safetensors.F32ToF16(d)
		df := safetensors.F16ToF32(dh)
		id = 0
		if df != 0 {
			id = 1 / df
		}
		for i, v := range blk {
			lv[i] = uint8(bestIQ4NLIndex(values, id*v))
		}
		binary.LittleEndian.PutUint16(rec[0:], dh)
		packIQ4NLNibbles(rec[2:], lv[:])
	}
}

// packIQ4NLNibbles is the pair split every block-32 format here uses:
// element j in the low nibble of byte j and element j+16 in its high one.
func packIQ4NLNibbles(dst []byte, lv []uint8) {
	for j := 0; j < 16; j++ {
		dst[j] = lv[j]&0xF | lv[j+16]&0xF<<4
	}
}

// pairIQ4NLRow rewrites a row of ggml `block_iq4_nl` records in place into
// **our** layout: blocks in pairs of 36 bytes — the two scales, then the two
// blocks' sixteen nibble bytes each.
//
// # Why the bank stops being ggml's here, and only here
//
// The MoE bank is staged as the checkpoint's own bytes, which is what makes
// ggml's record layout a constraint on the kernels rather than a choice. A
// **transcoded** tensor has no such constraint — the bytes are ours the
// moment we write them — and for this one format the difference is measured
// and large, in two directions that a single arrangement has to satisfy at
// once.
//
// An IQ4_NL block is eighteen bytes: a scale and sixteen nibble pairs. That
// is not a multiple of four, so every other block starts two bytes into a
// word and a payload dword comes out of a **two-word rotating window** — the
// arrangement Q8_0's 34-byte block already forced on this kernel. With the
// codebook in LDS the decode arm still read its bank at 180.2 GB/s where
// Q4_1's 20-byte block reads 211.4, on 10% fewer bytes.
//
// The obvious fix is a **planar** row — every scale, then every block's
// nibbles — and it is worth 180.2 → 203.3 GB/s at decode and **a 36% loss at
// prefill** (the `down` GEMM 2673 → 3637 us a layer at 512 tokens). The two
// kernels want opposite things: a GEMV lane group walks one row and reads the
// scale plane once per four dwords, while the GEMM's slab unpack touches one
// 32-element block of sixty-four *different* rows per K-step, so splitting a
// block's scale from its nibbles doubles the cache lines it opens.
//
// **Pairs satisfy both.** Two blocks are 4 + 32 = 36 bytes, which is a
// multiple of four — so every access is aligned, and a block's scale is
// within the same 36 bytes as its nibbles. Decode reads it at 205.6 GB/s and
// prefill at 2522 us a layer, both better than either of the other two
// arrangements and better than the Q5_1 bank this replaces.
//
// **It is a permutation, not a re-quantisation.** The levels, the scales and
// the bits are exactly what `packIQ4NLRow` wrote, and
// `TestIQ4NLPairIsAPermutation` inverts it back to ggml's records byte for
// byte — which is what keeps `TestMoEPackersAgainstGGML`'s bit-exactness
// against llama.cpp meaningful about the bank the device actually reads.
func pairIQ4NLRow(dst []byte) {
	const blk = 18
	nb := len(dst) / blk
	out := make([]byte, len(dst))
	for p := 0; p < nb/2; p++ {
		a, b := dst[p*2*blk:], dst[(p*2+1)*blk:]
		rec := out[p*36:]
		copy(rec[0:2], a[0:2])
		copy(rec[2:4], b[0:2])
		copy(rec[4:20], a[2:18])
		copy(rec[20:36], b[2:18])
	}
	copy(dst, out)
}

// unpairIQ4NLRow is the inverse, for the test that the pair is one.
func unpairIQ4NLRow(dst []byte) {
	const blk = 18
	nb := len(dst) / blk
	out := make([]byte, len(dst))
	for p := 0; p < nb/2; p++ {
		rec := dst[p*36:]
		a, b := out[p*2*blk:], out[(p*2+1)*blk:]
		copy(a[0:2], rec[0:2])
		copy(b[0:2], rec[2:4])
		copy(a[2:18], rec[4:20])
		copy(b[2:18], rec[20:36])
	}
	copy(dst, out)
}

// bestIQ4NLIndex is ggml's `best_index_int8` over a sorted codebook: the
// level nearest x, by a binary search rather than a scan.
func bestIQ4NLIndex(values [16]int8, x float32) int {
	if x <= float32(values[0]) {
		return 0
	}
	if x >= float32(values[15]) {
		return 15
	}
	ml, mu := 0, 15
	for mu-ml > 1 {
		mav := (ml + mu) / 2
		if x < float32(values[mav]) {
			mu = mav
		} else {
			ml = mav
		}
	}
	if x-float32(values[mu-1]) < float32(values[mu])-x {
		return mu - 1
	}
	return mu
}
