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
// format the down mode has no arm for — and it is written down in `LLM2.md`
// rather than built here.
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
	"sort"
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
// are the up mode's; Q5_1 is the down mode's. Asking for a format the kernel
// cannot read would stage a bank that reads as noise.
var moeBankFormats = map[string]gguf.Type{
	"q4_k": gguf.Q4_K,
	"q5_k": gguf.Q5_K,
	"q5_1": gguf.Q5_1,
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
			return MoEBankPlan{}, fmt.Errorf("llm: MoE family %s cannot be staged as %q; the kernels read q4_k, q5_k and q5_1", fam, want)
		}
		p.to[fam] = t
	}
	return p, nil
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
//	D19 + this     4.0970   +1.69%   4.279
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
const ShippedMoEBank = "gate_shexp=q4_k,up_shexp=q4_k,down_shexp=q5_1,down_exps=q5_1"

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
			}
		}
	})
	if perr != nil {
		return nil, perr
	}
	return out, nil
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
// The fit is **not** ggml's. `quantize_row_q5_1_ref` is plain min/max, which
// is what llama-quantize does because the legacy quants predate the
// importance matrix and were never wired to it. Here the matrix is available
// and the same weighted search the K-quants use applies unchanged to a single
// block — `makeQkxQuants` fits (scale, min) against a per-element weight —
// so the calibrated arm uses it and the `rtn` arm is ggml's own path. Both
// dequantise identically; only the levels differ.
//
// One asymmetry between the arms is worth stating rather than discovering:
// `makeQkxQuants` clamps the group's min to zero, which is ggml's *K-quant*
// convention, where `quantize_row_q5_1_ref` keeps the block's true min. On a
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
