package llm

import (
	"math"
	"math/rand"
	"testing"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/safetensors"
)

// P4's writers, gated the way L8c-4 gated its bank: **the format is the fit**.
//
// A transcode has two halves that can be wrong independently — the levels
// chosen, and where the bits are put — and only the second is a format
// question. So every test here packs a row, reads it back through
// `gguf.Dequantize` (which is the transcription of `dequantize_row_*` the
// shaders were written against), and demands it equal what the encoder itself
// says the block reconstructs to. Not a tolerance: an equality. If the two
// ever disagree the bank is not the format and every perplexity number
// measured on it is measuring something nobody wrote down.

// moebankRow is a plausible weight row: heavy-tailed, a few outliers, and a
// deterministic seed so a failure is reproducible.
func moebankRow(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		v := float32(r.NormFloat64()) * 0.02
		if r.Intn(97) == 0 {
			v *= 12 // the outliers a scale search exists for
		}
		x[i] = v
	}
	return x
}

func TestPackKRowIsGgmlsLayout(t *testing.T) {
	for _, c := range []struct {
		name string
		to   gguf.Type
		bits int
	}{
		{"q4_k", gguf.Q4_K, 4},
		{"q5_k", gguf.Q5_K, 5},
	} {
		t.Run(c.name, func(t *testing.T) {
			const k = 2560 // the shared expert's gate and up
			for _, mode := range []string{"rtn", "search"} {
				row := moebankRow(k, 7)
				qw := moebankRow(k, 11)
				for i := range qw {
					qw[i] = float32(math.Abs(float64(qw[i]))) + 1e-6
				}
				if mode == "rtn" {
					qw = nil
				}
				sim := QuantSim{Bits: c.bits, Group: 32, Mode: mode, Asym: true}
				enc := newAsymEnc(sim, asymSuperBlocks)
				dst := make([]byte, k/c.to.BlockElems()*c.to.BlockBytes())
				if err := packKRow(dst, row, qw, enc, c.to); err != nil {
					t.Fatal(err)
				}

				got, err := gguf.Dequantize(c.to, dst, k, nil)
				if err != nil {
					t.Fatal(err)
				}
				// The expected side re-runs the same encoder over the same
				// super-blocks and asks it what it stored.
				want := make([]float32, 0, k)
				check := newAsymEnc(sim, asymSuperBlocks)
				for b := 0; b*256 < k; b++ {
					var w []float32
					if qw != nil {
						w = qw[b*256 : (b+1)*256]
					}
					check.Encode(row[b*256:(b+1)*256], w)
					for i := 0; i < 256; i++ {
						want = append(want, check.Dequant(i))
					}
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%s/%s element %d: the bytes read back %g, the fit stored %g",
							c.name, mode, i, got[i], want[i])
					}
				}
				// And it is a real quantisation rather than a zeroed buffer.
				var num, den float64
				for i, v := range row {
					d := float64(v - got[i])
					num += d * d
					den += float64(v) * float64(v)
				}
				rms := math.Sqrt(num / den)
				if rms == 0 || rms > 0.1 {
					t.Fatalf("%s/%s: relative rms %.4g, want a real fit", c.name, mode, rms)
				}
				t.Logf("%s/%s: %d elements, relative rms %.4g", c.name, mode, k, rms)
			}
		})
	}
}

func TestPackQ51RowIsGgmlsLayout(t *testing.T) {
	const k = 640 // the down projection's row, the one no K-quant divides
	for _, mode := range []string{"rtn", "imatrix"} {
		row := moebankRow(k, 13)
		var qw []float32
		if mode == "imatrix" {
			qw = moebankRow(k, 17)
			for i := range qw {
				qw[i] = float32(math.Abs(float64(qw[i]))) + 1e-6
			}
		}
		dst := make([]byte, k/32*24)
		packQ51Row(dst, row, qw)
		got, err := gguf.Dequantize(gguf.Q5_1, dst, k, nil)
		if err != nil {
			t.Fatal(err)
		}
		// Every value has to be one of the 32 the block can represent, which
		// is the layout check that does not need a second implementation:
		// read d and m out of the record and confirm (v - m)/d is an integer
		// in [0, 31] to the last bit of a float32.
		for b := 0; b*32 < k; b++ {
			d := f16at(dst[b*24 : b*24+2])
			m := f16at(dst[b*24+2 : b*24+4])
			for i := 0; i < 32; i++ {
				v := got[b*32+i]
				if d == 0 {
					if v != m {
						t.Fatalf("%s block %d element %d: zero scale should read back the min %g, got %g", mode, b, i, m, v)
					}
					continue
				}
				l := (v - m) / d
				if l != float32(math.Round(float64(l))) || l < 0 || l > 31 {
					t.Fatalf("%s block %d element %d: %g is level %g of [0,31], which is not a Q5_1 value", mode, b, i, v, l)
				}
			}
		}
		var num, den float64
		for i, v := range row {
			d := float64(v - got[i])
			num += d * d
			den += float64(v) * float64(v)
		}
		rms := math.Sqrt(num / den)
		// A loose bound on purpose: this asks "is it a fit at all", and the
		// tight check is the level-membership loop above. Five bits over a
		// heavy-tailed block of 32 lands near 5%.
		if rms == 0 || rms > 0.08 {
			t.Fatalf("q5_1/%s: relative rms %.4g, want a real fit", mode, rms)
		}
		t.Logf("q5_1/%s: %d elements, relative rms %.4g", mode, k, rms)
	}
}

// TestPackQ51RtnIsGgmlsReference pins the uncalibrated arm to
// `quantize_row_q5_1_ref` itself: d = (max-min)/31 and m = min over the block,
// with no clamp of the min to zero — the departure that would make this Q5_0.
func TestPackQ51RtnIsGgmlsReference(t *testing.T) {
	const k = 64
	row := moebankRow(k, 19)
	dst := make([]byte, k/32*24)
	packQ51Row(dst, row, nil)
	for b := 0; b*32 < k; b++ {
		blk := row[b*32 : (b+1)*32]
		lo, hi := blk[0], blk[0]
		for _, v := range blk {
			lo, hi = minF(lo, v), maxF(hi, v)
		}
		wantD := f16of((hi - lo) / 31)
		wantM := f16of(lo)
		if d, m := f16at(dst[b*24:b*24+2]), f16at(dst[b*24+2:b*24+4]); d != wantD || m != wantM {
			t.Fatalf("block %d: (d, m) = (%g, %g), ggml's reference is (%g, %g)", b, d, m, wantD, wantM)
		}
	}
}

// TestMoEBankPlanIsACeiling is the grammar's one surprising rule, and the one
// that makes `down_exps=q5_1` mean the five Q8_0 layers and nothing else.
func TestMoEBankPlanIsACeiling(t *testing.T) {
	p, err := ParseMoEBankPlan("gate_shexp=q4_k,down_exps=q5_1", "imatrix")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		have gguf.Type
		want gguf.Type
		move bool
	}{
		{"blk.3.ffn_gate_shexp.weight", gguf.Q8_0, gguf.Q4_K, true},
		{"blk.3.ffn_up_shexp.weight", gguf.Q8_0, gguf.Q8_0, false},   // not named
		{"blk.2.ffn_down_exps.weight", gguf.Q8_0, gguf.Q5_1, true},   // one of the five
		{"blk.3.ffn_down_exps.weight", gguf.Q5_1, gguf.Q5_1, false},  // already there
		{"blk.3.ffn_gate_exps.weight", gguf.Q4_K, gguf.Q4_K, false},  // not named
		{"blk.3.ffn_down_shexp.weight", gguf.Q8_0, gguf.Q8_0, false}, // not named
	} {
		got, move := p.For(c.name, c.have)
		if got != c.want || move != c.move {
			t.Errorf("%s at %s: plan says (%s, %v), want (%s, %v)", c.name, c.have, got, move, c.want, c.move)
		}
	}
	if _, err := ParseMoEBankPlan("down_exps=q4_k", "imatrix"); err != nil {
		t.Fatalf("the plan parses; it is the transcode that refuses: %v", err)
	}
	if _, err := ParseMoEBankPlan("down_exps=q8_0", "imatrix"); err == nil {
		t.Error("q8_0 is not a format the kernels read as a target, and the plan should say so")
	}
	if _, err := ParseMoEBankPlan("nope=q4_k", "imatrix"); err == nil {
		t.Error("an unknown family should be refused")
	}
}

// f16at reads a stored half out of a record, which is what makes the layout
// checks above independent of the packer that wrote it.
func f16at(b []byte) float32 {
	return safetensors.F16ToF32(uint16(b[0]) | uint16(b[1])<<8)
}

// f16of is a float as it survives the round trip through a stored half.
func f16of(v float32) float32 { return safetensors.F16ToF32(safetensors.F32ToF16(v)) }
func minF(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
func maxF(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
