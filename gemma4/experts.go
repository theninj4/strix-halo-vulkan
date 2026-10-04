package gemma4

// The routed experts as GGUF Q8_0 banks (R4), for the LLM's grouped MoE GEMM
// (`llm/gpu_moe.go`, `shaders/llm_moe_gemm.comp` QFMT=FMT_Q8_0).
//
// Rune ships bf16 only, and the user's footprint target is Q8 (2026-10-03),
// so the experts are quantised here. Q8_0 is the format the LLM's kernels
// already read: blocks of 32 weights, an fp16 scale and 32 int8s, 8.5 bits a
// weight. The layout needs no transposition. HF's `gate_up_proj` is
// [experts][2 x 704][2816] with gate the first half and up the second (its
// forward is `.chunk(2, dim=-1)`), and `down_proj` is [experts][2816][704].
// Those are already a GGUF bank's [in, out, experts] in ggml order, where
// expert e's output row j is row e*out + j.
//
// `pre_feedforward_layernorm_2` is **not** folded into gate and up: its
// weights span eight orders of magnitude and folded they ruin the Q8_0
// groups; gemma4_post.comp applies it to the block's input instead.
//
// **`per_expert_scale` is folded into `down`.** Gemma 4's router multiplies
// each chosen expert's renormalised weight by a learned per-expert scalar,
// and the LLM's route kernel has no such step. The scalar multiplies that
// expert's whole output, so scaling its down rows before quantising is the
// same product, rounded once more in bf16 → fp32 → Q8_0, which is
// quantisation's own rounding anyway.

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/safetensors"
)

const q8Block = 32

// quantQ8_0 appends one row as Q8_0 blocks: ggml's quantize_row_q8_0_ref,
// d = max|x| / 127 stored as fp16, q = round(x / d) half away from zero
// (roundf), with the reciprocal taken from the fp32 d as ggml does.
func quantQ8_0(dst []byte, row []float32) []byte {
	for b := 0; b < len(row); b += q8Block {
		blk := row[b : b+q8Block]
		amax := float32(0)
		for _, v := range blk {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		d := amax / 127
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		h := safetensors.F32ToF16(d)
		dst = append(dst, byte(h), byte(h>>8))
		for _, v := range blk {
			dst = append(dst, byte(int8(math.Round(float64(v*id)))))
		}
	}
	return dst
}

// ExpertBanks is one layer's routed experts.
type ExpertBanks struct {
	Gate, Up, Down *gguf.Tensor
}

// LoadExperts quantises layer i's experts.
func LoadExperts(set *safetensors.Set, c *Config, i int) (*ExpertBanks, error) {
	pre := fmt.Sprintf("model.language_model.layers.%d.", i)
	gu, err := set.Get(pre + "experts.gate_up_proj")
	if err != nil {
		return nil, err
	}
	dn, err := set.Get(pre + "experts.down_proj")
	if err != nil {
		return nil, err
	}
	ps, err := set.Get(pre + "router.per_expert_scale")
	if err != nil {
		return nil, err
	}
	E, F, H := c.Experts, c.MoEInter, c.Hidden
	if want := []int{E, 2 * F, H}; !sameShape(gu.Shape, want) {
		return nil, fmt.Errorf("gemma4: %s is %v, want %v", gu.Name, gu.Shape, want)
	}
	if want := []int{E, H, F}; !sameShape(dn.Shape, want) {
		return nil, fmt.Errorf("gemma4: %s is %v, want %v", dn.Name, dn.Shape, want)
	}
	scale, err := ps.F32(nil)
	if err != nil {
		return nil, err
	}
	if len(scale) != E {
		return nil, fmt.Errorf("gemma4: %s has %d entries", ps.Name, len(scale))
	}
	guF, err := gu.F32(nil) // 128 x 1408 x 2816 = 507M floats, 2 GB; one layer at a time
	if err != nil {
		return nil, err
	}
	dnF, err := dn.F32(nil)
	if err != nil {
		return nil, err
	}
	rowBytes := func(n int) int { return n / q8Block * (2 + q8Block) }
	rbH, rbF := rowBytes(H), rowBytes(F)
	gate := make([]byte, E*F*rbH)
	up := make([]byte, E*F*rbH)
	down := make([]byte, E*H*rbF)
	// Every row lands at a fixed offset, so experts quantise in parallel:
	// quantQ8_0 appends into a zero-length slice capped at the row's bytes.
	parallelFor(E, func(e int) {
		base := e * 2 * F * H
		for j := range F {
			o := (e*F + j) * rbH
			quantQ8_0(gate[o:o:o+rbH], guF[base+j*H:base+(j+1)*H])
			quantQ8_0(up[o:o:o+rbH], guF[base+(F+j)*H:base+(F+j+1)*H])
		}
		row := make([]float32, F)
		for j := range H {
			src := dnF[(e*H+j)*F : (e*H+j+1)*F]
			for k, v := range src {
				row[k] = v * scale[e]
			}
			o := (e*H + j) * rbF
			quantQ8_0(down[o:o:o+rbF], row)
		}
	})
	mk := func(name string, in, out int, data []byte) *gguf.Tensor {
		return &gguf.Tensor{Name: fmt.Sprintf("blk.%d.%s", i, name), Type: gguf.Q8_0,
			Dims: []int64{int64(in), int64(out), int64(E)}, Data: data}
	}
	return &ExpertBanks{
		Gate: mk("ffn_gate_exps.weight", H, F, gate),
		Up:   mk("ffn_up_exps.weight", H, F, up),
		Down: mk("ffn_down_exps.weight", F, H, down),
	}, nil
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// parallelFor runs f(0..n-1) over every core.
func parallelFor(n int, f func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	next := atomic.Int64{}
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1) - 1)
				if i >= n {
					return
				}
				f(i)
			}
		}()
	}
	wg.Wait()
}

// tileQ8 lays a Q8_0 matrix ([rows][k], rows a multiple of 16) out as
// llm_moe_gemm.comp's Q8_TILED reads it: in each group of 16 rows, the
// blocks of one 32-wide K-step as one 544-byte tile, the 16 fp16 scales
// then the 16 rows' 32 int8s. A group occupies exactly its 16 rows' bytes,
// so the bank is Q8_0's size and every offset into it is unchanged.
func tileQ8(data []byte, rows, k int) []byte {
	kb := k / q8Block
	rowBytes := kb * (2 + q8Block)
	out := make([]byte, len(data))
	parallelFor(rows/16, func(gi int) {
		g := gi * 16
		base := g * rowBytes
		for b := range kb {
			tile := out[base+b*544:]
			for r := range 16 {
				src := data[(g+r)*rowBytes+b*(2+q8Block):]
				copy(tile[r*2:r*2+2], src[:2])
				copy(tile[32+r*q8Block:32+(r+1)*q8Block], src[2:2+q8Block])
			}
		}
	})
	return out
}
