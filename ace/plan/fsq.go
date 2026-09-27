package plan

// FSQ is the 5 Hz LM's audio-code codebook (vector_quantize_pytorch's
// ResidualFSQ with one quantizer): a code index is six mixed-radix digits
// over levels 8·8·8·5·5·5 = 64,000, each digit d mapped to d·(2/(L−1)) − 1
// (ResidualFSQ builds its FSQs with preserve_symmetry, so the grid is
// symmetric: an 8-level digit runs −1, −5/7, …, 1), and the six values
// projected to 2048 by project_out.
// The LM's vocabulary holds 65,535 code tokens; upstream clamps an index to
// FSQCodes − 1 before it gets here.

import (
	"fmt"
	"math"
)

// FSQLevels are the config's fsq_input_levels.
var FSQLevels = [6]int{8, 8, 8, 5, 5, 5}

// FSQCodes is the number of valid code indices.
const FSQCodes = 8 * 8 * 8 * 5 * 5 * 5

// FSQCode is the codebook row under index: FSQ._indices_to_codes.
func FSQCode(index int) ([6]float32, error) {
	var out [6]float32
	if index < 0 || index >= FSQCodes {
		return out, fmt.Errorf("plan: FSQ index %d out of [0, %d)", index, FSQCodes)
	}
	basis := 1
	for i, l := range FSQLevels {
		d := (index / basis) % l
		step := float32(2) / float32(l-1)
		out[i] = float32(float32(d)*step) - 1
		basis *= l
	}
	return out, nil
}

// FSQOutput is get_output_from_indices for one index: the code through
// project_out, whose weight is [2048, 6] row-major and bias [2048].
func FSQOutput(index int, w, b []float32) ([]float32, error) {
	code, err := FSQCode(index)
	if err != nil {
		return nil, err
	}
	if len(w) != len(b)*6 {
		return nil, fmt.Errorf("plan: project_out weight has %d values for %d outputs", len(w), len(b))
	}
	out := make([]float32, len(b))
	for o := range out {
		var s float32
		for i, c := range code {
			s += float32(w[o*6+i] * c)
		}
		out[o] = s + b[o]
	}
	return out, nil
}

// FSQIndex quantizes one pooled row's project_in output z (six values) as
// the audio tokenizer's ResidualFSQ does (one quantizer, scale 1): its
// bound_hard_clamp soft clamp tanh(z/c)·c with c = 1 + 1/(L−1), then FSQ's
// symmetry-preserving bound with the hard clamp, bracket = floor((L−1)·(z+1)/2
// + 0.5), and the index the mixed-radix sum of the brackets. It returns the
// index and, per digit, how far the bracket's argument sat from the floor's
// boundary (the fraction's distance to 0 or 1): what an fp16 path can flip.
func FSQIndex(z [6]float32) (index int, margin float32) {
	basis := 1
	margin = 1
	for i, l := range FSQLevels {
		lm1 := float32(l - 1)
		c := 1 + 1/lm1
		v := float32(math.Tanh(float64(z[i]/c))) * c
		v = max(-1, min(1, v))
		arg := lm1*(v+1)/2 + 0.5
		b := float32(math.Floor(float64(arg)))
		frac := arg - b
		margin = min(margin, frac, 1-frac)
		index += int(b) * basis
		basis *= l
	}
	return index, margin
}
