package audiobox

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// loadTargets reads config.json's target_transform: each axis's mean and
// std, which turn a head's output back into a 1-10 score.
func loadTargets(dir string) (map[string][2]float64, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c struct {
		NthLayer  int `json:"nth_layer"`
		ProjLayer int `json:"proj_num_layer"`
		Target    map[string]struct {
			Mean float64 `json:"mean"`
			Std  float64 `json:"std"`
		} `json:"target_transform"`
		Normalize bool `json:"normalize_embed"`
		Weighted  bool `json:"use_weighted_layer_sum"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("audiobox: %s: %w", dir, err)
	}
	if c.NthLayer != layers+1 || c.ProjLayer != 5 || !c.Normalize || !c.Weighted {
		return nil, fmt.Errorf("audiobox: config %+v is not the one this port is built for", c)
	}
	out := map[string][2]float64{}
	for _, a := range Axes {
		t, ok := c.Target[a]
		if !ok {
			return nil, fmt.Errorf("audiobox: no target_transform for %s", a)
		}
		out[a] = [2]float64{t.Mean, t.Std}
	}
	return out, nil
}

// score is one window's head for one axis, from the window's 13 pooled
// hidden states (each the mean over its valid frames): the softmax-weighted
// sum, L2-normalised, through the MLP, de-normalised.
func (h *head) score(pooled [layers + 1][]float32) float64 {
	x := make([]float64, hidden)
	for l, m := range pooled {
		for i, v := range m {
			x[i] += h.layerW[l] * float64(v)
		}
	}
	var n float64
	for _, v := range x {
		n += v * v
	}
	n = math.Max(math.Sqrt(n), 1e-12) // F.normalize's eps
	for i := range x {
		x[i] /= n
	}
	y := make([]float64, hidden)
	for b := 0; b < 4; b++ {
		for o := 0; o < hidden; o++ {
			s := float64(h.b[b][o])
			row := h.w[b][o*hidden : (o+1)*hidden]
			for i, v := range x {
				s += float64(row[i]) * v
			}
			y[o] = s
		}
		var mean, vr float64
		for _, v := range y {
			mean += v
		}
		mean /= hidden
		for _, v := range y {
			vr += (v - mean) * (v - mean)
		}
		inv := 1 / math.Sqrt(vr/hidden+lnEps)
		for o, v := range y {
			z := (v-mean)*inv*float64(h.lnG[b][o]) + float64(h.lnB[b][o])
			x[o] = 0.5 * z * math.Erfc(-z/math.Sqrt2)
		}
	}
	s := float64(h.outB)
	for i, v := range x {
		s += float64(h.outW[i]) * v
	}
	return s*h.std + h.mean
}
