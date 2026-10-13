package dit

// The DiT's half of upstream's auto score (MUSIC.md A14b): get_lyric_score
// asks the decoder for its cross-attention weights (output_attentions, the
// eager path) and reads the heads in lyric_alignment_layers_config, whose
// attention over the lyric tokens follows where each is sung.
//
// Our attention is a flash kernel and never forms the probabilities, so
// they are rebuilt on the host from what it reads: after each listed layer,
// that layer's packed cross q (normed and scaled by log2(e)/√128, as the
// kernel scores in base 2) and the condition's packed k, both fp16 in the
// fragment tiling qkpack writes. A forward stops after the last listed
// layer: upstream runs all 32 ("enable_early_exit" computes the bound and
// never breaks), but nothing after layer 7 reaches layers 3-7's attention.

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// Config is the checkpoint's config.json.
func (g *GPU) Config() *Config { return g.cfg }

// EncLen is the condition sequence's length since the last Begin.
func (g *GPU) EncLen() int { return g.encLen }

// CrossProbs runs a forward of x_t at t (Step's inputs, after Begin) up to
// the last layer heads lists, and returns each listed (layer, head)'s
// cross-attention probabilities as [tokens][encoder rows], row-major: the
// softmax over the condition sequence for each DiT token.
func (g *GPU) CrossProbs(x, ctx *qwen.Mat, t float32, heads [][2]int) ([][]float32, error) {
	byLayer := map[int][]int{}
	last := -1
	for i, lh := range heads {
		if lh[0] < 0 || lh[0] >= len(g.ditL) || lh[1] < 0 || lh[1] >= g.dit.heads {
			return nil, fmt.Errorf("dit: no head %v", lh)
		}
		byLayer[lh[0]] = append(byLayer[lh[0]], i)
		last = max(last, lh[0])
	}
	gr, n, err := g.inputGraph(x, ctx, t)
	if err != nil {
		return nil, err
	}
	L := g.encLen
	out := make([][]float32, len(heads))
	group := g.dit.heads / g.dit.kv
	for l := 0; l <= last; l++ {
		if err := gr.layer(g.dit, g.ditL[l], n, g.vecs(l), true); err != nil {
			return nil, err
		}
		if len(byLayer[l]) == 0 {
			continue
		}
		if _, err := gr.submit(); err != nil {
			return nil, err
		}
		keys := map[int][]float32{}
		for _, i := range byLayer[l] {
			h := heads[i][1]
			kv := h / group
			k, ok := keys[kv]
			if !ok {
				k = g.unpackPlane(g.ditL[l].crossK+uint32(kv*g.encRows*headDim), L)
				keys[kv] = k
			}
			// hQ holds this layer's cross q until the next layer's self q.
			q := g.unpackPlane(g.hQ+uint32(h*g.rows*headDim), n)
			out[i] = softmax2(q, k, n, L)
		}
	}
	return out, nil
}

// unpackPlane reads rows of one head's plane at off in qkpack's fragment
// tiling (tile t holds rows 16t.., component chunk c at c·256, element
// (row, component) at m·16 + k) as fp32 [rows][128].
func (g *GPU) unpackPlane(off uint32, rows int) []float32 {
	padded := roundUp(rows, tile)
	raw := g.hbuf.ReadUint16At(int(off), padded*headDim)
	out := make([]float32, rows*headDim)
	for r := 0; r < rows; r++ {
		blk := raw[(r/tile)*tile*headDim:]
		m := r % tile
		for c := 0; c < headDim/tile; c++ {
			for k := 0; k < tile; k++ {
				out[r*headDim+c*tile+k] = safetensors.F16ToF32(blk[c*tile*tile+m*tile+k])
			}
		}
	}
	return out
}

// softmax2 is each q row's softmax over the keys, the logits in base 2.
func softmax2(q, k []float32, n, L int) []float32 {
	p := make([]float32, n*L)
	parallel(n, func(i int) {
		qi := q[i*headDim : (i+1)*headDim]
		row := p[i*L : (i+1)*L]
		mx := float32(math.Inf(-1))
		for j := 0; j < L; j++ {
			kj := k[j*headDim : (j+1)*headDim]
			var s float32
			for d, v := range qi {
				s += v * kj[d]
			}
			row[j] = s
			mx = max(mx, s)
		}
		var sum float64
		for j, v := range row {
			e := math.Exp2(float64(v - mx))
			row[j] = float32(e)
			sum += e
		}
		for j := range row {
			row[j] = float32(float64(row[j]) / sum)
		}
	})
	return p
}
