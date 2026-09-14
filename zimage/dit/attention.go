package dit

import (
	"fmt"
	"math"
)

// Attention is Z-Image's single-stream self-attention: projections, per-head
// RMS norms on q and k, 3-D RoPE, then softmax attention over the whole
// sequence.
type Attention struct {
	Q, K, V, Out   *Linear
	NormQ, NormK   *RMSNorm
	Heads, HeadDim int
}

// Apply runs attention over x, which is [tokens, heads*headDim].
func (a *Attention) Apply(x *Mat, rope *RoPE) (*Mat, error) {
	q, err := a.Q.Apply(x)
	if err != nil {
		return nil, err
	}
	k, err := a.K.Apply(x)
	if err != nil {
		return nil, err
	}
	v, err := a.V.Apply(x)
	if err != nil {
		return nil, err
	}

	// The q/k norms are per head: their weight is headDim wide and the row
	// holds every head, so RMSNorm normalises each head's span on its own.
	if a.NormQ != nil {
		if _, err := a.NormQ.ApplyInPlace(q); err != nil {
			return nil, err
		}
	}
	if a.NormK != nil {
		if _, err := a.NormK.ApplyInPlace(k); err != nil {
			return nil, err
		}
	}
	if rope != nil {
		if err := rope.ApplyInPlace(q, a.Heads, a.HeadDim); err != nil {
			return nil, err
		}
		if err := rope.ApplyInPlace(k, a.Heads, a.HeadDim); err != nil {
			return nil, err
		}
	}

	ctx, err := a.scores(q, k, v)
	if err != nil {
		return nil, err
	}
	return a.Out.Apply(ctx)
}

// scores is the softmax attention itself, one head at a time.
func (a *Attention) scores(q, k, v *Mat) (*Mat, error) {
	n := q.Rows
	if k.Rows != n || v.Rows != n {
		return nil, fmt.Errorf("dit: attention operands disagree: q %s k %s v %s", q, k, v)
	}
	d := a.HeadDim
	scale := 1 / math.Sqrt(float64(d))
	out := NewMat(n, a.Heads*d)

	// One work item per (head, query token): the scores for one row are
	// computed, softmaxed and consumed without ever materialising the full
	// n x n matrix, which is also what the shader will have to do.
	parallelFor(a.Heads*n, func(w int) {
		h, i := w/n, w%n
		qi := q.Row(i)[h*d : (h+1)*d]
		scores := make([]float32, n)
		maxScore := float32(math.Inf(-1))
		for j := 0; j < n; j++ {
			kj := k.Row(j)[h*d : (h+1)*d]
			var s float32
			for t, qv := range qi {
				s += qv * kj[t]
			}
			s *= float32(scale)
			scores[j] = s
			if s > maxScore {
				maxScore = s
			}
		}
		var denom float32
		for j, s := range scores {
			e := float32(math.Exp(float64(s - maxScore)))
			scores[j] = e
			denom += e
		}
		dst := out.Row(i)[h*d : (h+1)*d]
		for j, wgt := range scores {
			if wgt == 0 {
				continue
			}
			wgt /= denom
			vj := v.Row(j)[h*d : (h+1)*d]
			for t := range dst {
				dst[t] += wgt * vj[t]
			}
		}
	})
	return out, nil
}

// FeedForward is SwiGLU: w2(silu(w1(x)) * w3(x)).
type FeedForward struct {
	W1, W2, W3 *Linear
}

// Apply runs the feed-forward.
func (f *FeedForward) Apply(x *Mat) (*Mat, error) {
	gate, err := f.W1.Apply(x)
	if err != nil {
		return nil, err
	}
	up, err := f.W3.Apply(x)
	if err != nil {
		return nil, err
	}
	parallelFor(gate.Rows, func(r int) {
		g, u := gate.Row(r), up.Row(r)
		for i, v := range g {
			g[i] = v / (1 + float32(math.Exp(float64(-v)))) * u[i]
		}
	})
	return f.W2.Apply(gate)
}

// Block is one Z-Image DiT transformer block.
type Block struct {
	AttnNorm1, AttnNorm2 *RMSNorm
	FFNNorm1, FFNNorm2   *RMSNorm
	Attn                 *Attention
	FFN                  *FeedForward
	AdaLN                *Linear // timestep embedding -> 4*dim of modulation
	Dim                  int
}

// Apply runs the block. x is [tokens, dim]; adaln is the timestep embedding,
// and is ignored by a block with no adaLN projection.
//
// A block without modulation -- the two context refiners, which diffusers
// builds with modulation=False -- is this block with its four modulation
// vectors absent rather than a different module: no scale on either branch
// input and both residuals ungated.
func (b *Block) Apply(x *Mat, adaln []float32, rope *RoPE) (*Mat, error) {
	if x.Cols != b.Dim {
		return nil, fmt.Errorf("dit: block takes %d features, got %s", b.Dim, x)
	}
	var scaleMSA, gateMSA, scaleMLP, gateMLP []float32
	if b.AdaLN != nil {
		modIn := &Mat{Rows: 1, Cols: len(adaln), Data: adaln}
		modMat, err := b.AdaLN.Apply(modIn)
		if err != nil {
			return nil, err
		}
		mod := modMat.Data
		if len(mod) != 4*b.Dim {
			return nil, fmt.Errorf("dit: adaLN produced %d values, want %d", len(mod), 4*b.Dim)
		}
		// The four chunks are scale_msa, gate_msa, scale_mlp, gate_mlp, in
		// that order. The scales are offset by one and the gates pass through
		// tanh, so a freshly trained block starts as the identity.
		scaleMSA = make([]float32, b.Dim)
		gateMSA = make([]float32, b.Dim)
		scaleMLP = make([]float32, b.Dim)
		gateMLP = make([]float32, b.Dim)
		for i := 0; i < b.Dim; i++ {
			scaleMSA[i] = 1 + mod[i]
			gateMSA[i] = float32(math.Tanh(float64(mod[b.Dim+i])))
			scaleMLP[i] = 1 + mod[2*b.Dim+i]
			gateMLP[i] = float32(math.Tanh(float64(mod[3*b.Dim+i])))
		}
	}

	h := x.Clone()
	if _, err := b.AttnNorm1.ApplyInPlace(h); err != nil {
		return nil, err
	}
	scaleRows(h, scaleMSA)

	attnOut, err := b.Attn.Apply(h, rope)
	if err != nil {
		return nil, err
	}
	if _, err := b.AttnNorm2.ApplyInPlace(attnOut); err != nil {
		return nil, err
	}
	out := x.Clone()
	gateAddRows(out, attnOut, gateMSA)

	h2 := out.Clone()
	if _, err := b.FFNNorm1.ApplyInPlace(h2); err != nil {
		return nil, err
	}
	scaleRows(h2, scaleMLP)
	ff, err := b.FFN.Apply(h2)
	if err != nil {
		return nil, err
	}
	if _, err := b.FFNNorm2.ApplyInPlace(ff); err != nil {
		return nil, err
	}
	gateAddRows(out, ff, gateMLP)
	return out, nil
}

// scaleRows multiplies every row elementwise by s; a nil s is the unmodulated
// block's missing scale, which is the identity rather than zero.
func scaleRows(x *Mat, s []float32) {
	if s == nil {
		return
	}
	parallelFor(x.Rows, func(r int) {
		row := x.Row(r)
		for i := range row {
			row[i] *= s[i]
		}
	})
}

// gateAddRows computes dst += gate * src, elementwise per row. A nil gate is
// the unmodulated block's ungated residual.
func gateAddRows(dst, src *Mat, gate []float32) {
	parallelFor(dst.Rows, func(r int) {
		d, s := dst.Row(r), src.Row(r)
		if gate == nil {
			for i := range d {
				d[i] += s[i]
			}
			return
		}
		for i := range d {
			d[i] += gate[i] * s[i]
		}
	})
}
