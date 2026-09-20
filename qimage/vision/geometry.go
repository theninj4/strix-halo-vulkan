package vision

import (
	"math"

	"strix-halo-vulkan/zimage/qwen"
)

// The tower's two geometric pieces, together because they share one fact:
// **the token order is 2x2-block-major, not raster**. Both the position
// grid and the rope table are indexed by that order, and so is the pixel
// packing; get it wrong and every stage still produces plausible numbers.

// blockOrder maps a token index to the (row, col) patch it holds.
//
// The reference builds it by reshaping an h x w index grid to
// (h/m, m, w/m, m), transposing the middle two axes and flattening — which
// is "walk the merge blocks in raster order, and inside each block walk its
// m x m patches in raster order". Written out rather than reshaped, because
// the transpose is the entire content of the layout and a reshape hides it.
func blockOrder(gridH, gridW, merge int) [][2]int {
	out := make([][2]int, 0, gridH*gridW)
	for by := 0; by < gridH/merge; by++ {
		for bx := 0; bx < gridW/merge; bx++ {
			for dy := 0; dy < merge; dy++ {
				for dx := 0; dx < merge; dx++ {
					out = append(out, [2]int{by*merge + dy, bx*merge + dx})
				}
			}
		}
	}
	return out
}

// positionEmbedding resamples the learned 48x48 grid onto this image's
// grid, bilinearly with align_corners.
//
// This is the piece with the most ways to be subtly wrong, so it is written
// the way the reference computes it rather than the way one would write a
// resampler: four (index, weight) pairs per patch, the weights being the
// bilinear products, and the four indices the corners of the source cell.
// align_corners means the *endpoints* map to endpoints, so the source
// coordinate of destination index i out of n is i*(48-1)/(n-1) — and for
// n == 1 that is 0 rather than a division by zero.
func (m *Model) positionEmbedding(gridH, gridW int) (*qwen.Mat, error) {
	side := m.Cfg.GridPerSide()
	dim := m.Cfg.HiddenSize
	order := blockOrder(gridH, gridW, m.Cfg.SpatialMergeSize)

	out := qwen.NewMat(len(order), dim)
	for t, rc := range order {
		sy := srcCoord(rc[0], gridH, side)
		sx := srcCoord(rc[1], gridW, side)
		y0, x0 := int(math.Floor(sy)), int(math.Floor(sx))
		y1, x1 := min(y0+1, side-1), min(x0+1, side-1)
		fy, fx := sy-float64(y0), sx-float64(x0)
		dst := out.Row(t)
		for _, c := range [4]struct {
			idx int
			w   float64
		}{
			{y0*side + x0, (1 - fy) * (1 - fx)},
			{y0*side + x1, (1 - fy) * fx},
			{y1*side + x0, fy * (1 - fx)},
			{y1*side + x1, fy * fx},
		} {
			if c.w == 0 {
				continue
			}
			row := m.PosEmbed[c.idx*dim : (c.idx+1)*dim]
			for i, v := range row {
				dst[i] += float32(c.w * float64(v))
			}
		}
	}
	return out, nil
}

// srcCoord is align_corners=true's destination-to-source map.
func srcCoord(i, n, side int) float64 {
	if n <= 1 {
		return 0
	}
	return float64(i) * float64(side-1) / float64(n-1)
}

// Rope is the tower's axial 2-D rotary table: cos and sin of [tokens,
// headDim], already in the layout the rotation reads.
//
// The frequencies are built over headDim/2 = 36 with a stride of 2, giving
// 18 per axis; h's 18 and w's 18 are concatenated to 36 and the 36
// duplicated to 72. The rotation is then **NeoX halves** -- the first half
// of the head against the second -- and not the adjacent-pair complex
// rotation the DiT's RoPE uses. Both conventions live in this repository
// and swapping them is a negative control in dit_test.go for exactly that
// reason.
type Rope struct {
	Cos, Sin *qwen.Mat
	// Pairs selects the *other* convention -- adjacent-pair complex
	// rotation, which is what qimage/dit's RoPE does. It is wrong here and
	// is selectable for exactly that reason: the two conventions live in
	// this repository, a swap between them produces numbers rather than an
	// error, and a bound that has never been shown to catch one is not a
	// bound. vision_test.go's negative control is its only caller.
	Pairs bool
}

// NewRope builds the table for one image's grid.
func NewRope(cfg *Config, gridH, gridW int) *Rope {
	hd := cfg.HeadDim()
	spatial := hd / 2
	nf := spatial / 2 // 18
	inv := make([]float64, nf)
	for i := range inv {
		inv[i] = 1 / math.Pow(10000, float64(2*i)/float64(spatial))
	}
	order := blockOrder(gridH, gridW, cfg.SpatialMergeSize)
	cos := qwen.NewMat(len(order), hd)
	sin := qwen.NewMat(len(order), hd)
	for t, rc := range order {
		c, s := cos.Row(t), sin.Row(t)
		for i, f := range inv {
			ah := float64(rc[0]) * f
			aw := float64(rc[1]) * f
			// freq_hw = [h's 18 ++ w's 18]; the table is freq_hw twice.
			for _, base := range [2]int{0, spatial} {
				c[base+i] = float32(math.Cos(ah))
				s[base+i] = float32(math.Sin(ah))
				c[base+nf+i] = float32(math.Cos(aw))
				s[base+nf+i] = float32(math.Sin(aw))
			}
		}
	}
	return &Rope{Cos: cos, Sin: sin}
}

// Apply rotates every head of every row in place, NeoX halves.
func (r *Rope) Apply(x *qwen.Mat, heads, headDim int) {
	half := headDim / 2
	for t := 0; t < x.Rows; t++ {
		row := x.Row(t)
		cos, sin := r.Cos.Row(t), r.Sin.Row(t)
		for h := 0; h < heads; h++ {
			v := row[h*headDim : (h+1)*headDim]
			if r.Pairs {
				for i := 0; i < half; i++ {
					a, b := float64(v[2*i]), float64(v[2*i+1])
					v[2*i] = float32(a*float64(cos[2*i]) - b*float64(sin[2*i]))
					v[2*i+1] = float32(b*float64(cos[2*i+1]) + a*float64(sin[2*i+1]))
				}
				continue
			}
			for i := 0; i < half; i++ {
				a, b := float64(v[i]), float64(v[i+half])
				v[i] = float32(a*float64(cos[i]) - b*float64(sin[i]))
				v[i+half] = float32(b*float64(cos[i+half]) + a*float64(sin[i+half]))
			}
		}
	}
}

// Patchify turns an image into the processor's `pixel_values`: one row per
// patch, in block-major order, each row channels-then-temporal-then-pixels.
//
// img is [3, H, W] already normalized to [-1, 1] (the processor's mean and
// std are both 0.5, so that is exactly its output), and the temporal axis is
// the same frame repeated TemporalPatchSize times — a still image entering a
// video model.
func (cfg *Config) Patchify(img []float32, h, w int) (*qwen.Mat, int, int, error) {
	p := cfg.PatchSize
	gridH, gridW := h/p, w/p
	order := blockOrder(gridH, gridW, cfg.SpatialMergeSize)
	out := qwen.NewMat(len(order), cfg.PatchElems())
	plane := h * w
	for t, rc := range order {
		dst := out.Row(t)
		i := 0
		for c := 0; c < cfg.InChannels; c++ {
			for tp := 0; tp < cfg.TemporalPatchSize; tp++ {
				for py := 0; py < p; py++ {
					srcRow := (rc[0]*p + py) * w
					for px := 0; px < p; px++ {
						dst[i] = img[c*plane+srcRow+rc[1]*p+px]
						i++
					}
				}
			}
		}
	}
	return out, gridH, gridW, nil
}
