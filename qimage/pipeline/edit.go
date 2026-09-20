package pipeline

// The condition-image half of an edit — IMAGE.md Q8.3.
//
// A generation's prefix is a prompt. An edit's is a prompt *and* one to ten
// reference images, and each of those is encoded twice: by the vision tower
// into the context the text tokens absorb (qimage/textenc's edit path), and
// by the VAE into latents the transformer attends over directly. This file
// is the second half — the geometry a condition image is resized to, and the
// packing that turns its latent grid into the `cond` rows Denoise prepends to
// the target at every step.
//
// One resize feeds both encoders, which is the fact that keeps the two
// halves consistent: the vision tower's patch grid and the VAE's latent grid
// are the *same numbers* (a 16x16 pixel tile is one latent token and one
// vision patch, and four of either make one VLM slot), so a condition image
// has one geometry and not two.

import (
	"fmt"
	"math"

	qvae "strix-halo-vulkan/qimage/vae"
	"strix-halo-vulkan/zimage/qwen"
	zvae "strix-halo-vulkan/zimage/vae"
)

// CondMultiple is the pixel granularity every condition image is snapped to.
// It is the VAE's 16 times the VLM's 2x2 grouping: below it a latent grid
// cannot be cut into whole slots.
const CondMultiple = 32

// CalcDimensions is diffusers' calculate_dimensions: the pixel width and
// height a condition image is resized to, preserving its aspect ratio at a
// target area and snapped to CondMultiple.
//
// The rounding is Python's, which breaks halves to even rather than away
// from zero — math.RoundToEven and not math.Round. It only bites on exact
// halves, which at this granularity means a source whose scaled side lands
// on a multiple of 16 that is not one of 32; rare, and free to get right.
func CalcDimensions(targetArea int, ratio float64) (int, int) {
	w := math.Sqrt(float64(targetArea) * ratio)
	h := w / ratio
	return int(math.RoundToEven(w/CondMultiple)) * CondMultiple,
		int(math.RoundToEven(h/CondMultiple)) * CondMultiple
}

// PackLatents flattens a [1, C, h, w] latent grid into one row per token, in
// raster order — diffusers' _pack_latents, which for 2.1 is a plain spatial
// flatten because the DiT takes latents unpatched.
func PackLatents(z *zvae.Tensor) *qwen.Mat {
	out := qwen.NewMat(z.H*z.W, z.C)
	for c := 0; c < z.C; c++ {
		plane := z.Plane(0, c)
		for i, v := range plane {
			out.Row(i)[c] = v
		}
	}
	return out
}

// UnpackLatents is the inverse, back to a [1, C, h, w] grid.
func UnpackLatents(m *qwen.Mat, h, w int) (*zvae.Tensor, error) {
	if m.Rows != h*w {
		return nil, fmt.Errorf("pipeline: %d latent rows for a %dx%d grid", m.Rows, h, w)
	}
	z := zvae.NewTensor(1, m.Cols, h, w)
	for c := 0; c < m.Cols; c++ {
		plane := z.Plane(0, c)
		for i := 0; i < m.Rows; i++ {
			plane[i] = m.Row(i)[c]
		}
	}
	return z, nil
}

// EncodeCondition turns one condition image into the rows the denoiser
// prepends: the VAE's posterior *mode* (never a sample — IMAGE.md decision
// 3, and it is what keeps a seeded edit reproducible), normalised per
// channel into the DiT's space, and packed.
//
// img is [1, 4, H, W] in [-1, 1] — all four channels, alpha included. The
// vision tower reads a copy of the same image flattened over white; this one
// is not flattened, and a port that flattens once and uses it twice is
// wrong in a way that only shows on transparent references.
func EncodeCondition(enc *qvae.Encoder, cfg *qvae.Config, img *zvae.Tensor) (*qwen.Mat, [3]int, error) {
	if img.H%CondMultiple != 0 || img.W%CondMultiple != 0 {
		return nil, [3]int{}, fmt.Errorf("pipeline: a %dx%d condition image is not a multiple of %d",
			img.W, img.H, CondMultiple)
	}
	z, err := enc.Encode(img)
	if err != nil {
		return nil, [3]int{}, fmt.Errorf("pipeline: encoding a condition image: %w", err)
	}
	cfg.Normalize(z)
	return PackLatents(z), [3]int{1, z.H, z.W}, nil
}

// ConcatConditions stacks several condition images' latents into the single
// `cond` matrix Denoise takes, in the order their runs appear in the prompt.
func ConcatConditions(conds []*qwen.Mat) (*qwen.Mat, error) {
	if len(conds) == 0 {
		return nil, nil
	}
	rows, cols := 0, conds[0].Cols
	for i, c := range conds {
		if c.Cols != cols {
			return nil, fmt.Errorf("pipeline: condition %d is %d wide, condition 0 is %d", i, c.Cols, cols)
		}
		rows += c.Rows
	}
	out := qwen.NewMat(rows, cols)
	at := 0
	for _, c := range conds {
		copy(out.Data[at:], c.Data)
		at += len(c.Data)
	}
	return out, nil
}
