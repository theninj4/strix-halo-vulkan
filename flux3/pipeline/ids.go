package pipeline

import "strix-halo-vulkan/zimage/qwen"

// ImageRopeScale scales the image tokens' row and column ids (the
// pipeline's image_rope_scale: the model was trained on half-step ids).
const ImageRopeScale = 0.5

// ImageIDs is a token grid's position ids [h*w, 4], row-major: (0, row·s,
// col·s, l) with s = ImageRopeScale and l = 0 for the target, −(1+k) for
// reference k (diffusers' _prepare_image_ids).
func ImageIDs(h, w int, l float32) *qwen.Mat {
	m := qwen.NewMat(h*w, 4)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r := m.Row(y*w + x)
			r[1], r[2], r[3] = float32(y)*ImageRopeScale, float32(x)*ImageRopeScale, l
		}
	}
	return m
}

// TextIDs is the text tokens' ids [n, 4]: (0, 0, 0, i) for every row of the
// padded window, pad rows included — the DiT's positions, not the encoder's.
func TextIDs(n int) *qwen.Mat {
	m := qwen.NewMat(n, 4)
	for i := 0; i < n; i++ {
		m.Row(i)[3] = float32(i)
	}
	return m
}
