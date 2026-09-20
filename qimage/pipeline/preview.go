package pipeline

import (
	"fmt"

	"strix-halo-vulkan/zimage/qwen"
	zvae "strix-halo-vulkan/zimage/vae"
)

// The in-progress preview — IMAGE.md Q7.
//
// Z-Image previewed through `madebyollin/taef1`, a distilled decoder, at
// 87 ms a frame. **No such decoder exists for this model**: taehv's taew2_1
// decodes the Wan-2.1 16-channel/8x latent that the *original* Qwen-Image
// borrowed, and 2.1's VAE is a new 64-channel/16x RGBA design. Training one
// is a project this repo does not want.
//
// So the preview is the other well-known thing to do with a latent, the
// "latent2rgb" trick: **one 64x4 matrix and a bias, per latent pixel**. It
// is a least-squares fit of the VAE's own decoder restricted to its constant
// and linear terms — see cmd/previewfit for how the pairs were collected and
// how the fit scores. What it costs is 260 multiply-adds a latent pixel on
// the host: 1.0 M FLOPs for a 1024x1024 image's 64x64 grid, against 2.3 s of
// device time for the step that produced it. The frame is therefore free in
// a way taef1's was not, which is why previews here are not behind a flag.
//
// What it cannot do is resolve anything the latent does not already carry
// per-pixel: a preview is 1/16 scale and has no texture, and it is upscaled
// by whatever is displaying it. What it does carry is composition, colour
// and layout, which is what an in-progress frame is for.
//
// **The tensor it decodes is the denoised estimate, not the current
// latent**, and that distinction is load-bearing rather than pedantic. The
// schedule is flow matching, so the sample at step k is an interpolation
// x_t = (1-sigma) x0 + sigma eps -- ten steps into a forty-step run it is
// still three quarters noise, and mapping *that* through any decoder gives a
// picture of noise. The estimate x0 = x_t - sigma v is what looks like the
// image, and the loop can form it for nothing because it has the velocity in
// hand. Step.Preview does it; see Run.

// PreviewDecode maps packed *normalized* latents -- the space the DiT works
// in, so no denormalisation happens first -- to an RGBA image at latent
// resolution: [1, 4, h, w] in [-1, 1], which is the range and layout the
// real decoder produces. Everything downstream (ToImage, the backend, the
// SSE frame) is therefore identical for a preview and a finished image.
func PreviewDecode(latents *qwen.Mat, h, w int) (*zvae.Tensor, error) {
	if latents.Rows != h*w {
		return nil, fmt.Errorf("pipeline: %d latent rows for a %dx%d preview grid", latents.Rows, h, w)
	}
	if latents.Cols != previewZDim {
		return nil, fmt.Errorf("pipeline: preview matrix is fitted for %d channels, latents have %d",
			previewZDim, latents.Cols)
	}
	out := zvae.NewTensor(1, previewChannels, h, w)
	plane := h * w
	for p := 0; p < plane; p++ {
		z := latents.Row(p)
		for c := 0; c < previewChannels; c++ {
			row := previewWeight[c]
			sum := previewBias[c]
			for k := 0; k < previewZDim; k++ {
				sum += row[k] * z[k]
			}
			// The real decoder clamps; so does this, for the same reason --
			// a value past the range wraps to the opposite end when it is
			// converted to 8 bits, and a white highlight comes out black.
			if sum > 1 {
				sum = 1
			} else if sum < -1 {
				sum = -1
			}
			out.Data[c*plane+p] = sum
		}
	}
	return out, nil
}
