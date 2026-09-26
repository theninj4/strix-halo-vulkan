package pipeline

import (
	"context"
	"fmt"

	qvae "strix-halo-vulkan/qimage/vae"
	"strix-halo-vulkan/zimage/qwen"
)

// The in-progress preview — IMAGE.md Q7, redone once Q-o4 came true.
//
// Q7 shipped a fitted 64x4 matrix because no tiny decoder existed for this
// VAE. madebyollin/taesd then shipped one, TAEQI2.1 (2026-09-25), and it is
// what previews run through now: qvae.TinyGPUDecoder, staged beside the full
// decoder. On the final latent of a real 1024² run it lands rms 0.087 from
// the full VAE's image where the matrix, upscaled, landed 0.20 — and on the
// matrix's own 1/16 grid 0.012 against 0.116, ten times closer at the one
// thing the matrix was fitted to do. It also decodes at the finished size,
// with texture, where the matrix gave a 64x64 thumbnail.
//
// What it costs is ~63 ms a frame at 1024² (qvae's TestTinyGPUTiming)
// against a ~2.2 s step, and 224 MB of activation arena and 29 MB of weights
// at a 1024² ceiling. That is enough to be lazy about (Step.Preview is a
// closure) and not enough to be a flag.
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
// in, and the one TAEQI2.1 was distilled on, so no denormalisation happens
// first -- to an RGBA image [1, 4, 16h, 16w] in [-1, 1], the range and
// layout the real decoder produces. Everything downstream (ToImage, the
// backend, the SSE frame) is therefore identical for a preview and a
// finished image. It runs on the device.
func (p *Pipeline) PreviewDecode(ctx context.Context, latents *qwen.Mat, h, w int) (*qvae.Tensor, error) {
	if latents.Rows != h*w {
		return nil, fmt.Errorf("pipeline: %d latent rows for a %dx%d preview grid", latents.Rows, h, w)
	}
	z := qvae.NewTensor(1, latents.Cols, h, w)
	for tk := 0; tk < latents.Rows; tk++ {
		row := latents.Row(tk)
		for c := 0; c < latents.Cols; c++ {
			z.Plane(0, c)[tk] = row[c]
		}
	}
	return p.tiny.Decode(ctx, z)
}
