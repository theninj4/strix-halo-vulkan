package vae

import (
	"fmt"
	"os"
	"testing"
)

// The encoder's oracle, produced by reference/dump_vae_encoder.py:
//
//	.venv/bin/python reference/dump_vae_encoder.py --size 256
//
// It is a *real image* rather than a random tensor, which matters more here
// than it did for the decoder: the encoder's first convolution sees pixels,
// and the downsamplers' asymmetric padding is only visible where the picture
// has edges.
const encRefDir = "../../reference/out/vaeenc"

func loadEncManifest(t *testing.T) *manifest {
	return loadManifestFrom(t, encRefDir, "reference/dump_vae_encoder.py")
}

// encRelTol is the encoder's stagewise bound, and it is 25x the decoder's for
// a reason that was measured rather than assumed.
//
// **The encoder's late stages are ill-conditioned in float32.** Its
// activations grow from an absmax of 3.6 after conv_in to 861 at the end of
// the down blocks, and a 512-channel 3x3 convolution there sums 4608 products
// of that size; the group norm at the end then divides by a standard
// deviation, which turns the accumulated absolute error into a large relative
// one. That is a property of the arithmetic and not of this port, so the
// reference has it too. Asking diffusers for the same encode in float64 and
// diffing its own float32 against it:
//
//	down.0.resnets.0   4.8e-06      down.3.resnets.1   8.1e-05
//	down.1.downsample  1.2e-05      mid.resnets.0      2.0e-04
//	down.2.downsample  3.4e-05      conv_norm_out      6.3e-04
//	                                conv_out           1.5e-05
//
// This port lands within about 3x of those numbers everywhere, which is what
// two independent float32 summation orders of the same graph cost. The bound
// is set above the worst of them with room and is still three orders of
// magnitude below what the negative controls measure.
//
// The three tensors the pipeline actually consumes are *not* held to it:
// conv_out, the mode and the scaled latent are all well conditioned again --
// conv_out divides the accumulated error by a tensor whose RMS is 14 rather
// than 1.1 -- and they get encOutTol.
const (
	encRelTol = 5e-3
	encOutTol = 5e-4
)

// TestEncoderAgainstDiffusers walks the encoder stage by stage. Same argument
// as TestDecoderAgainstDiffusers: an end-to-end check says the latent is
// wrong, this says which of the 27 convolutions is.
func TestEncoderAgainstDiffusers(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadEncManifest(t)

	enc, err := LoadEncoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}

	h, err := enc.ConvIn.Apply(loadRef(t, m, "image_in"))
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "conv_in", h, loadRef(t, m, "conv_in"), encRelTol)

	for i, down := range enc.DownBlocks {
		for j, r := range down.Resnets {
			if h, err = r.Apply(h); err != nil {
				t.Fatal(err)
			}
			compareTol(t, fmt.Sprintf("down.%d.resnets.%d", i, j), h, loadRef(t, m, fmt.Sprintf("down.%d.resnets.%d", i, j)), encRelTol)
		}
		if down.Downsampler != nil {
			if h, err = down.Downsampler.Apply(h); err != nil {
				t.Fatal(err)
			}
			compareTol(t, fmt.Sprintf("down.%d.downsample", i), h, loadRef(t, m, fmt.Sprintf("down.%d.downsample", i)), encRelTol)
		}
	}

	if h, err = enc.Mid.Resnet1.Apply(h); err != nil {
		t.Fatal(err)
	}
	compareTol(t, "mid.resnets.0", h, loadRef(t, m, "mid.resnets.0"), encRelTol)
	if h, err = enc.Mid.Attn.Apply(h); err != nil {
		t.Fatal(err)
	}
	compareTol(t, "mid.attn", h, loadRef(t, m, "mid.attn"), encRelTol)
	if h, err = enc.Mid.Resnet2.Apply(h); err != nil {
		t.Fatal(err)
	}
	compareTol(t, "mid.resnets.1", h, loadRef(t, m, "mid.resnets.1"), encRelTol)

	if h, err = enc.ConvNormOut.ApplyInPlace(h); err != nil {
		t.Fatal(err)
	}
	compareTol(t, "conv_norm_out", h, loadRef(t, m, "conv_norm_out"), encRelTol)
	SiLUInPlace(h)
	if h, err = enc.ConvOut.Apply(h); err != nil {
		t.Fatal(err)
	}
	compareTol(t, "conv_out", h, loadRef(t, m, "conv_out"), encRelTol)
}

// TestEncoderEndToEnd runs Encode as the pipeline calls it, and checks the
// mode as well as the moments -- the two differ by a channel slice, which is
// exactly the kind of thing that is right in one direction and silently wrong
// in the other.
func TestEncoderEndToEnd(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadEncManifest(t)
	cfg := FluxConfig()
	enc, err := LoadEncoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	img := loadRef(t, m, "image_in")

	moments, err := enc.Apply(img)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "moments", moments, loadRef(t, m, "moments"), encOutTol)

	mode, err := enc.Encode(img)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "mode", mode, loadRef(t, m, "mode"), encOutTol)

	// And the scaling that puts it in the transformer's space, which is the
	// half of the contract the pipeline depends on.
	cfg.ToDiffusion(mode.Data)
	compareTol(t, "latent", mode, loadRef(t, m, "latent"), encOutTol)
}

// TestToDiffusionRoundTrips pins the pair against each other. The decode side
// of the pipeline has done `z/scale + shift` since stage 6; the encode side is
// new, and getting the two out of step produces a washed-out picture rather
// than an error.
func TestToDiffusionRoundTrips(t *testing.T) {
	cfg := FluxConfig()
	orig := []float32{-3.5, -0.25, 0, 0.125, 2.75, 9.9325}
	z := append([]float32(nil), orig...)
	cfg.ToDiffusion(z)
	cfg.FromDiffusion(z)
	for i := range z {
		if d := float64(z[i] - orig[i]); d > 1e-5 || d < -1e-5 {
			t.Errorf("round trip of %g gave %g", orig[i], z[i])
		}
	}
	// And the direction: shift_factor is positive, so a zero latent maps
	// below zero going out. A sign flip here is the mistake worth catching.
	z = []float32{0}
	cfg.ToDiffusion(z)
	if z[0] >= 0 {
		t.Errorf("ToDiffusion(0) = %g, want (0 - shift)*scale < 0", z[0])
	}
}

// TestStrideTwoIsStrideOneSubsampled is the identity the Vulkan downsampler
// rests on, asserted on the real filter rather than argued in a comment.
//
// diffusers' Downsample2D pads (0, 1, 0, 1) and convolves 3x3 with stride 2.
// Output (oh, ow) therefore reads input rows 2oh..2oh+2 and columns
// 2ow..2ow+2, which is exactly what a stride-1 pad-1 convolution computes at
// (2oh+1, 2ow+1) -- including at the last row, where the stride-1 form's own
// zero padding supplies the pixel the (0,1,0,1) pad would have. So the device
// runs the same filter at stride 1 and keeps one pixel in four, which needs
// no new matrix-core kernel; see builder.downsample for what that costs.
func TestStrideTwoIsStrideOneSubsampled(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	enc, err := LoadEncoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	down := enc.DownBlocks[0].Downsampler
	if down.Stride != 2 || down.PadEnd != 1 || down.Pad != 0 {
		t.Fatalf("downsampler is pad %d/%d stride %d, want 0/1/2", down.Pad, down.PadEnd, down.Stride)
	}

	// A small but real input: 24x24 is a shape whose last output row is the
	// one the far-side padding reaches, which a 16x16 would also be and a
	// padded-to-even fixture would not.
	const n = 24
	x := NewTensor(1, down.InC, n, n)
	for i := range x.Data {
		x.Data[i] = float32((i%97)-48) / 32
	}
	strided, err := down.Apply(x)
	if err != nil {
		t.Fatal(err)
	}

	full := &Conv2D{InC: down.InC, OutC: down.OutC, KH: down.KH, KW: down.KW, Pad: 1,
		Weight: down.Weight, Bias: down.Bias}
	dense, err := full.Apply(x)
	if err != nil {
		t.Fatal(err)
	}
	sub := Subsample2x(dense)
	compare(t, "stride2 == stride1 subsampled", sub, strided) // bit-exact: same filter, same order

	// The negative control: the offset is the whole of the identity, so
	// taking the *even* pixels instead has to be caught.
	off := NewTensor(1, dense.C, dense.H/2, dense.W/2)
	for c := 0; c < dense.C; c++ {
		src, dst := dense.Plane(0, c), off.Plane(0, c)
		for h := 0; h < off.H; h++ {
			for w := 0; w < off.W; w++ {
				dst[h*off.W+w] = src[(h*2)*dense.W+w*2]
			}
		}
	}
	assertCaught(t, off, strided)
}

// TestEncoderValidationDetectsErrors is the negative control for the walk
// above. Each perturbation is a porting mistake someone would actually make,
// and two of them produce a tensor of the *right shape*, which is what makes
// them worth asserting rather than arguing about.
func TestEncoderValidationDetectsErrors(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadEncManifest(t)
	img := loadRef(t, m, "image_in")
	cfg := FluxConfig()

	// The symmetric pad. diffusers pads (0, 1, 0, 1) and convolves with no
	// padding of its own; the obvious reading is "pad 1, stride 2", which at
	// an even input gives a tensor of exactly the same shape one pixel out of
	// register. Nothing downstream of it can tell.
	t.Run("symmetric pad on the downsampler", func(t *testing.T) {
		enc, err := LoadEncoder(vaeDir, cfg)
		if err != nil {
			t.Fatal(err)
		}
		h, err := enc.ConvIn.Apply(img)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range enc.DownBlocks[0].Resnets {
			if h, err = r.Apply(h); err != nil {
				t.Fatal(err)
			}
		}
		d := enc.DownBlocks[0].Downsampler
		d.Pad, d.PadEnd = 1, 0
		got, err := d.Apply(h)
		if err != nil {
			t.Fatal(err)
		}
		want := loadRef(t, m, "down.0.downsample")
		if got.H != want.H || got.W != want.W {
			t.Fatalf("the wrong pad changed the shape to %s; it is meant to be the control that does not", got)
		}
		assertCaught(t, got, want)
	})

	// The log-variance taken for the mean. conv_out produces 32 channels and
	// the split is by convention, not by anything in the file, so the half
	// that is dropped is a coin-flip in a port. The wrong half is still a
	// [1, 16, H/8, W/8] tensor of plausible numbers.
	t.Run("logvar taken for the mean", func(t *testing.T) {
		moments := loadRef(t, m, "moments")
		c := cfg.LatentChannels
		wrong := NewTensor(1, c, moments.H, moments.W)
		for ch := 0; ch < c; ch++ {
			copy(wrong.Plane(0, ch), moments.Plane(0, c+ch))
		}
		assertCaught(t, wrong, loadRef(t, m, "mode"))
	})

	// A dropped residual in the first down block, which is the single most
	// likely thing to lose when a resnet becomes a dispatch graph.
	t.Run("dropped residual", func(t *testing.T) {
		enc, err := LoadEncoder(vaeDir, cfg)
		if err != nil {
			t.Fatal(err)
		}
		h, err := enc.ConvIn.Apply(img)
		if err != nil {
			t.Fatal(err)
		}
		got, err := applyResnetNoResidual(enc.DownBlocks[0].Resnets[0], h)
		if err != nil {
			t.Fatal(err)
		}
		assertCaught(t, got, loadRef(t, m, "down.0.resnets.0"))
	})

	// The scaling applied the decoder's way round. This one is the reason
	// ToDiffusion and FromDiffusion are a pair in one file: both directions
	// produce a latent of the right shape and sane magnitude, and the picture
	// that comes out the far end is merely washed out.
	t.Run("the decode convention on the way in", func(t *testing.T) {
		mode := loadRef(t, m, "mode")
		z := append([]float32(nil), mode.Data...)
		cfg.FromDiffusion(z)
		assertCaught(t, &Tensor{N: 1, C: mode.C, H: mode.H, W: mode.W, Data: z}, loadRef(t, m, "latent"))
	})
}
