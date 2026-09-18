package vae

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The reference is produced by reference/dump_vae.py, which decodes a fixed
// latent with diffusers on CPU in float32 and writes every submodule's
// output. Regenerate with:
//
//	.venv/bin/python reference/dump_vae.py --latent-size 16
const (
	refDir = "../../reference/out/vae"
	vaeDir = "../../models/Z-Image-Turbo/vae"
)

type manifest struct {
	// dir is where the tensors this manifest describes live, so that a test
	// reading a second dump -- taef1's -- uses the same loader rather than a
	// copy of it.
	dir        string
	Seed       int `json:"seed"`
	LatentSize int `json:"latent_size"`
	Tensors    map[string]struct {
		Shape  []int   `json:"shape"`
		Count  int     `json:"count"`
		Sum    float64 `json:"sum"`
		AbsMax float64 `json:"absmax"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	return loadManifestFrom(t, refDir, "reference/dump_vae.py")
}

// loadManifestFrom reads a dump directory's manifest, skipping the test if it
// has not been produced. gen is the script that produces it, so the skip says
// what to run.
func loadManifestFrom(t *testing.T, dir, gen string) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump (%v); run %s", err, gen)
	}
	m := manifest{dir: dir}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// loadRef reads one dumped tensor as NCHW float32.
func loadRef(t *testing.T, m *manifest, name string) *Tensor {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(m.dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != meta.Count*4 {
		t.Fatalf("%s: %d bytes for %d float32", name, len(raw), meta.Count)
	}
	data := make([]float32, meta.Count)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	sh := meta.Shape
	if len(sh) != 4 {
		t.Fatalf("%s: shape %v, want 4 dims", name, sh)
	}
	return &Tensor{N: sh[0], C: sh[1], H: sh[2], W: sh[3], Data: data}
}

// deviation measures the worst absolute error between two tensors and
// expresses it relative to the reference's RMS.
func deviation(got, want *Tensor) (maxAbs, rms, rel float64, worst int) {
	var sumSqWant float64
	for i := range want.Data {
		g, w := float64(got.Data[i]), float64(want.Data[i])
		sumSqWant += w * w
		if d := math.Abs(g - w); d > maxAbs {
			maxAbs, worst = d, i
		}
	}
	rms = math.Sqrt(sumSqWant / float64(len(want.Data)))
	return maxAbs, rms, maxAbs / math.Max(rms, 1e-12), worst
}

// relTol is the bound every stage must stay inside. It has to leave room for
// depth -- this is 40-odd convolutions of two independent float32 summation
// orders, not the same one twice -- but not so much room that a real error
// slips under it. Measured legitimate worst case is 6.2e-5; the negative
// control in TestValidationDetectsErrors pins the other side by checking
// that plausible porting mistakes land above this.
const relTol = 2e-4

// compare asserts a computed tensor matches the reference.
func compare(t *testing.T, name string, got, want *Tensor) {
	t.Helper()
	if got.N != want.N || got.C != want.C || got.H != want.H || got.W != want.W {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	maxAbs, rms, rel, worst := deviation(got, want)

	// Normalising by the tensor's RMS rather than by each element's own
	// magnitude is what makes this bound meaningful. A per-element relative
	// error is dominated by values near zero -- the activations here run to
	// a mean of -12 with plenty of crossings -- and reports 0.19 for a
	// tensor that agrees to seven digits.
	if rel > relTol {
		t.Errorf("%s %s: max abs %.6g (at %d: got %g want %g), rms %.6g, rel %.3g > %.0e",
			name, got, maxAbs, worst, got.Data[worst], want.Data[worst], rms, rel, relTol)
		return
	}
	t.Logf("%-22s %-22s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
}

// TestDecoderAgainstDiffusers walks the decoder stage by stage, checking each
// submodule's output against the diffusers dump. Doing it stagewise is the
// point: a single end-to-end check tells you the image is wrong, this tells
// you which of the 40 convolutions is.
func TestDecoderAgainstDiffusers(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)

	dec, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}

	latent := loadRef(t, m, "latent")

	h, err := dec.ConvIn.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "conv_in", h, loadRef(t, m, "conv_in"))

	if h, err = dec.Mid.Resnet1.Apply(h); err != nil {
		t.Fatal(err)
	}
	compare(t, "mid.resnets.0", h, loadRef(t, m, "mid.resnets.0"))

	if h, err = dec.Mid.Attn.Apply(h); err != nil {
		t.Fatal(err)
	}
	compare(t, "mid.attn", h, loadRef(t, m, "mid.attn"))

	if h, err = dec.Mid.Resnet2.Apply(h); err != nil {
		t.Fatal(err)
	}
	compare(t, "mid.resnets.1", h, loadRef(t, m, "mid.resnets.1"))

	for i, up := range dec.UpBlocks {
		for j, r := range up.Resnets {
			if h, err = r.Apply(h); err != nil {
				t.Fatal(err)
			}
			compare(t, fmt.Sprintf("up.%d.resnets.%d", i, j), h, loadRef(t, m, fmt.Sprintf("up.%d.resnets.%d", i, j)))
		}
		if up.Upsampler != nil {
			if h, err = up.Upsampler.Apply(UpsampleNearest2x(h)); err != nil {
				t.Fatal(err)
			}
			compare(t, fmt.Sprintf("up.%d.upsample", i), h, loadRef(t, m, fmt.Sprintf("up.%d.upsample", i)))
		}
	}

	if h, err = dec.ConvNormOut.ApplyInPlace(h); err != nil {
		t.Fatal(err)
	}
	compare(t, "conv_norm_out", h, loadRef(t, m, "conv_norm_out"))

	SiLUInPlace(h)
	if h, err = dec.ConvOut.Apply(h); err != nil {
		t.Fatal(err)
	}
	compare(t, "conv_out", h, loadRef(t, m, "conv_out"))
}

// TestDecoderEndToEnd runs Apply as the pipeline will call it and checks
// only the image, so that the composed graph is covered as well as its
// parts.
func TestDecoderEndToEnd(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dec, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	got, err := dec.Apply(loadRef(t, m, "latent"))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "image", got, loadRef(t, m, "image"))
}

// TestValidationDetectsErrors is the negative control: it perturbs the
// decoder in four ways that a plausible porting mistake would produce, and
// asserts each one is caught. Without this, the stagewise test above proves
// only that it runs, not that it can fail.
//
// The perturbations are deliberately small — a single weight moved by 1%, a
// dropped residual, an off-by-one pad — because the errors worth catching in
// a Vulkan port are subtle ones, not tensors of garbage.
func TestValidationDetectsErrors(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	latent := loadRef(t, m, "latent")
	wantConvIn := loadRef(t, m, "conv_in")
	wantResnet := loadRef(t, m, "mid.resnets.0")

	t.Run("one output channel off by 1%", func(t *testing.T) {
		dec, err := LoadDecoder(vaeDir, FluxConfig())
		if err != nil {
			t.Fatal(err)
		}
		// A whole output channel, which is what an indexing or layout bug
		// looks like. A single weight of 73728 moved by 1% measures 3.9e-4
		// and is deliberately not the bar -- no shader gets one weight
		// wrong.
		per := dec.ConvIn.InC * dec.ConvIn.KH * dec.ConvIn.KW
		for i := 0; i < per; i++ {
			dec.ConvIn.Weight[i] *= 1.01
		}
		got, err := dec.ConvIn.Apply(latent)
		if err != nil {
			t.Fatal(err)
		}
		assertCaught(t, got, wantConvIn)
	})

	t.Run("wrong padding", func(t *testing.T) {
		dec, err := LoadDecoder(vaeDir, FluxConfig())
		if err != nil {
			t.Fatal(err)
		}
		dec.ConvIn.Pad = 0
		got, err := dec.ConvIn.Apply(latent)
		if err != nil {
			t.Fatal(err)
		}
		// A pad change also changes the shape, which is itself the signal.
		if got.H == wantConvIn.H && got.W == wantConvIn.W {
			assertCaught(t, got, wantConvIn)
		}
	})

	t.Run("dropped residual", func(t *testing.T) {
		dec, err := LoadDecoder(vaeDir, FluxConfig())
		if err != nil {
			t.Fatal(err)
		}
		h, err := dec.ConvIn.Apply(latent)
		if err != nil {
			t.Fatal(err)
		}
		got, err := applyResnetNoResidual(dec.Mid.Resnet1, h)
		if err != nil {
			t.Fatal(err)
		}
		assertCaught(t, got, wantResnet)
	})

	t.Run("transposed attention value projection", func(t *testing.T) {
		dec, err := LoadDecoder(vaeDir, FluxConfig())
		if err != nil {
			t.Fatal(err)
		}
		// to_v and not to_q, and the reason is worth recording: this VAE's
		// mid-block attention is fully saturated. Scores reach 1.16e7 and
		// the softmax is a hard one-hot (entropy 0), with the argmax set by
		// ||k_j|| rather than by q's direction -- so transposing to_q leaves
		// the output unchanged to 1.5e-5 and no tolerance can catch it.
		// to_v feeds the output directly and is caught easily.
		//
		// That saturation is also the constraint the Vulkan port inherits:
		// activations stay inside fp16 (absmax 497 through the whole
		// decoder), but the score matrix does not, so scores must
		// accumulate in fp32 and the row max must be subtracted before any
		// exponential.
		q := dec.Mid.Attn.V
		if q.In != q.Out {
			t.Skip("projection is not square, transposing changes the shape")
		}
		w := append([]float32(nil), q.Weight...)
		for i := 0; i < q.Out; i++ {
			for j := 0; j < q.In; j++ {
				q.Weight[i*q.In+j] = w[j*q.In+i]
			}
		}
		h, err := dec.ConvIn.Apply(latent)
		if err != nil {
			t.Fatal(err)
		}
		if h, err = dec.Mid.Resnet1.Apply(h); err != nil {
			t.Fatal(err)
		}
		got, err := dec.Mid.Attn.Apply(h)
		if err != nil {
			t.Fatal(err)
		}
		assertCaught(t, got, loadRef(t, m, "mid.attn"))
	})
}

// assertCaught fails if a deliberately broken tensor slips under relTol.
func assertCaught(t *testing.T, got, want *Tensor) {
	t.Helper()
	maxAbs, rms, rel, _ := deviation(got, want)
	if rel <= relTol {
		t.Errorf("perturbation NOT caught: max abs %.6g, rms %.6g, rel %.3g <= %.0e", maxAbs, rms, rel, relTol)
		return
	}
	t.Logf("caught: rel %.3g (%.0fx the %.0e bound)", rel, rel/relTol, relTol)
}

// applyResnetNoResidual is ResnetBlock.Apply with the skip connection
// omitted, which is the single most likely thing to get wrong when porting a
// residual block to a shader graph.
func applyResnetNoResidual(r *ResnetBlock, x *Tensor) (*Tensor, error) {
	h := &Tensor{N: x.N, C: x.C, H: x.H, W: x.W, Data: append([]float32(nil), x.Data...)}
	var err error
	if h, err = r.Norm1.ApplyInPlace(h); err != nil {
		return nil, err
	}
	SiLUInPlace(h)
	if h, err = r.Conv1.Apply(h); err != nil {
		return nil, err
	}
	if h, err = r.Norm2.ApplyInPlace(h); err != nil {
		return nil, err
	}
	SiLUInPlace(h)
	return r.Conv2.Apply(h)
}
