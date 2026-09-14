package vae

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

// newTestDevice opens the Strix Halo iGPU, or skips.
func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("vae-test")
	if err != nil {
		t.Skipf("no Vulkan instance: %v", err)
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		t.Skipf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		t.Skipf("no compute queue: %v", err)
	}
	// fp16, cooperative matrices and subgroup-size control are what the mid
	// block's matrix-core path needs (stage 7); a device without them falls
	// back to stage 2b's fp32 kernels rather than failing, so they are
	// requested and not required.
	feat, err := phys.SupportedFeatures()
	if err != nil {
		inst.Destroy()
		t.Skipf("no feature query: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16:             feat.Float16,
		CoopMatrix:          feat.CoopMatrix,
		SubgroupSizeControl: feat.SubgroupSizeControl,
	})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// TestGPUDecoderAgainstDiffusers runs the fp32 Vulkan graph against the same
// reference the CPU implementation is held to. The CPU path already matched,
// so any failure here is the shaders and not the architecture -- which is
// the whole reason stage 2a came first.
func TestGPUDecoderAgainstDiffusers(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")

	gpu, err := NewGPUDecoderOpts(dev, cpu, latent.H, latent.W, Options{Attn: AttnScalar, Conv: ConvScalar})
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Destroy()

	n, err := gpu.Dispatches(latent.H, latent.W)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("graph: %d dispatches, %.1f MB weights, %.1f MB activations",
		n, float64(gpu.wbuf.Size())/1e6, float64(gpu.abuf.Size())/1e6)

	got, err := gpu.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "gpu image", got, loadRef(t, m, "image"))
}

// TestGPUMatchesCPU compares the two implementations directly, which is the
// check that keeps meaning something once the GPU path moves to fp16 and the
// diffusers tolerance has to widen.
func TestGPUMatchesCPU(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")

	gpu, err := NewGPUDecoderOpts(dev, cpu, latent.H, latent.W, Options{Attn: AttnScalar, Conv: ConvScalar})
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Destroy()

	gotGPU, err := gpu.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}
	gotCPU, err := cpu.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "gpu vs cpu", gotGPU, gotCPU)
}

// wmmaTol is what the mid block's fp16 operands cost the decoded image,
// measured rather than assumed: see TestGPUWMMAMatchesScalar, which reports
// the same number against the kernel that narrows nothing. It is 40x the
// fp32 graph's 6.2e-5 and 100x below what the pipeline's own end-to-end
// bound (2.5e-2, stage 6) leaves room for.
//
// The exposure is not the arithmetic, which accumulates in fp32 either way.
// It is that this softmax is a hard one-hot (stage 2: scores reach 1.16e7,
// entropy 0) whose argmax is set by ||k_j||, so a 5e-4 perturbation of the
// operands can hand a pixel a different key -- and when it does, that pixel's
// context changes by O(1) rather than by O(eps).
const wmmaTol = 2.5e-3

// compareTol is compare with an explicit bound, for the paths that narrow.
func compareTol(t *testing.T, name string, got, want *Tensor, tol float64) float64 {
	t.Helper()
	if got.N != want.N || got.C != want.C || got.H != want.H || got.W != want.W {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	maxAbs, rms, rel, worst := deviation(got, want)
	if rel > tol {
		t.Errorf("%s %s: max abs %.6g (at %d: got %g want %g), rms %.6g, rel %.3g > %.1e",
			name, got, maxAbs, worst, got.Data[worst], want.Data[worst], rms, rel, tol)
		return rel
	}
	t.Logf("%-24s %-22s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
	return rel
}

// newWMMADecoder builds a decoder on the matrix-core path, or skips.
func newWMMADecoder(t *testing.T, dev *vk.Device, cpu *Decoder, h, w int, opt Options) *GPUDecoder {
	t.Helper()
	g, err := NewGPUDecoderOpts(dev, cpu, h, w, opt)
	if err != nil {
		t.Fatal(err)
	}
	if !g.wmma() {
		g.Destroy()
		t.Skip("device has no matrix cores")
	}
	return g
}

// TestGPUWMMAAgainstDiffusers holds the matrix-core path to the same
// reference as the fp32 one, at the bound fp16 operands earn.
func TestGPUWMMAAgainstDiffusers(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")

	for _, k := range []AttnKernel{AttnQT1KT2, AttnQT1KT4, AttnQT1KT8, AttnQT2KT2, AttnQT1KT4W32} {
		t.Run(string(k), func(t *testing.T) {
			gpu := newWMMADecoder(t, dev, cpu, latent.H, latent.W, Options{Attn: k, Conv: ConvScalar})
			defer gpu.Destroy()
			got, err := gpu.Apply(latent)
			if err != nil {
				t.Fatal(err)
			}
			compareTol(t, "image "+string(k), got, loadRef(t, m, "image"), wmmaTol)
		})
	}
}

// TestGPUWMMAMatchesScalar is the measurement the tolerance above comes from:
// the same graph with and without the narrowing, on the same device, so the
// only difference between the two answers is fp16.
func TestGPUWMMAMatchesScalar(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")

	scalar, err := NewGPUDecoderOpts(dev, cpu, latent.H, latent.W, Options{Attn: AttnScalar, Conv: ConvScalar})
	if err != nil {
		t.Fatal(err)
	}
	defer scalar.Destroy()
	want, err := scalar.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}

	gpu := newWMMADecoder(t, dev, cpu, latent.H, latent.W, Options{Conv: ConvScalar})
	defer gpu.Destroy()
	got, err := gpu.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "wmma vs scalar", got, want, wmmaTol)
}

// TestGPUWMMAControls breaks the two things the new kernel does that no
// kernel in this package did before -- summing the waves' partial scores, and
// rescaling the running output onto a new maximum -- and asserts what each
// break does to the *image*. One of them does nothing at all, and that is the
// finding this test is here to pin:
//
//   - NO_RESCALE lands 634x above the bound, as a broken control should.
//   - NO_CROSS_WAVE, which throws away three quarters of every dot product,
//     decodes the same image to **8e-4** -- inside the tolerance the correct
//     kernel meets. This is stage 2's observation again and harder: the mid
//     block's softmax is a hard one-hot whose argmax is set by ||k_j|| rather
//     than by q's direction, so a quarter of each score is enough to pick the
//     same key, and no end-to-end tolerance can catch a score-side bug here.
//     TestGPUWMMAAttentionKernel is where it is caught instead, on inputs
//     whose softmax is soft: there the same build is 833x out.
//
// So the assertion below is deliberately two-sided. If a future change makes
// the image sensitive to the scores, this test fails and the right response
// is to promote the control, not to widen a bound.
func TestGPUWMMAControls(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")
	want := loadRef(t, m, "image")

	for _, c := range []struct {
		kernel AttnKernel
		caught bool
	}{
		{AttnNoRescale, true},
		{AttnNoCrossWave, false},
	} {
		t.Run(string(c.kernel), func(t *testing.T) {
			gpu := newWMMADecoder(t, dev, cpu, latent.H, latent.W, Options{Attn: c.kernel, Conv: ConvScalar})
			defer gpu.Destroy()
			got, err := gpu.Apply(latent)
			if err != nil {
				t.Fatal(err)
			}
			_, _, rel, _ := deviation(got, want)
			switch {
			case c.caught && rel <= wmmaTol:
				t.Errorf("%s: rel %.3g is inside the tolerance %.1e -- the control is not controlling anything", c.kernel, rel, wmmaTol)
			case !c.caught && rel > wmmaTol:
				t.Errorf("%s: rel %.3g is outside the tolerance %.1e -- the image has become sensitive to the scores, so this control now works and should be asserted as one", c.kernel, rel, wmmaTol)
			default:
				t.Logf("%-16s rel %.3g, %.2gx the bound", c.kernel, rel, rel/wmmaTol)
			}
		})
	}
}

// TestGPUWMMATailMask decodes at a latent whose pixel count is not a multiple
// of the kernel's key block, which is the one thing the 16x16 reference latent
// cannot exercise: 256 pixels is four whole blocks of 64. At 12x12 the mid
// block is 144 rows, so the last block is 16 keys of 64 and the other 48 are
// padding that must not reach the denominator.
func TestGPUWMMATailMask(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	const n = 12
	latent := NewTensor(1, cpu.ConvIn.InC, n, n)
	for i := range latent.Data {
		latent.Data[i] = float32(math.Sin(float64(i)*0.37)) * 2
	}

	scalar, err := NewGPUDecoderOpts(dev, cpu, n, n, Options{Attn: AttnScalar, Conv: ConvScalar})
	if err != nil {
		t.Fatal(err)
	}
	defer scalar.Destroy()
	want, err := scalar.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}

	for _, k := range []AttnKernel{AttnQT1KT2, AttnQT1KT4, AttnQT1KT8, AttnQT2KT2, AttnQT1KT4W32} {
		t.Run(string(k), func(t *testing.T) {
			gpu := newWMMADecoder(t, dev, cpu, n, n, Options{Attn: k, Conv: ConvScalar})
			defer gpu.Destroy()
			got, err := gpu.Apply(latent)
			if err != nil {
				t.Fatal(err)
			}
			compareTol(t, "tail "+string(k), got, want, wmmaTol)
		})
	}
}

// runAttention dispatches the mid block's three packs and its attention
// kernel on supplied rows, and returns the context. It writes into the
// decoder's own arenas at fixed offsets rather than through the builder,
// because what it is testing is one dispatch and not a graph.
func runAttention(t *testing.T, g *GPUDecoder, rows, dim int, q, k, v []float32, scale float64) []float32 {
	t.Helper()
	rowsPad := padRows(rows)
	plane := rowsPad * dim
	if need := 4 * plane * 4; g.abuf.Size() < need {
		t.Skipf("fp32 arena is %d MB, need %d", g.abuf.Size()>>20, need>>20)
	}
	if need := 3 * plane * 2; g.hbuf.Size() < need {
		t.Skipf("fp16 arena is %d MB, need %d", g.hbuf.Size()>>20, need>>20)
	}
	pad := func(src []float32) []float32 {
		out := make([]float32, plane)
		copy(out, src)
		return out
	}
	g.abuf.WriteFloat32At(0, pad(q))
	g.abuf.WriteFloat32At(plane, pad(k))
	g.abuf.WriteFloat32At(2*plane, pad(v))

	const log2e = 1.4426950408889634
	packs := []struct {
		pipe    string
		in, out uint32
		scale   float32
	}{
		{"pack", 0, 0, float32(scale * log2e)},
		{"pack", uint32(plane), uint32(plane), 1},
		{"packt", uint32(2 * plane), uint32(2 * plane), 1},
	}
	for _, p := range packs {
		pc := pushConstants{
			InOff: p.in, OutOff: p.out,
			Aux0: uint32(rows), Aux1: math.Float32bits(p.scale), Aux2: noBias,
		}
		if _, err := g.pipes[p.pipe].DispatchTimed(uint32(rowsPad/coopMatTile), 1, 1, 1, pc.bytes()); err != nil {
			t.Fatal(err)
		}
	}
	pc := pushConstants{
		InOff: 0, OutOff: uint32(3 * plane), ResOff: uint32(plane), Aux2: uint32(2 * plane),
		C: uint32(dim), Aux0: uint32(rows),
	}
	grid := uint32((rows + g.attn.qt*coopMatTile - 1) / (g.attn.qt * coopMatTile))
	if _, err := g.pipes["attn_wmma"].DispatchTimed(grid, 1, 1, 1, pc.bytes()); err != nil {
		t.Fatal(err)
	}
	return g.abuf.ReadFloat32At(3*plane, rows*dim)
}

// flooredDeviation is the worst per-element error relative to
// max(|want|, rms). The floor is what keeps elements near zero from
// dominating; the per-element numerator is what keeps a tensor whose RMS is
// far below its largest elements from hiding a real error in them.
func flooredDeviation(got, want *Tensor) float64 {
	var sumSq float64
	for _, w := range want.Data {
		sumSq += float64(w) * float64(w)
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var worst float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		worst = math.Max(worst, d/math.Max(math.Abs(float64(want.Data[i])), rms))
	}
	return worst
}

// cpuAttention is softmax(q k^T * scale) v in float64, the oracle for the
// kernel test below.
func cpuAttention(rows, dim int, q, k, v []float32, scale float64) []float32 {
	out := make([]float32, rows*dim)
	w := make([]float64, rows)
	for r := 0; r < rows; r++ {
		mx := math.Inf(-1)
		for j := 0; j < rows; j++ {
			var s float64
			for d := 0; d < dim; d++ {
				s += float64(q[r*dim+d]) * float64(k[j*dim+d])
			}
			w[j] = s * scale
			mx = math.Max(mx, w[j])
		}
		var den float64
		for j := range w {
			w[j] = math.Exp(w[j] - mx)
			den += w[j]
		}
		for d := 0; d < dim; d++ {
			var acc float64
			for j := 0; j < rows; j++ {
				acc += w[j] * float64(v[j*dim+d])
			}
			out[r*dim+d] = float32(acc / den)
		}
	}
	return out
}

// TestGPUWMMAAttentionKernel checks the kernel on inputs the model itself
// cannot supply, and it exists because of what the image-level controls
// found: at this VAE's real activations the softmax is a hard one-hot whose
// argmax is set by ||k_j|| (stage 2), so **a kernel that sums only a quarter
// of each dot product decodes the same image to 8e-4**. Every score-side bug
// is invisible there. On unit-normal rows the scores land near 1 and the
// distribution is soft, which is where a quarter of a dot product, a
// mis-ordered fragment tile or a wave writing another wave's output columns
// all show up.
func TestGPUWMMAAttentionKernel(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	dev, done := newTestDevice(t)
	defer done()
	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}

	const rows, dim = 144, 512
	scale := 1 / math.Sqrt(float64(dim))
	rnd := func(seed int64) []float32 {
		r := rand.New(rand.NewSource(seed))
		out := make([]float32, rows*dim)
		for i := range out {
			out[i] = float32(r.NormFloat64())
		}
		return out
	}
	q, k, v := rnd(1), rnd(2), rnd(3)
	want := cpuAttention(rows, dim, q, k, v, scale)
	wantT := &Tensor{N: 1, C: 1, H: rows, W: dim, Data: want}

	// The denominator is per element with an RMS floor rather than the RMS
	// alone, which is the distinction stage 5 paid for. Every build measures
	// 2.0e-3 of it and the bound is 3e-3; the controls below land 1250x and
	// 4077x above that, so the margin costs nothing.
	//
	// 2.0e-3 is fp16 on v and not the softmax: v's entries reach 4.5, where
	// the quantum is 2e-3, and a weighted average of 144 of them keeps a few
	// times 1e-4 of that -- which against this tensor's 0.137 RMS is exactly
	// what is measured. The scores contribute less, because P is bounded by 1
	// and its own narrowing is 5e-4 at worst.
	const kernelTol = 3e-3
	for _, kern := range []AttnKernel{AttnQT1KT2, AttnQT1KT4, AttnQT1KT8, AttnQT2KT2, AttnQT1KT4W32} {
		t.Run(string(kern), func(t *testing.T) {
			g := newWMMADecoder(t, dev, cpu, 16, 16, Options{Attn: kern, Conv: ConvScalar})
			defer g.Destroy()
			got := runAttention(t, g, rows, dim, q, k, v, scale)
			rel := flooredDeviation(&Tensor{N: 1, C: 1, H: rows, W: dim, Data: got}, wantT)
			if rel > kernelTol {
				t.Errorf("attention %s: rel %.3g > %.1e", kern, rel, kernelTol)
				return
			}
			t.Logf("attention %-12s rel %.3g", kern, rel)
		})
	}

	for _, kern := range []AttnKernel{AttnNoCrossWave, AttnNoRescale} {
		t.Run(string(kern), func(t *testing.T) {
			g := newWMMADecoder(t, dev, cpu, 16, 16, Options{Attn: kern, Conv: ConvScalar})
			defer g.Destroy()
			got := runAttention(t, g, rows, dim, q, k, v, scale)
			rel := flooredDeviation(&Tensor{N: 1, C: 1, H: rows, W: dim, Data: got}, wantT)
			if rel <= kernelTol {
				t.Errorf("%s: rel %.3g is inside the tolerance %.1e", kern, rel, kernelTol)
				return
			}
			t.Logf("%-16s rel %.3g, %.0fx the bound", kern, rel, rel/kernelTol)
		})
	}
}
