package vae

import (
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
	zvae "strix-halo-vulkan/zimage/vae"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("qi21-vae-test")
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
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// loadCPU loads the config and the CPU decoder, or skips.
func loadCPU(t *testing.T) (*Config, *Decoder) {
	t.Helper()
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Skipf("no VAE checkpoint at %s (%v)", vaeDir, err)
	}
	dec, err := LoadDecoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return cfg, dec
}

// loadGPU loads the decoder and builds its device graph for one latent size.
//
// The cleanup is registered rather than deferred by the caller, and the
// device's teardown has to be registered the same way and *first*: cleanups
// run last-in-first-out and after every deferred call in the test body, so a
// deferred dev.Destroy() would free the device out from under these
// pipelines and the shim would segfault destroying them.
func loadGPU(t *testing.T, dev *vk.Device, h, w int) (*Config, *Decoder, *GPUDecoder) {
	t.Helper()
	cfg, dec := loadCPU(t)
	g, err := NewGPUDecoder(dev, dec, h, w)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	return cfg, dec, g
}

// TestGPUDecoderStages walks the device graph stage by stage against the same
// dump the CPU decoder is gated on — conv_in, the mid block, every up block,
// the fused tail norm, conv_out and the clamped image — for both of the
// dump's cases. The bounds are vae_test.go's, unchanged: this path is fp32
// throughout, so a summation-order difference is all that separates it from
// the CPU port, and a structural mistake shows at rel >= 1e-1 on every one
// of these stages.
func TestGPUDecoderStages(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	for label, c := range m.Cases {
		lh, lw := c.Latent[len(c.Latent)-2], c.Latent[len(c.Latent)-1]
		t.Run(label, func(t *testing.T) {
			cfg, _, g := loadGPU(t, dev, lh, lw)
			n, err := g.Dispatches(lh, lw)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%s: %d dispatches, %d MB activations, %d MB weights",
				label, n, g.ActivationBytes()>>20, g.WeightBytes()>>20)

			z := loadRef(t, m, label+"_z_norm")
			cfg.Denormalize(z)

			stages, err := g.Stages(lh, lw)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range stages {
				got, err := g.RunTo(z, name)
				if err != nil {
					t.Fatal(err)
				}
				compare(t, label+"_dec_"+name, got, loadRef(t, m, label+"_dec_"+name))
			}

			img, err := g.Decode(z)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, label+"_decoded", img, loadRef(t, m, label+"_decoded"))
		})
	}
}

// TestGPUDecoderMatchesCPU is the same decode on both paths, end to end. It
// is the tighter instrument of the two: the dump carries the reference's own
// fp32 noise, while this isolates what the device graph does differently
// from the Go one it was ported from.
func TestGPUDecoderMatchesCPU(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	c, ok := m.Cases["s256"]
	if !ok {
		t.Skip("reference has no s256 case")
	}
	lh, lw := c.Latent[len(c.Latent)-2], c.Latent[len(c.Latent)-1]
	cfg, dec, g := loadGPU(t, dev, lh, lw)

	z := loadRef(t, m, "s256_z_norm")
	cfg.Denormalize(z)

	want, err := dec.Decode(z)
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.Decode(z)
	if err != nil {
		t.Fatal(err)
	}
	// Named to land on tolFor's loose bound, and for the reason that bound
	// exists: this comparison is downstream of the tail norm, where the model
	// rescales O(1e4) activations to O(1) and any summation-order difference
	// upstream surfaces elementwise. The number to read is the max abs, in
	// [-1, 1] — it comes out below the CPU port's own distance from the dump.
	compare(t, "s256_gpu_vs_cpu_decoded", got, want)
}

// TestGPUDecoderNegativeControl breaks the one piece of the graph that is
// neither z-image's nor obvious — the up blocks' parameter-free shortcut,
// whose channel mapping depends on a temporal factor that no longer has an
// axis — and checks the bound catches it. Without it, "the DupUp layout is
// right" rests on a stage comparison that could have been passing for the
// wrong reason.
func TestGPUDecoderNegativeControl(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	c, ok := m.Cases["s256"]
	if !ok {
		t.Skip("reference has no s256 case")
	}
	lh, lw := c.Latent[len(c.Latent)-2], c.Latent[len(c.Latent)-1]
	cfg, dec := loadCPU(t)

	// Every temporal factor set to 1: the spatial duplication is unchanged
	// and only which source channel feeds an output pixel moves.
	for i := range dec.Ups {
		if dec.Ups[i].Shortcut != nil {
			dec.Ups[i].Shortcut.FactorT = 1
		}
	}
	bad, err := NewGPUDecoder(dev, dec, lh, lw)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(bad.Destroy)

	z := loadRef(t, m, "s256_z_norm")
	cfg.Denormalize(z)
	got, err := bad.Decode(z)
	if err != nil {
		t.Fatal(err)
	}
	want := loadRef(t, m, "s256_decoded")
	var rel float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		if r := d / math.Max(math.Abs(float64(want.Data[i])), 1e-3); r > rel {
			rel = r
		}
	}
	if rel <= normTol {
		t.Fatalf("temporal factor 1 everywhere still decodes at rel %.3g, inside the %.0e bound", rel, normTol)
	}
	t.Logf("negative control (FactorT=1): rel %.3g, %.0fx the bound", rel, rel/normTol)
}

// TestGPUDecodeTiming is the wall clock a served image pays downstream of the
// DiT, at the two sizes Q6 has to choose between. Two runs, per the project's
// convention; the first also pays the arena's first touch.
func TestGPUDecodeTiming(t *testing.T) {
	if testing.Short() {
		t.Skip("timing runs the full-size graph")
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	for _, side := range []int{256, 1024} {
		lat := side / 16
		t.Run(fmt.Sprintf("%dx%d", side, side), func(t *testing.T) {
			_, _, g := loadGPU(t, dev, lat, lat)
			n, err := g.Dispatches(lat, lat)
			if err != nil {
				t.Fatal(err)
			}
			z := zvae.NewTensor(1, 64, lat, lat)
			for i := range z.Data {
				z.Data[i] = float32(math.Sin(float64(i)*0.001)) * 0.5
			}
			for run := 0; run < 2; run++ {
				start := time.Now()
				if _, err := g.Decode(z); err != nil {
					t.Fatal(err)
				}
				t.Logf("run %d: %v (%d dispatches, %d MB activations)",
					run, time.Since(start).Round(time.Millisecond), n, g.ActivationBytes()>>20)
			}
		})
	}
}

// TestGPUDecodeProfile attributes the decode by kind, which is what says
// whether the next percent is worth taking and where. It is not a gate.
func TestGPUDecodeProfile(t *testing.T) {
	if testing.Short() {
		t.Skip("profiling runs the full-size graph one dispatch at a time")
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	const lat = 64 // a 1024x1024 image
	_, _, g := loadGPU(t, dev, lat, lat)
	z := zvae.NewTensor(1, 64, lat, lat)
	for i := range z.Data {
		z.Data[i] = float32(math.Sin(float64(i)*0.001)) * 0.5
	}
	stages, err := g.Profile(z)
	if err != nil {
		t.Fatal(err)
	}

	type agg struct {
		d     time.Duration
		flops float64
		n     int
	}
	byOp := map[string]*agg{}
	var total time.Duration
	for _, s := range stages {
		op := s.Kind
		if i := indexByte(op, ' '); i > 0 {
			op = op[:i]
		}
		a := byOp[op]
		if a == nil {
			a = &agg{}
			byOp[op] = a
		}
		a.d += s.GPU
		a.flops += s.Flops
		a.n++
		total += s.GPU
	}
	ops := make([]string, 0, len(byOp))
	for op := range byOp {
		ops = append(ops, op)
	}
	sort.Slice(ops, func(i, j int) bool { return byOp[ops[i]].d > byOp[ops[j]].d })
	t.Logf("1024x1024 decode: %v over %d dispatches", total.Round(time.Millisecond), len(stages))
	for _, op := range ops {
		a := byOp[op]
		tf := a.flops / a.d.Seconds() / 1e12
		t.Logf("  %-12s %6d x  %9v  %5.1f%%  %6.1f TFLOP/s",
			op, a.n, a.d.Round(time.Millisecond), 100*float64(a.d)/float64(total), tf)
	}
	sort.Slice(stages, func(i, j int) bool { return stages[i].GPU > stages[j].GPU })
	t.Log("  slowest dispatches:")
	for _, s := range stages[:min(8, len(stages))] {
		t.Logf("    %-44s %9v", s.Kind, s.GPU.Round(time.Millisecond))
	}
}

func indexByte(s string, b byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			return i
		}
	}
	return -1
}

// TestConvInputRanges measures what a matrix-core convolution would have to
// narrow. z-image's decoder peaks at 497 and its whole graph moved to fp16
// operands on that number; IMAGE.md's Q0 finding says this model does not,
// and this is where "does not" becomes a list of which convolutions.
func TestConvInputRanges(t *testing.T) {
	if testing.Short() {
		t.Skip("reads a tensor back per convolution")
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	c, ok := m.Cases["s256"]
	if !ok {
		t.Skip("reference has no s256 case")
	}
	lh, lw := c.Latent[len(c.Latent)-2], c.Latent[len(c.Latent)-1]
	cfg, _, g := loadGPU(t, dev, lh, lw)

	z := loadRef(t, m, "s256_z_norm")
	cfg.Denormalize(z)
	ranges, err := g.ConvInputRanges(z)
	if err != nil {
		t.Fatal(err)
	}

	// fp16's largest finite value. A narrowed operand at or above it is an
	// infinity, and one infinity in a fragment makes the whole output tile a
	// NaN.
	const fp16Max = 65504.0
	var over int
	for _, r := range ranges {
		flag := ""
		if r.AbsMax >= fp16Max {
			flag = "  OVER fp16"
			over++
		}
		t.Logf("%-38s absmax %10.4g%s", r.Kind, r.AbsMax, flag)
	}
	t.Logf("%d of %d convolutions exceed fp16", over, len(ranges))
}

// TestArenaCeiling is the geometry ceiling Q6 has to serve under. The
// activation arena is a single storage buffer and this device caps one at
// 4 GiB - 4, so "how large an image can this decoder decode" is a question
// with an exact answer rather than a memory budget.
func TestArenaCeiling(t *testing.T) {
	_, dec := loadCPU(t)
	for _, side := range []int{256, 512, 1024, 1184, 1216, 1536, 2048} {
		lat := side / 16
		n, err := ArenaBytes(dec, lat, lat)
		if err != nil {
			t.Fatal(err)
		}
		verdict := "fits"
		if n > MaxStorageBufferBytes {
			verdict = "OVER maxStorageBufferRange"
		}
		t.Logf("%4dx%-4d latent %3dx%-3d  arena %6d MB  %s", side, side, lat, lat, n>>20, verdict)
	}
}

// TestArenaShape is what the server's geometry policy rests on, and it is a
// measurement rather than an argument: **the arena is a function of the pixel
// count alone**, at exactly 3060 bytes a pixel, whatever shape those pixels
// are in.
//
// It matters because the ceiling used to be a pair of side limits, inherited
// from z-image — whose blocked fp16 conv copies really are padded per axis,
// so 128x512 cost it more arena than 256x256. This decoder holds its
// activations flat, and nothing in the graph pads a row: an 8:1 picture plans
// to the same byte as the square of its area. A side ceiling would therefore
// have been charging every landscape request for a constraint that does not
// exist — 16:9 inside a 1024x1024 box is 1024x576, 56% of the pixels the same
// arena holds.
//
// The two controls are the extreme aspects (256x4096 and 4096x256), because
// an axis-dependent cost would show up there or nowhere.
func TestArenaShape(t *testing.T) {
	_, dec := loadCPU(t)
	const bytesPerPixel = 3060
	for _, s := range [][2]int{
		{1024, 1024}, {1344, 768}, {768, 1344}, {1152, 896}, {2048, 512},
		{512, 2048}, {256, 4096}, {4096, 256}, {1184, 1184}, {1536, 864},
	} {
		w, h := s[0], s[1]
		n, err := ArenaBytes(dec, h/16, w/16)
		if err != nil {
			t.Fatal(err)
		}
		if got := n / (w * h); got != bytesPerPixel || n%(w*h) != 0 {
			t.Errorf("%dx%d plans %d bytes, %g a pixel; want exactly %d",
				w, h, n, float64(n)/float64(w*h), bytesPerPixel)
		}
		t.Logf("%4dx%-4d  %.3f Mpx  arena %6d MB  %s",
			w, h, float64(w*h)/1e6, n>>20,
			map[bool]string{true: "fits", false: "OVER maxStorageBufferRange"}[n <= MaxStorageBufferBytes])
	}
	// The consequence, spelled out: the largest area this decoder decodes,
	// and the largest 16:9 inside it.
	t.Logf("budget %d pixels (%.2f Mpx); largest square 1184x1184, largest exact 16:9 1536x864",
		MaxStorageBufferBytes/bytesPerPixel, float64(MaxStorageBufferBytes/bytesPerPixel)/1e6)
}
