package vae

import (
	"math"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// The screen behind kernels.go's two defaults — IMAGE.md Q9b.
//
// It is the same shape as Q9's `TestGPUPackScreen`, and for the same reason
// that one gave: a screen that only timed its arms could pick one that
// computes the wrong thing quickly. Both of these replacements reorder no
// accumulator, so every arm is required to decode **bit-identically** to the
// pair of kernels it replaces — `ConvOC8` + `MidLinear`, which is the graph
// Q5g shipped and Q9's first pass left alone — and the timing is only read
// from arms that passed that.
//
// It runs at 1024², which is the served shape and the one the ledger's
// percentages are quoted at.
func TestGPUKernelScreen(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the full-size graph once per arm")
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	_, dec := loadCPU(t)

	const lat = 64 // a 1024x1024 image
	z := NewTensor(1, 64, lat, lat)
	for i := range z.Data {
		z.Data[i] = float32(math.Sin(float64(i)*0.001)) * 0.5
	}

	// The reference output: the kernels this stage replaces.
	base := runArm(t, dev, dec, z, Options{Conv: ConvOC8, Mid: MidLinear}, nil)

	t.Run("conv", func(t *testing.T) {
		for _, k := range ConvKernels() {
			runArm(t, dev, dec, z, Options{Conv: k, Mid: DefaultMidKernel}, base)
		}
	})
	t.Run("mid", func(t *testing.T) {
		for _, k := range MidKernels() {
			runArm(t, dev, dec, z, Options{Conv: DefaultConvKernel, Mid: k}, base)
		}
	})
}

// runArm stages one pair of kernels, checks its decode against want (when
// given), and reports the wall clock of two more decodes plus the two
// operators' share of a one-dispatch-at-a-time profile.
func runArm(t *testing.T, dev *vk.Device, dec *Decoder, z *Tensor, opt Options, want *Tensor) *Tensor {
	t.Helper()
	g, err := NewGPUDecoderOpts(dev, dec, z.H, z.W, opt)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	conv, mid := g.Kernels()

	got, err := g.Decode(t.Context(), z)
	if err != nil {
		t.Fatal(err)
	}
	if want != nil {
		for i := range want.Data {
			if got.Data[i] != want.Data[i] {
				t.Fatalf("%s/%s: element %d decodes to %v, the kernels it replaces give %v — "+
					"this port is supposed to move bytes, not numbers",
					conv, mid, i, got.Data[i], want.Data[i])
			}
		}
	}

	var walls [2]time.Duration
	for run := range walls {
		start := time.Now()
		if _, err := g.Decode(t.Context(), z); err != nil {
			t.Fatal(err)
		}
		walls[run] = time.Since(start).Round(time.Millisecond)
	}

	stages, err := g.Profile(z)
	if err != nil {
		t.Fatal(err)
	}
	var convD, midD, total time.Duration
	var convF, midF float64
	for _, s := range stages {
		total += s.GPU
		switch {
		case len(s.Kind) >= 7 && s.Kind[:7] == "conv3x3":
			convD += s.GPU
			convF += s.Flops
		case len(s.Kind) >= 6 && s.Kind[:6] == "linear":
			midD += s.GPU
			midF += s.Flops
		}
	}
	t.Logf("%-9s %-7s  decode %v / %v   conv3x3 %6v (%4.1f%%, %5.1f TFLOP/s)   proj %6v (%4.1f%%, %5.2f TFLOP/s)   profiled total %v",
		conv, mid, walls[0], walls[1],
		convD.Round(time.Millisecond), 100*float64(convD)/float64(total), convF/convD.Seconds()/1e12,
		midD.Round(time.Millisecond), 100*float64(midD)/float64(total), midF/midD.Seconds()/1e12,
		total.Round(time.Millisecond))
	return got
}
