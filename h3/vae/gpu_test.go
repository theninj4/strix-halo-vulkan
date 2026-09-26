package vae

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("h3-vae-test")
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
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Skipf("no subgroup size control query: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

func newTestGPU(t *testing.T, th, tw, seqs int) (*GPU, func()) {
	t.Helper()
	dev, done := newTestDevice(t)
	start := time.Now()
	g, err := NewGPU(dev, vaeDir, th, tw, seqs)
	if err != nil {
		done()
		t.Fatal(err)
	}
	t.Logf("staged %.2f GB of weights and %.2f GB of activations (%d × %d rows) in %v",
		float64(g.WeightBytes())/1e9, float64(g.ActivationBytes())/1e9, seqs, g.stride, time.Since(start).Round(time.Millisecond))
	return g, func() { g.Destroy(); done() }
}

// pixelGap is what a decode's error is in the video it becomes: the largest
// and rms difference in 8-bit levels after the pipeline's denormalise and
// clamp, and the PSNR over the whole clip.
func pixelGap(got, want *Tensor) (maxLevels, rmsLevels, psnr float64) {
	var sq float64
	for c := 0; c < 3; c++ {
		plane := want.T * want.H * want.W
		s, m := float64(PixelStd[c]), float64(PixelMean[c])
		for i := c * plane; i < (c+1)*plane; i++ {
			a := math.Min(math.Max(float64(got.Data[i])*s+m, 0), 1) * 255
			b := math.Min(math.Max(float64(want.Data[i])*s+m, 0), 1) * 255
			d := math.Abs(a - b)
			maxLevels = math.Max(maxLevels, d)
			sq += d * d
		}
	}
	mse := sq / float64(len(want.Data))
	return maxLevels, math.Sqrt(mse), 10 * math.Log10(255*255/mse)
}

// TestGPUDecoder is M5's gate on the device, against the fp32 oracle: the
// blocks teacher-forced at four depths, one tile-clip through the whole
// decoder, and `vae.decode` of the first 12 latent frames — two temporal
// clips of two tiles, so the tile blend and the temporal cross-fade both
// run, batched as one call of four sequences.
func TestGPUDecoder(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 5 GB")
	}
	m := loadManifest(t)
	c := testConfig(t)
	g, done := newTestGPU(t, 16, 16, 4)
	defer done()

	for _, b := range m.KeepBlocks {
		n := m.Tensors[fmt.Sprintf("tile_block%d_in", b)].Count
		in := readF32(t, filepath.Join(vaeRef, fmt.Sprintf("tile_block%d_in.bin", b)), n)
		want := readF32(t, filepath.Join(vaeRef, fmt.Sprintf("tile_block%d_out.bin", b)), n)
		got, err := g.Blocks(in, b, b+1)
		if err != nil {
			t.Fatal(err)
		}
		rel, rms := gap(got, want)
		t.Logf("block %2d, teacher-forced: rel %.2e rms %.2e", b, rel, rms)
		if rel > 5e-3 {
			t.Errorf("block %d rel %.2e", b, rel)
		}
	}

	tileIn := readTensor(t, m, "tile_in")
	out, took, err := g.DecodeTiles([]*Tensor{tileIn})
	if err != nil {
		t.Fatal(err)
	}
	want := readTensor(t, m, "tile_out")
	rel, rms := gap(out[0].Data, want.Data)
	mx, rmsL, psnr := pixelGap(out[0], want)
	t.Logf("tile-clip: rel %.2e rms %.2e; %.2f levels max, %.3f rms, PSNR %.1f dB; %v on the device", rel, rms, mx, rmsL, psnr, took)
	if rms > 5e-3 || psnr < 50 {
		t.Errorf("tile-clip rms %.2e, PSNR %.1f dB", rms, psnr)
	}

	z := readTensor(t, m, "z")
	pqc, err := LoadPostQuantConv(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	short := z.Slice(0, m.ShortFrames, 0, z.H, 0, z.W)
	start := time.Now()
	dec, st, err := g.Decode(short, pqc)
	if err != nil {
		t.Fatal(err)
	}
	wall := time.Since(start)
	want = readTensor(t, m, "short_dec")
	if dec.C != want.C || dec.T != want.T || dec.H != want.H || dec.W != want.W {
		t.Fatalf("decode is %dx%dx%dx%d, want %dx%dx%dx%d", dec.C, dec.T, dec.H, dec.W, want.C, want.T, want.H, want.W)
	}
	rel, rms = gap(dec.Data, want.Data)
	mx, rmsL, psnr = pixelGap(dec, want)
	t.Logf("decode of %d latent frames (%d calls, %d batch): rel %.2e rms %.2e; %.2f levels max, %.3f rms, PSNR %.1f dB; %v device, %v wall",
		m.ShortFrames, st.Calls, st.Batches, rel, rms, mx, rmsL, psnr, st.Device, wall.Round(time.Millisecond))
	if rms > 5e-3 || psnr < 50 {
		t.Errorf("decode rms %.2e, PSNR %.1f dB", rms, psnr)
	}
	_ = c
}

// TestGPUDecodeFull decodes the oracle's whole 124-frame clip (7 clips × 2
// tiles, batched 8) and, with H3_VAE_MP4=path, writes it out as frames for
// a look. Opt-in with H3_VAE_FULL=1.
func TestGPUDecodeFull(t *testing.T) {
	if os.Getenv("H3_VAE_FULL") == "" {
		t.Skip("set H3_VAE_FULL=1")
	}
	m := loadManifest(t)
	g, done := newTestGPU(t, 16, 16, 8)
	defer done()
	z := readTensor(t, m, "z")
	pqc, err := LoadPostQuantConv(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	dec, st, err := g.Decode(z, pqc)
	if err != nil {
		t.Fatal(err)
	}
	wall := time.Since(start)
	want := readTensor(t, m, "full_dec")
	rel, rms := gap(dec.Data, want.Data)
	mx, rmsL, psnr := pixelGap(dec, want)
	t.Logf("full decode (%d calls, %d batches): rel %.2e rms %.2e; %.2f levels max, %.3f rms, PSNR %.1f dB; %v device, %v wall",
		st.Calls, st.Batches, rel, rms, mx, rmsL, psnr, st.Device, wall.Round(time.Millisecond))
	if rms > 5e-3 || psnr < 50 {
		t.Errorf("decode rms %.2e, PSNR %.1f dB", rms, psnr)
	}
	if path := os.Getenv("H3_VAE_RAW"); path != "" {
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		for _, fr := range ToRGB8(dec) {
			f.Write(fr)
		}
		f.Close()
		t.Logf("wrote %s: rgb24 %dx%d, %d frames", path, dec.W, dec.H, dec.T)
	}
}

// TestGPUShapes times the decode at the served and trained canvases on
// random latents (the cost does not depend on the values). Opt-in with
// H3_VAE_SHAPES=1; H3_VAE_SEQS sets the batch.
func TestGPUShapes(t *testing.T) {
	if os.Getenv("H3_VAE_SHAPES") == "" {
		t.Skip("set H3_VAE_SHAPES=1")
	}
	seqs := 8
	if s := os.Getenv("H3_VAE_SEQS"); s != "" {
		fmt.Sscan(s, &seqs)
	}
	c := testConfig(t)
	pqc, err := LoadPostQuantConv(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	g, done := newTestGPU(t, 16, 16, seqs)
	defer done()
	r := rand.New(rand.NewPCG(1, 2))
	for _, s := range []struct {
		name   string
		lh, lw int
	}{{"480p", 30, 54}, {"768p", 48, 84}} {
		z := NewTensor(c.LatentChannels, 37, s.lh, s.lw)
		for i := range z.Data {
			z.Data[i] = float32(r.NormFloat64())
		}
		start := time.Now()
		_, st, err := g.Decode(z, pqc)
		if err != nil {
			t.Fatal(err)
		}
		wall := time.Since(start)
		flops := float64(st.Calls) * g.callFlops()
		t.Logf("%s × 124 frames: %d tile-clips in %d batches of ≤%d; %v device (%.1f TFLOP/s), %v wall",
			s.name, st.Calls, st.Batches, seqs, st.Device.Round(time.Millisecond), flops/st.Device.Seconds()/1e12, wall.Round(time.Millisecond))
	}
}
