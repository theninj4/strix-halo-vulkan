package vae

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"strix-halo-vulkan/h3/dit"
	"strix-halo-vulkan/safetensors"
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

// bits logs a hash of a decode's values: what a scheduling change to the
// kernels (KERNELS.md decision 3) must leave the same, across binaries.
func bits(t *testing.T, name string, v []float32) {
	t.Helper()
	h := sha256.New()
	binary.Write(h, binary.LittleEndian, v)
	t.Logf("%s bits %x", name, h.Sum(nil)[:12])
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
// run, batched as one call of four sequences. Then G3's fused epilogues
// (KERNELS.md) against their controls, bit for bit: the short decode again,
// and a batch of eight tile-clips, whose rows are whole GEMM tiles, so v is
// stored by its projection.
func TestGPUDecoder(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 5 GB")
	}
	m := loadManifest(t)
	c := testConfig(t)
	g, done := newTestGPU(t, 16, 16, 8)
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
	bits(t, "short decode", dec.Data)
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

	same := func(name string, a, b *Tensor) {
		t.Helper()
		for i := range a.Data {
			if math.Float32bits(a.Data[i]) != math.Float32bits(b.Data[i]) {
				t.Fatalf("%s: the fused epilogues differ from their controls at %d: %g against %g", name, i, a.Data[i], b.Data[i])
			}
		}
	}
	batch := make([]*Tensor, 8)
	for i := range batch {
		batch[i] = NewTensor(tileIn.C, tileIn.T, tileIn.H, tileIn.W)
		for j, v := range tileIn.Data {
			batch[i].Data[j] = v * (1 - float32(i)/16)
		}
	}
	fused, _, err := g.DecodeTiles(batch)
	if err != nil {
		t.Fatal(err)
	}
	g.packV, g.fuseGLU = false, false
	defer func() { g.packV, g.fuseGLU = true, true }()
	ctl, _, err := g.Decode(short, pqc)
	if err != nil {
		t.Fatal(err)
	}
	same("short decode", dec, ctl)
	ctlBatch, _, err := g.DecodeTiles(batch)
	if err != nil {
		t.Fatal(err)
	}
	for i := range fused {
		same(fmt.Sprintf("batch of 8, clip %d", i), fused[i], ctlBatch[i])
	}
	t.Logf("fused v and SwiGLU epilogues: bit-identical to the pack and SwiGLU passes")
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
	bits(t, "full decode", dec.Data)
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
// H3_VAE_SHAPES=1; H3_VAE_SEQS sets the batch, H3_VAE_ATTN_PLAIN=1 runs the
// plain attention (M11e's control arm), H3_VAE_FUSE=0 the fused epilogues'
// controls (KERNELS.md G3).
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
	if os.Getenv("H3_VAE_ATTN_PLAIN") != "" {
		if err := g.SetAttention(attnPlain); err != nil {
			t.Fatal(err)
		}
	}
	// H3_VAE_FUSE=0 runs the fused epilogues' controls (KERNELS.md G3).
	if os.Getenv("H3_VAE_FUSE") == "0" {
		g.packV, g.fuseGLU = false, false
	}
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

// TestGPUProfile times one batch of tile-clips a dispatch at a time and
// prints the decoder's cost by kind, with each streaming pass's GB/s against
// the 236 a copy gets (KERNELS.md G6). Opt-in with H3_VAE_SHAPES=1;
// H3_VAE_SEQS sets the batch.
func TestGPUProfile(t *testing.T) {
	if os.Getenv("H3_VAE_SHAPES") == "" {
		t.Skip("set H3_VAE_SHAPES=1")
	}
	seqs := 8
	if s := os.Getenv("H3_VAE_SEQS"); s != "" {
		fmt.Sscan(s, &seqs)
	}
	c := testConfig(t)
	g, done := newTestGPU(t, 16, 16, seqs)
	defer done()
	r := rand.New(rand.NewPCG(5, 6))
	clips := make([]*Tensor, seqs)
	for i := range clips {
		clips[i] = NewTensor(c.LatentChannels, c.ClipTokens(), 16, 16)
		for j := range clips[i].Data {
			clips[i].Data[j] = float32(r.NormFloat64())
		}
	}
	if _, _, err := g.DecodeTiles(clips); err != nil {
		t.Fatal(err)
	}
	st, err := g.Profile(clips)
	if err != nil {
		t.Fatal(err)
	}
	type agg struct {
		d      time.Duration
		fl, mv float64
		n      int
	}
	by := map[string]*agg{}
	var kinds []string
	var total time.Duration
	for _, s := range st {
		a := by[s.Kind]
		if a == nil {
			a = &agg{}
			by[s.Kind] = a
			kinds = append(kinds, s.Kind)
		}
		a.d += s.GPU
		a.fl += s.Flops
		a.mv += s.Bytes
		a.n++
		total += s.GPU
	}
	slices.SortFunc(kinds, func(a, b string) int { return int(by[b].d - by[a].d) })
	t.Logf("a batch of %d tile-clips: %v over %d dispatches", seqs, total.Round(time.Millisecond), len(st))
	for _, k := range kinds {
		a := by[k]
		rate := ""
		if a.fl > 0 {
			rate = fmt.Sprintf("%5.1f TFLOP/s", a.fl/a.d.Seconds()/1e12)
		} else if a.mv > 0 {
			rate = fmt.Sprintf("%5.0f GB/s", a.mv/a.d.Seconds()/1e9)
		}
		t.Logf("  %-14s %4d  %8.1f ms  %5.1f%%  %s", k, a.n, a.d.Seconds()*1e3, 100*a.d.Seconds()/total.Seconds(), rate)
	}
}

// TestGPUAttentionScreen times every attention build on block 0's planes of
// a batch of random tile-clips at the served tile size, and prices each
// against the plain build's context (VIDEO.md M11e). Opt-in with
// H3_VAE_SHAPES=1; H3_VAE_SEQS sets the batch.
func TestGPUAttentionScreen(t *testing.T) {
	if os.Getenv("H3_VAE_SHAPES") == "" {
		t.Skip("set H3_VAE_SHAPES=1")
	}
	seqs := 8
	if s := os.Getenv("H3_VAE_SEQS"); s != "" {
		fmt.Sscan(s, &seqs)
	}
	c := testConfig(t)
	g, done := newTestGPU(t, 16, 16, seqs)
	defer done()
	ok, err := dit.TransposedAttentionOK(g.dev)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("transposed element order: %v", ok)

	r := rand.New(rand.NewPCG(3, 4))
	clips := make([]*Tensor, seqs)
	for i := range clips {
		clips[i] = NewTensor(c.LatentChannels, c.ClipTokens(), 16, 16)
		for j := range clips[i].Data {
			clips[i].Data[j] = float32(r.NormFloat64())
		}
	}
	g.input(clips)
	if err := g.SetAttention(attnPlain); err != nil {
		t.Fatal(err)
	}
	if _, err := g.run(seqs, 0, 1, true, false); err != nil {
		t.Fatal(err)
	}
	ctx := func() []float32 {
		out := make([]float32, 0, seqs*g.seqLen*g.H)
		for s := 0; s < seqs; s++ {
			for row := 0; row < g.seqLen; row++ {
				h := g.hbuf.ReadUint16At(int(g.hCtx)+(s*g.stride+row)*g.ldaH, g.H)
				for _, v := range h {
					out = append(out, safetensors.F16ToF32(v))
				}
			}
		}
		return out
	}
	ref := ctx()

	flops := 4 * float64(seqs) * float64(g.seqLen) * float64(g.seqLen) * float64(g.headDim) * float64(g.heads)
	vs := AttnVariants()
	slices.SortFunc(vs, func(a, b AttnVariant) int { return strings.Compare(a.String(), b.String()) })
	for _, v := range vs {
		if v.T && !ok {
			continue
		}
		if err := g.SetAttention(v); err != nil {
			t.Fatal(err)
		}
		var times []time.Duration
		for rep := 0; rep < 7; rep++ {
			gr := g.newGraph()
			gr.attention(seqs)
			d, err := gr.submit()
			if err != nil {
				t.Fatal(err)
			}
			times = append(times, d)
		}
		slices.Sort(times)
		med := times[len(times)/2]
		rel, rms := gap(ctx(), ref)
		t.Logf("%-22v %8v median (%v..%v), %.1f TFLOP/s; against plain rel %.2e rms %.2e",
			v, med.Round(10*time.Microsecond), times[0].Round(10*time.Microsecond), times[len(times)-1].Round(10*time.Microsecond),
			flops/med.Seconds()/1e12, rel, rms)
		if rel > 5e-3 {
			t.Errorf("%v: rel %.2e against plain", v, rel)
		}
	}
}
