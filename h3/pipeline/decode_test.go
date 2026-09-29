package pipeline

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/h3/audiovae"
	"strix-halo-vulkan/h3/vae"
)

// TestDecodeOverlap prices moving the audio decode to the device (VIDEO.md
// M11f): Generate runs it on the CPU beside the video decode, so it costs a
// request only what it adds past the video decode's end, or what it slows
// that decode by. Times the served 480p × 124-frame decode alone, the
// 5.17 s stereo decode alone, and the two together, on random latents.
// Opt-in with H3_OVERLAP=1.
func TestDecodeOverlap(t *testing.T) {
	if os.Getenv("H3_OVERLAP") == "" {
		t.Skip("set H3_OVERLAP=1")
	}
	dev, done := newTestDevice(t)
	defer done()
	dir := filepath.Join(modelDir, "vae")
	cfg, err := vae.LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	pqc, err := vae.LoadPostQuantConv(dir)
	if err != nil {
		t.Fatal(err)
	}
	g, err := vae.NewGPU(dev, dir, 16, 16, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	avae, err := audiovae.Load(filepath.Join(modelDir, "audio_vae"))
	if err != nil {
		t.Fatal(err)
	}

	r := rand.New(rand.NewPCG(5, 6))
	z := vae.NewTensor(cfg.LatentChannels, 37, 30, 54)
	for i := range z.Data {
		z.Data[i] = float32(r.NormFloat64())
	}
	const latents = 207 // 5.17 s at 40 a second
	rows := make([]float32, 2*latents*avae.Cfg.LatentChannels)
	for i := range rows {
		rows[i] = float32(r.NormFloat64())
	}

	videoRun := func() (time.Duration, time.Duration) {
		start := time.Now()
		_, st, err := g.Decode(z, pqc)
		if err != nil {
			t.Fatal(err)
		}
		return time.Since(start), st.Device
	}
	audioRun := func() time.Duration {
		start := time.Now()
		if _, _, err := avae.Decode(rows); err != nil {
			t.Fatal(err)
		}
		return time.Since(start)
	}
	arms := []int{0, 8, 4}
	if v := os.Getenv("H3_OVERLAP_WORKERS"); v != "" {
		arms = nil
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			arms = append(arms, n)
		}
	}
	v, vd := videoRun()
	t.Logf("video alone %v (%v device)", v.Round(time.Millisecond), vd.Round(time.Millisecond))
	defer audiovae.SetWorkers(0)
	for pass := 0; pass < 2; pass++ {
		for _, w := range arms {
			audiovae.SetWorkers(w)
			a := audioRun()
			var aa time.Duration
			start := time.Now()
			var wg sync.WaitGroup
			wg.Add(1)
			go func() { defer wg.Done(); aa = audioRun() }()
			va, vad := videoRun()
			wg.Wait()
			both := time.Since(start)
			t.Logf("pass %d, %2d workers: audio alone %v; together %v (video %v, %v device; audio %v): %+v past the video alone",
				pass, w, a.Round(time.Millisecond), both.Round(time.Millisecond), va.Round(time.Millisecond),
				vad.Round(time.Millisecond), aa.Round(time.Millisecond), (both - v).Round(time.Millisecond))
		}
		v, vd = videoRun()
		t.Logf("video alone %v (%v device)", v.Round(time.Millisecond), vd.Round(time.Millisecond))
	}
}
