package pipeline

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/h3/audiovae"
	"strix-halo-vulkan/h3/textenc"
	"strix-halo-vulkan/h3/vae"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// TestStagings times the two stagings a request still makes that M11b's
// cache does not cover (VIDEO.md M11g): the text encoder's token embedding
// table, read from bf16 and widened to fp32 on the host at every encoder
// staging, and the video VAE's decoder, staged inside every decode at the
// served tile size. Opt-in with H3_STAGINGS=1; evict the files from the page
// cache first to time a cold read.
func TestStagings(t *testing.T) {
	if os.Getenv("H3_STAGINGS") == "" {
		t.Skip("set H3_STAGINGS=1")
	}
	dir := filepath.Join(modelDir, "text_encoder")
	cfg, err := textenc.LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	rows, embed, err := qwen.LoadEmbedding(set, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("embedding table: %d × %d, %.2f GB of fp32, in %v", rows, cfg.HiddenSize,
		float64(len(embed)*4)/1e9, time.Since(start).Round(time.Millisecond))
	set.Close()
	embed = nil

	dev, done := newTestDevice(t)
	defer done()
	stageVAE := func(beside string) {
		start := time.Now()
		g, err := vae.NewGPU(dev, filepath.Join(modelDir, "vae"), 16, 16, 8)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("video VAE decoder%s: %.2f GB of weights, %.2f GB of activations, in %v", beside,
			float64(g.WeightBytes())/1e9, float64(g.ActivationBytes())/1e9, time.Since(start).Round(time.Millisecond))
		g.Destroy()
	}
	stageVAE("")

	// Generate starts the audio decode on the CPU just before the video
	// decode, whose first act is this staging: time it beside one.
	avae, err := audiovae.Load(filepath.Join(modelDir, "audio_vae"))
	if err != nil {
		t.Fatal(err)
	}
	latents := make([]float32, 2*207*avae.Cfg.LatentChannels) // 5.17 s
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		if _, _, err := avae.Decode(latents); err != nil {
			t.Error(err)
		}
	}()
	stageVAE(", beside the audio decode")
	wg.Wait()
}
