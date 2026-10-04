package gemma4

import (
	"math"
	"math/rand/v2"
	"os"
	"testing"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/safetensors"
)

// TestConfig reads Rune's config.json and checks the numbers the port
// relies on (research/rune-vertical.md's table).
func TestConfig(t *testing.T) {
	c, err := LoadConfig("../models/rune-26b-a4b")
	if os.IsNotExist(err) {
		t.Skip("no models/rune-26b-a4b")
	}
	if err != nil {
		t.Fatal(err)
	}
	full := 0
	for i := range c.Layers {
		if c.Full(i) {
			full++
		}
	}
	if c.Layers != 30 || full != 5 || !c.Full(5) || c.Hidden != 2816 || c.Experts != 128 || c.TopK != 8 ||
		c.MoEInter != 704 || c.Intermediate != 2112 || c.HeadDim != 256 || c.GlobalHeadDim != 512 ||
		c.KVHeads != 8 || c.GlobalKVHeads != 2 || c.Window != 1024 || c.Softcap != 30 ||
		c.Rope["full_attention"].Partial != 0.25 {
		t.Errorf("config %+v", c)
	}
}

// TestQuantQ8_0 holds the packer to the format the LLM's kernels read: the
// gguf package's own dequantiser recovers every weight within half a step.
func TestQuantQ8_0(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	const rows, n = 7, 704
	src := make([]float32, rows*n)
	for i := range src {
		src[i] = float32(r.NormFloat64()) * 0.02
	}
	src[5] = 0.3 // one outlier block
	for i := 32; i < 64; i++ {
		src[i] = 0 // an all-zero block
	}
	var data []byte
	for j := range rows {
		data = quantQ8_0(data, src[j*n:(j+1)*n])
	}
	tn := &gguf.Tensor{Type: gguf.Q8_0, Dims: []int64{n, rows}, Data: data}
	got, err := tn.Dequantize(nil)
	if err != nil {
		t.Fatal(err)
	}
	for b := 0; b < len(src); b += q8Block {
		amax := float32(0)
		for _, v := range src[b : b+q8Block] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		step := safetensors.F16ToF32(safetensors.F32ToF16(amax / 127))
		for i := b; i < b+q8Block; i++ {
			if d := math.Abs(float64(got[i] - src[i])); d > float64(step)*0.5+1e-7+float64(amax)*1e-3 {
				t.Fatalf("weight %d: %v -> %v (step %v)", i, src[i], got[i], step)
			}
		}
	}
}
