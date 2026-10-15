package pipeline

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// BN is the VAE's non-affine BatchNorm over the 2x2-patchified latent: the
// DiT works in its normalised space, the VAE decoder in the raw one.
type BN struct {
	Mean, Std []float32 // [128]; Std = sqrt(running_var + eps) in float32
}

// LoadBN reads bn.running_mean/var and batch_norm_eps from the vae directory.
func LoadBN(dir string) (*BN, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Eps float64 `json:"batch_norm_eps"`
	}
	if err := json.Unmarshal(buf, &cfg); err != nil || cfg.Eps == 0 {
		return nil, fmt.Errorf("pipeline: vae config has no batch_norm_eps (%v)", err)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	get := func(name string) ([]float32, error) {
		t, err := set.Get(name)
		if err != nil {
			return nil, err
		}
		return t.F32(nil)
	}
	mean, err := get("bn.running_mean")
	if err != nil {
		return nil, err
	}
	v, err := get("bn.running_var")
	if err != nil {
		return nil, err
	}
	if len(mean) != len(v) || len(mean)%4 != 0 {
		return nil, fmt.Errorf("pipeline: bn has %d means and %d variances", len(mean), len(v))
	}
	b := &BN{Mean: mean, Std: make([]float32, len(v))}
	eps := float32(cfg.Eps)
	for i, x := range v {
		b.Std[i] = float32(math.Sqrt(float64(float32(x + eps))))
	}
	return b, nil
}

// ToVAE turns the DiT's tokens [h*w, 128] (row-major over the token grid)
// into the decoder's input [32, 2h, 2w], channel-major: de-normalise, then
// unpatchify — channel c*4 + 2*dy + dx of token (y, x) is pixel
// (2y+dy, 2x+dx) of latent channel c (diffusers' _unpatchify_latents).
func (b *BN) ToVAE(x *qwen.Mat, h, w int) ([]float32, error) {
	if x.Rows != h*w || x.Cols != len(b.Mean) {
		return nil, fmt.Errorf("pipeline: %s tokens for a %dx%d grid of %d channels", x, h, w, len(b.Mean))
	}
	c := x.Cols / 4
	out := make([]float32, c*4*h*w)
	W := 2 * w
	for y := 0; y < h; y++ {
		for xx := 0; xx < w; xx++ {
			tok := x.Row(y*w + xx)
			for ch, v := range tok {
				z := float32(float32(v*b.Std[ch]) + b.Mean[ch])
				lc, dy, dx := ch/4, (ch/2)%2, ch%2
				out[(lc*2*h+2*y+dy)*W+2*xx+dx] = z
			}
		}
	}
	return out, nil
}

// FromVAE is the inverse, for an encoded reference: the posterior mode
// [32, 2h, 2w] → normalised tokens [h*w, 128] (diffusers' Flux3VaeEncoderStep).
func (b *BN) FromVAE(z []float32, h, w int) (*qwen.Mat, error) {
	n := len(b.Mean)
	if len(z) != n*h*w {
		return nil, fmt.Errorf("pipeline: %d latent values for a %dx%d grid of %d channels", len(z), h, w, n)
	}
	out := qwen.NewMat(h*w, n)
	W := 2 * w
	for y := 0; y < h; y++ {
		for xx := 0; xx < w; xx++ {
			tok := out.Row(y*w + xx)
			for ch := range tok {
				lc, dy, dx := ch/4, (ch/2)%2, ch%2
				v := z[(lc*2*h+2*y+dy)*W+2*xx+dx]
				tok[ch] = float32(float32(v-b.Mean[ch]) / b.Std[ch])
			}
		}
	}
	return out, nil
}
