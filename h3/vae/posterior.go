package vae

// The keyframe's posterior, sampled as the released pipeline samples it
// (`encode_vae_condition`, VIDEO.md M10): the encoder's moments are a
// diagonal Gaussian, the conditioning latent is a *draw* from it under a
// generator of its own seeded 42, and the draw is rounded to fp16 before it
// is normalised. The draw is not a nicety. It moves the normalised latent
// by up to 0.16–0.33 against the posterior mean (dump_h3_fl2va.py's
// `sample_vs_mode`), so the anchor the transformer saw in training is this
// one and not the mode. Reproducing it means reproducing torch's CPU
// `randn`, which is what most of this file is.

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
)

// KeyframeSeed is the seed the released pipeline samples every keyframe's
// posterior under, independent of the request's own.
const KeyframeSeed = 42

// mt19937 is the 32-bit Mersenne Twister torch's CPU generator runs.
type mt19937 struct {
	mt  [624]uint32
	idx int
}

// newMT19937 seeds as std::mt19937 and torch's CPUGeneratorImpl do: the
// seed's low 32 bits, then Knuth's initialisation multiplier.
func newMT19937(seed uint64) *mt19937 {
	m := &mt19937{idx: 624}
	m.mt[0] = uint32(seed)
	for i := 1; i < 624; i++ {
		m.mt[i] = 1812433253*(m.mt[i-1]^(m.mt[i-1]>>30)) + uint32(i)
	}
	return m
}

func (m *mt19937) next() uint32 {
	if m.idx >= 624 {
		for i := 0; i < 624; i++ {
			y := m.mt[i]&0x80000000 | m.mt[(i+1)%624]&0x7fffffff
			v := m.mt[(i+397)%624] ^ y>>1
			if y&1 != 0 {
				v ^= 0x9908b0df
			}
			m.mt[i] = v
		}
		m.idx = 0
	}
	y := m.mt[m.idx]
	m.idx++
	y ^= y >> 11
	y ^= y << 7 & 0x9d2c5680
	y ^= y << 15 & 0xefc60000
	y ^= y >> 18
	return y
}

// uniform is `at::uniform_real_distribution<float>(0, 1)`: the low 24 bits
// of one 32-bit draw, times 2^-24.
func (m *mt19937) uniform() float32 {
	return float32(float64(m.next()&(1<<24-1)) / (1 << 24))
}

// TorchRandn is `torch.randn(n, generator=torch.Generator().manual_seed(seed))`
// for a contiguous float32 tensor of n ≥ 16 elements: torch's `normal_fill`.
// Every element is first a uniform; then each block of 16 is Box–Muller'd
// in place, element j with j + 8; and when n is not a multiple of 16 the
// last 16 are drawn *again* and transformed, overwriting the tail.
//
// torch's transcendental functions are Sleef's (1 ulp) or libm's, and these
// are Go's correctly rounded ones, so an element can differ in its last
// bit. That is below the fp16 rounding the sample goes through next.
func TorchRandn(seed uint64, n int) ([]float32, error) {
	if n < 16 {
		return nil, fmt.Errorf("vae: torch's normal_fill path needs 16 elements, got %d", n)
	}
	g := newMT19937(seed)
	out := make([]float32, n)
	for i := range out {
		out[i] = g.uniform()
	}
	for i := 0; i+16 <= n; i += 16 {
		boxMuller16(out[i : i+16])
	}
	if n%16 != 0 {
		tail := out[n-16:]
		for i := range tail {
			tail[i] = g.uniform()
		}
		boxMuller16(tail)
	}
	return out, nil
}

// boxMuller16 is `normal_fill_16` with mean 0 and std 1, in float32 steps.
func boxMuller16(d []float32) {
	const twoPi = float32(2 * math.Pi)
	for j := 0; j < 8; j++ {
		u1 := 1 - d[j]
		u2 := d[j+8]
		radius := float32(math.Sqrt(float64(-2 * float32(math.Log(float64(u1))))))
		theta := twoPi * u2
		d[j] = radius * float32(math.Cos(float64(theta)))
		d[j+8] = radius * float32(math.Sin(float64(theta)))
	}
}

// SampleLatent turns an encoded keyframe's moments — [2·C, h, w], the mean
// then the log-variance, as quant_conv writes them — into the normalised
// conditioning latent [C, h, w]: `DiagonalGaussianDistribution.sample` under
// a generator seeded `seed`, rounded to fp16, then (z − mean) / std with the
// checkpoint's latent statistics. noise, when non-nil, replaces the draw
// (a gate's negative control).
func (c *Config) SampleLatent(moments *Tensor, seed uint64, noise []float32) (*Tensor, error) {
	C := c.LatentChannels
	if moments.C != 2*C || moments.T != 1 {
		return nil, fmt.Errorf("vae: moments are %dx%d, want %dx1", moments.C, moments.T, 2*C)
	}
	plane := moments.H * moments.W
	n := C * plane
	if noise == nil {
		var err error
		if noise, err = TorchRandn(seed, n); err != nil {
			return nil, err
		}
	} else if len(noise) != n {
		return nil, fmt.Errorf("vae: %d noise values for %d latents", len(noise), n)
	}
	z := NewTensor(C, 1, moments.H, moments.W)
	for ch := 0; ch < C; ch++ {
		m, s := c.LatentsMean[ch], c.LatentsStd[ch]
		for i := 0; i < plane; i++ {
			mean := moments.Data[ch*plane+i]
			logvar := min(max(moments.Data[(C+ch)*plane+i], -30), 20)
			std := float32(math.Exp(float64(0.5 * logvar)))
			// The conversion keeps the product rounded on its own, as torch
			// computes it, rather than fused into the sum.
			x := mean + float32(std*noise[ch*plane+i])
			x = safetensors.F16ToF32(safetensors.F32ToF16(x))
			z.Data[ch*plane+i] = (x - m) / s
		}
	}
	return z, nil
}

// Patchify is Unpatchify's inverse for one latent frame: [C, 1, h, w] to
// the transformer's rows [(h/2)·(w/2), 4·C], row-major over the 2×2 patches,
// columns (channel, 1, 2, 2) — `patchify_video_latents`.
func Patchify(z *Tensor) ([]float32, error) {
	if z.T != 1 || z.H%2 != 0 || z.W%2 != 0 {
		return nil, fmt.Errorf("vae: cannot patchify a %dx%dx%d latent", z.T, z.H, z.W)
	}
	out := make([]float32, 0, len(z.Data))
	for y := 0; y < z.H/2; y++ {
		for x := 0; x < z.W/2; x++ {
			for ch := 0; ch < z.C; ch++ {
				for py := 0; py < 2; py++ {
					for px := 0; px < 2; px++ {
						out = append(out, z.At(ch, 0, 2*y+py, 2*x+px))
					}
				}
			}
		}
	}
	return out, nil
}
