package mm3

// The window sampler (N5, upstream's denoise.py): the frames are cut into
// windows of 200 frames (8 s) at a hop of 100, and each window is denoised
// on its own -- the condition encoder over its frames, fresh noise, 30 Euler
// steps of the DiT with CFG 1.7 against a zero condition -- with the
// previous window's latents as a prompt over the overlap:
//
//   - the condition over the overlap is the previous window's (the encoder
//     is not re-run there);
//   - before every step the overlap's latents are blended from the noise
//     toward the previous window's, (1 − (1 − 1e-6)·t)·noise + t·prev;
//   - after the last step they are put back as the previous window's;
//   - the carry to the next window is latents [L−344, L−172) and the
//     condition over the same span.
//
// Upstream's scheduler runs the flow time t = i/30 (its sigmas are
// linspace(1, 1/30, 30) inverted) and steps x += (t_{i+1} − t_i)·v with
// t_30 = 1; the model reads t directly, 0 being noise.

import (
	"fmt"
	"math"
	"math/rand/v2"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

const (
	// WindowFrames and WindowHop are upstream's _CHUNK_FRAMES and
	// _CHUNK_HOP.
	WindowFrames = 200
	WindowHop    = 100
	// OverlapLatents is _OVERLAP_LATENT_LENGTH.
	OverlapLatents = 172
	// DiTSteps and DiTGuidance are the reference recipe's.
	DiTSteps    = 30
	DiTGuidance = 1.7
)

// WindowStarts is upstream's chunk_starts for a song of frames frames.
func WindowStarts(frames int) []int {
	if frames <= WindowFrames {
		return []int{0}
	}
	var s []int
	for f := 0; f < frames-WindowHop; f += WindowHop {
		s = append(s, f)
	}
	return s
}

// LatentsFor is how many latents f frames resample to: f · 44100/24000 ·
// 960/512, truncated, at least 1 (the condition encoder's latent_length).
func LatentsFor(f int) int {
	return max(1, int(float64(f)*44100/24000*960/512))
}

// Carry is what a window leaves the next: its latents and condition over
// [L−344, L−172).
type Carry struct {
	Latents [][]float32 // [128][n]
	Cond    []float32   // [n][2048]
}

// Condition runs the condition encoder's convolution over a window's mixed
// frame conditions ([frames][4096], Song.Cond) on the device, then the
// nearest resample to the latent rate on the host: [L][2048].
func (g *DiT) Condition(frames [][]float32) ([]float32, time.Duration, error) {
	F := len(frames)
	if F == 0 || F > WindowFrames {
		return nil, 0, fmt.Errorf("mm3: a window of %d frames", F)
	}
	buf := make([]uint16, F*g.ldaC)
	one := safetensors.F32ToF16(1)
	parallelRows(F, func(f int) {
		row := buf[f*g.ldaC:]
		for k := 0; k < condTaps; k++ {
			src := f + k - 1
			if src < 0 || src >= F {
				continue
			}
			for c, v := range frames[src] {
				row[k*hidden+c] = safetensors.F32ToF16(v)
			}
		}
		row[condTaps*hidden] = one
	})
	g.hbuf.WriteUint16At(int(g.hCond), buf)
	d := []vk.MultiDispatch{g.gemm("big", g.cond.bank, g.hCond, g.aO, g.cond.off, F, CondChannels, g.kC, g.ldaC, nil)}
	took, err := vk.DispatchMultiTimed(d, 1, 1, true)
	if err != nil {
		return nil, took, fmt.Errorf("mm3: condition encoder: %w", err)
	}
	y := g.abuf.ReadFloat32At(int(g.aO), F*CondChannels)
	L := LatentsFor(F)
	out := make([]float32, L*CondChannels)
	scale := float64(F) / float64(L) // torch's nearest: src = floor(dst · in/out)
	for i := 0; i < L; i++ {
		src := min(int(math.Floor(float64(i)*scale)), F-1)
		copy(out[i*CondChannels:(i+1)*CondChannels], y[src*CondChannels:(src+1)*CondChannels])
	}
	return out, took, nil
}

// Noise draws a window's initial latents, [128][L], standard normal.
func Noise(L int, rng *rand.Rand) [][]float32 {
	out := make([][]float32, LatentChannels)
	for c := range out {
		out[c] = make([]float32, L)
		for i := range out[c] {
			out[c][i] = float32(rng.NormFloat64())
		}
	}
	return out
}

// Denoise runs one window: cond is its condition [L][2048] (the overlap's
// is replaced by prev's here), noise its initial latents (consumed), prev
// the previous window's carry or nil. step, if set, is called after every
// step, with nothing of this model's on the device; an error from it ends
// the window. It returns the window's latents and its carry.
func (g *DiT) Denoise(cond []float32, noise [][]float32, prev *Carry, step func(i int) error) ([][]float32, *Carry, time.Duration, error) {
	L := len(noise[0])
	if len(cond) != L*CondChannels {
		return nil, nil, 0, fmt.Errorf("mm3: %d latents and %d condition values", L, len(cond))
	}
	overlap := 0
	var prompt [][]float32
	if prev != nil {
		overlap = min(len(prev.Latents[0]), L)
		copy(cond[:overlap*CondChannels], prev.Cond[:overlap*CondChannels])
		prompt = make([][]float32, LatentChannels)
		for c := range prompt {
			prompt[c] = append([]float32(nil), noise[c][:overlap]...)
		}
	}
	lat := noise
	var total time.Duration
	for i := 0; i < DiTSteps; i++ {
		t := float32(float64(i) / DiTSteps)
		next := float32(float64(i+1) / DiTSteps)
		if overlap > 0 {
			a := 1 - (1-1e-6)*t
			for c := range lat {
				for j := 0; j < overlap; j++ {
					lat[c][j] = a*prompt[c][j] + t*prev.Latents[c][j]
				}
			}
		}
		pc, pu, d, err := g.Forward(lat, cond, t)
		total += d
		if err != nil {
			return nil, nil, total, err
		}
		dt := next - t
		for c := range lat {
			for j := range lat[c] {
				v := pu[c][j] + DiTGuidance*(pc[c][j]-pu[c][j])
				lat[c][j] += dt * v
			}
		}
		if step != nil {
			if err := step(i); err != nil {
				return nil, nil, total, err
			}
		}
	}
	if overlap > 0 {
		for c := range lat {
			copy(lat[c][:overlap], prev.Latents[c][:overlap])
		}
	}
	lo := max(0, L-2*OverlapLatents)
	hi := max(lo, L-OverlapLatents)
	carry := &Carry{Latents: make([][]float32, LatentChannels), Cond: append([]float32(nil), cond[lo*CondChannels:hi*CondChannels]...)}
	for c := range lat {
		carry.Latents[c] = append([]float32(nil), lat[c][lo:hi]...)
	}
	return lat, carry, total, nil
}
