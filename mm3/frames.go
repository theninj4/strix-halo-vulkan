package mm3

// The frame loop (N3): the global LM samples a semantic code a frame with
// CFG, the local LM its seven residual codes, and the frame's codes are fed
// back; each frame leaves its condition, the eight hiddens (the global LM's
// and the local LM's seven) mixed by the condition encoder's softmax
// weights and scale, which is linear and so is folded in here: 16 KB a
// frame kept, not 128 (upstream's encoders.py and the encoder's einsum).
//
// Sampling is upstream's (encoders.py at dafe3733) on our RNG:
//
//   - global: CFG 1.5 over the head span (AudioEnd and the codes; the four
//     ids between are masked), restricted to the conditional row's top 50,
//     then top-50 sampling of what is left;
//   - local: CFG 1.5 over a head's 1,024 codes, then top-50 sampling;
//   - the first step only advances past <|audio_start|>: its codes are fed
//     back but it is not a frame; AudioEnd at any step ends the song.

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"time"

	"strix-halo-vulkan/safetensors"
)

const (
	cfgScale = 1.5
	cfgTopK  = 50
	sampleK  = 50
	// MaxFrames is upstream's cap, 6 minutes at 25 frames a second.
	MaxFrames = 9000
	// FrameRate is frames a second of audio.
	FrameRate = 25
)

// Mix is the condition encoder's per-frame weights: softmax of its
// layer_weight_logits times layer_scale, global LM first.
type Mix [dCodes + 1]float32

// LoadMix reads the weights from models/MiniMax-Music3/condition_encoder.
func LoadMix(dir string) (Mix, error) {
	var m Mix
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return m, err
	}
	defer set.Close()
	read := func(name string, n int) ([]float32, error) {
		t, err := set.Get(name)
		if err != nil {
			return nil, err
		}
		if t.Elems() != n {
			return nil, fmt.Errorf("mm3: %s has %d elements, want %d", name, t.Elems(), n)
		}
		return t.F32(nil)
	}
	logits, err := read("layer_weight_logits", len(m))
	if err != nil {
		return m, err
	}
	scale, err := read("layer_scale", 1)
	if err != nil {
		return m, err
	}
	mx := math.Inf(-1)
	for _, v := range logits {
		mx = math.Max(mx, float64(v))
	}
	sum := 0.0
	for _, v := range logits {
		sum += math.Exp(float64(v) - mx)
	}
	for i, v := range logits {
		m[i] = float32(math.Exp(float64(v)-mx)/sum) * scale[0]
	}
	return m, nil
}

// Song is a frame loop's result.
type Song struct {
	// Cond is each frame's mixed condition, [frames][4096].
	Cond [][]float32
	// Codes is each frame's 8 codes (semantic, then c1..c7).
	Codes [][dCodes + 1]int
	// Ended says the LM chose AudioEnd (rather than running into the cap).
	Ended bool
	// Prefill and the per-frame times, wall.
	Prefill, LMTime, DepthTime, SampleTime time.Duration
}

// ErrNoFrames is a prompt that ended the song before its first frame.
var ErrNoFrames = errors.New("mm3: the LM ended the song before its first frame")

// Generate runs the frame loop: ids is the conditional prompt and uncond
// its CFG counterpart (the same length), maxFrames the cap (at most
// MaxFrames, and what the cache leaves after the prompt). progress, if set,
// is called after every frame, and between, if set, after every frame with
// nothing of this model's on the device: an error from it (a cancelled
// request) ends the loop with that error.
func Generate(g *LM, d *Depth, mix Mix, ids, uncond []int32, maxFrames int, rng *rand.Rand, progress func(frames int), between func() error) (*Song, error) {
	T := len(ids)
	if T == 0 || len(uncond) != T {
		return nil, fmt.Errorf("mm3: prompts of %d and %d tokens", T, len(uncond))
	}
	maxFrames = min(maxFrames, MaxFrames, g.o.MaxLen-T-1)
	if maxFrames <= 0 {
		return nil, fmt.Errorf("mm3: a %d-token prompt leaves no room in a %d-position cache", T, g.o.MaxLen)
	}
	s := &Song{}
	t0 := time.Now()
	var pre []Row
	for i := 0; i < T-1; i++ {
		pre = append(pre, Row{ID: ids[i], Slot: 0, Pos: i}, Row{ID: uncond[i], Slot: 1, Pos: i})
	}
	for len(pre) > 0 {
		n := min(len(pre), g.o.Rows)
		if _, _, err := g.Pass(pre[:n], false); err != nil {
			return nil, err
		}
		pre = pre[n:]
	}
	s.Prefill = time.Since(t0)

	step := []Row{{ID: ids[T-1], Slot: 0, Pos: T - 1}, {ID: uncond[T-1], Slot: 1, Pos: T - 1}}
	for f := 0; f <= maxFrames; f++ {
		t0 := time.Now()
		out, _, err := g.Pass(step, true)
		if err != nil {
			return nil, err
		}
		t1 := time.Now()
		sem := sampleGlobal(out.Logits[0], out.Logits[1], rng)
		s.SampleTime += time.Since(t1)
		s.LMTime += t1.Sub(t0)
		if sem < 0 {
			s.Ended = true
			break
		}
		semEmb := g.Embed(CodeOffset + sem)
		t2 := time.Now()
		var st time.Duration
		dout, _, err := d.Frame([dSlots][]float32{out.Hidden[0], out.Hidden[1]}, semEmb, func(_ int, c, u []float32) int {
			t := time.Now()
			defer func() { st += time.Since(t) }()
			return sampleDepth(c, u, rng)
		})
		if err != nil {
			return nil, err
		}
		s.DepthTime += time.Since(t2) - st
		s.SampleTime += st
		if f > 0 {
			cond := make([]float32, hidden)
			for j, v := range out.Hidden[0] {
				cond[j] = mix[0] * v
			}
			for k, h := range dout.Hidden {
				w := mix[k+1]
				for j, v := range h {
					cond[j] += w * v
				}
			}
			var codes [dCodes + 1]int
			codes[0] = sem
			copy(codes[1:], dout.Codes[:])
			s.Cond = append(s.Cond, cond)
			s.Codes = append(s.Codes, codes)
			if progress != nil {
				progress(len(s.Cond))
			}
			if len(s.Cond) >= maxFrames {
				break
			}
		}
		if between != nil {
			if err := between(); err != nil {
				return nil, err
			}
		}
		fb := d.Feedback(semEmb, dout.Codes)
		step = []Row{{Vec: fb, Slot: 0, Pos: T + f}, {Vec: fb, Slot: 1, Pos: T + f}}
	}
	if len(s.Cond) == 0 {
		return nil, ErrNoFrames
	}
	return s, nil
}

// sampleGlobal is upstream's global-LM draw over the head span: the
// semantic code, or -1 for AudioEnd.
func sampleGlobal(cond, uncond []float32, rng *rand.Rand) int {
	g, keep := globalCandidates(cond, uncond)
	i := sampleTopK(g, keep, rng)
	if i == 0 {
		return -1
	}
	return i - (CodeOffset - HeadLo)
}

// globalCandidates is the guided logits over the head span and the entries
// left after the conditional row's top-50 cut and the mask.
func globalCandidates(cond, uncond []float32) ([]float32, []int) {
	n := HeadHi - HeadLo
	idx := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if i == 0 || i >= CodeOffset-HeadLo {
			idx = append(idx, i)
		}
	}
	thr := kthLargest(cond, idx, cfgTopK)
	g := make([]float32, n)
	var keep []int
	for _, i := range idx {
		if cond[i] < thr {
			continue
		}
		g[i] = uncond[i] + cfgScale*(cond[i]-uncond[i])
		keep = append(keep, i)
	}
	return g, keep
}

// sampleDepth is upstream's local-LM draw: CFG over the 1,024 codes, then
// top-50 sampling.
func sampleDepth(cond, uncond []float32, rng *rand.Rand) int {
	g := make([]float32, len(cond))
	idx := make([]int, len(cond))
	for i := range g {
		g[i] = uncond[i] + cfgScale*(cond[i]-uncond[i])
		idx[i] = i
	}
	return sampleTopK(g, idx, rng)
}

// kthLargest is the k-th largest of x over idx (all of them if fewer).
func kthLargest(x []float32, idx []int, k int) float32 {
	v := make([]float32, len(idx))
	for j, i := range idx {
		v[j] = x[i]
	}
	sort.Slice(v, func(a, b int) bool { return v[a] > v[b] })
	return v[min(k, len(v))-1]
}

// sampleTopK is upstream's _sample_top_k over the candidates idx of x:
// those at or above the 50th largest, softmax, one draw.
func sampleTopK(x []float32, idx []int, rng *rand.Rand) int {
	keep := topKSupport(x, idx)
	mx := float32(math.Inf(-1))
	for _, i := range keep {
		mx = max(mx, x[i])
	}
	p := make([]float64, len(keep))
	sum := 0.0
	for j, i := range keep {
		p[j] = math.Exp(float64(x[i] - mx))
		sum += p[j]
	}
	u := rng.Float64() * sum
	for j, e := range p {
		if u < e {
			return keep[j]
		}
		u -= e
	}
	return keep[len(keep)-1]
}

// topKSupport is the candidates a top-50 draw can pick: idx at or above
// the 50th largest of x (more than 50 on a tie, as torch's threshold keeps).
func topKSupport(x []float32, idx []int) []int {
	thr := kthLargest(x, idx, sampleK)
	var keep []int
	for _, i := range idx {
		if x[i] >= thr {
			keep = append(keep, i)
		}
	}
	return keep
}
