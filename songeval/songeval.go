package songeval

import (
	"fmt"
	"math"
	"time"

	"strix-halo-vulkan/vk"
)

// Model is SongEval, staged for songs of up to a ceiling's seconds.
type Model struct {
	g *GPU
	w *weights
}

// Load stages MuQ (muqDir: models/MuQ-large-msd-iter) and SongEval's head
// (headPath: its ckpt/model.safetensors) for songs of up to maxSeconds.
func Load(dev *vk.Device, muqDir, headPath string, maxSeconds float64) (*Model, error) {
	w, err := loadWeights(muqDir, headPath)
	if err != nil {
		return nil, err
	}
	g, err := newGPU(dev, w, maxSeconds)
	if err != nil {
		return nil, err
	}
	return &Model{g: g, w: w}, nil
}

// Destroy releases the device state.
func (m *Model) Destroy() {
	if m.g != nil {
		m.g.Destroy()
		m.g = nil
	}
}

// Result is a song's five scores and the device time.
type Result struct {
	Scores
	Tokens int
	Took   time.Duration
}

// Score scores interleaved 48 kHz stereo (Mono24k, then ScoreMono).
func (m *Model) Score(stereo []float32) (*Result, error) {
	return m.ScoreMono(Mono24k(stereo))
}

// tokens is a song's 25 Hz length for its mel frames: two stride-2 convs.
func tokens(frames int) (t1, t2 int) {
	t1 = (frames + 1) / 2
	return t1, (t1 + 1) / 2
}

// ScoreMono scores 24 kHz mono.
func (m *Model) ScoreMono(wav []float32) (*Result, error) {
	g := m.g
	mel, frames := m.w.Mel(wav)
	t1, T := tokens(frames)
	if T == 0 {
		return nil, fmt.Errorf("songeval: %d samples is too short", len(wav))
	}
	if T > g.maxTok {
		return nil, fmt.Errorf("songeval: %d tokens; staged for %d", T, g.maxTok)
	}
	res := &Result{Tokens: T}
	gr := &graph{g: g}
	for a := 0; a < T; a += chunkTok {
		g.putMel(mel, frames, a)
		gr.chunk(a, frames, t1, T)
		took, err := gr.submit()
		if err != nil {
			return nil, err
		}
		res.Took += took
	}
	g.putRotary(m.w, T)
	hiddenOff := gr.encode(T, Layers)
	ssl, last := gr.head(hiddenOff, T)
	took, err := gr.submit()
	if err != nil {
		return nil, err
	}
	res.Took += took
	m.finish(ssl, last, T, &res.Scores)
	return res, nil
}

// finish is the head's tail on the host: the last attention's mean over
// time beside the FFN's max, the linear to five, tanh·2 + 3.
func (m *Model) finish(ssl, last uint32, T int, s *Scores) {
	f := m.g.abuf.ReadFloat32At(int(ssl), T*hidden)
	a := m.g.abuf.ReadFloat32At(int(last), T*hidden)
	feat := make([]float64, 2*hidden)
	for c := 0; c < hidden; c++ {
		var sum float64
		mx := math.Inf(-1)
		for t := 0; t < T; t++ {
			sum += float64(a[t*hidden+c])
			mx = math.Max(mx, float64(f[t*hidden+c]))
		}
		feat[c], feat[hidden+c] = sum/float64(T), mx
	}
	for d := 0; d < 5; d++ {
		v := float64(m.w.fcB[d])
		for i, x := range feat {
			v += float64(m.w.fcW[d*2*hidden+i]) * x
		}
		*s.dim(d) = math.Tanh(v)*2 + 3
	}
}
