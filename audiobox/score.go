package audiobox

import (
	"fmt"
	"time"

	"strix-halo-vulkan/vk"
)

// DefaultWindows is a batch's windows: 80 s of audio a submit, ~0.4 GB of
// arenas (mostly the bias planes and the first conv's scratch).
const DefaultWindows = 8

// Model is Audiobox Aesthetics, staged.
type Model struct {
	g *GPU
	w *weights
}

// Load stages the checkpoint in dir (models/audiobox-aesthetics: its
// model.safetensors and config.json) for batches of up to maxWin windows.
func Load(dev *vk.Device, dir string, maxWin int) (*Model, error) {
	if maxWin <= 0 {
		maxWin = DefaultWindows
	}
	w, err := loadWeights(dir)
	if err != nil {
		return nil, err
	}
	g, err := newGPU(dev, w, maxWin)
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

// Window is one 10 s window's scores and its weight in the song's mean:
// its share of a full window's samples.
type Window struct {
	Scores
	Weight float64
}

// Result is a piece of audio's scores, its windows', and the device time.
type Result struct {
	Scores
	Windows []Window
	Took    time.Duration
}

// windows is make_inference_batch over 16 kHz mono: 10 s at a 10 s hop,
// the last zero-padded, each with its valid frames (forward_padding_mask: a
// frame of 320 samples is padding when every sample is) and its weight.
func windows(wav []float32) (wins [][]float32, valid []int, weight []float64) {
	for at := 0; at < len(wav); at += WindowSamples {
		w := make([]float32, WindowSamples)
		n := copy(w, wav[at:])
		wins = append(wins, w)
		valid = append(valid, min(Frames, (n+319)/320))
		weight = append(weight, float64(n)/WindowSamples)
	}
	return wins, valid, weight
}

// Score is AesPredictor.forward on interleaved 48 kHz stereo: resampled to
// 16 kHz mono (Mono16k), then ScoreMono.
func (m *Model) Score(stereo []float32) (*Result, error) {
	return m.ScoreMono(Mono16k(stereo))
}

// ScoreMono scores 16 kHz mono audio: every window through the encoder and
// the four heads, and the song's scores the windows' mean weighted by their
// length.
func (m *Model) ScoreMono(wav []float32) (*Result, error) {
	if len(wav) == 0 {
		return nil, fmt.Errorf("audiobox: no audio")
	}
	wins, valid, weight := windows(wav)
	res := &Result{}
	var total float64
	for lo := 0; lo < len(wins); lo += m.g.maxWin {
		hi := min(lo+m.g.maxWin, len(wins))
		pooled, took, err := m.g.run(wins[lo:hi], valid[lo:hi], layers)
		if err != nil {
			return nil, err
		}
		res.Took += took
		for b, p := range pooled {
			win := Window{Weight: weight[lo+b]}
			for a := range Axes {
				v := m.w.heads[a].score(p)
				*win.axis(a) = v
				*res.axis(a) += v * win.Weight
			}
			total += win.Weight
			res.Windows = append(res.Windows, win)
		}
	}
	for a := range Axes {
		*res.axis(a) /= total
	}
	return res, nil
}
