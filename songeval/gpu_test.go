package songeval

import (
	"math"
	"strconv"
	"sync"
	"testing"

	"strix-halo-vulkan/vk"
)

var (
	once  sync.Once
	model *Model
	skip  string
)

func load(t *testing.T) *Model {
	t.Helper()
	once.Do(func() {
		inst, err := vk.NewInstance("songeval-test")
		if err != nil {
			skip = err.Error()
			return
		}
		devs, err := inst.PhysicalDevices()
		if err != nil || len(devs) == 0 {
			skip = "no Vulkan device"
			return
		}
		phys := &devs[0]
		qf, err := phys.ComputeQueueFamily()
		if err != nil {
			skip = err.Error()
			return
		}
		sgs, err := phys.SubgroupSizeControl()
		if err != nil {
			skip = err.Error()
			return
		}
		dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
		if err != nil {
			skip = err.Error()
			return
		}
		if model, err = Load(dev, muqDir, headPath, 130); err != nil {
			skip = err.Error()
		}
	})
	if skip != "" {
		t.Skip(skip)
	}
	return model
}

// TestStages is A14d's stage gate, each case from upstream's own 24 kHz
// input: the conv front end's output (hidden_states[0], through the chunks
// and their halos), the six layers, and the head's FFN.
func TestStages(t *testing.T) {
	m := loadManifest(t)
	md := load(t)
	g := md.g
	for _, c := range cases {
		mel, frames := md.w.Mel(readF32(t, c+"_wav24"))
		t1, T := tokens(frames)
		if T != m.Cases[c].Tokens {
			t.Fatalf("%s: %d tokens, upstream %d", c, T, m.Cases[c].Tokens)
		}
		gr := &graph{g: g}
		for a := 0; a < T; a += chunkTok {
			g.putMel(mel, frames, a)
			gr.chunk(a, frames, t1, T)
			if _, err := gr.submit(); err != nil {
				t.Fatal(err)
			}
		}
		g.putRotary(md.w, T)
		for stop := 0; stop <= Layers; stop++ {
			// The conformer runs in place over aX, so each depth is the front
			// end's output re-staged and that many layers.
			if stop > 0 {
				g.abuf.WriteFloat32At(int(g.aX), front)
			} else {
				front = g.abuf.ReadFloat32At(int(g.aX), T*hidden)
			}
			x := gr.encode(T, stop)
			if _, err := gr.submit(); err != nil {
				t.Fatal(err)
			}
			got := g.abuf.ReadFloat32At(int(x), T*hidden)
			r, mx := rel(got, readF32(t, c+"_hidden"+strconv.Itoa(stop)))
			t.Logf("%-9s hidden %d: rms %.2e max %.2e", c, stop, r, mx)
			if r > 2e-2 {
				t.Errorf("%s: hidden %d rms %.2e", c, stop, r)
			}
			if stop == Layers {
				ssl, _ := gr.head(x, T)
				if _, err := gr.submit(); err != nil {
					t.Fatal(err)
				}
				r, mx := rel(g.abuf.ReadFloat32At(int(ssl), T*hidden), readF32(t, c+"_ffd"))
				t.Logf("%-9s head FFN: rms %.2e max %.2e", c, r, mx)
			}
		}
	}
}

var front []float32

// TestScores: the five scores end to end from the 48 kHz source, against
// upstream from the same (torchaudio) resample, priced against the gap
// between that and eval.py's own (librosa soxr) input.
func TestScores(t *testing.T) {
	m := loadManifest(t)
	md := load(t)
	for _, c := range cases {
		res, err := md.Score(stereo(t, c))
		if err != nil {
			t.Fatal(err)
		}
		mc := m.Cases[c]
		var worst, bar float64
		for i, d := range Dims {
			worst = math.Max(worst, math.Abs(*res.dim(i)-mc.Torchaudio[d]))
			bar = math.Max(bar, math.Abs(mc.Librosa[d]-mc.Torchaudio[d]))
		}
		t.Logf("%-9s %d tokens in %v: %.3f %.3f %.3f %.3f %.3f; upstream %.3f %.3f %.3f %.3f %.3f; worst %.4f (resampler gap %.4f)",
			c, res.Tokens, res.Took, res.Coherence, res.Musicality, res.Memorability, res.Clarity, res.Naturalness,
			mc.Torchaudio["Coherence"], mc.Torchaudio["Musicality"], mc.Torchaudio["Memorability"],
			mc.Torchaudio["Clarity"], mc.Torchaudio["Naturalness"], worst, bar)
		if !(worst <= math.Max(bar, 0.01)) { // NaN fails
			t.Errorf("%s: worst score %.4f off fp32, the resampler moves it %.4f", c, worst, bar)
		}
	}
}
