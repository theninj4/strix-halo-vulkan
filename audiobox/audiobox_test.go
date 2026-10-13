package audiobox

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"strix-halo-vulkan/vk"
)

// The reference: reference/dump_aes.py (MUSIC.md A14c), upstream's
// AesPredictor in fp32 on the CPU:
//
//	.venv-aes/bin/python -I reference/dump_aes.py
const (
	ref      = "../reference/out/aes"
	modelDir = "../models/audiobox-aesthetics"
)

var cases = []string{"synthwave", "lofi", "partial", "noise"}

type manifest struct {
	Cases map[string]struct {
		Windows int                `json:"windows"`
		Weights []float64          `json:"weights"`
		Kept    []int              `json:"kept"`
		Scores  map[string]float64 `json:"scores"`
	} `json:"cases"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(ref, "manifest.json"))
	if err != nil {
		t.Skipf("no reference (%v); run reference/dump_aes.py", err)
	}
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func readRaw(t *testing.T, name string) []uint32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(ref, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]uint32, len(raw)/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(raw[4*i:])
	}
	return out
}

func readF32(t *testing.T, name string) []float32 {
	raw := readRaw(t, name)
	out := make([]float32, len(raw))
	for i, v := range raw {
		out[i] = math.Float32frombits(v)
	}
	return out
}

// rel is the rms error over the reference's rms, and the max abs error.
func rel(got, want []float32) (float64, float64) {
	var se, sw, mx float64
	for i, w := range want {
		d := float64(got[i]) - float64(w)
		se, sw, mx = se+d*d, sw+float64(w)*float64(w), math.Max(mx, math.Abs(d))
	}
	return math.Sqrt(se / sw), mx
}

// TestBuckets: the relative-position buckets of every offset a window has.
func TestBuckets(t *testing.T) {
	loadManifest(t)
	want := readRaw(t, "buckets")
	for i, w := range want {
		if d := i - (Frames - 1); bucket(d) != int(int32(w)) {
			t.Fatalf("offset %d: bucket %d, upstream %d", d, bucket(d), int32(w))
		}
	}
}

// TestResample: 48 kHz stereo to 16 kHz mono against torchaudio's.
func TestResample(t *testing.T) {
	loadManifest(t)
	for _, c := range cases {
		w48 := readF32(t, c+"_wav48") // [2][n]
		n := len(w48) / 2
		st := make([]float32, 2*n)
		for i := 0; i < n; i++ {
			st[2*i], st[2*i+1] = w48[i], w48[n+i]
		}
		got, want := Mono16k(st), readF32(t, c+"_wav16")
		if len(got) != len(want) {
			t.Fatalf("%s: %d samples, upstream %d", c, len(got), len(want))
		}
		r, mx := rel(got, want)
		t.Logf("%-9s %d samples: rms %.2e, max %.2e", c, len(got), r, mx)
		if r > 1e-6 {
			t.Errorf("%s: resample rms %.2e", c, r)
		}
	}
}

var (
	once  sync.Once
	model *Model
	skip  string
)

func load(t *testing.T) *Model {
	t.Helper()
	once.Do(func() {
		inst, err := vk.NewInstance("audiobox-test")
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
		if model, err = Load(dev, modelDir, DefaultWindows); err != nil {
			skip = err.Error()
		}
	})
	if skip != "" {
		t.Skip(skip)
	}
	return model
}

// TestEncoder is A14c's stage gate, over the first and the last window of
// each case from upstream's own 16 kHz audio: the extractor's output, the
// projection, and the encoder's 13 hidden states over each window's valid
// frames, against fp32's.
func TestEncoder(t *testing.T) {
	m := loadManifest(t)
	md := load(t)
	g := md.g
	for _, c := range cases {
		mc := m.Cases[c]
		wins, valid, _ := windows(readF32(t, c+"_wav16"))
		if len(wins) != mc.Windows {
			t.Fatalf("%s: %d windows, upstream %d", c, len(wins), mc.Windows)
		}
		var bw [][]float32
		var bv []int
		for _, w := range mc.Kept {
			bw, bv = append(bw, wins[w]), append(bv, valid[w])
			pad := readRaw(t, c+"_w"+itoa(w)+"_pad")
			n := 0
			for _, p := range pad {
				if p == 0 {
					n++
				}
			}
			if n != valid[w] {
				t.Fatalf("%s window %d: %d valid frames, upstream %d", c, w, valid[w], n)
			}
		}
		rows := func(off uint32, b, n, dim int) []float32 {
			return g.abuf.ReadFloat32At(int(off)+b*framesPad*dim, n*dim)
		}
		for stop := -1; stop <= layers; stop++ {
			if _, _, err := g.run(bw, bv, stop); err != nil {
				t.Fatal(err)
			}
			for b, w := range mc.Kept {
				pre := c + "_w" + itoa(w) + "_"
				n := bv[b]
				if stop == -1 {
					if b == len(mc.Kept)-1 { // the scratch holds the last window's
						r, _ := rel(g.abuf.ReadFloat32At(int(g.aF1), Frames*convDim), readF32(t, pre+"conv"))
						t.Logf("%s w%d extractor out: rms %.2e", c, w, r)
						if r > 2e-3 {
							t.Errorf("%s w%d: extractor rms %.2e", c, w, r)
						}
					}
					want := readF32(t, pre+"proj")[:n*hidden]
					r, _ := rel(rows(g.aX, b, n, hidden), want)
					t.Logf("%s w%d projection: rms %.2e", c, w, r)
					if r > 3e-3 {
						t.Errorf("%s w%d: projection rms %.2e", c, w, r)
					}
					continue
				}
				want := readF32(t, pre+"hidden"+itoa(stop))[:n*hidden]
				r, mx := rel(rows(g.aX, b, n, hidden), want)
				t.Logf("%s w%d hidden %2d: rms %.2e, max %.2e", c, w, stop, r, mx)
				if r > 1e-2 {
					t.Errorf("%s w%d: hidden %d rms %.2e", c, w, stop, r)
				}
			}
		}
	}
}

// TestScores: every window's four scores and the song's, end to end from
// upstream's 16 kHz audio and from the 48 kHz source.
func TestScores(t *testing.T) {
	m := loadManifest(t)
	md := load(t)
	order := map[string]int{"CE": 0, "CU": 1, "PC": 2, "PQ": 3} // upstream's AXES_NAME
	for _, c := range cases {
		mc := m.Cases[c]
		res, err := md.ScoreMono(readF32(t, c+"_wav16"))
		if err != nil {
			t.Fatal(err)
		}
		preds := readF32(t, c+"_preds") // [windows][CE CU PC PQ], normalised
		var worst float64
		for w, win := range res.Windows {
			for a, name := range Axes {
				h := md.w.heads[a]
				want := float64(preds[w*4+order[name]])*h.std + h.mean
				worst = math.Max(worst, math.Abs(*win.axis(a)-want))
			}
			if math.Abs(win.Weight-mc.Weights[w]) > 1e-12 {
				t.Errorf("%s window %d: weight %v, upstream %v", c, w, win.Weight, mc.Weights[w])
			}
		}
		t.Logf("%-9s %2d windows in %v: PQ %.3f PC %.3f CE %.3f CU %.3f (upstream %.3f %.3f %.3f %.3f); worst window %.4f",
			c, len(res.Windows), res.Took, res.PQ, res.PC, res.CE, res.CU,
			mc.Scores["PQ"], mc.Scores["PC"], mc.Scores["CE"], mc.Scores["CU"], worst)
		for a, name := range Axes {
			if d := math.Abs(*res.axis(a) - mc.Scores[name]); d > 0.02 {
				t.Errorf("%s %s: %.4f, upstream %.4f", c, name, *res.axis(a), mc.Scores[name])
			}
		}
	}
}

func itoa(i int) string {
	if i < 0 {
		return "-" + itoa(-i)
	}
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
}
