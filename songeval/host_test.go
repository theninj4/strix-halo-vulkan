package songeval

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The reference: reference/dump_songeval.py (MUSIC.md A14d), SongEval's
// eval.py path in fp32 on the CPU, from torchaudio's resample of each song:
//
//	.venv-aes/bin/python -I reference/dump_songeval.py
const (
	ref      = "../reference/out/songeval"
	muqDir   = "../models/MuQ-large-msd-iter"
	headPath = "../models/SongEval-src/ckpt/model.safetensors"
)

var cases = []string{"synthwave", "lofi", "noise"}

type manifest struct {
	Cases map[string]struct {
		Librosa    map[string]float64 `json:"librosa"`
		Torchaudio map[string]float64 `json:"torchaudio"`
		Tokens     int                `json:"tokens"`
	} `json:"cases"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(ref, "manifest.json"))
	if err != nil {
		t.Skipf("no reference (%v); run reference/dump_songeval.py", err)
	}
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func readF32(t *testing.T, name string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(ref, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

func rel(got, want []float32) (float64, float64) {
	var se, sw, mx float64
	for i, w := range want {
		d := float64(got[i]) - float64(w)
		se, sw, mx = se+d*d, sw+float64(w)*float64(w), math.Max(mx, math.Abs(d))
	}
	return math.Sqrt(se / sw), mx
}

func stereo(t *testing.T, c string) []float32 {
	w48 := readF32(t, c+"_wav48")
	n := len(w48) / 2
	st := make([]float32, 2*n)
	for i := 0; i < n; i++ {
		st[2*i], st[2*i+1] = w48[i], w48[n+i]
	}
	return st
}

// TestHost: the 48 -> 24 kHz resample and the normalised mel, against
// torchaudio's.
func TestHost(t *testing.T) {
	loadManifest(t)
	w, err := loadWeights(muqDir, headPath)
	if err != nil {
		t.Skip(err)
	}
	for _, c := range cases {
		got, want := Mono24k(stereo(t, c)), readF32(t, c+"_wav24")
		if len(got) != len(want) {
			t.Fatalf("%s: %d samples, upstream %d", c, len(got), len(want))
		}
		r, _ := rel(got, want)
		mel, frames := w.Mel(want)
		wm := readF32(t, c+"_mel")
		if len(mel) != len(wm) {
			t.Fatalf("%s: %d mel frames, upstream %d values", c, frames, len(wm))
		}
		rm, mx := rel(mel, wm)
		t.Logf("%-9s resample rms %.2e; mel %d frames rms %.2e max %.2e", c, r, frames, rm, mx)
		if r > 1e-6 || rm > 1e-5 {
			t.Errorf("%s: resample %.2e, mel %.2e", c, r, rm)
		}
	}
}
