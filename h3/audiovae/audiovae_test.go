package audiovae

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The reference is reference/dump_h3_audio.py: diffusers' decode in fp32 of
// M4's final audio latents, stage by stage. Regenerate with:
//
//	.venv/bin/python reference/dump_h3_audio.py   (seconds)
const (
	audioRef = "../../reference/out/h3audio"
	ditRef   = "../../reference/out/h3dit"
	vaeDir   = "../../models/MiniMax-H3/audio_vae"
)

type manifest struct {
	AudioLatents int `json:"audio_latents"`
	Tensors      map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// channel reads stereo channel s of a dumped [2, C, T] tensor as a
// channel-last [T, C] Mat.
func channel(t *testing.T, m *manifest, name string, s int) *Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok || len(meta.Shape) != 3 {
		t.Fatalf("no [2, C, T] tensor %q", name)
	}
	C, T := meta.Shape[1], meta.Shape[2]
	raw := readF32(t, filepath.Join(audioRef, name+".bin"))
	x := NewMat(T, C)
	for c := 0; c < C; c++ {
		for i := 0; i < T; i++ {
			x.Data[i*C+c] = raw[(s*C+c)*T+i]
		}
	}
	return x
}

func gap(got, want []float32) (rel, rmsRel float64) {
	var maxAbs, ref, sq, refSq float64
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		maxAbs = math.Max(maxAbs, d)
		ref = math.Max(ref, math.Abs(float64(want[i])))
		sq += d * d
		refSq += float64(want[i]) * float64(want[i])
	}
	return maxAbs / ref, math.Sqrt(sq / refSq)
}

func setup(t *testing.T) (*manifest, *Decoder) {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(audioRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_h3_audio.py", audioRef, err)
	}
	m := &manifest{}
	if err := json.Unmarshal(buf, m); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	d, err := Load(vaeDir)
	if err != nil {
		t.Skipf("no audio VAE: %v", err)
	}
	t.Logf("loaded in %v", time.Since(start).Round(time.Millisecond))
	return m, d
}

// TestStages is M6's gate, a piece at a time on the left channel, each fed
// the oracle's own input: the front, stage 0's first alias-free
// activation alone, every stage, and the tail.
func TestStages(t *testing.T) {
	m, d := setup(t)
	check := func(name string, got []float32, want *Mat, tol float64) {
		t.Helper()
		rel, rms := gap(got, want.Data)
		t.Logf("%-8s rel %.2e rms %.2e", name, rel, rms)
		if rel > tol {
			t.Errorf("%s rel %.2e > %.0e", name, rel, tol)
		}
	}
	check("front", d.Front(channel(t, m, "z", 0)).Data, channel(t, m, "pre", 0), 1e-5)
	check("act0", d.Blocks[0].Acts[0].Apply(channel(t, m, "act0_in", 0)).Data, channel(t, m, "act0_out", 0), 1e-5)
	in := channel(t, m, "pre", 0)
	for i := range d.Ups {
		start := time.Now()
		got := d.Stage(i, in)
		name := fmt.Sprintf("stage%d", i)
		want := channel(t, m, name, 0)
		t.Logf("stage %d: %v", i, time.Since(start).Round(time.Millisecond))
		check(name, got.Data, want, 1e-4)
		in = want
	}
	wave := readF32(t, filepath.Join(audioRef, "wave.bin"))
	check("tail", d.Tail(in), &Mat{Data: wave[:len(wave)/2]}, 1e-4)
}

// TestDecode runs the whole stereo decode from the transformer's rows.
func TestDecode(t *testing.T) {
	m, d := setup(t)
	rows := readF32(t, filepath.Join(ditRef, "f6_audio.bin"))
	start := time.Now()
	l, r, err := d.Decode(rows)
	if err != nil {
		t.Fatal(err)
	}
	took := time.Since(start)
	wave := readF32(t, filepath.Join(audioRef, "wave.bin"))
	if len(l)+len(r) != len(wave) {
		t.Fatalf("%d + %d samples, want %d", len(l), len(r), len(wave))
	}
	relL, rmsL := gap(l, wave[:len(l)])
	relR, rmsR := gap(r, wave[len(l):])
	// Signal-to-error over the clip, in dB.
	var sig, noise float64
	for i, v := range append(append([]float32(nil), l...), r...) {
		sig += float64(wave[i]) * float64(wave[i])
		e := float64(v) - float64(wave[i])
		noise += e * e
	}
	t.Logf("%.2f s of stereo at %d Hz in %v: left rel %.2e rms %.2e, right rel %.2e rms %.2e, SNR %.1f dB",
		float64(len(l))/float64(d.Cfg.SamplingRate), d.Cfg.SamplingRate, took.Round(time.Millisecond),
		relL, rmsL, relR, rmsR, 10*math.Log10(sig/noise))
	if rmsL > 1e-4 || rmsR > 1e-4 {
		t.Errorf("rms %.2e / %.2e", rmsL, rmsR)
	}
	if path := os.Getenv("H3_AUDIO_RAW"); path != "" {
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		buf := make([]byte, 8*len(l))
		for i := range l {
			binary.LittleEndian.PutUint32(buf[8*i:], math.Float32bits(l[i]))
			binary.LittleEndian.PutUint32(buf[8*i+4:], math.Float32bits(r[i]))
		}
		f.Write(buf)
		f.Close()
		t.Logf("wrote %s: f32le stereo", path)
	}
	_ = m
}
