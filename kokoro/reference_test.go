package kokoro

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The reference comes from reference/dump_kokoro.py, which runs hexgrad's own
// KModel in fp32 on CPU over the converted models/Kokoro-82M and dumps ALBERT
// layer by layer, the duration encoder block by block, the length regulator,
// the F0/energy stacks, the text encoder and the whole vocoder. Regenerate
// both of these, in this order:
//
//	.venv/bin/python reference/convert_kokoro.py
//	.venv/bin/python reference/dump_kokoro.py
const (
	refDir   = "../reference/out/kokoro"
	modelDir = "../models/Kokoro-82M"
)

type tensorMeta struct {
	Shape  []int   `json:"shape"`
	Count  int     `json:"count"`
	Sum    float64 `json:"sum"`
	AbsMax float64 `json:"absmax"`
}

type manifest struct {
	dir string

	Text       string  `json:"text"`
	Phonemes   string  `json:"phonemes"`
	InputIDs   []int   `json:"input_ids"`
	Voice      string  `json:"voice"`
	VoiceRow   int     `json:"voice_row"`
	Speed      float32 `json:"speed"`
	SampleRate int     `json:"sampling_rate"`

	BERT struct {
		Layers   int     `json:"layers"`
		Heads    int     `json:"heads"`
		HeadDim  int     `json:"head_dim"`
		LayerGap float64 `json:"layer0_gap"`
		StackGap float64 `json:"stack_gap"`
	} `json:"bert"`
	LengthRegulator struct {
		Tokens    int     `json:"tokens"`
		Frames    int     `json:"frames"`
		Durations []int   `json:"durations"`
		Samples   int     `json:"samples"`
		Seconds   float64 `json:"seconds"`
	} `json:"length_regulator"`
	Decoder struct {
		EndToEndGap float64 `json:"end_to_end_gap"`
		Samples     int     `json:"samples"`
	} `json:"decoder"`

	Tensors map[string]tensorMeta `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_kokoro.py", refDir, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	m.dir = refDir
	return &m
}

func loadModel(t *testing.T) *Model {
	t.Helper()
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors")); err != nil {
		t.Skipf("no converted checkpoint in %s (%v); run reference/convert_kokoro.py", modelDir, err)
	}
	m, err := Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// loadRef reads a dumped tensor as a flat float32 slice, with its metadata.
func loadRef(t *testing.T, m *manifest, name string) ([]float32, tensorMeta) {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(m.dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != meta.Count*4 {
		t.Fatalf("%s: %d bytes for %d float32", name, len(raw), meta.Count)
	}
	data := make([]float32, meta.Count)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return data, meta
}

// refMat reads a dumped tensor into this package's [frames, channels] layout.
//
// The reference dumps the convolutional path as [C, T], because that is what
// PyTorch holds; this package holds [T, C], because that is what makes the
// pointwise convolutions GEMMs. `channelFirst` says which of the two a given
// tensor is, and the transpose happens here rather than in the model.
func refMat(t *testing.T, m *manifest, name string, channelFirst bool) *Mat {
	t.Helper()
	data, meta := loadRef(t, m, name)
	if len(meta.Shape) != 2 {
		t.Fatalf("%s: shape %v, want 2 dims", name, meta.Shape)
	}
	if !channelFirst {
		return &Mat{Rows: meta.Shape[0], Cols: meta.Shape[1], Data: data}
	}
	c, n := meta.Shape[0], meta.Shape[1]
	out := NewMat(n, c)
	for i := 0; i < c; i++ {
		for j := 0; j < n; j++ {
			out.Data[j*c+i] = data[i*n+j]
		}
	}
	return out
}

// deviation is the same measure the parakeet stages report: an absolute bound
// is meaningless without the scale it is measured against, so both are
// carried and the caller bounds the one it cares about.
type deviation struct {
	MaxAbs float64
	RMS    float64 // of the reference, for scale
	ErrRMS float64 // of the difference
	At     int
}

// Rel is the deviation as a fraction of the tensor's own scale.
func (d deviation) Rel() float64 { return d.ErrRMS / d.RMS }

func compare(t *testing.T, got, want []float32) deviation {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, reference has %d", len(got), len(want))
	}
	var d deviation
	var sq, errSq float64
	for i := range got {
		diff := math.Abs(float64(got[i]) - float64(want[i]))
		if diff > d.MaxAbs {
			d.MaxAbs, d.At = diff, i
		}
		errSq += diff * diff
		sq += float64(want[i]) * float64(want[i])
	}
	d.RMS = math.Sqrt(sq / float64(len(want)))
	d.ErrRMS = math.Sqrt(errSq / float64(len(want)))
	return d
}

// check compares against a named reference tensor and fails if the relative
// deviation exceeds the bound.
func check(t *testing.T, m *manifest, name string, got *Mat, channelFirst bool, bound float64) deviation {
	t.Helper()
	want := refMat(t, m, name, channelFirst)
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: got %v, reference is %v", name, got, want)
	}
	d := compare(t, got.Data, want.Data)
	if d.Rel() > bound {
		t.Errorf("%s: relative %.3g (max abs %.3g at %d, rms %.3g) exceeds %.3g",
			name, d.Rel(), d.MaxAbs, d.At, d.RMS, bound)
	} else {
		t.Logf("%-14s %-12s max abs %.3g  rms %.3g  relative %.3g", name, got, d.MaxAbs, d.RMS, d.Rel())
	}
	return d
}
