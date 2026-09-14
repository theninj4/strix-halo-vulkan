package parakeet

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The reference comes from reference/dump_parakeet.py, which runs
// transformers' ParakeetForTDT in fp32 on CPU over models/parakeet-tdt-0.6b-v3
// and dumps the front end, one encoder layer stage by stage, every layer's
// output, the prediction network's LSTM state and the whole TDT decode trace.
// Regenerate with:
//
//	.venv/bin/python reference/dump_parakeet.py
const (
	refDir     = "../reference/out/parakeet"
	modelDir   = "../models/parakeet-tdt-0.6b-v3"
	fixture    = "../testdata/jfk.wav"
	transcript = "And so, my fellow Americans, ask not what your country can do for you, " +
		"ask what you can do for your country."
)

type tensorMeta struct {
	Shape  []int   `json:"shape"`
	Count  int     `json:"count"`
	Sum    float64 `json:"sum"`
	AbsMax float64 `json:"absmax"`
}

type manifest struct {
	dir string

	Audio    string `json:"audio"`
	Samples  int    `json:"samples"`
	Expected string `json:"expected"`
	FrontEnd struct {
		NFFT        int     `json:"n_fft"`
		HopLength   int     `json:"hop_length"`
		WinLength   int     `json:"win_length"`
		Preemphasis float64 `json:"preemphasis"`
		NMels       int     `json:"n_mels"`
		Frames      int     `json:"frames"`
		ValidFrames int     `json:"valid_frames"`
	} `json:"front_end"`
	Encoder struct {
		Layers      int     `json:"layers"`
		HiddenSize  int     `json:"hidden_size"`
		Heads       int     `json:"heads"`
		HeadDim     int     `json:"head_dim"`
		Frames      int     `json:"frames"`
		ValidFrames int     `json:"valid_frames"`
		StackGap    float64 `json:"stack_gap"`
		BNFoldGap   float64 `json:"batchnorm_fold_gap"`
		BNEps       float64 `json:"batchnorm_eps"`
		Attention   struct {
			MaskAllTrue bool `json:"mask_is_all_true_over_valid"`
			Masked      int  `json:"masked_positions"`
		} `json:"attention"`
	} `json:"encoder"`
	TDT struct {
		VocabSize    int   `json:"vocab_size"`
		BlankTokenID int   `json:"blank_token_id"`
		Durations    []int `json:"durations"`
	} `json:"tdt"`
	Decode struct {
		Steps int `json:"steps"`
		Trace []struct {
			T        int `json:"t"`
			Token    int `json:"token"`
			Duration int `json:"duration"`
		} `json:"trace"`
		Tokens   []int  `json:"tokens"`
		Emitted  []int  `json:"emitted"`
		Text     string `json:"text"`
		Matches  bool   `json:"matches_expected"`
		SameAsHF bool   `json:"manual_matches_generate"`
	} `json:"decode"`
	Tensors map[string]tensorMeta `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_parakeet.py", refDir, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	m.dir = refDir
	return &m
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

// compare reports the worst absolute and relative deviation between a
// computed tensor and the reference, in the form the other stages of this
// repository report it: an absolute bound is meaningless without the scale it
// is measured against, so both are printed and the caller bounds the one it
// cares about.
type deviation struct {
	MaxAbs float64
	RMS    float64 // of the reference, for scale
	ErrRMS float64 // of the difference, which is the measure fp16 needs
	At     int
}

// Rel is the deviation as a fraction of the tensor's own scale, in the rms
// sense: the measure to bound a *narrowed* path by.
//
// MaxAbs/RMS is the right measure for the CPU reference, where an error is a
// bug in one element and the rest of the tensor is exact. It is the wrong one
// for fp16 operands, where every element carries a relative error of its own
// magnitude: the largest element of these tensors is 5-30x their rms, so its
// rounding alone puts MaxAbs/RMS in the percent range with nothing wrong.
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
