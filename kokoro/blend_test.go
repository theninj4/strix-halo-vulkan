package kokoro

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The blend oracle is reference/dump_blend.py, which runs hexgrad's own
// KPipeline.load_voice over the local .pt packs -- upstream's code path, not a
// reimplementation of it -- and dumps the whole [510, 256] mixed pack for each
// case. Regenerate it with:
//
//	.venv/bin/python reference/dump_blend.py
const blendDir = "../reference/out/blend"

type blendCase struct {
	File    string    `json:"file"`
	Kind    string    `json:"kind"` // "upstream" or "weighted"
	Names   []string  `json:"names"`
	Weights []float64 `json:"weights"`
	Shape   []int     `json:"shape"`
	Count   int       `json:"count"`
	Sum     float64   `json:"sum"`
	AbsMax  float64   `json:"absmax"`
}

type blendManifest struct {
	Cases     map[string]blendCase `json:"cases"`
	SingleGap float64              `json:"single_gap"`
}

func loadBlendManifest(t *testing.T) *blendManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(blendDir, "manifest.json"))
	if err != nil {
		t.Skipf("no blend dump in %s (%v); run reference/dump_blend.py", blendDir, err)
	}
	var m blendManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func loadBlendPack(t *testing.T, c blendCase) []float32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(blendDir, c.File))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != c.Count*4 {
		t.Fatalf("%s: %d bytes for %d float32", c.File, len(raw), c.Count)
	}
	out := make([]float32, c.Count)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// TestBlendAgainstUpstream is the stage's measurement: every row of every
// mixed pack, against the mix upstream computes.
//
// It walks all 510 rows rather than the one an utterance would use, because
// the Go side mixes *per row* on the grounds that a weighted mean is linear
// and upstream mixes whole packs. That claim is either true at every length or
// it is a bug that a single-length check would not see.
func TestBlendAgainstUpstream(t *testing.T) {
	man := loadBlendManifest(t)
	m := loadModel(t)
	for spec, c := range man.Cases {
		t.Run(spec, func(t *testing.T) {
			want := loadBlendPack(t, c)
			dim := c.Shape[1]
			if dim != m.voiceDim {
				t.Fatalf("reference dim %d, model %d", dim, m.voiceDim)
			}
			// The gap between the two computations is the rounding error of
			// a float32 sum, and that error is set by the size of the
			// *terms* rather than of the answer: three style channels of
			// order 1 can cancel to 1e-11, and one ulp of the terms is then
			// thousands of ulps of the result. So the bound is N ulps of the
			// largest term at that channel -- the textbook bound for summing
			// N floats -- which is a statement about arithmetic and not a
			// tolerance chosen to pass.
			packs := make([][]float32, len(c.Names))
			for i, n := range c.Names {
				v, ok := m.Voices[n]
				if !ok {
					t.Fatalf("the checkpoint has no voice %q", n)
				}
				packs[i] = v
			}
			var worstUlps float64
			var differing, total int
			for row := 0; row < c.Shape[0]; row++ {
				dec, pred, err := m.Style(spec, row+1)
				if err != nil {
					t.Fatal(err)
				}
				got := append(append([]float32{}, dec...), pred...)
				want := want[row*dim : (row+1)*dim]
				if len(got) != len(want) {
					t.Fatalf("row %d: %d channels, reference has %d", row, len(got), len(want))
				}
				for i := range got {
					total++
					if got[i] == want[i] {
						continue
					}
					differing++
					var term float32
					for _, p := range packs {
						if x := float32(math.Abs(float64(p[row*dim+i]))); x > term {
							term = x
						}
					}
					u := math.Abs(float64(got[i])-float64(want[i])) / (float64(len(packs)) * ulp32(term))
					if u > worstUlps {
						worstUlps = u
					}
				}
			}
			t.Logf("%s (%s): %d of %d channels differ, worst %.3g of the N-ulp bound",
				spec, c.Kind, differing, total, worstUlps)
			if worstUlps > 1 {
				t.Errorf("worst disagreement is %.3gx the N-ulp bound, want at most 1", worstUlps)
			}
		})
	}
}

// ulp32 is the spacing of float32 at x: the gap between it and the next
// representable number away from zero.
func ulp32(x float32) float64 {
	a := float32(math.Abs(float64(x)))
	if a == 0 {
		return float64(math.Float32frombits(1))
	}
	return float64(math.Float32frombits(math.Float32bits(a)+1) - a)
}

// TestBlendOfOneIsThePack is the property the spelling has to keep: whatever
// blends mean, a single name must go on meaning exactly what it meant before
// they existed. The single path returns the checkpoint's own memory, so this
// is bit equality and not a tolerance.
func TestBlendOfOneIsThePack(t *testing.T) {
	m := loadModel(t)
	for _, spec := range []string{"af_heart", "af_heart:1", "af_heart:0.4", " af_heart "} {
		dec, pred, err := m.Style(spec, 48)
		if err != nil {
			t.Fatalf("%q: %v", spec, err)
		}
		want := m.Voices["af_heart"][47*m.voiceDim : 48*m.voiceDim]
		got := append(append([]float32{}, dec...), pred...)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%q: channel %d is %v, want %v", spec, i, got[i], want[i])
			}
		}
	}
}

// TestBlendWeightsAreNormalised: bare weights need not add up, which is the
// decision that removes an error class rather than adding one. 3:1 and
// 0.75:0.25 are the same mix, and both are checked against the dump above --
// here they are checked against each other, exactly, because the normalised
// weights are identical numbers.
func TestBlendWeightsAreNormalised(t *testing.T) {
	m := loadModel(t)
	a, _, err := m.Style("af_bella:3,af_sky:1", 48)
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := m.Style("af_bella:0.75,af_sky:0.25", 48)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("channel %d: %v against %v", i, a[i], b[i])
		}
	}
	// And the equal mix is the weighted one with equal weights, which is the
	// claim that upstream's comma form is a special case of ours rather than
	// a second code path.
	c, _, err := m.Style("af_bella,af_sky", 48)
	if err != nil {
		t.Fatal(err)
	}
	d, _, err := m.Style("af_bella:1,af_sky:1", 48)
	if err != nil {
		t.Fatal(err)
	}
	for i := range c {
		if c[i] != d[i] {
			t.Fatalf("equal mix channel %d: %v against %v", i, c[i], d[i])
		}
	}
}

func TestParseBlend(t *testing.T) {
	for _, tc := range []struct {
		spec    string
		names   []string
		weights []float32
	}{
		{"af_heart", []string{"af_heart"}, []float32{1}},
		{"af_bella,af_sky", []string{"af_bella", "af_sky"}, []float32{0.5, 0.5}},
		{" af_bella , af_sky ", []string{"af_bella", "af_sky"}, []float32{0.5, 0.5}},
		{"af_bella:3,af_sky:1", []string{"af_bella", "af_sky"}, []float32{0.75, 0.25}},
		{"af_bella:0.75,af_sky:0.25", []string{"af_bella", "af_sky"}, []float32{0.75, 0.25}},
		{"a:1,b:1,c:2", []string{"a", "b", "c"}, []float32{0.25, 0.25, 0.5}},
		{"a:0,b:1", []string{"a", "b"}, []float32{0, 1}},
	} {
		b, err := ParseBlend(tc.spec)
		if err != nil {
			t.Errorf("%q: %v", tc.spec, err)
			continue
		}
		if strings.Join(b.Names, "|") != strings.Join(tc.names, "|") {
			t.Errorf("%q: names %v, want %v", tc.spec, b.Names, tc.names)
		}
		for i, w := range tc.weights {
			if math.Abs(float64(b.Weights[i]-w)) > 1e-6 {
				t.Errorf("%q: weight %d is %v, want %v", tc.spec, i, b.Weights[i], w)
			}
		}
	}
}

func TestParseBlendRejects(t *testing.T) {
	for _, spec := range []string{
		"",                     // nothing at all
		"af_bella,",            // a trailing comma is an empty component
		"af_bella,,af_sky",     // and so is a doubled one
		"af_bella:0.7,af_sky",  // half-weighted: two readings, so neither
		"af_bella:x,af_sky:1",  // not a number
		"af_bella:-1,af_sky:2", // negative
		"af_bella:0,af_sky:0",  // every weight zero
		"af_bella: ,af_sky:1",  // a weight that is only whitespace
	} {
		if b, err := ParseBlend(spec); err == nil {
			t.Errorf("%q parsed as %v, want an error", spec, b)
		}
	}
}

// TestUnknownVoiceNamesTheComponent is the message that started T8: a 400
// reading `no voice "af_alloy,af_bella,af_heart"` does not say which third of
// it to fix.
func TestUnknownVoiceNamesTheComponent(t *testing.T) {
	m := loadModel(t)
	err := m.CheckVoice("af_bella,af_nobody,af_sky")
	if err == nil {
		t.Fatal("a blend naming a voice that does not exist was accepted")
	}
	var unknown *UnknownVoiceError
	if !errors.As(err, &unknown) {
		t.Fatalf("error %v is not an *UnknownVoiceError", err)
	}
	if unknown.Name != "af_nobody" {
		t.Errorf("blamed %q, want af_nobody", unknown.Name)
	}
	if !strings.Contains(unknown.Blend, "af_nobody") || unknown.Blend == "" {
		t.Errorf("the blend %q does not carry the whole specification", unknown.Blend)
	}
	if len(unknown.Have) != len(m.Voices) {
		t.Errorf("listed %d voices, the checkpoint has %d", len(unknown.Have), len(m.Voices))
	}
	if err := m.CheckVoice("af_heart"); err != nil {
		t.Errorf("a voice that exists was refused: %v", err)
	}
}
