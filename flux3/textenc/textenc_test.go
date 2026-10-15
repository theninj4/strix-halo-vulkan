package textenc

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// The reference comes from reference/dump_flux3_textenc.py: diffusers' own
// _get_qwen3_vl_prompt_embeds over the checkpoint's Qwen3-VL-4B in fp32 on
// the CPU, plus hidden states 1, 2 and 32 of the short prompt. Regenerate with
//
//	.venv-flux3/bin/python -I reference/dump_flux3_textenc.py
const (
	refDir = "../../reference/out/flux3textenc"
	modDir = "../../models/flux-3-image-klein/text_encoder"
)

type promptCase struct {
	Prompt    string  `json:"prompt"`
	Rendered  string  `json:"rendered"`
	Real      int     `json:"real"`
	Padded    int     `json:"padded"`
	IDs       []int32 `json:"ids"`
	Positions []int   `json:"positions"`
}

type manifest struct {
	PadMultiple int                   `json:"pad_multiple"`
	PadTokenID  int32                 `json:"pad_token_id"`
	Taps        []int                 `json:"layers_tapped"`
	Prompts     map[string]promptCase `json:"prompts"`
	Tensors     map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_flux3_textenc.py", refDir, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func loadRef(t *testing.T, m *manifest, name string) *qwen.Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(refDir, name+".bin"))
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
	cols := meta.Shape[len(meta.Shape)-1]
	return &qwen.Mat{Rows: meta.Count / cols, Cols: cols, Data: data}
}

// relTol is the per-layer fp32 bound zimage/qwen and qimage/textenc hold:
// both sides fp32, only summation order differs.
const relTol = 2e-4

// deepTol is the full-depth bound. qimage/textenc needed 1e-3 at depth 36
// on the 8B (its reference disagreed with itself at 3.2e-4); this port lands
// at rel 3.0e-5 / 4.2e-5 / 3.0e-5 (short, long, cjk; X1, 2026-10-15) after 32
// layers, so the per-layer bound holds all the way down and is kept.
const deepTol = relTol

func compare(t *testing.T, name string, got, want *qwen.Mat, tol float64) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	var sumSq float64
	for _, w := range want.Data {
		sumSq += float64(w) * float64(w)
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var maxAbs, rel float64
	worst := 0
	for i := range want.Data {
		g, w := float64(got.Data[i]), float64(want.Data[i])
		d := math.Abs(g - w)
		maxAbs = math.Max(maxAbs, d)
		if r := d / math.Max(math.Abs(w), math.Max(rms, 1e-12)); r > rel {
			rel, worst = r, i
		}
	}
	if rel > tol {
		t.Errorf("%s %s: max abs %.6g, worst rel %.3g at %d (got %g want %g), rms %.6g > %.0e",
			name, got, maxAbs, rel, worst, got.Data[worst], want.Data[worst], rms, tol)
		return
	}
	t.Logf("%-22s %-14s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
}

// TestTemplate checks the rendered prompt, the window rule and the pad id.
func TestTemplate(t *testing.T) {
	m := loadManifest(t)
	if m.PadMultiple != PadMultiple || m.PadTokenID != PadID {
		t.Fatalf("dump pads to %d with id %d; package says %d / %d", m.PadMultiple, m.PadTokenID, PadMultiple, PadID)
	}
	for i, k := range m.Taps {
		if Taps[i] != k {
			t.Fatalf("taps %v, dump %v", Taps, m.Taps)
		}
	}
	for label, p := range m.Prompts {
		if got := Template(p.Prompt); got != p.Rendered {
			t.Errorf("%s: template %q, want %q", label, got, p.Rendered)
		}
		if got := Window(p.Real); got != p.Padded {
			t.Errorf("%s: window %d for %d tokens, want %d", label, got, p.Real, p.Padded)
		}
		for i := p.Real; i < p.Padded; i++ {
			if p.Positions[i] != PadPosition || p.IDs[i] != PadID {
				t.Fatalf("%s: row %d is id %d at position %d", label, i, p.IDs[i], p.Positions[i])
			}
		}
	}
}

// TestTokenizer checks the real ids against the dump's.
func TestTokenizer(t *testing.T) {
	m := loadManifest(t)
	e := &Encoder{}
	var err error
	if e.Tok, err = tokenizer.Load(modDir); err != nil {
		t.Skipf("no tokenizer at %s (%v)", modDir, err)
	}
	for label, p := range m.Prompts {
		ids, err := e.IDs(p.Prompt)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != p.Real {
			t.Fatalf("%s: %d ids, want %d", label, len(ids), p.Real)
		}
		for i, id := range ids {
			if id != p.IDs[i] {
				t.Fatalf("%s: id %d is %d, want %d", label, i, id, p.IDs[i])
			}
		}
	}
}

// TestLayerWalk gates the first two layers of the short prompt, pad row
// included, at the per-layer fp32 bound.
func TestLayerWalk(t *testing.T) {
	m := loadManifest(t)
	e, err := LoadLayers(modDir, 2)
	if err != nil {
		t.Skipf("no checkpoint at %s (%v)", modDir, err)
	}
	p := m.Prompts["short"]
	hs, err := e.Hidden(p.IDs[:p.Real])
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range []int{1, 2} {
		want := loadRef(t, m, "short_hidden"+string(rune('0'+i)))
		compare(t, "hidden"+string(rune('0'+i)), expand(hs[i], p.Real, p.Padded), want, relTol)
	}
}

// TestEncoder gates the whole 20480-wide embedding of every prompt.
func TestEncoder(t *testing.T) {
	if testing.Short() {
		t.Skip("loads 32 layers in fp32 (~15 GB)")
	}
	m := loadManifest(t)
	e, err := Load(modDir)
	if err != nil {
		t.Skipf("no checkpoint at %s (%v)", modDir, err)
	}
	for _, label := range []string{"short", "long", "cjk"} {
		p := m.Prompts[label]
		got, err := e.Encode(p.IDs[:p.Real])
		if err != nil {
			t.Fatal(err)
		}
		compare(t, label, got, loadRef(t, m, label+"_prompt_embeds"), deepTol)
	}
}

// expand copies a [n+1, h] hidden state (pad row last) out to the window.
func expand(h *qwen.Mat, n, window int) *qwen.Mat {
	out := qwen.NewMat(window, h.Cols)
	for r := 0; r < window; r++ {
		copy(out.Row(r), h.Row(min(r, n)))
	}
	return out
}
