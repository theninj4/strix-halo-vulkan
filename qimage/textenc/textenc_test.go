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

// The reference comes from reference/dump_qi21_textenc.py, which runs the
// diffusers pipeline's own _get_qwen_prompt_embeds (final-norm hook and all)
// over the checkpoint's text_encoder/ in fp32 on CPU, plus a layer walk and
// the rope tables. Regenerate with:
//
//	.venv/bin/python reference/dump_qi21_textenc.py
const (
	refDir  = "../../reference/out/qi21textenc"
	encoder = "../../models/Qwen-Image-2.1/text_encoder"
	tokDir  = "../../models/Qwen-Image-2.1/processor"
)

type promptCase struct {
	Prompt      string  `json:"prompt"`
	Rendered    string  `json:"rendered"`
	IDs         []int32 `json:"ids"`
	Seq         int     `json:"seq"`
	EmbedTokens int     `json:"embed_tokens"`
	PrenormGap  float64 `json:"prenorm_vs_pipeline_gap"`
}

type manifest struct {
	dir string

	TemplateT2I string                `json:"template_t2i"`
	SysPrompt   string                `json:"sys_prompt"`
	DropIdx     int                   `json:"drop_idx"`
	HiddenSize  int                   `json:"hidden_size"`
	Layers      int                   `json:"layers"`
	Heads       int                   `json:"heads"`
	KVHeads     int                   `json:"kv_heads"`
	HeadDim     int                   `json:"head_dim"`
	RMSEps      float64               `json:"rms_eps"`
	RopeTheta   float64               `json:"rope_theta"`
	MRopeGap    float64               `json:"mrope_gap"`
	Prompts     map[string]promptCase `json:"prompts"`
	Tensors     map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_qi21_textenc.py", refDir, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	m.dir = refDir
	return &m
}

func loadRef(t *testing.T, m *manifest, name string) *qwen.Mat {
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
	cols := meta.Shape[len(meta.Shape)-1]
	return &qwen.Mat{Rows: meta.Count / cols, Cols: cols, Data: data}
}

// relTol is zimage/qwen's fp32 bound: both sides are fp32, so the only drift
// is summation order. The same massive-activation floor applies — this model
// carries even larger outliers (en_last_prenorm absmax 4596).
const relTol = 2e-4

// deepTol is the bound for the full 36-layer forward, and it is measured,
// not chosen: the reference disagrees with *itself* by rel 3.2e-4 (max abs
// 0.011) on the empty prompt when only the attention kernel's summation
// order changes (eager vs sdpa, both torch fp32 — measured 2026-09-20).
// 2e-4 is therefore below this model's own fp32 noise floor at depth 36;
// the Go port lands at 7.1e-4 on the same prompt, the same phenomenon with
// a summation order further from eager's. The shallow tests above hold the
// per-layer bound, which is what keeps a real bug from hiding under this.
const deepTol = 1e-3

func deviation(got, want *qwen.Mat) (maxAbs, rms, rel float64, worst int) {
	var sumSq float64
	for i := range want.Data {
		w := float64(want.Data[i])
		sumSq += w * w
	}
	rms = math.Sqrt(sumSq / float64(len(want.Data)))
	for i := range want.Data {
		g, w := float64(got.Data[i]), float64(want.Data[i])
		d := math.Abs(g - w)
		if d > maxAbs {
			maxAbs = d
		}
		if r := d / math.Max(math.Abs(w), math.Max(rms, 1e-12)); r > rel {
			rel, worst = r, i
		}
	}
	return maxAbs, rms, rel, worst
}

func compare(t *testing.T, name string, got, want *qwen.Mat) {
	t.Helper()
	compareAt(t, name, got, want, relTol)
}

func compareAt(t *testing.T, name string, got, want *qwen.Mat, tol float64) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	maxAbs, rms, rel, worst := deviation(got, want)
	if rel > tol {
		t.Errorf("%s %s: max abs %.6g, worst rel %.3g at %d (got %g want %g), rms %.6g > %.0e",
			name, got, maxAbs, rel, worst, got.Data[worst], want.Data[worst], rms, tol)
		return
	}
	t.Logf("%-22s %-14s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
}

func loadTok(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok, err := tokenizer.Load(tokDir)
	if err != nil {
		t.Skipf("no tokenizer at %s (%v)", tokDir, err)
	}
	return tok
}

// TestConfig checks the nested-config adapter against the dump, and the two
// facts of Q0's that this package is built on: text-only mrope is plain RoPE
// (gap 0), and the hooked pre-norm hidden state is exactly what the pipeline
// hands the DiT (gap 0).
func TestConfig(t *testing.T) {
	m := loadManifest(t)
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	for _, c := range []struct {
		name      string
		got, want int
	}{
		{"hidden_size", cfg.HiddenSize, m.HiddenSize},
		{"layers", cfg.NumLayers, m.Layers},
		{"heads", cfg.NumHeads, m.Heads},
		{"kv_heads", cfg.NumKVHeads, m.KVHeads},
		{"head_dim", cfg.HeadDim, m.HeadDim},
	} {
		if c.got != c.want {
			t.Errorf("%s is %d, want %d", c.name, c.got, c.want)
		}
	}
	if cfg.RopeTheta != m.RopeTheta {
		t.Errorf("rope_theta is %g, want %g", cfg.RopeTheta, m.RopeTheta)
	}
	if cfg.Prefix != "model.language_model." {
		t.Errorf("prefix is %q", cfg.Prefix)
	}
	if m.MRopeGap != 0 {
		t.Errorf("text-only mrope differs from plain RoPE by %g; this package assumes 0", m.MRopeGap)
	}
	if g := m.Prompts["en"].PrenormGap; g != 0 {
		t.Errorf("hooked pre-norm state differs from the pipeline's output by %g", g)
	}
	if m.SysPrompt != SysPrompt {
		t.Errorf("system prompt is %q, want %q", SysPrompt, m.SysPrompt)
	}
	t.Logf("%d layers of %d hidden, theta %g, drop %d", cfg.NumLayers, cfg.HiddenSize, cfg.RopeTheta, m.DropIdx)
}

// TestTemplate checks the raw template and the tokenized-system-message drop
// count against what the diffusers pipeline computed for itself.
func TestTemplate(t *testing.T) {
	m := loadManifest(t)
	for label, p := range m.Prompts {
		if got := TemplateT2I(p.Prompt); got != p.Rendered {
			t.Errorf("%s: template renders %q, want %q", label, got, p.Rendered)
		}
	}
	tok := loadTok(t)
	drop, err := DropTokens(tok)
	if err != nil {
		t.Fatal(err)
	}
	if drop != m.DropIdx {
		t.Errorf("DropTokens is %d, want %d", drop, m.DropIdx)
	}
	for label, p := range m.Prompts {
		if p.Seq-p.EmbedTokens != m.DropIdx {
			t.Errorf("%s: dump drops %d tokens, manifest says %d", label, p.Seq-p.EmbedTokens, m.DropIdx)
		}
	}
	t.Logf("drop %d tokens", drop)
}

// TestTokenizer checks the Go tokenizer against the processor's ids for all
// three prompts, template included.
func TestTokenizer(t *testing.T) {
	m := loadManifest(t)
	tok := loadTok(t)
	for label, p := range m.Prompts {
		got, err := EncodePrompt(tok, p.Prompt)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(p.IDs) {
			t.Fatalf("%s: %d ids, want %d", label, len(got), len(p.IDs))
		}
		for i := range got {
			if got[i] != p.IDs[i] {
				t.Fatalf("%s: id %d is %d, want %d", label, i, got[i], p.IDs[i])
			}
		}
		t.Logf("%s: %d ids", label, len(got))
	}
}

// TestRoPETable checks qwen's rotary table at theta 5e6 against the cos/sin
// the model itself used (captured by hook in the dump). The dump holds
// cat(freqs, freqs) over the full width; qwen keeps the distinct half.
func TestRoPETable(t *testing.T) {
	m := loadManifest(t)
	p := m.Prompts["en"]
	cos, sin := loadRef(t, m, "en_rope_cos"), loadRef(t, m, "en_rope_sin")
	rope := qwen.NewRoPE(m.HeadDim, p.Seq, m.RopeTheta)
	half := m.HeadDim / 2
	gotCos, gotSin := qwen.NewMat(p.Seq, m.HeadDim), qwen.NewMat(p.Seq, m.HeadDim)
	for pos := 0; pos < p.Seq; pos++ {
		for i := 0; i < half; i++ {
			gotCos.Row(pos)[i] = rope.Cos[pos*half+i]
			gotCos.Row(pos)[i+half] = rope.Cos[pos*half+i]
			gotSin.Row(pos)[i] = rope.Sin[pos*half+i]
			gotSin.Row(pos)[i+half] = rope.Sin[pos*half+i]
		}
	}
	compare(t, "en_rope_cos", gotCos, cos)
	compare(t, "en_rope_sin", gotSin, sin)
}

// TestLayerWalk runs the first two decoder layers over the English prompt
// and compares each hidden state — the test that says *which* layer is wrong
// when TestEncoder's output is. Two layers and the embedding table are
// ~4.4 GB fp32.
func TestLayerWalk(t *testing.T) {
	m := loadManifest(t)
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	model, err := qwen.LoadWith(encoder, cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	tr := qwen.Trace{}
	if _, err := model.Forward(m.Prompts["en"].IDs, tr); err != nil {
		t.Fatal(err)
	}
	compare(t, "en_embed_tokens", tr["embeddings"], loadRef(t, m, "en_embed_tokens"))
	compare(t, "en_layer0_out", tr["hidden_1"], loadRef(t, m, "en_layer0_out"))
	compare(t, "en_layer1_out", tr["hidden_2"], loadRef(t, m, "en_layer1_out"))
}

// TestEncoder is Q1's CPU acceptance gate: all 36 layers, no final norm,
// three prompts, against the embeddings the diffusers pipeline hands the
// DiT. Loads ~34 GB of fp32 weights.
func TestEncoder(t *testing.T) {
	if testing.Short() {
		t.Skip("loads 34 GB of fp32 weights")
	}
	m := loadManifest(t)
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	model, err := qwen.LoadWith(encoder, cfg, cfg.NumLayers)
	if err != nil {
		t.Fatal(err)
	}
	drop := m.DropIdx

	for _, label := range []string{"en", "cjk", "empty"} {
		p := m.Prompts[label]
		out, err := model.Forward(p.IDs, nil)
		if err != nil {
			t.Fatal(err)
		}
		if label == "en" {
			compareAt(t, "en_last_prenorm", out, loadRef(t, m, "en_last_prenorm"), deepTol)
		}
		embeds, err := Drop(out, drop)
		if err != nil {
			t.Fatal(err)
		}
		compareAt(t, label+"_prompt_embeds", embeds, loadRef(t, m, label+"_prompt_embeds"), deepTol)
	}
}
