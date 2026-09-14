package qwen

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/zimage/tokenizer"
)

// The reference comes from reference/dump_qwen.py, which runs transformers'
// Qwen3Model in fp32 on CPU over the checkpoint's own text_encoder/ and
// dumps layer 0 stage by stage plus the hidden states the pipeline uses.
// Regenerate with:
//
//	.venv/bin/python reference/dump_qwen.py
const (
	refDir  = "../../reference/out/qwen"
	encoder = "../../models/Z-Image-Turbo/text_encoder"
	tokDir  = "../../models/Z-Image-Turbo/tokenizer"
)

type manifest struct {
	dir string

	Prompt     string  `json:"prompt"`
	Rendered   string  `json:"rendered"`
	IDs        []int32 `json:"ids"`
	Seq        int     `json:"seq"`
	Prompt2    string  `json:"prompt2"`
	IDs2       []int32 `json:"ids2"`
	HiddenSize int     `json:"hidden_size"`
	Layers     int     `json:"layers"`
	Heads      int     `json:"heads"`
	KVHeads    int     `json:"kv_heads"`
	HeadDim    int     `json:"head_dim"`
	RMSEps     float64 `json:"rms_eps"`
	RopeTheta  float64 `json:"rope_theta"`
	DumpLayer  int     `json:"dump_layer"`
	Minus2Gap  float64 `json:"minus2_gap"`
	PadGap     float64 `json:"pad_gap"`
	Tensors    map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_qwen.py", refDir, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	m.dir = refDir
	return &m
}

// loadRef reads a dumped tensor, flattening every leading axis into rows.
func loadRef(t *testing.T, m *manifest, name string) *Mat {
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
	return &Mat{Rows: meta.Count / cols, Cols: cols, Data: data}
}

// relTol is the bound every stage must stay inside, set the way the VAE's
// and the DiT's were: from the measured legitimate drift, with
// TestValidationDetectsErrors pinning the other side. Both sides are fp32
// here, so the only drift is summation order.
const relTol = 2e-4

// deviation reports the largest absolute error and that error normalised by
// max(|want|, rms) at the element it lands on.
//
// The DiT's tests normalise by the tensor's RMS alone. That does not survive
// this model: Qwen3's hidden states carry massive activations -- `hidden_2`
// has an absmax of 59.8 against an RMS of 0.55, and `final` 13753 against
// 59.5 -- so a float32 summation-order difference on one of those outliers
// is 9e-6 of the element it sits on and 7e-4 of the tensor's RMS, and only
// the second number looks like a bug. The floor keeps the RMS as the
// denominator for the elements near zero, where dividing by the element
// itself would be meaningless.
func deviation(got, want *Mat) (maxAbs, rms, rel float64, worst int) {
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

// compare asserts a computed tensor matches the reference.
func compare(t *testing.T, name string, got, want *Mat) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	maxAbs, rms, rel, worst := deviation(got, want)
	if rel > relTol {
		t.Errorf("%s %s: max abs %.6g, worst rel %.3g at %d (got %g want %g), rms %.6g > %.0e",
			name, got, maxAbs, rel, worst, got.Data[worst], want.Data[worst], rms, relTol)
		return
	}
	t.Logf("%-26s %-14s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
}

func loadModel(t *testing.T, layers int) (*Model, *manifest) {
	t.Helper()
	m := loadManifest(t)
	if _, err := os.Stat(encoder); err != nil {
		t.Skipf("no text encoder checkpoint at %s", encoder)
	}
	model, err := Load(encoder, layers)
	if err != nil {
		t.Fatal(err)
	}
	return model, m
}

// TestConfig checks the shape table this package was written against, and
// the two facts the dump settles about what the pipeline actually asks for.
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
	if cfg.EncoderLayers() != cfg.NumLayers-1 {
		t.Errorf("EncoderLayers is %d, want %d", cfg.EncoderLayers(), cfg.NumLayers-1)
	}
	// Both are exact zeros in the dump; anything else means the assumptions
	// this package is built on stopped holding.
	if m.Minus2Gap != 0 {
		t.Errorf("hidden_states[-2] is not %d layers by hand: gap %g", cfg.NumLayers-1, m.Minus2Gap)
	}
	if m.PadGap != 0 {
		t.Errorf("padding to 512 changed the hidden states by %g; Forward skips the padding", m.PadGap)
	}
	t.Logf("%d layers, %d of which the pipeline runs; padding to 512 costs %g",
		cfg.NumLayers, cfg.EncoderLayers(), m.PadGap)
}

// TestTokenizer checks that the Go tokenizer produces the ids the reference
// ran, which is what makes the two halves of this stage one pipeline.
func TestTokenizer(t *testing.T) {
	m := loadManifest(t)
	tok, err := tokenizer.Load(tokDir)
	if err != nil {
		t.Skipf("no tokenizer at %s (%v)", tokDir, err)
	}
	for _, c := range []struct {
		prompt string
		want   []int32
	}{{m.Prompt, m.IDs}, {m.Prompt2, m.IDs2}} {
		got, err := tok.EncodePrompt(c.prompt)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(c.want) {
			t.Fatalf("%q: %d ids, want %d", c.prompt, len(got), len(c.want))
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("%q: id %d is %d, want %d", c.prompt, i, got[i], c.want[i])
			}
		}
		t.Logf("%d ids for %q", len(got), c.prompt)
	}
}

// TestRoPETable checks the rotary table before anything uses it. The dump
// holds transformers' cat(freqs, freqs) over the full head width; this keeps
// only the distinct half, so the check is that the half repeats.
func TestRoPETable(t *testing.T) {
	m := loadManifest(t)
	cos, sin := loadRef(t, m, "cos"), loadRef(t, m, "sin")
	rope := NewRoPE(m.HeadDim, m.Seq, m.RopeTheta)
	half := m.HeadDim / 2
	gotCos, gotSin := NewMat(m.Seq, m.HeadDim), NewMat(m.Seq, m.HeadDim)
	for p := 0; p < m.Seq; p++ {
		for i := 0; i < half; i++ {
			gotCos.Row(p)[i] = rope.Cos[p*half+i]
			gotCos.Row(p)[i+half] = rope.Cos[p*half+i]
			gotSin.Row(p)[i] = rope.Sin[p*half+i]
			gotSin.Row(p)[i+half] = rope.Sin[p*half+i]
		}
	}
	compare(t, "cos", gotCos, cos)
	compare(t, "sin", gotSin, sin)
}

// TestLayer walks one decoder layer stage by stage. It is the test that says
// which piece is wrong when the encoder's output is; the end-to-end check
// below only says that something is.
func TestLayer(t *testing.T) {
	model, m := loadModel(t, m0Layers)
	if m.DumpLayer != 0 {
		t.Skipf("the dump is of layer %d and this test loads layer 0", m.DumpLayer)
	}
	tr := Trace{}
	got, err := model.Forward(m.IDs, tr)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "embeddings", tr["embeddings"], loadRef(t, m, "embeddings"))
	for _, name := range []string{
		"input_layernorm", "q", "k", "v", "q_normed", "k_normed",
		"q_roped", "k_roped", "attn_ctx", "attn_out", "resid1",
		"post_attention_layernorm", "mlp", "layer_out",
	} {
		compare(t, name, tr["layer0."+name], loadRef(t, m, name))
	}
	compare(t, "hidden_1", got, loadRef(t, m, "hidden_1"))
}

// m0Layers is one layer: everything TestLayer needs, and 448 MB rather than
// 15.7 GB.
const m0Layers = 1

// TestEncoder runs what the pipeline runs -- 35 of the 36 layers, no final
// norm -- against the hidden state it hands the DiT, for two prompts. This
// is the stage's acceptance check.
func TestEncoder(t *testing.T) {
	if testing.Short() {
		t.Skip("loads 15.7 GB of fp32 weights")
	}
	m := loadManifest(t)
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	model, err := Load(encoder, cfg.EncoderLayers())
	if err != nil {
		t.Fatal(err)
	}
	tr := Trace{}
	got, err := model.Forward(m.IDs, tr)
	if err != nil {
		t.Fatal(err)
	}
	// The intermediate hidden states first: a drift that only shows up deep
	// in the stack is a different bug from one present at layer 2.
	compare(t, "hidden_1", tr["hidden_1"], loadRef(t, m, "hidden_1"))
	compare(t, "hidden_2", tr["hidden_2"], loadRef(t, m, "hidden_2"))
	compare(t, "hidden_18", tr["hidden_18"], loadRef(t, m, "hidden_18"))
	compare(t, "final", got, loadRef(t, m, "final"))
	// And the pipeline's own output, which is the padded run masked back
	// down -- the same tensor, by the dump's measurement.
	compare(t, "pipeline_out", got, loadRef(t, m, "pipeline_out"))

	got2, err := model.Forward(m.IDs2, nil)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "final2", got2, loadRef(t, m, "final2"))
}

// TestValidationDetectsErrors is the negative control. The four
// perturbations are the mistakes this model actually invites: the two rotary
// conventions look equally plausible and the DiT next door uses the other
// one, the q/k norms are per head rather than over the full width, grouped
// query heads can be mapped to their kv head two ways, and a causal mask is
// invisible in every shape.
func TestValidationDetectsErrors(t *testing.T) {
	model, m := loadModel(t, m0Layers)
	l := model.Layers[0]
	heads, kvHeads, hd := l.Heads, l.KVHeads, l.HeadDim
	normed := loadRef(t, m, "input_layernorm")

	t.Run("rope paired adjacent not by halves", func(t *testing.T) {
		q := loadRef(t, m, "q_normed")
		rope := NewRoPE(hd, m.Seq, m.RopeTheta)
		applyRoPEAdjacent(rope, q, heads, hd)
		assertCaught(t, q, loadRef(t, m, "q_roped"))
	})

	t.Run("qk norm over full width not per head", func(t *testing.T) {
		q, err := l.Q.Apply(normed)
		if err != nil {
			t.Fatal(err)
		}
		// The same weights tiled to the full width: one RMS over all 4096
		// components instead of thirty-two over 128 each.
		wide := make([]float32, q.Cols)
		for i := range wide {
			wide[i] = l.QNorm.Weight[i%hd]
		}
		bad := &RMSNorm{Weight: wide, Eps: l.QNorm.Eps}
		if err := bad.ApplyInPlace(q); err != nil {
			t.Fatal(err)
		}
		assertCaught(t, q, loadRef(t, m, "q_normed"))
	})

	t.Run("gqa heads tiled not repeated", func(t *testing.T) {
		ctx := attentionVariant(t, m, heads, kvHeads, hd, func(h int) int { return h % kvHeads }, true)
		assertCaught(t, ctx, loadRef(t, m, "attn_ctx"))
	})

	t.Run("attention not causal", func(t *testing.T) {
		ctx := attentionVariant(t, m, heads, kvHeads, hd, func(h int) int { return h / (heads / kvHeads) }, false)
		assertCaught(t, ctx, loadRef(t, m, "attn_ctx"))
	})
}

// assertCaught fails if a deliberately broken tensor slips under relTol.
func assertCaught(t *testing.T, got, want *Mat) {
	t.Helper()
	maxAbs, rms, rel, _ := deviation(got, want)
	if rel <= relTol {
		t.Errorf("perturbation NOT caught: max abs %.6g, rms %.6g, rel %.3g <= %.0e", maxAbs, rms, rel, relTol)
		return
	}
	t.Logf("caught: rel %.3g (%.0fx the %.0e bound)", rel, rel/relTol, relTol)
}

// applyRoPEAdjacent is the *other* rotary convention -- pairing component 2j
// with 2j+1, which is what the DiT uses. Both are in the wild and they
// differ only in index arithmetic, so this exists to prove the test can tell
// them apart.
func applyRoPEAdjacent(r *RoPE, x *Mat, heads, headDim int) {
	half := headDim / 2
	for p := 0; p < x.Rows; p++ {
		row := x.Row(p)
		cos, sin := r.Cos[p*half:(p+1)*half], r.Sin[p*half:(p+1)*half]
		for h := 0; h < heads; h++ {
			seg := row[h*headDim : (h+1)*headDim]
			for j := 0; j < half; j++ {
				re, im := seg[2*j], seg[2*j+1]
				seg[2*j] = re*cos[j] - im*sin[j]
				seg[2*j+1] = re*sin[j] + im*cos[j]
			}
		}
	}
}

// attentionVariant recomputes the dumped attention with one rule changed:
// which kv head serves a query head, and whether the mask is causal.
func attentionVariant(t *testing.T, m *manifest, heads, kvHeads, hd int, kvOf func(int) int, causal bool) *Mat {
	t.Helper()
	q, k, v := loadRef(t, m, "q_roped"), loadRef(t, m, "k_roped"), loadRef(t, m, "v")
	T := q.Rows
	scale := float32(1 / math.Sqrt(float64(hd)))
	out := NewMat(T, heads*hd)
	for h := 0; h < heads; h++ {
		kv := kvOf(h)
		for i := 0; i < T; i++ {
			last := T - 1
			if causal {
				last = i
			}
			scores := make([]float32, last+1)
			max := float32(math.Inf(-1))
			for s := 0; s <= last; s++ {
				var dot float32
				qv := q.Row(i)[h*hd : (h+1)*hd]
				kr := k.Row(s)[kv*hd : (kv+1)*hd]
				for c, qc := range qv {
					dot += qc * kr[c]
				}
				scores[s] = dot * scale
				if scores[s] > max {
					max = scores[s]
				}
			}
			var sum float32
			for s := range scores {
				scores[s] = float32(math.Exp(float64(scores[s] - max)))
				sum += scores[s]
			}
			dst := out.Row(i)[h*hd : (h+1)*hd]
			for s := range scores {
				p := scores[s] / sum
				vr := v.Row(s)[kv*hd : (kv+1)*hd]
				for c, vv := range vr {
					dst[c] += p * vv
				}
			}
		}
	}
	return out
}
