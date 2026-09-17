package embed

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/zimage/qwen"
)

// The reference comes from reference/dump_qwen_embed.py, which runs
// transformers' Qwen3Model in fp32 on CPU over this checkpoint and dumps
// layer 0 stage by stage, both ends of the stack, and the model card's own
// similarity matrix. Regenerate with:
//
//	.venv/bin/python reference/dump_qwen_embed.py
const (
	refDir    = "../reference/out/embed"
	modelDir  = "../models/Qwen3-Embedding-0.6B"
	eosTokenA = "<|endoftext|>" // what the post-processor appends
)

type manifest struct {
	dir string

	Texts        []string    `json:"texts"`
	Task         string      `json:"task"`
	Queries      []string    `json:"queries"`
	Documents    []string    `json:"documents"`
	IDs          [][]int32   `json:"ids"`
	IDsNoSpecial []int32     `json:"ids_no_special"`
	EOSID        int32       `json:"eos_id"`
	AppendedID   int32       `json:"appended_id"`
	AppendsEOS   bool        `json:"appends_eos"`
	Seq          int         `json:"seq"`
	HiddenSize   int         `json:"hidden_size"`
	Layers       int         `json:"layers"`
	Heads        int         `json:"heads"`
	KVHeads      int         `json:"kv_heads"`
	HeadDim      int         `json:"head_dim"`
	Intermediate int         `json:"intermediate_size"`
	RMSEps       float64     `json:"rms_eps"`
	RopeTheta    float64     `json:"rope_theta"`
	VocabSize    int         `json:"vocab_size"`
	DumpLayer    int         `json:"dump_layer"`
	HandGap      float64     `json:"hand_gap"`
	PreNormGap   float64     `json:"prenorm_gap"`
	PadGap       float64     `json:"pad_gap"`
	CardGap      float64     `json:"card_gap"`
	CardScores   [][]float64 `json:"card_scores"`
	Scores       [][]float64 `json:"scores"`
	Tensors      map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_qwen_embed.py", refDir, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	m.dir = refDir
	return &m
}

// loadRef reads a dumped tensor, flattening every leading axis into rows.
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

// relTol is the bound every stage must stay inside. Both sides are fp32, so
// the only legitimate drift is summation order; the denominator is the
// element or the tensor's RMS, whichever is larger, for the reason
// zimage/qwen's tests give -- Qwen3 hidden states carry enormous outliers
// (`hidden_14` here has an absmax of 5910 against an RMS of 30) and
// normalising by the RMS alone makes a 1e-6 relative error on one of them
// look like a bug.
const relTol = 2e-4

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
	return
}

func compare(t *testing.T, name string, got, want *qwen.Mat) {
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

func skipWithoutCheckpoint(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors")); err != nil {
		t.Skipf("no checkpoint at %s", modelDir)
	}
}

// TestConfig checks the shape table against the model the reference ran, and
// the two measurements that say a Go implementation may skip what PyTorch
// does around the edges.
func TestConfig(t *testing.T) {
	m := loadManifest(t)
	skipWithoutCheckpoint(t)
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Fatal(err)
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
		{"intermediate_size", cfg.IntermediateSize, m.Intermediate},
		{"vocab_size", cfg.VocabSize, m.VocabSize},
		{"hidden_size vs Dim", cfg.HiddenSize, Dim},
	} {
		if c.got != c.want {
			t.Errorf("%s is %d, want %d", c.name, c.got, c.want)
		}
	}
	if cfg.Prefix != "" {
		t.Errorf("tensor prefix is %q, want empty -- this checkpoint is a bare Qwen3Model", cfg.Prefix)
	}
	// Exact zero in the dump: last_hidden_state is every layer and then the
	// final norm, which is what Model.Hidden does.
	if m.HandGap != 0 {
		t.Errorf("last_hidden_state is not %d layers + norm by hand: gap %g", cfg.NumLayers, m.HandGap)
	}
	// And the trap: transformers v5 captures hidden states after the norm, so
	// hidden_states[-1] is *not* the last layer's output. If these two ever
	// coincide, the dump's `prenorm` stopped being the pre-norm tensor.
	if m.PreNormGap == 0 {
		t.Errorf("the pre-norm and post-norm tensors are identical; the dump is not measuring what it says")
	}
	// Right padding cannot reach a real token through a causal mask, which is
	// what lets Go run one unpadded sequence where PyTorch runs a padded batch.
	if m.PadGap > 1e-4 {
		t.Errorf("right padding moved a hidden state by %g", m.PadGap)
	}
	t.Logf("%d layers of %d, %d/%d heads of %d; final norm is worth %g, padding %g",
		cfg.NumLayers, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim,
		m.PreNormGap, m.PadGap)
}

// TestTokenizer is the one that would catch the vertical's likeliest silent
// bug: the post-processor's appended token is what gets pooled, so an
// encoder that stops at the last real token produces a plausible vector that
// is wrong.
func TestTokenizer(t *testing.T) {
	m := loadManifest(t)
	skipWithoutCheckpoint(t)
	tok, err := LoadTokenizer(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	if tok.AppendToken != eosTokenA || tok.Append != m.AppendedID {
		t.Fatalf("appending %q (%d), want %q (%d)", tok.AppendToken, tok.Append, eosTokenA, m.AppendedID)
	}
	// It is not the configured eos_token, and a port that used that would be
	// wrong in a way no shape check catches.
	if tok.Append == m.EOSID {
		t.Errorf("the appended id and eos_token_id are both %d; the dump says they differ", tok.Append)
	}
	for i, text := range m.Texts {
		got, err := tok.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		want := m.IDs[i]
		if len(got) != len(want) {
			t.Fatalf("text %d: %d ids, want %d", i, len(got), len(want))
		}
		for j := range got {
			if got[j] != want[j] {
				t.Fatalf("text %d: id %d is %d, want %d", i, j, got[j], want[j])
			}
		}
		t.Logf("%3d ids  %q", len(got), truncate(text, 48))
	}
	// And the other half of the same fact: without the post-processor the
	// ids are the reference's add_special_tokens=False run.
	bare, err := tok.EncodeBare(m.Texts[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(bare) != len(m.IDsNoSpecial) {
		t.Fatalf("bare encode is %d ids, want %d", len(bare), len(m.IDsNoSpecial))
	}
	for i := range bare {
		if bare[i] != m.IDsNoSpecial[i] {
			t.Fatalf("bare id %d is %d, want %d", i, bare[i], m.IDsNoSpecial[i])
		}
	}
}

// TestInstruct pins the query template character for character: it is part
// of the input text, so a stray space changes the tokens and therefore the
// vector.
func TestInstruct(t *testing.T) {
	m := loadManifest(t)
	got := Instruct(m.Task, m.Queries[0])
	if got != m.Texts[0] {
		t.Errorf("Instruct gives %q, want %q", got, m.Texts[0])
	}
	if DefaultTask != m.Task {
		t.Errorf("DefaultTask is %q, want %q", DefaultTask, m.Task)
	}
}

// TestLayer walks decoder layer 0 stage by stage. It is the test that says
// *which* piece is wrong when an embedding is; the end-to-end checks below
// only say that something is.
func TestLayer(t *testing.T) {
	m := loadManifest(t)
	skipWithoutCheckpoint(t)
	if m.DumpLayer != 0 {
		t.Skipf("the dump is of layer %d and this test loads layer 0", m.DumpLayer)
	}
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	model, err := qwen.LoadWith(modelDir, cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	tr := qwen.Trace{}
	got, err := model.Forward(m.IDs[0], tr)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "embeddings", tr["embeddings"], loadRef(t, m, "embeddings"))
	for _, name := range []string{
		"input_layernorm", "q", "k", "v", "q_normed", "k_normed",
		"q_roped", "k_roped", "attn_ctx", "attn_out", "resid1",
		"post_attention_layernorm", "swiglu", "mlp", "layer_out",
	} {
		compare(t, name, tr["layer0."+name], loadRef(t, m, name))
	}
	compare(t, "hidden_1", got, loadRef(t, m, "hidden_1"))
}

// TestModel is the acceptance check for the CPU path: the whole stack, both
// ends of it, and the model card's similarity matrix.
func TestModel(t *testing.T) {
	if testing.Short() {
		t.Skip("loads 2.4 GB of fp32 weights and runs 28 layers four times")
	}
	m := loadManifest(t)
	skipWithoutCheckpoint(t)
	model, err := Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}

	tr := qwen.Trace{}
	h, err := model.Hidden(m.IDs[0], tr)
	if err != nil {
		t.Fatal(err)
	}
	// The intermediates first: a drift that only appears deep in the stack is
	// a different bug from one present at layer 2.
	compare(t, "hidden_1", tr["hidden_1"], loadRef(t, m, "hidden_1"))
	compare(t, "hidden_2", tr["hidden_2"], loadRef(t, m, "hidden_2"))
	compare(t, "hidden_14", tr["hidden_14"], loadRef(t, m, "hidden_14"))
	// Then the two tensors the final norm sits between, which is the whole
	// difference between this model and the text encoder in zimage/qwen.
	compare(t, "prenorm", tr["hidden_28"], loadRef(t, m, "prenorm"))
	compare(t, "final", h, loadRef(t, m, "final"))

	pooled := Pool(h)
	wantPooled := loadRef(t, m, "pooled")
	compare(t, "pooled", &qwen.Mat{Rows: 1, Cols: len(pooled), Data: pooled},
		&qwen.Mat{Rows: 1, Cols: wantPooled.Cols, Data: wantPooled.Data})

	unit := Normalize(append([]float32(nil), pooled...))
	wantUnit := loadRef(t, m, "embedding")
	compare(t, "embedding", &qwen.Mat{Rows: 1, Cols: len(unit), Data: unit},
		&qwen.Mat{Rows: 1, Cols: wantUnit.Cols, Data: wantUnit.Data})

	// And the end-to-end number nobody here produced: the card's own
	// query/document similarity matrix, from text.
	vecs := make([][]float32, len(m.Texts))
	for i, text := range m.Texts {
		if vecs[i], err = model.Embed(text); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			got := float64(Cosine(vecs[i], vecs[2+j]))
			want := m.CardScores[i][j]
			if math.Abs(got-want) > 1e-3 {
				t.Errorf("score[%d][%d] is %.6f, want %.6f (the model card's)", i, j, got, want)
			}
			t.Logf("score[%d][%d] %.6f  card %.6f", i, j, got, want)
		}
	}
	// The matrix has to be right for the right reason: a query matches its
	// own document by a margin, rather than everything scoring 0.99.
	if Cosine(vecs[0], vecs[2]) <= Cosine(vecs[0], vecs[3]) {
		t.Errorf("query 0 does not prefer document 0")
	}
}

// TestEmbedIsUnit and the MRL trim, which have no reference dump because
// they are arithmetic rather than model.
func TestNormalizeAndTruncate(t *testing.T) {
	v := make([]float32, Dim)
	for i := range v {
		v[i] = float32(i%7) - 3
	}
	u := Normalize(append([]float32(nil), v...))
	// 1e-5 rather than an exact 1: Cosine accumulates 1024 terms in fp32, so
	// a genuinely unit vector reads back as 0.9999986.
	if got := Cosine(u, u); math.Abs(float64(got)-1) > 1e-5 {
		t.Errorf("normalised vector has norm^2 %g", got)
	}
	short := Truncate(append([]float32(nil), u...), 32)
	if len(short) != 32 {
		t.Fatalf("Truncate gave %d components, want 32", len(short))
	}
	if got := Cosine(short, short); math.Abs(float64(got)-1) > 1e-5 {
		t.Errorf("truncated vector has norm^2 %g, want 1 -- MRL renormalises", got)
	}
	if same := Truncate(append([]float32(nil), u...), Dim); len(same) != Dim {
		t.Errorf("Truncate to the full width changed the length to %d", len(same))
	}
	var zero [4]float32
	if got := Normalize(zero[:]); got[0] != 0 {
		t.Errorf("normalising a zero vector gave %v", got)
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
