package dit

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/safetensors"
)

// The reference comes from reference/dump_dit_block.py, which instantiates
// diffusers' ZImageTransformerBlock for layer 0 and dumps every stage plus
// the attention internals. Regenerate with:
//
//	.venv/bin/python reference/dump_dit_block.py --seq 320 --caption 64
const (
	refDir      = "../../reference/out/dit"
	transformer = "../../models/Z-Image-Turbo/transformer"
)

type manifest struct {
	Seq       int     `json:"seq"`
	Caption   int     `json:"caption"`
	Dim       int     `json:"dim"`
	Heads     int     `json:"heads"`
	HeadDim   int     `json:"head_dim"`
	AxesDims  [3]int  `json:"axes_dims"`
	AxesLens  [3]int  `json:"axes_lens"`
	RopeTheta float64 `json:"rope_theta"`
	NormEps   float64 `json:"norm_eps"`
	Tensors   map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump (%v); run reference/dump_dit_block.py", err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// loadRef reads a dumped tensor, flattening every leading axis into rows.
func loadRef(t *testing.T, m *manifest, name string) *Mat {
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
	return &Mat{Rows: meta.Count / cols, Cols: cols, Data: data}
}

// relTol is the bound every stage must stay inside, set the same way the
// VAE's was: from the measured legitimate drift, with the negative control
// in TestValidationDetectsErrors pinning the other side.
const relTol = 2e-4

func deviation(got, want *Mat) (maxAbs, rms, rel float64, worst int) {
	var sumSq float64
	for i := range want.Data {
		g, w := float64(got.Data[i]), float64(want.Data[i])
		sumSq += w * w
		if d := math.Abs(g - w); d > maxAbs {
			maxAbs, worst = d, i
		}
	}
	rms = math.Sqrt(sumSq / float64(len(want.Data)))
	return maxAbs, rms, maxAbs / math.Max(rms, 1e-12), worst
}

// compare asserts a computed tensor matches the reference. As in the VAE,
// the error is normalised by the tensor's RMS rather than per element: these
// activations cross zero constantly, so a per-element relative error is
// dominated by the values nearest zero.
func compare(t *testing.T, name string, got, want *Mat) {
	t.Helper()
	compareTol(t, name, got, want, relTol)
}

// compareTol is compare against an explicit bound, for the paths that do not
// carry fp32's drift: the matrix-core kernels narrow their operands to fp16
// and are held to their own measured figure (fp16RelTol in gpu_test.go).
func compareTol(t *testing.T, name string, got, want *Mat, tol float64) float64 {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	maxAbs, rms, rel, worst := deviation(got, want)
	if rel > tol {
		t.Errorf("%s %s: max abs %.6g (at %d: got %g want %g), rms %.6g, rel %.3g > %.0e",
			name, got, maxAbs, worst, got.Data[worst], want.Data[worst], rms, rel, tol)
		return rel
	}
	t.Logf("%-18s %-14s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
	return rel
}

// ropeFromRef rebuilds the rotary table from the dumped ids, so the table is
// checked against the reference's own freqs rather than assumed.
func ropeFromRef(t *testing.T, m *manifest) *RoPE {
	t.Helper()
	idsMat := loadRef(t, m, "ids")
	ids := make([][3]int32, idsMat.Rows)
	for i := range ids {
		r := idsMat.Row(i)
		ids[i] = [3]int32{int32(r[0]), int32(r[1]), int32(r[2])}
	}
	rope, err := NewRoPE(ids, m.AxesDims, m.AxesLens, m.RopeTheta)
	if err != nil {
		t.Fatal(err)
	}
	return rope
}

// TestRoPETable checks the rotary table itself before anything uses it. It
// is the one component with no module boundary in the reference to hook, and
// a wrong frequency schedule produces a block output that is wrong
// everywhere with no clue as to why.
func TestRoPETable(t *testing.T) {
	m := loadManifest(t)
	rope := ropeFromRef(t, m)
	cos := loadRef(t, m, "freqs_cos")
	sin := loadRef(t, m, "freqs_sin")
	compare(t, "rope cos", &Mat{Rows: rope.Tokens, Cols: rope.Pairs, Data: rope.Cos}, cos)
	compare(t, "rope sin", &Mat{Rows: rope.Tokens, Cols: rope.Pairs, Data: rope.Sin}, sin)
}

// TestBlockAgainstDiffusers walks the block stage by stage.
func TestBlockAgainstDiffusers(t *testing.T) {
	if _, err := os.Stat(transformer); err != nil {
		t.Skipf("no transformer checkpoint at %s", transformer)
	}
	m := loadManifest(t)
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenSet(transformer)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()

	blk, err := LoadBlock(set, "layers.0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	rope := ropeFromRef(t, m)

	x := loadRef(t, m, "x")
	adaln := loadRef(t, m, "adaln_input")

	// Modulation.
	modMat, err := blk.AdaLN.Apply(&Mat{Rows: 1, Cols: adaln.Cols, Data: adaln.Data})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "mod", modMat, loadRef(t, m, "mod"))

	mod := modMat.Data
	d := cfg.Dim
	scaleMSA := make([]float32, d)
	gateMSA := make([]float32, d)
	for i := 0; i < d; i++ {
		scaleMSA[i] = 1 + mod[i]
		gateMSA[i] = float32(math.Tanh(float64(mod[d+i])))
	}
	compare(t, "scale_msa", &Mat{Rows: 1, Cols: d, Data: scaleMSA}, loadRef(t, m, "scale_msa"))
	compare(t, "gate_msa", &Mat{Rows: 1, Cols: d, Data: gateMSA}, loadRef(t, m, "gate_msa"))

	// Attention input.
	h := x.Clone()
	if _, err := blk.AttnNorm1.ApplyInPlace(h); err != nil {
		t.Fatal(err)
	}
	compare(t, "attention_norm1", h, loadRef(t, m, "attention_norm1"))
	scaleRows(h, scaleMSA)
	compare(t, "attn_in", h, loadRef(t, m, "attn_in"))

	// Attention internals.
	q, err := blk.Attn.Q.Apply(h)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "q", q, loadRef(t, m, "q"))
	if _, err := blk.Attn.NormQ.ApplyInPlace(q); err != nil {
		t.Fatal(err)
	}
	compare(t, "q_normed", q, loadRef(t, m, "q_normed"))
	if err := rope.ApplyInPlace(q, cfg.NHeads, cfg.Dim/cfg.NHeads); err != nil {
		t.Fatal(err)
	}
	compare(t, "q_roped", q, loadRef(t, m, "q_roped"))

	k, err := blk.Attn.K.Apply(h)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := blk.Attn.NormK.ApplyInPlace(k); err != nil {
		t.Fatal(err)
	}
	if err := rope.ApplyInPlace(k, cfg.NHeads, cfg.Dim/cfg.NHeads); err != nil {
		t.Fatal(err)
	}
	compare(t, "k_roped", k, loadRef(t, m, "k_roped"))

	v, err := blk.Attn.V.Apply(h)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "v", v, loadRef(t, m, "v"))

	ctx, err := blk.Attn.scores(q, k, v)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "attn_ctx", ctx, loadRef(t, m, "attn_ctx"))

	attnOut, err := blk.Attn.Out.Apply(ctx)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "attn_out", attnOut, loadRef(t, m, "attn_out"))

	// Whole block.
	out, err := blk.Apply(x, adaln.Data, rope)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "out", out, loadRef(t, m, "out"))
}

// TestValidationDetectsErrors is the negative control. The four
// perturbations are the mistakes this block actually invites: RoPE's two
// pairing conventions look equally plausible, the q/k norms are per head
// rather than over the full width, adaLN's four chunks have no labels in the
// checkpoint, and a gated residual is easy to drop.
func TestValidationDetectsErrors(t *testing.T) {
	if _, err := os.Stat(transformer); err != nil {
		t.Skipf("no transformer checkpoint at %s", transformer)
	}
	m := loadManifest(t)
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenSet(transformer)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	blk, err := LoadBlock(set, "layers.0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	rope := ropeFromRef(t, m)
	x := loadRef(t, m, "x")
	adaln := loadRef(t, m, "adaln_input")
	headDim := cfg.Dim / cfg.NHeads

	// The attention input, shared by the q-side perturbations.
	modMat, err := blk.AdaLN.Apply(&Mat{Rows: 1, Cols: adaln.Cols, Data: adaln.Data})
	if err != nil {
		t.Fatal(err)
	}
	scaleMSA := make([]float32, cfg.Dim)
	for i := range scaleMSA {
		scaleMSA[i] = 1 + modMat.Data[i]
	}
	h := x.Clone()
	if _, err := blk.AttnNorm1.ApplyInPlace(h); err != nil {
		t.Fatal(err)
	}
	scaleRows(h, scaleMSA)

	t.Run("rope paired by halves not adjacent", func(t *testing.T) {
		q, err := blk.Attn.Q.Apply(h)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := blk.Attn.NormQ.ApplyInPlace(q); err != nil {
			t.Fatal(err)
		}
		applyRoPEHalfSplit(rope, q, cfg.NHeads, headDim)
		assertCaught(t, q, loadRef(t, m, "q_roped"))
	})

	t.Run("qk norm over full width not per head", func(t *testing.T) {
		q, err := blk.Attn.Q.Apply(h)
		if err != nil {
			t.Fatal(err)
		}
		// The same weights tiled to the full width, i.e. one RMS over all
		// 3840 components instead of thirty over 128 each.
		wide := make([]float32, cfg.Dim)
		for i := range wide {
			wide[i] = blk.Attn.NormQ.Weight[i%headDim]
		}
		bad := &RMSNorm{Weight: wide, Eps: blk.Attn.NormQ.Eps}
		if _, err := bad.ApplyInPlace(q); err != nil {
			t.Fatal(err)
		}
		assertCaught(t, q, loadRef(t, m, "q_normed"))
	})

	t.Run("adaLN chunks in the wrong order", func(t *testing.T) {
		// scale_mlp read where scale_msa belongs, which is what happens if
		// the four chunks are assumed to be interleaved per feature rather
		// than concatenated.
		bad := make([]float32, cfg.Dim)
		for i := range bad {
			bad[i] = 1 + modMat.Data[2*cfg.Dim+i]
		}
		h2 := x.Clone()
		if _, err := blk.AttnNorm1.ApplyInPlace(h2); err != nil {
			t.Fatal(err)
		}
		scaleRows(h2, bad)
		assertCaught(t, h2, loadRef(t, m, "attn_in"))
	})

	t.Run("dropped gated residual", func(t *testing.T) {
		out, err := blk.Apply(x, adaln.Data, rope)
		if err != nil {
			t.Fatal(err)
		}
		// The block without its first residual: subtract x back out.
		bad := out.Clone()
		for i := range bad.Data {
			bad.Data[i] -= x.Data[i]
		}
		assertCaught(t, bad, loadRef(t, m, "out"))
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

// applyRoPEHalfSplit is the *other* rotary convention -- pairing component j
// with j+headDim/2 instead of 2j with 2j+1. Both are used in the wild and
// they differ only in index arithmetic, so this exists to prove the test can
// tell them apart.
func applyRoPEHalfSplit(r *RoPE, x *Mat, heads, headDim int) {
	half := headDim / 2
	for t := 0; t < x.Rows; t++ {
		row := x.Row(t)
		cos, sin := r.Cos[t*r.Pairs:(t+1)*r.Pairs], r.Sin[t*r.Pairs:(t+1)*r.Pairs]
		for hd := 0; hd < heads; hd++ {
			seg := row[hd*headDim : (hd+1)*headDim]
			for j := 0; j < half; j++ {
				re, im := seg[j], seg[j+half]
				c, s := cos[j], sin[j]
				seg[j] = re*c - im*s
				seg[j+half] = re*s + im*c
			}
		}
	}
}
