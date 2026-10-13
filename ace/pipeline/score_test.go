package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/plan"
	h3vae "strix-halo-vulkan/h3/vae"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// The reference: reference/dump_ace_score.py (MUSIC.md A14b), upstream's
// get_lyric_score on A1's requests, generated and scored in fp32 on the
// CPU; --bf16 the same inputs scored in bf16.
const (
	scoreRef     = "../../reference/out/acescore"
	scoreBF16Ref = "../../reference/out/acescore_bf16"
)

var alignCases = []string{"full_metas", "zh_odd", "instrumental_60"}

type alignRow struct {
	Coverage     float64 `json:"coverage"`
	Monotonicity float64 `json:"monotonicity"`
	Confidence   float64 `json:"confidence"`
	Score        float64 `json:"score"`
}

type alignCase struct {
	LatentLength int                 `json:"latent_length"`
	EncoderLen   int                 `json:"encoder_len"`
	Heads        [][2]int            `json:"heads"`
	LyricStart   int                 `json:"lyric_start"`
	LyricEnd     int                 `json:"lyric_end"`
	TypeMask     []int               `json:"type_mask"`
	Rows         map[string]alignRow `json:"rows"`
	LMScore      float64             `json:"lm_score"`
	DiTScore     float64             `json:"dit_score"`
}

func loadAlign(t *testing.T, dir, label string) *alignCase {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Skipf("no score reference (%v); run reference/dump_ace_score.py", err)
	}
	var m struct {
		Cases map[string]*alignCase `json:"cases"`
	}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	c, ok := m.Cases["align_"+label]
	if !ok {
		t.Skipf("no case align_%s in %s", label, dir)
	}
	return c
}

func readI32(t *testing.T, path string) []int32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// oracleProbs splits a dumped [2, heads, tokens, encoder rows] into each
// row's per-head matrices.
func oracleProbs(t *testing.T, dir, label string, c *alignCase) [2][][]float32 {
	all := readF32(t, filepath.Join(dir, "align_"+label+"_probs.bin"))
	n := (c.LatentLength + 1) / 2
	per := n * c.EncoderLen
	if len(all) != 2*len(c.Heads)*per {
		t.Fatalf("%s: %d probabilities for 2 x %d heads x %d x %d", label, len(all), len(c.Heads), n, c.EncoderLen)
	}
	var out [2][][]float32
	for r := 0; r < 2; r++ {
		for h := range c.Heads {
			o := (r*len(c.Heads) + h) * per
			out[r] = append(out[r], all[o:o+per])
		}
	}
	return out
}

// TestAlignHost is the host half of A14b's alignment gate, from upstream's
// own attention probabilities: the lyric ids, span and type mask; the head
// average and normalisation against upstream's energy; the DTW path
// exactly, over upstream's energy and over ours; and the metrics.
func TestAlignHost(t *testing.T) {
	tk, err := tokenizer.Load(DefaultDirs(models).Text)
	if err != nil {
		t.Skip(err)
	}
	p := &Pipeline{Tok: tk}
	for _, label := range alignCases {
		c := loadAlign(t, scoreRef, label)
		r := request(t, label)
		_, ids, err := p.Ids(r)
		if err != nil {
			t.Fatal(err)
		}
		want := readI32(t, filepath.Join(scoreRef, "align_"+label+"_lyric_ids.bin"))
		if len(ids) != len(want) {
			t.Fatalf("%s: %d lyric ids, upstream %d", label, len(ids), len(want))
		}
		for i := range want {
			if ids[i] != want[i] {
				t.Fatalf("%s: lyric id %d is %d, upstream %d", label, i, ids[i], want[i])
			}
		}
		start, end, err := p.LyricSpan(r, ids)
		if err != nil {
			t.Fatal(err)
		}
		if start != c.LyricStart || end != c.LyricEnd {
			t.Fatalf("%s: lyric span [%d, %d), upstream [%d, %d)", label, start, end, c.LyricStart, c.LyricEnd)
		}
		mask := p.TypeMask(ids[start:end])
		for i := range c.TypeMask {
			if mask[i] != c.TypeMask[i] {
				t.Fatalf("%s: type mask %v, upstream %v", label, mask, c.TypeMask)
			}
		}

		noise, err := h3vae.TorchRandn(alignSeed, c.LatentLength*plan.LatentChannels)
		if err != nil {
			t.Fatal(err)
		}
		// TestNoise's bound: TorchRandn is torch's draw to an ulp.
		if rel, _, _ := gap(noise, readF32(t, filepath.Join(scoreRef, "align_"+label+"_noise.bin"))); rel > 1e-6 {
			t.Fatalf("%s: seed-42 noise differs by %.2e", label, rel)
		}

		probs := oracleProbs(t, scoreRef, label, c)
		n, rows := (c.LatentLength+1)/2, end-start
		for ri, name := range []string{"lm", "dit"} {
			wantE := readF32(t, filepath.Join(scoreRef, "align_"+label+"_"+name+"_energy.bin"))
			wantP := readI32(t, filepath.Join(scoreRef, "align_"+label+"_"+name+"_path.bin"))
			e := Normalize(HeadEnergy(probs[ri], n, c.EncoderLen, start, end))
			var worst float64
			for i := range wantE {
				worst = math.Max(worst, math.Abs(float64(e[i]-wantE[i])))
			}
			pathEq := func(path [][2]int) bool {
				if len(path) != len(wantP)/2 {
					return false
				}
				for i, rc := range path {
					if rc[0] != int(wantP[i*2]) || rc[1] != int(wantP[i*2+1]) {
						return false
					}
				}
				return true
			}
			theirs, ours := DTW(wantE, rows, n), DTW(e, rows, n)
			if !pathEq(theirs) {
				t.Errorf("%s %s: DTW over upstream's energy is not upstream's path", label, name)
			}
			if !pathEq(ours) {
				t.Errorf("%s %s: DTW over our energy is not upstream's path", label, name)
			}
			a := Metrics(e, rows, n, mask, ours)
			w := c.Rows[name]
			t.Logf("%s %s: energy %dx%d worst %.1e, path %d steps; cov %.4f mono %.4f conf %.6f score %.6f (upstream %.4f)",
				label, name, rows, n, worst, len(ours), a.Coverage, a.Monotonicity, a.Confidence, a.Score, w.Score)
			if worst > 1e-6 {
				t.Errorf("%s %s: energy %.1e off upstream's", label, name, worst)
			}
			if math.Abs(a.Coverage-w.Coverage) > 1e-12 || math.Abs(a.Monotonicity-w.Monotonicity) > 1e-12 ||
				math.Abs(a.Confidence-w.Confidence) > 1e-6 || math.Abs(a.Score-w.Score) > 5e-5 {
				t.Errorf("%s %s: %+v, upstream %+v", label, name, a, w)
			}
		}
	}
}

// TestLyricAlignment is A14b's DiT gate: the listed heads' probabilities
// from upstream's decoder inputs (its condition, context, latent and noise)
// against fp32's, priced against upstream bf16's; then LyricAlignment end to
// end, both scores within bf16's drift of fp32 (or 0.01).
func TestLyricAlignment(t *testing.T) {
	p := pipeline(t)
	heads, err := p.DiT.Config().AlignHeads()
	if err != nil {
		t.Fatal(err)
	}
	for _, label := range alignCases {
		c := loadAlign(t, scoreRef, label)
		cb := loadAlign(t, scoreBF16Ref, label)
		if len(heads) != len(c.Heads) {
			t.Fatalf("%d heads, upstream read %d", len(heads), len(c.Heads))
		}
		for i := range heads {
			if heads[i] != c.Heads[i] {
				t.Fatalf("heads %v, upstream %v", heads, c.Heads)
			}
		}
		pre := filepath.Join(scoreRef, "align_"+label+"_")
		enc := readF32(t, pre+"encoder_states.bin")
		if err := p.DiT.Begin(&qwen.Mat{Rows: c.EncoderLen, Cols: len(enc) / c.EncoderLen, Data: enc}); err != nil {
			t.Fatal(err)
		}
		T := c.LatentLength
		ctxD := readF32(t, pre+"context_latents.bin")
		ctx := &qwen.Mat{Rows: T, Cols: len(ctxD) / T, Data: ctxD}
		xt := readF32(t, pre+"xt.bin")
		want, bf := oracleProbs(t, scoreRef, label, c), oracleProbs(t, scoreBF16Ref, label, cb)
		for ri, tt := range []float32{1, 1.0 / 8} {
			x := &qwen.Mat{Rows: T, Cols: plan.LatentChannels, Data: xt[ri*T*plan.LatentChannels : (ri+1)*T*plan.LatentChannels]}
			got, err := p.DiT.CrossProbs(x, ctx, tt, heads)
			if err != nil {
				t.Fatal(err)
			}
			var se, sb, sw float64
			for h := range heads {
				for i, w := range want[ri][h] {
					d, db := float64(got[h][i]-w), float64(bf[ri][h][i]-w)
					se, sb, sw = se+d*d, sb+db*db, sw+float64(w)*float64(w)
				}
			}
			rel, relB := math.Sqrt(se/sw), math.Sqrt(sb/sw)
			t.Logf("%s t=%.3f: probabilities rms %.2e of fp32's (bf16 %.2e)", label, tt, rel, relB)
			if rel > relB {
				t.Errorf("%s t=%.3f: probabilities rms %.2e, bf16's %.2e", label, tt, rel, relB)
			}
		}

		lat := readF32(t, pre+"latents.bin")
		noise := readF32(t, pre+"noise.bin")
		ls, err := p.LyricAlignment(request(t, label), lat, ctx, 8, noise)
		if err != nil {
			t.Fatal(err)
		}
		for _, row := range []struct {
			name string
			got  Alignment
		}{{"lm", ls.LM}, {"dit", ls.DiT}} {
			w, b := c.Rows[row.name], cb.Rows[row.name]
			d, db := math.Abs(row.got.Score-w.Score), math.Abs(b.Score-w.Score)
			t.Logf("%s %s: score %.4f (cov %.3f mono %.3f conf %.4f); fp32 %.4f, bf16 %.4f", label, row.name,
				row.got.Score, row.got.Coverage, row.got.Monotonicity, row.got.Confidence, w.Score, b.Score)
			if d > math.Max(db, 0.01) {
				t.Errorf("%s %s: score %.4f, fp32 %.4f (bf16 is %.4f off)", label, row.name, row.got.Score, w.Score, db)
			}
		}
	}
}

// TestScoreGenerate runs the auto score where the server runs it, after a
// thinking-path song (Options.Score), and logs what it says and costs: A1's
// 30 s full_metas request and its 200 s long_caption one (12 verses).
func TestScoreGenerate(t *testing.T) {
	p := pipeline(t)
	if p.LM == nil {
		if err := p.LoadLM(shared.dev, LMDir(models), lm.DefaultOptions()); err != nil {
			t.Skip(err)
		}
	}
	if p.Aes == nil {
		if err := p.LoadAesthetics(shared.dev, AesDir(models)); err != nil {
			t.Skip(err)
		}
	}
	for _, label := range []string{"full_metas", "long_caption"} {
		res, err := p.Generate(request(t, label), Options{Seed: 42, LMSeed: 42, Think: true, Score: true})
		if err != nil {
			t.Fatal(err)
		}
		q := res.Quality
		if q == nil || q.LM == nil || q.Lyrics == nil || q.Aesthetics == nil {
			t.Fatalf("%s: quality %+v", label, q)
		}
		t.Logf("%s: %.0f s song in %v, scored in %v (LM %v); reward %.4f %v pmi %v; lyrics lm %+v dit %+v",
			label, res.Seconds, res.Timings.Total.Round(time.Millisecond), q.Took.Round(time.Millisecond), q.LM.Stats,
			q.LM.Reward, q.LM.Conditions, q.LM.PMI, q.Lyrics.LM, q.Lyrics.DiT)
		t.Logf("%s: aesthetics %+v", label, *q.Aesthetics)
		if a := q.Aesthetics; a.PQ < 1 || a.PQ > 10 || math.IsNaN(a.CE) {
			t.Errorf("%s: aesthetics %+v", label, *a)
		}
		for k, v := range q.LM.Conditions {
			if v < 0 || v > 1 || math.IsNaN(v) {
				t.Errorf("%s: %s scored %v", label, k, v)
			}
		}
		for _, a := range []Alignment{q.Lyrics.LM, q.Lyrics.DiT} {
			if a.Score < 0 || a.Score > 1 || math.IsNaN(a.Score) {
				t.Errorf("%s: alignment %+v", label, a)
			}
		}
	}
}
