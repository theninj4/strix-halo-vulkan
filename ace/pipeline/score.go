package pipeline

// Upstream's auto score over a finished song (MUSIC.md A14b): the Gradio
// app's "Auto Score" (ui/gradio/events/results/scoring.py), which its API
// does not serve. Two scorers, each optional:
//
//   - the LM's PMI and recall over the song's codes (lm.Planner.Score), on
//     the thinking path, where the LM wrote codes;
//   - the DiT's lyric alignment (get_lyric_score): one forward of the song's
//     final latent and one of pure noise, the cross-attention of the
//     checkpoint's lyric_alignment_layers_config heads over the lyric
//     tokens, a DTW path through it, and coverage² · monotonicity² ·
//     confidence (core/scoring/dit_score.MusicLyricScorer). It runs when
//     the song has lyrics.

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/loops"
	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/audiobox"
	h3vae "strix-halo-vulkan/h3/vae"
	"strix-halo-vulkan/songeval"
	"strix-halo-vulkan/zimage/qwen"
)

// Quality is a song's auto score. A part is nil when it did not run.
type Quality struct {
	// LM is the 5 Hz LM's verdict: per condition and the reward.
	LM *lm.Score
	// Lyrics is the DiT's lyric alignment.
	Lyrics *LyricScore
	// Aesthetics is Audiobox Aesthetics' four scores on the audio
	// (MUSIC.md A14c), 1-10.
	Aesthetics *audiobox.Scores
	// Song is SongEval's five whole-song scores (MUSIC.md A14d), 1-5.
	Song *songeval.Scores
	// Render is the LM's verdict over codes re-tokenized from the finished
	// song rather than the codes it planned (MUSIC.md A-o16): what was
	// made, not what was meant. It runs whenever the LM is loaded, so a
	// cover or a repaint has one too.
	Render *lm.Score
	// Loops is how much of the song repeats itself at one short lag
	// (MUSIC.md A14e), on the CPU.
	Loops *loops.Result
	Took  time.Duration
}

// LyricScore is get_lyric_score's pair: "lm" from pure noise at t = 1 (the
// condition alone), "dit" from the song's latent noised to the last step.
type LyricScore struct {
	LM, DiT Alignment
}

// Alignment is one row's metrics and their product.
type Alignment struct {
	Coverage, Monotonicity, Confidence float64
	// Score is coverage² · monotonicity² · confidence, in [0, 1]
	// (upstream rounds it to 4 places for display; this is unrounded).
	Score float64
}

// alignSeed is the noise get_lyric_score's Gradio caller asks for.
const alignSeed = 42

// The metrics' constants (_compute_alignment_metrics' defaults).
const (
	coverageMin  = 0.1  // a sung row counts when its peak energy passes this
	timeWeight   = 0.01 // energies at or under this carry no weight in a centroid
	overlapRows  = 9.0  // a centroid may step back this many frames
	instrumental = 1.0  // a tag row's weight on the path
)

// TypeMask is _generate_token_type_mask: 1 for a lyric token, 0 for one
// inside a [structure tag], each token decoded alone.
func (p *Pipeline) TypeMask(ids []int32) []int {
	mask := make([]int, len(ids))
	in := false
	for i, id := range ids {
		s, _ := p.Tok.Decode([]int32{id})
		mask[i] = 1
		if strings.Contains(s, "[") {
			in = true
		}
		if in {
			mask[i] = 0
		}
		if strings.Contains(s, "]") {
			in = false
			mask[i] = 0
		}
	}
	return mask
}

// LyricSpan is _extract_lyric_segment over a request's lyric prompt ids:
// the lyric tokens start after the "# Languages / # Lyric" header and end
// at <|endoftext|>.
func (p *Pipeline) LyricSpan(r *plan.Request, ids []int32) (start, end int, err error) {
	lang := r.Language
	if lang == "" {
		lang = "unknown" // LyricsPrompt's
	}
	head, err := p.Tok.Encode("# Languages\n" + lang + "\n\n# Lyric\n")
	if err != nil {
		return 0, 0, err
	}
	start, end = len(head), len(ids)
	for i, id := range ids {
		if id == lm.EndOfTextID {
			end = i
			break
		}
	}
	return start, end, nil
}

// HeadEnergy is _preprocess_attention's average: the listed heads'
// probabilities ([tokens][encoder rows] each) over the lyric rows
// [start, end), transposed and averaged into [end−start][tokens].
func HeadEnergy(probs [][]float32, tokens, encRows, start, end int) []float32 {
	rows := end - start
	out := make([]float32, rows*tokens)
	for j := 0; j < rows; j++ {
		for i := 0; i < tokens; i++ {
			var s float32
			for _, p := range probs {
				s += p[i*encRows+start+j]
			}
			out[j*tokens+i] = s / float32(len(probs))
		}
	}
	return out
}

// Normalize is the min-max normalisation in fp32, as numpy does it on the
// head average; a flat matrix is zeros.
func Normalize(avg []float32) []float32 {
	lo, hi := float32(math.Inf(1)), float32(math.Inf(-1))
	for _, v := range avg {
		lo, hi = min(lo, v), max(hi, v)
	}
	out := make([]float32, len(avg))
	if hi-lo > 1e-9 {
		for i, v := range avg {
			out[i] = (v - lo) / (hi - lo)
		}
	}
	return out
}

// DTW is _dtw.dtw_cpu over −energy²: the fp32 cost accumulation and its
// tie-breaking (diagonal only when strictly cheapest, then up, else left),
// and the backtrace. It returns (row, column) pairs from (0, 0).
func DTW(energy []float32, rows, cols int) [][2]int {
	W := cols + 1
	inf := float32(math.Inf(1))
	cost := make([]float32, (rows+1)*W)
	trace := make([]int8, (rows+1)*W)
	for i := range cost {
		cost[i], trace[i] = inf, -1
	}
	cost[0] = 0
	for j := 1; j <= cols; j++ {
		for i := 1; i <= rows; i++ {
			c0, c1, c2 := cost[(i-1)*W+j-1], cost[(i-1)*W+j], cost[i*W+j-1]
			var c float32
			var t int8
			switch {
			case c0 < c1 && c0 < c2:
				c, t = c0, 0
			case c1 < c0 && c1 < c2:
				c, t = c1, 1
			default:
				c, t = c2, 2
			}
			// The explicit conversion keeps Go from fusing the square
			// into the add: numba rounds energy² first.
			e := energy[(i-1)*cols+j-1]
			cost[i*W+j] = -float32(e*e) + c
			trace[i*W+j] = t
		}
	}
	for j := 0; j <= cols; j++ {
		trace[j] = 2
	}
	for i := 0; i <= rows; i++ {
		trace[i*W] = 1
	}
	var path [][2]int
	for i, j := rows, cols; i > 0 || j > 0; {
		path = append(path, [2]int{i - 1, j - 1})
		switch trace[i*W+j] {
		case 0:
			i, j = i-1, j-1
		case 1:
			i--
		case 2:
			j--
		default:
			i, j = 0, 0
		}
	}
	for a, b := 0, len(path)-1; a < b; a, b = a+1, b-1 {
		path[a], path[b] = path[b], path[a]
	}
	return path
}

// Metrics is _compute_alignment_metrics and calculate_score over the
// normalised energy [rows][cols], the type mask and the path.
func Metrics(energy []float32, rows, cols int, mask []int, path [][2]int) Alignment {
	var a Alignment
	sung, covered := 0, 0
	cent := make([]float64, rows)
	for r := 0; r < rows; r++ {
		row := energy[r*cols : (r+1)*cols]
		peak := math.Inf(-1)
		var sw, st float64
		for c, v := range row {
			e := float64(v)
			peak = math.Max(peak, e)
			if e > timeWeight {
				sw += e
				st += e * float64(c)
			}
		}
		cent[r] = -1
		if sw > 1e-9 {
			cent[r] = st / sw
		}
		if mask[r] == 1 {
			sung++
			if peak > coverageMin {
				covered++
			}
		}
	}
	a.Coverage = 1
	if sung > 0 {
		a.Coverage = float64(covered) / float64(sung)
	}
	var seq []float64
	for r := 0; r < rows; r++ {
		if mask[r] == 1 && cent[r] >= 0 {
			seq = append(seq, cent[r])
		}
	}
	a.Monotonicity = 1
	if len(seq) > 1 {
		ok := 0
		for i := 1; i < len(seq); i++ {
			if seq[i] >= seq[i-1]-overlapRows {
				ok++
			}
		}
		a.Monotonicity = float64(ok) / float64(len(seq)-1)
	}
	var se, sw float64
	for _, rc := range path {
		w := 1.0
		if mask[rc[0]] == 0 {
			w = instrumental
		}
		se += float64(energy[rc[0]*cols+rc[1]]) * w
		sw += w
	}
	if sw > 0 {
		a.Confidence = se / sw
	}
	a.Score = math.Min(math.Max(a.Coverage*a.Coverage*a.Monotonicity*a.Monotonicity*a.Confidence, 0), 1)
	return a
}

// Align scores one forward's head probabilities: HeadEnergy, Normalize,
// DTW, Metrics.
func Align(probs [][]float32, tokens, encRows, start, end int, mask []int) Alignment {
	rows := end - start
	e := Normalize(HeadEnergy(probs, tokens, encRows, start, end))
	return Metrics(e, rows, tokens, mask, DTW(e, rows, tokens))
}

// LyricAlignment is get_lyric_score over a finished song: r is the DiT's
// request, lat its final latent [T, 64], ctx the context latents it was
// made over, steps the sampler's length. The DiT must still hold r's
// condition (Begin), which generate leaves it holding. noise, when set,
// replaces the seed-42 draw (the gates feed the oracle's).
func (p *Pipeline) LyricAlignment(r *plan.Request, lat []float32, ctx *qwen.Mat, steps int, noise []float32) (*LyricScore, error) {
	if strings.TrimSpace(r.Lyrics) == "" {
		return nil, nil
	}
	heads, err := p.DiT.Config().AlignHeads()
	if err != nil {
		return nil, err
	}
	_, ids, err := p.Ids(r)
	if err != nil {
		return nil, err
	}
	start, end, err := p.LyricSpan(r, ids)
	if err != nil {
		return nil, err
	}
	if start >= end {
		return nil, fmt.Errorf("ace: no lyric tokens to align (%d..%d)", start, end)
	}
	mask := p.TypeMask(ids[start:end])
	T := len(lat) / plan.LatentChannels
	if noise == nil {
		if noise, err = h3vae.TorchRandn(alignSeed, len(lat)); err != nil {
			return nil, err
		}
	}
	tl := float32(1 / float64(steps))
	xt := make([]float32, len(lat))
	for i := range xt {
		xt[i] = float32(tl*noise[i]) + float32((1-tl)*lat[i]) // torch's two fp32 products, unfused
	}
	out := &LyricScore{}
	for _, row := range []struct {
		x []float32
		t float32
		a *Alignment
	}{{noise, 1, &out.LM}, {xt, tl, &out.DiT}} {
		probs, err := p.DiT.CrossProbs(&qwen.Mat{Rows: T, Cols: plan.LatentChannels, Data: row.x}, ctx, row.t, heads)
		if err != nil {
			return nil, err
		}
		*row.a = Align(probs, p.DiT.Tokens(T), p.DiT.EncLen(), start, end, mask)
	}
	return out, nil
}

// ScoreConditions are the conditions the LM's score reads for a song, as
// Gradio's _build_score_metadata assembles them: the CoT's metadata (its
// caption is the one the DiT got when phase 1 ran), the request's where
// the CoT left a field out, and the request's lyrics. One departure: a
// language is scored only when the request or the CoT names one, where
// Gradio scores its own "unknown" default.
func ScoreConditions(r, d *plan.Request, meta lm.Meta) lm.Conditions {
	m := lm.Meta{}
	for k, v := range meta {
		if k != "caption" {
			m[k] = v
		}
	}
	set := func(k, v string) {
		if _, ok := m[k]; !ok && v != "" {
			m[k] = v
		}
	}
	if d.BPM > 0 {
		set("bpm", strconv.Itoa(d.BPM))
	}
	if d.Duration > 0 {
		set("duration", strconv.Itoa(int(d.Duration)))
	}
	set("keyscale", strings.TrimSpace(d.KeyScale))
	set("timesignature", strings.TrimSpace(d.TimeSignature))
	if l := strings.TrimSpace(r.Language); l != "unknown" {
		set("language", l)
	}
	if l := m["language"]; l == "unknown" || l == "" {
		delete(m, "language")
	}
	return lm.Conditions{Caption: d.Caption, Lyrics: r.Lyrics, Meta: m}
}
