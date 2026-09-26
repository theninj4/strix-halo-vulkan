// Package plan is everything a MiniMax-H3 request decides before a weight is
// touched (VIDEO.md M1): the canvas a ratio resolves to, the 17n+5 frame
// snap and the latent counts, the packed sequence layout the transformer
// runs over, its rotary tables, and the two rectified-flow schedules.
//
// It is the Go counterpart of diffusers' modular_pipelines/minimax_h3
// arithmetic (resolve_canvas_size, align_num_frames, the layout step's
// build_packed_sequence, build_row_timesteps) and of MiniMaxH3Scheduler, and
// it is gated bit-exactly against them (reference/dump_h3_plan.py). That is
// why several things below are written the long way round:
//
//   - positions are float64 and non-integer — the spatial axes are an
//     aspect-normalised numpy linspace, and time advances 5/3·(1,4,4,4,4) per
//     latent frame from the text length — and the reference sums the same
//     series in two different orders (torch's sequential cumsum for the
//     frames, numpy's pairwise sum for a "last" keyframe's anchor), which
//     differ in the last ulp from 16 latent frames on, so both are kept;
//   - Python's round is round-half-to-even;
//   - torch's float32 linspace is a fused multiply-add on a float32 step;
//   - the Euler step is float32 elementwise with no fusion, so every product
//     is narrowed explicitly before it is added (Go may otherwise fuse).
package plan

import (
	"fmt"
	"math"
	"sort"
)

// The released checkpoint's constants, as MiniMaxH3ModularPipeline and the
// transformer config carry them.
const (
	FPS                = 24
	CanvasMultiple     = 32 // VAE spatial 16 × transformer patch 2
	SpatialCompression = 16
	Patch              = 2 // (1, 2, 2): time is not patched
	FramesPerChunk     = 17
	LatentsPerChunk    = 5
	AudioLatentsPerSec = 40
	AudioChannels      = 2
	ShortEdge          = 768
	MaxPixels          = 768 * 1344
	MinDuration        = 5.0
	MaxDuration        = 15.0
	KeyframeNoiseAug   = 0.999

	VideoTag = 0
	TextTag  = 1
	AudioTag = 2

	RopeFreqDim = 16
	RopeTheta   = 10000.0
)

// ropeFrameRescale and ropeFramesPerLatent give a latent frame's rotary span:
// 5/3 × (1, 4, 4, 4, 4), the VAE's 17-frames-to-5-latents grouping.
const ropeSpatialScale = 32

var (
	ropeFrameRescale    = 5.0 / 3.0
	ropeFramesPerLatent = [5]float64{1, 4, 4, 4, 4}
)

// Canvas resolves an aspect ratio (or a keyframe's pixel size) to the
// (height, width) MiniMax-H3 generates at: the short edge at shortEdge, the
// area capped at maxPixels, both axes rounded to CanvasMultiple.
func Canvas(aspectW, aspectH float64, shortEdge, maxPixels int) (h, w int, err error) {
	if aspectW <= 0 || aspectH <= 0 {
		return 0, 0, fmt.Errorf("h3: aspect ratio must be positive, got %g:%g", aspectW, aspectH)
	}
	ratio := aspectW / aspectH
	if ratio < 0.25 || ratio > 4 {
		return 0, 0, fmt.Errorf("h3: aspect ratios run from 1:4 to 4:1, got %g:%g", aspectW, aspectH)
	}
	var fw, fh float64
	if ratio >= 1 {
		fw, fh = float64(shortEdge)*ratio, float64(shortEdge)
	} else {
		fw, fh = float64(shortEdge), float64(shortEdge)/ratio
	}
	if area := fw * fh; area > float64(maxPixels) {
		scale := math.Sqrt(float64(maxPixels) / area)
		fw, fh = fw*scale, fh*scale
	}
	m := float64(CanvasMultiple)
	return max(CanvasMultiple, int(math.RoundToEven(fh/m)*m)), max(CanvasMultiple, int(math.RoundToEven(fw/m)*m)), nil
}

// AlignFrames snaps a frame count up to the next 17n+5 the video VAE codes.
func AlignFrames(n int) int {
	for n%FramesPerChunk != LatentsPerChunk {
		n++
	}
	return n
}

// LatentFrames is the video VAE's latent frame count for an aligned count:
// 5n+2 for 17n+5.
func LatentFrames(aligned int) int {
	return (aligned-LatentsPerChunk)/FramesPerChunk*LatentsPerChunk + 2
}

// AudioLatents is the audio latent count (per channel) covering n frames.
func AudioLatents(frames int) int {
	return int(math.RoundToEven(float64(frames) / FPS * AudioLatentsPerSec))
}

// Frames validates a requested frame count the way the layout step does: it
// is aligned first, and the aligned duration must lie in [5 s, 15 s].
func Frames(requested int) (int, error) {
	if requested < 1 {
		return 0, fmt.Errorf("h3: num_frames must be positive, got %d", requested)
	}
	aligned := AlignFrames(requested)
	if d := float64(aligned) / FPS; d < MinDuration || d > MaxDuration {
		return 0, fmt.Errorf("h3: %d frames align to %d (%.3g s), outside %g–%g s",
			requested, aligned, d, MinDuration, MaxDuration)
	}
	return aligned, nil
}

// spatialGrid is one aspect-normalised rotary axis: np.linspace(left,
// left+ratio, dim/patch, endpoint=False) × 32, reproduced as numpy computes
// it (start + i·((stop−start)/num)).
func spatialGrid(dim int, sqrtArea float64) []float64 {
	ratio := float64(dim) / sqrtArea
	left := (1.0 - ratio) / 2.0
	stop := left + ratio
	n := dim / Patch
	step := (stop - left) / float64(n)
	out := make([]float64, n)
	for i := range out {
		out[i] = (float64(i)*step + left) * ropeSpatialScale
	}
	return out
}

// temporalGrid is every latent frame's rotary time from origin: a sequential
// cumulative sum of 5/3·(1,4,4,4,4), as torch.cumsum over float64 runs it.
func temporalGrid(frames int, origin float64) []float64 {
	out := make([]float64, frames)
	var acc float64
	for i := range out {
		out[i] = origin + acc
		acc += ropeFrameRescale * ropeFramesPerLatent[i%len(ropeFramesPerLatent)]
	}
	return out
}

// pairwiseSum is numpy's float64 add.reduce over a contiguous array: blocks
// of up to 128 summed through eight accumulators, larger arrays split at a
// multiple of eight and recursed.
func pairwiseSum(a []float64) float64 {
	n := len(a)
	switch {
	case n < 8:
		var r float64
		for _, x := range a {
			r += x
		}
		return r
	case n <= 128:
		var r [8]float64
		copy(r[:], a[:8])
		i := 8
		for ; i < n-n%8; i += 8 {
			for j := range r {
				r[j] += a[i+j]
			}
		}
		res := ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
		for ; i < n; i++ {
			res += a[i]
		}
		return res
	default:
		n2 := n / 2
		n2 -= n2 % 8
		return pairwiseSum(a[:n2]) + pairwiseSum(a[n2:])
	}
}

// Anchor says which end of the video a keyframe conditioning block holds.
type Anchor int

const (
	First Anchor = iota
	Last
)

// Layout is one request's packed sequence:
// [text | keyframe conditions | audio L, audio R | video], every row with a
// float64 (t, h, w) rotary coordinate and a modality tag.
type Layout struct {
	TextTokens, LatentFrames, LatentH, LatentW, AudioLatents int
	RowsPerFrame                                             int
	CondVideoRows                                            int // leading video rows that are conditioning

	Pos  [][3]float64
	Tags []int32
	// Video lists the video rows' sequence positions, conditioning rows first;
	// Audio the audio rows' (L then R); Text the text rows'.
	Video, Audio, Text []int32
}

// NewLayout builds the t2va/fl2va layout. textTags is the modality tag of
// every text row (a keyframe's vision block inside the prompt is tagged
// video); latentH and latentW are in latent pixels (canvas / 16).
func NewLayout(textTags []int32, latentFrames, latentH, latentW, audioLatents int, anchors []Anchor) (*Layout, error) {
	if latentH%Patch != 0 || latentW%Patch != 0 {
		return nil, fmt.Errorf("h3: latent %dx%d is not a whole number of %d-patches", latentH, latentW, Patch)
	}
	nText := len(textTags)
	rowsPerFrame := (latentH / Patch) * (latentW / Patch)
	nCond := len(anchors) * rowsPerFrame
	nAudio := audioLatents * AudioChannels
	nVideo := latentFrames * rowsPerFrame
	seq := nText + nCond + nAudio + nVideo
	condStart := nText
	audioStart := condStart + nCond
	videoStart := audioStart + nAudio

	l := &Layout{
		TextTokens: nText, LatentFrames: latentFrames, LatentH: latentH, LatentW: latentW,
		AudioLatents: audioLatents, RowsPerFrame: rowsPerFrame, CondVideoRows: nCond,
		Pos: make([][3]float64, seq), Tags: make([]int32, seq),
	}
	for i := 0; i < nText; i++ {
		l.Pos[i][0] = float64(i)
	}

	sqrtArea := math.Sqrt(float64(latentH * latentW))
	hGrid := spatialGrid(latentH, sqrtArea)
	wGrid := spatialGrid(latentW, sqrtArea)
	frame := func(start int, t float64) {
		for r, hv := range hGrid {
			for c, wv := range wGrid {
				l.Pos[start+r*len(wGrid)+c] = [3]float64{t, hv, wv}
			}
		}
	}

	origin := float64(nText)
	for k, a := range anchors {
		var t float64
		switch a {
		case First:
			t = origin
		case Last:
			spans := make([]float64, latentFrames)
			for i := range spans {
				spans[i] = ropeFrameRescale * ropeFramesPerLatent[i%len(ropeFramesPerLatent)]
			}
			t = origin + pairwiseSum(spans) - ropeFrameRescale
		default:
			return nil, fmt.Errorf("h3: unknown keyframe anchor %d", a)
		}
		frame(condStart+k*rowsPerFrame, t)
	}

	for ch := 0; ch < AudioChannels; ch++ {
		w := wGrid[0]
		if ch > 0 {
			w = wGrid[len(wGrid)-1]
		}
		for i := 0; i < audioLatents; i++ {
			l.Pos[audioStart+ch*audioLatents+i] = [3]float64{origin + float64(i), 0, w}
		}
	}

	for f, t := range temporalGrid(latentFrames, origin) {
		frame(videoStart+f*rowsPerFrame, t)
	}

	l.Text = seqRange(0, nText)
	l.Audio = seqRange(audioStart, videoStart)
	l.Video = append(seqRange(condStart, audioStart), seqRange(videoStart, seq)...)
	copy(l.Tags, textTags)
	for _, i := range l.Audio {
		l.Tags[i] = AudioTag
	}
	for _, i := range l.Video {
		l.Tags[i] = VideoTag
	}
	return l, nil
}

func seqRange(lo, hi int) []int32 {
	out := make([]int32, hi-lo)
	for i := range out {
		out[i] = int32(lo + i)
	}
	return out
}

// InvFreq is the transformer's rotary inverse frequencies,
// 1 / θ^(arange(0, 32, 2) / 32) in float32.
func InvFreq() []float32 {
	out := make([]float32, RopeFreqDim)
	for i := range out {
		e := float32(2*i) / float32(2*RopeFreqDim)
		out[i] = float32(1.0 / float64(float32(math.Pow(RopeTheta, float64(e)))))
	}
	return out
}

// Rope returns the per-row cos and sin tables, [rows, 96] row-major: the
// 16 frequencies over t, then h, then w, and that 48 repeated, which is what
// rotate-half applies to the first 96 of each head's 128 channels. The
// positions are narrowed to float32 first, as the model does.
func (l *Layout) Rope(invFreq []float32) (cos, sin []float32) {
	const width = 2 * 3 * RopeFreqDim
	cos = make([]float32, len(l.Pos)*width)
	sin = make([]float32, len(l.Pos)*width)
	for r, p := range l.Pos {
		row := r * width
		for axis := 0; axis < 3; axis++ {
			x := float32(p[axis])
			for k, f := range invFreq {
				a := float64(x * f)
				c, s := float32(math.Cos(a)), float32(math.Sin(a))
				j := row + axis*RopeFreqDim + k
				cos[j], sin[j] = c, s
				cos[j+width/2], sin[j+width/2] = c, s
			}
		}
	}
	return cos, sin
}

// Schedule is one modality's rectified-flow grid: Sigmas holds the grid
// points with the terminal 0, Timesteps = 1 − σ for every point but the last,
// one per forward.
type Schedule struct {
	Sigmas    []float32
	Timesteps []float32
}

// NewSchedule builds linspace(1, 0, steps) in float32, shifted by
// σ' = sσ / (1 + (s−1)σ), with consecutive duplicates collapsed.
func NewSchedule(steps int, shift float32) (*Schedule, error) {
	if steps < 2 {
		return nil, fmt.Errorf("h3: a schedule needs at least 2 grid points, got %d", steps)
	}
	step := float32(0-1) / float32(steps-1) // torch: (end − start) / (steps − 1), in float32
	half := steps / 2
	var sig []float32
	for i := 0; i < steps; i++ {
		var b float32
		if i < half {
			b = float32(1 + float64(step)*float64(i)) // exact in float64, one rounding: torch's fma
		} else {
			b = float32(0 - float64(step)*float64(steps-i-1))
		}
		num := float32(shift * b)
		den := float32(1 + float32((shift-1)*b))
		s := num / den
		if len(sig) == 0 || s != sig[len(sig)-1] {
			sig = append(sig, s)
		}
	}
	ts := make([]float32, len(sig)-1)
	for i := range ts {
		ts[i] = 1 - sig[i]
	}
	return &Schedule{Sigmas: sig, Timesteps: ts}, nil
}

// Step is MiniMaxH3Scheduler.step for step index i, in place over x: the
// data-ward velocity gives x0 = x + (1 − t)·v, with t the timestep the model
// saw, and then x ← r·x + (1 − r)·x0 with r = σ[i+1]/σ[i] from the grid.
func (s *Schedule) Step(i int, v, x []float32) {
	sigmaT := 1 - s.Timesteps[i]
	r := s.Sigmas[i+1] / s.Sigmas[i]
	omr := 1 - r
	for k := range x {
		x0 := x[k] + float32(sigmaT*v[k])
		x[k] = float32(r*x[k]) + float32(omr*x0)
	}
}

// RowTimesteps assigns every row of l its timestep for one forward and
// reduces them to the distinct values, sorted, and each row's index into
// them: generated video rows and text rows at tVideo, generated audio rows at
// tAudio, keyframe rows at max(tVideo, 0.999).
func (l *Layout) RowTimesteps(tVideo, tAudio float32) (unique []float32, index []int32) {
	cond := max(tVideo, float32(KeyframeNoiseAug))
	row := make([]float32, len(l.Pos))
	for i := range row {
		row[i] = tVideo
	}
	for _, i := range l.Video[:l.CondVideoRows] {
		row[i] = cond
	}
	for _, i := range l.Audio {
		row[i] = tAudio
	}
	seen := map[float32]bool{}
	for _, t := range row {
		if !seen[t] {
			seen[t] = true
			unique = append(unique, t)
		}
	}
	sort.Slice(unique, func(a, b int) bool { return unique[a] < unique[b] })
	pos := make(map[float32]int32, len(unique))
	for i, t := range unique {
		pos[t] = int32(i)
	}
	index = make([]int32, len(row))
	for i, t := range row {
		index[i] = pos[t]
	}
	return unique, index
}
