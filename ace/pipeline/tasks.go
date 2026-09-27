package pipeline

// The audio-in tasks (MUSIC.md A11): a reference audio's timbre on any
// request, and the turbo model's cover, cover-nofsq and repaint. All of them
// run the DiT-only path with a different wiring: the VAE encoder turns audio
// into latents (ace/vae.Encoder, sampled as upstream samples its
// posterior), and what the DiT is conditioned on, starts from and is reset
// to between steps follows upstream's handler at ca1e85f:
//
//   - reference: process_reference_audio's three 10 s segments, encoded,
//     read by the timbre encoder where the silence was;
//   - cover: the source's latents through the audio tokenizer and back
//     (is_covers), the DiT's source; cover-nofsq: the latents themselves;
//     audio_cover_strength < 1 switches to a text2music condition part way,
//     cover_noise_strength > 0 starts from the renoised source;
//   - repaint: the source's latents with the span silenced, the region
//     outside the span reset to the noised source after early steps, a
//     latent crossfade at the edges, and the source waveform spliced back
//     after the decode.

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/ace/vae"
	h3vae "strix-halo-vulkan/h3/vae"
	"strix-halo-vulkan/zimage/qwen"
)

// Tasks the turbo model serves (upstream's TASK_TYPES_TURBO).
const (
	Text2Music = "text2music"
	Cover      = "cover"
	CoverNoFSQ = "cover-nofsq"
	Repaint    = "repaint"
)

// RepaintInstruction and the cover's instruction (lm.Instruction) are the
// caption prompt's first line for those tasks (TASK_INSTRUCTIONS).
const RepaintInstruction = "Repaint the mask area based on the given conditions:"

// Audio is the audio-in part of a request. The zero value is text2music
// with no reference, and every field past Task keeps upstream's default at
// its zero value except the two strengths (NewAudio sets them).
type Audio struct {
	Task string
	// Source is the cover or repaint source, Reference the timbre
	// reference: interleaved 48 kHz stereo (ReadAudio), in [-1, 1].
	Source, Reference []float32
	// CoverStrength is audio_cover_strength (1: cover conditioning at every
	// step), CoverNoise cover_noise_strength (0: start from pure noise).
	CoverStrength, CoverNoise float64
	// RepaintStart and RepaintEnd are the span in seconds of the source; an
	// end ≤ 0 is the source's end, and either past an end pads the source
	// with silence (outpainting).
	RepaintStart, RepaintEnd float64
	// RepaintMode is conservative, balanced ("") or aggressive, and
	// RepaintStrength balanced's knob (0 keeps the most of the source).
	RepaintMode     string
	RepaintStrength float64
	// ExplicitMask is chunk_mask_mode "explicit": the chunk mask is the
	// span; "auto" (false) leaves it 1.0 everywhere.
	ExplicitMask bool

	// For the gates: the oracle's sampled latents ([T, 64]) in place of
	// encoding Source and Reference, and its segment draws.
	SourceLatents, ReferenceLatents []float32
	RefOffsets                      []int
}

// NewAudio is a task with upstream's defaults.
func NewAudio(task string) *Audio {
	return &Audio{Task: task, CoverStrength: 1, RepaintStrength: 0.5}
}

// needsSource reports whether the task reads a source.
func (a *Audio) needsSource() bool {
	return a != nil && (a.Task == Cover || a.Task == CoverNoFSQ || a.Task == Repaint)
}

// Seconds is how long a task's song is, before it runs: the source's whole
// latents (padded by a repaint's span past either end), at least 5.12 s.
// Zero for text2music, whose length is the request's.
func (a *Audio) Seconds() float64 {
	if !a.needsSource() {
		return 0
	}
	n := len(a.Source) / 2
	if a.Task == Repaint {
		dur := float64(n) / plan.SampleRate
		end := a.RepaintEnd
		if end <= 0 {
			end = dur
		}
		n += int(max(0, -a.RepaintStart)*plan.SampleRate) + int(max(0, end-dur)*plan.SampleRate)
	}
	return float64(max(plan.MinLatents, n/plan.Hop)*plan.Hop) / plan.SampleRate
}

// Check validates the task against what the turbo model serves.
func (a *Audio) Check() error {
	if a == nil {
		return nil
	}
	switch a.Task {
	case "", Text2Music, Cover, CoverNoFSQ, Repaint:
	default:
		return fmt.Errorf("ace: task %q: the turbo model serves text2music, cover, cover-nofsq and repaint", a.Task)
	}
	if a.needsSource() && len(a.Source) == 0 && a.SourceLatents == nil {
		return fmt.Errorf("ace: task %s needs source audio", a.Task)
	}
	if a.CoverStrength < 0 || a.CoverStrength > 1 || a.CoverNoise < 0 || a.CoverNoise > 1 {
		return fmt.Errorf("ace: cover strengths %v and %v are outside [0, 1]", a.CoverStrength, a.CoverNoise)
	}
	switch a.RepaintMode {
	case "", "balanced", "conservative", "aggressive":
	default:
		return fmt.Errorf("ace: repaint mode %q (conservative, balanced or aggressive)", a.RepaintMode)
	}
	if a.Reference != nil && silent(a.Reference) {
		return fmt.Errorf("ace: the reference audio is silent")
	}
	return nil
}

// silent is upstream's is_silence: every sample under 1e-6.
func silent(wav []float32) bool {
	for _, v := range wav {
		if math.Abs(float64(v)) >= 1e-6 {
			return false
		}
	}
	return true
}

// ReadAudio decodes any file ffmpeg reads into interleaved 48 kHz stereo,
// clamped to [-1, 1] (_normalize_audio_to_stereo_48k). Two departures from
// upstream's soundfile/torchaudio: ffmpeg resamples (swr, not torchaudio's
// sinc), and a file of more than two channels is downmixed rather than cut
// to its first two. The path is read as a local file only (ffmpeg's file:
// protocol), never as a URL or another protocol's name.
func ReadAudio(path string) ([]float32, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(FFmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-protocol_whitelist", "file",
		"-i", "file:"+abs,
		"-f", "f32le", "-ac", "2", "-ar", fmt.Sprint(plan.SampleRate), "pipe:1")
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("ace: %s: %w", FFmpeg, err)
	}
	r := bufio.NewReaderSize(out, 1<<20)
	var wav []float32
	var b [4]byte
	for {
		if _, err := io.ReadFull(r, b[:]); err != nil {
			break
		}
		v := math.Float32frombits(binary.LittleEndian.Uint32(b[:]))
		wav = append(wav, max(-1, min(1, v)))
	}
	if err := cmd.Wait(); err != nil {
		// ffmpeg's last line says why; the rest is its internals.
		lines := strings.Split(strings.TrimSpace(stderr.String()), "\n")
		return nil, fmt.Errorf("ace: reading %s: %s", path, strings.TrimSpace(lines[len(lines)-1]))
	}
	if len(wav) < 2 {
		return nil, fmt.Errorf("ace: %s holds no audio", path)
	}
	return wav[:len(wav)/2*2], nil
}

// ReadAudioBytes is ReadAudio over a file's contents (an upload), through a
// temporary file: some containers (mp4, m4a) need to seek.
func ReadAudioBytes(data []byte) ([]float32, error) {
	f, err := os.CreateTemp("", "ace-audio-*")
	if err != nil {
		return nil, err
	}
	defer os.Remove(f.Name())
	_, err = f.Write(data)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return nil, err
	}
	wav, err := ReadAudio(f.Name())
	if err != nil {
		return nil, errors.New(strings.ReplaceAll(err.Error(), "reading "+f.Name(), "the upload is not audio ffmpeg reads"))
	}
	return wav, nil
}

// ReferenceClip is process_reference_audio after the load: the audio
// repeated to at least 30 s, then 10 s from a random offset in each of its
// thirds, concatenated. offsets, when three, are the draws (the gates feed
// the oracle's); otherwise rng draws them as random.randint does, both ends
// inclusive.
func ReferenceClip(wav []float32, offsets []int, rng *rand.Rand) []float32 {
	const target, seg = 30 * plan.SampleRate, 10 * plan.SampleRate
	n := len(wav) / 2
	if n < target {
		reps := (target + n - 1) / n
		w := make([]float32, 0, reps*len(wav))
		for i := 0; i < reps; i++ {
			w = append(w, wav...)
		}
		wav, n = w, reps*n
	}
	third := n / 3
	randint := func(i, hi int) int {
		if len(offsets) == 3 {
			return offsets[i]
		}
		return rng.IntN(hi + 1)
	}
	starts := [3]int{
		randint(0, max(0, third-seg)),
		third + randint(1, max(0, third-seg)),
		2*third + randint(2, max(0, (n-2*third)-seg)),
	}
	out := make([]float32, 0, 3*seg*2)
	for _, s := range starts {
		out = append(out, wav[2*s:2*min(s+seg, n)]...)
	}
	return out
}

// encode is _encode_audio_to_latents: the VAE posterior, sampled with
// seeded noise as upstream samples it (unseeded, there).
func (p *Pipeline) encode(wav []float32, seed uint64) ([]float32, error) {
	mean, scale, _, err := p.Enc.Encode(wav)
	if err != nil {
		return nil, err
	}
	noise, err := h3vae.TorchRandn(seed, len(mean))
	if err != nil {
		return nil, err
	}
	return vae.Sample(mean, scale, noise), nil
}

// timbre is the timbre encoder's input: the reference's latents, or the
// silence's first 750 when there is none.
func (p *Pipeline) timbre(a *Audio, seed uint64) ([]float32, error) {
	if a == nil || (a.Reference == nil && a.ReferenceLatents == nil) {
		return p.silence[:plan.TimbreLatents*plan.LatentChannels], nil
	}
	if a.ReferenceLatents != nil {
		return a.ReferenceLatents, nil
	}
	clip := ReferenceClip(a.Reference, a.RefOffsets, rand.New(rand.NewPCG(seed, 0x7e)))
	return p.encode(clip, seed^0x7e)
}

// ditPlan is what the DiT and the VAE run on, past the request: the
// context, the timbre, and the task's changes to the sampling.
type ditPlan struct {
	T      int
	src    []float32 // [T, 64] the context's source rows
	mask   []float32 // [T] chunk mask; nil is 1.0 everywhere
	timbre []float32 // [750, 64]
	// hidden is the source before any hints replace it (upstream's
	// src_latents argument): cover noise renoises it.
	hidden []float32
	// alt is the text2music request the cover strength switches to after
	// switchFrac of the steps (audio_cover_strength), over silence.
	alt        *plan.Request
	switchFrac float64
	coverNoise float64
	rp         *repaintPlan
}

// repaintPlan is a repaint's span and its three blends.
type repaintPlan struct {
	span       []bool    // [T] inside the span
	clean      []float32 // [T, 64] the source's latents
	ratio      float64   // injection: the first round(ratio·steps) steps
	cfFrames   int       // the latent crossfade
	wavCF      float64   // the waveform crossfade, seconds
	splice     bool      // not aggressive
	start, end float64   // the span in seconds of the padded source
	padded     []float32 // the padded source waveform
}

// repaintConfig is _resolve_repaint_config: injection ratio, latent and
// waveform crossfades by mode.
func repaintConfig(mode string, strength float64) (ratio float64, cf int, wavCF float64) {
	strength = max(0, min(1, strength))
	switch mode {
	case "aggressive":
		return 0, 0, 0
	case "conservative":
		return 1, 25, 0.05
	}
	inv := 1 - strength
	return inv, int(math.RoundToEven(25 * inv)), 0.05 * inv
}

// padSilence extends rows [n, 64] to T rows with the silence's first rows
// (_get_silence_latent_slice), as upstream pads a short target.
func (p *Pipeline) padSilence(rows []float32, T int) []float32 {
	out := make([]float32, T*plan.LatentChannels)
	n := copy(out, rows)
	copy(out[n:], p.silence)
	return out
}

// taskPlan builds the DiT's inputs for an audio-in task, and the request
// its caption prompt is built from (the task's instruction; the source's
// length as the duration).
func (p *Pipeline) taskPlan(r *plan.Request, a *Audio, seed uint64) (*plan.Request, *ditPlan, error) {
	d := *r
	src := a.Source
	var left, right int // repaint's padding, in frames
	var rs, re float64  // repaint's span in the padded source
	switch a.Task {
	case Cover, CoverNoFSQ:
		d.Instruction = lm.Instruction
	case Repaint:
		d.Instruction = RepaintInstruction
		dur := float64(len(src)/2) / plan.SampleRate
		end := a.RepaintEnd
		if end <= 0 {
			end = dur
		}
		lp := max(0, -a.RepaintStart)
		rp := max(0, end-dur)
		left, right = int(lp*plan.SampleRate), int(rp*plan.SampleRate)
		rs, re = a.RepaintStart+lp, end+lp
	}
	if src != nil {
		d.Duration = float64(len(src)/2) / plan.SampleRate
	}
	padded := src
	if left > 0 || right > 0 {
		padded = make([]float32, 2*(left+len(src)/2+right))
		copy(padded[2*left:], src)
	}
	var lat []float32
	var err error
	switch {
	case a.SourceLatents != nil:
		lat = a.SourceLatents
	case silent(padded):
		lat = p.silence[:len(padded)/2/plan.Hop*plan.LatentChannels]
	default:
		if lat, err = p.encode(padded, seed^0x5c); err != nil {
			return nil, nil, err
		}
	}
	n := len(lat) / plan.LatentChannels
	T := max(plan.MinLatents, n)
	if T*plan.LatentChannels > len(p.silence) || (T+1)/2 > p.DiT.MaxTokens() {
		return nil, nil, fmt.Errorf("ace: a %d-latent source is past this pipeline's ceiling", n)
	}
	target := p.padSilence(lat, T)
	timbre, err := p.timbre(a, seed)
	if err != nil {
		return nil, nil, err
	}
	dp := &ditPlan{T: T, timbre: timbre, coverNoise: a.CoverNoise}
	hasAudio := a.SourceLatents != nil || !silent(padded)

	// _build_chunk_masks_and_src_latents.
	dp.mask = make([]float32, T)
	for i := range dp.mask {
		dp.mask[i] = 1
	}
	isCover := a.Task == Cover
	if a.Task == Repaint && re > rs {
		s := max(0, min(int(math.Floor(rs*plan.SampleRate/plan.Hop)), T-1))
		e := max(s+1, min(int(math.Floor(re*plan.SampleRate/plan.Hop)), T))
		ratio, cf, wcf := repaintConfig(a.RepaintMode, a.RepaintStrength)
		rp := &repaintPlan{span: make([]bool, T), clean: target, ratio: ratio, cfFrames: cf, wavCF: wcf,
			splice: a.RepaintMode != "aggressive", start: rs, end: re, padded: padded}
		for i := s; i < e; i++ {
			rp.span[i] = true
		}
		if a.ExplicitMask {
			for i := range dp.mask {
				if !rp.span[i] {
					dp.mask[i] = 0
				}
			}
		}
		dp.src = append([]float32(nil), target...)
		if hasAudio {
			copy(dp.src[s*plan.LatentChannels:e*plan.LatentChannels], p.silence[s*plan.LatentChannels:e*plan.LatentChannels])
		} else {
			dp.src = p.silence[:T*plan.LatentChannels]
		}
		dp.rp = rp
	} else {
		dp.src = target
		if !hasAudio {
			dp.src = p.silence[:T*plan.LatentChannels]
		}
	}
	dp.hidden = dp.src
	if isCover {
		if dp.src, err = p.coverHints(dp.src, T); err != nil {
			return nil, nil, err
		}
	}
	if a.CoverStrength < 1 {
		alt := d
		alt.Instruction = plan.DefaultInstruction
		dp.alt, dp.switchFrac = &alt, a.CoverStrength
	}
	return &d, dp, nil
}

// coverHints is prepare_condition's is_covers path: the source through the
// audio tokenizer (padded to whole 5-latent groups with the silence's first
// rows), the codes back through the detokenizer, cut to T.
func (p *Pipeline) coverHints(src []float32, T int) ([]float32, error) {
	const P = 5
	n := (T + P - 1) / P * P
	x := make([]float32, n*plan.LatentChannels)
	k := copy(x, src[:T*plan.LatentChannels])
	copy(x[k:], p.silence)
	codes, _, _, err := p.DiT.Tokenize(&qwen.Mat{Rows: n, Cols: plan.LatentChannels, Data: x})
	if err != nil {
		return nil, err
	}
	h, err := p.DiT.Detokenize(codes)
	if err != nil {
		return nil, err
	}
	return h.Data[:T*plan.LatentChannels], nil
}

// linspaceInner is torch.linspace(a, b, n+2)[1:-1] in fp32.
func linspaceInner(a, b float32, n int) []float32 {
	steps := n + 2
	step := (b - a) / float32(steps-1)
	out := make([]float32, n)
	for i := 1; i <= n; i++ {
		if i < steps/2 {
			out[i-1] = a + step*float32(i)
		} else {
			out[i-1] = b - step*float32(steps-1-i)
		}
	}
	return out
}

// blendEdges is _repaint_boundary_blend: the generated latents inside the
// span, the source outside, ramped over cf frames each side.
func (rp *repaintPlan) blendEdges(x []float32) {
	T := len(rp.span)
	soft := make([]float32, T)
	left, right := -1, -1
	for i, in := range rp.span {
		if in {
			soft[i] = 1
			if left < 0 {
				left = i
			}
			right = i + 1
		}
	}
	if left >= 0 && !(left == 0 && right == T) && rp.cfFrames > 0 {
		if fs := max(left-rp.cfFrames, 0); left-fs > 0 {
			copy(soft[fs:left], linspaceInner(0, 1, left-fs))
		}
		if fe := min(right+rp.cfFrames, T); fe-right > 0 {
			copy(soft[right:fe], linspaceInner(1, 0, fe-right))
		}
	}
	C := plan.LatentChannels
	for i := 0; i < T; i++ {
		m := soft[i]
		for c := 0; c < C; c++ {
			j := i*C + c
			x[j] = float32(m*x[j]) + float32((1-m)*rp.clean[j])
		}
	}
}

// spliceWave is apply_repaint_waveform_splice: the source waveform outside
// the span, the decode inside, a linear crossfade of wavCF seconds at each
// edge.
func (rp *repaintPlan) spliceWave(wav []float32) {
	n := min(len(wav), len(rp.padded)) / 2
	s := max(0, min(int(rp.start*plan.SampleRate), n))
	e := max(s, min(int(rp.end*plan.SampleRate), n))
	if s == 0 && e >= n {
		return
	}
	mask := make([]float32, n)
	for i := s; i < e; i++ {
		mask[i] = 1
	}
	if cf := int(rp.wavCF * plan.SampleRate); cf > 0 {
		if fs := max(s-cf, 0); s-fs > 0 {
			copy(mask[fs:s], linspaceInner(0, 1, s-fs))
		}
		if fe := min(e+cf, n); fe-e > 0 {
			copy(mask[e:fe], linspaceInner(1, 0, fe-e))
		}
	}
	for i := 0; i < n; i++ {
		m := mask[i]
		for c := 0; c < 2; c++ {
			j := 2*i + c
			wav[j] = float32(m*wav[j]) + float32((1-m)*rp.padded[j])
		}
	}
}
