package kokoro

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"strix-halo-vulkan/audio"
)

// Snake is the generator's activation, `x + sin(ax)^2 / a`, with a learned
// per-channel `a`.
//
// It is periodic and unbounded above, which is the point: a ReLU family
// activation has no way to make a harmonic series out of a smooth input, and
// the vocoder's whole job is to turn an F0 curve into one. `a` controls the
// frequency of the ripple, so each channel learns a periodicity.
func Snake(x, alpha float32) float32 {
	s := float32(math.Sin(float64(alpha * x)))
	return x + s*s/alpha
}

func snakeInPlace(x *Mat, alpha []float32) {
	parallelFor(x.Rows, func(r int) {
		row := x.Row(r)
		for c, v := range row {
			row[c] = Snake(v, alpha[c])
		}
	})
}

// SnakeResBlock is the generator's residual block — upstream's
// `AdaINResBlock1`, which despite the name is a different animal from the
// predictor's `AdainResBlk1d`: three serial (normalise, snake, convolve)
// pairs, each adding into the running value, with no 1/sqrt(2) and no
// shortcut convolution.
//
// The first convolution of each pair is dilated by 1, 3 and 5 and the second
// is not, so one block sees 1 + 2*(k-1)*(1+3+5) samples of context — 37 taps
// at k=3 and 181 at k=11. The generator runs three of these in parallel per
// upsampling stage and averages them, which is HiFi-GAN's multi-receptive
// field fusion.
type SnakeResBlock struct {
	Channels int

	Convs1, Convs2 []*Conv1D
	Norm1, Norm2   []*AdaIN1d
	Alpha1, Alpha2 [][]float32
}

// Apply runs the block over a [T, Channels] activation.
func (b *SnakeResBlock) Apply(x *Mat, style []float32) (*Mat, error) {
	if x.Cols != b.Channels {
		return nil, fmt.Errorf("kokoro: resblock over %d channels, got %d", b.Channels, x.Cols)
	}
	for i := range b.Convs1 {
		xt := x.Clone()
		if err := b.Norm1[i].Apply(xt, style); err != nil {
			return nil, err
		}
		snakeInPlace(xt, b.Alpha1[i])
		xt, err := b.Convs1[i].Apply(xt)
		if err != nil {
			return nil, err
		}
		if err := b.Norm2[i].Apply(xt, style); err != nil {
			return nil, err
		}
		snakeInPlace(xt, b.Alpha2[i])
		if xt, err = b.Convs2[i].Apply(xt); err != nil {
			return nil, err
		}
		xt.AddInPlace(x)
		x = xt
	}
	return x, nil
}

// HarmonicSource is the neural source filter's excitation: an F0 curve turned
// into nine sinusoids and mixed down to one signal.
//
// This is the only part of the model that is signal processing rather than
// linear algebra, and it is the only part that is *stochastic* upstream —
// `SineGen` draws each harmonic's initial phase from a uniform and adds
// Gaussian noise. Noise here is optional and off by default, because the
// reference dump zeroes it to be reproducible (see T1 in SPEECH.md): with it
// off the excitation is deterministic and comparable, and with it on the
// output is 13.8 dB better conditioned and matches nothing exactly.
//
// The phase arithmetic is a round trip that looks redundant and is not.
// The F0 curve is upsampled to the sample rate, converted to a fraction of a
// cycle per sample, *decimated back down* by the same factor, integrated, and
// re-interpolated. Integrating at the low rate and interpolating the result is
// what keeps the phase continuous across a frame boundary; integrating at the
// sample rate would accumulate the interpolation's own error 300 times faster.
type HarmonicSource struct {
	SampleRate      int
	Harmonics       int     // 8 overtones, so 9 sinusoids
	SineAmp         float32 // 0.1
	NoiseStd        float32 // 0.003, in voiced regions only
	VoicedThreshold float32 // 10 Hz
	UpsampleScale   int     // 300 = 10 * 6 * 5
	Mix             *Linear // 9 -> 1

	// Seed is what Noise was seeded with, so that the device path can draw
	// the same distribution from its own counter-based generator. It is not
	// the same *sequence*: see kokoro_source.comp.
	Seed int64

	// Noise switches the excitation noise on. Nil is the reference dump's
	// configuration and the only reproducible one; a real utterance wants it,
	// because the noise is 13.8 dB of what the model was trained to hear.
	//
	// Upstream also draws a random initial phase per harmonic. It is dead
	// code: the phase is decimated 300:1 with a half-sample offset, so output
	// frame 0 reads input samples 149 and 150 and the sample the initial
	// phase is added to is never read. Verified against the reference —
	// changing it moves nothing — so it is not reproduced here.
	Noise *rand.Rand
}

// Apply turns an F0 curve at the alignment rate into an excitation at the
// sample rate, returning it along with the upsampled F0, the voiced mask and
// the individual sinusoids — all four of which the reference dumps.
//
// The arithmetic is float64 and the reference's is float32, which is the one
// place in this package where the port is deliberately *more* accurate than
// the thing it is checked against. Upstream integrates the phase in radians
// and then multiplies by 300, so by the end of a three-second utterance the
// accumulator holds 1.3e5 radians, where one float32 ulp is **0.016 radians**
// — a 1% error in the sine, and the reason the excitation here agrees with
// the dump to about a percent and no better. See TestHarmonicSource, which
// bounds the port's own contribution by substituting the reference's
// excitation and re-running the vocoder.
//
// Two consequences. The phase is kept in *cycles* and wrapped to [0, 1)
// before the sine, so nothing here is ever large. And this quantity must
// never be narrowed: fp16 cannot even represent 1.3e5, and at 6e4 its ulp is
// 64 radians, so a naive port of the upstream ordering to half precision
// produces noise rather than a harmonic series.
func (h *HarmonicSource) Apply(f0 []float32) (source, f0Up, uv []float32, sines *Mat) {
	n := len(f0) * h.UpsampleScale
	f0Up = make([]float32, n)
	uv = make([]float32, n)
	for i := range f0Up {
		f0Up[i] = f0[i/h.UpsampleScale] // nn.Upsample, nearest
		if f0Up[i] > h.VoicedThreshold {
			uv[i] = 1
		}
	}

	// Cycles per sample, per harmonic, wrapped into [0, 1). Python's % is the
	// floored modulus, not C's truncated one, so a negative F0 — which a
	// convolution can produce — wraps up rather than down.
	dim := h.Harmonics + 1
	rad := make([]float64, n*dim)
	for i, v := range f0Up {
		for d := 0; d < dim; d++ {
			x := float64(v) * float64(d+1) / float64(h.SampleRate)
			rad[i*dim+d] = x - math.Floor(x)
		}
	}

	// Decimated back to the alignment rate, integrated there, and
	// re-interpolated: integrating at the sample rate would accumulate the
	// interpolation's own error 300 times faster.
	low := interpolateLinear(rad, dim, len(f0))
	phase := make([]float64, len(f0)*dim)
	for d := 0; d < dim; d++ {
		var acc float64
		for t := 0; t < len(f0); t++ {
			acc += low[t*dim+d]
			phase[t*dim+d] = acc * float64(h.UpsampleScale)
		}
	}
	full := interpolateLinear(phase, dim, n)

	sines = NewMat(n, dim)
	for i, p := range full {
		sines.Data[i] = float32(math.Sin(2*math.Pi*(p-math.Floor(p)))) * h.SineAmp * uv[i/dim]
	}
	if h.Noise != nil {
		// Loud where the signal is unvoiced and quiet where it is not: the
		// sinusoids carry nothing in an unvoiced region, so the noise is what
		// the vocoder has to make a fricative out of.
		for i := range sines.Data {
			amp := h.SineAmp / 3
			if uv[i/dim] != 0 {
				amp = h.NoiseStd
			}
			sines.Data[i] += amp * float32(h.Noise.NormFloat64())
		}
	}
	source = make([]float32, n)
	row := make([]float32, 1)
	for t := 0; t < n; t++ {
		h.Mix.ApplyRow(row, sines.Row(t))
		source[t] = tanh(row[0])
	}
	return source, f0Up, uv, sines
}

// interpolateLinear reproduces `F.interpolate(mode="linear",
// align_corners=False)` over the time axis of a [frames, cols] float64 array.
//
// The half-sample offset is the whole content: output i reads input
// coordinate (i + 0.5)*in/out - 0.5, clamped at zero, so a 300x decimation is
// *not* taking every 300th sample — it is averaging the two around 149.5.
// Getting that wrong shifts the excitation by half a frame, which is audible
// and would not show up as a shape error anywhere.
func interpolateLinear(x []float64, cols, out int) []float64 {
	in := len(x) / cols
	dst := make([]float64, out*cols)
	scale := float64(in) / float64(out)
	for i := 0; i < out; i++ {
		src := scale*(float64(i)+0.5) - 0.5
		if src < 0 {
			src = 0
		}
		i0 := int(src)
		if i0 > in-1 {
			i0 = in - 1
		}
		i1 := i0 + 1
		if i1 > in-1 {
			i1 = in - 1
		}
		w1 := src - float64(i0)
		w0 := 1 - w1
		a, b := x[i0*cols:], x[i1*cols:]
		row := dst[i*cols:]
		for c := 0; c < cols; c++ {
			row[c] = w0*a[c] + w1*b[c]
		}
	}
	return dst
}

// Generator is the iSTFTNet vocoder proper: two transposed-convolution
// upsampling stages, each followed by three parallel residual blocks that are
// averaged, then a projection to a magnitude and a phase spectrogram and one
// inverse STFT.
//
// "iSTFT" rather than a third upsampling stage is the whole idea: HiFi-GAN
// would take the last 60 samples-per-frame to 600 with more transposed
// convolutions, and this replaces them with a 20-point inverse transform at a
// hop of 5. The last factor of 5 costs a DFT instead of a convolution stack.
type Generator struct {
	NumKernels int

	Source     *HarmonicSource
	NoiseConvs []*Conv1D
	NoiseRes   []*SnakeResBlock
	Ups        []*ConvTranspose1D
	ResBlocks  []*SnakeResBlock
	ConvPost   *Conv1D

	STFT  *audio.STFT
	ISTFT *audio.ISTFT
	NFFT  int

	// GPU, when set, holds one device object per upsampling stage, each
	// carrying that stage's three residual blocks and its noise block. It is
	// nil by default: the CPU path is the reference and stays the reference.
	GPU []*GPUBlocks

	// SrcGPU is the excitation on the device (SPEECH.md T7). It is separate
	// from GPU because it is the one part of the generator whose arenas do
	// not have to be sized per utterance: nothing in either of its kernels
	// depends on the length but the dispatch extent, so one object serves
	// every clip up to the ceiling it was built for.
	SrcGPU *GPUSource
}

// GeneratorTrace is what a caller wants to look at when the waveform is
// wrong: the excitation, its spectrogram and each upsampling stage's output.
type GeneratorTrace struct {
	F0Up, UV, Source []float32
	Sines            *Mat
	Harmonic         *Mat // [frames, 2*bins], magnitude then phase
	Stages           []*Mat
	Post             *Mat
	Spec, Phase      []float64

	// Where the time went. The split is the one the port is optimised
	// against: the decoder, the two upsampling stages, and the tail — the
	// excitation, conv_post and the inverse transform.
	Excitation, Stage, Tail time.Duration
}

// Apply runs the vocoder over the decoder's [T, 512] output, conditioned on
// the style vector and driven by the F0 curve at twice the alignment rate.
func (g *Generator) Apply(x *Mat, style, f0 []float32) ([]float32, *GeneratorTrace, error) {
	t0 := time.Now()
	var source []float32
	var f0Up, uv []float32
	var sines, har *Mat
	var err error
	if g.SrcGPU != nil {
		// The device path leaves f0Up, uv and the individual sinusoids
		// unbuilt: they are 5.6 MB of intermediate that only the trace ever
		// looked at, and the two kernels never form them.
		if source, har, err = g.SrcGPU.Apply(f0); err != nil {
			return nil, &GeneratorTrace{}, err
		}
	} else {
		source, f0Up, uv, sines = g.Source.Apply(f0)
		har = g.Harmonic(source)
	}
	src := time.Since(t0)
	out, tr, err := g.applyHarmonic(x, style, har, &GeneratorTrace{
		F0Up: f0Up, UV: uv, Source: source, Sines: sines,
	})
	tr.Excitation = src
	return out, tr, err
}

// ApplyWithSource runs the vocoder on an excitation the caller supplies
// instead of the one the source module would produce.
//
// It exists for one measurement: the excitation is the part of this model
// whose reference value is ill-conditioned (see HarmonicSource.Apply), so the
// way to bound the *port's* error is to feed it the reference's excitation
// and see what is left.
func (g *Generator) ApplyWithSource(x *Mat, style, source []float32) ([]float32, *GeneratorTrace, error) {
	return g.applyHarmonic(x, style, g.Harmonic(source), &GeneratorTrace{Source: source})
}

// ApplyWithHarmonic runs the vocoder on an excitation *spectrogram* the
// caller supplies, one step further in than ApplyWithSource. The step between
// them is the one that cannot be reproduced: see Harmonic.
func (g *Generator) ApplyWithHarmonic(x *Mat, style []float32, har *Mat) ([]float32, *GeneratorTrace, error) {
	return g.applyHarmonic(x, style, har, &GeneratorTrace{})
}

// Harmonic is the excitation's spectrogram: magnitude in the first bins
// channels and phase in the rest, which is the only form the excitation
// reaches the network in.
//
// The phase half is **numerically undefined wherever the magnitude is zero**,
// and with the reference's noise switched off (see T1 in SPEECH.md) large
// stretches of it are: in an unvoiced region the excitation is a constant —
// tanh of the mixer's bias — and a constant's windowed spectrum is exactly
// zero outside the three bins a Hann window occupies. atan2 of two zeros is
// whatever the rounding says, so the reference's float32 transform and this
// float64 one disagree by up to 2*pi over a quarter of the bins, and the
// network consumes those angles as ordinary numbers.
//
// That is a property of the zero-noise oracle rather than of the model —
// trained with its noise on, those bins were never zero — and it is the floor
// on how closely any port can reproduce the dumped waveform. It is a separate
// method so that the floor can be measured rather than argued about.
func (g *Generator) Harmonic(source []float32) *Mat {
	mag, phase := g.STFT.MagnitudePhase(source)
	bins := g.STFT.Bins()
	frames := g.STFT.Frames(len(source))
	har := NewMat(frames, 2*bins)
	for t := 0; t < frames; t++ {
		row := har.Row(t)
		for b := 0; b < bins; b++ {
			row[b] = float32(mag[t*bins+b])
			row[bins+b] = float32(phase[t*bins+b])
		}
	}
	return har
}

func (g *Generator) applyHarmonic(x *Mat, style []float32, har *Mat, tr *GeneratorTrace) ([]float32, *GeneratorTrace, error) {
	tr.Harmonic = har

	var err error
	t0 := time.Now()
	for i := range g.Ups {
		dev := g.device(i)
		src, err := g.NoiseConvs[i].Apply(har)
		if err != nil {
			return nil, tr, err
		}
		if dev != nil {
			// One upload of the stage's input and one of the excitation
			// projection, then everything — the rectifier, the excitation's
			// own residual block, the upsampling, the bias, the reflection
			// pad and the three averaged blocks — happens on the device.
			if err := dev.UploadNoise(src); err != nil {
				return nil, tr, err
			}
			if !dev.resident {
				if err := dev.UploadInput(x); err != nil {
					return nil, tr, err
				}
			}
			if dev.HasTail() {
				// The last stage keeps going: conv_post, the two
				// nonlinearities and the inverse transform are on the device
				// too, so what comes back is the waveform rather than a
				// [15601, 128] activation.
				tr.Stage = time.Since(t0)
				t0 = time.Now()
				out, err := dev.RunStageWave()
				tr.Tail = time.Since(t0)
				return out, tr, err
			}
			if dev.Resident() {
				// The next stage reads this one's output in place.
				if err := dev.RunStageResident(); err != nil {
					return nil, tr, err
				}
				x = nil
				continue
			}
			if x, err = dev.RunStage(); err != nil {
				return nil, tr, err
			}
			tr.Stages = append(tr.Stages, x)
			continue
		}

		x = leakyReLU(x, 0.1)
		if src, err = g.NoiseRes[i].Apply(src, style); err != nil {
			return nil, tr, err
		}
		if x, err = g.Ups[i].Apply(x); err != nil {
			return nil, tr, err
		}
		if i == len(g.Ups)-1 {
			x = reflectPadLeft(x)
		}
		if x.Rows != src.Rows {
			return nil, tr, fmt.Errorf("kokoro: stage %d upsampled to %d frames, the excitation has %d",
				i, x.Rows, src.Rows)
		}
		x.AddInPlace(src)

		// Multi-receptive-field fusion: three filters of 3, 7 and 11 taps over
		// the same activation, averaged.
		var sum *Mat
		for j := 0; j < g.NumKernels; j++ {
			b, err := g.ResBlocks[i*g.NumKernels+j].Apply(x, style)
			if err != nil {
				return nil, tr, err
			}
			if sum == nil {
				sum = b
			} else {
				sum.AddInPlace(b)
			}
		}
		for k := range sum.Data {
			sum.Data[k] /= float32(g.NumKernels)
		}
		x = sum
		tr.Stages = append(tr.Stages, x)
	}

	tr.Stage = time.Since(t0)
	t0 = time.Now()
	defer func() { tr.Tail += time.Since(t0) }()

	x = leakyReLU(x, 0.01) // F.leaky_relu's default slope, not the 0.1 above
	post, err := g.ConvPost.Apply(x)
	if err != nil {
		return nil, tr, err
	}
	tr.Post = post

	// The head does not predict a complex spectrum. It predicts a log
	// magnitude and something it takes the sine of, so the magnitude is
	// positive by construction and the phase is in [-1, 1] rather than
	// wrapped — neither of which an unconstrained real/imaginary pair would
	// give, and both of which matter to how the iSTFT behaves.
	half := g.NFFT/2 + 1
	spec := make([]float64, post.Rows*half)
	ph := make([]float64, post.Rows*half)
	for t := 0; t < post.Rows; t++ {
		row := post.Row(t)
		for b := 0; b < half; b++ {
			spec[t*half+b] = math.Exp(float64(row[b]))
			ph[t*half+b] = math.Sin(float64(row[half+b]))
		}
	}
	tr.Spec, tr.Phase = spec, ph
	out, err := g.ISTFT.Apply(spec, ph, post.Rows)
	return out, tr, err
}

// noiseBlock is where the excitation's residual block sits in a stage's
// device set, after that stage's three generator blocks.
const noiseBlock = 3

// device returns the device object for one upsampling stage, or nil.
func (g *Generator) device(stage int) *GPUBlocks {
	if stage < len(g.GPU) {
		return g.GPU[stage]
	}
	return nil
}

// reflectPadLeft is `nn.ReflectionPad1d((1, 0))`: one frame on the left,
// mirrored, so the reflected value is frame 1 rather than frame 0.
//
// It exists to reconcile two counts that are otherwise one apart — the last
// transposed convolution produces hop*frames and the excitation's STFT
// produces hop*frames + 1 — and the model would not run at all without it.
func reflectPadLeft(x *Mat) *Mat {
	out := NewMat(x.Rows+1, x.Cols)
	src := 0
	if x.Rows > 1 {
		src = 1
	}
	copy(out.Row(0), x.Row(src))
	copy(out.Data[x.Cols:], x.Data)
	return out
}

// Vocoder is upstream's `Decoder`: the four AdaIN blocks that mix the
// expanded phonemes with the prosody, and the generator they feed.
//
// The two curves are decimated on the way in by a stride-2 convolution,
// having just been *doubled* inside the predictor's AdaIN stack. The round
// trip is not redundant — the doubling happens where it can be learned and
// the halving where it cannot — and it means the decoder runs at the
// alignment rate while the generator's F0 conditioning runs at twice it.
type Vocoder struct {
	F0Conv, NConv *Conv1D
	Encode        *AdainResBlk1d
	Decode        []*AdainResBlk1d
	ASRRes        *Conv1D
	Generator     *Generator

	// GPU, when set, runs the five AdaIN blocks on the device. It is nil by
	// default: the CPU path is the reference and stays the reference.
	GPU *GPUDecoder

	// arena is the one fp32 buffer the decoder and both generator stages
	// share, owned here because no one of them owns it.
	arena *sharedArena
}

// VocoderTrace is the decoder's intermediates, for the same reason
// GeneratorTrace exists.
type VocoderTrace struct {
	F0, N, ASRRes *Mat
	Encode        *Mat
	Decode        []*Mat
	Generator     *GeneratorTrace

	// Decoder is the five AdaIN blocks, on whichever path ran them.
	Decoder time.Duration
}

// SetFrames sizes every staged device object for one utterance.
//
// The arenas are built for a ceiling (Model.AttachGPU) and an utterance is a
// prefix of them, so this is the whole of what a new clip costs on the device:
// a walk down the same chain AttachGPU laid out — the decoder's blocks, then
// 2L frames into the generator, the upsampling rates, and the reflection pad
// on the last stage — writing a frame count into each object.
//
// It is called by Apply, so nothing outside this package has to; it is
// exported because a caller that stages once and speaks many times may want
// to check a length against the ceiling before it runs.
func (v *Vocoder) SetFrames(frames int) error {
	if v.GPU != nil {
		if err := v.GPU.SetFrames(frames); err != nil {
			return err
		}
	}
	g := v.Generator
	n := 2 * frames
	for i, gb := range g.GPU {
		in := n
		n = g.Ups[i].OutFrames(n)
		if i == len(g.Ups)-1 {
			n++ // the reflection pad
		}
		if err := gb.SetFrames(in, n); err != nil {
			return fmt.Errorf("kokoro: generator stage %d: %w", i, err)
		}
	}
	return nil
}

// Apply runs the whole vocoder: [T, 512] of expanded phonemes and two
// [2T] curves in, samples out.
func (v *Vocoder) Apply(asr *Mat, f0, energy, style []float32) ([]float32, *VocoderTrace, error) {
	tr := &VocoderTrace{}
	curve := func(c *Conv1D, in []float32) (*Mat, error) {
		return c.Apply(&Mat{Rows: len(in), Cols: 1, Data: in})
	}
	f0c, err := curve(v.F0Conv, f0)
	if err != nil {
		return nil, tr, err
	}
	nc, err := curve(v.NConv, energy)
	if err != nil {
		return nil, tr, err
	}
	if f0c.Rows != asr.Rows {
		return nil, tr, fmt.Errorf("kokoro: %d F0 frames against %d phoneme frames", f0c.Rows, asr.Rows)
	}
	tr.F0, tr.N = f0c, nc

	// Every staged object is sized for this utterance before anything runs.
	// A shorter one than the arenas were built for is the common case — see
	// SetFrames — and one the same length is free.
	if v.GPU != nil || len(v.Generator.GPU) > 0 {
		if err := v.SetFrames(asr.Rows); err != nil {
			return nil, tr, err
		}
	}

	tDec := time.Now()
	var x *Mat
	if v.GPU == nil {
		if x, err = Concat(asr, f0c, nc); err != nil {
			return nil, tr, err
		}
		if x, err = v.Encode.Apply(x, style); err != nil {
			return nil, tr, err
		}
		tr.Encode = x
	}
	asrRes, err := v.ASRRes.Apply(asr)
	if err != nil {
		return nil, tr, err
	}
	tr.ASRRes = asrRes

	if v.GPU != nil {
		// Every block, one submit. The concatenation upstream does four times
		// is a row stride on the device, so the side channels go up once and
		// nothing between `encode` and the generator crosses the bus.
		if v.GPU.Resident() {
			// The generator's first stage reads the decoder's output in
			// place, so there is nothing to bring back.
			if err = v.GPU.Run(asr, f0c, nc, asrRes); err != nil {
				return nil, tr, err
			}
			x = nil
		} else if x, err = v.GPU.Apply(asr, f0c, nc, asrRes); err != nil {
			return nil, tr, err
		}
		tr.Decoder = time.Since(tDec)
		if x != nil {
			tr.Decode = append(tr.Decode, x)
		}
		out, gtr, err := v.Generator.Apply(x, style, f0)
		tr.Generator = gtr
		return out, tr, err
	}

	for i, block := range v.Decode {
		// Every block is handed the phoneme residual and both curves again.
		// Upstream stops once a block has upsampled, because the side
		// channels are at the old rate; here the upsampling block is last, so
		// all four get them.
		if x, err = Concat(x, asrRes, f0c, nc); err != nil {
			return nil, tr, err
		}
		if x, err = block.Apply(x, style); err != nil {
			return nil, tr, fmt.Errorf("kokoro: decode block %d: %w", i, err)
		}
		tr.Decode = append(tr.Decode, x)
		if block.Upsample && i != len(v.Decode)-1 {
			return nil, tr, fmt.Errorf("kokoro: decode block %d upsamples and is not last", i)
		}
	}
	tr.Decoder = time.Since(tDec)
	out, gtr, err := v.Generator.Apply(x, style, f0)
	tr.Generator = gtr
	return out, tr, err
}
