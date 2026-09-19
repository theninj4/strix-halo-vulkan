package kokoro

import (
	"fmt"
	"math"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// GPUSource is the neural source filter's excitation on the device: the sine
// bank that turns an F0 curve into a waveform, and the short-time transform
// that turns that waveform into the spectrogram the noise convolutions read
// (SPEECH.md T7).
//
// It is one object rather than two because the waveform between them is the
// only thing either kernel shares and there is no reason for it to cross the
// bus: `source` is written into the arena by the first dispatch and read out
// of it by the second, and what comes back is the [frames, 2*bins] harmonic
// spectrogram the generator actually consumes.
//
// **Its arena is HOST_CACHED.** The spectrogram is 1.4 MB and the rest of
// kokoro's arenas come from the write-combined type NewBuffer prefers, which
// this host reads at 0.18 GB/s — 7.7 ms for this download, which is the whole
// stage. From a cached type it is 25 GB/s and the readback stops being a
// term. See vk.NewHostCachedBuffer and LLM.md L6b.
type GPUSource struct {
	dev *vk.Device
	h   *HarmonicSource

	dim, scale      int // harmonics + 1, and the 300:1 upsampling
	nfft, hop, bins int
	maxF0           int // the longest F0 curve the arenas were built for

	wbuf, abuf *vk.Buffer
	mods       []*vk.ShaderModule
	pipes      map[string]*vk.ComputePipeline

	// Weight-arena offsets, fp32. base/delta/f0 are rewritten per utterance;
	// mix and tw depend on nothing but the model.
	wBase, wDelta, wF0, wMix, wTw uint32
	wElems                        int

	// Activation-arena offsets, fp32.
	aSource, aHar uint32
	aElems        int

	// The utterance currently staged.
	f0Frames, samples, frames int
}

// NewGPUSource builds the excitation for any F0 curve up to maxF0 frames.
//
// The arenas are sized by that ceiling rather than by the utterance, because
// everything here is geometry: nothing in either kernel depends on the length
// except the dispatch extent and three push constants, so one object serves
// every utterance a server sees. That is the opposite of the generator's
// stages, which stage per request — see the package comment.
func NewGPUSource(dev *vk.Device, g *Generator, maxF0 int) (*GPUSource, error) {
	if g.Source == nil || g.STFT == nil {
		return nil, fmt.Errorf("kokoro: the generator has no source module or no transform")
	}
	if maxF0 <= 0 {
		return nil, fmt.Errorf("kokoro: maxF0 %d", maxF0)
	}
	h := g.Source
	s := &GPUSource{
		dev: dev, h: h,
		dim: h.Harmonics + 1, scale: h.UpsampleScale,
		nfft: g.STFT.NFFT, hop: g.STFT.Hop, bins: g.STFT.Bins(),
		maxF0: maxF0,
		pipes: map[string]*vk.ComputePipeline{},
	}
	if s.h.Mix == nil || s.h.Mix.In != s.dim || s.h.Mix.Out != 1 {
		return nil, fmt.Errorf("kokoro: the excitation's mixer is not %d -> 1", s.dim)
	}
	if !g.STFT.Center {
		return nil, fmt.Errorf("kokoro: the excitation's transform is not centred")
	}
	s.alloc()
	if err := s.build(); err != nil {
		s.Destroy()
		return nil, err
	}
	if err := s.stage(g); err != nil {
		s.Destroy()
		return nil, err
	}
	return s, nil
}

// alloc lays both arenas out. Everything is fp32 and nothing is shared with
// the generator's arenas: this runs before the first stage is staged, and the
// two never overlap in time.
func (s *GPUSource) alloc() {
	maxSamples := s.maxF0 * s.scale
	maxFrames := 1 + (maxSamples+2*(s.nfft/2)-s.nfft)/s.hop

	var off uint32
	take := func(n int) uint32 { o := off; off += uint32(n); return o }
	s.wBase = take(s.maxF0 * s.dim)
	s.wDelta = take(s.maxF0 * s.dim)
	s.wF0 = take(s.maxF0)
	s.wMix = take(s.dim + 1)
	s.wTw = take(2*s.nfft*s.bins + s.nfft)
	s.wElems = int(off)

	off = 0
	s.aSource = take(maxSamples)
	s.aHar = take(maxFrames * 2 * s.bins)
	s.aElems = int(off)
}

func (s *GPUSource) build() error {
	var err error
	if s.wbuf, err = s.dev.NewBuffer(s.wElems * 4); err != nil {
		return fmt.Errorf("kokoro: excitation weight arena: %w", err)
	}
	// Read back, so cached rather than write-combined -- see the type comment.
	if s.abuf, err = s.dev.NewHostCachedBuffer(s.aElems * 4); err != nil {
		return fmt.Errorf("kokoro: excitation fp32 arena: %w", err)
	}
	spec := vk.PipelineSpec{
		Buffers:          []*vk.Buffer{s.wbuf, s.abuf},
		PushConstantSize: uint32(unsafe.Sizeof(pushConstants{})),
	}
	for _, n := range []struct {
		name  string
		spirv []byte
	}{{"source", shaders.KokoroSource}, {"stft", shaders.KokoroSrcSTFT}} {
		mod, err := s.dev.NewShaderModule(n.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", n.name, err)
		}
		s.mods = append(s.mods, mod)
		if s.pipes[n.name], err = s.dev.NewPipeline(mod, spec); err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", n.name, err)
		}
	}
	return nil
}

// stage writes the two tables that depend on nothing but the model: the
// mixer, and the transform's twiddles and window.
func (s *GPUSource) stage(g *Generator) error {
	mix := make([]float32, s.dim+1)
	copy(mix, s.h.Mix.Weight)
	if s.h.Mix.Bias != nil {
		mix[s.dim] = s.h.Mix.Bias[0]
	}
	s.wbuf.WriteFloat32At(int(s.wMix), mix)

	// [n_fft, bins] of cosine and of sine, then the window. The same table
	// kokoro_istft.comp reads, minus the squared window an inverse needs.
	win := g.STFT.Window()
	tw := make([]float32, 2*s.nfft*s.bins+s.nfft)
	for n := 0; n < s.nfft; n++ {
		for b := 0; b < s.bins; b++ {
			th := 2 * math.Pi * float64(b*n) / float64(s.nfft)
			tw[n*s.bins+b] = float32(math.Cos(th))
			tw[s.nfft*s.bins+n*s.bins+b] = float32(math.Sin(th))
		}
		tw[2*s.nfft*s.bins+n] = float32(win[n])
	}
	s.wbuf.WriteFloat32At(int(s.wTw), tw)
	return nil
}

// Upload writes the utterance's phase table.
//
// **This is the whole of T7's precision argument, and it is 20 us of float64
// on the host.** The reference upsamples the F0 curve 300:1, takes the
// fractional part of the cycles per sample, and decimates it back down -- and
// that round trip is the identity, because the decimation reads coordinate
// 300t + 149.5 and nearest upsampling made both samples it averages equal.
// So the low-rate value is frac(f0[t]*(d+1)/sr), there are 260*9 of them, and
// integrating them in float64 costs nothing measurable (the host profile puts
// the cumulative sum at 8 us against the sine bank's 4.2 ms).
//
// What goes to the device is that integral **wrapped into [0, 1)** together
// with its forward difference, so the largest number any thread evaluates is
// one frame's worth of phase rather than the utterance's. The difference is
// 300*frac(...) and so at most 300, and zero on the last frame -- which is
// what makes the interpolation's clamped right edge need no special case in
// the shader.
func (s *GPUSource) Upload(f0 []float32) error {
	if len(f0) == 0 || len(f0) > s.maxF0 {
		return fmt.Errorf("kokoro: %d F0 frames, the excitation was built for 1..%d", len(f0), s.maxF0)
	}
	t := len(f0)
	s.f0Frames = t
	s.samples = t * s.scale
	s.frames = 1 + (s.samples+2*(s.nfft/2)-s.nfft)/s.hop

	phase := make([]float64, t*s.dim)
	acc := make([]float64, s.dim)
	for i := 0; i < t; i++ {
		for d := 0; d < s.dim; d++ {
			x := float64(f0[i]) * float64(d+1) / float64(s.h.SampleRate)
			acc[d] += x - math.Floor(x)
			phase[i*s.dim+d] = acc[d] * float64(s.scale)
		}
	}
	base := make([]float32, t*s.dim)
	delta := make([]float32, t*s.dim)
	for i := 0; i < t; i++ {
		for d := 0; d < s.dim; d++ {
			p := phase[i*s.dim+d]
			base[i*s.dim+d] = float32(p - math.Floor(p))
			if i+1 < t {
				delta[i*s.dim+d] = float32(phase[(i+1)*s.dim+d] - p)
			}
		}
	}
	s.wbuf.WriteFloat32At(int(s.wBase), base)
	s.wbuf.WriteFloat32At(int(s.wDelta), delta)
	s.wbuf.WriteFloat32At(int(s.wF0), f0)
	return nil
}

// UploadSource writes a waveform straight into the arena, so that the
// transform can be run over an excitation this object did not synthesise.
//
// It is the device's ApplyWithSource, and it exists for the same reason that
// one does: the excitation and its transform are the two halves of this stage
// and their errors are of completely different kinds -- the sine bank's is
// rounding, the transform's is an angle that is undefined wherever the
// magnitude is zero -- so a test that cannot separate them is measuring their
// sum and calling it either one.
func (s *GPUSource) UploadSource(x []float32) error {
	if len(x) == 0 || len(x) > s.maxF0*s.scale {
		return fmt.Errorf("kokoro: %d samples, the excitation was built for 1..%d",
			len(x), s.maxF0*s.scale)
	}
	s.samples = len(x)
	s.frames = 1 + (s.samples+2*(s.nfft/2)-s.nfft)/s.hop
	s.f0Frames = (s.samples + s.scale - 1) / s.scale
	s.abuf.WriteFloat32At(int(s.aSource), x)
	return nil
}

// RunSTFT runs the transform alone, over whatever waveform is in the arena.
func (s *GPUSource) RunSTFT() error {
	if s.samples == 0 {
		return fmt.Errorf("kokoro: the excitation has no waveform staged")
	}
	d, _ := s.graph()
	_, err := vk.DispatchMultiTimed(d[1:], 1, 1, true)
	return err
}

// graph is the two dispatches, in order. They are separated by a barrier
// because the transform reads every sample the sine bank wrote.
func (s *GPUSource) graph() ([]vk.MultiDispatch, []string) {
	src := pushConstants{
		WOff: s.wBase, KOff: s.wDelta, VOff: s.wF0, BOff: s.wMix, OutOff: s.aSource,
		Tokens: uint32(s.samples), Dim: uint32(s.dim),
		Aux0: uint32(s.scale), Aux1: uint32(s.f0Frames),
		Eps:   math.Float32bits(s.h.SineAmp),
		Scale: math.Float32bits(s.h.VoicedThreshold),
		Span:  math.Float32bits(s.h.NoiseStd),
	}
	if s.h.Noise != nil {
		// A zero seed with the noise on would read as off, and the flag and
		// the seed are the same field -- so it becomes one. Any constant
		// would do; this is just not zero.
		if src.Heads = uint32(s.h.Seed); src.Heads == 0 {
			src.Heads = 0x5bf03635
		}
	}
	stft := pushConstants{
		InOff: s.aSource, OutOff: s.aHar, WOff: s.wTw,
		Tokens: uint32(s.frames * s.bins), Dim: uint32(s.bins),
		Aux0: uint32(s.nfft), Aux1: uint32(s.hop), Aux2: uint32(s.samples),
	}
	return []vk.MultiDispatch{
		{Pipeline: s.pipes["source"], GroupsX: groups(s.samples, 256), GroupsY: 1,
			PushConstants: src.bytes()},
		{Pipeline: s.pipes["stft"], GroupsX: groups(s.frames*s.bins, 256), GroupsY: 1,
			PushConstants: stft.bytes()},
	}, []string{"source", "stft"}
}

// Run executes both dispatches over the uploaded curve.
func (s *GPUSource) Run() error {
	if s.f0Frames == 0 {
		return fmt.Errorf("kokoro: the excitation has no F0 curve staged")
	}
	d, _ := s.graph()
	_, err := vk.DispatchMultiTimed(d, 1, 1, true)
	return err
}

// Source reads the excitation waveform back. The generator does not need it
// -- only the spectrogram crosses into the network -- so it is here for the
// trace and for the tests that check this port against the host's.
func (s *GPUSource) Source() []float32 {
	return s.abuf.ReadFloat32At(int(s.aSource), s.samples)
}

// Harmonic reads the spectrogram back, in the layout Generator.Harmonic
// builds: magnitude in the first bins channels of a row, phase in the rest.
func (s *GPUSource) Harmonic() *Mat {
	return &Mat{Rows: s.frames, Cols: 2 * s.bins,
		Data: s.abuf.ReadFloat32At(int(s.aHar), s.frames*2*s.bins)}
}

// Apply is the whole stage: upload, two dispatches, one download.
func (s *GPUSource) Apply(f0 []float32) (source []float32, har *Mat, err error) {
	if err = s.Upload(f0); err != nil {
		return nil, nil, err
	}
	if err = s.Run(); err != nil {
		return nil, nil, err
	}
	return s.Source(), s.Harmonic(), nil
}

// Profile runs the pair with a timestamp after each dispatch.
func (s *GPUSource) Profile() ([]Stage, error) {
	d, names := s.graph()
	_, each, err := vk.DispatchMultiMarked(d, true)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, len(each))
	for i := range each {
		out[i] = Stage{Kind: names[i], Time: each[i].Seconds()}
	}
	return out, nil
}

// Destroy releases the arenas and the pipelines.
func (s *GPUSource) Destroy() {
	for _, p := range s.pipes {
		p.Destroy()
	}
	for _, m := range s.mods {
		m.Destroy()
	}
	if s.wbuf != nil {
		s.wbuf.Destroy()
	}
	if s.abuf != nil {
		s.abuf.Destroy()
	}
	s.pipes, s.mods, s.wbuf, s.abuf = nil, nil, nil, nil
}
