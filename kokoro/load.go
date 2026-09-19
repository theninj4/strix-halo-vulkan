package kokoro

import (
	"fmt"
	"strings"

	"strix-halo-vulkan/audio"
	"strix-halo-vulkan/safetensors"
)

// Model is the whole of Kokoro-82M: the two phoneme encoders, the prosody
// predictor and the iSTFTNet vocoder.
type Model struct {
	Config      *Config
	BERT        *ALBERT
	BERTEncoder *Linear // 768 -> 512
	TextEncoder *TextEncoder
	Predictor   *Predictor
	Vocoder     *Vocoder

	// BERTGPU, when set, runs the twelve ALBERT layers on the device. It is
	// nil by default: the CPU path is the reference and stays the reference.
	BERTGPU *GPUAlbert

	// PhonemesGPU, when set, runs the whole phoneme side on the device: both
	// encoders, every recurrence, the duration head, the length regulator and
	// the F0/N stacks. Like BERTGPU it is nil by default, and it is sized for
	// one utterance's frame count, which is why AttachGPU takes the
	// durations' answer.
	PhonemesGPU *GPUPhonemes

	Voices              map[string][]float32 // [510*256] each, by name
	voiceRows, voiceDim int
}

// Load reads a converted checkpoint directory as float32.
//
// The directory holds two safetensors files — the model and the voices — and
// `OpenSet` globs both, so one open gives both. See
// reference/convert_kokoro.py, which is what produces them from the shipped
// pickle; a directory holding only `kokoro-v1_0.pth` will fail here with a
// missing tensor rather than anything more helpful, which is the intended
// signal to run the converter.
func Load(dir string) (*Model, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()

	l := &loader{set: set}
	m := &Model{Config: cfg}
	m.BERT = l.albert(&cfg.PLBERT)
	m.BERTEncoder = l.linear("bert_encoder")
	m.TextEncoder = l.textEncoder(cfg)
	m.Predictor = l.predictor(cfg)
	m.Vocoder = l.vocoder(cfg)
	m.loadVoices(l)
	if l.err != nil {
		return nil, l.err
	}
	return m, nil
}

// Style returns a voice's style vector for an utterance of n phonemes, split
// the way the model splits it: the first 128 channels condition the decoder
// and the last 128 the predictor.
//
// The voice is a name, or a mixture of names — see ParseBlend, and note that
// the comma form is upstream's rather than ours. A single name is resolved in
// place and copies nothing, so it is exactly the tensor it was before blends
// existed; a mixture allocates one row.
//
// The row index is the *phoneme* count minus one, not the token count — the
// pack has one row per length and KPipeline indexes it before the boundary
// tokens are added. Getting that off by one picks a neighbouring row, which
// is a voice that still sounds plausible and is wrong, so it is worth
// stating: this is the one index in the model that no downstream shape check
// would catch.
func (m *Model) Style(voice string, phonemes int) (decoder, predictor []float32, err error) {
	b, err := ParseBlend(voice)
	if err != nil {
		return nil, nil, err
	}
	row := phonemes - 1
	if row < 0 || row >= m.voiceRows {
		return nil, nil, fmt.Errorf("kokoro: %d phonemes outside the pack's %d rows", phonemes, m.voiceRows)
	}
	var s []float32
	if name, single := b.Single(); single {
		v, ok := m.Voices[name]
		if !ok {
			return nil, nil, m.unknownVoice(name, b)
		}
		s = v[row*m.voiceDim : (row+1)*m.voiceDim]
	} else if s, err = m.blendRow(b, row); err != nil {
		return nil, nil, err
	}
	half := m.voiceDim / 2
	return s[:half], s[half:], nil
}

// loader pulls named tensors as float32, holding the first error.
type loader struct {
	set *safetensors.Set
	err error
}

func (l *loader) f32(name string) []float32 {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(name)
	if err != nil {
		l.err = err
		return nil
	}
	v, err := t.F32(nil)
	if err != nil {
		l.err = err
	}
	return v
}

func (l *loader) shape(name string) []int {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(name)
	if err != nil {
		l.err = err
		return nil
	}
	return t.Shape
}

// linear reads a `prefix.weight` of shape [Out, In] and an optional
// `prefix.bias`. A trailing kernel axis of 1 is dropped: a pointwise
// convolution over a channel-last activation is a matrix.
func (l *loader) linear(prefix string) *Linear {
	sh := l.shape(prefix + ".weight")
	if l.err != nil {
		return nil
	}
	for len(sh) > 2 && sh[len(sh)-1] == 1 {
		sh = sh[:len(sh)-1]
	}
	if len(sh) != 2 {
		l.err = fmt.Errorf("kokoro: %s.weight has shape %v, want 2 dims", prefix, sh)
		return nil
	}
	lin := &Linear{Out: sh[0], In: sh[1], Weight: l.f32(prefix + ".weight")}
	if l.set.Has(prefix + ".bias") {
		lin.Bias = l.f32(prefix + ".bias")
	}
	return lin
}

func (l *loader) embedding(name string) *Embedding {
	sh := l.shape(name)
	if l.err != nil {
		return nil
	}
	if len(sh) != 2 {
		l.err = fmt.Errorf("kokoro: %s has shape %v, want 2 dims", name, sh)
		return nil
	}
	return &Embedding{Vocab: sh[0], Dim: sh[1], Weight: l.f32(name)}
}

func (l *loader) layerNorm(prefix, weightName, biasName string, width int, eps float64) *LayerNorm {
	return &LayerNorm{
		Weight: l.f32(prefix + "." + weightName),
		Bias:   l.f32(prefix + "." + biasName),
		Width:  width, Eps: eps,
	}
}

// conv1d reads a [Out, In/Groups, K] weight. Stride, padding and dilation are
// the caller's, because nothing in the checkpoint records them — they are
// properties of the Python that built the module, not of the tensor.
func (l *loader) conv1d(prefix string, stride, pad, dilation, groups int) *Conv1D {
	sh := l.shape(prefix + ".weight")
	if l.err != nil {
		return nil
	}
	if len(sh) != 3 {
		l.err = fmt.Errorf("kokoro: %s.weight has shape %v, want [Out, In/G, K]", prefix, sh)
		return nil
	}
	c := &Conv1D{
		Out: sh[0], In: sh[1] * groups, Kernel: sh[2],
		Stride: stride, Pad: pad, Dilation: dilation, Groups: groups,
		Weight: l.f32(prefix + ".weight"),
	}
	if l.set.Has(prefix + ".bias") {
		c.Bias = l.f32(prefix + ".bias")
	}
	return c
}

// convTranspose reads an [In, Out/Groups, K] weight — the axis order is the
// opposite of conv1d's, which is the single most error-prone thing in this
// checkpoint.
func (l *loader) convTranspose(prefix string, stride, pad, outPad, groups int) *ConvTranspose1D {
	sh := l.shape(prefix + ".weight")
	if l.err != nil {
		return nil
	}
	if len(sh) != 3 {
		l.err = fmt.Errorf("kokoro: %s.weight has shape %v, want [In, Out/G, K]", prefix, sh)
		return nil
	}
	c := &ConvTranspose1D{
		In: sh[0], Out: sh[1] * groups, Kernel: sh[2],
		Stride: stride, Pad: pad, OutputPadding: outPad, Groups: groups,
		Weight: l.f32(prefix + ".weight"),
	}
	if l.set.Has(prefix + ".bias") {
		c.Bias = l.f32(prefix + ".bias")
	}
	return c
}

// lstm reads one bidirectional layer, summing PyTorch's two bias vectors.
func (l *loader) lstm(prefix string) *LSTM {
	sh := l.shape(prefix + ".weight_ih_l0")
	if l.err != nil {
		return nil
	}
	if len(sh) != 2 || sh[0]%4 != 0 {
		l.err = fmt.Errorf("kokoro: %s.weight_ih_l0 has shape %v, want [4H, In]", prefix, sh)
		return nil
	}
	hidden, in := sh[0]/4, sh[1]
	dir := func(suffix string) *LSTMDirection {
		ih := l.f32(prefix + ".bias_ih_l0" + suffix)
		hh := l.f32(prefix + ".bias_hh_l0" + suffix)
		if l.err != nil {
			return nil
		}
		bias := make([]float32, len(ih))
		for i := range bias {
			bias[i] = ih[i] + hh[i]
		}
		return &LSTMDirection{
			In: in, Hidden: hidden,
			WIH:  l.f32(prefix + ".weight_ih_l0" + suffix),
			WHH:  l.f32(prefix + ".weight_hh_l0" + suffix),
			Bias: bias,
		}
	}
	return &LSTM{In: in, Hidden: hidden, Fwd: dir(""), Rev: dir("_reverse")}
}

func (l *loader) albert(cfg *PLBERTConfig) *ALBERT {
	const g = "bert.encoder.albert_layer_groups.0.albert_layers.0."
	a := &ALBERT{
		Config:    *cfg,
		Word:      l.embedding("bert.embeddings.word_embeddings.weight"),
		Position:  l.embedding("bert.embeddings.position_embeddings.weight"),
		TokenType: l.embedding("bert.embeddings.token_type_embeddings.weight"),
		EmbedNorm: l.layerNorm("bert.embeddings.LayerNorm", "weight", "bias",
			cfg.EmbeddingSize, albertLayerEps),
		MapIn: l.linear("bert.encoder.embedding_hidden_mapping_in"),
		Layer: &ALBERTLayer{
			Heads: cfg.NumAttentionHeads, HeadDim: cfg.HeadDim(),
			Q:     l.linear(g + "attention.query"),
			K:     l.linear(g + "attention.key"),
			V:     l.linear(g + "attention.value"),
			Dense: l.linear(g + "attention.dense"),
			AttnNorm: l.layerNorm(g+"attention.LayerNorm", "weight", "bias",
				cfg.HiddenSize, albertLayerEps),
			FFN:    l.linear(g + "ffn"),
			FFNOut: l.linear(g + "ffn_output"),
			OutNorm: l.layerNorm(g+"full_layer_layer_norm", "weight", "bias",
				cfg.HiddenSize, albertLayerEps),
		},
	}
	// `bert.pooler` is in the checkpoint and is dead weight: CustomAlbert
	// overrides forward to return last_hidden_state and nothing reads the
	// pooled output. 0.59 M parameters that are never loaded here.
	return a
}

func (l *loader) textEncoder(cfg *Config) *TextEncoder {
	t := &TextEncoder{
		Channels:  cfg.HiddenDim,
		Embedding: l.embedding("text_encoder.embedding.weight"),
		LSTM:      l.lstm("text_encoder.lstm"),
	}
	pad := (cfg.TextEncoderKernel - 1) / 2
	for i := 0; i < cfg.NLayer; i++ {
		p := fmt.Sprintf("text_encoder.cnn.%d.", i)
		t.Convs = append(t.Convs, l.conv1d(p+"0", 1, pad, 1, 1))
		// modules.py's own LayerNorm, whose affine is named gamma/beta.
		t.Norms = append(t.Norms, l.layerNorm(p+"1", "gamma", "beta", cfg.HiddenDim, layerNormEps))
	}
	return t
}

func (l *loader) adaIN(prefix string, channels int) *AdaIN1d {
	return &AdaIN1d{Channels: channels, FC: l.linear(prefix + ".fc")}
}

// adainBlock loads one AdainResBlk1d. `In` and `Out` come off the weights
// rather than from a table, so the shape of the stack is read from the
// checkpoint and the two upsampling cases are told apart by what exists:
// a `pool` means the block doubles its frames, a `conv1x1` means it changes
// channel count.
func (l *loader) adainBlock(prefix string) *AdainResBlk1d {
	conv1 := l.conv1d(prefix+".conv1", 1, 1, 1, 1)
	conv2 := l.conv1d(prefix+".conv2", 1, 1, 1, 1)
	if l.err != nil {
		return nil
	}
	b := &AdainResBlk1d{
		In: conv1.In, Out: conv2.Out,
		Norm1: l.adaIN(prefix+".norm1", conv1.In),
		Conv1: conv1,
		Norm2: l.adaIN(prefix+".norm2", conv1.Out),
		Conv2: conv2,
	}
	if l.set.Has(prefix + ".pool.weight") {
		b.Upsample = true
		// Depthwise, stride 2, kernel 3, padding 1, output padding 1: the one
		// combination that takes T frames to exactly 2T.
		b.Pool = l.convTranspose(prefix+".pool", 2, 1, 1, b.In)
	}
	if l.set.Has(prefix + ".conv1x1.weight") {
		b.Conv1x1 = l.conv1d(prefix+".conv1x1", 1, 0, 1, 1)
	}
	return b
}

func (l *loader) predictor(cfg *Config) *Predictor {
	p := &Predictor{
		TextEncoder:  &DurationEncoder{Channels: cfg.HiddenDim, StyleDim: cfg.StyleDim},
		LSTM:         l.lstm("predictor.lstm"),
		DurationHead: l.linear("predictor.duration_proj.linear_layer"),
		Shared:       l.lstm("predictor.shared"),
		F0Proj:       l.conv1d("predictor.F0_proj", 1, 0, 1, 1),
		NProj:        l.conv1d("predictor.N_proj", 1, 0, 1, 1),
	}
	// The ModuleList alternates LSTM and AdaLayerNorm, so the checkpoint's
	// indices are 0, 2, 4 for the recurrences and 1, 3, 5 for the norms.
	for i := 0; i < cfg.NLayer; i++ {
		p.TextEncoder.LSTMs = append(p.TextEncoder.LSTMs,
			l.lstm(fmt.Sprintf("predictor.text_encoder.lstms.%d", 2*i)))
		p.TextEncoder.Norms = append(p.TextEncoder.Norms, &AdaLayerNorm{
			Channels: cfg.HiddenDim,
			FC:       l.linear(fmt.Sprintf("predictor.text_encoder.lstms.%d.fc", 2*i+1)),
			norm:     LayerNorm{Width: cfg.HiddenDim, Eps: layerNormEps},
		})
	}
	for i := 0; i < 3; i++ {
		p.F0 = append(p.F0, l.adainBlock(fmt.Sprintf("predictor.F0.%d", i)))
		p.N = append(p.N, l.adainBlock(fmt.Sprintf("predictor.N.%d", i)))
	}
	return p
}

// vocoder loads the decoder and the generator.
//
// Almost nothing here is in config.json. The strides, paddings and dilations
// are properties of the Python that built the modules, and the two that are
// derived rather than written down — the noise convolutions' stride and the
// upsamplers' padding — are computed the same way istftnet.py computes them,
// with the shapes checked against the weights on the way through.
func (l *loader) vocoder(cfg *Config) *Vocoder {
	ic := &cfg.ISTFTNet
	v := &Vocoder{
		// Stride 2 over a 3-tap kernel: the curves the predictor just doubled
		// are halved again here, back to the alignment rate.
		F0Conv: l.conv1d("decoder.F0_conv", 2, 1, 1, 1),
		NConv:  l.conv1d("decoder.N_conv", 2, 1, 1, 1),
		Encode: l.adainBlock("decoder.encode"),
		ASRRes: l.conv1d("decoder.asr_res.0", 1, 0, 1, 1),
	}
	for i := 0; i < 4; i++ {
		v.Decode = append(v.Decode, l.adainBlock(fmt.Sprintf("decoder.decode.%d", i)))
	}

	g := &Generator{NumKernels: len(ic.ResblockKernelSizes), NFFT: ic.GenISTFTNFFT}
	g.Source = &HarmonicSource{
		SampleRate: cfg.SamplingRate, Harmonics: 8, SineAmp: 0.1, NoiseStd: 0.003, VoicedThreshold: 10,
		UpsampleScale: ic.UpsampleFactor() * ic.GenISTFTHopSize,
		Mix:           l.linear("decoder.generator.m_source.l_linear"),
	}
	for i, rate := range ic.UpsampleRates {
		k := ic.UpsampleKernelSizes[i]
		g.Ups = append(g.Ups, l.convTranspose(
			fmt.Sprintf("decoder.generator.ups.%d", i), rate, (k-rate)/2, 0, 1))

		// The excitation is resampled to each stage's rate by a convolution
		// whose stride is the product of the rates still to come — so the
		// last stage, with nothing after it, gets a 1-tap projection.
		stride := 1
		for _, r := range ic.UpsampleRates[i+1:] {
			stride *= r
		}
		p := fmt.Sprintf("decoder.generator.noise_convs.%d", i)
		if i+1 < len(ic.UpsampleRates) {
			g.NoiseConvs = append(g.NoiseConvs, l.conv1d(p, stride, (stride+1)/2, 1, 1))
		} else {
			g.NoiseConvs = append(g.NoiseConvs, l.conv1d(p, 1, 0, 1, 1))
		}
		g.NoiseRes = append(g.NoiseRes, l.snakeBlock(
			fmt.Sprintf("decoder.generator.noise_res.%d", i), []int{1, 3, 5}))

		for j := range ic.ResblockKernelSizes {
			g.ResBlocks = append(g.ResBlocks, l.snakeBlock(
				fmt.Sprintf("decoder.generator.resblocks.%d", i*g.NumKernels+j),
				ic.ResblockDilationSizes[j]))
		}
	}
	g.ConvPost = l.conv1d("decoder.generator.conv_post", 1, 3, 1, 1)

	// torch.stft's defaults, which istftnet.py takes without naming them: a
	// periodic Hann window, centred, reflected at the edges.
	win := audio.HannWindow(ic.GenISTFTNFFT, true)
	st, err := audio.NewSTFT(ic.GenISTFTNFFT, ic.GenISTFTHopSize, win, true)
	if err != nil && l.err == nil {
		l.err = err
	}
	if st != nil {
		st.Pad = audio.PadReflect
	}
	is, err := audio.NewISTFT(ic.GenISTFTNFFT, ic.GenISTFTHopSize, win, true)
	if err != nil && l.err == nil {
		l.err = err
	}
	g.STFT, g.ISTFT = st, is
	v.Generator = g
	return v
}

// snakeBlock loads one AdaINResBlock1: three (dilated convolution, plain
// convolution) pairs with a style normalisation and a snake before each.
func (l *loader) snakeBlock(prefix string, dilations []int) *SnakeResBlock {
	sh := l.shape(prefix + ".convs1.0.weight")
	if l.err != nil {
		return nil
	}
	if len(sh) != 3 {
		l.err = fmt.Errorf("kokoro: %s.convs1.0.weight has shape %v", prefix, sh)
		return nil
	}
	channels, kernel := sh[0], sh[2]
	b := &SnakeResBlock{Channels: channels}
	for i, d := range dilations {
		// get_padding(k, d) upstream: the dilated convolution keeps its
		// length, and the undilated one that follows it does too.
		b.Convs1 = append(b.Convs1, l.conv1d(fmt.Sprintf("%s.convs1.%d", prefix, i), 1, (kernel*d-d)/2, d, 1))
		b.Convs2 = append(b.Convs2, l.conv1d(fmt.Sprintf("%s.convs2.%d", prefix, i), 1, (kernel-1)/2, 1, 1))
		b.Norm1 = append(b.Norm1, l.adaIN(fmt.Sprintf("%s.adain1.%d", prefix, i), channels))
		b.Norm2 = append(b.Norm2, l.adaIN(fmt.Sprintf("%s.adain2.%d", prefix, i), channels))
		b.Alpha1 = append(b.Alpha1, l.f32(fmt.Sprintf("%s.alpha1.%d", prefix, i)))
		b.Alpha2 = append(b.Alpha2, l.f32(fmt.Sprintf("%s.alpha2.%d", prefix, i)))
	}
	return b
}

// loadVoices reads every `voice.*` tensor in the set.
func (m *Model) loadVoices(l *loader) {
	if l.err != nil {
		return
	}
	m.Voices = make(map[string][]float32)
	for _, name := range l.set.Names() {
		rest, ok := strings.CutPrefix(name, "voice.")
		if !ok {
			continue
		}
		sh := l.shape(name)
		if len(sh) != 2 {
			l.err = fmt.Errorf("kokoro: voice %s has shape %v, want [rows, dim]", rest, sh)
			return
		}
		if m.voiceRows == 0 {
			m.voiceRows, m.voiceDim = sh[0], sh[1]
		} else if sh[0] != m.voiceRows || sh[1] != m.voiceDim {
			l.err = fmt.Errorf("kokoro: voice %s has shape %v, others have [%d %d]",
				rest, sh, m.voiceRows, m.voiceDim)
			return
		}
		m.Voices[rest] = l.f32(name)
	}
	if len(m.Voices) == 0 {
		l.err = fmt.Errorf("kokoro: no voice.* tensors; run reference/convert_kokoro.py")
	}
}
