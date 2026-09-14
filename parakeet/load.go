package parakeet

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
)

// layerNormEps and batchNormEps are torch's defaults for nn.LayerNorm and
// nn.BatchNorm1d. Neither appears in config.json, and neither is the
// feature extractor's 1e-5 — that one is a different constant that happens to
// have the same value.
const (
	layerNormEps = 1e-5
	batchNormEps = 1e-5
)

// Model is the whole of parakeet-tdt-0.6b-v3: the front end, the encoder, the
// projection into the joint's width, the prediction network and the joint.
type Model struct {
	Config     *Config
	FrontEnd   *FrontEnd
	Encoder    *Encoder
	Projector  *Linear
	Prediction *Prediction
	Joint      *Joint
	Tokenizer  *Tokenizer
}

// Load reads a checkpoint directory into memory as float32.
//
// The weights are copied out of the mapping rather than aliased: this is the
// CPU reference, where every tensor is read many times in an order that has
// nothing to do with the file's, and 2.5 GB of fp32 is affordable on a part
// with 128 GB of unified memory. The Vulkan path (stage S6) will stage
// straight from the mapping into fp16 device buffers instead.
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

	fe, err := NewFrontEnd(cfg.Features)
	if err != nil {
		return nil, err
	}
	l := &loader{set: set}
	m := &Model{Config: cfg, FrontEnd: fe}

	m.Encoder = l.encoder(&cfg.Encoder)
	m.Projector = l.linear("encoder_projector")
	m.Prediction = l.prediction(cfg)
	m.Joint = &Joint{Head: l.linear("joint.head"), Vocab: cfg.VocabSize, Durations: cfg.Durations}
	if l.err != nil {
		return nil, l.err
	}

	tok, err := LoadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	m.Tokenizer = tok
	return m, nil
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
// `prefix.bias`. The bias is optional per tensor rather than per config flag
// because the two disagree in this checkpoint: the flags say the encoder has
// no biases, and the three projections that join stages have them anyway.
func (l *loader) linear(prefix string) *Linear {
	sh := l.shape(prefix + ".weight")
	if l.err != nil {
		return nil
	}
	// Pointwise convolutions are [Out, In, 1]: the trailing kernel axis is a
	// no-op and the weights are a plain matrix.
	for len(sh) > 2 && sh[len(sh)-1] == 1 {
		sh = sh[:len(sh)-1]
	}
	if len(sh) != 2 {
		l.err = fmt.Errorf("parakeet: %s.weight has shape %v, want 2 dims", prefix, sh)
		return nil
	}
	lin := &Linear{Out: sh[0], In: sh[1], Weight: l.f32(prefix + ".weight")}
	if l.set.Has(prefix + ".bias") {
		lin.Bias = l.f32(prefix + ".bias")
	}
	return lin
}

func (l *loader) layerNorm(prefix string) *LayerNorm {
	return &LayerNorm{
		Weight: l.f32(prefix + ".weight"),
		Bias:   l.f32(prefix + ".bias"),
		Eps:    layerNormEps,
	}
}

func (l *loader) conv2d(prefix string, index int, depthwise, relu bool) *Conv2D {
	sh := l.shape(prefix + ".weight")
	if l.err != nil {
		return nil
	}
	if len(sh) != 4 {
		l.err = fmt.Errorf("parakeet: %s.weight has shape %v, want 4 dims", prefix, sh)
		return nil
	}
	c := &Conv2D{
		Index: index, Out: sh[0], Kernel: sh[2], Depthwise: depthwise, ReLUAfter: relu,
		Weight: l.f32(prefix + ".weight"),
	}
	if depthwise {
		c.In = sh[0]
	} else {
		c.In = sh[1]
	}
	// Kernel 3 is strided and padded; kernel 1 is the pointwise convolution
	// between two depthwise ones and moves nothing.
	if c.Kernel > 1 {
		c.Stride, c.Pad = 2, (c.Kernel-1)/2
	} else {
		c.Stride, c.Pad = 1, 0
	}
	if l.set.Has(prefix + ".bias") {
		c.Bias = l.f32(prefix + ".bias")
	}
	return c
}

func (l *loader) encoder(cfg *EncoderConfig) *Encoder {
	e := &Encoder{Config: *cfg}

	// The `dw_striding` stack: one dense convolution, then two
	// depthwise/pointwise pairs, with a ReLU after the dense one and after
	// each pointwise one. The ModuleList indices are the checkpoint's, and
	// the gaps in them are the activations.
	const p = "encoder.subsampling.layers."
	e.Subsampling = &Subsampling{
		Convs: []*Conv2D{
			l.conv2d(p+"0", 0, false, true),
			l.conv2d(p+"2", 2, true, false),
			l.conv2d(p+"3", 3, false, true),
			l.conv2d(p+"5", 5, true, false),
			l.conv2d(p+"6", 6, false, true),
		},
		Linear: l.linear("encoder.subsampling.linear"),
	}

	heads, hd := cfg.NumAttentionHeads, cfg.HeadDim()
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		pre := fmt.Sprintf("encoder.layers.%d.", i)
		layer := &EncoderLayer{
			NormFF1:  l.layerNorm(pre + "norm_feed_forward1"),
			FF1:      &FeedForward{Linear1: l.linear(pre + "feed_forward1.linear1"), Linear2: l.linear(pre + "feed_forward1.linear2")},
			NormAttn: l.layerNorm(pre + "norm_self_att"),
			Attn: &Attention{
				Heads: heads, HeadDim: hd,
				Q: l.linear(pre + "self_attn.q_proj"), K: l.linear(pre + "self_attn.k_proj"),
				V: l.linear(pre + "self_attn.v_proj"), O: l.linear(pre + "self_attn.o_proj"),
				RelK:  l.linear(pre + "self_attn.relative_k_proj"),
				BiasU: l.f32(pre + "self_attn.bias_u"), BiasV: l.f32(pre + "self_attn.bias_v"),
			},
			NormConv: l.layerNorm(pre + "norm_conv"),
			Conv:     l.convModule(pre+"conv", cfg),
			NormFF2:  l.layerNorm(pre + "norm_feed_forward2"),
			FF2:      &FeedForward{Linear1: l.linear(pre + "feed_forward2.linear1"), Linear2: l.linear(pre + "feed_forward2.linear2")},
			NormOut:  l.layerNorm(pre + "norm_out"),
		}
		e.Layers = append(e.Layers, layer)
	}
	return e
}

// convModule loads the convolution branch, folding its BatchNorm into a
// per-channel affine.
//
// At inference BatchNorm is `(x - mean)/sqrt(var + eps) * weight + bias`,
// which is a scale and a shift that depend on nothing but the channel, so it
// costs nothing at run time once folded. The running statistics are in the
// checkpoint; `num_batches_tracked` is the I64 scalar beside them that
// nothing reads, and that safetensors carries only so that cmd/inspect can
// report it.
func (l *loader) convModule(prefix string, cfg *EncoderConfig) *ConvModule {
	dwShape := l.shape(prefix + ".depthwise_conv.weight")
	if l.err != nil {
		return nil
	}
	if len(dwShape) != 3 || dwShape[1] != 1 {
		l.err = fmt.Errorf("parakeet: %s.depthwise_conv.weight has shape %v, want [C, 1, K]", prefix, dwShape)
		return nil
	}
	c := &ConvModule{
		Channels: dwShape[0],
		Kernel:   dwShape[2],
		PW1:      l.linear(prefix + ".pointwise_conv1"),
		DW:       l.f32(prefix + ".depthwise_conv.weight"),
		PW2:      l.linear(prefix + ".pointwise_conv2"),
	}
	weight := l.f32(prefix + ".norm.weight")
	bias := l.f32(prefix + ".norm.bias")
	mean := l.f32(prefix + ".norm.running_mean")
	variance := l.f32(prefix + ".norm.running_var")
	if l.err != nil {
		return nil
	}
	c.BNScale = make([]float32, c.Channels)
	c.BNShift = make([]float32, c.Channels)
	for i := range c.BNScale {
		s := float32(float64(weight[i]) / math.Sqrt(float64(variance[i])+batchNormEps))
		c.BNScale[i] = s
		c.BNShift[i] = bias[i] - mean[i]*s
	}
	if c.Kernel != cfg.ConvKernelSize {
		l.err = fmt.Errorf("parakeet: %s has kernel %d, config says %d", prefix, c.Kernel, cfg.ConvKernelSize)
	}
	return c
}

func (l *loader) prediction(cfg *Config) *Prediction {
	sh := l.shape("decoder.embedding.weight")
	if l.err != nil {
		return nil
	}
	p := &Prediction{
		Vocab: sh[0], Hidden: sh[1],
		Embedding: l.f32("decoder.embedding.weight"),
		Projector: l.linear("decoder.decoder_projector"),
	}
	for i := 0; i < cfg.NumDecoderLayers; i++ {
		ih := l.f32(fmt.Sprintf("decoder.lstm.weight_ih_l%d", i))
		hh := l.f32(fmt.Sprintf("decoder.lstm.weight_hh_l%d", i))
		bih := l.f32(fmt.Sprintf("decoder.lstm.bias_ih_l%d", i))
		bhh := l.f32(fmt.Sprintf("decoder.lstm.bias_hh_l%d", i))
		if l.err != nil {
			return nil
		}
		bias := make([]float32, len(bih))
		for j := range bias {
			bias[j] = bih[j] + bhh[j]
		}
		p.Layers = append(p.Layers, &LSTMLayer{
			Hidden: cfg.DecoderHiddenSize, In: cfg.DecoderHiddenSize,
			WIH: ih, WHH: hh, Bias: bias,
		})
	}
	return p
}

// SetF16 narrows every weight the encoder multiplies by to fp16 precision,
// and makes each projection narrow its input the same way.
//
// This is an emulation of what stage S6 will actually run — fp16 operands,
// fp32 accumulators — asked of the CPU reference, where the answer can be
// compared against the reference dump and against the transcript itself. What
// it does not emulate is the *order* of the accumulation, which a GPU splits
// across a K loop and several waves; that changes the rounding of the sum,
// not its precision, and the bound it moves is far below the one fp16
// operands move.
//
// The norms, the residual stream and the front end stay float32. That is the
// shape of the port too: a LayerNorm is a reduction over 1024 values and the
// residual is what everything accumulates into, so neither is a matrix-core
// operand and neither is worth narrowing.
func (m *Model) SetF16(on bool) {
	narrow := func(l *Linear) {
		if l == nil || l.Narrow == on {
			return
		}
		l.Narrow = on
		if on {
			narrowSliceF16(l.Weight)
		}
	}
	for _, layer := range m.Encoder.Layers {
		narrow(layer.FF1.Linear1)
		narrow(layer.FF1.Linear2)
		narrow(layer.FF2.Linear1)
		narrow(layer.FF2.Linear2)
		narrow(layer.Attn.Q)
		narrow(layer.Attn.K)
		narrow(layer.Attn.V)
		narrow(layer.Attn.O)
		narrow(layer.Attn.RelK)
		narrow(layer.Conv.PW1)
		narrow(layer.Conv.PW2)
	}
	narrow(m.Encoder.Subsampling.Linear)
	narrow(m.Projector)
	narrow(m.Prediction.Projector)
	narrow(m.Joint.Head)
	if on {
		for _, c := range m.Encoder.Subsampling.Convs {
			narrowSliceF16(c.Weight)
		}
		for _, layer := range m.Encoder.Layers {
			narrowSliceF16(layer.Conv.DW)
		}
		for _, l := range m.Prediction.Layers {
			narrowSliceF16(l.WIH)
			narrowSliceF16(l.WHH)
		}
		narrowSliceF16(m.Prediction.Embedding)
	}
}
