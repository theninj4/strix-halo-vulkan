// Package kokoro implements hexgrad's Kokoro-82M text-to-speech model: an
// ALBERT phoneme encoder, a StyleTTS2 prosody predictor, and an iSTFTNet
// vocoder, driven by a 256-wide style vector picked per voice and per
// utterance length.
//
// It is written against the *converted* checkpoint — the shipped
// `kokoro-v1_0.pth` is a pickle, so reference/convert_kokoro.py folds its
// weight_norm and writes model.safetensors and voices.safetensors beside it.
// The oracle for every stage is hexgrad's own `KModel`; see
// reference/dump_kokoro.py, whose manifest and .bin files are what the tests
// here compare against.
//
// Layout: every activation is a [frames, channels] Mat — channel-last, the
// opposite of the [C, T] the PyTorch convolutions use. That is S7's finding
// carried over: held channel-last, the pointwise convolutions are GEMMs and
// the depthwise ones are coalesced. The reference dumps most of these tensors
// as [C, T], so the tests transpose rather than the model.
package kokoro

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Torch's defaults for the normalisations this model uses. None of the three
// is in config.json, and they are not the same number: ALBERT was trained
// with 1e-12 and everything StyleTTS2 built with 1e-5, so a single shared
// constant here would be wrong by six orders of magnitude in one of them.
const (
	layerNormEps    = 1e-5  // nn.LayerNorm, and modules.py's own LayerNorm/AdaLayerNorm
	instanceNormEps = 1e-5  // nn.InstanceNorm1d, inside every AdaIN1d
	albertLayerEps  = 1e-12 // AlbertConfig.layer_norm_eps
)

// PLBERTConfig mirrors config.json's `plbert`, plus the three AlbertConfig
// defaults it leaves out and which the dump's manifest pins.
type PLBERTConfig struct {
	HiddenSize        int `json:"hidden_size"`
	NumAttentionHeads int `json:"num_attention_heads"`
	IntermediateSize  int `json:"intermediate_size"`
	MaxPositionEmbed  int `json:"max_position_embeddings"`
	NumHiddenLayers   int `json:"num_hidden_layers"`
	EmbeddingSize     int `json:"-"` // AlbertConfig default, 128
	TypeVocabSize     int `json:"-"` // AlbertConfig default, 2
}

// HeadDim is the per-head width, 64 here.
func (p *PLBERTConfig) HeadDim() int { return p.HiddenSize / p.NumAttentionHeads }

// ISTFTNetConfig mirrors config.json's `istftnet`. It is read at T2 so that
// the frame arithmetic can be checked end to end before the vocoder exists.
type ISTFTNetConfig struct {
	UpsampleRates          []int   `json:"upsample_rates"`
	UpsampleKernelSizes    []int   `json:"upsample_kernel_sizes"`
	UpsampleInitialChannel int     `json:"upsample_initial_channel"`
	ResblockKernelSizes    []int   `json:"resblock_kernel_sizes"`
	ResblockDilationSizes  [][]int `json:"resblock_dilation_sizes"`
	GenISTFTNFFT           int     `json:"gen_istft_n_fft"`
	GenISTFTHopSize        int     `json:"gen_istft_hop_size"`
}

// UpsampleFactor is how many STFT frames the generator makes from one of its
// input frames: the product of its upsampling rates, 60 here.
func (i *ISTFTNetConfig) UpsampleFactor() int {
	n := 1
	for _, r := range i.UpsampleRates {
		n *= r
	}
	return n
}

// Config mirrors config.json.
type Config struct {
	NTokens           int            `json:"n_token"`
	HiddenDim         int            `json:"hidden_dim"`
	StyleDim          int            `json:"style_dim"`
	MaxDur            int            `json:"max_dur"`
	NLayer            int            `json:"n_layer"`
	NMels             int            `json:"n_mels"`
	TextEncoderKernel int            `json:"text_encoder_kernel_size"`
	PLBERT            PLBERTConfig   `json:"plbert"`
	ISTFTNet          ISTFTNetConfig `json:"istftnet"`
	Vocab             map[string]int `json:"vocab"`

	SamplingRate int `json:"-"` // 24000; not in config.json, fixed by the vocoder
}

// SamplesPerFrame is how many output samples one alignment frame becomes:
// **600** at 24 kHz, i.e. 25 ms. The factor is not the generator's alone —
// the chain is a 2x inside the decoder's last AdainResBlk1d, then the
// generator's 10 and 6, then the iSTFT hop of 5. That first doubling is the
// one that is easy to miss, because it happens in a residual block rather
// than in anything named "upsample" in config.json.
//
// Every length in the model is this number times a duration, so an off-by-one
// here desynchronises the whole vocoder.
func (c *Config) SamplesPerFrame() int {
	return 2 * c.ISTFTNet.UpsampleFactor() * c.ISTFTNet.GenISTFTHopSize
}

// LoadConfig reads a checkpoint directory's config.json.
func LoadConfig(dir string) (*Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(buf, &c); err != nil {
		return nil, fmt.Errorf("kokoro: parsing config.json: %w", err)
	}
	// AlbertConfig's defaults, which config.json does not carry because the
	// upstream builds AlbertConfig(**config['plbert']) and lets them stand.
	c.PLBERT.EmbeddingSize = 128
	c.PLBERT.TypeVocabSize = 2
	c.SamplingRate = 24000

	switch {
	case c.NTokens == 0 || c.HiddenDim == 0:
		return nil, fmt.Errorf("kokoro: %s: config.json has no n_token/hidden_dim", dir)
	case len(c.Vocab) == 0:
		return nil, fmt.Errorf("kokoro: %s: config.json has no vocab", dir)
	case c.PLBERT.HiddenSize%c.PLBERT.NumAttentionHeads != 0:
		return nil, fmt.Errorf("kokoro: %d hidden does not divide into %d heads",
			c.PLBERT.HiddenSize, c.PLBERT.NumAttentionHeads)
	case len(c.ISTFTNet.UpsampleRates) == 0:
		return nil, fmt.Errorf("kokoro: %s: config.json has no istftnet", dir)
	}
	return &c, nil
}

// Phonemes maps a phoneme string to input ids, the way KModel.forward does:
// one id per rune, characters outside the vocabulary dropped silently, and
// the whole thing wrapped in the id-0 boundary token at both ends.
//
// Dropping is the upstream's behaviour and is kept, but the count of what was
// dropped is returned rather than swallowed — the style vector is indexed by
// the *phoneme* count and the model by the *token* count, so a dropped rune
// makes those two disagree.
func (c *Config) Phonemes(s string) (ids []int, dropped int) {
	ids = append(ids, 0)
	for _, r := range s {
		if id, ok := c.Vocab[string(r)]; ok {
			ids = append(ids, id)
		} else {
			dropped++
		}
	}
	return append(ids, 0), dropped
}
