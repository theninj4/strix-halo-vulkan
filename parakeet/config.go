// Package parakeet implements NVIDIA's parakeet-tdt-0.6b-v3 speech-to-text
// model: a FastConformer encoder, an LSTM prediction network, and the
// token-and-duration transducer that joins them.
//
// It is written against the checkpoint at models/parakeet-tdt-0.6b-v3, which
// ships as HF `transformers` rather than NeMo, so the oracle for every stage
// is `ParakeetForTDT` itself — see reference/dump_parakeet.py, whose manifest
// and .bin files are what the tests here compare against.
package parakeet

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// EncoderConfig mirrors config.json's `encoder_config`.
type EncoderConfig struct {
	HiddenSize            int    `json:"hidden_size"`
	IntermediateSize      int    `json:"intermediate_size"`
	NumHiddenLayers       int    `json:"num_hidden_layers"`
	NumAttentionHeads     int    `json:"num_attention_heads"`
	ConvKernelSize        int    `json:"conv_kernel_size"`
	NumMelBins            int    `json:"num_mel_bins"`
	SubsamplingFactor     int    `json:"subsampling_factor"`
	SubsamplingChannels   int    `json:"subsampling_conv_channels"`
	SubsamplingKernel     int    `json:"subsampling_conv_kernel_size"`
	SubsamplingStride     int    `json:"subsampling_conv_stride"`
	MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	HiddenAct             string `json:"hidden_act"`
	AttentionBias         bool   `json:"attention_bias"`
	ConvolutionBias       bool   `json:"convolution_bias"`
	ScaleInput            bool   `json:"scale_input"`
}

// HeadDim is the per-head width, 128 here — which is what the existing WMMA
// attention kernel is compiled for.
func (e *EncoderConfig) HeadDim() int { return e.HiddenSize / e.NumAttentionHeads }

// Config mirrors config.json.
type Config struct {
	VocabSize         int           `json:"vocab_size"`
	BlankTokenID      int           `json:"blank_token_id"`
	DecoderHiddenSize int           `json:"decoder_hidden_size"`
	NumDecoderLayers  int           `json:"num_decoder_layers"`
	MaxSymbolsPerStep int           `json:"max_symbols_per_step"`
	Durations         []int         `json:"durations"`
	HiddenAct         string        `json:"hidden_act"` // the joint's activation
	Encoder           EncoderConfig `json:"encoder_config"`

	Features FeatureConfig `json:"-"` // from processor_config.json
}

// FeatureConfig mirrors processor_config.json's `feature_extractor`.
type FeatureConfig struct {
	SamplingRate int     `json:"sampling_rate"`
	HopLength    int     `json:"hop_length"`
	NFFT         int     `json:"n_fft"`
	WinLength    int     `json:"win_length"`
	NMels        int     `json:"feature_size"`
	Preemphasis  float64 `json:"preemphasis"`
}

// LoadConfig reads a checkpoint directory's config.json and
// processor_config.json.
//
// Both are read because the split between them is not a split in the model:
// the number of mel bins appears in each, and the front end's hop length is
// what decides how many encoder frames a clip becomes.
func LoadConfig(dir string) (*Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(buf, &c); err != nil {
		return nil, fmt.Errorf("parakeet: parsing config.json: %w", err)
	}

	buf, err = os.ReadFile(filepath.Join(dir, "processor_config.json"))
	if err != nil {
		return nil, err
	}
	var proc struct {
		FeatureExtractor FeatureConfig `json:"feature_extractor"`
	}
	if err := json.Unmarshal(buf, &proc); err != nil {
		return nil, fmt.Errorf("parakeet: parsing processor_config.json: %w", err)
	}
	c.Features = proc.FeatureExtractor

	switch {
	case c.Encoder.HiddenSize == 0 || c.Encoder.NumHiddenLayers == 0:
		return nil, fmt.Errorf("parakeet: %s: config.json has no encoder_config", dir)
	case c.Features.NMels == 0 || c.Features.NFFT == 0:
		return nil, fmt.Errorf("parakeet: %s: processor_config.json has no feature_extractor", dir)
	case c.Features.NMels != c.Encoder.NumMelBins:
		return nil, fmt.Errorf("parakeet: %s: front end makes %d mel bins, encoder wants %d",
			dir, c.Features.NMels, c.Encoder.NumMelBins)
	}
	return &c, nil
}
