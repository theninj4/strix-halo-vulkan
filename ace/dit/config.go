// Package dit runs ACE-Step 1.5 XL turbo's condition encoder and DiT on the
// device (MUSIC.md A2/A3): the lyric and timbre encoders, the packed
// cross-attention sequence, and the 32-layer DiT that turns noise into 25 Hz
// × 64-channel latents in eight Euler steps.
package dit

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config is the part of config.json the device path reads.
type Config struct {
	Hidden       int      `json:"hidden_size"`
	FFN          int      `json:"intermediate_size"`
	Heads        int      `json:"num_attention_heads"`
	KVHeads      int      `json:"num_key_value_heads"`
	HeadDim      int      `json:"head_dim"`
	Layers       int      `json:"num_hidden_layers"`
	EncHidden    int      `json:"encoder_hidden_size"`
	EncFFN       int      `json:"encoder_intermediate_size"`
	EncHeads     int      `json:"encoder_num_attention_heads"`
	EncKVHeads   int      `json:"encoder_num_key_value_heads"`
	LyricLayers  int      `json:"num_lyric_encoder_hidden_layers"`
	TimbreLayers int      `json:"num_timbre_encoder_hidden_layers"`
	DetokLayers  int      `json:"num_attention_pooler_hidden_layers"`
	PoolWindow   int      `json:"pool_window_size"`
	TextDim      int      `json:"text_hidden_dim"`
	TimbreDim    int      `json:"timbre_hidden_dim"`
	Latent       int      `json:"audio_acoustic_hidden_dim"`
	InChannels   int      `json:"in_channels"`
	Patch        int      `json:"patch_size"`
	Window       int      `json:"sliding_window"`
	UseWindow    bool     `json:"use_sliding_window"`
	Theta        float64  `json:"rope_theta"`
	Eps          float64  `json:"rms_norm_eps"`
	LayerTypes   []string `json:"layer_types"`
}

// LoadConfig reads dir/config.json.
func LoadConfig(dir string) (*Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("dit: %s: %w", dir, err)
	}
	if c.HeadDim != 128 || c.Heads%c.KVHeads != 0 || c.EncHeads%c.EncKVHeads != 0 {
		return nil, fmt.Errorf("dit: head layout %d/%d/%d (encoder %d/%d) is not one the kernels build",
			c.HeadDim, c.Heads, c.KVHeads, c.EncHeads, c.EncKVHeads)
	}
	if len(c.LayerTypes) < max(c.Layers, c.LyricLayers, c.TimbreLayers) {
		return nil, fmt.Errorf("dit: %d layer types for %d layers", len(c.LayerTypes), c.Layers)
	}
	return &c, nil
}

// Windowed reports whether layer i of any stack is a sliding-window layer.
// The encoders index the same list as the DiT (upstream deep-copies the
// config), so in all three stacks it is the even layers.
func (c *Config) Windowed(i int) bool {
	return c.UseWindow && c.LayerTypes[i] == "sliding_attention"
}

// PatchIn is the input projection's reduction extent: a token is Patch
// frames of [context | x_t].
func (c *Config) PatchIn() int { return c.Patch * c.InChannels }

// PatchOut is the output projection's width: Patch frames of Latent.
func (c *Config) PatchOut() int { return c.Patch * c.Latent }
