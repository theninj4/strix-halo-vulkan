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
	"sort"
	"strconv"
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
	// IsTurbo is the CFG-distilled checkpoint (8 fixed steps, no guidance);
	// the base model runs CFG against null_condition_emb (A15).
	IsTurbo bool `json:"is_turbo"`
	// AlignLayers is lyric_alignment_layers_config: the cross-attention
	// heads, by layer, whose attention follows the lyrics (MUSIC.md A14b).
	AlignLayers map[string][]int `json:"lyric_alignment_layers_config"`
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

// AlignHeads is AlignLayers as (layer, head) pairs, layers ascending and
// each layer's heads as listed: the order upstream averages them in.
func (c *Config) AlignHeads() ([][2]int, error) {
	var ls []int
	for k := range c.AlignLayers {
		l, err := strconv.Atoi(k)
		if err != nil || l < 0 || l >= c.Layers {
			return nil, fmt.Errorf("dit: lyric_alignment_layers_config layer %q", k)
		}
		ls = append(ls, l)
	}
	sort.Ints(ls)
	var hs [][2]int
	for _, l := range ls {
		for _, h := range c.AlignLayers[strconv.Itoa(l)] {
			if h < 0 || h >= c.Heads {
				return nil, fmt.Errorf("dit: lyric_alignment_layers_config head %d", h)
			}
			hs = append(hs, [2]int{l, h})
		}
	}
	if len(hs) == 0 {
		return nil, fmt.Errorf("dit: no lyric_alignment_layers_config in config.json")
	}
	return hs, nil
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
