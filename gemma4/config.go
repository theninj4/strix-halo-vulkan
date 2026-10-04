package gemma4

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config is the text tower's shape, from config.json's text_config.
type Config struct {
	Layers        int      `json:"num_hidden_layers"`
	Hidden        int      `json:"hidden_size"`
	Intermediate  int      `json:"intermediate_size"`     // the dense MLP
	MoEInter      int      `json:"moe_intermediate_size"` // one expert
	Experts       int      `json:"num_experts"`
	TopK          int      `json:"top_k_experts"`
	Heads         int      `json:"num_attention_heads"`
	KVHeads       int      `json:"num_key_value_heads"`        // sliding layers
	GlobalKVHeads int      `json:"num_global_key_value_heads"` // full layers
	HeadDim       int      `json:"head_dim"`                   // sliding layers
	GlobalHeadDim int      `json:"global_head_dim"`            // full layers
	Window        int      `json:"sliding_window"`
	Vocab         int      `json:"vocab_size"`
	Eps           float64  `json:"rms_norm_eps"`
	Softcap       float64  `json:"final_logit_softcapping"`
	Activation    string   `json:"hidden_activation"`
	KEqV          bool     `json:"attention_k_eq_v"`
	MoEBlock      bool     `json:"enable_moe_block"`
	Tied          bool     `json:"tie_word_embeddings"`
	LayerTypes    []string `json:"layer_types"`
	PLE           int      `json:"hidden_size_per_layer_input"`
	KVShared      int      `json:"num_kv_shared_layers"`
	Rope          map[string]struct {
		Theta   float64 `json:"rope_theta"`
		Type    string  `json:"rope_type"`
		Partial float64 `json:"partial_rotary_factor"`
	} `json:"rope_parameters"`
}

// Full reports whether layer i is a full-attention layer.
func (c *Config) Full(i int) bool { return c.LayerTypes[i] == "full_attention" }

// LoadConfig reads dir/config.json and holds it to the shape this port
// implements: the 26B-A4B text tower, nothing it would silently get wrong.
func LoadConfig(dir string) (*Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var f struct {
		Text Config `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("gemma4: config.json: %w", err)
	}
	c := &f.Text
	switch {
	case len(c.LayerTypes) != c.Layers:
		return nil, fmt.Errorf("gemma4: %d layer types for %d layers", len(c.LayerTypes), c.Layers)
	case !c.MoEBlock || c.Experts == 0 || c.TopK == 0:
		return nil, fmt.Errorf("gemma4: not a MoE checkpoint")
	case c.PLE != 0 || c.KVShared != 0:
		return nil, fmt.Errorf("gemma4: per-layer inputs and shared KV layers (the E-series) are not implemented")
	case !c.KEqV || !c.Tied:
		return nil, fmt.Errorf("gemma4: expected K = V on full layers and a tied head")
	case c.Activation != "gelu_pytorch_tanh":
		return nil, fmt.Errorf("gemma4: activation %q", c.Activation)
	case c.Hidden%32 != 0 || c.MoEInter%32 != 0 || c.Intermediate%32 != 0:
		return nil, fmt.Errorf("gemma4: widths must be whole Q8_0 blocks")
	}
	if c.Rope["sliding_attention"].Type != "default" || c.Rope["full_attention"].Type != "proportional" {
		return nil, fmt.Errorf("gemma4: unexpected rope types")
	}
	return c, nil
}
