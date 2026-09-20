// Package textenc implements Qwen-Image-2.1's text encoding for the t2i
// path: the raw prompt template, the system-prefix drop, and the config
// adapter that lets zimage/qwen's Qwen3 transformer run the checkpoint's
// Qwen3-VL-8B language model.
//
// It is the Go counterpart of QwenImage21Pipeline._get_qwen_prompt_embeds
// (reference/qwenimage21/pipeline_qwenimage21.py), with the parts the edit
// path adds — vision slots in the template, the vision tower, mrope over
// image positions — deferred to Q8 (IMAGE.md). Q0 measured that for a
// text-only prompt the checkpoint's interleaved mrope is *identical* to the
// plain NeoX RoPE zimage/qwen already applies (mrope_gap 0 in the textenc
// manifest), so nothing here compensates for it.
//
// What the DiT wants from the encoder is narrower than the model, and
// different from Z-Image's slice: **all 36 layers run and the final norm
// does not** (Z-Image stopped a layer short). qwen.Load already expresses
// that as Load(dir, cfg.NumLayers) — Forward never applies a final norm
// because none is loaded. The rows the DiT sees start after the tokenized
// system message: diffusers computes that drop from the processor's chat
// template at pipeline-construction time, and DropTokens computes it the
// same way, by tokenizing the rendered system message rather than
// hardcoding its length.
package textenc

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// SysPrompt is the fixed system message every prompt is wrapped in.
const SysPrompt = "Comprehend and analyze the provided prompt."

// sysRendered is the system message as the chat template renders it, which
// is also the exact prefix of every rendered prompt — that identity is what
// lets the drop count come from tokenizing this string. The reference
// derives it through processor.apply_chat_template; the tests check the
// token count against the dump's drop_idx rather than trusting either.
const sysRendered = "<|im_start|>system\n" + SysPrompt + "<|im_end|>\n"

// TemplateT2I renders a prompt into the raw t2i template. This is a plain
// format string, not apply_chat_template — the two tokenize differently and
// the checkpoint expects this one. An empty prompt becomes a single space:
// Qwen has no BOS token, and an empty user turn would leave the encoder
// nothing to read.
func TemplateT2I(prompt string) string {
	if prompt == "" {
		prompt = " "
	}
	return sysRendered + "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
}

// LoadConfig reads the checkpoint's text_encoder/config.json as a
// qwen.Config. A Qwen3-VL checkpoint nests the language model's dimensions
// under "text_config" (the top level describes the composite model) and
// names its tensors "model.language_model.…" — same layers, same loader,
// one string apart, as embed.LoadConfig is for Qwen3-Embedding.
func LoadConfig(dir string) (*qwen.Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var outer struct {
		Text *qwen.Config `json:"text_config"`
	}
	if err := json.Unmarshal(buf, &outer); err != nil {
		return nil, fmt.Errorf("textenc: parsing config.json: %w", err)
	}
	c := outer.Text
	if c == nil || c.HiddenSize == 0 || c.NumLayers == 0 || c.HeadDim == 0 {
		return nil, fmt.Errorf("textenc: config.json has no text_config with hidden_size, num_hidden_layers and head_dim")
	}
	c.Prefix = "model.language_model."
	return c, nil
}

// DropTokens is how many leading rows of the encoder's output the DiT never
// sees: the length of the tokenized system message.
func DropTokens(tok *tokenizer.Tokenizer) (int, error) {
	ids, err := tok.Encode(sysRendered)
	if err != nil {
		return 0, err
	}
	return len(ids), nil
}

// EncodePrompt tokenizes a prompt under the t2i template.
func EncodePrompt(tok *tokenizer.Tokenizer, prompt string) ([]int32, error) {
	return tok.Encode(TemplateT2I(prompt))
}

// Drop returns the rows of a forward pass after the system prefix — the
// prompt embedding the DiT's txt_in consumes.
func Drop(m *qwen.Mat, drop int) (*qwen.Mat, error) {
	if drop <= 0 || drop >= m.Rows {
		return nil, fmt.Errorf("textenc: dropping %d of %d rows", drop, m.Rows)
	}
	return &qwen.Mat{Rows: m.Rows - drop, Cols: m.Cols, Data: m.Data[drop*m.Cols:]}, nil
}
