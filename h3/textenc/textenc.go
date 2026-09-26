// Package textenc is MiniMax-H3's text conditioning (VIDEO.md M2): the
// hidden state after decoder layer 49 of the Qwen3-VL-32B conditioner,
// *unnormalised*, over the prompt tokenised verbatim — no chat template, no
// special tokens, nothing dropped.
//
// The conditioner is Qwen-Image-2.1's Qwen3-VL at four times the size, so
// this package is only the parts that differ: which layer, and the
// presentation. The transformer itself is zimage/qwen under
// qimage/textenc's nested-config adapter; loading Layers of its 64 layers
// makes the last one's output the conditioning, because zimage/qwen never
// applies a final norm.
package textenc

import (
	"strix-halo-vulkan/qimage/textenc"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// Layers is how many decoder layers run: the pipeline reads
// hidden_states[50], the output of layer index 49. Layers 50–63 and the
// language-model head are never loaded.
const Layers = 50

// Width is the conditioning's feature width, the transformer's text_dim.
const Width = 5120

// LoadConfig reads the checkpoint's text_encoder/config.json.
func LoadConfig(dir string) (*qwen.Config, error) { return textenc.LoadConfig(dir) }

// EncodePrompt tokenises a t2va prompt: the string itself, as
// `tokenizer(prompt, add_special_tokens=False)` does. The repository's
// tokenizer adds `<d>`/`</d>` for dialogue, which Context-IR prompts use.
func EncodePrompt(tok *tokenizer.Tokenizer, prompt string) ([]int32, error) {
	return tok.Encode(prompt)
}
