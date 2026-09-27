package ocr

import (
	"fmt"
	"strings"
)

// Tasks are the six prompts the checkpoint was trained on (the card's
// PROMPTS and PaddleX's pipeline), keyed by PaddleX's prompt labels.
var Tasks = map[string]string{
	"ocr":      "OCR:",
	"table":    "Table Recognition:",
	"formula":  "Formula Recognition:",
	"chart":    "Chart Recognition:",
	"seal":     "Seal Recognition:",
	"spotting": "Spotting:",
}

// The template's markers (chat_template.jinja). The image slot is
// <|IMAGE_PLACEHOLDER|> (config image_token_id 100295), **not** the
// <|image_pad|> the tokenizer also carries: the processor expands the
// placeholder, and a prompt built on image_pad would put the image rows
// nowhere.
const (
	tokBegin      = "<|begin_of_sentence|>"
	tokImageStart = "<|IMAGE_START|>"
	tokImage      = "<|IMAGE_PLACEHOLDER|>"
	tokImageEnd   = "<|IMAGE_END|>"
	tokEOS        = "</s>"
)

// Prompt is one request's token row: ids, the 3-D rope position of each,
// and where the image's rows go.
type Prompt struct {
	IDs []int32
	// Pos is (t, h, w) per token (OCR.md: positions).
	Pos [][3]int32
	// ImageAt is the index of the first image token; ImageLen of them follow.
	ImageAt, ImageLen int
	// Next is the position the first generated token takes. It is not
	// len(IDs): an image of mh x mw merged tokens advances the position by
	// max(mh, mw), not mh*mw. HF's rope delta is Next - len(IDs).
	Next int32
}

// BuildPrompt renders the chat template around one image of mh x mw merged
// tokens and a text, and tokenises it as the processor does: the whole
// string at once, the placeholder expanded to mh*mw copies.
func (t *Tokenizer) BuildPrompt(text string, mh, mw int) (*Prompt, error) {
	if mh <= 0 || mw <= 0 {
		return nil, fmt.Errorf("ocr: image of %dx%d merged tokens", mw, mh)
	}
	img, _ := t.ID(tokImage)
	var sb strings.Builder
	sb.WriteString(tokBegin + "User: " + tokImageStart)
	sb.WriteString(strings.Repeat(tokImage, mh*mw))
	sb.WriteString(tokImageEnd + text + "\nAssistant:\n")
	p := &Prompt{IDs: t.Encode(sb.String()), ImageAt: -1}
	p.Pos = make([][3]int32, len(p.IDs))
	var pos int32
	for i := 0; i < len(p.IDs); {
		if p.IDs[i] != img {
			p.Pos[i] = [3]int32{pos, pos, pos}
			pos++
			i++
			continue
		}
		if p.ImageAt >= 0 {
			return nil, fmt.Errorf("ocr: the prompt text holds an image token")
		}
		n := 0
		for i+n < len(p.IDs) && p.IDs[i+n] == img {
			n++
		}
		if n != mh*mw {
			return nil, fmt.Errorf("ocr: %d image tokens for a %dx%d image", n, mw, mh)
		}
		p.ImageAt, p.ImageLen = i, n
		for r := 0; r < mh; r++ {
			for c := 0; c < mw; c++ {
				p.Pos[i+r*mw+c] = [3]int32{pos, pos + int32(r), pos + int32(c)}
			}
		}
		pos += int32(max(mh, mw))
		i += n
	}
	p.Next = pos
	return p, nil
}

// EOS is `</s>`, the generation's end.
func (t *Tokenizer) EOS() int32 {
	id, _ := t.ID(tokEOS)
	return id
}
