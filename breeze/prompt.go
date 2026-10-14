// Package breeze runs BreezeBlue's Breeze-TTS-2 on the device: a text
// encoder, an autoregressive backbone emitting one 16-codebook audio frame a
// step, a depth decoder filling codebooks 1-15 within a frame, and Qwen3-TTS's
// 12.5 Hz codec (research/tts-instruct-vertical.md, the B-stages).
package breeze

// The prompt (B1): upstream's four templates (breeze_infer/templates.py) and
// how `_prepare_one` turns their segments into ids.
//
// A text segment is encoded with add_special_tokens=True (a leading <bos>),
// decoded back to a string, and every segment's string is joined and encoded
// again without specials. Segments always start at an added token (<bos>, or
// an audio segment's <|AUDIO|>), which BPE never merges across, so the joint
// encode is each segment's re-encode in turn; the round trip itself is kept,
// because decode is not always encode's inverse (byte fallback that is not
// UTF-8 decodes to U+FFFD).

import (
	"fmt"

	"strix-halo-vulkan/gemma4"
)

// Token ids the model adds to Gemma-3's vocabulary (config.json).
const (
	BOS       = 2
	AudioTok  = 262144 // <|AUDIO|>: a frame of reference codes
	AudioEOS  = 262145 // <|audio_eos|>: the all-zero frame
	SpeakerS0 = 262146
	InsBOS    = 262156 // <ins_bos>
	InsEOS    = 262157 // <ins_eos>
)

// Template is one of upstream's four prompt shapes, chosen by which of
// instruction and reference a request has (select_template_name).
type Template int

const (
	TTSPlain       Template = iota // [S0]{text}
	TTSInstruction                 // [S0]<ins_bos>{instruction}<ins_eos>{text}; CFG negative: plain
	RefClone                       // [S0]{ref_text} <ref frames> <|audio_eos|> [S0]{text}
	RefEdit                        // the clone with the instruction; CFG negative: the clone
)

var templateNames = [...]string{"tts_plain", "tts_instruction", "ref_clone_tata", "ref_edit_tata"}

func (t Template) String() string { return templateNames[t] }

// Request is what a prompt is built from.
type Request struct {
	Text        string
	Instruction string
	// RefText is the reference audio's exact transcript and RefFrames its
	// codes, [frame][16]; both or neither.
	RefText   string
	RefFrames [][16]int32
}

// TemplateFor is select_template_name.
func TemplateFor(r *Request) Template {
	ref := len(r.RefFrames) > 0
	switch {
	case ref && r.Instruction != "":
		return RefEdit
	case ref:
		return RefClone
	case r.Instruction != "":
		return TTSInstruction
	}
	return TTSPlain
}

// Branch is one row of the backbone's prompt: its ids, which positions are
// text (each text segment encoded on its own), and the reference frames its
// <|AUDIO|> positions take, in order.
type Branch struct {
	IDs      []int32
	TextMask []bool
	Segments []int // each text segment's length, in order
	Frames   [][16]int32
}

// Prompt is a request's branches: the prompt, and the CFG negative when the
// template has one (nil otherwise).
type Prompt struct {
	Template Template
	Cond     Branch
	Uncond   *Branch
}

// segment is a text string, or (text == "" && audio) the reference audio.
type segment struct {
	text  string
	audio bool
}

func plainSegs(r *Request) []segment { return []segment{{text: "[S0]" + r.Text}} }

func instructionSegs(r *Request) []segment {
	return []segment{{text: "[S0]<ins_bos>" + r.Instruction + "<ins_eos>" + r.Text}}
}

func cloneSegs(r *Request) []segment {
	return []segment{{text: "[S0]" + r.RefText}, {audio: true}, {text: "[S0]" + r.Text}}
}

func editSegs(r *Request) []segment {
	return []segment{{text: "[S0]" + r.RefText}, {audio: true},
		{text: "[S0]<ins_bos>" + r.Instruction + "<ins_eos>" + r.Text}}
}

// BuildPrompt lays a request out as upstream does.
func BuildPrompt(tok *gemma4.Tokenizer, r *Request) (*Prompt, error) {
	if (r.RefText != "") != (len(r.RefFrames) > 0) {
		return nil, fmt.Errorf("breeze: a reference needs both its audio and its transcript")
	}
	t := TemplateFor(r)
	p := &Prompt{Template: t}
	var pos, neg []segment
	switch t {
	case TTSPlain:
		pos = plainSegs(r)
	case TTSInstruction:
		pos, neg = instructionSegs(r), plainSegs(r)
	case RefClone:
		pos = cloneSegs(r)
	case RefEdit:
		pos, neg = editSegs(r), cloneSegs(r)
	}
	p.Cond = buildBranch(tok, pos, r.RefFrames)
	if neg != nil {
		b := buildBranch(tok, neg, r.RefFrames)
		p.Uncond = &b
	}
	return p, nil
}

func buildBranch(tok *gemma4.Tokenizer, segs []segment, frames [][16]int32) Branch {
	var b Branch
	for _, s := range segs {
		if s.audio {
			for range frames {
				b.IDs = append(b.IDs, AudioTok)
				b.TextMask = append(b.TextMask, false)
			}
			b.IDs = append(b.IDs, AudioEOS)
			b.TextMask = append(b.TextMask, false)
			b.Frames = append(b.Frames, frames...)
			continue
		}
		rendered := tok.Decode(append([]int32{BOS}, tok.Encode(s.text)...), false)
		ids := tok.Encode(rendered)
		b.IDs = append(b.IDs, ids...)
		for range ids {
			b.TextMask = append(b.TextMask, true)
		}
		b.Segments = append(b.Segments, len(ids))
	}
	return b
}

// TextSegments returns each text segment's ids, in order: what the encoder
// runs on, one segment at a time from position 0.
func (b *Branch) TextSegments() [][]int32 {
	var text []int32
	for i, id := range b.IDs {
		if b.TextMask[i] {
			text = append(text, id)
		}
	}
	out := make([][]int32, 0, len(b.Segments))
	for _, n := range b.Segments {
		out = append(out, text[:n])
		text = text[n:]
	}
	return out
}
