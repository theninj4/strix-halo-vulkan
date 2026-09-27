// Package plan is ACE-Step 1.5's request plan (MUSIC.md A1): everything a
// text2music request decides around the transformer -- the caption and lyric
// prompts the text encoder reads, the latent length a duration becomes, the
// turbo schedule, the sampler's Euler/x0 update and its DCW correction, and
// the FSQ codebook the LM's audio codes index. No weights are touched here
// except FSQ's 6 → 2048 output projection, which the caller supplies.
//
// Every rule is upstream's (ace-step/ACE-Step-1.5 at ca1e85f, driven through
// its public `inference.generate_music`), gated against
// reference/dump_ace_plan.py. Several of them are quirks, reproduced on
// purpose and named where they happen: the metas carry `int(duration)` while
// the latent length rounds to a tenth first; an unset duration means 120 s of
// audio under metas that say 30; and chunk_mask_mode "auto"'s 2.0 is written
// into a bool tensor, so the DiT sees 1.0.
package plan

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

const (
	SampleRate     = 48000
	Hop            = 1920 // samples a latent: the VAE's 2·4·4·6·10
	LatentRate     = SampleRate / Hop
	LatentChannels = 64

	// MinLatents is the shortest target upstream builds (5.12 s):
	// `max(128, max_latent_length)` in conditioning_target.
	MinLatents = 128
	// FallbackSeconds is the length of a request with no duration and no LM
	// (padding_utils, issue #929).
	FallbackSeconds = 120.0
	// MinSeconds and MaxSeconds are upstream's DURATION_MIN/MAX.
	MinSeconds = 10
	MaxSeconds = 600

	TextMaxTokens  = 256
	LyricMaxTokens = 2048
	// TimbreLatents is the silence slice used as the timbre reference when a
	// request has no reference audio: 30 s.
	TimbreLatents = 750

	DefaultInstruction = "Fill the audio semantic mask based on the given conditions:"
	EndOfText          = "<|endoftext|>"
	// ChunkMask is the value the DiT's chunk-mask channels carry for a whole
	// text2music target. See the package comment for why it is not 2.
	ChunkMask = 1.0
)

// Request is a DiT-only text2music request as the pipeline sees it after the
// LM (if any) has filled in what it filled in. Zero values mean "not given".
type Request struct {
	Caption  string
	Lyrics   string
	BPM      int
	KeyScale string
	// TimeSignature is passed through as given ("4", "3", "4/4"...).
	TimeSignature string
	// Duration is in seconds; ≤ 0 means unset.
	Duration float64
	// Language is the vocal language code; "" is upstream's "unknown".
	Language string
	// Instruction overrides DefaultInstruction; a missing trailing colon is
	// added, as upstream's _format_instruction does.
	Instruction string
}

var sftCaption = regexp.MustCompile(`(?s)#\s*Caption\s*\n(.*?)(?:\n\s*#\s*Metas|$)`)

// caption is extract_caption_from_sft_format: a caption that is itself a
// whole SFT prompt contributes only its caption section.
func (r *Request) caption() string {
	c := r.Caption
	if strings.Contains(c, "# Instruction") && strings.Contains(c, "# Caption") {
		if m := sftCaption.FindStringSubmatch(c); m != nil {
			return strings.TrimSpace(m[1])
		}
	}
	return c
}

// Metas is the metadata block of the caption prompt: _build_metadata_dict
// then _dict_to_meta_string. Duration is truncated, not rounded, and an
// unset one reads "30 seconds" whatever length is generated.
func (r *Request) Metas() string {
	bpm := "N/A"
	if r.BPM != 0 {
		bpm = strconv.Itoa(r.BPM)
	}
	ks := "N/A"
	if strings.TrimSpace(r.KeyScale) != "" {
		ks = r.KeyScale
	}
	ts := "N/A"
	if strings.TrimSpace(r.TimeSignature) != "" && r.TimeSignature != "N/A" {
		ts = r.TimeSignature
	}
	dur := 30
	if r.Duration > 0 {
		dur = int(r.Duration)
	}
	return fmt.Sprintf("- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n", bpm, ts, ks, dur)
}

// CaptionPrompt is SFT_GEN_PROMPT filled in: the text the caption encoder
// reads.
func (r *Request) CaptionPrompt() string {
	ins := r.Instruction
	if ins == "" {
		ins = DefaultInstruction
	}
	if !strings.HasSuffix(ins, ":") {
		ins += ":"
	}
	return "# Instruction\n" + ins + "\n\n# Caption\n" + r.caption() + "\n\n# Metas\n" + r.Metas() + EndOfText + "\n"
}

// LyricsPrompt is _format_lyrics: the text whose token embeddings the lyric
// encoder reads.
func (r *Request) LyricsPrompt() string {
	lang := r.Language
	if lang == "" {
		lang = "unknown"
	}
	return "# Languages\n" + lang + "\n\n# Lyric\n" + r.Lyrics + EndOfText
}

// Seconds is the length a request generates: its duration rounded to a
// tenth (Python's round-half-even on the exact binary value, which
// strconv's fixed-precision formatting also does), or the fallback.
func (r *Request) Seconds() float64 {
	if r.Duration <= 0 {
		return FallbackSeconds
	}
	s, _ := strconv.ParseFloat(strconv.FormatFloat(r.Duration, 'f', 1, 64), 64)
	return math.Max(0.1, s)
}

// LatentLength is the target's latent count: `int(seconds·48000) // 1920`,
// at least MinLatents. A request under 5.12 s is generated at 5.12 s;
// whether upstream trims the audio back afterwards is A6's to check.
func (r *Request) LatentLength() int {
	return max(MinLatents, int(r.Seconds()*SampleRate)/Hop)
}

// EndOfTextID is <|endoftext|> in the Qwen3 vocabulary.
const EndOfTextID = 151643

// Tokens is the text encoder's tokenizer call on an encoded prompt: its
// post-processor appends <|endoftext|> (Qwen3-Embedding pools the last
// token), and `truncation=True, max_length=n` keeps room for it -- so a long
// prompt keeps its first n−1 ids and still ends in the appended token, while
// the <|endoftext|> written into the prompt text is what gets cut.
func Tokens(ids []int32, n int) []int32 {
	if len(ids) > n-1 {
		ids = ids[:n-1]
	}
	return append(append([]int32(nil), ids...), EndOfTextID)
}
