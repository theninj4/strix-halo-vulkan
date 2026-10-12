package lm

// Understand mode (MUSIC.md A13), upstream's understand_music and its API's
// full_analysis_only: given a song's 5 Hz audio codes, the LM writes the
// caption, the metas, genres and the lyrics it hears -- the thinking path
// run backwards. It is sample mode's pass (create_sample_from_query) with a
// different prompt: the same FSM phase ("understand", genres generated, no
// stop at reasoning), no user metadata, one row, no CFG.

import (
	"fmt"
	"math/rand/v2"
	"strings"
)

// UnderstandInstruction is DEFAULT_LM_UNDERSTAND_INSTRUCTION.
const UnderstandInstruction = "Understand the given musical conditions and describe the audio semantics accordingly:"

// UnderstandSampling is what upstream's API analyses a song with
// (job_analysis_runtime.py): temperature 0.3, top-p off. Gradio's button
// and understand_music's default take 0.85.
var UnderstandSampling = Sampling{Temperature: 0.3, TopP: 1}

// UnderstandPrompt is build_formatted_prompt_for_understanding as token
// ids: the chat template with the codes as the whole user turn. The codes
// are spliced in as ids rather than encoded from their text: each is one
// added token, and an added token splits the text around it, so the two
// are the same ids (TestUnderstandPrompt), but the tokenizer takes tens of
// seconds over a song's worth of them.
func (p *Planner) UnderstandPrompt(codes []int) ([]int32, error) {
	head, err := p.Tk.Encode(strings.TrimSuffix(chatPrompt(UnderstandInstruction, ""), understandTail))
	if err != nil {
		return nil, err
	}
	tail, err := p.Tk.Encode(understandTail)
	if err != nil {
		return nil, err
	}
	ids := make([]int32, 0, len(head)+len(codes)+len(tail))
	ids = append(ids, head...)
	for _, c := range codes {
		if c < 0 || c >= NumCodes {
			return nil, fmt.Errorf("lm: audio code %d is outside [0, %d)", c, NumCodes)
		}
		ids = append(ids, int32(CodeBase+c))
	}
	return append(ids, tail...), nil
}

// understandTail is the chat template after the user turn.
const understandTail = "<|im_end|>\n<|im_start|>assistant\n"

// Understand is understand_audio_from_codes: the song upstream's LM hears
// in the codes, and the text it wrote. It needs the genres vocabulary, as
// sample mode does. Lyrics are what the LM wrote after `</think>`, with no
// "[Instrumental]" put in their place when it wrote none (upstream puts
// none there either).
func (p *Planner) Understand(codes []int, rng *rand.Rand) (Song, string, Stats, error) {
	var st Stats
	if !p.HasGenres() {
		return Song{}, "", st, fmt.Errorf("lm: understand mode needs the genres vocabulary (LoadGenres)")
	}
	if len(codes) == 0 {
		return Song{}, "", st, fmt.Errorf("lm: no audio codes to understand")
	}
	ids, err := p.UnderstandPrompt(codes)
	if err != nil {
		return Song{}, "", st, err
	}
	if len(ids) >= p.G.o.MaxLen {
		return Song{}, "", st, fmt.Errorf("lm: %d codes leave no room in a %d-position cache", len(codes), p.G.o.MaxLen)
	}
	text, st, err := p.decode(ids, newFSM(p.v, Meta{}, true), 3, rng)
	if err != nil {
		return Song{}, "", st, err
	}
	return SongOf(ParseCoT(text), ExtractLyrics(text), false), text, st, nil
}
