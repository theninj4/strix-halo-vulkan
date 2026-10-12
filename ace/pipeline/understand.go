package pipeline

// Understand mode (MUSIC.md A13): captioning a song. Upstream's
// full_analysis_only (job_analysis_runtime.py) is convert_src_audio_to_codes
// -- the audio through the VAE encoder and the audio tokenizer, A11's
// cover path up to its codes -- and then understand_audio_from_codes, where
// the 5 Hz LM writes the caption, metas, genres and lyrics it hears in
// them (ace/lm Understand). Neither the DiT's forward nor the VAE decoder
// runs.

import (
	"fmt"
	"math/rand/v2"
	"time"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/plan"
)

// Understanding is what understand mode heard in a song.
type Understanding struct {
	Song lm.Song
	// Text is what the LM wrote, Codes what it read.
	Text  string
	Codes []int
	// Seconds is the song's length.
	Seconds float64
	Timings struct {
		Encode, Tokenize, Total time.Duration
		LM                      lm.Stats
	}
}

// AudioCodes is convert_src_audio_to_codes after the encode: the latents
// ([T, 64], time-major) through the audio tokenizer, 5 a code.
func (p *Pipeline) AudioCodes(lat []float32) ([]int, error) {
	T := len(lat) / plan.LatentChannels
	if T == 0 {
		return nil, fmt.Errorf("ace: no latents to tokenize")
	}
	ids, err := p.tokenize(lat, T)
	if err != nil {
		return nil, err
	}
	codes := make([]int, len(ids))
	for i, c := range ids {
		codes[i] = int(c)
	}
	return codes, nil
}

// Understand captions a song: wav is interleaved 48 kHz stereo
// (ReadAudio). o.Seed draws the VAE posterior's sample (upstream's is
// unseeded), o.LMSeed the LM's; o.Sampling overrides lm.UnderstandSampling;
// o.Between and o.Progress (stage "understand") see every LM step. The
// rest of o is the generation's and is not read.
func (p *Pipeline) Understand(wav []float32, o Options) (*Understanding, error) {
	start := time.Now()
	if p.LM == nil {
		return nil, fmt.Errorf("ace: the 5 Hz LM is not loaded")
	}
	if !p.LM.HasGenres() {
		return nil, fmt.Errorf("ace: understand mode needs the genres vocabulary: %v", p.GenresErr)
	}
	secs := float64(len(wav)/2) / plan.SampleRate
	switch {
	case len(wav)/2 < plan.Hop:
		return nil, fmt.Errorf("ace: the audio is shorter than one latent (%d samples)", plan.Hop)
	case secs > plan.MaxSeconds:
		return nil, fmt.Errorf("ace: the audio is %.0f s, past the %d s ceiling", secs, plan.MaxSeconds)
	case Silent(wav):
		return nil, fmt.Errorf("ace: the audio is silent")
	}
	u := &Understanding{Seconds: secs}
	t0 := time.Now()
	lat, err := p.encode(wav, o.Seed)
	if err != nil {
		return nil, err
	}
	u.Timings.Encode = time.Since(t0)
	if err := o.between(); err != nil {
		return nil, err
	}
	t0 = time.Now()
	if u.Codes, err = p.AudioCodes(lat); err != nil {
		return nil, err
	}
	u.Timings.Tokenize = time.Since(t0)
	p.LM.S = lm.UnderstandSampling
	if o.Sampling != nil {
		p.LM.S = *o.Sampling
	}
	p.LM.Between = o.Between
	p.LM.Step = func(_, done, total int) { o.progress("understand", done, total) }
	defer func() { p.LM.Between, p.LM.Step = nil, nil }()
	rng := rand.New(rand.NewPCG(o.LMSeed, 0x5a))
	if u.Song, u.Text, u.Timings.LM, err = p.LM.Understand(u.Codes, rng); err != nil {
		return nil, err
	}
	u.Timings.Total = time.Since(start)
	return u, nil
}
