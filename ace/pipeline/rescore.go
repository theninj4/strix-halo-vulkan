package pipeline

// Scoring a song after the fact (MUSIC.md A14e): the auto score of a song
// against a request other than the one that made it (a swapped caption or
// lyrics), or of audio no request made (noise, silence, a loop), and the
// LM's half over codes re-tokenized from the render rather than the LM's
// own plan (A-o16).

import (
	"fmt"
	"time"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/loops"
	"strix-halo-vulkan/ace/plan"
)

// Rescore is the auto score of a finished song: wav is interleaved 48 kHz
// stereo, lat its latents ([T, 64], time-major; nil encodes wav with seed),
// r the request as asked and d as the DiT got it (DiTRequest), meta the
// CoT's. q's LM half reads codes (the render's own, AudioCodes of lat,
// when codes is nil), and the alignment's context is their hints, as the
// thinking path's is; render is the LM's half over the render's codes
// whatever codes was. The LM must be loaded.
func (p *Pipeline) Rescore(r, d *plan.Request, meta lm.Meta, wav, lat []float32, codes []int, seed uint64) (q *Quality, render *lm.Score, err error) {
	if p.LM == nil {
		return nil, nil, fmt.Errorf("ace: the 5 Hz LM is not loaded")
	}
	t0 := time.Now()
	if lat == nil {
		if lat, err = p.encode(wav, seed); err != nil {
			return nil, nil, err
		}
	}
	T := len(lat) / plan.LatentChannels
	if T*plan.LatentChannels > len(p.silence) {
		return nil, nil, fmt.Errorf("ace: %d latents is past the %d-latent silence", T, len(p.silence)/plan.LatentChannels)
	}
	own, err := p.AudioCodes(lat)
	if err != nil {
		return nil, nil, err
	}
	if codes == nil {
		codes = own
	}
	hints, err := p.Hints(codes, T)
	if err != nil {
		return nil, nil, err
	}
	dp := &ditPlan{T: T, src: hints, timbre: p.silence[:plan.TimbreLatents*plan.LatentChannels]}
	if _, _, err := p.condition(d, dp.timbre); err != nil {
		return nil, nil, err
	}
	q = &Quality{}
	if q.Lyrics, err = p.LyricAlignment(d, lat, dp.context(hints), len(plan.Schedule(plan.DefaultShift, nil)), nil); err != nil {
		return nil, nil, err
	}
	conds := ScoreConditions(r, d, meta)
	sc, err := p.LM.Score(codes, conds, lm.ScoreScale)
	if err != nil {
		return nil, nil, err
	}
	q.LM = &sc
	rs, err := p.LM.Score(own, conds, lm.ScoreScale)
	if err != nil {
		return nil, nil, err
	}
	if p.Aes != nil {
		a, err := p.Aes.Score(wav)
		if err != nil {
			return nil, nil, err
		}
		q.Aesthetics = &a.Scores
	}
	if p.SongEval != nil {
		s, err := p.SongEval.Score(wav)
		if err != nil {
			return nil, nil, err
		}
		q.Song = &s.Scores
	}
	l := loops.Measure(wav)
	q.Loops = &l
	q.Render = &rs
	q.Took = time.Since(t0)
	return q, &rs, nil
}
