package mm3

// The whole model (N7): caption and lyrics in, stereo at 44.1 kHz out, as
// upstream's modular pipeline runs it -- the prompt, the frame loop, the
// windows, the vocoder and the crops (decoders.py: 86 latents off the left
// of every window but the first, 258 off the right of every one but the
// last; then a clamp to [−1, 1]).

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/tokenizer"
)

const (
	// CropLeft and CropRight are decoders.py's crops, in latents.
	CropLeft  = 86
	CropRight = 258
	// MaxSeconds is the frame cap's song length.
	MaxSeconds = MaxFrames / FrameRate
)

// Pipeline is every part, resident: ~19 GB with the int8 LM.
type Pipeline struct {
	// Between, when set, is called between two submissions -- after every
	// frame, DiT step and window -- when nothing of this model's is on the
	// device. A server uses it to let other work in, and an error from it
	// (a cancelled request) abandons the song there.
	Between func() error

	Tok     *tokenizer.Tokenizer
	LM      *LM
	Depth   *Depth
	Mix     Mix
	DiT     *DiT
	Vocoder *Vocoder
}

// Load stages the model from root (models/MiniMax-Music3).
func Load(dev *vk.Device, root string, o Options) (*Pipeline, error) {
	p := &Pipeline{}
	var err error
	fail := func(e error) (*Pipeline, error) { p.Destroy(); return nil, e }
	if p.Tok, err = tokenizer.Load(filepath.Join(root, "tokenizer")); err != nil {
		return fail(err)
	}
	if p.LM, err = LoadLM(dev, filepath.Join(root, "language_model"), o); err != nil {
		return fail(err)
	}
	if p.Depth, err = LoadDepth(dev, filepath.Join(root, "rvq_depth_decoder"), o.Bank); err != nil {
		return fail(err)
	}
	if p.Mix, err = LoadMix(filepath.Join(root, "condition_encoder")); err != nil {
		return fail(err)
	}
	full := LatentsFor(WindowFrames)
	if p.DiT, err = LoadDiT(dev, root, full); err != nil {
		return fail(err)
	}
	if p.Vocoder, err = LoadVocoder(dev, filepath.Join(root, "vocoder"), full); err != nil {
		return fail(err)
	}
	return p, nil
}

// Destroy releases every part.
func (p *Pipeline) Destroy() {
	if p.LM != nil {
		p.LM.Destroy()
	}
	if p.Depth != nil {
		p.Depth.Destroy()
	}
	if p.DiT != nil {
		p.DiT.Destroy()
	}
	if p.Vocoder != nil {
		p.Vocoder.Destroy()
	}
}

// Request is one song.
type Request struct {
	Caption, Lyrics string
	// Seconds caps the song (the model chooses its length, up to this);
	// at most MaxSeconds.
	Seconds float64
	Seed    uint64
}

// Stats is where a song's time went, wall.
type Stats struct {
	PromptTokens, Frames, Windows int
	Ended                         bool // the LM ended the song (not the cap)
	Frames0, DiT, Vocoder         time.Duration
	Song                          *Song
}

// Progress reports a stage and how far through it a song is: "frames"
// (done of the cap), "planned" once (the frames made, of the cap), then
// "windows" (DiT steps of all the windows').
type Progress func(stage string, done, total int)

// Generate makes one song: stereo [2][samples] at SampleRate.
func (p *Pipeline) Generate(r Request, progress Progress) ([2][]float32, *Stats, error) {
	var out [2][]float32
	if r.Seconds <= 0 || r.Seconds > MaxSeconds {
		return out, nil, fmt.Errorf("mm3: %.1f s; 0 to %d", r.Seconds, MaxSeconds)
	}
	ids, unc, err := p.Prompt(r.Caption, r.Lyrics)
	if err != nil {
		return out, nil, err
	}
	st := &Stats{PromptTokens: len(ids)}
	rng := rand.New(rand.NewPCG(r.Seed, 0x6d6d33))
	maxFrames := int(r.Seconds * FrameRate)
	t0 := time.Now()
	song, err := Generate(p.LM, p.Depth, p.Mix, ids, unc, maxFrames, rng, func(n int) {
		if progress != nil {
			progress("frames", n, maxFrames)
		}
	}, p.Between)
	if err != nil {
		return out, st, err
	}
	st.Frames0 = time.Since(t0)
	st.Song, st.Frames, st.Ended = song, len(song.Cond), song.Ended
	if progress != nil {
		// The frames fix the song's length: the LM may end it early.
		progress("planned", len(song.Cond), maxFrames)
	}

	starts := WindowStarts(len(song.Cond))
	st.Windows = len(starts)
	var prev *Carry
	for k, s := range starts {
		e := min(s+WindowFrames, len(song.Cond))
		t1 := time.Now()
		cond, _, err := p.DiT.Condition(song.Cond[s:e])
		if err != nil {
			return out, st, err
		}
		L := len(cond) / CondChannels
		lat, carry, _, err := p.DiT.Denoise(cond, Noise(L, rng), prev, func(i int) error {
			if progress != nil {
				progress("windows", k*DiTSteps+i+1, len(starts)*DiTSteps)
			}
			if p.Between != nil {
				return p.Between()
			}
			return nil
		})
		if err != nil {
			return out, st, err
		}
		prev = carry
		t2 := time.Now()
		st.DiT += t2.Sub(t1)
		wav, _, err := p.Vocoder.Decode(lat)
		if err != nil {
			return out, st, err
		}
		st.Vocoder += time.Since(t2)
		if p.Between != nil {
			if err := p.Between(); err != nil {
				return out, st, err
			}
		}
		left, right := 0, 0
		if k > 0 {
			left = CropLeft * Hop
		}
		if k < len(starts)-1 {
			right = CropRight * Hop
		}
		n := L * Hop
		for ch := 0; ch < 2; ch++ {
			for _, v := range wav[ch][left : n-right] {
				out[ch] = append(out[ch], min(max(v, -1), 1))
			}
		}
	}
	return out, st, nil
}

// MaxSecondsFor is the longest song a prompt of n tokens leaves room for
// in the cache, capped at MaxSeconds.
func (p *Pipeline) MaxSecondsFor(n int) float64 {
	return min(float64(MaxSeconds), float64(p.LM.o.MaxLen-n-1)/FrameRate)
}

// Prompt is the request's CFG pair of token ids.
func (p *Pipeline) Prompt(caption, lyrics string) ([]int32, []int32, error) {
	return Prompt(p.Tok, caption, lyrics)
}

// WriteWAV writes stereo as a 16-bit PCM wav at SampleRate.
func WriteWAV(path string, s [2][]float32) error {
	n := len(s[0])
	data := make([]byte, 44+4*n)
	put := func(off int, v uint32) { binary.LittleEndian.PutUint32(data[off:], v) }
	copy(data[0:], "RIFF")
	put(4, uint32(36+4*n))
	copy(data[8:], "WAVEfmt ")
	put(16, 16)
	binary.LittleEndian.PutUint16(data[20:], 1) // PCM
	binary.LittleEndian.PutUint16(data[22:], 2)
	put(24, SampleRate)
	put(28, SampleRate*4)
	binary.LittleEndian.PutUint16(data[32:], 4)
	binary.LittleEndian.PutUint16(data[34:], 16)
	copy(data[36:], "data")
	put(40, uint32(4*n))
	for i := 0; i < n; i++ {
		for ch := 0; ch < 2; ch++ {
			v := int16(math.Round(float64(min(max(s[ch][i], -1), 1)) * 32767))
			binary.LittleEndian.PutUint16(data[44+4*i+2*ch:], uint16(v))
		}
	}
	return os.WriteFile(path, data, 0o644)
}
