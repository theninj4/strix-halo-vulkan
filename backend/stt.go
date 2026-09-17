package backend

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/audio"
	"strix-halo-vulkan/parakeet"
	"strix-halo-vulkan/vk"
)

// STTOptions is what cmd/serve's flags come to.
type STTOptions struct {
	// Model is the parakeet-tdt-0.6b-v3 checkpoint directory.
	Model string
	// Device, when set, stages the encoder and the transducer tail on it at
	// load time and keeps them there.
	Device *Device
	// MaxSeconds is the longest clip the device arenas are sized for. It is
	// a residency decision, not a policy one: the encoder's attention is
	// quadratic in the clip and its largest subsampling tensor grows with
	// it, so the arenas are built once for a ceiling and every shorter clip
	// runs inside them. A longer request is refused rather than silently
	// truncated.
	MaxSeconds float64
	// ID is the model id this backend answers to in /v1/models.
	ID string
}

const (
	defaultSTTModelID = "parakeet-tdt-0.6b-v3"
	defaultMaxSeconds = 60
)

// STT is the parakeet adapter: an api.TranscriptionBackend over the model
// cmd/asr drives.
//
// Unlike kokoro, this one is genuinely resident. The encoder is staged for a
// maximum clip length and every shorter request runs in the same arenas
// (UploadMel settles the geometry per run), so a request costs a front end on
// the host and two submits -- no staging, no weight upload.
type STT struct {
	opt   STTOptions
	id    string
	model *parakeet.Model

	mu  sync.Mutex
	enc *parakeet.GPUEncoder
	dec *parakeet.GPUDecoder
}

// NewSTT loads the checkpoint and, when a device was given, stages the whole
// model on it.
func NewSTT(opt STTOptions) (*STT, error) {
	if opt.ID == "" {
		opt.ID = defaultSTTModelID
	}
	if opt.MaxSeconds <= 0 {
		opt.MaxSeconds = defaultMaxSeconds
	}
	m, err := parakeet.Load(opt.Model)
	if err != nil {
		return nil, fmt.Errorf("backend: loading %s: %w", opt.Model, err)
	}
	s := &STT{opt: opt, id: opt.ID, model: m}
	if opt.Device == nil {
		return s, nil
	}

	frames := s.maxEncoderFrames()
	err = opt.Device.Do(func(dev *vk.Device) error {
		enc, err := parakeet.NewGPUEncoder(dev, m.Encoder, frames, nil)
		if err != nil {
			return fmt.Errorf("staging the encoder: %w", err)
		}
		s.enc = enc
		dec, err := parakeet.NewGPUDecoder(dev, m, enc.MaxFrames())
		if err != nil {
			return fmt.Errorf("staging the transducer tail: %w", err)
		}
		s.dec = dec
		// Attached, so the [T, 1024] hidden states never come back to the
		// host between the encoder and the tail (SPEECH.md S8).
		return dec.Attach(enc)
	})
	if err != nil {
		s.Close()
		return nil, fmt.Errorf("backend: %w", err)
	}
	return s, nil
}

// maxEncoderFrames is how many 80 ms encoder frames MaxSeconds of audio
// becomes.
//
// It reproduces audio.STFT.Frames and Subsampling.ValidLength rather than
// running the front end over a dummy clip, because the answer depends on
// nothing but the configuration: a centred STFT of n samples is
// 1 + n/hop frames, and the subsampling stack's two strided convolutions take
// that to one frame per 80 ms.
func (s *STT) maxEncoderFrames() int {
	f := s.model.Config.Features
	samples := int(s.opt.MaxSeconds * float64(f.SamplingRate))
	mel := 1 + samples/f.HopLength
	return s.model.Encoder.Subsampling.ValidLength(mel)
}

// Models reports the one model this backend serves.
func (s *STT) Models() []api.Model {
	return []api.Model{{ID: s.id, Object: "model", OwnedBy: "local"}}
}

// Close releases the device residency.
func (s *STT) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.opt.Device == nil {
		return
	}
	_ = s.opt.Device.Do(func(*vk.Device) error {
		if s.dec != nil {
			s.dec.Destroy()
			s.dec = nil
		}
		if s.enc != nil {
			s.enc.Destroy()
			s.enc = nil
		}
		return nil
	})
}

// Transcribe runs one clip.
//
// The context is checked on the way in and not again: neither the front end
// nor the graph has a cancellation point, so a client that hangs up mid-run
// still costs the run. What it does not cost is the reply, which
// api.backendError drops.
//
// The request's language is not used: this checkpoint detects it, and there
// is nothing in the graph to condition on a hint. The prompt and the
// temperature are not used either -- a TDT transducer's greedy decode has
// neither an initial context nor a sampler.
func (s *STT) Transcribe(ctx context.Context, clip *audio.Clip,
	_ *api.TranscriptionRequest,
) (*api.TranscriptionResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	want := s.model.Config.Features.SamplingRate
	if clip.Rate != want {
		// Resampling is a signal-processing decision and doing it silently
		// at the door would hide a client sending the wrong thing.
		return nil, fmt.Errorf("the clip is %d Hz and this model wants %d: %w",
			clip.Rate, want, api.ErrUnsupported)
	}

	if s.enc == nil {
		out, err := s.model.Transcribe(clip)
		if err != nil {
			return nil, err
		}
		return s.response(out, clip)
	}

	// The front end stays on the host: it is a few thousand 512-point FFTs,
	// and it is the pipeline's largest remaining host cost rather than a
	// device one (SPEECH.md S10).
	feats, err := s.model.FrontEnd.Features(clip)
	if err != nil {
		return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
	}
	if got := s.model.Encoder.Subsampling.ValidLength(feats.Frames); got > s.enc.MaxFrames() {
		return nil, fmt.Errorf(
			"the clip is %.1f s, and this server staged its arenas for %.0f s (%d encoder frames, not %d): %w",
			clip.Duration(), s.opt.MaxSeconds, s.enc.MaxFrames(), got, api.ErrUnsupported)
	}

	var out *parakeet.Transcript
	err = s.opt.Device.Do(func(*vk.Device) error {
		mel := &parakeet.Mat{Rows: feats.Frames, Cols: feats.Mels, Data: feats.Data}
		if _, err := s.enc.RunMel(mel, feats.Valid); err != nil {
			return err
		}
		// Resident: the encoder leaves its hidden states in the arena and
		// the tail reads them there, so the boundary between the two is a
		// fence and not a copy (SPEECH.md S8).
		t, err := s.dec.DecodeResident()
		if err != nil {
			return err
		}
		out = t
		return nil
	})
	if err != nil {
		return nil, err
	}
	return s.response(out, clip)
}

// response is the transcript with its timings, which every request gets and
// only `verbose_json` prints.
//
// They are computed here rather than in the handler because grouping tokens
// into words is a fact about this tokenizer -- SentencePiece marks a word
// start with U+2581 -- and `api` has no business knowing it. The cost is a
// walk over a few hundred emissions.
func (s *STT) response(t *parakeet.Transcript, clip *audio.Clip) (*api.TranscriptionResponse, error) {
	words, err := s.words(t, clip.Duration())
	if err != nil {
		return nil, err
	}
	return &api.TranscriptionResponse{
		Text:     t.Text,
		Duration: milliseconds(clip.Duration()),
		Words:    words,
		Segments: segments(words),
	}, nil
}

// secondsPerFrame is what an encoder frame is worth, derived rather than
// written down: the STFT's hop, times every stride in the subsampling stack,
// over the sample rate. For this checkpoint it is 160 * 8 / 16000 = 80 ms.
func (s *STT) secondsPerFrame() float64 {
	f := s.model.Config.Features
	stride := 1
	for _, c := range s.model.Encoder.Subsampling.Convs {
		if c.Stride > 1 {
			stride *= c.Stride
		}
	}
	return float64(f.HopLength) * float64(stride) / float64(f.SamplingRate)
}

// words groups the transducer's emissions into words and times them.
//
// **These are the model's own timings and not a forced alignment.** Every
// emission records the encoder frame it was made at, and a TDT transducer
// also predicts how many frames to skip afterwards -- that prediction is
// what gives a word an end rather than only a beginning. It is a duration
// head, so it is approximate in a way a Viterbi alignment against the audio
// would not be, and 80 ms is the finest it can be.
func (s *STT) words(t *parakeet.Transcript, duration float64) ([]api.TranscriptionWord, error) {
	spf := s.secondsPerFrame()
	tok := s.model.Tokenizer

	// The steps that begin each word, by the marker the tokenizer's own
	// decoder reads.
	var starts []int
	for i, step := range t.Steps {
		piece, err := tok.Piece(step.Token)
		if err != nil {
			return nil, err
		}
		if i == 0 || strings.HasPrefix(piece, parakeet.Metaspace) {
			starts = append(starts, i)
		}
	}

	out := make([]api.TranscriptionWord, 0, len(starts))
	for w, first := range starts {
		last := len(t.Steps) - 1
		if w+1 < len(starts) {
			last = starts[w+1] - 1
		}
		ids := make([]int, 0, last-first+1)
		for _, step := range t.Steps[first : last+1] {
			ids = append(ids, step.Token)
		}
		text, err := tok.Decode(ids)
		if err != nil {
			return nil, err
		}
		if text == "" {
			continue
		}
		end := float64(t.Steps[last].Frame+t.Steps[last].Duration) * spf
		out = append(out, api.TranscriptionWord{
			Word:  text,
			Start: milliseconds(min(float64(t.Steps[first].Frame)*spf, duration)),
			End:   milliseconds(min(end, duration)),
		})
	}
	// A word whose duration head said nothing ends where the next one
	// starts, rather than at the instant it began.
	for i := range out {
		if out[i].End > out[i].Start {
			continue
		}
		if i+1 < len(out) {
			out[i].End = out[i+1].Start
		} else {
			out[i].End = milliseconds(duration)
		}
	}
	return out, nil
}

// milliseconds rounds a time to the millisecond, which is three orders finer
// than the 80 ms frame it came from and is there for one reason: 41 frames of
// 0.08 s is 3.2800000000000002 in binary floating point, and nobody reading a
// transcript should have to see that.
func milliseconds(t float64) float64 { return math.Round(t*1000) / 1000 }

// segments groups words into sentences, which is what a caller asking for
// segment timestamps wants and the only division this model offers: it
// punctuates, so a sentence end is a token and not a guess about silence.
func segments(words []api.TranscriptionWord) []api.TranscriptionSegment {
	var out []api.TranscriptionSegment
	var cur *api.TranscriptionSegment
	for _, w := range words {
		if cur == nil {
			out = append(out, api.TranscriptionSegment{ID: len(out), Start: w.Start})
			cur = &out[len(out)-1]
		}
		if cur.Text != "" {
			cur.Text += " "
		}
		cur.Text += w.Word
		cur.End = w.End
		if strings.HasSuffix(w.Word, ".") || strings.HasSuffix(w.Word, "?") ||
			strings.HasSuffix(w.Word, "!") {
			cur = nil
		}
	}
	return out
}
