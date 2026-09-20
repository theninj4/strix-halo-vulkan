package wyoming

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/audio"
)

// Options is what cmd/serve's flags come to.
type Options struct {
	// Speech and Transcription are the same two backends api.Server holds.
	// At least one must be set: a Wyoming server with neither advertises an
	// empty `info` and Home Assistant refuses to add it, so New says so at
	// startup instead.
	Speech        api.SpeechBackend
	Transcription api.TranscriptionBackend

	// Name is the program name in `info`, which is what Home Assistant
	// calls the entities it creates. Empty takes DefaultName.
	Name string

	// Voice is the pack a request that names none gets. Empty takes the
	// backend's own default, by sending no voice at all.
	Voice string

	// AllVoices advertises every pack in the checkpoint rather than only
	// the ones the loaded front end can pronounce. See Voices.
	AllVoices bool

	// MaxSeconds bounds a transcription: audio past it is not buffered and
	// the stream is answered with an error. It should be the same number
	// backend.STT was staged for, since a longer clip would be refused
	// there anyway -- the point of having it here too is that this side
	// holds the samples in memory while they arrive.
	MaxSeconds float64

	// OutputRate resamples synthesised speech before sending it. Zero sends
	// the model's own rate, which is what Home Assistant wants; a satellite
	// playing the audio directly may want its sound card's instead.
	OutputRate int

	// MaxConns bounds concurrent connections. Zero takes defaultMaxConns.
	MaxConns int
}

const (
	// DefaultName is the program name in `info`.
	DefaultName = "strix-halo"

	// DefaultPort is the port cmd/serve suggests. Wyoming has no registered
	// ports, but 10300 is what every speech-to-text service in the ecosystem
	// listens on and 10200 is text-to-speech; this server is both, and a
	// client discovers which from `info` rather than from the port, so one
	// is enough and 10300 is the one a Home Assistant user will try first.
	DefaultPort = 10300

	// defaultMaxSeconds matches backend.STT's own default ceiling.
	defaultMaxSeconds = 60

	// defaultMaxConns is a bound on goroutines and, more to the point, on
	// buffered audio: a connection receiving a transcription holds its whole
	// utterance, so this times MaxSeconds is the worst case. At the defaults
	// that is 32 x 60 s x 16 kHz x 4 bytes, about 123 MB.
	defaultMaxConns = 32

	// defaultASRRate is the rate this server resamples to when a
	// transcription backend does not name its own. It is parakeet's.
	defaultASRRate = 16000

	// ttsChunkSamples is how much audio one `audio-chunk` carries. At
	// kokoro's 24 kHz it is 43 ms, small enough that a satellite can start
	// playing immediately and large enough that the ~90-byte header is
	// noise against it.
	ttsChunkSamples = 1024

	// writeTimeout bounds one event write. A client that stops reading --
	// a satellite that lost power mid-utterance -- must not hold a
	// goroutine, and the GPU queue behind it, forever.
	writeTimeout = 30 * time.Second
)

// rateProvider is a transcription backend that names the rate it reads.
//
// It is an optional interface rather than a method on
// api.TranscriptionBackend because the HTTP endpoint does not need it: there
// a clip arrives with a rate in its own header and a mismatch is a 400 that
// tells the caller to convert. Wyoming has no such conversation -- the client
// sends what its microphone gives it and expects a transcript -- so this side
// resamples, and to resample it has to know the target.
type rateProvider interface {
	SampleRate() int
}

// Server answers the Wyoming protocol for whichever speech backends the
// process loaded.
//
// It holds no state per client beyond the connection: `info` is built once at
// New and every request is answered from the same two backends the HTTP API
// uses, which serialise themselves. What that means in practice is that two
// Home Assistant pipelines talking to this port at once are two goroutines
// queueing on one GPU, exactly as two HTTP requests would be.
type Server struct {
	opt     Options
	info    *Info
	asrRate int

	conns atomic.Int64
	next  atomic.Int64

	mu     sync.Mutex
	open   map[net.Conn]struct{}
	closed bool
}

// New builds a server and the `info` message it will answer `describe` with.
//
// The info is built here rather than per request because it is a fact about
// what this process staged, and a client that polls `describe` -- Home
// Assistant does, to notice a voice being installed -- should see the same
// answer every time until the process is restarted.
func New(opt Options) (*Server, error) {
	if opt.Speech == nil && opt.Transcription == nil {
		return nil, errors.New("wyoming: neither a speech nor a transcription backend was loaded; " +
			"pass -tts and/or -stt beside -wyoming")
	}
	if opt.Name == "" {
		opt.Name = DefaultName
	}
	if opt.MaxSeconds <= 0 {
		opt.MaxSeconds = defaultMaxSeconds
	}
	if opt.MaxConns <= 0 {
		opt.MaxConns = defaultMaxConns
	}

	s := &Server{opt: opt, asrRate: defaultASRRate, open: map[net.Conn]struct{}{}}
	if r, ok := opt.Transcription.(rateProvider); ok && r.SampleRate() > 0 {
		s.asrRate = r.SampleRate()
	}

	info := &Info{
		ASR: []AsrProgram{}, TTS: []TtsProgram{},
		Handle: []any{}, Intent: []any{}, Wake: []any{}, Mic: []any{}, Snd: []any{},
	}
	if opt.Transcription != nil {
		id := modelID(opt.Transcription, "parakeet")
		desc := fmt.Sprintf("TDT transducer, %d languages", len(asrLanguages))
		info.ASR = append(info.ASR, AsrProgram{
			Artifact: Artifact{
				Name:        opt.Name,
				Description: strptr("Speech to text on Vulkan"),
				Attribution: parakeetAttribution,
				Installed:   true,
			},
			Models: []AsrModel{{
				Artifact: Artifact{
					Name:        id,
					Description: &desc,
					Attribution: parakeetAttribution,
					Installed:   true,
				},
				Languages: asrLanguages,
			}},
			// See AsrProgram: the transducer transcribes a clip somebody
			// else decided the ends of.
			RequiresExternalVAD:          true,
			PrefersAutoGainEnabled:       true,
			PrefersNoiseReductionEnabled: true,
		})
	}
	if opt.Speech != nil {
		voices := Voices(opt.Speech.Voices(), opt.AllVoices)
		if len(voices) == 0 {
			return nil, fmt.Errorf("wyoming: none of the %d voice packs this checkpoint holds "+
				"has a language the loaded front end can pronounce; pass -wyoming-all-voices to "+
				"advertise them anyway", len(opt.Speech.Voices()))
		}
		if opt.Voice != "" && !hasVoice(voices, opt.Voice) {
			// Not fatal: -voice may legitimately name a blend, which is
			// not a pack and never appears in the list. It is worth a line
			// because the other reason it is missing is a typo.
			log.Printf("wyoming: the default voice %q is not one of the %d advertised packs; "+
				"a request that names no voice will still get it", opt.Voice, len(voices))
		}
		info.TTS = append(info.TTS, TtsProgram{
			Artifact: Artifact{
				Name:        opt.Name,
				Description: strptr("Text to speech on Vulkan"),
				Attribution: kokoroAttribution,
				Installed:   true,
			},
			Voices: voices,
			// See TtsProgram: the whole utterance is faster than the first
			// sentence of it would be to play.
			SupportsSynthesizeStreaming: false,
		})
	}
	s.info = info
	return s, nil
}

// Info is what `describe` is answered with. It is exported so cmd/serve can
// log what it is about to advertise rather than describing it twice.
func (s *Server) Info() *Info { return s.info }

// Serve accepts connections until ctx is cancelled or the listener fails.
//
// Shutdown is the same shape as the HTTP server's drain and for the same
// reason: a request in flight owns the GPU queue. Cancelling ctx closes the
// listener and pushes every open connection's read deadline into the past, so
// an idle client unwinds at once and one mid-utterance finishes its model run
// first; Serve returns when the last of them has.
func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	var wg sync.WaitGroup
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			_ = ln.Close()
			s.drain()
		case <-done:
		}
	}()

	for {
		c, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil {
				wg.Wait()
				return nil
			}
			return err
		}
		if n := s.conns.Add(1); n > int64(s.opt.MaxConns) {
			s.conns.Add(-1)
			log.Printf("wyoming: refusing %s: %d connections already open (-wyoming-max-conns)",
				c.RemoteAddr(), s.opt.MaxConns)
			_ = c.Close()
			continue
		}
		if !s.track(c) {
			// Shutdown started between Accept and here.
			s.conns.Add(-1)
			_ = c.Close()
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer s.conns.Add(-1)
			defer s.untrack(c)
			s.handle(ctx, c)
		}()
	}
}

// track registers an open connection so drain can reach it, and reports
// whether the server was still accepting.
func (s *Server) track(c net.Conn) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return false
	}
	s.open[c] = struct{}{}
	return true
}

func (s *Server) untrack(c net.Conn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.open, c)
}

// drain unblocks every open connection's read without touching its write, so
// a connection that is in the middle of sending an utterance finishes sending
// it and then sees its next read fail.
func (s *Server) drain() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	for c := range s.open {
		_ = c.SetReadDeadline(time.Now())
	}
}

// conn is one client and whatever it is in the middle of.
//
// The audio fields are the only real state: a transcription arrives as a
// `transcribe`, then an `audio-start`, then chunks, and the samples have to
// be somewhere until `audio-stop` says the utterance is over.
type conn struct {
	s   *Server
	id  int64
	net net.Conn
	r   *bufio.Reader
	w   *bufio.Writer

	// language is what the last `transcribe` asked for, logged and not used.
	language string

	// receiving is set between audio-start (or the first chunk) and
	// audio-stop. format is what that stream declared; a chunk that
	// disagrees with it is refused rather than converted.
	receiving bool
	format    audioFormat
	samples   []float32
	// overflowed is set when the stream went past MaxSeconds. The rest of
	// it is read and dropped so the client is not left writing into a
	// stalled socket, and audio-stop answers with the error.
	overflowed bool
}

func (s *Server) handle(ctx context.Context, nc net.Conn) {
	c := &conn{
		s:   s,
		id:  s.next.Add(1),
		net: nc,
		r:   bufio.NewReader(nc),
		w:   bufio.NewWriter(nc),
	}
	defer func() { _ = nc.Close() }()
	c.logf("connected from %s", nc.RemoteAddr())

	for {
		ev, err := ReadEvent(c.r)
		if err != nil {
			switch {
			case errors.Is(err, net.ErrClosed), isTimeout(err):
				// Our own shutdown, or the deadline drain set.
				c.logf("disconnected")
			case isEOF(err):
				c.logf("disconnected")
			default:
				c.logf("dropping the connection: %v", err)
			}
			return
		}
		if err := c.dispatch(ctx, ev); err != nil {
			c.logf("dropping the connection: %v", err)
			return
		}
	}
}

// dispatch handles one event. An error from it is a *transport* failure and
// ends the connection; a failure to answer the request is sent back as an
// `error` event and the connection carries on, because a client that asked
// for an unknown voice should be told so and then be able to ask again.
func (c *conn) dispatch(ctx context.Context, ev *Event) error {
	switch ev.Type {
	case typeDescribe:
		return c.write(typeInfo, c.s.info)

	case typePing:
		var p pingData
		if err := ev.Unmarshal(&p); err != nil {
			return c.fail(err)
		}
		return c.write(typePong, p)

	case typeSelectProgram:
		var sel selectProgramData
		if err := ev.Unmarshal(&sel); err != nil {
			return c.fail(err)
		}
		// One program per domain is advertised, so the only thing to do
		// with this is check that it is the one.
		if sel.Name != "" && sel.Name != c.s.opt.Name {
			return c.fail(fmt.Errorf("no program named %q; this server advertises %q",
				sel.Name, c.s.opt.Name))
		}
		return nil

	case typeTranscribe:
		var t transcribeData
		if err := ev.Unmarshal(&t); err != nil {
			return c.fail(err)
		}
		if c.s.opt.Transcription == nil {
			return c.fail(errors.New("this server was started without -stt and cannot transcribe"))
		}
		c.language = t.Language
		c.reset()
		return nil

	case typeAudioStart:
		var f audioFormat
		if err := ev.Unmarshal(&f); err != nil {
			return c.fail(err)
		}
		if err := checkFormat(f); err != nil {
			return c.fail(err)
		}
		c.reset()
		c.receiving = true
		c.format = f
		return nil

	case typeAudioChunk:
		return c.audioChunk(ev)

	case typeAudioStop:
		return c.audioStop(ctx)

	case typeSynthesize:
		return c.synthesize(ctx, ev)

	default:
		// The protocol asks a server to ignore what it does not know, and
		// that is how a client several versions newer stays compatible
		// with this one. It is logged once per event so an unexpected
		// conversation is diagnosable.
		c.logf("ignoring a %q event", ev.Type)
		return nil
	}
}

// audioChunk appends one chunk of PCM to the utterance being received.
func (c *conn) audioChunk(ev *Event) error {
	var f audioFormat
	if err := ev.Unmarshal(&f); err != nil {
		return c.fail(err)
	}
	if err := checkFormat(f); err != nil {
		return c.fail(err)
	}
	if !c.receiving {
		// A chunk may open a stream: it carries the whole format, and not
		// every client sends `audio-start` first.
		c.reset()
		c.receiving = true
		c.format = f
	}
	if f.Rate != c.format.Rate || f.Width != c.format.Width || f.Channels != c.format.Channels {
		// Refused rather than converted. Concatenating two rates would
		// give a clip whose timeline is wrong in a way the transcript
		// would not obviously show, and no client does this by accident.
		err := fmt.Errorf("this chunk is %d Hz, %d-byte, %d-channel and the stream began as %d Hz, %d-byte, %d-channel",
			f.Rate, f.Width, f.Channels, c.format.Rate, c.format.Width, c.format.Channels)
		c.reset()
		return c.fail(err)
	}
	if c.overflowed {
		return nil
	}
	limit := int(c.s.opt.MaxSeconds * float64(c.format.Rate))
	samples, err := appendPCM(c.samples, ev.Payload, f.Width, f.Channels)
	if err != nil {
		c.reset()
		return c.fail(err)
	}
	if len(samples) > limit {
		c.overflowed = true
		c.samples = nil
		return nil
	}
	c.samples = samples
	return nil
}

// audioStop ends the utterance and answers with a transcript.
func (c *conn) audioStop(ctx context.Context) error {
	if !c.receiving {
		// Nothing was being received; a stray stop is not worth an error.
		return nil
	}
	overflowed, rate, samples := c.overflowed, c.format.Rate, c.samples
	c.reset()

	if overflowed {
		return c.fail(fmt.Errorf("the utterance went past %.0f s, which is what this server buffers "+
			"and what its encoder arenas are staged for (-max-audio)", c.s.opt.MaxSeconds))
	}
	if c.s.opt.Transcription == nil {
		return c.fail(errors.New("this server was started without -stt and cannot transcribe"))
	}
	if len(samples) == 0 {
		// Silence in, nothing said. An empty transcript is the honest
		// answer and is what Home Assistant expects for a pipeline that
		// heard nothing.
		return c.write(typeTranscript, transcriptData{Text: ""})
	}

	clip := &audio.Clip{Samples: samples, Rate: rate}
	if rate != c.s.asrRate {
		// The HTTP endpoint refuses a clip at the wrong rate, because
		// there the caller chose the file and can convert it. Here the
		// caller is a microphone and there is no conversation to have, so
		// the filter runs on this side -- see audio.Resample for why it is
		// a filter and not an interpolation.
		converted, err := audio.Resample(clip, c.s.asrRate)
		if err != nil {
			return c.fail(err)
		}
		c.logf("resampled %.2f s from %d Hz to %d", clip.Duration(), rate, c.s.asrRate)
		clip = converted
	}

	start := time.Now()
	resp, err := c.s.opt.Transcription.Transcribe(ctx, clip, &api.TranscriptionRequest{
		Language: c.language,
	})
	if err != nil {
		return c.fail(err)
	}
	// The same line the HTTP handler logs, so the two doors are comparable
	// in one journal.
	took := time.Since(start)
	c.logf("transcribe: %.2f s of audio at %d Hz -> %d characters, in %v (%.1fx real time)",
		clip.Duration(), clip.Rate, len(resp.Text), took.Round(time.Millisecond),
		ratio(clip.Duration(), took))
	return c.write(typeTranscript, transcriptData{Text: resp.Text})
}

// synthesize answers a `synthesize` with an audio stream.
func (c *conn) synthesize(ctx context.Context, ev *Event) error {
	var req synthesizeData
	if err := ev.Unmarshal(&req); err != nil {
		return c.fail(err)
	}
	if c.s.opt.Speech == nil {
		return c.fail(errors.New("this server was started without -tts and cannot synthesise"))
	}
	text := strings.TrimSpace(req.Text)
	if text == "" {
		return c.fail(errors.New("there is no text to speak"))
	}
	if f := req.TextFormat; f != "" && f != "text" {
		// SSML is the other value the protocol defines. Kokoro's front end
		// takes text or IPA and has no notion of a markup language, and
		// speaking the tags aloud would be worse than saying so.
		return c.fail(fmt.Errorf("text_format %q is not supported; this server speaks plain text", f))
	}

	voice := c.s.opt.Voice
	if v := req.Voice; v != nil {
		if v.Speaker != "" {
			return c.fail(fmt.Errorf("voice %q has no speaker %q: a kokoro voice pack is one speaker, "+
				"and a blend of packs is spelled in the voice name instead (\"af_bella,af_sky\")",
				v.Name, v.Speaker))
		}
		switch {
		case v.Name != "":
			voice = v.Name
		case v.Language != "":
			name, ok := c.voiceForLanguage(v.Language)
			if !ok {
				return c.fail(fmt.Errorf("no advertised voice speaks %q", v.Language))
			}
			voice = name
		}
	}

	start := time.Now()
	clip, err := c.s.opt.Speech.Speak(ctx, &api.SpeechRequest{Input: text, Voice: voice, Speed: 1})
	if err != nil {
		return c.fail(err)
	}
	if clip == nil || len(clip.Samples) == 0 {
		return c.fail(errors.New("the speech backend returned no samples"))
	}
	if rate := c.s.opt.OutputRate; rate > 0 && rate != clip.Rate {
		if clip, err = audio.Resample(clip, rate); err != nil {
			return c.fail(err)
		}
	}
	took := time.Since(start)
	c.logf("synthesize: %d characters -> %.2f s of audio at %d Hz, voice %q, in %v (%.1fx real time)",
		len(text), clip.Duration(), clip.Rate, voice, took.Round(time.Millisecond),
		ratio(clip.Duration(), took))

	return c.writeClip(clip)
}

// writeClip sends one utterance as audio-start, chunks and audio-stop.
//
// The timestamps are the stream's own clock in milliseconds, which Home
// Assistant ignores and a satellite playing the audio directly does not.
func (c *conn) writeClip(clip *audio.Clip) error {
	zero := 0
	if err := c.write(typeAudioStart, audioFormat{
		Rate: clip.Rate, Width: 2, Channels: 1, Timestamp: &zero,
	}); err != nil {
		return err
	}
	for off := 0; off < len(clip.Samples); off += ttsChunkSamples {
		end := min(off+ttsChunkSamples, len(clip.Samples))
		ts := int(math.Round(float64(off) * 1000 / float64(clip.Rate)))
		ev, err := event(typeAudioChunk, audioFormat{
			Rate: clip.Rate, Width: 2, Channels: 1, Timestamp: &ts,
		})
		if err != nil {
			return err
		}
		ev.Payload = encodePCM16(clip.Samples[off:end])
		if err := c.writeEvent(ev); err != nil {
			return err
		}
	}
	total := int(math.Round(clip.Duration() * 1000))
	return c.write(typeAudioStop, audioStopData{Timestamp: &total})
}

// voiceForLanguage picks a pack for a request that named a language instead
// of a voice. The advertised list is sorted, so the choice is stable.
//
// An exact tag wins; failing that a tag with the same primary subtag does, so
// a pipeline asking for "en" gets an English voice rather than nothing.
func (c *conn) voiceForLanguage(lang string) (string, bool) {
	if len(c.s.info.TTS) == 0 {
		return "", false
	}
	want := strings.ToLower(lang)
	primary, _, _ := strings.Cut(want, "-")
	var fallback string
	for _, v := range c.s.info.TTS[0].Voices {
		for _, tag := range v.Languages {
			got := strings.ToLower(tag)
			if got == want {
				return v.Name, true
			}
			if p, _, _ := strings.Cut(got, "-"); p == primary && fallback == "" {
				fallback = v.Name
			}
		}
	}
	return fallback, fallback != ""
}

// reset drops whatever utterance was being received.
func (c *conn) reset() {
	c.receiving = false
	c.overflowed = false
	c.samples = nil
	c.format = audioFormat{}
}

// write sends one event built from a data value.
func (c *conn) write(typ string, data any) error {
	ev, err := event(typ, data)
	if err != nil {
		return err
	}
	return c.writeEvent(ev)
}

func (c *conn) writeEvent(ev *Event) error {
	if err := c.net.SetWriteDeadline(time.Now().Add(writeTimeout)); err != nil {
		return err
	}
	return WriteEvent(c.w, ev)
}

// fail answers the client with an `error` event and keeps the connection.
//
// Every failure goes back this way rather than by hanging up, because Home
// Assistant reports a closed connection as "connection lost" and reports an
// error event with its text -- so the difference between this and a dropped
// socket is the difference between "no voice named bm_geroge" in the log and
// a user with nothing to go on.
func (c *conn) fail(err error) error {
	c.logf("error: %v", err)
	return c.write(typeError, errorData{Text: err.Error()})
}

func (c *conn) logf(format string, args ...any) {
	log.Printf("wyoming[%d]: %s", c.id, fmt.Sprintf(format, args...))
}

// checkFormat rejects a PCM description this server cannot read. Width is in
// bytes: 1 is unsigned 8-bit, 2 and 4 are signed.
func checkFormat(f audioFormat) error {
	switch {
	case f.Rate <= 0:
		return fmt.Errorf("sample rate %d", f.Rate)
	case f.Channels < 1:
		return fmt.Errorf("%d channels", f.Channels)
	case f.Width != 1 && f.Width != 2 && f.Width != 4:
		return fmt.Errorf("%d-byte samples; this server reads 1, 2 and 4", f.Width)
	}
	return nil
}

// appendPCM decodes one chunk into mono float32 in [-1, 1) and appends it.
//
// The scaling is audio.DecodeWAV's -- divide by the full-scale magnitude, not
// by the maximum sample -- so a clip that arrived over this protocol and the
// same clip read from a file hold identical values, and a difference in a
// transcript is the model's rather than the door's. Channels are averaged,
// which is what both front ends expect.
func appendPCM(dst []float32, payload []byte, width, channels int) ([]float32, error) {
	frame := width * channels
	if frame == 0 {
		return dst, fmt.Errorf("a %d-byte, %d-channel frame", width, channels)
	}
	if len(payload)%frame != 0 {
		return dst, fmt.Errorf("a %d-byte chunk is not a whole number of %d-byte frames",
			len(payload), frame)
	}
	n := len(payload) / frame
	dst = slicesGrow(dst, n)
	for i := range n {
		var sum float32
		for ch := range channels {
			off := i*frame + ch*width
			switch width {
			case 1:
				// 8-bit PCM is unsigned with a 128 bias, which is the one
				// place this differs from the WAV decoder's int16 path.
				sum += (float32(payload[off]) - 128) / 128
			case 2:
				sum += float32(int16(binary.LittleEndian.Uint16(payload[off:]))) / 32768
			case 4:
				sum += float32(int32(binary.LittleEndian.Uint32(payload[off:]))) / 2147483648
			}
		}
		dst = append(dst, sum/float32(channels))
	}
	return dst, nil
}

// encodePCM16 is api.encodePCM16: scaled by 32767 and clamped, so a full
// scale +1.0 stays at the top of the range instead of wrapping to the most
// negative sample.
func encodePCM16(samples []float32) []byte {
	out := make([]byte, 0, 2*len(samples))
	for _, s := range samples {
		v := math.Round(float64(s) * 32767)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		out = binary.LittleEndian.AppendUint16(out, uint16(int16(v)))
	}
	return out
}

// slicesGrow reserves room for n more samples. Chunks arrive at a few
// hundred a second and the utterance is appended to across all of them, so
// growing once per chunk beats growing per sample.
func slicesGrow(s []float32, n int) []float32 {
	if cap(s)-len(s) >= n {
		return s
	}
	grown := make([]float32, len(s), max(2*cap(s), len(s)+n))
	copy(grown, s)
	return grown
}

// ratio reports how much faster than real time a run was, the way the HTTP
// handlers do: 20x means a second of audio took 50 ms.
func ratio(seconds float64, took time.Duration) float64 {
	if took <= 0 {
		return 0
	}
	return seconds / took.Seconds()
}

// modelID is the backend's first model id, for the `info` message. The
// fallback is only reached by a backend that reports none, which none does.
func modelID(b api.Backend, fallback string) string {
	if b == nil {
		return fallback
	}
	if models := b.Models(); len(models) > 0 && models[0].ID != "" {
		return models[0].ID
	}
	return fallback
}

func hasVoice(voices []TtsVoice, name string) bool {
	for _, v := range voices {
		if v.Name == name {
			return true
		}
	}
	return false
}

func strptr(s string) *string { return &s }

func isTimeout(err error) bool {
	var ne net.Error
	return errors.As(err, &ne) && ne.Timeout()
}

// isEOF covers both ends of a connection that stopped: a clean close between
// events, and one that stopped in the middle of a frame it had declared the
// length of. Neither is worth more than a disconnection line.
func isEOF(err error) bool {
	return errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF)
}
