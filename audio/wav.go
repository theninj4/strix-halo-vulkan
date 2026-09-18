// Package audio is the host-side signal processing the two speech verticals
// need: WAV in and out, and the short-time Fourier transforms that sit at
// either end of them — a forward STFT for parakeet's mel front end and an
// inverse one for kokoro's vocoder.
//
// It is deliberately CPU-only and deliberately small. A 30 s clip is 3000
// 512-point FFTs, which is a few milliseconds of work against an encoder
// that is billions of flops a frame, so SPEECH.md's rule applies: build the
// thing that runs, and let the profiler decide what moves to the GPU.
package audio

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// Clip is mono PCM as float32 in [-1, 1).
//
// The scaling is the one every Python reference uses — `soundfile`, `librosa`
// and `torchaudio` all divide 16-bit samples by 32768 — so a Go-side clip and
// a reference dump of the same file hold bit-identical values and any
// difference downstream is the model's, not the decoder's.
type Clip struct {
	Samples []float32
	Rate    int // sample rate in Hz
}

// Duration is the clip's length in seconds.
func (c *Clip) Duration() float64 { return float64(len(c.Samples)) / float64(c.Rate) }

// ReadWAV decodes a RIFF/WAVE file. See DecodeWAV for what is supported.
func ReadWAV(path string) (*Clip, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	c, err := DecodeWAV(buf)
	if err != nil {
		return nil, fmt.Errorf("audio: %s: %w", path, err)
	}
	return c, nil
}

// DecodeWAV decodes 16-bit PCM RIFF/WAVE bytes to a mono Clip.
//
// Only 16-bit PCM is accepted, because it is what both verticals actually
// deal in — parakeet is fed 16 kHz s16 and kokoro emits 24 kHz s16 — and a
// decoder that quietly widens its input is a decoder whose bugs show up as
// model bugs. Multi-channel files are averaged down to mono, which is what
// the feature extractors expect; the sample rate is reported rather than
// resampled, since resampling is a filter design question and neither model
// wants one (both checkpoints fix their rate).
//
// Chunks other than `fmt ` and `data` are skipped, so the metadata that
// ffmpeg and friends write does not have to be modelled.
func DecodeWAV(buf []byte) (*Clip, error) {
	if len(buf) < 12 || string(buf[0:4]) != "RIFF" || string(buf[8:12]) != "WAVE" {
		// Name what did arrive. The overwhelmingly common cause of this
		// error is a client that sent a perfectly good file in a format
		// this decoder does not read, and "not a RIFF/WAVE file" alone
		// does not distinguish that from a corrupt upload.
		return nil, fmt.Errorf("not a RIFF/WAVE file: looks like %s", Sniff(buf))
	}

	var (
		format     uint16
		channels   int
		rate       int
		bits       uint16
		data       []byte
		haveFormat bool
		haveData   bool
	)
	for off := 12; off+8 <= len(buf); {
		id := string(buf[off : off+4])
		size := int(binary.LittleEndian.Uint32(buf[off+4 : off+8]))
		body := off + 8
		if size < 0 || body+size > len(buf) {
			// A truncated final chunk is common enough in streamed files
			// that taking what is there beats refusing the whole file.
			size = len(buf) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, fmt.Errorf("fmt chunk is %d bytes, need at least 16", size)
			}
			format = binary.LittleEndian.Uint16(buf[body:])
			channels = int(binary.LittleEndian.Uint16(buf[body+2:]))
			rate = int(binary.LittleEndian.Uint32(buf[body+4:]))
			bits = binary.LittleEndian.Uint16(buf[body+14:])
			// WAVE_FORMAT_EXTENSIBLE carries the real format code in its
			// sub-format GUID, whose first two bytes are the code itself.
			if format == 0xfffe && size >= 26 {
				format = binary.LittleEndian.Uint16(buf[body+24:])
			}
			haveFormat = true
		case "data":
			data = buf[body : body+size]
			haveData = true
		}
		// Chunks are word-aligned: an odd size is followed by a pad byte.
		off = body + size + size&1
	}

	switch {
	case !haveFormat:
		return nil, fmt.Errorf("no fmt chunk")
	case !haveData:
		return nil, fmt.Errorf("no data chunk")
	case format != 1:
		return nil, fmt.Errorf("format %d is not PCM", format)
	case bits != 16:
		return nil, fmt.Errorf("%d-bit samples, only 16-bit PCM is supported", bits)
	case channels < 1:
		return nil, fmt.Errorf("%d channels", channels)
	case rate <= 0:
		return nil, fmt.Errorf("sample rate %d", rate)
	}

	frame := 2 * channels
	n := len(data) / frame
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float32
		for ch := 0; ch < channels; ch++ {
			s := int16(binary.LittleEndian.Uint16(data[i*frame+2*ch:]))
			sum += float32(s) / 32768
		}
		out[i] = sum / float32(channels)
	}
	return &Clip{Samples: out, Rate: rate}, nil
}

// WriteWAV encodes a clip as a mono 16-bit PCM WAV file.
func (c *Clip) WriteWAV(path string) error {
	buf, err := c.EncodeWAV()
	if err != nil {
		return err
	}
	return os.WriteFile(path, buf, 0o644)
}

// EncodeWAV encodes a clip as mono 16-bit PCM RIFF/WAVE bytes.
//
// Samples are scaled by 32767 rather than 32768 and clamped, so that a full
// scale +1.0 — which a vocoder can produce and which 32768 would wrap to the
// most negative sample — stays at the top of the range instead of inverting.
func (c *Clip) EncodeWAV() ([]byte, error) {
	if c.Rate <= 0 {
		return nil, fmt.Errorf("audio: cannot encode a clip with sample rate %d", c.Rate)
	}
	const channels, bits = 1, 16
	dataLen := 2 * len(c.Samples)
	buf := make([]byte, 0, 44+dataLen)

	put32 := func(v uint32) { buf = binary.LittleEndian.AppendUint32(buf, v) }
	put16 := func(v uint16) { buf = binary.LittleEndian.AppendUint16(buf, v) }

	buf = append(buf, "RIFF"...)
	put32(uint32(36 + dataLen))
	buf = append(buf, "WAVE"...)
	buf = append(buf, "fmt "...)
	put32(16)
	put16(1) // PCM
	put16(channels)
	put32(uint32(c.Rate))
	put32(uint32(c.Rate * channels * bits / 8)) // byte rate
	put16(channels * bits / 8)                  // block align
	put16(bits)
	buf = append(buf, "data"...)
	put32(uint32(dataLen))
	for _, s := range c.Samples {
		v := math.Round(float64(s) * 32767)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		buf = binary.LittleEndian.AppendUint16(buf, uint16(int16(v)))
	}
	return buf, nil
}
