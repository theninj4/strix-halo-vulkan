package audio

import (
	"encoding/binary"
	"fmt"
	"strings"
	"unicode"
)

// Sniff names the container a byte slice looks like, for an error message
// that would otherwise say only that the bytes were not what was wanted.
//
// "not a RIFF/WAVE file" tells the caller their upload was rejected;
// "not a RIFF/WAVE file (looks like MP3)" tells them what to change. It reads
// magic numbers only -- it does not validate the file, and a name it returns
// means "this is what the first few bytes claim", not "this decodes".
func Sniff(buf []byte) string {
	has := func(off int, magic string) bool {
		return len(buf) >= off+len(magic) && string(buf[off:off+len(magic)]) == magic
	}

	switch {
	case has(0, "RIFF") && has(8, "WAVE"):
		return "WAV"
	case has(0, "RIFF"):
		// A RIFF container of some other kind: AVI, or a WAV whose header
		// was truncated before the form type.
		if len(buf) >= 12 {
			return fmt.Sprintf("a RIFF container of type %q, not WAVE", printable(buf[8:12]))
		}
		return "a truncated RIFF header"
	case has(0, "OggS"):
		return "Ogg (Vorbis or Opus)"
	case has(0, "fLaC"):
		return "FLAC"
	case has(0, "ID3"):
		return "MP3 (with an ID3 tag)"
	case len(buf) >= 2 && buf[0] == 0xff && buf[1]&0xe0 == 0xe0:
		return "MP3"
	case has(4, "ftyp"):
		brand := ""
		if len(buf) >= 12 {
			brand = " (brand " + printable(buf[8:12]) + ")"
		}
		return "MP4/M4A" + brand
	case len(buf) >= 4 && binary.BigEndian.Uint32(buf[0:4]) == 0x1a45dfa3:
		return "Matroska/WebM"
	case has(0, "FORM") && has(8, "AIFF"):
		return "AIFF"
	case has(0, "FORM") && has(8, "AIFC"):
		return "AIFF-C"
	case has(0, "#!AMR"):
		return "AMR"
	case has(0, ".snd"):
		return "AU/SND"
	case has(0, "\x1f\x8b"):
		return "gzip-compressed data"
	case has(0, "PK\x03\x04"):
		return "a zip archive"
	case has(0, "{"), has(0, "["):
		return "JSON, not audio"
	case has(0, "<"):
		return "markup, not audio"
	}

	// Nothing recognised. Raw PCM is the common case here -- it has no
	// header at all, which is exactly why it cannot be identified -- so say
	// what the bytes were rather than guessing.
	n := 8
	if len(buf) < n {
		n = len(buf)
	}
	if n == 0 {
		return "empty"
	}
	return fmt.Sprintf("unrecognised (first %d bytes %x); headerless PCM looks like this", n, buf[:n])
}

// printable renders a four-byte tag for an error message, replacing anything
// unprintable so a binary blob cannot scribble control characters into a log.
func printable(b []byte) string {
	var sb strings.Builder
	for _, c := range b {
		if c > unicode.MaxASCII || !unicode.IsPrint(rune(c)) {
			sb.WriteByte('.')
			continue
		}
		sb.WriteByte(c)
	}
	return sb.String()
}
