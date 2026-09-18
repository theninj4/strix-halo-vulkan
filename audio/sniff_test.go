package audio

import (
	"strings"
	"testing"
)

func TestSniff(t *testing.T) {
	wav := func() []byte {
		b := make([]byte, 12)
		copy(b[0:], "RIFF")
		copy(b[8:], "WAVE")
		return b
	}()

	for _, tt := range []struct {
		name, want string
		buf        []byte
	}{
		{"wav", "WAV", wav},
		{"ogg", "Ogg", []byte("OggS\x00\x02")},
		{"flac", "FLAC", []byte("fLaC\x00\x00")},
		{"id3 mp3", "MP3", []byte("ID3\x04\x00\x00")},
		{"bare mp3", "MP3", []byte{0xff, 0xfb, 0x90, 0x00}},
		{"m4a", "MP4/M4A", []byte("\x00\x00\x00\x20ftypM4A ")},
		{"webm", "Matroska/WebM", []byte{0x1a, 0x45, 0xdf, 0xa3, 0x01}},
		{"aiff", "AIFF", []byte("FORM\x00\x00\x00\x00AIFF")},
		{"json", "JSON", []byte(`{"file":"..."}`)},
		{"empty", "empty", nil},
		{"raw pcm", "unrecognised", []byte{0x00, 0x01, 0xff, 0xfe, 0x12, 0x34, 0x56, 0x78}},
	} {
		if got := Sniff(tt.buf); !strings.Contains(got, tt.want) {
			t.Errorf("Sniff(%s) = %q, want it to contain %q", tt.name, got, tt.want)
		}
	}
}

// A RIFF file that is not WAVE must not be reported as WAV, and its form type
// must not be able to write control characters into a log line.
func TestSniffRIFFNotWave(t *testing.T) {
	buf := []byte("RIFF\x00\x00\x00\x00AVI ")
	if got := Sniff(buf); !strings.Contains(got, "AVI") || strings.Contains(got, "WAV,") {
		t.Errorf("Sniff(avi) = %q", got)
	}
	buf = []byte("RIFF\x00\x00\x00\x00\x01\x02\n\x7f")
	got := Sniff(buf)
	if strings.ContainsAny(got, "\n\r\x00") {
		t.Errorf("Sniff put control characters in a log line: %q", got)
	}
}

// The decoder's error must name the format, since that is the whole point.
func TestDecodeWAVNamesTheFormat(t *testing.T) {
	_, err := DecodeWAV([]byte("ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00"))
	if err == nil || !strings.Contains(err.Error(), "MP3") {
		t.Fatalf("err = %v, want it to name MP3", err)
	}
}
