package audio

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// fixture is the clip every stage of the STT vertical is checked against:
// 11.000 s of 16 kHz mono speech with a known transcript (SPEECH.md).
const fixture = "../testdata/jfk.wav"

// TestFixtureMatchesPython pins the decode against an independent read of the
// same file by CPython's `wave` module scaled the way `soundfile` scales —
// `int16 / 32768` — because every reference dump from S2 onward starts from
// those samples. If this test and a dump disagree, no later comparison means
// anything.
//
//	>>> x = np.frombuffer(wave.open(f).readframes(n), '<i2').astype(np.float32) / 32768
//	>>> hashlib.sha256(x.tobytes()).hexdigest()
func TestFixtureMatchesPython(t *testing.T) {
	c, err := ReadWAV(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if c.Rate != 16000 {
		t.Errorf("rate = %d, want 16000", c.Rate)
	}
	if len(c.Samples) != 176000 {
		t.Fatalf("samples = %d, want 176000", len(c.Samples))
	}
	if d := c.Duration(); math.Abs(d-11) > 1e-9 {
		t.Errorf("duration = %v, want 11", d)
	}

	// The hash is over the little-endian float32 bytes, so it checks the
	// exact values and not only their statistics.
	raw := unsafe.Slice((*byte)(unsafe.Pointer(&c.Samples[0])), 4*len(c.Samples))
	if got := hex.EncodeToString(sha256Sum(raw)); got != "ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0" {
		t.Errorf("sha256 of samples = %s, want ebd5285…", got)
	}

	want := []float32{0.027130126953125, 0.01092529296875, -0.007659912109375, -0.0185546875, -0.013916015625}
	for i, w := range want {
		if got := c.Samples[len(c.Samples)-5+i]; got != w {
			t.Errorf("sample[n-%d] = %v, want %v", 5-i, got, w)
		}
	}
}

func sha256Sum(b []byte) []byte { h := sha256.Sum256(b); return h[:] }

// TestRoundTrip checks that encoding and decoding is the identity on values
// that are already on the 16-bit grid. It is not the identity in general —
// encode rounds to 1/32767 and decode divides by 32768 — so the input is
// built on the grid the encoder writes.
func TestRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	in := make([]float32, 1000)
	for i := range in {
		in[i] = float32(rng.Intn(65535)-32767) / 32767
	}
	c := &Clip{Samples: in, Rate: 24000}
	buf, err := c.EncodeWAV()
	if err != nil {
		t.Fatal(err)
	}
	if len(buf) != 44+2*len(in) {
		t.Errorf("encoded %d bytes, want %d", len(buf), 44+2*len(in))
	}
	got, err := DecodeWAV(buf)
	if err != nil {
		t.Fatal(err)
	}
	if got.Rate != 24000 {
		t.Errorf("rate = %d, want 24000", got.Rate)
	}
	for i := range in {
		// 1/32767 out and 1/32768 back in: the same integer, a hair smaller.
		if d := math.Abs(float64(got.Samples[i] - in[i])); d > 1.0/32767 {
			t.Fatalf("sample %d: %v -> %v", i, in[i], got.Samples[i])
		}
	}
}

// TestClipping pins the saturation rather than letting a vocoder overshoot
// wrap around to the opposite sign, which is what a naive int16 conversion
// does at +1.0 and what it sounds like when it happens.
func TestClipping(t *testing.T) {
	c := &Clip{Samples: []float32{1, -1, 2, -2, 0}, Rate: 24000}
	buf, err := c.EncodeWAV()
	if err != nil {
		t.Fatal(err)
	}
	want := []int16{32767, -32767, 32767, -32768, 0}
	for i, w := range want {
		if got := int16(binary.LittleEndian.Uint16(buf[44+2*i:])); got != w {
			t.Errorf("sample %d = %d, want %d", i, got, w)
		}
	}
}

// TestDownmix averages channels, which is what the feature extractors assume
// a stereo file means.
func TestDownmix(t *testing.T) {
	stereo := &Clip{Samples: []float32{0.5, -0.5, 0.25, 0.75}, Rate: 16000}
	buf, err := stereo.EncodeWAV()
	if err != nil {
		t.Fatal(err)
	}
	// Relabel the mono file as stereo: same data chunk, two channels.
	binary.LittleEndian.PutUint16(buf[22:], 2)
	binary.LittleEndian.PutUint32(buf[28:], 16000*2*2)
	binary.LittleEndian.PutUint16(buf[32:], 4)
	got, err := DecodeWAV(buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Samples) != 2 {
		t.Fatalf("frames = %d, want 2", len(got.Samples))
	}
	for i, w := range []float32{0, 0.5} {
		if d := math.Abs(float64(got.Samples[i] - w)); d > 1e-4 {
			t.Errorf("frame %d = %v, want %v", i, got.Samples[i], w)
		}
	}
}

// TestSkipsUnknownChunks covers the LIST/INFO blocks ffmpeg writes, including
// the pad byte after an odd-sized one.
func TestSkipsUnknownChunks(t *testing.T) {
	c := &Clip{Samples: []float32{0.5, -0.5}, Rate: 16000}
	buf, _ := c.EncodeWAV()
	fmtChunk, dataChunk := buf[12:36], buf[36:]

	var odd []byte
	odd = append(odd, "LIST"...)
	odd = binary.LittleEndian.AppendUint32(odd, 5)
	odd = append(odd, "INFOx"...)
	odd = append(odd, 0) // pad to a word boundary

	out := append([]byte{}, buf[0:12]...)
	out = append(out, odd...)
	out = append(out, fmtChunk...)
	out = append(out, dataChunk...)
	binary.LittleEndian.PutUint32(out[4:], uint32(len(out)-8))

	got, err := DecodeWAV(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Samples) != 2 || got.Rate != 16000 {
		t.Fatalf("got %d samples at %d Hz", len(got.Samples), got.Rate)
	}
}

// TestRejects covers the formats that are deliberately not supported, since
// silently coercing them is how a decoder bug becomes a model bug.
func TestRejects(t *testing.T) {
	c := &Clip{Samples: []float32{0.5, -0.5}, Rate: 16000}
	good, _ := c.EncodeWAV()

	cases := []struct {
		name   string
		mangle func([]byte)
	}{
		{"float32 samples", func(b []byte) { binary.LittleEndian.PutUint16(b[20:], 3); binary.LittleEndian.PutUint16(b[34:], 32) }},
		{"8-bit", func(b []byte) { binary.LittleEndian.PutUint16(b[34:], 8) }},
		{"24-bit", func(b []byte) { binary.LittleEndian.PutUint16(b[34:], 24) }},
		{"zero rate", func(b []byte) { binary.LittleEndian.PutUint32(b[24:], 0) }},
		{"not RIFF", func(b []byte) { copy(b[0:4], "RIFX") }},
		{"no data chunk", func(b []byte) { copy(b[36:40], "junk") }},
	}
	for _, tc := range cases {
		b := append([]byte{}, good...)
		tc.mangle(b)
		if _, err := DecodeWAV(b); err == nil {
			t.Errorf("%s: decoded without error", tc.name)
		}
	}
}
