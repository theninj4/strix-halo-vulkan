package safetensors

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestF32ToF16Golden checks the narrowing conversion against CPython's
// struct 'e', which is IEEE binary16 with round-half-to-even. The cases in
// testdata cover the subnormal ladder, the tie points between every 37th
// adjacent pair of halves, the overflow edge at 65504/65520, and 4000
// uniformly random bit patterns.
func TestF32ToF16Golden(t *testing.T) {
	f, err := os.Open(filepath.Join("testdata", "f32_to_f16.txt"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	n, bad := 0, 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) != 2 {
			t.Fatalf("malformed line %q", line)
		}
		inBits, err := strconv.ParseUint(parts[0], 16, 32)
		if err != nil {
			t.Fatal(err)
		}
		wantBits, err := strconv.ParseUint(parts[1], 16, 16)
		if err != nil {
			t.Fatal(err)
		}
		in := math.Float32frombits(uint32(inBits))
		got := f32ToF16(in)
		if got != uint16(wantBits) {
			if bad < 10 {
				t.Errorf("f32ToF16(%v /*%08x*/) = %04x, want %04x", in, inBits, got, wantBits)
			}
			bad++
		}
		n++
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if n < 10000 {
		t.Fatalf("only %d golden cases loaded, expected the full table", n)
	}
	if bad > 0 {
		t.Errorf("%d of %d golden cases wrong", bad, n)
	}
	t.Logf("%d golden cases matched CPython struct 'e'", n)
}

// TestF16RoundTrip checks that widening then narrowing is the identity for
// every one of the 65536 half bit patterns. It is the property that matters
// for a checkpoint already stored as F16: reading it must not perturb it.
func TestF16RoundTrip(t *testing.T) {
	for i := 0; i < 1<<16; i++ {
		h := uint16(i)
		exp, mant := (h>>10)&0x1f, h&0x3ff
		if exp == 0x1f && mant != 0 {
			continue // NaN: payload is not preserved by design
		}
		got := f32ToF16(f16ToF32(h))
		if got != h {
			t.Fatalf("round trip of %04x gave %04x (via %v)", h, got, f16ToF32(h))
		}
	}
}

// TestF16ToF32KnownValues pins the widening direction against values that
// can be written exactly, including both ends of the subnormal range.
func TestF16ToF32KnownValues(t *testing.T) {
	cases := []struct {
		h    uint16
		want float32
	}{
		{0x0000, 0},
		{0x8000, float32(math.Copysign(0, -1))},
		{0x0001, 5.9604644775390625e-08}, // smallest subnormal, 2^-24
		{0x03ff, 6.0975551605224609e-05}, // largest subnormal
		{0x0400, 6.103515625e-05},        // smallest normal, 2^-14
		{0x3c00, 1},
		{0xbc00, -1},
		{0x3555, 0.333251953125},
		{0x7bff, 65504}, // largest finite
		{0x7c00, float32(math.Inf(1))},
		{0xfc00, float32(math.Inf(-1))},
	}
	for _, c := range cases {
		got := f16ToF32(c.h)
		if got != c.want {
			t.Errorf("f16ToF32(%04x) = %v, want %v", c.h, got, c.want)
		}
		if math.Signbit(float64(got)) != math.Signbit(float64(c.want)) {
			t.Errorf("f16ToF32(%04x) sign mismatch: %v vs %v", c.h, got, c.want)
		}
	}
}

// TestOpenRoundTrip writes a small checkpoint by hand and reads it back,
// covering the header parse, the offset arithmetic and all three float
// dtypes the GOALS.md models arrive in.
func TestOpenRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tiny.safetensors")

	f32vals := []float32{1, -2.5, 0.125, 65504}
	bf16vals := []uint16{0x3f80, 0xc020, 0x3e00, 0x477f} // 1, -2.5, 0.125, 65280
	f16vals := []uint16{0x3c00, 0xc100, 0x3000, 0x7bff}  // 1, -2.5, 0.125, 65504

	var body []byte
	off := 0
	type ent struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	hdr := map[string]any{}

	buf := make([]byte, 4*len(f32vals))
	for i, v := range f32vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	hdr["a.f32"] = ent{"F32", []int{2, 2}, [2]int{off, off + len(buf)}}
	body = append(body, buf...)
	off += len(buf)

	buf = make([]byte, 2*len(bf16vals))
	for i, v := range bf16vals {
		binary.LittleEndian.PutUint16(buf[i*2:], v)
	}
	hdr["b.bf16"] = ent{"BF16", []int{4}, [2]int{off, off + len(buf)}}
	body = append(body, buf...)
	off += len(buf)

	buf = make([]byte, 2*len(f16vals))
	for i, v := range f16vals {
		binary.LittleEndian.PutUint16(buf[i*2:], v)
	}
	hdr["c.f16"] = ent{"F16", []int{4}, [2]int{off, off + len(buf)}}
	body = append(body, buf...)

	hdrJSON, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(hdrJSON)))
	out = append(out, hdrJSON...)
	out = append(out, body...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}

	set, err := OpenSet(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()

	if set.Len() != 3 {
		t.Fatalf("got %d tensors, want 3", set.Len())
	}
	if got, want := set.Params(), int64(12); got != want {
		t.Errorf("Params() = %d, want %d", got, want)
	}

	a, err := set.Get("a.f32")
	if err != nil {
		t.Fatal(err)
	}
	if got := a.Shape; len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Errorf("a.f32 shape = %v, want [2 2]", got)
	}
	gotF32, err := a.F32(nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range f32vals {
		if gotF32[i] != v {
			t.Errorf("a.f32[%d] = %v, want %v", i, gotF32[i], v)
		}
	}
	gotF16, err := a.F16(nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range f32vals {
		if f16ToF32(gotF16[i]) != v {
			t.Errorf("a.f32[%d] via fp16 = %v, want %v", i, f16ToF32(gotF16[i]), v)
		}
	}

	// BF16 widens exactly, so its fp32 view must equal the bit-shifted value.
	b, err := set.Get("b.bf16")
	if err != nil {
		t.Fatal(err)
	}
	gotB, err := b.F32(nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, raw := range bf16vals {
		want := math.Float32frombits(uint32(raw) << 16)
		if gotB[i] != want {
			t.Errorf("b.bf16[%d] = %v, want %v", i, gotB[i], want)
		}
	}

	// F16 must pass through untouched.
	c, err := set.Get("c.f16")
	if err != nil {
		t.Fatal(err)
	}
	gotC, err := c.F16(nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, raw := range f16vals {
		if gotC[i] != raw {
			t.Errorf("c.f16[%d] = %04x, want %04x", i, gotC[i], raw)
		}
	}

	if _, err := set.Get("nope"); err == nil {
		t.Error("Get on a missing tensor should fail")
	}
	if !set.Has("a.f32") || set.Has("nope") {
		t.Error("Has disagrees with Get")
	}
}

// TestOpenRejectsBadHeader checks the failure modes are caught at open time
// rather than at first use.
func TestOpenRejectsBadHeader(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, b []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}

	if _, err := Open(write("short.bin", []byte{1, 2, 3})); err == nil {
		t.Error("a 3 byte file should not parse")
	}

	hdr := []byte(`{"x":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}`)
	out := make([]byte, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(hdr)))
	out = append(out, hdr...)
	out = append(out, make([]byte, 8)...) // only half the promised bytes
	if _, err := Open(write("truncated.bin", out)); err == nil {
		t.Error("a tensor running past EOF should not parse")
	}

	hdr = []byte(`{"x":{"dtype":"I8","shape":[4],"data_offsets":[0,4]}}`)
	out = make([]byte, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(hdr)))
	out = append(out, hdr...)
	out = append(out, make([]byte, 4)...)
	if _, err := Open(write("dtype.bin", out)); err == nil {
		t.Error("an unsupported dtype should be rejected at open time")
	}

	hdr = []byte(`{"x":{"dtype":"F32","shape":[5],"data_offsets":[0,16]}}`)
	out = make([]byte, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(hdr)))
	out = append(out, hdr...)
	out = append(out, make([]byte, 16)...)
	if _, err := Open(write("shape.bin", out)); err == nil {
		t.Error("a shape disagreeing with the byte span should be rejected")
	}
}
