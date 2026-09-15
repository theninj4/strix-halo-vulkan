package gguf

import (
	"bytes"
	"encoding/binary"
	"flag"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The dequant paths are transcriptions of ggml-quants.c, so a test that
// restates the formats would only be checking the transcription against
// itself. The oracle is llama.cpp's own to_float, run over the same bytes by
// reference/dequant_ref.c and committed as testdata/dequant_ref.bin — see
// that file for how to regenerate the pair.

var updateDequant = flag.Bool("updatedequant", false, "rewrite gguf/testdata/dequant_in.bin")

// dequantCases are the eight formats a qwen3.8-flash-next checkpoint holds,
// with enough blocks of each to exercise every sub-block of a K-quant
// super-block and both halves of a nibble plane.
var dequantCases = []struct {
	typ    Type
	blocks int
}{
	{Q4_K, 4}, {Q5_K, 4}, {Q5_1, 8}, {Q8_0, 8}, {IQ4_NL, 8},
	{F32, 64}, {F16, 64}, {BF16, 64},
}

func TestDequantMatchesGGML(t *testing.T) {
	inPath := filepath.Join("testdata", "dequant_in.bin")
	refPath := filepath.Join("testdata", "dequant_ref.bin")

	if *updateDequant {
		if err := os.WriteFile(inPath, buildDequantInput(), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s; now run reference/dequant_ref.c over it", inPath)
	}

	in, err := os.ReadFile(inPath)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	// The input the reference was taken over has to be the input the test
	// generates, or the comparison means nothing.
	if want := buildDequantInput(); !bytes.Equal(in, want) {
		t.Fatalf("testdata/dequant_in.bin is %d bytes and the generator now makes %d; regenerate both files",
			len(in), len(want))
	}

	ri, rr := &recordReader{b: in}, &recordReader{b: ref}
	for _, c := range dequantCases {
		typ, n, payload := ri.next(t)
		refTyp, refN, refPayload := rr.next(t)
		if typ != c.typ || refTyp != c.typ {
			t.Fatalf("record order: input %s, reference %s, want %s", typ, refTyp, c.typ)
		}
		if n != refN {
			t.Fatalf("%s: input has %d elements, reference %d", typ, n, refN)
		}
		got, err := Dequantize(typ, payload, n, nil)
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		if int64(len(got)) != n {
			t.Fatalf("%s: got %d values for %d elements", typ, len(got), n)
		}
		// ggml's own arithmetic, in the same order, in the same precision:
		// the comparison is bit-exact rather than within a tolerance, and a
		// single ulp of difference is a real disagreement about the format.
		for i := int64(0); i < n; i++ {
			want := math.Float32frombits(binary.LittleEndian.Uint32(refPayload[i*4:]))
			if math.Float32bits(got[i]) != math.Float32bits(want) {
				t.Fatalf("%s: element %d = %v (%#08x), llama.cpp says %v (%#08x)",
					typ, i, got[i], math.Float32bits(got[i]), want, math.Float32bits(want))
			}
		}
		t.Logf("%-7s %5d elements exact against llama.cpp", typ, n)
	}
	if !ri.done() || !rr.done() {
		t.Errorf("trailing records: input %d bytes left, reference %d", len(ri.b)-ri.o, len(rr.b)-rr.o)
	}
}

func TestDequantRowAndErrors(t *testing.T) {
	w := newWriter()
	w.tensor("q", Q8_0, []int64{64, 3}, 0)
	f, err := Open(w.write(t, filepath.Join(t.TempDir(), "m.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tn, _ := f.Get("q")
	// Every byte is zero, so every block is a zero scale over zero quants.
	row, err := tn.DequantizeRow(2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(row) != 64 {
		t.Errorf("row is %d values, want 64", len(row))
	}
	for i, v := range row {
		if v != 0 {
			t.Fatalf("element %d of an all-zero block is %v", i, v)
		}
	}
	if _, err := tn.DequantizeRow(3, nil); err == nil {
		t.Error("dequantized a row past the end of the tensor")
	}
	if _, err := Dequantize(Q4_K, make([]byte, 144), 100, nil); err == nil {
		t.Error("dequantized a partial Q4_K block")
	}
	if _, err := Dequantize(Q8_0, make([]byte, 10), 32, nil); err == nil {
		t.Error("dequantized past the end of the data")
	}
	if _, err := Dequantize(Q2_K, make([]byte, 84), 256, nil); err == nil {
		t.Error("Q2_K has no path here and should say so")
	}
	if Dequantizable(Q2_K) || !Dequantizable(Q4_K) {
		t.Error("Dequantizable disagrees with Dequantize")
	}
}

// recordReader walks the `u32 type, u32 elements, u64 bytes, payload` stream
// both files are in.
type recordReader struct {
	b []byte
	o int
}

func (r *recordReader) next(t *testing.T) (Type, int64, []byte) {
	t.Helper()
	if r.o+16 > len(r.b) {
		t.Fatalf("record header past the end of a %d byte file", len(r.b))
	}
	typ := Type(binary.LittleEndian.Uint32(r.b[r.o:]))
	n := int64(binary.LittleEndian.Uint32(r.b[r.o+4:]))
	nbytes := int64(binary.LittleEndian.Uint64(r.b[r.o+8:]))
	r.o += 16
	if int64(r.o)+nbytes > int64(len(r.b)) {
		t.Fatalf("%s: payload of %d bytes past the end", typ, nbytes)
	}
	payload := r.b[r.o : int64(r.o)+nbytes]
	r.o += int(nbytes)
	return typ, n, payload
}

func (r *recordReader) done() bool { return r.o == len(r.b) }

// buildDequantInput generates the block bytes both sides read.
//
// The bytes are pseudorandom, but every fp16 scale field is forced to an
// ordinary magnitude first: a random 16-bit pattern is Inf or NaN 1 time in
// 32, and a NaN scale would make the comparison meaningless rather than
// strict. Everything else — nibbles, sign bits, the 6-bit packed scales, the
// fifth-bit planes — stays fully random, which is the part that matters.
func buildDequantInput() []byte {
	var out bytes.Buffer
	rng := newLCG(0x5150514e)
	for _, c := range dequantCases {
		n := int64(c.blocks) * int64(c.typ.BlockElems())
		nbytes, err := c.typ.SizeOf(n)
		if err != nil {
			panic(err)
		}
		payload := make([]byte, nbytes)
		for i := range payload {
			payload[i] = byte(rng.next())
		}
		bs := c.typ.BlockBytes()
		for b := 0; b < c.blocks; b++ {
			blk := payload[b*bs:]
			switch c.typ {
			case Q8_0, IQ4_NL:
				putF16(blk[0:], saneF16(rng.next()))
			case Q5_1, Q4_K, Q5_K:
				putF16(blk[0:], saneF16(rng.next()))
				putF16(blk[2:], saneF16(rng.next()))
			case F16:
				for i := 0; i+1 < bs; i += 2 {
					putF16(blk[i:], saneF16(rng.next()))
				}
			case BF16:
				for i := 0; i+1 < bs; i += 2 {
					binary.LittleEndian.PutUint16(blk[i:], uint16(saneF32(rng.next())>>16))
				}
			case F32:
				for i := 0; i+3 < bs; i += 4 {
					binary.LittleEndian.PutUint32(blk[i:], saneF32(rng.next()))
				}
			}
		}
		binary.Write(&out, binary.LittleEndian, uint32(c.typ))
		binary.Write(&out, binary.LittleEndian, uint32(n))
		binary.Write(&out, binary.LittleEndian, uint64(nbytes))
		out.Write(payload)
	}
	return out.Bytes()
}

// saneF16 keeps the sign and mantissa of a random word and replaces the
// exponent with one in [-10, 9], which is 2^-10 to just under 2^10.
func saneF16(r uint64) uint16 {
	h := uint16(r)
	exp := uint16(5 + (r>>16)%20)
	return h&0x83FF | exp<<10
}

// saneF32 is the same idea for float32: an exponent in [-17, 14].
func saneF32(r uint64) uint32 {
	b := uint32(r)
	exp := uint32(110 + (r>>32)%32)
	return b&0x807FFFFF | exp<<23
}

func putF16(b []byte, v uint16) { binary.LittleEndian.PutUint16(b, v) }

// lcg is a 64-bit linear congruential generator — Knuth's constants. Written
// out rather than taken from math/rand so the stream is pinned to this file
// and a Go release cannot change what the committed testdata means.
type lcg struct{ s uint64 }

func newLCG(seed uint64) *lcg { return &lcg{s: seed} }

func (g *lcg) next() uint64 {
	g.s = g.s*6364136223846793005 + 1442695040888963407
	return g.s >> 11
}
