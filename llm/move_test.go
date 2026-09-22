package llm

// The arena-to-arena move, against the host narrowing it replaced (LLM.md L6c).
//
// The whole graph's numbers did not move by one digit when the glue went from
// the host to the device — `l_last` at all six depths, `result_norm` and
// `result_output` are identical to the last place. That is the right outcome
// and it is not an accident: `float16_t(v)` in SPIR-V and `F32ToF16(v)` in Go
// are the same round-to-nearest-even. This is that claim in one test, run
// against the tensor shape the graph actually moves and without a 85 GB
// checkpoint in the way.
//
//	go test ./llm/ -v -run TestMove

import (
	"math"
	"testing"

	"strix-halo-vulkan/safetensors"
)

// TestMoveNarrow checks the f32→fp16 move value for value, including the pad
// rows a GEMM with no bounds check depends on.
func TestMoveNarrow(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()

	const (
		rows  = 37   // deliberately not a multiple of anything
		pad   = 64   // the consumer's row block
		width = 2560 // n_embd
	)
	lda := width + gemmPad

	src, err := newArena(dev, rows*width*4)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Destroy()
	dst, err := newArena(dev, pad*lda*2)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Destroy()
	// The destination starts poisoned, so a row the kernel fails to write is
	// a visible failure rather than a lucky zero.
	poison := make([]uint16, pad*lda)
	for i := range poison {
		poison[i] = 0x3c00 // 1.0
	}
	dst.WriteUint16At(0, poison)

	// Values that exercise the conversion rather than a smooth ramp: the
	// halfway cases round-to-even decides, and magnitudes across the fp16
	// range including the subnormals L4a-7 found two bugs in.
	in := make([]float32, rows*width)
	for i := range in {
		switch i % 7 {
		case 0:
			in[i] = float32(math.Float32frombits(uint32(0x38000000 + i*7919)))
		case 1:
			in[i] = -float32(i%2048) / 1024
		case 2:
			in[i] = float32(i%3) * 6.1035156e-05 // subnormal halves
		case 3:
			in[i] = 2049 + float32(i%4) // past fp16's exact-integer range
		default:
			in[i] = float32(math.Sin(float64(i)))
		}
	}
	src.WriteFloat32At(0, in)

	m, err := newMover(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Destroy()
	if err := m.Move(
		Port{Buf: dst, Off: 0, Stride: lda, Width: width, Rows: pad, Half: true},
		Port{Buf: src, Off: 0, Stride: width, Width: width},
		rows); err != nil {
		t.Fatal(err)
	}

	want := make([]uint16, pad*lda)
	narrowRows(want, in, rows, width, lda)
	got := dst.ReadUint16At(0, pad*lda)

	bad := 0
	for t0 := 0; t0 < pad; t0++ {
		for i := 0; i < width; i++ {
			j := t0*lda + i
			if got[j] != want[j] {
				if bad++; bad <= 5 {
					t.Errorf("row %d value %d: device %#04x, host %#04x (source %v)",
						t0, i, got[j], want[j], srcAt(in, t0, i, width, rows))
				}
			}
		}
	}
	if bad == 0 {
		t.Logf("%d rows of %d narrowed bit for bit, and %d pad rows zeroed",
			rows, width, pad-rows)
	} else {
		t.Errorf("%d of %d values differ", bad, pad*width)
	}

	// The columns past the width are the A operand's padding (§2.3) and the
	// move must not touch them: they were zeroed once at allocation and the
	// GEMM reads them.
	for t0 := 0; t0 < pad; t0++ {
		for i := width; i < lda; i++ {
			if got[t0*lda+i] != 0x3c00 {
				t.Fatalf("the move wrote pad column %d of row %d", i, t0)
				return
			}
		}
	}
}

// TestMoveWiden checks the f32→f32 move, which is how a sublayer's output
// reaches the hyper-connection block's combine.
func TestMoveWiden(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()

	const rows, width, stride = 11, 512, 640
	src, err := newArena(dev, rows*stride*4)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Destroy()
	dst, err := newArena(dev, rows*width*4)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Destroy()

	in := make([]float32, rows*stride)
	for i := range in {
		in[i] = float32(i)*1e-3 - 3
	}
	src.WriteFloat32At(0, in)

	m, err := newMover(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Destroy()
	// A source whose row stride is wider than the values it carries, which is
	// the shape every fused projection's output has.
	if err := m.Move(
		Port{Buf: dst, Off: 0, Stride: width, Width: width, Rows: rows},
		Port{Buf: src, Off: 0, Stride: stride, Width: width},
		rows); err != nil {
		t.Fatal(err)
	}
	got := dst.ReadFloat32At(0, rows*width)
	for r := 0; r < rows; r++ {
		for i := 0; i < width; i++ {
			if w := in[r*stride+i]; got[r*width+i] != w {
				t.Fatalf("row %d value %d: %v, want %v", r, i, got[r*width+i], w)
			}
		}
	}
	t.Logf("%d rows of %d moved exactly out of a %d-wide stride", rows, width, stride)
}

// srcAt is the source value behind a destination element, for an error
// message that says what was being converted.
func srcAt(in []float32, row, i, width, rows int) any {
	if row >= rows {
		return "pad row, should be zero"
	}
	v := in[row*width+i]
	return [2]any{v, safetensors.F32ToF16(v)}
}

// TestInPortPadsTheRunNotTheArena is P16's mechanism, asserted because no
// tolerance can see it: a port that pads to the arena's rows is *correct*, and
// it was a decode step writing zeros over the whole prefill arena twice a
// layer — 21% of decode at an 8192-row `-llm-batch`, and the whole of what a
// wide batch was measured to cost it (P12-7, P15). One row must pad to one
// row block, not to the arena.
func TestInPortPadsTheRunNotTheArena(t *testing.T) {
	// The fixtures' own arenas are one row block deep, which cannot tell the
	// two paddings apart, so both blocks are staged four times as wide.
	const maxTok = 512
	_, dc, dw, _ := dnFixtures(t, dnLayer)
	_, _, ac, aw, _, _ := attnFixtures(t)
	dev, done := newTestDevice(t)
	defer done()
	dn, err := NewDeltaNetGPU(dev, dc, maxTok, []DeltaNetWeights{dw}, denseQ8Test)
	if err != nil {
		t.Fatal(err)
	}
	defer dn.Destroy()
	at, err := NewAttnGPU(dev, ac, maxTok, maxTok, []AttnWeights{aw}, denseQ8Test)
	if err != nil {
		t.Fatal(err)
	}
	defer at.Destroy()

	for _, b := range []struct {
		name                 string
		resize               func(int) error
		port                 func() Port
		align, arena, maxTok int
	}{
		{"deltanet", dn.Resize, dn.InPort, dn.rowAlign, dn.arenaRows, dn.tokens},
		{"attention", at.Resize, at.InPort, at.rowAlign, at.arenaRows, at.tokens},
	} {
		if b.arena <= b.align {
			t.Fatalf("%s: arena of %d rows is one row block of %d — the fixture cannot tell the two paddings apart",
				b.name, b.arena, b.align)
		}
		if err := b.resize(1); err != nil {
			t.Fatal(err)
		}
		if got := b.port().Rows; got != b.align {
			t.Errorf("%s: one row pads to %d rows, want the row block %d (arena %d)", b.name, got, b.align, b.arena)
		}
		if err := b.resize(b.maxTok); err != nil {
			t.Fatal(err)
		}
		if got := b.port().Rows; got != b.arena {
			t.Errorf("%s: a full batch pads to %d rows, want the arena's %d", b.name, got, b.arena)
		}
	}
}
