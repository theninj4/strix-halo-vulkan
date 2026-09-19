package llm

// P4c's two packers against the quantiser they are a transcription of.
//
//	go test ./llm/ -v -run TestMoEPackersAgainstGGML
//
// `packQ41Row` and `packIQ4NLRow` are ports of `quantize_row_q4_1_impl` and
// `quantize_row_iq4_nl_impl`, which is a stronger claim than the level
// membership `TestPackQ51RowIsGgmlsLayout` checks: membership says the bytes
// are the format, and says nothing about whether the *levels* are the ones
// llama-quantize would have chosen. A search that optimises something will
// still produce valid blocks.
//
// So the oracle is llama.cpp's own, through `reference/quant_ref.c` — the
// same binary and the same record format `TestQuantSimMatchesGGML` uses, over
// its own pair of fixture files because the rows here are the down
// projection's 640 rather than a dense row:
//
//	L=/home/kube/repos/llama.cpp
//	gcc -O2 -o /tmp/quant_ref reference/quant_ref.c \
//	    -I$L/ggml/include -L$L/build/bin -lggml-base -Wl,-rpath,$L/build/bin
//	go test ./llm/ -run TestMoEPackersAgainstGGML -updatemoepack
//	/tmp/quant_ref llm/testdata/moepack_in.bin llm/testdata/moepack_ref.bin
//
// **It is not an equality, and the one place it cannot be is measured rather
// than excused.** Both packers choose their levels against the (d, m) pair as
// *stored* — halves, which is what the shader reads — where ggml chooses them
// against the f32 pair it then rounds (D10). That can only move a level by
// one, so the test bounds it at one level and then asks the question the
// departure exists for: under the importance weight the fit was made with,
// which of the two reconstructs the row better?
//
// The uncalibrated arms have no such departure on Q4_1 — `rtn` is
// `quantize_row_q4_1_ref`, whose own levels come from the f32 pair too — so
// the number to watch there is how few values differ, not that none do.

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"testing"

	"strix-halo-vulkan/gguf"
)

var updateMoEPack = flag.Bool("updatemoepack", false,
	"write llm/testdata/moepack_in.bin for reference/quant_ref.c and stop")

const (
	moePackInPath  = "testdata/moepack_in.bin"
	moePackRefPath = "testdata/moepack_ref.bin"
	// The tensor P4c narrows, and the row length that has no K-quant.
	moePackTensor = "blk.3.ffn_down_exps.weight"
	moePackRows   = 64
	// GGML_TYPE_Q4_1 and GGML_TYPE_IQ4_NL.
	ggmlTypeQ4_1   = 3
	ggmlTypeIQ4_NL = 20
)

var moePackTypes = []struct {
	ggml int
	typ  gguf.Type
	name string
}{
	{ggmlTypeQ4_1, gguf.Q4_1, "q4_1"},
	{ggmlTypeIQ4_NL, gguf.IQ4_NL, "iq4_nl"},
}

func TestMoEPackersAgainstGGML(t *testing.T) {
	if *updateMoEPack {
		writeMoEPackInput(t)
		return
	}
	in, err := readQuantRecords(moePackInPath, true)
	if err != nil {
		t.Skipf("%v; regenerate with -updatemoepack and reference/quant_ref.c", err)
	}
	ref, err := readQuantRecords(moePackRefPath, false)
	if err != nil {
		t.Skipf("%v; regenerate with reference/quant_ref.c", err)
	}
	if len(in) != len(ref) || len(in) == 0 {
		t.Fatalf("%d input records against %d reference ones", len(in), len(ref))
	}

	for i, rec := range in {
		var c struct {
			typ  gguf.Type
			name string
		}
		for _, mt := range moePackTypes {
			if mt.ggml == rec.typ {
				c.typ, c.name = mt.typ, mt.name
			}
		}
		if c.name == "" {
			t.Fatalf("record %d: ggml type %d is not one this test knows", i, rec.typ)
		}
		mode := "rtn"
		if rec.haveIm {
			mode = "imatrix"
		}
		t.Run(fmt.Sprintf("%s/%s", c.name, mode), func(t *testing.T) {
			rowBytes := rec.k / c.typ.BlockElems() * c.typ.BlockBytes()
			dst := make([]byte, rec.rows*rowBytes)
			for r := 0; r < rec.rows; r++ {
				row := rec.x[r*rec.k : (r+1)*rec.k]
				out := dst[r*rowBytes : (r+1)*rowBytes]
				qw := rec.im
				if !rec.haveIm {
					qw = nil
				}
				switch c.typ {
				case gguf.Q4_1:
					packQ41Row(out, row, qw)
				case gguf.IQ4_NL:
					packIQ4NLRow(out, row, qw)
				}
			}
			got, err := gguf.Dequantize(c.typ, dst, int64(rec.rows*rec.k), nil)
			if err != nil {
				t.Fatal(err)
			}
			want := ref[i].x
			if len(got) != len(want) {
				t.Fatalf("%d values against %d", len(got), len(want))
			}

			// A differing value has to be one level away, which is all the
			// stored-half assignment can move it. The level width is the
			// block's own scale, read out of our record.
			diff, worst := 0, 0.0
			var ourErr, ggmlErr float64
			for j := range got {
				r, b := j/rec.k, (j%rec.k)/c.typ.BlockElems()
				d := float64(f16at(dst[r*rowBytes+b*c.typ.BlockBytes():][:2]))
				step := math.Abs(d)
				if c.typ == gguf.IQ4_NL {
					// The codebook's widest gap, in units of d: 113 - 89.
					step *= 24
				}
				if got[j] != want[j] {
					diff++
					if e := math.Abs(float64(got[j] - want[j])); e > worst {
						worst = e
					}
					if e := math.Abs(float64(got[j] - want[j])); e > step*1.001 {
						t.Fatalf("value %d is %g against ggml's %g — %.3g, more than the one level (%.3g) the stored-half assignment can move it",
							j, got[j], want[j], e, step)
					}
				}
				// The error each fit leaves, weighted the way the fit was
				// made: this is the question the departure is for.
				w := 1.0
				if rec.haveIm {
					w = float64(rec.im[j%rec.k])
				}
				x := float64(rec.x[j])
				ourErr += w * (x - float64(got[j])) * (x - float64(got[j]))
				ggmlErr += w * (x - float64(want[j])) * (x - float64(want[j]))
			}
			pct := 100 * float64(diff) / float64(len(got))
			t.Logf("%s/%s: %d of %d values differ (%.3f%%, worst %.3e); weighted sq error ours %.6e against ggml's %.6e (%.4f)",
				c.name, mode, diff, len(got), pct, worst, ourErr, ggmlErr, ourErr/ggmlErr)
			// The departure is taken because it should not lose. A percent
			// of slack on the ratio, because the two fits are so close that
			// a single row could otherwise decide it.
			if ourErr > ggmlErr*1.01 {
				t.Errorf("%s/%s: choosing levels against the stored halves is %.4fx ggml's weighted error — the departure is not paying and should be dropped",
					c.name, mode, ourErr/ggmlErr)
			}
			// And it is a real quantisation of the real row rather than a
			// zeroed buffer or a copy of the reference.
			var num, den float64
			for j := range got {
				e := float64(rec.x[j] - got[j])
				num += e * e
				den += float64(rec.x[j]) * float64(rec.x[j])
			}
			if rms := math.Sqrt(num / den); rms == 0 || rms > 0.2 {
				t.Fatalf("%s/%s: relative rms %.4g, want a real fit", c.name, mode, rms)
			}
		})
	}
}

// writeMoEPackInput dumps the fixture the C oracle reads: real rows of the
// down projection, and the imatrix row of the expert they belong to.
func writeMoEPackInput(t *testing.T) {
	t.Helper()
	m, _ := fixtures(t)
	im, imErr := DefaultImatrix()
	if imErr != nil {
		t.Logf("no imatrix (%v): writing the uncalibrated records only", imErr)
	}
	if err := os.MkdirAll("testdata", 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(moePackInPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	tn, err := m.Set.Get(moePackTensor)
	if err != nil {
		t.Fatalf("%s: %v", moePackTensor, err)
	}
	k := int(tn.Dims[0])
	x := make([]float32, 0, moePackRows*k)
	for r := 0; r < moePackRows; r++ {
		if x, err = tn.DequantizeRow(int64(r), x); err != nil {
			t.Fatal(err)
		}
	}
	// Expert 0's importance row: the first `moePackRows` rows of the tensor
	// all belong to it, since a routed tensor's rows are `[k, n, nExpert]`.
	var cols []float32
	if imErr == nil {
		if cols, err = im.ExpertColumns(moePackTensor, 0); err != nil {
			t.Fatal(err)
		}
	}
	for _, typ := range moePackTypes {
		for _, withIm := range []bool{false, true} {
			if withIm && cols == nil {
				continue
			}
			hdr := []uint32{uint32(typ.ggml), uint32(moePackRows), uint32(k), 0}
			if withIm {
				hdr[3] = 1
			}
			if err := binary.Write(f, binary.LittleEndian, hdr); err != nil {
				t.Fatal(err)
			}
			if err := binary.Write(f, binary.LittleEndian, x); err != nil {
				t.Fatal(err)
			}
			if withIm {
				if err := binary.Write(f, binary.LittleEndian, cols); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	t.Logf("wrote %s (%d rows of %d); now run reference/quant_ref.c over it", moePackInPath, moePackRows, k)
}
