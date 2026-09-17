package llm

// L8c-2's port checked against the implementation it is a port of.
//
//	go test ./llm/ -v -run TestQuantSimMatchesGGML
//
// `makeQxQuants` is a transcription of ggml's `make_qx_quants`, and the
// simulation's whole claim is that it stages the values a real quantiser
// would produce. A transcription that optimises *something* will still look
// plausible on a reconstruction-error table — it will reduce error, just not
// ggml's error — so the claim needs the reference, not a re-reading of the
// same source. This is `gguf`'s TestDequantMatchesGGML on the other side of
// the round trip.
//
// The oracle is `reference/quant_ref.c`: `ggml_quantize_chunk`, the entry
// point `llama-quantize` uses, then `ggml_get_type_traits(t)->to_float`
// straight back. Passing an imatrix or not selects the calibrated and
// uncalibrated arms out of one binary, because `quantize_row_q4_0_impl`
// short-circuits to `quantize_row_q4_0_ref` when it is null.
//
//	L=/home/kube/repos/llama.cpp
//	gcc -O2 -o /tmp/quant_ref reference/quant_ref.c \
//	    -I$L/ggml/include -L$L/build/bin -lggml-base -Wl,-rpath,$L/build/bin
//	go test ./llm/ -run TestQuantSimMatchesGGML -updatequant
//	/tmp/quant_ref llm/testdata/quant_in.bin llm/testdata/quant_ref.bin
//
// **The comparison is against the half, not the float.** ggml's `to_float`
// returns `float(q) * float(d)` in f32; the simulation stores
// `fp16(float(q) * d)`, because that is what a kernel reading the tile
// forms (L8b-2). So the two agree when `F32ToF16(ours) == F32ToF16(theirs)`,
// which is the equality the simulation actually needs to be true.

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"strix-halo-vulkan/safetensors"
)

var updateQuant = flag.Bool("updatequant", false,
	"write llm/testdata/quant_in.bin for reference/quant_ref.c and stop")

const (
	quantInPath  = "testdata/quant_in.bin"
	quantRefPath = "testdata/quant_ref.bin"
	// ggmlTypeQ4_0 is GGML_TYPE_Q4_0, whose block is 32 with sixteen levels
	// — the format D7 now names, and the one make_qx_quants is reached
	// through for every 4-bit rung here.
	ggmlTypeQ4_0 = 2
	// ggmlTypeQ4_K and ggmlTypeQ5_K are L8c-3's asymmetric arm: a super-block
	// of eight groups of 32 with a 6-bit scale and min apiece, against one
	// fp16 pair — the form this checkpoint's own experts are in and the one
	// unsloth's imatrix was collected for.
	ggmlTypeQ4_K = 12
	ggmlTypeQ5_K = 13
	// quantRefRows keeps the fixture small enough to commit: 64 rows of the
	// two widths below is 819 200 weights, which is what L8c-2's claim is
	// stated over. L8c-3's two K arms take a quarter of that — 204 800 each
	// — because three formats times two arms at 64 rows would be a 39 MB
	// fixture, and a bit-exactness check does not get more true with rows.
	quantRefRows  = 64
	quantRefRowsK = 16
)

// quantRefTypes are the formats each case is quantised in, and the spec
// llm/sim.go is asked for the same thing by. Every one of them reaches a
// different ggml entry point.
var quantRefTypes = []struct {
	ggml int
	spec string
	rows int
}{
	{ggmlTypeQ4_0, "q4_0/32", quantRefRows},
	{ggmlTypeQ4_K, "q4_k/32", quantRefRowsK},
	{ggmlTypeQ5_K, "q5_k/32", quantRefRowsK},
}

// quantRefCases are real weights, one per shape that matters: the DeltaNet's
// fused projection at k = 2560 and the hyper-connection block's down
// projection at k = 10240 — the family L8c-1 found decides the whole plan.
var quantRefCases = []string{"blk.0.attn_qkv.weight", "blk.0.hc_attn_down.weight"}

type quantRecord struct {
	typ     int
	rows, k int
	haveIm  bool
	x       []float32
	im      []float32
}

// spec names the sim format a record's ggml type stands for.
func (r quantRecord) spec() string {
	for _, t := range quantRefTypes {
		if t.ggml == r.typ {
			return t.spec
		}
	}
	return ""
}

func TestQuantSimMatchesGGML(t *testing.T) {
	if *updateQuant {
		writeQuantInput(t)
		return
	}
	in, err := readQuantRecords(quantInPath, true)
	if err != nil {
		t.Skipf("%v; regenerate with -updatequant and reference/quant_ref.c", err)
	}
	ref, err := readQuantRecords(quantRefPath, false)
	if err != nil {
		t.Skipf("%v; regenerate with reference/quant_ref.c", err)
	}
	if len(in) != len(ref) || len(in) == 0 {
		t.Fatalf("%d input records against %d reference ones", len(in), len(ref))
	}

	for i, rec := range in {
		mode := "rtn"
		if rec.haveIm {
			mode = "imatrix"
		}
		spec := rec.spec()
		if spec == "" {
			t.Fatalf("record %d: ggml type %d is not one this test knows", i, rec.typ)
		}
		t.Run(fmt.Sprintf("%s/%s/k%d", strings.TrimSuffix(spec, "/32"), mode, rec.k), func(t *testing.T) {
			q, err := ParseQuantSim(spec)
			if err != nil {
				t.Fatal(err)
			}
			q.Mode = mode
			got := append([]float32(nil), rec.x...)
			if err := q.ApplyWeighted(got, rec.k, rec.im); err != nil {
				t.Fatal(err)
			}
			want := ref[i].x
			if len(got) != len(want) {
				t.Fatalf("%d values against %d", len(got), len(want))
			}
			bad, at, exact := 0, -1, 0
			var worst float64
			for j := range got {
				if got[j] == want[j] {
					exact++
				}
				// ggml hands back the f32 product; the bank holds the half.
				if safetensors.F32ToF16(got[j]) != safetensors.F32ToF16(want[j]) {
					bad++
					if at < 0 {
						at = j
					}
					if d := math.Abs(float64(got[j] - want[j])); d > worst {
						worst = d
					}
				}
			}
			if bad != 0 {
				t.Errorf("%s: %d of %d halves differ (worst %.3e), first at %d: %v against %v",
					mode, bad, len(got), worst, at, got[at], want[at])
				return
			}
			// The asymmetric arm reproduces ggml's dequant expression term
			// for term — `d*sc*l - dmin*m` in f32 — so it is exact in f32
			// and not only in the half, which is a stronger statement than
			// the gate needs and the one that says the port is the port.
			t.Logf("%s, k=%d: %d values identical to ggml's, half for half (%d of them in f32)",
				mode, rec.k, len(got), exact)
		})
	}
}

// writeQuantInput dumps the fixture the C oracle reads.
func writeQuantInput(t *testing.T) {
	t.Helper()
	m, _ := fixtures(t)
	im, imErr := DefaultImatrix()
	if imErr != nil {
		t.Logf("no imatrix (%v): writing the uncalibrated records only", imErr)
	}
	if err := os.MkdirAll("testdata", 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(quantInPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	for _, name := range quantRefCases {
		tn, err := m.Set.Get(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		k := int(tn.Dims[0])
		all, err := tn.Dequantize(nil)
		if err != nil {
			t.Fatal(err)
		}
		all = all[:quantRefRows*k]
		var cols []float32
		if imErr == nil {
			if cols, err = im.Columns(name); err != nil {
				t.Fatal(err)
			}
		}
		for _, typ := range quantRefTypes {
			for _, withIm := range []bool{false, true} {
				if withIm && cols == nil {
					continue
				}
				x := all[:typ.rows*k]
				hdr := []uint32{uint32(typ.ggml), uint32(typ.rows), uint32(k), 0}
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
	}
	t.Logf("wrote %s; now run reference/quant_ref.c over it", quantInPath)
}

// readQuantRecords reads either file; the input carries the imatrix row and
// the reference does not.
func readQuantRecords(path string, wantIm bool) ([]quantRecord, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []quantRecord
	for off := 0; off < len(raw); {
		if off+16 > len(raw) {
			return nil, fmt.Errorf("%s: short header at %d", path, off)
		}
		typ := int(binary.LittleEndian.Uint32(raw[off:]))
		rows := int(binary.LittleEndian.Uint32(raw[off+4:]))
		k := int(binary.LittleEndian.Uint32(raw[off+8:]))
		haveIm := binary.LittleEndian.Uint32(raw[off+12:]) == 1
		off += 16
		n := rows * k
		need := n * 4
		if wantIm && haveIm {
			need += k * 4
		}
		if off+need > len(raw) {
			return nil, fmt.Errorf("%s: short payload at %d", path, off)
		}
		rec := quantRecord{typ: typ, rows: rows, k: k, haveIm: haveIm, x: make([]float32, n)}
		for i := range rec.x {
			rec.x[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[off+4*i:]))
		}
		off += n * 4
		if wantIm && haveIm {
			rec.im = make([]float32, k)
			for i := range rec.im {
				rec.im[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[off+4*i:]))
			}
			off += k * 4
		}
		out = append(out, rec)
	}
	return out, nil
}
