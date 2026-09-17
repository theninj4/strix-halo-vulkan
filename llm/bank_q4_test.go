package llm

// The 4.5-bit dense bank against the simulation that chose it: LLM.md L8c-4.

import (
	"math/rand"
	"testing"
)

// mustQ4K is the format every rung of L8c-3 was measured with.
func mustQ4K(t *testing.T, spec, mode string) QuantSim {
	t.Helper()
	q, err := ParseQuantSim(spec)
	if err != nil {
		t.Fatalf("%s: %v", spec, err)
	}
	q.Mode = mode
	return q
}

// TestBankQ4KIsTheSim is the stage's first gate, and it is an equality.
//
// L8c-3's number is a perplexity measured through `sim.go`, which round-trips
// a weight to floats and lets the fp16 kernels multiply them. This bank is
// the same format *stored*, so every value a kernel reads out of it has to be
// the value the simulation staged — not close to it. Both go through
// `asymEnc`, so the test is really that the packing and the addressing are
// each other's inverse, which is exactly the thing a tolerance would hide.
func TestBankQ4KIsTheSim(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, tc := range []struct {
		name    string
		n, k    int
		spec    string
		mode    string
		weights bool
	}{
		{"q4_k rtn 64x2560", 64, 2560, "q4_k/32", "rtn", false},
		{"q4_k imatrix 64x2560", 64, 2560, "q4_k/32", "imatrix", true},
		{"q4_k rtn 32x6144", 32, 6144, "q4_k/32", "rtn", false},
		{"q4_k search 32x256", 32, 256, "q4_k/32", "search", false},
		// L8c-6's family: 320 is ten groups of 32 and one super-block for
		// the whole row, so the record is `packScaleMin12`'s twenty bytes
		// and not ggml's sixteen. Both arms, because the calibrated one is
		// the arm a bank is built at.
		{"q4_k rtn 64x320", 64, 320, "q4_k/32", "rtn", false},
		{"q4_k imatrix 64x320", 64, 320, "q4_k/32", "imatrix", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q := mustQ4K(t, tc.spec, tc.mode)
			x := make([]float32, tc.n*tc.k)
			for i := range x {
				x[i] = float32(rng.NormFloat64()) * 0.05
			}
			var qw []float32
			if tc.weights {
				qw = make([]float32, tc.k)
				for i := range qw {
					qw[i] = float32(rng.Float64()) + 1e-3
				}
			}
			// The simulation's answer: the floats a kernel would multiply.
			want := append([]float32(nil), x...)
			if err := q.ApplyWeighted(want, tc.k, qw); err != nil {
				t.Fatal(err)
			}
			// The bank's: the same format, stored and read back.
			bq := make([]byte, tc.n*tc.k/2)
			br := make([]byte, q4kRecPlane(tc.n, tc.k))
			// A row remapping, so the addressing is exercised rather than
			// assumed to be the identity.
			perm := make([]int, tc.n)
			for i := range perm {
				perm[i] = tc.n - coopMatTile - (i/coopMatTile)*coopMatTile + i%coopMatTile
			}
			if err := tileBQ4K(bq, br, x, tc.n, tc.k, func(i int) int { return perm[i] }, q, qw); err != nil {
				t.Fatal(err)
			}
			if got := q4kBytes(tc.n, tc.k); got != len(bq)+len(br) {
				t.Fatalf("q4kBytes says %d, the two planes are %d", got, len(bq)+len(br))
			}
			// 4.500 bits at both super-block lengths: eight groups carry
			// 4 + 12 = 16 bytes over 256 weights, ten carry 4 + 15 = 19
			// rounded up to a word over 320.
			if bits := float64(len(bq)+len(br)) * 8 / float64(tc.n*tc.k); bits != 4.5 {
				t.Fatalf("%.4f bits a weight, want 4.5", bits)
			}
			diff := 0
			for i := 0; i < tc.n; i++ {
				for c := 0; c < tc.k; c++ {
					got := q4kDequant(bq, br, tc.n, tc.k, perm[i], c)
					if got != want[i*tc.k+c] {
						diff++
					}
				}
			}
			if diff != 0 {
				t.Fatalf("%d of %d values differ from the simulation", diff, tc.n*tc.k)
			}
		})
	}
}

// TestQ4KRecordPackings is the two packings against each other's inverse,
// value for value over every representable six-bit pair: LLM.md L8c-6.
//
// It exists because the ten-group one is **ours** rather than ggml's — there
// is no oracle for it the way `reference/quant_ref.c` is the oracle for the
// levels — so the only thing that can be checked is that the writer and the
// two readers (here and in shaders/llm_q4k.glsl, which is the same
// arithmetic) agree. A packing that lost a bit would show up as a wrong
// scale on one group in sixty-four, which is exactly the sort of error a
// perplexity run reports as "the format costs a bit more than expected".
func TestQ4KRecordPackings(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, sub := range []int{8, 10} {
		ls, lm := make([]int, sub), make([]int, sub)
		rec := make([]byte, q4kRecordBytes(sub))
		for trial := 0; trial < 2000; trial++ {
			for j := 0; j < sub; j++ {
				ls[j], lm[j] = rng.Intn(64), rng.Intn(64)
			}
			for i := range rec {
				rec[i] = 0
			}
			switch sub {
			case 8:
				packScaleMinK4(rec[4:16], ls, lm)
			default:
				packScaleMin12(rec[4:19], ls, lm)
			}
			for j := 0; j < sub; j++ {
				_, _, sc, mn := unpackQ4KRecord(rec, sub, j)
				if sc != ls[j] || mn != lm[j] {
					t.Fatalf("sub %d group %d: wrote (%d, %d), read (%d, %d)",
						sub, j, ls[j], lm[j], sc, mn)
				}
			}
		}
		// Every bit the packing does not use has to stay zero, or the
		// padding byte would be carrying state the shader does not read.
		for j := range ls {
			ls[j], lm[j] = 63, 63
		}
		for i := range rec {
			rec[i] = 0
		}
		if sub == 10 {
			// Ten twelve-bit pairs are 120 bits, which is bytes 4..18 to the
			// last bit; byte 19 is the word alignment and nothing else.
			packScaleMin12(rec[4:19], ls, lm)
			if rec[19] != 0 {
				t.Fatalf("the ten-group record's twentieth byte is %02x, want the pad", rec[19])
			}
			for i := 4; i < 19; i++ {
				if rec[i] != 0xFF {
					t.Fatalf("all-63 pairs left byte %d at %02x, want ff", i, rec[i])
				}
			}
		}
	}
}
