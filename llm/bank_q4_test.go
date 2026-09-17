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
			nsb := tc.k / (q4kSuper * 32)
			br := make([]byte, (tc.n/coopMatTile)*nsb*coopMatTile*q4kRecord)
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
