package audio

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// TestAgainstDFT checks the radix-2 transform against the definition it is a
// fast version of, at a size small enough for the O(n^2) form to be cheap.
func TestAgainstDFT(t *testing.T) {
	const n = 64
	rng := rand.New(rand.NewSource(7))
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(rng.NormFloat64(), rng.NormFloat64())
	}
	for _, inverse := range []bool{false, true} {
		sign := -1.0
		if inverse {
			sign = 1.0
		}
		want := make([]complex128, n)
		for k := range want {
			var sum complex128
			for j, v := range x {
				ang := sign * 2 * math.Pi * float64(k) * float64(j) / n
				sum += v * cmplx.Exp(complex(0, ang))
			}
			want[k] = sum
		}
		f, err := NewFFT(n, inverse)
		if err != nil {
			t.Fatal(err)
		}
		got := append([]complex128(nil), x...)
		f.Transform(got)
		for k := range got {
			if d := cmplx.Abs(got[k] - want[k]); d > 1e-10 {
				t.Fatalf("inverse=%v bin %d: %v, want %v (%.3g off)", inverse, k, got[k], want[k], d)
			}
		}
	}
}

// TestRoundTripFFT is the property the vocoder depends on: an inverse
// transform of a forward transform is the input, scaled by n.
func TestRoundTripFFT(t *testing.T) {
	const n = 512
	rng := rand.New(rand.NewSource(3))
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(rng.NormFloat64(), 0)
	}
	fwd, _ := NewFFT(n, false)
	inv, _ := NewFFT(n, true)
	buf := append([]complex128(nil), x...)
	fwd.Transform(buf)
	inv.Transform(buf)
	for i := range buf {
		if d := cmplx.Abs(buf[i]/complex(n, 0) - x[i]); d > 1e-12 {
			t.Fatalf("sample %d: round trip off by %.3g", i, d)
		}
	}
}

// TestRealTransform checks the half-spectrum helper against the full one, and
// that the spectrum really is conjugate-symmetric — the property that makes
// keeping only n/2+1 bins lossless.
func TestRealTransform(t *testing.T) {
	const n = 128
	rng := rand.New(rand.NewSource(11))
	x := make([]float64, n)
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	f, _ := NewFFT(n, false)
	full := make([]complex128, n)
	for i, v := range x {
		full[i] = complex(v, 0)
	}
	f.Transform(full)
	half := f.RealTransform(nil, x, nil)
	if len(half) != n/2+1 {
		t.Fatalf("%d bins, want %d", len(half), n/2+1)
	}
	for k, v := range half {
		if d := cmplx.Abs(v - full[k]); d > 1e-12 {
			t.Fatalf("bin %d: %v vs %v", k, v, full[k])
		}
		if k > 0 && k < n/2 {
			if d := cmplx.Abs(v - cmplx.Conj(full[n-k])); d > 1e-12 {
				t.Fatalf("bin %d is not the conjugate of bin %d", k, n-k)
			}
		}
	}
}

func TestFFTRejectsNonPowerOfTwo(t *testing.T) {
	for _, n := range []int{0, 1, 3, 100, 1000} {
		if _, err := NewFFT(n, false); err == nil {
			t.Errorf("NewFFT(%d) was accepted", n)
		}
	}
}

// TestSTFTGeometry pins the frame count and the centring against the formula
// parakeet's extractor uses, since an off-by-one here is an off-by-one in
// every downstream length.
func TestSTFTGeometry(t *testing.T) {
	st, err := NewSTFT(512, 160, HannWindow(400, false), true)
	if err != nil {
		t.Fatal(err)
	}
	if got := st.Frames(176000); got != 1101 {
		t.Errorf("frames = %d, want 1101", got)
	}
	if got := st.Bins(); got != 257 {
		t.Errorf("bins = %d, want 257", got)
	}

	// A pure tone at bin 32 of a 512-point transform lands on that bin, with
	// the rest of its energy in the immediate neighbours: the window is 400
	// samples inside a 512-point frame, so the tone is not periodic in the
	// window and the leakage is the window's, not the transform's.
	x := make([]float32, 16000)
	for i := range x {
		x[i] = float32(math.Cos(2 * math.Pi * 32 * float64(i) / 512))
	}
	power := st.Power(x)
	row := power[50*st.Bins() : 51*st.Bins()]
	var total, peak, near float64
	peakBin := -1
	for b, v := range row {
		total += v
		if v > peak {
			peak, peakBin = v, b
		}
		if b >= 30 && b <= 34 {
			near += v
		}
	}
	if peakBin != 32 {
		t.Errorf("peak at bin %d, want 32", peakBin)
	}
	if near/total < 0.99 {
		t.Errorf("bins 30-34 hold %.2f%% of the frame's energy", 100*near/total)
	}
}
