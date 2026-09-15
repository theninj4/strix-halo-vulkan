package audio

import (
	"math"
	"math/rand"
	"testing"
)

// TestDFTMatchesFFT checks the direct transform against the radix-2 one at
// the sizes both can do, which is the only cross-check either has.
func TestDFTMatchesFFT(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for _, n := range []int{2, 4, 8, 16, 64} {
		for _, inverse := range []bool{false, true} {
			f, err := NewFFT(n, inverse)
			if err != nil {
				t.Fatal(err)
			}
			d := NewDFT(n, inverse)
			a := make([]complex128, n)
			for i := range a {
				a[i] = complex(r.NormFloat64(), r.NormFloat64())
			}
			b := append([]complex128(nil), a...)
			f.Transform(a)
			d.Transform(b)
			for i := range a {
				if math.Abs(real(a[i])-real(b[i])) > 1e-10 || math.Abs(imag(a[i])-imag(b[i])) > 1e-10 {
					t.Fatalf("n=%d inverse=%v bin %d: FFT %v, DFT %v", n, inverse, i, a[i], b[i])
				}
			}
		}
	}
}

// TestDFTRoundTrip covers the sizes the FFT cannot, including the 20 kokoro's
// vocoder needs.
func TestDFTRoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(8))
	for _, n := range []int{3, 5, 12, 20, 25} {
		fwd, inv := NewDFT(n, false), NewDFT(n, true)
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(r.NormFloat64(), 0)
		}
		got := append([]complex128(nil), x...)
		fwd.Transform(got)
		inv.Transform(got)
		for i := range got {
			if math.Abs(real(got[i])/float64(n)-real(x[i])) > 1e-10 {
				t.Fatalf("n=%d sample %d: %v after a round trip, want %v", n, i, got[i]/complex(float64(n), 0), x[i])
			}
		}
	}
}

// TestSTFTRoundTrip is the check that matters for the vocoder: analyse a
// signal and resynthesise it, which only comes back if the window-square
// normalisation and the centring are both right.
//
// The geometry is kokoro's — 20-point transforms at a hop of 5 with a
// periodic Hann window — so this is also the check that a 20-point transform
// works at all.
func TestSTFTRoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(9))
	const nfft, hop = 20, 5
	win := HannWindow(nfft, true)
	st, err := NewSTFT(nfft, hop, win, true)
	if err != nil {
		t.Fatal(err)
	}
	st.Pad = PadReflect
	is, err := NewISTFT(nfft, hop, win, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{600, 1205, 4000} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.07)) + 0.3*float32(r.NormFloat64())
		}
		frames := st.Frames(n)
		if got := is.Samples(frames); got != n {
			t.Errorf("%d samples make %d frames and come back as %d", n, frames, got)
		}
		mag, phase := st.MagnitudePhase(x)
		back, err := is.Apply(mag, phase, frames)
		if err != nil {
			t.Fatal(err)
		}
		// The first and last n_fft/2 samples are reconstructed from frames
		// that ran off the end of the signal, so they carry the padding's
		// contribution; torch has the same edge and the vocoder never looks
		// at it, since its input is a spectrogram rather than a signal.
		var worst float64
		for i := nfft; i < n-nfft; i++ {
			if d := math.Abs(float64(back[i] - x[i])); d > worst {
				worst = d
			}
		}
		if worst > 1e-5 {
			t.Errorf("n=%d: round trip differs by %.3g", n, worst)
		}
	}
}

// TestSTFTPadModes pins the difference between the two paddings, since it is
// a one-field choice that changes every frame at the edges and nothing in the
// middle — the kind of thing that is wrong for a long time before it is
// noticed. Parakeet's front end wants zeros; kokoro's vocoder wants torch's
// default reflection.
func TestSTFTPadModes(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	win := HannWindow(4, true)
	zero, err := NewSTFT(4, 2, win, true)
	if err != nil {
		t.Fatal(err)
	}
	refl, _ := NewSTFT(4, 2, win, true)
	refl.Pad = PadReflect

	// Index -1 is 0 under zero padding and x[1] = 2 under reflection.
	if got := zero.sample(x, -1); got != 0 {
		t.Errorf("zero padding reads %g before the start, want 0", got)
	}
	if got := refl.sample(x, -1); got != 2 {
		t.Errorf("reflection reads %g before the start, want 2", got)
	}
	if got := refl.sample(x, len(x)); got != 7 {
		t.Errorf("reflection reads %g past the end, want 7", got)
	}
	// The interior is untouched by either.
	a, b := zero.Power(x), refl.Power(x)
	mid := 2 * zero.Bins()
	for i := mid; i < mid+zero.Bins(); i++ {
		if math.Abs(a[i]-b[i]) > 1e-12 {
			t.Fatalf("the paddings disagree in the interior at %d: %g against %g", i, a[i], b[i])
		}
	}
	if math.Abs(a[0]-b[0]) < 1e-9 {
		t.Error("the paddings agree on the first frame; the test signal is degenerate")
	}
}
