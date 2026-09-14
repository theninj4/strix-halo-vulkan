package parakeet

import (
	"math"
	"testing"

	"strix-halo-vulkan/audio"
)

// melTol is what the front end has to stay inside of the reference features.
//
// The two paths are not the same arithmetic — this one transforms in float64
// and transformers' runs torch.stft in float32 — so the bound is set from the
// measured drift (3.1e-4) with headroom, not from a tolerance argument. The
// scale it sits against is in the test output: the features are normalised to
// unit variance, so an absolute bound is also a relative one here.
//
// The drift is dominated by the quietest mel bins, where the *reference* is
// the imprecise side: see the power ladder logged by TestFrontEndStages, on
// which float32 loses three digits between the loudest bins and those six
// decades down. TestFrontEndDetectsErrors pins the other end of the bound.
const melTol = 1e-3

func TestFrontEndMatchesReference(t *testing.T) {
	m := loadManifest(t)
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Skipf("no checkpoint in %s (%v)", modelDir, err)
	}
	// The config the Go side reads has to be the config the dump ran with.
	if cfg.Features.NFFT != m.FrontEnd.NFFT || cfg.Features.HopLength != m.FrontEnd.HopLength ||
		cfg.Features.WinLength != m.FrontEnd.WinLength || cfg.Features.NMels != m.FrontEnd.NMels ||
		cfg.Features.Preemphasis != m.FrontEnd.Preemphasis {
		t.Fatalf("config %+v does not match the dump's %+v", cfg.Features, m.FrontEnd)
	}

	clip, err := audio.ReadWAV(fixture)
	if err != nil {
		t.Fatal(err)
	}
	fe, err := NewFrontEnd(cfg.Features)
	if err != nil {
		t.Fatal(err)
	}
	feats, err := fe.Features(clip)
	if err != nil {
		t.Fatal(err)
	}

	if feats.Frames != m.FrontEnd.Frames || feats.Valid != m.FrontEnd.ValidFrames {
		t.Fatalf("%d frames (%d valid), reference has %d (%d valid)",
			feats.Frames, feats.Valid, m.FrontEnd.Frames, m.FrontEnd.ValidFrames)
	}

	want, _ := loadRef(t, m, "mel")
	d := compare(t, feats.Data, want)
	t.Logf("mel: max abs %.3g at %d, reference rms %.4g", d.MaxAbs, d.At, d.RMS)
	if d.MaxAbs > melTol {
		t.Errorf("mel deviates by %.3g, bound is %g", d.MaxAbs, melTol)
	}
}

// TestFrontEndStages walks the front end so a regression lands on the step
// that caused it rather than on the features. Each stage is compared against
// the same intermediate in transformers' extractor.
func TestFrontEndStages(t *testing.T) {
	m := loadManifest(t)
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Skipf("no checkpoint in %s (%v)", modelDir, err)
	}
	clip, err := audio.ReadWAV(fixture)
	if err != nil {
		t.Fatal(err)
	}
	fc := cfg.Features

	t.Run("window", func(t *testing.T) {
		want, _ := loadRef(t, m, "window")
		win := audio.HannWindow(fc.WinLength, false)
		got := make([]float32, len(win))
		for i, v := range win {
			got[i] = float32(v)
		}
		d := compare(t, got, want)
		t.Logf("max abs %.3g", d.MaxAbs)
		// A few ulp: torch evaluates the cosine in float32 and this
		// evaluates it in float64 and rounds once.
		if d.MaxAbs > 1e-6 {
			t.Errorf("hann window deviates by %.3g", d.MaxAbs)
		}
	})

	t.Run("mel_filters", func(t *testing.T) {
		want, meta := loadRef(t, m, "mel_filters")
		got := audio.MelFilters(fc.SamplingRate, fc.NFFT, fc.NMels, 0, float64(fc.SamplingRate)/2, audio.Slaney, true)
		if len(got) != meta.Count {
			t.Fatalf("%d weights, reference has %d (%v)", len(got), meta.Count, meta.Shape)
		}
		d := compare(t, got, want)
		t.Logf("max abs %.3g against a peak of %.4g", d.MaxAbs, meta.AbsMax)
		// librosa builds the bank in float64 and rounds to float32, which is
		// exactly what MelFilters does, so the only gap is the order the
		// float64 arithmetic happens in: one ulp at this magnitude.
		if d.MaxAbs > 1e-8 {
			t.Errorf("mel filterbank deviates by %.3g", d.MaxAbs)
		}
	})

	t.Run("preemphasis", func(t *testing.T) {
		want, _ := loadRef(t, m, "preemphasised")
		got := make([]float32, len(clip.Samples))
		got[0] = clip.Samples[0]
		for i := 1; i < len(got); i++ {
			got[i] = clip.Samples[i] - float32(fc.Preemphasis)*clip.Samples[i-1]
		}
		d := compare(t, got, want)
		t.Logf("max abs %.3g", d.MaxAbs)
		if d.MaxAbs > 1e-7 {
			t.Errorf("preemphasis deviates by %.3g", d.MaxAbs)
		}
	})

	t.Run("power", func(t *testing.T) {
		want, meta := loadRef(t, m, "stft_power")
		st, err := audio.NewSTFT(fc.NFFT, fc.HopLength, audio.HannWindow(fc.WinLength, false), true)
		if err != nil {
			t.Fatal(err)
		}
		pre := make([]float32, len(clip.Samples))
		pre[0] = clip.Samples[0]
		for i := 1; i < len(pre); i++ {
			pre[i] = clip.Samples[i] - float32(fc.Preemphasis)*clip.Samples[i-1]
		}
		power := st.Power(pre)
		if len(power) != meta.Count {
			t.Fatalf("%d bins, reference has %d (%v)", len(power), meta.Count, meta.Shape)
		}
		got := make([]float32, len(power))
		for i, v := range power {
			got[i] = float32(v)
		}
		d := compare(t, got, want)

		// The power spectrum spans nine decades, so the number that says
		// whether the transform agrees is the relative one — and it has to be
		// read against how far down the bin is, because below the loudest
		// bins it is the float32 reference that is losing digits, not this.
		// The ladder is logged rather than asserted on except at the top.
		var worst float64
		for _, floor := range []float64{1e-6, 1e-4, 1e-2, 1, 10} {
			var rel float64
			var n int
			for i, v := range power {
				w := math.Abs(float64(want[i]))
				if w <= floor {
					continue
				}
				n++
				if r := math.Abs(v-float64(want[i])) / w; r > rel {
					rel = r
				}
			}
			t.Logf("  bins above %-6g: %6d, worst relative %.3g", floor, n, rel)
			if floor == 1e-2 {
				worst = rel
			}
		}
		t.Logf("max abs %.3g on a peak of %.4g", d.MaxAbs, meta.AbsMax)
		if worst > 1e-4 {
			t.Errorf("power spectrum deviates relatively by %.3g six decades below the peak", worst)
		}
	})

	t.Run("log_mel", func(t *testing.T) {
		want, _ := loadRef(t, m, "log_mel")
		// The unnormalised log-mel, which is Features() without its last
		// pass; recomputed here through the same code path by undoing the
		// normalisation would be circular, so it is built directly.
		fe, err := NewFrontEnd(fc)
		if err != nil {
			t.Fatal(err)
		}
		got := fe.logMel(clip.Samples)
		d := compare(t, got, want)
		t.Logf("max abs %.3g, reference rms %.4g", d.MaxAbs, d.RMS)
		if d.MaxAbs > 1e-3 {
			t.Errorf("log mel deviates by %.3g", d.MaxAbs)
		}
	})
}

// TestFrontEndDetectsErrors is the other end of melTol: a bound only means
// something if a wrong front end fails it. Each case below is a mistake that
// is easy to make and impossible to see in the output — the window
// convention, the preemphasis, the normalisation denominator — and each has
// to miss by orders of magnitude, not by a hair.
func TestFrontEndDetectsErrors(t *testing.T) {
	m := loadManifest(t)
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Skipf("no checkpoint in %s (%v)", modelDir, err)
	}
	clip, err := audio.ReadWAV(fixture)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := loadRef(t, m, "mel")

	cases := []struct {
		name   string
		break_ func(*FrontEnd)
	}{
		{"periodic window", func(f *FrontEnd) {
			st, err := audio.NewSTFT(f.cfg.NFFT, f.cfg.HopLength, audio.HannWindow(f.cfg.WinLength, true), true)
			if err != nil {
				t.Fatal(err)
			}
			f.stft = st
		}},
		{"no preemphasis", func(f *FrontEnd) { f.cfg.Preemphasis = 0 }},
		{"htk mel scale", func(f *FrontEnd) {
			f.filters = audio.MelFilters(f.cfg.SamplingRate, f.cfg.NFFT, f.cfg.NMels,
				0, float64(f.cfg.SamplingRate)/2, audio.HTK, true)
		}},
		{"unnormalised filterbank", func(f *FrontEnd) {
			f.filters = audio.MelFilters(f.cfg.SamplingRate, f.cfg.NFFT, f.cfg.NMels,
				0, float64(f.cfg.SamplingRate)/2, audio.Slaney, false)
		}},
		{"uncentred stft", func(f *FrontEnd) {
			st, err := audio.NewSTFT(f.cfg.NFFT, f.cfg.HopLength, audio.HannWindow(f.cfg.WinLength, false), false)
			if err != nil {
				t.Fatal(err)
			}
			f.stft = st
		}},
	}
	for _, tc := range cases {
		fe, err := NewFrontEnd(cfg.Features)
		if err != nil {
			t.Fatal(err)
		}
		tc.break_(fe)
		feats, err := fe.Features(clip)
		if err != nil || len(feats.Data) != len(want) {
			t.Logf("%s: rejected outright (%v)", tc.name, err)
			continue
		}
		d := compare(t, feats.Data, want)
		t.Logf("%-24s max abs %.3g (%.0fx the bound)", tc.name, d.MaxAbs, d.MaxAbs/melTol)
		if d.MaxAbs < 10*melTol {
			t.Errorf("%s only moved the features by %.3g, which melTol=%g would not catch",
				tc.name, d.MaxAbs, melTol)
		}
	}
}
