package mm3

import (
	"math"
	"path/filepath"
	"testing"
	"time"
)

// snr is 10·log10(‖want‖² / ‖got − want‖²).
func snr(got, want []float32) float64 {
	var se, sw float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		se += d * d
		sw += float64(want[i]) * float64(want[i])
	}
	return 10 * math.Log10(sw/se)
}

// TestVocoder is N6's gate: each oracle window's final latents through the
// vocoder against fp32's waveform, then the windows cropped and
// concatenated as decoders.py does, against the oracle's song.
func TestVocoder(t *testing.T) {
	m := loadDiTManifest(t)
	dev, err := openDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	t0 := time.Now()
	g, err := LoadVocoder(dev, filepath.Join(modelDir, "vocoder"), LatentsFor(WindowFrames))
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	defer g.Destroy()
	t.Logf("staged the vocoder in %v", time.Since(t0))
	var song [2][]float32
	for k := range m.Starts {
		shape, lat := ditTensor(t, m, "w"+itoa(k)+"_final")
		L := shape[1]
		wav, took, err := g.Decode(channels(lat, L))
		if err != nil {
			t.Fatal(err)
		}
		_, want := ditTensor(t, m, "w"+itoa(k)+"_wave")
		n := L * Hop
		for ch := 0; ch < 2; ch++ {
			s := snr(wav[ch], want[ch*n:(ch+1)*n])
			t.Logf("window %d channel %d: %d samples, SNR %.1f dB (GPU %v the pair)", k, ch, n, s, took)
			if !(s >= 40) {
				t.Errorf("window %d channel %d: SNR %.1f dB, under 40", k, ch, s)
			}
		}
		left, right := 0, 0
		if k > 0 {
			left = m.CropLeft * Hop
		}
		if k < len(m.Starts)-1 {
			right = m.CropRight * Hop
		}
		for ch := 0; ch < 2; ch++ {
			song[ch] = append(song[ch], wav[ch][left:n-right]...)
		}
	}
	shape, want := ditTensor(t, m, "song")
	n := shape[1]
	if len(song[0]) != n {
		t.Fatalf("song of %d samples, upstream %d", len(song[0]), n)
	}
	for ch := 0; ch < 2; ch++ {
		for i, v := range song[ch] {
			song[ch][i] = min(max(v, -1), 1)
		}
		t.Logf("song channel %d: %.2f s, SNR %.1f dB", ch, float64(n)/SampleRate, snr(song[ch], want[ch*n:(ch+1)*n]))
	}
}
