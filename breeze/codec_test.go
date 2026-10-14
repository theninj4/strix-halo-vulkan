package breeze

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"
)

var (
	codecOnce sync.Once
	testCodec *Codec
	codecErr  error
)

// loadCodec stages the codec once for chunks of up to 128 frames.
func loadCodec(t testing.TB) *Codec {
	t.Helper()
	dev := device(t)
	codecOnce.Do(func() {
		t0 := time.Now()
		testCodec, codecErr = LoadCodec(dev, modelDir(), 128, 2048)
		if codecErr == nil {
			t.Logf("staged the codec in %v", time.Since(t0))
		}
	})
	if codecErr != nil {
		t.Skipf("no checkpoint: %v", codecErr)
	}
	return testCodec
}

// snr is 10 log10(sum want^2 / sum (got - want)^2).
func snr(got, want []float32) float64 {
	var s, e float64
	for i := range want {
		d := float64(got[i] - want[i])
		s += float64(want[i]) * float64(want[i])
		e += d * d
	}
	return 10 * math.Log10(s/e)
}

// decodeChunks streams frames through the codec chunk at a time.
func decodeChunks(t *testing.T, c *Codec, fr [][16]int32, chunk int) ([]float32, time.Duration) {
	c.Reset()
	var wav []float32
	var took time.Duration
	for i := 0; i < len(fr); i += chunk {
		w, dt, err := c.Decode(fr[i:min(i+chunk, len(fr))])
		if err != nil {
			t.Fatal(err)
		}
		wav = append(wav, w...)
		took += dt
	}
	return wav, took
}

// codecSNR is the bound against the fp32 oracle: measured 57.9-63.7 dB over
// the four dumps (fp16 operands, fp32 sums), the same at every chunk size.
const codecSNR = 54.0

// TestCodec decodes every dumped clip whole and in chunks of 1, 7 and 25
// frames, and holds each to the oracle's whole-clip decode (decoder.forward
// in fp32).
func TestCodec(t *testing.T) {
	c := loadCodec(t)
	for _, name := range []string{"design", "plain", "zh", "long"} {
		if _, err := os.Stat(filepath.Join(refDir, name, "tensors.safetensors")); err != nil {
			t.Logf("%s: no dump", name)
			continue
		}
		ref := refCase(t, name)
		fr := frames(t, ref)
		want := f32(t, ref["wav"])
		for _, chunk := range []int{len(fr), 25, 7, 1} {
			if chunk > c.Chunk {
				continue
			}
			got, took := decodeChunks(t, c, fr, chunk)
			if len(got) != len(want) {
				t.Fatalf("%s: %d samples, want %d", name, len(got), len(want))
			}
			db := snr(got, want)
			t.Logf("%s: %d frames in chunks of %d: %.1f dB in %v", name, len(fr), chunk, db, took.Round(time.Microsecond))
			if db < codecSNR {
				t.Errorf("%s: %.1f dB in chunks of %d; the bound is %.0f", name, db, chunk, codecSNR)
			}
		}
	}
}

// TestKthLargest holds the heap to a sort.
func TestKthLargest(t *testing.T) {
	var buf []float32
	for n := 1; n < 300; n += 7 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((i*7919)%101) - 50
		}
		s := append([]float32(nil), v...)
		slices.Sort(s)
		for _, k := range []int{1, 2, 50, n} {
			if k > n {
				continue
			}
			kth, top := kthLargest(v, k, &buf)
			if kth != s[n-k] || top != s[n-1] {
				t.Fatalf("n %d k %d: %v %v, want %v %v", n, k, kth, top, s[n-k], s[n-1])
			}
		}
	}
}
