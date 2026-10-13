// Package loops measures how much of a song repeats itself at one short
// lag (MUSIC.md A14e): the failure no auto-score component sees. A song
// that is a few seconds tiled scores like any other song on PMI, Audiobox
// and SongEval, and the user rated it 1.
//
// The audio becomes 64 log-mel bands (30 Hz-8 kHz, where mp3 does little
// damage) every 1024 samples, centred on the song's mean spectrum and
// normalised, so that a frame's similarity to another is a cosine. For
// every lag from 0.5 to 12 s the frame at t is compared with the frame a
// lag later (or one frame after that, so a period that is not a whole
// number of frames costs nothing). Two numbers come out:
//
//   - Repetition: the share of frames at least 0.9 alike at the lag where
//     most are. Graded, and the generated songs the user disliked repeat
//     more (Spearman -0.51 over 20 songs, A14e).
//   - the longest loop: the longest stretch whose frames are 0.98 alike at
//     one lag (gaps under 0.25 s bridged), plus the lag itself, in seconds
//     and as a share of the song. A tiled clip scores the whole song, an mp3
//     of it too; no generated song in the A14e set passes 0.54 of itself.
//
// Frames 50 dB under the song's loudest never match, so silence is not a
// loop. Where the thresholds turn into a flag is A14e's fit.
package loops

import (
	"math"
	"runtime"
	"sync"
	"time"

	"strix-halo-vulkan/audio"
)

// The geometry and thresholds, from the A14e prototype.
const (
	SampleRate = 48000
	fftSize    = 2048
	hop        = 1024
	bands      = 64
	fMin, fMax = 30, 8000
	quietDB    = 50 // under the loudest frame: never a match

	MinLag, MaxLag = 0.5, 12.0 // seconds
	RepeatAlike    = 0.9       // a frame repeats (Repetition)
	LoopAlike      = 0.98      // a frame is a copy (the loop)
	gapSeconds     = 0.25      // a loop survives misses this long
)

// Result is a song's repetition.
type Result struct {
	// Repetition is the share of frames that repeat at RepetitionLag.
	Repetition    float64 `json:"repetition"`
	RepetitionLag float64 `json:"repetition_lag"`
	// LoopSeconds is the longest stretch that copies what came LoopLag
	// seconds before it, the first period included; LoopShare is its share
	// of the song.
	LoopSeconds float64       `json:"loop_seconds"`
	LoopShare   float64       `json:"loop_share"`
	LoopLag     float64       `json:"loop_lag"`
	Took        time.Duration `json:"-"`
}

// Measure measures interleaved stereo at 48 kHz.
func Measure(wav []float32) Result {
	t0 := time.Now()
	mono := make([]float32, len(wav)/2)
	for i := range mono {
		mono[i] = (wav[2*i] + wav[2*i+1]) / 2
	}
	f := features(mono)
	r := measure(f, len(f)/bands)
	if secs := float64(len(mono)) / SampleRate; secs > 0 {
		r.LoopShare = min(1, r.LoopSeconds/secs)
	}
	r.Took = time.Since(t0)
	return r
}

// features is the [frames, bands] unit vectors, quiet frames zero.
func features(x []float32) []float32 {
	st, err := audio.NewSTFT(fftSize, hop, audio.HannWindow(fftSize, true), false)
	if err != nil || len(x) < fftSize {
		return nil
	}
	pow := st.Power(x)
	bins := st.Bins()
	n := len(pow) / bins
	fb := audio.MelFilters(SampleRate, fftSize, bands, fMin, fMax, audio.HTK, false)
	logs := make([]float64, n*bands)
	loud := make([]float64, n)
	peak := math.Inf(-1)
	for t := 0; t < n; t++ {
		row := pow[t*bins : (t+1)*bins]
		var total float64
		for m := 0; m < bands; m++ {
			var e float64
			for b, w := range fb[m*bins : (m+1)*bins] {
				if w != 0 {
					e += float64(w) * row[b]
				}
			}
			total += e
			logs[t*bands+m] = math.Log(e + 1e-6)
		}
		loud[t] = 10 * math.Log10(total+1e-20)
		peak = max(peak, loud[t])
	}
	mean := make([]float64, bands)
	kept := 0
	for t := 0; t < n; t++ {
		if loud[t] >= peak-quietDB {
			kept++
			for m := range mean {
				mean[m] += logs[t*bands+m]
			}
		}
	}
	out := make([]float32, n*bands)
	if kept == 0 {
		return out
	}
	for m := range mean {
		mean[m] /= float64(kept)
	}
	for t := 0; t < n; t++ {
		if loud[t] < peak-quietDB {
			continue
		}
		var norm float64
		for m := 0; m < bands; m++ {
			v := logs[t*bands+m] - mean[m]
			logs[t*bands+m] = v
			norm += v * v
		}
		norm = math.Sqrt(norm) + 1e-9
		for m := 0; m < bands; m++ {
			out[t*bands+m] = float32(logs[t*bands+m] / norm)
		}
	}
	return out
}

// lagResult is one lag's two measures, in frames.
type lagResult struct {
	cover float64
	run   int
}

func measure(f []float32, n int) Result {
	fps := float64(SampleRate) / hop
	lo, hi := int(MinLag*fps), int(MaxLag*fps)
	if n < lo+3 {
		return Result{}
	}
	hi = min(hi, n-3)
	gap := int(gapSeconds * fps)
	res := make([]lagResult, hi+1)
	var wg sync.WaitGroup
	next := make(chan int)
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for L := range next {
				res[L] = lag(f, n, L, gap)
			}
		}()
	}
	for L := lo; L <= hi; L++ {
		next <- L
	}
	close(next)
	wg.Wait()
	var r Result
	bestRun := 0
	for L := lo; L <= hi; L++ {
		if res[L].cover > r.Repetition {
			r.Repetition, r.RepetitionLag = res[L].cover, float64(L)/fps
		}
		if res[L].run > bestRun {
			bestRun, r.LoopLag = res[L].run, float64(L)/fps
		}
	}
	if bestRun > 0 {
		r.LoopSeconds = float64(bestRun)/fps + r.LoopLag
	}
	return r
}

// lag compares frame t with t+L and t+L+1 for every t that has both.
func lag(f []float32, n, L, gap int) lagResult {
	m := n - L - 1
	var cover, cur, miss, run int
	for t := 0; t < m; t++ {
		a := f[t*bands : (t+1)*bands]
		s := max(dot(a, f[(t+L)*bands:(t+L+1)*bands]), dot(a, f[(t+L+1)*bands:(t+L+2)*bands]))
		if s >= RepeatAlike {
			cover++
		}
		switch {
		case s >= LoopAlike:
			cur += 1 + miss
			miss = 0
		case cur > 0 && miss < gap:
			miss++
		default:
			cur, miss = 0, 0
		}
		run = max(run, cur)
	}
	return lagResult{cover: float64(cover) / float64(m), run: run}
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
