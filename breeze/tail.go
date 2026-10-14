package breeze

import (
	"math"
	"time"
)

// The breath the model ends on (research/tts-instruct-vertical.md, "The
// closing breath"). Breeze-TTS-2 closes about half its utterances, upstream's
// fp32 loop included, with 0.2-0.8 s of exhale after the last word, at
// speech level, which a listener (and parakeet) can hear as a word: "west,
// and". EOS is all or nothing (0 every frame, then 1), so no sampler setting
// moves it, and its log-probability is no higher at the breath than
// mid-word (~1e-7). The cut is on the audio. Nothing in the signal alone
// tells the breath from an excited, rough "show" or a whispered "us" (both
// measured as unvoiced as the breath), so where the speech ends is
// GenOptions.Ending's call, from the words.
//
// tailTrim passes audio on as it comes, except what follows the last voiced
// run (two 40 ms windows in a row, periodic at a pitch lag), which it holds,
// up to tailHold: a closing breath is unvoiced. At the end it hands Ending
// up to tailContext of the utterance with the held tail -- parakeet hears
// the breath as a word only with the sentence before it (on the last 3 s it
// heard "west." and nothing after) -- and cuts what is held past where
// Ending says the speech ends, behind a tailFade ramp.
const (
	tailWin     = SampleRate / 25  // 40 ms analysis window
	tailHop     = tailWin / 2      // 20 ms
	tailFade    = SampleRate / 25  // 40 ms ramp to zero after the cut
	tailHold    = SampleRate       // the longest unvoiced stretch held back
	tailContext = 30 * SampleRate  // audio before the held tail that Ending sees
	tailFloor   = 0.01             // RMS below which a window is silence (-40 dB)
	tailVoice   = 0.6              // normalised autocorrelation of a voiced window
	lagMin      = SampleRate / 400 // pitch 70-400 Hz
	lagMax      = SampleRate / 70
)

type tailTrim struct {
	emit   func([]float32) error
	ending func([]float32) (int, error)
	all    []float32 // from tailContext before the earlier of sent and next on
	sent   int       // samples handed on (in all)
	next   int       // the next window's start (in all)
	held   int       // where holding may start: the end of the last voiced run (in all)
	heard  bool      // a voiced run yet
	run    int       // voiced windows in a row
}

func (t *tailTrim) push(s []float32) error {
	t.all = append(t.all, s...)
	for ; t.next+tailWin <= len(t.all); t.next += tailHop {
		w := t.all[t.next : t.next+tailWin]
		if rms(w) < tailFloor || !voiced(w) {
			t.run = 0
			continue
		}
		if t.run++; t.run >= 2 {
			t.held, t.heard = t.next+tailWin, true
		}
	}
	lim := len(t.all)
	if t.heard {
		lim = min(lim, max(t.held, len(t.all)-tailHold))
	}
	if lim <= t.sent {
		return nil
	}
	err := t.emit(append([]float32(nil), t.all[t.sent:lim]...))
	t.sent = lim
	// Drop what neither the windows nor Ending's context still need.
	if off := min(t.sent, t.next) - tailContext; off >= 1<<16 {
		t.all = t.all[:copy(t.all, t.all[off:])]
		t.sent, t.next, t.held = t.sent-off, t.next-off, t.held-off
	}
	return err
}

// finish hands on what is held, cut where Ending says the speech ends; it
// returns what was dropped.
func (t *tailTrim) finish() (time.Duration, error) {
	n := len(t.all) - t.sent
	if n <= tailFade {
		if n > 0 {
			return 0, t.emit(append([]float32(nil), t.all[t.sent:]...))
		}
		return 0, nil
	}
	from := max(0, t.sent-tailContext)
	end, err := t.ending(t.all[from:])
	if err != nil {
		return 0, err
	}
	cut := max(t.sent, from+end)
	if cut+tailFade >= len(t.all) {
		return 0, t.emit(append([]float32(nil), t.all[t.sent:]...))
	}
	out := append([]float32(nil), t.all[t.sent:cut+tailFade]...)
	for i := range tailFade {
		out[cut-t.sent+i] *= float32(tailFade-i) / tailFade
	}
	return time.Duration(len(t.all)-cut-tailFade) * time.Second / SampleRate, t.emit(out)
}

func rms(w []float32) float64 {
	var e float64
	for _, v := range w {
		e += float64(v) * float64(v)
	}
	return math.Sqrt(e / float64(len(w)))
}

// voiced is a periodic window: its normalised autocorrelation peaks above
// tailVoice at a pitch lag.
func voiced(w []float32) bool {
	var m float64
	for _, v := range w {
		m += float64(v)
	}
	m /= float64(len(w))
	x := make([]float32, len(w))
	var e0 float64
	for i, v := range w {
		x[i] = v - float32(m)
		e0 += float64(x[i]) * float64(x[i])
	}
	for lag := lagMin; lag <= lagMax; lag++ {
		var c float32
		for i := 0; i+lag < len(x); i++ {
			c += x[i] * x[i+lag]
		}
		if float64(c) > tailVoice*e0 {
			return true
		}
	}
	return false
}
