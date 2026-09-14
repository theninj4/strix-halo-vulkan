// Command asr transcribes a WAV file with parakeet-tdt-0.6b-v3: audio in,
// text out, on the CPU reference (SPEECH.md stages S3-S5).
//
// It is also the profile the Vulkan port is aimed at. The stage timings it
// prints are the whole argument for what moves to the GPU first: the front
// end is a few thousand 512-point FFTs and the decode loop is one 8198-wide
// projection per emitted token, while the encoder is ~180 GFLOP for eleven
// seconds of audio, all of it in shapes the engine's GEMM ladder already
// covers.
//
//	go run ./cmd/asr testdata/jfk.wav
//	go run ./cmd/asr -v -reps 3 testdata/jfk.wav
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"strix-halo-vulkan/audio"
	"strix-halo-vulkan/parakeet"
)

func main() {
	log.SetFlags(0)
	model := flag.String("model", "models/parakeet-tdt-0.6b-v3", "checkpoint directory")
	reps := flag.Int("reps", 1, "timed repetitions; the best of each stage is reported")
	verbose := flag.Bool("v", false, "print the decode trace, one line per emission")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "usage: %s [-model dir] [-v] <file.wav>\n", os.Args[0])
		os.Exit(2)
	}

	clip, err := audio.ReadWAV(flag.Arg(0))
	must(err)
	fmt.Printf("%s: %.3f s at %d Hz\n", flag.Arg(0), clip.Duration(), clip.Rate)

	start := time.Now()
	m, err := parakeet.Load(*model)
	must(err)
	fmt.Printf("loaded %s in %v\n\n", *model, time.Since(start).Round(time.Millisecond))

	var best struct{ front, encode, project, decode time.Duration }
	var out *parakeet.Transcript
	for i := 0; i < *reps; i++ {
		t0 := time.Now()
		feats, err := m.FrontEnd.Features(clip)
		must(err)
		t1 := time.Now()

		mel := &parakeet.Mat{Rows: feats.Frames, Cols: feats.Mels, Data: feats.Data}
		hidden, valid, err := m.Encoder.Apply(mel, feats.Valid)
		must(err)
		t2 := time.Now()

		enc, err := m.Projector.Apply(hidden)
		must(err)
		t3 := time.Now()

		out, err = m.Decode(enc, valid)
		must(err)
		t4 := time.Now()

		if i == 0 {
			fmt.Printf("%d mel frames (%d valid) -> %d encoder frames (%d valid)\n",
				feats.Frames, feats.Valid, hidden.Rows, valid)
			best.front, best.encode, best.project, best.decode = t1.Sub(t0), t2.Sub(t1), t3.Sub(t2), t4.Sub(t3)
		}
		best.front = min(best.front, t1.Sub(t0))
		best.encode = min(best.encode, t2.Sub(t1))
		best.project = min(best.project, t3.Sub(t2))
		best.decode = min(best.decode, t4.Sub(t3))
	}

	total := best.front + best.encode + best.project + best.decode
	fmt.Printf("\n%-12s %9s  %5s\n", "stage", "time", "share")
	for _, s := range []struct {
		name string
		d    time.Duration
	}{
		{"front end", best.front},
		{"encoder", best.encode},
		{"projector", best.project},
		{"decode", best.decode},
		{"total", total},
	} {
		fmt.Printf("%-12s %9v  %4.1f%%\n", s.name, s.d.Round(time.Millisecond), 100*float64(s.d)/float64(total))
	}
	fmt.Printf("\n%.2fx real time (%d emissions, %d tokens)\n",
		clip.Duration()/total.Seconds(), len(out.Steps), len(out.Tokens))

	if *verbose {
		fmt.Println()
		for _, s := range out.Steps {
			piece := "<blank>"
			if s.Token != m.Config.BlankTokenID {
				piece, err = m.Tokenizer.Piece(s.Token)
				must(err)
			}
			// Each frame is one subsampling stride of mel hops: 8 x 10 ms.
			fmt.Printf("  %6.2fs  frame %3d  +%d  %5d  %q\n",
				float64(s.Frame)*8*float64(m.Config.Features.HopLength)/float64(m.Config.Features.SamplingRate),
				s.Frame, s.Duration, s.Token, piece)
		}
	}
	fmt.Printf("\n%s\n", out.Text)
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
