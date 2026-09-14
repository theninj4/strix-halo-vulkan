// Command asr transcribes a WAV file with parakeet-tdt-0.6b-v3: audio in,
// text out, on the CPU reference (SPEECH.md stages S3-S5) or on Vulkan (S6).
//
// The stage timings it prints are what decided the order of the port: the
// front end is a few thousand 512-point FFTs and the decode loop is one
// 8198-wide projection per emitted token, while the encoder is ~180 GFLOP for
// eleven seconds of audio -- 91.5% of the CPU time -- all of it in shapes the
// engine's GEMM ladder already covers.
//
// -gpu moves the 24 conformer layers onto the device and leaves the front
// end, the subsampling stack, the projector and the TDT loop on the host, so
// the "encoder" row it prints then covers the CPU subsampling *and* the GPU
// layers, which is what a caller actually waits for.
//
//	go run ./cmd/asr testdata/jfk.wav
//	go run ./cmd/asr -gpu -reps 3 testdata/jfk.wav
//	go run ./cmd/asr -gpu -profile testdata/jfk.wav
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"strix-halo-vulkan/audio"
	"strix-halo-vulkan/parakeet"
	"strix-halo-vulkan/vk"
)

func main() {
	log.SetFlags(0)
	model := flag.String("model", "models/parakeet-tdt-0.6b-v3", "checkpoint directory")
	reps := flag.Int("reps", 1, "timed repetitions; the best of each stage is reported")
	verbose := flag.Bool("v", false, "print the decode trace, one line per emission")
	gpu := flag.Bool("gpu", false, "run the conformer layers on Vulkan")
	profile := flag.Bool("profile", false, "with -gpu, time every dispatch on the device and print the total per kind")
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

	// The encoder the run uses: the CPU reference, or the Vulkan one with the
	// same signature (parakeet.GPUEncoder.ApplyMel).
	encode := m.Encoder.Apply
	if *gpu {
		dev, release := openDevice()
		defer release()
		// Sized for this clip: one front-end pass up front says how many
		// encoder frames it becomes. The arenas are a few MB either way; what
		// the length decides is the GEMM tile, and PlanFor picks that per run.
		feats, err := m.FrontEnd.Features(clip)
		must(err)
		g, err := parakeet.NewGPUEncoder(dev, m.Encoder, m.Encoder.Subsampling.ValidLength(feats.Frames), nil)
		must(err)
		defer g.Destroy()
		fmt.Printf("gpu: %d layers, %d MB of weights, %d MB of arenas, plan %s\n\n",
			g.Layers(), g.WeightBytes()>>20, g.ActivationBytes()>>20, g.Plan()[parakeet.ProjQ])
		encode = g.ApplyMel
		if *profile {
			defer func() { printProfile(g) }()
		}
	}

	var best struct{ front, encode, project, decode time.Duration }
	var out *parakeet.Transcript
	for i := 0; i < *reps; i++ {
		t0 := time.Now()
		feats, err := m.FrontEnd.Features(clip)
		must(err)
		t1 := time.Now()

		mel := &parakeet.Mat{Rows: feats.Frames, Cols: feats.Mels, Data: feats.Data}
		hidden, valid, err := encode(mel, feats.Valid)
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

// openDevice picks the compute device the rest of this repository picks: the
// integrated Strix Halo part when it is there, the first one otherwise.
func openDevice() (*vk.Device, func()) {
	inst, err := vk.NewInstance("asr")
	must(err)
	devices, err := inst.PhysicalDevices()
	must(err)
	if len(devices) == 0 {
		log.Fatal("no Vulkan devices")
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == 0x1586 {
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	must(err)
	sgs, err := phys.SubgroupSizeControl()
	must(err)
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	must(err)
	fmt.Printf("device: %s\n", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// printProfile times the graph one dispatch at a time and totals it by kind.
// Wall clock around the encoder is not a measurement of it: that also carries
// the host's subsampling, the upload and the read-back.
func printProfile(g *parakeet.GPUEncoder) {
	stages, _, err := g.Profile(g.Read(g.TensorX(), g.Dim()), g.Valid())
	must(err)
	type total struct {
		n int
		d time.Duration
	}
	byKind := map[string]total{}
	var sum time.Duration
	for _, s := range stages {
		e := byKind[s.Kind]
		e.n, e.d = e.n+1, e.d+s.GPU
		byKind[s.Kind] = e
		sum += s.GPU
	}
	fmt.Printf("\n%d dispatches, %v on the device, %.0f GFLOP, %.1f TFLOP/s\n",
		len(stages), sum.Round(time.Microsecond), g.FLOPs(g.Rows())/1e9,
		g.FLOPs(g.Rows())/sum.Seconds()/1e12)
	for _, k := range g.Labels() {
		e, ok := byKind[k]
		if !ok {
			continue
		}
		delete(byKind, k)
		fmt.Printf("  %-18s %3d  %8v  %4.1f%%\n", k, e.n, e.d.Round(time.Microsecond), 100*float64(e.d)/float64(sum))
	}
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
