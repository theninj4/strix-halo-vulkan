// Command asr transcribes a WAV file with parakeet-tdt-0.6b-v3: audio in,
// text out, on the CPU reference (SPEECH.md stages S3-S5) or on Vulkan (S6).
//
// The stage timings it prints are what decided the order of the port: the
// front end is a few thousand 512-point FFTs and the decode loop is one
// 8198-wide projection per emitted token, while the encoder is ~180 GFLOP for
// eleven seconds of audio -- 91.5% of the CPU time -- all of it in shapes the
// engine's GEMM ladder already covers.
//
// -gpu moves the whole model onto the device -- the subsampling stack (S7),
// the 24 conformer layers (S6), and the projector, prediction network, joint
// and TDT loop (S8) -- leaving the host the front end, the decode loop's
// cursor arithmetic and the tokenizer. -hostsub and -hostdec put the
// convolutions and the transducer tail back on the CPU, which is what the S7
// and S8 measurements are against.
//
//	go run ./cmd/asr testdata/jfk.wav
//	go run ./cmd/asr -gpu -reps 3 testdata/jfk.wav
//	go run ./cmd/asr -gpu -profile testdata/jfk.wav
//	go run ./cmd/asr -gpu -hostdec testdata/jfk.wav
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
	gpu := flag.Bool("gpu", false, "run the encoder on Vulkan")
	hostsub := flag.Bool("hostsub", false, "with -gpu, run the subsampling stack on the CPU as S6 did")
	hostdec := flag.Bool("hostdec", false, "with -gpu, run the projector and the TDT loop on the CPU as S7 did")
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
	// The transducer tail: the projector, the joint and the TDT loop, which
	// -gpu moves onto the device as one step. On the host they are two, so
	// the timing below reports them apart and the GPU path charges the whole
	// tail to "decode".
	decode := func(hidden *parakeet.Mat, valid int) (*parakeet.Transcript, error) {
		enc, err := m.Projector.Apply(hidden)
		if err != nil {
			return nil, err
		}
		return m.Decode(enc, valid)
	}
	var tail *parakeet.GPUDecoder
	var gpuEncoder *parakeet.GPUEncoder
	if *gpu {
		dev, release := openDevice()
		defer release()
		// Sized for this clip: one front-end pass up front says how many
		// encoder frames it becomes, and the subsampling arenas follow from
		// that. What the length decides is the GEMM tile, and PlanFor picks
		// that per run.
		feats, err := m.FrontEnd.Features(clip)
		must(err)
		g, err := parakeet.NewGPUEncoder(dev, m.Encoder, m.Encoder.Subsampling.ValidLength(feats.Frames), nil)
		must(err)
		defer g.Destroy()
		gpuEncoder = g
		g.HostSubsampling = *hostsub
		where := "on the device"
		if *hostsub {
			where = "on the host"
		}
		fmt.Printf("gpu: %d layers, %d MB of weights, %d MB of arenas, plan %s, subsampling %s\n\n",
			g.Layers(), g.WeightBytes()>>20, g.ActivationBytes()>>20, g.Plan()[parakeet.ProjQ], where)
		encode = g.ApplyMel
		if *profile {
			defer func() { printProfile(g, m, clip) }()
		}
		if !*hostdec {
			gd, err := parakeet.NewGPUDecoder(dev, m, g.MaxFrames())
			must(err)
			defer gd.Destroy()
			// Attached, so the [T, 1024] hidden states never come back to
			// the host: a 552 KB read out of a device-local host-visible
			// buffer is 2.4 ms on this part, which would be half the
			// pipeline's remaining decode time.
			must(gd.Attach(g))
			fmt.Printf("gpu: transducer tail on the device, %d MB of weights, %d MB of arenas, plan %s\n\n",
				gd.WeightBytes()>>20, gd.ActivationBytes()>>20, gd.Plan()[parakeet.DecJoint])
			tail = gd
		}
	}

	var best struct{ front, encode, decode time.Duration }
	var out *parakeet.Transcript
	for i := 0; i < *reps; i++ {
		t0 := time.Now()
		feats, err := m.FrontEnd.Features(clip)
		must(err)
		t1 := time.Now()

		mel := &parakeet.Mat{Rows: feats.Frames, Cols: feats.Mels, Data: feats.Data}
		var t2 time.Time
		if tail != nil {
			// Resident: the encoder leaves its hidden states in the arena and
			// the tail reads them there, so the boundary between the two rows
			// below is a fence and not a copy.
			_, err := gpuEncoder.RunMel(mel, feats.Valid)
			must(err)
			t2 = time.Now()
			out, err = tail.DecodeResident()
			must(err)
		} else {
			hidden, valid, err := encode(mel, feats.Valid)
			must(err)
			t2 = time.Now()
			out, err = decode(hidden, valid)
			must(err)
		}
		t4 := time.Now()

		if i == 0 {
			fmt.Printf("%d mel frames (%d valid) -> %d encoder frames\n", feats.Frames, feats.Valid, out.Frames)
			best.front, best.encode, best.decode = t1.Sub(t0), t2.Sub(t1), t4.Sub(t2)
		}
		best.front = min(best.front, t1.Sub(t0))
		best.encode = min(best.encode, t2.Sub(t1))
		best.decode = min(best.decode, t4.Sub(t2))
	}

	total := best.front + best.encode + best.decode
	fmt.Printf("\n%-12s %9s  %5s\n", "stage", "time", "share")
	for _, s := range []struct {
		name string
		d    time.Duration
	}{
		{"front end", best.front},
		{"encoder", best.encode},
		{"decode", best.decode},
		{"total", total},
	} {
		fmt.Printf("%-12s %9v  %4.1f%%\n", s.name, s.d.Round(time.Millisecond), 100*float64(s.d)/float64(total))
	}
	fmt.Printf("\n%.2fx real time (%d emissions, %d tokens",
		clip.Duration()/total.Seconds(), len(out.Steps), len(out.Tokens))
	if tail != nil {
		fmt.Printf(", %d submits, %v an emission",
			tail.Submits, (best.decode / time.Duration(max(tail.Submits, 1))).Round(time.Microsecond))
	}
	fmt.Println(")")

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
// the front end, the upload and the read-back.
//
// The subsampling stack's ten dispatches are reported first and separately,
// since they are once per clip where the other 960 are forty per layer.
func printProfile(g *parakeet.GPUEncoder, m *parakeet.Model, clip *audio.Clip) {
	feats, err := m.FrontEnd.Features(clip)
	must(err)
	mel := &parakeet.Mat{Rows: feats.Frames, Cols: feats.Mels, Data: feats.Data}
	stages, _, err := g.ProfileMel(mel, feats.Valid)
	must(err)
	type total struct {
		n int
		d time.Duration
	}
	byKind := map[string]total{}
	var sum, sub time.Duration
	for _, s := range stages {
		e := byKind[s.Kind]
		e.n, e.d = e.n+1, e.d+s.GPU
		byKind[s.Kind] = e
		sum += s.GPU
		if s.Layer < 0 {
			sub += s.GPU
		}
	}
	fmt.Printf("\n%d dispatches, %v on the device, %.0f GFLOP, %.1f TFLOP/s\n",
		len(stages), sum.Round(time.Microsecond), g.FLOPs(g.Rows())/1e9,
		g.FLOPs(g.Rows())/(sum-sub).Seconds()/1e12)
	show := func(labels []string) {
		for _, k := range labels {
			e, ok := byKind[k]
			if !ok {
				continue
			}
			delete(byKind, k)
			fmt.Printf("  %-18s %3d  %8v  %4.1f%%\n", k, e.n, e.d.Round(time.Microsecond), 100*float64(e.d)/float64(sum))
		}
	}
	if sub > 0 {
		fmt.Printf("\nsubsampling: %v, %.2f GFLOP, %.0f GFLOP/s\n",
			sub.Round(time.Microsecond), g.SubFLOPs()/1e9, g.SubFLOPs()/sub.Seconds()/1e9)
		show(g.SubLabels())
		fmt.Printf("\nlayers: %v\n", (sum - sub).Round(time.Microsecond))
	}
	show(g.Labels())
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
