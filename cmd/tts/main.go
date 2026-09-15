// Command tts synthesises speech with Kokoro-82M: phonemes in, a WAV out, on
// the CPU reference (SPEECH.md stages T2-T3).
//
// It takes phonemes by default, and **text** with -text. Grapheme-to-phoneme
// lives outside the model on purpose: the checkpoint's vocabulary is 178 IPA
// symbols, and every language it supports needs a different front end to
// reach them. `reference/dump_kokoro.py` records the phonemes misaki produces
// for its test sentence, and -phonemes defaults to those; -text runs `g2p`,
// which needs the lexicon `reference/convert_misaki.py` writes and, for words
// outside it, libespeak-ng — without which those words are dropped with a
// warning (SPEECH.md T5).
//
// The stage timings it prints are what T4 has to work with: ALBERT and the
// recurrences are a fifth of a second between them and the vocoder is the
// rest, which is the split between a latency-bound problem and a
// bandwidth-bound one.
//
//	go run ./cmd/tts -gpu -text 'Hello there.' -o out.wav
//	go run ./cmd/tts -gpu -o out.wav
//	go run ./cmd/tts -voice bm_george -noise 1 -o out.wav
//	go run ./cmd/tts -phonemes 'hɛlˈO wˈɜɹld.' -o out.wav
//	go run ./cmd/tts -list
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"time"

	"strix-halo-vulkan/audio"
	"strix-halo-vulkan/g2p"
	"strix-halo-vulkan/kokoro"
	"strix-halo-vulkan/vk"
)

// The phonemes misaki produces for "The quick brown fox jumps over the lazy
// dog.", which is what reference/out/kokoro was dumped from.
const defaultPhonemes = "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ."

func main() {
	log.SetFlags(0)
	dir := flag.String("model", "models/Kokoro-82M", "converted checkpoint directory")
	phonemes := flag.String("phonemes", defaultPhonemes, "IPA phonemes to speak")
	text := flag.String("text", "", "English text to speak; needs the lexicon in -lexicon (SPEECH.md T5)")
	lexdir := flag.String("lexicon", "models/misaki", "misaki lexicon directory, from reference/convert_misaki.py")
	voice := flag.String("voice", "af_heart", "voice pack name")
	speed := flag.Float64("speed", 1, "duration divisor; >1 is faster and shorter")
	noise := flag.Int64("noise", 0, "seed for the excitation noise; 0 leaves it off, which is what the reference dump used")
	out := flag.String("o", "", "write a 24 kHz WAV here")
	reps := flag.Int("reps", 1, "timed repetitions; the best of each stage is reported")
	list := flag.Bool("list", false, "print the available voices and exit")
	gpu := flag.Bool("gpu", false, "run the generator's residual blocks on Vulkan")
	flag.Parse()

	t0 := time.Now()
	model, err := kokoro.Load(*dir)
	if err != nil {
		log.Fatalf("%v\n(the checkpoint ships as a pickle; run reference/convert_kokoro.py first)", err)
	}
	load := time.Since(t0)

	if *list {
		names := make([]string, 0, len(model.Voices))
		for n := range model.Voices {
			names = append(names, n)
		}
		sort.Strings(names)
		fmt.Printf("%d voices: %s\n", len(names), strings.Join(names, " "))
		return
	}
	if *noise != 0 {
		model.SetExcitationNoise(*noise)
	}

	if *text != "" {
		lex, err := g2p.Load(*lexdir, false)
		if err != nil {
			log.Fatalf("%v\n(run reference/convert_misaki.py to write it)", err)
		}
		if e, err := g2p.Open(false); err == nil {
			defer e.Close()
			lex.Fallback = e
		} else {
			fmt.Fprintf(os.Stderr, "warning: no espeak fallback (%v); "+
				"words outside the dictionary will be dropped\n", err)
		}
		ps, unknown := lex.Phonemize(*text)
		if unknown > 0 {
			fmt.Fprintf(os.Stderr,
				"warning: %d token(s) could not be pronounced and were dropped\n", unknown)
		}
		fmt.Printf("%q\n  -> %q\n\n", *text, ps)
		*phonemes = ps
	}

	ids, dropped := model.Config.Phonemes(*phonemes)
	runes := len([]rune(*phonemes))
	if dropped > 0 {
		fmt.Fprintf(os.Stderr, "warning: %d of %d characters are outside the vocabulary and were dropped\n",
			dropped, runes)
	}
	decStyle, predStyle, err := model.Style(*voice, runes-dropped)
	if err != nil {
		log.Fatal(err)
	}

	if *gpu {
		// The frame count the arenas are sized for comes from the durations,
		// so the prosody has to run before the device can be staged.
		p, err := model.Prosody(ids, predStyle, float32(*speed))
		if err != nil {
			log.Fatal(err)
		}
		t := time.Now()
		dev, cleanup, err := openDevice()
		if err != nil {
			log.Fatal(err)
		}
		defer cleanup()
		if err := model.AttachGPU(dev, p.Frames, decStyle, kokoro.DefaultConvKernel); err != nil {
			log.Fatal(err)
		}
		defer model.DetachGPU()
		fmt.Printf("staged %d generator blocks on %s in %.0fms\n\n",
			2*4, dev.Physical().Name, ms(time.Since(t)))
	}

	var prosody *kokoro.Prosody
	var wav []float32
	best := struct {
		prosody, vocoder, total          time.Duration
		decoder, excitation, stage, tail time.Duration
	}{}
	for r := 0; r < *reps; r++ {
		t := time.Now()
		p, err := model.Prosody(ids, predStyle, float32(*speed))
		if err != nil {
			log.Fatal(err)
		}
		tp := time.Since(t)
		t = time.Now()
		samples, tr, err := model.Vocoder.Apply(p.ASR, p.F0, p.Energy, decStyle)
		if err != nil {
			log.Fatal(err)
		}
		tv := time.Since(t)
		if r == 0 || tp+tv < best.total {
			best.prosody, best.vocoder, best.total = tp, tv, tp+tv
			best.decoder, best.excitation = tr.Decoder, tr.Generator.Excitation
			best.stage, best.tail = tr.Generator.Stage, tr.Generator.Tail
		}
		prosody, wav = p, samples
	}

	seconds := prosody.Seconds(model.Config)
	fmt.Printf("%q\n", *phonemes)
	fmt.Printf("%s: %d phonemes -> %d tokens -> %d frames -> %d samples = %.3f s at %d Hz\n",
		*voice, runes-dropped, len(ids), prosody.Frames, len(wav), seconds, model.Config.SamplingRate)
	pt := prosody.Times
	fmt.Printf("\n    stage         time\n")
	fmt.Printf("    bert        %6.0fms\n", ms(pt.BERT))
	fmt.Printf("    dur enc     %6.0fms\n", ms(pt.DurEncoder))
	fmt.Printf("    durations   %6.0fms\n", ms(pt.Durations))
	fmt.Printf("    prosody     %6.0fms\n", ms(pt.Prosody))
	fmt.Printf("    text enc    %6.0fms\n", ms(pt.TextEncoder))
	fmt.Printf("    expand      %6.0fms\n", ms(pt.Expand))
	fmt.Printf("    phonemes    %6.0fms\n", ms(best.prosody))
	fmt.Printf("    decoder     %6.0fms\n", ms(best.decoder))
	fmt.Printf("    generator   %6.0fms\n", ms(best.stage))
	fmt.Printf("    excitation  %6.0fms\n", ms(best.excitation))
	fmt.Printf("    tail        %6.0fms\n", ms(best.tail))
	fmt.Printf("    vocoder     %6.0fms\n", ms(best.vocoder))
	fmt.Printf("    total       %6.0fms   %.2fx real time (load %.0fms)\n",
		ms(best.total), seconds/best.total.Seconds(), ms(load))

	if *out != "" {
		clip := &audio.Clip{Rate: model.Config.SamplingRate, Samples: wav}
		if err := clip.WriteWAV(*out); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("\nwrote %s\n", *out)
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// openDevice picks this machine's iGPU, or the first device that has a
// compute queue.
func openDevice() (*vk.Device, func(), error) {
	inst, err := vk.NewInstance("kokoro-tts")
	if err != nil {
		return nil, nil, err
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		return nil, nil, fmt.Errorf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == 0x1586 {
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		return nil, nil, err
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		return nil, nil, err
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		inst.Destroy()
		return nil, nil, err
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }, nil
}
