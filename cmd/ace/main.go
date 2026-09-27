// Command ace generates a song from a caption and lyrics with ACE-Step 1.5
// XL turbo, upstream's defaults: the 5 Hz LM plans it first ("thinking",
// MUSIC.md A8: a CoT of the metadata the request leaves out, then 5 audio
// codes a second), then the DiT (8 steps, shift 3, DCW) and the VAE. Out
// comes 48 kHz stereo peak-normalised to -1 dBFS, written through ffmpeg as
// wav, flac, mp3 or opus by the output's extension.
//
//	go run ./cmd/ace -caption "upbeat synth-pop, female vocals" -lyrics-file song.txt -duration 90 -out song.mp3
//	go run ./cmd/ace -caption "calm piano" -lyrics "[Instrumental]" -duration 30 -bpm 70 -key "C major"
//	go run ./cmd/ace -think=false -caption "..." -duration 30    # the DiT-only path (A6)
//	go run ./cmd/ace -sample "a melancholy synthwave song about a neon city"   # sample mode (A12)
//
// Lyrics use upstream's section tags ([Verse], [Chorus], [Instrumental]...).
// Without -duration the LM chooses it (10-600 s); with -think=false it is
// upstream's 120 s fallback. With -sample the LM first writes the caption,
// metas and lyrics from the description (upstream's sample mode); -caption
// and -lyrics are then not given, and -duration, -bpm, -key and -timesig are
// held in the song as it is written.
package main

import (
	"flag"
	"log"
	"os"
	"strings"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/pipeline"
	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/backend"
	"strix-halo-vulkan/vk"
)

func main() {
	models := flag.String("models", "models", "checkpoint root")
	caption := flag.String("caption", "", "the caption: genre, instruments, mood, vocals")
	lyrics := flag.String("lyrics", "", "the lyrics")
	lyricsFile := flag.String("lyrics-file", "", "read the lyrics from this file instead")
	duration := flag.Float64("duration", 0, "seconds, 10–600 (0: upstream's 120 s fallback)")
	bpm := flag.Int("bpm", 0, "tempo (0: unspecified)")
	key := flag.String("key", "", `key and scale, e.g. "C major" (empty: unspecified)`)
	timesig := flag.String("timesig", "", `time signature, e.g. "4" (empty: unspecified)`)
	lang := flag.String("lang", "", "vocal language code, e.g. en (empty: unknown)")
	seed := flag.Uint64("seed", 42, "noise seed")
	think := flag.Bool("think", true, "plan with the 5 Hz LM first (upstream's default)")
	lmSeed := flag.Uint64("lm-seed", 42, "the LM's sampling seed")
	sample := flag.String("sample", "", "sample mode: the LM writes the song from this description")
	sampleMode := flag.Bool("sample-mode", false, "sample mode, and no phase-1 CoT for a meta the song lacks (upstream's sample_mode); alone, the LM picks the song")
	out := flag.String("out", "ace.mp3", "output file: .wav, .flac, .mp3 or .opus")
	flag.Parse()

	text := *lyrics
	if *lyricsFile != "" {
		b, err := os.ReadFile(*lyricsFile)
		if err != nil {
			log.Fatal(err)
		}
		text = strings.TrimSpace(string(b))
	}
	var smp *pipeline.Sample
	if *sample != "" || *sampleMode {
		if *caption != "" || text != "" {
			log.Fatal("-sample writes the caption and lyrics: give neither")
		}
		smp = &pipeline.Sample{Query: *sample, SkipCoT: *sampleMode}
	} else if *caption == "" {
		log.Fatal("-caption is required (or -sample)")
	}
	if *duration != 0 && (*duration < plan.MinSeconds || *duration > plan.MaxSeconds) {
		log.Fatalf("-duration %.1f is outside %d–%d s", *duration, plan.MinSeconds, plan.MaxSeconds)
	}
	req := &plan.Request{Caption: *caption, Lyrics: text, BPM: *bpm, KeyScale: *key,
		TimeSignature: *timesig, Duration: *duration, Language: *lang}

	dev, err := backend.OpenDevice("ace")
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	err = dev.Do(func(d *vk.Device) error {
		ceiling := req.Seconds()
		if (*think || smp != nil) && req.Duration <= 0 {
			ceiling = plan.MaxSeconds // the LM chooses
		}
		p, err := pipeline.New(d, pipeline.DefaultDirs(*models), ceiling)
		if err != nil {
			return err
		}
		defer p.Destroy()
		if *think || smp != nil {
			if err := p.LoadLM(d, pipeline.LMDir(*models), lm.DefaultOptions()); err != nil {
				return err
			}
		}
		res, err := p.Generate(req, pipeline.Options{Seed: *seed, Think: *think, LMSeed: *lmSeed, Sample: smp})
		if err != nil {
			return err
		}
		if err := pipeline.Write(*out, res.Audio); err != nil {
			return err
		}
		t := res.Timings
		if res.Song != nil {
			log.Printf("LM sample pass (%v):\n%s", t.Sample, res.SampleText)
		}
		if *think {
			if res.CoT != "" {
				log.Printf("LM phase 1 (%v):\n%s", t.Think, res.CoT)
			}
			log.Printf("LM phase 2: %d codes (%v)", len(res.Codes), t.Codes)
			log.Printf("DiT metas: %q", res.DiT.Metas())
		}
		log.Printf("wrote %s: %.1f s of audio in %.2f s (text %.0f ms, encoders %.0f ms, DiT %.2f s, VAE %.2f s)",
			*out, res.Seconds, t.Total.Seconds(), t.Text.Seconds()*1e3, t.Encode.Seconds()*1e3, t.DiT.Seconds(), t.VAE.Seconds())
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}
