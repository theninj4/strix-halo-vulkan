// Command ace generates a song from a caption and lyrics with ACE-Step 1.5
// XL turbo (MUSIC.md A6): the DiT-only text2music path, upstream's defaults
// (8 steps, shift 3, DCW), 48 kHz stereo peak-normalised to -1 dBFS, written
// through ffmpeg as wav, flac, mp3 or opus by the output's extension.
//
//	go run ./cmd/ace -caption "upbeat synth-pop, female vocals" -lyrics-file song.txt -duration 90 -out song.mp3
//	go run ./cmd/ace -caption "calm piano" -lyrics "[Instrumental]" -duration 30 -bpm 70 -key "C major"
//
// Lyrics use upstream's section tags ([Verse], [Chorus], [Instrumental]...).
// A request without -duration is 120 s of audio (upstream's fallback with no
// LM); the 5 Hz LM that would plan it is A7.
package main

import (
	"flag"
	"log"
	"os"
	"strings"

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
	if *caption == "" {
		log.Fatal("-caption is required")
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
		p, err := pipeline.New(d, pipeline.DefaultDirs(*models), req.Seconds())
		if err != nil {
			return err
		}
		defer p.Destroy()
		res, err := p.Generate(req, pipeline.Options{Seed: *seed})
		if err != nil {
			return err
		}
		if err := pipeline.Write(*out, res.Audio); err != nil {
			return err
		}
		t := res.Timings
		log.Printf("wrote %s: %.1f s of audio in %.2f s (text %.0f ms, encoders %.0f ms, DiT %.2f s, VAE %.2f s)",
			*out, res.Seconds, t.Total.Seconds(), t.Text.Seconds()*1e3, t.Encode.Seconds()*1e3, t.DiT.Seconds(), t.VAE.Seconds())
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}
