// Command mm3 generates a song from a caption and lyrics with
// MiniMax-Music3 on the device (research/music-minimax.md N7): the global
// and local LMs sample 25 frames of codes a second, the DiT flow-matches the
// latents window by window, and the vocoder writes 44.1 kHz stereo as a
// 16-bit wav.
//
//	go run ./cmd/mm3 -caption-file caption.txt -lyrics-file song.txt -seconds 180 -out song.wav
//	go run ./cmd/mm3 -set out/mm3-n0c/set.json -captions out/mm3-n0c/captions.json -id h3 -out h3.wav
//
// The model chooses the song's length; -seconds caps it (at most 360). The
// caption works best long and structured, in the skill's style
// (MiniMax-Music3's music-caption-rewriter: Global Metadata, Basic
// Attributes, ... Arrangement).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"strix-halo-vulkan/backend"
	"strix-halo-vulkan/mm3"
	"strix-halo-vulkan/vk"
)

func main() {
	models := flag.String("models", "models/MiniMax-Music3", "the checkpoint")
	caption := flag.String("caption", "", "the caption")
	captionFile := flag.String("caption-file", "", "read the caption from this file")
	lyrics := flag.String("lyrics", "", "the lyrics, with [verse]/[chorus]... tags on their own lines")
	lyricsFile := flag.String("lyrics-file", "", "read the lyrics from this file")
	setFile := flag.String("set", "", "a set.json (cmd/acecal's shape) to take -id's caption, lyrics and duration from")
	captions := flag.String("captions", "", "a json of id -> caption replacing the set's")
	id := flag.String("id", "", "the song in -set")
	seconds := flag.Float64("seconds", 0, "the cap on the song's length (0: the set's duration, else 120)")
	seed := flag.Uint64("seed", 11, "the RNG seed")
	bank := flag.String("bank", "q8", "the LMs' weight bank: q8 or fp16")
	out := flag.String("out", "song.wav", "the wav to write")
	flag.Parse()

	read := func(s *string, path string) {
		if path == "" {
			return
		}
		b, err := os.ReadFile(path)
		if err != nil {
			log.Fatal(err)
		}
		*s = string(b)
	}
	read(caption, *captionFile)
	read(lyrics, *lyricsFile)
	if *setFile != "" {
		var set struct {
			Songs []struct {
				ID, Caption, Lyrics string
				Duration            float64
			}
		}
		b, err := os.ReadFile(*setFile)
		if err != nil {
			log.Fatal(err)
		}
		if err := json.Unmarshal(b, &set); err != nil {
			log.Fatal(err)
		}
		found := false
		for _, s := range set.Songs {
			if s.ID == *id {
				*caption, *lyrics, found = s.Caption, s.Lyrics, true
				if *seconds == 0 {
					*seconds = s.Duration
				}
			}
		}
		if !found {
			log.Fatalf("mm3: no song %q in %s", *id, *setFile)
		}
		if *captions != "" {
			var cs map[string]any
			b, err := os.ReadFile(*captions)
			if err != nil {
				log.Fatal(err)
			}
			if err := json.Unmarshal(b, &cs); err != nil {
				log.Fatal(err)
			}
			if c, ok := cs[*id].(string); ok {
				*caption = c
			}
		}
	}
	if *seconds == 0 {
		*seconds = 120
	}
	o := mm3.DefaultOptions()
	if *bank == "fp16" {
		o.Bank = mm3.BankFP16
	}

	dev, err := backend.OpenDevice("mm3")
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	err = dev.Do(func(d *vk.Device) error {
		t0 := time.Now()
		p, err := mm3.Load(d, *models, o)
		if err != nil {
			return err
		}
		defer p.Destroy()
		log.Printf("staged in %v", time.Since(t0))
		t0 = time.Now()
		last := time.Now()
		wav, st, err := p.Generate(mm3.Request{Caption: *caption, Lyrics: *lyrics, Seconds: *seconds, Seed: *seed},
			func(stage string, done, total int) {
				if time.Since(last) > 10*time.Second || done == total {
					log.Printf("%s %d/%d (%v)", stage, done, total, time.Since(t0).Round(time.Second))
					last = time.Now()
				}
			})
		if err != nil {
			return err
		}
		secs := float64(len(wav[0])) / mm3.SampleRate
		wall := time.Since(t0)
		log.Printf("%.1f s of audio (%d frames, ended by the LM: %v; %d prompt tokens, %d windows) in %v: frames %v, DiT %v, vocoder %v; %.2f s a second",
			secs, st.Frames, st.Ended, st.PromptTokens, st.Windows, wall.Round(time.Millisecond),
			st.Frames0.Round(time.Millisecond), st.DiT.Round(time.Millisecond), st.Vocoder.Round(time.Millisecond), wall.Seconds()/secs)
		if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
			return err
		}
		if err := mm3.WriteWAV(*out, wav); err != nil {
			return err
		}
		meta, _ := json.MarshalIndent(map[string]any{"caption": *caption, "lyrics": *lyrics, "seconds": *seconds,
			"seed": *seed, "bank": *bank, "audio_seconds": secs, "frames": st.Frames, "ended": st.Ended,
			"wall_s": wall.Seconds(), "frames_s": st.Frames0.Seconds(), "dit_s": st.DiT.Seconds(), "vocoder_s": st.Vocoder.Seconds()}, "", " ")
		return os.WriteFile(*out+".json", meta, 0o644)
	})
	if err != nil {
		log.Fatal(fmt.Errorf("mm3: %w", err))
	}
}
