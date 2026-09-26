// Command h3 generates a video with sound from a prompt with MiniMax-H3
// (VIDEO.md M8, M10): t2va, or fl2va from a first and/or last keyframe,
// end to end, muxed to mp4.
//
//	go run ./cmd/h3 -prompt "…" -out clip.mp4              # 864x480, 5 s, 20 steps (~13 min)
//	go run ./cmd/h3 -prompt-file p.txt -short 768 -steps 50  # the trained canvas (~2 h)
//	go run ./cmd/h3 -prompt "…" -first start.png -last end.jpg  # fl2va; the canvas follows start.png
//
// The prompt is used verbatim: MiniMax's own requests are long structured
// Context-IR prompts (models/MiniMax-H3/docs/VIDEO_PROMPT_WRITING_GUIDE_*),
// and a one-line prompt gets a noticeably plainer clip.
package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"os/signal"
	"strings"
	"time"

	"strix-halo-vulkan/backend"
	"strix-halo-vulkan/h3/pipeline"
	"strix-halo-vulkan/vk"
)

func main() {
	model := flag.String("model", "models/MiniMax-H3", "checkpoint root")
	prompt := flag.String("prompt", "", "the prompt")
	promptFile := flag.String("prompt-file", "", "read the prompt from this file instead")
	out := flag.String("out", "h3.mp4", "mp4 to write")
	aspect := flag.String("aspect", "", "aspect ratio, W:H (default: the first keyframe's, else 16:9)")
	first := flag.String("first", "", "fl2va: the picture the video starts from")
	last := flag.String("last", "", "fl2va: the picture the video ends on")
	short := flag.Int("short", pipeline.DefaultShortEdge, "short edge in pixels (256–768; 768 is the trained canvas)")
	seconds := flag.Float64("seconds", pipeline.DefaultSeconds, "duration, 5–15 s (snapped up to 17n+5 frames)")
	steps := flag.Int("steps", pipeline.DefaultSteps, "sampling steps N (N−1 forwards); the release's default is 50")
	seed := flag.Uint64("seed", 0, "noise seed")
	flag.Parse()

	text := *prompt
	if *promptFile != "" {
		b, err := os.ReadFile(*promptFile)
		if err != nil {
			log.Fatal(err)
		}
		text = strings.TrimSpace(string(b))
	}
	var aw, ah float64
	if *aspect != "" {
		if _, err := fmt.Sscanf(*aspect, "%g:%g", &aw, &ah); err != nil {
			log.Fatalf("-aspect %q: %v", *aspect, err)
		}
	}
	picture := func(path string) image.Image {
		if path == "" {
			return nil
		}
		f, err := os.Open(path)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		img, _, err := image.Decode(f)
		if err != nil {
			log.Fatalf("%s: %v", path, err)
		}
		return img
	}
	dev, err := backend.OpenDevice("h3")
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	req := &pipeline.Request{Prompt: text, AspectW: aw, AspectH: ah, ShortEdge: *short,
		Seconds: *seconds, Steps: *steps, Seed: *seed, First: picture(*first), Last: picture(*last)}
	err = dev.Do(func(d *vk.Device) error {
		p, err := pipeline.New(d, *model, pipeline.Options{})
		if err != nil {
			return err
		}
		defer p.Close()
		r, err := pipeline.Resolve(p.Tokenizer(), req)
		if err != nil {
			return err
		}
		log.Printf("%dx%d, %d frames, %d steps, %d prompt tokens, %d rows", r.Width, r.Height, r.Frames, r.Steps, len(r.Tokens), len(r.Layout.Pos))
		start := time.Now()
		res, err := p.Generate(ctx, req, func(s pipeline.Stage, done, total int) {
			log.Printf("%7.1f s  %s %d/%d", time.Since(start).Seconds(), s, done, total)
		})
		if err != nil {
			return err
		}
		if err := pipeline.WriteMP4(*out, res); err != nil {
			return err
		}
		log.Printf("wrote %s in %v (%v)", *out, time.Since(start).Round(time.Second), res.Timings)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}
