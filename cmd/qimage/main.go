// Command qimage generates an image with Qwen-Image-2.1 on the device, which
// is what cmd/zimage was for Z-Image-Turbo (IMAGE.md Q6).
//
//	go run ./cmd/qimage -prompt "a red fox in fresh snow" -out fox.png
//	go run ./cmd/qimage -steps 16 -reps 2            # the step/wall-clock trade
//	go run ./cmd/qimage -width 512 -height 768       # a portrait, inside the ceiling
//	go run ./cmd/qimage -transparent -out sticker.png
//
// It is the tool the step-count sweep and the eyeball are done with, so it
// reports the wall clock by stage rather than only the total: the text
// encoder, the DiT's prefill, the mean cached step, and the VAE. Those four
// are what IMAGE.md's per-stage numbers are made of.
//
// **The size is a request parameter, the ceiling is residency.** -width and
// -height set both here, because a one-shot command has no reason to build
// arenas larger than the image it was asked for. The ceiling is an *area*
// rather than a rectangle -- every arena is sized by the pixel count alone,
// measured -- and it stops at 1,403,584 pixels (1184x1184 square, 1536x864 at
// 16:9): the VAE decoder's activation arena is one storage buffer and this
// device caps one at 4 GiB - 4.
package main

import (
	"context"
	"flag"
	"fmt"
	"image/png"
	"log"
	"os"
	"os/signal"
	"time"

	"strix-halo-vulkan/qimage/pipeline"
	"strix-halo-vulkan/vk"
	zvae "strix-halo-vulkan/zimage/vae"
)

const strixHaloDeviceID = 0x1586

func main() {
	log.SetFlags(0)
	// Ctrl-C cancels the run rather than only the process: the sampler checks
	// between steps and the VAE between submit batches, so an interrupt
	// unwinds through the same path a client that hung up takes and the
	// device is left clean.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	model := flag.String("model", "models/Qwen-Image-2.1", "checkpoint root")
	prompt := flag.String("prompt", "a red fox sitting in fresh snow at dawn, photograph, shallow depth of field", "the prompt")
	out := flag.String("out", "qimage.png", "PNG to write; with -reps > 1 the repetition index is appended")
	width := flag.Int("width", 1024, "image width in pixels, a multiple of 32")
	height := flag.Int("height", 0, "image height; 0 is square")
	steps := flag.Int("steps", pipeline.DefaultSteps, "denoising steps; the checkpoint's default is 40")
	seed := flag.Int64("seed", 1, "seed for the initial latent")
	maxPrompt := flag.Int("maxprompt", 512, "longest prompt the text encoder is built for")
	reps := flag.Int("reps", 1, "generate this many times, reporting each; the first also pays the arenas' first touch")
	transparent := flag.Bool("transparent", false,
		"ask for an RGBA image with a transparent background, using the model card's recommended prompt phrasing")
	progress := flag.Bool("progress", false, "log every denoising step as it lands")
	flag.Parse()

	if *height == 0 {
		*height = *width
	}
	text := *prompt
	if *transparent {
		text = "This is an RGBA image with transparency. " + text +
			" The image has alpha channel and the background is transparent."
	}

	inst, err := vk.NewInstance("qimage")
	must(err)
	defer inst.Destroy()
	devs, err := inst.PhysicalDevices()
	must(err)
	phys := &devs[0]
	for i := range devs {
		if devs[i].DeviceID == strixHaloDeviceID {
			phys = &devs[i]
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
	defer dev.Destroy()

	fmt.Printf("device: %s\n", phys.Name)
	fmt.Printf("%dx%d, %d steps, seed %d\n", *width, *height, *steps, *seed)

	t0 := time.Now()
	p, err := pipeline.New(dev, pipeline.Options{
		Model: *model, Width: *width, Height: *height, Steps: *steps, MaxPrompt: *maxPrompt,
	})
	must(err)
	defer p.Destroy()
	enc, dt, vaeW, act := p.Residency()
	fmt.Printf("staged in %v: %.1f GB (%.1f encoder + %.1f transformer + %.1f vae + %.1f activations)\n",
		time.Since(t0).Round(time.Millisecond), float64(enc+dt+vaeW+act)/1e9,
		float64(enc)/1e9, float64(dt)/1e9, float64(vaeW)/1e9, float64(act)/1e9)

	for rep := 0; rep < *reps; rep++ {
		var onStep func(pipeline.Step)
		if *progress {
			onStep = func(st pipeline.Step) {
				fmt.Printf("  step %2d/%d  %7.1f ms  sigma %.4f -> %.4f\n",
					st.Index, st.Steps, float64(st.Wall.Microseconds())/1000, st.Sigma, st.NextSigma)
			}
		}
		img, tm, err := p.Run(ctx, pipeline.Request{
			Prompt: text, Width: *width, Height: *height, Steps: *steps,
			Seed: *seed + int64(rep), Progress: onStep,
		})
		must(err)
		report(tm)
		path := *out
		if *reps > 1 {
			path = fmt.Sprintf("%s.%d.png", *out, rep)
		}
		must(writePNG(path, img, !*transparent))
		fmt.Printf("wrote %s\n", path)
	}
}

// report is the per-stage wall clock, which is the point of this command.
// The prefill is separated from the cached steps because they are different
// work: step 0 runs the whole joint sequence block-causally and stores every
// block's prefix K/V, and the rest recompute only the target's rows.
func report(tm *pipeline.Timings) {
	var cached time.Duration
	for _, d := range tm.Steps[1:] {
		cached += d
	}
	mean := time.Duration(0)
	if len(tm.Steps) > 1 {
		mean = cached / time.Duration(len(tm.Steps)-1)
	}
	fmt.Printf("%dx%d in %v: encode %v (%d tokens), prefill %v, %d cached steps mean %v, decode %v\n",
		tm.Width, tm.Height, tm.Total.Round(time.Millisecond),
		tm.Encode.Round(time.Millisecond), tm.Tokens,
		tm.Prefill.Round(time.Millisecond), len(tm.Steps)-1, mean.Round(time.Millisecond),
		tm.Decode.Round(time.Millisecond))
}

func writePNG(path string, img *zvae.Tensor, opaque bool) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, pipeline.ToImage(img, opaque))
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
