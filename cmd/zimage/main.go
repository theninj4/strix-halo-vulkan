// Command zimage is the pipeline: a prompt in, a PNG out.
//
// It is stage 6 of PIPELINE.md, and what it adds over cmd/textenc,
// cmd/ditstack and cmd/vaedecode is only that it runs them in order -- the
// tokenizer and the text encoder for the caption, the transformer's three
// phases over its 34 blocks for eight denoising steps, and the VAE for the
// picture. Everything it prints is wall clock, because a pipeline's own
// question is where an image's seconds go and not what a dispatch costs.
//
//	go run ./cmd/zimage -prompt "a red fox in the snow" -out fox.png
//
// -latents takes an initial latent from a file instead of the seeded RNG,
// which is what makes an end-to-end comparison against diffusers possible:
// the two generators do not agree and nothing else in the pipeline is random.
//
// -init turns it into an *edit* (IMAGE.md I7): the PNG it names is encoded,
// noised to the sigma -strength picks out of the schedule, and only the tail
// of the schedule runs. The picture has to be exactly the size being
// rendered, because fitting one to the other is a policy question this
// command has no business deciding; -width and -height default to the input's
// own size when -init is given, so the ordinary case needs neither.
//
//	go run ./cmd/zimage -init fox.png -strength 0.6 -prompt "the same fox, at night"
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"
	"strings"
	"time"

	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/pipeline"
	"strix-halo-vulkan/zimage/vae"
)

const strixHaloDeviceID = 0x1586

func main() {
	log.SetFlags(0)
	model := flag.String("model", "models/Z-Image-Turbo", "checkpoint root")
	prompt := flag.String("prompt", "a red fox sitting in fresh snow, photograph", "the prompt")
	out := flag.String("out", "zimage.png", "PNG to write")
	width := flag.Int("width", 1024, "image width in pixels, a multiple of 16")
	height := flag.Int("height", 0, "image height; 0 is square")
	steps := flag.Int("steps", 8, "denoising steps; Z-Image-Turbo's NFE is 8")
	cpuHead := flag.Bool("cpuhead", false, "run the patch embedder and the final layer on the host, which is stage 9's slow path")
	unfusedLayout := flag.Bool("unfused-layout", false, "keep the block's two pure-layout dispatches, which is stage 10's slow path")
	seed := flag.Int64("seed", 1, "seed for the initial latent")
	maxPrompt := flag.Int("maxprompt", 512, "longest prompt the text encoder is built for")
	latentFile := flag.String("latents", "", "read the initial latent from this file instead of the RNG")
	dumpLatent := flag.String("dumplatent", "", "write the denoised latent here, before the VAE")
	reps := flag.Int("reps", 1, "generate this many times, reporting each")
	initPNG := flag.String("init", "", "a PNG to edit rather than generate from noise; see -strength")
	strength := flag.Float64("strength", 0, "with -init: how much of the schedule to run, (0, 1]; 0 takes the pipeline's default")
	preview := flag.String("preview", "",
		"a madebyollin/taef1 checkpoint; writes <out>.preview.<step>.png after every step and times the decode")
	flag.Parse()

	// An edit's geometry comes from the picture unless it was overridden,
	// which is the only sensible default: the encoder has to be handed the
	// image at the size being rendered, so anything else would be a resize
	// this command refuses to choose.
	var initImg *vae.Tensor
	if *initPNG != "" {
		var err error
		initImg, err = readPNG(*initPNG)
		must(err)
		if !flagSet("width") {
			*width = initImg.W
		}
		if !flagSet("height") {
			*height = initImg.H
		}
	}
	if *height == 0 {
		*height = *width
	}

	inst, err := vk.NewInstance("zimage")
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
	if initImg != nil {
		st := *strength
		if st == 0 {
			st = pipeline.DefaultStrength
		}
		fmt.Printf("editing %s (%dx%d) at strength %g: steps %d..%d of %d\n",
			*initPNG, initImg.W, initImg.H, st, pipeline.StartStep(*steps, st), *steps-1, *steps)
	}

	t0 := time.Now()
	p, err := pipeline.New(dev, pipeline.Options{
		Model: *model, Width: *width, Height: *height, Steps: *steps, MaxPrompt: *maxPrompt,
		CPUHead:       *cpuHead,
		UnfusedLayout: *unfusedLayout,
		Preview:       *preview,
		Encoder:       initImg != nil,
	})
	must(err)
	defer p.Destroy()
	load := time.Since(t0)

	enc, tr, vaeW, act := p.Residency()
	c, lh, lw := p.Latent()
	fmt.Printf("loaded in %s: %.2f GB text encoder + %.2f GB transformer + %.2f GB VAE + %.2f GB activations, latent [%d %d %d]\n",
		ms(load), float64(enc)/1e9, float64(tr)/1e9, float64(vaeW)/1e9, float64(act)/1e9, c, lh, lw)
	fmt.Printf("sigmas: %s\n\n", sigmas(p))

	for rep := 0; rep < *reps; rep++ {
		latents := p.Noise(*seed + int64(rep))
		if *latentFile != "" {
			latents, err = readLatent(*latentFile, len(latents))
			must(err)
		}
		fmt.Printf("prompt: %q\n", *prompt)
		var previewTotal time.Duration
		img, tm, err := p.Run(pipeline.Request{
			Prompt: *prompt, Latents: latents, Init: initImg, Strength: *strength,
			Progress: func(d pipeline.Step) {
				fmt.Printf("  step %d/%d  sigma %.4f -> %.4f  %8s  blocks %8s  head %7s",
					d.Index+1, *steps, d.Sigma, d.NextSig, ms(d.Blocks+d.Head), ms(d.Blocks), ms(d.Head))
				if d.Preview == nil {
					fmt.Println()
					return
				}
				// Timed and written here rather than inside the pipeline, because
				// what this flag is for is the *cost*: a preview is only worth
				// having if it is small beside the step it interrupts, and the
				// two numbers belong on the same line.
				t0 := time.Now()
				frame, err := d.Preview()
				must(err)
				dt := time.Since(t0)
				previewTotal += dt
				name := fmt.Sprintf("%s.preview.%d.png", strings.TrimSuffix(*out, ".png"), d.Index)
				must(writePNGOpt(name, frame, true))
				fmt.Printf("  preview %7s (%.1f%% of the step) -> %s\n",
					ms(dt), 100*dt.Seconds()/(d.Blocks+d.Head).Seconds(), name)
			}})
		must(err)
		if previewTotal > 0 {
			fmt.Printf("  previews: %d x taef1, %s total, %.1f%% of the image\n",
				len(tm.Steps), ms(previewTotal), 100*previewTotal.Seconds()/tm.Total.Seconds())
		}

		if *dumpLatent != "" {
			must(writeLatent(*dumpLatent, latents))
		}
		name := *out
		if *reps > 1 {
			name = fmt.Sprintf("%s.%d.png", *out, rep)
		}
		must(writePNG(name, img))
		report(tm, name, previewTotal)
	}
}

// report prints where the image's seconds went.
func report(tm *pipeline.Timings, name string, previews time.Duration) {
	var steps time.Duration
	for _, d := range tm.Steps {
		steps += d
	}
	fmt.Printf("\n%-22s %10s  %s\n", "STAGE", "WALL", "SHARE")
	row := func(label string, d time.Duration, note string) {
		fmt.Printf("%-22s %10s  %5.1f%%  %s\n", label, ms(d), 100*d.Seconds()/tm.Total.Seconds(), note)
	}
	row("text encoder", tm.Encode, fmt.Sprintf("%d tokens", tm.Tokens))
	if tm.VAEEncode > 0 {
		row("vae encode", tm.VAEEncode, fmt.Sprintf("the init image, once; the loop then starts at sigma %.4f", tm.Sigma))
	}
	row("caption refiners", tm.Caption, fmt.Sprintf("%d rows, once per image", tm.CapTotal))
	row("denoising", steps, fmt.Sprintf("steps %d..%d over %d tokens, %s each",
		tm.First, tm.First+len(tm.Steps)-1, tm.Unified, ms(steps/time.Duration(len(tm.Steps)))))
	if previews > 0 {
		// Inside TOTAL, because the callback runs inside the denoising loop
		// and the pipeline's clock is around the whole of it. It is its own
		// row rather than folded into the steps for the same reason it is
		// worth measuring at all.
		row("taef1 previews", previews, fmt.Sprintf("%d frames", len(tm.Steps)))
	}
	row("vae decode", tm.Decode, "")
	fmt.Printf("%-22s %10s\n", "TOTAL", ms(tm.Total))
	fmt.Printf("wrote %s\n\n", name)
}

func sigmas(p *pipeline.Pipeline) string {
	s := ""
	for _, v := range p.Scheduler().Sigmas {
		s += fmt.Sprintf("%.4f ", v)
	}
	return s
}

// writePNG maps the decoder's [-1, 1] output to 8-bit RGB.
//
// fast is png.BestSpeed, and it is set for previews only. At 1024x1024 the
// default encoder is 344 ms against BestSpeed's 60 for a file 15% larger, so
// writing eight previews the careful way would cost more than twice what
// generating them does and would put this flag's own timing line in the shade.
func writePNG(path string, img *vae.Tensor) error { return writePNGOpt(path, img, false) }

func writePNGOpt(path string, img *vae.Tensor, fast bool) error {
	rgb := image.NewRGBA(image.Rect(0, 0, img.W, img.H))
	for y := 0; y < img.H; y++ {
		for x := 0; x < img.W; x++ {
			var px [3]uint8
			for ch := 0; ch < 3; ch++ {
				v := (float64(img.Data[(ch*img.H+y)*img.W+x]) + 1) * 127.5
				px[ch] = uint8(math.Round(math.Min(255, math.Max(0, v))))
			}
			rgb.Set(x, y, color.RGBA{px[0], px[1], px[2], 255})
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := png.Encoder{}
	if fast {
		enc.CompressionLevel = png.BestSpeed
	}
	return enc.Encode(f, rgb)
}

func readLatent(path string, want int) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) != want*4 {
		return nil, fmt.Errorf("%s is %d bytes, want %d float32", path, len(raw), want)
	}
	out := make([]float32, want)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}

func writeLatent(path string, v []float32) error {
	raw := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(f))
	}
	return os.WriteFile(path, raw, 0o644)
}

func ms(d time.Duration) string {
	if d >= time.Second {
		return fmt.Sprintf("%.2f s", d.Seconds())
	}
	return fmt.Sprintf("%.1f ms", float64(d.Microseconds())/1000)
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// flagSet reports whether a flag was given on the command line, which is how
// -width tells "the caller asked for 1024" from "1024 is the default".
func flagSet(name string) bool {
	found := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			found = true
		}
	})
	return found
}

// readPNG is writePNG's inverse: an 8-bit image to the [3, H, W] tensor in
// [-1, 1] that the VAE's encoder takes.
func readPNG(path string) (*vae.Tensor, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	src, err := png.Decode(fh)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	b := src.Bounds()
	t := vae.NewTensor(1, 3, b.Dy(), b.Dx())
	plane := t.H * t.W
	for y := 0; y < t.H; y++ {
		for x := 0; x < t.W; x++ {
			r, g, bl, _ := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
			for ch, v := range [3]uint32{r, g, bl} {
				// RGBA() is 16-bit; 65535 maps to +1 and 0 to -1, which is
				// the inverse of what writePNG does going out.
				t.Data[ch*plane+y*t.W+x] = float32(v)/32767.5 - 1
			}
		}
	}
	return t, nil
}
