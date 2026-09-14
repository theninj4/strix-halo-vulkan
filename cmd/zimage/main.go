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
	seed := flag.Int64("seed", 1, "seed for the initial latent")
	maxPrompt := flag.Int("maxprompt", 512, "longest prompt the text encoder is built for")
	latentFile := flag.String("latents", "", "read the initial latent from this file instead of the RNG")
	dumpLatent := flag.String("dumplatent", "", "write the denoised latent here, before the VAE")
	reps := flag.Int("reps", 1, "generate this many times, reporting each")
	flag.Parse()

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

	t0 := time.Now()
	p, err := pipeline.New(dev, pipeline.Options{
		Model: *model, Width: *width, Height: *height, Steps: *steps, MaxPrompt: *maxPrompt,
		CPUHead: *cpuHead,
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
		img, tm, err := p.GenerateFrom(*prompt, latents, func(d pipeline.Step) {
			fmt.Printf("  step %d/%d  sigma %.4f -> %.4f  %8s  blocks %8s  head %7s\n",
				d.Index+1, *steps, d.Sigma, d.NextSig, ms(d.Blocks+d.Head), ms(d.Blocks), ms(d.Head))
		})
		must(err)

		if *dumpLatent != "" {
			must(writeLatent(*dumpLatent, latents))
		}
		name := *out
		if *reps > 1 {
			name = fmt.Sprintf("%s.%d.png", *out, rep)
		}
		must(writePNG(name, img))
		report(tm, name)
	}
}

// report prints where the image's seconds went.
func report(tm *pipeline.Timings, name string) {
	var steps time.Duration
	for _, d := range tm.Steps {
		steps += d
	}
	fmt.Printf("\n%-22s %10s  %s\n", "STAGE", "WALL", "SHARE")
	row := func(label string, d time.Duration, note string) {
		fmt.Printf("%-22s %10s  %5.1f%%  %s\n", label, ms(d), 100*d.Seconds()/tm.Total.Seconds(), note)
	}
	row("text encoder", tm.Encode, fmt.Sprintf("%d tokens", tm.Tokens))
	row("caption refiners", tm.Caption, fmt.Sprintf("%d rows, once per image", tm.CapTotal))
	row("denoising", steps, fmt.Sprintf("%d steps over %d tokens, %s each", len(tm.Steps), tm.Unified, ms(steps/time.Duration(len(tm.Steps)))))
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
func writePNG(path string, img *vae.Tensor) error {
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
	return png.Encode(f, rgb)
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
