// Command flux3 generates an image with FLUX 3 Image Klein on the device
// (research/flux3-vertical.md X5): prompt → Qwen3-VL-4B encoder → 16
// flow-map steps of the DiT → the VAE decoder → PNG, reporting each stage's
// wall clock.
package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"image/png"
	"log"
	"os"
	"time"

	"strix-halo-vulkan/flux3/pipeline"
	"strix-halo-vulkan/flux3/vae"
	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func main() {
	model := flag.String("model", "models/flux-3-image-klein", "checkpoint root (diffusers layout)")
	prompt := flag.String("prompt", "a red fox sitting in fresh snow at dawn, photograph, shallow depth of field", "the prompt")
	out := flag.String("out", "flux3.png", "PNG to write; with -reps > 1 the repetition index is appended")
	width := flag.Int("width", 1024, "image width in pixels, a multiple of 16")
	height := flag.Int("height", 0, "image height; 0 is square")
	steps := flag.Int("steps", pipeline.DefaultSteps, "flow-map steps; the checkpoint is distilled to 16")
	seed := flag.Int64("seed", 1, "seed for the initial latent")
	maxPrompt := flag.Int("maxprompt", 512, "longest prompt, in tokens, the encoder is staged for")
	reps := flag.Int("reps", 1, "generate this many times (seed, seed+1, ...), reporting each")
	previews := flag.Int("previews", 0, "also write the taef2 preview of every Nth step as <out>.stepK.png; 0 writes none")
	flag.Parse()
	if *height == 0 {
		*height = *width
	}

	inst, err := vk.NewInstance("flux3")
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
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	must(err)
	defer dev.Destroy()

	fmt.Printf("device: %s\n%dx%d, %d steps, seed %d\n", phys.Name, *width, *height, *steps, *seed)
	t0 := time.Now()
	p, err := pipeline.NewGPU(dev, pipeline.Config{Model: *model, MaxWidth: *width, MaxHeight: *height, MaxPrompt: *maxPrompt})
	must(err)
	defer p.Destroy()
	enc, tr, vaes, act := p.Residency()
	fmt.Printf("staged in %v: %.1f GB (%.1f encoder + %.1f transformer + %.2f decoders + %.1f activations)\n",
		time.Since(t0).Round(time.Millisecond), float64(enc+tr+vaes+act)/1e9,
		float64(enc)/1e9, float64(tr)/1e9, float64(vaes)/1e9, float64(act)/1e9)

	for rep := 0; rep < *reps; rep++ {
		start := time.Now()
		path := *out
		if *reps > 1 {
			path = fmt.Sprintf("%s.%d.png", *out, rep)
		}
		var progress func(pipeline.Step)
		if *previews > 0 {
			progress = func(st pipeline.Step) {
				if st.Index%*previews != 0 {
					return
				}
				pv, err := st.Preview()
				must(err)
				must(writePNG(fmt.Sprintf("%s.step%02d.png", path, st.Index), pv))
			}
		}
		img, tm, err := p.Generate(context.Background(), pipeline.Request{
			Prompt: *prompt, Width: *width, Height: *height, Steps: *steps, Seed: *seed + int64(rep),
			Progress: progress,
		})
		must(err)
		total := time.Since(start)
		fmt.Printf("encode %v  denoise %v (%.0f ms/step, %v on the device)  previews %v  decode %v  total %v\n",
			tm.Encode.Round(time.Millisecond), tm.Denoise.Round(time.Millisecond),
			tm.Denoise.Seconds()*1e3/float64(tm.Steps), tm.Device.Round(time.Millisecond),
			tm.Previews.Round(time.Millisecond), tm.Decode.Round(time.Millisecond), total.Round(time.Millisecond))
		must(writePNG(path, img))
		fmt.Printf("wrote %s\n", path)
	}
}

func writePNG(path string, img *vae.Tensor) error {
	rgb := vae.RGB8(img)
	pic := image.NewNRGBA(image.Rect(0, 0, img.W, img.H))
	for i := 0; i < img.W*img.H; i++ {
		copy(pic.Pix[i*4:], rgb[i*3:i*3+3])
		pic.Pix[i*4+3] = 255
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, pic); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
