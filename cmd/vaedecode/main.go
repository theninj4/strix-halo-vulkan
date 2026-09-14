// Command vaedecode decodes a latent dumped by reference/encode_image.py
// with the Vulkan VAE decoder and writes a PNG, so the port can be checked
// against a picture and not only against a relative error.
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
	"path/filepath"
	"time"

	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/vae"
)

const strixHaloDeviceID = 0x1586

func main() {
	log.SetFlags(0)
	dir := flag.String("vae", "models/Z-Image-Turbo/vae", "VAE checkpoint")
	in := flag.String("in", "reference/out/image", "directory holding latent.bin and shape.txt")
	out := flag.String("out", "reference/out/image/vulkan.png", "PNG to write")
	flag.Parse()

	var c, h, w int
	shape, err := os.ReadFile(filepath.Join(*in, "shape.txt"))
	must(err)
	_, err = fmt.Sscanf(string(shape), "%d %d %d", &c, &h, &w)
	must(err)

	raw, err := os.ReadFile(filepath.Join(*in, "latent.bin"))
	must(err)
	latent := vae.NewTensor(1, c, h, w)
	if len(raw) != len(latent.Data)*4 {
		log.Fatalf("latent.bin is %d bytes, want %d for [1 %d %d %d]", len(raw), len(latent.Data)*4, c, h, w)
	}
	for i := range latent.Data {
		latent.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}

	inst, err := vk.NewInstance("vaedecode")
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
	// The mid block's matrix-core path (stage 7) needs all three; without
	// them NewGPUDecoder falls back to stage 2b's fp32 kernels.
	feat, err := phys.SupportedFeatures()
	must(err)
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16:             feat.Float16,
		CoopMatrix:          feat.CoopMatrix,
		SubgroupSizeControl: feat.SubgroupSizeControl,
	})
	must(err)
	defer dev.Destroy()

	cpu, err := vae.LoadDecoder(*dir, vae.FluxConfig())
	must(err)
	g, err := vae.NewGPUDecoder(dev, cpu, h, w)
	must(err)
	defer g.Destroy()

	t0 := time.Now()
	img, err := g.Apply(latent)
	must(err)
	elapsed := time.Since(t0)

	rgb := image.NewRGBA(image.Rect(0, 0, img.W, img.H))
	for y := 0; y < img.H; y++ {
		for x := 0; x < img.W; x++ {
			px := [3]uint8{}
			for ch := 0; ch < 3; ch++ {
				v := (float64(img.Data[(ch*img.H+y)*img.W+x]) + 1) * 127.5
				px[ch] = uint8(math.Round(math.Min(255, math.Max(0, v))))
			}
			rgb.Set(x, y, color.RGBA{px[0], px[1], px[2], 255})
		}
	}
	f, err := os.Create(*out)
	must(err)
	defer f.Close()
	must(png.Encode(f, rgb))

	fmt.Printf("decoded [1 %d %d %d] -> %dx%d in %s\n", c, h, w, img.W, img.H, elapsed.Round(time.Millisecond))

	// Compare against the diffusers decode if it is there.
	if ref, err := os.ReadFile(filepath.Join(*in, "decoded.bin")); err == nil && len(ref) == len(img.Data)*4 {
		var maxAbs, sumSq float64
		for i := range img.Data {
			r := float64(math.Float32frombits(binary.LittleEndian.Uint32(ref[i*4:])))
			sumSq += r * r
			if d := math.Abs(float64(img.Data[i]) - r); d > maxAbs {
				maxAbs = d
			}
		}
		rms := math.Sqrt(sumSq / float64(len(img.Data)))
		fmt.Printf("vs diffusers: max abs %.3g, rms %.4g, rel %.2g\n", maxAbs, rms, maxAbs/rms)
	}
	fmt.Printf("wrote %s\n", *out)
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
