// Command ocr runs PaddleOCR-VL-1.6 over one image, element level (OCR.md
// O5): the image and a task prompt in, the recognised text on stdout, and
// where the time went on stderr.
//
//	go run ./cmd/ocr -image testdata/ocr/ocr_demo.jpg
//	go run ./cmd/ocr -image testdata/ocr/table_recognition.jpg -task table
//	go run ./cmd/ocr -image crop.png -prompt "OCR:" -max-pixels 1605632
//
// -task is one of PaddleX's labels (ocr, table, formula, chart, seal,
// spotting); -prompt overrides it with any text. Generation is greedy to
// </s> or -max-tokens. There is no page layout here: a whole page goes in
// as one image, as PaddleX's demo does, and the layout model is O7.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"time"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/ocr"
	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func main() {
	log.SetFlags(0)
	model := flag.String("model", "models/PaddleOCR-VL-1.6", "checkpoint directory")
	image := flag.String("image", "", "image file (PNG, JPEG or GIF)")
	task := flag.String("task", "ocr", "task label: "+strings.Join(taskNames(), ", "))
	prompt := flag.String("prompt", "", "prompt text, overriding -task")
	maxTokens := flag.Int("max-tokens", ocr.DefaultMaxTokens, "cap on generated tokens")
	minPixels := flag.Int("min-pixels", ocr.DefaultMinPixels, "resize lower bound, in pixels")
	maxPixels := flag.Int("max-pixels", ocr.DefaultMaxPixels, "resize upper bound, in pixels")
	flag.Parse()
	if *image == "" {
		flag.Usage()
		os.Exit(2)
	}
	text := *prompt
	if text == "" {
		var ok bool
		if text, ok = ocr.Tasks[*task]; !ok {
			log.Fatalf("unknown task %q; one of %s", *task, strings.Join(taskNames(), ", "))
		}
	}
	data, err := os.ReadFile(*image)
	must(err)
	img, _, err := pixels.Decode(data)
	must(err)

	dev, done := openDevice()
	defer done()
	o := ocr.DefaultOptions()
	o.MaxPatches = *maxPixels / (ocr.PatchSize * ocr.PatchSize)
	if o.MaxPatches < ocr.MaxPatches {
		o.MaxPatches = ocr.MaxPatches
	}
	start := time.Now()
	e, err := ocr.Load(dev, *model, o)
	must(err)
	defer e.Destroy()
	fmt.Fprintf(os.Stderr, "loaded   %v, %.2f GB on the device\n", time.Since(start).Round(time.Millisecond), float64(e.DeviceBytes())/1e9)

	start = time.Now()
	res, err := e.Recognize(context.Background(), ocr.Request{
		Image: img, Prompt: text, MinPixels: *minPixels, MaxPixels: *maxPixels, MaxTokens: *maxTokens,
	})
	must(err)
	wall := time.Since(start)
	fmt.Println(res.Text)
	fmt.Fprintf(os.Stderr, "image    %dx%d -> %dx%d patches, %d prompt tokens\n", img.W, img.H, res.GridW, res.GridH, res.PromptTokens)
	fmt.Fprintf(os.Stderr, "time     process %v, tower %v, prefill %v, decode %v\n",
		res.Process.Round(time.Millisecond), res.Tower.Round(time.Millisecond),
		res.Prefill.Round(time.Millisecond), res.Decode.Round(time.Millisecond))
	fmt.Fprintf(os.Stderr, "tokens   %d (%s), %.2f ms a token, %.0f tok/s; %v in all\n", len(res.IDs), res.Finish,
		float64(res.Decode.Microseconds())/1e3/float64(len(res.IDs)),
		float64(len(res.IDs))/res.Decode.Seconds(), wall.Round(time.Millisecond))
}

func taskNames() []string {
	var out []string
	for k := range ocr.Tasks {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func openDevice() (*vk.Device, func()) {
	inst, err := vk.NewInstance("ocr")
	must(err)
	devices, err := inst.PhysicalDevices()
	must(err)
	if len(devices) == 0 {
		log.Fatal("no Vulkan devices")
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
		}
	}
	qf, err := phys.ComputeQueueFamily()
	must(err)
	sgs, err := phys.SubgroupSizeControl()
	must(err)
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	must(err)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
