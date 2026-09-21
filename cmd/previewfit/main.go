// Command previewfit fits the linear 64->RGBA preview matrix that
// qimage/pipeline.PreviewDecode uses, and writes it as Go source
// (IMAGE.md Q7).
//
//	go run ./cmd/previewfit -out qimage/pipeline/preview_matrix.go
//	go run ./cmd/previewfit -size 512 -steps 8 -report   # refit and score it
//
// **It generates its own training data rather than reading a dump**, which
// is the one judgement in here. Q0's dumps carry two (latent, image) pairs;
// this runs the served pipeline over a prompt set chosen to span colour and
// content, and every pair it collects is a latent this decoder decoded into
// that image -- the two sides cannot disagree about normalisation, layout or
// range, which is exactly the class of mistake a hand-assembled pair set
// invites.
//
// The fit itself is ordinary least squares on the normal equations: one
// 65x65 system (64 latent channels plus a bias) with four right-hand sides,
// solved in float64 by Gaussian elimination with partial pivoting. At that
// size nothing about conditioning is interesting, and the residual it
// reports is what says whether the model is adequate -- a linear map of a
// 64-channel latent cannot represent a convolutional decoder, and the number
// to read is how much of the colour it does get.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"

	"strix-halo-vulkan/qimage/pipeline"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
	zvae "strix-halo-vulkan/zimage/vae"
)

const strixHaloDeviceID = 0x1586

// fitPrompts span what the matrix has to get right: saturated primaries,
// skin and foliage, night and snow, flat graphic colour and neutral greys.
// A latent2rgb matrix that saw only photographs of one palette maps
// everything else toward it, and the failure is invisible in the fit's own
// residual.
var fitPrompts = []string{
	"a red fox sitting in fresh snow at dawn, photograph",
	"an oil painting of a harbour at sunset, thick impasto brushwork",
	"a bowl of lemons and limes on a blue tablecloth, still life photograph",
	"a dense green rainforest canopy seen from above, aerial photograph",
	"a portrait of an elderly woman with grey hair, soft window light",
	"a neon-lit city street at night in the rain, cinematic",
	"a white ceramic teapot on a white table, high key studio lighting",
	"a black cat asleep on a dark sofa, low key photograph",
	"a flat vector illustration of a mountain range, bold orange and teal",
	"a close-up of rusted iron and peeling blue paint, texture photograph",
	"a field of purple lavender under a stormy grey sky",
	"a bright yellow taxi on a wet asphalt road, shallow depth of field",
	"an underwater photograph of a coral reef, turquoise water",
	"a slice of chocolate cake on a pink plate, food photography",
	"a snowy mountain peak against a deep blue sky, clear day",
	"a crowded market stall of colourful spices, warm light",
	"a single white daisy on a black background, macro photograph",
	"a desert dune landscape at midday, pale sand and hard shadows",
	"a stack of old books and a brass lamp, sepia tones",
	"a cartoon sticker of a smiling green frog, flat colour",
	"a stormy grey sea with white foam, long exposure",
	"a bright red London bus on a grey street",
	"a bunch of ripe bananas on a wooden board",
	"an abstract pastel gradient of pink and mint, smooth",
}

func main() {
	log.SetFlags(0)
	model := flag.String("model", "models/Qwen-Image-2.1", "checkpoint root")
	out := flag.String("out", "", "write the Go source here; empty prints it")
	size := flag.Int("size", 512, "fitting resolution; each image gives (size/16)^2 samples")
	steps := flag.Int("steps", 8, "denoising steps per fitting image; the picture only has to be a picture")
	seed := flag.Int64("seed", 20260920, "base seed")
	report := flag.Bool("report", false, "also score the fit against a held-out tail of the prompt set")
	flag.Parse()

	inst, err := vk.NewInstance("previewfit")
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

	p, err := pipeline.New(dev, pipeline.Options{
		Model: *model, Width: *size, Height: *size, Steps: *steps,
	})
	must(err)
	defer p.Destroy()

	side := *size / 16
	// Held out: the last quarter of the prompt set never enters the normal
	// equations, so -report's number is a generalisation error rather than a
	// residual. A 260-parameter model over 6000 samples would not overfit,
	// but "would not" is not a measurement.
	hold := len(fitPrompts)
	if *report {
		hold = len(fitPrompts) * 3 / 4
	}

	var train, test []sample
	start := time.Now()
	for i, prompt := range fitPrompts {
		latents, img, err := renderPair(p, prompt, *seed+int64(i), *size, *steps)
		must(err)
		s := pairSamples(latents, img, side)
		if i < hold {
			train = append(train, s...)
		} else {
			test = append(test, s...)
		}
		fmt.Fprintf(os.Stderr, "  [%2d/%d] %-52s %d samples\n", i+1, len(fitPrompts), truncate(prompt, 52), len(s))
	}
	fmt.Fprintf(os.Stderr, "collected %d training samples (%d held out) in %v\n",
		len(train), len(test), time.Since(start).Round(time.Second))

	w, b := fit(train)
	reportFit(os.Stderr, "train", train, w, b)
	if len(test) > 0 {
		reportFit(os.Stderr, "held out", test, w, b)
	}

	src := render(w, b)
	if *out == "" {
		fmt.Print(src)
		return
	}
	must(os.WriteFile(*out, []byte(src), 0o644))
	fmt.Fprintf(os.Stderr, "wrote %s\n", *out)
}

// sample is one latent pixel and the average colour of the 16x16 image block
// it produced.
type sample struct {
	z   []float32
	rgb [4]float64
}

// renderPair generates one image and returns the final latents beside it.
func renderPair(p *pipeline.Pipeline, prompt string, seed int64, size, steps int) (*qwen.Mat, *zvae.Tensor, error) {
	var final *qwen.Mat
	img, _, err := p.Run(context.Background(), pipeline.Request{
		Prompt: prompt, Width: size, Height: size, Steps: steps, Seed: seed,
		// The last step's latents are the ones that were decoded. Cloning in
		// the callback rather than reading the request back is what keeps the
		// pair honest: the loop integrates in place.
		Progress: func(st pipeline.Step) {
			if st.Index == st.Steps-1 {
				final = st.Latents.Clone()
			}
		},
	})
	if err != nil {
		return nil, nil, err
	}
	return final, img, nil
}

// pairSamples box-filters each 16x16 image block down to the latent pixel
// that produced it.
//
// A box filter and not something cleverer: the decoder's receptive field is
// far wider than its stride, so no resampling kernel makes this pairing
// exact, and the least-squares fit is what absorbs the difference. Pretending
// otherwise by picking a fancier filter would be choosing a number to make
// the residual look smaller.
func pairSamples(latents *qwen.Mat, img *zvae.Tensor, side int) []sample {
	const scale = 16
	out := make([]sample, 0, side*side)
	plane := img.H * img.W
	for ly := 0; ly < side; ly++ {
		for lx := 0; lx < side; lx++ {
			var s sample
			s.z = latents.Row(ly*side + lx)
			for c := 0; c < 4 && c < img.C; c++ {
				var acc float64
				for dy := 0; dy < scale; dy++ {
					row := (ly*scale + dy) * img.W
					for dx := 0; dx < scale; dx++ {
						acc += float64(img.Data[c*plane+row+lx*scale+dx])
					}
				}
				s.rgb[c] = acc / (scale * scale)
			}
			out = append(out, s)
		}
	}
	return out
}

const (
	zDim     = 64
	channels = 4
	dim      = zDim + 1 // the bias rides as a constant column
)

// fit solves the normal equations for all four output channels at once.
func fit(samples []sample) ([channels][zDim]float64, [channels]float64) {
	var ata [dim][dim]float64
	var atb [dim][channels]float64
	for _, s := range samples {
		var x [dim]float64
		for k := 0; k < zDim; k++ {
			x[k] = float64(s.z[k])
		}
		x[zDim] = 1
		for i := 0; i < dim; i++ {
			for j := i; j < dim; j++ {
				ata[i][j] += x[i] * x[j]
			}
			for c := 0; c < channels; c++ {
				atb[i][c] += x[i] * s.rgb[c]
			}
		}
	}
	for i := 0; i < dim; i++ {
		for j := 0; j < i; j++ {
			ata[i][j] = ata[j][i]
		}
	}
	sol := solve(ata, atb)

	var w [channels][zDim]float64
	var b [channels]float64
	for c := 0; c < channels; c++ {
		for k := 0; k < zDim; k++ {
			w[c][k] = sol[k][c]
		}
		b[c] = sol[zDim][c]
	}
	return w, b
}

// solve is Gaussian elimination with partial pivoting over four right-hand
// sides. 65x65 in float64: nothing here is worth a library.
func solve(a [dim][dim]float64, rhs [dim][channels]float64) [dim][channels]float64 {
	for col := 0; col < dim; col++ {
		pivot := col
		for r := col + 1; r < dim; r++ {
			if math.Abs(a[r][col]) > math.Abs(a[pivot][col]) {
				pivot = r
			}
		}
		a[col], a[pivot] = a[pivot], a[col]
		rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
		if a[col][col] == 0 {
			log.Fatalf("previewfit: singular normal matrix at column %d; the prompt set produced degenerate latents", col)
		}
		inv := 1 / a[col][col]
		for r := col + 1; r < dim; r++ {
			f := a[r][col] * inv
			if f == 0 {
				continue
			}
			for j := col; j < dim; j++ {
				a[r][j] -= f * a[col][j]
			}
			for c := 0; c < channels; c++ {
				rhs[r][c] -= f * rhs[col][c]
			}
		}
	}
	var sol [dim][channels]float64
	for i := dim - 1; i >= 0; i-- {
		for c := 0; c < channels; c++ {
			acc := rhs[i][c]
			for j := i + 1; j < dim; j++ {
				acc -= a[i][j] * sol[j][c]
			}
			sol[i][c] = acc / a[i][i]
		}
	}
	return sol
}

// reportFit is the number that says whether a linear map is adequate: the
// per-channel RMS error in [-1, 1], and the fraction of the target's own
// variance it explains.
func reportFit(w2 *os.File, label string, samples []sample, w [channels][zDim]float64, b [channels]float64) {
	var sse, sst, mean [channels]float64
	for _, s := range samples {
		for c := 0; c < channels; c++ {
			mean[c] += s.rgb[c]
		}
	}
	for c := range mean {
		mean[c] /= float64(len(samples))
	}
	for _, s := range samples {
		for c := 0; c < channels; c++ {
			pred := b[c]
			for k := 0; k < zDim; k++ {
				pred += w[c][k] * float64(s.z[k])
			}
			d := pred - s.rgb[c]
			sse[c] += d * d
			d0 := s.rgb[c] - mean[c]
			sst[c] += d0 * d0
		}
	}
	names := [channels]string{"R", "G", "B", "A"}
	fmt.Fprintf(w2, "%-9s (%d samples):\n", label, len(samples))
	for c := 0; c < channels; c++ {
		rms := math.Sqrt(sse[c] / float64(len(samples)))
		r2 := 0.0
		if sst[c] > 0 {
			r2 = 1 - sse[c]/sst[c]
		}
		fmt.Fprintf(w2, "  %s  rms %.4f in [-1,1] = %.1f of 255 8-bit levels  R2 %.3f\n",
			names[c], rms, rms*127.5, r2)
	}
}

func render(w [channels][zDim]float64, b [channels]float64) string {
	var s strings.Builder
	s.WriteString(`// Code generated by cmd/previewfit. DO NOT EDIT.

package pipeline

// The fitted linear preview -- IMAGE.md Q7, see preview.go for what it is
// and cmd/previewfit for how it was measured. Regenerate with:
//
//	go run ./cmd/previewfit -out qimage/pipeline/preview_matrix.go

const (
	previewZDim     = 64
	previewChannels = 4
)

// previewBias is the colour a zero latent maps to, per channel.
var previewBias = [previewChannels]float32{
`)
	for c := 0; c < channels; c++ {
		fmt.Fprintf(&s, "\t%s,\n", f32(b[c]))
	}
	s.WriteString(`}

// previewWeight is the 4x64 map, output channel major.
var previewWeight = [previewChannels][previewZDim]float32{
`)
	for c := 0; c < channels; c++ {
		s.WriteString("\t{\n")
		for k := 0; k < zDim; k++ {
			if k%4 == 0 {
				s.WriteString("\t\t")
			}
			fmt.Fprintf(&s, "%s,", f32(w[c][k]))
			if k%4 == 3 {
				s.WriteString("\n")
			} else {
				s.WriteString(" ")
			}
		}
		s.WriteString("\t},\n")
	}
	s.WriteString("}\n")
	return s.String()
}

func f32(v float64) string { return fmt.Sprintf("%+.6f", float32(v)) }

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
