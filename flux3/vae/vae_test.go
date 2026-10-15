package vae

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The reference is reference/dump_flux3_vae.py: diffusers' AutoencoderKLFlux2
// in fp32 on the CPU, hooked at every top-level stage — a decode of the t2i
// run oracle's own latent (1024²) and an encode of a 768x512 picture.
// Regenerate with
//
//	.venv-flux3/bin/python -I reference/dump_flux3_vae.py
const (
	dumpDir = "../../reference/out/flux3vae"
	runDir  = "../../reference/out/flux3run_t2i"
	vaeDir  = "../../models/flux-3-image-klein/vae"
)

type dump struct {
	dir     string
	Tensors map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func loadDump(t *testing.T, dir string) *dump {
	t.Helper()
	if _, err := os.Stat(filepath.Join(vaeDir, "config.json")); err != nil {
		t.Skipf("no vae at %s (%v)", vaeDir, err)
	}
	buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Skipf("no dump in %s (%v)", dir, err)
	}
	d := &dump{dir: dir}
	if err := json.Unmarshal(buf, d); err != nil {
		t.Fatal(err)
	}
	return d
}

// tensor reads a dumped [C, H, W] array as a batch-1 Tensor.
func (d *dump) tensor(t *testing.T, name string) *Tensor {
	t.Helper()
	meta, ok := d.Tensors[name]
	if !ok {
		t.Fatalf("%s has no tensor %q", d.dir, name)
	}
	raw, err := os.ReadFile(filepath.Join(d.dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	data := make([]float32, meta.Count)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	sh := meta.Shape
	if len(sh) != 3 {
		t.Fatalf("%s is %v, want [C H W]", name, sh)
	}
	return &Tensor{N: 1, C: sh[0], H: sh[1], W: sh[2], Data: data}
}

// stageTol is the per-stage fp32 bound. X4 measured every stage at ≤ 2.8e-5
// (up block 2) once the mid attention accumulates in float64; before that
// the decoder's mid block sat at 1.08e-4 while torch's fp32 was 1.8e-6 from
// float64, so the bound caught a real precision loss and is kept.
const stageTol = 1e-4

func compare(t *testing.T, name string, got, want *Tensor, tol float64) {
	t.Helper()
	if got.C != want.C || got.H != want.H || got.W != want.W {
		t.Fatalf("%s: got %s, want %s", name, got, want)
	}
	var sumSq float64
	for _, w := range want.Data {
		sumSq += float64(w) * float64(w)
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var maxAbs, rel float64
	for i, w := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(w))
		maxAbs = math.Max(maxAbs, d)
		rel = math.Max(rel, d/math.Max(math.Abs(float64(w)), rms))
	}
	if rel > tol || math.IsNaN(rel) {
		t.Errorf("%-20s %s: max abs %.4g, worst rel %.3g, rms %.4g > %.0e", name, got, maxAbs, rel, rms, tol)
		return
	}
	t.Logf("%-20s %-18s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
}

func step(t *testing.T, name string, f func() (*Tensor, error)) *Tensor {
	t.Helper()
	out, err := f()
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return out
}

// TestDecoderStages runs every decoder stage on the dump's input to that
// stage. Several minutes of CPU at 1024².
func TestDecoderStages(t *testing.T) {
	if testing.Short() {
		t.Skip("a 1024² decode on the CPU")
	}
	d := loadDump(t, dumpDir)
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := LoadDecoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	in := d.tensor(t, "dec_in")
	compare(t, "post_quant_conv", step(t, "post_quant_conv", func() (*Tensor, error) { return dec.PostQuantConv.Apply(in) }),
		d.tensor(t, "dec_post_quant_conv"), stageTol)
	compare(t, "conv_in", step(t, "conv_in", func() (*Tensor, error) { return dec.ConvIn.Apply(d.tensor(t, "dec_post_quant_conv")) }),
		d.tensor(t, "dec_conv_in"), stageTol)
	compare(t, "mid_block", step(t, "mid", func() (*Tensor, error) { return dec.Mid.Apply(d.tensor(t, "dec_conv_in")) }),
		d.tensor(t, "dec_mid_block"), stageTol)
	prev := "dec_mid_block"
	for i, up := range dec.UpBlocks {
		name := fmt.Sprintf("dec_up_blocks_%d", i)
		x := d.tensor(t, prev)
		compare(t, name[4:], step(t, name, func() (*Tensor, error) { return up.Apply(x) }), d.tensor(t, name), stageTol)
		prev = name
	}
	compare(t, "conv_norm_out", step(t, "norm", func() (*Tensor, error) { return dec.ConvNormOut.ApplyInPlace(d.tensor(t, prev)) }),
		d.tensor(t, "dec_conv_norm_out"), stageTol)
	compare(t, "conv_out", step(t, "conv_out", func() (*Tensor, error) {
		return dec.ConvOut.Apply(SiLUInPlace(d.tensor(t, "dec_conv_norm_out")))
	}), d.tensor(t, "dec_conv_out"), stageTol)
}

// TestEncoderStages runs every encoder stage on the dump's input to that
// stage, then the mode.
func TestEncoderStages(t *testing.T) {
	if testing.Short() {
		t.Skip("a 768x512 encode on the CPU")
	}
	d := loadDump(t, dumpDir)
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := LoadEncoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "conv_in", step(t, "conv_in", func() (*Tensor, error) { return enc.ConvIn.Apply(d.tensor(t, "enc_in")) }),
		d.tensor(t, "enc_conv_in"), stageTol)
	prev := "enc_conv_in"
	for i, down := range enc.DownBlocks {
		name := fmt.Sprintf("enc_down_blocks_%d", i)
		x := d.tensor(t, prev)
		compare(t, name[4:], step(t, name, func() (*Tensor, error) { return down.Apply(x) }), d.tensor(t, name), stageTol)
		prev = name
	}
	compare(t, "mid_block", step(t, "mid", func() (*Tensor, error) { return enc.Mid.Apply(d.tensor(t, prev)) }),
		d.tensor(t, "enc_mid_block"), stageTol)
	compare(t, "conv_norm_out", step(t, "norm", func() (*Tensor, error) { return enc.ConvNormOut.ApplyInPlace(d.tensor(t, "enc_mid_block")) }),
		d.tensor(t, "enc_conv_norm_out"), stageTol)
	compare(t, "conv_out", step(t, "conv_out", func() (*Tensor, error) {
		return enc.ConvOut.Apply(SiLUInPlace(d.tensor(t, "enc_conv_norm_out")))
	}), d.tensor(t, "enc_conv_out"), stageTol)
	compare(t, "quant_conv", step(t, "quant_conv", func() (*Tensor, error) { return enc.QuantConv.Apply(d.tensor(t, "enc_conv_out")) }),
		d.tensor(t, "enc_quant_conv"), stageTol)
	compare(t, "mode", step(t, "mode", func() (*Tensor, error) { return Mode(d.tensor(t, "enc_quant_conv"), cfg.LatentChannels) }),
		d.tensor(t, "enc_mode"), 0)
	// and the whole encoder free-running from the pixels
	compare(t, "encode (free)", step(t, "encode", func() (*Tensor, error) { return enc.Encode(d.tensor(t, "enc_in")) }),
		d.tensor(t, "enc_mode"), freeTol)
}

// freeTol is the free-running bound, encoder or decoder. X4 measured rel
// 1.4e-5 (768x512 encode) and 3.0e-5 (1024² decode); 1e-4 keeps 3x.
const freeTol = 1e-4

// TestDecodeImage decodes the t2i run's own latent free-running and compares
// the 8-bit picture with the PNG the run wrote.
func TestDecodeImage(t *testing.T) {
	if testing.Short() {
		t.Skip("a 1024² decode on the CPU")
	}
	d := loadDump(t, dumpDir)
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := LoadDecoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	img, err := dec.Apply(d.tensor(t, "dec_in"))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "decode (free)", img, d.tensor(t, "dec_out"), freeTol)

	png := loadDump(t, runDir)
	meta := png.Tensors["image"] // [H, W, 3] of 0..255 as float32
	raw, err := os.ReadFile(filepath.Join(runDir, "image.bin"))
	if err != nil {
		t.Fatal(err)
	}
	rgb := RGB8(img)
	diff, worst := 0, 0
	for i := 0; i < meta.Count; i++ {
		want := int(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
		if g := int(rgb[i]); g != want {
			diff++
			if a := abs(g - want); a > worst {
				worst = a
			}
		}
	}
	if worst > 1 {
		t.Errorf("8-bit picture: %d of %d levels differ, worst by %d", diff, meta.Count, worst)
		return
	}
	t.Logf("8-bit picture: %d of %d levels differ (%.3f%%), worst by %d", diff, meta.Count, 100*float64(diff)/float64(meta.Count), worst)
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
