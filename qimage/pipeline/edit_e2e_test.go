package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"strix-halo-vulkan/qimage/dit"
	"strix-halo-vulkan/qimage/textenc"
	qvae "strix-halo-vulkan/qimage/vae"
	"strix-halo-vulkan/qimage/vision"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// The edit oracle is reference/dump_qi21_edit.py: the whole diffusers
// pipeline in fp32 on CPU at 256²/4 steps with one condition image, kv-cache
// on, `output_resolution` set to the card's own size so the pipeline's
// condition resize is the identity and this stays a gate on the pipeline
// rather than on a resampler. Regenerate with:
//
//	.venv/bin/python reference/dump_qi21_edit.py
const editRef = "../../reference/out/qi21edit"

type editManifest struct {
	Prompt          string  `json:"prompt"`
	Size            int     `json:"size"`
	Steps           int     `json:"steps"`
	CondInputSize   []int   `json:"cond_input_size"`
	CondLatentShape []int   `json:"cond_latent_shape"`
	ImgShapes       [][]int `json:"img_shapes"`
	PromptTokens    int     `json:"prompt_tokens"`
	PadMaskCount    int     `json:"pad_mask_count"`
	ResizeOutSize   []int   `json:"resize_out_size"`
	CalcDimensions  []struct {
		Resolution int   `json:"resolution"`
		Src        []int `json:"src"`
		Out        []int `json:"out"`
	} `json:"calculate_dimensions"`
	Tensors map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadEditManifest(t *testing.T) *editManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(editRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_qi21_edit.py", editRef, err)
	}
	var m editManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func loadEditMat(t *testing.T, m *editManifest, name string) *qwen.Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(editRef, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != meta.Count*4 {
		t.Fatalf("%s: %d bytes for %d float32", name, len(raw), meta.Count)
	}
	data := make([]float32, meta.Count)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	cols := meta.Shape[len(meta.Shape)-1]
	return &qwen.Mat{Rows: meta.Count / cols, Cols: cols, Data: data}
}

// TestCalcDimensions checks the condition-image geometry against the
// pipeline's own `calculate_dimensions` over a spread of aspect ratios and
// two target resolutions. It is pure arithmetic and needs no weights, which
// is why it is separate: every condition image the server is ever handed
// goes through it, and the oracle below deliberately exercises only the
// identity case.
func TestCalcDimensions(t *testing.T) {
	m := loadEditManifest(t)
	if len(m.CalcDimensions) == 0 {
		t.Skip("the dump has no calculate_dimensions cases; re-run reference/dump_qi21_edit.py")
	}
	for _, c := range m.CalcDimensions {
		w, h := CalcDimensions(c.Resolution*c.Resolution, float64(c.Src[0])/float64(c.Src[1]))
		if w != c.Out[0] || h != c.Out[1] {
			t.Errorf("%dx%d at resolution %d: got %dx%d, want %dx%d",
				c.Src[0], c.Src[1], c.Resolution, w, h, c.Out[0], c.Out[1])
			continue
		}
		t.Logf("%4dx%-4d at %4d -> %4dx%-4d", c.Src[0], c.Src[1], c.Resolution, w, h)
	}
}

// TestLayoutFromPadMask checks the joint-sequence geometry an edit derives
// from the text encoder's pad mask, against the oracle's own img_shapes and
// token counts. No weights either — this is the piece that decides *where*
// every row of the prefix sits.
func TestLayoutFromPadMask(t *testing.T) {
	m := loadEditManifest(t)
	mask := loadEditMat(t, m, "image_pad_mask")
	pad := make([]bool, mask.Rows*mask.Cols)
	for i := range pad {
		pad[i] = mask.Data[i] != 0
	}
	cond := [3]int{m.CondLatentShape[0], m.CondLatentShape[1], m.CondLatentShape[2]}
	side := m.Size / 16
	lay, err := dit.LayoutFromPadMask(pad, [][3]int{cond}, [3]int{1, side, side})
	if err != nil {
		t.Fatal(err)
	}
	if len(lay.ImgShapes) != len(m.ImgShapes) {
		t.Fatalf("%d image blocks, the oracle has %d", len(lay.ImgShapes), len(m.ImgShapes))
	}
	for i, sh := range lay.ImgShapes {
		want := m.ImgShapes[i]
		if sh[0] != want[0] || sh[1] != want[1] || sh[2] != want[2] {
			t.Errorf("image block %d is %v, the oracle says %v", i, sh, want)
		}
	}
	// The prefix is the prompt's rows with each image slot expanded four-fold,
	// which is the whole point of the mask: 85 prompt rows holding 64 slots
	// become 21 text rows plus 256 latent rows.
	wantPrefix := (m.PromptTokens - m.PadMaskCount) + m.PadMaskCount*4
	if lay.PrefixLen != wantPrefix {
		t.Errorf("prefix is %d rows, want %d", lay.PrefixLen, wantPrefix)
	}
	if got := lay.Seq(); got != wantPrefix+side*side {
		t.Errorf("sequence is %d rows, want %d", got, wantPrefix+side*side)
	}
	// A t2i prompt has no slots, and the same constructor has to be the t2i
	// one — otherwise the edit path is a second layout implementation.
	flat, err := dit.LayoutFromPadMask(make([]bool, 17), nil, [3]int{1, side, side})
	if err != nil {
		t.Fatal(err)
	}
	plain, err := dit.NewLayout([]int{17}, [][3]int{{1, side, side}})
	if err != nil {
		t.Fatal(err)
	}
	if flat.PrefixLen != plain.PrefixLen || flat.Seq() != plain.Seq() || len(flat.Segments) != len(plain.Segments) {
		t.Errorf("the t2i case disagrees with NewLayout: %d/%d rows against %d/%d",
			flat.PrefixLen, flat.Seq(), plain.PrefixLen, plain.Seq())
	}
	t.Logf("%d prompt rows (%d slots) -> %d prefix rows + %d target = %d, segments %d",
		m.PromptTokens, m.PadMaskCount, lay.PrefixLen, side*side, lay.Seq(), len(lay.Segments))
}

// TestEndToEndEdit is Q8.3's gate: a condition image and a prompt through
// this repo's own tokenizer, vision tower, text encoder, VAE encoder, DiT,
// scheduler and VAE decoder — only the initial noise comes from the oracle,
// as data — against the 256²/4-step reference edit.
//
// It is the edit twin of TestEndToEndImage and is measured the same way: the
// component gates do not compose into this claim. ~68 GB of fp32 weights
// pass through, sequenced so the peak stays near the text encoder's 34 GB;
// it must run alone on the machine.
func TestEndToEndEdit(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the whole fp32 edit pipeline")
	}
	m := loadEditManifest(t)

	// --- the condition image's two copies. The VAE reads all four channels;
	// the tower reads the same picture flattened over white.
	rgba := loadEditMat(t, m, "cond_rgba") // [4, H*W] folded to rows
	cw, ch := m.CondInputSize[0], m.CondInputSize[1]
	cond := qvae.NewTensor(1, 4, ch, cw)
	copy(cond.Data, rgba.Data)
	flat := flattenOverWhite(cond)

	// --- text: our tower, our mrope, our deepstack injection.
	vcfg, err := vision.LoadConfig(model + "/text_encoder")
	if err != nil {
		t.Skipf("no checkpoint at %s (%v)", model, err)
	}
	tower, err := vision.Load(model+"/text_encoder", vcfg, 0)
	if err != nil {
		t.Fatal(err)
	}
	pixels, gridH, gridW, err := vcfg.Patchify(flat.Data, ch, cw)
	if err != nil {
		t.Fatal(err)
	}
	vout, err := tower.Forward(pixels, gridH, gridW, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The negative control for the 8-bit composite, run here because the
	// tower is staged and this is where the difference is born: the same
	// picture composited in float instead of on the uint8 levels PIL uses.
	// Half a level per pixel, and the port that did it this way first missed
	// the oracle's prompt embedding by rel 11.
	soft, _, _, err := vcfg.Patchify(flattenWith(cond, false).Data, ch, cw)
	if err != nil {
		t.Fatal(err)
	}
	softOut, err := tower.Forward(soft, gridH, gridW, nil)
	if err != nil {
		t.Fatal(err)
	}
	tower = nil
	runtime.GC()

	tok, err := tokenizer.Load(model + "/processor")
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := textenc.NewEditPrompt(tok, m.Prompt,
		[]textenc.Grid{{T: 1, H: gridH, W: gridW}}, vcfg.SpatialMergeSize)
	if err != nil {
		t.Fatal(err)
	}
	tcfg, err := textenc.LoadConfig(model + "/text_encoder")
	if err != nil {
		t.Fatal(err)
	}
	section, err := textenc.LoadMRope(model + "/text_encoder")
	if err != nil {
		t.Fatal(err)
	}
	rope, err := prompt.MRope(tcfg.HeadDim, tcfg.RopeTheta, section)
	if err != nil {
		t.Fatal(err)
	}
	drop, err := textenc.DropTokens(tok)
	if err != nil {
		t.Fatal(err)
	}
	te, err := qwen.LoadWith(model+"/text_encoder", tcfg, tcfg.NumLayers)
	if err != nil {
		t.Fatal(err)
	}
	embeds, padMask, err := prompt.Encode(te, rope,
		[]textenc.Condition{{Merged: vout.Merged, Deepstack: vout.Deepstack}}, drop, nil)
	if err != nil {
		t.Fatal(err)
	}
	embeds = embeds.Clone()
	compareStepAt(t, "prompt_embeds", embeds, loadEditMat(t, m, "prompt_embeds"), embedSpec)

	// The control, with the text model still staged: the float composite's
	// vision rows through the same encoder. It has to land far outside the
	// bound, or "composite on the levels" is a preference rather than a
	// requirement.
	softEmbeds, _, err := prompt.Encode(te, rope,
		[]textenc.Condition{{Merged: softOut.Merged, Deepstack: softOut.Deepstack}}, drop, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantEmbeds := loadEditMat(t, m, "prompt_embeds")
	if rel := relDeviation(softEmbeds, wantEmbeds); rel <= embedSpec {
		t.Errorf("the float composite still matches at rel %.3g, inside the %.0e bound", rel, embedSpec)
	} else {
		t.Logf("control %-28s rel %.3g, %.0fx the bound (merged rows differ by %.3g)",
			"float composite, not 8-bit", rel, rel/embedSpec, relDeviation(softOut.Merged, vout.Merged))
	}
	te = nil
	runtime.GC()

	// --- the condition latents, which are the other half of the prefix.
	vaecfg, err := qvae.LoadConfig(model + "/vae")
	if err != nil {
		t.Fatal(err)
	}
	enc, err := qvae.LoadEncoder(model+"/vae", vaecfg)
	if err != nil {
		t.Fatal(err)
	}
	condLatents, condShape, err := EncodeCondition(enc, vaecfg, cond)
	if err != nil {
		t.Fatal(err)
	}
	enc = nil
	runtime.GC()
	compareStepAt(t, "cond_latents", condLatents, loadEditMat(t, m, "cond_latents"), condSpec)
	if want := m.CondLatentShape; condShape[1] != want[1] || condShape[2] != want[2] {
		t.Fatalf("condition latents are %v, the oracle says %v", condShape, want)
	}

	// --- denoise from the oracle's noise, with the prefix in front.
	dcfg, err := dit.LoadConfig(transformer)
	if err != nil {
		t.Fatal(err)
	}
	dm, err := dit.Load(transformer, dcfg.NumLayers)
	if err != nil {
		t.Fatal(err)
	}
	side := m.Size / 16
	lay, err := dit.LayoutFromPadMask(padMask, [][3]int{condShape}, [3]int{1, side, side})
	if err != nil {
		t.Fatal(err)
	}
	scfg, err := LoadSchedConfig(model + "/scheduler")
	if err != nil {
		t.Fatal(err)
	}
	// The shift is a function of the *target* token count, not the joint
	// sequence — a bigger prefix does not move the schedule.
	sched, err := scfg.Timesteps(m.Steps, side*side)
	if err != nil {
		t.Fatal(err)
	}
	steps := map[int]*qwen.Mat{}
	latents, err := Denoise(dm, embeds, lay, condLatents, loadEditMat(t, m, "noise"), sched,
		func(step int, l *qwen.Mat) { steps[step] = l.Clone() })
	if err != nil {
		t.Fatal(err)
	}
	dm = nil
	runtime.GC()
	for i := 0; i < m.Steps; i++ {
		compareStepAt(t, fmt.Sprintf("step%d_latents", i), steps[i],
			loadEditMat(t, m, fmt.Sprintf("step%d_latents", i)), latentSpec)
	}

	// --- decode and compare the picture.
	dec, err := qvae.LoadDecoder(model+"/vae", vaecfg)
	if err != nil {
		t.Fatal(err)
	}
	z, err := UnpackLatents(latents, side, side)
	if err != nil {
		t.Fatal(err)
	}
	vaecfg.Denormalize(z)
	img, err := dec.Decode(z)
	if err != nil {
		t.Fatal(err)
	}
	want := loadEditMat(t, m, "image") // [H*W, 4] in [0, 1]
	var maxAbs, sumAbs float64
	for h := 0; h < img.H; h++ {
		for w := 0; w < img.W; w++ {
			for c := 0; c < img.C; c++ {
				got := (float64(img.Plane(0, c)[h*img.W+w]) + 1) / 2
				d := math.Abs(got - float64(want.Row(h*img.W + w)[c]))
				if d > maxAbs {
					maxAbs = d
				}
				sumAbs += d
			}
		}
	}
	mean := sumAbs / float64(img.C*img.H*img.W)
	if maxAbs > imgSpec {
		t.Errorf("edited image: max abs %.5f (mean %.6f) > %.3g against the oracle", maxAbs, mean, imgSpec)
		return
	}
	t.Logf("edited image: max abs %.5f, mean %.6f against the oracle (bound %.3g)", maxAbs, mean, imgSpec)
}

// relDeviation is compareStepAt's metric without the assertion, for the
// controls: the worst per-element deviation, floored by the reference's rms
// so a small element cannot dominate.
func relDeviation(got, want *qwen.Mat) float64 {
	var sumSq float64
	for i := range want.Data {
		sumSq += float64(want.Data[i]) * float64(want.Data[i])
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var rel float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		if r := d / math.Max(math.Abs(float64(want.Data[i])), math.Max(rms, 1e-12)); r > rel {
			rel = r
		}
	}
	return rel
}

// condSpec is the bound on the condition latents. The VAE encoder's own gate
// (qimage/vae) measures the posterior mode at 4.7e-5 against the dump, and
// nothing here is deeper than that — this is the same encoder over the same
// card, so the bound is the encoder's, not a new one.
const condSpec = 2e-4

// latentSpec is Q3's per-step bound, measured there at 2.1e-5 to 2.5e-4 over
// the t2i oracle's four steps.
const latentSpec = 1e-3

// flattenOverWhite composites an RGBA condition image over white and drops
// the alpha, which is the copy the *vision tower* reads. The VAE reads the
// four-channel original; the pipeline flattens once, for one of the two
// encoders, and a port that flattens for both is wrong only on transparent
// references — which is exactly the case this model exists to serve.
//
// **The composite is done in 8-bit**, because the reference's is: PIL's
// `paste` works on the uint8 image, so every composited pixel lands on a
// level. Doing the same arithmetic in float leaves 97% of the pixels half a
// level away from the reference's, and this file's own negative control
// measures what that costs downstream — it is not a rounding detail, it is
// the difference between reproducing the oracle and not. A served edit is
// handed an 8-bit PNG, so quantizing here is also what the real input
// looks like.
func flattenOverWhite(rgba *qvae.Tensor) *qvae.Tensor {
	return flattenWith(rgba, true)
}

func flattenWith(rgba *qvae.Tensor, quantize bool) *qvae.Tensor {
	out := qvae.NewTensor(1, 3, rgba.H, rgba.W)
	alpha := rgba.Plane(0, 3)
	for c := 0; c < 3; c++ {
		src, dst := rgba.Plane(0, c), out.Plane(0, c)
		for i := range dst {
			// [-1, 1] to [0, 1], composite over white, quantize, and back.
			a := (float64(alpha[i]) + 1) / 2
			v := (float64(src[i]) + 1) / 2
			c := v*a + (1 - a)
			if quantize {
				dst[i] = float32(math.Round(c*255)/127.5 - 1)
				continue
			}
			dst[i] = float32(c*2 - 1)
		}
	}
	return out
}
