package vision

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// The reference comes from reference/dump_qi21_vision.py: the real Qwen3-VL
// vision tower in fp32 on CPU over a deterministic 256x256 RGBA card, every
// stage hooked. Regenerate with:
//
//	.venv/bin/python reference/dump_qi21_vision.py
const (
	refDir  = "../../reference/out/qi21vision"
	encoder = "../../models/Qwen-Image-2.1/text_encoder"
)

type manifest struct {
	Seq               int     `json:"seq"`
	Size              int     `json:"size"`
	DropIdx           int     `json:"drop_idx"`
	GridTHW           [][]int `json:"grid_thw"`
	MultiGridTHW      [][]int `json:"multi_grid_thw"`
	MultiSizes        [][]int `json:"multi_sizes"`
	ImagePadPositions []int   `json:"image_pad_positions"`
	Vision            struct {
		Depth            int   `json:"depth"`
		DeepstackIndexes []int `json:"deepstack_visual_indexes"`
	} `json:"vision"`
	Tensors map[string]struct {
		Shape  []int   `json:"shape"`
		Count  int     `json:"count"`
		Sum    float64 `json:"sum"`
		AbsMax float64 `json:"absmax"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_qi21_vision.py", refDir, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// loadRef reads a dumped tensor as a matrix, folding every leading axis into
// the row count.
func loadRef(t *testing.T, m *manifest, name string) *qwen.Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(refDir, name+".bin"))
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

// deepTol is the bound for the tower's stages, and it is set the way every
// other bound in this vertical is: from the model's own arithmetic rather
// than from taste. Both sides are fp32; the drift is summation order through
// 27 pre-norm blocks whose residual stream grows to absmax 1.3e4 by the last
// hidden state (the dump records it), so the rms floor in `compare` is what
// keeps that growth from hiding an error on a small element.
//
// A structural mistake — the raster token order instead of block-major, the
// wrong GELU, the wrong merger norm placement, adjacent-pair rotation
// instead of halves — shows at rel >= 1e-1 on the first stage it touches.
// TestNegativeControls is that claim measured rather than asserted.
const deepTol = 2e-4

func compare(t *testing.T, name string, got, want *qwen.Mat) float64 {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %dx%d, want %dx%d", name, got.Rows, got.Cols, want.Rows, want.Cols)
	}
	var sumSq float64
	for _, v := range want.Data {
		sumSq += float64(v) * float64(v)
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var maxAbs, rel float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		if d > maxAbs {
			maxAbs = d
		}
		if r := d / math.Max(math.Abs(float64(want.Data[i])), math.Max(rms, 1e-12)); r > rel {
			rel = r
		}
	}
	if rel > deepTol {
		t.Errorf("%-20s %dx%d: max abs %.4g, rel %.3g, rms %.4g > %.0e",
			name, got.Rows, got.Cols, maxAbs, rel, rms, deepTol)
		return rel
	}
	t.Logf("%-20s %5dx%-5d max abs %.3g  rms %.4g  rel %.2g", name, got.Rows, got.Cols, maxAbs, rms, rel)
	return rel
}

func load(t *testing.T, blocks int) (*Config, *Model, *manifest) {
	t.Helper()
	m := loadManifest(t)
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	model, err := Load(encoder, cfg, blocks)
	if err != nil {
		t.Fatal(err)
	}
	return cfg, model, m
}

// TestPatchify checks the processor's pixel layout — block-major token
// order, channels-then-temporal-then-pixels inside a row, the still frame
// repeated on the temporal axis — against what the processor actually
// emitted. It needs no weights, which is why it runs first: every later
// stage is reading these rows.
func TestPatchify(t *testing.T) {
	m := loadManifest(t)
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	card := loadRef(t, m, "card_vision_rgb") // [1, 3, H, W] folded to rows
	h, w := m.Size, m.Size
	got, gridH, gridW, err := cfg.Patchify(card.Data, h, w)
	if err != nil {
		t.Fatal(err)
	}
	if g := m.GridTHW[0]; gridH != g[1] || gridW != g[2] {
		t.Fatalf("grid %dx%d, dump says %v", gridH, gridW, g)
	}
	compare(t, "pixel_values", got, loadRef(t, m, "pixel_values"))
}

// TestGeometry checks the two pieces that decide *where* everything goes:
// the bilinear resample of the 48x48 learned position grid, and the axial
// rope table. Both are pure geometry, so they are exact to fp32 rounding
// and a disagreement is a layout bug rather than drift.
func TestGeometry(t *testing.T) {
	cfg, model, m := load(t, 1)
	gridH, gridW := m.GridTHW[0][1], m.GridTHW[0][2]

	pos, err := model.positionEmbedding(gridH, gridW)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "pos_embeds", pos, loadRef(t, m, "vis_pos_embeds"))

	rope := NewRope(cfg, gridH, gridW)
	compare(t, "rope_cos", rope.Cos, loadRef(t, m, "vis_rope_cos"))
	compare(t, "rope_sin", rope.Sin, loadRef(t, m, "vis_rope_sin"))
}

// TestTowerPrefix walks the first two blocks, which is enough to catch every
// per-block mistake without staging the whole 1.0 GB tower: the patch
// embedding, the position sum, and both residual halves twice over.
func TestTowerPrefix(t *testing.T) {
	cfg, model, m := load(t, 2)
	card := loadRef(t, m, "card_vision_rgb")
	pixels, gridH, gridW, err := cfg.Patchify(card.Data, m.Size, m.Size)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"patch_embed": "vis_patch_embed",
		"block_input": "vis_block_input",
		"block0_out":  "vis_block0_out",
		"block1_out":  "vis_block1_out",
	}
	seen := 0
	_, err = model.Forward(pixels, gridH, gridW, func(name string, x *qwen.Mat) {
		ref, ok := want[name]
		if !ok {
			return
		}
		compare(t, name, x, loadRef(t, m, ref))
		seen++
	})
	if err != nil {
		t.Fatal(err)
	}
	if seen != len(want) {
		t.Fatalf("%d of %d stages compared", seen, len(want))
	}
}

// TestTower is the full gate: all 27 blocks, the three deepstack taps at
// layers 8/16/24, the last hidden state and the merged rows the text
// encoder scatters into its image slots.
func TestTower(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole 27-layer tower")
	}
	cfg, model, m := load(t, 0)
	card := loadRef(t, m, "card_vision_rgb")
	pixels, gridH, gridW, err := cfg.Patchify(card.Data, m.Size, m.Size)
	if err != nil {
		t.Fatal(err)
	}
	out, err := model.Forward(pixels, gridH, gridW, func(name string, x *qwen.Mat) {
		for _, i := range m.Vision.DeepstackIndexes {
			if name == fmt.Sprintf("block%d_out", i) {
				compare(t, name, x, loadRef(t, m, "vis_"+name))
			}
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "last_hidden", out.Last, loadRef(t, m, "vis_last_hidden"))
	if len(out.Deepstack) != len(m.Vision.DeepstackIndexes) {
		t.Fatalf("%d deepstack features, want %d", len(out.Deepstack), len(m.Vision.DeepstackIndexes))
	}
	for i, f := range out.Deepstack {
		compare(t, fmt.Sprintf("deepstack%d", i), f, loadRef(t, m, fmt.Sprintf("vis_deepstack%d", i)))
	}
	compare(t, "merged", out.Merged, loadRef(t, m, "vis_merged"))
	// The merged rows are exactly what the encoder scatters into the
	// `<|image_pad|>` slots — the dump measured that scatter at gap 0
	// against the model's own inputs_embeds, so matching here is matching
	// the tensor the text side actually reads.
	compare(t, "merged_as_scattered", out.Merged, loadRef(t, m, "edit_merged_rows"))
}

// TestNegativeControls breaks each of the four things this port could
// plausibly have got wrong and checks the bound catches it. Without them,
// "the token order is block-major", "the two GELUs differ" and "the mergers
// normalize in different places" rest on a comparison that might have been
// passing for another reason.
//
// Each runs on the two-block prefix, because every one of them is visible by
// block 0 and the whole tower costs 27x as much to say the same.
func TestNegativeControls(t *testing.T) {
	cfg, model, m := load(t, 2)
	card := loadRef(t, m, "card_vision_rgb")
	pixels, gridH, gridW, err := cfg.Patchify(card.Data, m.Size, m.Size)
	if err != nil {
		t.Fatal(err)
	}
	want := loadRef(t, m, "vis_block0_out")

	// 1. Raster token order instead of 2x2-block-major. The pixels are the
	// same bytes in a different order, which is exactly the mistake a port
	// that patchifies rastered would make.
	raster := qwen.NewMat(pixels.Rows, pixels.Cols)
	for i, rc := range blockOrder(gridH, gridW, cfg.SpatialMergeSize) {
		copy(raster.Row(rc[0]*gridW+rc[1]), pixels.Row(i))
	}
	control(t, "raster token order", model, raster, gridH, gridW, want)

	// 2. The exact-erf GELU in the block MLP, where the tanh one belongs.
	// Both are GELUs; they differ by ~1e-3 around the knee.
	for i := range model.Blocks {
		model.Blocks[i].MLP.Act = geluErf
	}
	control(t, "erf GELU in the MLP", model, pixels, gridH, gridW, want)
	for i := range model.Blocks {
		model.Blocks[i].MLP.Act = geluTanh
	}

	// 3. Adjacent-pair rotation instead of NeoX halves -- qimage/dit's
	// convention, which is one import away from this file.
	pairs := *model
	pairs.RopePairs = true
	control(t, "adjacent-pair rotation", &pairs, pixels, gridH, gridW, want)

	// 4. The deepstack mergers' norm placement on the output merger. This one
	// cannot even load -- the checkpoint's `merger.norm` is 1152 wide and a
	// post-shuffle merger wants 4608 -- so it is asserted as a load failure,
	// which is the stronger outcome and the reason the widths are checked at
	// load rather than assumed.
	set, err := safetensors.OpenSet(encoder)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	l := &loader{set: set, prefix: "model.visual."}
	l.merger("merger", cfg, true)
	if l.err == nil {
		t.Error("loading the output merger as post-shuffle succeeded; the norm widths should have refused it")
	} else {
		t.Logf("control %-24s refused at load: %v", "post-shuffle output merger", l.err)
	}
}

// control runs the two-block prefix and requires block 0's output to be far
// outside the bound.
func control(t *testing.T, what string, model *Model, pixels *qwen.Mat, gridH, gridW int, want *qwen.Mat) {
	t.Helper()
	var sumSq float64
	for _, v := range want.Data {
		sumSq += float64(v) * float64(v)
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var rel float64
	_, err := model.Forward(pixels, gridH, gridW, func(name string, x *qwen.Mat) {
		if name != "block0_out" {
			return
		}
		for i := range want.Data {
			d := math.Abs(float64(x.Data[i]) - float64(want.Data[i]))
			if r := d / math.Max(math.Abs(float64(want.Data[i])), rms); r > rel {
				rel = r
			}
		}
	})
	if err != nil {
		t.Fatalf("%s: %v", what, err)
	}
	if rel <= deepTol {
		t.Errorf("%s still matches at rel %.3g, inside the %.0e bound", what, rel, deepTol)
		return
	}
	t.Logf("control %-24s rel %.3g, %.0fx the bound", what, rel, rel/deepTol)
}

// TestFP16Ladder prices the device port this package does not have yet.
//
// The tower is the last piece of an edit still running on the CPU, and it
// costs minutes at a served condition size, so it has to move to the GPU.
// What is not obvious is at what precision. Everything else in qimage/vae is
// fp32 because the VAE's activations reach 1e5 and fp16 stops at 65504; the
// DiT and the text encoder are fp16 on the matrix cores because that is
// where their speed comes from. This tower sits between: its residual stream
// reaches absmax 1.4e4 — inside half's range, but with three decimal digits
// left at that magnitude — and Q8.2 measured the tower *amplifying* an input
// perturbation by about 1600x in this metric.
//
// So this runs the same picture twice on the CPU, once as the fp32 oracle
// and once with every linear taking fp16 operands and accumulating in fp32,
// which is what a matrix-core GEMM does. The gap it reports is the bound a
// device port could be gated at — measured before the port is written rather
// than discovered after — and the per-block walk says where it comes from.
func TestFP16Ladder(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole 27-layer tower twice")
	}
	cfg, model, m := load(t, 0)
	card := loadRef(t, m, "card_vision_rgb")
	pixels, gridH, gridW, err := cfg.Patchify(card.Data, m.Size, m.Size)
	if err != nil {
		t.Fatal(err)
	}
	fp16Ladder(t, cfg, model, pixels, gridH, gridW)

	// The *wide* card, on the tower already staged. It is not a second
	// reading of the same number: Q8.2 measured this picture amplifying fp32
	// rounding about ten times harder than the square one (the dumped fp32
	// tensors are themselves rel 8.9e-4 from float64 on it against 1.0e-4 on
	// the square card), and a device port is gated on both. Whatever this
	// says is what the grid-agnostic GPU tower should cost on it.
	if len(m.MultiSizes) == 2 {
		card2 := loadRef(t, m, "card2_vision_rgb")
		sz := m.MultiSizes[1]
		px2, gh2, gw2, err := cfg.Patchify(card2.Data, sz[0], sz[1])
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("--- the wide card, %dx%d patches", gh2, gw2)
		fp16Ladder(t, cfg, model, px2, gh2, gw2)
	}
}

func fp16Ladder(t *testing.T, cfg *Config, model *Model, pixels *qwen.Mat, gridH, gridW int) {
	t.Helper()
	blocks := map[string]*qwen.Mat{}
	fp32, err := model.Forward(pixels, gridH, gridW, func(name string, x *qwen.Mat) {
		blocks[name] = x.Clone()
	})
	if err != nil {
		t.Fatal(err)
	}

	model.SetFP16(true)
	defer model.SetFP16(false)
	var absmax float64
	half := map[string]*qwen.Mat{}
	fp16, err := model.Forward(pixels, gridH, gridW, func(name string, x *qwen.Mat) {
		half[name] = x.Clone()
		for _, v := range x.Data {
			if a := math.Abs(float64(v)); a > absmax {
				absmax = a
			}
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsInf(absmax, 0) || math.IsNaN(absmax) {
		t.Fatalf("the fp16 run overflowed: absmax %g", absmax)
	}

	for i := 0; i < cfg.Depth; i += 6 {
		name := fmt.Sprintf("block%d_out", i)
		t.Logf("%-14s rel %.3g", name, relGap(half[name], blocks[name]))
	}
	t.Logf("%-14s rel %.3g   (absmax %.4g, half's ceiling is 65504)",
		"last_hidden", relGap(fp16.Last, fp32.Last), absmax)
	t.Logf("%-14s rel %.3g   <- the bound a device port would be gated at",
		"merged", relGap(fp16.Merged, fp32.Merged))
	for i := range fp16.Deepstack {
		t.Logf("deepstack%d     rel %.3g", i, relGap(fp16.Deepstack[i], fp32.Deepstack[i]))
	}
}

// relGap is compare's metric without the assertion.
func relGap(got, want *qwen.Mat) float64 {
	var sumSq float64
	for _, v := range want.Data {
		sumSq += float64(v) * float64(v)
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
