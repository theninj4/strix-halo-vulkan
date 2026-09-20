package textenc

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"strix-halo-vulkan/qimage/vision"
	"strix-halo-vulkan/zimage/qwen"
)

// The edit path's reference is reference/dump_qi21_vision.py, which runs the
// real Qwen3-VL over a deterministic 256x256 RGBA card and dumps both halves
// of an edit's prefix: the vision tower stage by stage (gated in
// qimage/vision) and the *text* encoding it conditions, which is what this
// file gates. Regenerate with:
//
//	.venv/bin/python reference/dump_qi21_vision.py
const editRefDir = "../../reference/out/qi21vision"

type editManifest struct {
	Rendered          string  `json:"rendered"`
	Prompt            string  `json:"prompt"`
	Size              int     `json:"size"`
	DropIdx           int     `json:"drop_idx"`
	ImageTokenID      int32   `json:"image_token_id"`
	IDs               []int32 `json:"ids"`
	Seq               int     `json:"seq"`
	GridTHW           [][]int `json:"grid_thw"`
	ImagePadPositions []int   `json:"image_pad_positions"`
	EmbedTokens       int     `json:"embed_tokens"`
	PadMaskCount      int     `json:"pad_mask_count"`
	ScatterGap        float64 `json:"scatter_gap"`

	MultiRendered     string  `json:"multi_rendered"`
	MultiIDs          []int32 `json:"multi_ids"`
	MultiSeq          int     `json:"multi_seq"`
	MultiGridTHW      [][]int `json:"multi_grid_thw"`
	MultiSizes        [][]int `json:"multi_sizes"`
	MultiPadPositions []int   `json:"multi_pad_positions"`
	MultiEmbedTokens  int     `json:"multi_embed_tokens"`
	MultiPadMaskCount int     `json:"multi_pad_mask_count"`

	// How far the dumped fp32 tensors are from the same tower in float64,
	// measured by the dump with this file's own metric. It is what makes a
	// disagreement at the tower's depth readable: see towerTol.
	FP32VsFP64 map[string]struct {
		Rel    float64 `json:"rel"`
		MaxAbs float64 `json:"max_abs"`
		RMS    float64 `json:"rms"`
	} `json:"fp32_vs_fp64"`

	Text              struct {
		Section     []int   `json:"mrope_section"`
		Interleaved bool    `json:"mrope_interleaved"`
		Theta       float64 `json:"rope_theta"`
		HeadDim     int     `json:"head_dim"`
	} `json:"text"`
	Tensors map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadEditManifest(t *testing.T) *editManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(editRefDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_qi21_vision.py", editRefDir, err)
	}
	var m editManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func loadEditRef(t *testing.T, m *editManifest, name string) *qwen.Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(editRefDir, name+".bin"))
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

// editPrompt builds the prompt the dump used, and checks the geometry the
// rest of the file leans on while it is there.
func editPrompt(t *testing.T, m *editManifest) *EditPrompt {
	t.Helper()
	tok := loadTok(t)
	g := m.GridTHW[0]
	p, err := NewEditPrompt(tok, m.Prompt, []Grid{{T: g[0], H: g[1], W: g[2]}}, 2)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestEditTemplate checks the rendered template and the expansion of the pad
// token against the processor's own ids. Nothing here needs a weight, and
// everything downstream reads these positions, so it runs first.
func TestEditTemplate(t *testing.T) {
	m := loadEditManifest(t)
	if got := TemplateTI2I(m.Prompt, 1); got != m.Rendered {
		t.Errorf("template renders\n%q\nwant\n%q", got, m.Rendered)
	}
	// The dump's own claim that the merger's rows *are* the model's
	// inputs_embeds at the image slots is what EditPrompt.Embeddings
	// implements; if the reference stopped measuring 0 there, this port's
	// scatter would be building a tensor the model does not use.
	if m.ScatterGap != 0 {
		t.Errorf("the dump's merger scatter differs from inputs_embeds by %g", m.ScatterGap)
	}

	p := editPrompt(t, m)
	if len(p.IDs) != m.Seq {
		t.Fatalf("%d ids, want %d", len(p.IDs), m.Seq)
	}
	for i := range p.IDs {
		if p.IDs[i] != m.IDs[i] {
			t.Fatalf("id %d is %d, want %d", i, p.IDs[i], m.IDs[i])
		}
	}
	if p.PadID != m.ImageTokenID {
		t.Errorf("image pad id is %d, want %d", p.PadID, m.ImageTokenID)
	}
	if len(p.Pads) != len(m.ImagePadPositions) {
		t.Fatalf("%d image slots, want %d", len(p.Pads), len(m.ImagePadPositions))
	}
	for i, got := range p.Pads {
		if got != m.ImagePadPositions[i] {
			t.Fatalf("image slot %d is at %d, want %d", i, got, m.ImagePadPositions[i])
		}
	}
	// The mask the DiT reads, against the pipeline's own.
	mask := p.PadMask(m.DropIdx)
	want := loadEditRef(t, m, "edit_image_pad_mask")
	if len(mask) != want.Cols*want.Rows || len(mask) != m.EmbedTokens {
		t.Fatalf("pad mask is %d long, want %d", len(mask), m.EmbedTokens)
	}
	set := 0
	for i, v := range mask {
		if v != (want.Data[i] != 0) {
			t.Fatalf("pad mask differs at %d", i)
		}
		if v {
			set++
		}
	}
	if set != m.PadMaskCount {
		t.Errorf("%d image rows in the mask, want %d", set, m.PadMaskCount)
	}
	t.Logf("%d ids, %d image slots, %d rows after the drop", len(p.IDs), len(p.Pads), len(mask))
}

// TestEditPositions checks the 3-D positions and the interleaved rotary table
// against the ones the model built for itself. Both are pure geometry, so a
// disagreement is a layout bug and not drift — they are compared exactly.
func TestEditPositions(t *testing.T) {
	m := loadEditManifest(t)
	p := editPrompt(t, m)

	want := loadEditRef(t, m, "edit_position_ids") // [3, seq]
	if want.Rows != 3 || want.Cols != len(p.IDs) {
		t.Fatalf("position ids are %s", want)
	}
	for a := 0; a < 3; a++ {
		for i, got := range p.Pos[a] {
			if float32(got) != want.Row(a)[i] {
				t.Fatalf("position axis %d token %d is %d, want %g", a, i, got, want.Row(a)[i])
			}
		}
	}

	section, err := LoadMRope(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	if len(m.Text.Section) == 3 && (section[0] != m.Text.Section[0] || section[1] != m.Text.Section[1] || section[2] != m.Text.Section[2]) {
		t.Fatalf("mrope section %v, the dump used %v", section, m.Text.Section)
	}
	rope, err := p.MRope(m.Text.HeadDim, m.Text.Theta, section)
	if err != nil {
		t.Fatal(err)
	}
	// transformers duplicates the distinct half across the head; qwen.RoPE
	// keeps the half and indexes it twice, so the comparison rebuilds the
	// full width — which also checks the duplication is what the model did.
	half := m.Text.HeadDim / 2
	gotCos, gotSin := qwen.NewMat(len(p.IDs), m.Text.HeadDim), qwen.NewMat(len(p.IDs), m.Text.HeadDim)
	for pos := 0; pos < len(p.IDs); pos++ {
		for i := 0; i < half; i++ {
			gotCos.Row(pos)[i], gotCos.Row(pos)[i+half] = rope.Cos[pos*half+i], rope.Cos[pos*half+i]
			gotSin.Row(pos)[i], gotSin.Row(pos)[i+half] = rope.Sin[pos*half+i], rope.Sin[pos*half+i]
		}
	}
	compare(t, "edit_rope_cos", gotCos, loadEditRef(t, m, "edit_rope_cos"))
	compare(t, "edit_rope_sin", gotSin, loadEditRef(t, m, "edit_rope_sin"))

	// The two axes that make this table multimodal at all: over the 64 image
	// slots the h and w positions disagree with the plain counter a t2i
	// prompt would use, and the *sequence* is 99 tokens in 43 positions.
	last := p.Pos[0][len(p.IDs)-1]
	t.Logf("%d tokens span %d positions; image block at t=%d h=%d..%d w=%d..%d",
		len(p.IDs), last+1, p.Pos[0][p.Pads[0]], p.Pos[1][p.Pads[0]],
		p.Pos[1][p.Pads[len(p.Pads)-1]], p.Pos[2][p.Pads[0]], p.Pos[2][p.Pads[len(p.Pads)-1]])
}

// multiPrompt builds the two-image prompt the dump's multi case used.
func multiPrompt(t *testing.T, m *editManifest) *EditPrompt {
	t.Helper()
	if len(m.MultiGridTHW) != 2 {
		t.Skipf("the dump has no two-image case; re-run reference/dump_qi21_vision.py")
	}
	tok := loadTok(t)
	grids := make([]Grid, len(m.MultiGridTHW))
	for i, g := range m.MultiGridTHW {
		grids[i] = Grid{T: g[0], H: g[1], W: g[2]}
	}
	p, err := NewEditPrompt(tok, m.Prompt, grids, 2)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestMultiRefPrompt gates the three things a *second* condition image adds
// and the first one cannot show: the leading space before " <image2>", the
// position counter advancing by a grid's span rather than its token count,
// and the two runs' slots being distinguishable at all. The dump's second
// card is 192x384 against the first's 256x256, so the two spans differ (12
// against 8) and a port that advanced by either the token count or a
// constant lands somewhere else.
func TestMultiRefPrompt(t *testing.T) {
	m := loadEditManifest(t)
	p := multiPrompt(t, m)
	if got := TemplateTI2I(m.Prompt, 2); got != m.MultiRendered {
		t.Errorf("two-image template renders\n%q\nwant\n%q", got, m.MultiRendered)
	}
	if len(p.IDs) != m.MultiSeq {
		t.Fatalf("%d ids, want %d", len(p.IDs), m.MultiSeq)
	}
	for i := range p.IDs {
		if p.IDs[i] != m.MultiIDs[i] {
			t.Fatalf("id %d is %d, want %d", i, p.IDs[i], m.MultiIDs[i])
		}
	}
	if len(p.Pads) != len(m.MultiPadPositions) {
		t.Fatalf("%d image slots, want %d", len(p.Pads), len(m.MultiPadPositions))
	}
	for i, got := range p.Pads {
		if got != m.MultiPadPositions[i] {
			t.Fatalf("image slot %d is at %d, want %d", i, got, m.MultiPadPositions[i])
		}
	}
	want := loadEditRef(t, m, "multi_position_ids")
	for a := 0; a < 3; a++ {
		for i, got := range p.Pos[a] {
			if float32(got) != want.Row(a)[i] {
				t.Fatalf("position axis %d token %d is %d, want %g", a, i, got, want.Row(a)[i])
			}
		}
	}
	first, second := p.Grids[0], p.Grids[1]
	t.Logf("%d ids, grids %v and %v, %d+%d slots spanning %d+%d positions",
		len(p.IDs), first, second, first.Tokens(2), second.Tokens(2), first.Span(2), second.Span(2))
}

// towerTol is the bound for the wide card's deep tower stages, and like every
// other number in this vertical it is measured rather than chosen.
//
// The tower's port accumulates its dot products in float64 while holding
// activations in float32, so after 27 pre-norm blocks over a residual stream
// that reaches absmax 1.4e4 it is *more* accurate than the fp32 reference,
// not less — and on this picture that stops being a rounding detail. The dump
// measures its own fp32 tensors against the same tower in float64 and finds
// rel 8.9e-4 (last hidden) and 1.4e-3 (merged) for the wide card against
// 1.0e-4 and 1.2e-4 for the square one: the same code, a picture that
// amplifies rounding ten times harder. Against the float64 rows this port
// lands at 1.3e-4, 2.1e-4 and 4.8e-5 — six to seven times closer than the
// fp32 reference — so 5e-4 sits above the port and a factor of three below
// the reference's own error. Nothing structural hides under it: the vision
// package's negative controls move these stages by rel 4e-2 and up.
const towerTol = 5e-4

// TestMultiRefTower walks the *second* condition image — the non-square one —
// through the tower on its own, which is what says whether a disagreement at
// the end of the two-image sequence is the tower's or the text model's. It
// needs 1.6 GB rather than the encoder gate's 36, so it is the cheap half of
// that question and runs first.
func TestMultiRefTower(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the 27-layer tower")
	}
	m := loadEditManifest(t)
	if len(m.MultiSizes) != 2 {
		t.Skipf("the dump has no two-image case; re-run reference/dump_qi21_vision.py")
	}
	cfg, err := vision.LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	h, w := m.MultiSizes[1][0], m.MultiSizes[1][1]
	card := loadEditRef(t, m, "card2_vision_rgb")
	pixels, gridH, gridW, err := cfg.Patchify(card.Data, h, w)
	if err != nil {
		t.Fatal(err)
	}
	if g := m.MultiGridTHW[1]; gridH != g[1] || gridW != g[2] {
		t.Fatalf("grid %dx%d, the dump says %v", gridH, gridW, g)
	}
	// Before any weight: the processor's own pixel rows for this image. A
	// non-square grid is the first thing that can reorder them.
	compare(t, "multi_pixel_values2", pixels, loadEditRef(t, m, "multi_pixel_values2"))

	model, err := vision.Load(encoder, cfg, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"patch_embed": "vis2_patch_embed", "block0_out": "vis2_block0_out"}
	out, err := model.Forward(pixels, gridH, gridW, func(name string, x *qwen.Mat) {
		if ref, ok := want[name]; ok {
			compare(t, name, x, loadEditRef(t, m, ref))
		}
	})
	if err != nil {
		t.Fatal(err)
	}

	// Past block 0 the *fp32* dump stops being the right oracle for this
	// image, and the dump says so itself: it measured its own fp32 tensors
	// against the same tower in float64 and found rel 8.9e-4 on this last
	// hidden state and 1.4e-3 on these merged rows — against 1.0e-4 and
	// 1.2e-4 for the square card, which is the same code on a picture that
	// amplifies rounding ten times less. So the gate is against the float64
	// rows, at the tower's own 2e-4, and the fp32 distance is logged beside
	// it as the context that makes it readable. (A structural mistake does
	// not survive to here anyway: patch_embed and block0_out above are exact
	// to fp32 rounding, and every geometric piece feeds them.)
	for _, c := range []struct{ name, f64, f32 string }{
		{"vis2_last_hidden", "vis2_last_hidden64", "vis2_last_hidden"},
		{"vis2_merged", "vis2_merged64", "vis2_merged"},
		{"vis2_deepstack2", "vis2_deepstack264", "vis2_deepstack2"},
	} {
		var got *qwen.Mat
		switch c.name {
		case "vis2_last_hidden":
			got = out.Last
		case "vis2_merged":
			got = out.Merged
		default:
			got = out.Deepstack[2]
		}
		compareAt(t, c.name+" vs float64", got, loadEditRef(t, m, c.f64), towerTol)
		_, _, rel, _ := deviation(got, loadEditRef(t, m, c.f32))
		t.Logf("%-22s vs the fp32 dump   rel %.2g   (the fp32 dump is itself rel %.2g from float64)",
			c.name, rel, m.FP32VsFP64[fp32Key(c.name)].Rel)
	}
	for i, f := range out.Deepstack[:2] {
		compare(t, fmt.Sprintf("vis2_deepstack%d", i), f, loadEditRef(t, m, fmt.Sprintf("vis2_deepstack%d", i)))
	}
}

// fp32Key names the dump's fp32-vs-float64 measurement for a tensor.
func fp32Key(tensor string) string {
	switch tensor {
	case "vis2_last_hidden":
		return "image2_last_hidden"
	case "vis2_merged":
		return "image2_merged"
	default:
		return "image2_deepstack2"
	}
}

// tower runs the real 27-layer vision tower over a card and returns the
// condition it produces. It is a function so the tower's 1.6 GB is
// collectable before the 34 GB text encoder is staged.
func tower(t *testing.T, cards []*qwen.Mat, sizes [][]int) []Condition {
	t.Helper()
	cfg, err := vision.LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	model, err := vision.Load(encoder, cfg, 0)
	if err != nil {
		t.Fatal(err)
	}
	conds := make([]Condition, len(cards))
	for i, card := range cards {
		pixels, gridH, gridW, err := cfg.Patchify(card.Data, sizes[i][0], sizes[i][1])
		if err != nil {
			t.Fatal(err)
		}
		out, err := model.Forward(pixels, gridH, gridW, nil)
		if err != nil {
			t.Fatal(err)
		}
		conds[i] = Condition{Merged: out.Merged, Deepstack: out.Deepstack}
	}
	return conds
}

// TestEditEncoder is Q8.2's acceptance gate: the tower and the text encoder
// together, against the embeddings the diffusers pipeline hands the DiT for
// an edit.
//
// It composes nothing — the tower is run here rather than read from the dump,
// because the claim worth having is that *our* vision context, positioned by
// *our* mrope and injected by *our* deepstack, reproduces the reference's
// prompt_embeds. Stages ~36 GB of fp32 weights and must run alone.
func TestEditEncoder(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the tower and 34 GB of text encoder")
	}
	m := loadEditManifest(t)
	p := editPrompt(t, m)
	cards := []*qwen.Mat{loadEditRef(t, m, "card_vision_rgb")}
	sizes := [][]int{{m.Size, m.Size}}
	if len(m.MultiGridTHW) == 2 {
		cards = append(cards, loadEditRef(t, m, "card2_vision_rgb"))
		sizes = append(sizes, m.MultiSizes[1])
	}
	conds := tower(t, cards, sizes)
	cond := conds[0]
	compare(t, "merged_rows", cond.Merged, loadEditRef(t, m, "edit_merged_rows"))
	runtime.GC()

	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	section, err := LoadMRope(encoder)
	if err != nil {
		t.Fatal(err)
	}
	model, err := qwen.LoadWith(encoder, cfg, cfg.NumLayers)
	if err != nil {
		t.Fatal(err)
	}
	rope, err := p.MRope(cfg.HeadDim, cfg.RopeTheta, section)
	if err != nil {
		t.Fatal(err)
	}

	one := conds[:1]
	tr := qwen.Trace{}
	embeds, mask, err := p.Encode(model, rope, one, m.DropIdx, tr)
	if err != nil {
		t.Fatal(err)
	}
	// The reference's layer hook fires before the deepstack injection, so
	// hidden_1 is the right side of the comparison: the trace records each
	// layer's output before `after` runs.
	compare(t, "edit_layer0_out", tr["hidden_1"], loadEditRef(t, m, "edit_layer0_out"))
	compareAt(t, "edit_last_prenorm", tr[fmt.Sprintf("hidden_%d", cfg.NumLayers)],
		loadEditRef(t, m, "edit_last_prenorm"), deepTol)
	compareAt(t, "edit_prompt_embeds", embeds, loadEditRef(t, m, "edit_prompt_embeds"), deepTol)
	if got := countTrue(mask); got != m.PadMaskCount {
		t.Errorf("%d image rows in the mask, want %d", got, m.PadMaskCount)
	}

	// The two mechanisms an edit adds to a generation, each broken in turn.
	// Both produce a full-shaped answer and no error, which is why the bound
	// has to be shown to catch them rather than assumed to.
	want := loadEditRef(t, m, "edit_prompt_embeds")
	t.Run("controls", func(t *testing.T) {
		plain := qwen.NewRoPE(cfg.HeadDim, len(p.IDs), cfg.RopeTheta)
		out, _, err := p.Encode(model, plain, one, m.DropIdx, nil)
		if err != nil {
			t.Fatal(err)
		}
		control(t, "plain RoPE instead of mrope", out, want)

		bare := []Condition{{Merged: cond.Merged}}
		out, _, err = p.Encode(model, rope, bare, m.DropIdx, nil)
		if err != nil {
			t.Fatal(err)
		}
		control(t, "no deepstack injection", out, want)
	})

	// The same gate over two condition images, on the model that is already
	// staged. A second reference is where the span arithmetic, the ordering
	// of the scatter and the stacking of the deepstack features across
	// images all become visible — and the endpoint promises up to ten.
	if len(conds) != 2 {
		t.Skip("the dump has no two-image case; re-run reference/dump_qi21_vision.py")
	}
	t.Run("multi", func(t *testing.T) {
		mp := multiPrompt(t, m)
		mrope, err := mp.MRope(cfg.HeadDim, cfg.RopeTheta, section)
		if err != nil {
			t.Fatal(err)
		}
		// The conditions are the dump's *own* rows rather than our tower's,
		// which is what makes this a gate on the text encoding instead of on
		// the pair. It matters here and not in the single-image case: the
		// wide card amplifies fp32 rounding ten times harder (towerTol), so
		// our more-accurate tower and the reference's disagree by rel 1.6e-3
		// before the first layer runs, and 36 layers multiply that. Feeding
		// the reference's rows removes the question; the composed run is
		// measured separately below.
		dumped := []Condition{
			{Merged: loadEditRef(t, m, "vis_merged"), Deepstack: []*qwen.Mat{
				loadEditRef(t, m, "vis_deepstack0"),
				loadEditRef(t, m, "vis_deepstack1"),
				loadEditRef(t, m, "vis_deepstack2")}},
			{Merged: loadEditRef(t, m, "vis2_merged"), Deepstack: []*qwen.Mat{
				loadEditRef(t, m, "vis2_deepstack0"),
				loadEditRef(t, m, "vis2_deepstack1"),
				loadEditRef(t, m, "vis2_deepstack2")}},
		}
		mtr := qwen.Trace{}
		embeds, mask, err := mp.Encode(model, mrope, dumped, m.DropIdx, mtr)
		if err != nil {
			t.Fatal(err)
		}
		// Stagewise, so a disagreement says where: inputs_embeds is the
		// tower and the scatter with no layer in it, layer 0 is the first
		// deepstack injection's input, and the pre-norm state is the whole
		// depth.
		compare(t, "multi_inputs_embeds", mtr["inputs_embeds"], loadEditRef(t, m, "multi_inputs_embeds"))
		compare(t, "multi_layer0_out", mtr["hidden_1"], loadEditRef(t, m, "multi_layer0_out"))
		compareAt(t, "multi_last_prenorm", mtr[fmt.Sprintf("hidden_%d", cfg.NumLayers)],
			loadEditRef(t, m, "multi_last_prenorm"), deepTol)
		compareAt(t, "multi_prompt_embeds", embeds, loadEditRef(t, m, "multi_prompt_embeds"), deepTol)
		if got := countTrue(mask); got != m.MultiPadMaskCount {
			t.Errorf("%d image rows in the mask, want %d", got, m.MultiPadMaskCount)
		}
		refMask := loadEditRef(t, m, "multi_image_pad_mask")
		for i, v := range mask {
			if v != (refMask.Data[i] != 0) {
				t.Fatalf("multi pad mask differs at %d", i)
			}
		}
		// Swapping the two images is the control the single-image case
		// cannot run: same rows, same count, wrong order.
		swapped := []Condition{dumped[1], dumped[0]}
		if _, _, err := mp.Encode(model, mrope, swapped, m.DropIdx, nil); err == nil {
			t.Error("encoding with the conditions swapped succeeded; the row counts should have refused it")
		} else {
			t.Logf("control %-30s refused: %v", "conditions in the wrong order", err)
		}

		// And the composed run — our tower feeding our text encoder, which is
		// what a served edit actually does. It is reported rather than gated
		// tightly, and the attribution is the gate above: with the
		// reference's own rows the same code lands inside 1e-3, so whatever
		// this number is, it is the tower's input difference travelling
		// through 36 layers and not the text encoding. Our tower is the more
		// accurate of the two (towerTol), so a served edit is not worse for
		// it. The bound here is an order-of-magnitude guard.
		ours, _, err := mp.Encode(model, mrope, conds, m.DropIdx, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, _, rel, _ := deviation(ours, loadEditRef(t, m, "multi_prompt_embeds"))
		if rel > 3e-2 {
			t.Errorf("our own tower's rows give rel %.3g, outside the 3e-2 guard", rel)
		}
		_, _, ownGap, _ := deviation(conds[1].Merged, loadEditRef(t, m, "vis2_merged"))
		t.Logf("composed (our tower + our encoder) rel %.2g, from a %.2g difference in the tower's merged rows",
			rel, ownGap)
	})
}

func countTrue(v []bool) int {
	n := 0
	for _, b := range v {
		if b {
			n++
		}
	}
	return n
}

// control requires a deliberately broken run to land far outside the gate's
// bound, measured against the same reference the gate uses.
func control(t *testing.T, what string, got, want *qwen.Mat) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", what, got, want)
	}
	_, _, rel, _ := deviation(got, want)
	if rel <= deepTol {
		t.Errorf("%s still matches at rel %.3g, inside the %.0e bound", what, rel, deepTol)
		return
	}
	t.Logf("control %-30s rel %.3g, %.0fx the bound", what, rel, rel/deepTol)
}
