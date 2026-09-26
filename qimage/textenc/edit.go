package textenc

// The edit path's text encoding: everything a condition image changes about
// the prompt before the DiT sees it (IMAGE.md Q8).
//
// A generation's prompt is a flat run of text tokens at positions 0..T-1. An
// edit's is not, and the three differences are the whole of this file:
//
//  1. **The template carries image runs.** Each condition image contributes
//     `<imageN><|vision_start|>…<|vision_end|>`, and the processor expands
//     that single `<|image_pad|>` into one pad token per *merged* 2x2 patch
//     group — 64 of them for a 256x256 image. The pads are rendered here for
//     the same reason: the tokenizer must see the sequence the checkpoint was
//     trained on, not a placeholder.
//  2. **The positions are 3-D.** Text tokens take (p, p, p) off a counter,
//     but an image block lays its merged grid out as (start, start+h,
//     start+w) and advances the counter by max(H, W)/merge — so a 16x16-patch
//     image costs 8 positions, not 64. The rotary table is then *interleaved*
//     rather than blocked: frequency j reads the h axis when j%3 == 1 and the
//     w axis when j%3 == 2, below 3*section, and the t axis everywhere else.
//     Q0 measured that for a text-only prompt this collapses to the plain
//     theta-5e6 table exactly (mrope_gap 0), which is why the t2i path above
//     needs none of it.
//  3. **The tower reaches into the first three layers.** The merger's rows
//     are scattered into the image-pad slots before layer 0, and the three
//     deepstack features are *added* at those same slots after layers 0, 1
//     and 2. Nothing else carries vision content: the DiT overwrites the
//     image rows with VAE latents, so what survives is only what the
//     surrounding text tokens absorbed through attention.
//
// The vision tower itself is qimage/vision, and it is deliberately not
// imported here — a Condition is whatever produced those rows, which keeps
// this package's dependencies the tokenizer and the transformer, and lets
// the gate feed either the tower's output or the dump's.

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// ImagePad is the token every merged patch group of a condition image
// occupies in the text sequence.
const ImagePad = "<|image_pad|>"

// Grid is one condition image's patch grid, as the processor reports it in
// image_grid_thw: T frames of H x W patches, *before* the 2x2 spatial merge.
// A still image is T = 1 (the temporal patch size folds two copies of the
// same frame into one patch, so it does not appear here).
type Grid struct{ T, H, W int }

// Tokens is how many `<|image_pad|>` slots this grid occupies: the patch
// count divided by the merge square.
func (g Grid) Tokens(merge int) int { return g.T * g.H * g.W / (merge * merge) }

// Span is how far the position counter advances across this image — max(H, W)
// over the merge, not the token count. Two images of the same token count and
// different aspect ratios therefore occupy different position spans.
func (g Grid) Span(merge int) int { return max(g.H, g.W) / merge }

// MRopeSection is the checkpoint's [t, h, w] split of the rotary frequencies,
// 24/20/20 here, summing to head_dim/2.
type MRopeSection [3]int

// LoadMRope reads the text encoder's mrope configuration. It refuses a
// checkpoint that is not interleaved rather than quietly running the blocked
// layout, which produces numbers and not an error.
func LoadMRope(dir string) (MRopeSection, error) {
	var zero MRopeSection
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return zero, err
	}
	var outer struct {
		Text struct {
			RopeScaling struct {
				Section     []int `json:"mrope_section"`
				Interleaved bool  `json:"mrope_interleaved"`
			} `json:"rope_scaling"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(buf, &outer); err != nil {
		return zero, fmt.Errorf("textenc: parsing config.json: %w", err)
	}
	s := outer.Text.RopeScaling
	if len(s.Section) != 3 {
		return zero, fmt.Errorf("textenc: mrope_section is %v, want three axes", s.Section)
	}
	if !s.Interleaved {
		return zero, fmt.Errorf("textenc: mrope_interleaved is false; this port implements the interleaved layout")
	}
	return MRopeSection{s.Section[0], s.Section[1], s.Section[2]}, nil
}

// TemplateTI2I renders the edit template for n condition images, before the
// processor expands the pads — the string diffusers builds and formats.
func TemplateTI2I(prompt string, n int) string {
	pads := make([]int, n)
	for i := range pads {
		pads[i] = 1
	}
	return RenderEdit(prompt, pads)
}

// RenderEdit renders an edit prompt with each image's pad token already
// repeated the number of times the processor would repeat it.
//
// The leading space before the second and later images is the reference's,
// and it tokenizes: dropping it shifts every id after the first image.
func RenderEdit(prompt string, pads []int) string {
	if prompt == "" {
		prompt = " "
	}
	var b strings.Builder
	b.WriteString(sysRendered)
	b.WriteString("<|im_start|>user\n")
	for i, n := range pads {
		if i > 0 {
			b.WriteString(" ")
		}
		fmt.Fprintf(&b, "<image%d><|vision_start|>%s<|vision_end|>",
			i+1, strings.Repeat(ImagePad, n))
	}
	b.WriteString(prompt)
	b.WriteString("<|im_end|>\n<|im_start|>assistant\n")
	return b.String()
}

// EditPrompt is a tokenized edit prompt: the ids, where the image slots are,
// and the 3-D positions each token sits at.
type EditPrompt struct {
	IDs   []int32
	Grids []Grid
	Merge int
	PadID int32
	// Pads holds the sequence index of every image-pad token, in order, so
	// the conditions' rows scatter into them by position.
	Pads []int
	// Pos is [t, h, w] over the sequence — transformers' get_rope_index.
	Pos [3][]int32
}

// NewEditPrompt renders, tokenizes and positions an edit prompt.
//
// grids come from the image processor's own geometry (qimage/vision's
// Patchify returns them) and must be in the order the images were passed,
// which is the order their runs appear in the template.
func NewEditPrompt(tok *tokenizer.Tokenizer, prompt string, grids []Grid, merge int) (*EditPrompt, error) {
	if len(grids) == 0 {
		return nil, fmt.Errorf("textenc: an edit prompt needs at least one condition image")
	}
	padID, ok := tok.ID(ImagePad)
	if !ok {
		return nil, fmt.Errorf("textenc: the tokenizer has no %s token", ImagePad)
	}
	pads := make([]int, len(grids))
	for i, g := range grids {
		if g.T <= 0 || g.H%merge != 0 || g.W%merge != 0 {
			return nil, fmt.Errorf("textenc: image %d has grid %v, which does not merge by %d", i, g, merge)
		}
		pads[i] = g.Tokens(merge)
	}
	ids, err := tok.Encode(RenderEdit(prompt, pads))
	if err != nil {
		return nil, err
	}
	return NewPromptIDs(ids, grids, merge, padID)
}

// NewPromptIDs positions a prompt that is already tokenized with its image
// runs expanded in place, one `padID` per merged patch group: a
// presentation some other pipeline defines (MiniMax-H3's fl2va puts a
// label and a vision block per keyframe ahead of the bare prompt, with no
// template). The positions are get_rope_index's, as for an edit.
func NewPromptIDs(ids []int32, grids []Grid, merge int, padID int32) (*EditPrompt, error) {
	want := 0
	for i, g := range grids {
		if g.T <= 0 || g.H%merge != 0 || g.W%merge != 0 {
			return nil, fmt.Errorf("textenc: image %d has grid %v, which does not merge by %d", i, g, merge)
		}
		want += g.Tokens(merge)
	}
	p := &EditPrompt{IDs: ids, Grids: grids, Merge: merge, PadID: padID}
	for i, id := range ids {
		if id == padID {
			p.Pads = append(p.Pads, i)
		}
	}
	if len(p.Pads) != want {
		return nil, fmt.Errorf("textenc: tokenized %d image slots, the grids want %d", len(p.Pads), want)
	}
	var err error
	if p.Pos, err = p.positions(); err != nil {
		return nil, err
	}
	return p, nil
}

func sum(v []int) int {
	t := 0
	for _, x := range v {
		t += x
	}
	return t
}

// positions is transformers' get_rope_index over one sequence with no
// padding: runs of text advance all three axes together off a shared
// counter, and each image run lays its merged grid out in (t, h, w) raster
// order starting at the counter, which then advances by the grid's *span*.
func (p *EditPrompt) positions() ([3][]int32, error) {
	var pos [3][]int32
	for a := range pos {
		pos[a] = make([]int32, len(p.IDs))
	}
	cur, img, i := 0, 0, 0
	for i < len(p.IDs) {
		if p.IDs[i] != p.PadID {
			for ; i < len(p.IDs) && p.IDs[i] != p.PadID; i++ {
				for a := range pos {
					pos[a][i] = int32(cur)
				}
				cur++
			}
			continue
		}
		run := 0
		for j := i; j < len(p.IDs) && p.IDs[j] == p.PadID; j++ {
			run++
		}
		if img >= len(p.Grids) {
			return pos, fmt.Errorf("textenc: %d image runs for %d grids", img+1, len(p.Grids))
		}
		g := p.Grids[img]
		if run != g.Tokens(p.Merge) {
			return pos, fmt.Errorf("textenc: image %d has a run of %d pads, grid %v wants %d",
				img, run, g, g.Tokens(p.Merge))
		}
		lh, lw := g.H/p.Merge, g.W/p.Merge
		for t := 0; t < g.T; t++ {
			for h := 0; h < lh; h++ {
				for w := 0; w < lw; w++ {
					pos[0][i] = int32(cur + t)
					pos[1][i] = int32(cur + h)
					pos[2][i] = int32(cur + w)
					i++
				}
			}
		}
		cur += g.Span(p.Merge)
		img++
	}
	if img != len(p.Grids) {
		return pos, fmt.Errorf("textenc: %d image runs for %d grids", img, len(p.Grids))
	}
	return pos, nil
}

// MRope builds the interleaved multimodal rotary table for this prompt.
//
// The three axes each get the full set of head_dim/2 inverse frequencies, and
// the table is then recomposed frequency by frequency: index j reads the h
// axis when j%3 == 1 and j < 3*section[1], the w axis when j%3 == 2 and
// j < 3*section[2], and the t axis otherwise — which leaves section[0] of
// them on t, the four above 3*section[1] among them. transformers duplicates
// the result to the full head width; qwen.RoPE keeps the distinct half and
// indexes it twice, so this returns the half.
func (p *EditPrompt) MRope(headDim int, theta float64, section MRopeSection) (*qwen.RoPE, error) {
	half := headDim / 2
	if section[0]+section[1]+section[2] != half {
		return nil, fmt.Errorf("textenc: mrope section %v does not sum to %d", section, half)
	}
	T := len(p.IDs)
	r := &qwen.RoPE{HeadDim: headDim, Cos: make([]float32, T*half), Sin: make([]float32, T*half)}
	for j := 0; j < half; j++ {
		axis := 0
		switch {
		case j%3 == 1 && j < 3*section[1]:
			axis = 1
		case j%3 == 2 && j < 3*section[2]:
			axis = 2
		}
		freq := 1 / math.Pow(theta, float64(2*j)/float64(headDim))
		for t := 0; t < T; t++ {
			angle := float64(p.Pos[axis][t]) * freq
			r.Cos[t*half+j] = float32(math.Cos(angle))
			r.Sin[t*half+j] = float32(math.Sin(angle))
		}
	}
	return r, nil
}

// Condition is one condition image as the vision tower produced it: the
// merger's rows, which are scattered into that image's pad slots, and the
// deepstack features, which are added at those slots after the text model's
// first layers.
type Condition struct {
	Merged    *qwen.Mat
	Deepstack []*qwen.Mat
}

// Embeddings looks the ids up and scatters the conditions' merged rows into
// the image slots — the model's `inputs_embeds`, which the dump measured
// identical to this construction (scatter_gap 0).
func (p *EditPrompt) Embeddings(model *qwen.Model, conds []Condition) (*qwen.Mat, error) {
	x, err := model.Embeddings(p.IDs)
	if err != nil {
		return nil, err
	}
	if err := p.scatter(x, conds); err != nil {
		return nil, err
	}
	return x, nil
}

// scatter overwrites the image-pad rows of a token-embedding tensor with the
// conditions' merged rows, in the order the images appear in the template.
// It is the half of Embeddings that does not care where the lookup came from,
// which is what lets the device path (EncodeGPU) share it.
func (p *EditPrompt) scatter(x *qwen.Mat, conds []Condition) error {
	if len(conds) != len(p.Grids) {
		return fmt.Errorf("textenc: %d conditions for %d image runs", len(conds), len(p.Grids))
	}
	slot := 0
	for i, c := range conds {
		want := p.Grids[i].Tokens(p.Merge)
		if c.Merged == nil || c.Merged.Rows != want {
			return fmt.Errorf("textenc: condition %d has %v merged rows, grid %v wants %d",
				i, c.Merged, p.Grids[i], want)
		}
		if c.Merged.Cols != x.Cols {
			return fmt.Errorf("textenc: condition %d is %d wide, the model is %d", i, c.Merged.Cols, x.Cols)
		}
		for r := 0; r < c.Merged.Rows; r++ {
			copy(x.Row(p.Pads[slot]), c.Merged.Row(r))
			slot++
		}
	}
	return nil
}

// Encode runs the text encoder over the edit prompt and returns the rows the
// DiT consumes together with the image-pad mask that tells it which of them
// stand for a condition image's latents.
//
// rope is a parameter rather than built here so the gate can hand it the
// plain t2i table as a negative control: swapping the two changes numbers
// and not shapes.
func (p *EditPrompt) Encode(model *qwen.Model, rope *qwen.RoPE, conds []Condition, drop int, tr qwen.Trace) (*qwen.Mat, []bool, error) {
	x, err := p.Embeddings(model, conds)
	if err != nil {
		return nil, nil, err
	}
	// transformers' name for the post-scatter tensor, and deliberately not
	// qwen.Forward's "embeddings": that one is the raw lookup, this one has
	// the tower's rows in it.
	tr.Put("inputs_embeds", x)
	deep, err := p.deepstack(conds, x.Cols)
	if err != nil {
		return nil, nil, err
	}
	out, err := model.ForwardEmbeds(x, rope, tr, func(layer int, h *qwen.Mat) error {
		if layer >= len(deep) {
			return nil
		}
		f := deep[layer]
		for i, row := range p.Pads {
			dst, src := h.Row(row), f.Row(i)
			for c := range dst {
				dst[c] += src[c]
			}
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	embeds, err := Drop(out, drop)
	if err != nil {
		return nil, nil, err
	}
	return embeds, p.PadMask(drop), nil
}

// EncodeGPU is Encode on the device — IMAGE.md Q8.4, and the stage that makes
// a *served* edit possible: the CPU path above is minutes at a condition
// image's token count, where this is milliseconds.
//
// The three mechanisms are the same three, and each lands on one of the seams
// zimage/qwen's GPUEncoder grew for it: the scatter happens on the host
// before `UploadEmbeds` (the rows are the tower's, so the ids alone do not
// determine them), the mrope table is uploaded by `SetRoPE` instead of the
// encoder building 0..T-1 for itself, and the deepstack features are added at
// the pad slots between layers by `RunHooked` + `AddRows`. Nothing about the
// transformer changes, which is why this file and not zimage/qwen carries the
// edit.
//
// trace, when given, is called with each layer's index once that layer has
// run and *before* that layer's deepstack injection — where the reference's
// own forward hook fires — so a gate can read an intermediate off the device.
// Production passes nil.
func (p *EditPrompt) EncodeGPU(g *qwen.GPUEncoder, rope *qwen.RoPE, conds []Condition, drop int,
	trace func(layer int) error) (*qwen.Mat, []bool, error) {

	x, err := g.Embeddings(p.IDs)
	if err != nil {
		return nil, nil, err
	}
	if err := p.scatter(x, conds); err != nil {
		return nil, nil, err
	}
	deep, err := p.deepstack(conds, x.Cols)
	if err != nil {
		return nil, nil, err
	}
	if err := g.UploadEmbeds(x); err != nil {
		return nil, nil, err
	}
	if err := g.SetRoPE(rope); err != nil {
		return nil, nil, err
	}
	if err := g.RunHooked(func(layer int) error {
		if trace != nil {
			if err := trace(layer); err != nil {
				return err
			}
		}
		if layer >= len(deep) {
			return nil
		}
		return p.inject(g, deep[layer])
	}); err != nil {
		return nil, nil, err
	}
	embeds, err := Drop(g.Read(g.TensorX(), x.Cols), drop)
	if err != nil {
		return nil, nil, err
	}
	return embeds, p.PadMask(drop), nil
}

// inject adds one layer's deepstack features at the image-pad slots.
//
// The slots of one image are a contiguous run of the sequence — the template
// puts them between a `<|vision_start|>` and a `<|vision_end|>` and nothing
// else is interleaved — so this is one dispatch per image rather than one per
// row, and the contiguity is checked rather than assumed: a template change
// that broke it would otherwise add the right rows in the wrong places.
func (p *EditPrompt) inject(g *qwen.GPUEncoder, f *qwen.Mat) error {
	slot := 0
	for i, grid := range p.Grids {
		n := grid.Tokens(p.Merge)
		for r := 1; r < n; r++ {
			if p.Pads[slot+r] != p.Pads[slot]+r {
				return fmt.Errorf("textenc: image %d's pad slots are not contiguous at %d",
					i, p.Pads[slot+r])
			}
		}
		run := &qwen.Mat{Rows: n, Cols: f.Cols, Data: f.Data[slot*f.Cols : (slot+n)*f.Cols]}
		if err := g.AddRows(p.Pads[slot], run); err != nil {
			return fmt.Errorf("textenc: injecting image %d's deepstack: %w", i, err)
		}
		slot += n
	}
	return nil
}

// deepstack concatenates the conditions' features layer by layer, in pad
// order — the text model adds one tensor per layer over *all* the images'
// slots at once, so several images are one stacked matrix and not a loop.
func (p *EditPrompt) deepstack(conds []Condition, cols int) ([]*qwen.Mat, error) {
	n := len(conds[0].Deepstack)
	out := make([]*qwen.Mat, n)
	for l := 0; l < n; l++ {
		out[l] = qwen.NewMat(len(p.Pads), cols)
		slot := 0
		for i, c := range conds {
			if len(c.Deepstack) != n {
				return nil, fmt.Errorf("textenc: condition %d has %d deepstack features, condition 0 has %d",
					i, len(c.Deepstack), n)
			}
			f := c.Deepstack[l]
			if f.Rows != p.Grids[i].Tokens(p.Merge) || f.Cols != cols {
				return nil, fmt.Errorf("textenc: condition %d deepstack %d is %v", i, l, f)
			}
			for r := 0; r < f.Rows; r++ {
				copy(out[l].Row(slot), f.Row(r))
				slot++
			}
		}
	}
	return out, nil
}

// PadMask is the image_pad_mask the DiT reads: true where a row of the
// prompt embedding stands for a condition image's token, over the rows that
// survive the system-prefix drop.
func (p *EditPrompt) PadMask(drop int) []bool {
	mask := make([]bool, len(p.IDs)-drop)
	for _, i := range p.Pads {
		if i >= drop {
			mask[i-drop] = true
		}
	}
	return mask
}
