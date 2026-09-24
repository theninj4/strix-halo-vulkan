package llm

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/qimage/vision"
)

// V5's block gate: the device's attention layer at image positions against
// the CPU reference at the same positions (LLM-VISION.md V5).
//
// 64 cells: ten of text, a 6x8 image (p0 = 10, so its token (r, c) is
// (10, 10+r, 10+c)), and six more of text resuming at 18, delta -40. That is a
// real layout (cellPositions over one span), with indexer blocks whose first
// cell is an image cell, so the pooled keys are roped by an image row, and a
// q/k rotary that differs from the cell on 54 of the 64 rows. The input is
// the trace's seven rows of hc_mixed-3 tiled and scaled, so the scores are
// not degenerate.
//
// The control is the same device run on the *text* table, which must miss the
// CPU at image positions by far more than the bound. Otherwise the rows could
// be unused and the gate would still pass.
func TestAttnGPUImagePositions(t *testing.T) {
	const nTok = 64
	_, tr, c, w, _, nKV := attnFixtures(t)
	in7, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	src := in7.Vals
	rows := len(src) / c.NEmbd
	in := make([]float32, nTok*c.NEmbd)
	for i := 0; i < nTok; i++ {
		s := float32(1 + 0.1*math.Sin(float64(i)))
		for j := 0; j < c.NEmbd; j++ {
			in[i*c.NEmbd+j] = src[(i%rows)*c.NEmbd+j] * s
		}
	}
	spans := []ImageSpan{{Start: 10, End: 58, GridH: 6, GridW: 8}}
	pos := cellPositions(spans, 0, nTok)
	if pos[58] != TextPos(18) || pos[12] != (Pos3{10, 10, 12}) {
		t.Fatalf("the layout is not the one described: cell 12 %v, cell 58 %v", pos[12], pos[58])
	}

	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewAttnGPU(dev, c, nTok, nKV, []AttnWeights{w}, denseQ8Test)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)

	exact := c
	exact.Act = Exact
	cpu := AttnLayerAt(exact, w, in, nTok, nKV, pos)
	run := func() {
		t.Helper()
		if err := g.Upload(in, nTok); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
	}
	type check struct {
		name      string
		got, want []float32
		tol       float64
	}
	checks := func() []check {
		return []check{
			{"q (normed, rotated, scaled)", g.Q(), scaled(cpu.Q, attnLog2Scale(c)), 2e-3},
			{"k (normed, rotated)", g.K(), cpu.K, 2e-3},
			{"indexer q", g.IdxQ(), cpu.IdxQ, 2e-3},
			{"indexer pooled k", g.IdxK(), cpu.IdxK, 2e-3},
			{"attn_gated", g.Context(), cpu.Gated, 2e-3},
			{"attn_output", g.Out(), cpu.Out, 5e-3},
		}
	}

	// The control first, on the table as staged: text positions. It is also
	// the indexer's baseline. Its BF16 projections take a bf16 activation on
	// the device above eight rows (L4), so it sits ~3e-3 from the Exact
	// reference at *any* positions, and the claim at image positions is that
	// it gets no worse than it is on text.
	run()
	cpuText := AttnLayerAt(exact, w, in, nTok, nKV, nil)
	base := map[string]float64{}
	for _, tc := range []check{{"indexer q", g.IdxQ(), cpuText.IdxQ, 0}, {"indexer pooled k", g.IdxK(), cpuText.IdxK, 0}} {
		r, err := compare(tc.got, tc.want)
		if err != nil {
			t.Fatal(err)
		}
		base[tc.name] = r.rms
		t.Logf("baseline, text table vs CPU at text positions: %-18s %v", tc.name, r)
	}
	for _, tc := range checks()[:2] {
		r, err := compare(tc.got, tc.want)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("control, text table: %-28s %v", tc.name, r)
		if r.rms < 10*tc.tol {
			t.Errorf("control: %s on the text table is within %.0fx the bound of the image positions", tc.name, r.rms/tc.tol)
		}
	}

	if err := g.SetRopeRows(0, 0, pos); err != nil {
		t.Fatal(err)
	}
	run()
	for _, tc := range checks() {
		r, err := compare(tc.got, tc.want)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-28s vs CPU at image positions  %v", tc.name, r)
		if b, ok := base[tc.name]; ok {
			tc.tol = 1.5 * b
		}
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e against the CPU reference, over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}
}

// TestGraphImageIsAChunkSplit is V5's graph gate: TestGraphIsAChunkSplit
// with an image in the prompt. 512 tokens of the 4k trace with rows
// [100, 292) replaced by image pads under a 12x16 grid (192 cells, advance
// 16, so everything after it is roped 176 behind its cell). The prompt run
// whole has to equal it run in pieces: the spans, the rotary rows and the
// delta all carry across passes. One schedule ends in single tokens, which
// is how decode continues after an image.
//
// Two controls, which must *differ*: the same ids with no image declared
// (so the rows are read, and read as the image's), and the whole prompt
// again after the slot ran a *different* image sequence, which must be
// identical again (so a table dirtied by one sequence is repaired for the
// next: ropeHW).
func TestGraphImageIsAChunkSplit(t *testing.T) {
	const (
		layers     = 4
		nTok       = 512
		at, gh, gw = 100, 12, 16
		imgEnd     = at + gh*gw
	)
	m, tr := fixtures4k(t)
	all, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	if len(all) < nTok {
		t.Skipf("the 4k trace has %d tokens", len(all))
	}
	pc, ok, err := m.PLEConfig()
	if err != nil || !ok {
		t.Fatalf("PLE config: %v", err)
	}
	ids := append([]int32(nil), all[:nTok]...)
	for i := at; i < imgEnd; i++ {
		ids[i] = pc.Image
	}
	img := InputImage{At: at, GridH: gh, GridW: gw, Hash: 1}

	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: nTok, Layers: layers, NoHead: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	if err := g.PinSchedule(true); err != nil {
		t.Fatal(err)
	}

	whole := func() []float32 {
		t.Helper()
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		norm, err := g.HiddenExtendInput(Input{IDs: ids, Images: []InputImage{img}})
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), norm...)
	}
	want := whole()
	same := func(got []float32) (bad, first int) {
		first = -1
		for i := range got {
			if got[i] != want[i] {
				bad++
				if first < 0 {
					first = i
				}
			}
		}
		return bad, first
	}

	for _, tc := range []struct {
		name   string
		chunks []int
	}{
		{"text, the image, then text", []int{at, gh * gw, nTok - imgEnd}},
		{"text in 9s around the image", append(append(even(at, 10), gh*gw), even(nTok-imgEnd, 11)...)},
		{"the image with text on both sides, then one token at a time", append([]int{imgEnd + 20}, even(nTok-imgEnd-20, 1)...)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := g.Reset(); err != nil {
				t.Fatal(err)
			}
			var got []float32
			pos := 0
			for i, n := range tc.chunks {
				in := Input{IDs: ids[pos : pos+n]}
				if at >= pos && at < pos+n {
					im := img
					im.At = at - pos
					in.Images = []InputImage{im}
				}
				if i == len(tc.chunks)-1 {
					got, err = g.HiddenExtendInput(in)
				} else {
					err = g.AppendInputN(in, layers)
				}
				if err != nil {
					t.Fatalf("chunk %d of %d tokens at position %d: %v", i, n, pos, err)
				}
				pos += n
			}
			if bad, first := same(got); bad != 0 {
				t.Errorf("result_norm: %d of %d values differ, first at %d: %v against %v",
					bad, len(got), first, got[first], want[first])
				return
			}
			t.Logf("%d chunks: result_norm identical to the last place", len(tc.chunks))
		})
	}

	t.Run("control: the same ids with no image declared differ", func(t *testing.T) {
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		got, err := g.HiddenExtendInput(Input{IDs: ids})
		if err != nil {
			t.Fatal(err)
		}
		bad, _ := same(got)
		if bad == 0 {
			t.Fatal("declaring the image changed nothing: the rotary rows are not read")
		}
		t.Logf("%d of %d values move without the image's positions", bad, len(got))
	})

	t.Run("a table another image dirtied is repaired", func(t *testing.T) {
		// A different image sequence in the same slot: a 4x48 grid at 50,
		// which writes rows [50, 512) that are not this prompt's.
		other := append([]int32(nil), all[:nTok]...)
		for i := 50; i < 50+192; i++ {
			other[i] = pc.Image
		}
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		if _, err := g.HiddenExtendInput(Input{IDs: other, Images: []InputImage{{At: 50, GridH: 4, GridW: 48, Hash: 2}}}); err != nil {
			t.Fatal(err)
		}
		if bad, first := same(whole()); bad != 0 {
			t.Errorf("after another image: %d values differ, first at %d", bad, first)
		}
		// And text-only after it must be text-only's own answer.
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		plain, err := g.HiddenExtendInput(Input{IDs: all[:nTok]})
		if err != nil {
			t.Fatal(err)
		}
		plain = append([]float32(nil), plain...)
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		if _, err := g.HiddenExtendInput(Input{IDs: other, Images: []InputImage{{At: 50, GridH: 4, GridW: 48, Hash: 2}}}); err != nil {
			t.Fatal(err)
		}
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		again, err := g.HiddenExtendInput(Input{IDs: all[:nTok]})
		if err != nil {
			t.Fatal(err)
		}
		for i := range plain {
			if plain[i] != again[i] {
				t.Fatalf("a text prompt after an image sequence differs from itself at %d", i)
			}
		}
		t.Log("an image's rows are repaired for the next image and for text")
	})
}

// TestGraphImageDecode is V5's gate on the two decode paths that bypass
// appendIn: the recorded one-token step (P1c) and the batched rows (C5).
//
// Three slots: two different image prompts (their rotary rows differ at the
// same cells) and a text one. Each is prefilled and stepped greedily through
// Extend, which records its step and replays it from the second token. Then:
//
//   - DecodeRows over all three must be each slot's solo step, bit for bit.
//     A table shared between slots, or a row written into the wrong slot's,
//     would make the image rows read each other's angles;
//   - slot 0's prompt and its decoded tokens, run whole in one pass, must
//     give the last step's logits bit for bit: the recorded step read the
//     same rotary rows as a pass that wrote them. This one is pinned
//     (PinSchedule), the batched one is not, as their text twins are.
func TestGraphImageDecode(t *testing.T) {
	const (
		layers = 4
		nTok   = 400
		steps  = 6
	)
	m, tr := fixtures4k(t)
	all, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	pc, ok, err := m.PLEConfig()
	if err != nil || !ok {
		t.Fatalf("PLE config: %v", err)
	}
	withImage := func(from, at, gh, gw int, hash uint64) Input {
		ids := append([]int32(nil), all[from:from+nTok]...)
		for i := at; i < at+gh*gw; i++ {
			ids[i] = pc.Image
		}
		return Input{IDs: ids, Images: []InputImage{{At: at, GridH: gh, GridW: gw, Hash: hash}}}
	}
	prompts := []Input{
		withImage(0, 40, 12, 16, 1),
		{IDs: all[500 : 500+nTok]},
		withImage(1000, 120, 8, 20, 2),
	}

	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: 512, NKV: 1024, Layers: layers, Slots: 3})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	argmax := func(l []float32) int32 {
		id := int32(0)
		for j := range l {
			if l[j] > l[id] {
				id = int32(j)
			}
		}
		return id
	}
	prefill := func(s int) []float32 {
		t.Helper()
		if err := g.UseSlot(s); err != nil {
			t.Fatal(err)
		}
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		l, _, err := g.ExtendInput(prompts[s])
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), l...)
	}

	toks := make([][]int32, 3)
	want := make([][][]float32, 3)
	for pass := 0; pass < 2; pass++ { // the first only warms the arenas
		for s := range 3 {
			l := prefill(s)
			toks[s], want[s] = nil, nil
			for range steps {
				id := argmax(l)
				toks[s] = append(toks[s], id)
				if l, _, err = g.Extend([]int32{id}); err != nil {
					t.Fatal(err)
				}
				want[s] = append(want[s], append([]float32(nil), l...))
			}
			if !g.Prerecorded() {
				t.Fatalf("slot %d decoded without the recorded step", s)
			}
		}
	}
	vocab := len(want[0][0])

	t.Run("batched rows are each slot's solo step", func(t *testing.T) {
		for s := range 3 {
			prefill(s)
		}
		slots := []int{0, 1, 2}
		for i := range steps {
			in := []int32{toks[0][i], toks[1][i], toks[2][i]}
			l, err := g.DecodeRows(slots, in)
			if err != nil {
				t.Fatal(err)
			}
			for r, s := range slots {
				row := l[r*vocab : (r+1)*vocab]
				for j := range row {
					if row[j] != want[s][i][j] {
						t.Fatalf("step %d, slot %d: logit %d is %v batched and %v solo", i, s, j, row[j], want[s][i][j])
					}
				}
			}
		}
		t.Logf("%d steps of 3 rows, two of them image sequences: bit-identical to the solo steps", steps)
	})

	t.Run("decoding after an image is the prompt run whole", func(t *testing.T) {
		// Pinned, as TestGraphIsAChunkSplit is: a one-token step and a
		// many-row pass are bit-identical only on one schedule (the decode
		// GEMV splits K). So the steps are re-run pinned first.
		if err := g.PinSchedule(true); err != nil {
			t.Fatal(err)
		}
		defer g.PinSchedule(false)
		l := prefill(0)
		for i := range steps {
			if l, _, err = g.Extend([]int32{toks[0][i]}); err != nil {
				t.Fatal(err)
			}
		}
		last := append([]float32(nil), l...)
		in := prompts[0]
		in.IDs = append(append([]int32(nil), in.IDs...), toks[0][:steps]...)
		if err := g.UseSlot(0); err != nil {
			t.Fatal(err)
		}
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		whole, _, err := g.ExtendInput(in)
		if err != nil {
			t.Fatal(err)
		}
		for j := range whole {
			if whole[j] != last[j] {
				t.Fatalf("logit %d: %v whole, %v decoded a token at a time", j, whole[j], last[j])
			}
		}
		t.Logf("%d tokens after a %d-cell image: the recorded steps are the whole pass", steps, 12*16)
	})
}

// TestGraphRestoreKnowsItsImages: a checkpoint restores into a slot that
// still holds its image, and is refused by one that holds a different image
// under identical ids. The ids at an image are all the pad token, so without
// the spans' hashes the second would look like the same conversation.
func TestGraphRestoreKnowsItsImages(t *testing.T) {
	m, tr := fixtures4k(t)
	all, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	pc, ok, err := m.PLEConfig()
	if err != nil || !ok {
		t.Fatalf("PLE config: %v", err)
	}
	ids := append([]int32(nil), all[:200]...)
	for i := 20; i < 20+64; i++ {
		ids[i] = pc.Image
	}
	image := func(hash uint64) Input {
		return Input{IDs: ids, Images: []InputImage{{At: 20, GridH: 8, GridW: 8, Hash: hash}}}
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: 256, Layers: 4, NoHead: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)

	if err := g.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := g.AppendInputN(image(1), 4); err != nil {
		t.Fatal(err)
	}
	cp, err := g.Checkpoint(nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AppendN(all[300:310], 4); err != nil {
		t.Fatal(err)
	}
	if err := g.Restore(cp); err != nil {
		t.Fatalf("the slot still holds the checkpoint's image and refused it: %v", err)
	}
	if g.Past() != 200 {
		t.Fatalf("restored to %d, want 200", g.Past())
	}

	if err := g.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := g.AppendInputN(image(2), 4); err != nil {
		t.Fatal(err)
	}
	if err := g.AppendN(all[300:310], 4); err != nil {
		t.Fatal(err)
	}
	if err := g.Restore(cp); err == nil {
		t.Fatal("a checkpoint restored into a slot holding a different image under the same ids")
	}
	// And a rewind into the middle of an image is refused.
	if err := g.rewindPosition(50, 50); err == nil {
		t.Fatal("a rewind to the middle of an image was accepted")
	}
	t.Log("same image: restored; different image, same ids: refused; a cut inside an image: refused")
}

// The V6 oracle: reference/eval_dump_mtmd.c over a prompt holding the barn
// photo, llama.cpp's own tower and M-RoPE, dumped per decode.
const visionEvalDir = "../reference/out/llmvision_eval"

// visionTextDir is the same length of text through eval_dump, the drift
// baseline TestGraphImageLogits reads its bounds from.
const visionTextDir = "../reference/out/llmvision_text281"

// visionEval reads the oracle: the trace, the ids, and the image's place,
// grid and embedding rows (mtmd's own, so the comparison is the language
// model's alone).
func visionEval(t *testing.T) (*Model, *Trace, Input) {
	t.Helper()
	tr, err := OpenTrace(visionEvalDir)
	if err != nil {
		t.Skipf("no image oracle in %s (%v); see reference/eval_dump_mtmd.c", visionEvalDir, err)
	}
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	meta, err := os.ReadFile(filepath.Join(visionEvalDir, "image.txt"))
	if err != nil {
		t.Fatal(err)
	}
	var at, nx, ny, n, width int
	if _, err := fmt.Sscan(string(meta), &at, &nx, &ny, &n, &width); err != nil || n != nx*ny {
		t.Fatalf("image.txt: %q (%v)", meta, err)
	}
	raw, err := os.ReadFile(filepath.Join(visionEvalDir, "image_embd.bin"))
	if err != nil {
		t.Fatal(err)
	}
	embd := make([]float32, len(raw)/4)
	for i := range embd {
		embd[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	if len(embd) != n*width {
		t.Fatalf("image_embd.bin holds %d values, want %d x %d", len(embd), n, width)
	}
	if _, err := os.Stat(modelDir); err != nil {
		t.Skipf("checkpoint absent: %v", err)
	}
	m, err := Open(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m, tr, Input{IDs: ids, Images: []InputImage{{At: at, GridH: ny, GridW: nx, Hash: 1, Embd: embd}}}
}

// traceRows is a tensor's occurrences concatenated: mtmd decodes the text
// before the image, the image and the text after it as three calls.
func traceRows(t *testing.T, tr *Trace, name string) []float32 {
	t.Helper()
	var out []float32
	for i := 0; ; i++ {
		d, err := tr.Get(name, i)
		if err != nil {
			if i == 0 {
				t.Fatal(err)
			}
			return out
		}
		out = append(out, d.Vals...)
	}
}

// TestGraphImagePrefix is V6's gate at the 4-layer prefix: the graph over
// the oracle's prompt, with mtmd's image rows at mtmd's cells, against
// llama.cpp's residual after every layer, held to TestGraphPrefix's bound.
// The controls are the two things this stage adds: without the image's
// rows (the pad's own embedding there), and with the rows but a transposed
// grid (the same cells, other positions).
func TestGraphImagePrefix(t *testing.T) {
	const layers = 4
	m, tr, in := visionEval(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: len(in.IDs), Layers: layers, NoHead: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)

	run := func(in Input, n int) []float32 {
		t.Helper()
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		if err := g.AppendInputN(in, n); err != nil {
			t.Fatal(err)
		}
		return g.Residual()
	}
	var last cmpResult
	for n := 1; n <= layers; n++ {
		want := traceRows(t, tr, fmt.Sprintf("l_last-%d", n-1))
		r, err := compare(run(in, n), want)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("l_last-%d over %d cells (a %dx%d image at %d): %v  (%.3f%% of scale)",
			n-1, len(in.IDs), in.Images[0].GridH, in.Images[0].GridW, in.Images[0].At, r, 100*r.rms/r.refMax)
		last = r
	}
	if last.rms > 5e-3 {
		t.Errorf("l_last-%d: rms %.3e over 5e-3 (%v)", layers-1, last.rms, last)
	}

	want := traceRows(t, tr, fmt.Sprintf("l_last-%d", layers-1))
	noRows := in
	noRows.Images = []InputImage{in.Images[0]}
	noRows.Images[0].Embd = nil
	im := in.Images[0]
	transposed := in
	transposed.Images = []InputImage{{At: im.At, GridH: im.GridH * im.GridW, GridW: 1, Hash: 1, Embd: im.Embd}}
	asText := in
	asText.Images = []InputImage{in.Images[0]}
	asText.Images[0].textPositions = true
	// Positions reach the residual through one attention layer of four,
	// on 64 of its 256 head dims, so their controls sit at a small multiple
	// of the gate here. The whole model (TestGraphImageLogits) is where they
	// compound over twelve.
	for _, c := range []struct {
		name   string
		in     Input
		margin float64
	}{
		{"the pad's embedding in place of the image's rows", noRows, 100},
		{"the image's rows under a transposed grid", transposed, 1.5},
		{"the image's rows at text positions (vLLM's rule)", asText, 1.5},
	} {
		r, err := compare(run(c.in, layers), want)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("control, %s: %v (%.1fx the gate)", c.name, r, r.rms/last.rms)
		if r.rms < c.margin*last.rms {
			t.Errorf("control, %s: rms %.3e is within %.1fx the gate's %.3e", c.name, r.rms, c.margin, last.rms)
		}
	}
}

// TestGraphImageLogits is V6's whole-model gate: the oracle's image prompt
// through all 48 layers and the head, against llama.cpp's last-row logits,
// held to TestGraphLogits's bounds (result_norm within 1% of scale, logits
// within 5%, the argmax and the top ten). vLLM's text-position rule runs
// beside it as the control: where positions compound over twelve attention
// layers, it has to land measurably further from the reference.
func TestGraphImageLogits(t *testing.T) {
	if testing.Short() {
		t.Skip("the whole model is 85 GB resident")
	}
	m, tr, in := visionEval(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: len(in.IDs)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)

	run := func(in Input) ([]float32, []float32) {
		t.Helper()
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		l, norm, err := g.ExtendInput(in)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), l...), append([]float32(nil), norm...)
	}
	logits, norm := run(in)
	// The last decode's: only the call that ends the prompt asks for logits.
	last := func(name string) []float32 {
		var d *Dump
		for i := 0; ; i++ {
			next, err := tr.Get(name, i)
			if err != nil {
				break
			}
			d = next
		}
		if d == nil || len(d.Vals) == 0 {
			t.Fatalf("the oracle has no %s", name)
		}
		return d.Vals
	}
	wantNorm, wantLog := last("result_norm"), last("result_output")

	// The baseline: a text prompt of the same 281 cells through the same
	// staged graph against its own llama.cpp run
	// (reference/out/llmvision_text281, eval_dump -nt 281). TestGraphLogits's
	// bounds were measured on seven tokens, and drift grows with the
	// sequence. So the claim is that an image prompt drifts no more than
	// text does at its length.
	baseNorm, baseLog := 1e-2, 5e-2
	if bt, err := OpenTrace(visionTextDir); err != nil {
		t.Logf("no text baseline (%v); holding to TestGraphLogits's seven-token bounds", err)
	} else {
		bids, _, err := bt.Tokens()
		if err != nil {
			t.Fatal(err)
		}
		bl, bn := run(Input{IDs: bids})
		wn, err := bt.Get("result_norm", 0)
		if err != nil {
			t.Fatal(err)
		}
		wl, err := bt.Get("result_output", 0)
		if err != nil {
			t.Fatal(err)
		}
		n1, _ := compare(bn, wn.Vals)
		l1, _ := compare(bl, wl.Vals)
		t.Logf("text baseline, %d cells: result_norm %.3f%% of scale, result_output %.3f%%, argmax %d vs %d",
			len(bids), 100*n1.rms/n1.refMax, 100*l1.rms/l1.refMax, topLogits(bl, 1)[0], topLogits(wl.Vals, 1)[0])
		// Image tokens route on near-ties 3-5x as often as text
		// (TestImageRoutingTies), and every flip moves its row a few
		// percent, so the image prompt gets twice TestGraphLogits's bound on
		// the final norm rather than the text baseline's: measured 1.19%
		// against 0.34% for text of its length. The logits keep 5%, and
		// the decision below is held as tightly as text's.
		baseNorm = max(2e-2, 1.5*n1.rms/n1.refMax)
		baseLog = max(5e-2, 1.5*l1.rms/l1.refMax)
	}
	rn, err := compare(norm, wantNorm)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("result_norm: %v  (%.3f%% of scale, bound %.3f%%)", rn, 100*rn.rms/rn.refMax, 100*baseNorm)
	if rn.rms/rn.refMax > baseNorm {
		t.Errorf("result_norm: rms is %.3f%% of scale, over %.3f%%", 100*rn.rms/rn.refMax, 100*baseNorm)
	}
	rl, err := compare(logits, wantLog)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("result_output over %d logits: %v  (%.3f%% of scale, bound %.3f%%)", len(logits), rl, 100*rl.rms/rl.refMax, 100*baseLog)
	if rl.rms/rl.refMax > baseLog {
		t.Errorf("result_output: rms is %.3f%% of scale, over %.3f%%", 100*rl.rms/rl.refMax, 100*baseLog)
	}
	got, want := topLogits(logits, 10), topLogits(wantLog, 10)
	t.Logf("argmax: ours %d (%.4f), llama.cpp %d (%.4f)", got[0], logits[got[0]], want[0], wantLog[want[0]])
	t.Logf("top 10: ours      %v", got)
	t.Logf("        llama.cpp %v", want)
	if got[0] != want[0] {
		gap := float64(wantLog[want[0]] - wantLog[got[0]])
		if denseQ8Test && gap >= 0 && gap <= 3*rl.rms {
			t.Logf("argmax: a near-tie, %.4f apart against drift rms %.4f", gap, rl.rms)
		} else {
			t.Errorf("argmax %d, llama.cpp says %d (gap %.4f, drift rms %.4f)", got[0], want[0], gap, rl.rms)
		}
	}
	if n := sameSet(got, want); n < 8 {
		t.Errorf("top 10: %d of %d tokens in common", n, len(want))
	}

	asText := in
	asText.Images = []InputImage{in.Images[0]}
	asText.Images[0].textPositions = true
	cl, _ := run(asText)
	rc, err := compare(cl, wantLog)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("control, vLLM's text positions: result_output %v (%.2fx ours); argmax %d",
		rc, rc.rms/rl.rms, topLogits(cl, 1)[0])
	if rc.rms <= rl.rms {
		t.Errorf("control: text positions land nearer the reference than image positions (%.3e against %.3e)", rc.rms, rl.rms)
	}
}

// TestGraphImageDepth is the drift curve behind TestGraphImageLogits: our
// residual against llama.cpp's every four layers, for the image prompt and a
// text prompt of the same length, and for the image prompt its image rows and
// its text rows apart. It asserts nothing; it says where the two part.
func TestGraphImageDepth(t *testing.T) {
	if testing.Short() {
		t.Skip("the whole model is 85 GB resident")
	}
	imgTr, err := OpenTrace(visionEvalDir + "_depth")
	if err != nil {
		t.Skipf("no depth oracle (%v)", err)
	}
	txtTr, err := OpenTrace(visionTextDir + "_depth")
	if err != nil {
		t.Skipf("no depth oracle (%v)", err)
	}
	m, _, in := visionEval(t)
	tids, _, err := txtTr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: len(in.IDs), NoHead: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	wide := g.cfg.HCConfig().Wide()
	im := in.Images[0]
	lo, hi := im.At*wide, (im.At+im.GridH*im.GridW)*wide

	pct := func(got, want []float32) float64 {
		r, err := compare(got, want)
		if err != nil {
			t.Fatal(err)
		}
		return 100 * r.rms / r.refMax
	}
	t.Logf("%-6s %10s %10s %10s %10s", "layers", "text", "image", "img rows", "text rows")
	for _, n := range []int{4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47} {
		name := fmt.Sprintf("l_last-%d", n-1)
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		if err := g.AppendInputN(in, n); err != nil {
			t.Fatal(err)
		}
		gi := append([]float32(nil), g.Residual()...)
		wi := traceRows(t, imgTr, name)
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		if err := g.AppendN(tids, n); err != nil {
			t.Fatal(err)
		}
		wt, err := txtTr.Get(name, 0)
		if err != nil {
			t.Fatal(err)
		}
		textP := pct(g.Residual(), wt.Vals)
		imgP := pct(gi, wi)
		rowsP := pct(gi[lo:hi], wi[lo:hi])
		restP := pct(append(append([]float32(nil), gi[:lo]...), gi[hi:]...), append(append([]float32(nil), wi[:lo]...), wi[hi:]...))
		t.Logf("%-6d %9.3f%% %9.3f%% %9.3f%% %9.3f%%", n, textP, imgP, rowsP, restP)
	}
}

// TestImageRoutingTies explains TestGraphImageLogits's drift, and holds the
// explanation. An image prompt drifts from llama.cpp ~4x as far as text of
// its length (result_norm 1.19% against 0.34% at 281 cells), and
// TestGraphImageDepth put all of it on a handful of *image* rows that jump
// to ~5% after layer 0 alone, before any attention layer, while the image
// prompt's text rows drift exactly as text does.
//
// Those rows are MoE routing flips: one of a row's ten experts differs from
// llama.cpp's, always at a near-tie in the reference's own router. The flips
// are not ours to fix, because **our router is more accurate on image rows
// than on text** (rms-rel ~1e-3 against ~2.6e-3). Image tokens route on much
// tighter margins: the gap between the 10th and 11th expert has a median of
// 0.019 against text's 0.054, and a 10th percentile of 0.002 against 0.010.
// Any two implementations flip some of them, HF's bf16 and llama.cpp's
// included.
//
// Asserted: every row whose expert set differs is at a reference gap under
// 3e-2, and the router's error on image rows is no worse than 1.5x its error
// on text rows.
func TestImageRoutingTies(t *testing.T) {
	m, _, in := visionEval(t)
	tr, err := OpenTrace(visionEvalDir + "_moe")
	if err != nil {
		t.Skipf("no router oracle (%v); eval_dump_mtmd -n '^(ffn_moe_topk-0|ffn_moe_logits-0)$'", err)
	}
	var want []int32
	var ref []float32
	for i := 0; ; i++ {
		d, err := tr.Get("ffn_moe_topk-0", i)
		if err != nil {
			break
		}
		l, err := tr.Get("ffn_moe_logits-0", i)
		if err != nil {
			t.Fatal(err)
		}
		want, ref = append(want, d.Ints...), append(ref, l.Vals...)
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: len(in.IDs), Layers: 1, NoHead: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	if err := g.AppendInputN(in, 1); err != nil {
		t.Fatal(err)
	}
	got, ours := g.moe.TopK(), g.moe.Logits()
	n := len(in.IDs)
	nE, stride := len(ref)/n, len(ours)/g.moe.Tokens()
	used := len(want) / n
	im := in.Images[0]
	isImage := func(r int) bool { return r >= im.At && r < im.At+im.GridH*im.GridW }

	var err2 [2][2]float64 // [image?][diff, ref]
	var flips [2]int
	for r := 0; r < n; r++ {
		k := 0
		if isImage(r) {
			k = 1
		}
		for e := 0; e < nE; e++ {
			a, b := float64(ours[r*stride+e]), float64(ref[r*nE+e])
			err2[k][0] += (a - b) * (a - b)
			err2[k][1] += b * b
		}
		set := map[int32]bool{}
		for j := 0; j < used; j++ {
			set[want[r*used+j]] = true
		}
		differs := false
		for j := 0; j < used; j++ {
			differs = differs || !set[got[r*used+j]]
		}
		if !differs {
			continue
		}
		flips[k]++
		row := append([]float32(nil), ref[r*nE:(r+1)*nE]...)
		sort.Slice(row, func(a, b int) bool { return row[a] > row[b] })
		if gap := row[used-1] - row[used]; gap > 3e-2 {
			t.Errorf("row %d: an expert differs where the reference's 10th and 11th are %.3e apart", r, gap)
		}
	}
	textRel := math.Sqrt(err2[0][0] / err2[0][1])
	imgRel := math.Sqrt(err2[1][0] / err2[1][1])
	t.Logf("layer 0: %d of %d image rows and %d of %d text rows route one expert differently, all at near-ties",
		flips[1], im.GridH*im.GridW, flips[0], n-im.GridH*im.GridW)
	t.Logf("router logits vs llama.cpp: image rows rms-rel %.2e, text rows %.2e", imgRel, textRel)
	if imgRel > 1.5*textRel {
		t.Errorf("the router is %.1fx less accurate on image rows than on text", imgRel/textRel)
	}
}

// TestGraphImageOwnTower is V6's end-to-end claim: the barn photo through
// this repository's own decoder, processor and device tower (not mtmd's),
// then the whole model, against llama.cpp's logits for its own pipeline.
//
// The two pipelines see different pixels: stb_image and llama.cpp's resize
// against libjpeg's upsampling rules and torch's bicubic (llm/pixels). V3
// measured that this ViT turns invisible pixel differences into ~0.02-0.1 rms
// on its rows. So the rows are measured rather than gated, and the claim is
// the model's: the same argmax and a top ten mostly in common.
func TestGraphImageOwnTower(t *testing.T) {
	if testing.Short() {
		t.Skip("the whole model is 85 GB resident")
	}
	m, tr, in := visionEval(t)
	data, err := os.ReadFile("../reference/out/llmvision/barn.jpg")
	if err != nil {
		t.Skipf("no barn.jpg (%v)", err)
	}
	img, _, err := pixels.Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	planes, h, w, err := pixels.Default.Planes(img)
	if err != nil {
		t.Fatal(err)
	}
	_, tower, err := vision.LoadGGUF("../models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf", 0)
	if err != nil {
		t.Skipf("no mmproj (%v)", err)
	}
	rows, gh, gw, err := tower.Cfg.Patchify(planes, h, w)
	if err != nil {
		t.Fatal(err)
	}
	im := in.Images[0]
	if gh/2 != im.GridH || gw/2 != im.GridW {
		t.Fatalf("our grid is %dx%d merged, llama.cpp's %dx%d: the prompts would differ", gh/2, gw/2, im.GridH, im.GridW)
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	vg, err := vision.NewGPU(dev, tower, gh*gw)
	if err != nil {
		t.Fatal(err)
	}
	out, err := vg.Forward(t.Context(), rows, gh, gw)
	vg.Destroy()
	if err != nil {
		t.Fatal(err)
	}
	ours := out.Merged.Data
	r, err := compare(ours, im.Embd)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("our tower's %d rows against mtmd's: %v (rms-relative %.3f)", gh*gw/4, r, r.rms/rmsOf(im.Embd))

	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: len(in.IDs)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	mine := in
	mine.Images = []InputImage{im}
	mine.Images[0].Embd = ours
	logits, _, err := g.ExtendInput(mine)
	if err != nil {
		t.Fatal(err)
	}
	var want []float32
	for i := 0; ; i++ {
		d, err := tr.Get("result_output", i)
		if err != nil {
			break
		}
		want = d.Vals
	}
	rl, _ := compare(logits, want)
	got, ref := topLogits(logits, 10), topLogits(want, 10)
	t.Logf("result_output: %v (%.3f%% of scale)", rl, 100*rl.rms/rl.refMax)
	t.Logf("argmax: ours %d, llama.cpp %d; top 10 in common %d", got[0], ref[0], sameSet(got, ref))
	if got[0] != ref[0] {
		t.Errorf("our pipeline's answer starts with %d, llama.cpp's with %d", got[0], ref[0])
	}
	if n := sameSet(got, ref); n < 7 {
		t.Errorf("top 10: %d in common", n)
	}
}

func rmsOf(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s / float64(len(v)))
}
