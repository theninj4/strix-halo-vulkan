package dit

import (
	"fmt"
	"slices"
	"testing"

	"strix-halo-vulkan/safetensors"
)

// Stage 4c's tests. What is new here is not arithmetic -- a block is the same
// 18 dispatches it was in 4b -- but *addressing*: 34 blocks share one set of
// pipelines and one pair of activation arenas, their 12.0 GB of projection
// weights are spread over three storage buffers because one buffer on this
// device holds 4.29 GB, and every block is reached by an offset plus a bank.
// Every way of getting that wrong produces a plausible tensor, so each test
// below is paired with a control that breaks exactly that.
//
// The reference is reference/dump_dit_stack.py: the real modules with the
// real weights, applied in sequence. Regenerate with
//
//	.venv/bin/python reference/dump_dit_stack.py --seq 320 --caption 64

// oneBlockBank is a bank size that holds exactly one of this model's blocks,
// which is how a six-block test reaches the six-bank case a 34-block stack
// only reaches at 4.29 GB.
func oneBlockBank(dim, ffn int) int { return 2 * (4*dim*dim + 3*dim*ffn) }

// modelWidths reads the two widths the bank arithmetic needs out of the
// checkpoint's header -- dim from the config, the FFN's from a weight's
// shape, since config.json does not carry it.
func modelWidths(t *testing.T) (dim, ffn int) {
	t.Helper()
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Skipf("no transformer checkpoint at %s (%v)", transformer, err)
	}
	set, err := safetensors.OpenSet(transformer)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	w1, err := set.Get("layers.0.feed_forward.w1.weight")
	if err != nil {
		t.Fatal(err)
	}
	return cfg.Dim, w1.Shape[0]
}

// loadStack builds a stack of the named checkpoint blocks. bankBytes caps one
// bank; pass maxBankBytes for the arrangement the real model gets.
func loadStack(t *testing.T, prefixes []string, tokens int, bankBytes int, ctl blockControls) (*GPUStack, func()) {
	t.Helper()
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Skipf("no transformer checkpoint at %s (%v)", transformer, err)
	}
	set, err := safetensors.OpenSet(transformer)
	if err != nil {
		t.Fatal(err)
	}
	m := loadManifest(t, refStackDir)
	rope := ropeFromRef(t, m)
	specs := make([]blockSpec, len(prefixes))
	for i, p := range prefixes {
		specs[i] = specFor(p)
	}
	dev, done := newTestDevice(t)
	g, err := newStack(dev, set, cfg, specs, rope, tokens, nil, ctl, bankBytes)
	if err != nil {
		set.Close()
		done()
		t.Fatal(err)
	}
	return g, func() {
		g.Destroy()
		set.Close()
		done()
	}
}

// TestGPUStackAgainstDiffusers is the stage's end-to-end check: six real
// blocks, each with its own weights, applied in sequence, against diffusers
// doing the same thing.
//
// It is checked twice over, and the two are different questions. Each block
// fed the reference's own input says *which* block is wrong, the way RunTo
// does inside a block, and holds each to the single-block bound. Then the
// whole stack in one go, where the residual stream never leaves the device
// and 108 dispatches run back to back over arenas that are never re-zeroed,
// says the chaining itself is right -- at a looser bound, because six fp16
// blocks feed each other.
//
// Two of the six are context refiners, which diffusers builds with
// modulation=False: no adaLN projection, no scale on either branch input and
// both residuals ungated. They are the reason the dump interleaves the kinds
// rather than using six layers.
func TestGPUStackAgainstDiffusers(t *testing.T) {
	m := loadManifest(t, refStackDir)
	if len(m.Blocks) == 0 {
		t.Skip("the reference dump names no blocks; regenerate it")
	}
	g, done := loadStack(t, m.Blocks, m.Seq, maxBankBytes, blockControls{})
	defer done()
	t.Logf("stack: %v", g.Blocks())

	x := loadRef(t, m, "x")
	adaln := loadRef(t, m, "adaln_input")

	// Each block on its own, fed the reference's input rather than the
	// stack's: this is the stagewise walk RunTo does inside a block, done
	// across blocks, so a block whose weights or modulation are wrong shows
	// up as one row out of line instead of as a chain that drifts.
	for i := range m.Blocks {
		in := x
		if i > 0 {
			in = loadRef(t, m, fmt.Sprintf("block%d", i-1))
		}
		out, err := g.Apply(in, adaln.Data, []int{i})
		if err != nil {
			t.Fatal(err)
		}
		compareTol(t, fmt.Sprintf("block%d %s", i, m.Blocks[i]), out,
			loadRef(t, m, fmt.Sprintf("block%d", i)), fp16GEMMRelTol)
	}

	// And the chain, where the residual stream never leaves the device.
	out, err := g.Apply(x, adaln.Data, nil)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "stack out", out, loadRef(t, m, "out"), fp16StackRelTol)
}

// fp16StackRelTol is what six chained fp16 blocks are held to, against the
// 8e-2 one block gets (fp16GEMMRelTol).
//
// It is looser for the reason the block's was looser than attention's: each
// block's output is the next one's input, so the narrowing compounds. What
// says that is all it is doing is the test above, which runs each of the six
// blocks on the *reference's* input rather than the stack's: in isolation
// they measure 5.1e-2, 1.5e-2, 7.6e-2, 2.6e-2, 1.5e-3 and 6.2e-3 against the
// same reference, every one of them inside the single-block bound, while the
// chain of them lands at 1.8e-1. Nothing is wrong with any block; six of them
// in a row is five extra roundings of an input. (The spread across the six is
// the RMS normalisation again rather than the blocks: the two largest are the
// ones whose output carries outliers ninety times its RMS.)
//
// Two other things pin this number rather than leaving it as an assertion
// about fp16. TestStackMatchesCPU runs the identical chain in fp32 and lands
// at 4.2e-4, so the block order, the unmodulated variant and the weight
// addressing are right independently of the narrowing. And the three controls
// below -- all of them addressing mistakes, not arithmetic ones -- land at
// 263x, 351x and 496x this bound.
const fp16StackRelTol = 4e-1

// stackF32RelTol is the fp32 CPU chain's bound, against the 2e-4 a single
// fp32 stage gets (relTol).
//
// The measured figure is 4.2e-4 and it is not really drift: the worst element
// is 1.3e-3 off a value of 83, i.e. 1.6e-5 of itself, on a tensor whose RMS is
// 3.1. It looks larger than relTol only because this project normalises by
// the tensor's RMS, and a stack's activations grow outliers twenty-five times
// their RMS by the sixth block. 2e-3 is five times what six blocks measure.
const stackF32RelTol = 2e-3

// TestGPUStackDetectsErrors is the negative control, and without it the test
// above proves only that a stack runs.
//
// All three breakages are addressing, because that is what stage 4c adds.
// Each leaves every kernel, every shape and every weight in the arena exactly
// as they were and changes only which block reads which.
func TestGPUStackDetectsErrors(t *testing.T) {
	m := loadManifest(t, refStackDir)
	if len(m.Blocks) < 2 {
		t.Skip("the reference dump names fewer than two blocks")
	}
	ref := loadRef(t, m, "out")
	x := loadRef(t, m, "x")
	adaln := loadRef(t, m, "adaln_input")

	perBank := oneBlockBank(modelWidths(t))

	reversed := slices.Clone(m.Blocks)
	slices.Reverse(reversed)
	first := make([]string, len(m.Blocks))
	for i := range first {
		first[i] = m.Blocks[0]
	}

	for _, c := range []struct {
		name      string
		blocks    []string
		bankBytes int
		ctl       blockControls
	}{
		// The order of the weights, with everything else identical.
		{"blocks run in reverse order", reversed, maxBankBytes, blockControls{}},
		// One block's weights used for all of them: the mistake a shared
		// pipeline set invites, since nothing but the push constant says
		// which block a dispatch is.
		{"every block reads block 0's weights", first, maxBankBytes, blockControls{}},
		// One block per bank, and every block reading the *next* bank at its
		// own offsets. This is the one only stage 4c can make: the bank is in
		// the descriptor set rather than the push constants, so a stack that
		// picked the pipeline by the wrong index would look exactly like this.
		{"every block reads the next bank", m.Blocks, perBank, blockControls{shiftBank: true}},
	} {
		t.Run(c.name, func(t *testing.T) {
			g, done := loadStack(t, c.blocks, m.Seq, c.bankBytes, c.ctl)
			defer done()
			out, err := g.Apply(x, adaln.Data, nil)
			if err != nil {
				t.Fatal(err)
			}
			_, rms, rel, _ := deviation(out, ref)
			if rel <= fp16StackRelTol {
				t.Errorf("%s: rel %.3g is inside the %.0e bound -- the check cannot see this break",
					c.name, rel, fp16StackRelTol)
				return
			}
			t.Logf("%-34s rel %.3g = %.0fx the bound (rms %.4g)", c.name, rel, rel/fp16StackRelTol, rms)
		})
	}
}

// TestGPUStackBanksAgree is the stage's central claim: splitting the weight
// arena across storage buffers changes nothing.
//
// The same six blocks are built twice -- once in one bank, once one block per
// bank -- and the two have to agree *exactly*, not within a tolerance. They
// run the same kernels over the same numbers in the same order; all that
// differs is which VkBuffer a descriptor set points at, so any difference at
// all is a bug rather than rounding.
func TestGPUStackBanksAgree(t *testing.T) {
	m := loadManifest(t, refStackDir)
	if len(m.Blocks) < 2 {
		t.Skip("the reference dump names fewer than two blocks")
	}
	x := loadRef(t, m, "x")
	adaln := loadRef(t, m, "adaln_input")

	one, done := loadStack(t, m.Blocks, m.Seq, maxBankBytes, blockControls{})
	perBank := oneBlockBank(one.Dim(), one.FFN())
	wantBanks, wantByBlock := one.Banks()
	want, err := one.Apply(x, adaln.Data, nil)
	done()
	if err != nil {
		t.Fatal(err)
	}
	if len(wantBanks) != 1 {
		t.Fatalf("%d blocks at 4.29 GB a bank came to %d banks, want 1", len(m.Blocks), len(wantBanks))
	}
	t.Logf("one bank:  %d MB, blocks %v", wantBanks[0]>>20, wantByBlock)

	split, done := loadStack(t, m.Blocks, m.Seq, perBank, blockControls{})
	gotBanks, gotByBlock := split.Banks()
	got, err := split.Apply(x, adaln.Data, nil)
	done()
	if err != nil {
		t.Fatal(err)
	}
	if len(gotBanks) != len(m.Blocks) {
		t.Fatalf("one block per bank came to %d banks for %d blocks", len(gotBanks), len(m.Blocks))
	}
	t.Logf("per block: %d banks of %d MB, blocks %v", len(gotBanks), gotBanks[0]>>20, gotByBlock)

	for i := range want.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("element %d: %v across %d banks, %v in one -- the split is not free",
				i, got.Data[i], len(gotBanks), want.Data[i])
		}
	}
	t.Logf("%d elements identical across %d banks and one", len(want.Data), len(gotBanks))
}

// TestStackBankPlan checks the layout the real 34-block model gets, without a
// device and without reading 12 GB of weights: planBanks is arithmetic over
// the config, which is the whole point of it being a separate step.
//
// What has to hold is what the graph assumes. Every bank is inside the 4.29
// GB a storage buffer addresses here; every block is in exactly one bank; and
// within a bank the seven projections of every block occupy disjoint spans
// that fit. Getting the last one wrong is how a stack silently reads a
// neighbour's weights.
func TestStackBankPlan(t *testing.T) {
	const (
		dim, ffn, heads = 3840, 10240, 30
		adaIn, tokens   = 256, 4096
	)
	cfg := &Config{Dim: dim, NHeads: heads, NLayers: 30, NRefiner: 2}
	g := &GPUStack{
		dim: dim, ffn: ffn, heads: heads, headDim: dim / heads,
		adaIn: adaIn, tokens: tokens, plan: DefaultGEMMPlan(),
	}
	specs := StackBlocks(cfg)
	if len(specs) != 34 {
		t.Fatalf("the DiT is %d blocks, want 34", len(specs))
	}
	total32, bankElems, err := g.planBanks(specs, maxBankBytes)
	if err != nil {
		t.Fatal(err)
	}

	var total16 int
	for b, n := range bankElems {
		if n*2 > maxBankBytes {
			t.Errorf("bank %d is %d bytes, past the %d a storage buffer addresses", b, n*2, maxBankBytes)
		}
		total16 += n
	}
	t.Logf("34 blocks: %.2f GB of fp16 weights in %d banks, %.2f GB of fp32, %.2f GB in all",
		float64(total16*2)/1e9, len(bankElems), float64(total32*4)/1e9,
		float64(total16*2+total32*4)/1e9)
	for b, n := range bankElems {
		var held []int
		for i, w := range g.w {
			if w.bank == b {
				held = append(held, i)
			}
		}
		t.Logf("  bank %d: %.2f GB, %d blocks %v", b, float64(n*2)/1e9, len(held), held)
	}

	// Every block in exactly one bank, and its projections disjoint inside it.
	type span struct{ lo, hi int }
	occupied := make(map[int][]span)
	shapes := g.projShapes()
	for i, w := range g.w {
		if w.bank < 0 || w.bank >= len(bankElems) {
			t.Fatalf("block %d names bank %d of %d", i, w.bank, len(bankElems))
		}
		for _, r := range projOrder {
			v, _ := variantFor(g.plan[r])
			n := bElems(shapes[r][0], shapes[r][1], v.layout)
			s := span{int(w.bOff[r]), int(w.bOff[r]) + n}
			if s.hi > bankElems[w.bank] {
				t.Fatalf("block %d %s runs to %d in a bank of %d", i, r, s.hi, bankElems[w.bank])
			}
			for _, o := range occupied[w.bank] {
				if s.lo < o.hi && o.lo < s.hi {
					t.Fatalf("block %d %s [%d %d) overlaps [%d %d) in bank %d",
						i, r, s.lo, s.hi, o.lo, o.hi, w.bank)
				}
			}
			occupied[w.bank] = append(occupied[w.bank], s)
		}
	}

	// The two context refiners are the unmodulated ones, and they are the
	// only blocks with no adaLN projection in the fp32 arena.
	var unmod []string
	for _, w := range g.w {
		if !w.modulated {
			unmod = append(unmod, w.prefix)
		}
	}
	want := []string{"context_refiner.0", "context_refiner.1"}
	if !slices.Equal(unmod, want) {
		t.Errorf("unmodulated blocks %v, want %v", unmod, want)
	}
}

// TestStackMatchesCPU chains the CPU blocks over the same reference input.
//
// It answers a question the GPU stack cannot answer about itself: the fp32
// CPU implementation carries none of the fp16 narrowing, so if it reproduces
// the diffusers chain to the 2e-4 every fp32 stage is held to, the *wiring*
// of a stack -- the block order, the unmodulated variant, whose weights go
// where -- is right, and whatever the GPU stack's larger error is, it is
// arithmetic rather than addressing.
func TestStackMatchesCPU(t *testing.T) {
	m := loadManifest(t, refStackDir)
	if len(m.Blocks) == 0 {
		t.Skip("the reference dump names no blocks; regenerate it")
	}
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Skipf("no transformer checkpoint at %s (%v)", transformer, err)
	}
	set, err := safetensors.OpenSet(transformer)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	rope := ropeFromRef(t, m)
	adaln := loadRef(t, m, "adaln_input")

	h := loadRef(t, m, "x")
	for i, prefix := range m.Blocks {
		blk, err := LoadBlock(set, prefix, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := blk.AdaLN != nil, specFor(prefix).modulated; got != want {
			t.Fatalf("%s: adaLN present = %v, but the stack calls it modulated = %v", prefix, got, want)
		}
		h, err = blk.Apply(h, adaln.Data, rope)
		if err != nil {
			t.Fatal(err)
		}
		compareTol(t, fmt.Sprintf("cpu block%d %s", i, prefix), h,
			loadRef(t, m, fmt.Sprintf("block%d", i)), stackF32RelTol)
	}
}
