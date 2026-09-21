package vae

import (
	"fmt"
	"math"
	"testing"

	zvae "strix-halo-vulkan/zimage/vae"
)

// The instrument the conv port owes before it exists — IMAGE.md's Q9 ledger,
// in the paragraph headed "what that port has to price first".
//
// z-image moved its whole decoder's convolutions onto the matrix cores and
// kept its fp32 gates, on the strength of a graph whose activations peak at
// 497. This one's peak at 1.1e4 (TestConvInputRanges), 20x less headroom, and
// its tightest gates are 2e-4 on the conv stages and max abs 7.3e-4 on the
// decoded image. An fp16 operand carries ~5e-4 of relative error, so the port
// moves exactly the numbers those gates are written against, and "it worked
// there" is the composition this vertical's method refuses.
//
// So this runs the *CPU* decoder and encoder with convolutions narrowing both
// operands to binary16 and accumulating in float32 — what a matrix core does
// — and reports what it costs, stage by stage, against two references: the
// dump (which is what the device path is gated on) and the fp32 CPU port
// itself (which isolates the narrowing from the fp32 noise the dump already
// carries).
//
// Two arms, because the answer differs between them:
//
//   - **3x3** is what the port narrows. Every 3x3 convolution in either graph
//     reads a channel-normed tensor or an upsample of one, which is what
//     bounds it;
//   - **all** additionally narrows the 1x1 shortcuts, which read the raw
//     residual stream. It is here because it is the obvious port, it is the
//     one z-image took, and on this model it does not merely lose precision —
//     the decoder's last shortcut reads absmax 2.6e5 and comes back NaN.
//
// It is a measurement, not a gate. The one thing it asserts is the 3x3 arm's
// finiteness, because an infinity in a fragment makes a whole output tile NaN
// and that failure would not show up as a tolerance.
func TestConvFP16Ladder(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the CPU decoder and encoder twice per arm")
	}
	m := loadManifest(t)
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Skipf("no VAE checkpoint at %s (%v)", vaeDir, err)
	}
	dec, err := LoadDecoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := LoadEncoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}

	// The up blocks' learned upsamplers are the three 3x3 convolutions that
	// read a tensor the channel norm does not bound — the block's own output,
	// at absmax 114 / 159 / 1.0e4 against the normed tensors' ~10
	// (TestConvInputRanges) — so they are the obvious suspects for where a
	// narrowing hurts, and an arm that excludes them says whether they are.
	upconv := map[*zvae.Conv2D]bool{}
	for i := range dec.Ups {
		if dec.Ups[i].Up != nil {
			upconv[&dec.Ups[i].Up.Conv] = true
		}
	}
	is3x3 := func(c *zvae.Conv2D) bool { return c.KH == 3 && c.KW == 3 }

	arms := []struct {
		name string
		pick func(c *zvae.Conv2D) bool
	}{
		{"3x3", is3x3},
		{"3x3-noup", func(c *zvae.Conv2D) bool { return is3x3(c) && !upconv[c] }},
		{"upconv", func(c *zvae.Conv2D) bool { return upconv[c] }},
		{"convin", func(c *zvae.Conv2D) bool { return c == &dec.ConvIn }},
		{"all", func(c *zvae.Conv2D) bool { return true }},
	}

	for _, arm := range arms {
		// The three attribution arms are the decoder's, and one case is enough
		// to place the damage; the two whole-graph arms run everything.
		whole := arm.name == "3x3" || arm.name == "all"
		for label := range m.Cases {
			if !whole && label != "s256" {
				continue
			}
			t.Run(arm.name+"/decode/"+label, func(t *testing.T) {
				z := loadRef(t, m, label+"_z_norm")
				cfg.Denormalize(z)

				fp32 := map[string]*zvae.Tensor{}
				dec.Tap = func(name string, x *zvae.Tensor) { fp32[name] = cloneTensor(x) }
				img32, err := dec.Decode(z)
				dec.Tap = nil
				if err != nil {
					t.Fatal(err)
				}

				zvae.FP16ConvOperands = arm.pick
				defer func() { zvae.FP16ConvOperands = nil }()
				dec.Tap = func(name string, x *zvae.Tensor) {
					report(t, arm.name, name, x, fp32[name], loadRef(t, m, label+"_dec_"+name))
				}
				img16, err := dec.Decode(z)
				dec.Tap = nil
				zvae.FP16ConvOperands = nil
				if err != nil {
					t.Fatal(err)
				}
				report(t, arm.name, "decoded", img16, img32, loadRef(t, m, label+"_decoded"))
			})

			if !whole {
				continue
			}
			t.Run(arm.name+"/encode/"+label, func(t *testing.T) {
				card := loadRef(t, m, label+"_card")

				fp32 := map[string]*zvae.Tensor{}
				enc.Tap = func(name string, x *zvae.Tensor) { fp32[name] = cloneTensor(x) }
				mode32, err := enc.Encode(card)
				enc.Tap = nil
				if err != nil {
					t.Fatal(err)
				}

				zvae.FP16ConvOperands = arm.pick
				defer func() { zvae.FP16ConvOperands = nil }()
				enc.Tap = func(name string, x *zvae.Tensor) {
					report(t, arm.name, name, x, fp32[name], loadRef(t, m, label+"_enc_"+name))
				}
				mode16, err := enc.Encode(card)
				enc.Tap = nil
				zvae.FP16ConvOperands = nil
				if err != nil {
					t.Fatal(err)
				}
				report(t, arm.name, "encoded_mode", mode16, mode32, loadRef(t, m, label+"_encoded_mode"))
			})
		}
	}
}

func cloneTensor(x *zvae.Tensor) *zvae.Tensor {
	out := zvae.NewTensor(x.N, x.C, x.H, x.W)
	copy(out.Data, x.Data)
	return out
}

// report prints one stage's distance from both references. A non-finite
// element fails the 3x3 arm and is reported by the "all" arm, which is there
// to produce exactly that.
func report(t *testing.T, arm, name string, got, vsCPU, vsRef *zvae.Tensor) {
	t.Helper()
	for _, v := range got.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			if arm == "3x3" {
				t.Fatalf("%s: fp16 operands produced %v — an overflowed fragment", name, v)
			}
			t.Logf("%-18s %-18s  OVERFLOWED: %v", name, got.String(), v)
			return
		}
	}
	t.Logf("%-18s %-18s  vs fp32 CPU %s   vs dump %s",
		name, got.String(), distance(got, vsCPU), distance(got, vsRef))
}

// distance is compare()'s two numbers without its assertion: the largest
// absolute difference, and the largest relative one against an rms floor.
func distance(got, want *zvae.Tensor) string {
	if want == nil || got.Len() != want.Len() {
		return "n/a"
	}
	var sumSq float64
	for _, w := range want.Data {
		sumSq += float64(w) * float64(w)
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var maxAbs, rel, sumAbs float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		sumAbs += d
		if d > maxAbs {
			maxAbs = d
		}
		if r := d / math.Max(math.Abs(float64(want.Data[i])), math.Max(rms, 1e-12)); r > rel {
			rel = r
		}
	}
	return fmt.Sprintf("max abs %9.3g mean %9.3g rel %8.3g", maxAbs, sumAbs/float64(len(want.Data)), rel)
}
