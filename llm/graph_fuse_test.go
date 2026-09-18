package llm

// P1a's gate: the fused combine+norm is the pair, to the last place.
//
//	go test ./llm/ -v -run TestHCFusionIsTheCombineThenTheNorm   # the kernel, ~1 s
//	go test ./llm/ -v -run TestHCFusionRunsThePass               # 4 layers, ~4 s
//
// The fusion is a *scheduling* change and not an arithmetic one, and these
// are what say so. `llm_hc_cn.comp` runs one mixer's scatter and the next
// mixer's grouped RMSNorm in a single workgroup per (token, stream), keeping
// the combined residual in registers instead of going back to the arena — but
// it keeps the norm's 256-thread partition and its 256-way tree, because a
// wider workgroup would be a different sum over 2560 squares.
//
// A tolerance would not do here. The pair and the fusion read the same
// weights in the same order; if they disagree at all it is because a value
// took a different path, and 1e-7 of that is as much a defect as 1e-2. The
// first version of this kernel failed by exactly that much — one ulp on 23%
// of the residual, because a second reader of the combined value let the
// backend contract the scatter into a fused multiply-add. See the `precise`
// note in the shader.

import "testing"

// identical reports how far two tensors are from being the same bits.
func identical(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values fused, %d as the pair", name, len(got), len(want))
	}
	bad, first := 0, -1
	for i := range got {
		if got[i] != want[i] {
			bad++
			if first < 0 {
				first = i
			}
		}
	}
	if bad != 0 {
		t.Errorf("%s: %d of %d values differ, first at %d: fused %v, pair %v",
			name, bad, len(got), first, got[first], want[first])
		return
	}
	t.Logf("%s: %d values identical to the last place", name, len(got))
}

// TestHCFusionIsTheCombineThenTheNorm is the kernel-level gate: two mixers of
// layer 0 over llama.cpp's own residual, closed and opened once each way.
//
// It stages the attention mixer and the FFN mixer because that is the
// boundary the graph fuses 48 times — the scatter of one and the norm of the
// next — and it compares all three tensors the boundary produces: the wide
// residual the scatter updates, `xn`, which is the norm's own output and the
// only thing the fusion could get wrong on its own, and `mixed`, which is
// what the rest of the layer sees.
func TestHCFusionIsTheCombineThenTheNorm(t *testing.T) {
	m, tr := fixtures(t)
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	w0, err := m.HCWeights(0, "attn")
	if err != nil {
		t.Fatal(err)
	}
	w1, err := m.HCWeights(0, "ffn")
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewHCGPU(dev, m.Config.HCConfig(), len(ids), []HCWeights{w0, w1},
		HCOpts{Gate: true, Q8: denseQ8Test})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)

	in := mixerInput(t, m, tr, 0, "attn")
	out, err := tr.Get("linear_attn_out-0", 0)
	if err != nil {
		t.Fatal(err)
	}

	// Mixer 0, the layer's block output, then mixer 1 — one arm as two
	// dispatches and one as `cn`.
	arm := func(fused bool) (res, xn, mixed []float32) {
		t.Helper()
		if err := g.Upload(in, len(ids)); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0, false); err != nil {
			t.Fatal(err)
		}
		if err := g.UploadBlockOut(out.Vals); err != nil {
			t.Fatal(err)
		}
		if fused {
			if err := g.RunCombineMix(0, 1); err != nil {
				t.Fatal(err)
			}
		} else {
			if err := g.RunCombine(0); err != nil {
				t.Fatal(err)
			}
			if err := g.Run(1, false); err != nil {
				t.Fatal(err)
			}
		}
		return append([]float32(nil), g.Res()...),
			append([]float32(nil), g.Xn()...),
			append([]float32(nil), g.Mixed()...)
	}

	fRes, fXn, fMixed := arm(true)
	pRes, pXn, pMixed := arm(false)
	identical(t, "res", fRes, pRes)
	identical(t, "xn", fXn, pXn)
	identical(t, "mixed", fMixed, pMixed)
}

// TestHCFusionRunsThePass is the same equality through the graph, where the
// fusion is a schedule and not a call: four layers, nine mixers, eight
// combines, of which six fuse and two do not — the PLE block moves the
// residual out of this arena at layer 1 and the end of the pass has no mixer
// to open. `LLM_HC_NOFUSE=1` is the control arm.
func TestHCFusionRunsThePass(t *testing.T) {
	const layers = 4
	g, _, ids := graphFixture(t, GraphOpts{Layers: layers, NoHead: true})

	run := func(fuse bool) (res, mixed []float32) {
		t.Helper()
		if fuse {
			t.Setenv("LLM_HC_NOFUSE", "0")
		} else {
			t.Setenv("LLM_HC_NOFUSE", "1")
		}
		if err := g.PrefillN(ids, layers); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.Residual()...), append([]float32(nil), g.hc.Mixed()...)
	}

	fRes, fMixed := run(true)
	pRes, pMixed := run(false)
	identical(t, "l_last-3", fRes, pRes)
	identical(t, "mixed", fMixed, pMixed)
}
