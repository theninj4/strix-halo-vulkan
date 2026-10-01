package llm

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

const mtpDraftPath = "../models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"

// TestMTPSeedDeviceIsTheHost is P20d's gate for `nextn` on the device: the
// draft layer's seeded residual from `eh_proj` staged as an int8 bank and run
// over fp16 rows, against `Seed`'s fp32 matvecs on the host, for three rows
// of a trunk-shaped residual. It stages only `blk.48` (1.9 GB); the borrowed
// lm head is a stand-in the seed never touches.
//
// The two are not the same arithmetic — the A operand is narrowed to halves
// and the weight is rounded to int8 — so the bar is a tolerance, and the
// acceptance rate is the gate that matters (it measured count for count the
// host's on both of `cmd/llm -spec`'s arms, 2026-10-01). This catches a
// layout mistake: a stream or a row in the wrong place reads as rms ~1.
func TestMTPSeedDeviceIsTheHost(t *testing.T) {
	if _, err := os.Stat(mtpDraftPath); err != nil {
		t.Skipf("draft checkpoint absent: %v", err)
	}
	if _, err := os.Stat(modelDir); err != nil {
		t.Skipf("trunk checkpoint absent: %v", err)
	}
	dev, done := newTestDevice(t)
	defer done()
	trunk, err := Open(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer trunk.Close()
	draft, err := Open(mtpDraftPath)
	if err != nil {
		t.Fatal(err)
	}
	defer draft.Close()

	c := trunk.Config
	rng := rand.New(rand.NewSource(7))
	head, err := NewHeadGPU(dev, c.NEmbd, q8Tensor(rng, "output.weight", c.NEmbd, 64), 1, true)
	if err != nil {
		t.Fatal(err)
	}
	defer head.Destroy()
	d, err := NewMTPHead(dev, draft, c, head, 4, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Destroy()
	if d.eh == nil {
		t.Skip("LLM_MTP_HOST_NEXTN=1: nothing on the device to compare")
	}

	const n = 3
	hs := make([]float32, n*d.Wide())
	for i := range hs {
		hs[i] = float32(rng.NormFloat64()) * 4
	}
	es := make([]float32, 0, n*d.NEmbd())
	for _, id := range []int32{785, 1879, 13} {
		e, err := trunk.Embedding(id)
		if err != nil {
			t.Fatal(err)
		}
		es = append(es, e...)
	}
	for _, w := range []Wiring{{Name: "res"}, {Name: "res-flip", Flip: true}} {
		var want []float32
		for r := 0; r < n; r++ {
			s, err := d.Seed(hs[r*d.Wide():(r+1)*d.Wide()], es[r*d.NEmbd():(r+1)*d.NEmbd()], w)
			if err != nil {
				t.Fatal(err)
			}
			want = append(want, s...)
		}
		if err := d.seedDevice(hs, es, n, w); err != nil {
			t.Fatal(err)
		}
		got := d.hc.Res()
		if len(got) != len(want) {
			t.Fatalf("%s: %d values against %d", w.Name, len(got), len(want))
		}
		var se, sw float64
		for i := range want {
			e := float64(got[i] - want[i])
			se += e * e
			sw += float64(want[i]) * float64(want[i])
		}
		rel := math.Sqrt(se / sw)
		t.Logf("%s: rms error %.2e of the seed's rms", w.Name, rel)
		if rel > 1e-2 || math.IsNaN(rel) {
			t.Errorf("%s: the device seed is %.3g rms off the host's", w.Name, rel)
		}
	}
}
