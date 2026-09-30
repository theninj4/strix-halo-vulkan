package llm

import (
	"fmt"
	"testing"
)

// TestHCGPUUpGemvRowsAgree: the up GEMV's arithmetic is per row — the weight
// is read once and multiplied into ROWS accumulators, each row's dot product
// in the same order — so row r of a ROWS-row dispatch must be the bits of a
// one-row dispatch over that row alone. On the model's own mixer weights and
// llama.cpp's own input, for every row count the rung carries.
func TestHCGPUUpGemvRowsAgree(t *testing.T) {
	const layer, side = 1, "ffn"
	m, tr := fixtures(t)
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < GEMVMaxRows {
		t.Skipf("%d tokens", len(ids))
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	w, err := m.HCWeights(layer, side)
	if err != nil {
		t.Fatal(err)
	}
	cfg := m.Config.HCConfig()
	for _, spec := range []string{"q5_k/32", "q4_k/32"} {
		sim, err := ParseQuantSim(spec)
		if err != nil {
			t.Fatal(err)
		}
		sim.Mode = "rtn"
		bank, err := BankForSim(sim)
		if err != nil {
			t.Fatal(err)
		}
		g, err := NewHCGPU(dev, cfg, len(ids), []HCWeights{w}, HCOpts{Bank: bank, Sim: sim})
		if err != nil {
			t.Fatal(err)
		}
		in := mixerInput(t, m, tr, layer, side)
		wide, n := cfg.Wide(), cfg.NEmbd
		run := func(first, rows int, up HCKernel) []float32 {
			t.Helper()
			if err := g.Upload(in[first*wide:(first+rows)*wide], rows); err != nil {
				t.Fatal(err)
			}
			if err := g.SetPlan(HCDownGemv32, up); err != nil {
				t.Fatal(err)
			}
			if err := g.Run(0, false); err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), g.Mixed()...)
		}
		for rows := 2; rows <= GEMVMaxRows; rows++ {
			multi := run(0, rows, HCUpGemv)
			gemm := run(0, rows, HCUpM1)
			for r := 0; r < rows; r++ {
				single := run(r, 1, HCUpGemv)
				rs, err := compare(multi[r*n:(r+1)*n], single[:n])
				if err != nil {
					t.Fatal(err)
				}
				rg, err := compare(multi[r*n:(r+1)*n], gemm[r*n:(r+1)*n])
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("%s %d rows, row %d: against the one-row GEMV %v; against the GEMM %v", spec, rows, r, rs, rg)
				if rs.maxAbs != 0 {
					t.Errorf("%s %d rows, row %d: the %d-row GEMV differs from the one-row GEMV on the same row: %v", spec, rows, r, rows, rs)
				}
			}
		}
		g.Destroy()
		_ = fmt.Sprint
	}
}
