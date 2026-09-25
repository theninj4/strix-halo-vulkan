package kev

import (
	"math"
	"math/rand"
	"testing"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

// TestGPUGEMMPadRowsDoNotLeak runs one projection GEMM over 37 real rows,
// padded to its tile, with the pad rows of A set to zeros, to large finite
// values, to NaN and to ordinary activations, and requires the real rows of C
// to be the same bits every time. Every plain-store rung is run on its own
// grid, the row-block-fastest rb rungs (K7.6) included. A GEMM's rows are independent by its arithmetic; this checks
// the kernels agree.
func TestGPUGEMMPadRowsDoNotLeak(t *testing.T) {
	g := loadGPU(t)
	const n = 37
	lw := &g.w[0]
	p := lw.in
	lda := g.Cfg.Hidden + gemmPad
	rng := rand.New(rand.NewSource(1))
	real := make([]uint16, n*lda)
	for i := range real {
		real[i] = safetensors.F32ToF16(float32(rng.NormFloat64()))
	}
	rungs := gemmKernels
	if g.Bank == BankQ8 {
		rungs = append(append([]gemmKernel(nil), q8Kernels...), gluKernels[3:]...)
	}
	for _, rung := range rungs {
		tokPad := roundUp(n, rung.bm)
		var ref []float32
		for _, fill := range []float32{0, 30000, float32(math.NaN()), 1} {
			pad := make([]uint16, (tokPad-n)*lda)
			for i := range pad {
				pad[i] = safetensors.F32ToF16(fill)
				if fill == 1 { // ordinary activations, like a neighbour's stale rows
					pad[i] = safetensors.F32ToF16(float32(rng.NormFloat64()))
				}
			}
			g.hbuf.WriteUint16At(int(g.hA), real)
			g.hbuf.WriteUint16At(int(g.hA)+n*lda, pad)
			var pc []byte
			if g.Bank == BankQ8 {
				pc = llmPush{xnOff: g.hA, outOff: g.aP, bOff: p.off, lda: uint32(lda),
					gemmM: uint32(tokPad), gemmN: uint32(p.n), gemmK: uint32(p.k)}.bytes()
			} else {
				pc = push{InOff: g.hA, OutOff: g.aP, BOff: p.off, GemmM: uint32(tokPad), GemmN: uint32(p.n),
					GemmK: uint32(p.k), LDA: uint32(lda)}.bytes()
			}
			gx, gy := uint32(p.n/rung.bn), uint32(tokPad/rung.bm)
			if g.Bank == BankQ8 {
				gx, gy = rung.grid(p.n, tokPad)
			}
			d := []vk.MultiDispatch{{Pipeline: g.gemms[lw.bank][rung.name], GroupsX: gx, GroupsY: gy, PushConstants: pc}}
			if _, err := vk.DispatchMultiTimed(d, 1, 1, true); err != nil {
				t.Fatal(err)
			}
			got := g.abuf.ReadFloat32At(int(g.aP), n*p.n)
			if ref == nil {
				ref = got
				continue
			}
			diff := 0
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(ref[i]) {
					diff++
				}
			}
			t.Logf("%s, pad rows = %v: %d of %d real outputs differ from zero padding", rung.name, fill, diff, len(got))
			if diff > 0 {
				t.Errorf("%s: pad rows of %v leak into real rows", rung.name, fill)
			}
		}
	}
}

// TestGPUReadsOnlyWhatItWrote poisons every per-pass scratch tensor with NaN
// before each pass and requires every readout to come back finite: a kernel
// that reads memory the pass never wrote -- a pad row, a stale tail, a
// neighbour's slot -- would carry the NaN to a real row.
func TestGPUReadsOnlyWhatItWrote(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	g.Poison = true
	defer func() { g.Poison, g.PassRows = false, 0 }()
	for _, fx := range loadFixtures(t) {
		enc := encodeFixture(t, e, fx.Request)
		for _, rows := range []int{0, 42} {
			if rows > 0 && enc.StateLen > rows {
				continue
			}
			g.ClearCache()
			g.PassRows = rows
			for rep := 0; rep < 2; rep++ { // a miss, then a cache hit
				probs, p, err := g.Probs(enc)
				if err != nil {
					if rows > 0 {
						break
					}
					t.Fatal(err)
				}
				for k := range probs {
					for _, v := range probs[k] {
						if math.IsNaN(v) {
							t.Errorf("%s (pass rows %d, hit %v): question %d is NaN", fx.Name, rows, p.CacheHit, k)
							break
						}
					}
				}
			}
		}
	}
}
