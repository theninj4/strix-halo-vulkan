package vae

import (
	"os"
	"path/filepath"
	"testing"
)

// maxBufferBytes is one storage buffer's range on this device; a binding
// past it is clamped silently (a-benchmark-that-gets-faster).
const maxBufferBytes = 0xfffffffc

// TestArenaCeiling plans the full decoder's and taef2's graphs (no device:
// the planner records with no pipelines) at growing square sides and reports
// the largest whose every arena fits one storage buffer, and how the arenas
// scale with the shape at a fixed area (X-o3).
func TestArenaCeiling(t *testing.T) {
	if _, err := os.Stat(filepath.Join(vaeDir, "config.json")); err != nil {
		t.Skip("no vae")
	}
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := LoadDecoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	tiny, err := LoadTinyDecoder(taef2Dir, TAEF2Config())
	if err != nil {
		t.Fatal(err)
	}
	// The served kernels' arenas: the fp16 conv path packs each conv's input,
	// so plan with a conv variant chosen as on a device with matrix cores.
	full := &GPUDecoder{engine: newEngine(nil), cpu: cpu}
	full.conv, _ = convVariantFor(DefaultConvKernel)
	full.attn, _ = attnVariantFor(DefaultAttnKernel)
	full.gemm, _ = gemmVariantFor(DefaultGEMMKernel)
	full.flattenWeights()
	full.stageConvWeights(full.stageProjections(nil, cpu.Mid.Attn))
	small := &GPUTiny{engine: newEngine(nil), cpu: tiny}
	small.conv, _ = convVariantFor(DefaultTinyConvKernel)
	small.flattenWeights()
	small.stageConvWeights(nil)
	enc, err := LoadEncoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ref := &GPUEncoder{engine: newEngine(nil), cpu: enc}
	ref.conv, ref.attn, ref.gemm = full.conv, full.attn, full.gemm
	ref.flattenWeights()
	ref.stageConvWeights(ref.stageProjections(nil, enc.Mid.Attn))
	// The edit encoder (X7) is staged at the same area: one reference's
	// canvas is capped at it.
	encFits := func(h, w int) (bool, [2]int) {
		a, hh, err := ref.planSize(h, w)
		if err != nil {
			t.Fatal(err)
		}
		b := [2]int{int(a) * 4, int(hh) * 2}
		return b[0] <= maxBufferBytes && b[1] <= maxBufferBytes, b
	}
	fits := func(side, sideW int) (bool, [4]int) {
		a, h, err := full.planSize(side/8, sideW/8)
		if err != nil {
			t.Fatal(err)
		}
		ta, th, err := small.planSize(side/8, sideW/8)
		if err != nil {
			t.Fatal(err)
		}
		b := [4]int{int(a) * 4, int(h) * 2, int(ta) * 4, int(th) * 2}
		for _, v := range b {
			if v > maxBufferBytes {
				return false, b
			}
		}
		return true, b
	}
	for _, side := range []int{1024, 1536} {
		_, b := fits(side, side)
		t.Logf("%4d²: decoder fp32 %4d MB, fp16 %4d MB; taef2 fp32 %4d MB, fp16 %4d MB",
			side, b[0]>>20, b[1]>>20, b[2]>>20, b[3]>>20)
	}
	best := 0
	for side := 1024; side <= 4096; side += 16 {
		if ok, _ := fits(side, side); !ok {
			break
		}
		best = side
	}
	_, b := fits(best, best)
	t.Logf("largest square: %d² = %d pixels (decoder fp32 %d MB)", best, best*best, b[0]>>20)
	for _, wh := range [][2]int{{best * 2, best / 2}, {best / 2, best * 2}} {
		ok, b := fits(wh[1], wh[0])
		t.Logf("%dx%d (same area): fits %v, decoder fp32 %d MB", wh[0], wh[1], ok, b[0]>>20)
	}
	for _, wh := range [][2]int{{1024, 1024}, {best, best}, {best * 2, best / 2}, {best / 2, best * 2}} {
		ok, b := encFits(wh[1], wh[0])
		if !ok {
			t.Errorf("the edit encoder at %dx%d: fp32 %d MB, fp16 %d MB, past one buffer", wh[0], wh[1], b[0]>>20, b[1]>>20)
			continue
		}
		t.Logf("encoder %dx%d: fp32 %d MB, fp16 %d MB", wh[0], wh[1], b[0]>>20, b[1]>>20)
	}
}
