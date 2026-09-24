package vision

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/zimage/qwen"
)

// TestLLMGPUTower is V4's gate: qwen3.8-flash-next's tower from the mmproj,
// on the device, against HF's fp32 dump on three grids run through one
// staged tower. The bound is llmFP16Tol below.
func TestLLMGPUTower(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole tower")
	}
	ref := loadLLMRef(t)
	_, model, err := LoadGGUF(llmMMProj, 0)
	if err != nil {
		t.Skipf("no mmproj (%v)", err)
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	// Staged once for the largest card: every grid below runs inside the
	// same arenas, as a served tower does.
	g, err := NewGPU(dev, model, 128*128)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	cards := []string{"sq", "odd", "tiny", "big", "huge"}
	if env := os.Getenv("VISION_CARDS"); env != "" {
		cards = strings.Split(env, ",")
	}
	for _, card := range cards {
		gr := ref.Cards[card].GridTHW
		if gr == nil {
			t.Fatalf("the dump has no %q card; rerun reference/dump_llm_vision.py", card)
		}
		out, err := g.Forward(t.Context(), llmRef(t, ref, card+"_pixel_values"), gr[1], gr[2])
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: %dx%d patches", card, gr[1], gr[2])
		// The large cards dump only the merged rows (HF's SDPA at 4 096 and
		// 16 384 patches).
		if _, ok := ref.Tensors[card+"_last_hidden"]; ok {
			compareAt(t, card+" last_hidden", out.Last, llmRef(t, ref, card+"_last_hidden"), llmFP16Tol)
		}
		want := llmRef(t, ref, card+"_merged")
		if gr[1]*gr[2] <= 1024 {
			compareAt(t, card+" merged", out.Merged, want, llmFP16Tol)
		}
		rms, cos := rmsGap(out.Merged, want)
		bf, ok := ref.BF16[card]
		if !ok {
			t.Fatalf("the dump has no bf16 yardstick for %q; rerun reference/dump_llm_vision.py", card)
		}
		t.Logf("%-16s rms-relative %.3g (bf16 %.3g), worst row cosine %.6f (bf16 %.6f)",
			card+" merged", rms, bf.RMSRel, cos, bf.WorstCos)
		if rms > bf.RMSRel/4 || cos < bf.WorstCos {
			t.Errorf("%s: the device is not within a quarter of the checkpoint's own bf16 error", card)
		}
	}
}

// The worst-element bound below holds for grids up to 1 024 patches, where it
// was measured. A worst element grows with the element count by itself, and
// so does fp16 rounding over a longer softmax: 1024² reaches rel 0.11 and
// 2048² 0.31, with the scalar kernel as bad as the matrix cores (0.12 at
// 1024²). So every card, and the large ones only, is held to the
// checkpoint's own precision instead. HF's tower in bf16, which is how the
// model is served, is dumped against fp32 on the same pixels, and the
// device's rms-relative error must be under a quarter of that, with its worst
// row cosine no lower. Measured: bf16 is 0.027-0.10 rms and 0.75-0.998
// cosine; the device is 0.0015-0.0068 and 0.9964-0.99999.
//
// llmFP16Tol is the device's bound against HF's fp32 tower, set from the
// measurement at about three times the worst card: sq 0.0165, odd 0.030,
// tiny 0.015 on the merged rows (LLM-VISION.md V4). This tower is 5-20x
// tighter than Qwen-Image's under the same kernels (fp16Tol, 0.14-0.56),
// and a structural mistake moves these tensors by 0.76 or more
// (TestNegativeControls), so the bound separates the two by 7x.
const llmFP16Tol = 0.1

// TestLLMGPUTowerTiming prices the tower against image size: square grids
// from 32x32 patches (256 image tokens, 512²) up to VISION_MAX_SIDE patches a
// side (default 64, a 1024² image). The processor allows 256 a side, and the
// attention is full, so the cost is quadratic in patches. With the scalar
// attention kernel 128 a side does not finish: a dispatch outruns the
// driver's timeout (VK_TIMEOUT) (LLM-VISION.md V4).
func TestLLMGPUTowerTiming(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole tower")
	}
	_, model, err := LoadGGUF(llmMMProj, 0)
	if err != nil {
		t.Skipf("no mmproj (%v)", err)
	}
	maxSide := 64
	if s, err := strconv.Atoi(os.Getenv("VISION_MAX_SIDE")); err == nil {
		maxSide = s
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGPU(dev, model, maxSide*maxSide)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	t.Logf("budget %d patches: %d MB weights, %d MB activations",
		maxSide*maxSide, g.WeightBytes()>>20, g.ActivationBytes()>>20)
	for side := 32; side <= maxSide; side *= 2 {
		pixels := qwen.NewMat(side*side, model.Cfg.PatchElems())
		for i := range pixels.Data {
			pixels.Data[i] = float32(i%255)/127.5 - 1
		}
		var best time.Duration
		for run := 0; run < 2; run++ {
			start := time.Now()
			if _, err := g.Forward(t.Context(), pixels, side, side); err != nil {
				t.Fatal(err)
			}
			if d := time.Since(start); run == 0 || d < best {
				best = d
			}
		}
		t.Logf("%4dx%-4d patches (%5d tokens, %4d px square): %v",
			side, side, side*side/4, side*16, best.Round(time.Millisecond))
	}
}

// TestLLMGPUArenaSizes prints the two activation arenas at the processor's
// ceiling. Each must stay under maxStorageBufferRange (4 GiB - 4), which
// NewGPU now refuses to cross.
func TestLLMGPUArenaSizes(t *testing.T) {
	_, model, err := LoadGGUF(llmMMProj, 0)
	if err != nil {
		t.Skipf("no mmproj (%v)", err)
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	for _, side := range []int{128, 256} {
		g, err := NewGPU(dev, model, side*side)
		if err != nil {
			t.Fatalf("%d patches: %v", side*side, err)
		}
		t.Logf("%6d patches: fp32 arena %.2f GB, fp16 arena %.2f GB",
			side*side, float64(g.abuf.Size())/1e9, float64(g.hbuf.Size())/1e9)
		g.Destroy()
	}
}

// rmsGap is the error's rms over the reference's, and the lowest cosine
// similarity between a row and its reference: two views that, unlike
// relGap's worst element, do not grow with the element count.
func rmsGap(got, want *qwen.Mat) (rms, worstCos float64) {
	var dd, ww float64
	worstCos = 1
	for r := 0; r < want.Rows; r++ {
		var gw, gg, wr float64
		for c := 0; c < want.Cols; c++ {
			g, w := float64(got.Data[r*want.Cols+c]), float64(want.Data[r*want.Cols+c])
			dd += (g - w) * (g - w)
			ww += w * w
			gw += g * w
			gg += g * g
			wr += w * w
		}
		if cos := gw / math.Sqrt(gg*wr); cos < worstCos {
			worstCos = cos
		}
	}
	return math.Sqrt(dd / ww), worstCos
}

// TestLLMJPEGGap prices what is left of the JPEG decode gap (LLM-VISION.md
// V3) where it matters: each JPEG in the dump, decoded by PIL and by
// llm/pixels, through the same processor and the same device tower.
//
// The finding is that the tower itself sets the scale. The residual of at
// most 3 levels on ~2% of samples (Go's IDCT) moves the merged rows by
// 0.018-0.106 rms, which looks alarming until the control: **±1 level of
// noise on 2% of samples moves them by 0.019-0.073, with worst row cosines
// down to 0.28**. The tower amplifies any invisible perturbation that much,
// as qimage's Q8.3 found for the same ViT, so a decoder that is not
// libjpeg-turbo, which every serving stack has (llama.cpp's is stb_image), is
// noise and not a bug. The gate: reruns are bit-identical, and the decode gap
// is within twice what the noise does.
func TestLLMJPEGGap(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole tower")
	}
	ref := loadLLMRef(t)
	_, model, err := LoadGGUF(llmMMProj, 0)
	if err != nil {
		t.Skipf("no mmproj (%v)", err)
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGPU(dev, model, 20000)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	run := func(img *pixels.RGB) *qwen.Mat {
		planes, h, w, err := pixels.Default.Planes(img)
		if err != nil {
			t.Fatal(err)
		}
		rows, gh, gw, err := model.Cfg.Patchify(planes, h, w)
		if err != nil {
			t.Fatal(err)
		}
		out, err := g.Forward(t.Context(), rows, gh, gw)
		if err != nil {
			t.Fatal(err)
		}
		return out.Merged
	}
	for _, c := range ref.Decode {
		if !strings.HasSuffix(c.File, ".jpg") {
			continue
		}
		dir := filepath.Join(llmRefDir, "decode")
		data, err := os.ReadFile(filepath.Join(dir, c.File))
		if err != nil {
			t.Fatal(err)
		}
		pil, err := os.ReadFile(filepath.Join(dir, c.File+".rgb"))
		if err != nil {
			t.Fatal(err)
		}
		ours, _, err := pixels.Decode(data)
		if err != nil {
			t.Fatal(err)
		}
		want := run(&pixels.RGB{W: c.HW[1], H: c.HW[0], Pix: pil})
		again := run(&pixels.RGB{W: c.HW[1], H: c.HW[0], Pix: pil})
		got := run(ours)
		// The sensitivity control: PIL's pixels with one level of noise
		// on a fixed 2% of samples, which is the decode gap's size and
		// density without its structure.
		noisy := append([]uint8(nil), pil...)
		state := uint32(12345)
		for i := range noisy {
			state = state*1664525 + 1013904223
			if state>>24 < 5 { // 5/256 ~ 2%
				if state&(1<<12) != 0 && noisy[i] < 255 {
					noisy[i]++
				} else if noisy[i] > 0 {
					noisy[i]--
				}
			}
		}
		ctl := run(&pixels.RGB{W: c.HW[1], H: c.HW[0], Pix: noisy})
		rms, cos := rmsGap(got, want)
		rms0, _ := rmsGap(again, want)
		rmsN, cosN := rmsGap(ctl, want)
		t.Logf("%-28s %4dx%-4d  decode gap rms %.3g cos %.4f | ±1 noise on 2%%: rms %.3g cos %.4f | rerun %.3g",
			c.File, c.HW[1], c.HW[0], rms, cos, rmsN, cosN, rms0)
		if rms0 != 0 {
			t.Errorf("%s: two runs over the same pixels differ by %.3g", c.File, rms0)
		}
		if rms > 2*rmsN {
			t.Errorf("%s: the decode gap (%.3g) is over twice what invisible noise does (%.3g)", c.File, rms, rmsN)
		}
	}
}
