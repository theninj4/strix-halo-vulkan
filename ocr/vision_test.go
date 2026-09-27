package ocr

import (
	"math"
	"testing"
	"time"

	"strix-halo-vulkan/zimage/qwen"
)

// relErr is ||got - want|| / ||want|| over two equal-length slices.
func relErr(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(num / den)
}

// TestTowerCPU walks the line case through the embedding (patch projection
// plus the resized position grid) and the first two blocks, and runs the
// projector from the dump's own post-LN rows. The whole stack is gated on
// the device (O3): 27 blocks of scalar attention are minutes here.
func TestTowerCPU(t *testing.T) {
	if testing.Short() {
		t.Skip("scalar fp64 tower: ~20 s")
	}
	recs := records(t)
	var rec record
	for _, r := range recs {
		if r.Name == "line" {
			rec = r
		}
	}
	if rec.Name == "" {
		t.Skip("no line case in the reference")
	}
	tw, err := LoadTower(modelDir, 2)
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	pix, _ := refTensor(t, rec.Name, "pixels")
	im := &Image{Patches: pix, GridH: rec.GridTHW[1], GridW: rec.GridTHW[2]}
	start := time.Now()
	checked := 0
	_, err = tw.Forward(im, func(name string, got *qwen.Mat) {
		want, _ := refTensor(t, rec.Name, name)
		e := relErr(got.Data, want)
		t.Logf("%-12s rel %.2e", name, e)
		if e > 1e-5 {
			t.Errorf("%s: rel %.2e over 1e-5", name, e)
		}
		checked++
	})
	if err != nil {
		t.Fatal(err)
	}
	if checked != 3 {
		t.Fatalf("checked %d stages, want embed + 2 blocks", checked)
	}

	full, err := LoadTower(modelDir, -1)
	if err != nil {
		t.Fatal(err)
	}
	post, _ := refTensor(t, rec.Name, "vis.post_ln")
	got, err := full.Project(&qwen.Mat{Rows: im.GridH * im.GridW, Cols: VisionHidden, Data: post}, im.GridH, im.GridW)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := refTensor(t, rec.Name, "proj")
	e := relErr(got.Data, want)
	t.Logf("%-12s rel %.2e (from the dump's post-LN)  [%v]", "proj", e, time.Since(start).Round(time.Millisecond))
	if e > 1e-5 {
		t.Errorf("proj: rel %.2e over 1e-5", e)
	}
}
