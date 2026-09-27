package layout

import (
	"math"
	"testing"
	"time"
)

// TestStageTimes reports the CPU forward's time between taps (the enc.coord
// and enc.topk taps cost a little extra work of their own).
func TestStageTimes(t *testing.T) {
	if testing.Short() {
		t.Skip("timing")
	}
	m := load(t)
	px := refTensors(t, "page")["pixels"]
	last := time.Now()
	_, err := m.Forward(px, func(name string, _ []float32) {
		t.Logf("%-14s %v", name, time.Since(last).Round(time.Millisecond))
		last = time.Now()
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%-14s %v", "(order, masks)", time.Since(last).Round(time.Millisecond))
}

// TestFP16Ladder runs every case with fp16 operands on every conv and
// linear (fp32 sums) and reports what a matrix-core port would cost: the
// largest activation against half's range, each stage against the fp32
// dump, the query selection, the initial (mask) boxes and the detections.
func TestFP16Ladder(t *testing.T) {
	if testing.Short() {
		t.Skip("five CPU forwards")
	}
	m := load(t)
	defer func() { fp16Operands, fp16Linears = false, true }()
	for _, rec := range records(t) {
		ref := refTensors(t, rec.Name)
		for arm, on := range []bool{false, true, true} {
			fp16Operands = on
			fp16Linears = arm == 1 // arm 2: the convs alone
			absMax = 0
			stages := map[string]float64{}
			var topkSame int
			var initRef float64
			out, err := m.Forward(ref["pixels"], func(name string, got []float32) {
				want := ref[name]
				switch name {
				case "enc.topk":
					set := map[float32]bool{}
					for _, v := range want {
						set[v] = true
					}
					for _, v := range got {
						if set[v] {
							topkSame++
						}
					}
				case "dec.init_ref":
					for i := range want {
						initRef = max(initRef, math.Abs(float64(got[i]-want[i])))
					}
				case "enc.coord":
				default:
					stages[name] = relErr(got, want)
				}
			})
			if err != nil {
				t.Fatal(err)
			}
			regs := out.Detect(rec.Size[1], rec.Size[0], rec.Threshold)
			same := 0
			var ds, db float64
			for _, w := range rec.Detections {
				for _, g := range regs {
					if g.Query == w.Query && g.LabelID == w.LabelID {
						same++
						ds = max(ds, math.Abs(g.Score-w.Score))
						for k := range g.Box {
							db = max(db, math.Abs(g.Box[k]-w.Box[k]))
						}
					}
				}
			}
			t.Logf("%-8s arm %d fp16=%-5v absmax %8.0f | stage4 %.1e pan0 %.1e mask_feat %.1e layer5 %.1e logits %.1e | topk %d/300 same, init_ref max %.3f | %d/%d regions (got %d), score %.3f, box %.1f px",
				rec.Name, arm, on, absMax, stages["bb.stage4"], stages["enc.pan0"], stages["enc.mask_feat"], stages["dec.layer5"], stages["dec.logits"],
				topkSame, initRef, same, len(rec.Detections), len(regs), ds, db)
		}
	}
}
