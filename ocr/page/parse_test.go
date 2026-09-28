package page

import (
	"context"
	"testing"
	"time"

	"strix-halo-vulkan/ocr"
	"strix-halo-vulkan/ocr/layout"
	"strix-halo-vulkan/vk"
)

func testDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("page-test")
	if err != nil {
		t.Skipf("no Vulkan instance: %v", err)
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		t.Skipf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == 0x1586 {
			phys = &devices[i]
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		t.Skipf("no compute queue: %v", err)
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Skip(err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// TestParse runs every oracle page end to end in Go -- the image file, the
// device layout, the glue, the device engine -- and holds the markdown and
// the block list to PaddleX's.
func TestParse(t *testing.T) {
	os := oracles(t)
	dev, done := testDevice(t)
	defer done()
	lm, err := layout.Load("../../models/PP-DocLayoutV3")
	if err != nil {
		t.Skipf("no layout checkpoint (%v)", err)
	}
	lg, err := layout.NewGPU(dev, lm)
	if err != nil {
		t.Fatal(err)
	}
	defer lg.Destroy()
	eng, err := ocr.Load(dev, "../../models/PaddleOCR-VL-1.6", ocr.DefaultOptions())
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	defer eng.Destroy()
	p := &Parser{
		Layout: func(px []float32) (*layout.Output, error) {
			out, _, err := lg.Forward(px, nil)
			return out, err
		},
		Recognize: eng.RecognizeAll,
	}
	for _, o := range os {
		t.Run(o.Name, func(t *testing.T) {
			img, format := decode(t, o.Image)
			st0 := eng.Stats()
			start := time.Now()
			pg, err := p.Parse(context.Background(), img)
			if err != nil {
				t.Fatal(err)
			}
			wall := time.Since(start)
			same := 0
			for i := range pg.Texts {
				if i < len(o.VLM) && pg.Texts[i] == o.VLM[i].Raw {
					same++
				}
			}
			t.Logf("%s: %d/%d recognitions identical; layout %v, glue %v, recognition %v, %v in all",
				format, same, len(o.VLM), pg.Timings.Layout.Round(time.Millisecond), pg.Timings.Glue.Round(time.Millisecond),
				pg.Timings.Recognition.Round(time.Millisecond), wall.Round(time.Millisecond))
			st := eng.Stats()
			if n := st.Passes - st0.Passes; n > 0 {
				t.Logf("engine: %d towers %v; %d passes %v (%.1f rows, %.1f logit rows a pass), sampling %v; %d preemptions",
					st.Towers-st0.Towers, (st.Tower - st0.Tower).Round(time.Millisecond), n, (st.Pass - st0.Pass).Round(time.Millisecond),
					float64(st.Rows-st0.Rows)/float64(n), float64(st.LogitRows-st0.LogitRows)/float64(n),
					(st.Sample - st0.Sample).Round(time.Millisecond), st.Preemptions-st0.Preemptions)
			}
			if format == "jpeg" {
				return // the JPEG decodes differently; reported, not held
			}
			if pg.Markdown != o.Markdown {
				t.Errorf("markdown differs from PaddleX's:\n got %q\nwant %q", pg.Markdown, o.Markdown)
			}
		})
	}
}
