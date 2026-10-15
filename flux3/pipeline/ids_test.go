package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/zimage/qwen"
)

// TestIDs builds every id table the two run dumps hold and asks for them
// exactly: the t2i run's 64x64 grid and text window, the edit's 72x57
// target and its one 70x56 reference at l = −1.
func TestIDs(t *testing.T) {
	read := func(dir, name string) *qwen.Mat {
		t.Helper()
		var man struct {
			Tensors map[string]struct {
				Shape []int `json:"shape"`
				Count int   `json:"count"`
			} `json:"tensors"`
		}
		buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
		if err != nil {
			t.Skipf("no dump in %s", dir)
		}
		if err := json.Unmarshal(buf, &man); err != nil {
			t.Fatal(err)
		}
		raw, err := os.ReadFile(filepath.Join(dir, name+".bin"))
		if err != nil {
			t.Fatal(err)
		}
		m := qwen.NewMat(man.Tensors[name].Count/4, 4)
		for i := range m.Data {
			m.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return m
	}
	edit := "../../reference/out/flux3run_edit"
	for _, c := range []struct {
		name      string
		got, want *qwen.Mat
	}{
		{"t2i img", ImageIDs(64, 64, 0), read(runDir, "img_ids")},
		{"t2i txt", TextIDs(80), read(runDir, "txt_ids")},
		{"edit img", ImageIDs(72, 57, 0), read(edit, "img_ids")},
		{"edit cond", ImageIDs(70, 56, -1), read(edit, "cond_ids")},
	} {
		if c.got.Rows != c.want.Rows {
			t.Fatalf("%s: %d rows, want %d", c.name, c.got.Rows, c.want.Rows)
		}
		for i := range c.want.Data {
			if c.got.Data[i] != c.want.Data[i] {
				t.Fatalf("%s: element %d is %g, want %g", c.name, i, c.got.Data[i], c.want.Data[i])
			}
		}
	}
}
