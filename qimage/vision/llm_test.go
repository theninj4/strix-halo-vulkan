package vision

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/zimage/qwen"
)

// reference/dump_llm_vision.py: qwen3.8-flash-next's tower in fp32 (and the
// square card in float64) over three cards, LLM-VISION.md V2.
const llmRefDir = "../../reference/out/llmvision"

type llmManifest struct {
	Cards map[string]struct {
		HW      []int `json:"hw"`
		GridTHW []int `json:"grid_thw"`
	} `json:"cards"`
	Decode []struct {
		File string `json:"file"`
		HW   []int  `json:"hw"`
	} `json:"decode"`
	BF16 map[string]struct {
		RMSRel   float64 `json:"rms_rel"`
		WorstCos float64 `json:"worst_cos"`
	} `json:"bf16_vs_fp32"`
	Tensors map[string]struct {
		Shape []int  `json:"shape"`
		DType string `json:"dtype"`
		Count int    `json:"count"`
	} `json:"tensors"`
}

func loadLLMRef(t *testing.T) *llmManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(llmRefDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_llm_vision.py", llmRefDir, err)
	}
	var m llmManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// llmRef reads a dumped tensor as a float32 matrix (float64 dumps are
// narrowed), folding leading axes into rows.
func llmRef(t *testing.T, m *llmManifest, name string) *qwen.Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(llmRefDir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	data := make([]float32, meta.Count)
	for i := range data {
		if meta.DType == "float64" {
			data[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		} else {
			data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	}
	cols := meta.Shape[len(meta.Shape)-1]
	return &qwen.Mat{Rows: meta.Count / cols, Cols: cols, Data: data}
}

// TestLLMTower is V2's gate: the tower loaded from the mmproj, run on the
// processor's own pixel_values, against HF's fp32 tower at every tap. The
// square card is also held against the float64 run, and the resized
// 18x32 card exercises a non-square grid (block order, the interpolated
// position grid and the rope all read the two sides separately).
func TestLLMTower(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the whole 27-layer tower twice on the CPU")
	}
	ref := loadLLMRef(t)
	_, model, err := LoadGGUF(llmMMProj, 0)
	if err != nil {
		t.Skipf("no mmproj (%v)", err)
	}
	for _, card := range []string{"sq", "odd"} {
		t.Run(card, func(t *testing.T) {
			g := ref.Cards[card].GridTHW
			pixels := llmRef(t, ref, card+"_pixel_values")
			out, err := model.Forward(pixels, g[1], g[2], func(name string, x *qwen.Mat) {
				switch name {
				case "patch_embed", "block0_out", "block1_out", "block13_out":
					compare(t, name, x, llmRef(t, ref, card+"_"+name))
				}
			})
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "last_hidden", out.Last, llmRef(t, ref, card+"_last_hidden"))
			compare(t, "merged", out.Merged, llmRef(t, ref, card+"_merged"))
			if card == "sq" {
				compare(t, "merged_vs_f64", out.Merged, llmRef(t, ref, "sq_merged64"))
			}
		})
	}
}
