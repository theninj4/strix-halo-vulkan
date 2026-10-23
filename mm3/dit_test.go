package mm3

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

const ditRef = "../reference/out/mm3dit" // reference/dump_mm3_dit.py

type ditManifest struct {
	Frames    int     `json:"frames"`
	Steps     int     `json:"steps"`
	Guidance  float64 `json:"guidance"`
	Starts    []int   `json:"starts"`
	Overlap   int     `json:"overlap_latents"`
	CropLeft  int     `json:"crop_left"`
	CropRight int     `json:"crop_right"`
	Scheduler struct {
		Timesteps []float64 `json:"timesteps"`
		Sigmas    []float64 `json:"sigmas"`
	} `json:"scheduler"`
	FFOutMax []float64 `json:"ff_out_max"`
	Tensors  map[string]struct {
		Shape []int  `json:"shape"`
		DType string `json:"dtype"`
	} `json:"tensors"`
}

func loadDiTManifest(t testing.TB) *ditManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(ditRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference (run reference/dump_mm3_dit.py): %v", err)
	}
	var m ditManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// ditTensor reads one of the dump's float32 tensors.
func ditTensor(t testing.TB, m *ditManifest, name string) ([]int, []float32) {
	t.Helper()
	info, ok := m.Tensors[name]
	if !ok || info.DType != "float32" {
		t.Fatalf("reference has no float32 %s", name)
	}
	buf, err := os.ReadFile(filepath.Join(ditRef, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(buf)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(buf[4*i]) | uint32(buf[4*i+1])<<8 | uint32(buf[4*i+2])<<16 | uint32(buf[4*i+3])<<24)
	}
	return info.Shape, out
}

// channels splits [128][L] row-major into its rows.
func channels(x []float32, L int) [][]float32 {
	out := make([][]float32, len(x)/L)
	for c := range out {
		out[c] = x[c*L : (c+1)*L]
	}
	return out
}

func flatRel(got [][]float32, want []float32) float64 {
	var se, sw float64
	L := len(got[0])
	for c, row := range got {
		for i, v := range row {
			w := float64(want[c*L+i])
			se += (float64(v) - w) * (float64(v) - w)
			sw += w * w
		}
	}
	return math.Sqrt(se / sw)
}

// TestDiTForward is N4's gate: window 0's first step through the DiT, both
// CFG rows, against the fp32 oracle's predictions. The guided velocity's
// error is the one that moves the latents.
func TestDiTForward(t *testing.T) {
	m := loadDiTManifest(t)
	dev, err := openDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	shape, lat := ditTensor(t, m, "w0_lat0")
	L := shape[1]
	t0 := time.Now()
	g, err := LoadDiT(dev, modelDir, L)
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	defer g.Destroy()
	t.Logf("staged the DiT in %v (%d banks, attention %s); ff_out input max over the oracle's run %.1f",
		time.Since(t0), len(g.banks), g.attn, maxOf(m.FFOutMax))
	_, cond := ditTensor(t, m, "w0_condition")
	_, wc := ditTensor(t, m, "w0_out_c0")
	_, wu := ditTensor(t, m, "w0_out_u0")
	_, wv := ditTensor(t, m, "w0_v0")
	ts := float32(m.Scheduler.Timesteps[0])
	var took time.Duration
	var c, u [][]float32
	for i := 0; i < 3; i++ {
		var d time.Duration
		c, u, d, err = g.Forward(channels(lat, L), cond, ts)
		if err != nil {
			t.Fatal(err)
		}
		took = d
	}
	v := make([][]float32, len(c))
	for ch := range c {
		v[ch] = make([]float32, L)
		for i := range v[ch] {
			v[ch][i] = u[ch][i] + float32(m.Guidance)*(c[ch][i]-u[ch][i])
		}
	}
	rc, ru, rv := flatRel(c, wc), flatRel(u, wu), flatRel(v, wv)
	t.Logf("t=%.4f, %d latents: rel cond %.2e uncond %.2e guided %.2e; a CFG forward %v (GPU)", ts, L, rc, ru, rv, took)
	if !(rv <= 5e-3) {
		t.Errorf("guided velocity rel %.2e, over 5e-3", rv)
	}
}

func maxOf(x []float64) float64 {
	m := 0.0
	for _, v := range x {
		m = math.Max(m, v)
	}
	return m
}
