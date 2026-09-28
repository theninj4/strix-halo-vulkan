package ocr

import (
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestLMStepLadder times a decode step at depth ~200 (median of 64), and
// with OCR_SLABS=q,k,v,o,gate,up,down tries another split-K table (1s and
// 16s: the slab counts with 16-row builds).
func TestLMStepLadder(t *testing.T) {
	if testing.Short() {
		t.Skip("timing")
	}
	dev, done := newTestDevice(t)
	defer done()
	if s := os.Getenv("OCR_SLABS"); s != "" {
		for i, f := range strings.Split(s, ",") {
			n, _ := strconv.Atoi(f)
			gemvSlabs[i] = n
		}
	}
	lm, err := LoadLM(dev, modelDir, LMOptions{MaxLen: 1024, Slots: 1, Rows: 256})
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	defer lm.Destroy()
	rows := make([]Row, 200)
	for i := range rows {
		rows[i] = Row{ID: int32(1000 + i), Pos: i, Rope: [3]int32{int32(i), int32(i), int32(i)}}
	}
	if _, _, err := lm.Pass(rows, 0); err != nil {
		t.Fatal(err)
	}
	var walls, gpus []time.Duration
	for s := 0; s < 64; s++ {
		p := int32(200 + s)
		start := time.Now()
		_, g, err := lm.Pass([]Row{{ID: 1234, Pos: 200 + s, Rope: [3]int32{p, p, p}}}, 1)
		if err != nil {
			t.Fatal(err)
		}
		walls, gpus = append(walls, time.Since(start)), append(gpus, g)
	}
	med := func(d []time.Duration) time.Duration {
		c := append([]time.Duration(nil), d...)
		for i := range c {
			for j := i + 1; j < len(c); j++ {
				if c[j] < c[i] {
					c[i], c[j] = c[j], c[i]
				}
			}
		}
		return c[len(c)/2]
	}
	t.Logf("slabs %v: step %v wall, %v device", gemvSlabs, med(walls), med(gpus))
}

// TestLMRowLadder times a decode step of N rows, one a slot, each at depth
// ~200 (median of 32): what a page's regions decoding together cost.
// OCR_GEMV_ROWS=N moves the GEMV/GEMM crossover to N rows (0: GEMMs only).
func TestLMRowLadder(t *testing.T) {
	if testing.Short() {
		t.Skip("timing")
	}
	dev, done := newTestDevice(t)
	defer done()
	ns := []int{1, 2, 3, 4, 8, 12, 16, 24, 32, 64}
	if v := os.Getenv("OCR_GEMV_ROWS"); v != "" {
		gemvRows, _ = strconv.Atoi(v)
		defer func() { gemvRows = gemvMaxRows }()
	}
	lm, err := LoadLM(dev, modelDir, LMOptions{MaxLen: 512, Slots: 64, Rows: 256})
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	defer lm.Destroy()
	for s := 0; s < 64; s++ {
		rows := make([]Row, 200)
		for i := range rows {
			rows[i] = Row{ID: int32(1000 + i + s), Slot: s, Pos: i, Rope: [3]int32{int32(i), int32(i), int32(i)}}
		}
		if _, _, err := lm.Pass(rows, 0); err != nil {
			t.Fatal(err)
		}
	}
	for _, n := range ns {
		var walls, gpus []time.Duration
		for s := 0; s < 32; s++ {
			rows := make([]Row, n)
			for i := range rows {
				p := int32(200 + s)
				rows[i] = Row{ID: 1234, Slot: i, Pos: 200 + s, Rope: [3]int32{p, p, p}}
			}
			start := time.Now()
			_, g, err := lm.Pass(rows, n)
			if err != nil {
				t.Fatal(err)
			}
			walls, gpus = append(walls, time.Since(start)), append(gpus, g)
		}
		w, g := median(walls), median(gpus)
		t.Logf("rows %2d: step %v wall, %v device, %.2f ms a row", n, w, g, float64(w.Microseconds())/1000/float64(n))
	}
}

func median(d []time.Duration) time.Duration {
	c := append([]time.Duration(nil), d...)
	sort.Slice(c, func(i, j int) bool { return c[i] < c[j] })
	return c[len(c)/2]
}
