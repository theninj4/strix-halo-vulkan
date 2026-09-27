package ocr

import (
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestLMStepLadder times a decode step at depth ~200 (median of 64), and
// with OCR_SLABS=q,k,v,o,gate,up,down tries another split-K table.
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
