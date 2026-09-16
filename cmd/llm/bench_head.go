package main

// The lm head's benchmark: LLM.md L8a, the first dense bank that is not
// halves.
//
//	go run ./cmd/llm -head                          # both banks, every rung, 1..512 rows
//	go run ./cmd/llm -head -tokens 1                # the shape the model actually runs
//	go run ./cmd/llm -head -csv results/l8a_head.csv
//
// `output.weight` is [2560, 248320] Q8_0 — 675 MB in the checkpoint, 1.27 GB
// as the halves every GEMM in this vertical read before L8 and 0.68 GB as the
// int8-plus-scales bank that replaces them. It is staged twice here, so the
// two are compared on the same device in the same process against the same
// activation, and the only thing that differs is the format and the kernel
// build that reads it.
//
// What is timed is GPU dispatch time over `-iters` back-to-back repetitions.
// There is no sweep over staged copies as -hc has: one bank is 0.68 GB and
// the MALL is 32 MiB, so nothing here is ever warm.

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"

	"strix-halo-vulkan/llm"
)

type headRow struct {
	bank   string
	rung   llm.GEMMKernel
	rows   int
	us     float64
	gbs    float64
	weight int
}

// mib prints a byte count the way the resident plan does.
func mib(n int) string { return fmt.Sprintf("%.2f GB", float64(n)/1e9) }

func headBench(model string, rows []int, iters int, csvPath string) error {
	m, err := llm.Open(model)
	if err != nil {
		return err
	}
	defer m.Set.Close()
	c := m.Config
	t, err := m.Set.Get("output.weight")
	if err != nil {
		return fmt.Errorf("output.weight: %w", err)
	}
	fmt.Printf("head: %s %v, %s in the checkpoint\n", t.Type, t.Dims, mib(len(t.Data)))

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()

	maxRows := 1
	for _, r := range rows {
		maxRows = max(maxRows, r)
	}

	// One activation for both banks, so a disagreement could only be the
	// kernel's. It is the same shape `result_norm` has: one row per token.
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, maxRows*c.NEmbd)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	var out []headRow
	var logits [2][]float32
	for bi, q8 := range []bool{false, true} {
		name := "fp16"
		if q8 {
			name = "q8"
		}
		start := time.Now()
		g, err := llm.NewHeadGPU(dev, c.NEmbd, t, maxRows, q8)
		if err != nil {
			return fmt.Errorf("head (%s): %w", name, err)
		}
		fmt.Printf("%-4s bank %s staged in %s\n", name, mib(g.WeightBytes()),
			time.Since(start).Round(time.Millisecond))
		for _, r := range rows {
			if err := g.Upload(x[:r*c.NEmbd], r); err != nil {
				g.Destroy()
				return err
			}
			for _, k := range llm.GEMMKernels() {
				if err := g.SetKernel(k); err != nil {
					g.Destroy()
					return err
				}
				st, err := g.Profile(iters)
				if err != nil {
					g.Destroy()
					return err
				}
				us := float64(st[0].GPU.Nanoseconds()) / 1e3
				out = append(out, headRow{name, k, r, us,
					float64(g.WeightBytes()) / (us * 1e-6) / 1e9, g.WeightBytes()})
			}
			if r == rows[0] {
				// The gate: the two banks are the same numbers, so they are
				// the same logits, and this says so on the real tensor rather
				// than on the unit test's synthetic one.
				if err := g.SetKernel(llm.GEMMKernels()[0]); err != nil {
					g.Destroy()
					return err
				}
				if err := g.Run(); err != nil {
					g.Destroy()
					return err
				}
				logits[bi] = append([]float32(nil), g.Logits()[:g.Vocab()]...)
			}
		}
		g.Destroy()
	}

	diff := 0
	for i := range logits[0] {
		if logits[0][i] != logits[1][i] {
			diff++
		}
	}
	fmt.Printf("\nrow 0 logits: %d of %d differ between the two banks\n", diff, len(logits[0]))

	fmt.Printf("\n%-5s %-8s %6s %10s %8s\n", "bank", "rung", "rows", "us", "GB/s")
	for _, r := range out {
		fmt.Printf("%-5s %-8s %6d %10.1f %8.1f\n", r.bank, r.rung, r.rows, r.us, r.gbs)
	}

	// The comparison the stage is about: the best rung of each bank at each
	// row count, and what the format bought.
	fmt.Printf("\n%6s %10s %10s %8s %8s\n", "rows", "fp16 us", "q8 us", "speedup", "q8 GB/s")
	for _, r := range rows {
		best := func(bank string) headRow {
			var b headRow
			for _, o := range out {
				if o.bank == bank && o.rows == r && (b.us == 0 || o.us < b.us) {
					b = o
				}
			}
			return b
		}
		a, b := best("fp16"), best("q8")
		fmt.Printf("%6d %10.1f %10.1f %7.2fx %8.1f\n", r, a.us, b.us, a.us/b.us, b.gbs)
	}

	if csvPath == "" {
		return nil
	}
	f, err := os.Create(csvPath)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	if err := w.Write([]string{"bank", "rung", "rows", "us", "gb_s", "bank_bytes"}); err != nil {
		return err
	}
	for _, r := range out {
		if err := w.Write([]string{r.bank, string(r.rung), strconv.Itoa(r.rows),
			fmt.Sprintf("%.1f", r.us), fmt.Sprintf("%.1f", r.gbs), strconv.Itoa(r.weight)}); err != nil {
			return err
		}
	}
	return nil
}
