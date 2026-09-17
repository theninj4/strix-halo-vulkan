package main

// The lm head's benchmark: LLM.md L8a, the first dense bank that is not
// halves — and L8c-4, the first that is not the checkpoint's arithmetic.
//
//	go run ./cmd/llm -head                          # every bank, every rung, 1..512 rows
//	go run ./cmd/llm -head -tokens 1                # the shape the model actually runs
//	go run ./cmd/llm -head -csv results/l8c_head.csv
//
// `output.weight` is [2560, 248320] Q8_0 — 675 MB in the checkpoint, 1.27 GB
// as the halves every GEMM in this vertical read before L8, 0.68 GB as the
// int8-plus-scales bank that replaced them and **0.36 GB** as L8c-4's 4.5-bit
// K-quant. Each is staged in the same process against the same activation, so
// the only thing that differs is the format and the kernel build that reads
// it.
//
// Four arms, not three, and the fourth is the gate. `sim` is the fp16 bank
// staged through `llm/sim.go`'s round trip at `q4_k/32` — the format L8c-3
// measured and this stage built a kernel for — so the real bank's logits and
// the simulation's are produced side by side. They have to be **identical**:
// both encoders are `quantk.go`'s, the value staged is `d*sc*l - dmin*m` in
// f32 either way, and the GEMM's LDS store rounds it to the same half the
// host's `tileB` writes. A tolerance here would hide an addressing bug, which
// is the only kind of bug this bank can have.
//
// What is timed is GPU dispatch time over `-iters` back-to-back repetitions.
// There is no sweep over staged copies as -hc has: the smallest bank here is
// 0.36 GB and the MALL is 32 MiB, so nothing is ever warm.

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"strix-halo-vulkan/llm"
)

type headRow struct {
	bank   string
	rung   string
	rows   int
	us     float64
	gbs    float64
	weight int
}

// mib prints a byte count the way the resident plan does.
func mib(n int) string { return fmt.Sprintf("%.2f GB", float64(n)/1e9) }

// headArm is one staging of the head: which bank, and which format if the
// bank or the simulation wants one.
type headArm struct {
	name string
	bank llm.DenseBank
	spec string // "" for no simulation and no format
}

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

	// One activation for every bank, so a disagreement could only be the
	// kernel's. It is the same shape `result_norm` has: one row per token.
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, maxRows*c.NEmbd)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	arms := []headArm{
		{"fp16", llm.BankFP16, ""},
		{"q8", llm.BankQ8, ""},
		{"q4_k", llm.BankQ4K, "q4_k/32"},
		{"sim", llm.BankFP16, "q4_k/32"},
	}
	var out []headRow
	logits := make(map[string][]float32, len(arms))
	for _, arm := range arms {
		var sim llm.QuantSim
		if arm.spec != "" {
			if sim, err = llm.ParseQuantSim(arm.spec); err != nil {
				return err
			}
			// `lm_head` has no entry in unsloth's matrix, so this falls back
			// to round-to-nearest the way ggml does — which is what L8c-3's
			// +0.40% for this family was measured at.
			sim.Mode = "imatrix"
		}
		start := time.Now()
		g, err := llm.NewHeadGPUBank(dev, c.NEmbd, t, maxRows, arm.bank, sim)
		if err != nil {
			return fmt.Errorf("head (%s): %w", arm.name, err)
		}
		fmt.Printf("%-5s bank %s staged in %s\n", arm.name, mib(g.WeightBytes()),
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
				out = append(out, headRow{arm.name, string(k), r, us,
					float64(g.WeightBytes()) / (us * 1e-6) / 1e9, g.WeightBytes()})
			}
			// The decode kernel, where there is one row for it to read. At
			// N = 248320 the output width alone is 15520 workgroups, so the
			// rung is KSLABS = 1 and there are no partials to reduce (D11).
			if r == 1 && arm.bank != llm.BankFP16 {
				if err := g.SetGEMV(llm.GEMVK1); err != nil {
					g.Destroy()
					return err
				}
				st, err := g.Profile(iters)
				if err != nil {
					g.Destroy()
					return err
				}
				us := float64(st[0].GPU.Nanoseconds()) / 1e3
				out = append(out, headRow{arm.name, "gemv_k1", r, us,
					float64(g.WeightBytes()) / (us * 1e-6) / 1e9, g.WeightBytes()})
				if err := g.Run(); err != nil {
					g.Destroy()
					return err
				}
				logits[arm.name+"/gemv"] = append([]float32(nil), g.Logits()[:g.Vocab()]...)
				if err := g.SetGEMV(llm.GEMVOff); err != nil {
					g.Destroy()
					return err
				}
				// The GEMM at the same one row, so the pair below is a
				// comparison of kernels and not of row counts.
				if err := g.SetKernel(llm.GEMMKernels()[0]); err != nil {
					g.Destroy()
					return err
				}
				if err := g.Run(); err != nil {
					g.Destroy()
					return err
				}
				logits[arm.name+"/gemm1"] = append([]float32(nil), g.Logits()[:g.Vocab()]...)
			}
			if r == rows[0] {
				// The gate. fp16 against q8 is L8a's — the same numbers, so
				// the same logits — and q4_k against sim is L8c-4's.
				if err := g.SetKernel(llm.GEMMKernels()[0]); err != nil {
					g.Destroy()
					return err
				}
				if err := g.Run(); err != nil {
					g.Destroy()
					return err
				}
				logits[arm.name] = append([]float32(nil), g.Logits()[:g.Vocab()]...)
			}
		}
		g.Destroy()
	}

	fmt.Println()
	for _, p := range [][2]string{{"fp16", "q8"}, {"sim", "q4_k"}} {
		a, b := logits[p[0]], logits[p[1]]
		if a == nil || b == nil {
			continue
		}
		diff, worst := 0, float64(0)
		for i := range a {
			if a[i] != b[i] {
				diff++
				if d := math.Abs(float64(a[i] - b[i])); d > worst {
					worst = d
				}
			}
		}
		fmt.Printf("row 0 logits, %s vs %s: %d of %d differ (worst %.3g)\n",
			p[0], p[1], diff, len(a), worst)
	}
	// The GEMV is not the GEMM's arithmetic on this bank and says so: a
	// K-quant group is affine and the subtraction is done in f32 here where
	// the GEMM rounds it to a half in LDS, so the GEMV carries one fewer
	// rounding per weight. llm_moe_gemv.comp is in exactly this position
	// against llm_moe_gemm.comp, and the number is what makes it a floor
	// rather than a hope.
	for _, bank := range []string{"q8", "q4_k"} {
		a, b := logits[bank+"/gemm1"], logits[bank+"/gemv"]
		if a == nil || b == nil {
			continue
		}
		var num, den float64
		diff := 0
		for i := range a {
			d := float64(a[i] - b[i])
			num += d * d
			den += float64(a[i]) * float64(a[i])
			if a[i] != b[i] {
				diff++
			}
		}
		fmt.Printf("row 0 logits, %s GEMM vs GEMV: %d of %d differ, rms %.3g relative\n",
			bank, diff, len(a), math.Sqrt(num/den))
	}

	fmt.Printf("\n%-5s %-8s %6s %10s %8s\n", "bank", "rung", "rows", "us", "GB/s")
	for _, r := range out {
		fmt.Printf("%-5s %-8s %6d %10.1f %8.1f\n", r.bank, r.rung, r.rows, r.us, r.gbs)
	}

	// The comparison the stage is about: the best rung of each bank at each
	// row count, and what the format bought.
	best := func(bank string, r int) headRow {
		var b headRow
		for _, o := range out {
			if o.bank == bank && o.rows == r && (b.us == 0 || o.us < b.us) {
				b = o
			}
		}
		return b
	}
	fmt.Printf("\n%6s %10s %10s %10s %9s %9s %9s\n",
		"rows", "fp16 us", "q8 us", "q4_k us", "q8/fp16", "q4/fp16", "q4 GB/s")
	for _, r := range rows {
		a, b, q := best("fp16", r), best("q8", r), best("q4_k", r)
		fmt.Printf("%6d %10.1f %10.1f %10.1f %8.2fx %8.2fx %9.1f\n",
			r, a.us, b.us, q.us, a.us/b.us, a.us/q.us, q.gbs)
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
		if err := w.Write([]string{r.bank, r.rung, strconv.Itoa(r.rows),
			fmt.Sprintf("%.1f", r.us), fmt.Sprintf("%.1f", r.gbs), strconv.Itoa(r.weight)}); err != nil {
			return err
		}
	}
	return nil
}
