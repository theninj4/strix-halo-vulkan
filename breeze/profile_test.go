package breeze

import (
	"fmt"
	"sort"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// profile runs g with a timestamp after every dispatch and sums the time by
// pipeline label: where a pass's GPU time goes (B7).
func profile(t *testing.T, d *Decoder, name string, g []vk.MultiDispatch) {
	labels := map[*vk.ComputePipeline]string{}
	for k, p := range d.pipes {
		labels[p] = k
	}
	for b := range d.banks {
		for k, p := range d.gemvs[b] {
			labels[p] = fmt.Sprintf("gemv%d %s", b, k)
		}
		for k, p := range d.gemms[b] {
			labels[p] = fmt.Sprintf("gemm%d %s", b, k)
		}
	}
	for _, s := range []*stack{d.bb, d.dd} {
		labels[s.pPrep], labels[s.pAttn], labels[s.pCombine] = s.name+" prep", s.name+" attn", s.name+" combine"
	}
	best := map[string]time.Duration{}
	var total time.Duration
	for rep := 0; rep < 5; rep++ {
		_, marks, err := vk.DispatchMultiMarked(g, true)
		if err != nil || marks == nil {
			t.Fatalf("marked dispatch: %v (%d dispatches)", err, len(g))
		}
		sum := map[string]time.Duration{}
		var tot time.Duration
		for i, m := range marks {
			sum[labels[g[i].Pipeline]] += m
			tot += m
		}
		if rep == 0 || tot < total {
			total, best = tot, sum
		}
	}
	keys := make([]string, 0, len(best))
	for k := range best {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return best[keys[i]] > best[keys[j]] })
	t.Logf("%s: %d dispatches, %v", name, len(g), total.Round(time.Microsecond))
	for _, k := range keys {
		t.Logf("  %-24s %8.1f us", k, float64(best[k].Nanoseconds())/1e3)
	}
}

// TestProfileStep times one decode step of each stack at the CFG pair's two
// rows, dispatch by dispatch.
func TestProfileStep(t *testing.T) {
	d := loadDecoder(t)
	rows := []Row{{0, 100}, {1, 90}}
	g, err := d.stackGraph(d.bb, rows, 0)
	if err != nil {
		t.Fatal(err)
	}
	g = append(g, d.proj(headBank, 1, d.bb.hA, d.bb.ldaDim, d.aLogits, d.lmHead, d.headPad, bbHidden, 2)...)
	profile(t, d, "backbone step", g)
	rows = []Row{{0, 7}, {1, 7}}
	if g, err = d.stackGraph(d.dd, rows, 0); err != nil {
		t.Fatal(err)
	}
	g = append(d.proj(headBank, 8, d.hIn, d.ldaIn, d.dd.aX, d.projIn, ddHidden, bbHidden, 2), g...)
	g = append(g, d.proj(headBank, 1, d.dd.hA, d.dd.ldaDim, d.aLogits, d.ddHeads, d.headPad, ddHidden, 2)...)
	profile(t, d, "depth step", g)
}

// TestSlabLadder times each projection of each stack at two rows, over every
// layer in one submit (so no weight is in a cache), at each split-K the bank
// has a build for: the table in Load is this test's argmin (B7).
func TestSlabLadder(t *testing.T) {
	d := loadDecoder(t)
	for _, s := range []*stack{d.bb, d.dd} {
		for p := proj(0); p < nProj; p++ {
			nOut, k := s.projShape(p)
			kt := k / tile
			line := fmt.Sprintf("%s %v [%d x %d]:", s.name, projParts[p], nOut, k)
			best, bestS := time.Duration(1<<62), 0
			for _, slabs := range []int{1, 2, 4, 8, 16, 32} {
				if kt%slabs != 0 || (kt/slabs)%4 != 0 {
					continue
				}
				var g []vk.MultiDispatch
				for i := range s.w {
					g = append(g, d.proj(layerBank, slabs, s.hA, s.ldaDim, s.aGU, s.w[i].off[p], nOut, k, 2)...)
				}
				if p == pDown {
					g = g[:0]
					for i := range s.w {
						g = append(g, d.proj(layerBank, slabs, s.hFFN, s.ldaF, s.aFF, s.w[i].off[p], nOut, k, 2)...)
					}
				}
				var tbest time.Duration
				for rep := 0; rep < 7; rep++ {
					dt, err := vk.DispatchMultiTimed(g, 1, 1, true)
					if err != nil {
						t.Fatal(err)
					}
					if rep == 0 || dt < tbest {
						tbest = dt
					}
				}
				per := tbest / time.Duration(len(s.w))
				bytes := float64(d.o.Bank.projBytes(nOut, k))
				line += fmt.Sprintf("  k%d %.1fus (%.0f GB/s)", slabs, float64(per.Nanoseconds())/1e3, bytes/per.Seconds()/1e9)
				if per < best {
					best, bestS = per, slabs
				}
			}
			t.Logf("%s -> k%d", line, bestS)
		}
	}
}
