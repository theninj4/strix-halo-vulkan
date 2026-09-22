package main

// Rate against context depth: what prompt processing and token generation
// cost when the cache is already full of conversation.
//
//	go run ./cmd/llm -depth                                  # 0..128000
//	go run ./cmd/llm -depth -depths 0,8000 -tg 32            # the short one
//	go run ./cmd/llm -depth -depths 0,4096 -layers 4         # the shape, in 7 GB
//
// This is llama-bench's `-d`, and the two rates it reports are the two
// questions of LLM.md L7c asked at a *position* rather than from cell zero.
// **pp** is one batch of -pp tokens run with the cache already holding the
// depth, and **tg** is a batch of one repeated -tg times on top of it. Neither
// is a whole prompt's rate: a client that sends a 128k prompt pays the sum of
// every batch from zero, and the sum is the area under this curve, not the
// last point on it.
//
// The depths are walked **forwards through one sequence** rather than
// re-prefilled one at a time. The nested depths share their prefix, so a
// sweep to 128k costs 128k tokens of prefill once instead of 248k, and the
// cache at depth D holds the same corpus tokens either way. The cost is that
// the timed batches of the lower depths are in the sequence the higher ones
// read — 576 tokens of it per depth, under half a percent at 128k, and it is
// the same text the fill would have put there.
//
// Generation is teacher-forced on the corpus for the same reason -ppl is: a
// decode step's cost is the cache and the weights, not which token came out,
// and feeding the corpus keeps positions aligned with the text so that the
// next depth's fill is a continuation rather than a second prompt.

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"strix-halo-vulkan/llm"
)

// depthOpts is what the -depth flags come to.
type depthOpts struct {
	model, file string
	depths      []int
	pp, tg      int
	ubatch      int // arena rows; 0 means -pp
	ctx, layers int
	csv         string
}

// depthRun is one depth's pair of measurements.
type depthRun struct {
	depth    int
	ppTok    int
	ppWall   time.Duration
	ppStats  llm.GraphStats
	tgTok    int
	tgWall   time.Duration
	tgStats  llm.GraphStats
	fillWall time.Duration
	filled   int
}

// depthBench stages the model once and walks one sequence, stopping at each
// depth to time a prompt batch and a decode run.
func depthBench(o depthOpts) error {
	if o.pp <= 0 || o.tg <= 0 {
		return fmt.Errorf("-pp and -tg must both be positive")
	}
	depths := append([]int(nil), o.depths...)
	sort.Ints(depths)
	deepest := depths[len(depths)-1]

	m, err := llm.Open(o.model)
	if err != nil {
		return err
	}
	defer m.Close()
	tok, err := llm.LoadTokenizer(m.Set)
	if err != nil {
		return err
	}

	// The corpus, not a repeated paragraph: the QSA selection only starts
	// excluding cells past its window and the MoE routing is text-shaped, so
	// a prompt that repeats itself measures a model nobody runs.
	buf, err := os.ReadFile(o.file)
	if err != nil {
		return err
	}
	ids, err := tok.Encode(string(buf))
	if err != nil {
		return err
	}
	// Every depth spends its own pp batch and tg run inside the sequence, so
	// the tokens the walk consumes are the deepest depth plus one pair per
	// depth below it.
	want := deepest + len(depths)*(o.pp+o.tg)
	if len(ids) < want {
		return fmt.Errorf("%s tokenizes to %d tokens, this sweep needs %d", o.file, len(ids), want)
	}
	ctx := o.ctx
	if ctx < want {
		ctx = want
	}

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()

	fmt.Printf("%s\n%d layers, %d cells of context, pp %d, tg %d, depths %v\n",
		o.model, m.Config.NLayer, ctx, o.pp, o.tg, depths)

	ubatch := o.pp
	if o.ubatch > 0 {
		if o.ubatch < o.pp {
			return fmt.Errorf("-ubatch %d is narrower than -pp %d", o.ubatch, o.pp)
		}
		ubatch = o.ubatch
		fmt.Printf("arenas sized for %d rows\n", ubatch)
	}

	start := time.Now()
	g, err := llm.NewGraph(dev, m, llm.GraphOpts{MaxTokens: ubatch, NKV: ctx, Layers: o.layers})
	if err != nil {
		return err
	}
	defer g.Destroy()
	reportStaging(g, time.Since(start))

	// One thrown-away pass before anything is timed. Nothing of the 80 GB is
	// in any cache on the first pass and the Go heap has not yet grown to
	// hold a run's host buffers, so depth 0 would otherwise measure staging.
	if _, _, err := g.Forward(ids[:o.pp]); err != nil {
		return err
	}
	for i := 0; i < 4; i++ {
		if _, _, err := g.Extend(ids[o.pp+i : o.pp+i+1]); err != nil {
			return err
		}
	}
	if err := g.Reset(); err != nil {
		return err
	}

	var runs []depthRun
	pos, fresh := 0, true
	for _, d := range depths {
		// Fill the cache up to the depth, untimed, in batches the arenas
		// hold. This is the bulk of the wall clock at 128k and it is not a
		// measurement — the timed batch below is.
		r := depthRun{depth: d}
		t0 := time.Now()
		for pos < d {
			n := min(o.pp, d-pos)
			if err := depthRunBatch(g, ids[pos:pos+n], &fresh); err != nil {
				return err
			}
			pos += n
			if pos%16384 < o.pp {
				fmt.Printf("  filled %d/%d tokens in %v\n", pos, d, time.Since(t0).Round(time.Second))
			}
		}
		r.fillWall, r.filled = time.Since(t0), pos

		// Prompt processing at the depth: one batch, timed.
		g.ResetStats()
		t0 = time.Now()
		if err := depthRunBatch(g, ids[pos:pos+o.pp], &fresh); err != nil {
			return err
		}
		r.ppWall, r.ppStats, r.ppTok = time.Since(t0), g.Stats.Clone(), o.pp
		reportSelSkip(g)
		pos += o.pp

		// Token generation at the depth: a batch of one, -tg times.
		g.ResetStats()
		t0 = time.Now()
		for i := 0; i < o.tg; i++ {
			if err := depthRunBatch(g, ids[pos+i:pos+i+1], &fresh); err != nil {
				return err
			}
		}
		r.tgWall, r.tgStats, r.tgTok = time.Since(t0), g.Stats.Clone(), o.tg
		pos += o.tg

		runs = append(runs, r)
		reportDepthRun(r)
	}

	reportDepthTable(runs)
	if o.csv == "" {
		return nil
	}
	if err := writeCSV(o.csv, depthCSV(runs)); err != nil {
		return err
	}
	return writeCSV(depthLabelPath(o.csv), depthLabelCSV(runs))
}

// depthLabelPath is the main CSV's name with `-labels` before the extension,
// so a sweep writes its two tables side by side under one -csv flag.
func depthLabelPath(csv string) string {
	ext := filepath.Ext(csv)
	return strings.TrimSuffix(csv, ext) + "-labels" + ext
}

// depthRunBatch runs one batch, resetting the sequence for the very first one
// and extending it for every batch after.
func depthRunBatch(g *llm.Graph, ids []int32, fresh *bool) error {
	var err error
	if *fresh {
		_, _, err = g.Forward(ids)
		*fresh = false
	} else {
		_, _, err = g.Extend(ids)
	}
	return err
}

// reportDepthRun prints one depth as it finishes, so a sweep that takes
// minutes to reach 128k can be read while it runs.
func reportDepthRun(r depthRun) {
	fmt.Printf("\ndepth %d — filled to %d in %v\n", r.depth, r.filled, r.fillWall.Round(time.Millisecond))
	fmt.Printf("  pp %d tokens in %v — %.1f tok/s\n",
		r.ppTok, r.ppWall.Round(time.Millisecond), rate(r.ppTok, r.ppWall))
	reportPhases(r.ppStats, r.ppWall)
	fmt.Printf("  tg %d tokens in %v — %.2f tok/s, %.1f ms a token\n",
		r.tgTok, r.tgWall.Round(time.Millisecond), rate(r.tgTok, r.tgWall),
		float64(r.tgWall.Microseconds())/1000/float64(r.tgTok))
	reportPhases(r.tgStats, r.tgWall)
	fmt.Println()
}

// reportDepthTable is the summary: both rates against depth, and each one
// against its own depth-zero value, which is the number the question asks for.
func reportDepthTable(runs []depthRun) {
	if len(runs) == 0 {
		return
	}
	pp0, tg0 := rate(runs[0].ppTok, runs[0].ppWall), rate(runs[0].tgTok, runs[0].tgWall)
	fmt.Printf("\n%8s %12s %9s %12s %9s %10s\n", "depth", "pp tok/s", "vs d0", "tg tok/s", "vs d0", "tg ms/tok")
	for _, r := range runs {
		pp, tg := rate(r.ppTok, r.ppWall), rate(r.tgTok, r.tgWall)
		fmt.Printf("%8d %12.1f %8.2fx %12.2f %8.2fx %10.1f\n",
			r.depth, pp, pp/pp0, tg, tg/tg0, float64(r.tgWall.Microseconds())/1000/float64(r.tgTok))
	}
	fmt.Println()
}

// depthCSV is the table above plus where each measurement's wall clock went,
// one row a depth a phase.
func depthCSV(runs []depthRun) [][]string {
	rows := [][]string{{"depth", "phase", "tokens", "ms", "tok_s", "ms_per_token",
		"hc_ms", "ple_ms", "dn_ms", "attn_ms", "moe_ms", "head_ms", "move_ms", "gather_ms", "glue_ms", "gpu_ms"}}
	row := func(d int, phase string, n int, wall time.Duration, st llm.GraphStats) []string {
		ms := func(x time.Duration) string {
			return fmt.Sprintf("%.3f", float64(x.Microseconds())/1000/float64(max(n, 1)))
		}
		return []string{
			fmt.Sprint(d), phase, fmt.Sprint(n),
			fmt.Sprintf("%.1f", float64(wall.Microseconds())/1000),
			fmt.Sprintf("%.2f", rate(n, wall)),
			fmt.Sprintf("%.3f", float64(wall.Microseconds())/1000/float64(max(n, 1))),
			ms(st.HC), ms(st.PLE), ms(st.DeltaNet), ms(st.Attn), ms(st.MoE),
			ms(st.Head), ms(st.Move), ms(st.Gather), ms(st.Glue), ms(st.GPU),
		}
	}
	for _, r := range runs {
		rows = append(rows, row(r.depth, "pp", r.ppTok, r.ppWall, r.ppStats))
		rows = append(rows, row(r.depth, "tg", r.tgTok, r.tgWall, r.tgStats))
	}
	return rows
}

// depthLabelCSV is where a *decode step* went by dispatch label, one row a
// depth a label, sorted by cost within each depth.
//
// The block columns of depthCSV say attention owns the falloff; they cannot
// say which of the full-attention layer's six dispatches owns it, and after
// P7 that is the whole remaining question — the indexer scores every block,
// the selection radix-sorts every live cell, and the flash kernel walks every
// key block to test it, so three different O(depth) terms hide inside one
// `attn_ms`. This is the resolution that separates them.
func depthLabelCSV(runs []depthRun) [][]string {
	rows := [][]string{{"depth", "phase", "label", "n_per_pass", "ms_per_token", "us_each"}}
	add := func(d int, phase string, n int, st llm.GraphStats) {
		type row struct {
			name string
			s    llm.DispatchStat
		}
		rs := make([]row, 0, len(st.Kinds))
		for k, v := range st.Kinds {
			rs = append(rs, row{k, v})
		}
		sort.Slice(rs, func(i, j int) bool { return rs[i].s.GPU > rs[j].s.GPU })
		passes := float64(max(st.Runs, 1))
		for _, r := range rs {
			rows = append(rows, []string{
				fmt.Sprint(d), phase, r.name,
				fmt.Sprintf("%.1f", float64(r.s.Count)/passes),
				fmt.Sprintf("%.4f", float64(r.s.GPU.Microseconds())/1000/float64(max(n, 1))),
				fmt.Sprintf("%.1f", float64(r.s.GPU.Nanoseconds())/1000/float64(max(r.s.Count, 1))),
			})
		}
	}
	for _, r := range runs {
		add(r.depth, "pp", r.ppTok, r.ppStats)
		add(r.depth, "tg", r.tgTok, r.tgStats)
	}
	return rows
}

// rate is tokens a second, and zero for a run that took no measurable time.
func rate(n int, d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return float64(n) / d.Seconds()
}

// reportSelSkip prints how much of the key axis the attention kernel's block
// skip could skip on the selection the batch just left behind (P11).
//
// At prefill this is the whole question. Every other block of the model is
// flat in depth; attention is not, and what decides *its* slope is not the
// selection's density — 2051 cells however deep the cache, which at 64k is
// 3.2% — but the density of the **union** over the sixteen adjacent queries a
// cooperative-matrix fragment covers, because that is the granularity the
// kernel can skip at. The two columns are that union and the per-row floor a
// gather would reach, so the gap between them is what a restructuring is
// worth before anyone writes one.
func reportSelSkip(g *llm.Graph) {
	a := g.Attn()
	if a == nil || !a.Sparse() {
		return
	}
	mask := a.Selection()
	blk16, uni, row := a.SelUnion(mask, 16, 16)
	fmt.Printf("    selection live %%, (query tile) x (key block):\n")
	fmt.Printf("      %-6s %8s %8s %8s %8s\n", "bm\\bn", "16", "32", "64", "row")
	for _, bm := range []int{16, 32, 64} {
		fmt.Printf("      %-6d", bm)
		for _, bn := range []int{16, 32, 64} {
			live, _ := a.SelSkip(mask, bm, bn)
			fmt.Printf(" %7.1f%%", 100*live)
		}
		_, perRow := a.SelSkip(mask, bm, 64)
		fmt.Printf(" %7.1f%%\n", 100*perRow)
	}
	// And the same question in **cells**, which is what the gather is priced
	// in: the shipped kernel visits `live% x nKV` of them, a per-cell gather
	// over the tile's union visits `union`, and one query's own selection is
	// the floor a sixteen-row fragment cannot reach.
	fmt.Printf("      cells a 16-row tile: blocks(bn=16) %.0f  union %.0f  row %.0f  -> %.2fx\n",
		blk16, uni, row, blk16/math.Max(uni, 1))
}
