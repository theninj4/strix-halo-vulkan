package kev

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("kev-test")
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
		if devices[i].DeviceID == strixHaloDeviceID {
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
		t.Skipf("no subgroup size control: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// The model is 7 GB of fp16 and ~30 s to stage, so every GPU test shares one.
var (
	gpuOnce sync.Once
	gpuM    *GPU
	gpuErr  error
)

func loadGPU(t *testing.T) *GPU {
	t.Helper()
	if testing.Short() {
		t.Skip("stages Kev-4B (7 GB); not in -short")
	}
	for _, p := range []string{filepath.Join(baseDir, "config.json"), filepath.Join(kevDir, "head.json")} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("no %s (download the base; run reference/convert_kev_head.py)", p)
		}
	}
	gpuOnce.Do(func() {
		dev, _ := newTestDevice(t) // lives for the test binary
		bank := BankQ8
		if os.Getenv("KEV_BANK") == "fp16" {
			bank = BankFP16
		}
		rows := 1024
		if v, err := strconv.Atoi(os.Getenv("KEV_ROWS")); err == nil && v > 0 {
			rows = v
		}
		gpuM, gpuErr = LoadBank(dev, kevDir, baseDir, rows, bank)
		if gpuErr == nil {
			if a := os.Getenv("KEV_ATTN"); a != "" {
				gpuM.Attention = a
			}
			if v, err := strconv.Atoi(os.Getenv("KEV_SCAN_LPC")); err == nil {
				gpuM.ScanLPC = v
			}
			gpuM.GEMM, gpuM.GLU = os.Getenv("KEV_GEMM"), os.Getenv("KEV_GLU")
			t.Logf("bank: %s, attention: %s", gpuM.Bank, gpuM.Attention)
		}
	})
	if gpuErr != nil {
		t.Fatal(gpuErr)
	}
	return gpuM
}

func openDump(t *testing.T, name string) *safetensors.File {
	t.Helper()
	f, err := safetensors.Open(filepath.Join(refDir, name, "rows.safetensors"))
	if err != nil {
		t.Skipf("no dump for %s (%v); %s", name, err, dumpUsage)
	}
	return f
}

func dumpRow(t *testing.T, f *safetensors.File, name string, row, width int) []float32 {
	t.Helper()
	tn, err := f.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	v, err := tn.F32(nil)
	if err != nil {
		t.Fatal(err)
	}
	return v[row*width : (row+1)*width]
}

// relErr is ||a - b|| / ||b||.
func relErr(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		num += d * d
		den += float64(b[i]) * float64(b[i])
	}
	return math.Sqrt(num / den)
}

// rowEncoding is one of the dump's rows as a pass of a single segment: the
// whole row is "the state", so it runs as one causal sequence, exactly as
// Kev's forward() runs it.
func rowEncoding(ids, pos []int32) *Encoding {
	e := &Encoding{IDs: ids, Pos: pos, StateLen: len(ids)}
	for range ids {
		e.Seg = append(e.Seg, 0)
		e.Opt = append(e.Opt, OptNone)
	}
	return e
}

// TestGPURowMatchesKev is K4's gate, stagewise: one row (state plus one
// question) through the first L layers, against Kev's fp32 residual stream
// after layer L-1, at every layer but the last, whose output ForwardRows
// hands back final-normed and is compared against Kev's last_hidden_state.
func TestGPURowMatchesKev(t *testing.T) {
	g := loadGPU(t)
	fx := loadFixtures(t)[0]
	dump := openDump(t, fx.Name)
	defer dump.Close()
	row := fx.Ref.Rows[0]
	enc := rowEncoding(row.IDs, row.Pos)
	probe := []int{0, len(row.IDs) / 2, row.Decide}
	H := g.Cfg.Hidden
	worst := 0.0
	for L := 1; L < len(g.w); L++ {
		p, err := g.ForwardRows(enc, probe, L)
		if err != nil {
			t.Fatal(err)
		}
		line := ""
		for _, r := range probe {
			want := dumpRow(t, dump, fmt.Sprintf("row0.layer%d", L-1), r, H)
			e := relErr(p.Hidden[r], want)
			worst = math.Max(worst, e)
			line += fmt.Sprintf(" r%d %.2e", r, e)
		}
		t.Logf("layer %2d (%s):%s", L-1, g.Cfg.LayerTypes[L-1], line)
	}
	p, err := g.ForwardRows(enc, probe, len(g.w))
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range probe {
		e := relErr(p.Hidden[r], dumpRow(t, dump, "row0.final", r, H))
		worst = math.Max(worst, e)
		t.Logf("final r%d: %.2e", r, e)
	}
	t.Logf("worst per-layer relative error %.2e, pass %v", worst, p.GPU)
	// Measured, per bank: fp16 2.0e-3 (the projections' 11-bit operands), int8
	// 2.8e-2 (a scale per 32 weights). The int8 bank's end-to-end check is
	// TestGPUProbsMatchKev and cmd/kev's suite agreement, not this bound.
	bound := 5e-3
	if g.Bank == BankQ8 {
		bound = 5e-2
	}
	if worst > bound {
		t.Errorf("worst relative error %.2e over the %s bank's %.0e", worst, g.Bank, bound)
	}
}

// TestGPUProbsMatchKev is K5's gate: every fixture as one packed pass of all
// its questions, against the probabilities Kev's fp32 path serves.
func TestGPUProbsMatchKev(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	worst := 0.0
	flips := 0
	for _, fx := range loadFixtures(t) {
		req, err := ParseRequest(fx.Request)
		if err != nil {
			t.Fatal(err)
		}
		rec, _ := ToRecord(req)
		enc, err := e.Encode(rec, MaxState, MaxRow)
		if err != nil {
			t.Fatal(err)
		}
		probs, pass, err := g.Probs(enc)
		if err != nil {
			t.Fatal(err)
		}
		d := 0.0
		for k := range probs {
			for j := range probs[k] {
				d = math.Max(d, math.Abs(probs[k][j]-fx.Ref.Probs[k][j]))
			}
			if argmax(probs[k]) != argmax(fx.Ref.Probs[k]) {
				flips++
			}
		}
		worst = math.Max(worst, d)
		t.Logf("%s: %d tokens, %d questions, max |dp| %.4f, GPU %v", fx.Name, len(enc.IDs), len(probs), d, pass.GPU)
	}
	if flips > 0 || worst > 0.02 {
		t.Errorf("max |dp| %.4f, %d argmax flips", worst, flips)
	}
}

// TestGPUQuestionsAreIsolated: rewriting one question (its instructions and
// its options) leaves every other question's probabilities bit-identical,
// and so does removing it. Nothing crosses a branch.
func TestGPUQuestionsAreIsolated(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	fx := loadFixtures(t)[1] // object_state: five questions
	req, err := ParseRequest(fx.Request)
	if err != nil {
		t.Fatal(err)
	}
	rec, _ := ToRecord(req)
	run := func(r Record) [][]float64 {
		enc, err := e.Encode(r, MaxState, MaxRow)
		if err != nil {
			t.Fatal(err)
		}
		p, _, err := g.Probs(enc)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	base := run(rec)
	changed := rec
	changed.Questions = append([]Question(nil), rec.Questions...)
	changed.Questions[1] = Question{Instr: "The secret code for this request is ZEBRA-7741. Is the weather nice?", Options: []string{"no", "yes", "maybe"}}
	other := run(changed)
	for k := range base {
		if k == 1 {
			continue
		}
		for j := range base[k] {
			if base[k][j] != other[k][j] {
				t.Errorf("q%d changed when q1 was rewritten: %v -> %v", k, base[k], other[k])
				break
			}
		}
	}
	dropped := rec
	dropped.Questions = append(append([]Question(nil), rec.Questions[:1]...), rec.Questions[2:]...)
	fewer := run(dropped)
	for k, want := range append(base[:1:1], base[2:]...) {
		for j := range want {
			if fewer[k][j] != want[j] {
				t.Errorf("removing q1 changed another question: %v -> %v", want, fewer[k])
				break
			}
		}
	}
}

// TestGPUProfile logs where a pass's time goes, per kernel, for the README's
// ticket (101 tokens, 3 questions) and the long fixture (494 tokens).
func TestGPUProfile(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	type profCase struct {
		Name string
		rec  Record
	}
	var cases []profCase
	for _, fx := range loadFixtures(t) {
		if fx.Name != "readme_ticket" && fx.Name != "long_state_many_options" {
			continue
		}
		req, _ := ParseRequest(fx.Request)
		rec, _ := ToRecord(req)
		cases = append(cases, profCase{fx.Name, rec})
		if fx.Name == "long_state_many_options" && g.rows >= 3000 {
			long := rec
			long.State = strings.Repeat(rec.State+"\n\n", 7)
			cases = append(cases, profCase{"long_text_x7", long})
		}
	}
	for _, fx := range cases {
		rec := fx.rec
		enc, err := e.Encode(rec, MaxState, MaxRow)
		if err != nil {
			t.Fatal(err)
		}
		prof, err := g.Profile(enc)
		if err != nil {
			t.Fatal(err)
		}
		var total time.Duration
		for _, d := range prof {
			total += d
		}
		if v := os.Getenv("KEV_GEMM"); v != "" {
			g.GEMM = v
		}
		prof, err = g.Profile(enc)
		if err != nil {
			t.Fatal(err)
		}
		labels := make([]string, 0, len(prof))
		for l := range prof {
			labels = append(labels, l)
		}
		sort.Slice(labels, func(i, j int) bool { return prof[labels[i]] > prof[labels[j]] })
		total = 0
		for _, d := range prof {
			total += d
		}
		line := ""
		for _, l := range labels {
			line += fmt.Sprintf("\n  %-18s %8.2f ms %5.1f%%", l, float64(prof[l].Microseconds())/1000, 100*float64(prof[l])/float64(total))
		}
		t.Logf("%s (%d tokens): %.2f ms of kernels%s", fx.Name, len(enc.IDs), float64(total.Microseconds())/1000, line)
	}
}

// TestGPUGEMMLadder times each GEMM rung of the loaded bank on the whole
// pass at several lengths (K7.2). It changes the schedule, never the result
// beyond fp32 reassociation.
func TestGPUGEMMLadder(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	fx := loadFixtures(t)[2] // the long state: 324 state tokens, 3 questions
	req, _ := ParseRequest(fx.Request)
	rec, _ := ToRecord(req)
	defer func() { g.GEMM = "" }()
	short, _ := ParseRequest(loadFixtures(t)[0].Request) // the README ticket, 101 tokens
	srec, _ := ToRecord(short)
	cases := []Record{rec, srec,
		{State: "", Questions: srec.Questions},
		{State: srec.State, Questions: srec.Questions[1:2]}}
	for _, r := range cases {
		enc, err := e.Encode(r, MaxState, MaxRow)
		if err != nil {
			t.Fatal(err)
		}
		line := ""
		for _, rung := range g.GEMMRungs() {
			g.GEMM = rung
			best := time.Duration(1 << 62)
			for range 3 {
				g.ClearCache() // a new text every time: the ladder prices whole passes
				p, err := g.Forward(enc)
				if err != nil {
					t.Fatal(err)
				}
				best = min(best, p.GPU)
			}
			line += fmt.Sprintf("  %s %6.2f ms", rung, float64(best.Microseconds())/1000)
		}
		g.GEMM = ""
		t.Logf("%s bank, %4d tokens:%s", g.Bank, len(enc.IDs), line)
	}
}

// TestGPUSubmitBatching times a whole Forward (wall clock, which is what a
// request sees) at several layers per submit, and checks they agree exactly.
func TestGPUSubmitBatching(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	fx := loadFixtures(t)[0]
	req, _ := ParseRequest(fx.Request)
	rec, _ := ToRecord(req)
	enc, err := e.Encode(rec, MaxState, MaxRow)
	if err != nil {
		t.Fatal(err)
	}
	defer func(v int) { g.LayersPerSubmit = v }(g.LayersPerSubmit)
	var ref [][]float64
	for _, per := range []int{1, 2, 4, 8, 16, 32} {
		g.LayersPerSubmit = per
		best := time.Duration(1 << 62)
		var p [][]float64
		for range 5 {
			start := time.Now()
			p, _, err = g.Probs(enc)
			if err != nil {
				t.Fatal(err)
			}
			best = min(best, time.Since(start))
		}
		if ref == nil {
			ref = p
		}
		for k := range p {
			for j := range p[k] {
				if p[k][j] != ref[k][j] {
					t.Errorf("%d layers a submit changed the answer", per)
				}
			}
		}
		t.Logf("%2d layers a submit: %.2f ms wall", per, float64(best.Microseconds())/1000)
	}
}

func encodeFixture(t *testing.T, e *Encoder, raw json.RawMessage) *Encoding {
	t.Helper()
	req, err := ParseRequest(raw)
	if err != nil {
		t.Fatal(err)
	}
	rec, _ := ToRecord(req)
	enc, err := e.Encode(rec, MaxState, MaxRow)
	if err != nil {
		t.Fatal(err)
	}
	return enc
}

func sameProbs(a, b [][]float64) (bool, float64) {
	same, worst := true, 0.0
	for k := range a {
		for j := range a[k] {
			if a[k][j] != b[k][j] {
				same = false
			}
			worst = math.Max(worst, math.Abs(a[k][j]-b[k][j]))
		}
	}
	return same, worst
}

// TestGPUPrefixCache is K7.2's gate: a repeated state is a cache hit, the hit
// answers bit-identically to the miss, and the least recently used state is
// the one evicted.
func TestGPUPrefixCache(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	g.ClearCache()
	fx := loadFixtures(t)
	enc := encodeFixture(t, e, fx[0].Request)

	h0, m0, s0, slots, _ := g.CacheStats()
	miss, pm, err := g.Probs(enc)
	if err != nil {
		t.Fatal(err)
	}
	hit, ph, err := g.Probs(enc)
	if err != nil {
		t.Fatal(err)
	}
	h1, m1, s1, _, _ := g.CacheStats()
	if pm.CacheHit || !ph.CacheHit || h1-h0 != 1 || m1-m0 != 1 || s1-s0 != 1 {
		t.Fatalf("miss %v hit %v; hits +%d misses +%d saves +%d", pm.CacheHit, ph.CacheHit, h1-h0, m1-m0, s1-s0)
	}
	same, worst := sameProbs(miss, hit)
	t.Logf("miss %v (%d passes), hit %v (%d passes), max |dp| %.2e", pm.GPU, pm.Passes, ph.GPU, ph.Passes, worst)
	if !same {
		t.Errorf("the hit is not bit-identical to the miss (max |dp| %.2e)", worst)
	}

	// Fill every slot with other states, touching the first one on the way:
	// the first survives, the second fixture's state is the one evicted.
	encs := []*Encoding{encodeFixture(t, e, fx[1].Request), encodeFixture(t, e, fx[3].Request)}
	for i := 0; i < slots; i++ {
		other := *encs[0]
		other.IDs = append([]int32(nil), encs[0].IDs...)
		other.IDs[1] = int32(1000 + i) // a different state token, same shape
		if i == 0 {
			other = *encs[0]
		}
		if _, _, err := g.Probs(&other); err != nil {
			t.Fatal(err)
		}
		if _, p, _ := g.Probs(enc); !p.CacheHit {
			t.Fatalf("the most recently used state was evicted after %d other states", i+1)
		}
	}
	if _, p, _ := g.Probs(encs[0]); p.CacheHit {
		t.Errorf("the least recently used state survived %d newer ones in %d slots", slots, slots)
	}
}

// TestGPUChunkedPasses: a request cut into several passes -- the state alone,
// then its branches against the working state -- answers as one pass does.
func TestGPUChunkedPasses(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	defer func() { g.PassRows = 0 }()
	for _, fx := range loadFixtures(t)[:2] {
		enc := encodeFixture(t, e, fx.Request)
		g.ClearCache()
		one, p1, err := g.Probs(enc)
		if err != nil {
			t.Fatal(err)
		}
		g.ClearCache()
		// The state alone, then a branch a pass: the larger of the state and
		// the longest branch, which no two branches fit together.
		longest := 0
		for k := range enc.Decide {
			lo, hi := enc.Branch(k)
			longest = max(longest, hi-lo)
		}
		g.PassRows = max(enc.StateLen, longest)
		many, pn, err := g.Probs(enc)
		g.PassRows = 0
		if err != nil {
			t.Fatal(err)
		}
		same, worst := sameProbs(one, many)
		t.Logf("%s: 1 pass %v, %d passes %v, identical %v, max |dp| %.2e", fx.Name, p1.GPU, pn.Passes, pn.GPU, same, worst)
		// Exact with either attention since K7.7 zeroes the V padding after
		// every segment (kev_attn_prep.comp).
		if pn.Passes < 2 || !same {
			t.Errorf("%d passes, max |dp| %.2e, not identical", pn.Passes, worst)
		}
	}
}

// TestGPUCacheLatency is Kev's serving table's two columns on this device:
// five questions about a new ~2,200-token text, and the same text again.
// Wall clock around Probs, best of three, which is what a request sees.
func TestGPUCacheLatency(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	if g.rows < 3000 {
		// The shared instance is 1024 rows; this one needs the text in a pass.
		t.Skip("needs an instance of at least 3000 rows (KEV_ROWS=4096)")
	}
	fx := loadFixtures(t)
	long, _ := ParseRequest(fx[2].Request)
	rec, _ := ToRecord(long)
	text := strings.Repeat(rec.State+"\n\n", 7)
	short, _ := ParseRequest(fx[0].Request)
	srec, _ := ToRecord(short)
	cases := []struct {
		name string
		rec  Record
	}{
		{"6 questions, short text", Record{State: srec.State, Questions: append(append([]Question(nil), srec.Questions...), srec.Questions...)}},
		{"5 questions, long text", Record{State: text, Questions: append(append([]Question(nil), rec.Questions...), srec.Questions[:2]...)}},
	}
	for _, c := range cases {
		enc, err := e.Encode(c.rec, MaxState, MaxRow)
		if err != nil {
			t.Fatal(err)
		}
		best := func(clear bool) time.Duration {
			b := time.Duration(1 << 62)
			for range 3 {
				if clear {
					g.ClearCache()
				}
				start := time.Now()
				if _, _, err := g.Probs(enc); err != nil {
					t.Fatal(err)
				}
				b = min(b, time.Since(start))
			}
			return b
		}
		miss := best(true)
		g.ClearCache()
		g.Probs(enc)
		hit := best(false)
		t.Logf("%-24s %5d tokens (state %4d): new text %7.1f ms, same text again %6.1f ms",
			c.name, len(enc.IDs), enc.StateLen, float64(miss.Microseconds())/1000, float64(hit.Microseconds())/1000)
	}
}

// TestGPUScanLadder times the GDN scan's LPC rungs on the state scan of a
// long text and a short one (K7.4). Kernel time summed over the 24 layers.
func TestGPUScanLadder(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	fx := loadFixtures(t)
	long, _ := ParseRequest(fx[2].Request)
	rec, _ := ToRecord(long)
	short, _ := ParseRequest(fx[0].Request)
	srec, _ := ToRecord(short)
	cases := []Record{srec, rec}
	if g.rows >= 3000 {
		cases = append(cases, Record{State: strings.Repeat(rec.State+"\n\n", 7), Questions: rec.Questions})
	}
	defer func(v int) { g.ScanLPC = v }(g.ScanLPC)
	for _, r := range cases {
		enc, err := e.Encode(r, MaxState, MaxRow)
		if err != nil {
			t.Fatal(err)
		}
		line := ""
		for _, lpc := range []int{1, 2, 4, 8, 16} {
			g.ScanLPC = lpc
			best := map[string]time.Duration{}
			for range 3 {
				prof, err := g.Profile(enc)
				if err != nil {
					t.Fatal(err)
				}
				for _, l := range []string{"gdn scan state", "gdn scan branches"} {
					if d, ok := best[l]; !ok || prof[l] < d {
						best[l] = prof[l]
					}
				}
			}
			line += fmt.Sprintf("  l%d %.2f+%.2f", lpc, float64(best["gdn scan state"].Microseconds())/1000,
				float64(best["gdn scan branches"].Microseconds())/1000)
		}
		t.Logf("%4d tokens (state %4d), state+branches ms:%s", len(enc.IDs), enc.StateLen, line)
	}
}

// TestGPUProjLadder prices every GEMM rung per projection (K7.6): the
// profile's per-label kernel time with the rung forced, best of two.
func TestGPUProjLadder(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	fx := loadFixtures(t)
	long, _ := ParseRequest(fx[2].Request)
	rec, _ := ToRecord(long)
	short, _ := ParseRequest(fx[0].Request)
	srec, _ := ToRecord(short)
	cases := []Record{srec, rec}
	if g.rows >= 3000 {
		cases = append(cases, Record{State: strings.Repeat(rec.State+"\\n\\n", 7), Questions: rec.Questions})
	}
	defer func() { g.GEMM, g.GLU = "", "" }()
	for _, r := range cases {
		enc, err := e.Encode(r, MaxState, MaxRow)
		if err != nil {
			t.Fatal(err)
		}
		for _, label := range []string{"in proj", "out proj", "down", "gate+up+swiglu"} {
			line := ""
			rungs := g.GEMMRungs()
			if label == "gate+up+swiglu" {
				rungs = []string{"glum2", "glum4", "glum8"}
			}
			for _, rung := range rungs {
				g.GEMM, g.GLU = "", ""
				if label == "gate+up+swiglu" {
					g.GLU = rung
				} else {
					g.GEMM = rung
				}
				best := time.Duration(1 << 62)
				for range 2 {
					prof, err := g.Profile(enc)
					if err != nil {
						t.Fatal(err)
					}
					best = min(best, prof[label])
				}
				line += fmt.Sprintf("  %s %.1f", rung, float64(best.Microseconds())/1000)
			}
			t.Logf("%4d tokens %-15s%s", len(enc.IDs), label, line)
		}
	}
}
