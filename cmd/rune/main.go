// Command rune runs Rune v3 over one of Kev's frozen eval suites
// (research/rune-vertical.md R7) and scores it the way cmd/kev scored Kev
// (K8): accuracy, Brier and ECE over the `clean` variant's knowable rows,
// plus the share of unknowable rows answered at p >= 0.9. It fits the
// calibration temperature by mean NLL on the same rows.
//
//	go build -o /tmp/runebench ./cmd/rune
//	/tmp/runebench -suite models/kev-suites/transfer-v4/development.jsonl -out rows.jsonl
//	/tmp/runebench -compare reference/out/kev/suites/q8-transfer-v4-development.jsonl,rows.jsonl
//
// Each record is a System One request with a `label` (and `src`) on every
// question; it is answered through decide.ParseSystemOne, the translation
// the server runs for /v1/systemone. Rows are one JSON object per question,
// in cmd/kev's schema (so -compare reads Kev's rows too) plus the label
// logits `z` at T = 1 and, with -prompts, a file of each question's prompt
// ids for reference/rune_suite_ref.py to re-run in bf16.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"strix-halo-vulkan/decide"
	"strix-halo-vulkan/gemma4"
	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

// row is one scored question.
type row struct {
	ID       string    `json:"id"`
	Question string    `json:"question"`
	Variant  string    `json:"variant"`
	Source   string    `json:"source"`
	Task     string    `json:"task"`
	Type     string    `json:"type"`
	Keys     []string  `json:"keys"`
	Label    int       `json:"label"`
	P        []float64 `json:"p"`           // at T = 1
	Z        []float32 `json:"z,omitempty"` // the label logits
}

type promptRow struct {
	ID       string  `json:"id"`
	Question string  `json:"question"`
	IDs      []int32 `json:"ids"`
	Labels   []int32 `json:"labels"`
}

func main() {
	log.SetFlags(0)
	suite := flag.String("suite", "", "a suite partition, JSONL of labelled requests")
	model := flag.String("model", "models/rune-26b-a4b", "Rune checkpoint directory")
	out := flag.String("out", "", "write one row per question here (JSONL)")
	prompts := flag.String("prompts", "", "write each question's prompt ids here (JSONL), for reference/rune_suite_ref.py")
	compare := flag.String("compare", "", "two row files, comma-separated: report agreement and exit")
	limit := flag.Int("n", 0, "score only the first n records")
	every := flag.Int("every", 1, "score every k-th record (a sample for a slow reference)")
	profile := flag.String("profile", "", "a decisions request body (file): time it dispatch by dispatch and exit (R8)")
	reps := flag.Int("reps", 10, "with -profile: timed runs of the whole request")
	profBatch := flag.Int("profile-batch", 0, "with -suite: pack this many of its records into one pass (as the server batches), time it dispatch by dispatch and exit")
	profPass := flag.Int("profile-pass", 0, "time a synthetic pass of this many rows dispatch by dispatch and exit")
	passes := flag.String("passes", "", "comma-separated pass lengths: time a whole forward at each and exit (R8)")
	ladder := flag.String("ladder", "", "comma-separated pass lengths: time every GEMM rung on every projection at each and exit (R8)")
	flag.Parse()

	if *compare != "" {
		files := strings.Split(*compare, ",")
		if len(files) != 2 {
			log.Fatal("-compare takes two row files")
		}
		compareRows(readRows(files[0]), readRows(files[1]))
		return
	}
	if *suite == "" && *profile == "" && *ladder == "" && *passes == "" && *profPass == 0 {
		log.Fatal("-suite, -profile or -ladder is required")
	}
	dev, done := openDevice()
	defer done()
	start := time.Now()
	g, err := gemma4.Load(dev, *model, gemma4.Options{Rows: 8192})
	must(err)
	defer g.Destroy()
	codebook := g.Tok.Codebook()
	fmt.Printf("staged   %s in %v\n", *model, time.Since(start).Round(time.Millisecond))
	if *profile != "" {
		runProfile(g, codebook, *profile, *reps)
		return
	}
	if *profBatch > 0 {
		runProfileBatch(g, codebook, *suite, *profBatch, *reps)
		return
	}
	if *profPass > 0 {
		p, err := g.SyntheticPass(*profPass)
		must(err)
		printProfile(g, p)
		return
	}
	if *ladder != "" {
		runLadder(g, *ladder)
		return
	}
	if *passes != "" {
		for _, ls := range strings.Split(*passes, ",") {
			var n int
			_, err := fmt.Sscan(ls, &n)
			must(err)
			p, err := g.SyntheticPass(n)
			must(err)
			var ts []time.Duration
			for range 4 {
				d, err := g.Forward(p, 0)
				must(err)
				ts = append(ts, d)
			}
			sort.Slice(ts, func(i, j int) bool { return ts[i] < ts[j] })
			fmt.Printf("rows %5d  forward %8.1f ms  %6.3f ms a row\n", n, float64(ts[1].Microseconds())/1000,
				float64(ts[1].Microseconds())/1000/float64(n))
		}
		return
	}

	f, err := os.Open(*suite)
	must(err)
	defer f.Close()
	var rowsOut, promptsOut *bufio.Writer
	if *out != "" {
		fo, err := os.Create(*out)
		must(err)
		defer fo.Close()
		rowsOut = bufio.NewWriter(fo)
		defer rowsOut.Flush()
	}
	if *prompts != "" {
		fp, err := os.Create(*prompts)
		must(err)
		defer fp.Close()
		promptsOut = bufio.NewWriter(fp)
		defer promptsOut.Flush()
	}

	var rows []row
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	records, seen := 0, 0
	var busy time.Duration
	start = time.Now()
	for sc.Scan() {
		if *limit > 0 && records >= *limit {
			break
		}
		seen++
		if (seen-1)%*every != 0 {
			continue
		}
		line := append([]byte(nil), sc.Bytes()...)
		req, err := decide.ParseSystemOne(line)
		must(err)
		var lab labels
		must(json.Unmarshal(line, &lab))
		t0 := time.Now()
		ro, err := g.Readout(req, codebook)
		must(err)
		busy += time.Since(t0)
		for i := range req.Questions {
			q := &req.Questions[i]
			ql := lab.Questions[q.Name]
			p, err := decide.Probabilities(ro.Logits[i], 1)
			must(err)
			r := row{ID: lab.Meta.ID, Question: q.Name, Variant: lab.Meta.Variant, Source: lab.Meta.Source,
				Task: ql.Src, Type: q.Type.String(), Keys: q.Keys, Label: labelIndex(q, ql.Label), P: p, Z: ro.Logits[i]}
			rows = append(rows, r)
			if rowsOut != nil {
				b, _ := json.Marshal(r)
				rowsOut.Write(append(b, '\n'))
			}
			if promptsOut != nil {
				b, _ := json.Marshal(promptRow{ID: r.ID, Question: r.Question, IDs: ro.Prompts[i], Labels: ro.Labels[i]})
				promptsOut.Write(append(b, '\n'))
			}
		}
		records++
	}
	must(sc.Err())
	fmt.Printf("records  %d (%d questions) in %v, %.1f ms a record\n", records, len(rows),
		time.Since(start).Round(time.Millisecond), float64(busy.Microseconds())/1000/float64(max(records, 1)))
	report(rows)
}

// labels is what a suite record carries beside its request.
type labels struct {
	Questions map[string]struct {
		Label json.RawMessage `json:"label"`
		Src   string          `json:"src"`
	} `json:"questions"`
	Meta struct {
		ID      string `json:"id"`
		Source  string `json:"source"`
		Variant string `json:"variant"`
	} `json:"_meta"`
}

// labelIndex is kev.benchmark.labels: a choice's label is its key, a noul's
// is a bool (int(True) is 1, the "true" slot), a score's is the level index.
func labelIndex(q *decide.Question, raw json.RawMessage) int {
	switch q.Type {
	case decide.Choice:
		var k string
		must(json.Unmarshal(raw, &k))
		for i, key := range q.Keys {
			if key == k {
				return i
			}
		}
		log.Fatalf("label %q is not an option of %v", k, q.Keys)
	case decide.Noul:
		var b bool
		if json.Unmarshal(raw, &b) == nil {
			if b {
				return 1
			}
			return 0
		}
	}
	var n float64
	must(json.Unmarshal(raw, &n))
	return int(n)
}

func argmax(p []float64) int {
	best := 0
	for i, v := range p {
		if v > p[best] {
			best = i
		}
	}
	return best
}

// knowable is cmd/kev's filter: the clean variant, not the unknowable rows.
func knowable(r row) bool { return r.Variant == "clean" && r.Source != "unknowable" }

// temper is softmax(log p / T): p at T = 1 re-read at temperature T.
func temper(p []float64, t float64) []float64 {
	out := make([]float64, len(p))
	top := math.Inf(-1)
	for i, v := range p {
		out[i] = math.Log(v) / t
		top = math.Max(top, out[i])
	}
	sum := 0.0
	for i := range out {
		out[i] = math.Exp(out[i] - top)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func nll(rows []row, t float64) float64 {
	s, n := 0.0, 0
	for _, r := range rows {
		if !knowable(r) || r.Label < 0 || r.Label >= len(r.P) {
			continue
		}
		s -= math.Log(math.Max(temper(r.P, t)[r.Label], 1e-300))
		n++
	}
	return s / float64(max(n, 1))
}

// fitT is the NLL-minimising temperature, by golden section on [0.3, 10].
func fitT(rows []row) float64 {
	a, b := 0.3, 10.0
	phi := (math.Sqrt(5) - 1) / 2
	for range 60 {
		c, d := b-phi*(b-a), a+phi*(b-a)
		if nll(rows, c) < nll(rows, d) {
			b = d
		} else {
			a = c
		}
	}
	return (a + b) / 2
}

// report is cmd/kev's report over clean, knowable rows, at T = 1, 2 and the
// fitted T: accuracy (the same at every T), Brier, ECE and NLL.
func report(rows []row) {
	type acc struct{ n, right int }
	all := acc{}
	byTask := map[string]*acc{}
	var unk, unkHigh int
	for _, r := range rows {
		if r.Variant != "clean" {
			continue
		}
		if r.Source == "unknowable" {
			unk++
			if r.P[argmax(r.P)] >= 0.9 {
				unkHigh++
			}
			continue
		}
		a := byTask[r.Task]
		if a == nil {
			a = &acc{}
			byTask[r.Task] = a
		}
		ok := argmax(r.P) == r.Label
		for _, x := range []*acc{a, &all} {
			x.n++
			if ok {
				x.right++
			}
		}
	}
	n := float64(max(all.n, 1))
	fmt.Printf("accuracy %.4f over %d clean questions\n", float64(all.right)/n, all.n)
	tf := fitT(rows)
	for _, t := range []float64{1, 2, tf} {
		var brier float64
		var conf []float64
		var correct []bool
		for _, r := range rows {
			if !knowable(r) {
				continue
			}
			p := temper(r.P, t)
			brier += brierScore(p, r.Label)
			conf = append(conf, p[argmax(p)])
			correct = append(correct, argmax(p) == r.Label)
		}
		label := fmt.Sprintf("T=%g", t)
		if t == tf {
			label = fmt.Sprintf("T=%.3f (fitted)", t)
		}
		fmt.Printf("  %-17s Brier %.4f  ECE %.4f  NLL %.4f\n", label, brier/n, ece(conf, correct), nll(rows, t))
	}
	if unk > 0 {
		fmt.Printf("unknowable %d questions, %.3f answered at p >= 0.9 (T=1)\n", unk, float64(unkHigh)/float64(unk))
	}
	tasks := make([]string, 0, len(byTask))
	for t := range byTask {
		tasks = append(tasks, t)
	}
	sort.Strings(tasks)
	for _, t := range tasks {
		a := byTask[t]
		fmt.Printf("  %-36s %4d  %.3f\n", t, a.n, float64(a.right)/float64(a.n))
	}
}

func brierScore(p []float64, label int) float64 {
	s := 0.0
	for i, v := range p {
		if i == label {
			v -= 1
		}
		s += v * v
	}
	return s
}

// ece is kev.metrics.ece: ten equal-width confidence bins, the last closed.
func ece(conf []float64, correct []bool) float64 {
	e := 0.0
	for b := 0; b < 10; b++ {
		lo, hi := float64(b)/10, float64(b+1)/10
		var n, right int
		var c float64
		for i, v := range conf {
			if v >= lo && (v < hi || b == 9 && v <= hi) {
				n++
				c += v
				if correct[i] {
					right++
				}
			}
		}
		if n > 0 {
			e += float64(n) / float64(len(conf)) * math.Abs(float64(right)/float64(n)-c/float64(n))
		}
	}
	return e
}

func readRows(path string) map[string]row {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	out := map[string]row{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	for sc.Scan() {
		var r row
		must(json.Unmarshal(sc.Bytes(), &r))
		out[r.ID+"/"+r.Question] = r
	}
	must(sc.Err())
	return out
}

// compareRows reports how b's answers stand against a's on the questions
// both scored: argmax agreement, where each is right and the other wrong,
// the largest and mean max-|dp|, and the flips' margins in a.
func compareRows(a, b map[string]row) {
	var n, agree, aOnly, bOnly int
	var worst, sum float64
	var worstID string
	var flipMargins []float64
	for id, ra := range a {
		rb, ok := b[id]
		if !ok || len(ra.P) != len(rb.P) {
			continue
		}
		n++
		if argmax(ra.P) == argmax(rb.P) {
			agree++
		} else {
			// The flip's margin in a: its top two probabilities apart.
			s := append([]float64(nil), ra.P...)
			sort.Float64s(s)
			flipMargins = append(flipMargins, s[len(s)-1]-s[len(s)-2])
		}
		if knowable(ra) {
			okA, okB := argmax(ra.P) == ra.Label, argmax(rb.P) == rb.Label
			if okA && !okB {
				aOnly++
			}
			if okB && !okA {
				bOnly++
			}
		}
		d := 0.0
		for i := range ra.P {
			d = math.Max(d, math.Abs(ra.P[i]-rb.P[i]))
		}
		sum += d
		if d > worst {
			worst, worstID = d, id
		}
	}
	sort.Float64s(flipMargins)
	fmt.Printf("questions %d in both; argmax agreement %d/%d (%.4f); right in a only %d, in b only %d\n",
		n, agree, n, float64(agree)/float64(max(n, 1)), aOnly, bOnly)
	fmt.Printf("max |dp| %.4f (%s); mean max |dp| %.5f; flip margins in a %v\n", worst, worstID, sum/float64(max(n, 1)), roundAll(flipMargins))
	for _, side := range []struct {
		name string
		rows map[string]row
	}{{"a", a}, {"b", b}} {
		var rs []row
		for id, r := range side.rows {
			if _, both := a[id]; both {
				if _, both := b[id]; both {
					rs = append(rs, r)
				}
			}
		}
		fmt.Printf("%s: ", side.name)
		report(rs)
	}
}

func roundAll(v []float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = math.Round(x*1e4) / 1e4
	}
	return out
}

func openDevice() (*vk.Device, func()) {
	inst, err := vk.NewInstance("rune")
	must(err)
	devices, err := inst.PhysicalDevices()
	must(err)
	if len(devices) == 0 {
		log.Fatal("no Vulkan devices")
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
		}
	}
	qf, err := phys.ComputeQueueFamily()
	must(err)
	sgs, err := phys.SubgroupSizeControl()
	must(err)
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	must(err)
	fmt.Printf("device   %s\n", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// runProfile times one request: its wall time over reps runs, then one
// pass dispatch by dispatch.
func runProfile(g *gemma4.GPU, codebook []string, path string, reps int) {
	body, err := os.ReadFile(path)
	must(err)
	req, err := decide.Parse(body)
	must(err)
	var times []time.Duration
	var ro gemma4.Readout
	for range reps + 1 {
		t0 := time.Now()
		ro, err = g.Readout(req, codebook)
		must(err)
		times = append(times, time.Since(t0))
	}
	times = times[1:] // the first warms the pipelines
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	fmt.Printf("request  %d questions, %d input tokens: median %v (min %v, max %v) over %d runs\n", len(req.Questions),
		ro.Usage.InputTokens, times[len(times)/2].Round(10*time.Microsecond), times[0].Round(10*time.Microsecond),
		times[len(times)-1].Round(10*time.Microsecond), len(times))
	p, err := g.PassOf(req, codebook)
	must(err)
	printProfile(g, p)
}

func printProfile(g *gemma4.GPU, p *gemma4.Pass) {
	st, err := g.Profile(p)
	must(err)
	type kv struct {
		k string
		v time.Duration
	}
	var all []kv
	var total time.Duration
	for k, v := range st {
		all = append(all, kv{k, v})
		total += v
	}
	sort.Slice(all, func(i, j int) bool { return all[i].v > all[j].v })
	fmt.Printf("gpu      %v over %d rows, by dispatch (summed over the layers):\n", total.Round(10*time.Microsecond), p.Rows())
	for _, e := range all {
		fmt.Printf("  %-20s %9.3f ms  %5.1f%%\n", e.k, float64(e.v.Microseconds())/1000, 100*float64(e.v)/float64(total))
	}
}

// runLadder times every rung of every projection at each pass length: one
// profiled pass a rung index, all four projections moved together (each to
// its own family's rung of that index), the time read per projection.
func runLadder(g *gemma4.GPU, lengths string) {
	labels := []string{"qkv", "o", "gate+up", "down"}
	for _, ls := range strings.Split(lengths, ",") {
		var n int
		_, err := fmt.Sscan(ls, &n)
		must(err)
		p, err := g.SyntheticPass(n)
		must(err)
		fmt.Printf("rows %d\n", n)
		best := map[string]string{}
		bestT := map[string]time.Duration{}
		for _, label := range labels {
			for _, rung := range gemma4.GEMMRungs(label) {
				g.GEMM = map[string]string{label: rung}
				st, err := g.Profile(p)
				must(err)
				fmt.Printf("  %-8s %-6s %8.3f ms\n", label, rung, float64(st[label].Microseconds())/1000)
				if bestT[label] == 0 || st[label] < bestT[label] {
					best[label], bestT[label] = rung, st[label]
				}
			}
		}
		g.GEMM = nil
		st, err := g.Profile(p)
		must(err)
		for _, label := range labels {
			fmt.Printf("  best %-8s %-6s %8.3f ms (schedule %8.3f ms)\n", label, best[label],
				float64(bestT[label].Microseconds())/1000, float64(st[label].Microseconds())/1000)
		}
	}
}

// runProfileBatch packs the first n records of a suite into one pass, the
// shape the server's batching makes of concurrent requests (distinct
// conversations, so the experts they touch are realistic), and times it.
func runProfileBatch(g *gemma4.GPU, codebook []string, path string, n, reps int) {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	var plans []*gemma4.Plan
	var segs []gemma4.Segment
	rows := 0
	for len(plans) < n && sc.Scan() {
		req, err := decide.ParseSystemOne(append([]byte(nil), sc.Bytes()...))
		must(err)
		p, err := g.PlanRequest(req, codebook)
		must(err)
		plans, segs = append(plans, p), append(segs, p.Segment())
		rows += p.Rows()
	}
	var times []time.Duration
	for range reps + 1 {
		t0 := time.Now()
		_, err := g.RunPlans(plans)
		must(err)
		times = append(times, time.Since(t0))
	}
	times = times[1:]
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	fmt.Printf("batch    %d requests, %d rows: median %v a pass, %.1f req/s, %.3f ms a row\n", len(plans), rows,
		times[len(times)/2].Round(10*time.Microsecond), float64(len(plans))/times[len(times)/2].Seconds(),
		float64(times[len(times)/2].Microseconds())/1000/float64(rows))
	p, err := g.NewBatch(segs)
	must(err)
	printProfile(g, p)
}
