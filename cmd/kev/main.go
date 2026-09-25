// Command kev runs Kev-4B over one of Kev's frozen eval suites
// (CLASSIFICATION.md K7.1, K8) and scores it the way `kev.benchmark` does:
// accuracy, Brier (served and raw) and ECE over the `clean` variant's
// knowable rows, plus the share of unknowable rows answered at p >= 0.9.
//
//	go run ./cmd/kev -suite models/kev-suites/transfer-v4/development.jsonl -out /tmp/q8.jsonl
//	go run ./cmd/kev -suite ... -bank fp16 -out /tmp/fp16.jsonl
//	go run ./cmd/kev -compare /tmp/fp16.jsonl,/tmp/q8.jsonl
//
// Each record of a suite is a System One request with a `label` (and `src`)
// on every question and a `_meta` beside them; the request is parsed by the
// same code the server runs, which ignores the extra fields. The rows it
// writes are one JSON object per question, with the full distribution, so
// two runs -- two banks, or this and Kev's fp32 -- can be compared row by
// row with -compare.
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

	"strix-halo-vulkan/kev"
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
	P        []float64 `json:"p"`
}

func main() {
	log.SetFlags(0)
	suite := flag.String("suite", "", "a suite partition, JSONL of labelled requests")
	model := flag.String("model", "models/kev-4b", "Kev checkpoint directory")
	base := flag.String("base", "models/Qwen3.5-4B-Base", "the base it was trained on")
	bank := flag.String("bank", "q8", "weight width: q8 or fp16")
	out := flag.String("out", "", "write one row per question here (JSONL)")
	compare := flag.String("compare", "", "two row files, comma-separated: report agreement and exit")
	limit := flag.Int("n", 0, "score only the first n records")
	batch := flag.Int("batch", 1, "records a pass: run the suite through ProbsBatch n records at a time (K7.5)")
	flag.Parse()

	head, err := kev.LoadHead(*model)
	must(err)
	temperature = head.Temperature

	if *compare != "" {
		files := strings.Split(*compare, ",")
		if len(files) != 2 {
			log.Fatal("-compare takes two row files")
		}
		compareRows(readRows(files[0]), readRows(files[1]))
		return
	}
	if *suite == "" {
		log.Fatal("-suite is required")
	}
	b := kev.BankQ8
	switch *bank {
	case "q8":
	case "fp16":
		b = kev.BankFP16
	default:
		log.Fatalf("-bank %q: q8 or fp16", *bank)
	}

	enc, err := kev.LoadEncoder(*model)
	must(err)
	dev, done := openDevice()
	defer done()
	start := time.Now()
	g, err := kev.LoadBank(dev, *model, *base, 8192, b)
	must(err)
	defer g.Destroy()
	fmt.Printf("loaded   %s bank in %v\n", g.Bank, time.Since(start).Round(time.Millisecond))

	f, err := os.Open(*suite)
	must(err)
	defer f.Close()
	var w *bufio.Writer
	if *out != "" {
		of, err := os.Create(*out)
		must(err)
		defer of.Close()
		w = bufio.NewWriter(of)
		defer w.Flush()
	}

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	var rows []row
	var gpu time.Duration
	records := 0
	start = time.Now()
	type pending struct {
		enc  *kev.Encoding
		meta []kev.Meta
		lab  labels
	}
	var queue []pending
	flush := func() {
		if len(queue) == 0 {
			return
		}
		encs := make([]*kev.Encoding, len(queue))
		for i, q := range queue {
			encs[i] = q.enc
		}
		probs, passes, err := g.ProbsBatch(encs)
		must(err)
		gpu += passes[0].GPU
		for i, q := range queue {
			for k, m := range q.meta {
				ql := q.lab.Questions[m.ID]
				r := row{ID: q.lab.Meta.ID, Question: m.ID, Variant: q.lab.Meta.Variant, Source: q.lab.Meta.Source,
					Task: ql.Src, Type: m.Type, Keys: m.Keys, Label: labelIndex(m, ql.Label), P: probs[i][k]}
				rows = append(rows, r)
				if w != nil {
					b, _ := json.Marshal(r)
					w.Write(append(b, '\n'))
				}
			}
		}
		queue = queue[:0]
	}
	for sc.Scan() {
		if *limit > 0 && records >= *limit {
			break
		}
		line := sc.Bytes()
		req, err := kev.ParseRequest(line)
		must(err)
		var lab labels
		must(json.Unmarshal(line, &lab))
		rec, meta := kev.ToRecord(req)
		e, err := enc.Encode(rec, kev.MaxState, kev.MaxRow)
		must(err)
		queue = append(queue, pending{e, meta, lab})
		if len(queue) >= *batch {
			flush()
		}
		records++
	}
	flush()
	must(sc.Err())
	wall := time.Since(start)
	fmt.Printf("records  %d (%d questions, batches of %d) in %v, %v of GPU (%.1f ms a record)\n", records, len(rows), *batch,
		wall.Round(time.Millisecond), gpu.Round(time.Millisecond), float64(gpu.Microseconds())/1000/float64(records))
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
func labelIndex(m kev.Meta, raw json.RawMessage) int {
	switch m.Type {
	case kev.TypeChoice:
		var k string
		must(json.Unmarshal(raw, &k))
		for i, key := range m.Keys {
			if key == k {
				return i
			}
		}
		log.Fatalf("label %q is not an option of %v", k, m.Keys)
	case kev.TypeNoul:
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

// temperature is the head's fitted T, which the served probabilities carry;
// raw (T = 1) probabilities are p^T renormalised, which is what Kev's model
// card reports Brier on.
var temperature = 1.0

// report is kev.metrics over clean, knowable rows: accuracy, Brier and ECE as
// served, raw Brier, accuracy by task, and kev.metrics.unknowable_report's
// share of evidence-free records answered at p >= 0.9.
func report(rows []row) {
	type acc struct{ n, right int }
	all := acc{}
	byTask := map[string]*acc{}
	var brier, rawBrier float64
	var conf []float64
	var correct []bool
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
		brier += brierScore(r.P, r.Label)
		rawBrier += brierScore(untemper(r.P), r.Label)
		conf = append(conf, r.P[argmax(r.P)])
		correct = append(correct, ok)
	}
	n := float64(max(all.n, 1))
	fmt.Printf("accuracy %.4f over %d clean questions; Brier %.4f served, %.4f raw (T=1); ECE %.4f\n",
		float64(all.right)/n, all.n, brier/n, rawBrier/n, ece(conf, correct))
	if unk > 0 {
		fmt.Printf("unknowable %d questions, %.3f answered at p >= 0.9\n", unk, float64(unkHigh)/float64(unk))
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

// untemper undoes the served temperature: softmax(z/T)^T is softmax(z).
func untemper(p []float64) []float64 {
	out := make([]float64, len(p))
	sum := 0.0
	for i, v := range p {
		out[i] = math.Pow(v, temperature)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
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

// compareRows reports how far b's distributions are from a's: argmax
// agreement, the largest and mean max-|dp| per question, and each side's
// clean accuracy.
func compareRows(a, b map[string]row) {
	var n, agree int
	var worst, sum float64
	var worstID string
	for id, ra := range a {
		rb, ok := b[id]
		if !ok {
			continue
		}
		n++
		if argmax(ra.P) == argmax(rb.P) {
			agree++
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
	fmt.Printf("questions %d in both; argmax agreement %d/%d (%.4f); max |dp| %.4f (%s); mean max |dp| %.5f\n",
		n, agree, n, float64(agree)/float64(max(n, 1)), worst, worstID, sum/float64(max(n, 1)))
	for _, side := range []struct {
		name string
		rows map[string]row
	}{{"a", a}, {"b", b}} {
		var rs []row
		for _, r := range side.rows {
			rs = append(rs, r)
		}
		fmt.Printf("%s: ", side.name)
		report(rs)
	}
}

func openDevice() (*vk.Device, func()) {
	inst, err := vk.NewInstance("kev")
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
