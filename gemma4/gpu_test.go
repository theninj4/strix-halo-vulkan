package gemma4

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"strix-halo-vulkan/decide"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

const runeDir = "../models/rune-26b-a4b"

// The model is ~26 GB staged and takes minutes to quantise, so every test
// shares one.
var (
	gpuOnce sync.Once
	gpuM    *GPU
	gpuErr  error
	gpuDone func()
)

func loadGPU(t *testing.T) *GPU {
	t.Helper()
	if testing.Short() {
		t.Skip("stages Rune (~26 GB); not in -short")
	}
	if _, err := os.Stat(filepath.Join(runeDir, "model-00011-of-00011.safetensors")); err != nil {
		t.Skip("no models/rune-26b-a4b")
	}
	gpuOnce.Do(func() {
		dev, done := newTestDevice(t)
		gpuDone = done
		gpuM, gpuErr = Load(dev, runeDir, Options{Rows: 8192, Vision: true})
	})
	if gpuErr != nil {
		t.Fatal(gpuErr)
	}
	return gpuM
}

type oracleMeta struct {
	Name       string
	IDs        []int32 `json:"ids"`
	Labels     []string
	LabelIDs   []int32   `json:"label_ids"`
	LabelLogit []float64 `json:"label_logits"`
	ProbsT1    []float64 `json:"probs_T1"`
	LayersRun  int       `json:"layers_run"`
}

// TestOracle is R5's model gate: every prompt reference/dump_rune.py wrote,
// layer by layer against HF's fp32 forward, then the label logits and the
// decision probabilities. The prompt runs as a state of all but its last
// token and a one-token branch, so the branch path is exercised as well.
func TestOracle(t *testing.T) {
	g := loadGPU(t)
	dirs, _ := filepath.Glob("../reference/out/rune/*/*/meta.json")
	if len(dirs) == 0 {
		t.Skip("no reference/out/rune (run reference/dump_rune.py)")
	}
	for _, mp := range dirs {
		raw, err := os.ReadFile(mp)
		if err != nil {
			t.Fatal(err)
		}
		var m oracleMeta
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatal(err)
		}
		t.Run(m.Name, func(t *testing.T) {
			f, err := safetensors.Open(filepath.Join(filepath.Dir(mp), "rows.safetensors"))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			ids := m.IDs
			p, err := g.NewPass(ids[:len(ids)-1], [][]int32{ids[len(ids)-1:]})
			if err != nil {
				t.Fatal(err)
			}
			L := len(ids)
			H := g.Cfg.Hidden
			g.Trace = func(i int) {
				if i >= m.LayersRun {
					return
				}
				ref, err := f.Get("layer" + itoa(i))
				if err != nil {
					t.Fatal(err)
				}
				want, _ := ref.F32(nil)
				var e2, r2, worst float64
				for r := range L {
					got := g.Residual(r)
					for j := range H {
						d := float64(got[j] - want[r*H+j])
						e2 += d * d
						r2 += float64(want[r*H+j]) * float64(want[r*H+j])
						worst = max(worst, math.Abs(d))
					}
				}
				rel := math.Sqrt(e2 / r2)
				if i < 3 || i%5 == 4 || i == m.LayersRun-1 {
					t.Logf("layer %2d: rel rms %.2e, worst %.3g (ref rms %.3g)", i, rel, worst, math.Sqrt(r2/float64(L*H)))
				}
			}
			defer func() { g.Trace = nil }()
			took, err := g.Forward(p, m.LayersRun)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%d tokens in %v", L, took)
			if m.LabelLogit == nil {
				return
			}
			h := g.FinalNorm(g.Residual(p.readouts[0]))
			got := g.LabelLogits(h, m.LabelIDs)
			pr, err := decide.Probabilities(got, 1)
			if err != nil {
				t.Fatal(err)
			}
			var worstZ, worstP float64
			for k := range got {
				worstZ = max(worstZ, math.Abs(float64(got[k])-m.LabelLogit[k]))
				worstP = max(worstP, math.Abs(pr[k]-m.ProbsT1[k]))
			}
			t.Logf("labels %v: logits worst |d| %.3g, p(T=1) worst |d| %.3g; got %.4f want %.4f",
				m.Labels[:min(4, len(m.Labels))], worstZ, worstP, pr[:min(4, len(pr))], m.ProbsT1[:min(4, len(pr))])
			// The bar is bf16's own drift from fp32, doubled: Rune served
			// in bf16 (reference/dump_rune.py --bf16) moves these prompts'
			// probabilities by up to 0.021, and the Q8 banks by up to 0.042
			// (research/rune-vertical.md R5). The choice must not move.
			if worstP > 0.05 {
				t.Errorf("probabilities off by %.3g", worstP)
			}
			best := func(v []float64) int {
				b := 0
				for i := range v {
					if v[i] > v[b] {
						b = i
					}
				}
				return b
			}
			if best(pr) != best(m.ProbsT1) {
				t.Errorf("chose option %d, the reference %d", best(pr), best(m.ProbsT1))
			}
		})
	}
}

func itoa(i int) string {
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
}

// TestOracleTeacher feeds every layer the reference's own input (the
// oracle's output of the layer before) and measures that layer's error
// alone, to tell a wrong layer from amplification of small errors.
// GEMMA4_TEACHER names the prompt (default ticket/sentiment).
func TestOracleTeacher(t *testing.T) {
	g := loadGPU(t)
	name := os.Getenv("GEMMA4_TEACHER")
	if name == "" {
		name = "ticket/sentiment"
	}
	dir := filepath.Join("../reference/out/rune", name)
	raw, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		t.Skip("no dump for", name)
	}
	var m oracleMeta
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.Open(filepath.Join(dir, "rows.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	ids := m.IDs
	p, err := g.NewPass(ids[:len(ids)-1], [][]int32{ids[len(ids)-1:]})
	if err != nil {
		t.Fatal(err)
	}
	H, L := g.Cfg.Hidden, len(ids)
	get := func(name string) []float32 {
		ten, err := f.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		v, _ := ten.F32(nil)
		return v
	}
	cmp := func(got, want []float32) (float64, float64) {
		var e2, r2, worst float64
		for i := range want {
			d := float64(got[i] - want[i])
			e2 += d * d
			r2 += float64(want[i]) * float64(want[i])
			worst = max(worst, math.Abs(d))
		}
		return math.Sqrt(e2 / r2), worst
	}
	emb := make([]float32, L*H)
	for r, id := range ids {
		g.Embed(id, emb[r*H:(r+1)*H])
	}
	rel, worst := cmp(emb, get("embed"))
	t.Logf("embed: rel %.2e, worst %.3g", rel, worst)
	in := get("embed")
	for i := range g.Cfg.Layers {
		if _, err := g.ForwardFrom(p, in, i, i+1); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, 0, L*H)
		for r := range L {
			got = append(got, g.Residual(r)...)
		}
		want := get("layer" + itoa(i))
		rel, worst := cmp(got, want)
		kind := "sliding"
		if g.Cfg.Full(i) {
			kind = "full"
		}
		t.Logf("layer %2d (%s): rel %.2e, worst %.3g", i, kind, rel, worst)
		in = want
	}
}

// TestOracleInternals runs layers 0 and 5 teacher-forced, a dispatch range
// at a time, against the dump's internals: the attention block's output,
// the dense MLP's, the router's top-8 and the experts' sum.
func TestOracleInternals(t *testing.T) {
	g := loadGPU(t)
	dir := "../reference/out/rune/ticket/sentiment"
	raw, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		t.Skip("no dump")
	}
	var m oracleMeta
	_ = json.Unmarshal(raw, &m)
	f, err := safetensors.Open(filepath.Join(dir, "rows.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	ids := m.IDs
	p, _ := g.NewPass(ids[:len(ids)-1], [][]int32{ids[len(ids)-1:]})
	H, L := g.Cfg.Hidden, len(ids)
	get := func(name string) []float32 {
		ten, err := f.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		v, _ := ten.F32(nil)
		return v
	}
	rel := func(got, want []float32) float64 {
		var e2, r2 float64
		for i := range want {
			d := float64(got[i] - want[i])
			e2 += d * d
			r2 += float64(want[i]) * float64(want[i])
		}
		return math.Sqrt(e2 / r2)
	}
	for _, li := range []int{0, 5} {
		in := get("embed")
		if li > 0 {
			in = get("layer" + itoa(li-1))
		}
		g.abuf.WriteFloat32At(int(g.aX), in[:L*H])
		g.abuf.WriteUint32At(int(g.aMeta), p.meta)
		if err := g.moe.Resize(L); err != nil {
			t.Fatal(err)
		}
		d, labels := g.attnGraph(li, L)
		run := func(upto string) {
			for k, l := range labels {
				if l == upto {
					if _, err := vkDispatch(d[:k+1]); err != nil {
						t.Fatal(err)
					}
					return
				}
			}
			t.Fatalf("no dispatch %q", upto)
		}
		run("o")
		t.Logf("layer %d attn out: rel %.2e", li, rel(g.abuf.ReadFloat32At(int(g.aY), L*H), get("layer"+itoa(li)+".attn")))
		g.abuf.WriteFloat32At(int(g.aX), in[:L*H])
		run("down")
		t.Logf("layer %d dense mlp: rel %.2e", li, rel(g.abuf.ReadFloat32At(int(g.aY), L*H), get("layer"+itoa(li)+".mlp")))
		if err := g.moe.Run(li); err != nil {
			t.Fatal(err)
		}
		out := g.moe.OutPort()
		t.Logf("layer %d experts: rel %.2e", li, rel(out.Buf.ReadFloat32At(int(out.Off), L*H), get("layer"+itoa(li)+".experts")))
		top := g.moe.TopK()
		rt, err := f.Get("layer" + itoa(li) + ".router.index")
		if err != nil {
			t.Fatal(err)
		}
		ref := make([]int64, len(rt.Data)/8)
		for i := range ref {
			ref[i] = int64(binary.LittleEndian.Uint64(rt.Data[8*i:]))
		}
		K := g.Cfg.TopK
		agree := 0
		for r := range L {
			set := map[int32]bool{}
			for k := range K {
				set[top[r*K+k]] = true
			}
			for k := range K {
				if set[int32(ref[r*K+k])] {
					agree++
				}
			}
		}
		t.Logf("layer %d routing: %d of %d choices agree", li, agree, L*K)
	}
}

func vkDispatch(d []vk.MultiDispatch) (any, error) { return vk.DispatchMultiTimed(d, 1, 1, true) }

// TestBatch holds R8's batching: requests answered together in one pass give
// what each gives alone, up to the rounding of the rungs a longer pass picks.
func TestBatch(t *testing.T) {
	g := loadGPU(t)
	cb := g.Tok.Codebook()
	bodies := []string{
		`{"model": "rune", "state": {"ticket": "Order 8812 arrived late and the box was crushed. I want a refund."}, "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}, "tone": {"type": "choice", "instructions": "What is the tone?", "criteria": {"calm": "Calm", "annoyed": "Annoyed", "furious": "Furious"}}}}`,
		`{"model": "rune", "state": "The quarterly report shows revenue up 12% and margins steady.", "questions": {"sentiment": {"type": "choice", "instructions": "Overall sentiment?", "criteria": {"neg": "Negative", "neu": "Neutral", "pos": "Positive"}}}}`,
		`{"model": "rune", "state": {"email": "Hi, can we move tomorrow's 10am meeting to Thursday? Thanks, Ana"}, "order_averaging": true, "questions": {"meeting": {"type": "noul", "instructions": "Is this about scheduling?"}, "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}}}`,
	}
	var reqs []*decide.Request
	var alone [][]float64
	for _, b := range bodies {
		r, err := decide.Parse([]byte(b))
		if err != nil {
			t.Fatal(err)
		}
		reqs = append(reqs, r)
		ro, err := g.Readout(r, cb)
		if err != nil {
			t.Fatal(err)
		}
		for i := range r.Questions {
			p, _ := decide.Probabilities(ro.Logits[i], 1)
			alone = append(alone, p)
		}
	}
	ros, errs := g.ReadoutBatch(reqs, cb)
	k, worst := 0, 0.0
	for j, r := range reqs {
		if errs[j] != nil {
			t.Fatal(errs[j])
		}
		for i := range r.Questions {
			p, _ := decide.Probabilities(ros[j].Logits[i], 1)
			for o := range p {
				worst = max(worst, math.Abs(p[o]-alone[k][o]))
			}
			k++
		}
	}
	t.Logf("%d requests, %d questions: batched vs alone, worst |dp| %.2e", len(reqs), k, worst)
	if worst > 1e-2 {
		t.Errorf("batching moved a probability by %.3g", worst)
	}
}

// TestBatchDebug prints, per question, alone against batched with its
// neighbours and against a batch of two copies of itself.
func TestBatchDebug(t *testing.T) {
	g := loadGPU(t)
	cb := g.Tok.Codebook()
	parse := func(b string) *decide.Request {
		r, err := decide.Parse([]byte(b))
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	a := parse(`{"model": "rune", "state": {"ticket": "Order 8812 arrived late and the box was crushed. I want a refund."}, "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}, "tone": {"type": "choice", "instructions": "What is the tone?", "criteria": {"calm": "Calm", "annoyed": "Annoyed", "furious": "Furious"}}}}`)
	_ = parse(`{"model": "rune", "state": "The quarterly report shows revenue up 12% and margins steady.", "questions": {"sentiment": {"type": "choice", "instructions": "Overall sentiment?", "criteria": {"neg": "Negative", "neu": "Neutral", "pos": "Positive"}}}}`)
	show := func(name string, ros []Readout) {
		for j, ro := range ros {
			for i, z := range ro.Logits {
				t.Logf("%-10s req %d q %d: z %v", name, j, i, z)
			}
		}
	}
	for _, c := range []struct {
		name string
		reqs []*decide.Request
	}{{"a", []*decide.Request{a}}, {"a,a", []*decide.Request{a, a}}, {"a,a pin", []*decide.Request{a, a}}} {
		if c.name == "a,a pin" {
			g.GEMM = map[string]string{"qkv": "q8m4", "o": "q8m2", "gate+up": "gegm4", "down": "q8m2"}
		}
		ros, errs := g.ReadoutBatch(c.reqs, cb)
		for _, e := range errs {
			if e != nil {
				t.Fatal(e)
			}
		}
		show(c.name, ros)
		t.Logf("%-10s rows %d", c.name, func() int { p, _ := g.PassOf(c.reqs[0], cb); return p.Rows() }())
	}
	g.GEMM = nil
}
