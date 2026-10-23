package mm3

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

const (
	modelDir = "../models/MiniMax-Music3"
	refDir   = "../reference/out/mm3lm" // reference/dump_mm3.py
)

// lmManifest is reference/dump_mm3.py's manifest.
type lmManifest struct {
	Frames int `json:"frames"`
	Steps  []struct {
		KLCond   float64 `json:"kl_cond"`
		KLUncond float64 `json:"kl_uncond"`
		KLCFG    float64 `json:"kl_cfg"`
		Top50    int     `json:"top50_shared"`
		Argmax   bool    `json:"argmax"`
	} `json:"steps"`
	Tensors map[string]struct {
		Shape []int  `json:"shape"`
		DType string `json:"dtype"`
	} `json:"tensors"`
}

func loadManifest(t testing.TB) *lmManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference (run reference/dump_mm3.py): %v", err)
	}
	var m lmManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// refFloat32 and refInt32 read one of the dump's tensors.
func refFloat32(t testing.TB, m *lmManifest, name string) ([]int, []float32) {
	t.Helper()
	info, ok := m.Tensors[name]
	if !ok || info.DType != "float32" {
		t.Fatalf("reference has no float32 %s", name)
	}
	buf, err := os.ReadFile(filepath.Join(refDir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(buf)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(buf[4*i]) | uint32(buf[4*i+1])<<8 | uint32(buf[4*i+2])<<16 | uint32(buf[4*i+3])<<24)
	}
	return info.Shape, out
}

func refInt32(t testing.TB, m *lmManifest, name string) ([]int, []int32) {
	t.Helper()
	info, ok := m.Tensors[name]
	if !ok || info.DType != "int32" {
		t.Fatalf("reference has no int32 %s", name)
	}
	buf, err := os.ReadFile(filepath.Join(refDir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int32, len(buf)/4)
	for i := range out {
		out[i] = int32(uint32(buf[4*i]) | uint32(buf[4*i+1])<<8 | uint32(buf[4*i+2])<<16 | uint32(buf[4*i+3])<<24)
	}
	return info.Shape, out
}

const strixHaloDeviceID = 0x1586

func openDevice() (*vk.Device, error) {
	inst, err := vk.NewInstance("mm3-test")
	if err != nil {
		return nil, err
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		return nil, fmt.Errorf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		return nil, err
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		return nil, err
	}
	return vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
}

// testBank is the bank under test: int8 (the default, as served), or
// MM3_LM_BANK=fp16 for the control.
func testBank() Bank {
	if os.Getenv("MM3_LM_BANK") == "fp16" {
		return BankFP16
	}
	return BankQ8
}

// spanKL is KL(want || got) of softmax over the head span, skipping the
// entries want masks (-inf: the four ids between AudioEnd and the codes).
func spanKL(want, got []float32) float64 {
	lse := func(x []float32) float64 {
		m := math.Inf(-1)
		for i, v := range x {
			if !math.IsInf(float64(want[i]), -1) {
				m = math.Max(m, float64(v))
			}
		}
		s := 0.0
		for i, v := range x {
			if !math.IsInf(float64(want[i]), -1) {
				s += math.Exp(float64(v) - m)
			}
		}
		return m + math.Log(s)
	}
	lw, lg := lse(want), lse(got)
	d := 0.0
	for i := range want {
		if math.IsInf(float64(want[i]), -1) {
			continue
		}
		pw := float64(want[i]) - lw
		d += math.Exp(pw) * (pw - (float64(got[i]) - lg))
	}
	return d
}

// cfgSpan is upstream's guidance over the span, before the top-50 cut.
func cfgSpan(cond, uncond []float32) []float32 {
	out := make([]float32, len(cond))
	for i := range out {
		if math.IsInf(float64(cond[i]), -1) || math.IsInf(float64(uncond[i]), -1) {
			out[i] = float32(math.Inf(-1)) // masked: -inf - -inf would be NaN
			continue
		}
		out[i] = uncond[i] + 1.5*(cond[i]-uncond[i])
	}
	return out
}

// top50 is the span indices of x's 50 largest unmasked entries.
func top50(x, mask []float32) map[int]bool {
	idx := make([]int, 0, len(x))
	for i := range x {
		if !math.IsInf(float64(mask[i]), -1) {
			idx = append(idx, i)
		}
	}
	sort.Slice(idx, func(a, b int) bool { return x[idx[a]] > x[idx[b]] })
	out := map[int]bool{}
	for _, i := range idx[:50] {
		out[i] = true
	}
	return out
}

func relErr(got, want []float32) float64 {
	var se, sw float64
	for i := range want {
		if math.IsInf(float64(want[i]), 0) {
			continue
		}
		d := float64(got[i] - want[i])
		se += d * d
		sw += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(se / sw)
}

// TestLMFrames is N1's gate: the oracle's prompt prefilled into both CFG
// slots, then its own frames' feedback fed back one step at a time
// (teacher-forced), and every step's head logits and hidden compared with
// fp32's. On the fp16 bank it gates the implementation (KL ≤ 1e-3, ace/lm's
// A7 bound); on the int8 bank it is priced against upstream's bf16 (A10's
// rule: mean KL after CFG no worse than bf16's, worst within 2x of bf16's
// worst).
func TestLMFrames(t *testing.T) {
	m := loadManifest(t)
	dev, err := openDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	bank := testBank()
	t0 := time.Now()
	g, err := LoadLM(dev, filepath.Join(modelDir, "language_model"), Options{MaxLen: 2048, Slots: 2, Rows: 512, Bank: bank})
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	defer g.Destroy()
	t.Logf("staged the LM (%s bank) in %v", bank, time.Since(t0))

	shape, ids := refInt32(t, m, "prompt_ids")
	T := shape[1]
	_, wantC := refFloat32(t, m, "head_cond")
	_, wantU := refFloat32(t, m, "head_uncond")
	_, wantH := refFloat32(t, m, "hidden")
	_, feed := refFloat32(t, m, "feedback")
	span := HeadHi - HeadLo

	// Prefill every token but the last, both slots, in passes of the arena.
	var pre []Row
	for s := 0; s < 2; s++ {
		for i := 0; i < T-1; i++ {
			pre = append(pre, Row{ID: ids[s*T+i], Slot: s, Pos: i})
		}
	}
	t0 = time.Now()
	for len(pre) > 0 {
		n := min(len(pre), g.o.Rows)
		if _, _, err := g.Pass(pre[:n], false); err != nil {
			t.Fatal(err)
		}
		pre = pre[n:]
	}
	t.Logf("prefill: 2 x %d tokens in %v", T-1, time.Since(t0))

	step := []Row{{ID: ids[T-1], Slot: 0, Pos: T - 1}, {ID: ids[2*T-1], Slot: 1, Pos: T - 1}}
	var sumKL, sumBF, worstKL, worstBF, worstRel, worstHid float64
	var tDec time.Duration
	agree := 0
	for f := 0; f <= m.Frames; f++ {
		out, d, err := g.Pass(step, true)
		if err != nil {
			t.Fatal(err)
		}
		tDec += d
		wc, wu := wantC[f*span:(f+1)*span], wantU[f*span:(f+1)*span]
		gc, gu := out.Logits[0], out.Logits[1]
		k := spanKL(cfgSpan(wc, wu), cfgSpan(gc, gu))
		rel := math.Max(relErr(gc, wc), relErr(gu, wu))
		hid := math.Max(relErr(out.Hidden[0], wantH[(2*f)*hidden:(2*f+1)*hidden]),
			relErr(out.Hidden[1], wantH[(2*f+1)*hidden:(2*f+2)*hidden]))
		tw, tg := top50(wc, wc), top50(gc, wc)
		shared := 0
		for i := range tw {
			if tg[i] {
				shared++
			}
		}
		if shared == 50 {
			agree++
		}
		bf := m.Steps[f].KLCFG
		sumKL += k
		sumBF += bf
		worstKL, worstBF = math.Max(worstKL, k), math.Max(worstBF, bf)
		worstRel, worstHid = math.Max(worstRel, rel), math.Max(worstHid, hid)
		if f < 4 || f%8 == 0 || f == m.Frames {
			t.Logf("step %2d: KL cond %.2e uncond %.2e cfg %.2e (bf16 %.2e), rel %.2e, hidden %.2e, top-50 shared %d (bf16 %d)",
				f, spanKL(wc, gc), spanKL(wu, gu), k, bf, rel, hid, shared, m.Steps[f].Top50)
		}
		if f == m.Frames {
			break
		}
		v := feed[f*hidden : (f+1)*hidden]
		step = []Row{{Vec: v, Slot: 0, Pos: T + f}, {Vec: v, Slot: 1, Pos: T + f}}
	}
	n := float64(m.Frames + 1)
	t.Logf("%s: CFG KL mean %.2e worst %.2e (bf16 %.2e, %.2e); logits rel worst %.2e, hidden %.2e; top-50 identical on %d/%d; %.1f ms a 2-row step (GPU)",
		bank, sumKL/n, worstKL, sumBF/n, worstBF, worstRel, worstHid, agree, m.Frames+1, float64(tDec.Microseconds())/1e3/n)
	switch bank {
	case BankFP16:
		if !(worstKL <= 1e-3) {
			t.Errorf("fp16 bank: worst CFG KL %.2e, over 1e-3", worstKL)
		}
	case BankQ8:
		if !(sumKL <= sumBF && worstKL <= 2*worstBF) {
			t.Errorf("int8 bank: CFG KL mean %.2e worst %.2e against bf16's %.2e, %.2e", sumKL/n, worstKL, sumBF/n, worstBF)
		}
	}
}

// embedRows reads rows of the global LM's embedding table straight from the
// checkpoint (a test of the local LM needs the semantic codes' rows, not a
// 16 GB LM).
func embedRows(t testing.TB, ids []int) [][]float32 {
	t.Helper()
	set, err := safetensors.OpenSet(filepath.Join(modelDir, "language_model"))
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	defer set.Close()
	emb, err := set.Get("model.embed_tokens.weight")
	if err != nil {
		t.Fatal(err)
	}
	out := make([][]float32, len(ids))
	for i, id := range ids {
		out[i] = make([]float32, hidden)
		for j := range out[i] {
			out[i][j] = bf16(binary.LittleEndian.Uint16(emb.Data[2*(id*hidden+j):]))
		}
	}
	return out
}

// TestDepthFrames is N2's gate: every frame of the oracle's run through the
// local LM, teacher-forced (the oracle's global hidden in, its codes
// picked), each code's head logits and hidden compared with fp32's, and the
// frame's feedback rebuilt from the codes. On the fp16 bank the worst CFG KL
// is held to 1e-3, N1's bound.
func TestDepthFrames(t *testing.T) {
	m := loadManifest(t)
	dev, err := openDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	bank := testBank()
	t0 := time.Now()
	d, err := LoadDepth(dev, filepath.Join(modelDir, "rvq_depth_decoder"), bank)
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	defer d.Destroy()
	t.Logf("staged the local LM (%s bank) in %v", bank, time.Since(t0))

	_, codes := refInt32(t, m, "codes")
	_, wantH := refFloat32(t, m, "hidden")
	_, wantDH := refFloat32(t, m, "depth_hidden")
	_, wantC := refFloat32(t, m, "depth_cond")
	_, wantU := refFloat32(t, m, "depth_uncond")
	_, feed := refFloat32(t, m, "feedback")
	F := m.Frames + 1
	ids := make([]int, F)
	for f := range ids {
		ids[f] = CodeOffset + int(codes[f*8])
	}
	sems := embedRows(t, ids)

	var worstKL, sumKL, worstHid, worstFeed float64
	var took time.Duration
	agree, total := 0, 0
	for f := 0; f < F; f++ {
		last := [dSlots][]float32{wantH[(2*f)*hidden : (2*f+1)*hidden], wantH[(2*f+1)*hidden : (2*f+2)*hidden]}
		out, dur, err := d.Frame(last, sems[f], func(i int, _, _ []float32) int { return int(codes[f*8+1+i]) })
		if err != nil {
			t.Fatal(err)
		}
		took += dur
		var msg string
		for i := 0; i < dCodes; i++ {
			o := (f*dCodes + i) * dVocab
			wc, wu := wantC[o:o+dVocab], wantU[o:o+dVocab]
			k := spanKL(cfgSpan(wc, wu), cfgSpan(out.Logits[i][0], out.Logits[i][1]))
			hid := relErr(out.Hidden[i], wantDH[(f*dCodes+i)*hidden:(f*dCodes+i+1)*hidden])
			tw, tg := top50(cfgSpan(wc, wu), wc), top50(cfgSpan(out.Logits[i][0], out.Logits[i][1]), wc)
			same := true
			for j := range tw {
				same = same && tg[j]
			}
			if same {
				agree++
			}
			total++
			worstKL, worstHid = math.Max(worstKL, k), math.Max(worstHid, hid)
			sumKL += k
			if i == 0 || i == dCodes-1 {
				msg += fmt.Sprintf(" c%d KL %.2e hidden %.2e;", i+1, k, hid)
			}
		}
		var c [dCodes]int
		for i := range c {
			c[i] = int(codes[f*8+1+i])
		}
		fe := relErr(d.Feedback(sems[f], c), feed[f*hidden:(f+1)*hidden])
		worstFeed = math.Max(worstFeed, fe)
		if f < 3 || f%16 == 0 || f == F-1 {
			t.Logf("frame %2d:%s feedback %.2e", f, msg, fe)
		}
	}
	t.Logf("%s: CFG KL mean %.2e worst %.2e, hidden rel worst %.2e, feedback rel worst %.2e; top-50 identical %d/%d; %.1f ms a frame (7 passes, GPU)",
		bank, sumKL/float64(total), worstKL, worstHid, worstFeed, agree, total, float64(took.Microseconds())/1e3/float64(F))
	if worstFeed > 1e-6 {
		t.Errorf("feedback rel %.2e from the oracle's", worstFeed)
	}
	if bank == BankFP16 && !(worstKL <= 1e-3) {
		t.Errorf("fp16 bank: worst CFG KL %.2e, over 1e-3", worstKL)
	}
}

// TestSlabLadder times a 2-row decode step with each projection's split-K
// moved through the rungs that divide its K, the others held at the table's
// (ace/lm's A10 ladder at this model's widths). MM3_LADDER=1 runs it.
func TestSlabLadder(t *testing.T) {
	if os.Getenv("MM3_LADDER") == "" {
		t.Skip("MM3_LADDER=1 runs the ladder")
	}
	m := loadManifest(t)
	dev, err := openDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	bank := testBank()
	g, err := LoadLM(dev, filepath.Join(modelDir, "language_model"), Options{MaxLen: 2048, Slots: 2, Rows: 512, Bank: bank, allSlabs: true})
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	defer g.Destroy()
	shape, ids := refInt32(t, m, "prompt_ids")
	T := shape[1]
	var pre []Row
	for i := 0; i < T-1; i++ {
		pre = append(pre, Row{ID: ids[i], Slot: 0, Pos: i}, Row{ID: ids[T+i], Slot: 1, Pos: i})
	}
	for len(pre) > 0 {
		n := min(len(pre), g.o.Rows)
		if _, _, err := g.Pass(pre[:n], false); err != nil {
			t.Fatal(err)
		}
		pre = pre[n:]
	}
	step := []Row{{ID: ids[T-1], Slot: 0, Pos: T - 1}, {ID: ids[2*T-1], Slot: 1, Pos: T - 1}}
	timeStep := func() float64 {
		var ds []float64
		for i := 0; i < 15; i++ {
			_, d, err := g.Pass(step, true)
			if err != nil {
				t.Fatal(err)
			}
			ds = append(ds, float64(d.Microseconds())/1e3)
		}
		sort.Float64s(ds)
		return ds[len(ds)/2]
	}
	t.Logf("%s table: %.2f ms", bank, timeStep())
	for p := pQ; p < nProj; p++ {
		kt := projShape[p][1] / tile
		var msg string
		best, bestS := math.Inf(1), 0
		for _, s := range []int{1, 2, 4, 8, 16, 20, 32, 40} {
			if kt%s != 0 || (kt/s)%4 != 0 {
				continue
			}
			g.slabOverride[p] = s
			ms := timeStep()
			msg += fmt.Sprintf(" k%d %.2f", s, ms)
			if ms < best {
				best, bestS = ms, s
			}
		}
		g.slabOverride[p] = 0
		t.Logf("%s (table k%d):%s -> k%d", projNames[p], g.slabs(p), msg, bestS)
	}
}
