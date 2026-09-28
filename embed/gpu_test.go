package embed

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// safetensorsOpen opens the checkpoint for the tests that build an encoder
// themselves rather than through NewGPU.
func safetensorsOpen(t *testing.T) (*safetensors.Set, error) {
	t.Helper()
	return safetensors.OpenSet(modelDir)
}

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("embed-test")
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
			break
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
		t.Skipf("no subgroup size control query: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// The device path is held to two bounds rather than to a per-element one,
// and TestGPULadder is why. Worst-element relative error on this model is a
// misleading statistic: fp16 projections leave 5.4e-2 of the tensor RMS on a
// single near-zero component of `prenorm` while the error *as a whole* is
// 1.9e-3 of it and every row still points where the reference's does to
// 1.2e-5. So what is asserted is the size of the error tensor (errRMSTol)
// and the direction of each row (rowCosTol), which are the two things a
// pooled, normalised vector is actually made of.
const (
	errRMSTol = 5e-3
	rowCosTol = 1e-4
)

// cosTol is what matters more than any tensor: an embedding that agrees with
// the reference to this in *cosine* is the same vector for retrieval
// purposes, whatever the per-element drift is.
const cosTol = 1e-4

func newGPU(t *testing.T, dev *vk.Device, maxTokens int) *GPU {
	t.Helper()
	skipWithoutCheckpoint(t)
	g, err := NewGPU(dev, modelDir, maxTokens)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// TestGPUHidden checks the device's whole hidden state against the dump,
// which is the tensor-level acceptance check for the stack.
func TestGPUHidden(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	g := newGPU(t, dev, m.Seq)
	defer g.Destroy()

	h, err := g.Hidden(m.IDs[0])
	if err != nil {
		t.Fatal(err)
	}
	want := loadRef(t, m, "final")
	maxAbs, rms, rel, _ := deviation(h, want)
	relRMS, worstCos := errStats(h, want)
	if relRMS > errRMSTol {
		t.Errorf("final %s: the error is %.3g of the tensor RMS, over %.0e", h, relRMS, errRMSTol)
	}
	if 1-worstCos > rowCosTol {
		t.Errorf("final %s: worst row cosine %.7f", h, worstCos)
	}
	t.Logf("final %s max abs %.3g  rms %.4g  worst-element rel %.2g  err/rms %.2g  worst row cos %.7f",
		h, maxAbs, rms, rel, relRMS, worstCos)
}

// TestGPUEmbedding is the one that decides whether this vertical works: the
// four texts of the model card, embedded on the device, scored against each
// other, against the card's own numbers.
func TestGPUEmbedding(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	longest := 0
	for _, ids := range m.IDs {
		longest = max(longest, len(ids))
	}
	g := newGPU(t, dev, longest)
	defer g.Destroy()

	vecs := make([][]float32, len(m.Texts))
	for i, text := range m.Texts {
		v, err := g.Embed(text)
		if err != nil {
			t.Fatal(err)
		}
		vecs[i] = v
	}

	// Against the reference vectors first, element for element and then as a
	// cosine, because those two can disagree: a vector that is 1e-3 off
	// everywhere is still the same direction.
	all := loadRef(t, m, "embeddings_all")
	for i := range vecs {
		ref := all.Row(i)
		got := &qwen.Mat{Rows: 1, Cols: len(vecs[i]), Data: vecs[i]}
		wantMat := &qwen.Mat{Rows: 1, Cols: len(ref), Data: ref}
		maxAbs, rms, rel, _ := deviation(got, wantMat)
		cos := float64(Cosine(vecs[i], ref))
		if 1-cos > cosTol {
			t.Errorf("text %d: cosine against the reference embedding is %.6f", i, cos)
		}
		t.Logf("text %d: cos %.6f  max abs %.3g  rms %.4g  rel %.2g", i, cos, maxAbs, rms, rel)
	}

	// And the card's matrix, which is the end-to-end number.
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			got := float64(Cosine(vecs[i], vecs[2+j]))
			want := m.CardScores[i][j]
			if math.Abs(got-want) > 2e-3 {
				t.Errorf("score[%d][%d] is %.6f, want %.6f (the model card's)", i, j, got, want)
			}
			t.Logf("score[%d][%d] %.6f  card %.6f", i, j, got, want)
		}
	}
	if Cosine(vecs[0], vecs[2]) <= Cosine(vecs[0], vecs[3]) {
		t.Errorf("query 0 does not prefer document 0")
	}
}

// TestGPULatency is not a benchmark -- it is the number EMBEDDING.md quotes,
// measured the way a caller experiences it: tokenize, run, pool, normalise.
// The GPU-side breakdown is cmd/embed -profile's.
func TestGPULatency(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the model onto the device")
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	const maxTokens = 512
	start := time.Now()
	g := newGPU(t, dev, maxTokens)
	defer g.Destroy()
	load := time.Since(start)

	text := m.Texts[3]
	ids, err := g.Tok.Encode(text)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.EmbedIDs(ids); err != nil { // warm the clock and the pipelines
		t.Fatal(err)
	}
	const iters = 20
	start = time.Now()
	for i := 0; i < iters; i++ {
		if _, err := g.EmbedIDs(ids); err != nil {
			t.Fatal(err)
		}
	}
	each := time.Since(start) / iters
	t.Logf("load %v; %d tokens in %v (%.0f texts/s, %.0f tok/s)",
		load.Round(time.Millisecond), len(ids), each.Round(time.Microsecond),
		float64(time.Second)/float64(each), float64(len(ids))*float64(time.Second)/float64(each))
}

// errStats is the pair of numbers a per-element bound does not give: the RMS
// of the error against the RMS of the tensor, and the worst row-wise cosine.
// Drift that is spread over a tensor moves the first and not the second; a
// kernel that is wrong moves both.
func errStats(got, want *qwen.Mat) (relRMS, worstCos float64) {
	var errSq, refSq float64
	for i := range want.Data {
		d := float64(got.Data[i]) - float64(want.Data[i])
		errSq += d * d
		refSq += float64(want.Data[i]) * float64(want.Data[i])
	}
	worstCos = 1
	for r := 0; r < want.Rows; r++ {
		var dot, ga, wa float64
		g, w := got.Row(r), want.Row(r)
		for i := range w {
			dot += float64(g[i]) * float64(w[i])
			ga += float64(g[i]) * float64(g[i])
			wa += float64(w[i]) * float64(w[i])
		}
		if ga > 0 && wa > 0 {
			worstCos = math.Min(worstCos, dot/math.Sqrt(ga*wa))
		}
	}
	return math.Sqrt(errSq / refSq), worstCos
}

// TestGPULadder is the diagnostic behind the tolerance above: the same run
// truncated to 1, 2, 14 and 28 layers, each against its own dumped hidden
// state. Fp16 drift down a residual stream compounds smoothly; a kernel that
// is wrong does not, so the shape of this ladder is what says which of the
// two the final tensor's error is.
func TestGPULadder(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the encoder four times")
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	skipWithoutCheckpoint(t)
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensorsOpen(t)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	for _, c := range []struct {
		layers int
		ref    string
	}{{1, "hidden_1"}, {2, "hidden_2"}, {14, "hidden_14"}, {cfg.NumLayers, "prenorm"}} {
		enc, err := qwen.NewGPUEncoder(dev, set, cfg, c.layers, m.Seq, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, err := enc.Forward(m.IDs[0])
		if err != nil {
			t.Fatal(err)
		}
		want := loadRef(t, m, c.ref)
		maxAbs, rms, rel, _ := deviation(got, want)
		relRMS, worstCos := errStats(got, want)
		t.Logf("%2d layers  %-10s max abs %9.3g  rms %9.4g  rel %.2g  err/rms %.2g  worst row cos %.6f",
			c.layers, c.ref, maxAbs, rms, rel, relRMS, worstCos)
		enc.Destroy()
	}
}

// batchTexts is a set of inputs whose lengths cover the cases RunBatch lays
// out differently: one token short of a key block, exactly one, one over,
// several blocks, and the model card's four.
func batchTexts(t *testing.T, g *GPU, m *manifest) [][]int32 {
	t.Helper()
	var seqs [][]int32
	for _, text := range m.Texts {
		ids, err := g.Tok.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		seqs = append(seqs, ids)
	}
	words := "the quick brown fox jumps over a lazy dog while seven engineers argue about memory bandwidth "
	for _, n := range []int{63, 64, 65, 1, 150, 17} {
		ids, err := g.Tok.Encode(strings.Repeat(words, 20))
		if err != nil {
			t.Fatal(err)
		}
		// Keep the appended end-of-text token last, as EncodeLimit does.
		s := append(append([]int32{}, ids[:n-1]...), ids[len(ids)-1])
		seqs = append(seqs, s)
	}
	return seqs
}

// TestGPUBatchMatchesSingle is E7's gate: a text's vector out of a batch is
// the vector it gets alone, **bit for bit**, under one plan. That is the
// strongest statement available and the right one -- rows never meet in a
// projection, and a text's attention region is laid out as its own run lays
// it out -- so any difference at all is a leak between texts or a layout
// bug, not rounding. Then again under AutoPlan, where the batch runs a
// wider GEMM rung than a lone text and only the direction can be held.
func TestGPUBatchMatchesSingle(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	g := newGPU(t, dev, 1024)
	defer g.Destroy()
	seqs := batchTexts(t, g, m)

	single := func() [][]float32 {
		out := make([][]float32, len(seqs))
		for i, s := range seqs {
			v, err := g.EmbedIDs(s)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = v
		}
		return out
	}

	for _, k := range []qwen.GEMMKernel{qwen.GEMMReg16x64K8, qwen.GEMMReg64} {
		g.AutoPlan = false
		if err := g.Enc.SetPlan(qwen.UniformGEMMPlan(k)); err != nil {
			t.Fatal(err)
		}
		want := single()
		// Forwards, then reversed, so every text has different neighbours
		// and a different place in the planes.
		for pass, order := range [][]int{identity(len(seqs)), reversed(len(seqs))} {
			batch := make([][]int32, len(order))
			for i, j := range order {
				batch[i] = seqs[j]
			}
			got, err := g.EmbedBatch(batch)
			if err != nil {
				t.Fatal(err)
			}
			for i, j := range order {
				for e := range got[i] {
					if got[i][e] != want[j][e] {
						t.Errorf("%s pass %d: text %d (%d tokens) differs from its lone run at element %d: %g vs %g",
							k, pass, j, len(seqs[j]), e, got[i][e], want[j][e])
						break
					}
				}
			}
		}
	}

	g.AutoPlan = true
	want := single()
	got, err := g.EmbedBatch(seqs)
	if err != nil {
		t.Fatal(err)
	}
	worst := 1.0
	for i := range seqs {
		worst = min(worst, float64(Cosine(got[i], want[i])))
	}
	if 1-worst > 1e-5 {
		t.Errorf("auto plan: worst cosine between batched and lone vectors is %.7f", worst)
	}
	t.Logf("auto plan: worst cosine batched vs lone %.8f over %d texts", worst, len(seqs))

	// And the card's matrix out of one batch of its four texts.
	card, err := g.EmbedBatch(seqs[:4])
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			got := float64(Cosine(card[i], card[2+j]))
			if want := m.CardScores[i][j]; math.Abs(got-want) > 2e-3 {
				t.Errorf("batched score[%d][%d] is %.6f, want %.6f (the model card's)", i, j, got, want)
			}
		}
	}
}

// TestGPUBatchLeakIsCaught is the control for the gate above: the same texts
// run as *one* sequence -- what batching without per-text attention would
// be -- must come back different, or the bit-for-bit comparison could not
// see a leak.
func TestGPUBatchLeakIsCaught(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	g := newGPU(t, dev, 1024)
	defer g.Destroy()
	seqs := batchTexts(t, g, m)[:4]
	lone, err := g.EmbedIDs(seqs[1])
	if err != nil {
		t.Fatal(err)
	}
	leaked, err := g.EmbedIDs(append(append([]int32{}, seqs[0]...), seqs[1]...))
	if err != nil {
		t.Fatal(err)
	}
	cos := float64(Cosine(lone, leaked))
	if cos > 0.999 {
		t.Errorf("text 1 after text 0 in one sequence is cosine %.6f to text 1 alone; a leak would not be seen", cos)
	}
	t.Logf("text 1 with text 0 visible: cosine %.4f to its lone vector", cos)
}

// TestGPUBatchThroughput is the E7 number: short texts one at a time against
// the same texts as batches.
func TestGPUBatchThroughput(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the model onto the device")
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	g := newGPU(t, dev, 4096)
	defer g.Destroy()
	g.Enc.PerSubmit = 1 << 10

	words := strings.Fields("the quick brown fox jumps over a lazy dog while seven engineers argue about memory bandwidth and caches")
	for _, size := range []int{12, 40, 120} {
		for _, b := range []int{1, 8, 32, 64} {
			seqs := make([][]int32, b)
			for i := range seqs {
				text := ""
				for w := 0; len(text) < size*5; w++ {
					text += words[(w*7+i*3)%len(words)] + " "
				}
				ids, err := g.Tok.Encode(text + m.Texts[i%len(m.Texts)][:8])
				if err != nil {
					t.Fatal(err)
				}
				seqs[i] = ids
			}
			lens := make([]int, b)
			rows := 0
			for i := range seqs {
				lens[i] = len(seqs[i])
				rows += lens[i]
			}
			if !g.Fits(lens) {
				continue
			}
			best := func(f func()) time.Duration {
				f()
				d := time.Duration(math.MaxInt64)
				for r := 0; r < 3; r++ {
					start := time.Now()
					f()
					d = min(d, time.Since(start))
				}
				return d
			}
			alone := best(func() {
				for _, s := range seqs {
					if _, err := g.EmbedIDs(s); err != nil {
						t.Fatal(err)
					}
				}
			})
			batched := best(func() {
				if _, err := g.EmbedBatch(seqs); err != nil {
					t.Fatal(err)
				}
			})
			t.Logf("%3d texts x ~%3d tokens (%5d rows): alone %8v (%6.2f ms/text)  batched %8v (%6.2f ms/text)  %5.2fx",
				b, rows/b, rows, alone.Round(10*time.Microsecond), ms(alone)/float64(b),
				batched.Round(10*time.Microsecond), ms(batched)/float64(b), float64(alone)/float64(batched))
		}
	}
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func identity(n int) []int {
	o := make([]int, n)
	for i := range o {
		o[i] = i
	}
	return o
}

func reversed(n int) []int {
	o := make([]int, n)
	for i := range o {
		o[i] = n - 1 - i
	}
	return o
}

// TestGPUBatchLadder prices every GEMM rung at batch sizes past the 602
// tokens embed.PlanFor was fitted to, and the command-buffer size at one
// text, which is what a lone query's latency is made of.
func TestGPUBatchLadder(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the model onto the device")
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	g := newGPU(t, dev, 4096)
	defer g.Destroy()
	best := func(f func()) time.Duration {
		f()
		d := time.Duration(math.MaxInt64)
		for r := 0; r < 5; r++ {
			start := time.Now()
			f()
			d = min(d, time.Since(start))
		}
		return d
	}
	ids, err := g.Tok.Encode(m.Texts[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, every := range []int{8, 16, 32, 1 << 10} {
		g.Enc.PerSubmit = every
		d := best(func() {
			if _, err := g.EmbedIDs(ids); err != nil {
				t.Fatal(err)
			}
		})
		t.Logf("one text, %d tokens, %4d dispatches a submit: %v", len(ids), every, d.Round(10*time.Microsecond))
	}
	g.Enc.PerSubmit = 1 << 10

	seqs := batchTexts(t, g, m)
	g.AutoPlan = false
	for _, reps := range []int{1, 2, 4} {
		var batch [][]int32
		rows := 0
		for r := 0; r < reps; r++ {
			for _, s := range seqs {
				batch = append(batch, s)
				rows += len(s)
			}
		}
		line := ""
		for _, k := range qwen.GEMMKernels() {
			if err := g.Enc.SetPlan(qwen.UniformGEMMPlan(k)); err != nil {
				t.Fatal(err)
			}
			d := best(func() {
				if _, err := g.EmbedBatch(batch); err != nil {
					t.Fatal(err)
				}
			})
			line += fmt.Sprintf("  %s %.2f", k, ms(d))
		}
		t.Logf("%4d rows, %3d texts:%s", rows, len(batch), line)
	}
}
