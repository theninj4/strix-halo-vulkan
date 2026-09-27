package lm

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

var (
	devOnce sync.Once
	testDev *vk.Device
	devErr  error

	gpuMu sync.Mutex
	gpuLM = map[Bank]*GPU{}
)

// testBank is the bank the tests stage: int8 (served), or ACE_LM_BANK=fp16
// for the control.
func testBank() Bank {
	if os.Getenv("ACE_LM_BANK") == "fp16" {
		return BankFP16
	}
	return BankQ8
}

// loadGPU stages the LM in testBank's width, once for the package's tests.
func loadGPU(t testing.TB) *GPU { return loadGPUBank(t, testBank()) }

// loadGPUBank stages the LM in a given width once (3.9 GB of int8 layers or
// 7.3 of fp16, and a 1.1 GB fp16 head); the two share a device.
func loadGPUBank(t testing.TB, bank Bank) *GPU {
	t.Helper()
	devOnce.Do(func() { testDev, devErr = openDevice() })
	if devErr != nil {
		t.Skipf("no device: %v", devErr)
	}
	gpuMu.Lock()
	defer gpuMu.Unlock()
	if g := gpuLM[bank]; g != nil {
		return g
	}
	t0 := time.Now()
	g, err := Load(testDev, lmDir, Options{MaxLen: 4096, Slots: 2, Rows: 256, Bank: bank,
		allSlabs: os.Getenv("ACE_LM_LADDER") != ""})
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	t.Logf("staged the LM (%s bank) in %v", bank, time.Since(t0))
	gpuLM[bank] = g
	return g
}

// logitStats compares a row of logits with the oracle's over [lo, hi):
// the error relative to the oracle's rms, and the KL divergence of the
// sampling distribution (temperature 0.85) from the oracle's.
type logitStats struct {
	rel, maxAbs, kl float64
	argmax          bool
}

func compareLogits(got, want []float32, lo, hi int) logitStats {
	var se, sw, mx float64
	ga, wa := lo, lo
	for i := lo; i < hi; i++ {
		d := float64(got[i] - want[i])
		se += d * d
		sw += float64(want[i]) * float64(want[i])
		mx = math.Max(mx, math.Abs(d))
		if got[i] > got[ga] {
			ga = i
		}
		if want[i] > want[wa] {
			wa = i
		}
	}
	return logitStats{rel: math.Sqrt(se / sw), maxAbs: mx, kl: kl(got[lo:hi], want[lo:hi], 0.85), argmax: ga == wa}
}

// kl is KL(want || got) of softmax(x / temp).
func kl(got, want []float32, temp float64) float64 {
	lse := func(x []float32) float64 {
		m := math.Inf(-1)
		for _, v := range x {
			m = math.Max(m, float64(v)/temp)
		}
		s := 0.0
		for _, v := range x {
			s += math.Exp(float64(v)/temp - m)
		}
		return m + math.Log(s)
	}
	lg, lw := lse(got), lse(want)
	d := 0.0
	for i := range want {
		pw := float64(want[i])/temp - lw
		pg := float64(got[i])/temp - lg
		d += math.Exp(pw) * (pw - pg)
	}
	return d
}

// cfg is upstream's phase-2 guidance over the codes: uncond + s·(cond − uncond).
func cfg(cond, uncond []float32, s float32) []float32 {
	out := make([]float32, len(cond))
	for i := range out {
		out[i] = uncond[i] + s*(cond[i]-uncond[i])
	}
	return out
}

// TestLogits is A7a and A7b's gate, on the fp16 bank whatever testBank
// says: it gates the implementation against fp32, and TestBankAgainstBF16
// prices the int8 quantisation (MUSIC.md A10).
//
// It is the prompts prefilled, then the oracle's
// own sampled tokens fed back one step at a time (teacher-forced), and the
// logits at the steps it dumped compared -- phase 1 over the text tokens
// (all the FSM ever allows there), phase 2 over the codes, each row and
// after CFG.
func TestLogits(t *testing.T) {
	m := loadLMManifest(t)
	g := loadGPUBank(t, BankFP16)
	steps := map[int]bool{0: true, 1: true, 2: true, 3: true, 10: true, 50: true, 100: true}
	for _, label := range []string{"given_duration", "all_metas", "sample", "sample_ja"} {
		c, ok := m.Cases[label]
		if !ok {
			t.Logf("%s: not in the reference", label)
			continue
		}
		for ph, p := range c.Phases {
			if len(p.Steps) == 0 {
				continue // a sample case stops at the codes phase's door
			}
			rows := promptRows(t, m, label, ph)
			_, toks := refInt32(t, m, fmt.Sprintf("%s_p%d_tokens", label, ph))
			codes := p.Rows == 2
			lo, hi := 0, CodeBase
			if codes {
				lo, hi = CodeBase, CodeBase+NumCodes
			}
			// Prefill every row but its last token, all slots in one pass.
			var pre []Tok
			for s, r := range rows {
				for i, id := range r[:len(r)-1] {
					pre = append(pre, Tok{ID: id, Slot: s, Pos: i})
				}
			}
			t0 := time.Now()
			if _, _, err := g.Pass(pre, Range{}); err != nil {
				t.Fatal(err)
			}
			tPre := time.Since(t0)
			step := make([]Tok, len(rows))
			for s, r := range rows {
				step[s] = Tok{ID: r[len(r)-1], Slot: s, Pos: len(r) - 1}
			}
			var worst logitStats
			var tDec time.Duration
			nDec := 0
			for i := 0; i <= 100 && i <= len(toks); i++ {
				t0 := time.Now()
				out, _, err := g.Pass(step, Range{Lo: 0, Hi: g.vocab})
				if err != nil {
					t.Fatal(err)
				}
				tDec += time.Since(t0)
				nDec++
				if steps[i] {
					_, want := refFloat32(t, m, fmt.Sprintf("%s_p%d_logits%d", label, ph, i))
					var msg string
					for r := range rows {
						st := compareLogits(out[r], want[r*g.vocab:(r+1)*g.vocab], lo, hi)
						msg += fmt.Sprintf(" row %d: rel %.2e max %.3f KL %.2e argmax %v;", r, st.rel, st.maxAbs, st.kl, st.argmax)
						worst.rel = nanMax(worst.rel, st.rel)
						worst.kl = nanMax(worst.kl, st.kl)
					}
					if codes {
						gc := cfg(out[0][lo:hi], out[1][lo:hi], 2)
						wc := cfg(want[lo:hi], want[g.vocab+lo:g.vocab+hi], 2)
						st := compareLogits(gc, wc, 0, len(gc))
						msg += fmt.Sprintf(" cfg: rel %.2e KL %.2e", st.rel, st.kl)
						worst.kl = nanMax(worst.kl, st.kl)
					}
					t.Logf("%s phase %d step %3d:%s", label, ph, i, msg)
				}
				if i == len(toks) {
					break
				}
				for s := range step {
					step[s].ID = toks[i]
					step[s].Pos++
				}
			}
			t.Logf("%s phase %d: prefill %d rows %v; %d steps of %d rows, %.1f ms a step (wall)",
				label, ph, len(pre), tPre, nDec, len(rows), float64(tDec.Microseconds())/1e3/float64(nDec))
			// rel was gated at 2e-3, A7's observed worst (given_duration's
			// phase 1 step 3, the bpm value); the sample pass's same step is
			// 2.5e-3 at KL 5.7e-5 (MUSIC.md A12). KL is the bound that matters.
			if !(worst.kl <= 1e-3 && worst.rel <= 3e-3) { // NaN fails too
				t.Errorf("%s phase %d: worst rel %.2e KL %.2e", label, ph, worst.rel, worst.kl)
			}
		}
	}
}

// nanMax is max that keeps a NaN.
func nanMax(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}
	return math.Max(a, b)
}

// TestDeterministic runs one prefill and a few decode steps several times
// and wants the logits bit-identical: a race between a kernel's threads (the
// one ace_lm_prep had, a shared-memory reuse without a barrier, NaN'd a row
// about one pass in twelve) passes a tolerance gate most of the time.
func TestDeterministic(t *testing.T) {
	m := loadLMManifest(t)
	g := loadGPU(t)
	rows := promptRows(t, m, "given_duration", 1)
	var ref [][]float32
	for trial := 0; trial < 8; trial++ {
		var pre []Tok
		for s, r := range rows {
			for i, id := range r[:len(r)-1] {
				pre = append(pre, Tok{ID: id, Slot: s, Pos: i})
			}
		}
		if _, _, err := g.Pass(pre, Range{}); err != nil {
			t.Fatal(err)
		}
		var got [][]float32
		step := []Tok{{ID: rows[0][len(rows[0])-1], Slot: 0, Pos: len(rows[0]) - 1},
			{ID: rows[1][len(rows[1])-1], Slot: 1, Pos: len(rows[1]) - 1}}
		for i := 0; i < 4; i++ {
			out, _, err := g.Pass(step, Range{Lo: CodeBase, Hi: CodeBase + NumCodes})
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, out...)
			for s := range step {
				step[s].ID = CodeBase + int32(i*977)
				step[s].Pos++
			}
		}
		if trial == 0 {
			ref = got
			continue
		}
		for r := range got {
			for j := range got[r] {
				if math.Float32bits(got[r][j]) != math.Float32bits(ref[r][j]) {
					t.Fatalf("trial %d row %d col %d: %v vs %v", trial, r, j, got[r][j], ref[r][j])
				}
			}
		}
	}
}

const strixHaloDeviceID = 0x1586

// openDevice opens the Strix Halo's device with what the kernels need.
func openDevice() (*vk.Device, error) {
	inst, err := vk.NewInstance("ace-lm-test")
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

// TestPlanner runs both phases live for the oracle's given_duration request
// (caption, lyrics, 30 s): a CoT that parses into valid metadata, then
// exactly 150 codes. What the LM samples is its own (our RNG); this gates
// the loop, not the draws.
func TestPlanner(t *testing.T) {
	m := loadLMManifest(t)
	g := loadGPU(t)
	p, err := NewPlanner(g, loadTokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cases["given_duration"]
	req := requestOf(c.Request)
	rng := rand.New(rand.NewPCG(42, 1))
	meta, text, st1, err := p.Think(req.Caption, req.Lyrics, UserMeta(req), rng)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("phase 1: %v\n%s", st1, text)
	for _, k := range []string{"bpm", "caption", "duration", "keyscale", "language", "timesignature"} {
		if meta[k] == "" {
			t.Errorf("the CoT has no %s: %q", k, meta)
		}
	}
	if meta["duration"] != "30" {
		t.Errorf("duration %q, want the request's 30", meta["duration"])
	}
	target, err := TargetCodes(req.Duration, meta)
	if err != nil {
		t.Fatal(err)
	}
	codes, st2, err := p.Codes(req.Caption, req.Lyrics, meta, target, rng)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("phase 2: %v; first codes %v", st2, codes[:min(10, len(codes))])
	if len(codes) != 150 {
		t.Errorf("%d codes, want 150", len(codes))
	}
}

// TestSamplePlanner runs sample mode live (MUSIC.md A12) for a description
// that names its language and one that asks for an instrumental: a song
// with every field the FSM holds valid, genres from upstream's vocabulary,
// the named language injected, and lyrics (or "[Instrumental]"). The draws
// are our RNG's; this gates the loop and what it parses, not the words.
func TestSamplePlanner(t *testing.T) {
	g := loadGPU(t)
	p, err := NewPlanner(g, loadTokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	if err := p.LoadGenres(genresPath); err != nil {
		t.Skip(err)
	}
	for _, q := range []string{"a gentle japanese city pop song for a summer evening", "pure music: a slow ambient piano piece"} {
		lang, inst := DescriptionHints(q)
		user := Meta{}
		if lang != "" {
			user["language"] = lang
		}
		song, text, st, err := p.Sample(q, inst, user, rand.New(rand.NewPCG(42, 1)))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%q: %v\n%s", q, st, text)
		if song.Caption == "" || song.BPM < 30 || song.BPM > 300 || song.Duration < 10 || song.Duration > 600 ||
			song.KeyScale == "" || song.TimeSignature == "" || song.Genres == "" {
			t.Errorf("%q: song %+v", q, song)
		}
		if lang != "" && song.Language != lang {
			t.Errorf("%q: language %q, want the injected %q", q, song.Language, lang)
		}
		if _, whole := p.v.genres.children(pyLower(song.Genres)); !whole {
			t.Errorf("%q: genres %q are not in the vocabulary", q, song.Genres)
		}
		if song.Lyrics == "" || (inst && song.Lyrics != "[Instrumental]" && !strings.Contains(strings.ToLower(song.Lyrics), "instrumental")) {
			t.Logf("%q: lyrics %q", q, song.Lyrics)
		}
		if !inst && song.Lyrics == "" {
			t.Errorf("%q: no lyrics", q)
		}
	}
}
