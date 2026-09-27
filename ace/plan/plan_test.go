package plan

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/h3/vae"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/tokenizer"
)

// The reference comes from reference/dump_ace_plan.py: upstream's own
// inference.generate_music driven to the DiT with thinking off, the real
// generate_audio loop over a stub decoder, and ResidualFSQ. Regenerate with:
//
//	HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_plan.py
const (
	planRef    = "../../reference/out/aceplan"
	encoderDir = "../../models/Qwen3-Embedding-0.6B"
	ditDir     = "../../models/acestep-v15-xl-turbo"
)

type planManifest struct {
	Cases map[string]struct {
		Params struct {
			Caption       string  `json:"caption"`
			Lyrics        string  `json:"lyrics"`
			BPM           int     `json:"bpm"`
			KeyScale      string  `json:"keyscale"`
			TimeSignature string  `json:"timesignature"`
			Duration      float64 `json:"duration"`
			Language      string  `json:"vocal_language"`
		} `json:"params"`
		TextPrompt      string         `json:"text_prompt"`
		LyricsPrompt    string         `json:"lyrics_prompt"`
		LatentLength    int            `json:"latent_length"`
		SrcIsSilence    bool           `json:"src_is_silence"`
		ChunkMaskValues []float64      `json:"chunk_mask_values"`
		IsCovers        []bool         `json:"is_covers"`
		ReferShape      []int          `json:"refer_shape"`
		ReferSilence750 bool           `json:"refer_is_silence750"`
		Scalars         map[string]any `json:"scalars"`
	} `json:"cases"`
	Sampler map[string]struct {
		T         int       `json:"T"`
		Shift     float64   `json:"shift"`
		DCW       bool      `json:"dcw"`
		Timesteps []float64 `json:"timesteps"`
	} `json:"sampler"`
	FSQ struct {
		Indices []int `json:"indices"`
	} `json:"fsq"`
	Tensors map[string]struct {
		Shape []int  `json:"shape"`
		Dtype string `json:"dtype"`
	} `json:"tensors"`
}

func loadPlan(t *testing.T) *planManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(planRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_ace_plan.py", planRef, err)
	}
	var m planManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func readRaw(t *testing.T, m *planManifest, name, dtype string) []byte {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	if meta.Dtype != dtype {
		t.Fatalf("%s is %s, want %s", name, meta.Dtype, dtype)
	}
	raw, err := os.ReadFile(filepath.Join(planRef, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func f32s(t *testing.T, m *planManifest, name string) []float32 {
	raw := readRaw(t, m, name, "float32")
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

func i32s(t *testing.T, m *planManifest, name string) []int32 {
	raw := readRaw(t, m, name, "int32")
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// exact32 fails on the first value that differs by so much as a bit.
func exact32(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] = %.9g, want %.9g", name, i, got[i], want[i])
		}
	}
}

func request(m *planManifest, label string) *Request {
	p := m.Cases[label].Params
	return &Request{Caption: p.Caption, Lyrics: p.Lyrics, BPM: p.BPM, KeyScale: p.KeyScale,
		TimeSignature: p.TimeSignature, Duration: p.Duration, Language: p.Language}
}

// TestRequests: the prompts, token ids and latent lengths of every captured
// request, and the fixed inputs (silence source, timbre, masks, flags) the
// code assumes.
func TestRequests(t *testing.T) {
	m := loadPlan(t)
	tok, err := tokenizer.Load(encoderDir)
	if err != nil {
		t.Skipf("no text encoder tokenizer: %v", err)
	}
	for label, c := range m.Cases {
		r := request(m, label)
		if got := r.CaptionPrompt(); got != c.TextPrompt {
			t.Errorf("%s: caption prompt\n got %q\nwant %q", label, got, c.TextPrompt)
		}
		if got := r.LyricsPrompt(); got != c.LyricsPrompt {
			t.Errorf("%s: lyrics prompt\n got %q\nwant %q", label, got, c.LyricsPrompt)
		}
		if got := r.LatentLength(); got != c.LatentLength {
			t.Errorf("%s: latent length %d, want %d", label, got, c.LatentLength)
		}
		for _, p := range []struct {
			name, text string
			max        int
		}{{"_text_ids", c.TextPrompt, TextMaxTokens}, {"_lyric_ids", c.LyricsPrompt, LyricMaxTokens}} {
			ids, err := tok.Encode(p.text)
			if err != nil {
				t.Fatal(err)
			}
			ids = Tokens(ids, p.max)
			want := i32s(t, m, label+p.name)
			if len(ids) != len(want) {
				t.Errorf("%s%s: %d ids, want %d", label, p.name, len(ids), len(want))
				continue
			}
			for i := range ids {
				if ids[i] != want[i] {
					t.Errorf("%s%s[%d] = %d, want %d", label, p.name, i, ids[i], want[i])
					break
				}
			}
		}
		// What the code takes as given for a text2music request.
		if !c.SrcIsSilence || !c.ReferSilence750 || len(c.IsCovers) != 1 || c.IsCovers[0] {
			t.Errorf("%s: src silence %v, timbre silence %v, covers %v", label, c.SrcIsSilence, c.ReferSilence750, c.IsCovers)
		}
		if len(c.ChunkMaskValues) != 1 || c.ChunkMaskValues[0] != ChunkMask {
			t.Errorf("%s: chunk mask values %v, want [%v]", label, c.ChunkMaskValues, ChunkMask)
		}
		if s := c.Scalars; s["shift"] != DefaultShift || s["dcw_enabled"] != true || s["dcw_mode"] != "double" ||
			s["dcw_scaler"] != DefaultDCW.Low || s["dcw_high_scaler"] != DefaultDCW.High || s["infer_method"] != "ode" ||
			s["timesteps"] != nil {
			t.Errorf("%s: sampler arguments %v", label, s)
		}
	}
}

// TestNoise: a CPU run's seed-42 noise is torch's CPU randn, which h3/vae
// reproduces to within Sleef's last bit.
func TestNoise(t *testing.T) {
	m := loadPlan(t)
	want := f32s(t, m, "noise_seed42")
	got, err := vae.TorchRandn(42, len(want))
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for i := range got {
		worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
	}
	if worst > 1e-6 {
		t.Fatalf("noise max abs diff %.3g", worst)
	}
	t.Logf("noise: %d values, max abs diff %.3g", len(got), worst)
}

// TestSampler: upstream's whole generate_audio loop over the stub decoder
// v = 0.5·x + t·c, for each shift, with and without DCW, at an odd and an
// even length -- bit-exact.
func TestSampler(t *testing.T) {
	m := loadPlan(t)
	for label, s := range m.Sampler {
		sched := Schedule(s.Shift, nil)
		if len(sched) != len(s.Timesteps) {
			t.Fatalf("%s: %d steps, want %d", label, len(sched), len(s.Timesteps))
		}
		for i := range sched {
			if float64(sched[i]) != s.Timesteps[i] {
				t.Fatalf("%s: timestep %d = %v, want %v", label, i, sched[i], s.Timesteps[i])
			}
		}
		c := f32s(t, m, "sampler_c_T"+itoa(s.T))
		stub := func(x []float32, tt float32) ([]float32, error) {
			v := make([]float32, len(x))
			for i := range v {
				v[i] = 0.5*x[i] + float32(tt*c[i])
			}
			return v, nil
		}
		dcw := DCW{}
		if s.DCW {
			dcw = DefaultDCW
		}
		out, err := Sample(f32s(t, m, label+"_noise"), sched, dcw, stub, nil)
		if err != nil {
			t.Fatal(err)
		}
		exact32(t, label, out, f32s(t, m, label+"_out"))
	}
}

func itoa(n int) string { return string(rune('0'+n/10)) + string(rune('0'+n%10)) }

// TestCustomSchedule: the snapping rules, on cases worked by hand from
// generate_audio's code.
func TestCustomSchedule(t *testing.T) {
	for _, c := range []struct {
		shift  float64
		custom []float64
		want   []float64
	}{
		{2.4, nil, ShiftTimesteps[2]},
		{2.5, nil, ShiftTimesteps[2]}, // min() keeps the first of a tie
		{5, nil, ShiftTimesteps[3]},
		{3, []float64{0.97, 0.76, 0.615, 0.5, 0.395, 0.28, 0.18, 0.085, 0}, []float64{0.9545454545454546, 0.7692307692307693, 0.625, 0.5, 0.4, 0.3, 0.2222222222222222, 0.125}},
		{3, []float64{0, 0}, ShiftTimesteps[3]},
	} {
		got := Schedule(c.shift, c.custom)
		if len(got) != len(c.want) {
			t.Fatalf("shift %v custom %v: %v, want %v", c.shift, c.custom, got, c.want)
		}
		for i := range got {
			if got[i] != float32(c.want[i]) {
				t.Fatalf("shift %v custom %v: %v, want %v", c.shift, c.custom, got, c.want)
			}
		}
	}
}

// TestFSQ: the codebook rows exactly, and the projected outputs to within
// the order of a 6-term sum.
func TestFSQ(t *testing.T) {
	m := loadPlan(t)
	codes := f32s(t, m, "fsq_codes")
	outs := f32s(t, m, "fsq_out")
	set, err := safetensors.OpenSet(ditDir)
	if err != nil {
		t.Skipf("no DiT checkpoint: %v", err)
	}
	defer set.Close()
	get := func(name string) []float32 {
		tt, err := set.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		v, err := tt.F32(nil)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	w, b := get("tokenizer.quantizer.project_out.weight"), get("tokenizer.quantizer.project_out.bias")
	worst := 0.0
	for n, idx := range m.FSQ.Indices {
		code, err := FSQCode(idx)
		if err != nil {
			t.Fatal(err)
		}
		exact32(t, "fsq code", code[:], codes[n*6:(n+1)*6])
		out, err := FSQOutput(idx, w, b)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range out {
			want := outs[n*len(b)+i]
			worst = math.Max(worst, math.Abs(float64(v-want))/math.Max(1e-3, math.Abs(float64(want))))
		}
	}
	if worst > 1e-5 {
		t.Fatalf("FSQ output worst relative diff %.3g", worst)
	}
	t.Logf("FSQ: %d indices, codes exact, outputs within %.3g", len(m.FSQ.Indices), worst)
}
