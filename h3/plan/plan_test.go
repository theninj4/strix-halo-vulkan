package plan

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The reference comes from reference/dump_h3_plan.py: diffusers' own canvas,
// frame, layout, rotary and scheduler code for MiniMax-H3, no weights.
// Regenerate with:
//
//	.venv/bin/python reference/dump_h3_plan.py
const planRef = "../../reference/out/h3plan"

type planManifest struct {
	Canvases []struct {
		Aspect []float64 `json:"aspect"`
		Height int       `json:"height"`
		Width  int       `json:"width"`
	} `json:"canvases"`
	Frames []struct {
		Requested    int `json:"requested"`
		Aligned      int `json:"aligned"`
		LatentFrames int `json:"latent_frames"`
		AudioLatents int `json:"audio_latents"`
	} `json:"frames"`
	Layouts map[string]struct {
		Height, Width, Frames int
		TextTokens            int      `json:"text_tokens"`
		Anchors               []string `json:"anchors"`
		LatentFrames          int      `json:"latent_frames"`
		LatentHeight          int      `json:"latent_height"`
		LatentWidth           int      `json:"latent_width"`
		AudioLatents          int      `json:"audio_latents"`
		SequenceLength        int      `json:"sequence_length"`
		CondVideoRows         int      `json:"condition_video_rows"`
	} `json:"layouts"`
	Schedules map[string]struct {
		Steps    int `json:"steps"`
		Forwards int `json:"forwards"`
	} `json:"schedules"`
	Shift      float32 `json:"shift"`
	AudioShift float32 `json:"audio_shift"`
	Tensors    map[string]struct {
		Shape []int  `json:"shape"`
		Dtype string `json:"dtype"`
	} `json:"tensors"`
}

func loadPlan(t *testing.T) *planManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(planRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_h3_plan.py", planRef, err)
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

func f64s(t *testing.T, m *planManifest, name string) []float64 {
	raw := readRaw(t, m, name, "float64")
	out := make([]float64, len(raw)/8)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return out
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

func exactI32(t *testing.T, name string, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %d, want %d", name, i, got[i], want[i])
		}
	}
}

func TestCanvas(t *testing.T) {
	m := loadPlan(t)
	for _, c := range m.Canvases {
		h, w, err := Canvas(c.Aspect[0], c.Aspect[1], ShortEdge, MaxPixels)
		if err != nil {
			t.Fatal(err)
		}
		if h != c.Height || w != c.Width {
			t.Errorf("%v: %dx%d, want %dx%d", c.Aspect, h, w, c.Height, c.Width)
		}
	}
}

func TestFrames(t *testing.T) {
	m := loadPlan(t)
	for _, f := range m.Frames {
		a := AlignFrames(f.Requested)
		if a != f.Aligned || LatentFrames(a) != f.LatentFrames || AudioLatents(a) != f.AudioLatents {
			t.Errorf("%d frames: aligned %d, %d latent, %d audio; want %d, %d, %d", f.Requested,
				a, LatentFrames(a), AudioLatents(a), f.Aligned, f.LatentFrames, f.AudioLatents)
		}
	}
	// The duration check is on the aligned count: 346 would align to 362,
	// 15.08 s, and is refused; 345 is the longest request that passes.
	if _, err := Frames(345); err != nil {
		t.Errorf("345 frames: %v", err)
	}
	for _, n := range []int{346, 100} {
		if _, err := Frames(n); err == nil {
			t.Errorf("%d frames: accepted, want refused", n)
		}
	}
}

func buildLayout(t *testing.T, m *planManifest, label string) *Layout {
	t.Helper()
	c := m.Layouts[label]
	var anchors []Anchor
	for _, a := range c.Anchors {
		switch a {
		case "first":
			anchors = append(anchors, First)
		case "last":
			anchors = append(anchors, Last)
		}
	}
	l, err := NewLayout(i32s(t, m, label+"_text_tags"), c.LatentFrames, c.LatentHeight, c.LatentWidth, c.AudioLatents, anchors)
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Pos) != c.SequenceLength || l.CondVideoRows != c.CondVideoRows {
		t.Fatalf("%s: %d rows, %d conditioning; want %d, %d", label, len(l.Pos), l.CondVideoRows,
			c.SequenceLength, c.CondVideoRows)
	}
	return l
}

// TestLayout is the packed sequence bit for bit — every float64 rotary
// coordinate, tag and index — on every case the dump holds.
func TestLayout(t *testing.T) {
	m := loadPlan(t)
	for label := range m.Layouts {
		t.Run(label, func(t *testing.T) {
			l := buildLayout(t, m, label)
			want := f64s(t, m, label+"_pos")
			for r, p := range l.Pos {
				for a := 0; a < 3; a++ {
					if math.Float64bits(p[a]) != math.Float64bits(want[r*3+a]) {
						t.Fatalf("row %d axis %d: %.17g, want %.17g", r, a, p[a], want[r*3+a])
					}
				}
			}
			exactI32(t, "tags", l.Tags, i32s(t, m, label+"_tags"))
			exactI32(t, "video", l.Video, i32s(t, m, label+"_video_idx"))
			exactI32(t, "audio", l.Audio, i32s(t, m, label+"_audio_idx"))
			exactI32(t, "text", l.Text, i32s(t, m, label+"_text_idx"))
		})
	}
}

// TestRope checks the rotary tables. inv_freq and the float32 angles are
// exact; cos and sin are torch's vectorised Sleef (1-ulp) against Go's
// correctly rounded math, so a value may differ by one ulp and no more.
func TestRope(t *testing.T) {
	m := loadPlan(t)
	inv := InvFreq()
	exact32(t, "inv_freq", inv, f32s(t, m, "inv_freq"))
	for label := range m.Layouts {
		l := buildLayout(t, m, label)
		cos, sin := l.Rope(inv)
		for _, tab := range []struct {
			name      string
			got, want []float32
		}{{"cos", cos, f32s(t, m, label+"_cos")}, {"sin", sin, f32s(t, m, label+"_sin")}} {
			if len(tab.got) != len(tab.want) {
				t.Fatalf("%s %s: %d values, want %d", label, tab.name, len(tab.got), len(tab.want))
			}
			var off int
			for i := range tab.got {
				d := int64(math.Float32bits(tab.got[i])) - int64(math.Float32bits(tab.want[i]))
				if d > 1 || d < -1 {
					t.Fatalf("%s %s[%d] = %.9g, want %.9g", label, tab.name, i, tab.got[i], tab.want[i])
				}
				if d != 0 {
					off++
				}
			}
			t.Logf("%s %s: %d of %d values one ulp off", label, tab.name, off, len(tab.got))
		}
	}
}

func TestSchedule(t *testing.T) {
	m := loadPlan(t)
	if m.Shift != 12 || m.AudioShift != 3 {
		t.Fatalf("shifts %g/%g, want 12/3", m.Shift, m.AudioShift)
	}
	lay := buildLayout(t, m, "first_256x448")
	for label, c := range m.Schedules {
		t.Run(label, func(t *testing.T) {
			v, err := NewSchedule(c.Steps, m.Shift)
			if err != nil {
				t.Fatal(err)
			}
			a, err := NewSchedule(c.Steps, m.AudioShift)
			if err != nil {
				t.Fatal(err)
			}
			exact32(t, "sigmas", v.Sigmas, f32s(t, m, label+"_sigmas"))
			exact32(t, "timesteps", v.Timesteps, f32s(t, m, label+"_timesteps"))
			exact32(t, "audio sigmas", a.Sigmas, f32s(t, m, label+"_audio_sigmas"))
			exact32(t, "audio timesteps", a.Timesteps, f32s(t, m, label+"_audio_timesteps"))

			uniq := f32s(t, m, label+"_plan_unique")
			idx := i32s(t, m, label+"_plan_index")
			rows := len(lay.Pos)
			for i := range v.Timesteps {
				u, x := lay.RowTimesteps(v.Timesteps[i], a.Timesteps[i])
				want := uniq[i*4 : i*4+4]
				n := 0
				for n < 4 && want[n] != -1 {
					n++
				}
				exact32(t, "unique", u, want[:n])
				exactI32(t, "index", x, idx[i*rows:(i+1)*rows])
			}
		})
	}
}

func TestStep(t *testing.T) {
	m := loadPlan(t)
	sample := f32s(t, m, "step_sample")
	v := f32s(t, m, "step_model_out")
	for _, c := range []struct {
		name  string
		shift float32
	}{{"video", m.Shift}, {"audio", m.AudioShift}} {
		s, err := NewSchedule(8, c.shift)
		if err != nil {
			t.Fatal(err)
		}
		chain := f32s(t, m, "step_"+c.name+"_chain")
		x := append([]float32(nil), sample...)
		for i := range s.Timesteps {
			s.Step(i, v, x)
			exact32(t, c.name+" step", x, chain[i*len(x):(i+1)*len(x)])
		}
	}
}

// TestLastAnchorOrder is the negative control for the "last" keyframe case:
// at 102 latent frames numpy's pairwise sum and a sequential sum of the same
// spans differ, so TestLayout's last_256x448 case would catch a port that
// summed them the other way.
func TestLastAnchorOrder(t *testing.T) {
	spans := make([]float64, 102)
	var seq float64
	for i := range spans {
		spans[i] = ropeFrameRescale * ropeFramesPerLatent[i%len(ropeFramesPerLatent)]
		seq += spans[i]
	}
	if pairwiseSum(spans) == seq {
		t.Fatalf("pairwise and sequential sums agree at 102 frames (%.17g); the layout case no longer tells them apart", seq)
	}
}
