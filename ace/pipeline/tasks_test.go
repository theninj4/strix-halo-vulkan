package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/zimage/qwen"
)

// copy2 copies src into dst and returns what is left of dst.
func copy2(dst, src []float32) []float32 { return dst[copy(dst, src):] }

func readI64(t *testing.T, path string) []int32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no reference (%v)", err)
	}
	out := make([]int32, len(raw)/8)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return out
}

// tasksRef is reference/dump_ace_tasks.py: upstream's handler running a
// reference's timbre, cover (twice), cover-nofsq and repaint (twice), fp32
// on the CPU, seed 42 (MUSIC.md A11).
//
//	HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv-acestep/bin/python reference/dump_ace_tasks.py   (~4 min)
const tasksRef = "../../reference/out/acetasks"

// tasksBF16 is the same cases the way upstream serves them on CUDA:
// dump_ace_tasks.py --bf16.
const tasksBF16 = "../../reference/out/acetasks_bf16"

type taskCase struct {
	Params struct {
		Caption       string   `json:"caption"`
		BPM           int      `json:"bpm"`
		KeyScale      string   `json:"keyscale"`
		TimeSignature string   `json:"timesignature"`
		Duration      float64  `json:"duration"`
		Language      string   `json:"vocal_language"`
		Task          string   `json:"task_type"`
		Reference     string   `json:"reference_audio"`
		CoverStrength *float64 `json:"audio_cover_strength"`
		CoverNoise    float64  `json:"cover_noise_strength"`
		RepaintStart  float64  `json:"repainting_start"`
		RepaintEnd    float64  `json:"repainting_end"`
		RepaintMode   string   `json:"repaint_mode"`
		ChunkMaskMode string   `json:"chunk_mask_mode"`
	} `json:"params"`
	RefOffsets   []int `json:"ref_offsets"`
	Conditions   int   `json:"conditions"`
	LatentLength int   `json:"latent_length"`
}

var taskLabels = []string{"ref_t2m", "cover", "cover_mix", "cover_nofsq", "repaint", "repaint_out"}

// taskRequest is a case's request and audio-in options, fed the oracle's
// sampled latents (the VAE's posterior noise is unseeded upstream) and its
// segment draws, and its noise.
func taskRequest(t *testing.T, label string) (*plan.Request, Options, taskCase) {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(tasksRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_ace_tasks.py", tasksRef, err)
	}
	var m struct {
		Cases map[string]taskCase `json:"cases"`
	}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	c, ok := m.Cases[label]
	if !ok {
		t.Skipf("no case %q; run reference/dump_ace_tasks.py %s", label, label)
	}
	p := c.Params
	lyrics := request(t, "full_metas").Lyrics // the oracle's cases all sing A1's LYRICS
	r := &plan.Request{Caption: p.Caption, Lyrics: lyrics, BPM: p.BPM, KeyScale: p.KeyScale,
		TimeSignature: p.TimeSignature, Duration: p.Duration, Language: p.Language}
	task := p.Task
	if task == "" {
		task = Text2Music
	}
	a := NewAudio(task)
	if p.CoverStrength != nil {
		a.CoverStrength = *p.CoverStrength
	}
	a.CoverNoise = p.CoverNoise
	a.RepaintStart, a.RepaintEnd, a.RepaintMode = p.RepaintStart, p.RepaintEnd, p.RepaintMode
	a.ExplicitMask = p.ChunkMaskMode == "explicit"
	if p.Reference != "" {
		a.ReferenceLatents = readF32(t, filepath.Join(tasksRef, label+"_refer.bin"))
	}
	if a.needsSource() {
		a.Source = interleaved(t, filepath.Join(tasksRef, label+"_src_wav.bin"))
		a.SourceLatents = sourceLatents(t, label, task)
	}
	o := Options{Seed: 42, Audio: a, Noise: readF32(t, filepath.Join(tasksRef, label+"_noise.bin"))}
	return r, o, c
}

// sourceLatents are the oracle's sampled latents of the (padded) source,
// before the pad to 128 latents and before any span is silenced: the
// repaint's clean_src, a cover's src_latents.
func sourceLatents(t *testing.T, label, task string) []float32 {
	t.Helper()
	if task == Repaint {
		return readF32(t, filepath.Join(tasksRef, label+"_clean_src.bin"))
	}
	return readF32(t, filepath.Join(tasksRef, label+"_src_latents.bin"))
}

// TestReferenceClip: process_reference_audio's segments from the oracle's
// draws, sample for sample.
func TestReferenceClip(t *testing.T) {
	_, _, c := taskRequest(t, "ref_t2m")
	src := interleaved(t, filepath.Join(vaeRef, "defaults_final.bin"))
	for i := range src {
		src[i] = max(-1, min(1, src[i]))
	}
	got := ReferenceClip(src, c.RefOffsets, nil)
	want := interleaved(t, filepath.Join(tasksRef, "ref_t2m_ref_wav.bin"))
	if len(got) != len(want) {
		t.Fatalf("clip is %d samples, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sample %d: %v, want %v", i, got[i], want[i])
		}
	}
	t.Logf("30 s clip from offsets %v: exact", c.RefOffsets)
}

// TestTaskInputs is A11b-d's input gate: for every case, the caption
// prompt (through the text encoder), the context the DiT reads (the
// source rows -- a cover's through our tokenizer and detokenizer -- and the
// chunk mask), the repaint span, and the cover-strength switch's condition,
// against what upstream's handler handed generate_audio.
func TestTaskInputs(t *testing.T) {
	p := pipeline(t)
	for _, label := range taskLabels {
		r, o, c := taskRequest(t, label)
		var d *plan.Request
		var dp *ditPlan
		var err error
		if o.Audio.needsSource() {
			d, dp, err = p.taskPlan(r, o.Audio, o.Seed)
		} else {
			d = r
			dp, err = p.textPlan(o, r.LatentLength(), nil)
		}
		if err != nil {
			t.Fatal(err)
		}
		if dp.T != c.LatentLength {
			t.Fatalf("%s: %d latents, want %d", label, dp.T, c.LatentLength)
		}
		text, _, err := p.Condition(d)
		if err != nil {
			t.Fatal(err)
		}
		want := readF32(t, filepath.Join(tasksRef, label+"_text_hidden.bin"))
		if len(want) != len(text.Data) {
			t.Fatalf("%s: caption is %d tokens, want %d", label, text.Rows, len(want)/text.Cols)
		}
		_, rms, _ := gap(text.Data, want)
		t.Logf("%-12s caption hidden rms %.2e", label, rms)
		if rms > 5e-3 {
			t.Errorf("%s: caption hidden rms %.2e", label, rms)
		}
		ctx := readF32(t, filepath.Join(tasksRef, label+"_cond0_ctx.bin"))
		got := dp.context(dp.src).Data
		srcGot, srcWant := make([]float32, 0, dp.T*64), make([]float32, 0, dp.T*64)
		for i := 0; i < dp.T; i++ {
			srcGot = append(srcGot, got[i*128:i*128+64]...)
			srcWant = append(srcWant, ctx[i*128:i*128+64]...)
			for j := 64; j < 128; j++ {
				if got[i*128+j] != ctx[i*128+j] {
					t.Fatalf("%s: chunk mask row %d is %v, want %v", label, i, got[i*128+j], ctx[i*128+j])
				}
			}
		}
		if o.Audio.Task == Cover {
			// The hints are our tokenizer's codes through the detokenizer.
			// A code at a quantisation boundary can flip (TestGPUTokenize
			// in ace/dit) and replace its 5 rows; the rest is fp16's.
			codes := readI64(t, filepath.Join(tasksRef, label+"_codes.bin"))
			n := (dp.T + 4) / 5 * 5
			x := make([]float32, n*64)
			copy(copy2(x, sourceLatents(t, label, Cover)), p.silence)
			ours, _, _, err := p.DiT.Tokenize(&qwen.Mat{Rows: n, Cols: 64, Data: x})
			if err != nil {
				t.Fatal(err)
			}
			flips := 0
			var g, w []float32
			for i := 0; i < dp.T; i++ {
				if ours[i/5] != codes[i/5] {
					if i%5 == 0 {
						flips++
					}
					continue
				}
				g = append(g, srcGot[i*64:(i+1)*64]...)
				w = append(w, srcWant[i*64:(i+1)*64]...)
			}
			_, rms, _ := gap(g, w)
			t.Logf("%-12s hints rms %.2e outside %d flipped codes of %d", label, rms, flips, len(codes))
			if rms > 3e-3 || flips*50 > len(codes) {
				t.Errorf("%s: hints rms %.2e, %d flipped codes", label, rms, flips)
			}
		} else {
			_, rms, _ := gap(srcGot, srcWant)
			t.Logf("%-12s context source rms %.2e", label, rms)
			if rms != 0 {
				t.Errorf("%s: context source rms %.2e, want exact", label, rms)
			}
		}
		if dp.rp != nil {
			span := readF32(t, filepath.Join(tasksRef, label+"_repaint_mask.bin"))
			for i, v := range span {
				if (v != 0) != dp.rp.span[i] {
					t.Fatalf("%s: repaint span row %d", label, i)
				}
			}
		}
		if c.Conditions == 3 {
			if dp.alt == nil {
				t.Fatalf("%s: no cover-strength switch", label)
			}
			text, _, err := p.Condition(dp.alt)
			if err != nil {
				t.Fatal(err)
			}
			_, rms, _ := gap(text.Data, readF32(t, filepath.Join(tasksRef, label+"_non_cover_text_hidden.bin")))
			t.Logf("%-12s non-cover caption hidden rms %.2e", label, rms)
			if rms > 5e-3 {
				t.Errorf("%s: non-cover caption hidden rms %.2e", label, rms)
			}
		}
	}
}

// TestTasks is the end-to-end gate: every case from the oracle's sampled
// latents and noise to the latents and the -1 dBFS audio, against the fp32
// oracle's, held under half of upstream bf16's own drift from them (its
// CUDA dtype, dump_ace_tasks.py --bf16, from the same DiT noise), as A6's
// gate is. A cover drifts further than text2music (0.13 rms even from the
// oracle's own hints), and bf16 further still: 0.66, with 73 of 150 codes
// flipped where ours flips 1.
func TestTasks(t *testing.T) {
	p := pipeline(t)
	for _, label := range taskLabels {
		r, o, _ := taskRequest(t, label)
		res, err := p.Generate(r, o)
		if err != nil {
			t.Fatal(err)
		}
		want := readF32(t, filepath.Join(tasksRef, label+"_latents.bin"))
		if len(want) != len(res.Latents) {
			t.Fatalf("%s: %d latent values, want %d", label, len(res.Latents), len(want))
		}
		_, rms, _ := gap(res.Latents, want)
		audio := interleaved(t, filepath.Join(tasksRef, label+"_audio.bin"))
		if len(audio) != len(res.Audio) {
			t.Fatalf("%s: %d samples, want %d", label, len(res.Audio), len(audio))
		}
		_, _, snr := gap(res.Audio, audio)
		tm := res.Timings
		t.Logf("%-12s latents rms %.2e, audio SNR %5.1f dB; %.1f s of audio in %.2f s (DiT %.2f, VAE %.2f)",
			label, rms, snr, res.Seconds, tm.Total.Seconds(), tm.DiT.Seconds(), tm.VAE.Seconds())
		bf := readF32(t, filepath.Join(tasksBF16, label+"_latents.bin"))
		_, bfRMS, _ := gap(bf, want)
		t.Logf("%-12s upstream bf16: latents rms %.2e (%.1fx ours)", label, bfRMS, bfRMS/rms)
		if rms > bfRMS/2 {
			t.Errorf("%s: latents rms %.2e, more than half of bf16's %.2e", label, rms, bfRMS)
		}
	}
}

// TestTasksOwnEncode is the whole audio-in path: the oracle's 48 kHz audio
// through our clip (from its draws), encoder and posterior sampling rather
// than its sampled latents. Upstream bf16's run encodes for itself too, so
// the bar is the same.
func TestTasksOwnEncode(t *testing.T) {
	p := pipeline(t)
	src := interleaved(t, filepath.Join(vaeRef, "full_metas_final.bin"))
	ref := interleaved(t, filepath.Join(vaeRef, "defaults_final.bin"))
	for _, w := range [][]float32{src, ref} {
		for i := range w {
			w[i] = max(-1, min(1, w[i]))
		}
	}
	for _, label := range []string{"ref_t2m", "cover", "repaint"} {
		r, o, c := taskRequest(t, label)
		a := o.Audio
		a.SourceLatents, a.ReferenceLatents = nil, nil
		if a.needsSource() {
			a.Source = src
		}
		if label == "ref_t2m" {
			a.Reference, a.RefOffsets = ref, c.RefOffsets
		}
		res, err := p.Generate(r, o)
		if err != nil {
			t.Fatal(err)
		}
		want := readF32(t, filepath.Join(tasksRef, label+"_latents.bin"))
		_, rms, _ := gap(res.Latents, want)
		_, bfRMS, _ := gap(readF32(t, filepath.Join(tasksBF16, label+"_latents.bin")), want)
		t.Logf("%-12s own encode: latents rms %.2e; upstream bf16 %.2e (%.1fx)", label, rms, bfRMS, bfRMS/rms)
		if rms > bfRMS/2 {
			t.Errorf("%s: latents rms %.2e, more than half of bf16's %.2e", label, rms, bfRMS)
		}
	}
}
