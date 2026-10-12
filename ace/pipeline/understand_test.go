package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/zimage/qwen"
)

// understandRef reads an understand case of reference/dump_ace_lm.py
// (MUSIC.md A13): upstream's posterior sample of the source ([T, 64]) and
// the codes its tokenizer made of it.
func understandRef(t *testing.T, label string) (lat []float32, codes []int32) {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(lmRef, "manifest.json"))
	if err != nil {
		t.Skipf("no LM reference (%v)", err)
	}
	var m struct {
		Tensors map[string]json.RawMessage `json:"tensors"`
	}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m.Tensors[label+"_codes"]; !ok {
		t.Skipf("no %s case; run reference/dump_ace_lm.py %s", label, label)
	}
	raw, err := os.ReadFile(filepath.Join(lmRef, label+"_codes.bin"))
	if err != nil {
		t.Fatal(err)
	}
	codes = make([]int32, len(raw)/4)
	for i := range codes {
		codes[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return readF32(t, filepath.Join(lmRef, label+"_latents.bin")), codes
}

// TestUnderstandCodes is convert_src_audio_to_codes after the encode: the
// oracle's latents through our audio tokenizer, against its codes. Every
// code must match or sit within 0.02 of the floor that separates it from
// the oracle's (ace/dit's TestGPUTokenize, flipMargin), and at most one in
// fifty may flip (TestTaskInputs' cover gate).
func TestUnderstandCodes(t *testing.T) {
	p := pipeline(t)
	for _, label := range []string{"understand", "understand_lofi"} {
		lat, want := understandRef(t, label)
		got, err := p.AudioCodes(lat)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(want) {
			t.Fatalf("%s: %d codes, want %d", label, len(got), len(want))
		}
		T := len(lat) / plan.LatentChannels
		n := (T + 4) / 5 * 5
		x := make([]float32, n*plan.LatentChannels)
		copy(copy2(x, lat), p.silence)
		_, _, margins, err := p.DiT.Tokenize(&qwen.Mat{Rows: n, Cols: plan.LatentChannels, Data: x})
		if err != nil {
			t.Fatal(err)
		}
		flips, worst := 0, float32(0)
		for i := range got {
			if int32(got[i]) != want[i] {
				flips++
				worst = max(worst, margins[i])
			}
		}
		t.Logf("%-15s %d of %d codes flipped, their widest margin %.4f", label, flips, len(want), worst)
		if flips*50 > len(want) || worst > 0.02 {
			t.Errorf("%s: %d of %d codes flipped, one %.4f from its boundary", label, flips, len(want), worst)
		}
	}
}

// TestUnderstand captions A11's source end to end -- ffmpeg, the encoder,
// the tokenizer and the LM -- and wants every field the FSM holds. The
// song is full_metas' (synthwave, female vocals, 120 bpm, C major, 4/4,
// 30 s, English); what the LM hears is logged, not gated.
func TestUnderstand(t *testing.T) {
	p := pipeline(t)
	if p.LM == nil {
		if err := p.LoadLM(shared.dev, LMDir(models), lm.DefaultOptions()); err != nil {
			t.Skip(err)
		}
	}
	wav, err := ReadAudio(filepath.Join(tasksRef, "src.wav"))
	if err != nil {
		t.Skip(err)
	}
	u, err := p.Understand(wav, Options{Seed: 42, LMSeed: 42})
	if err != nil {
		t.Fatal(err)
	}
	s := u.Song
	t.Logf("%d codes; encode %v, tokenize %v, LM %v, %v in all\n%s", len(u.Codes), u.Timings.Encode,
		u.Timings.Tokenize, u.Timings.LM, u.Timings.Total, u.Text)
	if len(u.Codes) != 150 || s.Caption == "" || s.BPM == 0 || s.KeyScale == "" || s.TimeSignature == "" ||
		s.Duration == 0 || s.Genres == "" || s.Language == "" {
		t.Errorf("%d codes, song %+v", len(u.Codes), s)
	}
}
