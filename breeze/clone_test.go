package breeze

import (
	"path/filepath"
	"testing"
	"time"

	"strix-halo-vulkan/safetensors"
)

// TestCodecEncoder runs the two dumped clips through the host encoder and
// holds every stage to the oracle, and the codes exactly (B8).
func TestCodecEncoder(t *testing.T) {
	f, err := safetensors.Open(filepath.Join(refDir, "enc", "tensors.safetensors"))
	if err != nil {
		t.Skipf("no dump (reference/dump_breeze_enc.py): %v", err)
	}
	defer f.Close()
	get := func(n string) []float32 {
		ten, err := f.Get(n)
		if err != nil {
			t.Fatal(err)
		}
		return f32(t, ten)
	}
	t0 := time.Now()
	e, err := LoadCodecEncoder(modelDir())
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	t.Logf("loaded in %v", time.Since(t0))
	flat := func(x [][]float32) []float32 {
		var out []float32
		for _, r := range x {
			out = append(out, r...)
		}
		return out
	}
	for _, name := range []string{"jfk", "design"} {
		var st Stages
		t0 := time.Now()
		codes, err := e.Encode(get(name+"_wav"), &st)
		if err != nil {
			t.Fatal(err)
		}
		took := time.Since(t0)
		for _, s := range []struct {
			n string
			x [][]float32
		}{{"seanet", st.SEANet}, {"transformer", st.Transformer}, {"downsample", st.Downsample}} {
			c := compare(flat(s.x), get(name+"_"+s.n), 0)
			t.Logf("%s %s: rel %.2e max %.2e", name, s.n, c.rel, c.maxAbs)
		}
		want := get(name + "_codes")
		miss := 0
		for i := range codes {
			for k := 0; k < Codebooks; k++ {
				if float32(codes[i][k]) != want[i*Codebooks+k] {
					miss++
				}
			}
		}
		t.Logf("%s: %d frames in %v, %d of %d codes differ", name, len(codes), took.Round(time.Millisecond), miss, len(codes)*Codebooks)
		if len(codes)*Codebooks != len(want) || miss > 0 {
			t.Errorf("%s: codes differ from the oracle", name)
		}
	}
}

// TestCloneEmbeds builds the clone and direction prompts in Go -- the
// reference codes from the oracle's own encode -- and holds every row of
// each branch's backbone input to the dump: the encoder's text rows, each
// reference frame's summed code rows, and <|audio_eos|>'s all-zero frame.
func TestCloneEmbeds(t *testing.T) {
	e := loadEncoder(t)
	d := loadDecoder(t)
	tok := loadTokenizer(t)
	m := &Model{Tok: tok, Enc: e, Dec: d}
	for _, name := range []string{"clone", "direct"} {
		ref := refCase(t, name)
		rc := f32(t, ref["ref_codes"])
		frames := make([][16]int32, len(rc)/16)
		for i := range frames {
			for k := range frames[i] {
				frames[i][k] = int32(rc[i*16+k])
			}
		}
		req := &Request{Text: "We choose to build this machine not because it is easy, but because it is hard.",
			RefText:   "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
			RefFrames: frames}
		if name == "direct" {
			req.Instruction = "Speak slowly, in a quiet, intimate whisper."
		}
		p, err := BuildPrompt(tok, req)
		if err != nil {
			t.Fatal(err)
		}
		for _, br := range []struct {
			n string
			b *Branch
		}{{"cond", &p.Cond}, {"uncond", p.Uncond}} {
			if br.b == nil || ref[br.n+"_embeds"] == nil {
				continue
			}
			want := f32(t, ref[br.n+"_embeds"])
			got, _, err := m.embedPrompt(br.b, func(f func() error) error { return f() })
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("%s/%s: %d rows, want %d", name, br.n, len(got)/bbHidden, len(want)/bbHidden)
			}
			var text, audio []float32
			var wantText, wantAudio []float32
			for i, m := range br.b.TextMask {
				r := got[i*bbHidden : (i+1)*bbHidden]
				w := want[i*bbHidden : (i+1)*bbHidden]
				if m {
					text, wantText = append(text, r...), append(wantText, w...)
				} else {
					audio, wantAudio = append(audio, r...), append(wantAudio, w...)
				}
			}
			ct, ca := compare(text, wantText, 0), compare(audio, wantAudio, 0)
			t.Logf("%s/%s: %d rows (%d reference frames): text rel %.2e, audio rel %.2e (max %.2e)",
				name, br.n, len(br.b.IDs), len(br.b.Frames), ct.rel, ca.rel, ca.maxAbs)
			if ct.rel > encTol || ca.rel > 1e-6 {
				t.Errorf("%s/%s: past the bounds", name, br.n)
			}
		}
	}
}
