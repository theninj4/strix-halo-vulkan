package dit

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// tasksRef is reference/dump_ace_tasks.py: upstream's audio-in tasks run
// through its own handler (MUSIC.md A11).
const tasksRef = "../../reference/out/acetasks"

func readI64(t *testing.T, path string) []int32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no reference (%v); run reference/dump_ace_tasks.py", err)
	}
	out := make([]int32, len(raw)/8)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return out
}

// flipMargin is how close to a quantisation boundary a code may be and
// still come out different: the pooled rows are fp16 GEMMs and attention
// over fp32 ones, 1e-3 of their scale apart, and a digit's bracket argument
// moves by about (L−1)/2 of that.
const flipMargin = 0.02

// TestGPUTokenize is A11c's tokenizer gate: 25 Hz latents to 5 Hz codes
// against the model's own tokenizer. The inputs are A4's (the fp32 DiT's
// latents for two songs; its codes are what A4's detokenizer gate reads)
// and the cover oracle's source latents (a VAE-encoded song), whose pooled
// rows are dumped too. Every code must match, or sit within flipMargin of
// the floor that separates it from the oracle's.
func TestGPUTokenize(t *testing.T) {
	g := sharedGPU(t)
	check := func(name string, got, want []int32, margins []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d codes, want %d", name, len(got), len(want))
		}
		flips, worst := 0, float32(0)
		for i := range want {
			if got[i] != want[i] {
				flips++
				worst = max(worst, margins[i])
			}
		}
		t.Logf("%-28s %d codes, %d differ (their widest margin %.4f)", name, len(want), flips, worst)
		if worst > flipMargin {
			t.Errorf("%s: a code %.4f from its boundary differs", name, worst)
		}
		if flips*50 > len(want) {
			t.Errorf("%s: %d of %d codes differ", name, flips, len(want))
		}
	}
	for _, label := range cases {
		p := label + "_"
		lat := readMat(t, ditRef, p+"latents")
		codes, _, margins, err := g.Tokenize(lat)
		if err != nil {
			t.Fatal(err)
		}
		check(label, codes, readI32(t, filepath.Join(detokRef, p+"codes.bin")), margins)
	}
	src := readMat(t, tasksRef, "cover_src_latents")
	codes, pooled, margins, err := g.Tokenize(src)
	if err != nil {
		t.Fatal(err)
	}
	gate(t, "cover pooled", pooled, readMat(t, tasksRef, "cover_pool"), 2e-2, 5e-3)
	check("cover", codes, readI64(t, filepath.Join(tasksRef, "cover_codes.bin")), margins)
}
