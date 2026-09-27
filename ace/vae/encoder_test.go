package vae

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"strix-halo-vulkan/vk"
)

// The reference is reference/dump_ace_vae_enc.py: diffusers' OobleckEncoder
// in fp32 on the audio A5's oracle wrote.
//
//	HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_vae_enc.py   (~1 min)
const encRef = "../../reference/out/acevaeenc"

func loadEncManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(encRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_ace_vae_enc.py", encRef, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// encChannelLast reads a dumped [C, T] tensor of the encoder's oracle as
// [T, C].
func encChannelLast(t *testing.T, m *manifest, name string) ([]float32, int, int) {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok || len(meta.Shape) != 2 {
		t.Fatalf("no [C, T] tensor %q", name)
	}
	C, T := meta.Shape[0], meta.Shape[1]
	raw := readF32(t, filepath.Join(encRef, name+".bin"))
	out := make([]float32, len(raw))
	for c := 0; c < C; c++ {
		for i := 0; i < T; i++ {
			out[i*C+c] = raw[c*T+i]
		}
	}
	return out, T, C
}

func newEncoder(t *testing.T) (*Encoder, func()) {
	t.Helper()
	g, done := newGPU(t, 512)
	e, err := NewEncoder(g, vaeDir)
	if err != nil {
		done()
		t.Fatal(err)
	}
	return e, func() { e.Destroy(); done() }
}

// stageSNR is the stagewise floor. With the audio and block 0 carried as
// fp16 pairs, every stage lands within 0.3 dB of torch's fp32 encoder with
// fp16 weights and every later convolution's input rounded to fp16
// (MUSIC.md A11a); the lowest, block 2, is 54 dB.
const stageSNR = 50

// TestEncoderStages: the encode of the 4 s excerpt, run up to the end of
// conv1, of each block's residual units and of each block, read back there,
// against the oracle's hooks; and the whole excerpt's posterior.
func TestEncoderStages(t *testing.T) {
	m := loadEncManifest(t)
	e, done := newEncoder(t)
	defer done()
	wav, n, _ := encChannelLast(t, m, "stage_in")
	gr := e.tileGraph(n)
	upTo := func(prefix string) int {
		last := -1
		for i, k := range gr.kinds {
			if strings.HasPrefix(k, prefix) {
				last = i
			}
		}
		return last
	}
	run := func(name, prefix string, rows, ch int) {
		end := upTo(prefix)
		e.upload(wav, 0, n) // the last run's Snake passes wrote over it
		if _, err := vk.DispatchMultiTimed(gr.d[:end+1], 1, 1, true); err != nil {
			t.Fatal(err)
		}
		want, T, C := encChannelLast(t, m, name)
		if T != rows || C != ch {
			t.Fatalf("%s is [%d %d], the graph has [%d %d]", name, C, T, ch, rows)
		}
		check(t, name, e.k.abuf.ReadFloat32At(int(e.k.aX), rows*ch), want, stageSNR)
	}
	rows := n
	run("stage_conv1", "conv1 bias", rows, encWidths[0])
	for i, s := range encStrides {
		run(fmt.Sprintf("stage_block%d_res", i), fmt.Sprintf("b%d r2 add", i), rows, encWidths[i])
		rows /= s
		run(fmt.Sprintf("stage_block%d", i), fmt.Sprintf("b%d down bias", i), rows, encWidths[i+1])
	}
	run("stage_out", "conv2 bias", rows, Posterior)
}

// TestEncode: three songs' whole posterior against the untiled oracle: 30 s
// in one window, 120 s tiled at 512 latents, and 100 s plus 1,234 frames
// (the frames past the last latent read, in a tiled tail); and
// how far upstream's own tiling moves the mean, for scale.
func TestEncode(t *testing.T) {
	m := loadEncManifest(t)
	e, done := newEncoder(t)
	defer done()
	for _, label := range []string{"full_metas", "defaults", "odd"} {
		src := label
		if label == "odd" {
			src = "defaults"
		}
		wav, _, _ := channelLast(t, loadManifest(t), src+"_final")
		if label == "odd" {
			wav = wav[:2*(2500*Hop+1234)] // not whole latents, and a tiled tail
		}
		for i := range wav {
			wav[i] = max(-1, min(1, wav[i]))
		}
		mean, scale, took, err := e.Encode(wav)
		if err != nil {
			t.Fatal(err)
		}
		T := len(mean) / Latent
		t.Logf("%s: %.0f s of audio to %d latents in %.0f ms on the device", label, float64(len(wav)/2)/Rate, T, took.Seconds()*1e3)
		// Priced against upstream's own bf16 encode, and at least 50 dB.
		for _, part := range []struct {
			name string
			got  []float32
		}{{"mean", mean}, {"scale", scale}} {
			want, _, _ := encChannelLast(t, m, label+"_"+part.name)
			bf, _, _ := encChannelLast(t, m, label+"_bf16_"+part.name)
			_, _, bfSNR := gap(bf, want)
			check(t, label+" "+part.name, part.got, want, max(50, bfSNR))
			t.Logf("%-26s SNR %5.1f dB  (upstream bf16)", label+" "+part.name, bfSNR)
		}
		tiled, _, _ := encChannelLast(t, m, label+"_tiled_mean")
		mw, _, _ := encChannelLast(t, m, label+"_mean")
		rel, rms, snr := gap(tiled, mw)
		t.Logf("%-26s max %.2e  rms %.2e  SNR %5.1f dB  (upstream's tiling vs untiled)", label+" upstream tiled", rel, rms, snr)
	}
}
