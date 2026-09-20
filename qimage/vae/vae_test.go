package vae

import (
	"encoding/binary"
	"encoding/json"

	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	zvae "strix-halo-vulkan/zimage/vae"
)

// The reference comes from reference/dump_qi21_vae.py: diffusers' VAE in
// fp32 on CPU, decode from seeded noise and encode of a deterministic RGBA
// test card, with every stage hooked. Regenerate with:
//
//	.venv/bin/python reference/dump_qi21_vae.py
const (
	refDir = "../../reference/out/qi21vae"
	vaeDir = "../../models/Qwen-Image-2.1/vae"
)

type manifest struct {
	Cases map[string]struct {
		H      int   `json:"h"`
		W      int   `json:"w"`
		Latent []int `json:"latent"`
	} `json:"cases"`
	Tensors map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_qi21_vae.py", refDir, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// loadRef reads a dumped [1, C, 1, H, W] tensor with the frame axis folded
// away.
func loadRef(t *testing.T, m *manifest, name string) *zvae.Tensor {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(refDir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != meta.Count*4 {
		t.Fatalf("%s: %d bytes for %d float32", name, len(raw), meta.Count)
	}
	sh := meta.Shape
	if len(sh) != 5 || sh[0] != 1 || sh[2] != 1 {
		t.Fatalf("%s: shape %v, want [1 C 1 H W]", name, sh)
	}
	out := zvae.NewTensor(1, sh[1], sh[3], sh[4])
	for i := range out.Data {
		out.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// relTol is the fp32 bound, the same instrument as the DiT tests': both
// sides fp32, summation order the only drift, the rms floor keeping the
// decoder's enormous intermediates (absmax ~3e5) from hiding real errors on
// small elements.
const relTol = 2e-4

// normTol is the bound for the stages where this model amplifies fp32
// noise, and it is measured, not chosen (all numbers 2026-09-20):
//
//   - the decoder past its tail norm: the up blocks run at magnitudes ~1e4,
//     where a summation-order difference is a real absolute quantity; the
//     per-pixel L2 norm rescales to O(1) and the inherited drift surfaces
//     elementwise. Against a float64 decode the dumped fp32 run is itself
//     rel 1.3e-3 at norm_out (3.4e-4 at the image), and changing only its
//     thread count moves it 8.6e-4. This port's decoded image sits max abs
//     8.7e-4 from the dump in [-1, 1] — under a quarter of one 8-bit
//     quantization step;
//   - the encoder's deep middle: the reference's own thread-order noise
//     jumps 47x at down_blocks.3 (4.6e-6 → 2.2e-4, still 2.1e-4 at the mid
//     block) and re-contracts to 8e-6 at the posterior mode. This port
//     tracks at a consistent ~5x that self-noise through the same stages
//     and lands at 4.7e-5 on the mode, which keeps the tight bound.
//
// A structural mistake shows at rel ≥ 1e-1 on every one of these stages, so
// the loosened bound still catches every modeled error.
const normTol = 8e-3

// tolFor picks the bound by stage, per the measurements above.
func tolFor(name string) float64 {
	loose := []string{
		"dec_norm_out", "dec_nonlinearity", "dec_conv_out", "decoded", "roundtrip",
		"enc_down_blocks.3", "enc_down_blocks.4", "enc_mid_block", "enc_norm_out", "enc_nonlinearity",
	}
	for _, s := range loose {
		if strings.HasSuffix(name, s) {
			return normTol
		}
	}
	return relTol
}

func compare(t *testing.T, name string, got, want *zvae.Tensor) {
	t.Helper()
	if got.C != want.C || got.H != want.H || got.W != want.W {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	var sumSq float64
	for i := range want.Data {
		w := float64(want.Data[i])
		sumSq += w * w
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var maxAbs, rel float64
	var worst int
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		if d > maxAbs {
			maxAbs = d
		}
		if r := d / math.Max(math.Abs(float64(want.Data[i])), math.Max(rms, 1e-12)); r > rel {
			rel, worst = r, i
		}
	}
	if tol := tolFor(name); rel > tol {
		t.Errorf("%s %s: max abs %.6g, worst rel %.3g at %d (got %g want %g), rms %.6g > %.0e",
			name, got, maxAbs, rel, worst, got.Data[worst], want.Data[worst], rms, tol)
		return
	}
	t.Logf("%-22s %-18s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
}

// TestDecoder walks the decoder stage by stage — conv_in, mid, every up
// block, the tail — and the clamped image, for the square and rectangular
// cases. The DupUp shortcuts, the reference's temporal-slice semantics
// included, are inside up_blocks.0-3.
func TestDecoder(t *testing.T) {
	m := loadManifest(t)
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Skipf("no VAE checkpoint at %s (%v)", vaeDir, err)
	}
	dec, err := LoadDecoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for label := range m.Cases {
		z := loadRef(t, m, label+"_z_norm")
		cfg.Denormalize(z)
		dec.Tap = func(name string, x *zvae.Tensor) {
			compare(t, label+"_dec_"+name, x, loadRef(t, m, label+"_dec_"+name))
		}
		img, err := dec.Decode(z)
		dec.Tap = nil
		if err != nil {
			t.Fatal(err)
		}
		compare(t, label+"_decoded", img, loadRef(t, m, label+"_decoded"))
	}
}

// TestEncoder walks the encoder the same way — the AvgDown shortcuts with
// their zero-frame means are inside down_blocks.1-3 — through to the
// posterior mode, raw and normalized.
func TestEncoder(t *testing.T) {
	m := loadManifest(t)
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Skipf("no VAE checkpoint at %s (%v)", vaeDir, err)
	}
	enc, err := LoadEncoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for label := range m.Cases {
		card := loadRef(t, m, label+"_card")
		enc.Tap = func(name string, x *zvae.Tensor) {
			compare(t, label+"_enc_"+name, x, loadRef(t, m, label+"_enc_"+name))
		}
		mode, err := enc.Encode(card)
		enc.Tap = nil
		if err != nil {
			t.Fatal(err)
		}
		compare(t, label+"_encoded_mode", mode, loadRef(t, m, label+"_encoded_mode"))
		cfg.Normalize(mode)
		compare(t, label+"_encoded_norm", mode, loadRef(t, m, label+"_encoded_norm"))
	}
}

// TestRoundtrip is decode(encode(image)) against the reference's own — the
// RGBA round trip in one number.
func TestRoundtrip(t *testing.T) {
	m := loadManifest(t)
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Skipf("no VAE checkpoint at %s (%v)", vaeDir, err)
	}
	enc, err := LoadEncoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := LoadDecoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	mode, err := enc.Encode(loadRef(t, m, "s256_card"))
	if err != nil {
		t.Fatal(err)
	}
	img, err := dec.Decode(mode)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "s256_roundtrip", img, loadRef(t, m, "s256_roundtrip"))
}
