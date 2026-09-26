package vae

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// The reference is reference/dump_h3_vae.py: diffusers' decode in fp32 on
// M4's final latents (256x448 x 124 frames). Regenerate with:
//
//	.venv/bin/python reference/dump_h3_vae.py   (~12 GB RSS, ~3 min)
const (
	vaeRef = "../../reference/out/h3vae"
	ditRef = "../../reference/out/h3dit"
	vaeDir = "../../models/MiniMax-H3/vae"
)

type vaeManifest struct {
	LatentFrames int                  `json:"latent_frames"`
	LatentHeight int                  `json:"latent_height"`
	LatentWidth  int                  `json:"latent_width"`
	ShortFrames  int                  `json:"short_frames"`
	KeepBlocks   []int                `json:"keep_blocks"`
	BlockStats   []map[string]float64 `json:"block_stats"`
	Tensors      map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T) *vaeManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(vaeRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_h3_vae.py", vaeRef, err)
	}
	m := &vaeManifest{}
	if err := json.Unmarshal(buf, m); err != nil {
		t.Fatal(err)
	}
	return m
}

func readF32(t *testing.T, path string, n int) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if n >= 0 && len(raw) != 4*n {
		t.Fatalf("%s: %d bytes, want %d floats", path, len(raw), n)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// readTensor reads a dumped [C, T, H, W] volume.
func readTensor(t *testing.T, m *vaeManifest, name string) *Tensor {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok || len(meta.Shape) != 4 {
		t.Fatalf("%s: no 4-d tensor %q", vaeRef, name)
	}
	s := meta.Shape
	return &Tensor{C: s[0], T: s[1], H: s[2], W: s[3], Data: readF32(t, filepath.Join(vaeRef, name+".bin"), meta.Count)}
}

// gap is max abs error over the reference's absmax, and rms error over its
// rms.
func gap(got, want []float32) (rel, rmsRel float64) {
	var maxAbs, ref, sq, refSq float64
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		maxAbs = math.Max(maxAbs, d)
		ref = math.Max(ref, math.Abs(float64(want[i])))
		sq += d * d
		refSq += float64(want[i]) * float64(want[i])
	}
	return maxAbs / ref, math.Sqrt(sq / refSq)
}

func testConfig(t *testing.T) *Config {
	t.Helper()
	c, err := LoadConfig(vaeDir)
	if err != nil {
		t.Skipf("no VAE config: %v", err)
	}
	return c
}

// TestSplitTiles holds `_split_tiles` to diffusers' own output for the
// served and trained canvases and a few around them.
func TestSplitTiles(t *testing.T) {
	for _, tc := range []struct {
		length   int
		starts   []int
		overlaps []int
	}{
		{256, []int{0}, nil},
		{448, []int{0, 192}, []int{64}},
		{480, []int{0, 112, 224}, []int{144, 144}},
		{544, []int{0, 144, 288}, []int{112, 112}},
		{768, []int{0, 160, 336, 512}, []int{96, 80, 80}},
		{864, []int{0, 144, 288, 448, 608}, []int{112, 112, 96, 96}},
		{960, []int{0, 176, 352, 528, 704}, []int{80, 80, 80, 80}},
		{1344, []int{0, 176, 352, 528, 704, 896, 1088}, []int{80, 80, 80, 80, 64, 64}},
	} {
		starts, size, overlaps := splitTiles(tc.length, tileSize, tileOverlap, 16)
		if !reflect.DeepEqual(starts, tc.starts) || !reflect.DeepEqual(overlaps, tc.overlaps) || size != min(tileSize, tc.length) {
			t.Errorf("%d: starts %v overlaps %v size %d, want %v %v", tc.length, starts, overlaps, size, tc.starts, tc.overlaps)
		}
	}
}

func TestFrames(t *testing.T) {
	c := testConfig(t)
	for lf, want := range map[int]int{7: 22, 12: 39, 37: 124, 102: 345} {
		got, err := c.Frames(lf)
		if err != nil || got != want {
			t.Errorf("Frames(%d) = %d, %v; want %d", lf, got, err, want)
		}
	}
	for _, lf := range []int{2, 11, 13} {
		if _, err := c.Frames(lf); err == nil {
			t.Errorf("Frames(%d) accepted a count the pipeline never makes", lf)
		}
	}
}

// TestLatent: the transformer's rows, unpatchified and denormalised, are
// the oracle's z bit for bit, and post_quant_conv on its first tile-clip is
// the decoder's input.
func TestLatent(t *testing.T) {
	m := loadManifest(t)
	c := testConfig(t)
	rows := readF32(t, filepath.Join(ditRef, "f6_latents.bin"), -1)
	z, err := Unpatchify(rows, c.LatentChannels, m.LatentFrames, m.LatentHeight, m.LatentWidth)
	if err != nil {
		t.Fatal(err)
	}
	c.Denormalize(z)
	want := readTensor(t, m, "z")
	for i := range want.Data {
		if z.Data[i] != want.Data[i] {
			t.Fatalf("z[%d] = %g, want %g", i, z.Data[i], want.Data[i])
		}
	}
	pqc, err := LoadPostQuantConv(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	tile := pqc.Apply(z.Slice(0, 7, 0, 16, 0, 16))
	rel, _ := gap(tile.Data, readTensor(t, m, "tile_in").Data)
	t.Logf("post_quant_conv: rel %.2e", rel)
	if rel > 1e-6 {
		t.Errorf("post_quant_conv rel %.2e", rel)
	}
}

func TestPlan(t *testing.T) {
	c := testConfig(t)
	for _, tc := range []struct{ lf, lh, lw, clips, tiles, tokens, frames int }{
		{37, 16, 28, 7, 2, 1797, 124},  // the oracle's 256x448
		{37, 30, 54, 7, 15, 1797, 124}, // 480p served
		{37, 48, 84, 7, 28, 1797, 124}, // 768p trained
		{12, 8, 8, 2, 1, 7*64 + 5, 39}, // one small tile
	} {
		p, err := c.NewPlan(tc.lf, tc.lh, tc.lw)
		if err != nil {
			t.Fatal(err)
		}
		if p.Clips != tc.clips || p.Tiles() != tc.tiles || p.Tokens(c) != tc.tokens || p.Frames != tc.frames {
			t.Errorf("%dx%dx%d: %d clips × %d tiles of %d tokens, %d frames; want %d × %d of %d, %d",
				tc.lf, tc.lh, tc.lw, p.Clips, p.Tiles(), p.Tokens(c), p.Frames, tc.clips, tc.tiles, tc.tokens, tc.frames)
		}
	}
}
