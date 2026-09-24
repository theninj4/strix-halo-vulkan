package vision

import (
	"fmt"
	"math"
	"testing"
)

// qwen3.8-flash-next's tower, twice: llama.cpp's mmproj and the HF
// checkpoint's first shard, which holds all of `model.visual.*`
// (LLM-VISION.md V0).
const (
	llmMMProj = "../../models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf"
	llmHF     = "../../models/Qwen3.8-Flash-Next-HF-vision"
)

// TestGGUFMatchesSafetensors is V1's gate. Both files are BF16 widened to
// fp32 (and the mmproj's F32 tensors are BF16 values widened by the
// converter), so the two loaders have to agree **bit for bit** on every
// tensor. That proves the name map, the per-frame patch kernels'
// interleave and the merger's placement before any arithmetic runs on them.
// The config the GGUF implies has to be the one config.json states.
func TestGGUFMatchesSafetensors(t *testing.T) {
	want, err := LoadConfig(llmHF)
	if err != nil {
		t.Skipf("no HF vision shard at %s (%v)", llmHF, err)
	}
	cfg, got, err := LoadGGUF(llmMMProj, 0)
	if err != nil {
		t.Skipf("no mmproj at %s (%v)", llmMMProj, err)
	}
	if fmt.Sprint(*cfg) != fmt.Sprint(*want) {
		t.Fatalf("config from the mmproj:\n  %+v\nconfig.json:\n  %+v", *cfg, *want)
	}
	ref, err := Load(llmHF, want, 0)
	if err != nil {
		t.Fatal(err)
	}

	n := 0
	same := func(name string, a, b []float32) {
		t.Helper()
		n++
		if len(a) != len(b) {
			t.Errorf("%s: %d values, HF has %d", name, len(a), len(b))
			return
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Errorf("%s[%d] = %g, HF has %g", name, i, a[i], b[i])
				return
			}
		}
	}
	lin := func(name string, a, b Linear) {
		same(name+".weight", a.Weight, b.Weight)
		same(name+".bias", a.Bias, b.Bias)
	}
	ln := func(name string, a, b LayerNorm) {
		same(name+".weight", a.Weight, b.Weight)
		same(name+".bias", a.Bias, b.Bias)
	}

	lin("patch_embed", got.PatchProj, ref.PatchProj)
	same("pos_embed", got.PosEmbed, ref.PosEmbed)
	if len(got.Blocks) != len(ref.Blocks) {
		t.Fatalf("%d blocks, HF has %d", len(got.Blocks), len(ref.Blocks))
	}
	for i := range got.Blocks {
		g, r := &got.Blocks[i], &ref.Blocks[i]
		p := fmt.Sprintf("blocks.%d.", i)
		ln(p+"norm1", g.Norm1, r.Norm1)
		ln(p+"norm2", g.Norm2, r.Norm2)
		lin(p+"attn.qkv", g.Attn.QKV, r.Attn.QKV)
		lin(p+"attn.proj", g.Attn.Proj, r.Attn.Proj)
		lin(p+"mlp.fc1", g.MLP.FC1, r.MLP.FC1)
		lin(p+"mlp.fc2", g.MLP.FC2, r.MLP.FC2)
	}
	if got.Merger.PostShuffleNorm != ref.Merger.PostShuffleNorm || got.Merger.Merge != ref.Merger.Merge {
		t.Errorf("merger kind: mmproj post=%v merge=%d, HF post=%v merge=%d",
			got.Merger.PostShuffleNorm, got.Merger.Merge, ref.Merger.PostShuffleNorm, ref.Merger.Merge)
	}
	ln("merger.norm", got.Merger.Norm, ref.Merger.Norm)
	lin("merger.fc1", got.Merger.FC1, ref.Merger.FC1)
	lin("merger.fc2", got.Merger.FC2, ref.Merger.FC2)
	if len(got.Deepstack) != 0 || len(ref.Deepstack) != 0 {
		t.Errorf("deepstack mergers: mmproj %d, HF %d, want none", len(got.Deepstack), len(ref.Deepstack))
	}
	t.Logf("%d tensors bit-identical; tower %d blocks x %d, merger -> %d",
		n, cfg.Depth, cfg.HiddenSize, cfg.OutHiddenSize)
}
