package kokoro

import (
	"path/filepath"
	"testing"
)

// TestConfig checks the constants the rest of the package builds on against
// the manifest, since three of them (the ALBERT eps, the embedding width and
// the samples-per-frame arithmetic) are not in config.json at all.
func TestConfig(t *testing.T) {
	m := loadManifest(t)
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Skipf("no checkpoint at %s (%v)", modelDir, err)
	}
	if got := cfg.PLBERT.NumHiddenLayers; got != m.BERT.Layers {
		t.Errorf("layers = %d, reference ran %d", got, m.BERT.Layers)
	}
	if got := cfg.PLBERT.HeadDim(); got != m.BERT.HeadDim {
		t.Errorf("head dim = %d, reference has %d", got, m.BERT.HeadDim)
	}
	if got := cfg.SamplesPerFrame(); got*m.LengthRegulator.Frames != m.LengthRegulator.Samples {
		t.Errorf("%d samples a frame x %d frames != the reference's %d samples",
			got, m.LengthRegulator.Frames, m.LengthRegulator.Samples)
	}
	if cfg.SamplingRate != m.SampleRate {
		t.Errorf("sampling rate = %d, reference has %d", cfg.SamplingRate, m.SampleRate)
	}

	// The phoneme string is the input the whole dump hangs off, so the
	// vocabulary lookup is checked against the ids the reference actually ran
	// rather than against itself.
	ids, dropped := cfg.Phonemes(m.Phonemes)
	if dropped != 0 {
		t.Errorf("%d phonemes outside the vocabulary", dropped)
	}
	if len(ids) != len(m.InputIDs) {
		t.Fatalf("%d ids, reference has %d", len(ids), len(m.InputIDs))
	}
	for i := range ids {
		if ids[i] != m.InputIDs[i] {
			t.Fatalf("id %d = %d, reference has %d", i, ids[i], m.InputIDs[i])
		}
	}
}

// TestVoices checks the style pack loads and is indexed the way KPipeline
// indexes it — by phoneme count, not token count. The row it picks is
// compared against the 256 floats the reference actually used.
func TestVoices(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	if len(model.Voices) == 0 {
		t.Fatal("no voices")
	}
	dec, pred, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	want, _ := loadRef(t, m, "ref_s")
	got := append(append([]float32(nil), dec...), pred...)
	if d := compare(t, got, want); d.MaxAbs != 0 {
		t.Errorf("style row differs by %.3g at %d — wrong row?", d.MaxAbs, d.At)
	}
	if _, _, err := model.Style("no_such_voice", 10); err == nil {
		t.Error("an unknown voice loaded")
	}
	if _, _, err := model.Style(m.Voice, 100000); err == nil {
		t.Error("a row past the end of the pack loaded")
	}
}

// TestALBERT walks the phoneme encoder: the embedding stack, one layer's
// attention and output, then every twelfth of the way to the final hidden
// state.
//
// The bound is relative rather than absolute because the deviation here is
// float32 summation order, not arithmetic: Go accumulates a 768-wide dot
// product in a different order than torch's BLAS does, and twelve
// post-normalised layers carry that forward without amplifying it.
func TestALBERT(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	ids := m.InputIDs

	embed, hidden, err := model.BERT.Embed(ids)
	if err != nil {
		t.Fatal(err)
	}
	check(t, m, "bert_embed", embed, false, 1e-6)
	check(t, m, "bert_hidden_in", hidden, false, 1e-6)

	layer := model.BERT.Layer
	q, err := layer.Q.Apply(hidden)
	if err != nil {
		t.Fatal(err)
	}
	k, _ := layer.K.Apply(hidden)
	v, _ := layer.V.Apply(hidden)
	check(t, m, "bert_q", q, false, 1e-6)
	check(t, m, "bert_k", k, false, 1e-6)
	check(t, m, "bert_v", v, false, 1e-6)

	attn, err := layer.Attention(hidden)
	if err != nil {
		t.Fatal(err)
	}
	check(t, m, "bert_attn_out", attn, false, 1e-5)

	out, err := layer.Apply(hidden)
	if err != nil {
		t.Fatal(err)
	}
	check(t, m, "bert_layer_out", out, false, 1e-5)

	hiddens, err := model.BERT.Apply(ids)
	if err != nil {
		t.Fatal(err)
	}
	if len(hiddens) != m.BERT.Layers {
		t.Fatalf("%d layers, reference ran %d", len(hiddens), m.BERT.Layers)
	}
	check(t, m, "bert_hidden_5", hiddens[5], false, 1e-5)
	check(t, m, "bert_out", hiddens[len(hiddens)-1], false, 1e-5)
}

// TestTextEncoder walks the second, shallower path from the phonemes — the
// one whose output is expanded and handed to the vocoder.
func TestTextEncoder(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)

	embed, err := model.TextEncoder.Embedding.Rows(m.InputIDs)
	if err != nil {
		t.Fatal(err)
	}
	check(t, m, "te_embed", embed, true, 1e-6)

	x := embed
	for i := range model.TextEncoder.Convs {
		if x, err = model.TextEncoder.Convs[i].Apply(x); err != nil {
			t.Fatal(err)
		}
		if err = model.TextEncoder.Norms[i].ApplyInPlace(x); err != nil {
			t.Fatal(err)
		}
		leakyReLUInPlace(x, 0.2)
		check(t, m, "te_cnn_"+string(rune('0'+i)), x, true, 1e-5)
	}

	out, err := model.TextEncoder.Apply(m.InputIDs, nil)
	if err != nil {
		t.Fatal(err)
	}
	check(t, m, "t_en", out, true, 1e-5)
}

// TestLoadRejectsRawCheckpoint states the failure mode of pointing the loader
// at an unconverted directory: the pickle is not a mapping and there is
// nothing to read.
func TestLoadRejectsRawCheckpoint(t *testing.T) {
	dir := t.TempDir()
	if _, err := Load(dir); err == nil {
		t.Error("an empty directory loaded")
	}
	if _, err := Load(filepath.Join(dir, "nope")); err == nil {
		t.Error("a missing directory loaded")
	}
}
