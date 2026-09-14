package parakeet

import (
	"testing"

	"strix-halo-vulkan/audio"
)

// TestPredictionMatchesReference checks the LSTM against two steps of the
// reference: the blank start token from a zero state, and a real token from
// the state that leaves behind. One step alone would not exercise the
// recurrence at all — and it would hide a swapped gate order, since the
// embedding of the blank is a row of zeros in this checkpoint.
func TestPredictionMatchesReference(t *testing.T) {
	m := testModel(t)
	ref := loadManifest(t)

	state := m.Prediction.NewState()
	out, err := m.Prediction.Step(m.Config.BlankTokenID, state)
	if err != nil {
		t.Fatal(err)
	}
	checkVec(t, ref, "dec_out", out)
	checkState(t, ref, "dec_h", state.H)
	checkState(t, ref, "dec_c", state.C)

	out, err = m.Prediction.Step(1, state)
	if err != nil {
		t.Fatal(err)
	}
	checkVec(t, ref, "dec_out1", out)
	checkState(t, ref, "dec_h1", state.H)
	checkState(t, ref, "dec_c1", state.C)

	// The blank's embedding row really is zero, which is why the first step
	// has to be checked through its state rather than through its input.
	emb := m.Prediction.Embedding[m.Config.BlankTokenID*m.Prediction.Hidden:]
	for i := 0; i < m.Prediction.Hidden; i++ {
		if emb[i] != 0 {
			t.Fatalf("the blank's embedding is not zero at %d: %v", i, emb[i])
		}
	}
}

func checkVec(t *testing.T, ref *manifest, name string, got []float32) {
	t.Helper()
	want, meta := loadRef(t, ref, name)
	if len(got) != meta.Count {
		t.Fatalf("%s: %d values, reference has %d %v", name, len(got), meta.Count, meta.Shape)
	}
	d := compare(t, got, want)
	t.Logf("%-10s max abs %.3g, rms %.4g", name, d.MaxAbs, d.RMS)
	if d.MaxAbs > encTol*d.RMS {
		t.Errorf("%s deviates by %.3g against an rms of %.4g", name, d.MaxAbs, d.RMS)
	}
}

func checkState(t *testing.T, ref *manifest, name string, layers [][]float32) {
	t.Helper()
	var flat []float32
	for _, l := range layers {
		flat = append(flat, l...)
	}
	checkVec(t, ref, name, flat)
}

// TestJointMatchesReference runs the joint at the first frame against the
// first prediction state, which is the (t=0, u=0) cell of the transducer
// lattice and the first thing the decode loop evaluates.
func TestJointMatchesReference(t *testing.T) {
	m := testModel(t)
	ref := loadManifest(t)

	enc, _ := loadRef(t, ref, "encoder_projected")
	state := m.Prediction.NewState()
	dec, err := m.Prediction.Step(m.Config.BlankTokenID, state)
	if err != nil {
		t.Fatal(err)
	}
	width := m.Config.DecoderHiddenSize
	logits := m.Joint.Logits(make([]float32, m.Joint.Head.Out), enc[:width], dec)
	checkVec(t, ref, "joint_logits", logits)

	// The head is 8193 token logits followed by five duration logits, and
	// the two argmaxes are independent. Both are checked against the
	// reference's first decode step rather than against the tensor, since
	// that is the form the loop consumes them in.
	token, duration := m.Joint.Argmax(logits, m.Config.BlankTokenID)
	first := ref.Decode.Trace[0]
	if token != first.Token || duration != first.Duration {
		t.Errorf("first step emits (%d, %d), reference has (%d, %d)", token, duration, first.Token, first.Duration)
	}
	if got, want := m.Joint.Head.Out, ref.TDT.VocabSize+len(ref.TDT.Durations); got != want {
		t.Errorf("joint head is %d wide, want %d token + %d duration logits",
			got, ref.TDT.VocabSize, len(ref.TDT.Durations))
	}
}

// TestDecodeMatchesReference runs the TDT loop over the reference's own
// encoder output, so that a difference here is the loop's and not the
// encoder's.
func TestDecodeMatchesReference(t *testing.T) {
	m := testModel(t)
	ref := loadManifest(t)

	enc, meta := loadRef(t, ref, "encoder_projected")
	frames := meta.Shape[0]
	got, err := m.Decode(&Mat{Rows: frames, Cols: meta.Shape[1], Data: enc}, ref.Encoder.ValidFrames)
	if err != nil {
		t.Fatal(err)
	}
	compareTrace(t, ref, got)
	if got.Text != ref.Decode.Text {
		t.Errorf("text\n got %q\nwant %q", got.Text, ref.Decode.Text)
	}
}

func compareTrace(t *testing.T, ref *manifest, got *Transcript) {
	t.Helper()
	if len(got.Steps) != len(ref.Decode.Trace) {
		t.Errorf("%d steps, reference has %d", len(got.Steps), len(ref.Decode.Trace))
	}
	for i := range got.Steps {
		if i >= len(ref.Decode.Trace) {
			break
		}
		want := ref.Decode.Trace[i]
		if got.Steps[i].Frame != want.T || got.Steps[i].Token != want.Token || got.Steps[i].Duration != want.Duration {
			t.Fatalf("step %d: frame %d token %d duration %d, reference has frame %d token %d duration %d",
				i, got.Steps[i].Frame, got.Steps[i].Token, got.Steps[i].Duration, want.T, want.Token, want.Duration)
		}
	}
}

// TestTranscribeMatchesReference is the whole vertical: a WAV file in, a
// string out, compared against the reference's transcript of the same clip.
//
// This is the bar SPEECH.md set for the STT side — the correctness bound is
// an exact string, not a tolerance — so it is the one test in this package
// whose failure means the model is wrong rather than drifting.
func TestTranscribeMatchesReference(t *testing.T) {
	if testing.Short() {
		t.Skip("the whole model is ~180 GFLOP on the CPU")
	}
	m := testModel(t)
	ref := loadManifest(t)
	clip, err := audio.ReadWAV(fixture)
	if err != nil {
		t.Fatal(err)
	}

	got, err := m.Transcribe(clip)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d steps over %d frames, %d tokens: %q", len(got.Steps), got.Frames, len(got.Tokens), got.Text)

	if got.Text != transcript {
		t.Errorf("transcript\n got %q\nwant %q", got.Text, transcript)
	}
	if got.Text != ref.Decode.Text {
		t.Errorf("transcript differs from the reference's\n got %q\nwant %q", got.Text, ref.Decode.Text)
	}
	compareTrace(t, ref, got)

	// The reference itself has to have been checked against the known
	// transcript and against transformers' own generate(), or the comparison
	// above is circular.
	if !ref.Decode.Matches || !ref.Decode.SameAsHF {
		t.Errorf("the reference dump does not agree with itself: matches_expected=%v manual_matches_generate=%v",
			ref.Decode.Matches, ref.Decode.SameAsHF)
	}
}

// TestTokenizerDecode checks the Metaspace rule on the reference's own
// emitted ids, independently of the model producing them.
func TestTokenizerDecode(t *testing.T) {
	tok, err := LoadTokenizer(modelDir)
	if err != nil {
		t.Skipf("no checkpoint in %s (%v)", modelDir, err)
	}
	ref := loadManifest(t)
	got, err := tok.Decode(ref.Decode.Emitted)
	if err != nil {
		t.Fatal(err)
	}
	if got != ref.Decode.Text {
		t.Errorf("decode\n got %q\nwant %q", got, ref.Decode.Text)
	}

	// The blank has a vocabulary entry of its own — `<blank>` at 8192, flagged
	// special — so the whole emission stream, blanks included, decodes to the
	// same string as the filtered one.
	if tok.Size() != ref.TDT.VocabSize {
		t.Errorf("vocabulary covers %d ids, the model has %d", tok.Size(), ref.TDT.VocabSize)
	}
	if piece, err := tok.Piece(ref.TDT.BlankTokenID); err != nil || piece != "<blank>" {
		t.Errorf("piece %d = %q, %v; want \"<blank>\"", ref.TDT.BlankTokenID, piece, err)
	}
	raw, err := tok.Decode(ref.Decode.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	if raw != ref.Decode.Text {
		t.Errorf("decoding the unfiltered stream\n got %q\nwant %q", raw, ref.Decode.Text)
	}
}
