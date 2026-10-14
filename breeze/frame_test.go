package breeze

import (
	"context"
	"math/rand/v2"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// TestDeviceSampler prices breeze_frame.comp's sampler against the host's
// on random logits with the same uniform: CFG on and off, the penalty, the
// backbone's EOS column. A draw within rounding of a boundary may land on
// the neighbouring id; anything more is a bug.
func TestDeviceSampler(t *testing.T) {
	d := loadDecoder(t)
	const trials = 3000
	gen := rand.New(rand.NewPCG(1, 2))
	hostRNG := rand.New(rand.NewPCG(3, 4))
	devRNG := rand.New(rand.NewPCG(3, 4))
	s := &sampler{temp: 0.9, topK: 50, rng: hostRNG}
	miss := 0
	for i := 0; i < trials; i++ {
		nb, n := 1+i%2, Vocab+(i/2)%2
		backbone := n == Vocab+1
		sp := sampling{cfg: 1, temp: 0.9, topK: 50, pen: 1.1}
		if nb == 2 {
			sp.cfg = 4
		}
		lg := make([][]float32, nb)
		for b := range lg {
			lg[b] = make([]float32, n)
			for j := range lg[b] {
				lg[b][j] = float32(gen.NormFloat64() * 2)
			}
			// a peaked head, as real logits are
			for j := 0; j < 8; j++ {
				lg[b][gen.IntN(n)] += 6
			}
			d.abuf.WriteFloat32At(int(d.aLogits)+b*d.headPad, lg[b])
		}
		var history []int32
		seen := make([]uint32, Vocab+1)
		if backbone {
			for j := 0; j < 40; j++ {
				c := int32(gen.IntN(Codes))
				history = append(history, c)
				seen[c] = 1
			}
		}
		d.abuf.WriteUint32At(int(d.aSeen), seen)
		d.abuf.WriteFloat32At(int(d.aUniform), []float32{float32(devRNG.Float64())})

		g := []vk.MultiDispatch{d.sampleDispatch(0, nb, n, 0, 5, 0, sp, backbone)}
		if _, err := vk.DispatchMultiTimed(g, 1, 1, true); err != nil {
			t.Fatal(err)
		}
		got := int32(d.abuf.ReadUint32At(int(d.aFrame)+5, 1)[0])

		in := make([]float32, n)
		if nb == 1 {
			copy(in, lg[0])
		} else {
			for j := range in {
				in[j] = lg[1][j] + sp.cfg*(lg[0][j]-lg[1][j])
			}
		}
		pen := 1.0
		if backbone {
			pen = sp.pen
		}
		want := s.sample(in, history, pen, Codes, Vocab)
		if got != want {
			miss++
			if miss <= 5 {
				t.Logf("trial %d (nb %d, n %d): device %d, host %d", i, nb, n, got, want)
			}
		}
		if backbone && d.abuf.ReadUint32At(int(d.aSeen)+int(got), 1)[0] != 1 {
			t.Fatalf("trial %d: the sampled id %d was not marked", i, got)
		}
	}
	t.Logf("%d of %d draws differ", miss, trials)
	if miss > trials/1000 {
		t.Fatalf("%d of %d draws differ from the host sampler", miss, trials)
	}
}

// TestDeviceFrames generates the same seeded utterance with the device frame
// and the host loop: the same draws give the same frames, so the audio is
// identical unless a draw fell within rounding of a boundary.
func TestDeviceFrames(t *testing.T) {
	m := loadModel(t)
	for _, c := range []struct {
		name string
		req  Request
	}{
		{"design", Request{Text: "Good morning everybody, and welcome back to the breakfast show!", Instruction: "An enthusiastic radio DJ"}},
		{"plain", Request{Text: "The quick brown fox jumps over the lazy dog."}},
	} {
		t.Run(c.name, func(t *testing.T) {
			var wavs [2][]float32
			var sts [2]*Stats
			for i, host := range []bool{true, false} {
				st, err := m.Generate(context.Background(), &c.req, GenOptions{Seed: 7, HostSampler: host},
					func(s []float32) error { wavs[i] = append(wavs[i], s...); return nil })
				if err != nil {
					t.Fatal(err)
				}
				sts[i] = st
			}
			for i, name := range []string{"host", "device"} {
				st := sts[i]
				f := float64(st.Frames)
				t.Logf("%s: %d frames in %v, %.2f ms a frame (wall, after the prefill), device frame %.2f ms",
					name, st.Frames, st.Total, (st.Total-st.Encode-st.Prefill).Seconds()*1e3/f, st.Frame.Seconds()*1e3/f)
			}
			if len(wavs[0]) != len(wavs[1]) {
				t.Errorf("host %d samples, device %d", len(wavs[0]), len(wavs[1]))
			}
			for j := range min(len(wavs[0]), len(wavs[1])) {
				if wavs[0][j] != wavs[1][j] {
					t.Fatalf("the audio differs from frame %d of %d", j/1920, len(wavs[0])/1920)
				}
			}
		})
	}
}

var (
	modelOnce sync.Once
	testModel *Model
)

// loadModel is the package's one Model over the shared test parts.
func loadModel(t *testing.T) *Model {
	t.Helper()
	tok, enc, dec, codec := loadTokenizer(t), loadEncoder(t), loadDecoder(t), loadCodec(t)
	modelOnce.Do(func() { testModel = &Model{Tok: tok, Enc: enc, Dec: dec, Codec: codec} })
	return testModel
}

// TestSharedFrames runs two utterances alone and then at once, sharing
// every frame's passes: each must come out the same as it did alone, and
// the pair should take little more than the longer one.
func TestSharedFrames(t *testing.T) {
	m := loadModel(t)
	reqs := []struct {
		req Request
		o   GenOptions
	}{
		{Request{Text: "Good morning everybody, and welcome back to the breakfast show!", Instruction: "An enthusiastic radio DJ"}, GenOptions{Seed: 7}},
		{Request{Text: "The quick brown fox jumps over the lazy dog, and then it naps in the sun."}, GenOptions{Seed: 3}},
	}
	run := func(i int) ([]float32, *Stats) {
		var wav []float32
		st, err := m.Generate(context.Background(), &reqs[i].req, reqs[i].o,
			func(s []float32) error { wav = append(wav, s...); return nil })
		if err != nil {
			t.Error(err)
		}
		return wav, st
	}
	var alone [2][]float32
	var aloneT time.Duration
	for i := range reqs {
		var st *Stats
		alone[i], st = run(i)
		aloneT += st.Total
		t.Logf("alone %d: %d frames in %v", i, st.Frames, st.Total)
	}
	var both [2][]float32
	var wg sync.WaitGroup
	t0 := time.Now()
	for i := range reqs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var st *Stats
			both[i], st = run(i)
			t.Logf("together %d: %d frames in %v, first audio %v, %.2f ms device a frame",
				i, st.Frames, st.Total, st.FirstAudio, st.Frame.Seconds()*1e3/float64(st.Frames))
		}()
	}
	wg.Wait()
	t.Logf("alone, one after the other: %v; together: %v", aloneT, time.Since(t0))
	for i := range reqs {
		if len(alone[i]) != len(both[i]) {
			t.Errorf("request %d: %d samples alone, %d together", i, len(alone[i]), len(both[i]))
		}
		for k := range min(len(alone[i]), len(both[i])) {
			if alone[i][k] != both[i][k] {
				t.Errorf("request %d: the audio differs from frame %d of %d", i, k/FrameSamples, len(alone[i])/FrameSamples)
				break
			}
		}
	}
}
