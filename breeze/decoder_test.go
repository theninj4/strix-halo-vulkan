package breeze

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

const refDir = "../reference/out/breeze"

const strixHaloDeviceID = 0x1586

var (
	devOnce sync.Once
	testDev *vk.Device
	devErr  error

	decOnce sync.Once
	testDec *Decoder
	decErr  error
)

func openDevice() (*vk.Device, error) {
	inst, err := vk.NewInstance("breeze-test")
	if err != nil {
		return nil, err
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		return nil, fmt.Errorf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		return nil, err
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		return nil, err
	}
	return vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
}

func device(t testing.TB) *vk.Device {
	t.Helper()
	devOnce.Do(func() { testDev, devErr = openDevice() })
	if devErr != nil {
		t.Skipf("no device: %v", devErr)
	}
	return testDev
}

// loadDecoder stages the decoder once for the package's tests.
func loadDecoder(t testing.TB) *Decoder {
	t.Helper()
	dev := device(t)
	decOnce.Do(func() {
		t0 := time.Now()
		o := DefaultOptions()
		if os.Getenv("BREEZE_BANK") == "fp16" {
			o.Bank = BankFP16
		}
		testDec, decErr = Load(dev, modelDir(), o)
		if decErr == nil {
			t.Logf("staged the decoder (%s bank) in %v", testDec.o.Bank, time.Since(t0))
		}
	})
	if decErr != nil {
		t.Skipf("no checkpoint: %v", decErr)
	}
	return testDec
}

// refCase reads a case's dump.
func refCase(t testing.TB, name string) map[string]*safetensors.Tensor {
	t.Helper()
	f, err := safetensors.Open(filepath.Join(refDir, name, "tensors.safetensors"))
	if err != nil {
		t.Skipf("no dump (reference/dump_breeze.py): %v", err)
	}
	t.Cleanup(func() { f.Close() })
	out := map[string]*safetensors.Tensor{}
	for _, n := range f.Names() {
		out[n], _ = f.Get(n)
	}
	return out
}

// refBF16 reads a case's bf16 teacher-forced logits
// (reference/dump_breeze.py --bf16), or nil.
func refBF16(t testing.TB, name string) map[string][]float32 {
	t.Helper()
	f, err := safetensors.Open(filepath.Join(refDir, name, "tensors_bf16.safetensors"))
	if err != nil {
		return nil
	}
	defer f.Close()
	out := map[string][]float32{}
	for _, n := range f.Names() {
		ten, _ := f.Get(n)
		out[n] = f32(t, ten)
	}
	return out
}

func f32(t testing.TB, ten *safetensors.Tensor) []float32 {
	t.Helper()
	v, err := ten.F32(nil)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// frames reads a dump's frames as [F][16] codes.
func frames(t testing.TB, ref map[string]*safetensors.Tensor) [][16]int32 {
	v := f32(t, ref["frames"])
	out := make([][16]int32, len(v)/16)
	for i := range out {
		for k := range out[i] {
			out[i][k] = int32(v[i*16+k])
		}
	}
	return out
}

// stats compares got with want: the error relative to want's rms, the max
// abs error and, for logits, the KL of the sampling distribution
// (temperature 0.9) from the oracle's and whether the argmax agrees.
type stats struct {
	rel, maxAbs, kl float64
	argmax          bool
}

func compare(got, want []float32, temp float64) stats {
	var se, sw, mx float64
	ga, wa := 0, 0
	for i := range want {
		e := float64(got[i] - want[i])
		se += e * e
		sw += float64(want[i]) * float64(want[i])
		mx = math.Max(mx, math.Abs(e))
		if got[i] > got[ga] {
			ga = i
		}
		if want[i] > want[wa] {
			wa = i
		}
	}
	s := stats{rel: math.Sqrt(se / sw), maxAbs: mx, argmax: ga == wa}
	if temp > 0 {
		lse := func(x []float32) float64 {
			m := math.Inf(-1)
			for _, v := range x {
				m = math.Max(m, float64(v)/temp)
			}
			var z float64
			for _, v := range x {
				z += math.Exp(float64(v)/temp - m)
			}
			return m + math.Log(z)
		}
		lw, lg := lse(want), lse(got)
		for i := range want {
			pw := float64(want[i])/temp - lw
			s.kl += math.Exp(pw) * (pw - (float64(got[i])/temp - lg))
		}
	}
	return s
}

// The bounds, measured over the three dumped cases (B2-B4, all fp16
// operands with fp32 sums against the fp32 oracle): the backbone's worst
// hidden state 5.1e-4 relative, its logits' KL at temperature 0.9 1.2e-6,
// every argmax; the depth decoder's logits 1.1e-3 and KL 6.8e-6, 3 argmax
// missed of 4,590 (near-ties); the encoder 7.5e-4 and its projection 1.3e-3.
// Each bound is a few times the measured worst.
const (
	backboneTol = 2e-3
	backboneKL  = 1e-5
	depthTol    = 4e-3
	depthKL     = 5e-5
	depthMiss   = 0.005
	encTol      = 4e-3
	// q8VsBF16 bounds the int8 bank's mean KL against the fp32 oracle as a
	// multiple of upstream bf16's own (its serving precision, B7).
	q8VsBF16 = 1.0
)

// TestBackboneTeacher runs each branch's prompt and the frames the oracle
// sampled through the backbone as one causal pass, and holds the normed
// hidden state and the head's logits after the prompt and after every frame
// to the fp32 dump.
func TestBackboneTeacher(t *testing.T) {
	d := loadDecoder(t)
	for _, name := range []string{"design", "plain", "zh", "long", "clone", "direct"} {
		ref := refCase(t, name)
		fr := frames(t, ref)
		for _, br := range []string{"cond", "uncond"} {
			if ref[br+"_embeds"] == nil {
				continue
			}
			emb := f32(t, ref[br+"_embeds"])
			L := len(emb) / bbHidden
			n := L + len(fr)
			x := make([]float32, n*bbHidden)
			copy(x, emb)
			for f, codes := range fr {
				d.frameEmbed(x[(L+f)*bbHidden:(L+f+1)*bbHidden], codes)
			}
			rows := make([]Row, n)
			for i := range rows {
				rows[i] = Row{0, i}
			}
			hid, lg, took, err := d.Backbone(x, rows, len(fr)+1)
			if err != nil {
				t.Fatal(err)
			}
			wantH, wantL := f32(t, ref[br+"_hidden"]), f32(t, ref[br+"_logits"])
			bf := refBF16(t, name)[br+"_logits"]
			var worstH, worstL, worstKL, meanKL, bfKL float64
			miss, bfMiss := 0, 0
			for i := 0; i <= len(fr); i++ {
				w := wantL[i*(Vocab+1) : (i+1)*(Vocab+1)]
				h := compare(hid[L-1+i], wantH[i*bbHidden:(i+1)*bbHidden], 0)
				l := compare(lg[i], w, 0.9)
				worstH, worstL, worstKL = math.Max(worstH, h.rel), math.Max(worstL, l.rel), math.Max(worstKL, l.kl)
				meanKL += l.kl / float64(len(fr)+1)
				if !l.argmax {
					miss++
				}
				if bf != nil {
					b := compare(bf[i*(Vocab+1):(i+1)*(Vocab+1)], w, 0.9)
					bfKL += b.kl / float64(len(fr)+1)
					if !b.argmax {
						bfMiss++
					}
				}
			}
			t.Logf("%s/%s: %d rows in %v: hidden rel %.2e, logits rel %.2e, KL worst %.2e mean %.2e (bf16 %.2e: %.2fx), argmax missed %d/%d (bf16 %d)",
				name, br, n, took.Round(time.Microsecond), worstH, worstL, worstKL, meanKL, bfKL, meanKL/bfKL, miss, len(fr)+1, bfMiss)
			if d.o.Bank == BankQ8 {
				if bf != nil && meanKL > q8VsBF16*bfKL {
					t.Errorf("%s/%s: int8's mean KL is %.2fx bf16's; the bound is %.1fx", name, br, meanKL/bfKL, q8VsBF16)
				}
			} else if worstH > backboneTol || worstKL > backboneKL || miss > 0 {
				t.Errorf("%s/%s: past the bounds (rel %.0e, KL %.0e, no argmax missed)", name, br, backboneTol, backboneKL)
			}
		}
	}
}

// TestDepthTeacher runs every frame of each case through the depth decoder
// as one 16-row pass -- the oracle's backbone hidden state, then codes 0-14
// -- and holds the 15 codebooks' logits to the dump.
func TestDepthTeacher(t *testing.T) {
	d := loadDecoder(t)
	for _, name := range []string{"design", "plain", "zh", "long", "clone", "direct"} {
		ref := refCase(t, name)
		fr := frames(t, ref)
		for _, br := range []string{"cond", "uncond"} {
			if ref[br+"_embeds"] == nil {
				continue
			}
			wantH, want := f32(t, ref[br+"_hidden"]), f32(t, ref[br+"_depth_logits"])
			bf := refBF16(t, name)[br+"_depth_logits"]
			var meanKL, bfKL float64
			bfMiss := 0
			in := make([]float32, Codebooks*bbHidden)
			rows := make([]Row, Codebooks)
			heads := make([]int, Codebooks)
			for i := range rows {
				rows[i] = Row{0, i}
				heads[i] = i - 1
			}
			var worstL, worstKL float64
			miss, total := 0, 0
			var took time.Duration
			for f, codes := range fr {
				copy(in, wantH[f*bbHidden:(f+1)*bbHidden])
				for k := 0; k < Codebooks-1; k++ {
					d.codeEmbed(in[(k+1)*bbHidden:(k+2)*bbHidden], k, codes[k])
				}
				lg, dt, err := d.Depth(in, rows, heads)
				if err != nil {
					t.Fatal(err)
				}
				took += dt
				for k := 0; k < Codebooks-1; k++ {
					w := want[(f*(Codebooks-1)+k)*Vocab : (f*(Codebooks-1)+k+1)*Vocab]
					s := compare(lg[k+1], w, 0.9)
					worstL, worstKL = math.Max(worstL, s.rel), math.Max(worstKL, s.kl)
					total++
					meanKL += s.kl
					if !s.argmax {
						miss++
					}
					if bf != nil {
						b := compare(bf[(f*(Codebooks-1)+k)*Vocab:(f*(Codebooks-1)+k+1)*Vocab], w, 0.9)
						bfKL += b.kl
						if !b.argmax {
							bfMiss++
						}
					}
				}
			}
			meanKL /= float64(total)
			bfKL /= float64(total)
			t.Logf("%s/%s: %d frames in %v: logits rel %.2e, KL worst %.2e mean %.2e (bf16 %.2e: %.2fx), argmax missed %d/%d (bf16 %d)",
				name, br, len(fr), took.Round(time.Microsecond), worstL, worstKL, meanKL, bfKL, meanKL/bfKL, miss, total, bfMiss)
			if d.o.Bank == BankQ8 {
				if bf != nil && meanKL > q8VsBF16*bfKL {
					t.Errorf("%s/%s: int8's mean KL is %.2fx bf16's; the bound is %.1fx", name, br, meanKL/bfKL, q8VsBF16)
				}
			} else if worstL > depthTol || worstKL > depthKL || float64(miss) > depthMiss*float64(total) {
				t.Errorf("%s/%s: past the bounds (rel %.0e, KL %.0e, argmax missed under %.1f%%)", name, br,
					depthTol, depthKL, 100*depthMiss)
			}
		}
	}
}

var (
	encOnce sync.Once
	testEnc *Encoder
	encErr  error
)

func loadEncoder(t testing.TB) *Encoder {
	t.Helper()
	dev := device(t)
	encOnce.Do(func() {
		t0 := time.Now()
		testEnc, encErr = LoadEncoder(dev, modelDir(), 1024)
		if encErr == nil {
			t.Logf("staged the encoder in %v", time.Since(t0))
		}
	})
	if encErr != nil {
		t.Skipf("no checkpoint: %v", encErr)
	}
	return testEnc
}

// TestEncoder runs every text segment of every case's branches through the
// encoder and the projection, and holds the encoder's normed output and the
// projected rows (the backbone's prompt) to the dump.
func TestEncoder(t *testing.T) {
	e := loadEncoder(t)
	for _, name := range []string{"design", "plain", "zh", "long", "clone", "direct"} {
		ref := refCase(t, name)
		for _, br := range []string{"cond", "uncond"} {
			if ref[br+"_embeds"] == nil {
				continue
			}
			emb := f32(t, ref[br+"_embeds"])
			mask := f32(t, ref[br+"_text_mask"])
			var textRows []int
			for i, m := range mask {
				if m != 0 {
					textRows = append(textRows, i)
				}
			}
			at := 0
			for s := 0; ref[fmt.Sprintf("%s_seg%d_ids", br, s)] != nil; s++ {
				idf := f32(t, ref[fmt.Sprintf("%s_seg%d_ids", br, s)])
				ids := make([]int32, len(idf))
				for i, v := range idf {
					ids[i] = int32(v)
				}
				var states []float32
				out, took, err := e.Encode(ids, &states)
				if err != nil {
					t.Fatal(err)
				}
				st := compare(states, f32(t, ref[fmt.Sprintf("%s_seg%d_enc", br, s)]), 0)
				var want []float32
				for i := range ids {
					r := textRows[at+i]
					want = append(want, emb[r*bbHidden:(r+1)*bbHidden]...)
				}
				at += len(ids)
				pj := compare(out, want, 0)
				t.Logf("%s/%s seg %d: %d tokens in %v: encoder rel %.2e (max %.2e), projected rel %.2e (max %.2e)",
					name, br, s, len(ids), took.Round(time.Microsecond), st.rel, st.maxAbs, pj.rel, pj.maxAbs)
				if st.rel > encTol || pj.rel > encTol {
					t.Errorf("%s/%s seg %d: past rel %.0e", name, br, s, encTol)
				}
			}
		}
	}
}
