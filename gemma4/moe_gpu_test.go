package gemma4

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"time"
	"math/rand/v2"
	"sort"
	"testing"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("gemma4-test")
	if err != nil {
		t.Skipf("no Vulkan instance: %v", err)
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		t.Skipf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		t.Skipf("no compute queue: %v", err)
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Skipf("no subgroup size control: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

func geluTanh(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(0.7978845608028654*(x+0.044715*x*x*x)))
}

func f16(v float32) float32 { return safetensors.F16ToF32(safetensors.F32ToF16(v)) }

// TestMoEBlockGemmaShapes is R5's first block gate: the LLM's MoE block
// (llm.MoEGPU) at Rune's shapes (hidden 2816, 128 experts, top 8, expert
// 704), Q8_0 banks, GELU-tanh on the gate (WithMoEGELU), and the shared
// expert slot filled by a 64-wide expert whose down is zero. Against a
// float64 CPU reference over the same fp16 input, fp16 router and dequantised
// banks: the routing must agree wherever the reference's 8th and 9th logits
// are not a near-tie, and the output to fp16-intermediate accuracy.
func TestMoEBlockGemmaShapes(t *testing.T) {
	if testing.Short() {
		t.Skip("GPU")
	}
	dev, done := newTestDevice(t)
	defer done()

	const H, E, K, F, S, T = 2816, 128, 8, 704, 64, 100
	r := rand.New(rand.NewPCG(7, 8))
	randn := func(n int, s float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(r.NormFloat64() * s)
		}
		return v
	}
	q8 := func(name string, rows []float32, in, out, exps int) *gguf.Tensor {
		var data []byte
		for i := 0; i < len(rows); i += in {
			data = quantQ8_0(data, rows[i:i+in])
		}
		dims := []int64{int64(in), int64(out)}
		if exps > 0 {
			dims = append(dims, int64(exps))
		}
		return &gguf.Tensor{Name: name, Type: gguf.Q8_0, Dims: dims, Data: data}
	}
	gate := randn(E*F*H, 0.02)
	up := randn(E*F*H, 0.02)
	down := randn(E*H*F, 0.02)
	router := randn(E*H, 0.02)
	w := llm.MoEWeights{
		Router:     router,
		SharedGate: make([]float32, H),
		Gate:       &llm.ExpertBank{T: q8("gate", gate, H, F, E), In: H, Out: F, NExp: E},
		Up:         &llm.ExpertBank{T: q8("up", up, H, F, E), In: H, Out: F, NExp: E},
		Down:       &llm.ExpertBank{T: q8("down", down, F, H, E), In: F, Out: H, NExp: E},
		GateShexpT: q8("shgate", make([]float32, S*H), H, S, 0),
		UpShexpT:   q8("shup", make([]float32, S*H), H, S, 0),
		DownShexpT: q8("shdown", make([]float32, H*S), S, H, 0),
	}
	cfg := llm.MoEConfig{NEmbd: H, NExpert: E, NExpertUsed: K, FFNExpert: F, FFNShared: S}
	g, err := llm.NewMoEGPU(dev, cfg, T, []llm.MoEWeights{w}, llm.WithMoEGELU())
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	g.PinGemv(true)
	x := randn(T*H, 1)
	if err := g.Upload(x, T); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	out := g.Out()
	topk := g.TopK()

	deq := func(tn *gguf.Tensor) []float32 {
		v, err := tn.Dequantize(nil)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	gq, uq, dq := deq(w.Gate.T), deq(w.Up.T), deq(w.Down.T)
	var worst, rmsErr, rmsRef float64
	ties, mism := 0, 0
	for tk := range T {
		xn := make([]float64, H)
		for i := range xn {
			xn[i] = float64(f16(x[tk*H+i]))
		}
		logits := make([]float64, E)
		for e := range E {
			s := 0.0
			for i := range H {
				s += float64(f16(router[e*H+i])) * xn[i]
			}
			logits[e] = s
		}
		idx := make([]int, E)
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
		if logits[idx[K-1]]-logits[idx[K]] < 1e-3 {
			ties++
			continue // a near-tie at the cut; fp16 rounding may pick either
		}
		got := map[int]bool{}
		for k := range K {
			got[int(topk[tk*K+k])] = true
		}
		for k := range K {
			if !got[idx[k]] {
				mism++
				t.Errorf("token %d: expert %d (logit %.5f) not chosen", tk, idx[k], logits[idx[k]])
			}
		}
		mx := logits[idx[0]]
		den := 0.0
		for _, l := range logits {
			den += math.Exp(l - mx)
		}
		sel := 0.0
		for k := range K {
			sel += math.Exp(logits[idx[k]]-mx) / den
		}
		ref := make([]float64, H)
		for k := range K {
			e := idx[k]
			wt := math.Exp(logits[e]-mx) / den / sel
			h := make([]float64, F)
			for j := range F {
				var a, b float64
				gr, ur := gq[(e*F+j)*H:(e*F+j+1)*H], uq[(e*F+j)*H:(e*F+j+1)*H]
				for i := range H {
					a += float64(gr[i]) * xn[i]
					b += float64(ur[i]) * xn[i]
				}
				h[j] = float64(f16(float32(geluTanh(a) * b)))
			}
			for j := range H {
				dr := dq[(e*H+j)*F : (e*H+j+1)*F]
				s := 0.0
				for i := range F {
					s += float64(dr[i]) * h[i]
				}
				ref[j] += wt * s
			}
		}
		for j := range H {
			d := float64(out[tk*H+j]) - ref[j]
			worst = max(worst, math.Abs(d))
			rmsErr += d * d
			rmsRef += ref[j] * ref[j]
		}
	}
	rel := math.Sqrt(rmsErr / rmsRef)
	t.Logf("%d tokens, %d near-ties skipped, %d routing mismatches; rel rms err %.2e, worst |d| %.2e (ref rms %.3f)",
		T, ties, mism, rel, worst, math.Sqrt(rmsRef/float64((T-ties)*H)))
	if rel > 2e-3 {
		t.Errorf("relative rms error %.2e", rel)
	}
}

// TestMoEGemmaTiming times one Rune-shaped MoE layer (random Q8_0 experts,
// GELU, no shared expert) at RUNE_MOE_T rows (default 1024, a batched pass):
// the harness for screening the grouped GEMM's Q8_0 builds through
// LLM_MOE_SPV without staging the model.
func TestMoEGemmaTiming(t *testing.T) {
	if testing.Short() {
		t.Skip("GPU")
	}
	T := 1024
	if s := os.Getenv("RUNE_MOE_T"); s != "" {
		fmt.Sscan(s, &T)
	}
	dev, done := newTestDevice(t)
	defer done()
	const H, E, K, F, S = 2816, 128, 8, 704, 64
	r := rand.New(rand.NewPCG(7, 8))
	q8 := func(in, rows int) []byte {
		row := make([]float32, in)
		data := make([]byte, 0, rows*in/32*34)
		for range rows {
			for i := range row {
				row[i] = float32(r.NormFloat64() * 0.02)
			}
			data = quantQ8_0(data, row)
		}
		return data
	}
	bank := func(in, out int) *llm.ExpertBank {
		return &llm.ExpertBank{T: &gguf.Tensor{Type: gguf.Q8_0, Dims: []int64{int64(in), int64(out), E}, Data: q8(in, out*E)},
			In: in, Out: out, NExp: E}
	}
	zero := func(in, out int) *gguf.Tensor {
		return &gguf.Tensor{Type: gguf.Q8_0, Dims: []int64{int64(in), int64(out)}, Data: make([]byte, in*out/32*34)}
	}
	router := make([]float32, E*H)
	for i := range router {
		router[i] = float32(r.NormFloat64() * 0.02)
	}
	w := llm.MoEWeights{Router: router, SharedGate: make([]float32, H),
		Gate: bank(H, F), Up: bank(H, F), Down: bank(F, H),
		GateShexpT: zero(H, S), UpShexpT: zero(H, S), DownShexpT: zero(S, H)}
	if os.Getenv("RUNE_MOE_TILED") == "1" {
		// The same values in Q8_TILED's layout, for builds compiled with it.
		for _, b := range []*llm.ExpertBank{w.Gate, w.Up, w.Down} {
			b.T.Data = tileQ8(b.T.Data, b.Out*E, b.In)
		}
	}
	cfg := llm.MoEConfig{NEmbd: H, NExpert: E, NExpertUsed: K, FFNExpert: F, FFNShared: S}
	opts := []llm.MoEOption{llm.WithMoEGELU(), llm.WithMoENoShared()}
	if os.Getenv("RUNE_MOE_TILED") == "1" {
		opts = append(opts, llm.WithMoEQ8Tiled())
	}
	g, err := llm.NewMoEGPU(dev, cfg, T, []llm.MoEWeights{w}, opts...)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	g.PinGemv(true)
	x := make([]float32, T*H)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	if err := g.Upload(x, T); err != nil {
		t.Fatal(err)
	}
	for range 3 {
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
	}
	ref := append([]float32(nil), g.Out()[:T*H]...)
	st, err := g.Profile(0, 20)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range st {
		if s.GPU > 50*time.Microsecond {
			t.Logf("T=%d %-14s %8.3f ms a layer", T, s.Kind, float64(s.GPU.Microseconds())/1000)
		}
	}
	if path := os.Getenv("RUNE_MOE_REF"); path != "" {
		// The bit-identity check across builds: the first run writes the
		// output, later ones compare.
		if old, err := os.ReadFile(path); err == nil {
			diff := 0
			for i := range ref {
				if math.Float32bits(ref[i]) != binary.LittleEndian.Uint32(old[4*i:]) {
					diff++
				}
			}
			t.Logf("against %s: %d of %d values differ", path, diff, len(ref))
			if diff > 0 {
				t.Errorf("not bit-identical")
			}
		} else {
			buf := make([]byte, 4*len(ref))
			for i, v := range ref {
				binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
			}
			os.WriteFile(path, buf, 0o644)
		}
	}
}
