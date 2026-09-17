package llm

import (
	"math"
	"math/rand"
	"testing"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/safetensors"
)

// q8Tensor builds a [k, n] Q8_0 tensor out of thin air: n rows of k values,
// each 32-element block an fp16 scale and 32 levels, exactly as the
// checkpoint stores `output.weight`. It is a tensor and not a fixture because
// the head is 675 MB and the claim under test — that the int8 bank and the
// fp16 bank are the same numbers — is a property of the format, not of this
// model's values.
func q8Tensor(rng *rand.Rand, name string, k, n int) *gguf.Tensor {
	blocks := n * k / 32
	data := make([]byte, blocks*34)
	for b := 0; b < blocks; b++ {
		d := float32(math.Abs(rng.NormFloat64())) * 1e-2
		h := safetensors.F32ToF16(d)
		data[b*34] = byte(h)
		data[b*34+1] = byte(h >> 8)
		peak := rng.Intn(32)
		for j := 0; j < 32; j++ {
			q := int8(rng.Intn(255) - 127)
			if j == peak {
				q = 127
			}
			data[b*34+2+j] = byte(q)
		}
	}
	return &gguf.Tensor{Name: name, Type: gguf.Q8_0,
		Dims: []int64{int64(k), int64(n)}, Data: data}
}

// TestHeadGPUQ8IsTheHalves runs the head twice over the same weight — once
// with L8's int8 bank and once with the fp16 tiling it replaces — and demands
// the **same logits, bit for bit**.
//
// That is a stronger gate than a tolerance and it is the right one: the two
// banks hold the same halves (TestBankQ8IsTheHalves), the kernel puts them in
// LDS instead of reading them from global, and a cooperative-matrix multiply
// over identical operands in the same order is identical. Anything less than
// equality here would mean the unpack had moved a value, not that the format
// had cost accuracy.
func TestHeadGPUQ8IsTheHalves(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()

	const nEmbd, vocab, rows = 256, 640, 4
	rng := rand.New(rand.NewSource(3))
	w := q8Tensor(rng, "output.weight", nEmbd, vocab)

	x := make([]float32, rows*nEmbd)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	run := func(q8 bool) []float32 {
		t.Helper()
		g, err := NewHeadGPU(dev, nEmbd, w, rows, q8)
		if err != nil {
			t.Fatalf("head (q8=%v): %v", q8, err)
		}
		defer g.Destroy()
		if err := g.Upload(x, rows); err != nil {
			t.Fatalf("upload (q8=%v): %v", q8, err)
		}
		if err := g.Run(); err != nil {
			t.Fatalf("run (q8=%v): %v", q8, err)
		}
		out := append([]float32(nil), g.Logits()...)
		if got, want := g.WeightBytes(), vocab*nEmbd*2; q8 && got >= want {
			t.Fatalf("q8 bank is %d bytes, the fp16 one is %d", got, want)
		}
		return out
	}

	half, q8 := run(false), run(true)
	if len(half) != len(q8) {
		t.Fatalf("%d logits against %d", len(q8), len(half))
	}
	for i := range half {
		if half[i] != q8[i] {
			t.Fatalf("logit %d: q8 %.9g, fp16 %.9g", i, q8[i], half[i])
		}
	}
}

// TestHeadGPUQ8BankSize states the bank L8 stages against the one it
// replaces: 8.5 bits a weight rather than 16, which for this model's
// [2560, 248320] head is 0.68 GB against 1.27.
func TestHeadGPUQ8BankSize(t *testing.T) {
	const nEmbd, vocab = 2560, 248320
	q8, half := q8Bytes(vocab, nEmbd), vocab*nEmbd*2
	if ratio := float64(half) / float64(q8); ratio < 1.87 || ratio > 1.89 {
		t.Fatalf("the q8 bank is %.3fx smaller, want 16/8.5", ratio)
	}
}

// TestHeadGPUQ4IsTheSim is L8c-4's gate, and it is the same shape as the one
// above: the 4.5-bit bank against **the simulation that chose it**, bit for
// bit.
//
// L8c-3's +4.24% is a perplexity measured through `sim.go`, which stages a
// candidate format's floats and lets the fp16 kernels multiply them. This
// bank stores that format instead. Both encode through `quantk.go`, the value
// is `d*sc*l - dmin*m` in f32 either way, and the shader's LDS store rounds
// it to the same half the host's `tileB` writes — so the two have to agree
// exactly. A tolerance would hide the only kind of bug a new bank can have,
// which is an address.
func TestHeadGPUQ4IsTheSim(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()

	const nEmbd, vocab, rows = 256, 640, 4
	rng := rand.New(rand.NewSource(11))
	w := q8Tensor(rng, "output.weight", nEmbd, vocab)
	sim, err := ParseQuantSim("q4_k/32")
	if err != nil {
		t.Fatal(err)
	}
	sim.Mode = "rtn"

	x := make([]float32, rows*nEmbd)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	run := func(bank DenseBank) ([]float32, int) {
		t.Helper()
		g, err := NewHeadGPUBank(dev, nEmbd, w, rows, bank, sim)
		if err != nil {
			t.Fatalf("head (%s): %v", bank, err)
		}
		defer g.Destroy()
		if err := g.Upload(x, rows); err != nil {
			t.Fatalf("upload (%s): %v", bank, err)
		}
		if err := g.Run(); err != nil {
			t.Fatalf("run (%s): %v", bank, err)
		}
		return append([]float32(nil), g.Logits()...), g.WeightBytes()
	}

	simLogits, simBytes := run(BankFP16)
	q4, q4Bytes := run(BankQ4K)
	if bits := float64(q4Bytes) * 8 / float64(vocab*nEmbd); bits != 4.5 {
		t.Fatalf("the bank is %.4f bits a weight, want 4.5", bits)
	}
	if simBytes != vocab*nEmbd*2 {
		t.Fatalf("the simulation's bank is %d bytes, want halves", simBytes)
	}
	for i := range simLogits {
		if simLogits[i] != q4[i] {
			t.Fatalf("logit %d: bank %.9g, simulation %.9g", i, q4[i], simLogits[i])
		}
	}
}

// TestHeadGPUQ4Gemv prices the decode kernel against the GEMM on the same
// bank, and pins the refusal that keeps it honest.
//
// It is **not** an equality and cannot be, for the reason llm_moe_gemv.comp
// gives about the expert bank: a K-quant group is affine, the GEMV forms
// `d*sc*l - dmin*m` in a register and multiplies it by the fp16 activation in
// f32, where the GEMM rounds that same value to a half on its way into LDS.
// So the GEMV carries one *fewer* rounding per weight, and what the test
// states is how far apart that puts them.
func TestHeadGPUQ4Gemv(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()

	const nEmbd, vocab = 256, 640
	rng := rand.New(rand.NewSource(13))
	w := q8Tensor(rng, "output.weight", nEmbd, vocab)
	sim, err := ParseQuantSim("q4_k/32")
	if err != nil {
		t.Fatal(err)
	}
	sim.Mode = "rtn"

	g, err := NewHeadGPUBank(dev, nEmbd, w, 4, BankQ4K, sim)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	x := make([]float32, nEmbd)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	if err := g.Upload(x, 1); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(); err != nil {
		t.Fatal(err)
	}
	gemm := append([]float32(nil), g.Logits()[:vocab]...)

	if err := g.SetGEMV(GEMVK1); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(); err != nil {
		t.Fatal(err)
	}
	gemv := append([]float32(nil), g.Logits()[:vocab]...)

	var num, den float64
	for i := range gemm {
		d := float64(gemm[i] - gemv[i])
		num += d * d
		den += float64(gemm[i]) * float64(gemm[i])
	}
	rel := math.Sqrt(num / den)
	if rel > 1e-3 {
		t.Fatalf("the GEMV is %.3g from the GEMM relative, want one rounding's worth", rel)
	}
	t.Logf("GEMV against GEMM on the q4_k bank: rms %.3g relative", rel)

	// D15: the GEMV reads one row of A, so more than one is refused rather
	// than silently dropped.
	if err := g.Upload(make([]float32, 2*nEmbd), 2); err == nil {
		t.Fatal("the GEMV took two rows")
	}
	// And there is no fp16 arm of it at all.
	h, err := NewHeadGPUBank(dev, nEmbd, w, 1, BankFP16, QuantSim{})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Destroy()
	if err := h.SetGEMV(GEMVK1); err == nil {
		t.Fatal("the fp16 bank offered a GEMV")
	}
}

// TestHeadGPUQ4BankSize states L8c-4's bank against the two it follows: 4.500
// bits a weight against the checkpoint's 8.5 and the halves' 16, which for
// this model's [2560, 248320] head is 0.36 GB against 0.68 and 1.27.
func TestHeadGPUQ4BankSize(t *testing.T) {
	const nEmbd, vocab = 2560, 248320
	q4, q8, half := q4kBytes(vocab, nEmbd), q8Bytes(vocab, nEmbd), vocab*nEmbd*2
	if bits := float64(q4) * 8 / float64(vocab*nEmbd); bits != 4.5 {
		t.Fatalf("%.4f bits a weight, want 4.500", bits)
	}
	if ratio := float64(q8) / float64(q4); ratio < 1.88 || ratio > 1.89 {
		t.Fatalf("the q4_k bank is %.3fx the q8 one, want 8.5/4.5", ratio)
	}
	if ratio := float64(half) / float64(q4); ratio < 3.55 || ratio > 3.56 {
		t.Fatalf("the q4_k bank is %.3fx the halves, want 16/4.5", ratio)
	}
}
