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
