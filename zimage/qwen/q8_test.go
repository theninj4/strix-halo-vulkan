package qwen

import (
	"math"
	"math/rand/v2"
	"testing"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// TestPackQ8 checks the quantisation on the host: every value comes back
// within half a step of its group's scale, and a group of zeros is zeros.
func TestPackQ8(t *testing.T) {
	n, k := 48, 192
	rng := rand.New(rand.NewPCG(1, 2))
	w := make([]float32, n*k)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * float32(math.Pow(10, float64(i%5-2)))
	}
	for j := 0; j < Q8Group; j++ {
		w[5*k+64+j] = 0
	}
	buf := make([]byte, Q8Bytes(n, k))
	if err := PackQ8(buf, w, n, k); err != nil {
		t.Fatal(err)
	}
	got := DequantQ8(buf, n, k)
	for r := 0; r < n; r++ {
		for gi := 0; gi < k/Q8Group; gi++ {
			var amax float64
			for j := 0; j < Q8Group; j++ {
				amax = max(amax, math.Abs(float64(w[r*k+gi*Q8Group+j])))
			}
			for j := 0; j < Q8Group; j++ {
				c := gi*Q8Group + j
				i := (r/16*(k/16)+c/16)*256 + r%16*16 + c%16
				v := float64(safetensors.F16ToF32(got[i]))
				// Half a step of the fp16 scale, plus fp16's own rounding
				// of the product.
				tol := amax/127*0.5*1.001 + math.Abs(v)*1e-3
				if d := math.Abs(v - float64(w[r*k+c])); d > tol {
					t.Fatalf("(%d, %d): %g back as %g, off by %g > %g", r, c, w[r*k+c], v, d, tol)
				}
			}
		}
	}
}

// TestGPUDequantQ8 holds the device's expansion to the host's, bit for bit,
// at two offsets in one bank and at a destination offset.
func TestGPUDequantQ8(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	shapes := [][2]int{{64, 320}, {1024, 5120}}
	rng := rand.New(rand.NewPCG(3, 4))
	var packed [][]byte
	var offs []int
	total := 0
	for _, sh := range shapes {
		w := make([]float32, sh[0]*sh[1])
		for i := range w {
			w[i] = float32(rng.NormFloat64() * 0.02)
		}
		buf := make([]byte, Q8Bytes(sh[0], sh[1]))
		if err := PackQ8(buf, w, sh[0], sh[1]); err != nil {
			t.Fatal(err)
		}
		packed = append(packed, buf)
		offs = append(offs, total)
		total += len(buf)
	}
	qb, err := dev.NewBuffer(total)
	if err != nil {
		t.Fatal(err)
	}
	defer qb.Destroy()
	const hOff = 512 // halves
	out, err := dev.NewBuffer((hOff + 1024*5120) * 2)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Destroy()
	dummy, err := dev.NewBuffer(256)
	if err != nil {
		t.Fatal(err)
	}
	defer dummy.Destroy()
	for i, b := range packed {
		qb.WriteBytesAt(offs[i], b)
	}
	mod, err := dev.NewShaderModule(shaders.DiTDequantQ8)
	if err != nil {
		t.Fatal(err)
	}
	defer mod.Destroy()
	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{dummy, dummy, qb, out},
		PushConstantSize: uint32(unsafe.Sizeof(pushConstants{})),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Destroy()
	for i, sh := range shapes {
		n, k := sh[0], sh[1]
		out.ZeroUint16At(0, hOff+n*k)
		if _, err := vk.DispatchMultiTimed([]vk.MultiDispatch{Q8Dispatch(pipe, uint32(offs[i]), hOff, n, k)}, 1, 1, true); err != nil {
			t.Fatal(err)
		}
		got := out.ReadUint16At(hOff, n*k)
		want := DequantQ8(packed[i], n, k)
		bad := 0
		for j := range want {
			if got[j] != want[j] {
				if bad < 5 {
					t.Errorf("[%d %d] half %d: device %#04x, host %#04x", n, k, j, got[j], want[j])
				}
				bad++
			}
		}
		if bad > 0 {
			t.Fatalf("[%d %d]: %d of %d halves differ", n, k, bad, n*k)
		}
		if lead := out.ReadUint16At(0, hOff); lead[hOff-1] != 0 {
			t.Fatalf("wrote before its offset")
		}
	}
}
