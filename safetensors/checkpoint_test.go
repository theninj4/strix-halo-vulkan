package safetensors

import (
	"math"
	"os"
	"testing"
)

// checkpointDir is the local Z-Image-Turbo checkout. These tests are skipped
// when it is absent so the package still tests on a machine without 30 GB of
// weights checked out.
const checkpointDir = "../models/Z-Image-Turbo"

// TestRealCheckpointValues pins a few tensors against values read by an
// independent CPython implementation of the format (struct-based, no
// safetensors library). The point is the offset arithmetic: these cover a
// sharded F32 checkpoint, a sharded BF16 one, and a single-file BF16 one,
// and a tensor that lives in a shard other than the first.
//
// The tensors are deliberately small. The same cross-check was run against
// layers.0.attention.to_q.weight (14.7 M elements, sum -805.15519) and
// layers.29.feed_forward.w2.weight (39.3 M, sum 759.369712) and matched to
// every digit, but reading 24 GB does not belong in a unit test.
func TestRealCheckpointValues(t *testing.T) {
	if _, err := os.Stat(checkpointDir); err != nil {
		t.Skipf("no checkpoint at %s", checkpointDir)
	}
	cases := []struct {
		dir, name        string
		dtype            DType
		shape            []int
		first, last, sum float64
	}{
		{
			dir: checkpointDir + "/transformer", name: "cap_pad_token",
			dtype: F32, shape: []int{1, 3840},
			first: -0.0732421875, last: 0.0280761719, sum: 12.381859,
		},
		{
			dir: checkpointDir + "/vae", name: "decoder.conv_in.weight",
			dtype: BF16, shape: []int{512, 16, 3, 3},
			first: -0.00823974609, last: -0.0139770508, sum: -16.1314038,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			set, err := OpenSet(c.dir)
			if err != nil {
				t.Fatal(err)
			}
			defer set.Close()

			ten, err := set.Get(c.name)
			if err != nil {
				t.Fatal(err)
			}
			if ten.DType != c.dtype {
				t.Errorf("dtype = %s, want %s", ten.DType, c.dtype)
			}
			if len(ten.Shape) != len(c.shape) {
				t.Fatalf("shape = %v, want %v", ten.Shape, c.shape)
			}
			for i := range c.shape {
				if ten.Shape[i] != c.shape[i] {
					t.Fatalf("shape = %v, want %v", ten.Shape, c.shape)
				}
			}
			v, err := ten.F32(nil)
			if err != nil {
				t.Fatal(err)
			}
			var sum float64
			for _, x := range v {
				sum += float64(x)
			}
			const eps = 1e-6
			if math.Abs(float64(v[0])-c.first) > eps {
				t.Errorf("first = %.9g, want %.9g", v[0], c.first)
			}
			if math.Abs(float64(v[len(v)-1])-c.last) > eps {
				t.Errorf("last = %.9g, want %.9g", v[len(v)-1], c.last)
			}
			if math.Abs(sum-c.sum) > 1e-4*math.Abs(c.sum) {
				t.Errorf("sum = %.9g, want %.9g", sum, c.sum)
			}
		})
	}
}

// TestRealCheckpointInventory guards the facts the pipeline is being built
// against, which bench/modelshapes.go got wrong by transcribing config.json
// rather than reading the weights: the Z-Image DiT has 34 attention blocks,
// not 30, because two context_refiner and two noise_refiner layers sit
// alongside the 30 `layers`.
func TestRealCheckpointInventory(t *testing.T) {
	if _, err := os.Stat(checkpointDir); err != nil {
		t.Skipf("no checkpoint at %s", checkpointDir)
	}
	set, err := OpenSet(checkpointDir + "/transformer")
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()

	if got, want := set.Len(), 521; got != want {
		t.Errorf("tensors = %d, want %d", got, want)
	}
	if got := set.Params(); got < 6.15e9 || got > 6.16e9 {
		t.Errorf("params = %d, want ~6.155 B", got)
	}
	blocks := 0
	for _, prefix := range []string{"layers", "context_refiner", "noise_refiner"} {
		for i := 0; ; i++ {
			if !set.Has(prefixName(prefix, i)) {
				break
			}
			blocks++
		}
	}
	if want := 34; blocks != want {
		t.Errorf("attention blocks = %d, want %d (30 layers + 2 context_refiner + 2 noise_refiner)", blocks, want)
	}
}

func prefixName(prefix string, i int) string {
	return prefix + "." + itoa(i) + ".attention.to_q.weight"
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [8]byte
	n := len(b)
	for i > 0 {
		n--
		b[n] = byte('0' + i%10)
		i /= 10
	}
	return string(b[n:])
}
