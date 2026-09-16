package llm

import (
	"math/rand"
	"testing"

	"strix-halo-vulkan/safetensors"
)

// q8Source builds an [n, k] matrix of genuine Q8_0 values: every group of 32
// has an fp16 scale and integer levels, with one level at 127 because that is
// what d = amax/127 forces. It returns the matrix and the (d, q) it was made
// from, so a test can say what the bank ought to hold without re-deriving it.
func q8Source(rng *rand.Rand, n, k int) (src []float32, ds []float32, qs []int8) {
	src = make([]float32, n*k)
	ds = make([]float32, n*k/q8Group)
	qs = make([]int8, n*k)
	for g := range ds {
		// A scale that is exactly representable as fp16, as the checkpoint's is.
		d := safetensors.F16ToF32(safetensors.F32ToF16(float32(rng.NormFloat64()) * 1e-3))
		if d < 0 {
			d = -d
		}
		ds[g] = d
		peak := rng.Intn(q8Group)
		for j := 0; j < q8Group; j++ {
			q := int8(rng.Intn(255) - 127)
			if j == peak {
				q = 127
				if rng.Intn(2) == 0 {
					q = -127
				}
			}
			qs[g*q8Group+j] = q
			src[g*q8Group+j] = float32(q) * d
		}
	}
	return src, ds, qs
}

// TestBankQ8RoundTrip is the identity L8 rests on: re-deriving (d, q) from a
// dequantised Q8_0 block returns the checkpoint's own pair, because ggml
// chooses d = amax/127 and so the block's largest level is always 127.
func TestBankQ8RoundTrip(t *testing.T) {
	const n, k = 48, 256
	rng := rand.New(rand.NewSource(7))
	src, ds, qs := q8Source(rng, n, k)

	dstQ := make([]byte, n*k)
	dstS := make([]uint16, n*k/q8Group)
	tileBQ8(dstQ, dstS, src, n, k, func(i int) int { return i })

	const tile = coopMatTile
	kt, kg := k/tile, k/q8Group
	for i := 0; i < n; i++ {
		for c := 0; c < k; c++ {
			got := int8(dstQ[(i/tile)*kt*tile*tile+(c/tile)*tile*tile+(i%tile)*tile+c%tile])
			if want := qs[i*k+c]; got != want {
				t.Fatalf("row %d col %d: level %d, want %d", i, c, got, want)
			}
		}
		for g := 0; g < kg; g++ {
			got := safetensors.F16ToF32(dstS[(i/tile)*kg*tile+g*tile+i%tile])
			if want := ds[i*k/q8Group+g]; got != want {
				t.Fatalf("row %d group %d: scale %g, want %g", i, g, got, want)
			}
		}
	}
}

// TestBankQ8IsTheHalves is the claim the kernel's -DQ8B arm is built on: the
// halves it puts in LDS are the halves the fp16 arm loads out of the bank,
// exactly, tile for tile and element for element. It is an equality and not a
// tolerance — see the note at the top of bank.go.
func TestBankQ8IsTheHalves(t *testing.T) {
	const n, k = 64, 128
	rng := rand.New(rand.NewSource(11))
	src, _, _ := q8Source(rng, n, k)

	// The same permutation packUpB uses, so the row remapping is covered too.
	row := func(i int) int { return (i%4)*16 + i/4 }

	half := make([]uint16, n*k)
	tileB(half, src, n, k, row)

	dstQ := make([]byte, n*k)
	dstS := make([]uint16, n*k/q8Group)
	tileBQ8(dstQ, dstS, src, n, k, row)

	const tile = coopMatTile
	kt, kg := k/tile, k/q8Group
	for nt := 0; nt < n/tile; nt++ {
		for ktile := 0; ktile < kt; ktile++ {
			for e := 0; e < tile*tile; e++ {
				at := (nt*kt+ktile)*tile*tile + e
				q := int8(dstQ[at])
				// A 32-element group is two whole k-tiles, so one scale
				// serves the tile: the kernel reads it once per four bytes.
				d := safetensors.F16ToF32(dstS[nt*kg*tile+(ktile/2)*tile+e/tile])
				got := safetensors.F32ToF16(float32(q) * d)
				if got != half[at] {
					t.Fatalf("tile (%d,%d) element %d: q8 gives %g, the fp16 bank holds %g",
						nt, ktile, e, safetensors.F16ToF32(got), safetensors.F16ToF32(half[at]))
				}
			}
		}
	}
}

// TestBankQ8Bytes states the staged size the kernel's derived scale offset
// depends on: n*k bytes of tiles and then the plane, nothing between them.
func TestBankQ8Bytes(t *testing.T) {
	if got, want := q8Bytes(248320, 2560), 248320*2560+248320*2560/32*2; got != want {
		t.Fatalf("q8Bytes = %d, want %d", got, want)
	}
	if q8Bytes(16, 32) != 16*32+16*2 {
		t.Fatalf("q8Bytes(16,32) = %d", q8Bytes(16, 32))
	}
}
