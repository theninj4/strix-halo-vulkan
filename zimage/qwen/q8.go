package qwen

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

// Bank is how a model's projection weights are held on the device.
type Bank int

const (
	// BankFP16 is every projection in fp16 fragment tiles, read by the GEMMs
	// directly.
	BankFP16 Bank = iota
	// BankQ8 is every projection as int8 with one fp16 scale per 32 k of a
	// row (PackQ8), expanded into one fp16 scratch a layer at a time
	// (shaders/dit_dequant_q8.comp) just before that layer's GEMMs. Half
	// the memory, the same GEMMs, and ~1 ms a GB of weights on the device
	// (VIDEO.md M11a).
	BankQ8
)

func (b Bank) String() string {
	if b == BankQ8 {
		return "q8"
	}
	return "fp16"
}

// ParseBank reads "fp16" or "q8".
func ParseBank(s string) (Bank, error) {
	switch s {
	case "fp16":
		return BankFP16, nil
	case "q8", "int8":
		return BankQ8, nil
	}
	return 0, fmt.Errorf("qwen: bank %q; fp16 or q8", s)
}

// Q8Group is how many k of a row share one scale.
const Q8Group = 32

// Q8Bytes is how many bytes PackQ8 writes for an [n, k] weight: the int8
// tiles, then the scale plane, rounded up to 256 so the next matrix starts
// on a word and a tile.
func Q8Bytes(n, k int) int { return (n*k + n*k/Q8Group*2 + 255) &^ 255 }

// PackQ8 quantises a row-major [n, k] weight into dst (Q8Bytes(n, k) long):
// int8 in the fragment tiling -- tile (nt, kt) is 256 bytes holding (k, n)
// at (n%16)*16 + k%16, tiles kt-fastest -- then the fp16 scales, scale
// (nt, group, n%16) at (nt*k/32 + group)*16 + n%16. It is Kev's K7.1
// quantisation (kev.packQ8) and the LLM's L8 layout: q is rounded against
// the fp16 scale the expander multiplies by, amax/127 narrowed.
func PackQ8(dst []byte, w []float32, n, k int) error {
	if n%coopMatTile != 0 || k%64 != 0 {
		return fmt.Errorf("qwen: [%d %d] does not tile for the int8 bank", n, k)
	}
	if len(w) != n*k || len(dst) < Q8Bytes(n, k) {
		return fmt.Errorf("qwen: packing %d values into [%d %d] (%d bytes)", len(w), n, k, len(dst))
	}
	const tile = coopMatTile
	kt, kg := k/tile, k/Q8Group
	sc := dst[n*k:]
	parallelFor((n+packChunk-1)/packChunk, func(c int) {
		for r := c * packChunk; r < min((c+1)*packChunk, n); r++ {
			base := (r / tile) * kt * tile * tile
			lane := (r % tile) * tile
			sbase := (r/tile)*kg*tile + r%tile
			x := w[r*k : (r+1)*k]
			for gi := 0; gi < kg; gi++ {
				blk := x[gi*Q8Group : (gi+1)*Q8Group]
				var amax float32
				for _, v := range blk {
					amax = max(amax, float32(math.Abs(float64(v))))
				}
				dh := safetensors.F32ToF16(amax / 127)
				d := safetensors.F16ToF32(dh)
				si := sbase + gi*tile
				sc[2*si], sc[2*si+1] = byte(dh), byte(dh>>8)
				for j, v := range blk {
					var qi int32
					if d != 0 {
						qi = int32(math.Round(float64(v / d)))
					}
					qi = min(max(qi, -127), 127)
					col := gi*Q8Group + j
					dst[base+(col/tile)*tile*tile+lane+col%tile] = byte(int8(qi))
				}
			}
		}
	})
	return nil
}

// DequantQ8 is the expander's arithmetic on the host: PackQ8's bytes back
// to fp16 halves in the same fragment tiling, float(q)·float(scale) rounded
// once. The device's halves equal these bit for bit.
func DequantQ8(src []byte, n, k int) []uint16 {
	const tile = coopMatTile
	kt, kg := k/tile, k/Q8Group
	out := make([]uint16, n*k)
	sc := src[n*k:]
	// One tile row (16 output rows) a work item.
	parallelFor(n/tile, func(nt int) {
		for i := nt * kt * tile * tile; i < (nt+1)*kt*tile*tile; i++ {
			in := i % (tile * tile)
			ktile := i/(tile*tile) - nt*kt
			si := (nt*kg+ktile/2)*tile + in/tile
			d := safetensors.F16ToF32(uint16(sc[2*si]) | uint16(sc[2*si+1])<<8)
			out[i] = safetensors.F32ToF16(float32(int8(src[i])) * d)
		}
	})
	return out
}

// Q8Dispatch is one matrix's expansion: the PackQ8 bytes at byte qOff of
// the int8 bank the pipeline binds at 2, into halves at hOff of the fp16
// scratch it binds at 3. The pipeline is shaders.DiTDequantQ8, built with
// the dit_common push-constant block (88 bytes).
func Q8Dispatch(pipe *vk.ComputePipeline, qOff, hOff uint32, n, k int) vk.MultiDispatch {
	pc := pushConstants{
		InOff: qOff / 4, WOff: (qOff + uint32(n*k)) / 2, OutOff: hOff,
		GemmN: uint32(n), GemmK: uint32(k),
	}
	return vk.MultiDispatch{Pipeline: pipe, GroupsX: uint32(k / 64), GroupsY: uint32(n / coopMatTile), PushConstants: pc.bytes()}
}
