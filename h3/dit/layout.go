package dit

import (
	"errors"
	"fmt"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// errLayout is a device whose cooperative matrices hold their elements in
// another order than h3_attn_t.comp is written against.
var errLayout = errors.New("dit: cooperative-matrix element order is not the transposed attention's")

// coopMatLayout runs shaders/coopmat_layout_probe.comp and returns, per Use
// (A, B, fp32 accumulator, fp16 accumulator), per lane, the (row, column)
// each element index holds, as row*16 + column.
func coopMatLayout(dev *vk.Device) ([4][32][]int, error) {
	var lay [4][32][]int
	const slots = 32
	src32 := make([]float32, 256)
	src16 := make([]uint16, 256)
	for i := range src32 {
		src32[i] = float32(i)
		src16[i] = safetensors.F32ToF16(float32(i))
	}
	var bufs []*vk.Buffer
	defer func() {
		for _, b := range bufs {
			b.Destroy()
		}
	}()
	for _, size := range []int{256 * 4, 256 * 2, 4 * 32 * slots * 4} {
		b, err := dev.NewBuffer(size)
		if err != nil {
			return lay, err
		}
		bufs = append(bufs, b)
	}
	bufs[0].WriteFloat32(src32)
	bufs[1].WriteUint16At(0, src16)
	mod, err := dev.NewShaderModule(shaders.CoopMatLayoutProbe)
	if err != nil {
		return lay, err
	}
	defer mod.Destroy()
	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{Buffers: bufs, RequiredSubgroupSize: 32})
	if err != nil {
		return lay, err
	}
	defer pipe.Destroy()
	if _, err := vk.DispatchMultiTimed([]vk.MultiDispatch{{Pipeline: pipe, GroupsX: 1, GroupsY: 1}}, 1, 1, true); err != nil {
		return lay, err
	}
	got := bufs[2].ReadFloat32(4 * 32 * slots)
	for u := range lay {
		for l := range lay[u] {
			for e := 0; e < slots; e++ {
				v := got[(u*32+l)*slots+e]
				if v < 0 {
					break
				}
				lay[u][l] = append(lay[u][l], int(v))
			}
		}
	}
	return lay, nil
}

// checkCoopMatLayout says whether the device's element order is the one
// h3_attn_t.comp is written against (VIDEO.md M11c), on RADV/gfx1151 at
// wave32:
//
//   - A (fp16): lane l holds row l%16, element e its column e.
//   - B (fp16): lane l holds column l%16, element e its row e.
//   - accumulators (fp32 and fp16): lane l holds column l%16, element e
//     its row 2e + l/16.
//
// A different order wraps errLayout, and the build falls back to the plain
// kernel: the transposed one would then be wrong, not slow.
func checkCoopMatLayout(dev *vk.Device) error {
	lay, err := coopMatLayout(dev)
	if err != nil {
		return err
	}
	names := []string{"A", "B", "fp32 accumulator", "fp16 accumulator"}
	for l := 0; l < 32; l++ {
		for u, want := range []func(e int) int{
			func(e int) int { return (l%16)*16 + e },
			func(e int) int { return e*16 + l%16 },
			func(e int) int { return (2*e+l/16)*16 + l%16 },
			func(e int) int { return (2*e+l/16)*16 + l%16 },
		} {
			n := 16
			if u >= 2 {
				n = 8
			}
			if len(lay[u][l]) != n {
				return fmt.Errorf("%w: %s lane %d has %d elements, want %d", errLayout, names[u], l, len(lay[u][l]), n)
			}
			for e, v := range lay[u][l] {
				if v != want(e) {
					return fmt.Errorf("%w: %s lane %d element %d holds (%d, %d), want (%d, %d)", errLayout,
						names[u], l, e, v/16, v%16, want(e)/16, want(e)%16)
				}
			}
		}
	}
	return nil
}

// TransposedAttentionOK says whether h3_attn_t.comp's builds are right on dev
// (checkCoopMatLayout), for the other callers of that kernel (the video
// VAE's decoder). A false with a nil error is a device with another element
// order; an error is a probe that did not run.
func TransposedAttentionOK(dev *vk.Device) (bool, error) {
	err := checkCoopMatLayout(dev)
	if errors.Is(err, errLayout) {
		return false, nil
	}
	return err == nil, err
}
