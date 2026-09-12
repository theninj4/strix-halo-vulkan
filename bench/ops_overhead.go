package bench

import (
	"fmt"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// overheadGroupCounts sweeps the dispatch size for a shader that does
// nothing, separating the fixed per-dispatch cost from the cost that scales
// with the number of workgroups scheduled.
var overheadGroupCounts = []int{1, 40, 640, 4096, 65536}

// RunOverhead measures the floor under every other number in this suite:
// the cost of launching one dispatch and the pipeline barrier
// bench.TimeDispatch places between iterations, with no actual work.
//
// This matters well beyond benchmark hygiene. Decode runs a long chain of
// small kernels — a 48-layer model at ~10 dispatches per layer is ~500
// dispatches per token — so if a dispatch costs tens of microseconds, that
// chain has a throughput ceiling entirely independent of how fast the
// kernels are. It also explains bandwidth measurements at small sizes: the
// existing 1M-element copy reports a *lower* GB/s than the 4M-element one,
// which is what happens when a measurement is dominated by a fixed cost
// rather than by memory traffic.
func RunOverhead(dev *vk.Device, warmup, iters uint32) ([]Result, error) {
	mod, err := dev.NewShaderModule(shaders.Empty)
	if err != nil {
		return nil, err
	}
	defer mod.Destroy()

	// Sized for the largest sweep entry, though with writeFlag=0 no
	// invocation ever writes to it.
	maxGroups := 0
	for _, g := range overheadGroupCounts {
		if g > maxGroups {
			maxGroups = g
		}
	}
	sink, err := dev.NewBuffer(maxGroups * 64 * 4)
	if err != nil {
		return nil, err
	}
	defer sink.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{sink},
		PushConstantSize: 4,
	})
	if err != nil {
		return nil, err
	}
	defer pipe.Destroy()

	pc := newPC().U32(0).Bytes() // writeFlag = 0: no stores, but not provably so at compile time

	var results []Result
	for _, groups := range overheadGroupCounts {
		ns, clocks, err := TimeDispatch(pipe, uint32(groups), 1, 1, warmup, iters, pc)
		if err != nil {
			return nil, fmt.Errorf("overhead groups=%d: %w", groups, err)
		}
		results = append(results, Result{
			Op: "overhead", Variant: "empty_dispatch", Size: groups,
			NsPerIter: ns,
			Clocks:    clocks,
		})
	}
	return results, nil
}
