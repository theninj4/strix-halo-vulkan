package bench

import (
	"fmt"
	"os"

	"strix-halo-vulkan/vk"
)

// filterWaveVariants drops the variants of a benchmark table that pin a
// subgroup size the device will not let a pipeline require (IDEAS §6.2).
//
// This is a filter rather than an error because the wave32 rows are an extra
// arm of an ablation, not the ablation: on a device without
// VK_EXT_subgroup_size_control, or one whose reported size range excludes 32,
// every wave64 row is still a complete measurement. Dropping them here also
// keeps the check in one place instead of once per op family, and prints which
// rows went missing so a short CSV is explained rather than merely noticed.
func filterWaveVariants[T any](variants []T, sgs vk.SubgroupSizeControl, key func(T) (string, uint32)) []T {
	out := variants[:0:0]
	for _, v := range variants {
		name, size := key(v)
		if size != 0 && (!sgs.Supported || size < sgs.MinSubgroupSize || size > sgs.MaxSubgroupSize) {
			fmt.Fprintf(os.Stderr, "%s: device cannot require subgroup size %d, skipping\n", name, size)
			continue
		}
		out = append(out, v)
	}
	return out
}

// mustSubgroupSizeControl is SubgroupSizeControl for the callers that only
// want it to decide which rows to run. The query reads two structs the driver
// fills in from the physical device and cannot fail for any reason a
// benchmark could act on, so a failure is reported as "no control available"
// — which drops the wave32 rows, the same as an old driver would.
func mustSubgroupSizeControl(phys *vk.PhysicalDevice) vk.SubgroupSizeControl {
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		fmt.Fprintf(os.Stderr, "subgroup size control query failed (%v); wave32 rows will be skipped\n", err)
		return vk.SubgroupSizeControl{}
	}
	return sgs
}
