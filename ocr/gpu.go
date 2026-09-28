package ocr

// The vision tower and projector on the device (OCR.md O3).
//
// This is qimage/vision's GPU tower, not a copy of it: the same SigLIP widths,
// the same kernels, the same matrix-core attention with 72-wide heads padded
// to 80. What differs is carried by that package's device-path hooks, and
// one choice made here:
//
//   - **The tower runs 2x2-block-major, although this checkpoint is raster.**
//     Full attention does not care what order its rows come in so long as
//     each row carries its own position grid and rope angles, so the pixels,
//     the grid and the rope table are all permuted to block order on the
//     host. Then the projector's 2x2 gather is the free concatenation of four
//     consecutive rows that qimage's merger already reads, and the merged
//     rows come out in the raster order of the merged grid, which is the
//     order the image tokens take. Taps are permuted back to raster.
//   - Patch 14 is 588 values a patch, which is padded to the GEMM's 640.
//   - The post_layernorm is Model.PostNorm (eps 1e-6, in place), and the
//     projector's pre_norm is the merger's norm at eps 1e-5.

import (
	"context"
	"fmt"
	"strings"

	"strix-halo-vulkan/qimage/vision"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// MaxPatches is the patch budget at the checkpoint's max_pixels.
const MaxPatches = DefaultMaxPixels / (PatchSize * PatchSize)

// GPUTower is the tower and projector staged on a device.
type GPUTower struct {
	g *vision.GPU
	// order is the current grid's block order: row t of the device's
	// stream is raster patch order[t].
	order []int
	// Tap, if set, sees each stage under the dump's names ("vis.embed",
	// "vis.block{i}", "vis.post_ln") in raster order.
	Tap func(name string, x *qwen.Mat)
	// rasterControl feeds the patches in their own raster order, so the
	// merger concatenates four consecutive raster rows: the negative control
	// for the permutation (gpu_test.go).
	rasterControl bool
}

// NewGPUTower stages t for images of up to maxPatches patches (0 is
// MaxPatches).
func NewGPUTower(dev *vk.Device, t *Tower, maxPatches int) (*GPUTower, error) {
	if len(t.Blocks) != VisionDepth {
		return nil, fmt.Errorf("ocr: %d of %d vision blocks loaded", len(t.Blocks), VisionDepth)
	}
	if maxPatches <= 0 {
		maxPatches = MaxPatches
	}
	gt := &GPUTower{}
	m := &vision.Model{
		Cfg: vision.Config{
			Depth: VisionDepth, HiddenSize: VisionHidden, NumHeads: VisionHeads,
			IntermediateSize: VisionFFN, HiddenAct: "gelu_pytorch_tanh",
			PatchSize: PatchSize, SpatialMergeSize: MergeSize, TemporalPatchSize: 1,
			InChannels: 3, NumPositionEmbeddings: visionGrid * visionGrid, OutHiddenSize: TextHidden,
		},
		PatchProj: t.PatchProj,
		PosEmbed:  t.PosEmbed,
		Blocks:    t.Blocks,
		Merger: vision.Merger{
			Merge: MergeSize * MergeSize, Norm: t.ProjNorm, FC1: t.FC1, FC2: t.FC2,
			Act: geluErf, NormEps: projNormEps,
		},
		PostNorm:    &t.PostLN,
		PostNormEps: 1e-6,
		Geometry: func(gh, gw int) (*qwen.Mat, *vision.Rope, error) {
			pos, rope := t.positionEmbedding(gh, gw), rasterRope(gh, gw)
			return gather(pos, gt.order), &vision.Rope{Cos: gather(rope.Cos, gt.order), Sin: gather(rope.Sin, gt.order)}, nil
		},
	}
	g, err := vision.NewGPU(dev, m, maxPatches)
	if err != nil {
		return nil, err
	}
	gt.g = g
	return gt, nil
}

// Destroy releases the device objects.
func (gt *GPUTower) Destroy() { gt.g.Destroy() }

// WeightBytes and ActivationBytes are the tower's device memory.
func (gt *GPUTower) WeightBytes() int     { return gt.g.WeightBytes() }
func (gt *GPUTower) ActivationBytes() int { return gt.g.ActivationBytes() }

// Forward runs one processed image and returns its [MergedH*MergedW, 1024]
// image-token rows.
func (gt *GPUTower) Forward(ctx context.Context, im *Image) (*qwen.Mat, error) {
	gh, gw := im.GridH, im.GridW
	if gh%MergeSize != 0 || gw%MergeSize != 0 || len(im.Patches) != gh*gw*PatchElems {
		return nil, fmt.Errorf("ocr: %d values for a %dx%d patch grid", len(im.Patches), gw, gh)
	}
	gt.order = blockOrder(gh, gw)
	if gt.rasterControl {
		for i := range gt.order {
			gt.order[i] = i
		}
	}
	px := gather(&qwen.Mat{Rows: gh * gw, Cols: PatchElems, Data: im.Patches}, gt.order)
	// A submit's work about what eight dispatches of a full-budget image
	// are: a 600-patch crop's tower goes from 32 to 29 ms (O11).
	gt.g.PerSubmit = 8 * max(1, gt.g.MaxRows()/(gh*gw))
	gt.g.Tap = nil
	if gt.Tap != nil {
		gt.g.Tap = func(name string, x *qwen.Mat) {
			switch {
			case name == "embed":
				name = "vis.embed"
			case name == "post_norm":
				name = "vis.post_ln"
			case strings.HasPrefix(name, "block"):
				name = "vis." + name
			}
			gt.Tap(name, scatter(x, gt.order))
		}
	}
	out, err := gt.g.Forward(ctx, px, gh, gw)
	if err != nil {
		return nil, err
	}
	return out.Merged, nil
}

// blockOrder lists the raster index of each patch in 2x2-block-major order:
// merge blocks in raster order, and each block's four patches in raster
// order inside it.
func blockOrder(gh, gw int) []int {
	out := make([]int, 0, gh*gw)
	for by := 0; by < gh/MergeSize; by++ {
		for bx := 0; bx < gw/MergeSize; bx++ {
			for dy := 0; dy < MergeSize; dy++ {
				for dx := 0; dx < MergeSize; dx++ {
					out = append(out, (by*MergeSize+dy)*gw+bx*MergeSize+dx)
				}
			}
		}
	}
	return out
}

// gather returns x's rows in order: row t is x's row order[t].
func gather(x *qwen.Mat, order []int) *qwen.Mat {
	out := qwen.NewMat(len(order), x.Cols)
	for t, r := range order {
		copy(out.Row(t), x.Row(r))
	}
	return out
}

// scatter undoes gather: row order[t] of the result is x's row t.
func scatter(x *qwen.Mat, order []int) *qwen.Mat {
	out := qwen.NewMat(len(order), x.Cols)
	for t, r := range order {
		copy(out.Row(r), x.Row(t))
	}
	return out
}
