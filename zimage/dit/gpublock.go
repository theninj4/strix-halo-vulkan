package dit

import (
	"strix-halo-vulkan/vk"
)

// GPUBlock is a GPUStack holding a single block, which is the unit every
// stage before 4c was built and validated as.
//
// It exists because the *validation* unit and the *residency* unit are not
// the same thing. reference/dump_dit_block.py instantiates one block and
// dumps every submodule's output, so one block is the largest thing the
// implementation can be checked against something other than itself; the
// stack is what has to hold 12.0 GB of weights across several storage
// buffers. Sharing one implementation between them is what makes the
// stagewise tests -- RunTo, the tensor offsets, the negative controls -- apply
// unchanged to a block that is one of 34.
type GPUBlock struct {
	*GPUStack
}

// NewGPUBlock uploads one block's weights and builds its graph for a fixed
// sequence length. plan may be nil, which takes DefaultGEMMPlan.
func NewGPUBlock(dev *vk.Device, blk *Block, rope *RoPE, tokens int, plan GEMMPlan) (*GPUBlock, error) {
	return newGPUBlock(dev, blk, rope, tokens, plan, blockControls{})
}

func newGPUBlock(dev *vk.Device, blk *Block, rope *RoPE, tokens int, plan GEMMPlan, ctl blockControls) (*GPUBlock, error) {
	spec := blockSpec{prefix: "block", modulated: blk.AdaLN != nil, blk: blk}
	s, err := newStack(dev, nil, nil, []blockSpec{spec}, rope, tokens, plan, ctl, maxBankBytes)
	if err != nil {
		return nil, err
	}
	return &GPUBlock{s}, nil
}

// Apply runs the block over x [tokens, dim] with the timestep embedding
// adaln, and returns the new residual stream.
func (g *GPUBlock) Apply(x *Mat, adaln []float32) (*Mat, error) {
	return g.GPUStack.Apply(x, adaln, nil)
}

// Profile runs the block one dispatch at a time and times each on the GPU.
func (g *GPUBlock) Profile(x *Mat, adaln []float32) ([]Stage, *Mat, error) {
	return g.GPUStack.Profile(x, adaln, nil)
}

// Labels lists the block's dispatches in order.
func (g *GPUBlock) Labels() []string { return g.GPUStack.Labels(0) }

// RunTo runs the graph up to and including the dispatch with the given label.
func (g *GPUBlock) RunTo(x *Mat, adaln []float32, label string) error {
	return g.GPUStack.RunTo(x, adaln, 0, label)
}

// FLOPs is the block's multiply-add count at the length it was built for.
func (g *GPUBlock) FLOPs() float64 { return g.GPUStack.FLOPs(g.tokens) }
