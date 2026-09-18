package llm

// Moving an activation between two blocks' arenas, on the device (LLM.md L6c).
//
// L6b's graph was correct and 22.7% host plumbing. Every block here owns its
// own four buffers, so an activation leaving one block and entering the next
// crossed a buffer boundary — and until now it crossed it through the host:
// read out of one mapped arena, narrowed to halves on sixteen cores, written
// into the other, twice a sublayer and 96 times a pass. `shaders/llm_move.comp`
// is the same move as one dispatch.
//
// **The alternative was one shared arena, and this is not it.** Making the
// five blocks bump-allocate out of one pair of buffers means splitting every
// block's `alloc` into a pure layout pass and a materialisation, because a
// descriptor set needs its buffer before a pipeline exists and a Vulkan buffer
// cannot grow — a two-phase construction through five constructors and every
// caller of them. The move costs what it costs instead, and L6c-2 prices it:
// **3.6% of the graph at ubatch 2048**, against the 22.7% it replaces. A
// shared arena would buy that last 3.6% and nothing else, so it is written
// down here rather than built.

import (
	"fmt"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// Port is where an activation lives: which buffer, at what element offset,
// with what row stride and in what precision.
//
// It is the graph's view of a block's arena, and it is deliberately narrow.
// A block exposes a Port for the two or three tensors that cross its
// boundary and for nothing else, so "what may the graph reach into" is a
// list in each block's own file rather than an exported arena.
type Port struct {
	Buf *vk.Buffer
	// Off is in elements of the buffer's own kind: floats for an fp32 arena,
	// halves for an fp16 one.
	Off uint32
	// Stride is elements between rows, Width the values a row carries. They
	// differ wherever a GEMM's A operand is padded (§2.3).
	Stride, Width int
	// Rows is how many rows the destination must have written when the move
	// is done — the run's tokens rounded up to the consumer's row block,
	// since the GEMM rungs have no bounds check and a shorter run must not
	// read what a longer one left. Zero on a source, which is only read.
	Rows int
	// Half is whether the buffer is addressed as halves.
	Half bool
}

// movePush is llm_move.comp's push block, which is its own and not the
// vertical's 64-uint one: this kernel is built over two blocks' buffers
// rather than one block's five, so it shares no descriptor layout with
// anything and has no reason to share a push layout either.
type movePush struct {
	SrcOff, DstOff       uint32
	Rows, PadRows, Width uint32
	SrcStride, DstStride uint32
	Narrow               uint32
}

// bytes pads the block out to the vertical's own push size, because a
// sequence recorded into one command buffer pushes the same number of bytes
// for every dispatch in it (L7d) — the shim reads the blocks out of one flat
// array — and the moves now share a command buffer with the five blocks. The
// pipeline layout below declares the same range, so the bytes past this
// struct are inside it and no shader reads them.
func (p movePush) bytes() []byte {
	out := make([]byte, movePushSize)
	*(*movePush)(unsafe.Pointer(&out[0])) = p
	return out
}

// movePushSize is that shared size: the vertical's 64-uint block.
var movePushSize = int(unsafe.Sizeof(push{}))

// bufPair keys the pipeline cache. A compute pipeline here owns its
// descriptor set, so one is needed per (source, destination) buffer pair —
// nine for the five blocks and the head, built on first use.
type bufPair struct{ src, dst *vk.Buffer }

// mover holds those pipelines.
type mover struct {
	// rec, when set, collects these dispatches into the pass's one command
	// buffer instead of submitting them (record.go).
	rec   *recorder
	dev   *vk.Device
	mod   *vk.ShaderModule
	pipes map[bufPair]*vk.ComputePipeline
	// moves counts the dispatches a graph issued, which is the number L6c's
	// write-up quotes beside what they cost.
	moves int
}

func newMover(dev *vk.Device) (*mover, error) {
	mod, err := dev.NewShaderModule(shaders.LLMMove)
	if err != nil {
		return nil, fmt.Errorf("llm: shader move: %w", err)
	}
	return &mover{dev: dev, mod: mod, pipes: map[bufPair]*vk.ComputePipeline{}}, nil
}

// pipeline returns the pipeline for one buffer pair, building it on first use.
//
// Bindings 1 and 2 are the same VkBuffer twice: a destination is either an
// fp32 arena or an fp16 one, and `narrow` decides which declaration the
// kernel writes through.
func (m *mover) pipeline(src, dst *vk.Buffer) (*vk.ComputePipeline, error) {
	key := bufPair{src, dst}
	if p, ok := m.pipes[key]; ok {
		return p, nil
	}
	p, err := m.dev.NewPipeline(m.mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{src, dst, dst},
		PushConstantSize: uint32(movePushSize),
	})
	if err != nil {
		return nil, fmt.Errorf("llm: move pipeline: %w", err)
	}
	m.pipes[key] = p
	return p, nil
}

// Move copies rows of an activation from one block's arena into another's,
// narrowing to halves where the destination is an fp16 operand.
//
// Rows past `rows` are written as zeros up to `dst.Rows`, which is the
// consumer's row block and not the prompt: the GEMM rungs have no bounds
// check, so those rows have to carry the products of zeros.
func (m *mover) Move(dst, src Port, rows int) error {
	if src.Half {
		return fmt.Errorf("llm: move reads an fp32 arena, the source is halves")
	}
	if src.Width != dst.Width {
		return fmt.Errorf("llm: move of %d values into %d", src.Width, dst.Width)
	}
	pad := maxInt(dst.Rows, rows)
	pc := movePush{
		SrcOff: src.Off, DstOff: dst.Off,
		Rows: uint32(rows), PadRows: uint32(pad), Width: uint32(src.Width),
		SrcStride: uint32(src.Stride), DstStride: uint32(dst.Stride),
	}
	if dst.Half {
		pc.Narrow = 1
	}
	p, err := m.pipeline(src.Buf, dst.Buf)
	if err != nil {
		return err
	}
	m.moves++
	d := []vk.MultiDispatch{{
		Pipeline: p, GroupsX: uint32(pad), GroupsY: 1, PushConstants: pc.bytes(),
	}}
	if m.rec.add(ownMove, nil, d) {
		return nil
	}
	_, err = vk.DispatchMultiTimed(d, 1, 1, true)
	if err != nil {
		return fmt.Errorf("llm: move of %d x %d: %w", rows, src.Width, err)
	}
	return nil
}

// Destroy releases the pipelines and the module.
func (m *mover) Destroy() {
	for _, p := range m.pipes {
		p.Destroy()
	}
	if m.mod != nil {
		m.mod.Destroy()
	}
}
