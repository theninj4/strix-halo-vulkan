// Package vk is a minimal Go wrapper around shim.c/shim.h, which does the
// actual Vulkan calls. See shim.h for why the struct-building work lives in
// C rather than here.
package vk

/*
#cgo pkg-config: vulkan
#include "shim.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"time"
	"unsafe"
)

// Error wraps a non-VK_SUCCESS VkResult.
type Error struct {
	Op     string
	Result C.VkResult
}

func (e *Error) Error() string {
	return fmt.Sprintf("vk: %s failed: VkResult(%d)", e.Op, int32(e.Result))
}

func check(op string, r C.VkResult) error {
	if r != C.VK_SUCCESS {
		return &Error{Op: op, Result: r}
	}
	return nil
}

func boolToVk(b bool) C.VkBool32 {
	if b {
		return C.VK_TRUE
	}
	return C.VK_FALSE
}

// Instance wraps a VkInstance.
type Instance struct {
	handle C.VkInstance
}

// NewInstance creates a VkInstance with no extensions or layers — enough for
// headless compute. appName is purely informational (shows up in tools like
// validation layers or GPU profilers).
func NewInstance(appName string) (*Instance, error) {
	cAppName := C.CString(appName)
	defer C.free(unsafe.Pointer(cAppName))

	var handle C.VkInstance
	if err := check("vkCreateInstance", C.shim_create_instance(cAppName, &handle)); err != nil {
		return nil, err
	}
	return &Instance{handle: handle}, nil
}

// Destroy releases the instance.
func (i *Instance) Destroy() {
	C.shim_destroy_instance(i.handle)
}

// DeviceFeatures are the optional compute-relevant device features this
// benchmark suite cares about: fp16 math/storage, int8 storage, accelerated
// integer dot-product instructions, and cooperative-matrix (matrix-multiply
// acceleration).
type DeviceFeatures struct {
	Float16           bool
	Int8              bool
	IntegerDotProduct bool
	CoopMatrix        bool
	// SubgroupSizeControl enables VK_EXT_subgroup_size_control, without which
	// PipelineSpec.RequiredSubgroupSize must stay 0 and every shader runs at
	// whatever wave size the driver picks (IDEAS.md §6.2).
	SubgroupSizeControl bool
}

func (f DeviceFeatures) toShim() C.ShimDeviceFeatures {
	return C.ShimDeviceFeatures{
		float16:             boolToVk(f.Float16),
		int8:                boolToVk(f.Int8),
		integerDotProduct:   boolToVk(f.IntegerDotProduct),
		coopMatrix:          boolToVk(f.CoopMatrix),
		subgroupSizeControl: boolToVk(f.SubgroupSizeControl),
	}
}

// SubgroupSizeControl is what wave sizes a compute pipeline on this device
// may require. Supported folds together the extension's presence, the
// feature bit and whether the compute stage accepts a required size, because
// a pipeline may only name one when all three hold.
type SubgroupSizeControl struct {
	Supported                    bool
	ComputeFullSubgroups         bool
	MinSubgroupSize              uint32
	MaxSubgroupSize              uint32
	MaxComputeWorkgroupSubgroups uint32
}

// ComponentType mirrors VkComponentTypeKHR — the element type of a
// cooperative-matrix operand.
type ComponentType uint32

const (
	ComponentFloat16 ComponentType = 0
	ComponentFloat32 ComponentType = 1
	ComponentFloat64 ComponentType = 2
	ComponentSInt8   ComponentType = 3
	ComponentSInt16  ComponentType = 4
	ComponentSInt32  ComponentType = 5
	ComponentSInt64  ComponentType = 6
	ComponentUInt8   ComponentType = 7
	ComponentUInt16  ComponentType = 8
	ComponentUInt32  ComponentType = 9
	ComponentUInt64  ComponentType = 10
)

// Scope mirrors VkScopeKHR — which set of invocations cooperate on a
// cooperative-matrix operation.
type Scope uint32

const (
	ScopeDevice      Scope = 1
	ScopeWorkgroup   Scope = 2
	ScopeSubgroup    Scope = 3
	ScopeQueueFamily Scope = 5
)

// CoopMatShape is one MxNxK/type/scope combination VK_KHR_cooperative_matrix
// supports on a given device.
type CoopMatShape struct {
	M, N, K                         int
	AType, BType, CType, ResultType ComponentType
	Scope                           Scope
}

// PhysicalDevice wraps a VkPhysicalDevice plus its cached properties.
type PhysicalDevice struct {
	handle          C.VkPhysicalDevice
	instance        C.VkInstance
	Name            string
	VendorID        uint32
	DeviceID        uint32
	DeviceType      C.VkPhysicalDeviceType
	TimestampPeriod float64 // nanoseconds per timestamp-query tick
}

const maxPhysicalDevices = 64

// PhysicalDevices enumerates every physical device the instance can see.
func (i *Instance) PhysicalDevices() ([]PhysicalDevice, error) {
	raw := make([]C.ShimPhysicalDevice, maxPhysicalDevices)
	var count C.uint32_t
	if err := check("vkEnumeratePhysicalDevices",
		C.shim_enumerate_physical_devices(i.handle, &raw[0], C.uint32_t(len(raw)), &count)); err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, fmt.Errorf("vk: no physical devices found")
	}

	devices := make([]PhysicalDevice, 0, count)
	for _, d := range raw[:count] {
		devices = append(devices, PhysicalDevice{
			handle:          d.handle,
			instance:        i.handle,
			Name:            C.GoString(&d.name[0]),
			VendorID:        uint32(d.vendorID),
			DeviceID:        uint32(d.deviceID),
			DeviceType:      d.deviceType,
			TimestampPeriod: float64(d.timestampPeriod),
		})
	}
	return devices, nil
}

// ComputeQueueFamily returns the index of the first queue family on this
// device that supports compute.
func (p *PhysicalDevice) ComputeQueueFamily() (uint32, error) {
	var family C.uint32_t
	if err := check("vkGetPhysicalDeviceQueueFamilyProperties",
		C.shim_find_compute_queue_family(p.handle, &family)); err != nil {
		return 0, fmt.Errorf("vk: device %s has no compute-capable queue family", p.Name)
	}
	return uint32(family), nil
}

// SupportedFeatures reports which optional DeviceFeatures this physical
// device actually supports, independent of what any logical device enables.
func (p *PhysicalDevice) SupportedFeatures() (DeviceFeatures, error) {
	var out C.ShimDeviceFeatures
	if err := check("vkGetPhysicalDeviceFeatures2", C.shim_query_device_features(p.handle, &out)); err != nil {
		return DeviceFeatures{}, err
	}
	return DeviceFeatures{
		Float16:             out.float16 != 0,
		Int8:                out.int8 != 0,
		IntegerDotProduct:   out.integerDotProduct != 0,
		CoopMatrix:          out.coopMatrix != 0,
		SubgroupSizeControl: out.subgroupSizeControl != 0,
	}, nil
}

// SubgroupSizeControl reports the wave sizes this device will let a compute
// pipeline pin itself to. Supported == false means the default wave size is
// all there is.
func (p *PhysicalDevice) SubgroupSizeControl() (SubgroupSizeControl, error) {
	var out C.ShimSubgroupSizeControl
	if err := check("vkGetPhysicalDeviceProperties2", C.shim_query_subgroup_size_control(p.handle, &out)); err != nil {
		return SubgroupSizeControl{}, err
	}
	return SubgroupSizeControl{
		Supported:                    out.supported != 0,
		ComputeFullSubgroups:         out.computeFullSubgroups != 0,
		MinSubgroupSize:              uint32(out.minSubgroupSize),
		MaxSubgroupSize:              uint32(out.maxSubgroupSize),
		MaxComputeWorkgroupSubgroups: uint32(out.maxComputeWorkgroupSubgroups),
	}, nil
}

const maxCoopMatShapes = 64

// CooperativeMatrixShapes enumerates the MxNxK/type/scope combinations
// VK_KHR_cooperative_matrix supports on this device. Returns an empty slice
// (not an error) if the extension isn't supported — check SupportedFeatures
// first.
func (p *PhysicalDevice) CooperativeMatrixShapes() ([]CoopMatShape, error) {
	raw := make([]C.ShimCoopMatProperty, maxCoopMatShapes)
	var count C.uint32_t
	if err := check("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR",
		C.shim_query_cooperative_matrix_properties(p.instance, p.handle, &raw[0], C.uint32_t(len(raw)), &count)); err != nil {
		return nil, err
	}

	shapes := make([]CoopMatShape, 0, count)
	for _, s := range raw[:count] {
		shapes = append(shapes, CoopMatShape{
			M: int(s.MSize), N: int(s.NSize), K: int(s.KSize),
			AType:      ComponentType(s.AType),
			BType:      ComponentType(s.BType),
			CType:      ComponentType(s.CType),
			ResultType: ComponentType(s.ResultType),
			Scope:      Scope(s.scope),
		})
	}
	return shapes, nil
}

// Device wraps a VkDevice and the single compute queue we pulled from it.
type Device struct {
	handle      C.VkDevice
	phys        *PhysicalDevice
	queue       C.VkQueue
	queueFamily uint32
}

// NewDevice creates a logical device on phys with a single queue drawn from
// queueFamily, enabling exactly the optional features set in the request.
func NewDevice(phys *PhysicalDevice, queueFamily uint32, features DeviceFeatures) (*Device, error) {
	req := features.toShim()
	var handle C.VkDevice
	var queue C.VkQueue
	if err := check("vkCreateDevice",
		C.shim_create_device(phys.handle, C.uint32_t(queueFamily), &req, &handle, &queue)); err != nil {
		return nil, err
	}
	return &Device{handle: handle, phys: phys, queue: queue, queueFamily: queueFamily}, nil
}

// Destroy releases the device.
func (d *Device) Destroy() {
	C.shim_destroy_device(d.handle)
}

// Handle returns the raw VkDevice.
func (d *Device) Handle() C.VkDevice { return d.handle }

// Queue returns the VkQueue pulled from the device's queue family.
func (d *Device) Queue() C.VkQueue { return d.queue }

// QueueFamily returns the index of the queue family backing Queue().
func (d *Device) QueueFamily() uint32 { return d.queueFamily }

// WaitIdle blocks until every queue on the device is idle.
func (d *Device) WaitIdle() error {
	return check("vkDeviceWaitIdle", C.shim_device_wait_idle(d.handle))
}

// Buffer is a VkBuffer bound to host-visible memory (device-local when the
// hardware supports it — see shim_create_storage_buffer), mapped for the
// lifetime of the buffer.
type Buffer struct {
	dev    *Device
	handle C.VkBuffer
	memory C.VkDeviceMemory
	mapped unsafe.Pointer
	size   int
}

// NewBuffer allocates a buffer of size bytes usable as a storage buffer in a
// compute shader, left mapped.
func (d *Device) NewBuffer(size int) (*Buffer, error) {
	var handle C.VkBuffer
	var memory C.VkDeviceMemory
	var mapped unsafe.Pointer
	if err := check("vkCreateBuffer/vkAllocateMemory",
		C.shim_create_storage_buffer(d.handle, d.phys.handle, C.VkDeviceSize(size), &handle, &memory, &mapped)); err != nil {
		return nil, err
	}
	return &Buffer{dev: d, handle: handle, memory: memory, mapped: mapped, size: size}, nil
}

// Destroy unmaps and releases the buffer's memory and handle.
func (b *Buffer) Destroy() {
	C.shim_destroy_buffer(b.dev.handle, b.handle, b.memory)
}

// Size returns the buffer's size in bytes.
func (b *Buffer) Size() int { return b.size }

// WriteFloat32 copies src into the buffer's mapped memory starting at byte 0.
func (b *Buffer) WriteFloat32(src []float32) {
	dst := unsafe.Slice((*float32)(b.mapped), len(src))
	copy(dst, src)
}

// ReadFloat32 reads n float32s back out of the buffer's mapped memory.
func (b *Buffer) ReadFloat32(n int) []float32 {
	src := unsafe.Slice((*float32)(b.mapped), n)
	out := make([]float32, n)
	copy(out, src)
	return out
}

// ReadUint32 reads n uint32s back out of the buffer's mapped memory.
func (b *Buffer) ReadUint32(n int) []uint32 {
	src := unsafe.Slice((*uint32)(b.mapped), n)
	out := make([]uint32, n)
	copy(out, src)
	return out
}

// MappedPointer returns the start of the buffer's host mapping, for callers
// that want to fill or read it as a type the Write*/Read* helpers don't
// cover, or to fill a multi-hundred-MB buffer in place rather than building
// a host-side copy of it first.
func (b *Buffer) MappedPointer() unsafe.Pointer { return b.mapped }

// WriteBytes copies src into the buffer's mapped memory starting at byte 0.
func (b *Buffer) WriteBytes(src []byte) {
	dst := unsafe.Slice((*byte)(b.mapped), len(src))
	copy(dst, src)
}

// FillRepeating tiles block across the whole buffer, in place. It exists for
// the multi-gigabyte weight banks IDEAS §3.5 needs: a 512-expert fp16 bank is
// 1.8 GB, and building a host-side copy of that to hand to WriteBytes costs
// more than the measurement it feeds. Only the addresses matter for a timing
// run, so a repeating block of real values is as good as a unique one — and
// it must be real values, because reinterpreting whatever the allocation held
// as fp16 can produce infinities that change what the arithmetic costs.
func (b *Buffer) FillRepeating(block []byte) {
	if len(block) == 0 {
		return
	}
	dst := unsafe.Slice((*byte)(b.mapped), b.size)
	for off := 0; off < len(dst); off += len(block) {
		copy(dst[off:], block)
	}
}

// ReadBytes reads n bytes back out of the buffer's mapped memory.
func (b *Buffer) ReadBytes(n int) []byte {
	src := unsafe.Slice((*byte)(b.mapped), n)
	out := make([]byte, n)
	copy(out, src)
	return out
}

// Handle returns the raw VkBuffer.
func (b *Buffer) Handle() C.VkBuffer { return b.handle }

// ShaderModule wraps a VkShaderModule.
type ShaderModule struct {
	dev    *Device
	handle C.VkShaderModule
}

// NewShaderModule loads pre-compiled SPIR-V bytecode.
func (d *Device) NewShaderModule(spirv []byte) (*ShaderModule, error) {
	if len(spirv)%4 != 0 {
		return nil, fmt.Errorf("vk: SPIR-V length %d is not a multiple of 4", len(spirv))
	}
	var handle C.VkShaderModule
	if err := check("vkCreateShaderModule",
		C.shim_create_shader_module(d.handle, unsafe.Pointer(&spirv[0]), C.size_t(len(spirv)), &handle)); err != nil {
		return nil, err
	}
	return &ShaderModule{dev: d, handle: handle}, nil
}

// Destroy releases the shader module.
func (s *ShaderModule) Destroy() {
	C.shim_destroy_shader_module(s.dev.handle, s.handle)
}

// Handle returns the raw VkShaderModule.
func (s *ShaderModule) Handle() C.VkShaderModule { return s.handle }

// SpecConstant is one uint32 specialization constant, identified by the
// constant ID it's declared with in GLSL (`layout(constant_id = ID)`).
type SpecConstant struct {
	ID    uint32
	Value uint32
}

// PipelineSpec describes the resources a ComputePipeline binds: one storage
// buffer per entry in Buffers (bound to bindings 0..len(Buffers)-1), an
// optional push-constant range, and optional specialization constants.
type PipelineSpec struct {
	Buffers          []*Buffer
	PushConstantSize uint32
	SpecConstants    []SpecConstant
	// RequiredSubgroupSize pins the wave size the shader runs at instead of
	// letting the driver choose (IDEAS.md §6.2). Zero keeps the default. It
	// must be a power of two inside the device's reported
	// [MinSubgroupSize, MaxSubgroupSize], the shader's local_size_x must be a
	// multiple of it, and the device must have been created with
	// DeviceFeatures.SubgroupSizeControl — otherwise pipeline creation fails.
	RequiredSubgroupSize uint32
}

// ComputePipeline is a full pipeline (descriptor set layout, pipeline,
// descriptor set, command buffer, fence and timestamp query pool) built
// from a PipelineSpec.
type ComputePipeline struct {
	dev    *Device
	handle C.ShimComputePipeline
}

// NewPipeline builds a compute pipeline for shader bound according to spec.
func (d *Device) NewPipeline(shader *ShaderModule, spec PipelineSpec) (*ComputePipeline, error) {
	p := &ComputePipeline{dev: d}

	bufHandles := make([]C.VkBuffer, len(spec.Buffers))
	for i, b := range spec.Buffers {
		bufHandles[i] = b.handle
	}
	var bufPtr *C.VkBuffer
	if len(bufHandles) > 0 {
		bufPtr = &bufHandles[0]
	}

	var specC []C.ShimSpecConstant
	if len(spec.SpecConstants) > 0 {
		specC = make([]C.ShimSpecConstant, len(spec.SpecConstants))
		for i, sc := range spec.SpecConstants {
			specC[i] = C.ShimSpecConstant{constantID: C.uint32_t(sc.ID), value: C.uint32_t(sc.Value)}
		}
	}
	var specPtr *C.ShimSpecConstant
	if len(specC) > 0 {
		specPtr = &specC[0]
	}

	if err := check("vkCreateComputePipelines",
		C.shim_create_compute_pipeline(d.handle, shader.handle, bufPtr, C.uint32_t(len(bufHandles)),
			C.uint32_t(spec.PushConstantSize), specPtr, C.uint32_t(len(specC)),
			C.uint32_t(spec.RequiredSubgroupSize),
			C.uint32_t(d.queueFamily), &p.handle)); err != nil {
		p.Destroy()
		return nil, err
	}
	return p, nil
}

// NewComputePipeline is a convenience wrapper for the common case of a
// single storage-buffer binding and no push constants or specialization
// constants.
func (d *Device) NewComputePipeline(shader *ShaderModule, buf *Buffer) (*ComputePipeline, error) {
	return d.NewPipeline(shader, PipelineSpec{Buffers: []*Buffer{buf}})
}

// DispatchTimed records iterations back-to-back dispatches of
// (groupsX,groupsY,groupsZ) workgroups (with a barrier between each, so
// in-place steady-state throughput is what gets measured), submits them,
// blocks until complete, and returns the GPU-side elapsed time measured via
// timestamp queries — not wall-clock time, so it excludes CPU submission and
// scheduling overhead.
func (p *ComputePipeline) DispatchTimed(groupsX, groupsY, groupsZ, iterations uint32, pushConstants []byte) (time.Duration, error) {
	var start, end C.uint64_t
	var pcPtr unsafe.Pointer
	var pcLen C.uint32_t
	if len(pushConstants) > 0 {
		pcPtr = unsafe.Pointer(&pushConstants[0])
		pcLen = C.uint32_t(len(pushConstants))
	}
	if err := check("vkQueueSubmit", C.shim_dispatch_timed(p.dev.handle, p.dev.queue, &p.handle,
		C.uint32_t(groupsX), C.uint32_t(groupsY), C.uint32_t(groupsZ), C.uint32_t(iterations),
		pcPtr, pcLen, &start, &end)); err != nil {
		return 0, err
	}
	ticks := float64(uint64(end)) - float64(uint64(start))
	ns := ticks * p.dev.phys.TimestampPeriod
	return time.Duration(ns), nil
}

// DispatchSequenceTimed is DispatchTimed for a sequence of dispatches that
// differ in their push constants: the i-th covers groupsX[i] workgroups on
// the X axis and pushes pushConstants[i], and a barrier separates every
// dispatch from the next — within one iteration and across iterations alike.
//
// It exists for IDEAS §3.5's baseline. A grouped kernel's claim is that one
// dispatch over a tile table beats one dispatch per expert, and the losing
// arm of that comparison is a sequence of 512 dispatches whose only
// difference is where in the table each starts. Expressing that with
// DispatchTimed is impossible (one constant block per command buffer), and
// expressing it as 512 separate submits would measure the CPU's submit path
// instead of the GPU's.
//
// Every element of pushConstants must be the same length, since the pipeline
// layout declares one range.
func (p *ComputePipeline) DispatchSequenceTimed(groupsX []uint32, groupsY, groupsZ, iterations uint32, pushConstants [][]byte) (time.Duration, error) {
	if len(groupsX) == 0 {
		return 0, fmt.Errorf("DispatchSequenceTimed: empty dispatch sequence")
	}
	if len(pushConstants) != 0 && len(pushConstants) != len(groupsX) {
		return 0, fmt.Errorf("DispatchSequenceTimed: %d push-constant blocks for %d dispatches",
			len(pushConstants), len(groupsX))
	}
	// Flattened into one contiguous buffer because cgo will not let a Go
	// slice of Go slices cross the boundary.
	var flat []byte
	var pcSize int
	if len(pushConstants) > 0 {
		pcSize = len(pushConstants[0])
		flat = make([]byte, 0, pcSize*len(pushConstants))
		for i, b := range pushConstants {
			if len(b) != pcSize {
				return 0, fmt.Errorf("DispatchSequenceTimed: push-constant block %d is %d bytes, block 0 is %d",
					i, len(b), pcSize)
			}
			flat = append(flat, b...)
		}
	}

	var start, end C.uint64_t
	var pcPtr unsafe.Pointer
	if len(flat) > 0 {
		pcPtr = unsafe.Pointer(&flat[0])
	}
	if err := check("vkQueueSubmit", C.shim_dispatch_seq_timed(p.dev.handle, p.dev.queue, &p.handle,
		(*C.uint32_t)(unsafe.Pointer(&groupsX[0])), C.uint32_t(len(groupsX)),
		C.uint32_t(groupsY), C.uint32_t(groupsZ), C.uint32_t(iterations),
		pcPtr, C.uint32_t(pcSize), &start, &end)); err != nil {
		return 0, err
	}
	ticks := float64(uint64(end)) - float64(uint64(start))
	return time.Duration(ticks * p.dev.phys.TimestampPeriod), nil
}

// Dispatch records and submits a single dispatch covering groupsX
// workgroups on the X axis, then blocks until it completes. Its GPU timing
// is discarded — use DispatchTimed for benchmarking.
func (p *ComputePipeline) Dispatch(groupsX uint32) error {
	_, err := p.DispatchTimed(groupsX, 1, 1, 1, nil)
	return err
}

// Destroy releases every object the pipeline owns.
func (p *ComputePipeline) Destroy() {
	C.shim_destroy_compute_pipeline(p.dev.handle, &p.handle)
}
