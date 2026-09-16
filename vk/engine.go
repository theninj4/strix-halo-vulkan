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
	"strings"
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
	features    DeviceFeatures
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
	return &Device{handle: handle, phys: phys, queue: queue, queueFamily: queueFamily, features: features}, nil
}

// Physical returns the physical device this logical device was created on, so
// a caller holding only the Device can still ask what the hardware reports —
// cooperative-matrix shapes, for instance.
func (d *Device) Physical() *PhysicalDevice { return d.phys }

// Features returns the optional features this device was *created* with,
// which is what a shader may actually use: a feature the hardware supports
// but NewDevice was not asked for is not enabled.
func (d *Device) Features() DeviceFeatures { return d.features }

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

// Memory property bits, as reported per memory type. Named here rather than
// imported from cgo so a caller can test them without the C types.
const (
	MemoryDeviceLocal    = 0x0001
	MemoryHostVisible    = 0x0002
	MemoryHostCoherent   = 0x0004
	MemoryHostCached     = 0x0008
	MemoryDeviceCoherent = 0x0040 // VK_AMD_device_coherent_memory
	MemoryDeviceUncached = 0x0080 // VK_AMD_device_coherent_memory
)

// MemoryType is one entry of the device's memory-type table, with the heap it
// draws from. This device exposes eleven of them across two heaps that overlap
// the same physical RAM, and which one a weight bank lands in decides both how
// large it may be (heap 1 stops at 83.79 GiB) and, possibly, how fast it
// reads — IDEAS §5.1, LLM.md L0b.
type MemoryType struct {
	Index            uint32
	HeapIndex        uint32
	Flags            uint32
	HeapSize         uint64
	BufferCompatible bool // can back a plain storage buffer
}

// Has reports whether every bit in want is set.
func (m MemoryType) Has(want uint32) bool { return m.Flags&want == want }

// String renders the property bits the way vulkaninfo names them, shortened.
func (m MemoryType) String() string {
	names := []struct {
		bit  uint32
		name string
	}{
		{MemoryDeviceLocal, "DEVICE_LOCAL"},
		{MemoryHostVisible, "HOST_VISIBLE"},
		{MemoryHostCoherent, "HOST_COHERENT"},
		{MemoryHostCached, "HOST_CACHED"},
		{MemoryDeviceCoherent, "DEVICE_COHERENT"},
		{MemoryDeviceUncached, "DEVICE_UNCACHED"},
	}
	var parts []string
	for _, n := range names {
		if m.Flags&n.bit != 0 {
			parts = append(parts, n.name)
		}
	}
	if len(parts) == 0 {
		return "-"
	}
	return strings.Join(parts, "|")
}

// MemoryTypes returns the device's memory-type table.
func (d *Device) MemoryTypes() ([]MemoryType, error) {
	const max = 32
	raw := make([]C.ShimMemoryType, max)
	var count C.uint32_t
	if err := check("vkGetPhysicalDeviceMemoryProperties",
		C.shim_memory_types(d.handle, d.phys.handle, &raw[0], C.uint32_t(max), &count)); err != nil {
		return nil, err
	}
	n := int(count)
	if n > max {
		n = max
	}
	out := make([]MemoryType, n)
	for i := 0; i < n; i++ {
		out[i] = MemoryType{
			Index:            uint32(raw[i].index),
			HeapIndex:        uint32(raw[i].heapIndex),
			Flags:            uint32(raw[i].propertyFlags),
			HeapSize:         uint64(raw[i].heapSize),
			BufferCompatible: raw[i].bufferCompatible != 0,
		}
	}
	return out, nil
}

// NewHostCachedBuffer allocates a storage buffer from a HOST_CACHED memory
// type when the device has one, and falls back to NewBuffer when it does not.
//
// It exists for **activation arenas the host reads back**, and the difference
// is not marginal. On this device the type NewBuffer prefers —
// DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT, write-combined — is written at
// 50 GB/s and read at **0.18**, because an uncached mapping turns a read into
// one uncached load per cache line. A HOST_CACHED type reads the same buffer
// at **24.8 GB/s**, 138x, and writes it at 70. See LLM.md L6b.
//
// It is not the right choice for a weight bank: those are written once and
// never read, so the cached type buys only its faster write, and the GPU-side
// rate is what a bank is about (L0b: every type reads at 236-237 GB/s from
// the device, so the choice is free there and it is not free here).
func (d *Device) NewHostCachedBuffer(size int) (*Buffer, error) {
	types, err := d.MemoryTypes()
	if err != nil {
		return nil, err
	}
	const want = MemoryHostVisible | MemoryHostCoherent | MemoryHostCached
	for _, mt := range types {
		if mt.BufferCompatible && mt.Has(want) {
			return d.NewBufferOfType(size, mt.Index)
		}
	}
	return d.NewBuffer(size)
}

// NewBufferOfType allocates a storage buffer from a named memory type rather
// than from the one NewBuffer prefers. A type without HOST_VISIBLE cannot be
// mapped, so MappedPointer and every Write/Read method on the result are
// unusable — such a buffer is only good for a kernel that reads whatever the
// allocation happened to contain, which is exactly what the bandwidth probes
// want.
func (d *Device) NewBufferOfType(size int, memType uint32) (*Buffer, error) {
	var handle C.VkBuffer
	var memory C.VkDeviceMemory
	var mapped unsafe.Pointer
	if err := check("vkCreateBuffer/vkAllocateMemory",
		C.shim_create_storage_buffer_of_type(d.handle, C.VkDeviceSize(size), C.uint32_t(memType),
			&handle, &memory, &mapped)); err != nil {
		return nil, err
	}
	return &Buffer{dev: d, handle: handle, memory: memory, mapped: mapped, size: size}, nil
}

// Mapped reports whether the buffer's memory could be mapped, which is false
// only for a NewBufferOfType on a type without HOST_VISIBLE.
func (b *Buffer) Mapped() bool { return b.mapped != nil }

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

// WriteFloat32At copies src into the buffer's mapped memory at an element
// offset, so several tensors can share one arena.
func (b *Buffer) WriteFloat32At(off int, src []float32) {
	dst := unsafe.Slice((*float32)(b.mapped), off+len(src))
	copy(dst[off:], src)
}

// ReadFloat32At reads n float32s starting at an element offset. Reading a
// whole buffer to recover a tensor near its end is the difference between
// copying a few megabytes and copying gigabytes, which for the VAE decoder's
// activation arena was 23x the GPU time of the decode itself.
func (b *Buffer) ReadFloat32At(off, n int) []float32 {
	src := unsafe.Slice((*float32)(b.mapped), off+n)
	out := make([]float32, n)
	copy(out, src[off:])
	return out
}

// ZeroFloat32At clears n float32s at an element offset, in place in the
// mapped memory.
//
// It exists because `WriteFloat32At(off, make([]float32, n))` is two costs,
// and on a hot path the allocation is the larger one: the gated DeltaNet's
// recurrent state is 3.1 MB a layer and 36 layers of it are reset at the
// start of every prefill, so the obvious spelling was **113 MB of Go
// allocation a graph** — a fixed ~55 ms whatever the prompt length, which is
// 12% of a 128-token pass and invisible at 2048 (LLM.md L6c-3).
func (b *Buffer) ZeroFloat32At(off, n int) {
	clear(unsafe.Slice((*float32)(b.mapped), off+n)[off:])
}

// ZeroUint16At clears n uint16s at an element offset, for the fp16 arenas.
func (b *Buffer) ZeroUint16At(off, n int) {
	clear(unsafe.Slice((*uint16)(b.mapped), off+n)[off:])
}

// Zero clears the whole buffer, which is what an arena wants once at
// allocation: the pad columns of every A operand and the pad rows of a short
// run come from here, and the kernels have no bounds checks.
func (b *Buffer) Zero() {
	clear(unsafe.Slice((*byte)(b.mapped), b.size))
}

// WriteUint16At copies src into the buffer's mapped memory at a uint16
// element offset. fp16 is the matrix cores' only operand type, so a weight
// arena is filled through this rather than WriteFloat32At; the caller does
// the narrowing, because the layout it writes into is usually a transpose or
// a retiling of the tensor it came from.
func (b *Buffer) WriteUint16At(off int, src []uint16) {
	dst := unsafe.Slice((*uint16)(b.mapped), off+len(src))
	copy(dst[off:], src)
}

// ReadUint16At reads n uint16s starting at a uint16 element offset.
func (b *Buffer) ReadUint16At(off, n int) []uint16 {
	src := unsafe.Slice((*uint16)(b.mapped), off+n)
	out := make([]uint16, n)
	copy(out, src[off:])
	return out
}

// ReadUint32At reads n uint32s starting at a uint32 element offset. Storage
// buffers that hold bit-packed data rather than numbers — a mask, an index
// list — are read through this instead of ReadFloat32At, so that nothing goes
// through a float on the way.
func (b *Buffer) ReadUint32At(off, n int) []uint32 {
	src := unsafe.Slice((*uint32)(b.mapped), off+n)
	out := make([]uint32, n)
	copy(out, src[off:])
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

// WriteBytesAt copies src into the buffer's mapped memory at a byte offset.
// A quantised weight bank is staged through this rather than through
// WriteFloat32At: its blocks are 24, 34, 144 or 176 bytes and nothing about
// them is a float, so the host copies the checkpoint's own bytes and the
// shader unpacks them (LLM.md L5b).
func (b *Buffer) WriteBytesAt(off int, src []byte) {
	dst := unsafe.Slice((*byte)(b.mapped), off+len(src))
	copy(dst[off:], src)
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
	Buffers []*Buffer
	// Counts, when set, groups Buffers into descriptor *arrays*: binding i
	// holds Counts[i] consecutive buffers, and the shader declares it as
	// `buffer B { ... } b[N]` and indexes it with a dynamically uniform
	// expression. Its sum must be len(Buffers). Nil is the plain
	// arrangement — one buffer per binding, in order — which is what every
	// kernel here but one uses.
	//
	// It exists because `maxStorageBufferRange` on this device is 4 GiB - 4
	// and the MoE bank is 77 GB (LLM.md L6a): the bank is one buffer a layer,
	// bound as one array, and the layer index selects.
	Counts           []uint32
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

	var countPtr *C.uint32_t
	var counts []C.uint32_t
	if len(spec.Counts) > 0 {
		total := 0
		for _, n := range spec.Counts {
			total += int(n)
		}
		if total != len(spec.Buffers) {
			return nil, fmt.Errorf("vk: PipelineSpec.Counts sums to %d, but there are %d buffers", total, len(spec.Buffers))
		}
		counts = make([]C.uint32_t, len(spec.Counts))
		for i, n := range spec.Counts {
			counts[i] = C.uint32_t(n)
		}
		countPtr = &counts[0]
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
			countPtr, C.uint32_t(len(spec.Counts)),
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

// MultiDispatch is one dispatch of a mixed-pipeline sequence: which pipeline
// runs, over what grid, with which push constants.
type MultiDispatch struct {
	Pipeline      *ComputePipeline
	GroupsX       uint32
	GroupsY       uint32
	PushConstants []byte
}

// DispatchMultiTimed is DispatchSequenceTimed for a sequence whose dispatches
// use *different pipelines* and *different grid extents*.
//
// It exists for the mixed-width MoE decode dispatch (IDEAS §1.12). The M block
// a grouped GEMV carries is compiled in, so covering a routing histogram with
// several widths at once is one dispatch per width, each binding the build for
// that width and naming its own contiguous slice of the shared slot table. The
// dispatches write disjoint output rows, so `barriers` is a knob rather than a
// requirement: false lets the widths overlap the way an engine would let them,
// true charges the split a compute->compute barrier between every pair, which
// is what it costs if the overlap cannot be had.
//
// Every pipeline must have been built over the same buffers and the same
// push-constant size — the recording uses the first one's command buffer,
// query pool and fence — and every push-constant block must be the same
// length, since each pipeline layout declares one range.
func DispatchMultiTimed(dispatches []MultiDispatch, groupsZ, iterations uint32, barriers bool) (time.Duration, error) {
	if len(dispatches) == 0 {
		return 0, fmt.Errorf("DispatchMultiTimed: empty dispatch sequence")
	}
	dev := dispatches[0].Pipeline.dev
	// The pipelines are copied by value rather than referenced: a
	// ShimComputePipeline holds nothing but Vulkan handles, so a Go slice of
	// them carries no Go pointers and can cross the cgo boundary, which a
	// slice of *C.ShimComputePipeline could not.
	handles := make([]C.ShimComputePipeline, len(dispatches))
	groupsX := make([]C.uint32_t, len(dispatches))
	groupsY := make([]C.uint32_t, len(dispatches))
	var flat []byte
	pcSize := len(dispatches[0].PushConstants)
	for i, d := range dispatches {
		if d.Pipeline == nil {
			return 0, fmt.Errorf("DispatchMultiTimed: dispatch %d has no pipeline", i)
		}
		if d.Pipeline.dev != dev {
			return 0, fmt.Errorf("DispatchMultiTimed: dispatch %d is on a different device", i)
		}
		if len(d.PushConstants) != pcSize {
			return 0, fmt.Errorf("DispatchMultiTimed: push-constant block %d is %d bytes, block 0 is %d",
				i, len(d.PushConstants), pcSize)
		}
		handles[i] = d.Pipeline.handle
		groupsX[i] = C.uint32_t(d.GroupsX)
		groupsY[i] = C.uint32_t(d.GroupsY)
		flat = append(flat, d.PushConstants...)
	}

	var start, end C.uint64_t
	var pcPtr unsafe.Pointer
	if len(flat) > 0 {
		pcPtr = unsafe.Pointer(&flat[0])
	}
	var barrierFlag C.uint32_t
	if barriers {
		barrierFlag = 1
	}
	if err := check("vkQueueSubmit", C.shim_dispatch_multi_timed(dev.handle, dev.queue,
		&handles[0],
		&groupsX[0], &groupsY[0], C.uint32_t(len(dispatches)),
		C.uint32_t(groupsZ), C.uint32_t(iterations), barrierFlag,
		pcPtr, C.uint32_t(pcSize), &start, &end)); err != nil {
		return 0, err
	}
	ticks := float64(uint64(end)) - float64(uint64(start))
	return time.Duration(ticks * dev.phys.TimestampPeriod), nil
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
