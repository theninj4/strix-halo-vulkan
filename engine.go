// This file is a minimal Go wrapper around shim.c/shim.h, which does the
// actual Vulkan calls. See shim.h for why the struct-building work lives in
// C rather than here.
package main

/*
#cgo pkg-config: vulkan
#include "shim.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
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

// PhysicalDevice wraps a VkPhysicalDevice plus its cached properties.
type PhysicalDevice struct {
	handle     C.VkPhysicalDevice
	Name       string
	VendorID   uint32
	DeviceID   uint32
	DeviceType C.VkPhysicalDeviceType
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
			handle:     d.handle,
			Name:       C.GoString(&d.name[0]),
			VendorID:   uint32(d.vendorID),
			DeviceID:   uint32(d.deviceID),
			DeviceType: d.deviceType,
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

// Device wraps a VkDevice and the single compute queue we pulled from it.
type Device struct {
	handle      C.VkDevice
	phys        *PhysicalDevice
	queue       C.VkQueue
	queueFamily uint32
}

// NewDevice creates a logical device on phys with a single queue drawn from
// queueFamily.
func NewDevice(phys *PhysicalDevice, queueFamily uint32) (*Device, error) {
	var handle C.VkDevice
	var queue C.VkQueue
	if err := check("vkCreateDevice",
		C.shim_create_device(phys.handle, C.uint32_t(queueFamily), &handle, &queue)); err != nil {
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

// Buffer is a VkBuffer bound to host-visible, host-coherent memory, mapped
// for the lifetime of the buffer.
type Buffer struct {
	dev    *Device
	handle C.VkBuffer
	memory C.VkDeviceMemory
	mapped unsafe.Pointer
}

// NewBuffer allocates a buffer of size bytes usable as a storage buffer in a
// compute shader, backed by host-visible coherent memory and left mapped.
func (d *Device) NewBuffer(size int) (*Buffer, error) {
	var handle C.VkBuffer
	var memory C.VkDeviceMemory
	var mapped unsafe.Pointer
	if err := check("vkCreateBuffer/vkAllocateMemory",
		C.shim_create_storage_buffer(d.handle, d.phys.handle, C.VkDeviceSize(size), &handle, &memory, &mapped)); err != nil {
		return nil, err
	}
	return &Buffer{dev: d, handle: handle, memory: memory, mapped: mapped}, nil
}

// Destroy unmaps and releases the buffer's memory and handle.
func (b *Buffer) Destroy() {
	C.shim_destroy_buffer(b.dev.handle, b.handle, b.memory)
}

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

// ComputePipeline is a full pipeline bound to a single storage-buffer
// binding (binding 0), matching shaders/double.comp.
type ComputePipeline struct {
	dev    *Device
	handle C.ShimComputePipeline
}

// NewComputePipeline builds the descriptor set layout, pipeline, descriptor
// set and command buffer needed to dispatch shader against buf.
func (d *Device) NewComputePipeline(shader *ShaderModule, buf *Buffer) (*ComputePipeline, error) {
	p := &ComputePipeline{dev: d}
	if err := check("vkCreateComputePipelines",
		C.shim_create_compute_pipeline(d.handle, shader.handle, buf.handle, C.uint32_t(d.queueFamily), &p.handle)); err != nil {
		p.Destroy()
		return nil, err
	}
	return p, nil
}

// Dispatch records and submits a single dispatch covering groupsX workgroups
// on the X axis, then blocks until it completes.
func (p *ComputePipeline) Dispatch(groupsX uint32) error {
	return check("vkQueueSubmit", C.shim_dispatch(p.dev.handle, p.dev.queue, &p.handle, C.uint32_t(groupsX)))
}

// Destroy releases every object the pipeline owns.
func (p *ComputePipeline) Destroy() {
	C.shim_destroy_compute_pipeline(p.dev.handle, &p.handle)
}
