// Small C shim around the Vulkan compute path used by this program.
//
// Why this exists: Go's cgo forbids passing a Go struct across the C
// boundary when one of its fields is itself a pointer into other Go memory
// (the "Go pointer to Go pointer" check). Vulkan's *CreateInfo structs are
// built almost entirely out of such chains (pApplicationInfo,
// pQueueCreateInfos, pBindings, ...), so building them in Go and taking
// their address would trip that check on nearly every call. Building them
// here instead means every pointer crossing the Go/C boundary is either an
// opaque Vulkan handle or a flat byte buffer — both of which cgo allows.
#ifndef STRIX_HALO_VULKAN_SHIM_H
#define STRIX_HALO_VULKAN_SHIM_H

#include <vulkan/vulkan.h>

typedef struct {
    VkPhysicalDevice handle;
    char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint32_t vendorID;
    uint32_t deviceID;
    VkPhysicalDeviceType deviceType;
} ShimPhysicalDevice;

typedef struct {
    VkDescriptorSetLayout setLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;
    VkCommandPool cmdPool;
    VkCommandBuffer cmdBuf;
    VkFence fence;
} ShimComputePipeline;

VkResult shim_create_instance(const char *app_name, VkInstance *out_instance);
void shim_destroy_instance(VkInstance instance);

VkResult shim_enumerate_physical_devices(VkInstance instance, ShimPhysicalDevice *out, uint32_t max, uint32_t *count);
VkResult shim_find_compute_queue_family(VkPhysicalDevice phys, uint32_t *out_family);

VkResult shim_create_device(VkPhysicalDevice phys, uint32_t queueFamily, VkDevice *out_device, VkQueue *out_queue);
void shim_destroy_device(VkDevice device);
VkResult shim_device_wait_idle(VkDevice device);

// Allocates a host-visible, host-coherent storage buffer and leaves it
// mapped for the caller (there's no discrete VRAM to stage through on an
// APU, so this is the natural way to move data).
VkResult shim_create_storage_buffer(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size,
                                     VkBuffer *out_buffer, VkDeviceMemory *out_memory, void **out_mapped);
void shim_destroy_buffer(VkDevice device, VkBuffer buffer, VkDeviceMemory memory);

VkResult shim_create_shader_module(VkDevice device, const void *code, size_t size, VkShaderModule *out);
void shim_destroy_shader_module(VkDevice device, VkShaderModule module);

// Builds a full compute pipeline bound to a single storage-buffer binding
// (binding 0), matching shaders/double.comp.
VkResult shim_create_compute_pipeline(VkDevice device, VkShaderModule shader, VkBuffer buffer,
                                       uint32_t queueFamily, ShimComputePipeline *out);
VkResult shim_dispatch(VkDevice device, VkQueue queue, const ShimComputePipeline *p, uint32_t groupsX);
void shim_destroy_compute_pipeline(VkDevice device, ShimComputePipeline *p);

#endif
