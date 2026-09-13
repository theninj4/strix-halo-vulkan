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
    double timestampPeriod; // nanoseconds per timestamp tick (VkPhysicalDeviceLimits.timestampPeriod)
} ShimPhysicalDevice;

// Optional compute-relevant device features a caller may request/query.
// Maps onto VkPhysicalDeviceShaderFloat16Int8Features, 16BitStorageFeatures,
// ShaderIntegerDotProductFeatures and CooperativeMatrixFeaturesKHR.
typedef struct {
    VkBool32 float16;
    VkBool32 int8;
    VkBool32 integerDotProduct;
    VkBool32 coopMatrix;
    VkBool32 subgroupSizeControl;
} ShimDeviceFeatures;

// What VK_EXT_subgroup_size_control will let a *compute* pipeline ask for
// (IDEAS.md §6.2). `supported` folds together three separate conditions the
// caller would otherwise have to check by hand — the device extension is
// present, the subgroupSizeControl feature is set, and COMPUTE is in
// requiredSubgroupSizeStages — because a pipeline may only pass a
// requiredSubgroupSize when all three hold.
typedef struct {
    VkBool32 supported;
    VkBool32 computeFullSubgroups;
    uint32_t minSubgroupSize;
    uint32_t maxSubgroupSize;
    uint32_t maxComputeWorkgroupSubgroups;
} ShimSubgroupSizeControl;

// Mirrors VkCooperativeMatrixPropertiesKHR (AType/BType/CType/ResultType are
// VkComponentTypeKHR values, scope is a VkScopeKHR value) as flat uint32s so
// this struct has no pointer members and is safe to hand back over cgo.
typedef struct {
    uint32_t MSize;
    uint32_t NSize;
    uint32_t KSize;
    uint32_t AType;
    uint32_t BType;
    uint32_t CType;
    uint32_t ResultType;
    uint32_t scope;
} ShimCoopMatProperty;

// A single specialization constant: constantID -> a 4-byte uint32 value.
// Sufficient for the workgroup/tile-size and cooperative-matrix M/N/K
// tuning this benchmark suite needs.
typedef struct {
    uint32_t constantID;
    uint32_t value;
} ShimSpecConstant;

typedef struct {
    VkDescriptorSetLayout setLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;
    VkCommandPool cmdPool;
    VkCommandBuffer cmdBuf;
    VkFence fence;
    VkQueryPool queryPool; // 2 timestamp queries: dispatch-block start/end
} ShimComputePipeline;

VkResult shim_create_instance(const char *app_name, VkInstance *out_instance);
void shim_destroy_instance(VkInstance instance);

VkResult shim_enumerate_physical_devices(VkInstance instance, ShimPhysicalDevice *out, uint32_t max, uint32_t *count);
VkResult shim_find_compute_queue_family(VkPhysicalDevice phys, uint32_t *out_family);

// Queries which of ShimDeviceFeatures the physical device actually supports
// (independent of what's been enabled on any logical device).
VkResult shim_query_device_features(VkPhysicalDevice phys, ShimDeviceFeatures *out);

// Queries the subgroup sizes a compute pipeline on this device may require.
// Always fills *out; out->supported == VK_FALSE means requiredSubgroupSize
// must be left at 0 and the driver's default wave size is all there is.
VkResult shim_query_subgroup_size_control(VkPhysicalDevice phys, ShimSubgroupSizeControl *out);

// Enumerates the MxNxK/type/scope combinations VK_KHR_cooperative_matrix
// supports on this device. Returns VK_SUCCESS with *count == 0 if the
// extension isn't supported at all (callers should check
// shim_query_device_features first). instance is needed because this is a
// KHR extension function the loader doesn't export as a link-time symbol —
// it's resolved at runtime via vkGetInstanceProcAddr.
VkResult shim_query_cooperative_matrix_properties(VkInstance instance, VkPhysicalDevice phys, ShimCoopMatProperty *out, uint32_t max, uint32_t *count);

// Creates a logical device with exactly the optional features set in
// request enabled (plus VK_KHR_cooperative_matrix as a device extension
// when request->coopMatrix is set — float16/int8/integer-dot-product are
// core as of the Vulkan 1.2 instance this app requests, so no extension
// strings are needed for those).
VkResult shim_create_device(VkPhysicalDevice phys, uint32_t queueFamily, const ShimDeviceFeatures *request,
                             VkDevice *out_device, VkQueue *out_queue);
void shim_destroy_device(VkDevice device);
VkResult shim_device_wait_idle(VkDevice device);

// Allocates a storage buffer preferring DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT
// memory (true zero-copy on this unified-memory APU) and falling back to
// HOST_VISIBLE|HOST_COHERENT if no such memory type exists.
VkResult shim_create_storage_buffer(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size,
                                     VkBuffer *out_buffer, VkDeviceMemory *out_memory, void **out_mapped);
void shim_destroy_buffer(VkDevice device, VkBuffer buffer, VkDeviceMemory memory);

VkResult shim_create_shader_module(VkDevice device, const void *code, size_t size, VkShaderModule *out);
void shim_destroy_shader_module(VkDevice device, VkShaderModule module);

// Builds a full compute pipeline bound to `bufferCount` storage-buffer
// bindings (0..bufferCount-1, one per entry in `buffers`), an optional
// push-constant range of `pushConstantSize` bytes, and an optional set of
// uint32 specialization constants.
//
// requiredSubgroupSize, when non-zero, pins the wave size this shader runs at
// via VK_EXT_subgroup_size_control (IDEAS.md §6.2) instead of letting the
// driver pick. It must be a power of two within the device's
// [minSubgroupSize, maxSubgroupSize] and the shader's local_size_x must be a
// multiple of it; the device must have been created with
// ShimDeviceFeatures.subgroupSizeControl. Zero keeps the driver's default,
// byte for byte the same pipeline this shim built before the knob existed.
VkResult shim_create_compute_pipeline(VkDevice device, VkShaderModule shader,
                                       const VkBuffer *buffers, uint32_t bufferCount,
                                       uint32_t pushConstantSize,
                                       const ShimSpecConstant *specConstants, uint32_t specConstantCount,
                                       uint32_t requiredSubgroupSize,
                                       uint32_t queueFamily, ShimComputePipeline *out);

// Records, in one command buffer: a timestamp, `iterations` dispatches of
// (groupsX,groupsY,groupsZ) workgroups each (with a compute->compute
// barrier between iterations so in-place steady-state throughput is what
// gets measured), and a closing timestamp; submits it and blocks until
// done. out_start/out_end are raw timestamp-query ticks — multiply their
// difference by ShimPhysicalDevice.timestampPeriod for nanoseconds.
VkResult shim_dispatch_timed(VkDevice device, VkQueue queue, const ShimComputePipeline *p,
                              uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ, uint32_t iterations,
                              const void *pushConstants, uint32_t pushConstantSize,
                              uint64_t *out_start, uint64_t *out_end);

void shim_destroy_compute_pipeline(VkDevice device, ShimComputePipeline *p);

#endif
