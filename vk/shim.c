#include "shim.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

VkResult shim_create_instance(const char *app_name, VkInstance *out_instance) {
    VkApplicationInfo appInfo = {0};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = app_name;
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.pEngineName = "strix-halo-vulkan";
    appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    return vkCreateInstance(&createInfo, NULL, out_instance);
}

void shim_destroy_instance(VkInstance instance) {
    vkDestroyInstance(instance, NULL);
}

VkResult shim_enumerate_physical_devices(VkInstance instance, ShimPhysicalDevice *out, uint32_t max, uint32_t *count) {
    uint32_t total = 0;
    VkResult r = vkEnumeratePhysicalDevices(instance, &total, NULL);
    if (r != VK_SUCCESS) {
        return r;
    }
    if (total > max) {
        total = max;
    }

    VkPhysicalDevice handles[64];
    if (total > 64) {
        total = 64;
    }
    r = vkEnumeratePhysicalDevices(instance, &total, handles);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        return r;
    }

    for (uint32_t i = 0; i < total; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(handles[i], &props);
        out[i].handle = handles[i];
        strncpy(out[i].name, props.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE);
        out[i].vendorID = props.vendorID;
        out[i].deviceID = props.deviceID;
        out[i].deviceType = props.deviceType;
        out[i].timestampPeriod = (double)props.limits.timestampPeriod;
    }
    *count = total;
    return VK_SUCCESS;
}

VkResult shim_find_compute_queue_family(VkPhysicalDevice phys, uint32_t *out_family) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, NULL);
    if (count == 0) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    VkQueueFamilyProperties families[64];
    if (count > 64) {
        count = 64;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, families);

    for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            *out_family = i;
            return VK_SUCCESS;
        }
    }
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult shim_query_device_features(VkPhysicalDevice phys, ShimDeviceFeatures *out) {
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeat = {0};
    coopFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;

    VkPhysicalDeviceShaderIntegerDotProductFeatures dotFeat = {0};
    dotFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    dotFeat.pNext = &coopFeat;

    VkPhysicalDeviceShaderFloat16Int8Features f16Feat = {0};
    f16Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    f16Feat.pNext = &dotFeat;

    VkPhysicalDeviceFeatures2 features2 = {0};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &f16Feat;

    vkGetPhysicalDeviceFeatures2(phys, &features2);

    out->float16 = f16Feat.shaderFloat16;
    out->int8 = f16Feat.shaderInt8;
    out->integerDotProduct = dotFeat.shaderIntegerDotProduct;
    out->coopMatrix = coopFeat.cooperativeMatrix;

    ShimSubgroupSizeControl sgs;
    shim_query_subgroup_size_control(phys, &sgs);
    out->subgroupSizeControl = sgs.supported;
    return VK_SUCCESS;
}

// True if `name` appears in this device's extension list. Needed because the
// instance is created at Vulkan 1.2, where subgroup size control is still an
// EXT rather than core — so the feature/property structs below are only
// meaningful once the extension is known to be there.
static VkBool32 shim_has_device_extension(VkPhysicalDevice phys, const char *name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, NULL, &count, NULL) != VK_SUCCESS || count == 0) {
        return VK_FALSE;
    }
    VkExtensionProperties *props = calloc(count, sizeof(*props));
    if (!props) {
        return VK_FALSE;
    }
    VkResult r = vkEnumerateDeviceExtensionProperties(phys, NULL, &count, props);
    VkBool32 found = VK_FALSE;
    if (r == VK_SUCCESS || r == VK_INCOMPLETE) {
        for (uint32_t i = 0; i < count; i++) {
            if (strcmp(props[i].extensionName, name) == 0) {
                found = VK_TRUE;
                break;
            }
        }
    }
    free(props);
    return found;
}

VkResult shim_query_subgroup_size_control(VkPhysicalDevice phys, ShimSubgroupSizeControl *out) {
    memset(out, 0, sizeof(*out));
    if (!shim_has_device_extension(phys, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
        return VK_SUCCESS;
    }

    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT feat = {0};
    feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    VkPhysicalDeviceFeatures2 features2 = {0};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &feat;
    vkGetPhysicalDeviceFeatures2(phys, &features2);

    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sizeProps = {0};
    sizeProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 props2 = {0};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &sizeProps;
    vkGetPhysicalDeviceProperties2(phys, &props2);

    // Two more conditions on top of the extension check above, all three of
    // which must hold before a pipeline may name a size: the feature bit, and
    // COMPUTE among the stages that accept one.
    out->supported = feat.subgroupSizeControl &&
                     (sizeProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) ? VK_TRUE : VK_FALSE;
    out->computeFullSubgroups = feat.computeFullSubgroups;
    out->minSubgroupSize = sizeProps.minSubgroupSize;
    out->maxSubgroupSize = sizeProps.maxSubgroupSize;
    out->maxComputeWorkgroupSubgroups = sizeProps.maxComputeWorkgroupSubgroups;
    return VK_SUCCESS;
}

VkResult shim_query_cooperative_matrix_properties(VkInstance instance, VkPhysicalDevice phys, ShimCoopMatProperty *out, uint32_t max, uint32_t *count) {
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR fn =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
            instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!fn) {
        *count = 0;
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    uint32_t total = 0;
    VkResult r = fn(phys, &total, NULL);
    if (r != VK_SUCCESS) {
        *count = 0;
        return r;
    }
    if (total == 0) {
        *count = 0;
        return VK_SUCCESS;
    }

    VkCooperativeMatrixPropertiesKHR *props = calloc(total, sizeof(*props));
    if (!props) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    for (uint32_t i = 0; i < total; i++) {
        props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    }

    r = fn(phys, &total, props);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        free(props);
        *count = 0;
        return r;
    }

    uint32_t n = total < max ? total : max;
    for (uint32_t i = 0; i < n; i++) {
        out[i].MSize = props[i].MSize;
        out[i].NSize = props[i].NSize;
        out[i].KSize = props[i].KSize;
        out[i].AType = (uint32_t)props[i].AType;
        out[i].BType = (uint32_t)props[i].BType;
        out[i].CType = (uint32_t)props[i].CType;
        out[i].ResultType = (uint32_t)props[i].ResultType;
        out[i].scope = (uint32_t)props[i].scope;
    }
    *count = n;
    free(props);
    return VK_SUCCESS;
}

VkResult shim_create_device(VkPhysicalDevice phys, uint32_t queueFamily, const ShimDeviceFeatures *request,
                             VkDevice *out_device, VkQueue *out_queue) {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {0};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    // Chained first so it is last in the pNext list; the driver ignores an
    // all-false struct, so this is harmless when the caller didn't ask.
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sizeFeat = {0};
    sizeFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    sizeFeat.subgroupSizeControl = request->subgroupSizeControl;
    sizeFeat.computeFullSubgroups = request->subgroupSizeControl;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeat = {0};
    coopFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopFeat.cooperativeMatrix = request->coopMatrix;
    if (request->subgroupSizeControl) {
        coopFeat.pNext = &sizeFeat;
    }

    VkPhysicalDeviceShaderIntegerDotProductFeatures dotFeat = {0};
    dotFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    dotFeat.pNext = &coopFeat;
    dotFeat.shaderIntegerDotProduct = request->integerDotProduct;

    VkPhysicalDevice16BitStorageFeatures storage16Feat = {0};
    storage16Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16Feat.pNext = &dotFeat;
    storage16Feat.storageBuffer16BitAccess = request->float16;
    storage16Feat.uniformAndStorageBuffer16BitAccess = request->float16;

    VkPhysicalDeviceShaderFloat16Int8Features f16Feat = {0};
    f16Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    f16Feat.pNext = &storage16Feat;
    f16Feat.shaderFloat16 = request->float16;
    f16Feat.shaderInt8 = request->int8;

    VkPhysicalDeviceFeatures2 features2 = {0};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &f16Feat;

    const char *extensions[4];
    uint32_t extCount = 0;
    if (request->coopMatrix) {
        extensions[extCount++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    }
    if (request->integerDotProduct) {
        extensions[extCount++] = VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
    }
    if (request->subgroupSizeControl) {
        extensions[extCount++] = VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
    }

    VkDeviceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &features2;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueInfo;
    createInfo.enabledExtensionCount = extCount;
    createInfo.ppEnabledExtensionNames = extCount ? extensions : NULL;
    // pEnabledFeatures left NULL: features are requested via the pNext chain above.

    VkResult r = vkCreateDevice(phys, &createInfo, NULL, out_device);
    if (r != VK_SUCCESS) {
        return r;
    }
    vkGetDeviceQueue(*out_device, queueFamily, 0, out_queue);
    return VK_SUCCESS;
}

void shim_destroy_device(VkDevice device) {
    vkDestroyDevice(device, NULL);
}

VkResult shim_device_wait_idle(VkDevice device) {
    return vkDeviceWaitIdle(device);
}

static VkResult find_memory_type(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags want, uint32_t *out) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if (!(typeBits & (1u << i))) {
            continue;
        }
        if ((props.memoryTypes[i].propertyFlags & want) == want) {
            *out = i;
            return VK_SUCCESS;
        }
    }
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult shim_create_storage_buffer(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size,
                                     VkBuffer *out_buffer, VkDeviceMemory *out_memory, void **out_mapped) {
    VkBufferCreateInfo bufInfo = {0};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(device, &bufInfo, NULL, out_buffer);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, *out_buffer, &reqs);

    // Prefer memory that is DEVICE_LOCAL as well as host-visible: on this
    // unified-memory APU that combination exists (true zero-copy, full
    // memory-controller bandwidth) and is strictly better than plain
    // HOST_VISIBLE|HOST_COHERENT, which can land in a slower, non-local heap.
    uint32_t memType;
    r = find_memory_type(phys, reqs.memoryTypeBits,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          &memType);
    if (r != VK_SUCCESS) {
        r = find_memory_type(phys, reqs.memoryTypeBits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &memType);
    }
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(device, *out_buffer, NULL);
        return r;
    }

    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = reqs.size;
    allocInfo.memoryTypeIndex = memType;

    r = vkAllocateMemory(device, &allocInfo, NULL, out_memory);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(device, *out_buffer, NULL);
        return r;
    }

    r = vkBindBufferMemory(device, *out_buffer, *out_memory, 0);
    if (r != VK_SUCCESS) {
        vkFreeMemory(device, *out_memory, NULL);
        vkDestroyBuffer(device, *out_buffer, NULL);
        return r;
    }

    r = vkMapMemory(device, *out_memory, 0, reqs.size, 0, out_mapped);
    if (r != VK_SUCCESS) {
        vkFreeMemory(device, *out_memory, NULL);
        vkDestroyBuffer(device, *out_buffer, NULL);
        return r;
    }

    return VK_SUCCESS;
}

VkResult shim_create_storage_buffer_of_type(VkDevice device, VkDeviceSize size, uint32_t memType,
                                             VkBuffer *out_buffer, VkDeviceMemory *out_memory, void **out_mapped) {
    VkBufferCreateInfo bufInfo = {0};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(device, &bufInfo, NULL, out_buffer);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, *out_buffer, &reqs);
    if (!(reqs.memoryTypeBits & (1u << memType))) {
        vkDestroyBuffer(device, *out_buffer, NULL);
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = reqs.size;
    allocInfo.memoryTypeIndex = memType;

    r = vkAllocateMemory(device, &allocInfo, NULL, out_memory);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(device, *out_buffer, NULL);
        return r;
    }
    r = vkBindBufferMemory(device, *out_buffer, *out_memory, 0);
    if (r != VK_SUCCESS) {
        vkFreeMemory(device, *out_memory, NULL);
        vkDestroyBuffer(device, *out_buffer, NULL);
        return r;
    }

    // A type without HOST_VISIBLE cannot be mapped; the caller gets NULL and
    // must fill the buffer some other way (or not at all, for a read probe).
    *out_mapped = NULL;
    r = vkMapMemory(device, *out_memory, 0, VK_WHOLE_SIZE, 0, out_mapped);
    if (r != VK_SUCCESS) {
        *out_mapped = NULL;
    }
    return VK_SUCCESS;
}

VkResult shim_memory_types(VkDevice device, VkPhysicalDevice phys, ShimMemoryType *out, uint32_t max, uint32_t *count) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    *count = props.memoryTypeCount;

    // "Can this type back a storage buffer?" is a property of a buffer, not
    // of the device, so ask a throwaway one. Any size works; the bits do not
    // depend on it.
    uint32_t bufferBits = 0;
    VkBufferCreateInfo bufInfo = {0};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = 4096;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer probe;
    if (vkCreateBuffer(device, &bufInfo, NULL, &probe) == VK_SUCCESS) {
        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(device, probe, &reqs);
        bufferBits = reqs.memoryTypeBits;
        vkDestroyBuffer(device, probe, NULL);
    }

    uint32_t n = props.memoryTypeCount < max ? props.memoryTypeCount : max;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t h = props.memoryTypes[i].heapIndex;
        out[i].index = i;
        out[i].heapIndex = h;
        out[i].propertyFlags = props.memoryTypes[i].propertyFlags;
        out[i].heapSize = props.memoryHeaps[h].size;
        out[i].heapFlags = props.memoryHeaps[h].flags;
        out[i].bufferCompatible = (bufferBits & (1u << i)) ? 1 : 0;
    }
    return VK_SUCCESS;
}

void shim_destroy_buffer(VkDevice device, VkBuffer buffer, VkDeviceMemory memory) {
    vkUnmapMemory(device, memory);
    vkFreeMemory(device, memory, NULL);
    vkDestroyBuffer(device, buffer, NULL);
}

VkResult shim_create_shader_module(VkDevice device, const void *code, size_t size, VkShaderModule *out) {
    VkShaderModuleCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = size;
    createInfo.pCode = (const uint32_t *)code;
    return vkCreateShaderModule(device, &createInfo, NULL, out);
}

void shim_destroy_shader_module(VkDevice device, VkShaderModule module) {
    vkDestroyShaderModule(device, module, NULL);
}

#define SHIM_MAX_BINDINGS 16
// A binding may hold an *array* of storage buffers, so the flat buffer list is
// far longer than the binding list: LLM.md L6a binds the MoE bank one buffer a
// layer, because `maxStorageBufferRange` is 4 GiB - 4 and the bank is 77.
#define SHIM_MAX_BUFFERS 256
#define SHIM_MAX_SPEC_CONSTANTS 32

VkResult shim_create_compute_pipeline(VkDevice device, VkShaderModule shader,
                                       const VkBuffer *buffers, uint32_t bufferCount,
                                       const uint32_t *counts, uint32_t bindingCount,
                                       uint32_t pushConstantSize,
                                       const ShimSpecConstant *specConstants, uint32_t specConstantCount,
                                       uint32_t requiredSubgroupSize,
                                       uint32_t queueFamily, ShimComputePipeline *out) {
    memset(out, 0, sizeof(*out));

    // counts == NULL is the plain arrangement every other kernel here uses:
    // one buffer per binding, in order.
    if (counts == NULL) {
        bindingCount = bufferCount;
    }
    if (bindingCount > SHIM_MAX_BINDINGS || bufferCount > SHIM_MAX_BUFFERS ||
        specConstantCount > SHIM_MAX_SPEC_CONSTANTS) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkDescriptorSetLayoutBinding bindings[SHIM_MAX_BINDINGS];
    uint32_t total = 0;
    for (uint32_t i = 0; i < bindingCount; i++) {
        uint32_t n = counts == NULL ? 1 : counts[i];
        if (n == 0) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        memset(&bindings[i], 0, sizeof(bindings[i]));
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = n;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        total += n;
    }
    if (total != bufferCount) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindingCount;
    layoutInfo.pBindings = bindings;

    VkResult r = vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &out->setLayout);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkPushConstantRange pcRange = {0};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = pushConstantSize;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &out->setLayout;
    if (pushConstantSize > 0) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pcRange;
    }

    r = vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &out->pipelineLayout);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkSpecializationMapEntry mapEntries[SHIM_MAX_SPEC_CONSTANTS];
    uint32_t specValues[SHIM_MAX_SPEC_CONSTANTS];
    VkSpecializationInfo specInfo = {0};
    for (uint32_t i = 0; i < specConstantCount; i++) {
        mapEntries[i].constantID = specConstants[i].constantID;
        mapEntries[i].offset = i * (uint32_t)sizeof(uint32_t);
        mapEntries[i].size = sizeof(uint32_t);
        specValues[i] = specConstants[i].value;
    }
    if (specConstantCount > 0) {
        specInfo.mapEntryCount = specConstantCount;
        specInfo.pMapEntries = mapEntries;
        specInfo.dataSize = specConstantCount * sizeof(uint32_t);
        specInfo.pData = specValues;
    }

    // IDEAS §6.2: pin the wave size rather than take the driver's default.
    // REQUIRE_FULL_SUBGROUPS goes with it because every shader here declares a
    // local_size_x that is an exact multiple of the size being asked for, and
    // the flag makes a future one that isn't fail at pipeline creation instead
    // of silently running a partly-inactive last wave through subgroup ops.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT sizeInfo = {0};
    sizeInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    sizeInfo.requiredSubgroupSize = requiredSubgroupSize;

    VkPipelineShaderStageCreateInfo stageInfo = {0};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shader;
    stageInfo.pName = "main";
    if (specConstantCount > 0) {
        stageInfo.pSpecializationInfo = &specInfo;
    }
    if (requiredSubgroupSize > 0) {
        stageInfo.pNext = &sizeInfo;
        stageInfo.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
    }

    VkComputePipelineCreateInfo pipelineInfo = {0};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = out->pipelineLayout;

    r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &out->pipeline);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkDescriptorPoolSize poolSize = {0};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = bufferCount;

    VkDescriptorPoolCreateInfo descPoolInfo = {0};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.maxSets = 1;
    descPoolInfo.poolSizeCount = 1;
    descPoolInfo.pPoolSizes = &poolSize;

    r = vkCreateDescriptorPool(device, &descPoolInfo, NULL, &out->descPool);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkDescriptorSetAllocateInfo descSetAllocInfo = {0};
    descSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descSetAllocInfo.descriptorPool = out->descPool;
    descSetAllocInfo.descriptorSetCount = 1;
    descSetAllocInfo.pSetLayouts = &out->setLayout;

    r = vkAllocateDescriptorSets(device, &descSetAllocInfo, &out->descSet);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkDescriptorBufferInfo bufInfos[SHIM_MAX_BUFFERS];
    VkWriteDescriptorSet writes[SHIM_MAX_BINDINGS];
    for (uint32_t i = 0; i < bufferCount; i++) {
        memset(&bufInfos[i], 0, sizeof(bufInfos[i]));
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].offset = 0;
        bufInfos[i].range = VK_WHOLE_SIZE;
    }
    uint32_t first = 0;
    for (uint32_t i = 0; i < bindingCount; i++) {
        uint32_t n = counts == NULL ? 1 : counts[i];
        memset(&writes[i], 0, sizeof(writes[i]));
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = out->descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = n;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[first];
        first += n;
    }
    if (bindingCount > 0) {
        vkUpdateDescriptorSets(device, bindingCount, writes, 0, NULL);
    }

    VkCommandPoolCreateInfo cmdPoolInfo = {0};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = queueFamily;

    r = vkCreateCommandPool(device, &cmdPoolInfo, NULL, &out->cmdPool);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkCommandBufferAllocateInfo cmdBufAllocInfo = {0};
    cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.commandPool = out->cmdPool;
    cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandBufferCount = 1;

    r = vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &out->cmdBuf);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkFenceCreateInfo fenceInfo = {0};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(device, &fenceInfo, NULL, &out->fence);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkQueryPoolCreateInfo queryPoolInfo = {0};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    // Big enough to mark every dispatch of a whole forward pass, not just the
    // two ends: see shim_dispatch_multi_timed's out_marks. Only the pipeline
    // that records a sequence uses more than two of them, and which one that
    // is depends on the caller, so they all carry the pool. It is 16 KB a
    // pipeline.
    queryPoolInfo.queryCount = SHIM_QUERY_SLOTS;
    return vkCreateQueryPool(device, &queryPoolInfo, NULL, &out->queryPool);
}

// SHIM_QUERY_TIMEOUT_NS bounds the wait for a pass's timestamps. It is not a
// performance knob: the fence is already signalled by the time anything below
// runs, so a slot that is not ready within ten seconds is never going to be.
#define SHIM_QUERY_TIMEOUT_NS 10000000000ULL

static uint64_t shim_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// shim_query_results_deadline reads `marks` timestamps, waiting but not
// forever.
//
// VK_QUERY_RESULT_WAIT_BIT looks like a driver-side wait and is not one on
// RADV: radv_GetQueryPoolResults polls the mapped slot in userspace, with no
// timeout and no way out, so a slot the GPU never wrote spins a core at 100%
// on a queue that has gone idle and a fence that has already signalled. That
// is what >2560 rows did (LLM.md P0) and why the submit's own 20-second fence
// timeout never fired -- the wait it would have bounded was already over.
// Polling to a deadline turns that hang into an error, and names the slots.
static VkResult shim_query_results_deadline(VkDevice device, VkQueryPool pool, uint32_t marks,
                                             uint64_t *out) {
    uint64_t deadline = shim_now_ns() + SHIM_QUERY_TIMEOUT_NS;
    for (;;) {
        VkResult r = vkGetQueryPoolResults(device, pool, 0, marks, (size_t)marks * sizeof(uint64_t),
                                            out, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        if (r != VK_NOT_READY) {
            return r;
        }
        if (shim_now_ns() >= deadline) {
            uint32_t missing = 0;
            uint32_t first = UINT32_MAX, last = 0;
            for (uint32_t i = 0; i < marks; i++) {
                uint64_t v = 0;
                if (vkGetQueryPoolResults(device, pool, i, 1, sizeof(v), &v, sizeof(v),
                                           VK_QUERY_RESULT_64_BIT) == VK_NOT_READY) {
                    if (first == UINT32_MAX) {
                        first = i;
                    }
                    last = i;
                    missing++;
                }
            }
            fprintf(stderr, "shim: %u of %u timestamp slots never became ready (first %u, last %u)\n",
                    missing, marks, first, last);
            return VK_TIMEOUT;
        }
    }
}

VkResult shim_dispatch_timed(VkDevice device, VkQueue queue, const ShimComputePipeline *p,
                              uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ, uint32_t iterations,
                              const void *pushConstants, uint32_t pushConstantSize,
                              uint64_t *out_start, uint64_t *out_end) {
    if (iterations == 0) {
        iterations = 1;
    }

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VkResult r = vkBeginCommandBuffer(p->cmdBuf, &beginInfo);
    if (r != VK_SUCCESS) {
        return r;
    }

    vkCmdResetQueryPool(p->cmdBuf, p->queryPool, 0, 2);
    vkCmdBindPipeline(p->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    vkCmdBindDescriptorSets(p->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipelineLayout, 0, 1, &p->descSet, 0, NULL);
    if (pushConstantSize > 0 && pushConstants != NULL) {
        vkCmdPushConstants(p->cmdBuf, p->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, pushConstants);
    }

    vkCmdWriteTimestamp(p->cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, p->queryPool, 0);

    VkMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    for (uint32_t i = 0; i < iterations; i++) {
        vkCmdDispatch(p->cmdBuf, groupsX, groupsY, groupsZ);
        if (i + 1 < iterations) {
            vkCmdPipelineBarrier(p->cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &barrier, 0, NULL, 0, NULL);
        }
    }

    vkCmdWriteTimestamp(p->cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->queryPool, 1);

    r = vkEndCommandBuffer(p->cmdBuf);
    if (r != VK_SUCCESS) {
        return r;
    }

    r = vkResetFences(device, 1, &p->fence);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &p->cmdBuf;

    r = vkQueueSubmit(queue, 1, &submitInfo, p->fence);
    if (r != VK_SUCCESS) {
        return r;
    }

    r = vkWaitForFences(device, 1, &p->fence, VK_TRUE, 20000000000ULL);
    if (r != VK_SUCCESS) {
        return r;
    }

    uint64_t timestamps[2];
    r = shim_query_results_deadline(device, p->queryPool, 2, timestamps);
    if (r != VK_SUCCESS) {
        return r;
    }

    *out_start = timestamps[0];
    *out_end = timestamps[1];
    return VK_SUCCESS;
}

VkResult shim_dispatch_seq_timed(VkDevice device, VkQueue queue, const ShimComputePipeline *p,
                                  const uint32_t *groupsX, uint32_t count,
                                  uint32_t groupsY, uint32_t groupsZ, uint32_t iterations,
                                  const void *pushConstants, uint32_t pushConstantSize,
                                  uint64_t *out_start, uint64_t *out_end) {
    if (iterations == 0) {
        iterations = 1;
    }
    if (count == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VkResult r = vkBeginCommandBuffer(p->cmdBuf, &beginInfo);
    if (r != VK_SUCCESS) {
        return r;
    }

    vkCmdResetQueryPool(p->cmdBuf, p->queryPool, 0, 2);
    vkCmdBindPipeline(p->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    vkCmdBindDescriptorSets(p->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipelineLayout, 0, 1, &p->descSet, 0, NULL);

    vkCmdWriteTimestamp(p->cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, p->queryPool, 0);

    VkMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    const uint8_t *pc = (const uint8_t *)pushConstants;
    for (uint32_t it = 0; it < iterations; it++) {
        for (uint32_t i = 0; i < count; i++) {
            if (pushConstantSize > 0 && pc != NULL) {
                vkCmdPushConstants(p->cmdBuf, p->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    pushConstantSize, pc + (size_t)i * pushConstantSize);
            }
            vkCmdDispatch(p->cmdBuf, groupsX[i], groupsY, groupsZ);
            if (it + 1 < iterations || i + 1 < count) {
                vkCmdPipelineBarrier(p->cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      0, 1, &barrier, 0, NULL, 0, NULL);
            }
        }
    }

    vkCmdWriteTimestamp(p->cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->queryPool, 1);

    r = vkEndCommandBuffer(p->cmdBuf);
    if (r != VK_SUCCESS) {
        return r;
    }

    r = vkResetFences(device, 1, &p->fence);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &p->cmdBuf;

    r = vkQueueSubmit(queue, 1, &submitInfo, p->fence);
    if (r != VK_SUCCESS) {
        return r;
    }

    r = vkWaitForFences(device, 1, &p->fence, VK_TRUE, 20000000000ULL);
    if (r != VK_SUCCESS) {
        return r;
    }

    uint64_t timestamps[2];
    r = shim_query_results_deadline(device, p->queryPool, 2, timestamps);
    if (r != VK_SUCCESS) {
        return r;
    }

    *out_start = timestamps[0];
    *out_end = timestamps[1];
    return VK_SUCCESS;
}

VkResult shim_dispatch_multi_timed(VkDevice device, VkQueue queue, const ShimComputePipeline *pipes,
                                    const uint32_t *groupsX, const uint32_t *groupsY, uint32_t count,
                                    uint32_t groupsZ, uint32_t iterations, uint32_t barriers,
                                    const void *pushConstants, uint32_t pushConstantSize,
                                    uint64_t *out_start, uint64_t *out_end, uint64_t *out_marks) {
    if (iterations == 0) {
        iterations = 1;
    }
    if (count == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    // Per-dispatch marks are only meaningful for a single pass, and the pool
    // bounds how many there can be. Either condition falls back to the two
    // ends, which is what every caller before L7d asked for.
    uint32_t marks = (out_marks != NULL && iterations == 1 && count + 1 <= SHIM_QUERY_SLOTS) ? count + 1 : 0;

    // The recording lives in the first pipeline's command buffer; every other
    // pipeline contributes only its VkPipeline, layout and descriptor set.
    const ShimComputePipeline *rec = &pipes[0];

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VkResult r = vkBeginCommandBuffer(rec->cmdBuf, &beginInfo);
    if (r != VK_SUCCESS) {
        return r;
    }

    vkCmdResetQueryPool(rec->cmdBuf, rec->queryPool, 0, marks > 0 ? marks : 2);
    vkCmdWriteTimestamp(rec->cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, rec->queryPool, 0);

    VkMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    const uint8_t *pc = (const uint8_t *)pushConstants;
    for (uint32_t it = 0; it < iterations; it++) {
        for (uint32_t i = 0; i < count; i++) {
            const ShimComputePipeline *p = &pipes[i];
            vkCmdBindPipeline(rec->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
            vkCmdBindDescriptorSets(rec->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipelineLayout, 0, 1,
                                     &p->descSet, 0, NULL);
            if (pushConstantSize > 0 && pc != NULL) {
                vkCmdPushConstants(rec->cmdBuf, p->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    pushConstantSize, pc + (size_t)i * pushConstantSize);
            }
            vkCmdDispatch(rec->cmdBuf, groupsX[i], groupsY[i], groupsZ);
            if (marks > 0) {
                vkCmdWriteTimestamp(rec->cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, rec->queryPool, i + 1);
            }
            // Within an iteration the barrier is the caller's choice; between
            // iterations it is not, since the next iteration overwrites what
            // this one wrote.
            int last = (i + 1 == count);
            if ((!last && barriers) || (last && it + 1 < iterations)) {
                vkCmdPipelineBarrier(rec->cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
            }
        }
    }

    if (marks == 0) {
        vkCmdWriteTimestamp(rec->cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, rec->queryPool, 1);
    }

    r = vkEndCommandBuffer(rec->cmdBuf);
    if (r != VK_SUCCESS) {
        return r;
    }

    r = vkResetFences(device, 1, &rec->fence);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &rec->cmdBuf;

    r = vkQueueSubmit(queue, 1, &submitInfo, rec->fence);
    if (r != VK_SUCCESS) {
        return r;
    }

    r = vkWaitForFences(device, 1, &rec->fence, VK_TRUE, 20000000000ULL);
    if (r != VK_SUCCESS) {
        return r;
    }

    if (marks > 0) {
        r = shim_query_results_deadline(device, rec->queryPool, marks, out_marks);
        if (r != VK_SUCCESS) {
            return r;
        }
        *out_start = out_marks[0];
        *out_end = out_marks[marks - 1];
        return VK_SUCCESS;
    }

    uint64_t timestamps[2];
    r = shim_query_results_deadline(device, rec->queryPool, 2, timestamps);
    if (r != VK_SUCCESS) {
        return r;
    }

    *out_start = timestamps[0];
    *out_end = timestamps[1];
    return VK_SUCCESS;
}

VkResult shim_prerecord_multi(VkDevice device, uint32_t queueFamily, const ShimComputePipeline *pipes,
                               const uint32_t *groupsX, const uint32_t *groupsY, uint32_t count,
                               uint32_t barriers, uint32_t wantMarks,
                               const void *pushConstants, uint32_t pushConstantSize,
                               ShimPrerecorded *out) {
    if (count == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    memset(out, 0, sizeof(*out));
    out->marks = wantMarks ? count + 1 : 2;

    VkCommandPoolCreateInfo cmdPoolInfo = {0};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = queueFamily;
    VkResult r = vkCreateCommandPool(device, &cmdPoolInfo, NULL, &out->cmdPool);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkCommandBufferAllocateInfo cmdBufAllocInfo = {0};
    cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.commandPool = out->cmdPool;
    cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &out->cmdBuf);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkFenceCreateInfo fenceInfo = {0};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(device, &fenceInfo, NULL, &out->fence);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkQueryPoolCreateInfo queryPoolInfo = {0};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = out->marks;
    r = vkCreateQueryPool(device, &queryPoolInfo, NULL, &out->queryPool);
    if (r != VK_SUCCESS) {
        return r;
    }

    // The recording is shim_dispatch_multi_timed's, one iteration, minus the
    // submit. The query-pool reset is *inside* the command buffer, so every
    // replay re-arms and re-writes the same slots.
    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    r = vkBeginCommandBuffer(out->cmdBuf, &beginInfo);
    if (r != VK_SUCCESS) {
        return r;
    }

    vkCmdResetQueryPool(out->cmdBuf, out->queryPool, 0, out->marks);
    vkCmdWriteTimestamp(out->cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, out->queryPool, 0);

    VkMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    const uint8_t *pc = (const uint8_t *)pushConstants;
    for (uint32_t i = 0; i < count; i++) {
        const ShimComputePipeline *p = &pipes[i];
        vkCmdBindPipeline(out->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
        vkCmdBindDescriptorSets(out->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipelineLayout, 0, 1,
                                 &p->descSet, 0, NULL);
        if (pushConstantSize > 0 && pc != NULL) {
            vkCmdPushConstants(out->cmdBuf, p->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                pushConstantSize, pc + (size_t)i * pushConstantSize);
        }
        vkCmdDispatch(out->cmdBuf, groupsX[i], groupsY[i], 1);
        if (wantMarks) {
            vkCmdWriteTimestamp(out->cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, out->queryPool, i + 1);
        }
        if (barriers && i + 1 < count) {
            vkCmdPipelineBarrier(out->cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
        }
    }

    if (!wantMarks) {
        vkCmdWriteTimestamp(out->cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, out->queryPool, 1);
    }
    return vkEndCommandBuffer(out->cmdBuf);
}

VkResult shim_submit_prerecorded(VkDevice device, VkQueue queue, const ShimPrerecorded *p,
                                  uint64_t *out_ticks) {
    VkResult r = vkResetFences(device, 1, &p->fence);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &p->cmdBuf;
    r = vkQueueSubmit(queue, 1, &submitInfo, p->fence);
    if (r != VK_SUCCESS) {
        return r;
    }

    r = vkWaitForFences(device, 1, &p->fence, VK_TRUE, 20000000000ULL);
    if (r != VK_SUCCESS) {
        return r;
    }

    return shim_query_results_deadline(device, p->queryPool, p->marks, out_ticks);
}

void shim_destroy_prerecorded(VkDevice device, ShimPrerecorded *p) {
    if (p->queryPool) vkDestroyQueryPool(device, p->queryPool, NULL);
    if (p->fence) vkDestroyFence(device, p->fence, NULL);
    if (p->cmdPool) vkDestroyCommandPool(device, p->cmdPool, NULL);
    memset(p, 0, sizeof(*p));
}

void shim_destroy_compute_pipeline(VkDevice device, ShimComputePipeline *p) {
    if (p->queryPool) vkDestroyQueryPool(device, p->queryPool, NULL);
    if (p->fence) vkDestroyFence(device, p->fence, NULL);
    if (p->cmdPool) vkDestroyCommandPool(device, p->cmdPool, NULL);
    if (p->descPool) vkDestroyDescriptorPool(device, p->descPool, NULL);
    if (p->pipeline) vkDestroyPipeline(device, p->pipeline, NULL);
    if (p->pipelineLayout) vkDestroyPipelineLayout(device, p->pipelineLayout, NULL);
    if (p->setLayout) vkDestroyDescriptorSetLayout(device, p->setLayout, NULL);
}
