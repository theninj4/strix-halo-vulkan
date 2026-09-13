#include "shim.h"

#include <stdlib.h>
#include <string.h>

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

#define SHIM_MAX_BINDINGS 8
#define SHIM_MAX_SPEC_CONSTANTS 32

VkResult shim_create_compute_pipeline(VkDevice device, VkShaderModule shader,
                                       const VkBuffer *buffers, uint32_t bufferCount,
                                       uint32_t pushConstantSize,
                                       const ShimSpecConstant *specConstants, uint32_t specConstantCount,
                                       uint32_t requiredSubgroupSize,
                                       uint32_t queueFamily, ShimComputePipeline *out) {
    memset(out, 0, sizeof(*out));

    if (bufferCount > SHIM_MAX_BINDINGS || specConstantCount > SHIM_MAX_SPEC_CONSTANTS) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkDescriptorSetLayoutBinding bindings[SHIM_MAX_BINDINGS];
    for (uint32_t i = 0; i < bufferCount; i++) {
        memset(&bindings[i], 0, sizeof(bindings[i]));
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bufferCount;
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

    VkDescriptorBufferInfo bufInfos[SHIM_MAX_BINDINGS];
    VkWriteDescriptorSet writes[SHIM_MAX_BINDINGS];
    for (uint32_t i = 0; i < bufferCount; i++) {
        memset(&bufInfos[i], 0, sizeof(bufInfos[i]));
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].offset = 0;
        bufInfos[i].range = VK_WHOLE_SIZE;

        memset(&writes[i], 0, sizeof(writes[i]));
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = out->descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    if (bufferCount > 0) {
        vkUpdateDescriptorSets(device, bufferCount, writes, 0, NULL);
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
    queryPoolInfo.queryCount = 2;
    return vkCreateQueryPool(device, &queryPoolInfo, NULL, &out->queryPool);
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
    r = vkGetQueryPoolResults(device, p->queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t),
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r != VK_SUCCESS) {
        return r;
    }

    *out_start = timestamps[0];
    *out_end = timestamps[1];
    return VK_SUCCESS;
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
