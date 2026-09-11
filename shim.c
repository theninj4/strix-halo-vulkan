#include "shim.h"

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

VkResult shim_create_device(VkPhysicalDevice phys, uint32_t queueFamily, VkDevice *out_device, VkQueue *out_queue) {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {0};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueInfo;

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

    uint32_t memType;
    r = find_memory_type(phys, reqs.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &memType);
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

VkResult shim_create_compute_pipeline(VkDevice device, VkShaderModule shader, VkBuffer buffer,
                                       uint32_t queueFamily, ShimComputePipeline *out) {
    memset(out, 0, sizeof(*out));

    VkDescriptorSetLayoutBinding binding = {0};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    VkResult r = vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &out->setLayout);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &out->setLayout;

    r = vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &out->pipelineLayout);
    if (r != VK_SUCCESS) {
        return r;
    }

    VkPipelineShaderStageCreateInfo stageInfo = {0};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shader;
    stageInfo.pName = "main";

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
    poolSize.descriptorCount = 1;

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

    VkDescriptorBufferInfo bufInfo = {0};
    bufInfo.buffer = buffer;
    bufInfo.offset = 0;
    bufInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write = {0};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = out->descSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);

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

    return vkCreateFence(device, &fenceInfo, NULL, &out->fence);
}

VkResult shim_dispatch(VkDevice device, VkQueue queue, const ShimComputePipeline *p, uint32_t groupsX) {
    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VkResult r = vkBeginCommandBuffer(p->cmdBuf, &beginInfo);
    if (r != VK_SUCCESS) {
        return r;
    }

    vkCmdBindPipeline(p->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    vkCmdBindDescriptorSets(p->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipelineLayout, 0, 1, &p->descSet, 0, NULL);
    vkCmdDispatch(p->cmdBuf, groupsX, 1, 1);

    r = vkEndCommandBuffer(p->cmdBuf);
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

    return vkWaitForFences(device, 1, &p->fence, VK_TRUE, 5000000000ULL);
}

void shim_destroy_compute_pipeline(VkDevice device, ShimComputePipeline *p) {
    if (p->fence) vkDestroyFence(device, p->fence, NULL);
    if (p->cmdPool) vkDestroyCommandPool(device, p->cmdPool, NULL);
    if (p->descPool) vkDestroyDescriptorPool(device, p->descPool, NULL);
    if (p->pipeline) vkDestroyPipeline(device, p->pipeline, NULL);
    if (p->pipelineLayout) vkDestroyPipelineLayout(device, p->pipelineLayout, NULL);
    if (p->setLayout) vkDestroyDescriptorSetLayout(device, p->setLayout, NULL);
}
