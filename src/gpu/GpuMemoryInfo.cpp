#include "gpu/GpuMemoryInfo.h"

#include <glad/gl.h>

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <string>
#include <string_view>

namespace chmv::gpu {
namespace {

[[nodiscard]] bool HasGlExtension(std::string_view name) {
    GLint extensionCount = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &extensionCount);
    for (GLint i = 0; i < extensionCount; ++i) {
        const auto* extension = reinterpret_cast<const char*>(
            glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i)));
        if (extension != nullptr && name == extension) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::string LowerAscii(const char* value) {
    if (value == nullptr) {
        return {};
    }
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

[[nodiscard]] bool ContainsAny(std::string_view value,
                               std::initializer_list<std::string_view> needles) {
    for (const auto needle : needles) {
        if (value.find(needle) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] GPUMemoryArchitecture DetectMemoryArchitecture(std::string_view vendor,
                                                               std::string_view renderer) {
    if (vendor.find("intel") != std::string_view::npos ||
        renderer.find("intel") != std::string_view::npos ||
        ContainsAny(renderer, {"llvmpipe", "softpipe", "swiftshader"})) {
        return GPUMemoryArchitecture::Unified;
    }
    if (vendor.find("nvidia") != std::string_view::npos) {
        return GPUMemoryArchitecture::Dedicated;
    }
    if (vendor.find("amd") != std::string_view::npos ||
        vendor.find("ati") != std::string_view::npos ||
        renderer.find("radeon") != std::string_view::npos) {
        // AMD reports the same vendor for APUs and discrete boards. Discrete product names are
        // usually explicit, current APUs commonly show a generic "Radeon Graphics" or a mobile
        // iGPU name.
        if (ContainsAny(renderer, {" radeon rx ", "radeon rx", "radeon pro", "firepro",
                                   "instinct", "w6800", "w7800", "w7900"})) {
            return GPUMemoryArchitecture::Dedicated;
        }
        if (ContainsAny(renderer, {"radeon(tm) graphics", "radeon graphics", "780m", "760m",
                                   "680m", "660m", "890m", "880m", "vega 3", "vega 6",
                                   "vega 7", "vega 8", "vega 10", "vega 11"})) {
            return GPUMemoryArchitecture::Unified;
        }
        return GPUMemoryArchitecture::Dedicated;
    }
    return GPUMemoryArchitecture::Unknown;
}

} // namespace

GPUMemoryInfo QueryGpuMemoryInfo() {
    GPUMemoryInfo result;
    const auto vendor = LowerAscii(reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
    const auto renderer = LowerAscii(reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
    result.architecture = DetectMemoryArchitecture(vendor, renderer);

    constexpr GLenum kNvxTotalAvailableMemory = 0x9048;
    constexpr GLenum kNvxCurrentAvailableVidmem = 0x9049;
    if (HasGlExtension("GL_NVX_gpu_memory_info")) {
        GLint totalKiB = 0;
        GLint availableKiB = 0;
        glGetIntegerv(kNvxTotalAvailableMemory, &totalKiB);
        glGetIntegerv(kNvxCurrentAvailableVidmem, &availableKiB);
        result.reportedTotalBytes = static_cast<std::uint64_t>(std::max(totalKiB, 0)) * 1024ull;
        result.reportedAvailableBytes =
            static_cast<std::uint64_t>(std::max(availableKiB, 0)) * 1024ull;
        return result;
    }

    constexpr GLenum kAtiTextureFreeMemory = 0x87FC;
    if (HasGlExtension("GL_ATI_meminfo")) {
        GLint freeKiB[4] = {};
        glGetIntegerv(kAtiTextureFreeMemory, freeKiB);
        result.reportedAvailableBytes =
            static_cast<std::uint64_t>(std::max(freeKiB[0], 0)) * 1024ull;
    }
    return result;
}

} // namespace chmv::gpu
