#include "data/SystemMemory.h"

#include <fstream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace chmv::data {

SystemMemoryInfo QuerySystemMemoryInfo() {
    SystemMemoryInfo result;
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        result.totalPhysicalBytes = status.ullTotalPhys;
        result.availablePhysicalBytes = status.ullAvailPhys;
    }
#elif defined(__linux__)
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string key;
        std::uint64_t valueKiB = 0;
        std::string unit;
        while (meminfo >> key >> valueKiB >> unit) {
            if (key == "MemTotal:") {
                result.totalPhysicalBytes = valueKiB * 1024ull;
            } else if (key == "MemAvailable:") {
                result.availablePhysicalBytes = valueKiB * 1024ull;
            }
        }
    }
    if (result.totalPhysicalBytes == 0 || result.availablePhysicalBytes == 0) {
        struct sysinfo info {};
        if (sysinfo(&info) == 0) {
            result.totalPhysicalBytes = static_cast<std::uint64_t>(info.totalram) * info.mem_unit;
            result.availablePhysicalBytes =
                static_cast<std::uint64_t>(info.freeram + info.bufferram) * info.mem_unit;
        }
    }
#endif
    return result;
}

} // namespace chmv::data
