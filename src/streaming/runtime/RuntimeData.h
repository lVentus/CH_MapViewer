#pragma once

#include "streaming/index/CHIndex.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>

namespace chmv::streaming::runtime {

inline constexpr std::uint32_t kRuntimeDataVersion = 1u;
inline constexpr std::uint64_t kRuntimeDataHeaderBytes = 4096u;

struct RuntimeNodeRecord {
    float x = 0.0f;
    float y = 0.0f;
};
static_assert(sizeof(RuntimeNodeRecord) == 8u);

struct RuntimeEdgeRecord {
    std::uint32_t source = 0;
    std::uint32_t target = 0;
    std::uint32_t childA = 0xffffffffu;
    std::uint32_t childB = 0xffffffffu;
    float weight = 0.0f;
    float geometryError = 0.0f;
    std::int16_t birthLevel = -1;
    std::int16_t deathLevel = -1;
    std::int16_t type = 0;
    std::int16_t maxSpeed = 0;
};
static_assert(sizeof(RuntimeEdgeRecord) == 32u);

struct RuntimeDataHeader {
    std::uint32_t version = kRuntimeDataVersion;
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    std::uint64_t rootRecordCount = 0;
    std::uint64_t nodeSectionOffset = 0;
    std::uint64_t edgeSectionOffset = 0;
    std::uint64_t rootSectionOffset = 0;
    std::uint64_t fileBytes = 0;
};

class RuntimeData {
public:
    [[nodiscard]] static std::filesystem::path DefaultPath(const std::filesystem::path& graphPath);
    [[nodiscard]] static RuntimeDataHeader MakeHeader(std::uint64_t nodeCount,
                                                       std::uint64_t edgeCount,
                                                       std::uint64_t rootRecordCount);
    static void WriteHeader(std::ostream& output, const RuntimeDataHeader& header);
    [[nodiscard]] static RuntimeDataHeader ReadHeader(const std::filesystem::path& path);
    [[nodiscard]] static bool IsValid(const std::filesystem::path& path,
                                      std::uint64_t nodeCount,
                                      std::uint64_t edgeCount,
                                      std::uint64_t rootRecordCount);

    template <typename T>
    static void ReadRecords(std::ifstream& input, std::uint64_t byteOffset,
                            std::span<T> records) {
        if (records.empty()) return;
        if (byteOffset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
            throw std::runtime_error("runtime-data offset exceeds stream range");
        input.clear();
        input.seekg(static_cast<std::streamoff>(byteOffset), std::ios::beg);
        input.read(reinterpret_cast<char*>(records.data()),
                   static_cast<std::streamsize>(records.size_bytes()));
        if (!input) throw std::runtime_error("truncated runtime-data section");
    }
};

} // namespace chmv::streaming::runtime
