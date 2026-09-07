#pragma once

#include <cstdint>

namespace chmv::renderer {

class LODController {
public:
    void SetLevelRange(std::int32_t minLevel, std::int32_t maxLevel);
    void SetAutomatic(bool automatic) { automatic_ = automatic; }
    void SetManualLevel(std::int32_t level);

    [[nodiscard]] bool Automatic() const { return automatic_; }
    [[nodiscard]] std::int32_t ManualLevel() const { return manualLevel_; }
    [[nodiscard]] std::int32_t MinLevel() const { return minLevel_; }
    [[nodiscard]] std::int32_t MaxLevel() const { return maxLevel_; }
    [[nodiscard]] std::int32_t Level(float zoom) const;

private:
    std::int32_t minLevel_ = 0;
    std::int32_t maxLevel_ = 0;
    std::int32_t manualLevel_ = 0;
    bool automatic_ = true;
};

} // namespace chmv::renderer
