#include "renderer/RoadStyle.h"

#include <charconv>
#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace chmv::renderer {
namespace {

constexpr std::uint8_t kNoClass = std::numeric_limits<std::uint8_t>::max();

[[nodiscard]] bool IsHexColor(std::string_view token) {
    if (token.size() != 7u && token.size() != 9u) {
        return false;
    }
    for (const auto c : token.substr(1)) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::array<float, 4> ParseHexColor(std::string_view token) {
    std::array<float, 4> color{1.0f, 1.0f, 1.0f, 1.0f};
    for (std::size_t channel = 0; channel * 2u + 1u < token.size(); ++channel) {
        unsigned value = 0;
        std::from_chars(token.data() + 1 + channel * 2u, token.data() + 3 + channel * 2u, value,
                        16);
        color[channel] = static_cast<float>(value) / 255.0f;
    }
    return color;
}

// Whitespace-separated tokens; a token starting with '#' that is not a color starts a comment.
[[nodiscard]] std::vector<std::string> Tokens(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream stream(line);
    std::string token;
    while (stream >> token) {
        if (token.front() == '#' && !IsHexColor(token)) {
            break;
        }
        tokens.push_back(token);
    }
    return tokens;
}

} // namespace

std::array<std::array<float, 4>, data::RoadStyleTypeCount> RoadStyleConfig::TypeColors() const {
    std::array<std::array<float, 4>, data::RoadStyleTypeCount> colors{};
    for (std::size_t type = 0; type < colors.size(); ++type) {
        colors[type] = classes[classOfType[type]].color;
    }
    return colors;
}

std::array<float, data::RoadStyleTypeCount> RoadStyleConfig::TypeWidths() const {
    std::array<float, data::RoadStyleTypeCount> widths{};
    for (std::size_t type = 0; type < widths.size(); ++type) {
        widths[type] = classes[classOfType[type]].widthPixels;
    }
    return widths;
}

RoadStyleConfig UniformRoadStyle() {
    RoadStyleConfig styles;
    styles.classes.push_back({"road", {0.82f, 0.84f, 0.87f, 1.0f}, 1.5f});
    styles.classOfType.fill(0u);
    return styles;
}

RoadStyleConfig LoadRoadStyles(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open " + path.filename().string());
    }
    RoadStyleConfig styles;
    styles.classOfType.fill(kNoClass);
    std::uint8_t others = kNoClass;
    std::string line;
    for (std::size_t number = 1; std::getline(input, line); ++number) {
        const auto fail = [&](const std::string& message) {
            throw std::runtime_error(path.filename().string() + " line " + std::to_string(number) +
                                     ": " + message);
        };
        const auto tokens = Tokens(line);
        if (tokens.empty()) {
            continue;
        }
        if (tokens.size() < 4u) {
            fail("expected <name> <#color> <width> <types...>");
        }
        if (!IsHexColor(tokens[1])) {
            fail("color must be #RRGGBB or #RRGGBBAA");
        }
        float width = 0.0f;
        const auto& widthToken = tokens[2];
        const auto parsed =
            std::from_chars(widthToken.data(), widthToken.data() + widthToken.size(), width);
        if (parsed.ec != std::errc{} || parsed.ptr != widthToken.data() + widthToken.size() ||
            !(width > 0.0f) || width > kMaxRoadWidthPixels) {
            fail("width must be a number of pixels above 0 and at most " +
                 std::to_string(static_cast<int>(kMaxRoadWidthPixels)));
        }
        if (styles.classes.size() == kNoClass) {
            fail("too many road classes");
        }
        const auto index = static_cast<std::uint8_t>(styles.classes.size());
        styles.classes.push_back({tokens[0], ParseHexColor(tokens[1]), width});
        for (std::size_t t = 3; t < tokens.size(); ++t) {
            const auto& token = tokens[t];
            if (token == "*") {
                if (others != kNoClass) {
                    fail("only one class can take the remaining types (*)");
                }
                others = index;
                continue;
            }
            unsigned type = 0;
            const auto result = std::from_chars(token.data(), token.data() + token.size(), type);
            if (result.ec != std::errc{} || result.ptr != token.data() + token.size() ||
                type >= data::RoadStyleFallbackType) {
                fail("type '" + token + "' is not a number from 0 to " +
                     std::to_string(data::RoadStyleFallbackType - 1u));
            }
            if (styles.classOfType[type] != kNoClass) {
                fail("type " + token + " is already in another class");
            }
            styles.classOfType[type] = index;
        }
    }
    if (others == kNoClass) {
        throw std::runtime_error(path.filename().string() +
                                 ": one class has to take the remaining types (*)");
    }
    for (auto& assigned : styles.classOfType) {
        if (assigned == kNoClass) {
            assigned = others;
        }
    }
    return styles;
}

} // namespace chmv::renderer
