#include "reference/CPUReferenceUnfolder.h"

#include <stdexcept>

namespace chmv::reference {

std::vector<std::uint32_t>
CPUReferenceUnfolder::UnfoldFully(std::span<const std::uint32_t> roots) const {
    std::vector<std::uint32_t> output;
    UnfoldFully(roots, output);
    return output;
}

void CPUReferenceUnfolder::UnfoldFully(std::span<const std::uint32_t> roots,
                                       std::vector<std::uint32_t>& output) const {
    std::vector<std::uint32_t> stack;
    stack.reserve(128);
    UnfoldFully(roots, output, stack);
}

void CPUReferenceUnfolder::UnfoldFully(std::span<const std::uint32_t> roots,
                                       std::vector<std::uint32_t>& output,
                                       std::vector<std::uint32_t>& stack) const {
    output.clear();
    stack.clear();
    if (stack.capacity() < 128) {
        stack.reserve(128);
    }
    if (output.capacity() < roots.size()) {
        output.reserve(roots.size());
    }

    const auto& edges = graph_.Edges();
    const auto edgeCount = edges.size();

    for (const auto root : roots) {
        stack.push_back(root);

        while (!stack.empty()) {
            const auto edgeId = stack.back();
            stack.pop_back();

            const auto& edge = edges[edgeId];
            if (!edge.IsShortcut()) {
                output.push_back(edgeId);
                continue;
            }

            if (edge.childA >= edgeCount || edge.childB >= edgeCount) {
                throw std::runtime_error("shortcut references an invalid child edge");
            }

            stack.push_back(edge.childB);
            stack.push_back(edge.childA);
        }
    }
}

void CPUReferenceUnfolder::UnfoldAdaptive(std::span<const std::uint32_t> roots,
                                          std::vector<std::uint32_t>& output,
                                          std::vector<std::uint32_t>& stack,
                                          float screenPixelScale,
                                          float maxScreenErrorPixels) const {
    output.clear();
    stack.clear();
    if (stack.capacity() < 128) {
        stack.reserve(128);
    }
    if (output.capacity() < roots.size()) {
        output.reserve(roots.size());
    }

    const auto& edges = graph_.Edges();
    const auto edgeCount = edges.size();

    for (const auto root : roots) {
        stack.push_back(root);

        while (!stack.empty()) {
            const auto edgeId = stack.back();
            stack.pop_back();

            const auto& edge = edges[edgeId];
            if (!edge.IsShortcut() ||
                edge.geometryError * screenPixelScale <= maxScreenErrorPixels) {
                output.push_back(edgeId);
                continue;
            }

            if (edge.childA >= edgeCount || edge.childB >= edgeCount) {
                throw std::runtime_error("shortcut references an invalid child edge");
            }

            stack.push_back(edge.childB);
            stack.push_back(edge.childA);
        }
    }
}

} // namespace chmv::reference
