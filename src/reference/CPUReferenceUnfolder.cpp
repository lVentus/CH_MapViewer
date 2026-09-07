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
    output.clear();

    std::vector<std::uint32_t> stack;
    stack.reserve(128);

    for (const auto root : roots) {
        stack.push_back(root);

        while (!stack.empty()) {
            const auto edgeId = stack.back();
            stack.pop_back();

            const auto& edge = graph_.Edge(edgeId);
            if (!edge.IsShortcut()) {
                output.push_back(edgeId);
                continue;
            }

            if (edge.childA >= graph_.EdgeCount() || edge.childB >= graph_.EdgeCount()) {
                throw std::runtime_error("shortcut references an invalid child edge");
            }

            stack.push_back(edge.childB);
            stack.push_back(edge.childA);
        }
    }
}

} // namespace chmv::reference
