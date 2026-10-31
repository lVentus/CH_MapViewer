#include "streaming/lod/LodPagePlanner.h"

namespace chmv::streaming::lod {

void CollectLodPages(std::span<const PageGroup> groups, std::span<const PageEntry> pages,
                     const LodPageQuery& query, std::vector<std::uint32_t>& out) {
    for (const auto& group : groups) {
        if (!MayDraw(group, query)) {
            continue;
        }
        const auto end = group.firstPage + group.pageCount;
        for (auto page = group.firstPage; page < end; ++page) {
            if (MayDraw(pages[page], query)) {
                out.push_back(page);
            }
        }
    }
}

} // namespace chmv::streaming::lod
