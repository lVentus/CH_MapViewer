#include "data/DatasetCatalog.h"

#include <algorithm>
#include <utility>

namespace chmv::data {

DatasetCatalog::DatasetCatalog(std::filesystem::path directory)
    : directory_(std::move(directory)) {
    Refresh();
}

void DatasetCatalog::Refresh() {
    entries_.clear();

    if (!std::filesystem::exists(directory_)) {
        return;
    }

    for (const auto& item : std::filesystem::recursive_directory_iterator(directory_)) {
        if (!item.is_regular_file()) {
            continue;
        }

        const auto& graphPath = item.path();
        if (graphPath.extension() == ".ranges" || graphPath.extension() == ".bz2") {
            continue;
        }

        auto rangesPath = graphPath;
        rangesPath += ".ranges";
        if (!std::filesystem::is_regular_file(rangesPath)) {
            continue;
        }

        DatasetEntry entry;
        entry.name = std::filesystem::relative(graphPath, directory_).generic_string();
        entry.graphPath = graphPath;
        entry.rangesPath = std::move(rangesPath);
        entries_.push_back(std::move(entry));
    }

    std::sort(entries_.begin(), entries_.end(), [](const DatasetEntry& left, const DatasetEntry& right) {
        return left.name < right.name;
    });
}

} // namespace chmv::data
