#include "data/DatasetCatalog.h"

#include "streaming/lod/LodPageFile.h"

#include <algorithm>
#include <set>
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

    std::vector<std::filesystem::path> pageFiles;
    std::set<std::filesystem::path> builtPageFiles;
    std::filesystem::recursive_directory_iterator iterator(directory_);
    const std::filesystem::recursive_directory_iterator end;
    for (; iterator != end; ++iterator) {
        const auto& item = *iterator;
        if (item.is_directory() && item.path().filename() == ".chmv") {
            iterator.disable_recursion_pending();
            continue;
        }
        if (!item.is_regular_file()) {
            continue;
        }

        const auto& graphPath = item.path();
        if (graphPath.extension() == ".chlod") {
            pageFiles.push_back(graphPath);
            continue;
        }
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
        entry.pageFilePath = streaming::lod::LodPageFile::DefaultPath(graphPath);
        builtPageFiles.insert(entry.pageFilePath);
        entries_.push_back(std::move(entry));
    }
    // A page file is complete on its own: listed by itself once its sources are gone.
    for (auto& pageFile : pageFiles) {
        if (!builtPageFiles.contains(pageFile)) {
            DatasetEntry entry;
            entry.name = std::filesystem::relative(pageFile, directory_).generic_string();
            entry.pageFilePath = std::move(pageFile);
            entries_.push_back(std::move(entry));
        }
    }

    std::sort(entries_.begin(), entries_.end(), [](const DatasetEntry& left, const DatasetEntry& right) {
        return left.name < right.name;
    });
}

} // namespace chmv::data
