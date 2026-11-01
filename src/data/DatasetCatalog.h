#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace chmv::data {

// A .sch/.ranges pair (its page file is built next to it when needed), or a page file whose
// sources are gone (graphPath and rangesPath empty).
struct DatasetEntry {
    std::string name;
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;
    std::filesystem::path pageFilePath;

    [[nodiscard]] bool HasSources() const { return !graphPath.empty(); }
};

class DatasetCatalog {
public:
    explicit DatasetCatalog(std::filesystem::path directory);

    void Refresh();

    [[nodiscard]] const std::filesystem::path& Directory() const { return directory_; }
    [[nodiscard]] const std::vector<DatasetEntry>& Entries() const { return entries_; }

private:
    std::filesystem::path directory_;
    std::vector<DatasetEntry> entries_;
};

} // namespace chmv::data
