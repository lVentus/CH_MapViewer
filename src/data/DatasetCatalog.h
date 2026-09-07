#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace chmv::data {

struct DatasetEntry {
    std::string name;
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;
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
