#pragma once

#include "streaming/lod/LodPageBuilder.h"

#include <filesystem>
#include <string>

namespace chmv::streaming::lod {

class LodPageFile;

// Reports of a page file for later analysis (tables and plots), as text and CSV in a results
// directory, never next to the data; nothing reads them back.
//
// The layout report (layout_*.txt / .csv) only uses the header and the directories, which are in
// memory once the file is open, so it can be written for any file at any time. The build report
// (preprocess_*.txt / .csv) adds what only the build knows: stage times, RAM, threads; every
// build also appends a line to preprocess_history.csv.
void WriteLayoutReport(const LodPageFile& file, const std::filesystem::path& directory);
void WriteBuildReport(const BuildResult& result, const std::filesystem::path& directory);

// Where the reports of a page file go: a folder named like the file under `root`.
[[nodiscard]] std::filesystem::path ReportDirectory(const std::filesystem::path& root,
                                                    const std::filesystem::path& pageFile);

// The layout report in `directory` is missing or older than the file.
[[nodiscard]] bool LayoutReportOutdated(const LodPageFile& file,
                                        const std::filesystem::path& directory);

// Current time (UTC) for reports and file names: "2026-11-02 21:30:05" or, with fileName,
// "20261102-213005".
[[nodiscard]] std::string ReportTimestamp(bool fileName = false);

} // namespace chmv::streaming::lod
