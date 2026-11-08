#include "streaming/lod/LodPageReport.h"

#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPageFormat.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace chmv::streaming::lod {
namespace {

constexpr double kMiB = 1024.0 * 1024.0;
constexpr double kGiB = kMiB * 1024.0;

constexpr std::array<const char*, kStoredCount> kStoredNames{
    "NodeOsmId",    "NodeLatitude",  "NodeLongitude",  "NodeElevation", "NodeLevel",
    "EdgeSource",   "EdgeTarget",    "EdgeWeight",     "EdgeType",      "EdgeMaxSpeed",
    "EdgeChildA",   "EdgeChildB",    "EdgeBirth",      "EdgeDeath",     "EdgeError",
    "EdgeCoverLow", "EdgeCoverHigh", "RecordEdge",     "PageDirectory", "PageGroups",
    "Pages",        "PageChildren",  "VisitDirectory", "VisitGroups"};

[[nodiscard]] std::ofstream OpenReport(const std::filesystem::path& directory, const char* name,
                                       bool append = false) {
    std::filesystem::create_directories(directory);
    std::ofstream out(directory / name, append ? std::ios::app : std::ios::trunc);
    if (!out) {
        throw std::runtime_error("could not write " + (directory / name).string());
    }
    return out;
}

// Value at fraction q of sorted values.
template <typename T> [[nodiscard]] T Percentile(const std::vector<T>& sorted, double q) {
    if (sorted.empty()) {
        return T{};
    }
    const auto index = static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1u) + 0.5);
    return sorted[std::min(index, sorted.size() - 1u)];
}

[[nodiscard]] double Share(double part, double whole) {
    return whole > 0.0 ? 100.0 * part / whole : 0.0;
}

// Pages of one quality layer and zoom octave.
struct LayerStats {
    std::uint64_t pages = 0;
    std::uint64_t walkOnlyPages = 0;
    std::uint64_t records = 0;
    std::uint64_t vertices = 0;
    std::uint64_t wordBytes = 0;
    std::uint64_t storedBytes = 0;
    std::vector<double> boxKm;
};

} // namespace

std::string ReportTimestamp(bool fileName) {
    const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    return fileName ? std::format("{:%Y%m%d-%H%M%S}", now)
                    : std::format("{:%Y-%m-%d %H:%M:%S}", now);
}

std::filesystem::path ReportDirectory(const std::filesystem::path& root,
                                      const std::filesystem::path& pageFile) {
    return root / pageFile.stem();
}

bool LayoutReportOutdated(const LodPageFile& file, const std::filesystem::path& directory) {
    std::error_code reportError;
    std::error_code fileError;
    const auto report =
        std::filesystem::last_write_time(directory / "layout_summary.txt", reportError);
    const auto written = std::filesystem::last_write_time(file.Path(), fileError);
    return reportError || fileError || report < written;
}

void WriteLayoutReport(const LodPageFile& file, const std::filesystem::path& directory) {
    const auto& header = file.Header();
    const auto pages = file.Pages();
    const auto visitPages = file.VisitPages();
    const auto metersPerUnit = header.MetersPerNormalizedUnit();

    // Per quality layer and zoom octave, and over all pages.
    std::map<std::pair<int, int>, LayerStats> layers;
    std::vector<std::uint32_t> recordsPerPage;
    std::vector<double> boxKm;
    std::array<std::uint64_t, 8> fillHistogram{};
    recordsPerPage.reserve(pages.size());
    boxKm.reserve(pages.size());
    std::uint64_t storedPageBytes = 0;
    std::uint64_t vertices = 0;
    for (std::uint32_t page = 0; page < pages.size(); ++page) {
        const auto& entry = pages[page];
        auto& layer = layers[{entry.layer, entry.octave}];
        const auto stored = file.PageStorage(page).bytes;
        const auto km =
            std::max(entry.maxX - entry.minX, entry.maxY - entry.minY) * metersPerUnit / 1000.0;
        ++layer.pages;
        layer.walkOnlyPages += entry.minLevel > entry.maxLevel ? 1u : 0u;
        layer.records += entry.recordCount;
        layer.vertices += entry.vertexCount;
        layer.wordBytes += std::uint64_t{entry.wordCount} * 4u;
        layer.storedBytes += stored;
        layer.boxKm.push_back(km);
        recordsPerPage.push_back(entry.recordCount);
        boxKm.push_back(km);
        storedPageBytes += stored;
        vertices += entry.vertexCount;
        ++fillHistogram[std::min<std::size_t>((entry.recordCount * 8u) / (kPageRecordCapacity + 1u),
                                              fillHistogram.size() - 1u)];
    }
    std::sort(recordsPerPage.begin(), recordsPerPage.end());
    std::sort(boxKm.begin(), boxKm.end());

    // Pages a level can select, in the main directory and in the one traversals use.
    const auto levels = static_cast<std::size_t>(std::max(header.maxLevel, 0)) + 2u;
    std::vector<std::int64_t> mainChange(levels, 0);
    std::vector<std::int64_t> visitChange(levels, 0);
    const auto count = [&](std::span<const PageEntry> directory,
                           std::vector<std::int64_t>& change) {
        for (const auto& entry : directory) {
            if (entry.minLevel <= entry.maxLevel && entry.minLevel < levels - 1u) {
                ++change[entry.minLevel];
                --change[std::min<std::size_t>(entry.maxLevel + 1u, levels - 1u)];
            }
        }
    };
    count(pages, mainChange);
    count(visitPages, visitChange);

    std::uint64_t uncompressed = kHeaderBytes;
    std::uint64_t stored = kHeaderBytes;
    {
        auto csv = OpenReport(directory, "layout_sections.csv");
        csv << "section,uncompressed_bytes,stored_bytes,stored_percent\n";
        for (std::size_t index = 0; index < kStoredCount; ++index) {
            const auto bytes = header.Layout(index).bytes;
            const auto onDisk =
                header.compression == Compression::None ? bytes : header.stored[index].bytes;
            uncompressed += bytes;
            stored += onDisk;
            csv << kStoredNames[index] << ',' << bytes << ',' << onDisk << ','
                << std::format("{:.1f}",
                               Share(static_cast<double>(onDisk), static_cast<double>(bytes)))
                << '\n';
        }
    }
    {
        auto csv = OpenReport(directory, "layout_layers.csv");
        csv << "quality_layer,zoom_octave,pages,walk_only_pages,records,records_per_page,"
               "fill_percent,vertices_per_record,box_km_p50,box_km_p95,box_km_max,"
               "uncompressed_mib,stored_mib\n";
        for (auto& [key, layer] : layers) {
            std::sort(layer.boxKm.begin(), layer.boxKm.end());
            const auto perPage =
                static_cast<double>(layer.records) / static_cast<double>(layer.pages);
            csv << key.first << ',' << key.second << ',' << layer.pages << ','
                << layer.walkOnlyPages << ',' << layer.records << ','
                << std::format("{:.1f},{:.1f},{:.3f},{:.3f},{:.3f},{:.3f},{:.2f},{:.2f}", perPage,
                               Share(perPage, kPageRecordCapacity),
                               layer.records != 0u ? static_cast<double>(layer.vertices) /
                                                         static_cast<double>(layer.records)
                                                   : 0.0,
                               Percentile(layer.boxKm, 0.5), Percentile(layer.boxKm, 0.95),
                               layer.boxKm.empty() ? 0.0 : layer.boxKm.back(),
                               static_cast<double>(layer.wordBytes) / kMiB,
                               static_cast<double>(layer.storedBytes) / kMiB)
                << '\n';
        }
    }
    {
        auto csv = OpenReport(directory, "layout_levels.csv");
        csv << "level,pages,traversal_pages\n";
        std::int64_t main = 0;
        std::int64_t visit = 0;
        for (std::size_t level = 0; level + 1u < levels; ++level) {
            main += mainChange[level];
            visit += visitChange[level];
            csv << level << ',' << main << ',' << visit << '\n';
        }
    }
    {
        auto csv = OpenReport(directory, "layout_page_fill.csv");
        csv << "records_from,records_to,pages\n";
        for (std::size_t bucket = 0; bucket < fillHistogram.size(); ++bucket) {
            csv << bucket * kPageRecordCapacity / fillHistogram.size() << ','
                << (bucket + 1u) * kPageRecordCapacity / fillHistogram.size() << ','
                << fillHistogram[bucket] << '\n';
        }
    }

    const auto records = static_cast<double>(header.recordCount);
    const auto pageCount = static_cast<double>(std::max<std::uint64_t>(header.pageCount, 1u));
    const auto widthKm = (header.maxLongitude - header.minLongitude) * header.longitudeScale *
                         kMetersPerProjectedDegree / 1000.0;
    const auto heightKm =
        (header.maxLatitude - header.minLatitude) * kMetersPerProjectedDegree / 1000.0;
    auto out = OpenReport(directory, "layout_summary.txt");
    out << "LOD page file layout\n"
        << "Written:            " << ReportTimestamp() << " UTC\n"
        << "File:               " << file.Path().string() << '\n'
        << "Format version:     " << header.version
        << (header.compression == Compression::None ? " (uncompressed)" : " (zstd)") << '\n'
        << std::format("Region:             lat {:.4f} .. {:.4f}, lon {:.4f} .. {:.4f} "
                       "(about {:.0f} x {:.0f} km)\n",
                       header.minLatitude, header.maxLatitude, header.minLongitude,
                       header.maxLongitude, widthKm, heightKm)
        << "\nCH\n"
        << "Nodes:              " << header.nodeCount << '\n'
        << "Edges:              " << header.edgeCount << '\n'
        << "Levels:             0 .. " << header.maxLevel << '\n'
        << "Hierarchy height:   " << header.maxHierarchyHeight << '\n'
        << "\nRecords and pages\n"
        << "Records:            " << header.recordCount
        << std::format(" ({:.3f} per edge)\n",
                       records / static_cast<double>(std::max<std::uint64_t>(header.edgeCount, 1u)))
        << "Extra contexts:     " << header.extraContextRecords << '\n'
        << "Split covers:       " << header.coverGapEdges << '\n'
        << "Pages:              " << header.pageCount << " (" << header.groupCount << " groups of "
        << kPagesPerGroup << ")\n"
        << std::format("Records per page:   mean {:.1f}, p50 {}, p5 {}, min {} (capacity {}; "
                       "fill {:.1f} %)\n",
                       records / pageCount, Percentile(recordsPerPage, 0.5),
                       Percentile(recordsPerPage, 0.05),
                       recordsPerPage.empty() ? 0u : recordsPerPage.front(), kPageRecordCapacity,
                       Share(records / pageCount, kPageRecordCapacity))
        << std::format("Vertices per record: {:.3f} (shared end points within a page)\n",
                       records > 0.0 ? static_cast<double>(vertices) / records : 0.0)
        << std::format("Page box size:      p50 {:.3f} km, p95 {:.3f} km, max {:.1f} km\n",
                       Percentile(boxKm, 0.5), Percentile(boxKm, 0.95),
                       boxKm.empty() ? 0.0 : boxKm.back())
        << "Traversal pages:    " << visitPages.size() << " (directory of the hierarchy walk)\n"
        << "\nSizes\n"
        << std::format("File on disk:       {:.2f} GiB ({:.1f} % of {:.2f} GiB uncompressed)\n",
                       static_cast<double>(file.FileBytes()) / kGiB,
                       Share(static_cast<double>(stored), static_cast<double>(uncompressed)),
                       static_cast<double>(uncompressed) / kGiB)
        << std::format("Render pages:       {:.1f} MiB uncompressed, {:.1f} MiB stored, "
                       "{:.1f} bytes per record stored\n",
                       static_cast<double>(file.PageSectionBytes()) / kMiB,
                       static_cast<double>(storedPageBytes) / kMiB,
                       records > 0.0 ? static_cast<double>(storedPageBytes) / records : 0.0)
        << "\nlayout_layers.csv: pages per quality layer (0 = 1 px, 1 = 2 px, ...) and zoom "
           "octave\nlayout_levels.csv: pages each level can select\n"
           "layout_page_fill.csv: pages by records per page\n"
           "layout_sections.csv: every section uncompressed and stored\n";
}

void WriteBuildReport(const BuildResult& result, const std::filesystem::path& directory) {
    const auto when = ReportTimestamp();
    double total = 0.0;
    for (const auto seconds : result.stageSeconds) {
        total += seconds;
    }
    const auto readSeconds = result.stageSeconds[static_cast<std::size_t>(BuildStage::ReadNodes)] +
                             result.stageSeconds[static_cast<std::size_t>(BuildStage::ReadEdges)] +
                             result.stageSeconds[static_cast<std::size_t>(BuildStage::ReadRanges)];
    const auto sourceBytes = static_cast<double>(result.graphBytes + result.rangesBytes);
    {
        auto csv = OpenReport(directory, "preprocess_stages.csv");
        csv << "stage,seconds,percent\n";
        for (std::size_t stage = 0; stage < kBuildStageCount; ++stage) {
            csv << BuildStageName(static_cast<BuildStage>(stage)) << ','
                << std::format("{:.3f},{:.1f}", result.stageSeconds[stage],
                               Share(result.stageSeconds[stage], total))
                << '\n';
        }
    }
    {
        const auto history = directory / "preprocess_history.csv";
        const bool fresh = !std::filesystem::exists(history);
        auto csv = OpenReport(directory, "preprocess_history.csv", true);
        if (fresh) {
            csv << "time_utc,seconds,threads,peak_ram_mib,ram_budget_mib,page_passes,nodes,edges,"
                   "records,pages,source_mib,file_mib,uncompressed_mib";
            for (std::size_t stage = 0; stage < kBuildStageCount; ++stage) {
                csv << ",s_" << BuildStageName(static_cast<BuildStage>(stage));
            }
            csv << '\n';
        }
        csv << when << ',' << std::format("{:.2f}", result.seconds) << ',' << result.threads << ','
            << std::format("{:.0f},{:.0f}", static_cast<double>(result.peakResidentBytes) / kMiB,
                           static_cast<double>(result.memoryBudgetBytes) / kMiB)
            << ',' << result.pagePasses << ',' << result.nodeCount << ',' << result.edgeCount << ','
            << result.recordCount << ',' << result.pageCount << ','
            << std::format("{:.1f},{:.1f},{:.1f}", sourceBytes / kMiB,
                           static_cast<double>(result.fileBytes) / kMiB,
                           static_cast<double>(result.uncompressedBytes) / kMiB);
        for (const auto seconds : result.stageSeconds) {
            csv << std::format(",{:.3f}", seconds);
        }
        csv << '\n';
    }

    auto out = OpenReport(directory, "preprocess_summary.txt");
    out << "LOD page file build\n"
        << "Finished:           " << when << " UTC\n"
        << "Output:             " << result.path.string() << '\n'
        << std::format("Sources:            {:.1f} MiB .sch + {:.1f} MiB .ranges\n",
                       static_cast<double>(result.graphBytes) / kMiB,
                       static_cast<double>(result.rangesBytes) / kMiB)
        << "Threads:            " << result.threads << " (hardware "
        << std::thread::hardware_concurrency() << ")\n"
        << std::format("RAM:                peak {:.0f} MiB held by the process (sampled), "
                       "budget {:.0f} MiB\n",
                       static_cast<double>(result.peakResidentBytes) / kMiB,
                       static_cast<double>(result.memoryBudgetBytes) / kMiB)
        << "Page passes:        " << result.pagePasses << " (writing pages within the budget)\n"
        << "\nTime\n"
        << std::format("Total:              {:.1f} s\n", result.seconds);
    for (std::size_t stage = 0; stage < kBuildStageCount; ++stage) {
        out << std::format("  {:<20}{:9.2f} s  {:5.1f} %\n",
                           BuildStageName(static_cast<BuildStage>(stage)),
                           result.stageSeconds[stage], Share(result.stageSeconds[stage], total));
    }
    out << std::format("Reading sources:    {:.1f} MiB/s\n",
                       readSeconds > 0.0 ? sourceBytes / kMiB / readSeconds : 0.0)
        << std::format("Overall:            {:.2f} M edges/s\n",
                       result.seconds > 0.0
                           ? static_cast<double>(result.edgeCount) / 1e6 / result.seconds
                           : 0.0)
        << "\nResult\n"
        << "Nodes / edges:      " << result.nodeCount << " / " << result.edgeCount << '\n'
        << "Records / pages:    " << result.recordCount << " / " << result.pageCount << '\n'
        << "Extra contexts:     " << result.extraContextRecords << '\n'
        << "Split covers:       " << result.coverGapEdges << '\n'
        << "Walk-only records:  " << result.visitOnlyRecords << '\n'
        << "Checked cuts:       " << result.verifiedCuts << " reference cuts equal to DFS\n"
        << std::format("File:               {:.1f} MiB ({:.1f} % of {:.1f} MiB uncompressed)\n",
                       static_cast<double>(result.fileBytes) / kMiB,
                       Share(static_cast<double>(result.fileBytes),
                             static_cast<double>(result.uncompressedBytes)),
                       static_cast<double>(result.uncompressedBytes) / kMiB)
        << "\npreprocess_stages.csv: the times above; preprocess_history.csv: one line per "
           "build\n";
}

} // namespace chmv::streaming::lod
