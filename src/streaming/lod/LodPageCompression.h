#pragma once

// zstd frames of the LOD page file (layout in LodPageFormat.h). Compression and decompression
// contexts are kept per thread, so every function here may run on any number of threads at once.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <vector>

namespace chmv::streaming::lod {

class LodPageFile;

// zstd levels; decoding is equally fast for every level. On bw, level 9 makes pages only 0.3 %
// smaller than level 6 but takes more than twice as long to compress; level 3 is 3 % larger.
inline constexpr int kFrameLevel = 6;
inline constexpr int kBlockLevel = 6;
// Uncompressed bytes per block of a compressed column. RecordEdge is read a page at a time (by
// the validation) and gets small blocks.
inline constexpr std::uint32_t kColumnBlockBytes = 1u << 20u;
inline constexpr std::uint32_t kSmallBlockBytes = 64u << 10u;

// Appends one frame holding `input` to `out`. With shuffleBytes > 1 the bytes are shuffled by
// that element size first (a tail shorter than one element is kept as it is).
void AppendFrame(std::span<const std::byte> input, int level, std::uint32_t shuffleBytes,
                 std::vector<std::byte>& out);

// Appends the smaller of the plain frame and the one shuffled by elementBytes; returns the
// shuffle used (0 = plain).
[[nodiscard]] std::uint32_t AppendSmallerFrame(std::span<const std::byte> input, int level,
                                               std::uint32_t elementBytes,
                                               std::vector<std::byte>& out);

// Decodes one frame into exactly `output`; false if the frame is damaged or holds another size.
[[nodiscard]] bool DecodeFrame(std::span<const std::byte> frame, std::uint32_t shuffleBytes,
                               std::span<std::byte> output);

// Writes `output`: the checked uncompressed file `raw` (also a version 3 file), compressed.
// `progress(done, total)` counts uncompressed bytes; it is called from the calling thread only
// and may throw to cancel.
void CompressPageFile(const LodPageFile& raw, const std::filesystem::path& output,
                      const std::function<void(std::uint64_t, std::uint64_t)>& progress);

} // namespace chmv::streaming::lod
