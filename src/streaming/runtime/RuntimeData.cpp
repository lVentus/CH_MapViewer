#include "streaming/runtime/RuntimeData.h"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace chmv::streaming::runtime {
namespace {
constexpr std::array<char, 8> kMagic{'C','H','M','V','B','I','N','1'};
constexpr std::uint32_t kEndianTag = 0x01020304u;
std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}
template<class T> void WriteValue(std::ostream& out, T v) {
    static_assert(std::is_arithmetic_v<T>);
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    if (!out) throw std::runtime_error("failed to write runtime-data header");
}
template<class T> T ReadValue(std::istream& in) {
    static_assert(std::is_arithmetic_v<T>);
    T v{}; in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("truncated runtime-data header");
    return v;
}
}

std::filesystem::path RuntimeData::DefaultPath(const std::filesystem::path& graphPath) {
    if (graphPath.extension() == ".sch") { auto p=graphPath; p.replace_extension(".chbin"); return p; }
    return std::filesystem::path(graphPath.string()+".chbin");
}
RuntimeDataHeader RuntimeData::MakeHeader(std::uint64_t nodes,std::uint64_t edges,std::uint64_t roots) {
    RuntimeDataHeader h; h.nodeCount=nodes; h.edgeCount=edges; h.rootRecordCount=roots;
    h.nodeSectionOffset=kRuntimeDataHeaderBytes;
    h.edgeSectionOffset=AlignUp(h.nodeSectionOffset+nodes*sizeof(RuntimeNodeRecord),4096u);
    h.rootSectionOffset=AlignUp(h.edgeSectionOffset+edges*sizeof(RuntimeEdgeRecord),4096u);
    h.fileBytes=h.rootSectionOffset+roots*sizeof(index::CHIndexRootRecord);
    return h;
}
void RuntimeData::WriteHeader(std::ostream& out,const RuntimeDataHeader& h) {
    out.seekp(0,std::ios::beg); out.write(kMagic.data(),kMagic.size());
    WriteValue(out,h.version); WriteValue(out,kEndianTag);
    WriteValue(out,h.nodeCount); WriteValue(out,h.edgeCount); WriteValue(out,h.rootRecordCount);
    WriteValue(out,h.nodeSectionOffset); WriteValue(out,h.edgeSectionOffset);
    WriteValue(out,h.rootSectionOffset); WriteValue(out,h.fileBytes);
    WriteValue(out,static_cast<std::uint32_t>(sizeof(RuntimeNodeRecord)));
    WriteValue(out,static_cast<std::uint32_t>(sizeof(RuntimeEdgeRecord)));
    WriteValue(out,static_cast<std::uint32_t>(sizeof(index::CHIndexRootRecord)));
    WriteValue(out,std::uint32_t{0});
    const auto written=static_cast<std::uint64_t>(out.tellp());
    if (written>kRuntimeDataHeaderBytes) throw std::runtime_error("runtime-data header overflow");
    std::array<char,4096> zero{}; auto remaining=kRuntimeDataHeaderBytes-written;
    while (remaining) { const auto n=static_cast<std::streamsize>(std::min<std::uint64_t>(remaining,zero.size())); out.write(zero.data(),n); remaining-=n; }
    if (!out) throw std::runtime_error("failed to finalize runtime-data header");
}
RuntimeDataHeader RuntimeData::ReadHeader(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary); if(!in) throw std::runtime_error("could not open runtime data: "+path.string());
    std::array<char,8> magic{}; in.read(magic.data(),magic.size()); if(!in||magic!=kMagic) throw std::runtime_error("invalid runtime-data magic");
    RuntimeDataHeader h; h.version=ReadValue<std::uint32_t>(in); const auto endian=ReadValue<std::uint32_t>(in);
    if(h.version!=kRuntimeDataVersion||endian!=kEndianTag||std::endian::native!=std::endian::little) throw std::runtime_error("unsupported runtime-data format");
    h.nodeCount=ReadValue<std::uint64_t>(in); h.edgeCount=ReadValue<std::uint64_t>(in); h.rootRecordCount=ReadValue<std::uint64_t>(in);
    h.nodeSectionOffset=ReadValue<std::uint64_t>(in); h.edgeSectionOffset=ReadValue<std::uint64_t>(in); h.rootSectionOffset=ReadValue<std::uint64_t>(in); h.fileBytes=ReadValue<std::uint64_t>(in);
    const auto ns=ReadValue<std::uint32_t>(in), es=ReadValue<std::uint32_t>(in), rs=ReadValue<std::uint32_t>(in); (void)ReadValue<std::uint32_t>(in);
    if(ns!=sizeof(RuntimeNodeRecord)||es!=sizeof(RuntimeEdgeRecord)||rs!=sizeof(index::CHIndexRootRecord)) throw std::runtime_error("runtime-data record-size mismatch");
    const auto actual=std::filesystem::file_size(path);
    if(h.fileBytes!=actual || h.nodeSectionOffset<kRuntimeDataHeaderBytes ||
       h.edgeSectionOffset<h.nodeSectionOffset+h.nodeCount*sizeof(RuntimeNodeRecord) ||
       h.rootSectionOffset<h.edgeSectionOffset+h.edgeCount*sizeof(RuntimeEdgeRecord) ||
       h.rootSectionOffset+h.rootRecordCount*sizeof(index::CHIndexRootRecord)>h.fileBytes) throw std::runtime_error("runtime-data section layout is invalid");
    return h;
}
bool RuntimeData::IsValid(const std::filesystem::path& path,std::uint64_t nodes,std::uint64_t edges,std::uint64_t roots) {
    try { const auto h=ReadHeader(path); return h.nodeCount==nodes&&h.edgeCount==edges&&h.rootRecordCount==roots; } catch(...) { return false; }
}
} // namespace chmv::streaming::runtime
