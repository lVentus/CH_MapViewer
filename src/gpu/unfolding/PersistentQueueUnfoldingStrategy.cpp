#include "gpu/unfolding/PersistentQueueUnfoldingStrategy.h"

#include "gpu/unfolding/ExpansionKernels.h"

#include <glad/gl.h>

#include <algorithm>

namespace chmv::gpu::unfolding {
namespace {

constexpr std::uint32_t kGroupSize = 64;
// Resident workgroups: enough to fill a large GPU, few enough that waiting invocations do not
// crowd the queue counters.
constexpr std::uint32_t kMinGroups = 32;
constexpr std::uint32_t kMaxGroups = 256;
// Queue header words (lod_unfold_queue_start.comp); the dispatch arguments start at word 4.
constexpr std::size_t kHeaderWords = 8;
constexpr std::size_t kGroupsWord = 4;
constexpr std::uint32_t kEmptySlot = 0xffffffffu;

} // namespace

PersistentQueueUnfoldingStrategy::PersistentQueueUnfoldingStrategy(
    const std::filesystem::path& shaderDirectory)
    : startProgram_(shaderDirectory / "lod_unfold_queue_start.comp"),
      program_(shaderDirectory / "lod_unfold_queue.comp") {
    GLint64 maxBlockBytes = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockBytes);
    // 16 MiB is the least GL 4.3 guarantees.
    maxCapacity_ =
        static_cast<std::size_t>(std::max<GLint64>(maxBlockBytes, 16 << 20) / 4) - kHeaderWords;
}

void PersistentQueueUnfoldingStrategy::Execute(const UnfoldingInput& input) {
    // Every node visited in a pass gets a slot: the drawn ones and the expanded ones above them,
    // about twice what is drawn. Overflows are counted in the status, and the segment room (and
    // with it this) grows for the next pass.
    const auto wanted = std::min(
        std::max<std::size_t>(std::size_t{input.segmentCapacity} * 2u, 4096u), maxCapacity_);
    if (wanted > capacity_) {
        capacity_ = wanted;
        queue_.Allocate((kHeaderWords + capacity_) * 4u);
        // Slots start empty and are emptied again as they are taken.
        glBindBuffer(GL_COPY_WRITE_BUFFER, queue_.Id());
        glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT,
                          &kEmptySlot);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }

    startProgram_.Bind();
    const auto start = startProgram_.Id();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input.candidateList);
    queue_.BindBase(GL_SHADER_STORAGE_BUFFER, 7);
    glUniform1ui(glGetUniformLocation(start, "uCandidateCapacity"), input.candidateCapacity);
    glUniform1ui(glGetUniformLocation(start, "uGroupSize"), kGroupSize);
    glUniform1ui(glGetUniformLocation(start, "uMinGroups"), kMinGroups);
    glUniform1ui(glGetUniformLocation(start, "uMaxGroups"), kMaxGroups);
    startProgram_.Dispatch(1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    program_.Bind();
    const auto id = program_.Id();
    BindExpansionKernel(id, input, input.candidateList);
    BindTraversalKernel(id, input);
    queue_.BindBase(GL_SHADER_STORAGE_BUFFER, 7);
    glUniform1ui(glGetUniformLocation(id, "uQueueCapacity"), static_cast<std::uint32_t>(capacity_));
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, queue_.Id());
    glDispatchComputeIndirect(static_cast<GLintptr>(kGroupsWord * 4u));
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
}

} // namespace chmv::gpu::unfolding
