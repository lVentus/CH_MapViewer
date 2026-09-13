#include "benchmark/GPUTimer.h"

#include <glad/gl.h>

namespace chmv::benchmark {

GPUTimer::GPUTimer() {
    GLint bits = 0;
    glGetQueryiv(GL_TIMESTAMP, GL_QUERY_COUNTER_BITS, &bits);
    timestampBits_ = bits;
    if (timestampBits_ <= 0) {
        return;
    }

    glGenQueries(static_cast<GLsizei>(startQueries_.size()), startQueries_.data());
    glGenQueries(static_cast<GLsizei>(endQueries_.size()), endQueries_.data());
}

GPUTimer::~GPUTimer() {
    if (timestampBits_ <= 0) {
        return;
    }

    glDeleteQueries(static_cast<GLsizei>(startQueries_.size()), startQueries_.data());
    glDeleteQueries(static_cast<GLsizei>(endQueries_.size()), endQueries_.data());
}

void GPUTimer::Begin() {
    CollectReadyResults();
    activeQuery_ = -1;

    if (timestampBits_ <= 0) {
        return;
    }

    for (std::size_t attempt = 0; attempt < QueryCount; ++attempt) {
        const auto index = (nextQuery_ + attempt) % QueryCount;
        if (!pending_[index]) {
            activeQuery_ = static_cast<int>(index);
            nextQuery_ = (index + 1) % QueryCount;
            glQueryCounter(startQueries_[index], GL_TIMESTAMP);
            return;
        }
    }
}

void GPUTimer::End() {
    if (activeQuery_ < 0) {
        return;
    }

    const auto index = static_cast<std::size_t>(activeQuery_);
    glQueryCounter(endQueries_[index], GL_TIMESTAMP);
    pending_[index] = true;
    sequence_[index] = nextSequence_++;
    activeQuery_ = -1;
}

void GPUTimer::CollectReadyResults() {
    if (timestampBits_ <= 0) {
        return;
    }

    for (std::size_t i = 0; i < QueryCount; ++i) {
        if (!pending_[i]) {
            continue;
        }

        GLint available = GL_FALSE;
        glGetQueryObjectiv(endQueries_[i], GL_QUERY_RESULT_AVAILABLE, &available);
        if (available == GL_FALSE) {
            continue;
        }

        GLuint64 start = 0;
        GLuint64 end = 0;
        glGetQueryObjectui64v(startQueries_[i], GL_QUERY_RESULT, &start);
        glGetQueryObjectui64v(endQueries_[i], GL_QUERY_RESULT, &end);

        if (sequence_[i] > lastSequence_) {
            const auto nanoseconds = ElapsedNanoseconds(start, end);
            lastMilliseconds_ = static_cast<double>(nanoseconds) / 1'000'000.0;
            lastSequence_ = sequence_[i];
        }

        pending_[i] = false;
    }
}

std::uint64_t GPUTimer::ElapsedNanoseconds(std::uint64_t start, std::uint64_t end) const {
    if (end >= start) {
        return end - start;
    }

    if (timestampBits_ > 0 && timestampBits_ < 64) {
        return (std::uint64_t{1} << timestampBits_) - start + end;
    }

    return end - start;
}

} // namespace chmv::benchmark
