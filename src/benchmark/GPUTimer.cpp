#include "benchmark/GPUTimer.h"

#include <glad/gl.h>

namespace chmv::benchmark {

GPUTimer::GPUTimer() {
    glGenQueries(static_cast<GLsizei>(queries_.size()), queries_.data());
}

GPUTimer::~GPUTimer() {
    glDeleteQueries(static_cast<GLsizei>(queries_.size()), queries_.data());
}

void GPUTimer::Begin() {
    CollectReadyResults();
    activeQuery_ = -1;

    for (std::size_t attempt = 0; attempt < QueryCount; ++attempt) {
        const auto index = (nextQuery_ + attempt) % QueryCount;
        if (!pending_[index]) {
            activeQuery_ = static_cast<int>(index);
            nextQuery_ = (index + 1) % QueryCount;
            glBeginQuery(GL_TIME_ELAPSED, queries_[index]);
            return;
        }
    }
}

void GPUTimer::End() {
    if (activeQuery_ < 0) {
        return;
    }

    glEndQuery(GL_TIME_ELAPSED);
    pending_[static_cast<std::size_t>(activeQuery_)] = true;
    activeQuery_ = -1;
}

void GPUTimer::CollectReadyResults() {
    for (std::size_t i = 0; i < QueryCount; ++i) {
        if (!pending_[i]) {
            continue;
        }

        GLint available = GL_FALSE;
        glGetQueryObjectiv(queries_[i], GL_QUERY_RESULT_AVAILABLE, &available);
        if (available == GL_FALSE) {
            continue;
        }

        GLuint64 nanoseconds = 0;
        glGetQueryObjectui64v(queries_[i], GL_QUERY_RESULT, &nanoseconds);
        lastMilliseconds_ = static_cast<double>(nanoseconds) / 1'000'000.0;
        pending_[i] = false;
    }
}

} // namespace chmv::benchmark
