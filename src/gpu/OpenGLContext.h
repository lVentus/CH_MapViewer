#pragma once

namespace chmv::core {
class Window;
}

namespace chmv::gpu {

class OpenGLContext {
public:
    explicit OpenGLContext(const core::Window& window);

    [[nodiscard]] int MajorVersion() const { return majorVersion_; }
    [[nodiscard]] int MinorVersion() const { return minorVersion_; }

private:
    int majorVersion_ = 0;
    int minorVersion_ = 0;
};

} // namespace chmv::gpu
