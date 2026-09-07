#include "gpu/OpenGLContext.h"

#include "core/Window.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <stdexcept>

namespace chmv::gpu {

OpenGLContext::OpenGLContext(const core::Window& window) {
    glfwMakeContextCurrent(window.Handle());

    const auto version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        throw std::runtime_error("failed to load OpenGL functions");
    }

    majorVersion_ = GLAD_VERSION_MAJOR(version);
    minorVersion_ = GLAD_VERSION_MINOR(version);

    if (majorVersion_ < 4 || (majorVersion_ == 4 && minorVersion_ < 3)) {
        throw std::runtime_error("CH_MapViewer requires OpenGL 4.3 or newer");
    }
}

} // namespace chmv::gpu
