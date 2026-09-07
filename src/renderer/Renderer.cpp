#include "renderer/Renderer.h"

#include <glad/gl.h>

namespace chmv::renderer {

void Renderer::BeginFrame(int framebufferWidth, int framebufferHeight) const {
    glViewport(0, 0, framebufferWidth, framebufferHeight);
    glClearColor(0.055f, 0.06f, 0.07f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

} // namespace chmv::renderer
