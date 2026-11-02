#pragma once

namespace chmv::renderer {

class Renderer {
public:
    void BeginFrame(int framebufferWidth, int framebufferHeight) const;
    // Draws go to this part of the framebuffer from now on.
    void SetViewport(int x, int y, int width, int height) const;
};

} // namespace chmv::renderer
