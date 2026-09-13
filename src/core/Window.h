#pragma once

#include <string>

struct GLFWwindow;

namespace chmv::core {

class Window {
public:
    explicit Window(std::string title);
    Window(int width, int height, std::string title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    [[nodiscard]] GLFWwindow* Handle() const { return window_; }
    [[nodiscard]] bool ShouldClose() const;
    [[nodiscard]] float ContentScale() const;
    [[nodiscard]] bool LeftMouseDown() const;

    void FramebufferSize(int& width, int& height) const;
    void CursorPosition(double& x, double& y) const;
    double ConsumeScrollY();
    void PollEvents() const;
    void SwapBuffers() const;

private:
    static void ScrollCallback(GLFWwindow* window, double xOffset, double yOffset);

    GLFWwindow* window_ = nullptr;
    double scrollY_ = 0.0;
};

} // namespace chmv::core
