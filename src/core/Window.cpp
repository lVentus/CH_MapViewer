#include "core/Window.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace chmv::core {
namespace {

constexpr float kInitialWorkAreaScale = 0.8f;

} // namespace

Window::Window(std::string title) : Window(0, 0, std::move(title)) {}

Window::Window(int width, int height, std::string title) {
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

    int workX = 0;
    int workY = 0;
    int workWidth = 0;
    int workHeight = 0;
    const bool useWorkArea = width <= 0 || height <= 0;

    if (useWorkArea) {
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        if (!monitor) {
            glfwTerminate();
            throw std::runtime_error("failed to get primary monitor");
        }

        glfwGetMonitorWorkarea(monitor, &workX, &workY, &workWidth, &workHeight);
        width = static_cast<int>(static_cast<float>(workWidth) * kInitialWorkAreaScale);
        height = static_cast<int>(static_cast<float>(workHeight) * kInitialWorkAreaScale);
    }

    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("failed to create GLFW window");
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetScrollCallback(window_, ScrollCallback);

    if (useWorkArea) {
        glfwSetWindowPos(window_,
                         workX + (workWidth - width) / 2,
                         workY + (workHeight - height) / 2);
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);
}

Window::~Window() {
    if (window_) {
        glfwDestroyWindow(window_);
    }
    glfwTerminate();
}

bool Window::ShouldClose() const {
    return glfwWindowShouldClose(window_) != 0;
}

float Window::ContentScale() const {
    float xScale = 1.0f;
    float yScale = 1.0f;
    glfwGetWindowContentScale(window_, &xScale, &yScale);
    return std::max(xScale, yScale);
}

bool Window::MiddleMouseDown() const {
    return glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
}

void Window::FramebufferSize(int& width, int& height) const {
    glfwGetFramebufferSize(window_, &width, &height);
}

void Window::CursorPosition(double& x, double& y) const {
    glfwGetCursorPos(window_, &x, &y);
}

double Window::ConsumeScrollY() {
    const auto scroll = scrollY_;
    scrollY_ = 0.0;
    return scroll;
}

void Window::PollEvents() const {
    glfwPollEvents();
}

void Window::SwapBuffers() const {
    glfwSwapBuffers(window_);
}

void Window::ScrollCallback(GLFWwindow* window, double, double yOffset) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self) {
        self->scrollY_ += yOffset;
    }
}

} // namespace chmv::core
