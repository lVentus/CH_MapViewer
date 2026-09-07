cmake_minimum_required(VERSION 3.24)

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

project(CH_MapViewer VERSION 0.1.0 LANGUAGES C CXX)

if(MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    set(CHMV_MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}")
endif()

option(CHMV_BUILD_APP "Build the interactive viewer" ON)
option(CHMV_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR
        "CH_MapViewer requires a 64-bit build. Select an x64 Visual Studio/CMake kit instead of Win32."
    )
endif()

add_library(chmv_data
    src/benchmark/CorrectnessValidator.cpp
    src/data/DatasetCatalog.cpp
    src/data/ch/CHGraph.cpp
    src/data/ch/CHLoader.cpp
    src/pipeline/CPUReferencePipeline.cpp
    src/reference/CPURangeFilter.cpp
    src/reference/CPUReferenceUnfolder.cpp
)

target_include_directories(chmv_data PUBLIC src)

target_compile_features(chmv_data PUBLIC cxx_std_20)

if(DEFINED CHMV_MSVC_RUNTIME_LIBRARY)
    set_property(TARGET chmv_data PROPERTY MSVC_RUNTIME_LIBRARY "${CHMV_MSVC_RUNTIME_LIBRARY}")
endif()

if(MSVC)
    target_compile_options(chmv_data PRIVATE /W4 /permissive-)
else()
    target_compile_options(chmv_data PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CHMV_BUILD_APP)
    include(cmake/Dependencies.cmake)
    find_package(OpenGL REQUIRED)

    add_executable(CH_MapViewer
        src/main.cpp
        src/core/Application.cpp
        src/core/Window.cpp
        src/gpu/OpenGLContext.cpp
        src/gpu/GPUBuffer.cpp
        src/gpu/ComputeProgram.cpp
        src/gpu/GraphicsProgram.cpp
        src/benchmark/GPUTimer.cpp
        src/gpu/unfolding/UnfoldingStrategyManager.cpp
        src/gpu/unfolding/NoUnfoldingStrategy.cpp
        src/gpu/unfolding/IterativeDFSUnfoldingStrategy.cpp
        src/pipeline/GPUDrivenPipeline.cpp
        src/renderer/LODController.cpp
        src/renderer/MapCamera2D.cpp
        src/renderer/Renderer.cpp
        src/renderer/RoadRenderer.cpp
        src/ui/DebugUI.cpp
    )

    target_include_directories(CH_MapViewer PRIVATE src)
    if(DEFINED CHMV_MSVC_RUNTIME_LIBRARY)
        set_property(TARGET CH_MapViewer PROPERTY MSVC_RUNTIME_LIBRARY "${CHMV_MSVC_RUNTIME_LIBRARY}")
    endif()

    target_link_libraries(CH_MapViewer PRIVATE
        chmv_data
        glfw
        chmv_glad
        imgui
        OpenGL::GL
    )

    if(MSVC)
        target_compile_options(CH_MapViewer PRIVATE /W4 /permissive-)
    else()
        target_compile_options(CH_MapViewer PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_custom_command(TARGET CH_MapViewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/assets
            $<TARGET_FILE_DIR:CH_MapViewer>/assets
    )
endif()

if(CHMV_BUILD_TESTS)
    enable_testing()

    add_executable(chmv_tests
        tests/TestMain.cpp
    )
    target_link_libraries(chmv_tests PRIVATE chmv_data)
    target_include_directories(chmv_tests PRIVATE src)
    if(DEFINED CHMV_MSVC_RUNTIME_LIBRARY)
        set_property(TARGET chmv_tests PROPERTY MSVC_RUNTIME_LIBRARY "${CHMV_MSVC_RUNTIME_LIBRARY}")
    endif()

    add_test(NAME chmv_tests COMMAND chmv_tests)
endif()
