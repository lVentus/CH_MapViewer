# zstd, for the compressed LOD page files: an installed one (a CMake package, or zstd.h and the
# library on the usual search paths; ZSTD_INCLUDE_DIR / ZSTD_LIBRARY point at others), otherwise
# built from source. On Windows it is always built from source: copies found there come with
# other tools and may use another C runtime or need a DLL next to the viewer.
add_library(chmv_zstd INTERFACE)

if(NOT WIN32)
    find_package(zstd CONFIG QUIET)
    if(TARGET zstd::libzstd)
        target_link_libraries(chmv_zstd INTERFACE zstd::libzstd)
    elseif(TARGET zstd::libzstd_static)
        target_link_libraries(chmv_zstd INTERFACE zstd::libzstd_static)
    elseif(TARGET zstd::libzstd_shared)
        target_link_libraries(chmv_zstd INTERFACE zstd::libzstd_shared)
    else()
        find_path(ZSTD_INCLUDE_DIR zstd.h)
        find_library(ZSTD_LIBRARY NAMES zstd zstd_static libzstd)
        if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
            target_include_directories(chmv_zstd SYSTEM INTERFACE "${ZSTD_INCLUDE_DIR}")
            target_link_libraries(chmv_zstd INTERFACE "${ZSTD_LIBRARY}")
        endif()
    endif()
endif()

get_target_property(CHMV_ZSTD_LINKED chmv_zstd INTERFACE_LINK_LIBRARIES)
if(NOT CHMV_ZSTD_LINKED)
    include(FetchContent)
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
    set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        zstd
        GIT_REPOSITORY https://github.com/facebook/zstd.git
        GIT_TAG v1.5.7
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR build/cmake
    )
    # zstd's CMake files may name an older CMake minimum than CMake 4 accepts or warns about.
    set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
    FetchContent_MakeAvailable(zstd)
    unset(CMAKE_POLICY_VERSION_MINIMUM)
    if(DEFINED CHMV_MSVC_RUNTIME_LIBRARY)
        set_property(TARGET libzstd_static PROPERTY
            MSVC_RUNTIME_LIBRARY "${CHMV_MSVC_RUNTIME_LIBRARY}")
    endif()
    target_include_directories(chmv_zstd SYSTEM INTERFACE "${zstd_SOURCE_DIR}/lib")
    target_link_libraries(chmv_zstd INTERFACE libzstd_static)
endif()
