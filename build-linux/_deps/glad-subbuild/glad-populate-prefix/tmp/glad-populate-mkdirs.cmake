# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/ventus/Project/CH_MapViewer/build-linux/_deps/glad-src")
  file(MAKE_DIRECTORY "/home/ventus/Project/CH_MapViewer/build-linux/_deps/glad-src")
endif()
file(MAKE_DIRECTORY
  "/home/ventus/Project/CH_MapViewer/build-linux/_deps/glad-build"
  "/home/ventus/Project/CH_MapViewer/build-linux/_deps/glad-subbuild/glad-populate-prefix"
  "/home/ventus/Project/CH_MapViewer/build-linux/_deps/glad-subbuild/glad-populate-prefix/tmp"
  "/home/ventus/Project/CH_MapViewer/build-linux/_deps/glad-subbuild/glad-populate-prefix/src/glad-populate-stamp"
  "/home/ventus/Project/CH_MapViewer/build-linux/_deps/glad-subbuild/glad-populate-prefix/src"
  "/home/ventus/Project/CH_MapViewer/build-linux/_deps/glad-subbuild/glad-populate-prefix/src/glad-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ventus/Project/CH_MapViewer/build-linux/_deps/glad-subbuild/glad-populate-prefix/src/glad-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ventus/Project/CH_MapViewer/build-linux/_deps/glad-subbuild/glad-populate-prefix/src/glad-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
