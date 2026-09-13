# CMake generated Testfile for 
# Source directory: /home/ventus/Project/CH_MapViewer
# Build directory: /home/ventus/Project/CH_MapViewer/build-linux
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("chmv_tests" "/home/ventus/Project/CH_MapViewer/build-linux/chmv_tests")
set_tests_properties("chmv_tests" PROPERTIES  _BACKTRACE_TRIPLES "/home/ventus/Project/CH_MapViewer/CMakeLists.txt;126;add_test;/home/ventus/Project/CH_MapViewer/CMakeLists.txt;0;")
subdirs("_deps/glfw-build")
subdirs("_deps/glad-build")
