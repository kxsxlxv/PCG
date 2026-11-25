cmake_minimum_required(VERSION 3.19)
project(PCGVisualLab LANGUAGES CXX)

set(PCG_SDK_PROJECT "" CACHE PATH "Existing UNIGINE C++ double-precision SDK project")
set(PCG_RUNTIME_DIR "" CACHE PATH "Isolated directory for the test executable and SDK DLLs")
if(NOT EXISTS "${PCG_SDK_PROJECT}/source/cmake/modules/FindEngine.cmake" OR NOT PCG_RUNTIME_DIR)
    message(FATAL_ERROR "Set PCG_SDK_PROJECT and PCG_RUNTIME_DIR")
endif()

set(UNIGINE_DOUBLE 1)
set(UNIGINE_BIN_DIR "${PCG_SDK_PROJECT}/bin")
set(UNIGINE_LIB_DIR "${PCG_SDK_PROJECT}/lib")
set(UNIGINE_INCLUDE_DIR "${PCG_SDK_PROJECT}/include")
list(APPEND CMAKE_MODULE_PATH "${PCG_SDK_PROJECT}/source/cmake/modules")
find_package(Engine REQUIRED MODULE)

add_executable(PCGVisualLab preview.cpp)
target_compile_features(PCGVisualLab PRIVATE cxx_std_17)
target_link_libraries(PCGVisualLab PRIVATE Unigine::Engine)
set_target_properties(PCGVisualLab PROPERTIES
    DEBUG_POSTFIX "d"
    RUNTIME_OUTPUT_DIRECTORY "$<1:${PCG_RUNTIME_DIR}>"
)
