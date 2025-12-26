cmake_minimum_required(VERSION 3.16)

project(pbf
    VERSION 0.1
    LANGUAGES CXX
)

# Use modern C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable testing infrastructure
include(CTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

