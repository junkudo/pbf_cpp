cmake_minimum_required(VERSION 3.16)

project(pbf
    VERSION 0.1
    LANGUAGES CXX
)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_GIT_PROGRESS TRUE)

# # Add Raylib dependency
include(FetchContent)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.0
    GIT_SHALLOW TRUE
    )
FetchContent_MakeAvailable(raylib)

# add_subdirectory(external/raylib)

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

