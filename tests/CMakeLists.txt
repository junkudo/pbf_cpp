include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# Prevent GoogleTest from overriding your compiler flags
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Test-only library
add_library(test_utils STATIC
    particle_fixtures.cpp
)

target_include_directories(test_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)


add_executable(pbf_tests
    sph_kernels_tests.cpp
    pbf_kernels_tests.cpp
    spatial_hash_tests.cpp
    visualization_tests.cpp
    particle_system_tests.cpp
    vec2f_tests.cpp
    vec3f_tests.cpp
)



target_link_libraries(pbf_tests
    PRIVATE
        pbf
        test_utils
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(pbf_tests)
