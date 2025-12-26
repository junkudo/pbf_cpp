include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# Prevent GoogleTest from overriding your compiler flags
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(pbf_tests
    sph_kernels_tests.cpp
)

target_link_libraries(pbf_tests
    PRIVATE
        pbf
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(pbf_tests)
