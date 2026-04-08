add_library(pbf STATIC
  pbf_kernels.cpp
  sph_kernels.cpp
  visualization.cpp
)


target_include_directories(pbf
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/raylib/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/raylib/examples/shaders
)

# Warnings
if (MSVC)
    target_compile_options(pbf PRIVATE /W4)
else()
    target_compile_options(pbf PRIVATE -Wall -Wextra -Wpedantic)
endif()


add_executable(pbf_app_2d
    main_2d.cpp
)

target_link_libraries(pbf_app_2d
    PRIVATE pbf
    PRIVATE raylib
)

# find_package(OpenMP)
# if (OpenMP_CXX_FOUND)
#     target_link_libraries(pbf_app_2d PRIVATE OpenMP::OpenMP_CXX)
# endif()
