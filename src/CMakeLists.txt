add_library(pbf STATIC
  pbf_kernels.cpp
  sph_kernels.cpp
  visualization.cpp
)


target_include_directories(pbf
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/raylib/src
)

# Warnings
if (MSVC)
    target_compile_options(pbf PRIVATE /W4)
else()
    target_compile_options(pbf PRIVATE -Wall -Wextra -Wpedantic)
endif()


# --- Executable ---
add_executable(pbf_app
    main_2d.cpp
)

target_link_libraries(pbf_app
    PRIVATE pbf
    PRIVATE raylib
)
