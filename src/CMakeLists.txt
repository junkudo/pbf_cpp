add_library(pbf STATIC
  PositionBasedFluids.cpp
)


target_include_directories(pbf
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Warnings
if (MSVC)
    target_compile_options(pbf PRIVATE /W4)
else()
    target_compile_options(pbf PRIVATE -Wall -Wextra -Wpedantic)
endif()


# --- Executable ---
add_executable(pbf_app
    main.cpp
)

target_link_libraries(pbf_app
    PRIVATE pbf
)
