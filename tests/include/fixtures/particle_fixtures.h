#pragma once
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "pbf/sph_kernels.h"
#include "pbf/pbf_kernels.h"
#include "pbf/vec2f.h"
#include "pbf/vec3f.h"
#include "test_helpers.h"

namespace testing {

    static std::mt19937 rng(0);

    // Note - this thing is not designed to be able to inject a rng.
    // I'm too lazy to do that so im just going to use a global static
    // with a single seed.
    struct DensityGridFixture : ::testing::Test
{
    using Kernel = pbf::sph::Poly6<2>;
    // Grid parameters
    int nx = 5;
    int ny = 5;
    int nparticles = nx * ny;
    float dx = 0.7f;
    float h  = 1.8f * dx;

    // Particle positions
    std::vector<pbf::vec2f> positions;

    void SetUp() override
    {
        positions = jittered_grid<2>(
            {nx, ny},
            dx,
            0.2f * dx,
            rng
        );
    }
};

    struct DensityGridFixture3D : ::testing::Test
{
    using Kernel = pbf::sph::Poly6<3>;
    // Grid parameters
    int nx = 4;
    int ny = 4;
    int nz = 4;
    int nparticles = nx * ny * nz;
    float dx = 0.7f;
    float h  = 1.8f * dx;

    // Particle positions
    std::vector<pbf::vec3f> positions;

    void SetUp() override
    {
        positions = jittered_grid<3>(
            {nx, ny, nz},
            dx,
            0.2f * dx,
            rng
        );
    }
};
}

