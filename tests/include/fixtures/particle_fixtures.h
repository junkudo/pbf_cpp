#pragma once
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "pbf/sph_kernels.h"
#include "pbf/pbf_kernels.h"
#include "pbf/vec2f.h"

namespace testing {

    static std::mt19937 rng(0);

    template <class URBG>
    std::vector<pbf::vec2f>
    jittered_grid(int nx, int ny, float dx, float jitter, URBG& rng)
    {
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);

        std::vector<pbf::vec2f> pos;
        pos.reserve(nx * ny);

        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                pbf::vec2f p{i * dx, j * dx};
                p.x += (uni(rng) - 0.5f) * dx * jitter;
                p.y += (uni(rng) - 0.5f) * dx * jitter;
                pos.push_back(p);
            }
        }

        return pos;
    }

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
        positions = jittered_grid(
            nx,
            ny,
            dx,
            0.2f * dx,
            rng
        );
    }
};
}

