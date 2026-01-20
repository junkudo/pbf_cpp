#include <numbers>
#include <random>
#include <gtest/gtest.h>
#include "pbf/sph_kernels.h"
#include "pbf/pbf_kernels.h"
#include "pbf/vec2f.h"

namespace {
    constexpr float pi = std::numbers::pi_v<float>;
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


}
struct DensityGridFixture2 : ::testing::Test
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


TEST_F(DensityGridFixture2, density_constraint)
{
    float mass = 1.2f;
    std::vector<int> neighbors;
    float rest_density = 0.95f;
    float density = pbf::sph::computeDensity<2, Kernel>(0, mass, h, neighbors, positions);
    float density_constraint = pbf::sph::computeDensityConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    float tol = 1.0e-4f;
    EXPECT_NEAR(density_constraint, density / rest_density - 1.0f, tol);
}