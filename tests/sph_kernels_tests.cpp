#include <numbers>
#include <random>
#include <gtest/gtest.h>
#include "pbf/sph_kernels.h"
#include "pbf/vec2f.h"

namespace {
    constexpr float pi = std::numbers::pi_v<float>;
    static std::mt19937 rng(0);

}

template <typename SPHKernel>
void testSPHKernelNormalization(std::string const & implName) {
SCOPED_TRACE(testing::Message() << "Implementation: " << implName);
    // We expect that the sph kernels integrate to 1 over
    // the circle with radius h
    float h = 3.2f;
    float dr = 0.001f;
    float integral = 0.0f;

    float r = 0.0f;
    pbf::vec2f unitVec = pbf::vec2f(1.0f, 0.0f);
    while (r <= h) {

        float f = SPHKernel::eval(unitVec*r, h);
        integral += 2.0f * pi * r * dr * f;
        r += dr;
    }

    float tolerance = 1.0e-3f;
    EXPECT_NEAR(integral, 1.0f, tolerance);
}

TEST(sph_kernels, normalization)
{
    testSPHKernelNormalization<pbf::sph::Poly6<2>>("Poly6");
    testSPHKernelNormalization<pbf::sph::Spikey<2>>("Spikey");
}

template <typename SPHKernel>
void testSPHKernelFiniteDifference(std::string const & implName) {
SCOPED_TRACE(testing::Message() << "Implementation: " << implName);
    // We expect the gradients of the kernels to be consistent
    // with the function evaluation
    float h = 3.2f;
    float dr = 0.001f;

    // --------------------------
    // Test a random inner point
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    pbf::vec2f unitVec = pbf::vec2f(dist(rng), dist(rng));
    unitVec /= unitVec.length();

    {
        float r = dist(rng);
        pbf::vec2f rvec = r*unitVec;
        pbf::vec2f dfdr = SPHKernel::deriv(rvec, h);
        float eps = 1.0e-3f;

        for (int dim = 0; dim < 2; ++dim) {
            pbf::vec2f rvec_plus = rvec;
            rvec_plus[dim] += eps;
            float fplus = SPHKernel::eval(rvec_plus, h);

            pbf::vec2f rvec_minus = rvec;
            rvec_minus[dim] -= eps;
            float fminus = SPHKernel::eval(rvec_minus, h);

            float dfdx = (fplus - fminus) / (2.0f * eps);
            float tolerance = 1.0e-4f;
            EXPECT_NEAR(dfdx, dfdr[dim], tolerance);
        }
    }

    // --------------------------
    // Test at boundaries
    { // Testing at r = 0.0
        float r = 0.0f;
        pbf::vec2f rvec = r*unitVec;
        pbf::vec2f dfdr = SPHKernel::deriv(rvec, h);
        float tolerance = 1.0e-4f;
        for (int dim = 0; dim < 2; ++dim) {
            EXPECT_NEAR(0.0f, dfdr[dim], tolerance);
        }
    }

    { // Testing at r = h
        float r = h;
        pbf::vec2f rvec = r*unitVec;
        pbf::vec2f dfdr = SPHKernel::deriv(rvec, h);
        float tolerance = 1.0e-4f;
        for (int dim = 0; dim < 2; ++dim) {
            EXPECT_NEAR(0.0f, dfdr[dim], tolerance);
        }
    }
}

TEST(sph_kernels, derivative_consistency) {
    testSPHKernelFiniteDifference<pbf::sph::Poly6<2>>("Poly6");
    testSPHKernelFiniteDifference<pbf::sph::Spikey<2>>("Spikey");
}