#include <numbers>
#include <gtest/gtest.h>
#include "pbf/sph_kernels.h"
#include "pbf/vec2f.h"

namespace {
    constexpr float pi = std::numbers::pi_v<float>;
}
TEST(sph_kernels, poly6_2D)
{
    // We expect that the sph kernels integrate to 1 over
    // the circle with radius h
    float h = 3.2f;
    float dr = 0.001f;
    float integral = 0.0f;

    float r = 0.0f;
    pbf::vec2f unitVec = pbf::vec2f(1.0f, 0.0f);
    while (r <= h) {

        float f = pbf::sph::Poly6<2>::eval(unitVec*r, h);
        integral += 2.0f * pi * r * dr * f;
        r += dr;
    }

    float tolerance = 1.0e-3f;
    EXPECT_NEAR(integral, 1.0f, tolerance);
}