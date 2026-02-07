#include <gtest/gtest.h>
#include <cmath>
#include "pbf/vec2f.h"

using pbf::vec2f;

TEST(vec2f, constructors_and_access) {
    vec2f v;
    v.x = 1.0f;
    v.y = -2.0f;
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, -2.0f);

    vec2f w(3.5f, -4.5f);
    EXPECT_FLOAT_EQ(w.x, 3.5f);
    EXPECT_FLOAT_EQ(w.y, -4.5f);

    w[0] = -1.0f;
    w[1] = 2.0f;
    EXPECT_FLOAT_EQ(w[0], -1.0f);
    EXPECT_FLOAT_EQ(w[1], 2.0f);
}

TEST(vec2f, arithmetic_operators) {
    vec2f a(1.0f, 2.0f);
    vec2f b(-3.0f, 4.0f);

    vec2f sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, -2.0f);
    EXPECT_FLOAT_EQ(sum.y, 6.0f);

    vec2f diff = a - b;
    EXPECT_FLOAT_EQ(diff.x, 4.0f);
    EXPECT_FLOAT_EQ(diff.y, -2.0f);

    vec2f scaled = a * 2.0f;
    EXPECT_FLOAT_EQ(scaled.x, 2.0f);
    EXPECT_FLOAT_EQ(scaled.y, 4.0f);

    vec2f scaled_left = 3.0f * a;
    EXPECT_FLOAT_EQ(scaled_left.x, 3.0f);
    EXPECT_FLOAT_EQ(scaled_left.y, 6.0f);

    vec2f divided = a / 2.0f;
    EXPECT_FLOAT_EQ(divided.x, 0.5f);
    EXPECT_FLOAT_EQ(divided.y, 1.0f);
}

TEST(vec2f, compound_assignment) {
    vec2f v(1.0f, -2.0f);
    vec2f w(0.5f, 4.0f);

    v += w;
    EXPECT_FLOAT_EQ(v.x, 1.5f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);

    v -= w;
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, -2.0f);

    v *= 2.0f;
    EXPECT_FLOAT_EQ(v.x, 2.0f);
    EXPECT_FLOAT_EQ(v.y, -4.0f);

    v /= 4.0f;
    EXPECT_FLOAT_EQ(v.x, 0.5f);
    EXPECT_FLOAT_EQ(v.y, -1.0f);
}

TEST(vec2f, dot_and_lengths) {
    vec2f v(3.0f, -4.0f);
    vec2f w(-2.0f, 5.0f);

    EXPECT_FLOAT_EQ(v.dot(w), -26.0f);
    EXPECT_FLOAT_EQ(v.lengthSquared(), 25.0f);
    EXPECT_FLOAT_EQ(v.length(), 5.0f);
}

TEST(vec2f, normalized) {
    vec2f v(3.0f, 4.0f);
    vec2f n = v.normalized();
    EXPECT_NEAR(n.length(), 1.0f, 1.0e-6f);
    EXPECT_NEAR(n.x, 0.6f, 1.0e-6f);
    EXPECT_NEAR(n.y, 0.8f, 1.0e-6f);

    vec2f zero = vec2f::zero();
    vec2f nz = zero.normalized();
    EXPECT_FLOAT_EQ(nz.x, 0.0f);
    EXPECT_FLOAT_EQ(nz.y, 0.0f);
}

TEST(vec2f, zero_factory) {
    vec2f zero = vec2f::zero();
    EXPECT_FLOAT_EQ(zero.x, 0.0f);
    EXPECT_FLOAT_EQ(zero.y, 0.0f);
}
