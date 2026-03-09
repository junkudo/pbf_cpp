#include <gtest/gtest.h>
#include <cmath>
#include "pbf/vec3f.h"

using pbf::vec3f;

TEST(vec3f, constructors_and_access) {
    vec3f v;
    v.x = 1.0f;
    v.y = -2.0f;
    v.z = 3.5f;
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, -2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.5f);

    vec3f w(3.5f, -4.5f, 6.0f);
    EXPECT_FLOAT_EQ(w.x, 3.5f);
    EXPECT_FLOAT_EQ(w.y, -4.5f);
    EXPECT_FLOAT_EQ(w.z, 6.0f);

    w[0] = -1.0f;
    w[1] = 2.0f;
    w[2] = -3.0f;
    EXPECT_FLOAT_EQ(w[0], -1.0f);
    EXPECT_FLOAT_EQ(w[1], 2.0f);
    EXPECT_FLOAT_EQ(w[2], -3.0f);
}

TEST(vec3f, arithmetic_operators) {
    vec3f a(1.0f, 2.0f, -1.0f);
    vec3f b(-3.0f, 4.0f, 0.5f);

    vec3f sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, -2.0f);
    EXPECT_FLOAT_EQ(sum.y, 6.0f);
    EXPECT_FLOAT_EQ(sum.z, -0.5f);

    vec3f diff = a - b;
    EXPECT_FLOAT_EQ(diff.x, 4.0f);
    EXPECT_FLOAT_EQ(diff.y, -2.0f);
    EXPECT_FLOAT_EQ(diff.z, -1.5f);

    vec3f scaled = a * 2.0f;
    EXPECT_FLOAT_EQ(scaled.x, 2.0f);
    EXPECT_FLOAT_EQ(scaled.y, 4.0f);
    EXPECT_FLOAT_EQ(scaled.z, -2.0f);

    vec3f scaled_left = 3.0f * a;
    EXPECT_FLOAT_EQ(scaled_left.x, 3.0f);
    EXPECT_FLOAT_EQ(scaled_left.y, 6.0f);
    EXPECT_FLOAT_EQ(scaled_left.z, -3.0f);

    vec3f divided = a / 2.0f;
    EXPECT_FLOAT_EQ(divided.x, 0.5f);
    EXPECT_FLOAT_EQ(divided.y, 1.0f);
    EXPECT_FLOAT_EQ(divided.z, -0.5f);
}

TEST(vec3f, unary_negation) {
    vec3f v(1.5f, -2.5f, 3.25f);
    vec3f neg = -v;
    EXPECT_FLOAT_EQ(neg.x, -1.5f);
    EXPECT_FLOAT_EQ(neg.y, 2.5f);
    EXPECT_FLOAT_EQ(neg.z, -3.25f);

    vec3f zero = vec3f::zero();
    vec3f neg_zero = -zero;
    EXPECT_FLOAT_EQ(neg_zero.x, 0.0f);
    EXPECT_FLOAT_EQ(neg_zero.y, 0.0f);
    EXPECT_FLOAT_EQ(neg_zero.z, 0.0f);
}

TEST(vec3f, compound_assignment) {
    vec3f v(1.0f, -2.0f, 3.0f);
    vec3f w(0.5f, 4.0f, -1.5f);

    v += w;
    EXPECT_FLOAT_EQ(v.x, 1.5f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, 1.5f);

    v -= w;
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, -2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);

    v *= 2.0f;
    EXPECT_FLOAT_EQ(v.x, 2.0f);
    EXPECT_FLOAT_EQ(v.y, -4.0f);
    EXPECT_FLOAT_EQ(v.z, 6.0f);

    v /= 4.0f;
    EXPECT_FLOAT_EQ(v.x, 0.5f);
    EXPECT_FLOAT_EQ(v.y, -1.0f);
    EXPECT_FLOAT_EQ(v.z, 1.5f);
}

TEST(vec3f, dot_and_lengths) {
    vec3f v(3.0f, -4.0f, 12.0f);
    vec3f w(-2.0f, 5.0f, 1.0f);

    EXPECT_FLOAT_EQ(v.dot(w), -14.0f);
    EXPECT_FLOAT_EQ(v.lengthSquared(), 169.0f);
    EXPECT_FLOAT_EQ(v.length(), 13.0f);
}

TEST(vec3f, normalized) {
    vec3f v(3.0f, 4.0f, 12.0f);
    vec3f n = v.normalized();
    EXPECT_NEAR(n.length(), 1.0f, 1.0e-6f);
    EXPECT_NEAR(n.x, 3.0f / 13.0f, 1.0e-6f);
    EXPECT_NEAR(n.y, 4.0f / 13.0f, 1.0e-6f);
    EXPECT_NEAR(n.z, 12.0f / 13.0f, 1.0e-6f);

    vec3f zero = vec3f::zero();
    vec3f nz = zero.normalized();
    EXPECT_FLOAT_EQ(nz.x, 0.0f);
    EXPECT_FLOAT_EQ(nz.y, 0.0f);
    EXPECT_FLOAT_EQ(nz.z, 0.0f);
}

TEST(vec3f, zero_factory) {
    vec3f zero = vec3f::zero();
    EXPECT_FLOAT_EQ(zero.x, 0.0f);
    EXPECT_FLOAT_EQ(zero.y, 0.0f);
    EXPECT_FLOAT_EQ(zero.z, 0.0f);
}
