#pragma once

#include <cmath>

namespace pbf {
struct vec2f
{
    float x;
    float y;

    // --- constructors ---
    vec2f() {}
    constexpr vec2f(float x_, float y_) : x(x_), y(y_) {}

    // --- basic arithmetic ---
    constexpr vec2f operator+(const vec2f& o) const
    {
        return { x + o.x, y + o.y };
    }

    constexpr vec2f operator-(const vec2f& o) const
    {
        return { x - o.x, y - o.y };
    }

    constexpr vec2f operator*(float s) const
    {
        return { x * s, y * s };
    }

    constexpr vec2f operator/(float s) const
    {
        return { x / s, y / s };
    }

    vec2f& operator+=(const vec2f& o)
    {
        x += o.x;
        y += o.y;
        return *this;
    }

    vec2f& operator-=(const vec2f& o)
    {
        x -= o.x;
        y -= o.y;
        return *this;
    }

    vec2f& operator*=(float s)
    {
        x *= s;
        y *= s;
        return *this;
    }

    vec2f& operator/=(float s)
    {
        x /= s;
        y /= s;
        return *this;
    }

    // --- vector ops ---
    constexpr float dot(const vec2f& o) const
    {
        return x * o.x + y * o.y;
    }

    float length() const
    {
        return std::sqrt(dot(*this));
    }

    float lengthSquared() const
    {
        return dot(*this);
    }

    vec2f normalized() const
    {
        float len = length();
        if (len > 0.0f)
            return *this / len;
        return vec2f{};
    }
};

// allow scalar * vector
constexpr vec2f operator*(float s, const vec2f& v)
{
    return v * s;
}

template<int Dim>
struct VecForDim; // primary template (undefined)

template<>
struct VecForDim<2>
{
    using type = vec2f;
};

template<int Dim>
using Vec = typename VecForDim<Dim>::type;
}
