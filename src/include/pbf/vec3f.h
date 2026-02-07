#pragma once

#include <cmath>
#include <cassert>

namespace pbf {
struct vec3f
{
    float x;
    float y;
    float z;

    // --- constructors ---
    vec3f() {}
    constexpr vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    float& operator[](int i) {
        assert(i == 0 || i == 1 || i == 2);
        return (&x)[i];
    }

    const float& operator[](int i) const {
        assert(i == 0 || i == 1 || i == 2);
        return (&x)[i];
    }

    // --- basic arithmetic ---
    constexpr vec3f operator+(const vec3f& o) const
    {
        return { x + o.x, y + o.y, z + o.z };
    }

    constexpr vec3f operator-(const vec3f& o) const
    {
        return { x - o.x, y - o.y, z - o.z };
    }

    constexpr vec3f operator*(float s) const
    {
        return { x * s, y * s, z * s };
    }

    constexpr vec3f operator/(float s) const
    {
        return { x / s, y / s, z / s };
    }

    vec3f& operator+=(const vec3f& o)
    {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    vec3f& operator-=(const vec3f& o)
    {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }

    vec3f& operator*=(float s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    vec3f& operator/=(float s)
    {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    // --- vector ops ---
    constexpr float dot(const vec3f& o) const
    {
        return x * o.x + y * o.y + z * o.z;
    }

    float length() const
    {
        return std::sqrt(dot(*this));
    }

    float lengthSquared() const
    {
        return dot(*this);
    }

    vec3f normalized() const
    {
        float len = length();
        if (len > 0.0f)
            return *this / len;
        return vec3f{};
    }

    static vec3f zero()
    {
        return vec3f(0.0f, 0.0f, 0.0f);
    }
};

constexpr vec3f operator*(float s, const vec3f& v)
{
    return v * s;
}
}
