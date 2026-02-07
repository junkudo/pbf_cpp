#include <numbers>
#include "pbf/sph_kernels.h"
#include "pbf/vec.h"

namespace pbf::sph {

    float Poly6<2>::eval(Vec<2> const & rVec, const float h) {
        float r = rVec.length();
        if (r > h)
            return 0.0f;
        float r2 = r * r;
        float h2 = h * h;
        float h8 = h2 * h2 * h2 * h2;
        return 4.0f / (std::numbers::pi_v<float> * h8) * std::pow(h2 - r2, 3);
    }

    float Poly6<2>::evalAtZero(const float h) {
        float h2 = h * h;
        float h6 = h2 * h2 * h2;
        float h8 = h2 * h2 * h2 * h2;
        return 4.0f / (std::numbers::pi_v<float> * h8) * h6;
    }

    Vec<2> Poly6<2>::deriv(Vec<2> const & rVec, const float h) {
        float r = rVec.length();
        if (r > h)
            return Vec<2>(0.0f, 0.0f);

        float r2 = r * r;
        float h2 = h * h;
        float h8 = h2 * h2 * h2 * h2;
        float coeff = 4.0f / (std::numbers::pi_v<float> * h8);
        return -6.0f * coeff * std::pow(h2 - r2, 2) * rVec;
    }

    float Poly6<3>::eval(Vec<3> const & rVec, const float h) {
        float r = rVec.length();
        if (r > h)
            return 0.0f;
        float r2 = r * r;
        float h2 = h * h;
        float h9 = h2 * h2 * h2 * h2 * h;
        return 315.0f / (64.0f * std::numbers::pi_v<float> * h9) * std::pow(h2 - r2, 3);
    }

    float Poly6<3>::evalAtZero(const float h) {
        float h2 = h * h;
        float h9 = h2 * h2 * h2 * h2 * h;
        float h6 = h2 * h2 * h2;
        return 315.0f / (64.0f * std::numbers::pi_v<float> * h9) * h6;
    }

    Vec<3> Poly6<3>::deriv(Vec<3> const & rVec, const float h) {
        float r = rVec.length();
        if (r > h)
            return Vec<3>(0.0f, 0.0f, 0.0f);

        float r2 = r * r;
        float h2 = h * h;
        float h9 = h2 * h2 * h2 * h2 * h;
        float coeff = 315.0f / (64.0f * std::numbers::pi_v<float> * h9);
        return -6.0f * coeff * std::pow(h2 - r2, 2) * rVec;
    }

    float Spikey<2>::eval(Vec<2> const & rVec, const float h) {
        float r = rVec.length();
        if (r > h)
            return 0.0f;
        float h5 = h*h*h*h*h;
        float coeff = 10.0f / (std::numbers::pi_v<float> * h5);
        return coeff * std::pow(h-r, 3);
    }

    float Spikey<2>::evalAtZero(const float h) {
        float h3 = h*h*h;
        float h5 = h*h*h*h*h;
        float coeff = 10.0f / (std::numbers::pi_v<float> * h5);
        return coeff * h3;
    }

    Vec<2> Spikey<2>::deriv(Vec<2> const & rVec, const float h) {
        float r = rVec.length();
        if (r > h || r <= 0.0f)
            return Vec<2>(0.0f, 0.0f);
        float h5 = h*h*h*h*h;
        float coeff = 10.0f / (std::numbers::pi_v<float> * h5);
        float dfdr = -3.0 * coeff * (h - r) * (h - r);
        return dfdr * rVec / r;
    }

    float Spikey<3>::eval(Vec<3> const & rVec, const float h) {
        float r = rVec.length();
        if (r > h)
            return 0.0f;
        float h6 = h*h*h*h*h*h;
        float coeff = 15.0f / (std::numbers::pi_v<float> * h6);
        return coeff * std::pow(h - r, 3);
    }

    float Spikey<3>::evalAtZero(const float h) {
        float h3 = h*h*h;
        float h6 = h*h*h*h*h*h;
        float coeff = 15.0f / (std::numbers::pi_v<float> * h6);
        return coeff * h3;
    }

    Vec<3> Spikey<3>::deriv(Vec<3> const & rVec, const float h) {
        float r = rVec.length();
        if (r > h || r <= 0.0f)
            return Vec<3>(0.0f, 0.0f, 0.0f);
        float h6 = h*h*h*h*h*h;
        float coeff = 15.0f / (std::numbers::pi_v<float> * h6);
        float dfdr = -3.0f * coeff * (h - r) * (h - r);
        return dfdr * rVec / r;
    }
}
