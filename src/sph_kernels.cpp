#include <numbers>
#include "pbf/sph_kernels.h"
#include "pbf/vec2f.h"

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
    
    float Spikey<2>::eval(Vec<2> const & rVec, const float h) {
        float r = rVec.length();
        if (r > h)
            return 0.0f;
        float h5 = h*h*h*h*h;
        float coeff = 10.0f / (std::numbers::pi_v<float> * h5);
        return coeff * std::pow(h-r, 3);
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
}
