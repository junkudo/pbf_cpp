#pragma once
#include "pbf/vec2f.h"
// SPH functions used in position based fluids code
namespace pbf::sph {
    template<int Dim>
    struct Poly6;

    template <>
    struct Poly6<2> {
        static float eval(Vec<2> const & rVec, const float h);
        static Vec<2> deriv(Vec<2> const & rVec, const float h);
    };

    template <int Dim>
    struct Spikey;

    template <>
    struct Spikey<2> {
        static float eval(Vec<2> const & rVec, const float h);
        static Vec<2> deriv(Vec<2> const & rVec, const float h);        
    };

}