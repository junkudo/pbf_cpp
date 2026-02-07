#pragma once

#include "pbf/vec2f.h"
#include "pbf/vec3f.h"

namespace pbf {
template<int Dim>
struct VecForDim; // primary template (undefined)

template<>
struct VecForDim<2>
{
    using type = vec2f;
};

template<>
struct VecForDim<3>
{
    using type = vec3f;
};

template<int Dim>
using Vec = typename VecForDim<Dim>::type;
}
