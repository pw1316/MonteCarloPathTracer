#pragma once
#include <stdafx.h>

#include <math.h>
#include <float.h>

namespace PW
{
    namespace CUDA
    {
        __inline__ __device__ void normalize(PWVector3f& v)
        {
            PWfloat len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            if (abs(len) > FLT_EPSILON)
            {
                v.x /= len;
                v.y /= len;
                v.z /= len;
            }
        }
        
        __inline__ __device__ PWVector3f sampleHemi(curandState *RNG)
        {
            PWfloat x = curand_uniform(RNG);
            PWfloat y = curand_uniform(RNG);
            float sinTheta = sqrt(1 - x * x);
            float phi = 2 * PW_PI * y;
            return PWVector3f(sinTheta * cos(phi), x, sinTheta * sin(phi));
        }
    }
}