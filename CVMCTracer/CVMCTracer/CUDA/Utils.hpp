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

        __inline__ __device__ PWVector3f sampleHemi(curandState *RNG, const PWVector3f& normal)
        {
            PWfloat x = curand_uniform(RNG);
            PWfloat y = curand_uniform(RNG);
            PWfloat sinTheta = sqrt(1 - x * x);
            PWfloat phi = 2 * PW_PI * y;
            /* No rotate */
            if (abs(normal.y - 1) < FLT_EPSILON)
            {
                return PWVector3f(sinTheta * cos(phi), x, sinTheta * sin(phi));
            }
            /* Inverse */
            else if (abs(normal.y + 1) < FLT_EPSILON)
            {
                return PWVector3f(-sinTheta * cos(phi), -x, -sinTheta * sin(phi));
            }
            /* Rotate */
            else
            {
                PWVector3f dir(sinTheta * cos(phi), x, sinTheta * sin(phi));
                PWfloat invlen = 1.0f / (1.0f - normal.y * normal.y);
                PWfloat xx = (normal.z * dir.x + normal.x * dir.y + normal.x * normal.y * dir.z) * invlen;
                PWfloat yy = normal.y * dir.y * invlen - dir.z;
                PWfloat zz = (-normal.x * dir.x + normal.z * dir.y + normal.z * normal.y * dir.z) * invlen;
                return PWVector3f(xx, yy, zz);
            }
        }
    }
}