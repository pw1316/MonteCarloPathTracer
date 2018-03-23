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

        __inline__ __device__ PWVector3f samplePhong(curandState *RNG, const PWVector3f& normal, const PWVector3f& indir, const PWuint Ns)
        {
            PWVector3f outdir;
            outdir.x = indir.x - 2 * (indir.x * normal.x + indir.y * normal.y + indir.z * normal.z) * normal.x;
            outdir.y = indir.y - 2 * (indir.x * normal.x + indir.y * normal.y + indir.z * normal.z) * normal.y;
            outdir.z = indir.z - 2 * (indir.x * normal.x + indir.y * normal.y + indir.z * normal.z) * normal.z;
            PWfloat x = curand_uniform(RNG);
            PWfloat y = curand_uniform(RNG);
            PWfloat sinTheta = sqrt(1 - pow(x, 2.0f / (Ns + 1)));
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
                PWfloat invlen = 1.0f / (1.0f - outdir.y * outdir.y);
                PWfloat xx = (outdir.z * dir.x + outdir.x * dir.y + outdir.x * outdir.y * dir.z) * invlen;
                PWfloat yy = outdir.y * dir.y * invlen - dir.z;
                PWfloat zz = (-outdir.x * dir.x + outdir.z * dir.y + outdir.z * outdir.y * dir.z) * invlen;
                return PWVector3f(xx, yy, zz);
            }
        }

        __inline__ __device__ PWVector3f sampleFresnel(curandState *RNG, const PWVector3f& normal, const PWVector3f& indir, const PWfloat Tr, const PWfloat Ni)
        {
            PWfloat x = curand_uniform(RNG);
            PWVector3f outdir;
            /* Refract */
            if (x < Tr)
            {
                PWfloat ndoti = indir.x * normal.x + indir.y * normal.y + indir.z * normal.z;
                /* In */
                if (ndoti <= 0)
                {
                    PWfloat alpha = -ndoti / Ni - sqrt(1 - (1 - ndoti * ndoti) / Ni / Ni);
                    outdir.x = alpha * normal.x + indir.x / Ni;
                    outdir.y = alpha * normal.y + indir.y / Ni;
                    outdir.z = alpha * normal.z + indir.z / Ni;
                    normalize(outdir);
                }
                /* Out */
                else
                {
                    PWfloat test = 1 - (1 - ndoti * ndoti) * Ni * Ni;
                    if (test < 0)
                    {
                        return PWVector3f(0, 0, 0);
                    }
                    PWfloat alpha = -ndoti * Ni - sqrt(test);
                    outdir.x = alpha * normal.x + indir.x * Ni;
                    outdir.y = alpha * normal.y + indir.y * Ni;
                    outdir.z = alpha * normal.z + indir.z * Ni;
                    normalize(outdir);
                }
            }
            /* Reflect */
            else
            {
                outdir.x = indir.x - 2 * (indir.x * normal.x + indir.y * normal.y + indir.z * normal.z) * normal.x;
                outdir.y = indir.y - 2 * (indir.x * normal.x + indir.y * normal.y + indir.z * normal.z) * normal.y;
                outdir.z = indir.z - 2 * (indir.x * normal.x + indir.y * normal.y + indir.z * normal.z) * normal.z;
            }
            return outdir;
        }
    }
}