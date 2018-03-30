#pragma once
#include <stdafx.h>

#include <math.h>
#include <float.h>

namespace PW
{
    namespace CUDA
    {
        /* Vector3 helper BEGIN */
        __inline__ __device__ PWfloat dot(const PWVector3f &lhs, const PWVector3f &rhs)
        { 
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
        }

        __inline__  __device__ PWfloat lengthSquare(const PWVector3f &lhs)
        {
            return lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z;
        }

        __inline__  __device__ PWfloat length(const PWVector3f &lhs)
        {
            return sqrt(lengthSquare(lhs));
        }

        __inline__ __device__ void normalize(PWVector3f &lhs)
        {
            PWfloat len = length(lhs);
            if (fabsf(len) > FLT_EPSILON)
            {
                lhs /= len;
            }
        }

        __inline__ __device__ PWVector3f cross(const PWVector3f &lhs, const PWVector3f &rhs)
        {
            return _PWVector3f(
                lhs.y * rhs.z - lhs.z * rhs.y,
                lhs.z * rhs.x - lhs.x * rhs.z,
                lhs.x * rhs.y - lhs.y * rhs.x
            );
        }
        /* Vector3 helper END */

        __inline__ __device__ PWVector3f sampleHemi(curandState *RNG, const PWVector3f& normal)
        {
            PWfloat x = curand_uniform(RNG);
            PWfloat y = curand_uniform(RNG);
            PWfloat sinT = sqrtf(x);
            PWfloat cosT = sqrtf(1 - x);
            PWfloat phi = 2 * PW_PI * y;
            PWVector3f outdir(sinT * cosf(phi), cosT, sinT * sinf(phi));
            /* Inverse */
            if (fabsf(normal.y + 1) < FLT_EPSILON)
            {
                outdir = -outdir;
            }
            /* Rotate */
            else if(fabsf(normal.y - 1) >= FLT_EPSILON)
            {
                PWVector3f dir = outdir;
                PWfloat invlen = 1.0f / sqrtf(1.0f - normal.y * normal.y);
                PWfloat len = 1.0f / invlen;
                outdir.x = (normal.z * dir.x + normal.x * normal.y * dir.z) * invlen + normal.x * dir.y;
                outdir.y = normal.y * dir.y - dir.z * len;
                outdir.z = (-normal.x * dir.x + normal.z * normal.y * dir.z) * invlen + normal.z * dir.y;
            }
            return outdir;
        }

        __inline__ __device__ PWVector3f samplePhong(curandState *RNG, const PWVector3f& normal, const PWVector3f& indir, const PWuint Ns)
        {
            PWfloat x = curand_uniform(RNG);
            PWfloat y = curand_uniform(RNG);
            PWfloat cosT = powf(x, 1.0f / (Ns + 1));
            PWfloat sinT = sqrtf(1 - cosT * cosT);
            PWfloat phi = 2 * PW_PI * y;
            PWVector3f halfdir(sinT * cosf(phi), cosT, sinT * sinf(phi));
            /* Inverse */
            if (fabsf(normal.y + 1) < FLT_EPSILON)
            {
                halfdir = -halfdir;
            }
            /* Rotate */
            else if (fabsf(normal.y - 1) >= FLT_EPSILON)
            {
                PWVector3f dir = halfdir;
                PWfloat invlen = 1.0f / sqrtf(1.0f - normal.y * normal.y);
                halfdir.x = (normal.z * dir.x + normal.x * normal.y * dir.z) * invlen + normal.x * dir.y;
                halfdir.y = normal.y * dir.y - dir.z / invlen;
                halfdir.z = (-normal.x * dir.x + normal.z * normal.y * dir.z) * invlen + normal.z * dir.y;
            }
            return indir - halfdir * dot(indir, halfdir) * 2;
        }

        __inline__ __device__ PWVector3f sampleFresnel(curandState *RNG, const PWVector3f& normal, const PWVector3f& indir, PWfloat Tr, const PWfloat Ni)
        {
            PWfloat x = curand_uniform(RNG);
            PWVector3f outdir;
            PWfloat ndoti = dot(indir, normal);
            Tr = Tr * (1 - powf(1 - fabsf(ndoti), 5));
            /* Refract */
            if (x < Tr)
            {
                /* In */
                if (ndoti <= 0)
                {
                    PWfloat alpha = -ndoti / Ni - sqrt(1 - (1 - ndoti * ndoti) / Ni / Ni);
                    outdir = normal * alpha + indir / Ni;
                    normalize(outdir);
                }
                /* Out */
                else
                {
                    PWfloat test = 1 - (1 - ndoti * ndoti) * Ni * Ni;
                    /* Full reflect */
                    if (test < 0)
                    {
                        outdir = indir - normal * dot(indir, normal) * 2;
                    }
                    /* With refract */
                    else
                    {
                        PWfloat alpha = -ndoti * Ni + sqrt(test);
                        outdir = normal * alpha + indir * Ni;
                        normalize(outdir);
                    }
                }
            }
            /* Reflect */
            else
            {
                outdir = indir - normal * dot(indir, normal) * 2;
            }
            return outdir;
        }
    }
}
