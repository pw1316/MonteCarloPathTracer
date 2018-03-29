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
            PWfloat sinTheta = sqrt(1 - x * x);
            PWfloat phi = 2 * PW_PI * y;
            /* No rotate */
            if (fabsf(normal.y - 1) < FLT_EPSILON)
            {
                return PWVector3f(sinTheta * cosf(phi), x, sinTheta * sinf(phi));
            }
            /* Inverse */
            else if (fabsf(normal.y + 1) < FLT_EPSILON)
            {
                return PWVector3f(-sinTheta * cosf(phi), -x, -sinTheta * sinf(phi));
            }
            /* Rotate */
            else
            {
                PWVector3f dir(sinTheta * cosf(phi), x, sinTheta * sinf(phi));
                PWfloat invlen = 1.0f / (1.0f - normal.y * normal.y);
                PWfloat xx = (normal.z * dir.x + normal.x * dir.y + normal.x * normal.y * dir.z) * invlen;
                PWfloat yy = normal.y * dir.y * invlen - dir.z;
                PWfloat zz = (-normal.x * dir.x + normal.z * dir.y + normal.z * normal.y * dir.z) * invlen;
                return PWVector3f(xx, yy, zz);
            }
        }

        __inline__ __device__ PWVector3f samplePhong(curandState *RNG, const PWVector3f& normal, const PWVector3f& indir, const PWuint Ns)
        {
            PWVector3f outdir = indir - normal * dot(indir, normal) * 2;
            PWfloat x = curand_uniform(RNG);
            PWfloat y = curand_uniform(RNG);
            PWfloat sinTheta = sqrt(1 - powf(x, 2.0f / (Ns + 1)));
            PWfloat phi = 2 * PW_PI * y;
            /* No rotate */
            if (fabsf(outdir.y - 1) < FLT_EPSILON)
            {
                return PWVector3f(sinTheta * cosf(phi), x, sinTheta * sinf(phi));
            }
            /* Inverse */
            else if (fabsf(outdir.y + 1) < FLT_EPSILON)
            {
                return PWVector3f(-sinTheta * cosf(phi), -x, -sinTheta * sinf(phi));
            }
            /* Rotate */
            else
            {
                PWVector3f dir(sinTheta * cosf(phi), x, sinTheta * sinf(phi));
                PWfloat invlen = 1.0f / (1.0f - outdir.y * outdir.y);
                PWfloat xx = (outdir.z * dir.x + outdir.x * dir.y + outdir.x * outdir.y * dir.z) * invlen;
                PWfloat yy = outdir.y * dir.y * invlen - dir.z;
                PWfloat zz = (-outdir.x * dir.x + outdir.z * dir.y + outdir.z * outdir.y * dir.z) * invlen;
                return PWVector3f(xx, yy, zz);
            }
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
