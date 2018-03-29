#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
using PWbool = bool;
using PWbyte = unsigned char;
using PWint = int;
using PWuint = unsigned int;
using PWlong = long;
using PWfloat = float;
using PWdouble = double;

using PWVector2f = struct _PWVector2f
{
    __host__  __device__ _PWVector2f() {}
    __host__  __device__ _PWVector2f(PWfloat xx, PWfloat yy) : x(xx), y(yy) {}
    PWfloat x, y;
};
using PWVector3f = struct _PWVector3f
{
    __host__  __device__ _PWVector3f() {}
    __host__  __device__ _PWVector3f(PWfloat xx, PWfloat yy, PWfloat zz) : x(xx), y(yy), z(zz) {}
    /* Arithmetic */
    __host__  __device__ _PWVector3f operator-() { return _PWVector3f(-x, -y, -z); }
    __host__  __device__ _PWVector3f operator+ (const _PWVector3f &rhs) const { return _PWVector3f(x + rhs.x, y + rhs.y, z + rhs.z); }
    __host__  __device__ _PWVector3f &operator+= (const _PWVector3f &rhs) { x += rhs.x; y += rhs.y; z += rhs.z; return *this; }
    __host__  __device__ _PWVector3f operator- (const _PWVector3f &rhs) const { return _PWVector3f(x - rhs.x, y - rhs.y, z - rhs.z); }
    __host__  __device__ _PWVector3f &operator-= (const _PWVector3f &rhs) { x -= rhs.x; x -= rhs.x; x -= rhs.x; return *this; }
    __host__  __device__ _PWVector3f operator* (PWfloat a) const { return _PWVector3f(x * a, y * a, z * a); }
    __host__  __device__ _PWVector3f &operator*= (PWfloat a) { x *= a; y *= a; z *= a; return *this; }
    __host__  __device__ _PWVector3f operator/ (PWfloat a) const { return _PWVector3f(x / a, y / a, z / a); }
    __host__  __device__ _PWVector3f &operator/= (PWfloat a) { x /= a; y /= a; z /= a; return *this; }
    PWfloat x, y, z;
};
using PWVector4f = struct _PWVector4f
{
    __host__  __device__ _PWVector4f() {}
    __host__  __device__ _PWVector4f(PWfloat xx, PWfloat yy, PWfloat zz, PWfloat ww) : x(xx), y(yy), z(zz), w(ww) {}
    PWfloat x, y, z, w;
};

const PWuint IMG_WIDTH = 800;
const PWuint IMG_HEIGHT = 600;
const PWuint NUM_SAMPLES = 10;
#define ILLUM 10
#define PW_PI 3.14159265359f
