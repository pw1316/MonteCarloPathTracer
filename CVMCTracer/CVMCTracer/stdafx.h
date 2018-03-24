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
    __host__  __device__ _PWVector2f() :x(0), y(0) {}
    __host__  __device__ _PWVector2f(PWfloat xx, PWfloat yy) : x(xx), y(yy) {}
    PWfloat x, y;
};
using PWVector3f = struct _PWVector3f
{
    __host__  __device__ _PWVector3f() :x(0), y(0), z(0) {}
    __host__  __device__ _PWVector3f(PWfloat xx, PWfloat yy, PWfloat zz) : x(xx), y(yy), z(zz) {}
    __host__  __device__ _PWVector3f operator-() { return _PWVector3f(-x, -y, -z); }
    PWfloat x, y, z;
};
using PWVector4f = struct _PWVector4f
{
    __host__  __device__ _PWVector4f() :x(0), y(0), z(0), w(0) {}
    __host__  __device__ _PWVector4f(PWfloat xx, PWfloat yy, PWfloat zz, PWfloat ww) : x(xx), y(yy), z(zz), w(ww) {}
    PWfloat x, y, z, w;
};

const PWuint IMG_WIDTH = 800;
const PWuint IMG_HEIGHT = 600;
const PWuint NUM_SAMPLES = 1000;
#define ILLUM 50
#define PW_PI 3.14159265359f
