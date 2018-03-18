#pragma once
using PWbool = bool;
using PWbyte = unsigned char;
using PWint = int;
using PWuint = unsigned int;
using PWlong = long;
using PWfloat = float;
using PWdouble = double;
using PWVector2f = struct
{
    PWfloat x, y;
};
using PWVector3f = struct _PWVector3f
{
    _PWVector3f() :x(0), y(0), z(0) {}
    _PWVector3f(PWfloat xx, PWfloat yy, PWfloat zz) :x(xx), y(yy), z(zz) {}
    PWfloat x, y, z;
};
using PWVector4f = struct _PWVector4f
{
    _PWVector4f() :x(0), y(0), z(0), w(0) {}
    _PWVector4f(PWfloat xx, PWfloat yy, PWfloat zz, PWfloat ww) :x(xx), y(yy), z(zz), w(ww) {}
    PWfloat x, y, z, w;
};