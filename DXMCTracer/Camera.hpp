#pragma once
#include "stdafx.h"
#include "Math.hpp"

class Camera
{
public:
    static Math::Matrix44d lookAt(PWdouble x, PWdouble y, PWdouble z,
        PWdouble at_x, PWdouble at_y, PWdouble at_z,
        PWdouble up_x, PWdouble up_y, PWdouble up_z)
    {
        Math::Vector3d eye(x, y, z);
        Math::Vector3d forward(at_x - x, at_y - y, at_z - z);
        Math::Vector3d up(up_x, up_y, up_z);

        forward.normalize();
        if (forward.length() == 0.0f) forward.set(0, 0, -1);
        up = forward.cross(up).cross(forward).normal();
        if (up.length() == 0.0f) up.set(0.0f, 1.0f, 0.0f);
        Math::Vector3d right = forward.cross(up).normal();
        up = right.cross(forward).normal();

        Math::Matrix44d mat(right.getX(), right.getY(), right.getZ(), 0,
            up.getX(), up.getY(), up.getZ(), 0,
            -forward.getX(), -forward.getY(), -forward.getZ(), 0,
            0, 0, 0, 1);
        Math::Matrix44d translate;
        translate.setTranslate(-eye);
        mat *= translate;
        return mat;
    }
};
