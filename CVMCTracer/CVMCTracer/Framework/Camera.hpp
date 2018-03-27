#pragma once
#include "stdafx.h"
#include "Math.hpp"

namespace PW
{
    namespace Math
    {
        class Camera
        {
        public:
            static Math::Matrix44f lookAt(PWdouble x, PWdouble y, PWdouble z,
                PWdouble at_x, PWdouble at_y, PWdouble at_z,
                PWdouble up_x, PWdouble up_y, PWdouble up_z)
            {
                Math::Vector3f eye(x, y, z);
                Math::Vector3f forward(at_x - x, at_y - y, at_z - z);
                Math::Vector3f up(up_x, up_y, up_z);

                forward.normalize();
                if (forward.length() == 0.0f) forward = Math::Vector3f(0.0f, 0.0f, -1.0f);
                up = forward.cross(up).cross(forward).normal();
                if (up.length() == 0.0f) up = Math::Vector3f(0.0f, 1.0f, 0.0f);
                Math::Vector3f right = forward.cross(up).normal();
                up = right.cross(forward).normal();

                Math::Matrix44f mat(right.x, right.y, right.x, 0,
                    up.x, up.y, up.z, 0,
                    -forward.x, -forward.y, -forward.z, 0,
                    0, 0, 0, 1);
                Math::Matrix44f translate;
                translate.setTranslate(-eye);
                mat *= translate;
                return mat;
            }
        };
    }
}
