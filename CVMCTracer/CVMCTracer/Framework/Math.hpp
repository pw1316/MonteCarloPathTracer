#pragma once
#include "stdafx.h"
#include <cmath>
#include <cfloat>
#include <cstring>

/**
Linear Algebra

A vector in E^3 would be like that:
x = [x, y, z, ...]^T
which means the vector x here is a column vector

A transform matrix would be left producted by vector x
x' = ABx
This means x makes a transform with B, and then A.
So the MVP matrix would be like:
MVP = ViewProject * WorldView * ModelWorld

Anyway
Vector is column form
Transform is Left Product
*/

/* NO GPU USE!!! */
namespace PW {
    namespace Math
    {
        inline PWbool equal(PWfloat x, PWfloat y)
        {
            return std::fabs(x - y) < FLT_EPSILON;
        }

        struct Vector2i
        {
            Vector2i() :x(0), y(0) {}
            Vector2i(PWint xx, PWint yy) :x(xx), y(yy) {}

            /* Arithmetic */
            Vector2i operator- () const { return Vector2i(-x, -y); }
            Vector2i operator+ (const Vector2i &rhs) const { return Vector2i(x + rhs.x, y + rhs.y); }
            Vector2i &operator+= (const Vector2i &rhs) { x += rhs.x; y += rhs.y; return *this; }
            Vector2i operator- (const Vector2i &rhs) const { return Vector2i(x - rhs.x, y - rhs.y); }
            Vector2i &operator-= (const Vector2i &rhs) { x -= rhs.x; y -= rhs.y; return *this; }
            Vector2i operator* (PWint a) const { return Vector2i(x * a, y * a); }
            Vector2i &operator*= (PWint a) { x *= a; y *= a; return *this; }
            Vector2i operator/ (PWint a) const { return Vector2i(x / a, y / a); }
            Vector2i &operator/= (PWint a) { x /= a; y /= a; return *this; }

            /* Logic */
            PWbool operator== (const Vector2i &rhs) { return (x == rhs.x) && (y == rhs.y); }
            PWbool operator!= (const Vector2i &rhs) { return !operator==(rhs); }

            /* Vector */
            PWint lengthSquare() const { return x * x + y * y; }
            PWfloat length() const { return std::sqrt((PWfloat)lengthSquare()); }
            PWint dot(const Vector2i &rhs) const { return x * rhs.x + y * rhs.y; }

            PWint x, y;
        };

        struct Vector2f
        {
            Vector2f() :x(0), y(0) {}
            Vector2f(PWfloat xx, PWfloat yy) :x(xx), y(yy) {}

            /* Arithmetic */
            Vector2f operator- () const { return Vector2f(-x, -y); }
            Vector2f operator+ (const Vector2f &rhs) const { return Vector2f(x + rhs.x, y + rhs.y); }
            Vector2f &operator+= (const Vector2f &rhs) { x += rhs.x; y += rhs.y; return *this; }
            Vector2f operator- (const Vector2f &rhs) const { return Vector2f(x - rhs.x, y - rhs.y); }
            Vector2f &operator-= (const Vector2f &rhs) { x -= rhs.x; y -= rhs.y; return *this; }
            Vector2f operator* (PWfloat a) const { return Vector2f(x * a, y * a); }
            Vector2f &operator*= (PWfloat a) { x *= a; y *= a; return *this; }
            Vector2f operator/ (PWfloat a) const { return Vector2f(x / a, y / a); }
            Vector2f &operator/= (PWfloat a) { x /= a; y /= a; return *this; }

            /* Logic */
            PWbool operator== (const Vector2f &rhs) { return equal(x, rhs.x) && equal(y, rhs.y); }
            PWbool operator!= (const Vector2f &rhs) { return !operator==(rhs); }

            /* Vector */
            PWfloat lengthSquare() const { return x * x + y * y; }
            PWfloat length() const { return std::sqrt(lengthSquare()); }
            Vector2f normal() const { return *this / std::fmax(length(), FLT_MIN); }
            PWfloat dot(const Vector2f &rhs) const { return x * rhs.x + y * rhs.y; }
            void normalize() { *this /= std::fmax(length(), FLT_MIN); }

            PWfloat x, y;
        };

        struct Vector3f
        {
            Vector3f() :x(0), y(0), z(0) {}
            Vector3f(PWfloat xx, PWfloat yy, PWfloat zz) : x(xx), y(yy), z(zz) {}

            /* Arithmetic */
            Vector3f operator- () const { return Vector3f(-x, -y, -z); }
            Vector3f operator+ (const Vector3f &rhs) const { return Vector3f(x + rhs.x, y + rhs.y, z + rhs.z); }
            Vector3f &operator+= (const Vector3f &rhs) { x += rhs.x; y += rhs.y; z += rhs.z; return *this; }
            Vector3f operator- (const Vector3f &rhs) const { return Vector3f(x - rhs.x, y - rhs.y, z - rhs.z); }
            Vector3f &operator-= (const Vector3f &rhs) { x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }
            Vector3f operator* (PWfloat a) const { return Vector3f(x*a, y*a, z*a); }
            Vector3f &operator*= (PWfloat a) { x *= a; y *= a; z *= a; return *this; }
            Vector3f operator/ (PWfloat a) const { return Vector3f(x / a, y / a, z / a); }
            Vector3f &operator/= (PWfloat a) { x /= a; y /= a; z /= a; return *this; }

            /* Logic */
            PWbool operator== (const Vector3f &rhs) { return equal(x, rhs.x) && equal(y, rhs.y) && equal(z, rhs.z); }
            PWbool operator!= (const Vector3f &rhs) { return !operator==(rhs); }

            /* Vector */
            PWfloat lengthSquare() const { return x * x + y * y + z * z; }
            PWfloat length() const { return std::sqrt(lengthSquare()); }
            Vector3f normal() const { return *this / std::fmax(length(), FLT_MIN); }
            PWfloat dot(const Vector3f &rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }
            Vector3f cross(const Vector3f &rhs) const
            {
                return Vector3f(y * rhs.z - z * rhs.y,
                    z * rhs.x - x * rhs.z,
                    x * rhs.y - y * rhs.x);
            }
            void normalize() { *this /= std::fmax(length(), FLT_MIN); }

            PWfloat x, y, z;
        };

        struct Vector4f
        {
            Vector4f() :x(0), y(0), z(0), w(0) {}
            Vector4f(PWfloat xx, PWfloat yy, PWfloat zz, PWfloat ww) : x(xx), y(yy), z(zz), w(ww) {}

            /* Arithmetic */
            Vector4f operator/ (PWfloat w) const { return Vector4f(x / w, y / w, z / w, w / w); }
            Vector4f &operator/= (PWfloat w) { x /= w; y /= w; z /= w; w /= w; return *this; }

            /* Vector */
            Vector4f normal() const
            {
                if (!equal(w, 0.0f))
                {
                    return *this / w;
                }
                return *this;
            }
            void normalize()
            {
                if (!equal(w, 0.0f))
                {
                    *this /= w;
                }
            }

            PWfloat x, y, z, w;
        };

        struct Matrix33f
        {
            __host__ __device__ Matrix33f() { setIdendity(); }
            __host__ __device__ Matrix33f(PWfloat i_00, PWfloat i_01, PWfloat i_02,
                PWfloat i_10, PWfloat i_11, PWfloat i_12,
                PWfloat i_20, PWfloat i_21, PWfloat i_22)
            {
                m_data[0] = i_00; m_data[1] = i_01; m_data[2] = i_02;
                m_data[3] = i_10; m_data[4] = i_11; m_data[5] = i_12;
                m_data[6] = i_20; m_data[7] = i_21; m_data[8] = i_22;
            }

            __host__ __device__ PWfloat det()
            {
                PWfloat res1 = m_data[0] * (m_data[4] * m_data[8] - m_data[5] * m_data[7]);
                PWfloat res2 = -m_data[1] * (m_data[3] * m_data[8] - m_data[5] * m_data[6]);
                PWfloat res3 = m_data[2] * (m_data[3] * m_data[7] - m_data[4] * m_data[6]);
                return res1 + res2 + res3;
            }

            __host__ __device__ void setIdendity()
            {
                m_data[0] = 1; m_data[1] = 0; m_data[2] = 0;
                m_data[3] = 0; m_data[4] = 1; m_data[5] = 0;
                m_data[6] = 0; m_data[7] = 0; m_data[8] = 1;
            }

            PWfloat m_data[9];
        };

        struct Matrix44f
        {
            Matrix44f() { setIdentity(); }
            Matrix44f(Vector4f r0, Vector4f r1, Vector4f r2, Vector4f r3)
            {
                row(0) = r0;
                row(1) = r1;
                row(2) = r2;
                row(3) = r3;
            }
            Matrix44f(PWfloat i_00, PWfloat i_01, PWfloat i_02, PWfloat i_03,
                PWfloat i_10, PWfloat i_11, PWfloat i_12, PWfloat i_13,
                PWfloat i_20, PWfloat i_21, PWfloat i_22, PWfloat i_23,
                PWfloat i_30, PWfloat i_31, PWfloat i_32, PWfloat i_33)
            {
                data[0] = i_00; data[1] = i_01; data[2] = i_02; data[3] = i_03;
                data[4] = i_10; data[5] = i_11; data[6] = i_12; data[7] = i_13;
                data[8] = i_20; data[9] = i_21; data[10] = i_22; data[11] = i_23;
                data[12] = i_30; data[13] = i_31; data[14] = i_32; data[15] = i_33;
            }

            /* Arithmetic */
            Matrix44f operator- () const
            {
                return Matrix44f(-data[0], -data[1], -data[2], -data[3],
                    -data[4], -data[5], -data[6], -data[7],
                    -data[8], -data[9], -data[10], -data[11],
                    -data[12], -data[13], -data[14], -data[15]);
            }
            Matrix44f operator+ (const Matrix44f &rhs) const
            {
                return Matrix44f(data[0] + rhs.data[0], data[1] + rhs.data[1], data[2] + rhs.data[2], data[3] + rhs.data[3],
                    data[4] + rhs.data[4], data[5] + rhs.data[5], data[6] + rhs.data[6], data[7] + rhs.data[7],
                    data[8] + rhs.data[8], data[9] + rhs.data[9], data[10] + rhs.data[10], data[11] + rhs.data[11],
                    data[12] + rhs.data[12], data[13] + rhs.data[13], data[14] + rhs.data[14], data[15] + rhs.data[15]);
            }
            Matrix44f &operator+= (const Matrix44f &rhs)
            {
                for (int i = 0; i < 16; i++)
                {
                    data[i] += rhs.data[i];
                }
                return *this;
            }
            Matrix44f operator- (const Matrix44f &rhs) const
            {
                return Matrix44f(data[0] - rhs.data[0], data[1] - rhs.data[1], data[2] - rhs.data[2], data[3] - rhs.data[3],
                    data[4] - rhs.data[4], data[5] - rhs.data[5], data[6] - rhs.data[6], data[7] - rhs.data[7],
                    data[8] - rhs.data[8], data[9] - rhs.data[9], data[10] - rhs.data[10], data[11] - rhs.data[11],
                    data[12] - rhs.data[12], data[13] - rhs.data[13], data[14] - rhs.data[14], data[15] - rhs.data[15]);
            }
            Matrix44f &operator-= (const Matrix44f &rhs)
            {
                for (int i = 0; i < 16; i++)
                {
                    data[i] -= rhs.data[i];
                }
                return *this;
            }
            Matrix44f operator* (const Matrix44f &rhs) const
            {
                const auto &r_data_ = rhs.data;
                return Matrix44f(data[0] * r_data_[0] + data[1] * r_data_[4] + data[2] * r_data_[8] + data[3] * r_data_[12],
                    data[0] * r_data_[1] + data[1] * r_data_[5] + data[2] * r_data_[9] + data[3] * r_data_[13],
                    data[0] * r_data_[2] + data[1] * r_data_[6] + data[2] * r_data_[10] + data[3] * r_data_[14],
                    data[0] * r_data_[3] + data[1] * r_data_[7] + data[2] * r_data_[11] + data[3] * r_data_[15],
                    data[4] * r_data_[0] + data[5] * r_data_[4] + data[6] * r_data_[8] + data[7] * r_data_[12],
                    data[4] * r_data_[1] + data[5] * r_data_[5] + data[6] * r_data_[9] + data[7] * r_data_[13],
                    data[4] * r_data_[2] + data[5] * r_data_[6] + data[6] * r_data_[10] + data[7] * r_data_[14],
                    data[4] * r_data_[3] + data[5] * r_data_[7] + data[6] * r_data_[11] + data[7] * r_data_[15],
                    data[8] * r_data_[0] + data[9] * r_data_[4] + data[10] * r_data_[8] + data[11] * r_data_[12],
                    data[8] * r_data_[1] + data[9] * r_data_[5] + data[10] * r_data_[9] + data[11] * r_data_[13],
                    data[8] * r_data_[2] + data[9] * r_data_[6] + data[10] * r_data_[10] + data[11] * r_data_[14],
                    data[8] * r_data_[3] + data[9] * r_data_[7] + data[10] * r_data_[11] + data[11] * r_data_[15],
                    data[12] * r_data_[0] + data[13] * r_data_[4] + data[14] * r_data_[8] + data[15] * r_data_[12],
                    data[12] * r_data_[1] + data[13] * r_data_[5] + data[14] * r_data_[9] + data[15] * r_data_[13],
                    data[12] * r_data_[2] + data[13] * r_data_[6] + data[14] * r_data_[10] + data[15] * r_data_[14],
                    data[12] * r_data_[3] + data[13] * r_data_[7] + data[14] * r_data_[11] + data[15] * r_data_[15]);
            }
            Matrix44f &operator*= (const Matrix44f &rhs)
            {
                const auto &r_data_ = rhs.data;
                PWfloat tmp[16] = { data[0] * r_data_[0] + data[1] * r_data_[4] + data[2] * r_data_[8] + data[3] * r_data_[12],
                    data[0] * r_data_[1] + data[1] * r_data_[5] + data[2] * r_data_[9] + data[3] * r_data_[13],
                    data[0] * r_data_[2] + data[1] * r_data_[6] + data[2] * r_data_[10] + data[3] * r_data_[14],
                    data[0] * r_data_[3] + data[1] * r_data_[7] + data[2] * r_data_[11] + data[3] * r_data_[15],
                    data[4] * r_data_[0] + data[5] * r_data_[4] + data[6] * r_data_[8] + data[7] * r_data_[12],
                    data[4] * r_data_[1] + data[5] * r_data_[5] + data[6] * r_data_[9] + data[7] * r_data_[13],
                    data[4] * r_data_[2] + data[5] * r_data_[6] + data[6] * r_data_[10] + data[7] * r_data_[14],
                    data[4] * r_data_[3] + data[5] * r_data_[7] + data[6] * r_data_[11] + data[7] * r_data_[15],
                    data[8] * r_data_[0] + data[9] * r_data_[4] + data[10] * r_data_[8] + data[11] * r_data_[12],
                    data[8] * r_data_[1] + data[9] * r_data_[5] + data[10] * r_data_[9] + data[11] * r_data_[13],
                    data[8] * r_data_[2] + data[9] * r_data_[6] + data[10] * r_data_[10] + data[11] * r_data_[14],
                    data[8] * r_data_[3] + data[9] * r_data_[7] + data[10] * r_data_[11] + data[11] * r_data_[15],
                    data[12] * r_data_[0] + data[13] * r_data_[4] + data[14] * r_data_[8] + data[15] * r_data_[12],
                    data[12] * r_data_[1] + data[13] * r_data_[5] + data[14] * r_data_[9] + data[15] * r_data_[13],
                    data[12] * r_data_[2] + data[13] * r_data_[6] + data[14] * r_data_[10] + data[15] * r_data_[14],
                    data[12] * r_data_[3] + data[13] * r_data_[7] + data[14] * r_data_[11] + data[15] * r_data_[15]
                };
                memcpy(data, tmp, 16 * sizeof(PWfloat));
                return *this;
            }
            Vector4f operator*(const Vector4f &rhs) const
            {
                return Vector4f(data[0] * rhs.x + data[1] * rhs.y + data[2] * rhs.z + data[3] * rhs.w,
                    data[4] * rhs.x + data[5] * rhs.y + data[6] * rhs.z + data[7] * rhs.w,
                    data[8] * rhs.x + data[9] * rhs.y + data[10] * rhs.z + data[11] * rhs.w,
                    data[12] * rhs.x + data[13] * rhs.y + data[14] * rhs.z + data[15] * rhs.w);
            }

            /* Logic */
            PWbool operator== (const Matrix44f &rhs) const
            {
                return equal(data[0], rhs.data[0]) &&
                    equal(data[1], rhs.data[1]) &&
                    equal(data[2], rhs.data[2]) &&
                    equal(data[3], rhs.data[3]) &&
                    equal(data[4], rhs.data[4]) &&
                    equal(data[5], rhs.data[5]) &&
                    equal(data[6], rhs.data[6]) &&
                    equal(data[7], rhs.data[7]) &&
                    equal(data[8], rhs.data[8]) &&
                    equal(data[9], rhs.data[9]) &&
                    equal(data[10], rhs.data[10]) &&
                    equal(data[11], rhs.data[11]) &&
                    equal(data[12], rhs.data[12]) &&
                    equal(data[13], rhs.data[13]) &&
                    equal(data[14], rhs.data[14]) &&
                    equal(data[15], rhs.data[15]);
            }
            PWbool operator!= (const Matrix44f &rhs) const { return !operator==(rhs); }
            PWfloat operator()(int r, int c) const
            {
                if (r >= 0 && r <= 3 && c >= 0 && c <= 3)
                {
                    return data[r * 4 + c];
                }
                throw 1;
            }

            /* Matrix */
            /**
                setXX: Set the matrix as input
                addXX: Current matrix left product the input
            */
            void setIdentity()
            {
                data[0] = 1; data[1] = 0; data[2] = 0; data[3] = 0;
                data[4] = 0; data[5] = 1; data[6] = 0; data[7] = 0;
                data[8] = 0; data[9] = 0; data[10] = 1; data[11] = 0;
                data[12] = 0; data[13] = 0; data[14] = 0; data[15] = 1;
            }
            void setTranslate(const PWfloat &x, const PWfloat &y, const PWfloat &z)
            {
                setIdentity();
                data[3] = x;
                data[7] = y;
                data[11] = z;
            }
            void addTranslate(const PWfloat &x, const PWfloat &y, const PWfloat &z)
            {
                Matrix44f tmp = *this;
                setTranslate(x, y, z);
                this->operator*=(tmp);
            }
            void setTranslate(const Vector3f &v)
            {
                setIdentity();
                data[3] = v.x;
                data[7] = v.y;
                data[11] = v.z;
            }
            void addTranslate(const Vector3f &v)
            {
                Matrix44f tmp = *this;
                setTranslate(v.x, v.y, v.z);
                this->operator*=(tmp);
            }
            /* angle is rad */
            void setRotate(PWfloat x, PWfloat y, PWfloat z, PWfloat angle)
            {
                auto cosA = std::cos(angle);
                auto sinA = std::sin(angle);
                data[0] = x * x * (1 - cosA) + cosA; data[1] = y * x * (1 - cosA) - z * sinA; data[2] = z * x * (1 - cosA) + y * sinA; data[3] = 0;
                data[4] = x * y * (1 - cosA) + z * sinA; data[5] = y * y * (1 - cosA) + cosA; data[6] = z * y * (1 - cosA) - x * sinA; data[7] = 0;
                data[8] = x * z * (1 - cosA) - y * sinA; data[9] = y * z * (1 - cosA) + x * sinA; data[10] = z * z * (1 - cosA) + cosA; data[11] = 0;
                data[12] = 0; data[13] = 0; data[14] = 0; data[15] = 1;
            }
            /* angle is rad */
            void addRotate(PWfloat x, PWfloat y, PWfloat z, PWfloat angle)
            {
                Matrix44f tmp = *this;
                setRotate(x, y, z, angle);
                this->operator*=(tmp);
            }
            /* angle is rad */
            void setRotate(const Vector3f &v, PWfloat angle)
            {
                auto cosA = std::cos(angle);
                auto sinA = std::sin(angle);
                data[0] = v.x * v.x * (1 - cosA) + cosA; data[1] = v.y * v.x * (1 - cosA) - v.z * sinA; data[2] = v.z * v.x * (1 - cosA) + v.y * sinA; data[3] = 0;
                data[4] = v.x * v.y * (1 - cosA) + v.z * sinA; data[5] = v.y * v.y * (1 - cosA) + cosA; data[6] = v.z * v.y * (1 - cosA) - v.x * sinA; data[7] = 0;
                data[8] = v.x * v.z * (1 - cosA) - v.y * sinA; data[9] = v.y * v.z * (1 - cosA) + v.x * sinA; data[10] = v.z * v.z * (1 - cosA) + cosA; data[11] = 0;
                data[12] = 0; data[13] = 0; data[14] = 0; data[15] = 1;
            }
            /* angle is rad */
            void addRotate(const Vector3f &v, PWfloat angle)
            {
                Matrix44f tmp = *this;
                setRotate(v.x, v.y, v.z, angle);
                this->operator*=(tmp);
            }
            Vector4f &row(int i)
            {
                if (i >= 0 && i <= 3)
                {
                    return *reinterpret_cast<Vector4f *>(&data[i * 4]);
                }
                throw 1;
            }
            const Vector4f &row(int i) const
            {
                if (i >= 0 && i <= 3)
                {
                    return *reinterpret_cast<const Vector4f *>(&data[i * 4]);
                }
                throw 1;
            }
            PWfloat data[16];
        };

        /* Vector3f outer call */
        inline PWfloat lengthSquare(Vector3f &lhs)
        {
            return lhs.lengthSquare();
        }
        inline PWfloat length(Vector3f &lhs)
        {
            return lhs.length();
        }
        inline Vector3f normal(Vector3f &lhs)
        {
            return lhs.normal();
        }
        inline PWfloat dot(const Vector3f &lhs, const Vector3f &rhs)
        {
            return lhs.dot(rhs);
        }
        inline Vector3f cross(const Vector3f &lhs, const Vector3f &rhs)
        {
            return lhs.cross(rhs);
        }

        /* Vector4f outer call */
        inline Vector4f normal(Vector4f &lhs)
        {
            return lhs.normal();
        }
    }
}