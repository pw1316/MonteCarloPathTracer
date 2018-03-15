#pragma once
#include "stdafx.h"
#include <cmath>
#include <cfloat>

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
namespace Math
{
    const PWdouble EPSILON = 1e-7;
    
    inline PWbool equal(PWdouble x, PWdouble y)
    {
        return std::fabs(x - y) < EPSILON;
    }

    class Vector2i;
    class Vector2d;
    class Vector3d;
    class Vector4d;
    class Matrix44d;

    class Vector2i
    {
    public:
        Vector2i() :m_x_(0), m_y_(0) {}
        Vector2i(PWint x, PWint y) :m_x_(x), m_y_(y) {}

        /* Arithmetic */
        Vector2i operator- () const { return Vector2i(-m_x_, -m_y_); }
        Vector2i operator+ (const Vector2i &rhs) const { return Vector2i(m_x_ + rhs.m_x_, m_y_ + rhs.m_y_); }
        Vector2i &operator+= (const Vector2i &rhs) { m_x_ += rhs.m_x_; m_y_ += rhs.m_y_; return *this; }
        Vector2i operator- (const Vector2i &rhs) const { return Vector2i(m_x_ - rhs.m_x_, m_y_ - rhs.m_y_); }
        Vector2i &operator-= (const Vector2i &rhs) { m_x_ -= rhs.m_x_; m_y_ -= rhs.m_y_; return *this; }
        Vector2i operator* (PWint a) const { return Vector2i(m_x_ * a, m_y_ * a); }
        Vector2i &operator*= (PWint a) { m_x_ *= a; m_y_ *= a; return *this; }
        Vector2i operator/ (PWint a) const { return Vector2i(m_x_ / a, m_y_ / a); }
        Vector2i &operator/= (PWint a) { m_x_ /= a; m_y_ /= a; return *this; }

        /* Logic */
        PWbool operator== (const Vector2i &rhs) { return (m_x_ == rhs.m_x_) && (m_y_ == rhs.m_y_); }
        PWbool operator!= (const Vector2i &rhs) { return !operator==(rhs); }

        /* Vector */
        PWint lengthSquare() const { return m_x_ * m_x_ + m_y_ * m_y_; }
        PWdouble length() const { return std::sqrt(lengthSquare()); }
        PWdouble dot(const Vector2i &rhs) const { return m_x_ * rhs.m_x_ + m_y_*rhs.m_y_; }

        /* getter/setter */
        PWint getX() const { return m_x_; }
        PWint getY() const { return m_y_; }
        void setX(PWint v) { m_x_ = v; }
        void setY(PWint v) { m_y_ = v; }
        void set(PWint x, PWint y) { m_x_ = x; m_y_ = y; }
    private:
        PWint m_x_, m_y_;
    };

    class Vector2d
    {
    public:
        Vector2d() :m_x_(0), m_y_(0) {}
        Vector2d(PWdouble x, PWdouble y) :m_x_(x), m_y_(y) {}

        /* Arithmetic */
        Vector2d operator- () const { return Vector2d(-m_x_, -m_y_); }
        Vector2d operator+ (const Vector2d &rhs) const { return Vector2d(m_x_ + rhs.m_x_, m_y_ + rhs.m_y_); }
        Vector2d &operator+= (const Vector2d &rhs) { m_x_ += rhs.m_x_; m_y_ += rhs.m_y_; return *this; }
        Vector2d operator- (const Vector2d &rhs) const { return Vector2d(m_x_ - rhs.m_x_, m_y_ - rhs.m_y_); }
        Vector2d &operator-= (const Vector2d &rhs) { m_x_ -= rhs.m_x_; m_y_ -= rhs.m_y_; return *this; }
        Vector2d operator* (PWdouble a) const { return Vector2d(m_x_ * a, m_y_ * a); }
        Vector2d &operator*= (PWdouble a) { m_x_ *= a; m_y_ *= a; return *this; }
        Vector2d operator/ (PWdouble a) const { return Vector2d(m_x_ / a, m_y_ / a); }
        Vector2d &operator/= (PWdouble a) { m_x_ /= a; m_y_ /= a; return *this; }

        /* Logic */
        PWbool operator== (const Vector2d &rhs) { return equal(m_x_, rhs.m_x_) && equal(m_y_, rhs.m_y_); }
        PWbool operator!= (const Vector2d &rhs) { return !operator==(rhs); }

        /* Vector */
        PWdouble lengthSquare() const { return m_x_ * m_x_ + m_y_ * m_y_; }
        PWdouble length() const { return std::sqrt(lengthSquare()); }
        Vector2d normal() const { return *this / std::fmax(length(), DBL_MIN); }
        PWdouble dot(const Vector2d &rhs) const { return m_x_ * rhs.m_x_ + m_y_*rhs.m_y_; }
        void normalize() { *this /= std::fmax(length(), DBL_MIN); }

        /* getter/setter */
        PWdouble getX() const { return m_x_; }
        PWdouble getY() const { return m_y_; }
        void setX(PWdouble v) { m_x_ = v; }
        void setY(PWdouble v) { m_y_ = v; }
        void set(PWdouble x, PWdouble y) { m_x_ = x; m_y_ = y; }
    private:
        PWdouble m_x_, m_y_;
    };

    class Vector3d
    {
    public:
        Vector3d() :m_x_(0), m_y_(0), m_z_(0) {}
        Vector3d(PWdouble x, PWdouble y, PWdouble z) : m_x_(x), m_y_(y), m_z_(z) {}

        /* Arithmetic */
        Vector3d operator- () const { return Vector3d(-m_x_, -m_y_, -m_z_); }
        Vector3d operator+ (const Vector3d &rhs) const { return Vector3d(m_x_ + rhs.m_x_, m_y_ + rhs.m_y_, m_z_ + rhs.m_z_); }
        Vector3d &operator+= (const Vector3d &rhs) { m_x_ += rhs.m_x_; m_y_ += rhs.m_y_; m_z_ += rhs.m_z_; return *this; }
        Vector3d operator- (const Vector3d &rhs) const { return Vector3d(m_x_ - rhs.m_x_, m_y_ - rhs.m_y_, m_z_ - rhs.m_z_); }
        Vector3d &operator-= (const Vector3d &rhs) { m_x_ -= rhs.m_x_; m_y_ -= rhs.m_y_; m_z_ -= rhs.m_z_; return *this; }
        Vector3d operator* (PWdouble a) const { return Vector3d(m_x_*a, m_y_*a, m_z_*a); }
        Vector3d &operator*= (PWdouble a) { m_x_ *= a; m_y_ *= a; m_z_ *= a; return *this; }
        Vector3d operator/ (PWdouble a) const { return Vector3d(m_x_ / a, m_y_ / a, m_z_ / a); }
        Vector3d &operator/= (PWdouble a) { m_x_ /= a; m_y_ /= a; m_z_ /= a; return *this; }

        /* Logic */
        PWbool operator== (const Vector3d &rhs) { return equal(m_x_, rhs.m_x_) && equal(m_y_, rhs.m_y_) && equal(m_z_, rhs.m_z_); }
        PWbool operator!= (const Vector3d &rhs) { return !operator==(rhs); }

        /* Vector */
        PWdouble lengthSquare() const { return m_x_ * m_x_ + m_y_ * m_y_ + m_z_ * m_z_; }
        PWdouble length() const { return std::sqrt(lengthSquare()); }
        Vector3d normal() const { return *this / std::fmax(length(), DBL_MIN); }
        PWdouble dot(const Vector3d &rhs) const { return m_x_ * rhs.m_x_ + m_y_ * rhs.m_y_ + m_z_* rhs.m_z_; }
        Vector3d cross(const Vector3d &rhs) const
        {
            return Vector3d(m_y_ * rhs.m_z_ - m_z_ * rhs.m_y_,
                m_z_ * rhs.m_x_ - m_x_ * rhs.m_z_,
                m_x_ * rhs.m_y_ - m_y_ * rhs.m_x_);
        }
        void normalize() { *this /= std::fmax(length(), DBL_MIN); }

        /* getter/setter */
        PWdouble getX() const { return m_x_; }
        PWdouble getY() const { return m_y_; }
        PWdouble getZ() const { return m_z_; }
        void setX(PWdouble v) { m_x_ = v; }
        void setY(PWdouble v) { m_y_ = v; }
        void setZ(PWdouble v) { m_z_ = v; }
        void set(PWdouble x, PWdouble y, PWdouble z) { m_x_ = x; m_y_ = y; m_z_ = z; }

        /* To Other Types */
        inline Vector4d toVector4d0() const;
        inline Vector4d toVector4d1() const;
    private:
        PWdouble m_x_, m_y_, m_z_;
    };

    class Vector4d
    {
    public:
        Vector4d() :m_x_(0), m_y_(0), m_z_(0), m_w_(0) {}
        Vector4d(PWdouble x, PWdouble y, PWdouble z, PWdouble w) : m_x_(x), m_y_(y), m_z_(z), m_w_(w) {}

        /* Arithmetic */
        Vector4d operator/ (PWdouble w) const { return Vector4d(m_x_ / w, m_y_ / w, m_z_ / w, m_w_ / w); }
        Vector4d &operator/= (PWdouble w) { m_x_ /= w; m_y_ /= w; m_z_ /= w; m_w_ /= w; return *this; }

        /* Vector */
        Vector4d normal() const
        {
            if (!equal(m_w_, 0))
            {
                return *this / m_w_;
            }
            return *this;
        }
        void normalize()
        {
            if (!equal(m_w_, 0))
            {
                *this /= m_w_;
            }
        }

        /* getter/setter */
        PWdouble getX() const { return m_x_; }
        PWdouble getY() const { return m_y_; }
        PWdouble getZ() const { return m_z_; }
        PWdouble getW() const { return m_w_; }
        void setX(PWdouble v) { m_x_ = v; }
        void setY(PWdouble v) { m_y_ = v; }
        void setZ(PWdouble v) { m_z_ = v; }
        void setW(PWdouble v) { m_w_ = v; }
        void set(PWdouble x, PWdouble y, PWdouble z, PWdouble w) { m_x_ = x; m_y_ = y; m_z_ = z; m_w_ = w; }

        /* To Other Types */
        inline Vector3d toVector3d() const;
    private:
        PWdouble m_x_, m_y_, m_z_, m_w_;
    };

    class Matrix44d
    {
    public:
        Matrix44d() { setIdentity(); }
        Matrix44d(Vector4d r0, Vector4d r1, Vector4d r2, Vector4d r3)
        {
            row(0) = r0;
            row(1) = r1;
            row(2) = r2;
            row(3) = r3;
        }
        Matrix44d(PWdouble i_00, PWdouble i_01, PWdouble i_02, PWdouble i_03,
            PWdouble i_10, PWdouble i_11, PWdouble i_12, PWdouble i_13,
            PWdouble i_20, PWdouble i_21, PWdouble i_22, PWdouble i_23,
            PWdouble i_30, PWdouble i_31, PWdouble i_32, PWdouble i_33)
        {
            m_data_[0] = i_00; m_data_[1] = i_01; m_data_[2] = i_02; m_data_[3] = i_03;
            m_data_[4] = i_10; m_data_[5] = i_11; m_data_[6] = i_12; m_data_[7] = i_13;
            m_data_[8] = i_20; m_data_[9] = i_21; m_data_[10] = i_22; m_data_[11] = i_23;
            m_data_[12] = i_30; m_data_[13] = i_31; m_data_[14] = i_32; m_data_[15] = i_33;
        }

        /* Arithmetic */
        Matrix44d operator- () const
        {
            return Matrix44d(-m_data_[0], -m_data_[1], -m_data_[2], -m_data_[3],
                -m_data_[4], -m_data_[5], -m_data_[6], -m_data_[7],
                -m_data_[8], -m_data_[9], -m_data_[10], -m_data_[11],
                -m_data_[12], -m_data_[13], -m_data_[14], -m_data_[15]);
        }
        Matrix44d operator+ (const Matrix44d &rhs) const
        {
            return Matrix44d(m_data_[0] + rhs.m_data_[0], m_data_[1] + rhs.m_data_[1], m_data_[2] + rhs.m_data_[2], m_data_[3] + rhs.m_data_[3],
                m_data_[4] + rhs.m_data_[4], m_data_[5] + rhs.m_data_[5], m_data_[6] + rhs.m_data_[6], m_data_[7] + rhs.m_data_[7],
                m_data_[8] + rhs.m_data_[8], m_data_[9] + rhs.m_data_[9], m_data_[10] + rhs.m_data_[10], m_data_[11] + rhs.m_data_[11],
                m_data_[12] + rhs.m_data_[12], m_data_[13] + rhs.m_data_[13], m_data_[14] + rhs.m_data_[14], m_data_[15] + rhs.m_data_[15]);
        }
        Matrix44d &operator+= (const Matrix44d &rhs)
        {
            for (int i = 0; i < 16; i++)
            {
                m_data_[i] += rhs.m_data_[i];
            }
            return *this;
        }
        Matrix44d operator- (const Matrix44d &rhs) const
        {
            return Matrix44d(m_data_[0] - rhs.m_data_[0], m_data_[1] - rhs.m_data_[1], m_data_[2] - rhs.m_data_[2], m_data_[3] - rhs.m_data_[3],
                m_data_[4] - rhs.m_data_[4], m_data_[5] - rhs.m_data_[5], m_data_[6] - rhs.m_data_[6], m_data_[7] - rhs.m_data_[7],
                m_data_[8] - rhs.m_data_[8], m_data_[9] - rhs.m_data_[9], m_data_[10] - rhs.m_data_[10], m_data_[11] - rhs.m_data_[11],
                m_data_[12] - rhs.m_data_[12], m_data_[13] - rhs.m_data_[13], m_data_[14] - rhs.m_data_[14], m_data_[15] - rhs.m_data_[15]);
        }
        Matrix44d &operator-= (const Matrix44d &rhs)
        {
            for (int i = 0; i < 16; i++)
            {
                m_data_[i] -= rhs.m_data_[i];
            }
            return *this;
        }
        Matrix44d operator* (const Matrix44d &rhs) const
        {
            const auto &r_data_ = rhs.m_data_;
            return Matrix44d(m_data_[0] * r_data_[0] + m_data_[1] * r_data_[4] + m_data_[2] * r_data_[8] + m_data_[3] * r_data_[12],
                m_data_[0] * r_data_[1] + m_data_[1] * r_data_[5] + m_data_[2] * r_data_[9] + m_data_[3] * r_data_[13],
                m_data_[0] * r_data_[2] + m_data_[1] * r_data_[6] + m_data_[2] * r_data_[10] + m_data_[3] * r_data_[14],
                m_data_[0] * r_data_[3] + m_data_[1] * r_data_[7] + m_data_[2] * r_data_[11] + m_data_[3] * r_data_[15],
                m_data_[4] * r_data_[0] + m_data_[5] * r_data_[4] + m_data_[6] * r_data_[8] + m_data_[7] * r_data_[12],
                m_data_[4] * r_data_[1] + m_data_[5] * r_data_[5] + m_data_[6] * r_data_[9] + m_data_[7] * r_data_[13],
                m_data_[4] * r_data_[2] + m_data_[5] * r_data_[6] + m_data_[6] * r_data_[10] + m_data_[7] * r_data_[14],
                m_data_[4] * r_data_[3] + m_data_[5] * r_data_[7] + m_data_[6] * r_data_[11] + m_data_[7] * r_data_[15],
                m_data_[8] * r_data_[0] + m_data_[9] * r_data_[4] + m_data_[10] * r_data_[8] + m_data_[11] * r_data_[12],
                m_data_[8] * r_data_[1] + m_data_[9] * r_data_[5] + m_data_[10] * r_data_[9] + m_data_[11] * r_data_[13],
                m_data_[8] * r_data_[2] + m_data_[9] * r_data_[6] + m_data_[10] * r_data_[10] + m_data_[11] * r_data_[14],
                m_data_[8] * r_data_[3] + m_data_[9] * r_data_[7] + m_data_[10] * r_data_[11] + m_data_[11] * r_data_[15],
                m_data_[12] * r_data_[0] + m_data_[13] * r_data_[4] + m_data_[14] * r_data_[8] + m_data_[15] * r_data_[12],
                m_data_[12] * r_data_[1] + m_data_[13] * r_data_[5] + m_data_[14] * r_data_[9] + m_data_[15] * r_data_[13],
                m_data_[12] * r_data_[2] + m_data_[13] * r_data_[6] + m_data_[14] * r_data_[10] + m_data_[15] * r_data_[14],
                m_data_[12] * r_data_[3] + m_data_[13] * r_data_[7] + m_data_[14] * r_data_[11] + m_data_[15] * r_data_[15]);
        }
        Matrix44d &operator*= (const Matrix44d &rhs)
        {
            const auto &r_data_ = rhs.m_data_;
            PWdouble tmp[16] = { m_data_[0] * r_data_[0] + m_data_[1] * r_data_[4] + m_data_[2] * r_data_[8] + m_data_[3] * r_data_[12],
                m_data_[0] * r_data_[1] + m_data_[1] * r_data_[5] + m_data_[2] * r_data_[9] + m_data_[3] * r_data_[13],
                m_data_[0] * r_data_[2] + m_data_[1] * r_data_[6] + m_data_[2] * r_data_[10] + m_data_[3] * r_data_[14],
                m_data_[0] * r_data_[3] + m_data_[1] * r_data_[7] + m_data_[2] * r_data_[11] + m_data_[3] * r_data_[15],
                m_data_[4] * r_data_[0] + m_data_[5] * r_data_[4] + m_data_[6] * r_data_[8] + m_data_[7] * r_data_[12],
                m_data_[4] * r_data_[1] + m_data_[5] * r_data_[5] + m_data_[6] * r_data_[9] + m_data_[7] * r_data_[13],
                m_data_[4] * r_data_[2] + m_data_[5] * r_data_[6] + m_data_[6] * r_data_[10] + m_data_[7] * r_data_[14],
                m_data_[4] * r_data_[3] + m_data_[5] * r_data_[7] + m_data_[6] * r_data_[11] + m_data_[7] * r_data_[15],
                m_data_[8] * r_data_[0] + m_data_[9] * r_data_[4] + m_data_[10] * r_data_[8] + m_data_[11] * r_data_[12],
                m_data_[8] * r_data_[1] + m_data_[9] * r_data_[5] + m_data_[10] * r_data_[9] + m_data_[11] * r_data_[13],
                m_data_[8] * r_data_[2] + m_data_[9] * r_data_[6] + m_data_[10] * r_data_[10] + m_data_[11] * r_data_[14],
                m_data_[8] * r_data_[3] + m_data_[9] * r_data_[7] + m_data_[10] * r_data_[11] + m_data_[11] * r_data_[15],
                m_data_[12] * r_data_[0] + m_data_[13] * r_data_[4] + m_data_[14] * r_data_[8] + m_data_[15] * r_data_[12],
                m_data_[12] * r_data_[1] + m_data_[13] * r_data_[5] + m_data_[14] * r_data_[9] + m_data_[15] * r_data_[13],
                m_data_[12] * r_data_[2] + m_data_[13] * r_data_[6] + m_data_[14] * r_data_[10] + m_data_[15] * r_data_[14],
                m_data_[12] * r_data_[3] + m_data_[13] * r_data_[7] + m_data_[14] * r_data_[11] + m_data_[15] * r_data_[15]
            };
            memcpy(m_data_, tmp, 16 * sizeof(PWdouble));
            return *this;
        }
        Vector4d operator*(const Vector4d &rhs) const
        {
            const auto &x = rhs.getX();
            const auto &y = rhs.getY();
            const auto &z = rhs.getZ();
            const auto &w = rhs.getW();
            return Vector4d(m_data_[0] * x + m_data_[1] * y + m_data_[2] * z + m_data_[3] * w,
                m_data_[4] * x + m_data_[5] * y + m_data_[6] * z + m_data_[7] * w,
                m_data_[8] * x + m_data_[9] * y + m_data_[10] * z + m_data_[11] * w,
                m_data_[12] * x + m_data_[13] * y + m_data_[14] * z + m_data_[15] * w);
        }

        /* Logic */
        PWbool operator== (const Matrix44d &rhs) const
        {
            return equal(m_data_[0], rhs.m_data_[0]) &&
                equal(m_data_[1], rhs.m_data_[1]) &&
                equal(m_data_[2], rhs.m_data_[2]) &&
                equal(m_data_[3], rhs.m_data_[3]) &&
                equal(m_data_[4], rhs.m_data_[4]) &&
                equal(m_data_[5], rhs.m_data_[5]) &&
                equal(m_data_[6], rhs.m_data_[6]) &&
                equal(m_data_[7], rhs.m_data_[7]) &&
                equal(m_data_[8], rhs.m_data_[8]) &&
                equal(m_data_[9], rhs.m_data_[9]) &&
                equal(m_data_[10], rhs.m_data_[10]) &&
                equal(m_data_[11], rhs.m_data_[11]) &&
                equal(m_data_[12], rhs.m_data_[12]) &&
                equal(m_data_[13], rhs.m_data_[13]) &&
                equal(m_data_[14], rhs.m_data_[14]) &&
                equal(m_data_[15], rhs.m_data_[15]);
        }
        PWbool operator!= (const Matrix44d &rhs) const { return !operator==(rhs); }
        PWdouble operator()(int r, int c) const
        {
            if (r >= 0 && r <= 3 && c >= 0 && c <= 3)
            {
                return m_data_[r * 4 + c];
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
            m_data_[0] = 1; m_data_[1] = 0; m_data_[2] = 0; m_data_[3] = 0;
            m_data_[4] = 0; m_data_[5] = 1; m_data_[6] = 0; m_data_[7] = 0;
            m_data_[8] = 0; m_data_[9] = 0; m_data_[10] = 1; m_data_[11] = 0;
            m_data_[12] = 0; m_data_[13] = 0; m_data_[14] = 0; m_data_[15] = 1;
        }
        void setTranslate(const PWdouble &x, const PWdouble &y, const PWdouble &z)
        {
            setIdentity();
            m_data_[3] = x;
            m_data_[7] = y;
            m_data_[11] = z;
        }
        void addTranslate(const PWdouble &x, const PWdouble &y, const PWdouble &z)
        {
            Matrix44d tmp = *this;
            setTranslate(x, y, z);
            this->operator*=(tmp);
        }
        void setTranslate(const Vector3d &v)
        {
            setIdentity();
            m_data_[3] = v.getX();
            m_data_[7] = v.getY();
            m_data_[11] = v.getZ();
        }
        void addTranslate(const Vector3d &v)
        {
            Matrix44d tmp = *this;
            setTranslate(v.getX(), v.getY(), v.getZ());
            this->operator*=(tmp);
        }
        /* angle is rad */
        void setRotate(PWdouble x, PWdouble y, PWdouble z, PWdouble angle)
        {
            auto cosA = std::cos(angle);
            auto sinA = std::sin(angle);
            m_data_[0] = x * x * (1 - cosA) + cosA; m_data_[1] = y * x * (1 - cosA) - z * sinA; m_data_[2] = z * x * (1 - cosA) + y * sinA; m_data_[3] = 0;
            m_data_[4] = x * y * (1 - cosA) + z * sinA; m_data_[5] = y * y * (1 - cosA) + cosA; m_data_[6] = z * y * (1 - cosA) - x * sinA; m_data_[7] = 0;
            m_data_[8] = x * z * (1 - cosA) - y * sinA; m_data_[9] = y * z * (1 - cosA) + x * sinA; m_data_[10] = z * z * (1 - cosA) + cosA; m_data_[11] = 0;
            m_data_[12] = 0; m_data_[13] = 0; m_data_[14] = 0; m_data_[15] = 1;
        }
        /* angle is rad */
        void addRotate(PWdouble x, PWdouble y, PWdouble z, PWdouble angle)
        {
            Matrix44d tmp = *this;
            setRotate(x, y, z, angle);
            this->operator*=(tmp);
        }
        /* angle is rad */
        void setRotate(const Vector3d &v, PWdouble angle)
        {
            auto cosA = std::cos(angle);
            auto sinA = std::sin(angle);
            const auto &x = v.getX();
            const auto &y = v.getY();
            const auto &z = v.getZ();
            m_data_[0] = x * x * (1 - cosA) + cosA; m_data_[1] = y * x * (1 - cosA) - z * sinA; m_data_[2] = z * x * (1 - cosA) + y * sinA; m_data_[3] = 0;
            m_data_[4] = x * y * (1 - cosA) + z * sinA; m_data_[5] = y * y * (1 - cosA) + cosA; m_data_[6] = z * y * (1 - cosA) - x * sinA; m_data_[7] = 0;
            m_data_[8] = x * z * (1 - cosA) - y * sinA; m_data_[9] = y * z * (1 - cosA) + x * sinA; m_data_[10] = z * z * (1 - cosA) + cosA; m_data_[11] = 0;
            m_data_[12] = 0; m_data_[13] = 0; m_data_[14] = 0; m_data_[15] = 1;
        }
        /* angle is rad */
        void addRotate(const Vector3d &v, PWdouble angle)
        {
            Matrix44d tmp = *this;
            setRotate(v.getX(), v.getY(), v.getZ(), angle);
            this->operator*=(tmp);
        }
    private:
        Vector4d &row(int i)
        {
            if (i >= 0 && i <= 3)
            {
                return *reinterpret_cast<Vector4d *>(&m_data_[i * 4]);
            }
            throw 1;
        }
        const Vector4d &row(int i) const
        {
            if (i >= 0 && i <= 3)
            {
                return *reinterpret_cast<const Vector4d *>(&m_data_[i * 4]);
            }
            throw 1;
        }
        PWdouble m_data_[16];
    };

    /* Outside inline member function */
    inline Vector4d Vector3d::toVector4d0() const { return Vector4d(m_x_, m_y_, m_z_, 0); }
    inline Vector4d Vector3d::toVector4d1() const { return Vector4d(m_x_, m_y_, m_z_, 1); }
    inline Vector3d Vector4d::toVector3d() const { return Vector3d(m_x_, m_y_, m_z_); }

    /* Vector3d outer call */
    inline PWdouble lengthSquare(Vector3d &lhs)
    {
        return lhs.lengthSquare();
    }
    inline PWdouble length(Vector3d &lhs)
    {
        return lhs.length();
    }
    inline Vector3d normal(Vector3d &lhs)
    {
        return lhs.normal();
    }
    inline PWdouble dot(const Vector3d &lhs, const Vector3d &rhs)
    {
        return lhs.dot(rhs);
    }
    inline Vector3d cross(const Vector3d &lhs, const Vector3d &rhs)
    {
        return lhs.cross(rhs);
    }

    /* Vector4d outer call */
    inline Vector4d normal(Vector4d &lhs)
    {
        return lhs.normal();
    }
}
