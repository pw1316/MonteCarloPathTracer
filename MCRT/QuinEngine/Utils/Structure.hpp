#pragma once
#include <stdafx.h>
#include <algorithm>

namespace Quin::Utils
{
    struct QuinModel
    {
        QuinModel(const std::string& objPath, const std::string& mtlBase)
        {
            tinyobj::LoadObj(&attr, &shapes, &materials, nullptr, objPath.c_str(), mtlBase.c_str());
        }
        tinyobj::attrib_t attr;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
    };

    class QuinPoint3f
    {
    public:
        QuinPoint3f() :m_p{ 0.0f, 0.0f, 0.0f } {}
        explicit QuinPoint3f(FLOAT xx) :m_p{ xx, xx, xx } {}
        explicit QuinPoint3f(const FLOAT xx[]) :m_p{ xx[0], xx[1], xx[2] } {}
        QuinPoint3f(FLOAT xx, FLOAT yy, FLOAT zz) :m_p{ xx, yy, zz } {}

        const FLOAT& x() const { return m_p[0]; }
        const FLOAT& y() const { return m_p[1]; }
        const FLOAT& z() const { return m_p[2]; }
        FLOAT& x() { return m_p[0]; }
        FLOAT& y() { return m_p[1]; }
        FLOAT& z() { return m_p[2]; }
        const FLOAT& operator[](int idx) const
        {
            assert(idx >= 0 && idx <= 2);
            return m_p[idx];
        }
        FLOAT& operator[](int idx)
        {
            assert(idx >= 0 && idx <= 2);
            return m_p[idx];
        }

        QuinPoint3f operator+(const QuinPoint3f& rhs) const
        {
            return QuinPoint3f(m_p[0] + rhs[0], m_p[1] + rhs[1], m_p[2] + rhs[2]);
        }
        QuinPoint3f operator-(const QuinPoint3f& rhs) const
        {
            return QuinPoint3f(m_p[0] - rhs[0], m_p[1] - rhs[1], m_p[2] - rhs[2]);
        }
        QuinPoint3f operator*(FLOAT rhs) const
        {
            return QuinPoint3f(m_p[0] * rhs, m_p[1] * rhs, m_p[2] * rhs);
        }
        BOOL operator==(const QuinPoint3f& rhs) const
        {
            if (m_p[0] == rhs[0] && m_p[1] == rhs[1] && m_p[2] == rhs[2])
            {
                return true;
            }
            return false;
        }
        BOOL operator<(const QuinPoint3f& rhs) const
        {
            return (*this <= rhs) && !(*this == rhs);
        }
        BOOL operator<=(const QuinPoint3f& rhs) const
        {
            if (m_p[0] <= rhs[0] && m_p[1] <= rhs[1] && m_p[2] <= rhs[2])
            {
                return true;
            }
            return false;
        }
        BOOL operator>(const QuinPoint3f& rhs) const
        {
            return (*this >= rhs) && !(*this == rhs);
        }
        BOOL operator>=(const QuinPoint3f& rhs) const
        {
            if (m_p[0] >= rhs[0] && m_p[1] >= rhs[1] && m_p[2] >= rhs[2])
            {
                return true;
            }
            return false;
        }
    private:
        FLOAT m_p[3];
    };

    class QuinAABB
    {
    public:
        QuinAABB(BOOL inf = false) :m_min(inf ? -FLT_MAX : FLT_MAX), m_max(inf ? FLT_MAX : -FLT_MAX) {}
        void Clear()
        {
            m_min.x() = FLT_MAX;
            m_min.y() = FLT_MAX;
            m_min.z() = FLT_MAX;
            m_max.x() = -FLT_MAX;
            m_max.y() = -FLT_MAX;
            m_max.z() = -FLT_MAX;
        }
        void ClearInfinity()
        {
            m_min.x() = -FLT_MAX;
            m_min.y() = -FLT_MAX;
            m_min.z() = -FLT_MAX;
            m_max.x() = FLT_MAX;
            m_max.y() = FLT_MAX;
            m_max.z() = FLT_MAX;
        }
        void Add(const QuinPoint3f& rhs)
        {
            m_min.x() = std::min(m_min.x(), rhs.x());
            m_min.y() = std::min(m_min.y(), rhs.y());
            m_min.z() = std::min(m_min.z(), rhs.z());
            m_max.x() = std::max(m_max.x(), rhs.x());
            m_max.y() = std::max(m_max.y(), rhs.y());
            m_max.z() = std::max(m_max.z(), rhs.z());
        }
        const QuinPoint3f& min() const { return m_min; }
        const QuinPoint3f& max() const { return m_max; }
        QuinPoint3f& min() { return m_min; }
        QuinPoint3f& max() { return m_max; }

        QuinAABB& operator+=(const QuinAABB& rhs)
        {
            Add(rhs.m_min);
            Add(rhs.m_max);
            return *this;
        }
        QuinAABB& operator*=(const QuinAABB& rhs)
        {
            for (int i = 0; i < 3; ++i)
            {
                m_min[i] = std::max(m_min[i], rhs.m_min[i]);
                m_max[i] = std::min(m_max[i], rhs.m_max[i]);
            }
            return *this;
        }
    private:
        QuinPoint3f m_min;
        QuinPoint3f m_max;
    };

    class QuinTriangle
    {
    public:
        QuinTriangle() :m_v{ {}, {}, {} }, m_aabb() {}
        const QuinPoint3f& v0() const { return m_v[0]; }
        const QuinPoint3f& v1() const { return m_v[1]; }
        const QuinPoint3f& v2() const { return m_v[2]; }
        const QuinAABB& aabb() const
        {
            CalcAABB();
            return m_aabb;
        }
        QuinPoint3f& v0() { return m_v[0]; }
        QuinPoint3f& v1() { return m_v[1]; }
        QuinPoint3f& v2() { return m_v[2]; }
        const QuinPoint3f& operator[](int idx) const
        {
            assert(idx >= 0 && idx <= 2);
            return m_v[idx];
        }
        QuinPoint3f& operator[](int idx)
        {
            assert(idx >= 0 && idx <= 2);
            return m_v[idx];
        }
    private:
        void CalcAABB() const
        {
            m_aabb.Clear();
            m_aabb.Add(m_v[0]);
            m_aabb.Add(m_v[1]);
            m_aabb.Add(m_v[2]);
        }

        QuinPoint3f m_v[3];
        mutable QuinAABB m_aabb;
    };
    using QuinTriangleList = std::vector<QuinTriangle>;

    struct CB0
    {
        D3DXMATRIX viewMatrix;
        D3DXMATRIX projMatrix;
        UINT seed;
        UINT prevCount;
        UINT wndW;
        UINT wndH;
    };

    struct CSMaterial
    {
        D3DXVECTOR3 Ka;
        D3DXVECTOR3 Kd;
        D3DXVECTOR3 Ks;
        FLOAT Ns;
        FLOAT Tr;
        FLOAT Ni;
    };

    struct CSTriangle
    {
        UINT v0, v1, v2;
        UINT n0, n1, n2;
        UINT matId;
    };

    struct CSKDTree
    {
        UINT left, right;
        UINT parent;
        D3DXVECTOR3 aabbMin;
        D3DXVECTOR3 aabbMax;
        UINT splitAxis;
        FLOAT splitValue;
        UINT triIds[64];
        UINT numTri;
    };
}
