#pragma once
#include <stdafx.h>

#include <algorithm>
#include <list>
#include <set>
#include <vector>

#include <tiny_obj_loader.h>

namespace Quin::Utils
{
    enum class KDAxis
    {
        AXIS_NONE,
        AXIS_X,
        AXIS_Y,
        AXIS_Z
    };

    struct KDNode
    {
        KDAxis axis = KDAxis::AXIS_NONE;
        FLOAT split = 0.0f;
        std::set<UINT> triangleIds;
        KDNode* left = nullptr;
        KDNode* right = nullptr;
    };

    class KDPoint3f
    {
    public:
        KDPoint3f() :m_p{ 0.0f, 0.0f, 0.0f } {}
        KDPoint3f(FLOAT xx, FLOAT yy, FLOAT zz) :m_p{ xx, yy, zz } {}
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
    private:
        FLOAT m_p[3];
    };

    class KDAABB
    {
    public:
        KDAABB() :m_min(FLT_MAX, FLT_MAX, FLT_MAX), m_max(-FLT_MAX, -FLT_MAX, -FLT_MAX) {}
        void Clear()
        {
            m_min.x() = FLT_MAX;
            m_min.y() = FLT_MAX;
            m_min.z() = FLT_MAX;
            m_max.x() = -FLT_MAX;
            m_max.y() = -FLT_MAX;
            m_max.z() = -FLT_MAX;
        }
        void Add(const KDPoint3f& rhs)
        {
            m_min.x() = std::min(m_min.x(), rhs.x());
            m_min.y() = std::min(m_min.y(), rhs.y());
            m_min.z() = std::min(m_min.z(), rhs.z());
            m_max.x() = std::max(m_max.x(), rhs.x());
            m_max.y() = std::max(m_max.y(), rhs.y());
            m_max.z() = std::max(m_max.z(), rhs.z());
        }
        void Add(const KDAABB& rhs)
        {
            Add(rhs.m_min);
            Add(rhs.m_max);
        }
        const KDPoint3f& min() const { return m_min; }
        const KDPoint3f& max() const { return m_max; }
        KDPoint3f& min() { return m_min; }
        KDPoint3f& max() { return m_max; }
    private:
        KDPoint3f m_min;
        KDPoint3f m_max;
    };

    class KDTriangle
    {
    public:
        KDTriangle() :m_v{ {}, {}, {} }, m_aabb() {}
        void CalcAABB()
        {
            m_aabb.Clear();
            m_aabb.Add(m_v[0]);
            m_aabb.Add(m_v[1]);
            m_aabb.Add(m_v[2]);
        }
        const KDPoint3f& v0() const { return m_v[0]; }
        const KDPoint3f& v1() const { return m_v[1]; }
        const KDPoint3f& v2() const { return m_v[2]; }
        const KDAABB& aabb() const { return m_aabb; }
        KDPoint3f& v0() { return m_v[0]; }
        KDPoint3f& v1() { return m_v[1]; }
        KDPoint3f& v2() { return m_v[2]; }
        KDAABB& aabb() { return m_aabb; }
        const KDPoint3f& operator[](int idx) const
        {
            assert(idx >= 0 && idx <= 2);
            return m_v[idx];
        }
        KDPoint3f& operator[](int idx)
        {
            assert(idx >= 0 && idx <= 2);
            return m_v[idx];
        }
    private:
        KDPoint3f m_v[3];
        KDAABB m_aabb;
    };
    using KDTriangleList = std::vector<KDTriangle>;

    class KDTree
    {
    public:
        static BOOL GetNodeAABB(const KDTriangleList& triangles, KDNode* inNode, KDAABB& outAABB)
        {
            if (inNode == nullptr)
            {
                return false;
            }
            outAABB.Clear();
            for (auto& triId : inNode->triangleIds)
            {
                outAABB.Add(triangles[triId].aabb());
            }
            return true;
        }
        static void BuildTree(const tinyobj::attrib_t& attr, const std::vector<tinyobj::shape_t>& shapes)
        {
            KDTriangleList triangles;
            for (auto& shape : shapes)
            {
                for (size_t i = 0; i < shape.mesh.indices.size(); i += 3)
                {
                    KDTriangle tri;
                    for (UINT vId = 0; vId < 3; ++vId)
                    {
                        for (UINT coordId = 0; coordId < 3; ++coordId)
                        {
                            tri[vId][coordId] = attr.vertices[3 * shape.mesh.indices[i + vId].vertex_index + coordId];
                        }
                    }
                    triangles.push_back(tri);
                }
            }

            std::list<KDNode*> activeList;
            std::list<UINT> depthList;

            KDNode* root = new KDNode;
            activeList.push_back(root);
            depthList.push_back(0);

            for (UINT i = 0U; i < static_cast<UINT>(triangles.size()); ++i)
            {
                root->triangleIds.insert(i);
                triangles[i].CalcAABB();
            }

            while (!activeList.empty())
            {
                KDNode* node = activeList.front();
                KDAABB nodeAABB;
                activeList.pop_front();
                if (GetNodeAABB(triangles, node, nodeAABB))
                {
                    //TODO
                }
            }
        }
    };
}