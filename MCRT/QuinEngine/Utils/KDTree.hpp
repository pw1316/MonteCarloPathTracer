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

    class KDPoint3f
    {
    public:
        KDPoint3f() :m_p{ 0.0f, 0.0f, 0.0f } {}
        explicit KDPoint3f(FLOAT xx) :m_p{ xx, xx, xx } {}
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

        KDPoint3f operator+(const KDPoint3f& rhs) const
        {
            return KDPoint3f(m_p[0] + rhs[0], m_p[1] + rhs[1], m_p[2] + rhs[2]);
        }
        KDPoint3f operator-(const KDPoint3f& rhs) const
        {
            return KDPoint3f(m_p[0] - rhs[0], m_p[1] - rhs[1], m_p[2] - rhs[2]);
        }
        BOOL operator==(const KDPoint3f& rhs) const
        {
            if (m_p[0] == rhs[0] && m_p[1] == rhs[1] && m_p[2] == rhs[2])
            {
                return true;
            }
            return false;
        }
        BOOL operator<(const KDPoint3f& rhs) const
        {
            return (*this <= rhs) && !(*this == rhs);
        }
        BOOL operator<=(const KDPoint3f& rhs) const
        {
            if (m_p[0] <= rhs[0] && m_p[1] <= rhs[1] && m_p[2] <= rhs[2])
            {
                return true;
            }
            return false;
        }
        BOOL operator>(const KDPoint3f& rhs) const
        {
            return (*this >= rhs) && !(*this == rhs);
        }
        BOOL operator>=(const KDPoint3f& rhs) const
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

    class KDAABB
    {
    public:
        KDAABB(BOOL inf = false) :m_min(inf ? -FLT_MAX : FLT_MAX), m_max(inf ? FLT_MAX : -FLT_MAX) {}
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
        void Add(const KDPoint3f& rhs)
        {
            m_min.x() = std::min(m_min.x(), rhs.x());
            m_min.y() = std::min(m_min.y(), rhs.y());
            m_min.z() = std::min(m_min.z(), rhs.z());
            m_max.x() = std::max(m_max.x(), rhs.x());
            m_max.y() = std::max(m_max.y(), rhs.y());
            m_max.z() = std::max(m_max.z(), rhs.z());
        }
        const KDPoint3f& min() const { return m_min; }
        const KDPoint3f& max() const { return m_max; }
        KDPoint3f& min() { return m_min; }
        KDPoint3f& max() { return m_max; }

        KDAABB& operator+=(const KDAABB& rhs)
        {
            Add(rhs.m_min);
            Add(rhs.m_max);
            return *this;
        }
        KDAABB& operator*=(const KDAABB& rhs)
        {
            for (int i = 0; i < 3; ++i)
            {
                m_min[i] = std::max(m_min[i], rhs.m_min[i]);
                m_max[i] = std::min(m_max[i], rhs.m_max[i]);
            }
            return *this;
        }
    private:
        KDPoint3f m_min;
        KDPoint3f m_max;
    };

    class KDTriangle
    {
    public:
        KDTriangle() :m_v{ {}, {}, {} }, m_aabb() {}
        const KDPoint3f& v0() const { return m_v[0]; }
        const KDPoint3f& v1() const { return m_v[1]; }
        const KDPoint3f& v2() const { return m_v[2]; }
        const KDAABB& aabb() const
        {
            CalcAABB();
            return m_aabb;
        }
        KDPoint3f& v0() { return m_v[0]; }
        KDPoint3f& v1() { return m_v[1]; }
        KDPoint3f& v2() { return m_v[2]; }
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
        void CalcAABB() const
        {
            m_aabb.Clear();
            m_aabb.Add(m_v[0]);
            m_aabb.Add(m_v[1]);
            m_aabb.Add(m_v[2]);
        }

        KDPoint3f m_v[3];
        mutable KDAABB m_aabb;
    };
    using KDTriangleList = std::vector<KDTriangle>;

    struct KDNode
    {
        KDAxis axis = KDAxis::AXIS_NONE;
        FLOAT split = 0.0f;
        KDAABB aabb{ true };
        std::set<UINT> triangleIds;
        KDNode* left = nullptr;
        KDNode* right = nullptr;
    };

    class KDTree
    {
    public:
        static KDAABB GetNodeAABB(const KDTriangleList& triangles, KDNode* inNode)
        {
            if (inNode == nullptr)
            {
                return KDAABB(true);
            }
            KDAABB outAABB;
            for (auto& triId : inNode->triangleIds)
            {
                outAABB += triangles[triId].aabb();
            }
            return outAABB;
        }
        static KDNode* BuildTree(const tinyobj::attrib_t& attr, const std::vector<tinyobj::shape_t>& shapes)
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
            }

            while (!activeList.empty())
            {
                KDNode* node = activeList.front();
                activeList.pop_front();

                node->aabb *= GetNodeAABB(triangles, node);
                assert(node->aabb.min() <= node->aabb.max());
                /* Large node, Spatial median split */
                if (node->triangleIds.size() > 64U)
                {
                    FLOAT len[3];
                    for (int i = 0; i < 3; ++i)
                    {
                        len[i] = node->aabb.max()[i] - node->aabb.min()[i];
                    }
                    FLOAT maxLen = len[0];
                    UINT axis = 0;
                    for (int i = 1; i < 3; ++i)
                    {
                        if (len[i] > maxLen)
                        {
                            maxLen = len[i];
                            axis = i;
                        }
                    }
                    node->axis = static_cast<KDAxis>(axis + 1);
                    node->split = 0.5f * (node->aabb.max()[axis] + node->aabb.min()[axis]);
                    node->left = new KDNode;
                    node->left->aabb = node->aabb;
                    node->left->aabb.max()[axis] = node->split;
                    node->right = new KDNode;
                    node->right->aabb = node->aabb;
                    node->right->aabb.min()[axis] = node->split;
                    for (auto triId : node->triangleIds)
                    {
                        if (triangles[triId].aabb().min()[axis] < node->split)
                        {
                            node->left->triangleIds.insert(triId);
                        }
                        if (triangles[triId].aabb().max()[axis] > node->split)
                        {
                            node->right->triangleIds.insert(triId);
                        }
                    }
                    node->triangleIds.clear();
                    activeList.push_back(node->left);
                    activeList.push_back(node->right);
                }
                /* Small node, SAH split */
                else
                {
                    //TODO SAH split
                }
            }
            return root;
        }
        static void DestroyTree(KDNode** ppRoot)
        {
            if (ppRoot == nullptr || *ppRoot == nullptr)
            {
                return;
            }
            std::list<KDNode*> bfs;
            bfs.push_back(*ppRoot);
            while (!bfs.empty())
            {
                KDNode* node = bfs.front();
                bfs.pop_front();
                if (node->left)
                {
                    assert(node->right);
                    bfs.push_back(node->left);
                    bfs.push_back(node->right);
                }
                delete node;
            }
            *ppRoot = nullptr;
        }
    };
}