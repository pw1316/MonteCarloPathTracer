#pragma once
#include <stdafx.h>

#include <algorithm>
#include <fstream>
#include <list>
#include <set>
#include <vector>

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

    struct KDSplit
    {
        KDSplit() : axis(KDAxis::AXIS_NONE), value(0.0f) {}
        KDSplit(KDAxis ax, FLOAT v) : axis(ax), value(v) {}
        BOOL operator<(const KDSplit& rhs) const
        {
            if (axis < rhs.axis)
            {
                return true;
            }
            if (axis > rhs.axis)
            {
                return false;
            }
            if (value < rhs.value)
            {
                return true;
            }
            return false;
        }
        KDAxis axis;
        FLOAT value;
    };

    struct KDNode
    {
        KDAABB aabb{ true };
        KDSplit split;
        KDNode* left = nullptr;
        KDNode* right = nullptr;
        std::set<UINT> triangleIds;
    };

    class KDTree
    {
    public:
        KDTree(const tinyobj::attrib_t& attr, const std::vector<tinyobj::shape_t>& shapes)
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

            m_root = new KDNode;
            activeList.push_back(m_root);
            depthList.push_back(0);

            for (UINT i = 0U; i < static_cast<UINT>(triangles.size()); ++i)
            {
                m_root->triangleIds.insert(i);
            }
            m_root->aabb = GetNodeAABB(triangles, m_root);

            while (!activeList.empty())
            {
                KDNode* node = activeList.front();
                UINT depth = depthList.front();
                activeList.pop_front();
                depthList.pop_front();

                assert(node->aabb.min() <= node->aabb.max());
                /* Large node, Spatial median split */
                if (node->triangleIds.size() > 64U)
                {
                    KDPoint3f bbsize = node->aabb.max() - node->aabb.min();
                    FLOAT maxSize = bbsize[0];
                    UINT iaxis = 0;
                    for (int i = 1; i < 3; ++i)
                    {
                        if (bbsize[i] > maxSize)
                        {
                            maxSize = bbsize[i];
                            iaxis = i;
                        }
                    }
                    node->split.axis = static_cast<KDAxis>(iaxis + 1);
                    node->split.value = 0.5f * (node->aabb.max()[iaxis] + node->aabb.min()[iaxis]);
                    node->left = new KDNode;
                    node->left->aabb = node->aabb;
                    node->left->aabb.max()[iaxis] = node->split.value;
                    node->right = new KDNode;
                    node->right->aabb = node->aabb;
                    node->right->aabb.min()[iaxis] = node->split.value;
                    for (auto triId : node->triangleIds)
                    {
                        BOOL good = false;
                        /* On the split plane, To left */
                        if (triangles[triId].aabb().min()[iaxis] == triangles[triId].aabb().max()[iaxis] &&
                            triangles[triId].aabb().min()[iaxis] == node->split.value)
                        {
                            node->left->triangleIds.insert(triId);
                            good = true;
                        }
                        else
                        {
                            if (triangles[triId].aabb().min()[iaxis] < node->split.value)
                            {
                                node->left->triangleIds.insert(triId);
                                good = true;
                            }
                            if (triangles[triId].aabb().max()[iaxis] > node->split.value)
                            {
                                node->right->triangleIds.insert(triId);
                                good = true;
                            }
                        }
                        assert(good);
                    }
                    node->left->aabb *= GetNodeAABB(triangles, node->left);// Clip tri
                    node->right->aabb *= GetNodeAABB(triangles, node->right);//Clip tri
                    
                    node->triangleIds.clear();
                    activeList.push_back(node->left);
                    depthList.push_back(depth + 1);
                    activeList.push_back(node->right);
                    depthList.push_back(depth + 1);
                }
                /* Small node, SAH split */
                else
                {
                    constexpr FLOAT Cts = 0.0f;
                    std::set<KDSplit> splitList;
                    for (UINT axis = 0U; axis < 3U; ++axis)
                    {
                        for (const auto tri : node->triangleIds)
                        {
                            splitList.insert(KDSplit(static_cast<KDAxis>(axis + 1), triangles[tri][0][axis]));
                            splitList.insert(KDSplit(static_cast<KDAxis>(axis + 1), triangles[tri][1][axis]));
                            splitList.insert(KDSplit(static_cast<KDAxis>(axis + 1), triangles[tri][2][axis]));
                        }
                    }
                    KDPoint3f bbsize = node->aabb.max() - node->aabb.min();
                    FLOAT A0 = bbsize[0] * bbsize[1] + bbsize[1] * bbsize[2] + bbsize[2] * bbsize[0];
                    FLOAT SAH0 = static_cast<FLOAT>(node->triangleIds.size());

                    KDSplit minSplit;
                    FLOAT minSAH = FLT_MAX;
                    for (auto& split : splitList)
                    {
                        INT iaxis = static_cast<INT>(split.axis) - 1;
                        if (split.value < node->aabb.min()[iaxis] || split.value > node->aabb.max()[iaxis])
                        {
                            continue;
                        }
                        UINT numL = 0U;
                        UINT numR = 0U;
                        KDAABB aabbL = node->aabb;
                        KDAABB aabbR = node->aabb;
                        aabbL.max()[iaxis] = split.value;
                        aabbR.min()[iaxis] = split.value;
                        KDAABB aabbLt;
                        KDAABB aabbRt;
                        for (auto triId : node->triangleIds)
                        {
                            BOOL good = false;
                            /* On the split plane, To left */
                            if (triangles[triId].aabb().min()[iaxis] == triangles[triId].aabb().max()[iaxis] &&
                                triangles[triId].aabb().min()[iaxis] == split.value)
                            {
                                ++numL;
                                aabbLt += triangles[triId].aabb();
                                good = true;
                            }
                            else
                            {
                                if (triangles[triId].aabb().min()[iaxis] < split.value)
                                {
                                    ++numL;
                                    aabbLt += triangles[triId].aabb();
                                    good = true;
                                }
                                if (triangles[triId].aabb().max()[iaxis] > split.value)
                                {
                                    ++numR;
                                    aabbRt += triangles[triId].aabb();
                                    good = true;
                                }
                            }
                            assert(good);
                        }
                        aabbL *= aabbLt;
                        aabbR *= aabbRt;

                        KDPoint3f bbsizeL = aabbL.max() - aabbL.min();
                        KDPoint3f bbsizeR = aabbR.max() - aabbR.min();
                        FLOAT AL = bbsizeL[0] * bbsizeL[1] + bbsizeL[1] * bbsizeL[2] + bbsizeL[2] * bbsizeL[0];
                        FLOAT AR = bbsizeR[0] * bbsizeR[1] + bbsizeR[1] * bbsizeR[2] + bbsizeR[2] * bbsizeR[0];
                        FLOAT SAH = (AL * numL + AR * numR) / A0 + Cts;
                        if (SAH < minSAH)
                        {
                            minSAH = SAH;
                            minSplit = split;
                        }
                    }
                    if (minSAH < SAH0)
                    {
                        INT iaxis = static_cast<INT>(minSplit.axis) - 1;
                        node->split.axis = minSplit.axis;
                        node->split.value = minSplit.value;
                        node->left = new KDNode;
                        node->left->aabb = node->aabb;
                        node->left->aabb.max()[iaxis] = node->split.value;
                        node->right = new KDNode;
                        node->right->aabb = node->aabb;
                        node->right->aabb.min()[iaxis] = node->split.value;
                        for (auto triId : node->triangleIds)
                        {
                            BOOL good = false;
                            /* On the split plane, To left */
                            if (triangles[triId].aabb().min()[iaxis] == triangles[triId].aabb().max()[iaxis] &&
                                triangles[triId].aabb().min()[iaxis] == node->split.value)
                            {
                                node->left->triangleIds.insert(triId);
                                good = true;
                            }
                            else
                            {
                                if (triangles[triId].aabb().min()[iaxis] < node->split.value)
                                {
                                    node->left->triangleIds.insert(triId);
                                    good = true;
                                }
                                if (triangles[triId].aabb().max()[iaxis] > node->split.value)
                                {
                                    node->right->triangleIds.insert(triId);
                                    good = true;
                                }
                            }
                            assert(good);
                        }
                        node->left->aabb *= GetNodeAABB(triangles, node->left);
                        node->right->aabb *= GetNodeAABB(triangles, node->right);

                        node->triangleIds.clear();
                        activeList.push_back(node->left);
                        depthList.push_back(depth + 1);
                        activeList.push_back(node->right);
                        depthList.push_back(depth + 1);
                    }
                }
            }
        }
        ~KDTree()
        {
            if (m_root == nullptr)
            {
                return;
            }
            std::list<KDNode*> bfs;
            bfs.push_back(m_root);
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
            m_root = nullptr;
        }
        KDTree(const KDTree& rhs) = delete;
        KDTree(KDTree&& rhs)
        {
            m_root = rhs.m_root;
            rhs.m_root = nullptr;
        }
        KDTree& operator=(const KDTree& rhs) = delete;
        KDTree& operator=(KDTree&& rhs)
        {
            std::swap(m_root, rhs.m_root);
        }
        void Dump()
        {
            std::ofstream dumpFile("kdtree.obj");
            dumpFile << "g default\n";

        }
    private:
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
        KDNode* m_root = nullptr;
    };
}