#pragma once
#include <stdafx.h>

#include <algorithm>
#include <fstream>
#include <list>
#include <set>
#include <vector>

#include <Utils/Structure.hpp>

namespace Quin::Utils
{
    enum class KDAxis
    {
        AXIS_NONE,
        AXIS_X,
        AXIS_Y,
        AXIS_Z
    };

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
        QuinAABB aabb{ true };
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
            QuinTriangleList triangles;
            for (auto& shape : shapes)
            {
                for (size_t i = 0; i < shape.mesh.indices.size(); i += 3)
                {
                    QuinTriangle tri;
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
            UINT maxDepth = 0U;

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
                if (depth > maxDepth)
                {
                    maxDepth = depth;
                }

                assert(node->aabb.min() <= node->aabb.max());
                if (depth >= 32)
                {
                    continue;
                }
                /* Large node, Spatial median split */
                if (node->triangleIds.size() > 64U)
                {
                    QuinPoint3f bbsize = node->aabb.max() - node->aabb.min();
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
                    QuinPoint3f bbsize = node->aabb.max() - node->aabb.min();
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
                        QuinAABB aabbL = node->aabb;
                        QuinAABB aabbR = node->aabb;
                        aabbL.max()[iaxis] = split.value;
                        aabbR.min()[iaxis] = split.value;
                        QuinAABB aabbLt;
                        QuinAABB aabbRt;
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

                        QuinPoint3f bbsizeL = aabbL.max() - aabbL.min();
                        QuinPoint3f bbsizeR = aabbR.max() - aabbR.min();
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
        void Dump() const
        {
            if (m_root == nullptr)
            {
                return;
            }
            std::ofstream dumpFile("kdtree.obj");
            dumpFile << "g default\n";
            std::list<KDNode*> bfsNode;
            std::list<QuinAABB> bfsAABB;
            bfsNode.push_back(m_root);
            bfsAABB.push_back(m_root->aabb);
            UINT numV = 1U;

            while (!bfsNode.empty())
            {
                auto node = bfsNode.front();
                auto aabb = bfsAABB.front();
                bfsNode.pop_front();
                bfsAABB.pop_front();

                if (node->split.axis != KDAxis::AXIS_NONE)
                {
                    assert(node->left && node->right);
                    switch (node->split.axis)
                    {
                    case KDAxis::AXIS_X:
                        dumpFile << "v " << node->split.value << " " << aabb.min()[1] << " " << aabb.min()[2] << "\n";
                        dumpFile << "v " << node->split.value << " " << aabb.min()[1] << " " << aabb.max()[2] << "\n";
                        dumpFile << "v " << node->split.value << " " << aabb.max()[1] << " " << aabb.max()[2] << "\n";
                        dumpFile << "v " << node->split.value << " " << aabb.max()[1] << " " << aabb.min()[2] << "\n";
                        dumpFile << "f " << numV << " " << numV + 1 << " " << numV + 2 << " " << numV + 3 << "\n";
                        break;
                    case KDAxis::AXIS_Y:
                        dumpFile << "v " << aabb.min()[0] << " " << node->split.value << " " << aabb.min()[2] << "\n";
                        dumpFile << "v " << aabb.max()[0] << " " << node->split.value << " " << aabb.min()[2] << "\n";
                        dumpFile << "v " << aabb.max()[0] << " " << node->split.value << " " << aabb.max()[2] << "\n";
                        dumpFile << "v " << aabb.min()[0] << " " << node->split.value << " " << aabb.max()[2] << "\n";
                        dumpFile << "f " << numV << " " << numV + 1 << " " << numV + 2 << " " << numV + 3 << "\n";
                        break;
                    case KDAxis::AXIS_Z:
                        dumpFile << "v " << aabb.min()[0] << " " << aabb.min()[1] << " " << node->split.value << "\n";
                        dumpFile << "v " << aabb.min()[0] << " " << aabb.max()[1] << " " << node->split.value << "\n";
                        dumpFile << "v " << aabb.max()[0] << " " << aabb.max()[1] << " " << node->split.value << "\n";
                        dumpFile << "v " << aabb.max()[0] << " " << aabb.min()[1] << " " << node->split.value << "\n";
                        dumpFile << "f " << numV << " " << numV + 1 << " " << numV + 2 << " " << numV + 3 << "\n";
                        break;
                    default:
                        assert(false);
                        break;
                    }
                    numV += 4;
                    int iaxis = static_cast<int>(node->split.axis) - 1;
                    QuinAABB aabbL = aabb;
                    aabbL.max()[iaxis] = node->split.value;
                    QuinAABB aabbR = aabb;
                    aabbR.min()[iaxis] = node->split.value;
                    bfsNode.push_back(node->left);
                    bfsAABB.push_back(aabbL);
                    bfsNode.push_back(node->right);
                    bfsAABB.push_back(aabbR);
                }
            }
            dumpFile.close();
        }

        KDNode* m_root = nullptr;
    private:
        static QuinAABB GetNodeAABB(const QuinTriangleList& triangles, KDNode* inNode)
        {
            if (inNode == nullptr)
            {
                return QuinAABB(true);
            }
            QuinAABB outAABB;
            for (auto& triId : inNode->triangleIds)
            {
                outAABB += triangles[triId].aabb();
            }
            return outAABB;
        }
    };
}