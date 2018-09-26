#pragma once
#include <stdafx.h>

#include <algorithm>
#include <list>
#include <set>
#include <vector>

#include <tiny_obj_loader.h>

namespace Quin::Utils
{
    enum class Axis
    {
        AXIS_NONE,
        AXIS_X,
        AXIS_Y,
        AXIS_Z
    };

    struct KDTreeNode
    {
        Axis axis = Axis::AXIS_NONE;
        FLOAT split = 0.0f;
        std::set<UINT> triangleIds;
        KDTreeNode* left = nullptr;
        KDTreeNode* right = nullptr;
    };

    class KDTree
    {
    private:
        union Point3
        {
            Point3() :x(0.0f), y(0.0f), z(0.0f) {}
            Point3(FLOAT xx, FLOAT yy, FLOAT zz) :x(xx), y(yy), z(zz) {}
            struct
            {
                FLOAT x, y, z;
            };
            FLOAT p[3];
        };

        struct AABB
        {
            AABB() :min(FLT_MAX, FLT_MAX, FLT_MAX), max(-FLT_MAX, -FLT_MAX, -FLT_MAX) {}
            void Clear()
            {
                min.x = FLT_MAX;
                min.y = FLT_MAX;
                min.z = FLT_MAX;
                max.x = -FLT_MAX;
                max.y = -FLT_MAX;
                max.z = -FLT_MAX;
            }
            void Add(const Point3& rhs)
            {
                min.x = std::min(min.x, rhs.x);
                min.y = std::min(min.y, rhs.y);
                min.z = std::min(min.z, rhs.z);
                max.x = std::max(max.x, rhs.x);
                max.y = std::max(max.y, rhs.y);
                max.z = std::max(max.z, rhs.z);
            }
            Point3 min;
            Point3 max;
        };

        class Triangle
        {
        public:
            Triangle()
            {
                new (&v0) Point3;
                new (&v1) Point3;
                new (&v2) Point3;
            }
            union
            {
                struct
                {
                    Point3 v0, v1, v2;
                };
                Point3 v[3];
            };
            void CalcAABB()
            {
                m_aabb.min.x = std::min(std::min(v0.x, v1.x), v2.x);
                m_aabb.min.y = std::min(std::min(v0.y, v1.y), v2.y);
                m_aabb.min.z = std::min(std::min(v0.z, v1.z), v2.z);
                m_aabb.max.x = std::max(std::max(v0.x, v1.x), v2.x);
                m_aabb.max.y = std::max(std::max(v0.y, v1.y), v2.y);
                m_aabb.max.z = std::max(std::max(v0.z, v1.z), v2.z);
            }
        private:
            AABB m_aabb;
        };
    public:
        static AABB GetNodeAABB(const std::vector<Triangle>& triangles, KDTreeNode* node)
        {

        }
        static void BuildTree(const tinyobj::attrib_t& attr, const std::vector<tinyobj::shape_t>& shapes)
        {
            std::vector<Triangle> triangles;
            for (auto& shape : shapes)
            {
                for (size_t i = 0; i < shape.mesh.indices.size(); i += 3)
                {
                    Triangle tri;
                    for (UINT vId = 0; vId < 3; ++vId)
                    {
                        for (UINT coordId = 0; coordId < 3; ++coordId)
                        {
                            tri.v[vId].p[coordId] = attr.vertices[3 * shape.mesh.indices[i + vId].vertex_index + coordId];
                        }
                    }
                    triangles.push_back(tri);
                }
            }

            std::list<KDTreeNode*> activeList;

            KDTreeNode* root = new KDTreeNode;
            activeList.push_back(root);

            for (UINT i = 0U; i < static_cast<UINT>(triangles.size()); ++i)
            {
                root->triangleIds.insert(i);
                triangles[i].CalcAABB();
            }

            while (!activeList.empty())
            {

            }
        }
    };
}