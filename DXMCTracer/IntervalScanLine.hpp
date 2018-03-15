#pragma once
#include "stdafx.h"

#include <list>
#include <set>
#include <vector>

#include "Math.hpp"

struct EdgeNode
{
    PWint m_ymax;
    PWint m_y;
    PWdouble m_x;
    PWdouble m_dx;
    PWint m_polygonId;
};

struct PolygonNode
{
    PWint m_id;
    PWdouble m_a, m_b, m_c, m_d;
    Math::Vector3d m_color;
};

using EdgeTable = std::vector<EdgeNode>;
using ActiveEdgeTable = std::list<EdgeNode>;
using PolygonTable = std::vector<PolygonNode>;
using InPolygonTable = std::list<PWint>;

inline PWint getNearestPoly(const PolygonTable &pt, const InPolygonTable &ipl, const ActiveEdgeTable &aet, const ActiveEdgeTable::iterator &edge1, const ActiveEdgeTable::iterator &edge2)
{
    if (edge2 == aet.end())
    {
        return -1;
    }
    auto x = (edge1->m_x + edge2->m_x) / 2;
    auto y = edge1->m_y;
    PWdouble max = -1;
    PWint idx = -1;
    for (auto poly : ipl)
    {
        PWdouble depth = -(pt[poly].m_a * x + pt[poly].m_b * y + pt[poly].m_d) / pt[poly].m_c;
        if (depth > max)
        {
            max = depth;
            idx = poly;
        }
    }
    return idx;
}
