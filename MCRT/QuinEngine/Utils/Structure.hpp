#pragma once
#include <stdafx.h>

namespace Quin::Utils
{
    struct Model
    {
        Model(const std::string& objPath, const std::string& mtlBase)
        {
            tinyobj::LoadObj(&attr, &shapes, &materials, nullptr, objPath.c_str(), mtlBase.c_str());
        }
        tinyobj::attrib_t attr;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
    };

    struct CBGeometry
    {
        UINT nVertex;
        UINT nNormal;
        UINT nTriangle;
        UINT nGeometry;
    };

    struct CSTriangle
    {
        UINT v0, v1, v2;
        UINT n0, n1, n2;
    };
}
