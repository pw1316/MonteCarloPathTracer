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

    struct CB0
    {
        D3DXMATRIX viewMatrix;
        D3DXMATRIX projMatrix;
        UINT seed;
        UINT prevCount;
        UINT padding0;
        UINT padding1;
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

    struct CSGeometry
    {
        UINT startTri;
        UINT numTries;
    };
}
